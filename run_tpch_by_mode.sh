#!/usr/bin/env bash
#
# Run all TPC-H queries mode-major: every query under mode 0, then every query
# under mode 1, then every query under mode 2.
#
#   0 = native DuckDB    1 = unoptimized plan (MLIR)    2 = optimized plan (MLIR)
#
# Execution order is mode-major, but the CSV is written query-major: each query's
# rows appear together, ordered mode 0, 1, 2, so the three timings for a query sit
# side by side. Modes 1 and 2 carry compilation_ms and execution_ms in addition to
# the wall/real time; mode 0 has no compile step so those columns are empty.
#
# ClientContext::readCompileConfig() caches the mode in the process-global
# COMPILE_QUERIES, so the mode can only be chosen once per process. Each
# (query, mode) therefore gets its own CLI invocation -- which also keeps a
# crash in one compiled query from taking down the rest of the sweep.
#
# Usage:
#   ./run_tpch_by_mode.sh                    # all 22 queries, modes 0 then 1 then 2
#   ./run_tpch_by_mode.sh --dry-run          # print the plan, run nothing
#   ./run_tpch_by_mode.sh --queries q01,q06
#   ./run_tpch_by_mode.sh --modes 0,2
#   ./run_tpch_by_mode.sh --timeout 300 --db other.duckdb --out results/mine

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUCKDB="$REPO/build-fast/duckdb"
DB="$REPO/local.duckdb"
# LingoDB session directory, passed as the 2nd positional arg to the binary.
# Used by the compiled path (modes 1 & 2); ignored by native mode 0. The binary
# itself falls back to "bench_db" if this is empty.
LINGODB_DIR="bench_db"
QUERY_DIR="$REPO/extension/tpch/dbgen/queries"
OUT_DIR="$REPO/tpch_results"
TIMEOUT=600
DRY_RUN=0
QUERIES=""
MODES="0 1 2"

while [ $# -gt 0 ]; do
	case "$1" in
		--db)      DB="$2"; shift 2 ;;
		--duckdb)  DUCKDB="$2"; shift 2 ;;
		--lingodb-dir) LINGODB_DIR="$2"; shift 2 ;;
		--out)     OUT_DIR="$2"; shift 2 ;;
		--timeout) TIMEOUT="$2"; shift 2 ;;
		--queries) QUERIES="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
		--modes)   MODES="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
		--dry-run) DRY_RUN=1; shift ;;
		-h|--help) sed -n '2,27p' "$0"; exit 0 ;;
		*) echo "unknown argument: $1" >&2; exit 2 ;;
	esac
done

# ---------------------------------------------------------------- preflight --
if [ ! -x "$DUCKDB" ]; then
	echo "ERROR: duckdb binary not found or not executable: $DUCKDB" >&2
	echo "       build it first with ./build.sh" >&2
	exit 1
fi
if [ ! -f "$DB" ]; then
	echo "ERROR: database file not found: $DB" >&2
	exit 1
fi
if [ ! -s "$DB" ]; then
	# On an empty/new database the mode digit races the shell's SQL parser and
	# gets read as SQL instead of by readCompileConfig()'s scanf.
	echo "ERROR: database file is empty: $DB" >&2
	echo "       point --db at a populated TPC-H database." >&2
	exit 1
fi
if [ ! -d "$QUERY_DIR" ]; then
	echo "ERROR: TPC-H query directory not found: $QUERY_DIR" >&2
	exit 1
fi

if [ -z "$QUERIES" ]; then
	for n in $(seq -w 1 22); do QUERIES="$QUERIES q$n"; done
fi

if [ "$DRY_RUN" -eq 0 ]; then
	lock_check="$(printf '0\n.quit\n' | "$DUCKDB" "$DB" 2>&1)"
	if echo "$lock_check" | grep -q "Conflicting lock"; then
		echo "ERROR: database is locked by another DuckDB process." >&2
		echo "$lock_check" | grep "Conflicting lock" >&2
		echo "       Close that session (.quit) and re-run." >&2
		exit 1
	fi
fi

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_DIR/tpch_bymode_$STAMP.log"
CSV="$OUT_DIR/tpch_bymode_$STAMP.csv"
STDIN_FILE="$OUT_DIR/.stdin_$$"
RAW="$OUT_DIR/.raw_$$"

mode_name() {
	case "$1" in
		0) echo "native" ;;
		1) echo "unoptimized" ;;
		2) echo "optimized" ;;
		*) echo "mode$1" ;;
	esac
}

# stdin for one (query, mode):
#   line 1     : mode digit, consumed by readCompileConfig()'s scanf
#   lines 2..5 : session setup -- timer on, untruncated duckbox output
#   then       : the query, then .quit
build_stdin() {
	local mode="$1" qfile="$2" dest="$3"
	{
		echo "$mode"
		echo ".timer on"
		echo ".mode duckbox"
		echo ".maxrows -1"
		echo ".maxwidth -1"
		cat "$qfile"
		echo ""
		echo ".quit"
	} > "$dest"
}

# ------------------------------------------------------------------ dry run --
if [ "$DRY_RUN" -eq 1 ]; then
	echo "DRY RUN -- nothing will be executed"
	echo
	echo "binary   : $DUCKDB"
	echo "database : $DB"
	echo "queries  : $QUERIES"
	echo "modes    : $MODES"
	echo "timeout  : ${TIMEOUT}s per invocation"
	echo "log      : $OUT_DIR/tpch_bymode_<stamp>.log"
	echo "csv      : $OUT_DIR/tpch_bymode_<stamp>.csv"
	n=0
	for m in $MODES; do for q in $QUERIES; do n=$((n + 1)); done; done
	echo "invocations: $n"
	echo
	echo "execution order (mode-major):"
	for m in $MODES; do
		echo "  --- mode $m ($(mode_name "$m")) ---"
		for q in $QUERIES; do printf '    %s\n' "$q"; done
	done
	echo
	echo "csv row order (query-major, modes ascending):"
	for q in $QUERIES; do
		for m in $(echo "$MODES" | tr ' ' '\n' | sort -n); do
			printf '  %s,%s\n' "$q" "$m"
		done
	done
	exit 0
fi

# ------------------------------------------------------------------- run it --
cleanup() { rm -f "$STDIN_FILE" "$RAW"; }
trap cleanup EXIT

: > "$RAW"

{
	echo "TPC-H sweep (mode-major)"
	echo "started  : $(date)"
	echo "binary   : $DUCKDB"
	echo "database : $DB"
	echo "lingodb  : $LINGODB_DIR (modes 1 & 2)"
	echo "queries  : $QUERIES"
	echo "modes    : $MODES (0=native 1=unoptimized 2=optimized)"
	echo "timeout  : ${TIMEOUT}s per invocation"
	echo "git      : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo n/a)"
} | tee "$LOG"

total=0; ok=0; failed=0

for m in $MODES; do
	mname="$(mode_name "$m")"
	{
		echo
		echo "##############################################################"
		echo "## PASS: MODE $m ($mname) -- all queries"
		echo "##############################################################"
	} | tee -a "$LOG"

	for q in $QUERIES; do
		qfile="$QUERY_DIR/$q.sql"
		if [ ! -f "$qfile" ]; then
			echo "SKIP $q -- no such file: $qfile" | tee -a "$LOG"
			continue
		fi

		total=$((total + 1))
		build_stdin "$m" "$qfile" "$STDIN_FILE"

		{
			echo
			echo "=============================================================="
			echo "QUERY $q | MODE $m ($mname)"
			echo "=============================================================="
		} | tee -a "$LOG"

		start_ns=$(date +%s)
		# perl's alarm is our timeout; macOS ships no timeout(1).
		out="$(perl -e 'alarm shift @ARGV; exec @ARGV' "$TIMEOUT" \
			"$DUCKDB" "$DB" "$LINGODB_DIR" < "$STDIN_FILE" 2>&1)"
		rc=$?
		wall=$(( $(date +%s) - start_ns ))

		echo "$out" | tee -a "$LOG"

		# ---- classify ----
		status="OK"
		if [ "$rc" -eq 142 ]; then
			status="TIMEOUT"
		elif [ "$rc" -gt 128 ]; then
			status="CRASH_sig$((rc - 128))"
		elif [ "$rc" -ne 0 ]; then
			status="EXIT$rc"
		elif echo "$out" | grep -qE '^(Error|.*Error:)'; then
			status="ERROR"
		fi
		[ "$status" = "OK" ] && ok=$((ok + 1)) || failed=$((failed + 1))

		# ---- timings ----
		# native   : ".timer on" -> "Run Time (s): real 0.024 user ... sys ..."
		# compiled : "compilation: 131.782 [ms] execution: 41.624 [ms]"
		real_s="$(echo "$out"  | sed -n 's/.*Run Time (s): real \([0-9.]*\).*/\1/p' | tail -1)"
		comp_ms="$(echo "$out" | sed -n 's/.*compilation: \([0-9.]*\) \[ms\].*/\1/p' | tail -1)"
		exec_ms="$(echo "$out" | sed -n 's/.*execution: \([0-9.]*\) \[ms\].*/\1/p'   | tail -1)"

		# Buffered, not written straight to the CSV: execution is mode-major but
		# the CSV is emitted query-major once every pass has finished.
		printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
			"$q" "$m" "$mname" "$status" "$rc" "$wall" \
			"${real_s:-}" "${comp_ms:-}" "${exec_ms:-}" >> "$RAW"

		echo "--- $q mode $m ($mname): $status  wall=${wall}s  real=${real_s:--}s  compile=${comp_ms:--}ms  exec=${exec_ms:--}ms" | tee -a "$LOG"
	done
done

# --------------------------------------------------------- emit CSV in order --
# Sort by query name, then numerically by mode, so each query's 0/1/2 rows are
# adjacent and in ascending mode order.
{
	echo "query,mode,mode_name,status,exit_code,wall_s,duckdb_real_s,compilation_ms,execution_ms"
	sort -t',' -k1,1 -k2,2n "$RAW"
} > "$CSV"

{
	echo
	echo "=============================================================="
	echo "DONE  $(date)"
	echo "invocations: $total   ok: $ok   failed: $failed"
	echo "full log : $LOG"
	echo "csv      : $CSV"
	echo "=============================================================="
	echo
	column -t -s',' "$CSV" 2>/dev/null || cat "$CSV"
} | tee -a "$LOG"
