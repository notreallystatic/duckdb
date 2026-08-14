#!/usr/bin/env bash
#
# Run all TPC-H queries mode-major: every query under mode 0, then mode 1, then
# mode 2 -- but now with ONE process per mode instead of one per (query, mode).
#
#   0 = native DuckDB    1 = unoptimized plan (MLIR)    2 = optimized plan (MLIR)
#
# Why one process per mode:
#   The compiled path (modes 1 & 2) loads the whole LingoDB session (all Arrow
#   table data, ~seconds at SF10) inside runMLIR() on the first query. runMLIR()
#   caches that session in a process-global static, so subsequent queries in the
#   SAME process reuse it for free. Running one process per (query, mode) -- as the
#   old harness did -- defeats that cache: every query paid the full cold load,
#   which dominated the wall/real time. Feeding all queries into one process per
#   mode pays the load once (absorbed by a warm-up query) and lets the real
#   per-query cost show through.
#
#   ClientContext::readCompileConfig() reads the mode digit once from stdin (line 1)
#   and caches it in the process-global COMPILE_QUERIES, so a process runs a single
#   mode -- hence still one process PER MODE, but all queries within it.
#
# Trade-off (crash isolation): because a whole mode now shares one process, a
# segfault/timeout in one compiled query aborts the rest of that mode's run. Queries
# that never produced output are reported as INCOMPLETE. Run a single flaky query on
# its own (--queries qNN) if you need to isolate it.
#
# Per-query attribution (from the client_context.cpp timers + LingoDB timing):
#   compilation_ms      MLIR lowering + JIT codegen              (LingoDB)
#   execution_ms        JIT kernel run over the data             (LingoDB)
#   runmlir_total_ms    whole runMLIR() wall: session load(once) + compile + exec
#                       + executer setup + result readback       ([attrib])
#   residual_ms         runmlir_total - compilation - execution  (setup/scheduling/
#                       readback; for the warm-up row this is ~the session load)
#   real_s              DuckDB's .timer for the statement
#
# Usage:
#   ./run_tpch_by_mode.sh                    # all 22 queries, modes 0 then 1 then 2
#   ./run_tpch_by_mode.sh --dry-run          # print the plan, run nothing
#   ./run_tpch_by_mode.sh --queries q01,q06
#   ./run_tpch_by_mode.sh --skip q11          # run all queries except q11
#   ./run_tpch_by_mode.sh --modes 0,2
#   ./run_tpch_by_mode.sh --timeout 1200 --db other.duckdb --out results/mine
#   ./run_tpch_by_mode.sh --no-warmup        # skip the warm-up query
#   ./run_tpch_by_mode.sh --exec-mode DEFAULT  # verify MLIR (default is SPEED)
#   ./run_tpch_by_mode.sh --repeats 7          # loop the suite 7x; min+median per query
#   ./run_tpch_by_mode.sh --cooldown 60        # idle 60s between modes (thermal)
#
# NOTE: --timeout is now the budget for a WHOLE mode's process (all queries), not
# per query. Default 600s is ample for 22 SF10 queries; raise it for larger SFs.

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUCKDB="$REPO/build-fast/duckdb"
DB="$REPO/tpch_10.duckdb"
# LingoDB session directory, passed as the 2nd positional arg to the binary.
# Used by the compiled path (modes 1 & 2); ignored by native mode 0. The binary
# itself falls back to "bench_db" if this is empty.
LINGODB_DIR="/Users/sachinsheoran/Documents/masters_curriculum/project/lingo-db/bench_db_10"
QUERY_DIR="$REPO/extension/tpch/dbgen/queries"
OUT_DIR="$REPO/tpch_results"
TIMEOUT=600
DRY_RUN=0
QUERIES=""
# Queries to exclude from the sweep (comma/space separated, e.g. "q11"). Applied after
# the query list is resolved, so it works with the default 22 or an explicit --queries.
# Useful to drop a query that crashes the mode process (e.g. q11 at --cores 1 mode 2).
# Does NOT affect the warm-up query.
SKIP=""
MODES="0 1 2"
WARMUP=1
# The warm-up query runs first in every process to absorb the one-time session load,
# so the first *real* query isn't charged for it. Its own row makes the load cost
# visible (as its residual_ms). It must be a real TPC-H query (a name from
# QUERY_DIR) that compiles in every mode -- a synthetic `count(*)` is no good because
# its optimized plan collapses to a DUMMY_SCAN the MLIR translation can't handle.
# Default q01: a full lineitem scan+aggregate, so it also warms the biggest table.
WARMUP_QUERY="q01"
# LingoDB execution mode for the compiled path (modes 1 & 2), forwarded via the
# LINGODB_EXECUTION_MODE env var (read at process start). SPEED skips MLIR
# verification after every lowering step, cutting compilation_ms; DEFAULT verifies
# (safer, slower). Other valid values: PERF, DEBUGGING, CHEAP, C, NONE. Ignored by
# native mode 0.
EXEC_MODE="SPEED"
# Eager loading (LINGODB_BACKEND_ONLY): when set, createSession loads ALL tables at
# session open (during the warm-up), instead of lazily on first touch. This moves the
# per-table first-touch cost off individual queries (e.g. q02/q03, which otherwise pay
# to load part/orders/customer/...) and into the warm-up, so residual_ms is consistent
# and small across every real query. On by default; --lazy restores per-touch loading.
EAGER=1
# LingoDB worker-thread count for the compiled path (LINGODB_PARALLELISM). Empty means
# "let LingoDB decide" = std::thread::hardware_concurrency() (all logical cores). On
# Apple Silicon (e.g. M1: 4 performance + 4 efficiency cores) "all cores" oversubscribes
# the slow E-cores: memory-bandwidth-bound scans (q06/q01) saturate at ~2 P-cores and get
# WORSE with more workers, while compute-bound joins scale to all 8. There is no single
# best value; pin this to A/B P-core count (4) vs all (8). "OFF" forces single-threaded.
PARALLELISM=""
# Fixed core/thread count applied to BOTH engines for a fair comparison: DuckDB native via
# `PRAGMA threads=N` (mode 0) and LingoDB via LINGODB_PARALLELISM (modes 1 & 2). Default 4
# (the M1's performance-core count). Set --cores "" (empty) to let each engine use all
# cores. --parallelism (above), if given, overrides this for LingoDB only -- e.g. to A/B
# LingoDB thread counts against a fixed DuckDB.
CORES=4
# Number of full-suite passes per mode: the whole query list is looped REPEATS times
# (q01..q22, then q01..q22 again, ...), so each query runs REPEATS times but with the other
# queries interleaved between its runs -- realistic cache/TLB conditions, not artificially
# hot. Reported compilation/execution/residual are the MIN across a query's runs (noise is
# one-sided, so min is the truest speed) plus median execution; the warm-up runs once
# regardless; a per-mode geomean is printed. Default 1 (single pass).
REPEATS=1
# Seconds to idle at boundaries to let the machine settle (thermals/caches): between each
# full-suite pass (in-session via `.system sleep`, so the process/session stays alive) AND
# between mode processes (shell sleep). Counts against --timeout, since the in-session
# sleeps run inside the mode process. Default 0.
COOLDOWN=0

while [ $# -gt 0 ]; do
	case "$1" in
		--db)      DB="$2"; shift 2 ;;
		--duckdb)  DUCKDB="$2"; shift 2 ;;
		--lingodb-dir) LINGODB_DIR="$2"; shift 2 ;;
		--out)     OUT_DIR="$2"; shift 2 ;;
		--timeout) TIMEOUT="$2"; shift 2 ;;
		--queries) QUERIES="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
		--skip)    SKIP="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
		--modes)   MODES="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
		--no-warmup) WARMUP=0; shift ;;
		--warmup-query) WARMUP_QUERY="$2"; shift 2 ;;
		--exec-mode) EXEC_MODE="$2"; shift 2 ;;
		--eager) EAGER=1; shift ;;
		--lazy)  EAGER=0; shift ;;
		--parallelism) PARALLELISM="$2"; shift 2 ;;
		--cores) CORES="$2"; shift 2 ;;
		--repeats) REPEATS="$2"; shift 2 ;;
		--cooldown) COOLDOWN="$2"; shift 2 ;;
		--dry-run) DRY_RUN=1; shift ;;
		-h|--help) sed -n '2,49p' "$0"; exit 0 ;;
		*) echo "unknown argument: $1" >&2; exit 2 ;;
	esac
done

# Forward the LingoDB execution mode to the compiled path. GlobalSetting reads this
# at process start (static init), so it must be exported before the binary launches.
export LINGODB_EXECUTION_MODE="$EXEC_MODE"
# Eager vs lazy table loading. runMLIR() treats the mere presence of LINGODB_BACKEND_ONLY
# as "eager", so we must UNSET it (not set it to 0) to get lazy loading.
if [ "$EAGER" -eq 1 ]; then
	export LINGODB_BACKEND_ONLY=1
else
	unset LINGODB_BACKEND_ONLY
fi
# LingoDB worker count: explicit --parallelism wins, else the shared --cores value; empty
# means "all logical cores".
LINGO_WORKERS="${PARALLELISM:-$CORES}"
if [ -n "$LINGO_WORKERS" ]; then
	export LINGODB_PARALLELISM="$LINGO_WORKERS"
else
	unset LINGODB_PARALLELISM
fi

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

# Drop any --skip queries from the resolved list (leaves the warm-up untouched).
if [ -n "$SKIP" ]; then
	kept=""
	for q in $QUERIES; do
		drop=0
		for s in $SKIP; do [ "$q" = "$s" ] && drop=1 && break; done
		[ "$drop" -eq 0 ] && kept="$kept $q"
	done
	QUERIES="$(echo $kept)"   # collapse leading/extra whitespace
	if [ -z "$QUERIES" ]; then
		echo "ERROR: --skip removed every query; nothing to run." >&2
		exit 2
	fi
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
MODEOUT="$OUT_DIR/.modeout_$$"
PARSED="$OUT_DIR/.parsed_$$"

mode_name() {
	case "$1" in
		0) echo "native" ;;
		1) echo "unoptimized" ;;
		2) echo "optimized" ;;
		*) echo "mode$1" ;;
	esac
}

# The ordered list of queries actually fed to a process (warm-up first, if enabled).
run_list() {
	[ "$WARMUP" -eq 1 ] && printf '%s ' "__warmup__"
	printf '%s ' $QUERIES
}

# Build the single stdin stream for one mode's process:
#   line 1     : mode digit, consumed by readCompileConfig()'s scanf
#   then       : session setup, then each query preceded by a ".print" marker so the
#                combined output can be split back into per-query segments.
build_mode_stdin() {
	local mode="$1" dest="$2"
	{
		echo "$mode"
		echo ".timer on"
		echo ".mode duckbox"
		echo ".maxrows -1"
		echo ".maxwidth -1"
		# Pin DuckDB's native executor thread count (mode 0) to match LingoDB's worker count
		# for a fair comparison. Runs natively even under compiled modes (a PRAGMA is not a
		# SELECT, so it bypasses runMLIR) where it's a harmless no-op. Only when CORES is numeric.
		case "$CORES" in '' | *[!0-9]*) ;; *) echo "PRAGMA threads=$CORES;" ;; esac
		if [ "$WARMUP" -eq 1 ]; then
			echo ".print ===QUERY __warmup__==="
			local wfile="$QUERY_DIR/$WARMUP_QUERY.sql"
			if [ -f "$wfile" ]; then
				cat "$wfile"; echo ""
			else
				echo "SELECT 'missing warm-up query file: $WARMUP_QUERY';"
			fi
		fi
		# Loop the WHOLE query suite REPEATS times (q01..q22, then q01..q22 again, ...) --
		# NOT each query N times back-to-back. Interleaving the other queries between a
		# query's runs keeps caches/TLB from being artificially hot, for realistic
		# conditions. Each run is marked; the parser emits one row per marker occurrence and
		# the aggregation groups by query name (order-independent) into min/median. The
		# warm-up above still runs once.
		local r q qfile
		for r in $(seq 1 "$REPEATS"); do
			for q in $QUERIES; do
				qfile="$QUERY_DIR/$q.sql"
				echo ".print ===QUERY $q==="
				if [ -f "$qfile" ]; then
					cat "$qfile"; echo ""
				else
					echo "SELECT 'missing query file: $q';"
				fi
			done
			# Cool down BETWEEN passes (not after the last), from inside the session via a
			# shell sleep, so the machine settles before the suite is replayed. sleep produces
			# no output and the note is not a query marker, so neither affects parsing.
			if [ "$COOLDOWN" -gt 0 ] && [ "$r" -lt "$REPEATS" ]; then
				echo ".print --- cooldown ${COOLDOWN}s after pass $r ---"
				echo ".system sleep $COOLDOWN"
			fi
		done
		echo ".quit"
	} > "$dest"
}

# Split one mode's combined stdout+stderr into per-query rows. The ".print" markers
# delimit segments; timing lines that fall inside a segment belong to that query.
# BSD-awk compatible (no gawk match(...,arr)). Emits TSV: query real comp exec runmlir
parse_mode_output() {
	awk '
		function emit() {
			if (curq == "") return
			printf "%s\t%s\t%s\t%s\t%s\n", curq, real, comp, exec, runmlir
		}
		index($0, "===QUERY") > 0 {
			emit()
			line = $0
			sub(/.*===QUERY /, "", line)
			sub(/===.*/, "", line)
			curq = line; real = ""; comp = ""; exec = ""; runmlir = ""
			next
		}
		index($0, "compilation:") > 0 && index($0, "execution:") > 0 { comp = $2; exec = $5; next }
		index($0, "runmlir_total:") > 0 { runmlir = $3; next }
		/Run Time/ { for (i = 1; i <= NF; i++) if ($i == "real") real = $(i + 1); next }
		END { emit() }
	' "$1"
}

# Reduce the N parsed rows for a single query (piped in as TSV: query real comp exec
# runmlir) to one line of aggregates. Min is the headline (noise is one-sided); median of
# execution is also reported. residual is computed per run (runmlir-comp-exec) then min'd.
# BSD-awk compatible (manual insertion-sort for the median). Emits TSV:
#   n  real_min  comp_min  exec_min  exec_med  runmlir_min  residual_min
aggregate_query() {
	awk -F'\t' '
		function mn(a, k,   i, m) { if (k == 0) return ""; m = a[1]; for (i = 2; i <= k; i++) if (a[i] < m) m = a[i]; return m }
		function med(a, k,   i, j, t, b) {
			if (k == 0) return ""
			for (i = 1; i <= k; i++) b[i] = a[i]
			for (i = 2; i <= k; i++) { t = b[i]; j = i - 1; while (j >= 1 && b[j] > t) { b[j+1] = b[j]; j-- } b[j+1] = t }
			return (k % 2) ? b[(k+1)/2] : (b[k/2] + b[k/2+1]) / 2.0
		}
		{
			rows++
			if ($2 != "") R[++nr] = $2
			if ($3 != "") C[++nc] = $3
			if ($4 != "") E[++ne] = $4
			if ($5 != "") M[++nm] = $5
			if ($3 != "" && $4 != "" && $5 != "") D[++nd] = $5 - $3 - $4
		}
		END {
			printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\n", rows, mn(R, nr), mn(C, nc), mn(E, ne),
				(ne ? med(E, ne) : ""), mn(M, nm), mn(D, nd)
		}
	'
}

# Geometric mean of the whitespace-separated positive numbers on stdin (empty -> "").
geomean() {
	awk '{ for (i = 1; i <= NF; i++) if ($i + 0 > 0) { s += log($i); n++ } } END { if (n) printf "%.3f", exp(s / n); else printf "" }'
}

# ------------------------------------------------------------------ dry run --
if [ "$DRY_RUN" -eq 1 ]; then
	echo "DRY RUN -- nothing will be executed"
	echo
	echo "binary   : $DUCKDB"
	echo "database : $DB"
	echo "lingodb  : $LINGODB_DIR (modes 1 & 2)"
	echo "queries  : $QUERIES"
	echo "skip     : ${SKIP:-(none)}"
	echo "modes    : $MODES"
	echo "warm-up  : $([ "$WARMUP" -eq 1 ] && echo "on ($WARMUP_QUERY.sql)" || echo off)"
	echo "exec_mode: $EXEC_MODE (LINGODB_EXECUTION_MODE; modes 1 & 2)"
	echo "loading  : $([ "$EAGER" -eq 1 ] && echo "eager (all tables at warm-up)" || echo "lazy (per first-touch)")"
	echo "cores    : ${CORES:-all} (both engines: DuckDB PRAGMA threads + LingoDB workers)"
	echo "parallel : lingodb=${LINGO_WORKERS:-all} (LINGODB_PARALLELISM), duckdb=${CORES:-all} (PRAGMA threads)"
	echo "repeats  : $REPEATS full-suite pass(es) per mode (min + median execution per query)"
	echo "cooldown : ${COOLDOWN}s between passes (in-session) and between modes"
	echo "timeout  : ${TIMEOUT}s per MODE process"
	echo "log      : $OUT_DIR/tpch_bymode_<stamp>.log"
	echo "csv      : $OUT_DIR/tpch_bymode_<stamp>.csv"
	echo "invocations: $(echo $MODES | wc -w | tr -d ' ') (one process per mode)"
	echo
	echo "per-mode stdin (order): mode-digit, setup, [warm-up], then $REPEATS pass(es) of $(echo $QUERIES | wc -w | tr -d ' ') queries, .quit"
	exit 0
fi

# ------------------------------------------------------------------- run it --
cleanup() { rm -f "$STDIN_FILE" "$RAW" "$MODEOUT" "$PARSED"; }
trap cleanup EXIT

: > "$RAW"

{
	echo "TPC-H sweep (mode-major, one process per mode)"
	echo "started  : $(date)"
	echo "binary   : $DUCKDB"
	echo "database : $DB"
	echo "lingodb  : $LINGODB_DIR (modes 1 & 2)"
	echo "queries  : $QUERIES"
	echo "skip     : ${SKIP:-(none)}"
	echo "modes    : $MODES (0=native 1=unoptimized 2=optimized)"
	echo "warm-up  : $([ "$WARMUP" -eq 1 ] && echo "on ($WARMUP_QUERY.sql)" || echo off)"
	echo "exec_mode: $EXEC_MODE (LINGODB_EXECUTION_MODE; modes 1 & 2)"
	echo "loading  : $([ "$EAGER" -eq 1 ] && echo "eager (all tables at warm-up)" || echo "lazy (per first-touch)")"
	echo "cores    : ${CORES:-all} (both engines: DuckDB PRAGMA threads + LingoDB workers)"
	echo "parallel : lingodb=${LINGO_WORKERS:-all} (LINGODB_PARALLELISM), duckdb=${CORES:-all} (PRAGMA threads)"
	echo "repeats  : $REPEATS full-suite pass(es) per mode (min + median execution per query)"
	echo "cooldown : ${COOLDOWN}s between passes (in-session) and between modes"
	echo "timeout  : ${TIMEOUT}s per MODE process"
	echo "git      : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo n/a)"
} | tee "$LOG"

total=0; ok=0; failed=0

for m in $MODES; do
	mname="$(mode_name "$m")"
	{
		echo
		echo "##############################################################"
		echo "## PASS: MODE $m ($mname) -- one process, all queries"
		echo "##############################################################"
	} | tee -a "$LOG"

	build_mode_stdin "$m" "$STDIN_FILE"

	start_ns=$(date +%s)
	# perl's alarm is our timeout; macOS ships no timeout(1). One process for the
	# whole mode now, so this budget covers every query in it.
	perl -e 'alarm shift @ARGV; exec @ARGV' "$TIMEOUT" \
		"$DUCKDB" "$DB" "$LINGODB_DIR" < "$STDIN_FILE" > "$MODEOUT" 2>&1
	prc=$?
	mode_wall=$(( $(date +%s) - start_ns ))

	cat "$MODEOUT" | tee -a "$LOG" >/dev/null
	{
		echo "--- mode $m ($mname) process: exit=$prc  wall=${mode_wall}s"
		if [ "$prc" -eq 142 ]; then echo "--- WARNING: mode process TIMED OUT after ${TIMEOUT}s; later queries incomplete"; fi
		if [ "$prc" -gt 128 ] && [ "$prc" -ne 142 ]; then echo "--- WARNING: mode process CRASHED (sig $((prc - 128))); later queries incomplete"; fi
	} | tee -a "$LOG"

	# Parse per-query timings out of the combined stream.
	parse_mode_output "$MODEOUT" > "$PARSED"

	# Report the warm-up separately: its residual is the one-time session load.
	if [ "$WARMUP" -eq 1 ]; then
		wline="$(grep -E '^__warmup__	' "$PARSED" || true)"
		if [ -n "$wline" ]; then
			wrun="$(echo "$wline" | cut -f5)"; wcomp="$(echo "$wline" | cut -f3)"; wexec="$(echo "$wline" | cut -f4)"
			wload="$(awk -v r="$wrun" -v c="$wcomp" -v e="$wexec" 'BEGIN{ if(r!=""&&c!=""&&e!="") printf "%.3f", r-c-e; else print "" }')"
			echo "--- mode $m warm-up: session_load≈${wload:-?}ms (runmlir_total=${wrun:-?} comp=${wcomp:-?} exec=${wexec:-?})" | tee -a "$LOG"
		fi
	fi

	# Emit one CSV-buffer row per expected query (warm-up included so its load shows).
	# Values are the MIN across the query's REPEATS runs (plus median execution).
	geo_vals=""   # exec_min per real query, for this mode's geometric mean
	for q in $(run_list); do
		total=$((total + 1))
		# All rows for this query (REPEATS of them, tab-anchored so q01 != q010).
		rows="$(grep -E "^$q	" "$PARSED" || true)"
		agg="$(printf '%s\n' "$rows" | aggregate_query)"
		n="$(echo "$agg" | cut -f1)"
		real="$(echo "$agg" | cut -f2)"; comp="$(echo "$agg" | cut -f3)"
		exec="$(echo "$agg" | cut -f4)"; exec_med="$(echo "$agg" | cut -f5)"
		runmlir="$(echo "$agg" | cut -f6)"; residual="$(echo "$agg" | cut -f7)"
		if [ -z "$rows" ] || [ "${n:-0}" -eq 0 ]; then
			# Marker never appeared -> the process died before reaching this query.
			status="INCOMPLETE"; n=0
			failed=$((failed + 1))
		else
			# Completion criterion: native needs real; compiled needs runmlir_total.
			if [ "$m" -eq 0 ]; then
				[ -n "$real" ] && status="OK" || status="INCOMPLETE"
			else
				[ -n "$runmlir" ] && status="OK" || status="INCOMPLETE"
			fi
			[ "$status" = "OK" ] && ok=$((ok + 1)) || failed=$((failed + 1))
			# Collect the real queries' exec_min for the geomean (skip warm-up).
			if [ "$status" = "OK" ] && [ "$q" != "__warmup__" ] && [ -n "$exec" ]; then
				geo_vals="$geo_vals $exec"
			fi
		fi
		printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
			"$q" "$m" "$mname" "$status" "$n" \
			"${real:-}" "${comp:-}" "${exec:-}" "${exec_med:-}" "${runmlir:-}" "${residual:-}" >> "$RAW"
		echo "--- $q mode $m ($mname): $status n=$n  real=${real:--}s comp=${comp:--}ms exec_min=${exec:--}ms exec_med=${exec_med:--}ms runmlir=${runmlir:--}ms residual=${residual:--}ms" | tee -a "$LOG"
	done

	# Per-mode geometric mean of min execution across the real queries.
	if [ "$m" -ne 0 ]; then
		gm="$(printf '%s' "$geo_vals" | geomean)"
		echo "--- mode $m ($mname): geomean(execution_ms_min) = ${gm:-n/a} ms over $(echo $geo_vals | wc -w | tr -d ' ') queries" | tee -a "$LOG"
	fi

	# Cool down between modes to limit thermal bias in the mode-major ordering.
	if [ "$COOLDOWN" -gt 0 ]; then
		echo "--- cooling down ${COOLDOWN}s before next mode" | tee -a "$LOG"
		sleep "$COOLDOWN"
	fi
done

# --------------------------------------------------------- emit CSV in order --
# Sort by query name, then numerically by mode, so each query's rows are adjacent
# and in ascending mode order (warm-up sorts to the top).
{
	# Numeric columns are the MIN across the query's repeats (== the single value when
	# --repeats 1); execution_ms_median is the median of execution across repeats; n is the
	# number of completed runs.
	echo "query,mode,mode_name,status,n,duckdb_real_s,compilation_ms,execution_ms,execution_ms_median,runmlir_total_ms,residual_ms"
	sort -t',' -k1,1 -k2,2n "$RAW"
} > "$CSV"

{
	echo
	echo "=============================================================="
	echo "DONE  $(date)"
	echo "rows: $total   ok: $ok   incomplete: $failed"
	echo "full log : $LOG"
	echo "csv      : $CSV"
	echo "=============================================================="
	echo
	column -t -s',' "$CSV" 2>/dev/null || cat "$CSV"
} | tee -a "$LOG"
