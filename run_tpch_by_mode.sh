#!/usr/bin/env bash
#
# Run a benchmark suite iteration-major: with --repeats 2 and modes 0,1,2 the order
# is m0 m1 m2 L m0 m1 m2 L, and every (iteration, mode) pair is a FRESH process running
# the whole query list once. Reported numbers are the min over the iterations.
#
# Why iteration-major with a fresh process each time: caches. A new process starts
# with an empty DuckDB buffer pool and no LingoDB session, and interleaving the modes
# means a mode is never measured immediately after itself, so one mode cannot inherit
# a warm state left by its own previous run. What this does NOT clear is the OS page
# cache, which survives process exit -- see the note at the end of this header.
#
# Two suites ship with the harness, selected with --suite:
#   tpch   (default) the 22 TPC-H queries in extension/tpch/dbgen/queries
#   final            the curated set in microbench/final -- seven groups of three
#                    single-table queries plus TPC-H Q1; see microbench/final/README.md
#   micro            the micro-benchmark in microbench/queries -- single-table
#                    queries in seven categories of three, each isolating one
#                    effect; see microbench/README.md
#   test             scratch suite: multi-column GROUP BY plus the sort queries
#   presentation     the aggregate scans transcribed from the Nov 2025 presentation
#   count            the row-count queries from the same presentation
#   exploratory      the first-pass probes behind the Aug 2026 sweeps (has joins)
# Any other directory of .sql files works too: --query-dir DIR. When the suite is
# not tpch the query list defaults to every .sql file in the directory, sorted, so
# a new query is picked up by dropping the file in. Query NAMES are the file names
# without .sql, and that is what --queries/--skip/--warmup-query take.
#
#   0 = native DuckDB    1 = unoptimized plan (MLIR)    2 = optimized plan (MLIR)
#   L = LingoDB on its own SQL frontend (--no-lingodb-native to skip)
#
# Why all of a mode's queries still share one process:
#   The compiled path (modes 1 & 2) loads the whole LingoDB session (all Arrow
#   table data, ~seconds at SF10) inside runMLIR() on the first query. runMLIR()
#   caches that session in a process-global static, so subsequent queries in the
#   SAME process reuse it for free. One process per (query, mode) would make every
#   query pay the full cold load, which dominates the measurement. One process per
#   (iteration, mode) pays it once per iteration, absorbed by the warm-up query.
#
#   ClientContext::readCompileConfig() reads the mode digit once from stdin (line 1)
#   and caches it in the process-global COMPILE_QUERIES, so a process runs a single
#   mode -- hence one process per (iteration, mode), all queries within it.
#
#   The cost of a fresh process per iteration is that the session load is paid
#   --repeats times rather than once. At SF10 that is roughly 9s per compiled mode
#   per iteration.
#
# Crash recovery: because a whole mode shares one process, a segfault or a hang in
# one compiled query takes down every query behind it. The harness therefore retries:
# it identifies the query the process died on, drops it, and restarts the mode from
# just after it, keeping everything already measured. A dropped query is reported as
# FAILED, one that never got a turn before --max-retries ran out as INCOMPLETE.
#
# A retry is a new process, so it reloads the whole LingoDB session. The warm-up runs
# again to absorb that, and the re-warm-up's timings are discarded rather than folded
# into any query's numbers -- only the first attempt's warm-up is reported, where the
# residual is the genuine one-time session load.
#
# Per-query attribution (from the client_context.cpp timers + LingoDB timing):
#   plan_ms             DuckDB parse+bind+plan (+ MLIR build)    ([attrib]; modes 1,2)
#                       Mode L has no counterpart: its shell sums only the phases
#                       its backend times, the earliest being the relalg
#                       optimization, so its SQL parsing and module construction
#                       are never measured and appear in no column. Do NOT add
#                       plan_ms and compare the sum against mode L -- that sets a
#                       frontend-inclusive figure against a frontend-exclusive one.
#                       Only compilation_ms is comparable across modes 1, 2 and L.
#   compilation_ms      MLIR lowering + JIT codegen              (LingoDB)
#   execution_ms        JIT kernel run over the data             (LingoDB)
#   runmlir_total_ms    whole runMLIR() wall: session load(once) + compile + exec
#                       + executer setup + result readback       ([attrib])
#   residual_ms         runmlir_total - compilation - execution  (setup/scheduling/
#                       readback; for the warm-up row this is ~the session load).
#                       For mode L this column instead holds compilation + execution,
#                       since that mode has no runMLIR wrapper and hence no residual.
#   duckdb_real_s       end-to-end wall for the statement, in seconds. Modes 0/1/2 read
#                       it from DuckDB's .timer, so it covers everything: parse, bind,
#                       plan and then either the vectorized engine or the MLIR build
#                       plus the LingoDB compile and execute. Mode L has no such timer,
#                       so it carries (compilation + execution) / 1000 instead. That
#                       omits mode L's own SQL parsing, which nothing times, so mode
#                       L's total is understated relative to the other three by the
#                       cost of its frontend. Treat cross-mode totals involving L as
#                       slightly generous to L rather than as like for like.
#
# Usage:
#   ./run_tpch_by_mode.sh                    # all 22 queries, modes 0 then 1 then 2
#   ./run_tpch_by_mode.sh --suite micro      # the micro-benchmark instead
#   ./run_tpch_by_mode.sh --suite micro --queries c01,c04,v02
#   ./run_tpch_by_mode.sh --query-dir my/queries --warmup-query first
#   ./run_tpch_by_mode.sh --dry-run          # print the plan, run nothing
#   ./run_tpch_by_mode.sh --queries q01,q06
#   ./run_tpch_by_mode.sh --skip q11          # run all queries except q11
#   ./run_tpch_by_mode.sh --modes 0,2
#   ./run_tpch_by_mode.sh --suite micro --perfect-ht 0   # disable DuckDB perfect hashing
#   ./run_tpch_by_mode.sh --modes 0 --duckdb-profile     # per-operator profiles of DuckDB's plans
#   ./run_tpch_by_mode.sh --timeout 120 --db other.duckdb --out results/mine
#   ./run_tpch_by_mode.sh --no-warmup        # skip the warm-up query
#   ./run_tpch_by_mode.sh --exec-mode DEFAULT  # verify MLIR (default is SPEED)
#   ./run_tpch_by_mode.sh --no-inlj            # LingoDB never joins through its primary key hash index
#   ./run_tpch_by_mode.sh --repeats 7          # 7 iterations of every mode; min per query
#   ./run_tpch_by_mode.sh --cooldown 60        # idle 60s between processes (thermal)
#   ./run_tpch_by_mode.sh --max-retries 1      # do not retry after a crash
#   ./run_tpch_by_mode.sh --no-lingodb-native  # only the three DuckDB-side modes
#   ./run_tpch_by_mode.sh --lingodb-sql /path/to/sql
#
# NOTE: --timeout is a PER-QUERY budget. A whole mode still shares one process, so
# it is enforced by a watchdog on the progress markers rather than by an alarm on the
# process: a query that has not finished within the budget takes the process down and
# is then dropped by the retry logic. The warm-up is charged under the same budget and
# on the compiled path pays the one-time session load, so keep the budget above it.

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUCKDB="$REPO/build-fast/duckdb"
DB="$REPO/tpch_10.duckdb"
# LingoDB session directory, passed as the 2nd positional arg to the binary.
# Used by the compiled path (modes 1 & 2); ignored by native mode 0. The binary
# itself falls back to "bench_db" if this is empty.
LINGODB_DIR="/Users/sachinsheoran/Documents/masters_curriculum/project/lingo-db/bench_db_10"
# Which suite to run. Sets the query directory, the default query list and the
# default warm-up query; see resolve-suite below. --query-dir overrides the
# directory without changing the rest.
SUITE="tpch"
QUERY_DIR=""
QUERY_DIR_GIVEN=0
# Mode L only. When a file of the same name exists here it is used instead of the one
# in QUERY_DIR; otherwise QUERY_DIR is used. Modes 0, 1 and 2 always read QUERY_DIR,
# so the override never changes what the DuckDB-side modes execute. It exists because
# a few of DuckDB's TPC-H texts are not accepted the same way by LingoDB's own
# frontend: q15 groups by the select-list alias supplier_no, which LingoDB binds
# differently and plans into a query that exhausts memory at scale factor 10. The
# override holds LingoDB's own wording of the same query, which groups by l_suppkey.
# Left empty here and filled in from the suite below; --lingodb-query-dir overrides it.
LINGO_QUERY_DIR=""
LINGO_QUERY_DIR_GIVEN=0
# Print the module after query optimization and after every lowering step, in all
# three compiled modes (LINGODB_DUMP_IR). Off by default. The dumps are large and the
# printing happens inside the interval the compilation timers cover, so a run made
# with this on is for reading plans, not for reporting times.
DUMP_IR=0
# Count the tuples each relalg operator actually produces, in all three compiled
# modes (LINGODB_TRACK_TUPLES). Off by default. The counters are inserted into the
# plan and updated during execution, so a run made with this on measures
# cardinalities, not times. Pair it with --dump-ir: the counter ids are assigned
# just before the qopt module is printed, so that dump maps ids to operators.
TRACK_TUPLES=0
# Write DuckDB's per-operator profile of every query as JSON, in mode 0 only (PRAGMA
# enable_profiling='json'). Off by default. The profile records each operator's actual
# output cardinality, which is how join and aggregation sizes are read off DuckDB's own
# plan. Profiling times every operator, so a run made with this on is for reading
# cardinalities, not for reporting times. One file per query and iteration lands in
# <out>/<suite>_bymode_<stamp>_duckdb_profile/<query>_iter<N>.json. The warm-up is not
# profiled.
DUCKDB_PROFILE=0
OUT_DIR="$REPO/tpch_results"
# Per-QUERY budget in seconds. A query still running this long after its marker was
# printed is assumed hung: the mode process is killed and the retry logic drops that
# query and restarts the mode. The warm-up is charged under the same budget, and on
# the compiled path its time includes the one-time LingoDB session load, so this must
# comfortably exceed that load.
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
# Left empty here and filled in from the suite below, so that --suite micro picks
# its own warm-up without the caller having to name one.
WARMUP_QUERY=""
# LingoDB execution mode for the compiled path (modes 1 & 2), forwarded via the
# LINGODB_EXECUTION_MODE env var (read at process start). SPEED skips MLIR
# verification after every lowering step, cutting compilation_ms; DEFAULT verifies
# (safer, slower). Other valid values: PERF, DEBUGGING, CHEAP, C, NONE. Ignored by
# native mode 0.
EXEC_MODE="SPEED"
# Index nested loop joins in LingoDB (LINGODB_OPT_INDEX_NESTED_LOOP, read at process
# start). A primary key gives every LingoDB table a prebuilt hash index, and LingoDB's
# optimizer may join through it instead of building a hash table at query time. DuckDB
# has no counterpart: it never joins through an index. --no-inlj forces every such join
# to a plain hash join while the primary keys stay in place, so LingoDB keeps the row
# estimates and functional dependencies they provide and loses only the index itself.
# Modes 1 and 2 honour it through lingodb_ext; mode L only if its shell (--lingodb-sql)
# was built from a LingoDB tree carrying the same patch, otherwise it is silently
# ignored there. Default 1 (allowed), which is LingoDB's normal behaviour.
INLJ=1
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
# DuckDB perfect-hash-aggregate threshold, in bits (its `perfect_ht_threshold` setting,
# default 12). DuckDB uses PERFECT_HASH_GROUP_BY -- a direct array index with no hashing,
# probing or key comparison -- when every grouping key is an integer whose range fits
# inside this bit budget; otherwise it falls back to HASH_GROUP_BY. Since the compiled
# path has no equivalent strategy and always builds a hash table, this setting decides
# whether a grouping query is compared like for like. Set --perfect-ht 0 to disable
# perfect hashing and isolate its contribution; leave empty to use DuckDB's default.
# Applies to mode 0 only (it is a native-executor decision). See microbench grp1/mgrp1.
PERFECT_HT=""
# Number of ITERATIONS. One iteration runs the whole query list once under every
# mode, each in its own process, in the order the --modes list gives. So --repeats 3
# with modes 0,1,2 is nine processes: m0 m1 m2 m0 m1 m2 m0 m1 m2. Reported
# compilation/execution/residual are the MIN across a query's iterations (noise is
# one-sided, so min is the truest speed) plus median execution; a per-mode geomean is
# printed. Default 1.
REPEATS=1
# Seconds to idle after every mode process, to let the machine settle
# (thermals/caches). With iteration-major ordering this boundary is also the boundary
# between two measurements of the same query, so it is where a pause is worth most.
# Counts against nothing but the total wall. Default 0.
COOLDOWN=0
# A query that takes the mode process down (a crash, or a hang that hits --timeout)
# is dropped and the mode is restarted from just after it, so the queries behind it
# still get measured. Each restart is a new process and pays the LingoDB session load
# again, which is why the warm-up is re-run and its metrics discarded. This caps how
# many times that may happen per mode; 1 disables retrying.
MAX_RETRIES=5
# Also benchmark LingoDB through its OWN SQL frontend, as a fourth mode appended to
# every iteration: m0 m1 m2 L m0 m1 m2 L. Mode L shares nothing with modes 1 and 2
# except the LingoDB backend -- it parses, binds and plans the query itself instead of
# receiving a plan from DuckDB, so it is the reference for what the integration costs.
LINGODB_NATIVE=1
# The standalone LingoDB SQL shell used for mode L.
LINGODB_SQL="/Users/sachinsheoran/Documents/masters_curriculum/project/lingo-db/build/lingodb-release/sql"

while [ $# -gt 0 ]; do
	case "$1" in
		--db)      DB="$2"; shift 2 ;;
		--duckdb)  DUCKDB="$2"; shift 2 ;;
		--lingodb-dir) LINGODB_DIR="$2"; shift 2 ;;
		--suite)   SUITE="$2"; shift 2 ;;
		--query-dir) QUERY_DIR="$2"; QUERY_DIR_GIVEN=1; shift 2 ;;
		--lingodb-query-dir) LINGO_QUERY_DIR="$2"; LINGO_QUERY_DIR_GIVEN=1; shift 2 ;;
		--dump-ir) DUMP_IR=1; shift ;;
		--track-tuples) TRACK_TUPLES=1; shift ;;
		--duckdb-profile) DUCKDB_PROFILE=1; shift ;;
		--out)     OUT_DIR="$2"; shift 2 ;;
		--timeout) TIMEOUT="$2"; shift 2 ;;
		--queries) QUERIES="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
		--skip)    SKIP="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
		--modes)   MODES="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
		--no-warmup) WARMUP=0; shift ;;
		--warmup-query) WARMUP_QUERY="$2"; shift 2 ;;
		--exec-mode) EXEC_MODE="$2"; shift 2 ;;
		--inlj)    INLJ=1; shift ;;
		--no-inlj) INLJ=0; shift ;;
		--eager) EAGER=1; shift ;;
		--lazy)  EAGER=0; shift ;;
		--parallelism) PARALLELISM="$2"; shift 2 ;;
		--cores) CORES="$2"; shift 2 ;;
		--perfect-ht) PERFECT_HT="$2"; shift 2 ;;
		--repeats) REPEATS="$2"; shift 2 ;;
		--cooldown) COOLDOWN="$2"; shift 2 ;;
		--max-retries) MAX_RETRIES="$2"; shift 2 ;;
		--lingodb-sql) LINGODB_SQL="$2"; shift 2 ;;
		--lingodb-native) LINGODB_NATIVE=1; shift ;;
		--no-lingodb-native) LINGODB_NATIVE=0; shift ;;
		--dry-run) DRY_RUN=1; shift ;;
		-h|--help) sed -n '2,95p' "$0"; exit 0 ;;
		*) echo "unknown argument: $1" >&2; exit 2 ;;
	esac
done

# ------------------------------------------------------------ resolve suite --
# A suite supplies a query directory and a warm-up query. Both can be overridden:
# --query-dir replaces the directory, --warmup-query replaces the warm-up. The
# query list itself is resolved further down, once the directory is known.
case "$SUITE" in
	tpch)
		SUITE_DIR="$REPO/extension/tpch/dbgen/queries"
		# q01: a full lineitem scan+aggregate, so the warm-up also touches the
		# biggest table. It must be a query that compiles in every mode -- a bare
		# count(*) is no good, its optimized plan collapses to a DUMMY_SCAN the
		# MLIR translation cannot handle.
		SUITE_WARMUP="q01" ;;
	micro)
		SUITE_DIR="$REPO/microbench/queries"
		# agg1: sum over lineitem, the largest table in the suite, for the same reason.
		SUITE_WARMUP="agg1" ;;
	presentation)
		# The queries transcribed from the Nov 2025 project presentation, kept so the
		# numbers reported there stay reproducible.
		SUITE_DIR="$REPO/microbench/presentation"
		SUITE_WARMUP="a01" ;;
	final)
		# The curated set: seven groups of three, plus TPC-H Q1 as an anchor.
		SUITE_DIR="$REPO/microbench/final"
		# agg1: a full lineitem scan, so the warm-up touches the biggest table, and it
		# compiles in every mode.
		SUITE_WARMUP="agg1" ;;
	test)
		# Scratch suite: multi-column GROUP BY and the sort queries, kept apart from the
		# curated micro suite so it can be churned freely.
		SUITE_DIR="$REPO/microbench/test"
		SUITE_WARMUP="mgrp1" ;;
	count)
		# The row-count queries, split out of the presentation set.
		SUITE_DIR="$REPO/microbench/count"
		SUITE_WARMUP="n01" ;;
	exploratory)
		# The first-pass probes behind the Aug 2026 sweeps. Includes joins, so these
		# are not single-table like the micro suite.
		SUITE_DIR="$REPO/microbench/exploratory"
		SUITE_WARMUP="c01" ;;
	*)
		# An unknown suite name is only an error if no --query-dir was given.
		SUITE_DIR=""
		SUITE_WARMUP="" ;;
esac
[ -z "$QUERY_DIR" ]    && QUERY_DIR="$SUITE_DIR"
[ "$LINGO_QUERY_DIR_GIVEN" -eq 0 ] && LINGO_QUERY_DIR="$REPO/lingodb_queries/$SUITE"
[ -z "$WARMUP_QUERY" ] && WARMUP_QUERY="$SUITE_WARMUP"
if [ -z "$QUERY_DIR" ]; then
	echo "ERROR: unknown suite '$SUITE' and no --query-dir given." >&2
	echo "       known suites: tpch, micro" >&2
	exit 2
fi

# Forward the LingoDB execution mode to the compiled path. GlobalSetting reads this
# at process start (static init), so it must be exported before the binary launches.
export LINGODB_EXECUTION_MODE="$EXEC_MODE"
# Index nested loop joins, also read at static init. Unset when allowed, so an
# inherited "false" cannot leak into a run that did not ask for it.
if [ "$INLJ" -eq 0 ]; then
	export LINGODB_OPT_INDEX_NESTED_LOOP=false
else
	unset LINGODB_OPT_INDEX_NESTED_LOOP
fi
# Per-stage IR dumps. Like the execution mode, this is read at static init, so it has
# to be exported before the binary launches. Unset rather than set to 0 when off, to
# keep an inherited value from leaking into the run.
if [ "$DUMP_IR" -eq 1 ]; then
	export LINGODB_DUMP_IR=1
else
	unset LINGODB_DUMP_IR
fi
if [ "$TRACK_TUPLES" -eq 1 ]; then
	export LINGODB_TRACK_TUPLES=1
else
	unset LINGODB_TRACK_TUPLES
fi
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

# Mode L talks to the LingoDB shell rather than to duckdb. REPORT_TIMES makes it print
# "compilation: X [ms] execution: Y [ms]" on stderr after every statement, which is the
# only timing it offers; PROMPT=0 keeps the interactive prompt out of the output.
export LINGODB_SQL_REPORT_TIMES=1
export LINGODB_SQL_PROMPT=0

# The full mode list for one iteration: the numeric modes, then LingoDB native. L is
# appended only when --modes did not already name it, so it never runs twice.
RUN_MODES="$MODES"
case " $RUN_MODES " in
	*" L "*) ;;
	*) [ "$LINGODB_NATIVE" -eq 1 ] && RUN_MODES="$RUN_MODES L" ;;
esac

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
if [ "$LINGODB_NATIVE" -eq 1 ] && [ ! -x "$LINGODB_SQL" ]; then
	echo "ERROR: LingoDB sql shell not found or not executable: $LINGODB_SQL" >&2
	echo "       pass --lingodb-sql PATH, or --no-lingodb-native to skip that mode." >&2
	exit 1
fi
if [ ! -d "$QUERY_DIR" ]; then
	echo "ERROR: TPC-H query directory not found: $QUERY_DIR" >&2
	exit 1
fi

# Default query list. TPC-H is the fixed 22; any other suite is every .sql file in
# the directory, sorted, so adding a query is a matter of adding a file.
if [ -z "$QUERIES" ]; then
	# The fixed 22 only when the tpch suite is running out of its own directory;
	# --query-dir points somewhere else, so the names there are what count.
	if [ "$SUITE" = "tpch" ] && [ "$QUERY_DIR_GIVEN" -eq 0 ]; then
		for n in $(seq -w 1 22); do QUERIES="$QUERIES q$n"; done
		QUERIES="$(echo $QUERIES)"
	else
		for f in "$QUERY_DIR"/*.sql; do
			[ -e "$f" ] || continue
			QUERIES="$QUERIES $(basename "$f" .sql)"
		done
		QUERIES="$(echo $QUERIES)"
		if [ -z "$QUERIES" ]; then
			echo "ERROR: no .sql files found in $QUERY_DIR" >&2
			exit 2
		fi
	fi
fi

if [ "$WARMUP" -eq 1 ] && [ ! -f "$QUERY_DIR/$WARMUP_QUERY.sql" ]; then
	echo "ERROR: warm-up query not found: $QUERY_DIR/$WARMUP_QUERY.sql" >&2
	echo "       pass --warmup-query NAME or --no-warmup." >&2
	exit 2
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
LOG="$OUT_DIR/${SUITE}_bymode_$STAMP.log"
CSV="$OUT_DIR/${SUITE}_bymode_$STAMP.csv"
PROFILE_DIR="$OUT_DIR/${SUITE}_bymode_${STAMP}_duckdb_profile"
[ "$DUCKDB_PROFILE" -eq 1 ] && [ "$DRY_RUN" -eq 0 ] && mkdir -p "$PROFILE_DIR"
# Header note for --duckdb-profile when mode 0 is not among the modes being run. A
# function rather than an inline case, since bash 3.2 misparses a case pattern's ')'
# inside $(...).
profile_note() {
	case " $RUN_MODES " in
		*" 0 "*) ;;
		*) printf '; mode 0 not selected, nothing profiled' ;;
	esac
}
STDIN_FILE="$OUT_DIR/.stdin_$$"
RAW="$OUT_DIR/.raw_$$"
MODEOUT="$OUT_DIR/.modeout_$$"
PARSED="$OUT_DIR/.parsed_$$"
ATTEMPT="$OUT_DIR/.attempt_$$"
SCHED="$OUT_DIR/.sched_$$"
FAILEDF="$OUT_DIR/.failed_$$"
MODEERR="$OUT_DIR/.modeerr_$$"
NAMEF="$OUT_DIR/.names_$$"

mode_name() {
	case "$1" in
		0) echo "native" ;;
		1) echo "unoptimized" ;;
		2) echo "optimized" ;;
		L) echo "lingodb_native" ;;
		*) echo "mode$1" ;;
	esac
}

# The queries that get a row in the report. The warm-up is included so that its
# residual (the one-time session load) stays visible; the re-warm-up that every
# retry performs is deliberately absent, since its numbers describe a reload
# rather than a query.
run_list() {
	[ "$WARMUP" -eq 1 ] && printf '%s ' "__warmup__"
	printf '%s ' $QUERIES
}

# Write the run schedule: one entry per line, in execution order. An entry is a
# query name, or the sentinel __cooldown__ for the pause between passes.
#
# The schedule is a file rather than a loop inside the stdin builder because a
# retry has to resume from part-way through it: on a crash the harness drops the
# offending query and rebuilds the stdin from whatever is left.
build_schedule() {
	local dest="$1" failed_file="$2" q bad=""
	[ -f "$failed_file" ] && bad="$(cat "$failed_file")"
	: > "$dest"
	for q in $QUERIES; do
		# A query that already took this mode down is left out of later iterations;
		# it would only walk into the same wall again.
		case " $bad " in *" $q "*) continue ;; esac
		echo "$q" >> "$dest"
	done
	return 0
}

# How many queries a schedule still holds. awk, not `grep -c || echo 0`: that
# prints two numbers on an empty file.
sched_len() { awk 'END { print NR + 0 }' "$1" 2>/dev/null || echo 0; }

# The Nth query of a schedule.
sched_nth() { sed -n "${2}p" "$1"; }

# Drop everything up to and including the Nth query, then drop every occurrence of
# the named queries from what remains. This turns a crashed run into the next attempt.
sched_resume() {
	awk -v cut="$2" -v bad="$3" '
		BEGIN { n = split(bad, F, " "); for (i = 1; i <= n; i++) BAD[F[i]] = 1 }
		NR > cut && !($0 in BAD) { print }
	' "$1"
}

# Build the single stdin stream for one attempt at one mode:
#   line 1     : mode digit, consumed by readCompileConfig()'s scanf
#   then       : session setup, the warm-up, then each scheduled query preceded by a
#                ".print" marker so the combined output can be split back into
#                per-query segments.
# The warm-up marker is a parameter: a retry uses a different one so its timings
# are parsed and then discarded rather than mixed into the query's own numbers.
build_mode_stdin() {
	local mode="$1" dest="$2" sched="$3" wmark="$4" iter="${5:-1}"
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
		# Perfect-hash aggregate budget (mode 0). A SET is not a SELECT, so like the PRAGMA
		# above it bypasses runMLIR and is a harmless no-op under the compiled modes.
		case "$PERFECT_HT" in '' | *[!0-9]*) ;; *) echo "SET perfect_ht_threshold=$PERFECT_HT;" ;; esac
		if [ "$WARMUP" -eq 1 ]; then
			# Every attempt warms up, because a retry is a brand new process and pays the
			# LingoDB session load again from scratch.
			echo ".print ===QUERY $wmark==="
			local wfile="$QUERY_DIR/$WARMUP_QUERY.sql"
			if [ -f "$wfile" ]; then
				cat "$wfile"; echo ""
			else
				echo "SELECT 'missing warm-up query file: $WARMUP_QUERY';"
			fi
		fi
		local q qfile
		while IFS= read -r q; do
			[ -z "$q" ] && continue
			qfile="$QUERY_DIR/$q.sql"
			echo ".print ===QUERY $q==="
			# The output file is set before profiling is switched on, so that no statement
			# is ever profiled with no file to write to, which would print the profile into
			# the output this script parses. Each query overwrites whatever a preceding
			# PRAGMA left in its file.
			if [ "$DUCKDB_PROFILE" -eq 1 ] && [ "$mode" = "0" ]; then
				echo "PRAGMA profiling_output='$PROFILE_DIR/${q}_iter${iter}.json';"
				echo "PRAGMA enable_profiling='json';"
			fi
			if [ -f "$qfile" ]; then
				cat "$qfile"; echo ""
			else
				echo "SELECT 'missing query file: $q';"
			fi
		done < "$sched"
		echo ".quit"
	} > "$dest"
}

# Append one query so the LingoDB shell actually executes it: that shell runs a
# statement only when a line ends in ';', and the query files are not all terminated.
# Appending one unconditionally would leave an empty statement behind, and an empty
# statement gets its own timing line, which breaks the pairing the parser relies on.
emit_sql() {
	local f="$1" last
	cat "$f"
	last="$(tr -d '[:space:]' < "$f" | tail -c 1)"
	[ "$last" = ";" ] || echo ";"
	echo
}

# stdin for one LingoDB-native attempt. There are no dot commands in that shell, so
# the per-query marker has to be a SQL statement of its own.
# The file mode L reads for one query: the override if there is one, else QUERY_DIR.
lingodb_sql_file() {
	if [ -n "$LINGO_QUERY_DIR" ] && [ -f "$LINGO_QUERY_DIR/$1.sql" ]; then
		echo "$LINGO_QUERY_DIR/$1.sql"
	else
		echo "$QUERY_DIR/$1.sql"
	fi
}

build_lingodb_stdin() {
	local dest="$1" sched="$2" wmark="$3" q
	{
		if [ "$WARMUP" -eq 1 ]; then
			echo "select '===QUERY $wmark===';"
			emit_sql "$(lingodb_sql_file "$WARMUP_QUERY")"
		fi
		while IFS= read -r q; do
			[ -z "$q" ] && continue
			echo "select '===QUERY $q===';"
			emit_sql "$(lingodb_sql_file "$q")"
		done < "$sched"
	} > "$dest"
}

# The names of the statements fed to a LingoDB attempt, in order, one per line. The
# parser pairs these with the timing lines positionally.
build_name_list() {
	local sched="$1" wmark="$2"
	[ "$WARMUP" -eq 1 ] && echo "$wmark"
	cat "$sched"
}

# Split a LingoDB session's stderr into per-query rows. The shell prints one timing
# line per statement, and every query is preceded by its marker statement, so the
# lines arrive in pairs: marker, query, marker, query. The second of each pair is the
# query. Emits the same seven columns as parse_mode_output, with the total in the
# residual column and nothing in the DuckDB-only ones.
parse_lingodb_output() {
	awk '
		NR == FNR { name[++nn] = $0; next }
		/compilation:/ {
			c = ""; e = ""
			for (i = 1; i <= NF; i++) {
				if ($i == "compilation:") c = $(i + 1)
				if ($i == "execution:")   e = $(i + 1)
			}
			if (c == "" || e == "") next
			k++
			if (k % 2 == 0) {                       # even = the query, odd = its marker
				q = k / 2
				# The total goes in the real column too, converted to seconds to match the
				# unit the other modes report there, so one column holds end-to-end time for
				# every mode and a plot needs no per-mode special case. More decimals than
				# duckdb prints, because this is derived rather than read off a 1ms timer.
				# The LingoDB shell sums every non-execution phase it TIMES, and the SQL
				# frontend is not timed (Frontend.cpp SQLFrontend::loadFromString writes
				# no timing entry), so this total excludes the mode L parsing and module
				# construction. It is understated by that amount. NOTE: no apostrophes
				# in this block, it sits inside a single-quoted awk program.
				if (q <= nn) printf "%s\t%.6f\t%s\t%s\t\t%.3f\t\n", name[q], (c + e) / 1000, c, e, c + e
			}
		}
	' "$2" "$1"
}

# Split one mode's combined stdout+stderr into per-query rows. The ".print" markers
# delimit segments; timing lines that fall inside a segment belong to that query.
# BSD-awk compatible (no gawk match(...,arr)). Emits TSV: query real comp exec runmlir
parse_mode_output() {
	awk '
		function emit(   resid) {
			if (curq == "") return
			# residual: what runMLIR spent outside compile and execute (executer setup,
			# scheduling, result readback). Computed per run here so the aggregation only
			# has to take a minimum.
			resid = ""
			if (runmlir != "" && comp != "" && exec != "") resid = runmlir - comp - exec
			printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", curq, real, comp, exec, runmlir, resid, plan
		}
		index($0, "===QUERY") > 0 {
			emit()
			line = $0
			sub(/.*===QUERY /, "", line)
			sub(/===.*/, "", line)
			curq = line; real = ""; comp = ""; exec = ""; runmlir = ""; plan = ""
			next
		}
		index($0, "compilation:") > 0 && index($0, "execution:") > 0 { comp = $2; exec = $5; next }
		index($0, "runmlir_total:") > 0 { runmlir = $3; next }
		# Everything DuckDB spent before the LingoDB backend saw the query. Mode 1 reports
		# it as one figure because the MLIR is built inside the planner; mode 2 reports the
		# planner and the MLIR build separately, so they are summed.
		index($0, "duckdb_plan+mlir_build:") > 0 { plan = $3; next }
		index($0, "duckdb_plan:") > 0 { plan = (plan == "" ? 0 : plan) + $3; next }
		index($0, "mlir_build:") > 0 { plan = (plan == "" ? 0 : plan) + $3; next }
		/Run Time/ { for (i = 1; i <= NF; i++) if ($i == "real") real = $(i + 1); next }
		END { emit() }
	' "$1"
}

# Reduce the N parsed rows for a single query (piped in as TSV: query real comp exec
# runmlir residual plan) to one line of aggregates. Min is the headline (noise is
# one-sided); median of execution is also reported. BSD-awk compatible (manual
# insertion-sort for the median). Emits TSV:
#   n  real_min  comp_min  exec_min  exec_med  runmlir_min  residual_min  plan_min
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
			if ($6 != "") D[++nd] = $6
			if ($7 != "") P[++np] = $7
		}
		END {
			printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", rows, mn(R, nr), mn(C, nc), mn(E, ne),
				(ne ? med(E, ne) : ""), mn(M, nm), mn(D, nd), mn(P, np)
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
	echo "lingodb  : $LINGODB_DIR (modes 1, 2 and L)"
	echo "suite    : $SUITE ($QUERY_DIR)"
	[ -d "$LINGO_QUERY_DIR" ] && echo "modeLsql : $LINGO_QUERY_DIR (overrides, mode L only)"
	echo "queries  : $QUERIES"
	echo "skip     : ${SKIP:-(none)}"
	echo "modes    : $RUN_MODES"
	echo "warm-up  : $([ "$WARMUP" -eq 1 ] && echo "on ($WARMUP_QUERY.sql)" || echo off)"
	echo "exec_mode: $EXEC_MODE (LINGODB_EXECUTION_MODE; modes 1, 2 and L)"
	echo "inlj     : $([ "$INLJ" -eq 1 ] && echo "allowed (LingoDB default)" || echo "off (LINGODB_OPT_INDEX_NESTED_LOOP=false; hash joins only)")"
	[ "$DUMP_IR" -eq 1 ] && echo "dump_ir  : on (LINGODB_DUMP_IR; per-stage modules, TIMINGS NOT COMPARABLE)"
	[ "$TRACK_TUPLES" -eq 1 ] && echo "trktuples: on (LINGODB_TRACK_TUPLES; per-operator counts, TIMINGS NOT COMPARABLE)"
	[ "$DUCKDB_PROFILE" -eq 1 ] && echo "ddbprof  : on (mode 0 JSON profiles in $PROFILE_DIR, TIMINGS NOT COMPARABLE)$(profile_note)"
	echo "ldb shell: $([ "$LINGODB_NATIVE" -eq 1 ] && echo "$LINGODB_SQL" || echo "(mode L disabled)")"
	echo "loading  : $([ "$EAGER" -eq 1 ] && echo "eager (all tables at warm-up)" || echo "lazy (per first-touch)")"
	echo "cores    : ${CORES:-all} (both engines: DuckDB PRAGMA threads + LingoDB workers)"
	echo "parallel : lingodb=${LINGO_WORKERS:-all} (LINGODB_PARALLELISM), duckdb=${CORES:-all} (PRAGMA threads)"
	echo "perfectht: ${PERFECT_HT:-default 12} (DuckDB perfect_ht_threshold, bits; mode 0 only)"
	echo "repeats  : $REPEATS iteration(s); order $(for i in $(seq 1 $REPEATS); do for mm in $RUN_MODES; do printf '%s ' "$mm"; done; done)"
	echo "cooldown : ${COOLDOWN}s after every mode process"
	echo "timeout  : ${TIMEOUT}s per QUERY (watchdog on progress markers)"
	echo "retries  : up to $MAX_RETRIES attempt(s) per mode; a crashing query is dropped"
	echo "log      : $OUT_DIR/${SUITE}_bymode_<stamp>.log"
	echo "csv      : $OUT_DIR/${SUITE}_bymode_<stamp>.csv"
	echo "invocations: $(( $(echo $RUN_MODES | wc -w | tr -d ' ') * REPEATS )) (one per iteration+mode, plus one per retry)"
	echo
	echo "per-process stdin (order): mode-digit, setup, [warm-up], then $(echo $QUERIES | wc -w | tr -d ' ') queries, .quit"
	exit 0
fi

# ------------------------------------------------------------------- run it --
cleanup() { rm -f "$STDIN_FILE" "$RAW" "$MODEOUT" "$ATTEMPT" "$SCHED" "$SCHED.next" "$PARSED" "$PARSED".* "$FAILEDF".* "$MODEERR" "$NAMEF"; }
trap cleanup EXIT

: > "$RAW"

{
	echo "$SUITE sweep (iteration-major, fresh process per iteration+mode)"
	echo "started  : $(date)"
	echo "binary   : $DUCKDB"
	echo "database : $DB"
	echo "lingodb  : $LINGODB_DIR (modes 1, 2 and L)"
	echo "suite    : $SUITE ($QUERY_DIR)"
	[ -d "$LINGO_QUERY_DIR" ] && echo "modeLsql : $LINGO_QUERY_DIR (overrides, mode L only)"
	echo "queries  : $QUERIES"
	echo "skip     : ${SKIP:-(none)}"
	echo "modes    : $RUN_MODES (0=native 1=unoptimized 2=optimized L=lingodb_native)"
	echo "warm-up  : $([ "$WARMUP" -eq 1 ] && echo "on ($WARMUP_QUERY.sql)" || echo off)"
	echo "exec_mode: $EXEC_MODE (LINGODB_EXECUTION_MODE; modes 1, 2 and L)"
	echo "inlj     : $([ "$INLJ" -eq 1 ] && echo "allowed (LingoDB default)" || echo "off (LINGODB_OPT_INDEX_NESTED_LOOP=false; hash joins only)")"
	[ "$DUMP_IR" -eq 1 ] && echo "dump_ir  : on (LINGODB_DUMP_IR; per-stage modules, TIMINGS NOT COMPARABLE)"
	[ "$TRACK_TUPLES" -eq 1 ] && echo "trktuples: on (LINGODB_TRACK_TUPLES; per-operator counts, TIMINGS NOT COMPARABLE)"
	[ "$DUCKDB_PROFILE" -eq 1 ] && echo "ddbprof  : on (mode 0 JSON profiles in $PROFILE_DIR, TIMINGS NOT COMPARABLE)$(profile_note)"
	echo "ldb shell: $([ "$LINGODB_NATIVE" -eq 1 ] && echo "$LINGODB_SQL" || echo "(mode L disabled)")"
	echo "loading  : $([ "$EAGER" -eq 1 ] && echo "eager (all tables at warm-up)" || echo "lazy (per first-touch)")"
	echo "cores    : ${CORES:-all} (both engines: DuckDB PRAGMA threads + LingoDB workers)"
	echo "parallel : lingodb=${LINGO_WORKERS:-all} (LINGODB_PARALLELISM), duckdb=${CORES:-all} (PRAGMA threads)"
	echo "perfectht: ${PERFECT_HT:-default 12} (DuckDB perfect_ht_threshold, bits; mode 0 only)"
	echo "repeats  : $REPEATS iteration(s); order $(for i in $(seq 1 $REPEATS); do for mm in $RUN_MODES; do printf '%s ' "$mm"; done; done)"
	echo "cooldown : ${COOLDOWN}s after every mode process"
	echo "timeout  : ${TIMEOUT}s per QUERY (watchdog on progress markers)"
	echo "retries  : up to $MAX_RETRIES attempt(s) per mode; a crashing query is dropped"
	echo "git      : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo n/a)"
} | tee "$LOG"

total=0; ok=0; failed=0; skipped=0

# Per-mode accumulators. Rows from every iteration of a mode land in one file, so the
# min/median at the end is taken across iterations.
for m in $RUN_MODES; do
	: > "$PARSED.$m"
	: > "$FAILEDF.$m"
	eval "wall_$m=0; attempts_$m=0"
done

# ------------------------------------------------------------ iteration-major --
# The outer loop is the iteration and the inner loop is the mode, so the order is
# m0 m1 m2 m0 m1 m2 rather than m0 m0 m1 m1 m2 m2. Every (iteration, mode) pair is a
# brand new process. Both properties are about keeping caches out of the result: a
# fresh process starts with an empty DuckDB buffer pool and no LingoDB session, and
# interleaving the modes means a mode is never measured immediately after itself.
for it in $(seq 1 "$REPEATS"); do
	for m in $RUN_MODES; do
		mname="$(mode_name "$m")"
		{
			echo
			echo "##############################################################"
			echo "## ITERATION $it/$REPEATS  MODE $m ($mname) -- fresh process"
			echo "##############################################################"
		} | tee -a "$LOG"

		build_schedule "$SCHED" "$FAILEDF.$m"
		if [ "$(sched_len "$SCHED")" -eq 0 ]; then
			echo "--- iteration $it mode $m: every query has already failed for this mode; skipping" | tee -a "$LOG"
			continue
		fi

		mode_failed="$(cat "$FAILEDF.$m")"
		attempt=1

		# One attempt per surviving crash. A crash kills the whole mode process, so the
		# only way to reach the queries behind the offender is to start over without it.
		# Everything already measured is kept: parsed rows accumulate.
		while [ "$(sched_len "$SCHED")" -gt 0 ]; do
			if [ "$attempt" -eq 1 ]; then
				# Every iteration reports its own warm-up: with a fresh process each time,
				# the session load is paid again, and its residual is worth seeing per run.
				wmark="__warmup__"
			else
				# A retry within an iteration reloads the session again. Parsed, then dropped.
				wmark="__rewarmup__"
				echo "--- iteration $it mode $m: attempt $attempt, $(sched_len "$SCHED") query run(s) left, warming up again (metrics discarded)" | tee -a "$LOG"
			fi

			if [ "$m" = "L" ]; then
				build_lingodb_stdin "$STDIN_FILE" "$SCHED" "$wmark"
			else
				build_mode_stdin "$m" "$STDIN_FILE" "$SCHED" "$wmark" "$it"
			fi

			start_ns=$(date +%s)
			# --timeout is a per-QUERY budget, but a whole mode shares one process, so it
			# cannot be a single alarm on that process. Instead the process runs in the
			# background and a watchdog watches the ".print" markers: each new marker means
			# the previous query finished, so the time since the last marker is the time the
			# current query has been running. Exceeding the budget kills the process, which
			# the retry logic below then treats like any other crash. duckdb flushes a
			# .print immediately even when stdout is a file, so the markers are a live signal
			# rather than something that only appears at exit.
			: > "$MODEOUT"; : > "$MODEERR"
			if [ "$m" = "L" ]; then
				# The LingoDB shell puts its timings on stderr, so the two streams are kept
				# apart here rather than merged as they are for duckdb.
				"$LINGODB_SQL" "$LINGODB_DIR" < "$STDIN_FILE" > "$MODEOUT" 2> "$MODEERR" &
			else
				"$DUCKDB" "$DB" "$LINGODB_DIR" < "$STDIN_FILE" > "$MODEOUT" 2>&1 &
			fi
			dpid=$!
			timed_out=0; seen=0; slow_q=""
			last_tick=$(date +%s)
			while kill -0 "$dpid" 2>/dev/null; do
				sleep 1
				now=$(date +%s)
				# Progress signal: for duckdb the ".print" markers on stdout; for LingoDB the
				# per-statement timing lines on stderr, which arrive two per query.
				if [ "$m" = "L" ]; then
					marks="$(awk '/compilation:/{ n++ } END { print n + 0 }' "$MODEERR" 2>/dev/null || echo 0)"
				else
					marks="$(awk '/===QUERY /{ n++ } END { print n + 0 }' "$MODEOUT" 2>/dev/null || echo 0)"
				fi
				if [ "$marks" -ne "$seen" ]; then
					seen="$marks"; last_tick="$now"
				elif [ $(( now - last_tick )) -ge "$TIMEOUT" ]; then
					slow_q="$(sed -n 's/.*===QUERY \(.*\)===.*/\1/p' "$MODEOUT" | tail -1)"
					slow_q="${slow_q%%\'*}"
					timed_out=1
					kill -9 "$dpid" 2>/dev/null
					break
				fi
			done
			wait "$dpid" 2>/dev/null; prc=$?
			# Report a watchdog kill as a timeout rather than as signal 9.
			[ "$timed_out" -eq 1 ] && prc=142
			attempt_wall=$(( $(date +%s) - start_ns ))
			eval "wall_$m=\$(( wall_$m + attempt_wall )); attempts_$m=\$(( attempts_$m + 1 ))"

			if [ "$m" = "L" ]; then
				cat "$MODEOUT" "$MODEERR" >> "$LOG"
				build_name_list "$SCHED" "$wmark" > "$NAMEF"
				parse_lingodb_output "$MODEERR" "$NAMEF" > "$ATTEMPT"
			else
				cat "$MODEOUT" >> "$LOG"
				parse_mode_output "$MODEOUT" > "$ATTEMPT"
			fi
			# Keep every row except the re-warm-up, which describes a session reload.
			grep -v '^__rewarmup__	' "$ATTEMPT" >> "$PARSED.$m" || true

			{
				echo "--- iteration $it mode $m ($mname) attempt $attempt: exit=$prc  wall=${attempt_wall}s"
				if [ "$prc" -eq 142 ]; then echo "--- WARNING: query '${slow_q:-?}' ran longer than the ${TIMEOUT}s per-query budget; process killed"; fi
				if [ "$prc" -gt 128 ] && [ "$prc" -ne 142 ]; then echo "--- WARNING: attempt CRASHED (sig $((prc - 128)))"; fi
			} | tee -a "$LOG"

			# How far the attempt got, counted from the parsed rows so it cannot disagree
			# with what was recorded.
			reached="$(awk -F'\t' '$1 != "__warmup__" && $1 != "__rewarmup__" { n++ } END { print n + 0 }' "$ATTEMPT" 2>/dev/null || echo 0)"
			pending="$(sched_len "$SCHED")"

			if [ "$prc" -eq 0 ] && [ "$reached" -ge "$pending" ]; then
				break                   # the whole remaining schedule ran
			fi

			if [ "$reached" -eq 0 ]; then
				echo "--- iteration $it mode $m: attempt produced nothing; abandoning this mode for this iteration" | tee -a "$LOG"
				break
			fi

			# Blame a query. The marker is printed before its SQL runs, so the query that
			# took the process down is normally the last one with a row but no timing.
			last_row="$(grep -v '^__warmup__	\|^__rewarmup__	' "$ATTEMPT" | tail -1)"
			last_real="$(printf '%s' "$last_row" | cut -f2)"
			last_runmlir="$(printf '%s' "$last_row" | cut -f5)"
			last_exec="$(printf '%s' "$last_row" | cut -f4)"
			case "$m" in
				0) last_done="$last_real" ;;
				L) last_done="$last_exec" ;;
				*) last_done="$last_runmlir" ;;
			esac

			if [ -n "$last_done" ]; then
				# The last query finished and the process still died, so it went down between
				# queries. The next one never printed a marker; blame it, since there is
				# nothing else to blame and leaving it in would crash the retry too.
				cut_at="$reached"; blame_idx=$(( reached + 1 ))
				presumed=" (presumed; the process died between queries)"
			else
				cut_at="$reached"; blame_idx="$reached"; presumed=""
			fi

			if [ "$blame_idx" -gt "$pending" ]; then
				break                   # nothing left to blame; the death was at .quit
			fi
			badq="$(sched_nth "$SCHED" "$blame_idx")"
			if [ -z "$badq" ]; then
				echo "--- iteration $it mode $m: could not identify the failing query; abandoning this mode for this iteration" | tee -a "$LOG"
				break
			fi

			case " $mode_failed " in *" $badq "*) ;; *) mode_failed="$mode_failed $badq" ;; esac
			# Remembered for every later iteration of this mode, not just this attempt.
			echo "$mode_failed" > "$FAILEDF.$m"
			echo "--- iteration $it mode $m: '$badq' took the process down$presumed; skipping it and restarting" | tee -a "$LOG"

			attempt=$(( attempt + 1 ))
			if [ "$attempt" -gt "$MAX_RETRIES" ]; then
				echo "--- iteration $it mode $m: --max-retries ($MAX_RETRIES) reached; remaining queries not run" | tee -a "$LOG"
				break
			fi

			sched_resume "$SCHED" "$cut_at" "$mode_failed" > "$SCHED.next"
			mv "$SCHED.next" "$SCHED"
		done

		# Settle between processes: this now falls between every mode of every
		# iteration, which is also the boundary between two measurements of a query.
		if [ "$COOLDOWN" -gt 0 ]; then
			echo "--- cooling down ${COOLDOWN}s" | tee -a "$LOG"
			sleep "$COOLDOWN"
		fi
	done
done

# ------------------------------------------------------------------- report --
# One row per (query, mode), aggregated over the iterations of that mode.
for m in $RUN_MODES; do
	mname="$(mode_name "$m")"
	eval "mw=\${wall_$m}; ma=\${attempts_$m}"
	mode_failed="$(cat "$FAILEDF.$m")"
	{
		echo
		echo "=== MODE $m ($mname): $ma process(es) over $REPEATS iteration(s), wall=${mw}s, skipped:${mode_failed:- none}"
	} | tee -a "$LOG"

	# The warm-up row is the one-time session load, now paid once per iteration.
	if [ "$WARMUP" -eq 1 ]; then
		wline="$(grep -E '^__warmup__	' "$PARSED.$m" | head -1 || true)"
		if [ -n "$wline" ]; then
			wrun="$(echo "$wline" | cut -f5)"; wcomp="$(echo "$wline" | cut -f3)"; wexec="$(echo "$wline" | cut -f4)"
			wload="$(awk -v r="$wrun" -v c="$wcomp" -v e="$wexec" 'BEGIN{ if(r!=""&&c!=""&&e!="") printf "%.3f", r-c-e; else print "" }')"
			echo "--- mode $m warm-up: session_load≈${wload:-?}ms (runmlir_total=${wrun:-?} comp=${wcomp:-?} exec=${wexec:-?})" | tee -a "$LOG"
		fi
	fi

	geo_vals=""
	for q in $(run_list); do
		total=$((total + 1))
		rows="$(grep -E "^$q	" "$PARSED.$m" || true)"
		agg="$(printf '%s\n' "$rows" | aggregate_query)"
		n="$(echo "$agg" | cut -f1)"
		real="$(echo "$agg" | cut -f2)"; comp="$(echo "$agg" | cut -f3)"
		exec="$(echo "$agg" | cut -f4)"; exec_med="$(echo "$agg" | cut -f5)"
		runmlir="$(echo "$agg" | cut -f6)"; residual="$(echo "$agg" | cut -f7)"
		plan="$(echo "$agg" | cut -f8)"
		case " $mode_failed " in *" $q "*) was_skipped=1 ;; *) was_skipped=0 ;; esac
		if [ "$was_skipped" -eq 1 ]; then
			status="FAILED"; n=0
			skipped=$((skipped + 1))
		elif [ -z "$rows" ] || [ "${n:-0}" -eq 0 ]; then
			status="INCOMPLETE"; n=0
			failed=$((failed + 1))
		else
			# Completion criterion: native DuckDB needs its Run Time; the compiled modes
			# need runmlir_total; LingoDB-native only ever reports compile and execute.
			case "$m" in
				0) [ -n "$real" ] && status="OK" || status="INCOMPLETE" ;;
				L) [ -n "$exec" ] && status="OK" || status="INCOMPLETE" ;;
				*) [ -n "$runmlir" ] && status="OK" || status="INCOMPLETE" ;;
			esac
			[ "$status" = "OK" ] && ok=$((ok + 1)) || failed=$((failed + 1))
			if [ "$status" = "OK" ] && [ "$q" != "__warmup__" ] && [ -n "$exec" ]; then
				geo_vals="$geo_vals $exec"
			fi
		fi
		printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
			"$q" "$m" "$mname" "$status" "$n" \
			"${real:-}" "${comp:-}" "${exec:-}" "${exec_med:-}" "${runmlir:-}" "${residual:-}" "${plan:-}" >> "$RAW"
		echo "--- $q mode $m ($mname): $status n=$n  real=${real:--}s comp=${comp:--}ms exec_min=${exec:--}ms exec_med=${exec_med:--}ms runmlir=${runmlir:--}ms residual=${residual:--}ms" | tee -a "$LOG"
	done

	if [ "$m" != "0" ]; then
		gm="$(printf '%s' "$geo_vals" | geomean)"
		echo "--- mode $m ($mname): geomean(execution_ms_min) = ${gm:-n/a} ms over $(echo $geo_vals | wc -w | tr -d ' ') queries" | tee -a "$LOG"
	fi
done

# --------------------------------------------------------- emit CSV in order --
# Sort by query name, then by mode, so each query's rows are adjacent and in mode
# order (warm-up sorts to the top). The mode sort is lexical rather than numeric
# because mode L is not a number: numerically it would collate as 0 and land in front
# of mode 1, while lexically 0 1 2 L is exactly the order the modes ran in.
{
	# Numeric columns are the MIN across the query's repeats (== the single value when
	# --repeats 1); execution_ms_median is the median of execution across repeats; n is the
	# number of completed runs.
	echo "query,mode,mode_name,status,n,duckdb_real_s,compilation_ms,execution_ms,execution_ms_median,runmlir_total_ms,residual_ms,plan_ms"
	sort -t',' -k1,1 -k2,2 "$RAW"
} > "$CSV"

{
	echo
	echo "=============================================================="
	echo "DONE  $(date)"
	echo "rows: $total   ok: $ok   failed: $skipped   incomplete: $failed"
	echo "full log : $LOG"
	echo "csv      : $CSV"
	echo "=============================================================="
	echo
	column -t -s',' "$CSV" 2>/dev/null || cat "$CSV"
} | tee -a "$LOG"
