#!/bin/bash
# filepath: run_multiple_queries.sh

# Output file
OUTPUT_FILE="bench_results.txt"

# Clear/create output file
> "$OUTPUT_FILE"

# Array of queries
QUERIES=(
"SELECT COUNT(*) FROM 'bench_db/lineitem.arrow';"
"SELECT COUNT(*) FROM 'bench_db/lineitem.arrow';"
"SELECT COUNT(*) FROM 'bench_db/lineitem.arrow';"
"SELECT COUNT(*) FROM 'bench_db/orders.arrow';"
"SELECT COUNT(*) FROM 'bench_db/orders.arrow';"
"SELECT COUNT(*) FROM 'bench_db/orders.arrow';"
"SELECT COUNT(*) FROM 'bench_db/customer.arrow';"
"SELECT COUNT(*) FROM 'bench_db/customer.arrow';"
"SELECT COUNT(*) FROM 'bench_db/customer.arrow';"
"SELECT COUNT(*) FROM 'bench_db/partsupp.arrow';"
"SELECT COUNT(*) FROM 'bench_db/partsupp.arrow';"
"SELECT COUNT(*) FROM 'bench_db/partsupp.arrow';"
"SELECT COUNT(*) FROM 'bench_db/part.arrow';"
"SELECT COUNT(*) FROM 'bench_db/part.arrow';"
"SELECT COUNT(*) FROM 'bench_db/part.arrow';"
"SELECT COUNT(*) FROM 'bench_db/supplier.arrow';"
"SELECT COUNT(*) FROM 'bench_db/supplier.arrow';"
"SELECT COUNT(*) FROM 'bench_db/supplier.arrow';"
"SELECT COUNT(*) FROM 'bench_db/nation.arrow';"
"SELECT COUNT(*) FROM 'bench_db/nation.arrow';"
"SELECT COUNT(*) FROM 'bench_db/nation.arrow';"
"SELECT COUNT(*) FROM 'bench_db/region.arrow';"
"SELECT COUNT(*) FROM 'bench_db/region.arrow';"
"SELECT COUNT(*) FROM 'bench_db/region.arrow';"
"SELECT sum(l_linenumber) FROM 'bench_db/lineitem.arrow';"
"SELECT sum(l_linenumber) FROM 'bench_db/lineitem.arrow';"
"SELECT sum(l_linenumber) FROM 'bench_db/lineitem.arrow';"
"SELECT sum(l_linenumber), max(l_orderkey), min(l_suppkey) FROM 'bench_db/lineitem.arrow';"
"SELECT sum(l_linenumber), max(l_orderkey), min(l_suppkey) FROM 'bench_db/lineitem.arrow';"
"SELECT sum(l_linenumber), max(l_orderkey), min(l_suppkey) FROM 'bench_db/lineitem.arrow';"
"SELECT max(o_orderkey), min(o_custkey) FROM 'bench_db/orders.arrow';"
"SELECT max(o_orderkey), min(o_custkey) FROM 'bench_db/orders.arrow';"
"SELECT max(o_orderkey), min(o_custkey) FROM 'bench_db/orders.arrow';"
"SELECT sum(o_shippriority) FROM 'bench_db/orders.arrow';"
"SELECT sum(o_shippriority) FROM 'bench_db/orders.arrow';"
"SELECT sum(o_shippriority) FROM 'bench_db/orders.arrow';"
"SELECT sum(c_nationkey) FROM 'bench_db/customer.arrow';"
"SELECT sum(c_nationkey) FROM 'bench_db/customer.arrow';"
"SELECT sum(c_nationkey) FROM 'bench_db/customer.arrow';"
"SELECT sum(c_custkey), min(c_custkey), max(c_custkey) FROM 'bench_db/customer.arrow';"
"SELECT sum(c_custkey), min(c_custkey), max(c_custkey) FROM 'bench_db/customer.arrow';"
"SELECT sum(c_custkey), min(c_custkey), max(c_custkey) FROM 'bench_db/customer.arrow';"
"SELECT sum(p_size) FROM 'bench_db/part.arrow';"
"SELECT sum(p_size) FROM 'bench_db/part.arrow';"
"SELECT sum(p_size) FROM 'bench_db/part.arrow';"
"SELECT sum(s_suppkey), sum(s_nationkey), min(s_nationkey), max(s_nationkey) FROM 'bench_db/supplier.arrow';"
"SELECT sum(s_suppkey), sum(s_nationkey), min(s_nationkey), max(s_nationkey) FROM 'bench_db/supplier.arrow';"
"SELECT sum(s_suppkey), sum(s_nationkey), min(s_nationkey), max(s_nationkey) FROM 'bench_db/supplier.arrow';"
"SELECT * FROM 'bench_db/region.arrow';"
"SELECT * FROM 'bench_db/region.arrow';"
"SELECT * FROM 'bench_db/region.arrow';"
"SELECT sum(r_regionkey) FROM 'bench_db/region.arrow';"
"SELECT sum(r_regionkey) FROM 'bench_db/region.arrow';"
"SELECT sum(r_regionkey) FROM 'bench_db/region.arrow';"
)

# Function to run query and capture output
run_query() {
    local query="$1"
    
    echo "==========================================" | tee -a "$OUTPUT_FILE"
    echo "Running query: $query" | tee -a "$OUTPUT_FILE"
    echo "==========================================" | tee -a "$OUTPUT_FILE"
    
    # Create command file for this query
    cat << EOF > /tmp/duckdb_commands.txt
.timer on
load nanoarrow;
0
$query
.quit
EOF
    
    # Run DuckDB and capture output to both stdout and file
    ./build-fast/duckdb < /tmp/duckdb_commands.txt 2>&1 | tee -a "$OUTPUT_FILE"
    
    echo "" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
}

# Run each query
for query in "${QUERIES[@]}"; do
    run_query "$query"
done

# Clean up
rm -f /tmp/duckdb_commands.txt

echo "All queries completed! Results saved to $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
