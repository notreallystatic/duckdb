#!/bin/bash
# filepath: run_multiple_queries.sh

# Array of queries
QUERIES=(
"SELECT sum(o_shippriority) FROM 'bench_db/orders.arrow';"
)

# Run each query in a separate DuckDB session
for query in "${QUERIES[@]}"; do
    echo "=========================================="
    echo "Running query: $query"
    echo "=========================================="
    
    # Create command file for this query
    cat << EOF > /tmp/duckdb_commands.txt
.timer on
load nanoarrow;
1
$query
.quit
EOF
    
    # Run DuckDB with the commands
    ./build-fast/duckdb < /tmp/duckdb_commands.txt
    
    echo ""
    echo ""
done

# Clean up
rm -f /tmp/duckdb_commands.txt

echo "All queries completed!"
