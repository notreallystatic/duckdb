# Benchmarking Hybrid DuckDB

## Tables and Schemas

### Lineitem
  
**Records**: 6,001,215 (6.00M)

| column_name     | column_type   | null | key  | default | extra |
| --------------- | ------------- | ---- | ---- | ------- | ----- |
| l_orderkey      | INTEGER       | YES  | NULL | NULL    | NULL  |
| l_partkey       | INTEGER       | YES  | NULL | NULL    | NULL  |
| l_suppkey       | INTEGER       | YES  | NULL | NULL    | NULL  |
| l_linenumber    | INTEGER       | YES  | NULL | NULL    | NULL  |
| l_quantity      | DECIMAL(12,2) | YES  | NULL | NULL    | NULL  |
| l_extendedprice | DECIMAL(12,2) | YES  | NULL | NULL    | NULL  |
| l_discount      | DECIMAL(12,2) | YES  | NULL | NULL    | NULL  |
| l_tax           | DECIMAL(12,2) | YES  | NULL | NULL    | NULL  |
| l_returnflag    | BLOB          | YES  | NULL | NULL    | NULL  |
| l_linestatus    | BLOB          | YES  | NULL | NULL    | NULL  |
| l_shipdate      | DATE          | YES  | NULL | NULL    | NULL  |
| l_commitdate    | DATE          | YES  | NULL | NULL    | NULL  |
| l_receiptdate   | DATE          | YES  | NULL | NULL    | NULL  |
| l_shipinstruct  | VARCHAR       | YES  | NULL | NULL    | NULL  |
| l_shipmode      | VARCHAR       | YES  | NULL | NULL    | NULL  |
| l_comment       | VARCHAR       | YES  | NULL | NULL    | NULL  |

### Orders

**Records**: 1,500,000

| column_name     | column_type   | null | key  | default | extra |
| --------------- | ------------- | ---- | ---- | ------- | ----- |
| o_orderkey      | INTEGER       | YES  | NULL | NULL    | NULL  |
| o_custkey       | INTEGER       | YES  | NULL | NULL    | NULL  |
| o_orderstatus   | BLOB          | YES  | NULL | NULL    | NULL  |
| o_totalprice    | DECIMAL(12,2) | YES  | NULL | NULL    | NULL  |
| o_orderdate     | DATE          | YES  | NULL | NULL    | NULL  |
| o_orderpriority | VARCHAR       | YES  | NULL | NULL    | NULL  |
| o_clerk         | VARCHAR       | YES  | NULL | NULL    | NULL  |
| o_shippriority  | INTEGER       | YES  | NULL | NULL    | NULL  |
| o_comment       | VARCHAR       | YES  | NULL | NULL    | NULL  |

### Customer

**Records**: 150,000

| column_name  | column_type   | null | key  | default | extra |
| ------------ | ------------- | ---- | ---- | ------- | ----- |
| c_custkey    | INTEGER       | YES  | NULL | NULL    | NULL  |
| c_name       | VARCHAR       | YES  | NULL | NULL    | NULL  |
| c_address    | VARCHAR       | YES  | NULL | NULL    | NULL  |
| c_nationkey  | INTEGER       | YES  | NULL | NULL    | NULL  |
| c_phone      | VARCHAR       | YES  | NULL | NULL    | NULL  |
| c_acctbal    | DECIMAL(12,2) | YES  | NULL | NULL    | NULL  |
| c_mktsegment | VARCHAR       | YES  | NULL | NULL    | NULL  |
| c_comment    | VARCHAR       | YES  | NULL | NULL    | NULL  |

### PartSupp

**Records**: 800,000

| column_name   | column_type   | null | key  | default | extra |
| ------------- | ------------- | ---- | ---- | ------- | ----- |
| ps_partkey    | INTEGER       | YES  | NULL | NULL    | NULL  |
| ps_suppkey    | INTEGER       | YES  | NULL | NULL    | NULL  |
| ps_availqty   | INTEGER       | YES  | NULL | NULL    | NULL  |
| ps_supplycost | DECIMAL(12,2) | YES  | NULL | NULL    | NULL  |
| ps_comment    | VARCHAR       | YES  | NULL | NULL    | NULL  |

### Part

**Records**: 200,000

| column_name   | column_type   | null | key  | default | extra |
| ------------- | ------------- | ---- | ---- | ------- | ----- |
| p_partkey     | INTEGER       | YES  | NULL | NULL    | NULL  |
| p_name        | VARCHAR       | YES  | NULL | NULL    | NULL  |
| p_mfgr        | VARCHAR       | YES  | NULL | NULL    | NULL  |
| p_brand       | VARCHAR       | YES  | NULL | NULL    | NULL  |
| p_type        | VARCHAR       | YES  | NULL | NULL    | NULL  |
| p_size        | INTEGER       | YES  | NULL | NULL    | NULL  |
| p_container   | VARCHAR       | YES  | NULL | NULL    | NULL  |
| p_retailprice | DECIMAL(12,2) | YES  | NULL | NULL    | NULL  |
| p_comment     | VARCHAR       | YES  | NULL | NULL    | NULL  |

### Supplier

**Records**: 10,000

| column_name | column_type   | null | key  | default | extra |
| ----------- | ------------- | ---- | ---- | ------- | ----- |
| s_suppkey   | INTEGER       | YES  | NULL | NULL    | NULL  |
| s_name      | VARCHAR       | YES  | NULL | NULL    | NULL  |
| s_address   | VARCHAR       | YES  | NULL | NULL    | NULL  |
| s_nationkey | INTEGER       | YES  | NULL | NULL    | NULL  |
| s_phone     | VARCHAR       | YES  | NULL | NULL    | NULL  |
| s_acctbal   | DECIMAL(12,2) | YES  | NULL | NULL    | NULL  |
| s_comment   | VARCHAR       | YES  | NULL | NULL    | NULL  |

### Nation

**Records**: 25

| column_name | column_type | null | key  | default | extra |
| ----------- | ----------- | ---- | ---- | ------- | ----- |
| n_nationkey | INTEGER     | YES  | NULL | NULL    | NULL  |
| n_name      | VARCHAR     | YES  | NULL | NULL    | NULL  |
| n_regionkey | INTEGER     | YES  | NULL | NULL    | NULL  |
| n_comment   | VARCHAR     | YES  | NULL | NULL    | NULL  |

### Region

**Records**: 5

| column_name | column_type | null | key  | default | extra |
| ----------- | ----------- | ---- | ---- | ------- | ----- |
| r_regionkey | INTEGER     | YES  | NULL | NULL    | NULL  |
| r_name      | VARCHAR     | YES  | NULL | NULL    | NULL  |
| r_comment   | VARCHAR     | YES  | NULL | NULL    | NULL  |

## Benchmark Queries

- SELECT COUNT(*) FROM 'bench_db/lineitem.arrow';
- SELECT COUNT(*) FROM 'bench_db/orders.arrow';
- SELECT COUNT(*) FROM 'bench_db/customer.arrow';
- SELECT COUNT(*) FROM 'bench_db/partsupp.arrow';
- SELECT COUNT(*) FROM 'bench_db/part.arrow';
- SELECT COUNT(*) FROM 'bench_db/supplier.arrow';
- SELECT COUNT(*) FROM 'bench_db/nation.arrow';
- SELECT COUNT(*) FROM 'bench_db/region.arrow';

## Benchmark Results

| Query                                                                                                       | Raw Execution | Details                                             | Compiled Query Execution | Details                                                                                            |
| ----------------------------------------------------------------------------------------------------------- | ------------- | --------------------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------------------- |
| COUNT on Lineitem                                                                                           | 196ms         | Run Time (s): real 0.196 user 0.050256 sys 0.137721 | 235ms                    | compilation: 12.659 [ms] execution: 1.506 [ms] Run Time (s): real 0.235 user 0.031030 sys 0.160976 |
| COUNT on Orders                                                                                             | 48ms          | Run Time (s): real 0.048 user 0.009800 sys 0.034918 | 48ms                     | compilation: 10.241 [ms] execution: 0.54 [ms] Run Time (s): real 0.048 user 0.017768 sys 0.035620  |
| COUNT on Customer                                                                                           | 6ms           | Run Time (s): real 0.006 user 0.001985 sys 0.004404 | 22ms                     | compilation: 10.239 [ms] execution: 0.201 [ms] Run Time (s): real 0.022 user 0.014189 sys 0.011379 |
| COUNT on PartSupp                                                                                           | 25ms          | Run Time (s): real 0.025 user 0.006826 sys 0.018233 | 39ms                     | compilation: 10.679 [ms] execution: 0.434 [ms] Run Time (s): real 0.039 user 0.015939 sys 0.026396 |
| COUNT on Part                                                                                               | 7ms           | Run Time (s): real 0.007 user 0.002059 sys 0.004731 | 23ms                     | compilation: 11.071 [ms] execution: 0.171 [ms] Run Time (s): real 0.023 user 0.014944 sys 0.010969 |
| COUNT on Supplier                                                                                           | 1ms           | Run Time (s): real 0.000 user 0.000463 sys 0.000514 | 18ms                     | compilation: 11.292 [ms] execution: 0.075 [ms] Run Time (s): real 0.018 user 0.014416 sys 0.007292 |
| COUNT on Nation                                                                                             | 1ms           | Run Time (s): real 0.001 user 0.000416 sys 0.000246 | 17ms                     | compilation: 10.549 [ms] execution: 0.175 [ms] Run Time (s): real 0.017 user 0.014049 sys 0.006291 |
| COUNT on Region                                                                                             | 1ms           | Run Time (s): real 0.000 user 0.000339 sys 0.000185 | 17ms                     | compilation: 9.941 [ms] execution: 0.11 [ms] Run Time (s): real 0.017 user 0.013658 sys 0.006900   |
| SELECT sum(l_linenumber) FROM 'bench_db/lineitem.arrow';                                                    | 184ms         | Run Time (s): real 0.184 user 0.051605 sys 0.129610 | 165ms                    | compilation: 13.726 [ms] execution: 3.408 [ms] Run Time (s): real 0.165 user 0.046299 sys 0.142123 |
| SELECT sum(l_linenumber), max(l_orderkey), min(l_suppkey) FROM 'bench_db/lineitem.arrow';                   | 200ms         | Run Time (s): real 0.200 user 0.065417 sys 0.129782 | 170ms                    | compilation: 18.943 [ms] execution: 6.406 [ms] Run Time (s): real 0.170 user 0.072960 sys 0.140506 |
| SELECT max(o_orderkey), min(o_custkey) FROM 'bench_db/orders.arrow';                                        | 37ms          | Run Time (s): real 0.037 user 0.011001 sys 0.025829 | 54ms                     | compilation: 16.402 [ms] execution: 1.713 [ms] Run Time (s): real 0.054 user 0.030036 sys 0.035906 |
| SELECT sum(o_shippriority) FROM 'bench_db/orders.arrow';                                                    | 37ms          | Run Time (s): real 0.037 user 0.010830 sys 0.026057 | 50ms                     | compilation: 13.72 [ms] execution: 0.915 [ms] Run Time (s): real 0.050 user 0.024349 sys 0.034210  |
| SELECT sum(c_nationkey) FROM 'bench_db/customer.arrow';                                                     | 6ms           | Run Time (s): real 0.006 user 0.002035 sys 0.004240 | 26ms                     | compilation: 14.706 [ms] execution: 0.211 [ms] Run Time (s): real 0.026 user 0.018895 sys 0.011383 |
| SELECT sum(p_size) FROM 'bench_db/part.arrow';                                                              | 6ms           | Run Time (s): real 0.006 user 0.002128 sys 0.004488 | 27ms                     | compilation: 14.7 [ms] execution: 0.359 [ms] Run Time (s): real 0.027 user 0.019727 sys 0.012126   |
| SELECT sum(s_suppkey), sum(s_nationkey), min(s_nationkey), max(s_nationkey) FROM 'bench_db/supplier.arrow'; | 1ms           | Run Time (s): real 0.001 user 0.000599 sys 0.000570 | 31ms                     | compilation: 21.47 [ms] execution: 0.188 [ms] Run Time (s): real 0.031 user 0.023660 sys 0.008363  |
| SELECT * FROM 'bench_db/region.arrow';                                                                      | 1ms           | Run Time (s): real 0.001 user 0.001232 sys 0.000335 | 28ms                     | compilation: 20.574 [ms] execution: 0.149 [ms] Run Time (s): real 0.028 user 0.024686 sys 0.007478 |
| SELECT sum(r_regionkey) FROM 'bench_db/region.arrow';                                                       | 1ms           | Run Time (s): real 0.000 user 0.000377 sys 0.000202 | 21ms                     | compilation: 14.608 [ms] execution: 0.117 [ms] Run Time (s): real 0.021 user 0.018049 sys 0.007503 |
