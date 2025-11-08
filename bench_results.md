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

| Query                                                                                     | Raw Execution | Details                                             | Compiled Query Execution | Details                                                                                             |
| ----------------------------------------------------------------------------------------- | ------------- | --------------------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------- |
| COUNT on Lineitem                                                                         | 387ms         | real 0.387 user 0.058721 sys 0.241634               | 864ms                    | compilation: 283.251 [ms] execution: 4.234 [ms] Run Time (s): real 0.864 user 0.040090 sys 0.281848 |
| COUNT on Orders                                                                           | 104ms         | Run Time (s): real 0.104 user 0.018683 sys 0.074474 | 194ms                    | compilation: 23.344 [ms] execution: 0.479 [ms] Run Time (s): real 0.194 user 0.027491 sys 0.068038  |
| COUNT on Customer                                                                         | 22ms          | Run Time (s): real 0.022 user 0.005160 sys 0.014300 | 55ms                     | compilation: 21.272 [ms] execution: 0.201 [ms] Run Time (s): real 0.055 user 0.029594 sys 0.023761  |
| COUNT on PartSupp                                                                         | 60ms          | Run Time (s): real 0.060 user 0.014345 sys 0.043006 | 92ms                     | compilation: 16.113 [ms] execution: 0.299 [ms] Run Time (s): real 0.092 user 0.027592 sys 0.049786  |
| COUNT on Part                                                                             | 24ms          | Run Time (s): real 0.024 user 0.005654 sys 0.017083 | 46ms                     | compilation: 21.009 [ms] execution: 0.231 [ms] Run Time (s): real 0.046 user 0.026585 sys 0.020006  |
| COUNT on Supplier                                                                         | 5ms           | Run Time (s): real 0.005 user 0.001196 sys 0.002494 | 44ms                     | compilation: 21.529 [ms] execution: 0.148 [ms] Run Time (s): real 0.044 user 0.030035 sys 0.013348  |
| COUNT on Nation                                                                           | 3ms           | Run Time (s): real 0.003 user 0.001335 sys 0.001183 | 20ms                     | compilation: 10.798 [ms] execution: 0.074 [ms] Run Time (s): real 0.020 user 0.013110 sys 0.007225  |
| COUNT on Region                                                                           | 2ms           | Run Time (s): real 0.002 user 0.001254 sys 0.000730 | 17ms                     | compilation: 9.877 [ms] execution: 0.076 [ms] Run Time (s): real 0.017 user 0.013302 sys 0.007236   |
| SELECT sum(l_linenumber) FROM 'bench_db/lineitem.arrow';                                  | 306ms         | Run Time (s): real 0.306 user 0.064696 sys 0.193842 | 427ms                    | compilation: 24.917 [ms] execution: 9.072 [ms] Run Time (s): real 0.427 user 0.048933 sys 0.235221  |
| SELECT sum(l_linenumber), max(l_orderkey), min(l_suppkey) FROM 'bench_db/lineitem.arrow'; | 305ms         | Run Time (s): real 0.305 user 0.079295 sys 0.173922 | 286ms                    | compilation: 26.035 [ms] execution: 21.047 [ms] Run Time (s): real 0.286 user 0.076962 sys 0.216998 |
| SELECT max(o_orderkey), min(o_custkey) FROM 'bench_db/orders.arrow';                      | 97ms          | Run Time (s): real 0.097 user 0.105416 sys 0.057317 | 84ms                     | compilation: 18.257 [ms] execution: 1.58 [ms] Run Time (s): real 0.084 user 0.030285 sys 0.041091   |
| SELECT sum(o_shippriority) FROM 'bench_db/orders.arrow';                                  | 93ms          | Run Time (s): real 0.093 user 0.020639 sys 0.063502 | 48ms                     | compilation: 13.228 [ms] execution: 0.998 [ms] Run Time (s): real 0.048 user 0.023862 sys 0.032714  |
