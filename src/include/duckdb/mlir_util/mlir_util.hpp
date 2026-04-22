#pragma once

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/frontend/SQL/Parser.h"
#include "lingodb/runtime/Session.h"

#include "mlir/IR/BuiltinDialect.h"
// #include "duckdb/main/client_context.hpp"

#include "duckdb/planner/column_binding.hpp"

#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <unordered_set>

namespace duckdb {

struct DefineScope;
struct TupleScope;


void runMLIR();

struct MLIRStringInfo {
	static bool isEqual(std::string a, std::string b);
	static std::string getEmptyKey();
	static std::string getTombstoneKey();
	static size_t getHashValue(std::string str);
};

class ClientContext;

struct MLIRTranslationContext {
public:
	// static MLIRTranslationContext &getInstance() {
	// 	static MLIRTranslationContext instance;
	// 	return instance;
	// }
	duckdb::ClientContext* clientContext;
	std::stack<mlir::Value> currTuple;
	std::unordered_set<const lingodb::compiler::dialect::tuples::Column *> useZeroInsteadNull;
	std::stack<std::vector<std::pair<std::string, const lingodb::compiler::dialect::tuples::Column *>>>
	    definedAttributes;

	llvm::ScopedHashTable<std::string, const lingodb::compiler::dialect::tuples::Column *, MLIRStringInfo> resolver;
	using ResolverScope =
	    llvm::ScopedHashTable<std::string, const lingodb::compiler::dialect::tuples::Column *, MLIRStringInfo>::ScopeTy;
	MLIRTranslationContext();
	mlir::Value getCurrentTuple();
	void setCurrentTuple(mlir::Value v);
	void mapAttribute(ResolverScope &scope, std::string name, const lingodb::compiler::dialect::tuples::Column *attr);
	const lingodb::compiler::dialect::tuples::Column *getAttribute(std::string name);
	TupleScope createTupleScope();
	ResolverScope createResolverScope();

	DefineScope createDefineScope();
	const std::vector<std::pair<std::string, const lingodb::compiler::dialect::tuples::Column *>> &
	getAllDefinedColumns();
	void removeFromDefinedColumns(const lingodb::compiler::dialect::tuples::Column *col);

	void replace(ResolverScope &scope, const lingodb::compiler::dialect::tuples::Column *col,
	             const lingodb::compiler::dialect::tuples::Column *col2);

	// Map from ColumnBinding to pre-computed scalar MLIR values (for uncorrelated scalar subqueries)
	struct ColumnBindingHash {
		size_t operator()(const duckdb::ColumnBinding &b) const {
			return std::hash<idx_t>()(b.table_index) ^ (std::hash<idx_t>()(b.column_index) << 32);
		}
	};
	struct ColumnBindingEqual {
		bool operator()(const duckdb::ColumnBinding &a, const duckdb::ColumnBinding &b) const {
			return a.table_index == b.table_index && a.column_index == b.column_index;
		}
	};

	// Deferred scalar subquery info: stores the stream and column needed to emit relalg.getscalar
	// lazily inside the predicate block where it's actually used.
	struct DeferredScalarInfo {
		mlir::Value subqueryStream;
		const lingodb::compiler::dialect::tuples::Column *column;
	};
	std::unordered_map<duckdb::ColumnBinding, DeferredScalarInfo, ColumnBindingHash, ColumnBindingEqual> deferredScalarSubqueries;

	// Deferred EXISTS callbacks: keyed by the MARK join's "mark" ColumnBinding.
	// Set by LogicalDependentJoin(MARK) so that BoundColumnRefExpression can intercept
	// the mark column reference and emit relalg.exists inside the predicate block.
	std::unordered_map<duckdb::ColumnBinding,
	                   std::function<mlir::Value(mlir::OpBuilder &)>,
	                   ColumnBindingHash, ColumnBindingEqual>
	    deferredExistsCallbacks;

	// Deferred scalar callbacks: keyed by the SINGLE join's output ColumnBinding.
	// Set by LogicalDependentJoin(SINGLE) so that BoundColumnRefExpression can intercept
	// the scalar column reference, build the right child inside the predicate block,
	// and emit relalg.getscalar there (ensuring correlated refs are in scope).
	std::unordered_map<duckdb::ColumnBinding,
	                   std::function<mlir::Value(mlir::OpBuilder &)>,
	                   ColumnBindingHash, ColumnBindingEqual>
	    deferredScalarCallbacks;
};
struct DefineScope {
public:
	MLIRTranslationContext &context;
	DefineScope(MLIRTranslationContext &context);
	~DefineScope();
};

struct TupleScope {
public:
	MLIRTranslationContext *context;
	bool active;
	TupleScope(MLIRTranslationContext *context);
	~TupleScope();
};

} // namespace duckdb
