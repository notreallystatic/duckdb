

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/frontend/SQL/Parser.h"
#include "lingodb/runtime/Session.h"

#include "mlir/IR/BuiltinDialect.h"

#include <stack>
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

struct MLIRTranslationContext {
public:
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