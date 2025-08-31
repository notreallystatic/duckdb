

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/frontend/SQL/Parser.h"
#include "lingodb/runtime/Session.h"

#include "mlir/IR/BuiltinDialect.h"

namespace duckdb {

struct DefineScope;
struct TupleScope;
struct MLIRTranslationContext {
	std::stack<mlir::Value> currTuple;
	std::unordered_set<const lingodb::compiler::dialect::tuples::Column *> useZeroInsteadNull;
	std::stack<std::vector<std::pair<std::string, const lingodb::compiler::dialect::tuples::Column *>>>
	    definedAttributes;

	llvm::ScopedHashTable<std::string, const lingodb::compiler::dialect::tuples::Column *, StringInfo> resolver;
	using ResolverScope =
	    llvm::ScopedHashTable<std::string, const lingodb::compiler::dialect::tuples::Column *, StringInfo>::ScopeTy;
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
	MLIRTranslationContext &context;
	DefineScope(MLIRTranslationContext &context);
	~DefineScope();
};

struct TupleScope {
	MLIRTranslationContext *context;
	bool active;
	TupleScope(MLIRTranslationContext *context);
	~TupleScope();
};

class MLIRContainer {
private:
	MLIRContainer();
	MLIRContainer(const MLIRContainer &) = delete;
	MLIRContainer &operator=(const MLIRContainer &) = delete;

public:
	static MLIRContainer &GetInstance() {
		static MLIRContainer instance;
		return instance;
	}

	static mlir::MLIRContext context;
	static mlir::DialectRegistry registry;
	static mlir::OpBuilder builder; // Declaration only
	static mlir::ModuleOp moduleOp;
	static mlir::OpPrintingFlags flags;

	static void init();
	static void print();
	static void createMainFuncBlock();

	// static mlir::Type convertDuckDBTypeToMLIRType(const LogicalType &type);
	// static mlir::Type convertDuckDBTypeToNullableType(const LogicalType &type);
};

} // namespace duckdb