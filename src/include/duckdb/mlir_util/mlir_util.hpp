

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

	// static mlir::Type convertDuckDBTypeToMLIRType(const LogicalType &type);
	// static mlir::Type convertDuckDBTypeToNullableType(const LogicalType &type);
};

} // namespace duckdb