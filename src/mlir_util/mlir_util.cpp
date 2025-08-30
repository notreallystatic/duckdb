#include "duckdb/mlir_util/mlir_util.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"

#include <iostream>

namespace duckdb {
MLIRContainer::MLIRContainer() {
}

mlir::MLIRContext MLIRContainer::context;
mlir::DialectRegistry MLIRContainer::registry;
mlir::OpBuilder MLIRContainer::builder(&context);
mlir::ModuleOp MLIRContainer::moduleOp;
mlir::OpPrintingFlags MLIRContainer::flags;

void MLIRContainer::init() {
	registry.insert<mlir::BuiltinDialect>();
	registry.insert<lingodb::compiler::dialect::relalg::RelAlgDialect>();
	registry.insert<lingodb::compiler::dialect::subop::SubOperatorDialect>();
	registry.insert<lingodb::compiler::dialect::tuples::TupleStreamDialect>();
	registry.insert<lingodb::compiler::dialect::db::DBDialect>();
	registry.insert<mlir::func::FuncDialect>();
	registry.insert<mlir::arith::ArithDialect>();

	registry.insert<mlir::memref::MemRefDialect>();
	registry.insert<lingodb::compiler::dialect::util::UtilDialect>();
	registry.insert<mlir::scf::SCFDialect>();
	registry.insert<mlir::LLVM::LLVMDialect>();
	context.appendDialectRegistry(registry);
	context.loadAllAvailableDialects();
	context.loadDialect<lingodb::compiler::dialect::relalg::RelAlgDialect>();

	builder = mlir::OpBuilder(&context);
	moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

	builder.setInsertionPointToStart(moduleOp.getBody());
	auto *queryBlock = new mlir::Block();
	mlir::func::FuncOp funcOp =
	    builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", builder.getFunctionType({}, {}));
	funcOp.getBody().push_back(queryBlock);
}

void MLIRContainer::print() {
	std::cout << "MLIR so far :: \n";
	flags.assumeVerified();
	moduleOp->print(llvm::outs(), flags);
}

// mlir::Type convertDuckDBTypeToMLIRType(const LogicalType &type) {
// 	switch (type.id()) {
// 	case LogicalTypeId::BOOLEAN:
// 		return mlir::IntegerType::get(&MLIRContainer::context, 1);
// 	case LogicalTypeId::TINYINT:
// 		return mlir::IntegerType::get(&MLIRContainer::context, 8);
// 	case LogicalTypeId::SMALLINT:
// 		return mlir::IntegerType::get(&MLIRContainer::context, 16);
// 	case LogicalTypeId::INTEGER:
// 		return mlir::IntegerType::get(&MLIRContainer::context, 32);
// 	case LogicalTypeId::BIGINT:
// 		return mlir::IntegerType::get(&MLIRContainer::context, 64);
// 	case LogicalTypeId::HUGEINT:
// 		return mlir::IntegerType::get(&MLIRContainer::context, 128);
// 	case LogicalTypeId::FLOAT:
// 		return mlir::Float32Type::get(&MLIRContainer::context);
// 	case LogicalTypeId::DOUBLE:
// 		return mlir::Float64Type::get(&MLIRContainer::context);
// 	case LogicalTypeId::VARCHAR:
// 		return lingodb::compiler::dialect::db::StringType::get(&MLIRContainer::context);
// 	default:
// 		throw InternalException("Unsupported type for MLIR conversion");
// 	}
// }

// mlir::Type convertDuckDBTypeToNullableType(const LogicalType &type) {
// 	auto baseType = convertDuckDBTypeToMLIRType(type);
// 	return lingodb::compiler::dialect::db::NullableType::get(&MLIRContainer::context, baseType);
// }

} // namespace duckdb