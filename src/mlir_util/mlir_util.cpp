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

#include <stack>
#include <unordered_set>
#include <vector>

namespace duckdb {

bool MLIRStringInfo::isEqual(std::string a, std::string b) {
	return a == b;
}
std::string MLIRStringInfo::getEmptyKey() {
	return "";
}
std::string MLIRStringInfo::getTombstoneKey() {
	return "-";
}

size_t MLIRStringInfo::getHashValue(std::string str) {
	return std::hash<std::string> {}(str);
}

TupleScope::TupleScope(MLIRTranslationContext *context) : context(context) {
	context->currTuple.push(context->currTuple.top());
}

TupleScope::~TupleScope() {
	context->currTuple.pop();
}

DefineScope::DefineScope(MLIRTranslationContext &context) : context(context) {
	context.definedAttributes.push({});
}

DefineScope::~DefineScope() {
	context.definedAttributes.pop();
}

MLIRTranslationContext::MLIRTranslationContext() : currTuple(), resolver() {
	currTuple.push(mlir::Value());
	definedAttributes.push({});
}
mlir::Value MLIRTranslationContext::getCurrentTuple() {
	return currTuple.top();
}
void MLIRTranslationContext::setCurrentTuple(mlir::Value v) {
	currTuple.top() = v;
}
void MLIRTranslationContext::mapAttribute(ResolverScope &scope, std::string name,
                                          const lingodb::compiler::dialect::tuples::Column *attr) {
	definedAttributes.top().push_back({name, attr});
	resolver.insertIntoScope(&scope, std::move(name), attr);
}
const lingodb::compiler::dialect::tuples::Column *MLIRTranslationContext::getAttribute(std::string name) {
	const auto *res = resolver.lookup(name);
	if (!res) {
		// error("could not resolve '" + name + "'");
		throw std::runtime_error("could not resolve '" + name + "'");
	}
	return res;
}
TupleScope MLIRTranslationContext::createTupleScope() {
	return TupleScope(this);
}
MLIRTranslationContext::ResolverScope MLIRTranslationContext::createResolverScope() {
	return ResolverScope(resolver);
}

DefineScope MLIRTranslationContext::createDefineScope() {
	return DefineScope(*this);
}
const std::vector<std::pair<std::string, const lingodb::compiler::dialect::tuples::Column *>> &
MLIRTranslationContext::getAllDefinedColumns() {
	return definedAttributes.top();
}
void MLIRTranslationContext::removeFromDefinedColumns(const lingodb::compiler::dialect::tuples::Column *col) {
	auto &currDefinedColumns = definedAttributes.top();
	auto start = currDefinedColumns.begin();
	auto end = currDefinedColumns.end();
	auto position = std::find_if(start, end, [&](auto el) { return el.second == col; });
	if (position != currDefinedColumns.end()) {
		currDefinedColumns.erase(position);
	}
}

void MLIRTranslationContext::replace(ResolverScope &scope, const lingodb::compiler::dialect::tuples::Column *col,
                                     const lingodb::compiler::dialect::tuples::Column *col2) {
	auto &currDefinedColumns = definedAttributes.top();
	auto start = currDefinedColumns.begin();
	auto end = currDefinedColumns.end();
	std::vector<std::string> toReplace;
	while (start != end) {
		auto position = std::find_if(start, end, [&](auto el) { return el.second == col; });
		if (position != currDefinedColumns.end()) {
			start = position + 1;
			toReplace.push_back(position->first);
		} else {
			start = end;
		}
	}
	for (auto s : toReplace) {
		mapAttribute(scope, s, col2);
	}
}

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
	std::cout << "dumping module :: \n";
	moduleOp.dump();
}

void MLIRContainer::createMainFuncBlock() {
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