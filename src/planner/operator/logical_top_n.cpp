#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/execution/Frontend.h"

#include <iostream>

namespace relalg = lingodb::compiler::dialect::relalg;
namespace tuples = lingodb::compiler::dialect::tuples;

namespace duckdb {

LogicalTopN::LogicalTopN(vector<BoundOrderByNode> orders, idx_t limit, idx_t offset)
    : LogicalOperator(LogicalOperatorType::LOGICAL_TOP_N), orders(std::move(orders)), limit(limit), offset(offset) {
}

LogicalTopN::~LogicalTopN() {
}

idx_t LogicalTopN::EstimateCardinality(ClientContext &context) {
	auto child_cardinality = LogicalOperator::EstimateCardinality(context);
	if (child_cardinality < limit) {
		return child_cardinality;
	}
	return limit;
}

void LogicalTopN::resolveMLIRValue(MLIRTranslationContext& translationContext, MLIRTranslationContext::ResolverScope& scope) {
	std::cout << "[LogicalTopN](resolveMLIRValue) :: " << LogicalOperatorToString(type) << std::endl;
	std::cout.flush();

	if (children.size() != 1) {
		std::cout << "[LogicalTopN](resolveMLIRValue) :: Expected exactly one child but found " << children.size() << std::endl;
		throw InternalException("LogicalTopN operator should have exactly one child");
	}

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto &builder = mlirContainerInstance.getBuilder();
	auto module = mlirContainerInstance.getModuleOp();
	auto loc = builder.getUnknownLoc();
	tuples::ColumnManager &attrManager =
	    module.getContext()
	        ->getLoadedDialect<tuples::TupleStreamDialect>()
	        ->getColumnManager();

	auto child = children[0].get();
	child->resolveMLIRValue(translationContext, scope);
	mlir::Value childValue = child->getMLIRValue();

	// Build sort specifications from the orders vector, mirroring logical_order.cpp.
	std::vector<mlir::Attribute> orderAttributes;
	for (const auto &order : orders) {
		auto *expr = order.expression.get();
		if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
			auto &node = expr->Cast<BoundColumnRefExpression>();
			auto &mlirAttrInfo = children[0]->resolveColumnBindingToAttributeInfo(node.binding);
			relalg::SortSpec spec = (order.type == OrderType::ASCENDING) ? relalg::SortSpec::asc : relalg::SortSpec::desc;
			orderAttributes.push_back(relalg::SortSpecificationAttr::get(
			    builder.getContext(), attrManager.createRef(mlirAttrInfo.column), spec));
		}
	}

	this->mlirValue = builder.create<relalg::TopKOp>(
	    loc,
	    tuples::TupleStreamType::get(builder.getContext()),
	    static_cast<uint32_t>(limit),
	    childValue,
	    builder.getArrayAttr(orderAttributes));

	std::cout << "[LogicalTopN](resolveMLIRValue) :: Resolved MLIR Value for LogicalTopN: " << std::endl;
	std::cout.flush();
	this->mlirValue.print(llvm::outs());
	std::cout << std::endl;
}



} // namespace duckdb
