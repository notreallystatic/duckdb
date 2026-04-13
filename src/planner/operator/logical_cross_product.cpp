#include "duckdb/planner/operator/logical_cross_product.hpp"
#include <algorithm>

namespace duckdb {

LogicalCrossProduct::LogicalCrossProduct(unique_ptr<LogicalOperator> left, unique_ptr<LogicalOperator> right)
    : LogicalUnconditionalJoin(LogicalOperatorType::LOGICAL_CROSS_PRODUCT, std::move(left), std::move(right)) {
}

MLIRAttributeInfo& LogicalCrossProduct::resolveColumnBindingToAttributeInfo(ColumnBinding& binding) {
	auto leftBindings = children[0]->GetColumnBindings();
	auto rightBindings = children[1]->GetColumnBindings();
	auto leftIt = std::find(leftBindings.begin(), leftBindings.end(), binding);
	if (leftIt != leftBindings.end()) {
		return children[0]->resolveColumnBindingToAttributeInfo(binding);
	}
	return children[1]->resolveColumnBindingToAttributeInfo(binding);
}

void LogicalCrossProduct::resolveMLIRValue(MLIRTranslationContext& context, MLIRTranslationContext::ResolverScope& scope) {
	std::cout << "[LogicalCrossProduct](resolveMLIRValue) :: Resolving MLIR value for LogicalCrossProduct" << std::endl;
	children[0]->resolveMLIRValue(context, scope);
	children[1]->resolveMLIRValue(context, scope);

	auto leftValue = children[0]->getMLIRValue();
	auto rightValue = children[1]->getMLIRValue();

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	D_ASSERT(mlirContainerInstance.getContextPtr() != nullptr);

	auto &mlirContext = mlirContainerInstance.getContext();
	auto &builder = mlirContainerInstance.getBuilder();
	auto module = mlirContainerInstance.getModuleOp();

	this->mlirValue = builder.create<lingodb::compiler::dialect::relalg::CrossProductOp>(
		builder.getUnknownLoc(),
		lingodb::compiler::dialect::tuples::TupleStreamType::get(builder.getContext()),
		leftValue,
		rightValue);
}

unique_ptr<LogicalOperator> LogicalCrossProduct::Create(unique_ptr<LogicalOperator> left,
                                                        unique_ptr<LogicalOperator> right) {
	if (left->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN) {
		return right;
	}
	if (right->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN) {
		return left;
	}
	return make_uniq<LogicalCrossProduct>(std::move(left), std::move(right));
}

} // namespace duckdb
