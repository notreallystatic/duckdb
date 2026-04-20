#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include <algorithm>

#include "lingodb/execution/Frontend.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "mlir/IR/Builders.h"

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

	// Check if the right child is a scalar subquery wrapper (PROJECTION with CASE that was skipped)
	if (children[1]->hasMLIRResolutionSkipped && children[1]->defaultMLIRAttributeInfo != nullptr) {
		std::cout << "[LogicalCrossProduct](resolveMLIRValue) :: Right child is a scalar subquery (CASE projection was skipped), deferring relalg.getscalar" << std::endl;

		// The right child's getMLIRValue() falls through to the actual subquery aggregate/map
		auto subqueryStream = children[1]->getMLIRValue();
		auto *subqueryColumn = children[1]->defaultMLIRAttributeInfo->column;

		// Register deferred scalar info for the right child's column bindings
		// The actual relalg.getscalar will be emitted inside the predicate block where it's used
		auto rightBindings = children[1]->GetColumnBindings();
		for (auto &binding : rightBindings) {
			std::cout << "[LogicalCrossProduct](resolveMLIRValue) :: Registering deferred scalar subquery for binding " << binding.ToString() << std::endl;
			context.deferredScalarSubqueries[binding] = {subqueryStream, subqueryColumn};
		}

		// The cross product's MLIR value is just the left child — no cross product needed
		this->mlirValue = children[0]->getMLIRValue();
		return;
	}

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
