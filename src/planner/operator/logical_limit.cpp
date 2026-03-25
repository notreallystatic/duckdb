#include "duckdb/planner/operator/logical_limit.hpp"

namespace relalg = lingodb::compiler::dialect::relalg;
namespace tuples = lingodb::compiler::dialect::tuples;

namespace duckdb {

LogicalLimit::LogicalLimit(BoundLimitNode limit_val, BoundLimitNode offset_val)
    : LogicalOperator(LogicalOperatorType::LOGICAL_LIMIT), limit_val(std::move(limit_val)),
      offset_val(std::move(offset_val)) {
}

vector<ColumnBinding> LogicalLimit::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

idx_t LogicalLimit::EstimateCardinality(ClientContext &context) {
	auto child_cardinality = children[0]->EstimateCardinality(context);
	switch (limit_val.Type()) {
	case LimitNodeType::CONSTANT_VALUE:
		if (limit_val.GetConstantValue() < child_cardinality) {
			child_cardinality = limit_val.GetConstantValue();
		}
		break;
	case LimitNodeType::CONSTANT_PERCENTAGE:
		child_cardinality = idx_t(double(child_cardinality) * limit_val.GetConstantPercentage());
		break;
	default:
		break;
	}
	return child_cardinality;
}

void LogicalLimit::ResolveTypes() {
	types = children[0]->types;
}

void LogicalLimit::resolveMLIRValue(MLIRTranslationContext& translationContext, MLIRTranslationContext::ResolverScope& scope) {
	std::cout << "[LogicalLimit](resolveMLIRValue) :: " << LogicalOperatorToString(type) << std::endl;
	std::cout.flush();

	if (children.size() != 1) {
		std::cout << "[LogicalOrder](resolveMLIRValue) :: Expected exactly one child for LogicalOrder but found " << children.size() << std::endl;
		throw InternalException("LogicalOrder operator should have exactly one child");
	}

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto &builder = mlirContainerInstance.getBuilder();
	auto loc = builder.getUnknownLoc();

	auto child = children[0].get();
	child->resolveMLIRValue(translationContext, scope);
	mlir::Value childValue = child->getMLIRValue();

	mlir::Value limitValue = builder.create<relalg::LimitOp>(
		loc,
		tuples::TupleStreamType::get(builder.getContext()),
		limit_val.GetConstantValue(),
		childValue
	);
	this->mlirValue = limitValue;
	std::cout << "[LogicalLimit](resolveMLIRValue) :: Resolved MLIR Value for LogicalLimit: " << std::endl;
	std::cout.flush();
	this->mlirValue.print(llvm::outs());
	std::cout << std::endl;
}

} // namespace duckdb
