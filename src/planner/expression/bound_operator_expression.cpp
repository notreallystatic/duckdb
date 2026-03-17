#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/operator_expression.hpp"

namespace duckdb {

BoundOperatorExpression::BoundOperatorExpression(ExpressionType type, LogicalType return_type)
    : Expression(type, ExpressionClass::BOUND_OPERATOR, std::move(return_type)) {
}

mlir::Value BoundOperatorExpression::translateExpression(MLIRTranslationContext& translationContext, mlir::OpBuilder& predBuilder) {
	auto& mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto loc = predBuilder.getUnknownLoc();

	if (children.size() != 1) {
		std::cout << "[BoundOperatorExpression::translateExpression] unknown children size :: "
			<< ExpressionTypeToString(GetExpressionType()) << std::endl;
		throw std::runtime_error("BOUND_OPERATOR with unhandled children size");
	}
	auto exprResult = children[0]->translateExpression(translationContext, predBuilder);
	if (mlir::isa<lingodb::compiler::dialect::db::NullableType>(exprResult.getType())) {
		auto isNull = predBuilder.create<lingodb::compiler::dialect::db::IsNullOp>(loc, exprResult);
		if (type == ExpressionType::OPERATOR_IS_NOT_NULL) {
			return predBuilder.create<lingodb::compiler::dialect::db::NotOp>(loc, isNull);
		}
		else if (type == ExpressionType::OPERATOR_IS_NULL) {
			return isNull;
		}
		else {
			std::cout << "[BoundOperatorExpression::translateExpression] Unhandled bound operator type :: "
				<< ExpressionTypeToString(type) << std::endl;
			throw std::runtime_error("Unhandled bound operator type");
		}
	}
	else {
		return predBuilder.create<lingodb::compiler::dialect::db::ConstantOp>(
			loc, predBuilder.getI1Type(),
			predBuilder.getIntegerAttr(predBuilder.getI1Type(),
				type == ExpressionType::OPERATOR_IS_NOT_NULL));
	}

}

string BoundOperatorExpression::ToString() const {
	return OperatorExpression::ToString<BoundOperatorExpression, Expression>(*this);
}

bool BoundOperatorExpression::Equals(const BaseExpression &other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<BoundOperatorExpression>();
	if (!Expression::ListEquals(children, other.children)) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundOperatorExpression::Copy() const {
	auto copy = make_uniq<BoundOperatorExpression>(type, return_type);
	copy->CopyProperties(*this);
	for (auto &child : children) {
		copy->children.push_back(child->Copy());
	}
	return std::move(copy);
}

} // namespace duckdb
