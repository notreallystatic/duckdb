#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/operator_expression.hpp"

namespace duckdb {

BoundOperatorExpression::BoundOperatorExpression(ExpressionType type, LogicalType return_type)
    : Expression(type, ExpressionClass::BOUND_OPERATOR, std::move(return_type)) {
}

mlir::Value BoundOperatorExpression::translateExpression(MLIRTranslationContext& translationContext, mlir::OpBuilder& predBuilder, LogicalOperator *op) {
	auto& mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto loc = predBuilder.getUnknownLoc();

	switch (type) {
	case ExpressionType::OPERATOR_IS_NULL:
	case ExpressionType::OPERATOR_IS_NOT_NULL: {
		auto exprResult = children[0]->translateExpression(translationContext, predBuilder, op);
		if (mlir::isa<lingodb::compiler::dialect::db::NullableType>(exprResult.getType())) {
			auto isNull = predBuilder.create<lingodb::compiler::dialect::db::IsNullOp>(loc, exprResult);
			if (type == ExpressionType::OPERATOR_IS_NOT_NULL) {
				return predBuilder.create<lingodb::compiler::dialect::db::NotOp>(loc, isNull);
			}
			return isNull;
		}
		else {
			return predBuilder.create<lingodb::compiler::dialect::db::ConstantOp>(
				loc, predBuilder.getI1Type(),
				predBuilder.getIntegerAttr(predBuilder.getI1Type(),
					type == ExpressionType::OPERATOR_IS_NOT_NULL));
		}
	}
	case ExpressionType::OPERATOR_NOT: {
		auto childVal = children[0]->translateExpression(translationContext, predBuilder, op);
		return predBuilder.create<lingodb::compiler::dialect::db::NotOp>(loc, childVal);
	}
	case ExpressionType::COMPARE_IN: {
		auto leftVal = children[0]->translateExpression(translationContext, predBuilder, op);
		std::vector<mlir::Value> rightVals;
		for (size_t i = 1; i < children.size(); i++) {
			rightVals.push_back(children[i]->translateExpression(translationContext, predBuilder, op));
		}
		return predBuilder.create<lingodb::compiler::dialect::db::OneOfOp>(loc, leftVal, rightVals);
	}
	}
	std::cout << "[BoundOperatorExpression::translateExpression] Unhandled operator :: " << ExpressionTypeToString(type) << std::endl;
	throw std::runtime_error("Unhandled operator in MLIR translation :: " + ExpressionTypeToString(type));
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
