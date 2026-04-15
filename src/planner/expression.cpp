#include "duckdb/planner/expression.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/planner/expression/list.hpp"
#include "duckdb/parser/expression_util.hpp"

#include <iostream>

namespace duckdb {

void printExpression(const unique_ptr<Expression> &expression, int depth) {
	string indent = std::string(depth * 4, ' ');

	std::cout << indent << "Expression Type :: " << ExpressionTypeToString(expression->type) << std::endl;
	std::cout << indent << "Expression Class :: " << ExpressionClassToString(expression->GetExpressionClass())
	          << std::endl;
	switch (expression->GetExpressionClass()) {
	case ExpressionClass::BOUND_COMPARISON: {
		auto &bound_comparison = (BoundComparisonExpression &)*expression;
		std::cout << indent << "Expression Details :: " << expression->ToString() << std::endl;
		printExpression(bound_comparison.left, depth + 1);
		printExpression(bound_comparison.right, depth + 1);
		std::cout << indent << "Left Operand :: " << bound_comparison.left->ToString() << std::endl;
		std::cout << indent << "Right Operand :: " << bound_comparison.right->ToString() << std::endl;
		printExpression(bound_comparison.left, depth + 1);
		printExpression(bound_comparison.right, depth + 1);
		break;
	}
	case ExpressionClass::BOUND_FUNCTION: {
		auto &bound_function = (BoundFunctionExpression &)*expression;
		std::cout << indent << "Expression Details :: " << expression->ToString() << std::endl;
		std::cout << indent << "Function :: " << bound_function.function.ToString() << std::endl;
		for (auto &child : bound_function.children) {
			printExpression(child, depth + 1);
		}
		break;
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &bound_op = (BoundOperatorExpression &)*expression;
		std::cout << indent << "Operator :: " << ExpressionTypeToString(bound_op.type) << std::endl;
		std::cout << indent << "Expression Details :: " << expression->ToString() << std::endl;

		for (auto &child : bound_op.children) {
			printExpression(child, depth + 1);
		}
		break;
	}
	case ExpressionClass::BOUND_COLUMN_REF: {
		auto &bound_col_ref = (BoundColumnRefExpression &)*expression;
		std::cout << indent << "Expression Details :: " << expression->ToString() << std::endl;
		std::cout << indent << "Column Binding :: " << bound_col_ref.binding.ToString() << std::endl;
		break;
	}
	case ExpressionClass::BOUND_SUBQUERY: {
		auto &bound_subquery = (BoundSubqueryExpression &)*expression;
		std::cout << indent << "Expression Details :: " << expression->ToString() << std::endl;
		std::cout << indent << "Subquery Plan :: " << std::endl;
		auto comparison_type = bound_subquery.comparison_type;
		std::cout << indent << "Subquery Comparison Type :: " << ExpressionTypeToString(comparison_type) << std::endl;
		break;
	}
	case ExpressionClass::BOUND_CONJUNCTION: {
		auto &bound_conjunction = (BoundConjunctionExpression &)*expression;
		std::cout << indent << "Expression Details :: " << expression->ToString() << std::endl;
		std::cout << indent << "Conjunction Type :: " << ExpressionTypeToString(bound_conjunction.type) << std::endl;
		for (auto &child : bound_conjunction.children) {
			printExpression(child, depth + 1);
		}
		break;
	}
	case ExpressionClass::BOUND_CASE: {
		auto &bound_case = (BoundCaseExpression &)*expression;
		std::cout << indent << "Expression Details :: " << expression->ToString() << std::endl;
		std::cout << indent << "CASE checks:: " << std::endl;
		for (auto &check: bound_case.case_checks) {
			std::cout << indent << "WHEN :: " << check.when_expr->ToString() << std::endl;
			std::cout << indent << "THEN :: " << check.then_expr->ToString() << std::endl;
			printExpression(check.when_expr, depth + 1);
			printExpression(check.then_expr, depth + 1);
		}
		std::cout << indent << "ELSE :: " << bound_case.else_expr->ToString() << std::endl;
		printExpression(bound_case.else_expr, depth + 1);
		break;
	}
	default: {
		std::cout << indent << "Expression Details :: " << expression->ToString() << std::endl;
		break;
	}
	}
	std::cout << indent << "Expression Alias :: " << expression->alias << std::endl;
}

Expression::Expression(ExpressionType type, ExpressionClass expression_class, LogicalType return_type)
    : BaseExpression(type, expression_class), return_type(std::move(return_type)) {
}

Expression::~Expression() {
}

mlir::Value Expression::translateExpression(MLIRTranslationContext &context, mlir::OpBuilder &builder, LogicalOperator *op) {
	// Base implementation does nothing, individual expression types can override this to provide their own translation logic.
	std::cout << "[Expression::translateExpression] Base Expression translation called for expression of type :: " << ToString()
	          << std::endl;
	return mlir::Value();
}

bool Expression::IsAggregate() const {
	bool is_aggregate = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) { is_aggregate |= child.IsAggregate(); });
	return is_aggregate;
}

bool Expression::IsWindow() const {
	bool is_window = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) { is_window |= child.IsWindow(); });
	return is_window;
}

bool Expression::IsScalar() const {
	bool is_scalar = true;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (!child.IsScalar()) {
			is_scalar = false;
		}
	});
	return is_scalar;
}

bool Expression::IsVolatile() const {
	bool is_volatile = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (child.IsVolatile()) {
			is_volatile = true;
		}
	});
	return is_volatile;
}

bool Expression::IsConsistent() const {
	bool is_consistent = true;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (!child.IsConsistent()) {
			is_consistent = false;
		}
	});
	return is_consistent;
}

bool Expression::CanThrow() const {
	bool can_throw = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) { can_throw |= child.CanThrow(); });
	return can_throw;
}

bool Expression::PropagatesNullValues() const {
	if (type == ExpressionType::OPERATOR_IS_NULL || type == ExpressionType::OPERATOR_IS_NOT_NULL ||
	    type == ExpressionType::COMPARE_NOT_DISTINCT_FROM || type == ExpressionType::COMPARE_DISTINCT_FROM ||
	    type == ExpressionType::CONJUNCTION_OR || type == ExpressionType::CONJUNCTION_AND ||
	    type == ExpressionType::OPERATOR_COALESCE || type == ExpressionType::CASE_EXPR) {
		return false;
	}
	bool propagate_null_values = true;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (!child.PropagatesNullValues()) {
			propagate_null_values = false;
		}
	});
	return propagate_null_values;
}

bool Expression::IsFoldable() const {
	bool is_foldable = true;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (!child.IsFoldable()) {
			is_foldable = false;
		}
	});
	return is_foldable;
}

bool Expression::HasParameter() const {
	bool has_parameter = false;
	ExpressionIterator::EnumerateChildren(*this,
	                                      [&](const Expression &child) { has_parameter |= child.HasParameter(); });
	return has_parameter;
}

bool Expression::HasSubquery() const {
	bool has_subquery = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) { has_subquery |= child.HasSubquery(); });
	return has_subquery;
}

hash_t Expression::Hash() const {
	hash_t hash = duckdb::Hash<uint32_t>(static_cast<uint32_t>(type));
	hash = CombineHash(hash, return_type.Hash());
	ExpressionIterator::EnumerateChildren(*this,
	                                      [&](const Expression &child) { hash = CombineHash(child.Hash(), hash); });
	return hash;
}

bool Expression::Equals(const unique_ptr<Expression> &left, const unique_ptr<Expression> &right) {
	if (left.get() == right.get()) {
		return true;
	}
	if (!left || !right) {
		return false;
	}
	return left->Equals(*right);
}

bool Expression::ListEquals(const vector<unique_ptr<Expression>> &left, const vector<unique_ptr<Expression>> &right) {
	return ExpressionUtil::ListEquals(left, right);
}

} // namespace duckdb
