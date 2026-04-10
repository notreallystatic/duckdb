#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"

#include "duckdb/main/config.hpp"

#include <iostream>

namespace relalg = lingodb::compiler::dialect::relalg;
namespace tuples = lingodb::compiler::dialect::tuples;
namespace subop = lingodb::compiler::dialect::subop;

namespace duckdb {

LogicalProjection::LogicalProjection(idx_t table_index, vector<unique_ptr<Expression>> select_list)
    : LogicalOperator(LogicalOperatorType::LOGICAL_PROJECTION, std::move(select_list)), table_index(table_index) {
}

MLIRAttributeInfo& LogicalProjection::resolveColumnBindingToAttributeInfo(ColumnBinding& binding) {
	if (binding.table_index == table_index) {
		auto columnIndex = binding.column_index;
		if (expressions.size() <= columnIndex) {
			throw std::runtime_error("Column index " + to_string(columnIndex) + " out of bounds for projection expressions of size " + to_string(expressions.size()));
		}
		auto expr = expressions[columnIndex].get();
		if (expr->type != ExpressionType::BOUND_COLUMN_REF) {
			throw std::runtime_error("Expression for column index " + to_string(columnIndex) + " is not a column reference expression");
		}
		auto &columnRefExpr = expr->Cast<BoundColumnRefExpression>();
		auto columnBinding = columnRefExpr.binding;
		if (children.empty()) {
			throw std::runtime_error("No children to resolve column binding to attribute info for binding " + binding.ToString());
		}
		return children[0]->resolveColumnBindingToAttributeInfo(columnBinding);
	}
	std::cout << "[LogicalProjection](resolveColumnBindingToAttributeInfo) :: table index did not match projection's table index for binding " << binding.ToString() << std::endl;
	if (children.empty()) {
		throw std::runtime_error("No children to resolve column binding to attribute info for binding " + binding.ToString());
	}
	return children[0]->resolveColumnBindingToAttributeInfo(binding);
}

void LogicalProjection::Walk(int depth) {
	string indent = string(depth * 4, ' ');
	std::cout << indent << "[LogicalProjection](Walk) type :: " << LogicalOperatorToString(type) << std::endl;
	std::cout << indent << "[LogicalProjection](Walk) table_index :: " << table_index << std::endl;

	std::cout << indent << "[LogicalProjection](Walk) Expressions :: " << std::endl;
	for (const auto &ex : this->expressions) {
		printExpression(ex, depth + 1);
	}
	auto column_bindings = this->GetColumnBindings();
	std::cout << indent << "[LogicalProjection](Walk) Column Bindings :: " << ColumnBindingsToString(column_bindings) << std::endl;
	std::cout << std::endl;
	std::cout << indent << "[LogicalProjection](Walk) children length :: " << children.size() << std::endl;

for (const auto &child : children) {
		child->Walk(depth + 1);
	}
}

vector<ColumnBinding> LogicalProjection::GetColumnBindings() {
	return GenerateColumnBindings(table_index, expressions.size());
}

void LogicalProjection::ResolveTypes() {
	for (auto &expr : expressions) {
		types.push_back(expr->return_type);
	}
}

vector<idx_t> LogicalProjection::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalProjection::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

void LogicalProjection::resolveMLIRValue(MLIRTranslationContext& translationContext, MLIRTranslationContext::ResolverScope& scope) {
	std::cout << "[LogicalProjection](resolveMLIRValue) :: " << LogicalOperatorToString(type) << std::endl;
	for (const auto &child: children) {
		child->resolveMLIRValue(translationContext, scope);
	}
}

} // namespace duckdb
