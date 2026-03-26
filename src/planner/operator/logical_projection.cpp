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

string LogicalProjection::resolveColumnBinding(ColumnBinding& binding) {
	if (binding.table_index > 0) {
		auto newBinding = ColumnBinding(binding.table_index - 1, binding.column_index);
		return children.empty() ? "" : children[0]->resolveColumnBinding(newBinding);
	}
	string tableName = this->resolveTableIndex(binding.table_index);
	auto columnIndex = binding.column_index;
	if (expressions.size() >= columnIndex) {
		auto expr = expressions[columnIndex].get();
		if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
			auto &columnRefExpr = expr->Cast<BoundColumnRefExpression>();
			auto columnName = columnRefExpr.ToString();
			std::cout << "[LogicalProjection](resolveColumnBinding) :: Resolving column binding for table index " << binding.table_index
			          << " and column index " << columnIndex << " with column name " << columnName << std::endl;
			return tableName.empty() ? columnName : tableName + "." + columnName;
		}
	}
	return children.empty() ? "" : children[0]->resolveColumnBinding(binding);
}
} // namespace duckdb
