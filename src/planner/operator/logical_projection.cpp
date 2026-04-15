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
		if (this->hasMLIRResolutionSkipped) {
			std::cout << "[LogicalProjection](resolveColumnBindingToAttributeInfo) :: MLIR resolution has been skipped for this projection, returning default MLIR attribute info for binding " << binding.ToString() << std::endl;
			if (!defaultMLIRAttributeInfo) {
				throw std::runtime_error("Default MLIR attribute info is not set for this projection");
			}
			return *defaultMLIRAttributeInfo;
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
	/*
		For subqueries with scalar result like `SELECT * FROM t1 WHERE t1.a > (SELECT avg(t1.a) FROM t1) LIMIT 10`
		DuckDB wraps this subquery with an aggregate over `avg` and also adds a `count`.
		It then adds a projection on top of that with a `CASE` expression to handle the case where the subquery returns no rows (i.e. `avg` returns NULL) and the case where it does return rows (i.e. `avg` returns a non-NULL value).
		The projection looks something like this:
		```
		Projection
		├── CASE
		│   ├── WHEN count > 1 THEN error(`More than one row returned...`)
		│   └── ELSE avg
		└── Aggregate: count_start(), avg
		|___Projection (avg)
		|___Aggregate (avg)
		|___Scan (t1)

		```
		We need to ignore this in case of codegen because we don't have any mechanism to throw this error in the generated code.
	*/
	auto hasCaseExpression = std::any_of(expressions.begin(), expressions.end(), [](const auto &expr) {
		return expr->type == ExpressionType::CASE_EXPR;
	});
	if (hasCaseExpression) {
		std::cout << "[LogicalProjection](resolveMLIRValue) :: Projection has a case expression, skipping MLIR value resolution for this node and its child nodes" << std::endl;
		auto nextChild = children.empty() ? nullptr : children[0].get();
		if (nextChild) {
			std::cout << "[LogicalProjection](resolveMLIRValue) :: Checking child node of projection for case expression :: " << LogicalOperatorToString(nextChild->type) << std::endl;
			if (nextChild->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
				std::cout << "[LogicalProjection](resolveMLIRValue) :: Child node of projection is an aggregate and group by node, skipping MLIR value resolution for this child node as well" << std::endl;
				auto nextNextChild = nextChild->children.empty() ? nullptr : nextChild->children[0].get();
				if (nextNextChild) {
					std::cout << "[LogicalProjection](resolveMLIRValue) :: Checking child node of aggregate and group by node for case expression :: " << LogicalOperatorToString(nextNextChild->type) << std::endl;
					nextNextChild->resolveMLIRValue(translationContext, scope);
					auto columnBindings = nextNextChild->GetColumnBindings();
					std::cout << "[LogicalProjection](resolveMLIRValue) :: Column bindings of child node of aggregate and group by node :: " << ColumnBindingsToString(columnBindings) << std::endl;
					auto &resolvedColumnBinding = nextNextChild->resolveColumnBindingToAttributeInfo(columnBindings[0]);
					this->defaultMLIRAttributeInfo = &resolvedColumnBinding;
					this->hasMLIRResolutionSkipped = true;
					return;
				}
			}
		}
	}
	for (const auto &child: children) {
		child->resolveMLIRValue(translationContext, scope);
	}
}

} // namespace duckdb
