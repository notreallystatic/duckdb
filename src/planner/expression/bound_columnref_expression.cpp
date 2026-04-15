#include "duckdb/planner/expression/bound_columnref_expression.hpp"

#include "duckdb/common/types/hash.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

BoundColumnRefExpression::BoundColumnRefExpression(string alias_p, LogicalType type, ColumnBinding binding, idx_t depth)
    : Expression(ExpressionType::BOUND_COLUMN_REF, ExpressionClass::BOUND_COLUMN_REF, std::move(type)),
      binding(binding), depth(depth) {
	this->alias = std::move(alias_p);
}

mlir::Value BoundColumnRefExpression::translateExpression(MLIRTranslationContext &translationContext, mlir::OpBuilder &predBuilder, LogicalOperator *op) {
	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto moduleOp = mlirContainerInstance.getModuleOp();
	lingodb::compiler::dialect::tuples::ColumnManager &attrManager =
	    moduleOp->getContext()
	        ->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()
	        ->getColumnManager();
	auto loc = predBuilder.getUnknownLoc();

	auto column_name = this->ToString();
	auto binding = this->binding;
	std::cout << "[BoundColumnRefExpression](translateExpression) :: column name :: " << column_name << " binding :: " << binding.ToString() << std::endl;
	auto& columnAttr = op->resolveColumnBindingToAttributeInfo(binding);
	auto currentTuple = translationContext.getCurrentTuple();
	return predBuilder.create<lingodb::compiler::dialect::tuples::GetColumnOp>(
		loc, columnAttr.column->type, attrManager.createRef(columnAttr.column), translationContext.getCurrentTuple());
}

BoundColumnRefExpression::BoundColumnRefExpression(LogicalType type, ColumnBinding binding, idx_t depth)
    : BoundColumnRefExpression(string(), std::move(type), binding, depth) {
}

unique_ptr<Expression> BoundColumnRefExpression::Copy() const {
	return make_uniq<BoundColumnRefExpression>(alias, return_type, binding, depth);
}

hash_t BoundColumnRefExpression::Hash() const {
	auto result = Expression::Hash();
	result = CombineHash(result, duckdb::Hash<uint64_t>(binding.column_index));
	result = CombineHash(result, duckdb::Hash<uint64_t>(binding.table_index));
	return CombineHash(result, duckdb::Hash<uint64_t>(depth));
}

bool BoundColumnRefExpression::Equals(const BaseExpression &other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<BoundColumnRefExpression>();
	return other.binding == binding && other.depth == depth;
}

string BoundColumnRefExpression::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return StringUtil::Format("%s (%s)", binding.ToString(), return_type.ToString());
	}
#endif
	return Expression::GetName();
}

string BoundColumnRefExpression::ToString() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return binding.ToString();
	}
#endif
	if (!alias.empty()) {
		return alias;
	}
	return binding.ToString();
}

} // namespace duckdb
