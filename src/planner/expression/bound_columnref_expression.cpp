#include "duckdb/planner/expression/bound_columnref_expression.hpp"

#include "duckdb/common/types/hash.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/logical_operator.hpp"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"

namespace duckdb {

BoundColumnRefExpression::BoundColumnRefExpression(string alias_p, LogicalType type, ColumnBinding binding, idx_t depth)
    : Expression(ExpressionType::BOUND_COLUMN_REF, ExpressionClass::BOUND_COLUMN_REF, std::move(type)),
      binding(binding), depth(depth) {
	this->alias = std::move(alias_p);
}

mlir::Value BoundColumnRefExpression::translateExpression(MLIRTranslationContext &translationContext, mlir::OpBuilder &predBuilder, LogicalOperator *op) {
	auto column_name = this->ToString();
	auto binding = this->binding;
	std::cout << "[BoundColumnRefExpression](translateExpression) :: column name :: " << column_name << " binding :: " << binding.ToString() << std::endl;

	// Check if this column binding is the mark column from a MARK (EXISTS) join.
	// If so, build the right side inside the predicate block and emit relalg.exists.
	auto existsIt = translationContext.deferredExistsCallbacks.find(binding);
	if (existsIt != translationContext.deferredExistsCallbacks.end()) {
		std::cout << "[BoundColumnRefExpression](translateExpression) :: Found deferred EXISTS callback for binding "
		          << binding.ToString() << ", emitting relalg.exists" << std::endl;
		auto rightValue = existsIt->second(predBuilder);
		translationContext.deferredExistsCallbacks.erase(existsIt);
		return predBuilder.create<lingodb::compiler::dialect::relalg::ExistsOp>(
		    predBuilder.getUnknownLoc(), predBuilder.getI1Type(), rightValue);
	}

	// Check if this is a deferred scalar callback (SINGLE dependent join):
	// builds the right child inside the predicate block, then emits relalg.getscalar.
	auto scalarCbIt = translationContext.deferredScalarCallbacks.find(binding);
	if (scalarCbIt != translationContext.deferredScalarCallbacks.end()) {
		std::cout << "[BoundColumnRefExpression](translateExpression) :: Found deferred scalar callback for binding "
		          << binding.ToString() << ", invoking callback to build subquery inside predicate block" << std::endl;
		auto result = scalarCbIt->second(predBuilder);
		translationContext.deferredScalarCallbacks.erase(scalarCbIt);
		return result;
	}

	// Check if this column binding refers to a deferred scalar subquery
	// If so, emit relalg.getscalar here inside the current predicate block
	auto scalarIt = translationContext.deferredScalarSubqueries.find(binding);
	if (scalarIt != translationContext.deferredScalarSubqueries.end()) {
		std::cout << "[BoundColumnRefExpression](translateExpression) :: Found deferred scalar subquery for binding " << binding.ToString() << ", emitting relalg.getscalar" << std::endl;
		auto &info = scalarIt->second;

		auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
		auto moduleOp = mlirContainerInstance.getModuleOp();
		lingodb::compiler::dialect::tuples::ColumnManager &attrManager =
		    moduleOp->getContext()
		        ->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()
		        ->getColumnManager();

		// Determine the result type: wrap in nullable if not already
		mlir::Type resType = info.column->type;
		if (!mlir::isa<lingodb::compiler::dialect::db::NullableType>(resType)) {
			resType = lingodb::compiler::dialect::db::NullableType::get(predBuilder.getContext(), resType);
		}

		auto getScalarValue = predBuilder.create<lingodb::compiler::dialect::relalg::GetScalarOp>(
		    predBuilder.getUnknownLoc(),
		    resType,
		    attrManager.createRef(info.column),
		    info.subqueryStream);
		return getScalarValue;
	}
	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto moduleOp = mlirContainerInstance.getModuleOp();
	lingodb::compiler::dialect::tuples::ColumnManager &attrManager =
	    moduleOp->getContext()
	        ->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()
	        ->getColumnManager();
	auto loc = predBuilder.getUnknownLoc();
	if (this->depth > 0) {
		std::cout << "[BoundColumnRefExpression](translateExpression) :: Column reference has depth " << depth << ", looking up in parent column bindings" << std::endl;
		if (op->parentColumnBindings.empty()) {
			throw std::runtime_error("No parent column bindings found for column reference with binding " + binding.ToString());
		}
		int depthIndex = this->depth - 1;
 		if (op->parentColumnBindings.size() <= depthIndex) {
			throw std::runtime_error("Column reference with binding " + binding.ToString() + " has depth " + to_string(this->depth) + " but only " + to_string(op->parentColumnBindings.size()) + " parent column bindings available");
		}
		auto& columnAttr = op->parentColumnBindings[depthIndex]->resolveColumnBindingToAttributeInfo(binding);
		std::cout << "[BoundColumnRefExpression](translateExpression) :: Resolved column reference with binding " << binding.ToString() << " to parent attribute " << columnAttr.col_name << std::endl;
		return predBuilder.create<lingodb::compiler::dialect::tuples::GetColumnOp>(
			loc, columnAttr.column->type, attrManager.createRef(columnAttr.column), translationContext.getCurrentTuple());
	}

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
