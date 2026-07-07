#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/parser/expression/comparison_expression.hpp"

namespace duckdb {

BoundComparisonExpression::BoundComparisonExpression(ExpressionType type, unique_ptr<Expression> left,
                                                     unique_ptr<Expression> right)
    : Expression(type, ExpressionClass::BOUND_COMPARISON, LogicalType::BOOLEAN), left(std::move(left)),
      right(std::move(right)) {
}

mlir::Value BoundComparisonExpression::translateExpression(MLIRTranslationContext &translationContext,
															mlir::OpBuilder &predBuilder, LogicalOperator *op) {

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto moduleOp = mlirContainerInstance.getModuleOp();
	lingodb::compiler::dialect::tuples::ColumnManager &attrManager =
	    moduleOp->getContext()
	        ->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()
	        ->getColumnManager();
	auto loc = predBuilder.getUnknownLoc();

	// Strip CAST(x AS DOUBLE) wrappers from decimal operands before translation.
	// DOUBLE maps to decimal<38,19> which inflates toCommonBaseTypes to decimal<51,19>,
	// causing comparison failures. Translating the inner decimal directly lets type
	// inference work on the native precision instead.
	auto stripDoubleCast = [](Expression *expr) -> Expression * {
		if (expr->expression_class == ExpressionClass::BOUND_CAST &&
		    expr->return_type.id() == LogicalTypeId::DOUBLE) {
			auto &inner = expr->Cast<BoundCastExpression>().child;
			if (inner->return_type.id() == LogicalTypeId::DECIMAL) {
				return inner.get();
			}
		}
		return expr;
	};

	auto leftMLIRValue = stripDoubleCast(left.get())->translateExpression(translationContext, predBuilder, op);
	auto rightMLIRValue = stripDoubleCast(right.get())->translateExpression(translationContext, predBuilder, op);

	lingodb::compiler::dialect::db::DBCmpPredicate dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::eq;
	auto comparison_type = GetExpressionType();
	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::eq;
		break;
	case ExpressionType::COMPARE_NOTEQUAL:
		dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::neq;
		break;
	case ExpressionType::COMPARE_LESSTHAN:
		dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::lt;
		break;
	case ExpressionType::COMPARE_GREATERTHAN:
		dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::gt;
		break;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::lte;
		break;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::gte;
		break;
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		// Null-safe equality (DuckDB emits this for DELIM_JOIN conditions, e.g.
		// "p_partkey IS NOT DISTINCT FROM p_partkey"). db's `isa` predicate matches
		// SQL "IS NOT DISTINCT FROM" semantics (NULL = NULL is true) and never itself
		// returns null.
		dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::isa;
		break;
	default:
		std::cout << "[translateExpression] Unhandled comparison type :: " <<
		ExpressionTypeToString(comparison_type) << std::endl;
		throw std::runtime_error("Unhandled comparison type");
	}
	auto ct = lingodb::compiler::frontend::sql::SQLTypeInference::toCommonBaseTypes(predBuilder, {leftMLIRValue, rightMLIRValue});
	return predBuilder.create<lingodb::compiler::dialect::db::CmpOp>(loc, dbPred, ct[0], ct[1]);
}

string BoundComparisonExpression::ToString() const {
	return ComparisonExpression::ToString<BoundComparisonExpression, Expression>(*this);
}

bool BoundComparisonExpression::Equals(const BaseExpression &other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<BoundComparisonExpression>();
	if (!Expression::Equals(*left, *other.left)) {
		return false;
	}
	if (!Expression::Equals(*right, *other.right)) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundComparisonExpression::Copy() const {
	auto copy = make_uniq<BoundComparisonExpression>(type, left->Copy(), right->Copy());
	copy->CopyProperties(*this);
	return std::move(copy);
}

} // namespace duckdb
