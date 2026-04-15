#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/expression_util.hpp"

namespace duckdb {

BoundConjunctionExpression::BoundConjunctionExpression(ExpressionType type)
    : Expression(type, ExpressionClass::BOUND_CONJUNCTION, LogicalType::BOOLEAN) {
}

mlir::Value BoundConjunctionExpression::translateExpression(MLIRTranslationContext &translationContext,
															mlir::OpBuilder &predBuilder, LogicalOperator *op) {

    auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto moduleOp = mlirContainerInstance.getModuleOp();
	lingodb::compiler::dialect::tuples::ColumnManager &attrManager =
	    moduleOp->getContext()
	        ->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()
	        ->getColumnManager();
	auto loc = predBuilder.getUnknownLoc();

    std::vector<mlir::Value> childExprs;
	for (auto& child : children) {
		childExprs.push_back(child->translateExpression(translationContext, predBuilder, op));
	}

	switch (type) {
	case ExpressionType::CONJUNCTION_AND: {
		return predBuilder.create<lingodb::compiler::dialect::db::AndOp>(loc, childExprs);
	case ExpressionType::CONJUNCTION_OR: {
		return predBuilder.create<lingodb::compiler::dialect::db::OrOp>(loc, childExprs);
	}
	default: {
		std::cout << "[BoundConjunctionExpression::translateExpression] Unhandled conjunction type :: " <<
			ExpressionTypeToString(type) << std::endl;
		throw std::runtime_error("Unhandled conjunction type");
	}
	}
	}
}

BoundConjunctionExpression::BoundConjunctionExpression(ExpressionType type, unique_ptr<Expression> left,
                                                       unique_ptr<Expression> right)
    : BoundConjunctionExpression(type) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

string BoundConjunctionExpression::ToString() const {
	return ConjunctionExpression::ToString<BoundConjunctionExpression, Expression>(*this);
}

bool BoundConjunctionExpression::Equals(const BaseExpression &other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<BoundConjunctionExpression>();
	return ExpressionUtil::SetEquals(children, other.children);
}

bool BoundConjunctionExpression::PropagatesNullValues() const {
	return false;
}

unique_ptr<Expression> BoundConjunctionExpression::Copy() const {
	auto copy = make_uniq<BoundConjunctionExpression>(type);
	for (auto &expr : children) {
		copy->children.push_back(expr->Copy());
	}
	copy->CopyProperties(*this);
	return std::move(copy);
}

} // namespace duckdb
