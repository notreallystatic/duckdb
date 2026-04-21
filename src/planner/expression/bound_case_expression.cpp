#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/parser/expression/case_expression.hpp"
#include "duckdb/common/types.hpp"

#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <iostream>

namespace duckdb {

namespace db = lingodb::compiler::dialect::db;

// Pick the "wider" of two decimal types — prefer higher precision, then higher scale.
// Non-decimal types fall back to the left type.
static mlir::Type commonDecimalType(mlir::Type a, mlir::Type b) {
    auto da = mlir::dyn_cast<db::DecimalType>(a);
    auto db_ = mlir::dyn_cast<db::DecimalType>(b);
    if (!da || !db_) return a;
    if (da.getP() > db_.getP()) return a;
    if (db_.getP() > da.getP()) return b;
    return da.getS() >= db_.getS() ? a : b;
}

// Cast value to targetType only if needed and safe (upcast only for decimals).
static mlir::Value castToCommon(mlir::OpBuilder &builder, mlir::Value val, mlir::Type target) {
    if (val.getType() == target) return val;
    return builder.create<db::CastOp>(builder.getUnknownLoc(), target, val);
}

// Recursively translate CASE checks starting at index `idx`.
static mlir::Value translateCaseChecks(
    const std::vector<BoundCaseCheck> &checks,
    const std::unique_ptr<Expression> &elseExpr,
    MLIRTranslationContext &ctx,
    mlir::OpBuilder &builder,
    LogicalOperator *op,
    size_t idx)
{
    auto loc = builder.getUnknownLoc();

    // Build then block
    auto *thenBlock = new mlir::Block();
    mlir::OpBuilder thenBuilder(builder.getContext());
    thenBuilder.setInsertionPointToStart(thenBlock);
    mlir::Value thenVal = checks[idx].then_expr->translateExpression(ctx, thenBuilder, op);

    // Build else block
    auto *elseBlock = new mlir::Block();
    mlir::OpBuilder elseBuilder(builder.getContext());
    elseBuilder.setInsertionPointToStart(elseBlock);
    mlir::Value elseVal;
    if (idx + 1 >= checks.size()) {
        elseVal = elseExpr->translateExpression(ctx, elseBuilder, op);
    } else {
        elseVal = translateCaseChecks(checks, elseExpr, ctx, elseBuilder, op, idx + 1);
    }

    // Pick the widest common type to avoid narrowing casts (which trigger LingoDB lowering bugs).
    mlir::Type common = commonDecimalType(thenVal.getType(), elseVal.getType());

    thenVal = castToCommon(thenBuilder, thenVal, common);
    elseVal = castToCommon(elseBuilder, elseVal, common);
    thenBuilder.create<mlir::scf::YieldOp>(loc, thenVal);
    elseBuilder.create<mlir::scf::YieldOp>(loc, elseVal);

    // Translate condition
    auto cond = checks[idx].when_expr->translateExpression(ctx, builder, op);
    cond = builder.create<db::DeriveTruth>(loc, cond);

    auto ifOp = builder.create<mlir::scf::IfOp>(loc, common, cond, true);
    ifOp.getThenRegion().getBlocks().clear();
    ifOp.getElseRegion().getBlocks().clear();
    ifOp.getThenRegion().push_back(thenBlock);
    ifOp.getElseRegion().push_back(elseBlock);
    return ifOp.getResult(0);
}

mlir::Value BoundCaseExpression::translateExpression(
    MLIRTranslationContext &ctx, mlir::OpBuilder &builder, LogicalOperator *op)
{
    std::cout << "[BoundCaseExpression::translateExpression] Translating CASE expression with "
              << case_checks.size() << " check(s)" << std::endl;
    D_ASSERT(!case_checks.empty());
    return translateCaseChecks(case_checks, else_expr, ctx, builder, op, 0);
}

BoundCaseExpression::BoundCaseExpression(LogicalType type)
    : Expression(ExpressionType::CASE_EXPR, ExpressionClass::BOUND_CASE, std::move(type)) {
}

BoundCaseExpression::BoundCaseExpression(unique_ptr<Expression> when_expr, unique_ptr<Expression> then_expr,
                                         unique_ptr<Expression> else_expr_p)
    : Expression(ExpressionType::CASE_EXPR, ExpressionClass::BOUND_CASE, then_expr->return_type),
      else_expr(std::move(else_expr_p)) {
	BoundCaseCheck check;
	check.when_expr = std::move(when_expr);
	check.then_expr = std::move(then_expr);
	case_checks.push_back(std::move(check));
}

string BoundCaseExpression::ToString() const {
	return CaseExpression::ToString<BoundCaseExpression, Expression>(*this);
}

bool BoundCaseExpression::Equals(const BaseExpression &other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<BoundCaseExpression>();
	if (case_checks.size() != other.case_checks.size()) {
		return false;
	}
	for (idx_t i = 0; i < case_checks.size(); i++) {
		if (!Expression::Equals(*case_checks[i].when_expr, *other.case_checks[i].when_expr)) {
			return false;
		}
		if (!Expression::Equals(*case_checks[i].then_expr, *other.case_checks[i].then_expr)) {
			return false;
		}
	}
	if (!Expression::Equals(*else_expr, *other.else_expr)) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundCaseExpression::Copy() const {
	auto new_case = make_uniq<BoundCaseExpression>(return_type);
	for (auto &check : case_checks) {
		BoundCaseCheck new_check;
		new_check.when_expr = check.when_expr->Copy();
		new_check.then_expr = check.then_expr->Copy();
		new_case->case_checks.push_back(std::move(new_check));
	}
	new_case->else_expr = else_expr->Copy();

	new_case->CopyProperties(*this);
	return std::move(new_case);
}

} // namespace duckdb
