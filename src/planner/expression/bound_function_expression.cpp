#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/function/function_serialization.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/function/lambda_functions.hpp"
#include "duckdb/function/scalar/string_functions.hpp"

namespace duckdb {

BoundFunctionExpression::BoundFunctionExpression(LogicalType return_type, ScalarFunction bound_function,
                                                 vector<unique_ptr<Expression>> arguments,
                                                 unique_ptr<FunctionData> bind_info, bool is_operator)
    : Expression(ExpressionType::BOUND_FUNCTION, ExpressionClass::BOUND_FUNCTION, std::move(return_type)),
      function(std::move(bound_function)), children(std::move(arguments)), bind_info(std::move(bind_info)),
      is_operator(is_operator) {
	D_ASSERT(!function.name.empty());
}

mlir::Value BoundFunctionExpression::translateExpression(MLIRTranslationContext& translationContext,
	mlir::OpBuilder& builder, LogicalOperator *op) {
	auto loc = builder.getUnknownLoc();
	std::cout << "[BoundFunctionExpression::translateExpression] Translating function :: " << function.name << " children :: " << children.size() << std::endl;
	// to_days(CAST(trunc(CAST('90' AS DOUBLE)) AS INTEGER)))
	// For now, just add support for the above function.

	std::cout << "[BoundFunctionExpression::translateExpression] Function has " << children.size() << " arguments" << std::endl;
	for (size_t i = 0; i < children.size(); i++) {
		std::cout << "[BoundFunctionExpression::translateExpression] Argument " << i << " :: " << children[i]->ToString() << std::endl;
	}

	if (function.name == "to_days") {
		Value childValue;
		string childValueStr;
		D_ASSERT(children.size() == 1);
		bool isChildEvaluated = ExpressionExecutor::TryEvaluateScalar(*translationContext.clientContext, *children[0], childValue);
		std::cout << "[BoundFunctionExpression::translateExpression] Is child function evaluated? :: " << isChildEvaluated << std::endl;
		childValueStr = childValue.ToString();
		return builder.create<lingodb::compiler::dialect::db::ConstantOp>(
			loc, lingodb::compiler::dialect::db::IntervalType::get(builder.getContext(), lingodb::compiler::dialect::db::IntervalUnitAttr::daytime),
			builder.getStringAttr(childValueStr + "days"));
	} else if (function.name == "to_years") {
		Value childValue;
		string childValueStr;
		D_ASSERT(children.size() == 1);
		bool isChildEvaluated = ExpressionExecutor::TryEvaluateScalar(*translationContext.clientContext, *children[0], childValue);
		std::cout << "[BoundFunctionExpression::translateExpression] Is child function evaluated? :: " << isChildEvaluated << std::endl;
		auto yearValue = childValue.GetValueUnsafe<int32_t>();
		auto monthValue = yearValue * 12; // Convert years to months
		childValueStr = to_string(monthValue);
		return builder.create<lingodb::compiler::dialect::db::ConstantOp>(
			loc, lingodb::compiler::dialect::db::IntervalType::get(builder.getContext(), lingodb::compiler::dialect::db::IntervalUnitAttr::months),
			builder.getStringAttr(childValueStr));
	}
	 else if (function.name == "date_part") {
		auto datePart = children[0]->translateExpression(translationContext, builder, op);
		auto columnVal = children[1]->translateExpression(translationContext, builder, op);
		return builder.create<lingodb::compiler::dialect::db::RuntimeCall>(loc, builder.getI64Type(), "ExtractFromDate", mlir::ValueRange({ datePart, columnVal })).getRes();
	}
	else if (function.name == "-") {
		if (children.size() == 2) {
			// If left is date and right is a bound_function to_days, then we are doing date subtraction
			if (children[0]->return_type.id() == LogicalTypeId::DATE &&
				children[1]->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
				Value dateValue;
				string dateValueStr;
				bool isDateEvaluated = ExpressionExecutor::TryEvaluateScalar(*translationContext.clientContext, *children[0], dateValue);
				std::cout << "[BoundFunctionExpression::translateExpression] Is date child function evaluated? :: " << isDateEvaluated << std::endl;
				dateValueStr = dateValue.ToString();
				auto dateType = lingodb::compiler::dialect::db::DateType::get(builder.getContext(), lingodb::compiler::dialect::db::DateUnitAttr::day);
				auto leftVal = builder.create<lingodb::compiler::dialect::db::ConstantOp>(
					loc, dateType, builder.getStringAttr(dateValueStr));
				auto rightVal = children[1]->translateExpression(translationContext, builder, op);
				return builder.create<lingodb::compiler::dialect::db::RuntimeCall>(loc, leftVal.getType(), "DateSubtract", mlir::ValueRange({ leftVal, rightVal })).getRes();
			} else  {
				auto leftVal = children[0]->translateExpression(translationContext, builder, op);
				auto rightVal = children[1]->translateExpression(translationContext, builder, op);
				return builder.create<lingodb::compiler::dialect::db::SubOp>(loc, leftVal, rightVal);
			}
		}
	}
	else if (function.name == LikeFun::Name) {
		D_ASSERT(children.size() == 2);
		auto leftVal = children[0]->translateExpression(translationContext, builder, op);
		auto rightVal = children[1]->translateExpression(translationContext, builder, op);
		// Match LingoDB's own frontend (Parser.cpp COMPARE_LIKE): normalize both
		// operands to a common base type before handing them to the Like runtime
		// call. Now that VARCHAR literals are emitted as !db.char<len>, the
		// pattern would otherwise reach the runtime call as char<N>.
		auto likeOperands = lingodb::compiler::frontend::sql::SQLTypeInference::toCommonBaseTypes(
			builder, {leftVal, rightVal});
		return builder.create<lingodb::compiler::dialect::db::RuntimeCall>(loc, builder.getI1Type(), "Like", mlir::ValueRange({ likeOperands[0], likeOperands[1] })).getRes();
	}
	else if (function.name == NotLikeFun::Name) {
		D_ASSERT(children.size() == 2);
		auto leftVal = children[0]->translateExpression(translationContext, builder, op);
		auto rightVal = children[1]->translateExpression(translationContext, builder, op);
		auto likeOperands = lingodb::compiler::frontend::sql::SQLTypeInference::toCommonBaseTypes(
			builder, {leftVal, rightVal});
		auto result = builder.create<lingodb::compiler::dialect::db::RuntimeCall>(loc, builder.getI1Type(), "Like", mlir::ValueRange({ likeOperands[0], likeOperands[1] })).getRes();
		return builder.create<lingodb::compiler::dialect::db::NotOp>(loc, result).getRes();
	}
	else if (function.name == "*") {
		if (children.size() == 2) {
			auto leftVal = children[0]->translateExpression(translationContext, builder, op);
			auto rightVal = children[1]->translateExpression(translationContext, builder, op);
			return builder.create<lingodb::compiler::dialect::db::MulOp>(loc, leftVal, rightVal);
		}
	}
	else if (function.name == "+") {
		if (children.size() == 2) {
			if (children[0]->return_type.id() == LogicalTypeId::DATE &&
				children[1]->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
				Value dateValue;
				string dateValueStr;
				bool isDateEvaluated = ExpressionExecutor::TryEvaluateScalar(*translationContext.clientContext, *children[0], dateValue);
				std::cout << "[BoundFunctionExpression::translateExpression] Is date child function evaluated? :: " << isDateEvaluated << std::endl;
				dateValueStr = dateValue.ToString();
				auto dateType = lingodb::compiler::dialect::db::DateType::get(builder.getContext(), lingodb::compiler::dialect::db::DateUnitAttr::day);
				auto leftVal = builder.create<lingodb::compiler::dialect::db::ConstantOp>(
					loc, dateType, builder.getStringAttr(dateValueStr));
				auto rightVal = children[1]->translateExpression(translationContext, builder, op);
				return builder.create<lingodb::compiler::dialect::db::RuntimeCall>(loc, leftVal.getType(), "DateAdd", mlir::ValueRange({ leftVal, rightVal })).getRes();
			}
			else {
				auto leftVal = children[0]->translateExpression(translationContext, builder, op);
				auto rightVal = children[1]->translateExpression(translationContext, builder, op);
				return builder.create<lingodb::compiler::dialect::db::AddOp>(loc, leftVal, rightVal);
			}
		}
	}
	else if (function.name == "/") {
		if (children.size() == 2) {
			// Strip CAST(x AS DOUBLE) wrappers to keep native decimal precision.
			auto stripDoubleCast = [](Expression *expr) -> Expression * {
				if (expr->expression_class == ExpressionClass::BOUND_CAST &&
				    expr->return_type.id() == LogicalTypeId::DOUBLE) {
					return expr->Cast<BoundCastExpression>().child.get();
				}
				return expr;
			};
			auto leftExpr = stripDoubleCast(children[0].get());
			auto rightExpr = stripDoubleCast(children[1].get());
			auto leftVal = leftExpr->translateExpression(translationContext, builder, op);
			auto rightVal = rightExpr->translateExpression(translationContext, builder, op);
			// Ensure both operands share the same decimal type to avoid mixed i64/i128 lowering.
			namespace db = lingodb::compiler::dialect::db;
			auto leftDecimal = mlir::dyn_cast<db::DecimalType>(leftVal.getType());
			auto rightDecimal = mlir::dyn_cast<db::DecimalType>(rightVal.getType());
			if (leftDecimal && rightDecimal && leftDecimal != rightDecimal) {
				// Cast the narrower operand to the wider type.
				bool leftNarrow = leftDecimal.getP() < rightDecimal.getP() ||
				                  (leftDecimal.getP() == rightDecimal.getP() && leftDecimal.getS() < rightDecimal.getS());
				if (leftNarrow) {
					leftVal = builder.create<db::CastOp>(loc, rightDecimal, leftVal);
				} else {
					rightVal = builder.create<db::CastOp>(loc, leftDecimal, rightVal);
				}
			}
			return builder.create<db::DivOp>(loc, leftVal, rightVal);
		}
	}
	std::cout << "[BoundFunctionExpression::translateExpression] Unhandled function :: " << function.name << std::endl;
	throw std::runtime_error("Unhandled function in MLIR translation :: " + function.name);
}

bool BoundFunctionExpression::IsVolatile() const {
	return function.stability == FunctionStability::VOLATILE ? true : Expression::IsVolatile();
}

bool BoundFunctionExpression::IsConsistent() const {
	return function.stability != FunctionStability::CONSISTENT ? false : Expression::IsConsistent();
}

bool BoundFunctionExpression::IsFoldable() const {
	// functions with side effects cannot be folded: they have to be executed once for every row
	if (function.bind_lambda) {
		// This is a lambda function
		D_ASSERT(bind_info);
		auto &lambda_bind_data = bind_info->Cast<ListLambdaBindData>();
		if (lambda_bind_data.lambda_expr) {
			auto &expr = *lambda_bind_data.lambda_expr;
			if (expr.IsVolatile()) {
				return false;
			}
		}
	}
	return function.stability == FunctionStability::VOLATILE ? false : Expression::IsFoldable();
}

bool BoundFunctionExpression::CanThrow() const {
	if (function.errors == FunctionErrors::CAN_THROW_RUNTIME_ERROR) {
		return true;
	}
	return Expression::CanThrow();
}

string BoundFunctionExpression::ToString() const {
	return FunctionExpression::ToString<BoundFunctionExpression, Expression>(*this, string(), string(), function.name,
	                                                                         is_operator);
}
bool BoundFunctionExpression::PropagatesNullValues() const {
	return function.null_handling == FunctionNullHandling::SPECIAL_HANDLING ? false
	                                                                        : Expression::PropagatesNullValues();
}

hash_t BoundFunctionExpression::Hash() const {
	hash_t result = Expression::Hash();
	return CombineHash(result, function.Hash());
}

bool BoundFunctionExpression::Equals(const BaseExpression &other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<BoundFunctionExpression>();
	if (other.function != function) {
		return false;
	}
	if (!Expression::ListEquals(children, other.children)) {
		return false;
	}
	if (!FunctionData::Equals(bind_info.get(), other.bind_info.get())) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundFunctionExpression::Copy() const {
	vector<unique_ptr<Expression>> new_children;
	new_children.reserve(children.size());
	for (auto &child : children) {
		new_children.push_back(child->Copy());
	}
	unique_ptr<FunctionData> new_bind_info = bind_info ? bind_info->Copy() : nullptr;

	auto copy = make_uniq<BoundFunctionExpression>(return_type, function, std::move(new_children),
	                                               std::move(new_bind_info), is_operator);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundFunctionExpression::Verify() const {
	D_ASSERT(!function.name.empty());
}

void BoundFunctionExpression::Serialize(Serializer &serializer) const {
	Expression::Serialize(serializer);
	serializer.WriteProperty(200, "return_type", return_type);
	serializer.WriteProperty(201, "children", children);
	FunctionSerializer::Serialize(serializer, function, bind_info.get());
	serializer.WriteProperty(202, "is_operator", is_operator);
}

unique_ptr<Expression> BoundFunctionExpression::Deserialize(Deserializer &deserializer) {
	auto return_type = deserializer.ReadProperty<LogicalType>(200, "return_type");
	auto children = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(201, "children");

	auto entry = FunctionSerializer::Deserialize<ScalarFunction, ScalarFunctionCatalogEntry>(
	    deserializer, CatalogType::SCALAR_FUNCTION_ENTRY, children, return_type);
	auto function_return_type = entry.first.return_type;

	auto is_operator = deserializer.ReadProperty<bool>(202, "is_operator");

	if (entry.first.bind_expression) {
		// bind the function expression
		auto &context = deserializer.Get<ClientContext &>();
		auto bind_input = FunctionBindExpressionInput(context, entry.second, children);
		// replace the function expression with the bound expression
		auto bound_expression = entry.first.bind_expression(bind_input);
		if (bound_expression) {
			return bound_expression;
		}
		// Otherwise, fall thorugh and continue on normally
	}
	auto result = make_uniq<BoundFunctionExpression>(std::move(function_return_type), std::move(entry.first),
	                                                 std::move(children), std::move(entry.second));
	result->is_operator = is_operator;
	if (result->return_type != return_type) {
		// return type mismatch - push a cast
		auto &context = deserializer.Get<ClientContext &>();
		return BoundCastExpression::AddCastToType(context, std::move(result), return_type);
	}
	return std::move(result);
}

} // namespace duckdb
