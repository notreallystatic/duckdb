#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/value_operations/value_operations.hpp"

namespace duckdb {

BoundConstantExpression::BoundConstantExpression(Value value_p)
    : Expression(ExpressionType::VALUE_CONSTANT, ExpressionClass::BOUND_CONSTANT, value_p.type()),
      value(std::move(value_p)) {
}

mlir::Value BoundConstantExpression::translateExpression(MLIRTranslationContext& translationContext,
	mlir::OpBuilder& builder) {
	auto& mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto loc = builder.getUnknownLoc();

	switch (return_type.id()) {
	case LogicalTypeId::INTEGER: {
		return builder.create<lingodb::compiler::dialect::db::ConstantOp>(
			loc, builder.getI32Type(), builder.getI32IntegerAttr(value.GetValue<int32_t>()));
		break;
	}
	case LogicalTypeId::VARCHAR: {
		auto strVal = value.GetValue<string>();
		auto strType = lingodb::compiler::dialect::db::StringType::get(builder.getContext());
		return builder.create<lingodb::compiler::dialect::db::ConstantOp>(loc, strType,
			builder.getStringAttr(strVal));
		break;
	}
	case LogicalTypeId::CHAR: {
		auto strVal = value.GetValue<string>();
		auto strType = lingodb::compiler::dialect::db::CharType::get(builder.getContext(), strVal.size());
		return builder.create<lingodb::compiler::dialect::db::ConstantOp>(loc, strType,
			builder.getStringAttr(strVal));
		break;
	}
	case LogicalTypeId::FLOAT: { // TODO: Testing pending
		string expressionValue = value.ToString();
		auto floatVal = value.GetValue<float>();
		// get the integer part and decimal part of the floatVal
		auto intPart = static_cast<unsigned long>(floatVal);
		auto decimalPart = floatVal - intPart;
		// convert the decimal part to an integer by multiplying it with 10^6
		auto decimalIntPart = static_cast<unsigned long>(decimalPart * 10000000);
		return builder.create<lingodb::compiler::dialect::db::ConstantOp>(
			loc,
			lingodb::compiler::dialect::db::DecimalType::get(builder.getContext(), intPart, decimalIntPart),
			builder.getStringAttr(expressionValue));
	}
	case LogicalTypeId::DATE: {
		auto dateVal = value.GetValue<date_t>();
		auto type = getMLIRTypeFromDuckDBLogicalType(value.type(), builder.getContext());
		return builder.create<lingodb::compiler::dialect::db::ConstantOp>(
			loc, type,
			builder.getStringAttr(value.ToString()));
	}
	default: {
		std::cout << "[BoundConstantExpression::translateExpression] Unhandled constant type :: " << value.type().ToString()
			<< std::endl;
		break;
	}
	}
}

string BoundConstantExpression::ToString() const {
	return value.ToSQLString();
}

bool BoundConstantExpression::Equals(const BaseExpression &other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<BoundConstantExpression>();
	return value.type() == other.value.type() && !ValueOperations::DistinctFrom(value, other.value);
}

hash_t BoundConstantExpression::Hash() const {
	hash_t result = Expression::Hash();
	return CombineHash(value.Hash(), result);
}

unique_ptr<Expression> BoundConstantExpression::Copy() const {
	auto copy = make_uniq<BoundConstantExpression>(value);
	copy->CopyProperties(*this);
	return std::move(copy);
}

} // namespace duckdb
