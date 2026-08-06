#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/value_operations/value_operations.hpp"

namespace duckdb {

BoundConstantExpression::BoundConstantExpression(Value value_p)
    : Expression(ExpressionType::VALUE_CONSTANT, ExpressionClass::BOUND_CONSTANT, value_p.type()),
      value(std::move(value_p)) {
}

mlir::Value BoundConstantExpression::translateExpression(MLIRTranslationContext& translationContext,
	mlir::OpBuilder& builder, LogicalOperator *op) {
	auto& mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto loc = builder.getUnknownLoc();

	switch (return_type.id()) {
	case LogicalTypeId::INTEGER: {
		return builder.create<lingodb::compiler::dialect::db::ConstantOp>(
			loc, builder.getI32Type(), builder.getI32IntegerAttr(value.GetValue<int32_t>()));
		break;
	}
	case LogicalTypeId::BIGINT: {
		return builder.create<lingodb::compiler::dialect::db::ConstantOp>(
			loc, builder.getI64Type(), builder.getI64IntegerAttr(value.GetValue<int64_t>()));
		break;
	}
	case LogicalTypeId::VARCHAR: {
		auto strVal = value.GetValue<string>();
		// Match LingoDB's own SQL frontend (see lingodb_ext Parser.cpp T_String case):
		// string literals are emitted as !db.char<len>, not !db.string.
		//
		// This is required because LingoDB's QueryGraph::buildEvalExpr asserts
		// that char<1> values are never db.cast-ed to !db.string — the
		// expectation is that the constant on the other side of the comparison
		// is itself a char<1>, so no cast is ever emitted on the column value.
		// When a string literal is later compared against a genuine !db.string
		// column (e.g. c_comment), SQLTypeInference::castValueToType mutates
		// this ConstantOp's type in place to !db.string instead of inserting a
		// db.cast, so the invariant still holds.
		//
		// Empty string literals fall back to !db.string since char<0> is not a
		// meaningful LingoDB type.
		mlir::Type strType;
		if (strVal.empty()) {
			strType = lingodb::compiler::dialect::db::StringType::get(builder.getContext());
		} else {
			strType = lingodb::compiler::dialect::db::CharType::get(builder.getContext(), strVal.size());
		}
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
	case LogicalTypeId::DECIMAL: {
		auto type = getMLIRTypeFromDuckDBLogicalType(value.type(), builder.getContext());
		return builder.create<lingodb::compiler::dialect::db::ConstantOp>(
			loc, type,
			builder.getStringAttr(value.ToString()));
	}
	case LogicalTypeId::DOUBLE: {
		// The optimizer's constant folding can emit raw DOUBLE literals (e.g. the "0.2" in
		// Q17's "0.2 * avg(...)"), where the unoptimized plan wrapped the same literal in a
		// CAST over a small DECIMAL constant. We must NOT map DOUBLE to the wide
		// !db.decimal<38,19> that getMLIRTypeFromDuckDBLogicalType() uses for columns:
		// multiplying such a wide constant overflows the decimal precision and yields a
		// garbage result. Instead derive a tight !db.decimal<precision,scale> straight from
		// the literal's shortest string form, matching how LingoDB's own frontend represents
		// these (e.g. 0.2 -> !db.decimal<2,1>).
		std::string s = value.ToString();
		if (s.find('e') == string::npos && s.find('E') == string::npos && s.find("inf") == string::npos &&
		    s.find("nan") == string::npos) {
			std::string body = s;
			if (!body.empty() && (body[0] == '-' || body[0] == '+')) {
				body = body.substr(1);
			}
			auto dot = body.find('.');
			uint8_t scale = 0;
			uint8_t precision = 1;
			if (dot == string::npos) {
				precision = static_cast<uint8_t>(std::max<size_t>(1, body.size()));
			} else {
				auto intDigits  = dot;
				auto fracDigits = body.size() - dot - 1;
				scale     = static_cast<uint8_t>(fracDigits);
				precision = static_cast<uint8_t>(std::max<size_t>(1, intDigits + fracDigits));
			}
			if (precision < scale) {
				precision = scale;
			}
			auto type = lingodb::compiler::dialect::db::DecimalType::get(builder.getContext(), precision, scale);
			return builder.create<lingodb::compiler::dialect::db::ConstantOp>(
				loc, type, builder.getStringAttr(s));
		}
		// Fallback for scientific/special forms: use the column-style wide decimal mapping.
		auto type = getMLIRTypeFromDuckDBLogicalType(value.type(), builder.getContext());
		return builder.create<lingodb::compiler::dialect::db::ConstantOp>(
			loc, type,
			builder.getStringAttr(s));
	}
	default: {
		// std::cout << "[BoundConstantExpression::translateExpression] Unhandled constant type :: " << value.type().ToString()
			// << std::endl;
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
