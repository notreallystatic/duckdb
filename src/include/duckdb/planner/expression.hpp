//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/expression.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/parser/base_expression.hpp"
#include "duckdb/common/types.hpp"

#include "duckdb/mlir_util/mlir_util.hpp"

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/frontend/SQL/Parser.h"
#include "lingodb/runtime/Session.h"

#include "lingodb/execution/Frontend.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"



namespace duckdb {
class BaseStatistics;
class ClientContext;
class LogicalOperator;

//!  The Expression class represents a bound Expression with a return type
class Expression : public BaseExpression {
public:
	Expression(ExpressionType type, ExpressionClass expression_class, LogicalType return_type);
	~Expression() override;

	//! The return type of the expression
	LogicalType return_type;
	//! Expression statistics (if any) - ONLY USED FOR VERIFICATION
	unique_ptr<BaseStatistics> verification_stats;

	virtual mlir::Value translateExpression(MLIRTranslationContext&, mlir::OpBuilder&, LogicalOperator *op);

public:
	bool IsAggregate() const override;
	bool IsWindow() const override;
	bool HasSubquery() const override;
	bool IsScalar() const override;
	bool HasParameter() const override;

	virtual bool IsVolatile() const;
	virtual bool IsConsistent() const;
	virtual bool PropagatesNullValues() const;
	virtual bool IsFoldable() const;
	virtual bool CanThrow() const;

	hash_t Hash() const override;

	bool Equals(const BaseExpression &other) const override {
		if (!BaseExpression::Equals(other)) {
			return false;
		}
		return return_type == reinterpret_cast<const Expression &>(other).return_type;
	}
	static bool Equals(const Expression &left, const Expression &right) {
		return left.Equals(right);
	}
	static bool Equals(const unique_ptr<Expression> &left, const unique_ptr<Expression> &right);
	static bool ListEquals(const vector<unique_ptr<Expression>> &left, const vector<unique_ptr<Expression>> &right);
	//! Create a copy of this expression
	virtual unique_ptr<Expression> Copy() const = 0;

	virtual void Serialize(Serializer &serializer) const;
	static unique_ptr<Expression> Deserialize(Deserializer &deserializer);

protected:
	//! Copy base Expression properties from another expression to this one,
	//! used in Copy method
	void CopyProperties(const Expression &other) {
		type = other.type;
		expression_class = other.expression_class;
		alias = other.alias;
		return_type = other.return_type;
		query_location = other.query_location;
	}
};

void printExpression(const unique_ptr<Expression> &expression, int depth);
} // namespace duckdb
