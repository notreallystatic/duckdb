//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/logical_operator.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/enums/logical_operator_type.hpp"
#include "duckdb/common/enums/explain_format.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/insertion_order_preserving_map.hpp"
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


#include <algorithm>
#include <functional>
#include <optional>

namespace duckdb {

//! LogicalOperator is the base class of the logical operators present in the
//! logical query tree
class LogicalOperator {
public:
	explicit LogicalOperator(LogicalOperatorType type);
	LogicalOperator(LogicalOperatorType type, vector<unique_ptr<Expression>> expressions);
	virtual ~LogicalOperator();

	//! The type of the logical operator
	LogicalOperatorType type;
	//! The set of children of the operator
	vector<unique_ptr<LogicalOperator>> children;
	//! The set of expressions contained within the operator, if any
	vector<unique_ptr<Expression>> expressions;
	//! The types returned by this logical operator. Set by calling LogicalOperator::ResolveTypes.
	vector<LogicalType> types;
	//! Estimated Cardinality
	idx_t estimated_cardinality;
	bool has_estimated_cardinality;

	mlir::Value mlirValue; // Node's currect MLIR value

public:
	mlir::Value getMLIRValue();

	// Resolve the current MLIR value of the node and recursively resolve the MLIR values of the children
	virtual void resolveMLIRValue(MLIRTranslationContext&, MLIRTranslationContext::ResolverScope&);
	virtual string resolveColumnBinding(ColumnBinding& binding);
	virtual string resolveTableIndex(idx_t table_index);
	//! Print the operator tree, for debugging purposes
	virtual void Walk(int depth = 0);

	virtual vector<ColumnBinding> GetColumnBindings();
	static string ColumnBindingsToString(const vector<ColumnBinding> &bindings);
	void PrintColumnBindings();
	static vector<ColumnBinding> GenerateColumnBindings(idx_t table_idx, idx_t column_count);
	static vector<LogicalType> MapTypes(const vector<LogicalType> &types, const vector<idx_t> &projection_map);
	static vector<ColumnBinding> MapBindings(const vector<ColumnBinding> &types, const vector<idx_t> &projection_map);

	//! Resolve the types of the logical operator and its children
	void ResolveOperatorTypes();

	virtual string GetName() const;
	virtual InsertionOrderPreservingMap<string> ParamsToString() const;
	virtual string ToString(ExplainFormat format = ExplainFormat::DEFAULT) const;
	DUCKDB_API void Print();
	//! Debug method: verify that the integrity of expressions & child nodes are maintained
	virtual void Verify(ClientContext &context);

	void AddChild(unique_ptr<LogicalOperator> child);
	virtual idx_t EstimateCardinality(ClientContext &context);
	void SetEstimatedCardinality(idx_t _estimated_cardinality);
	void SetParamsEstimatedCardinality(InsertionOrderPreservingMap<string> &result) const;

	virtual void Serialize(Serializer &serializer) const;
	static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);

	virtual unique_ptr<LogicalOperator> Copy(ClientContext &context) const;

	virtual bool RequireOptimizer() const {
		return true;
	}

	//! Allows LogicalOperators to opt out of serialization
	virtual bool SupportSerialization() const {
		return true;
	};

	virtual bool HasProjectionMap() const {
		return false;
	}

	//! Returns the set of table indexes of this operator
	virtual vector<idx_t> GetTableIndex() const;

protected:
	//! Resolve types for this specific operator
	virtual void ResolveTypes() = 0;

public:
	template <class TARGET>
	TARGET &Cast() {
		if (TARGET::TYPE != LogicalOperatorType::LOGICAL_INVALID && type != TARGET::TYPE) {
			throw InternalException("Failed to cast logical operator to type - logical operator type mismatch");
		}
		return reinterpret_cast<TARGET &>(*this);
	}

	template <class TARGET>
	const TARGET &Cast() const {
		if (TARGET::TYPE != LogicalOperatorType::LOGICAL_INVALID && type != TARGET::TYPE) {
			throw InternalException("Failed to cast logical operator to type - logical operator type mismatch");
		}
		return reinterpret_cast<const TARGET &>(*this);
	}
};
} // namespace duckdb
