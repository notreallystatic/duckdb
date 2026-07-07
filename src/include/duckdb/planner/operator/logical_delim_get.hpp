//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_delim_get.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! LogicalDelimGet represents a duplicate eliminated scan belonging to a DelimJoin
class LogicalDelimGet : public LogicalOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_DELIM_GET;

public:
	LogicalDelimGet(idx_t table_index, vector<LogicalType> types)
	    : LogicalOperator(LogicalOperatorType::LOGICAL_DELIM_GET), table_index(table_index) {
		D_ASSERT(!types.empty());
		chunk_types = std::move(types);
	}

	//! The table index in the current bind context
	idx_t table_index;
	//! The types of the chunk
	vector<LogicalType> chunk_types;
	//! For a value-carrying DELIM_JOIN (scalar/aggregate correlated subquery, e.g. Q17)
	//! the owning LogicalComparisonJoin resolves the distinct correlated keys into a real
	//! relation and stashes it here before resolving this DELIM_GET's subtree. resolveMLIRValue
	//! then simply aliases it — no new relation is materialized for the DELIM_GET itself.
	mlir::Value aliasedRelation;

public:
	vector<ColumnBinding> GetColumnBindings() override {
		return GenerateColumnBindings(table_index, chunk_types.size());
	}

	void Serialize(Serializer &serializer) const override;
	static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);
	vector<idx_t> GetTableIndex() const override;
	string GetName() const override;

	MLIRAttributeInfo& resolveColumnBindingToAttributeInfo(ColumnBinding& binding) override;
	void resolveMLIRValue(MLIRTranslationContext& context, MLIRTranslationContext::ResolverScope& scope) override;

protected:
	void ResolveTypes() override {
		// types are resolved in the constructor
		this->types = chunk_types;
	}
};
} // namespace duckdb
