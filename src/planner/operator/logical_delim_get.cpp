#include "duckdb/planner/operator/logical_delim_get.hpp"

#include "duckdb/main/config.hpp"
#include "duckdb/planner/column_binding.hpp"

namespace duckdb {

vector<idx_t> LogicalDelimGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalDelimGet::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

MLIRAttributeInfo& LogicalDelimGet::resolveColumnBindingToAttributeInfo(ColumnBinding& binding) {
	if (binding.table_index == table_index && binding.column_index < mlirAttributeInfos.size()) {
		return mlirAttributeInfos[binding.column_index];
	}
	throw std::runtime_error("[LogicalDelimGet] Cannot resolve binding " + binding.ToString()
	                         + " (table_index=" + std::to_string(table_index)
	                         + ", attrs=" + std::to_string(mlirAttributeInfos.size()) + ")");
}

void LogicalDelimGet::resolveMLIRValue(MLIRTranslationContext&, MLIRTranslationContext::ResolverScope&) {
	// A DELIM_GET never materializes its own relation. For a value-carrying DELIM_JOIN the
	// owning LogicalComparisonJoin sets aliasedRelation to the (distinct, renamed) correlated
	// keys before resolving this subtree; we simply alias it here.
	if (!aliasedRelation) {
		throw std::runtime_error("[LogicalDelimGet] resolveMLIRValue called before aliasedRelation was set");
	}
	this->mlirValue = aliasedRelation;
}

} // namespace duckdb
