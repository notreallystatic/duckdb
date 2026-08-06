#include "duckdb/planner/operator/logical_materialized_cte.hpp"

#include "lingodb/execution/Frontend.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/IR/Builders.h"

namespace duckdb {

InsertionOrderPreservingMap<string> LogicalMaterializedCTE::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["CTE Name"] = ctename;
	result["Table Index"] = StringUtil::Format("%llu", table_index);
	SetParamsEstimatedCardinality(result);
	return result;
}

vector<idx_t> LogicalMaterializedCTE::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

MLIRAttributeInfo& LogicalMaterializedCTE::resolveColumnBindingToAttributeInfo(ColumnBinding& binding) {
	return children[1]->resolveColumnBindingToAttributeInfo(binding);
}

void LogicalMaterializedCTE::resolveMLIRValue(MLIRTranslationContext& context, MLIRTranslationContext::ResolverScope& scope) {
	// std::cout << "[LogicalMaterializedCTE](resolveMLIRValue) :: CTE name=" << ctename
	          // << " table_index=" << table_index << std::endl;

	// Resolve the CTE body (child[0]: PROJECTION → AGGREGATE → FILTER → SCAN).
	children[0]->parentColumnBindings = this->parentColumnBindings;
	children[0]->resolveMLIRValue(context, scope);

	// Store the CTE body in the context so CTE_REF nodes can emit relalg.renaming.
	MLIRTranslationContext::CTEBodyInfo bodyInfo;
	bodyInfo.bodyValue = children[0]->getMLIRValue();
	bodyInfo.cteName   = ctename;

	auto bodyBindings = children[0]->GetColumnBindings();
	for (auto& binding : bodyBindings) {
		bodyInfo.columns.push_back(children[0]->resolveColumnBindingToAttributeInfo(binding).column);
	}
	context.cteValues[table_index] = std::move(bodyInfo);
	// std::cout << "[LogicalMaterializedCTE](resolveMLIRValue) :: Stored CTE body with "
	          // << bodyBindings.size() << " column(s)" << std::endl;

	// Resolve the main query (child[1]) — it contains CTE_REF nodes that read cteValues.
	children[1]->parentColumnBindings = this->parentColumnBindings;
	children[1]->resolveMLIRValue(context, scope);

	this->mlirValue = children[1]->getMLIRValue();
}

} // namespace duckdb
