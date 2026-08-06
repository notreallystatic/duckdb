#include "duckdb/planner/operator/logical_cteref.hpp"

#include "duckdb/main/config.hpp"

#include "lingodb/execution/Frontend.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/IR/Builders.h"

namespace relalg = lingodb::compiler::dialect::relalg;
namespace tuples = lingodb::compiler::dialect::tuples;

namespace duckdb {

InsertionOrderPreservingMap<string> LogicalCTERef::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["CTE Index"] = StringUtil::Format("%llu", cte_index);
	SetParamsEstimatedCardinality(result);
	return result;
}

vector<idx_t> LogicalCTERef::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalCTERef::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

MLIRAttributeInfo& LogicalCTERef::resolveColumnBindingToAttributeInfo(ColumnBinding& binding) {
	idx_t col_idx = binding.column_index;
	if (col_idx < mlirAttributeInfos.size()) {
		return mlirAttributeInfos[col_idx];
	}
	throw std::runtime_error("[LogicalCTERef] Could not resolve column binding " + binding.ToString());
}

void LogicalCTERef::resolveMLIRValue(MLIRTranslationContext& context, MLIRTranslationContext::ResolverScope& scope) {
	// std::cout << "[LogicalCTERef](resolveMLIRValue) :: cte_index=" << cte_index
	          // << " table_index=" << table_index << std::endl;

	auto it = context.cteValues.find(cte_index);
	if (it == context.cteValues.end()) {
		throw std::runtime_error("[LogicalCTERef] CTE body not found for cte_index=" + std::to_string(cte_index));
	}
	auto& bodyInfo = it->second;

	auto& mlirContainer = lingodb::execution::MLIRContainer::getInstance();
	auto& builder = mlirContainer.getBuilder();
	auto& mlirContext = mlirContainer.getContext();
	auto moduleOp = mlirContainer.getModuleOp();
	auto loc = builder.getUnknownLoc();

	tuples::ColumnManager& attrManager =
	    moduleOp->getContext()
	        ->getLoadedDialect<tuples::TupleStreamDialect>()
	        ->getColumnManager();

	// Assign a unique scope name: first use gets @cteName, subsequent uses get @cteName_u_N.
	int useCount = bodyInfo.useCount++;
	std::string scopeName = (useCount == 0) ? bodyInfo.cteName
	                                         : bodyInfo.cteName + "_u_" + std::to_string(useCount);

	// Build the renaming columns: each new col def maps from the existing CTE body column.
	std::vector<mlir::Attribute> renamingCols;
	mlirAttributeInfos.clear();

	for (idx_t i = 0; i < bound_columns.size() && i < bodyInfo.columns.size(); i++) {
		auto* sourceCol = bodyInfo.columns[i];
		auto colName    = bound_columns[i];

		auto fromExisting = builder.getArrayAttr({attrManager.createRef(sourceCol)});
		auto newDef = attrManager.createDef(scopeName, colName, fromExisting);
		newDef.getColumn().type = sourceCol->type;

		renamingCols.push_back(newDef);

		context.mapAttribute(scope, colName, &newDef.getColumn());
		context.mapAttribute(scope, scopeName + "." + colName, &newDef.getColumn());

		mlirAttributeInfos.push_back(MLIRAttributeInfo{scopeName, colName, &newDef.getColumn()});

		// std::cout << "[LogicalCTERef](resolveMLIRValue) :: Renamed col[" << i << "]"
		          // << " -> " << scopeName << "." << colName << std::endl;
	}

	auto renamingOp = builder.create<relalg::RenamingOp>(
	    loc,
	    tuples::TupleStreamType::get(&mlirContext),
	    bodyInfo.bodyValue,
	    builder.getArrayAttr(renamingCols));

	this->mlirValue = renamingOp.getResult();
	// std::cout << "[LogicalCTERef](resolveMLIRValue) :: Created relalg.renaming as @"
	          // << scopeName << std::endl;
}

} // namespace duckdb
