#include "duckdb/planner/operator/logical_aggregate.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/main/config.hpp"

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

LogicalAggregate::LogicalAggregate(idx_t group_index, idx_t aggregate_index, vector<unique_ptr<Expression>> select_list)
    : LogicalOperator(LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY, std::move(select_list)),
      group_index(group_index), aggregate_index(aggregate_index), groupings_index(DConstants::INVALID_INDEX),
      distinct_validity(TupleDataValidityType::CAN_HAVE_NULL_VALUES) {
}

mlir::Type getBaseType(mlir::Type t) {
	if (auto nullableT = mlir::dyn_cast_or_null<lingodb::compiler::dialect::db::NullableType>(t)) {
		return nullableT.getType();
	}
	return t;
}

void LogicalAggregate::AddMLIRSpecific(ClientContext &context, LogicalOperatorType operator_to_process,
                                       unique_ptr<LogicalOperator> &og_tree, MLIRTranslationContext &translationContext,
                                       int depth) {

	if (type != operator_to_process) {
		if (children.size() > 0) {
			children[0]->AddMLIRSpecific(context, operator_to_process, og_tree, translationContext, depth + 1);
		}
		return;
	}
	string indent = string(depth * 4, ' ');
	// FIXME: need groupByAttrs.

	// Currently empty as we are not supporting `group by` yet.
	std::vector<mlir::Attribute> groupByAttrs;

	std::cout << indent << "[LogicalAggregate](AddMLIRSpecific) :: " << LogicalOperatorToString(type) << std::endl;
	std::cout.flush();

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto &mapping = mlirContainerInstance.getColumnMapping();
	auto &builder = mlirContainerInstance.getBuilder();
	auto loc = builder.getUnknownLoc();
	auto module = mlirContainerInstance.getModuleOp();
	// lingodb::compiler::dialect::

	static size_t groupById = 0;
	auto tupleStreamType = lingodb::compiler::dialect::tuples::TupleStreamType::get(builder.getContext());
	auto tupleType = lingodb::compiler::dialect::tuples::TupleType::get(builder.getContext());

	lingodb::compiler::dialect::tuples::ColumnManager &attrManager =
	    module->getContext()
	        ->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()
	        ->getColumnManager();
	auto tupleScope = translationContext.createTupleScope();
	auto *block = new mlir::Block();

	block->addArgument(tupleStreamType, loc);
	block->addArgument(tupleType, loc);

	mlir::Value relation = block->getArgument(0);
	mlir::Value temp;
	mlir::OpBuilder aggrBuilder(builder.getContext());

	aggrBuilder.setInsertionPointToStart(block);
	std::vector<mlir::Value> createdValues;
	std::vector<mlir::Attribute> createdCols;
	// std::unordered_map<std::string, lingodb::compiler::dialect::tuples::Column *> mapping;

	int tempNodeId = 0;
	string tempNodePrefix = "tmp_attr";
	string groupByName = "aggr" + std::to_string(groupById++);

	for (const auto &ex : this->expressions) {
		auto &bound_agg = ex->Cast<BoundAggregateExpression>();
		auto functionName = bound_agg.function.name;

		mlir::Value expr;
		// TODO: verify if this is always the case or not
		std::string colName =
		    bound_agg.children[0]->GetName(); // string columnName = tempNodePrefix + std::to_string(tempNodeId++);
		string columnName = functionName + "(" + colName + ")";
		auto attrDef = attrManager.createDef(groupByName, columnName);

		auto aggrFunc = llvm::StringSwitch<lingodb::compiler::dialect::relalg::AggrFunc>(functionName)
		                    .Case("sum", lingodb::compiler::dialect::relalg::AggrFunc::sum)
		                    .Case("min", lingodb::compiler::dialect::relalg::AggrFunc::min)
		                    .Case("max", lingodb::compiler::dialect::relalg::AggrFunc::max)
		                    .Case("avg", lingodb::compiler::dialect::relalg::AggrFunc::avg)
		                    .Case("count", lingodb::compiler::dialect::relalg::AggrFunc::count)
		                    .Default(lingodb::compiler::dialect::relalg::AggrFunc::count);

		// get column name from the first child expression

		auto *column = translationContext.getAttribute(colName);
		lingodb::compiler::dialect::tuples::ColumnRefAttr refAttr = attrManager.createRef(column);
		mlir::Value curRel = relation;
		mlir::Type aggrResultType;
		if (functionName == "count") {
			aggrResultType = builder.getI64Type();
		} else {
			aggrResultType = refAttr.getColumn().type;
			if (functionName == "avg") {
				auto baseType = getBaseType(aggrResultType);
				// TODO: process this.
			}
			if (!mlir::isa<lingodb::compiler::dialect::db::NullableType>(aggrResultType)) {
				aggrResultType =
				    lingodb::compiler::dialect::db::NullableType::get(builder.getContext(), aggrResultType);
			}
			expr = aggrBuilder.create<lingodb::compiler::dialect::relalg::AggrFuncOp>(loc, aggrResultType, aggrFunc,
			                                                                          curRel, refAttr);
		}
		attrDef.getColumn().type = expr.getType();
		mapping.push_back({columnName, &attrDef.getColumn()});
		createdCols.push_back(attrDef);
		createdValues.push_back(expr);
	}
	auto baseTableOp = mlirContainerInstance.baseTableOp;
	std::cout << indent << "[LogicalAggregate](AddMLIRSpecific) baseTableOp :: ";
	std::cout.flush();
	baseTableOp.print(llvm::outs());
	std::cout << std::endl;
	aggrBuilder.create<lingodb::compiler::dialect::tuples::ReturnOp>(loc, createdValues);
	auto groupByOp = builder.create<lingodb::compiler::dialect::relalg::AggregationOp>(
	    loc, tupleStreamType, baseTableOp, builder.getArrayAttr(groupByAttrs), builder.getArrayAttr(createdCols));
	groupByOp.getAggrFunc().push_back(block);
	mlirContainerInstance.aggrBlock = block;
	mlirContainerInstance.aggrOp = groupByOp.getResult();
}

void LogicalAggregate::Walk(int depth) {
	string indent = string(depth * 4, ' ');
	std::cout << indent << "[LogicalAggregate](Walk) type :: " << LogicalOperatorToString(type) << std::endl;

	std::cout << indent << "[LogicalAggregate](Walk) Groups :: " << std::endl;
	for (const auto &ex : this->groups) {
		printExpression(ex, depth + 1);
	}
	std::cout << std::endl;

	std::cout << indent << "[LogicalAggregate](Walk) group_index :: " << group_index
	          << " aggregate_index :: " << aggregate_index << " groupings_index :: " << groupings_index << std::endl;

	std::cout << indent << "[LogicalAggregate](Walk) Expressions :: " << std::endl;
	for (const auto &ex : this->expressions) {
		auto &bound_agg = ex->Cast<BoundAggregateExpression>();
		std::cout << indent << "  [Aggregate Function] function name :: " << bound_agg.function.name << std::endl;
		for (auto &ch : bound_agg.children) {
			std::cout << indent << "    [Child Expression] :: ";
			printExpression(ch, depth + 2);
		}
		printExpression(ex, depth + 1);
	}
	std::cout << std::endl;

	std::cout << indent << "[LogicalAggregate](Walk) Grouping Functions :: " << std::endl;
	for (const auto &gf : this->grouping_functions) {
		std::cout << indent << "  [Grouping Function] :: ";
		for (const auto &idx : gf) {
			std::cout << idx << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	for (const auto &child : children) {
		child->Walk(depth + 1);
	}
}

void LogicalAggregate::ResolveTypes() {
	D_ASSERT(groupings_index != DConstants::INVALID_INDEX || grouping_functions.empty());
	for (auto &expr : groups) {
		types.push_back(expr->return_type);
	}
	// get the chunk types from the projection list
	for (auto &expr : expressions) {
		types.push_back(expr->return_type);
	}
	for (idx_t i = 0; i < grouping_functions.size(); i++) {
		types.emplace_back(LogicalType::BIGINT);
	}
}

vector<ColumnBinding> LogicalAggregate::GetColumnBindings() {
	D_ASSERT(groupings_index != DConstants::INVALID_INDEX || grouping_functions.empty());
	vector<ColumnBinding> result;
	result.reserve(groups.size() + expressions.size() + grouping_functions.size());
	for (idx_t i = 0; i < groups.size(); i++) {
		result.emplace_back(group_index, i);
	}
	for (idx_t i = 0; i < expressions.size(); i++) {
		result.emplace_back(aggregate_index, i);
	}
	for (idx_t i = 0; i < grouping_functions.size(); i++) {
		result.emplace_back(groupings_index, i);
	}
	return result;
}

InsertionOrderPreservingMap<string> LogicalAggregate::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	string groups_info;
	for (idx_t i = 0; i < groups.size(); i++) {
		if (i > 0) {
			groups_info += "\n";
		}
		groups_info += groups[i]->GetName();
	}
	result["Groups"] = groups_info;

	string expressions_info;
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (i > 0) {
			expressions_info += "\n";
		}
		expressions_info += expressions[i]->GetName();
	}
	result["Expressions"] = expressions_info;
	SetParamsEstimatedCardinality(result);
	return result;
}

idx_t LogicalAggregate::EstimateCardinality(ClientContext &context) {
	if (groups.empty()) {
		// ungrouped aggregate
		return 1;
	}
	return LogicalOperator::EstimateCardinality(context);
}

vector<idx_t> LogicalAggregate::GetTableIndex() const {
	vector<idx_t> result {group_index, aggregate_index};
	if (groupings_index != DConstants::INVALID_INDEX) {
		result.push_back(groupings_index);
	}
	return result;
}

string LogicalAggregate::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() +
		       StringUtil::Format(" #%llu, #%llu, #%llu", group_index, aggregate_index, groupings_index);
	}
#endif
	return LogicalOperator::GetName();
}

} // namespace duckdb
