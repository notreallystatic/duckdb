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

namespace relalg = lingodb::compiler::dialect::relalg;
namespace tuples = lingodb::compiler::dialect::tuples;

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

idx_t LogicalAggregate::EstimateCardinality(ClientContext &context) { if (groups.empty()) { // ungrouped aggregate
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

relalg::AggrFunc getAggrFunc(const string& functionName) {
	std::cout << "[LogicalAggregate](getAggrFunc) :: functionName :: " << functionName << std::endl;
	return llvm::StringSwitch<relalg::AggrFunc>(functionName)
		.Case("sum", relalg::AggrFunc::sum)
		.Case("sum_no_overflow", relalg::AggrFunc::sum)
		.Case("min", relalg::AggrFunc::min)
		.Case("max", relalg::AggrFunc::max)
		.Case("avg", relalg::AggrFunc::avg)
		.Case("count_star", relalg::AggrFunc::count)
		.Default(relalg::AggrFunc::count);
}

MLIRAttributeInfo& LogicalAggregate::resolveColumnBindingToAttributeInfo(ColumnBinding& binding) {
	if (binding.table_index == aggregate_index) {
		if (binding.column_index >= expressions.size()) {
			std::cout << "[LogicalAggregate](resolveColumnBindingToAttributeInfo) :: Invalid column index " << binding.column_index << " for aggregate_index " << aggregate_index << std::endl;
			throw std::runtime_error("Invalid column index " + std::to_string(binding.column_index) + " for aggregate_index " + std::to_string(aggregate_index));
		}
		return mlirAttributeInfos[binding.column_index];
	}
	else {
		std::cout << "[LogicalAggregate](resolveColumnBindingToAttributeInfo) :: Invalid table index " << binding.table_index << " for LogicalAggregate" << std::endl;
		throw std::runtime_error("Invalid table index " + std::to_string(binding.table_index) + " for LogicalAggregate");
	}
}

/**

SELECT
    l_returnflag,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge
FROM lineitem
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
GROUP BY
    l_returnflag,
    l_linestatus
LIMIT 10;

┌─────────────────────────────┐
│┌───────────────────────────┐│
││ Unoptimized Logical Plan  ││
│└───────────────────────────┘│
└─────────────────────────────┘
┌───────────────────────────┐
│           LIMIT           │
│    ────────────────────   │
└─────────────┬─────────────┘
┌─────────────┴─────────────┐
│         PROJECTION        │
│    ────────────────────   │
│        Expressions:       │
│        l_returnflag       │
│          sum_qty          │
│         sum_charge        │
└─────────────┬─────────────┘
┌─────────────┴─────────────┐
│         AGGREGATE         │
│    ────────────────────   │
│          Groups:          │
│        l_returnflag       │
│        l_linestatus       │
│                           │
│        Expressions:       │
│      sum(l_quantity)      │
│  sum(((l_extendedprice *  │
│ (CAST(1 AS DECIMAL(16,2)) │
│  - l_discount)) * (CAST(1 │
│  AS DECIMAL(16,2)) + l_tax│
│            )))            │
└─────────────┬─────────────┘
┌─────────────┴─────────────┐
│           FILTER          │
│    ────────────────────   │
│        Expressions:       │
│    (CAST(l_shipdate AS    │
│  TIMESTAMP) <= (CAST('1998│
│ -12-01' AS DATE) - to_days│
│  (CAST(trunc(CAST('90' AS │
│   DOUBLE)) AS INTEGER)))) │
└─────────────┬─────────────┘
┌─────────────┴─────────────┐
│          SEQ_SCAN         │
│    ────────────────────   │
│      Table: lineitem      │
│   Type: Sequential Scan   │
└───────────────────────────┘
 */
void LogicalAggregate::resolveMLIRValue(MLIRTranslationContext &translationContext, MLIRTranslationContext::ResolverScope &scope) {
	std::cout << "[LogicalAggregate](resolveMLIRValue) :: Resolving MLIR Value for LogicalAggregate" << std::endl;
	std::cout.flush();

	if (children.size() != 1) {
		std::cout << "[LogicalAggregate](resolveMLIRValue) :: Expected exactly one child for LogicalAggregate but found " << children.size() << std::endl;
		throw std::runtime_error("Expected exactly one child for LogicalAggregate but found " + std::to_string(children.size()));
	}

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto &builder = mlirContainerInstance.getBuilder();
	auto &context = mlirContainerInstance.getContext();
	auto module = mlirContainerInstance.getModuleOp();
	auto loc = builder.getUnknownLoc();
	tuples::ColumnManager& attrManager =
		module.getContext()
		->getLoadedDialect<tuples::TupleStreamDialect>()
		->getColumnManager();
	auto tupleStreamType = tuples::TupleStreamType::get(builder.getContext());
	std::unordered_map<int, const tuples::Column*> resolvedColumnAttrs;

	children[0]->resolveMLIRValue(translationContext, scope);
	auto childValue = children[0]->getMLIRValue();

	bool isMapOperationRequired = false;
	for (int i = 0; i < expressions.size(); ++i) {
		auto& expr = expressions[i];
		if (expr->expression_class == ExpressionClass::BOUND_AGGREGATE) {
			auto& bound_agg = expr->Cast<BoundAggregateExpression>();
			for (const auto& child : bound_agg.children) {
				if (child->expression_class == ExpressionClass::BOUND_COLUMN_REF) {
					auto& colRefExpr = child->Cast<BoundColumnRefExpression>();
					auto columnName = colRefExpr.ToString();
					auto columnAttr = translationContext.getAttribute(columnName);
					resolvedColumnAttrs[i] = columnAttr;
				}
				else {
					isMapOperationRequired = true;
				}
			}
		}
	}
	std::cout << "[LogicalAggregate](resolveMLIRValue) :: isMapOperationRequired :: " << isMapOperationRequired << std::endl;

	if (isMapOperationRequired) {
		std::cout << "[LogicalAggregate](resolveMLIRValue) :: Map operation is required to resolve expressions" << std::endl;
		std::cout.flush();
		auto* block = new mlir::Block();
		static size_t mapOpId = 0;
		static size_t mapArgId = 0;
		string mapOpName = "map_op_" + std::to_string(mapOpId++);

		mlir::OpBuilder mapBuilder(&context);
		block->addArgument(tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
		auto tupleScope = translationContext.createTupleScope();
		mlir::Value tuple = block->getArgument(0);
		translationContext.setCurrentTuple(tuple);

		mapBuilder.setInsertionPointToStart(block);
		std::vector<mlir::Value> createdValues;
		std::vector<mlir::Attribute> createdCols;
		for (int i = 0; i < expressions.size(); ++i) {
			auto& expr = expressions[i];

			if (expr->expression_class != ExpressionClass::BOUND_AGGREGATE) {
				std::cout << "[LogicalAggregate](resolveMLIRValue) :: Currently only supporting BOUND_AGGREGATE expressions but found expression of class " << ExpressionClassToString(expr->expression_class) << std::endl;
				continue;
			}
			auto& bound_agg = expr->Cast<BoundAggregateExpression>();
			if (bound_agg.children.size() != 1) {
				std::cout << "[LogicalAggregate](resolveMLIRValue) :: Currently only supporting BOUND_AGGREGATE expressions with exactly one child but found " << bound_agg.children.size() << " children" << std::endl;
				continue;
			}
			auto childExpr = bound_agg.children[0].get();
			if (childExpr->expression_class == ExpressionClass::BOUND_COLUMN_REF) {
				continue;
			}
			auto resolvedValue = childExpr->translateExpression(translationContext, mapBuilder);
			createdValues.push_back(resolvedValue);
			string columnName = "expr_" + std::to_string(mapArgId++);
			auto attrDef = attrManager.createDef(mapOpName, columnName);
			attrDef.getColumn().type = resolvedValue.getType();
			createdCols.push_back(attrDef);
			resolvedColumnAttrs[i] = &attrDef.getColumn();
		}
		auto mapOp = builder.create<relalg::MapOp>(builder.getUnknownLoc(), tupleStreamType, childValue, builder.getArrayAttr(createdCols));
		mapOp.getRegion().push_back(block);
		mapBuilder.create<tuples::ReturnOp>(builder.getUnknownLoc(), createdValues);
		childValue = mapOp.getResult();
	}

	std::cout << "[LogicalAggregate](resolveMLIRValue) :: child value so far " << std::endl;
	childValue.print(llvm::outs());
	std::cout << std::endl;
	std::cout << "[LogicalAggregate](resolveMLIRValue) :: Creating AggregationOp" << std::endl;

	static size_t aggrOpId = 0;
	static size_t aggrArgId = 0;
	string aggrOpName = "aggr_op_" + std::to_string(aggrOpId++);

	std::vector<mlir::Attribute> groupByAttrs;
	std::vector<mlir::Attribute> aggrAttrs;

	for (int i = 0; i < groups.size(); ++i) {
		auto& groupByExpr = groups[i];
		if (groupByExpr->expression_class != ExpressionClass::BOUND_COLUMN_REF) {
			std::cout << "[LogicalAggregate](resolveMLIRValue) :: Skipping group by expression of class " << ExpressionClassToString(groupByExpr->expression_class) << " since only column reference expressions are supported in group by for now" << std::endl;
			continue;
		}
		auto& colRefExpr = groupByExpr->Cast<BoundColumnRefExpression>();
		auto columnName = colRefExpr.ToString();
		std::cout << "[LogicalAggregate](resolveMLIRValue) :: Group by column name :: " << columnName << std::endl;
		auto columnAttr = translationContext.getAttribute(columnName);
		groupByAttrs.push_back(attrManager.createRef(columnAttr));
	}
	for (int i = 0; i < expressions.size(); ++i) {
		string columnName = "aggr_arg_" + std::to_string(aggrArgId++);
		auto attrDef = attrManager.createDef(aggrOpName, columnName);
		auto columnDef = resolvedColumnAttrs[i];
		attrDef.getColumn().type = columnDef->type;
		aggrAttrs.push_back(attrDef);
		mlirAttributeInfos.push_back(MLIRAttributeInfo{aggrOpName, columnName, &attrDef.getColumn()});
	}

	auto aggrOp = builder.create<relalg::AggregationOp>(loc,
		tupleStreamType,
		childValue,
		builder.getArrayAttr(groupByAttrs),
		builder.getArrayAttr(aggrAttrs)
	);

	mlir::Block *aggrBlock = new mlir::Block();
	aggrBlock->addArgument(tupleStreamType, loc);
	aggrBlock->addArgument(tuples::TupleType::get(builder.getContext()), loc);

	mlir::OpBuilder aggrBuilder(builder.getContext());
	aggrBuilder.setInsertionPointToStart(aggrBlock);

	mlir::Value relArg = aggrBlock->getArgument(0);

	std::vector<mlir::Value> resultValues;
	for (int i = 0; i < expressions.size(); ++i) {
		auto &expr = expressions[i];
		if (expr->expression_class != ExpressionClass::BOUND_AGGREGATE) {
			std::cout << "[LogicalAggregate](resolveMLIRValue) :: Skipping expression of class " << ExpressionClassToString(expr->expression_class) << " since only BOUND_AGGREGATE expressions are supported" << std::endl;
			continue;
		}
		auto &bound_agg = expr->Cast<BoundAggregateExpression>();
		auto functionName = bound_agg.function.name;
		relalg::AggrFunc aggrFunc = getAggrFunc(functionName);
		auto columnDef = resolvedColumnAttrs[i];

		auto val = aggrBuilder.create<relalg::AggrFuncOp>(builder.getUnknownLoc(),
			columnDef->type,
			aggrFunc,
			relArg,
			attrManager.createRef(columnDef)
		);
		resultValues.push_back(val);
	}

	aggrBuilder.create<tuples::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange(resultValues));
	aggrOp.getAggrFunc().push_back(aggrBlock);
	this->mlirValue = aggrOp.getResult();

	std::cout << "[LogicalAggregate](resolveMLIRValue) :: Created AggregationOp with name " << aggrOpName << std::endl;
	std::cout.flush();
	this->mlirValue.print(llvm::outs());
	std::cout << std::endl;
}

} // namespace duckdb
