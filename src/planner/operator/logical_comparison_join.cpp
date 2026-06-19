#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_column_data_get.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/common/enum_util.hpp"

#include "lingodb/execution/Frontend.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace relalg = lingodb::compiler::dialect::relalg;
namespace tuples = lingodb::compiler::dialect::tuples;
namespace db     = lingodb::compiler::dialect::db;

namespace duckdb {

LogicalComparisonJoin::LogicalComparisonJoin(JoinType join_type, LogicalOperatorType logical_type)
    : LogicalJoin(join_type, logical_type) {
}

InsertionOrderPreservingMap<string> LogicalComparisonJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Join Type"] = EnumUtil::ToChars(join_type);

	string conditions_info;
	for (idx_t i = 0; i < conditions.size(); i++) {
		if (i > 0) {
			conditions_info += "\n";
		}
		auto &condition = conditions[i];
		auto expr =
		    make_uniq<BoundComparisonExpression>(condition.comparison, condition.left->Copy(), condition.right->Copy());
		conditions_info += expr->ToString();
	}
	if (predicate) {
		if (!conditions.empty()) {
			conditions_info += "\n";
		}
		conditions_info += predicate->ToString();
	}
	result["Conditions"] = conditions_info;
	SetParamsEstimatedCardinality(result);

	return result;
}

bool LogicalComparisonJoin::HasEquality(idx_t &range_count) const {
	bool result = false;
	for (size_t c = 0; c < conditions.size(); ++c) {
		auto &cond = conditions[c];
		switch (cond.comparison) {
		case ExpressionType::COMPARE_EQUAL:
		case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
			result = true;
			break;
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_GREATERTHAN:
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			++range_count;
			break;
		case ExpressionType::COMPARE_NOTEQUAL:
		case ExpressionType::COMPARE_DISTINCT_FROM:
			break;
		default:
			throw NotImplementedException("Unimplemented comparison join");
		}
	}
	return result;
}

MLIRAttributeInfo& LogicalComparisonJoin::resolveColumnBindingToAttributeInfo(ColumnBinding& binding) {
	auto leftBindings  = children[0]->GetColumnBindings();
	auto rightBindings = children[1]->GetColumnBindings();

	if (join_type == JoinType::RIGHT) {
		// RIGHT outer join: children[1] is preserved (non-nullable), children[0] is nullable.
		// mlirAttributeInfos holds nullable attrs for children[0]'s bindings.
		if (std::find(rightBindings.begin(), rightBindings.end(), binding) != rightBindings.end()) {
			return children[1]->resolveColumnBindingToAttributeInfo(binding);
		}
		if (!mlirAttributeInfos.empty()) {
			for (idx_t i = 0; i < leftBindings.size(); i++) {
				if (leftBindings[i] == binding) {
					return mlirAttributeInfos[i];
				}
			}
		}
		return children[0]->resolveColumnBindingToAttributeInfo(binding);
	}

	// LEFT join (and INNER/others): children[0] is non-nullable, children[1] may be nullable.
	// mlirAttributeInfos holds nullable attrs for children[1]'s bindings (populated after mapping).
	if (std::find(leftBindings.begin(), leftBindings.end(), binding) != leftBindings.end()) {
		return children[0]->resolveColumnBindingToAttributeInfo(binding);
	}
	if (!mlirAttributeInfos.empty()) {
		for (idx_t i = 0; i < rightBindings.size(); i++) {
			if (rightBindings[i] == binding) {
				return mlirAttributeInfos[i];
			}
		}
	}
	return children[1]->resolveColumnBindingToAttributeInfo(binding);
}

// ── DELIM_JOIN helpers ────────────────────────────────────────────────────────

// Recursively locate the LogicalDelimGet leaf in a subtree.
static LogicalDelimGet* findDelimGet(LogicalOperator* node) {
	if (node->type == LogicalOperatorType::LOGICAL_DELIM_GET)
		return &node->Cast<LogicalDelimGet>();
	for (auto& child : node->children) {
		if (auto* r = findDelimGet(child.get()))
			return r;
	}
	return nullptr;
}

static bool subtreeContainsDelimGet(LogicalOperator* node) {
	return findDelimGet(node) != nullptr;
}

// Walk the left subtree of a DELIM_JOIN, skipping the DELIM_GET placeholder
// and any narrowing PROJECTION nodes added by DuckDB. Returns the MLIR value
// for the real inner relation and a pointer to the inner COMPARISON_JOIN node
// (used as 'op' in translateExpression so its resolveColumnBindingToAttributeInfo
// can find both lineitem columns and the pre-populated DELIM_GET attrs).
struct DelimInnerInfo {
	mlir::Value innerValue;
	LogicalOperator* innerJoinOp; // the inner COMPARISON_JOIN node
};

static DelimInnerInfo extractDelimInnerPipeline(
    LogicalOperator* node,
    const std::deque<LogicalOperator*>& parentBindings,
    MLIRTranslationContext& ctx,
    MLIRTranslationContext::ResolverScope& scope)
{
	node->parentColumnBindings = parentBindings;

	switch (node->type) {
	case LogicalOperatorType::LOGICAL_PROJECTION:
		// Skip: DuckDB adds this only to narrow columns for its hash table.
		return extractDelimInnerPipeline(node->children[0].get(), parentBindings, ctx, scope);

	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		// One child leads to DELIM_GET (outer key feed), the other is the real inner table.
		bool leftIsDelim = subtreeContainsDelimGet(node->children[0].get());
		auto* innerChild = leftIsDelim ? node->children[1].get() : node->children[0].get();

		innerChild->parentColumnBindings = parentBindings;
		innerChild->resolveMLIRValue(ctx, scope);

		return {innerChild->getMLIRValue(), node};
	}

	default:
		// Unexpected node between DELIM_JOIN and COMPARISON_JOIN — resolve normally.
		node->resolveMLIRValue(ctx, scope);
		return {node->getMLIRValue(), node};
	}
}

// ─────────────────────────────────────────────────────────────────────────────

static int compJoinOuterCounter = 0;

void LogicalComparisonJoin::resolveMLIRValue(MLIRTranslationContext& context, MLIRTranslationContext::ResolverScope& scope) {
	// ── DELIM_JOIN (correlated subquery decorrelated by the optimizer) ────────
	if (this->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		// Step 1: resolve the outer relation (right child, e.g. orders + date filter).
		children[1]->parentColumnBindings = this->parentColumnBindings;
		children[1]->resolveMLIRValue(context, scope);
		auto outerValue = children[1]->getMLIRValue();

		// Step 2: pre-populate DELIM_GET's mlirAttributeInfos so that the join
		// condition expressions inside the inner COMPARISON_JOIN can resolve
		// DELIM_GET column bindings to the outer relation's already-resolved attrs.
		auto* delimGet = findDelimGet(children[0].get());
		D_ASSERT(delimGet != nullptr && "DELIM_JOIN left child must contain a DELIM_GET");
		delimGet->mlirAttributeInfos.clear();
		for (idx_t i = 0; i < duplicate_eliminated_columns.size(); i++) {
			auto& elimExpr = duplicate_eliminated_columns[i];
			auto& bcre = elimExpr->Cast<BoundColumnRefExpression>();
			auto binding = bcre.binding;
			auto& attrInfo = children[1]->resolveColumnBindingToAttributeInfo(binding);
			delimGet->mlirAttributeInfos.push_back(attrInfo);
		}

		// Step 3: walk the left child, skipping DELIM_GET and narrowing projections,
		// and extract the real inner relation + a pointer to the inner COMPARISON_JOIN.
		auto innerInfo    = extractDelimInnerPipeline(
		    children[0].get(), this->parentColumnBindings, context, scope);
		auto  innerValue  = innerInfo.innerValue;
		auto* innerJoinOp = innerInfo.innerJoinOp;

		// Conditions live on the inner COMPARISON_JOIN node.
		auto& joinConditions = innerJoinOp->Cast<LogicalComparisonJoin>().conditions;

		// Step 4: build the semijoin predicate block from the extracted conditions.
		auto& container = lingodb::execution::MLIRContainer::getInstance();
		auto& builder   = container.getBuilder();
		auto& mlirCtx   = container.getContext();
		auto  loc       = builder.getUnknownLoc();

		auto* predBlock = new mlir::Block();
		predBlock->addArgument(tuples::TupleType::get(&mlirCtx), loc);
		{
			mlir::OpBuilder predBuilder(&mlirCtx);
			predBuilder.setInsertionPointToStart(predBlock);
			auto tupleScope = context.createTupleScope();
			context.setCurrentTuple(predBlock->getArgument(0));

			if (joinConditions.empty()) {
				// No explicit conditions — predicate is trivially true.
				auto trueVal = predBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
				predBuilder.create<tuples::ReturnOp>(loc, trueVal.getResult());
			} else {
				std::vector<mlir::Value> condVals;
				for (auto& cond : joinConditions) {
					auto condExpr = make_uniq<BoundComparisonExpression>(
					    cond.comparison, cond.left->Copy(), cond.right->Copy());
					condVals.push_back(condExpr->translateExpression(context, predBuilder, innerJoinOp));
				}
				mlir::Value combined = condVals.size() == 1
				    ? condVals[0]
				    : predBuilder.create<db::AndOp>(loc, condVals).getResult();
				predBuilder.create<tuples::ReturnOp>(loc, combined);
			}
		}

		// Step 5: emit the appropriate relalg join op.
		if (join_type == JoinType::RIGHT_SEMI) {
			auto semiJoin = builder.create<relalg::SemiJoinOp>(
			    loc, tuples::TupleStreamType::get(&mlirCtx), outerValue, innerValue);
			semiJoin.getPredicate().push_back(predBlock);
			this->mlirValue = semiJoin.getResult();
		} else if (join_type == JoinType::RIGHT_ANTI) {
			auto antiJoin = builder.create<relalg::AntiSemiJoinOp>(
			    loc, tuples::TupleStreamType::get(&mlirCtx), outerValue, innerValue);
			antiJoin.getPredicate().push_back(predBlock);
			this->mlirValue = antiJoin.getResult();
		} else {
			throw NotImplementedException(
			    "[LogicalComparisonJoin] DELIM_JOIN: unsupported join_type: " + JoinTypeToString(join_type));
		}
		return;
	}

	if (join_type == JoinType::LEFT) {
		// LEFT OUTER JOIN: emit relalg::OuterJoinOp with nullable column mapping.
		// Pattern mirrors LogicalAnyJoin::resolveMLIRValue but uses structured conditions
		// (this->conditions) plus an optional extra predicate (this->predicate).
		children[0]->parentColumnBindings = this->parentColumnBindings;
		children[1]->parentColumnBindings = this->parentColumnBindings;
		children[0]->resolveMLIRValue(context, scope);
		children[1]->resolveMLIRValue(context, scope);

		auto leftValue  = children[0]->getMLIRValue();
		auto rightValue = children[1]->getMLIRValue();

		auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
		auto &mlirContext = mlirContainerInstance.getContext();
		auto &builder     = mlirContainerInstance.getBuilder();
		auto  module      = mlirContainerInstance.getModuleOp();
		auto  loc         = builder.getUnknownLoc();

		tuples::ColumnManager &attrManager =
		    module.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

		// Build predicate block before creating the mapping so right-side column refs
		// still resolve to the original (non-nullable) columns inside the predicate.
		auto *predBlock = new mlir::Block();
		mlir::OpBuilder predBuilder(builder.getContext());
		predBlock->addArgument(tuples::TupleType::get(builder.getContext()), loc);
		{
			auto tupleScope = context.createTupleScope();
			context.setCurrentTuple(predBlock->getArgument(0));
			predBuilder.setInsertionPointToStart(predBlock);

			std::vector<mlir::Value> condVals;
			for (auto &cond : conditions) {
				auto condExpr = make_uniq<BoundComparisonExpression>(
				    cond.comparison, cond.left->Copy(), cond.right->Copy());
				condVals.push_back(condExpr->translateExpression(context, predBuilder, this));
			}
			if (predicate) {
				condVals.push_back(predicate->translateExpression(context, predBuilder, this));
			}

			mlir::Value condVal = condVals.size() == 1
			    ? condVals[0]
			    : predBuilder.create<db::AndOp>(loc, condVals).getResult();
			predBuilder.create<tuples::ReturnOp>(loc, condVal);
		}

		// Build nullable mapping: each right-side column gets a new nullable output column.
		auto rightBindings = children[1]->GetColumnBindings();
		std::vector<mlir::Attribute> mappingAttrs;
		mlirAttributeInfos.clear();

		std::string ojName = "ojcj" + std::to_string(compJoinOuterCounter++);
		for (idx_t i = 0; i < rightBindings.size(); i++) {
			auto &rightAttrInfo  = children[1]->resolveColumnBindingToAttributeInfo(rightBindings[i]);
			auto *rightColumn    = rightAttrInfo.column;
			std::string attrName = rightAttrInfo.col_name;

			auto fromExisting = builder.getArrayAttr({attrManager.createRef(rightColumn)});
			auto newDef       = attrManager.createDef(ojName, attrName, fromExisting);

			auto originalType       = rightColumn->type;
			newDef.getColumn().type = mlir::isa<db::NullableType>(originalType)
			                              ? originalType
			                              : db::NullableType::get(&mlirContext, originalType);

			mappingAttrs.push_back(newDef);
			context.mapAttribute(scope, attrName, &newDef.getColumn());
			context.mapAttribute(scope, ojName + "." + attrName, &newDef.getColumn());
			mlirAttributeInfos.push_back(MLIRAttributeInfo{ojName, attrName, &newDef.getColumn()});
		}

		auto mapping   = builder.getArrayAttr(mappingAttrs);
		auto outerJoin = builder.create<relalg::OuterJoinOp>(
		    loc, tuples::TupleStreamType::get(&mlirContext), leftValue, rightValue, mapping);
		outerJoin.getPredicate().push_back(predBlock);

		this->mlirValue = outerJoin.getResult();
		return;
	}

	if (join_type == JoinType::RIGHT) {
		// RIGHT OUTER JOIN: children[1] is preserved, children[0] is nullable.
		// Expressed as relalg.outerjoin(preserved=children[1], nullable=children[0], mapping_for_children[0]).
		children[0]->parentColumnBindings = this->parentColumnBindings;
		children[1]->parentColumnBindings = this->parentColumnBindings;
		children[0]->resolveMLIRValue(context, scope);
		children[1]->resolveMLIRValue(context, scope);

		auto nullableValue  = children[0]->getMLIRValue();
		auto preservedValue = children[1]->getMLIRValue();

		auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
		auto &mlirContext = mlirContainerInstance.getContext();
		auto &builder     = mlirContainerInstance.getBuilder();
		auto  module      = mlirContainerInstance.getModuleOp();
		auto  loc         = builder.getUnknownLoc();

		tuples::ColumnManager &attrManager =
		    module.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

		// Build predicate before mapping so both sides still resolve as non-nullable inside it.
		auto *predBlock = new mlir::Block();
		mlir::OpBuilder predBuilder(builder.getContext());
		predBlock->addArgument(tuples::TupleType::get(builder.getContext()), loc);
		{
			auto tupleScope = context.createTupleScope();
			context.setCurrentTuple(predBlock->getArgument(0));
			predBuilder.setInsertionPointToStart(predBlock);

			std::vector<mlir::Value> condVals;
			for (auto &cond : conditions) {
				auto condExpr = make_uniq<BoundComparisonExpression>(
				    cond.comparison, cond.left->Copy(), cond.right->Copy());
				condVals.push_back(condExpr->translateExpression(context, predBuilder, this));
			}
			if (predicate) {
				condVals.push_back(predicate->translateExpression(context, predBuilder, this));
			}

			mlir::Value condVal = condVals.size() == 1
			    ? condVals[0]
			    : predBuilder.create<db::AndOp>(loc, condVals).getResult();
			predBuilder.create<tuples::ReturnOp>(loc, condVal);
		}

		// Build nullable mapping for children[0] (the non-preserved left side).
		auto leftBindings = children[0]->GetColumnBindings();
		std::vector<mlir::Attribute> mappingAttrs;
		mlirAttributeInfos.clear();

		std::string ojName = "ojcj" + std::to_string(compJoinOuterCounter++);
		for (idx_t i = 0; i < leftBindings.size(); i++) {
			auto &leftAttrInfo   = children[0]->resolveColumnBindingToAttributeInfo(leftBindings[i]);
			auto *leftColumn     = leftAttrInfo.column;
			std::string attrName = leftAttrInfo.col_name;

			auto fromExisting = builder.getArrayAttr({attrManager.createRef(leftColumn)});
			auto newDef       = attrManager.createDef(ojName, attrName, fromExisting);

			auto originalType       = leftColumn->type;
			newDef.getColumn().type = mlir::isa<db::NullableType>(originalType)
			                              ? originalType
			                              : db::NullableType::get(&mlirContext, originalType);

			mappingAttrs.push_back(newDef);
			context.mapAttribute(scope, attrName, &newDef.getColumn());
			context.mapAttribute(scope, ojName + "." + attrName, &newDef.getColumn());
			mlirAttributeInfos.push_back(MLIRAttributeInfo{ojName, attrName, &newDef.getColumn()});
		}

		auto mapping   = builder.getArrayAttr(mappingAttrs);
		auto outerJoin = builder.create<relalg::OuterJoinOp>(
		    loc, tuples::TupleStreamType::get(&mlirContext), preservedValue, nullableValue, mapping);
		outerJoin.getPredicate().push_back(predBlock);

		this->mlirValue = outerJoin.getResult();
		return;
	}

	if (join_type == JoinType::MARK) {
		// Special case: MARK join against CHUNK_GET = IN (literal list).
		// Translate as db.oneof inside a deferredScalarCallback so the parent FILTER
		// receives an i1 directly rather than going through relalg.exists.
		if (children[1]->type == LogicalOperatorType::LOGICAL_CHUNK_GET) {
			auto& chunkGet = children[1]->Cast<LogicalColumnDataGet>();

			// Collect all values from the materialized IN list (one column, N rows).
			std::vector<Value> inValues;
			ColumnDataScanState scanState;
			DataChunk chunk;
			chunkGet.collection->InitializeScan(scanState);
			chunkGet.collection->InitializeScanChunk(scanState, chunk);
			while (chunkGet.collection->Scan(scanState, chunk)) {
				chunk.Flatten();
				for (idx_t i = 0; i < chunk.size(); i++) {
					inValues.push_back(chunk.data[0].GetValue(i));
				}
			}

			// Resolve left child (the stream to filter).
			children[0]->parentColumnBindings = this->parentColumnBindings;
			children[0]->resolveMLIRValue(context, scope);
			this->mlirValue = children[0]->getMLIRValue();

			// Find the mark ColumnBinding (the one output binding not in the left child).
			auto leftBindings = children[0]->GetColumnBindings();
			auto allBindings  = this->GetColumnBindings();
			ColumnBinding markBinding;
			for (auto& b : allBindings) {
				if (std::find(leftBindings.begin(), leftBindings.end(), b) == leftBindings.end()) {
					markBinding = b;
					break;
				}
			}

			// Wrap in shared_ptr so the lambda (stored in std::function) stays copyable.
			auto lhsExpr = std::shared_ptr<Expression>(conditions[0].left->Copy());
			auto* thisOp = this;

			// Register a scalar callback: when the parent FILTER translates the mark
			// binding, this builds db.oneof(col ? v1, v2, ...) and returns i1 directly.
			context.deferredScalarCallbacks[markBinding] =
			    [lhsExpr, inValues, thisOp, &context](mlir::OpBuilder& predBuilder) -> mlir::Value {
				auto loc = predBuilder.getUnknownLoc();

				// Translate the column being tested (e.g. p_size).
				auto colVal = lhsExpr->translateExpression(context, predBuilder, thisOp);

				// Build one db.constant per IN value, then normalize all to common type.
				std::vector<mlir::Value> operands;
				operands.push_back(colVal);
				for (auto& v : inValues) {
					auto constExpr = make_uniq<BoundConstantExpression>(v);
					operands.push_back(constExpr->translateExpression(context, predBuilder, thisOp));
				}
				auto normalizedOperands =
				    lingodb::compiler::frontend::sql::SQLTypeInference::toCommonBaseTypes(predBuilder, operands);

				return predBuilder.create<db::OneOfOp>(loc, normalizedOperands).getResult();
			};

			return;
		}

		// 1. Resolve left child
		children[0]->parentColumnBindings = this->parentColumnBindings;
		children[0]->resolveMLIRValue(context, scope);

		// 2. Set up right child's parent bindings so it can access the outer scope
		children[1]->parentColumnBindings = this->parentColumnBindings;
		children[1]->parentColumnBindings.push_front(children[0].get());

		auto* rightChild = children[1].get();
		auto* thisOp = this;

		// 3. Find mark column binding: the one output binding NOT in the left child
		auto leftBindings = children[0]->GetColumnBindings();
		auto allBindings  = this->GetColumnBindings();
		ColumnBinding markBinding;
		for (auto& b : allBindings) {
			if (std::find(leftBindings.begin(), leftBindings.end(), b) == leftBindings.end()) {
				markBinding = b;
				break;
			}
		}

		// 4. Register deferred callback: builds the right side + correlated selection inside the predicate block.
		//    BoundColumnRefExpression will intercept the mark binding and invoke this to emit relalg.exists.
		context.deferredExistsCallbacks[markBinding] = [rightChild, &context, &scope, thisOp](mlir::OpBuilder& predBuilder) -> mlir::Value {
			auto& mlirContainer = lingodb::execution::MLIRContainer::getInstance();
			auto& globalBuilder = mlirContainer.getBuilder();

			auto savedPoint = globalBuilder.saveInsertionPoint();
			globalBuilder.setInsertionPoint(predBuilder.getBlock(), predBuilder.getInsertionPoint());

			// Build the right child (e.g. supplier filtered by s_comment LIKE ...)
			rightChild->resolveMLIRValue(context, scope);
			auto rightValue = rightChild->getMLIRValue();

			// Build a correlated selection that enforces the join condition(s) (ps_suppkey = s_suppkey)
			auto loc = globalBuilder.getUnknownLoc();
			auto* condBlock = new mlir::Block();
			mlir::OpBuilder condBuilder(globalBuilder.getContext());
			condBlock->addArgument(tuples::TupleType::get(globalBuilder.getContext()), loc);

			{
				auto tupleScope = context.createTupleScope();
				context.setCurrentTuple(condBlock->getArgument(0));
				condBuilder.setInsertionPointToStart(condBlock);

				// Translate each join condition and combine with AND
				std::vector<mlir::Value> condVals;
				for (auto& cond : thisOp->conditions) {
					auto condExpr = make_uniq<BoundComparisonExpression>(
					    cond.comparison, cond.left->Copy(), cond.right->Copy());
					condVals.push_back(condExpr->translateExpression(context, condBuilder, thisOp));
				}

				mlir::Value condVal = condVals.size() == 1
				    ? condVals[0]
				    : condBuilder.create<db::AndOp>(loc, condVals).getResult();
				condBuilder.create<tuples::ReturnOp>(loc, condVal);
			}

			auto selOp = globalBuilder.create<relalg::SelectionOp>(
			    loc, tuples::TupleStreamType::get(globalBuilder.getContext()), rightValue);
			selOp.getPredicate().push_back(condBlock);

			globalBuilder.restoreInsertionPoint(savedPoint);
			return selOp.getResult();
		};

		this->mlirValue = children[0]->getMLIRValue();
		return;
	}

	if (join_type == JoinType::INNER) {
		children[0]->parentColumnBindings = this->parentColumnBindings;
		children[1]->parentColumnBindings = this->parentColumnBindings;
		children[0]->resolveMLIRValue(context, scope);
		children[1]->resolveMLIRValue(context, scope);

		auto leftValue  = children[0]->getMLIRValue();
		auto rightValue = children[1]->getMLIRValue();

		auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
		auto &mlirContext = mlirContainerInstance.getContext();
		auto &builder     = mlirContainerInstance.getBuilder();
		auto  loc         = builder.getUnknownLoc();

		auto *predBlock = new mlir::Block();
		mlir::OpBuilder predBuilder(builder.getContext());
		predBlock->addArgument(tuples::TupleType::get(builder.getContext()), loc);
		{
			auto tupleScope = context.createTupleScope();
			context.setCurrentTuple(predBlock->getArgument(0));
			predBuilder.setInsertionPointToStart(predBlock);

			std::vector<mlir::Value> condVals;
			for (auto &cond : conditions) {
				auto condExpr = make_uniq<BoundComparisonExpression>(
				    cond.comparison, cond.left->Copy(), cond.right->Copy());
				condVals.push_back(condExpr->translateExpression(context, predBuilder, this));
			}
			if (predicate) {
				condVals.push_back(predicate->translateExpression(context, predBuilder, this));
			}

			mlir::Value condVal = condVals.size() == 1
			    ? condVals[0]
			    : predBuilder.create<db::AndOp>(loc, condVals).getResult();
			predBuilder.create<tuples::ReturnOp>(loc, condVal);
		}

		auto innerJoin = builder.create<relalg::InnerJoinOp>(
		    loc, tuples::TupleStreamType::get(&mlirContext), leftValue, rightValue);
		innerJoin.getPredicate().push_back(predBlock);

		this->mlirValue = innerJoin.getResult();
		return;
	}

	throw NotImplementedException("[LogicalComparisonJoin] Unsupported join type in MLIR codegen: " + JoinTypeToString(join_type));
}

} // namespace duckdb
