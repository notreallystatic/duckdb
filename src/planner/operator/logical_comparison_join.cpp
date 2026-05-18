#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/common/enum_util.hpp"

#include "lingodb/execution/Frontend.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "mlir/IR/Builders.h"

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
	auto leftBindings = children[0]->GetColumnBindings();
	if (std::find(leftBindings.begin(), leftBindings.end(), binding) != leftBindings.end()) {
		return children[0]->resolveColumnBindingToAttributeInfo(binding);
	}
	// After the outer join mapping is built, right-side bindings resolve to nullable columns.
	// Before the mapping is built (during predicate construction), fall through to the original
	// right-child attribute so the predicate accesses non-nullable types.
	if (!mlirAttributeInfos.empty()) {
		auto rightBindings = children[1]->GetColumnBindings();
		for (idx_t i = 0; i < rightBindings.size(); i++) {
			if (rightBindings[i] == binding) {
				return mlirAttributeInfos[i];
			}
		}
	}
	return children[1]->resolveColumnBindingToAttributeInfo(binding);
}

static int compJoinOuterCounter = 0;

void LogicalComparisonJoin::resolveMLIRValue(MLIRTranslationContext& context, MLIRTranslationContext::ResolverScope& scope) {
	std::cout << "[LogicalComparisonJoin](resolveMLIRValue) :: join_type=" << JoinTypeToString(join_type) << std::endl;

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
		std::cout << "[LogicalComparisonJoin] Created OuterJoinOp (LEFT) with " << rightBindings.size()
		          << " nullable column(s)" << std::endl;
		return;
	}

	if (join_type == JoinType::MARK) {
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
		std::cout << "[LogicalComparisonJoin](resolveMLIRValue) :: MARK join mark binding = " << markBinding.ToString() << std::endl;

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
		std::cout << "[LogicalComparisonJoin](resolveMLIRValue) :: MARK join: deferred relalg.exists registered" << std::endl;
		return;
	}

	throw NotImplementedException("[LogicalComparisonJoin] Only LEFT and MARK join types are supported in MLIR codegen, got: " + JoinTypeToString(join_type));
}

} // namespace duckdb
