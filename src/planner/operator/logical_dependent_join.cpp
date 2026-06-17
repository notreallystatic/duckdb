#include "duckdb/planner/operator/logical_dependent_join.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/common/types.hpp"

#include "lingodb/execution/Frontend.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "mlir/IR/Builders.h"

namespace duckdb {

LogicalDependentJoin::LogicalDependentJoin(unique_ptr<LogicalOperator> left, unique_ptr<LogicalOperator> right,
                                           vector<CorrelatedColumnInfo> correlated_columns, JoinType type,
                                           unique_ptr<Expression> condition)
    : LogicalComparisonJoin(type, LogicalOperatorType::LOGICAL_DEPENDENT_JOIN), join_condition(std::move(condition)),
      correlated_columns(std::move(correlated_columns)) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

LogicalDependentJoin::LogicalDependentJoin(JoinType join_type)
    : LogicalComparisonJoin(join_type, LogicalOperatorType::LOGICAL_DEPENDENT_JOIN) {
}

unique_ptr<LogicalOperator> LogicalDependentJoin::Create(unique_ptr<LogicalOperator> left,
                                                         unique_ptr<LogicalOperator> right,
                                                         vector<CorrelatedColumnInfo> correlated_columns, JoinType type,
                                                         unique_ptr<Expression> condition) {
	return make_uniq<LogicalDependentJoin>(std::move(left), std::move(right), std::move(correlated_columns), type,
	                                       std::move(condition));
}

MLIRAttributeInfo& LogicalDependentJoin::resolveColumnBindingToAttributeInfo(ColumnBinding& binding) {
	auto leftBindings = children[0]->GetColumnBindings();
	auto leftIt = std::find(leftBindings.begin(), leftBindings.end(), binding);
	if (leftIt != leftBindings.end()) {
		return children[0]->resolveColumnBindingToAttributeInfo(binding);
	}

	// Right-side bindings: when using singlejoin, they are mapped through mlirAttributeInfos.
	// When using deferred scalar (SINGLE join), mlirAttributeInfos is empty — delegate to right child.
	auto rightBindings = children[1]->GetColumnBindings();
	for (idx_t i = 0; i < rightBindings.size(); i++) {
		if (rightBindings[i] == binding) {
			if (i < mlirAttributeInfos.size()) {
				return mlirAttributeInfos[i];
			}
			return children[1]->resolveColumnBindingToAttributeInfo(binding);
		}
	}
	throw std::runtime_error("[LogicalDependentJoin] Could not resolve column binding " + binding.ToString());
}

void LogicalDependentJoin::resolveMLIRValue(MLIRTranslationContext& context, MLIRTranslationContext::ResolverScope& scope) {
	// The optimizer rewrites DEPENDENT_JOIN into a DELIM_JOIN by changing this->type.
	// Delegate to the base class handler which contains the DELIM_JOIN → semijoin/antisemijoin logic.
	if (this->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		this->LogicalComparisonJoin::resolveMLIRValue(context, scope);
		return;
	}

	if (join_type == JoinType::MARK) {
		// MARK join = EXISTS subquery. We do NOT emit a join op. Instead:
		// 1. Resolve the left child normally (produces the outer tuple stream).
		// 2. Store a callback in the context so that BoundSubqueryExpression can
		//    build the right side *inside* the enclosing selection's predicate block
		//    and emit relalg.exists on it.
		children[0]->parentColumnBindings = this->parentColumnBindings;
		children[0]->resolveMLIRValue(context, scope);

		// Set up right child's parent bindings so depth-1 column refs resolve against
		// the left side (e.g. o_orderkey referenced inside the lineitem filter).
		children[1]->parentColumnBindings = this->parentColumnBindings;
		children[1]->parentColumnBindings.push_front(children[0].get());

		auto* rightChild = children[1].get();

		// Find the mark column binding: the one output binding that is NOT in the left child.
		auto leftBindings  = children[0]->GetColumnBindings();
		auto allBindings   = this->GetColumnBindings();
		ColumnBinding markBinding;
		for (auto& b : allBindings) {
			if (std::find(leftBindings.begin(), leftBindings.end(), b) == leftBindings.end()) {
				markBinding = b;
				break;
			}
		}

		// Store a callback keyed by the mark binding. BoundColumnRefExpression will
		// intercept #[14.0] (or whatever the mark binding is), invoke this callback
		// inside the outer selection's predicate block, and emit relalg.exists.
		context.deferredExistsCallbacks[markBinding] = [rightChild, &context, &scope](mlir::OpBuilder& predBuilder) -> mlir::Value {
			auto& mlirContainer = lingodb::execution::MLIRContainer::getInstance();
			auto& globalBuilder = mlirContainer.getBuilder();

			auto savedPoint = globalBuilder.saveInsertionPoint();
			globalBuilder.setInsertionPoint(predBuilder.getBlock(), predBuilder.getInsertionPoint());

			rightChild->resolveMLIRValue(context, scope);
			auto rightValue = rightChild->getMLIRValue();

			globalBuilder.restoreInsertionPoint(savedPoint);
			return rightValue;
		};

		this->mlirValue = children[0]->getMLIRValue();
		return;
	}

	// SINGLE join (scalar subquery): defer the right child's resolution to inside the
	// outer selection's predicate block. This ensures correlated column refs (e.g. p_partkey)
	// resolve against the outer tuple context, and that relalg.getscalar is only emitted
	// inside the predicate block where all its inputs are in scope.
	children[0]->parentColumnBindings = this->parentColumnBindings;
	children[0]->resolveMLIRValue(context, scope);

	// Set up right child's parent bindings so depth-1 refs resolve against the left side.
	children[1]->parentColumnBindings = this->parentColumnBindings;
	children[1]->parentColumnBindings.push_front(children[0].get());

	auto* rightChild = children[1].get();
	auto rightBindings = children[1]->GetColumnBindings();

	for (idx_t bi = 0; bi < rightBindings.size(); bi++) {
		auto binding = rightBindings[bi];

		context.deferredScalarCallbacks[binding] = [rightChild, bi, &context, &scope](mlir::OpBuilder& predBuilder) -> mlir::Value {
			auto& mlirContainer = lingodb::execution::MLIRContainer::getInstance();
			auto& globalBuilder = mlirContainer.getBuilder();

			// Redirect the global builder into the predicate block so that all ops
			// created by resolveMLIRValue (basetable, selection, agg, map) land inside it.
			auto savedPoint = globalBuilder.saveInsertionPoint();
			globalBuilder.setInsertionPoint(predBuilder.getBlock(), predBuilder.getInsertionPoint());

			rightChild->resolveMLIRValue(context, scope);
			auto subqueryStream = rightChild->getMLIRValue();

			auto innerBindings = rightChild->GetColumnBindings();
			auto& rightAttrInfo = rightChild->resolveColumnBindingToAttributeInfo(innerBindings[bi]);
			auto* rightColumn = rightAttrInfo.column;

			auto moduleOp = mlirContainer.getModuleOp();
			lingodb::compiler::dialect::tuples::ColumnManager& attrManager =
			    moduleOp->getContext()
			        ->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()
			        ->getColumnManager();

			mlir::Type resType = rightColumn->type;
			if (!mlir::isa<lingodb::compiler::dialect::db::NullableType>(resType)) {
				resType = lingodb::compiler::dialect::db::NullableType::get(predBuilder.getContext(), resType);
			}

			auto getScalar = globalBuilder.create<lingodb::compiler::dialect::relalg::GetScalarOp>(
			    globalBuilder.getUnknownLoc(),
			    resType,
			    attrManager.createRef(rightColumn),
			    subqueryStream);

			globalBuilder.restoreInsertionPoint(savedPoint);
			return getScalar.getResult();
		};
	}

	this->mlirValue = children[0]->getMLIRValue();
}

} // namespace duckdb
