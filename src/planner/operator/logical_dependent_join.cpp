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

	// Right-side bindings are mapped through the SingleJoinOp mapping.
	// Find the index of this binding in the right child's bindings.
	auto rightBindings = children[1]->GetColumnBindings();
	for (idx_t i = 0; i < rightBindings.size(); i++) {
		if (rightBindings[i] == binding) {
			return mlirAttributeInfos[i];
		}
	}
	throw std::runtime_error("[LogicalDependentJoin] Could not resolve column binding " + binding.ToString());
}

void LogicalDependentJoin::resolveMLIRValue(MLIRTranslationContext& context, MLIRTranslationContext::ResolverScope& scope) {
	std::cout << "[LogicalDependentJoin](resolveMLIRValue) :: Resolving MLIR value for LogicalDependentJoin, join_type=" << JoinTypeToString(join_type) << std::endl;

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
		std::cout << "[LogicalDependentJoin](resolveMLIRValue) :: MARK join: mark binding = " << markBinding.ToString() << std::endl;

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
		std::cout << "[LogicalDependentJoin](resolveMLIRValue) :: MARK join: deferred right side for relalg.exists" << std::endl;
		return;
	}

	// SINGLE join (scalar subquery) path — original implementation below.
	children[0]->resolveMLIRValue(context, scope);

	children[1]->parentColumnBindings = this->parentColumnBindings; // Pass down parent column bindings to the right child
	children[1]->parentColumnBindings.push_front(children[0].get()); // Add left child to parent column bindings for the right child
	children[1]->resolveMLIRValue(context, scope);


	auto leftValue = children[0]->getMLIRValue();
	auto rightValue = children[1]->getMLIRValue();

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	D_ASSERT(mlirContainerInstance.getContextPtr() != nullptr);

	auto &mlirContext = mlirContainerInstance.getContext();
	auto &builder = mlirContainerInstance.getBuilder();
	auto module = mlirContainerInstance.getModuleOp();
	auto loc = builder.getUnknownLoc();

	tuples::ColumnManager& attrManager =
		module.getContext()
		->getLoadedDialect<tuples::TupleStreamDialect>()
		->getColumnManager();

	// Build the mapping: for each right-side column, create a new ColumnDefAttr
	// that maps from the original right-side column, with a nullable type.
	auto rightBindings = children[1]->GetColumnBindings();
	std::vector<mlir::Attribute> mappingAttrs;
	mlirAttributeInfos.clear();

	static int singleJoinCounter = 0;

	for (idx_t i = 0; i < rightBindings.size(); i++) {
		auto& rightAttrInfo = children[1]->resolveColumnBindingToAttributeInfo(rightBindings[i]);
		auto* rightColumn = rightAttrInfo.column;

		std::string scopeName = "singlejoin_" + std::to_string(singleJoinCounter++);
		std::string attrName = rightAttrInfo.col_name;

		// Create fromExisting reference to the original right-side column
		auto fromExisting = builder.getArrayAttr({attrManager.createRef(rightColumn)});

		// Create new column def
		auto newDef = attrManager.createDef(scopeName, attrName, fromExisting);

		// Make the type nullable (SingleJoin may produce NULL if right side has 0 rows)
		auto originalType = rightColumn->type;
		if (!mlir::isa<db::NullableType>(originalType)) {
			newDef.getColumn().type = db::NullableType::get(&mlirContext, originalType);
		} else {
			newDef.getColumn().type = originalType;
		}

		mappingAttrs.push_back(newDef);

		// Register the mapped column in the translation context
		context.mapAttribute(scope, attrName, &newDef.getColumn());
		context.mapAttribute(scope, scopeName + "." + attrName, &newDef.getColumn());

		// Store in mlirAttributeInfos for resolveColumnBindingToAttributeInfo
		mlirAttributeInfos.push_back(MLIRAttributeInfo{scopeName, attrName, &newDef.getColumn()});

		std::cout << "[LogicalDependentJoin](resolveMLIRValue) :: Mapped right column "
			<< rightAttrInfo.table_name << "." << rightAttrInfo.col_name
			<< " -> " << scopeName << "." << attrName << std::endl;
	}

	auto mapping = builder.getArrayAttr(mappingAttrs);

	// Create the SingleJoinOp
	auto singleJoin = builder.create<relalg::SingleJoinOp>(
		loc,
		tuples::TupleStreamType::get(&mlirContext),
		leftValue,
		rightValue,
		mapping);

	// Initialize an empty (always-true) predicate
	singleJoin.initPredicate();

	this->mlirValue = singleJoin.getResult();
	std::cout << "[LogicalDependentJoin](resolveMLIRValue) :: Created SingleJoinOp with "
		<< rightBindings.size() << " mapped column(s)" << std::endl;
}

} // namespace duckdb
