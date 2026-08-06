#include "duckdb/planner/operator/logical_any_join.hpp"

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

LogicalAnyJoin::LogicalAnyJoin(JoinType type) : LogicalJoin(type, LogicalOperatorType::LOGICAL_ANY_JOIN) {
}

void LogicalAnyJoin::Walk(int depth) {
	std::cout << string(depth * 2, ' ') << "LogicalAnyJoin :: JoinType :: " << JoinTypeToString(join_type) << std::endl;
	if (condition) {
		std::cout << string(depth * 2, ' ') << "Condition: " << condition->ToString() << std::endl;
	}
	children[0]->Walk(depth + 1);
	children[1]->Walk(depth + 1);
}

InsertionOrderPreservingMap<string> LogicalAnyJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Condition"] = condition->ToString();
	SetParamsEstimatedCardinality(result);
	return result;
}

MLIRAttributeInfo &LogicalAnyJoin::resolveColumnBindingToAttributeInfo(ColumnBinding &binding) {
	auto leftBindings = children[0]->GetColumnBindings();
	if (std::find(leftBindings.begin(), leftBindings.end(), binding) != leftBindings.end()) {
		return children[0]->resolveColumnBindingToAttributeInfo(binding);
	}

	// After the mapping is built, return the nullable output column for downstream operators.
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

void LogicalAnyJoin::resolveMLIRValue(MLIRTranslationContext &context,
                                       MLIRTranslationContext::ResolverScope &scope) {
	// std::cout << "[LogicalAnyJoin](resolveMLIRValue) :: join_type=" << JoinTypeToString(join_type) << std::endl;

	if (join_type != JoinType::LEFT) {
		throw NotImplementedException(
		    "[LogicalAnyJoin] Only LEFT outer join is supported in MLIR codegen, got: " + JoinTypeToString(join_type));
	}

	children[0]->parentColumnBindings = this->parentColumnBindings;
	children[1]->parentColumnBindings = this->parentColumnBindings;
	children[0]->resolveMLIRValue(context, scope);
	children[1]->resolveMLIRValue(context, scope);

	auto leftValue  = children[0]->getMLIRValue();
	auto rightValue = children[1]->getMLIRValue();

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	D_ASSERT(mlirContainerInstance.getContextPtr() != nullptr);

	auto &mlirContext = mlirContainerInstance.getContext();
	auto &builder     = mlirContainerInstance.getBuilder();
	auto  module      = mlirContainerInstance.getModuleOp();
	auto  loc         = builder.getUnknownLoc();

	tuples::ColumnManager &attrManager =
	    module.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

	// Build the predicate block first, while mlirAttributeInfos is still empty.
	// This ensures resolveColumnBindingToAttributeInfo delegates right-side bindings
	// to children[1] (original, non-nullable columns) for the predicate opcodes.
	auto *predBlock = new mlir::Block();
	mlir::OpBuilder predBuilder(builder.getContext());
	predBlock->addArgument(tuples::TupleType::get(builder.getContext()), loc);
	{
		auto tupleScope = context.createTupleScope();
		context.setCurrentTuple(predBlock->getArgument(0));
		predBuilder.setInsertionPointToStart(predBlock);
		mlir::Value condVal = condition->translateExpression(context, predBuilder, this);
		predBuilder.create<tuples::ReturnOp>(loc, condVal);
	}

	// Build nullable mapping: each right-side column gets a new nullable output column.
	auto rightBindings = children[1]->GetColumnBindings();
	std::vector<mlir::Attribute> mappingAttrs;
	mlirAttributeInfos.clear();

	static int outerJoinCounter = 0;
	std::string ojName = "oj" + std::to_string(outerJoinCounter++);

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

		// Shadow original name in scope so downstream operators see the nullable version.
		context.mapAttribute(scope, attrName, &newDef.getColumn());
		context.mapAttribute(scope, ojName + "." + attrName, &newDef.getColumn());

		mlirAttributeInfos.push_back(MLIRAttributeInfo {ojName, attrName, &newDef.getColumn()});

		// std::cout << "[LogicalAnyJoin] Mapped " << rightAttrInfo.table_name << "." << attrName
		          // << " -> " << ojName << "." << attrName << " (nullable)" << std::endl;
	}

	auto mapping = builder.getArrayAttr(mappingAttrs);

	// Create the OuterJoinOp and attach the predicate block.
	auto outerJoin = builder.create<relalg::OuterJoinOp>(
	    loc, tuples::TupleStreamType::get(&mlirContext), leftValue, rightValue, mapping);
	outerJoin.getPredicate().push_back(predBlock);

	this->mlirValue = outerJoin.getResult();
	// std::cout << "[LogicalAnyJoin] Created OuterJoinOp with " << rightBindings.size()
	          // << " nullable mapped column(s)" << std::endl;
}

} // namespace duckdb
