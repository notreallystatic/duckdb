#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"

#include "duckdb/main/config.hpp"

#include <iostream>

namespace relalg = lingodb::compiler::dialect::relalg;
namespace tuples = lingodb::compiler::dialect::tuples;
namespace subop = lingodb::compiler::dialect::subop;

namespace duckdb {

LogicalProjection::LogicalProjection(idx_t table_index, vector<unique_ptr<Expression>> select_list)
    : LogicalOperator(LogicalOperatorType::LOGICAL_PROJECTION, std::move(select_list)), table_index(table_index) {
}

vector<ColumnBinding> LogicalProjection::GetColumnBindings() {
	return GenerateColumnBindings(table_index, expressions.size());
}

void LogicalProjection::ResolveTypes() {
	for (auto &expr : expressions) {
		types.push_back(expr->return_type);
	}
}

vector<idx_t> LogicalProjection::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalProjection::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

void LogicalProjection::resolveMLIRValue(MLIRTranslationContext& translationContext, MLIRTranslationContext::ResolverScope& scope) {
	std::cout << "[LogicalProjection](resolveMLIRValue) :: " << LogicalOperatorToString(type) << std::endl;
	for (const auto &child: children) {
		child->resolveMLIRValue(translationContext, scope);
	}
}

string LogicalProjection::resolveColumnBinding(ColumnBinding& binding) {
	string tableName = this->resolveTableIndex(binding.table_index - 1);
	auto columnIndex = binding.column_index;
	if (expressions.size() >= columnIndex) {
		auto expr = expressions[columnIndex].get();
		if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
			auto &columnRefExpr = expr->Cast<BoundColumnRefExpression>();
			auto columnName = columnRefExpr.ToString();
			std::cout << "[LogicalProjection](resolveColumnBinding) :: Resolving column binding for table index " << binding.table_index
			          << " and column index " << columnIndex << " with column name " << columnName << std::endl;
			return tableName.empty() ? columnName : tableName + "." + columnName;
		}
	}
	return children.empty() ? "" : children[0]->resolveColumnBinding(binding);
}

void LogicalProjection::materializeMLIRValue(MLIRTranslationContext& translationContext, MLIRTranslationContext::ResolverScope& scope) {
	std::cout << "[LogicalProjection](materializeMLIRValue) :: " << LogicalOperatorToString(type) << std::endl;
	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto &builder = mlirContainerInstance.getBuilder();
	auto &context = mlirContainerInstance.getContext();
	auto module = mlirContainerInstance.getModuleOp();
	auto *mainBlock = mlirContainerInstance.mainBlock;
	auto *queryBlock = mlirContainerInstance.queryBlock;
	auto loc = builder.getUnknownLoc();
	tuples::ColumnManager& attrManager =
		module.getContext()
		->getLoadedDialect<tuples::TupleStreamDialect>()
		->getColumnManager();
	auto &memberManager = builder.getContext()
		->getLoadedDialect<subop::SubOperatorDialect>()
		->getMemberManager();

	llvm::SmallVector<subop::Member> members;
	std::vector<mlir::Attribute> names;
	std::vector<mlir::Attribute> attrs;
	for (const auto &expr: expressions) {
		if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
			auto& node = expr->Cast<BoundColumnRefExpression>();
			auto binding = node.binding;
			string columnNameWithTable = children[0]->resolveColumnBinding(binding); // table_name.column_name
			string columnName = columnNameWithTable.substr(columnNameWithTable.find(".") + 1); // column_name
			auto columnAttr = translationContext.getAttribute(columnNameWithTable);

			names.push_back(builder.getStringAttr(columnName));
			members.push_back(memberManager.createMember(columnName, columnAttr->type));
			attrs.push_back(attrManager.createRef(columnAttr));
		}
	}
	auto localTableType = subop::LocalTableType::get(
		builder.getContext(),
		subop::StateMembersAttr::get(builder.getContext(), members),
		builder.getArrayAttr(names)
	);

	mlir::Value childValue = children[0]->getMLIRValue();
	std::cout << "[LogicalProjection](materializeMLIRValue) :: Child MLIR Value before materialization: " << std::endl;
	childValue.print(llvm::outs());
	std::cout << std::endl;
	mlir::Value result = builder.create<relalg::MaterializeOp>(
		loc,
		localTableType,
		childValue,
		builder.getArrayAttr(attrs),
		builder.getArrayAttr(names)
	);
	builder.create<relalg::QueryReturnOp>(loc, result);

	if (!mainBlock || !queryBlock) {
		throw InternalException("materializeMLIRValue requires both mainBlock and queryBlock to be initialized");
	}

	// QueryOp and result wiring must live in the function/main block, not inside queryBlock.
	{
		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPointToStart(mainBlock);

		auto queryOp = builder.create<relalg::QueryOp>(
			loc,
			mlir::TypeRange {localTableType},
			mlir::ValueRange {}
		);
		queryOp.getQueryOps().getBlocks().clear();
		queryOp.getQueryOps().push_back(queryBlock);
		std::optional<mlir::Value> queryOpResult = queryOp.getResults()[0];
		builder.create<subop::SetResultOp>(loc, 0, queryOpResult.value());
		builder.create<mlir::func::ReturnOp>(loc);
	}
}
} // namespace duckdb
