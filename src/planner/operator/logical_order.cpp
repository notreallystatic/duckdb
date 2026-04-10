#include "duckdb/planner/operator/logical_order.hpp"

namespace relalg = lingodb::compiler::dialect::relalg;
namespace tuples = lingodb::compiler::dialect::tuples;

namespace duckdb {

LogicalOrder::LogicalOrder(vector<BoundOrderByNode> orders)
    : LogicalOperator(LogicalOperatorType::LOGICAL_ORDER_BY), orders(std::move(orders)) {
}

void LogicalOrder::resolveMLIRValue(MLIRTranslationContext &translationContext, MLIRTranslationContext::ResolverScope &scope) {
	std::cout << "[LogicalOrder](resolveMLIRValue) :: Resolving MLIR Value for LogicalOrder" << std::endl;
	std::cout.flush();

	if (children.size() != 1) {
		std::cout << "[LogicalOrder](resolveMLIRValue) :: Expected exactly one child for LogicalOrder but found " << children.size() << std::endl;
		throw InternalException("LogicalOrder operator should have exactly one child");
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

	auto projectionMap = this->projection_map;
	auto child = children[0].get();
	child->resolveMLIRValue(translationContext, scope);
	auto childValue = child->getMLIRValue();
	auto childColBindings = child->GetColumnBindings();
	std::vector<mlir::Attribute> orderAttributes;
	for (const auto& order : orders) {
		auto expr = order.expression.get();
		if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
			auto& node = expr->Cast<BoundColumnRefExpression>();
			auto binding = node.binding;
			auto &mlirAttrInfo = this->children[0]->resolveColumnBindingToAttributeInfo(binding);
			string columnNameWithTable = mlirAttrInfo.table_name; // table_name.column_name
			string columnName = mlirAttrInfo.col_name; // column_name
			auto columnAttr = mlirAttrInfo.column;

			relalg::SortSpec spec;
			if (order.type == OrderType::ASCENDING) {
				spec = relalg::SortSpec::asc;
			}
			else {
				spec = relalg::SortSpec::desc;
			}
			orderAttributes.push_back(
				relalg::SortSpecificationAttr::get(
					builder.getContext(),
					attrManager.createRef(columnAttr),
					spec
				)
			);
		}
	}
	this->mlirValue = builder.create<relalg::SortOp>(
		loc,
		tuples::TupleStreamType::get(builder.getContext()),
		childValue,
		builder.getArrayAttr(orderAttributes)
	);
	std::cout << "[LogicalOrder](resolveMLIRValue) :: Resolved MLIR Value for LogicalOrder: " << std::endl;
	std::cout.flush();
	this->mlirValue.print(llvm::outs());
	std::cout << std::endl;
}

vector<ColumnBinding> LogicalOrder::GetColumnBindings() {
	auto child_bindings = children[0]->GetColumnBindings();
	if (!HasProjectionMap()) {
		return child_bindings;
	}
	return MapBindings(child_bindings, projection_map);
}

InsertionOrderPreservingMap<string> LogicalOrder::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	string orders_info;
	for (idx_t i = 0; i < orders.size(); i++) {
		if (i > 0) {
			orders_info += "\n";
		}
		orders_info += orders[i].expression->GetName();
	}
	result["__order_by__"] = orders_info;
	SetParamsEstimatedCardinality(result);
	return result;
}

void LogicalOrder::ResolveTypes() {
	const auto child_types = children[0]->types;
	if (!HasProjectionMap()) {
		types = child_types;
	} else {
		types = MapTypes(child_types, projection_map);
	}
}

} // namespace duckdb
