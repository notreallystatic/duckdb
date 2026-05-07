#include "duckdb/planner/logical_operator.hpp"

#include "duckdb/original/std/sstream.hpp"
#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/tree_renderer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/expression.hpp"

#include <iostream>

namespace relalg = lingodb::compiler::dialect::relalg;
namespace tuples = lingodb::compiler::dialect::tuples;
namespace subop = lingodb::compiler::dialect::subop;

namespace duckdb {

void LogicalOperator::Walk(int depth) {
	std::string indent = std::string(depth * 4, ' ');
	std::cout << indent << "[LogicalOperator](Walk) type :: " << LogicalOperatorToString(type) << std::endl;
	std::cout << indent << "[LogicalOperator](Walk) Expressions :: " << std::endl;
	for (const auto &ex : this->expressions) {
		printExpression(ex, depth + 1);
	}
	auto column_bindings = this->GetColumnBindings();
	std::cout << indent << "[LogicalOperator](Walk) Column Bindings :: " << ColumnBindingsToString(column_bindings) << std::endl;
	std::cout << std::endl;
	std::cout << indent << "[LogicalOperator](Walk) children length :: " << children.size() << std::endl;

	for (const auto &child : children) {
		child->Walk(depth + 1);
	}
}

MLIRAttributeInfo& LogicalOperator::resolveColumnBindingToAttributeInfo(ColumnBinding& binding) {
	if (children.empty()) {
		throw std::runtime_error("resolveColumnBindingToAttributeInfo not implemented for operator type " + LogicalOperatorToString(type));
	}
	return children[0]->resolveColumnBindingToAttributeInfo(binding);
}

mlir::Value LogicalOperator::getMLIRValue() {
	// Check if the value is present or not, otherwise return the child value.
	if (mlirValue) {
		return mlirValue;
	} else if (!children.empty()) {
		return children[0]->getMLIRValue();
	} else {
		throw std::runtime_error("No MLIR value found for this operator and it has no children to derive a value from.");
	}
}

LogicalOperator::LogicalOperator(LogicalOperatorType type)
    : type(type), estimated_cardinality(0), has_estimated_cardinality(false) {
}

LogicalOperator::LogicalOperator(LogicalOperatorType type, vector<unique_ptr<Expression>> expressions)
    : type(type), expressions(std::move(expressions)), estimated_cardinality(0), has_estimated_cardinality(false) {
}

LogicalOperator::~LogicalOperator() {
}

vector<ColumnBinding> LogicalOperator::GetColumnBindings() {
	return {ColumnBinding(0, 0)};
}

void LogicalOperator::SetParamsEstimatedCardinality(InsertionOrderPreservingMap<string> &result) const {
	if (has_estimated_cardinality) {
		result[RenderTreeNode::ESTIMATED_CARDINALITY] = StringUtil::Format("%llu", estimated_cardinality);
	}
}

void LogicalOperator::SetEstimatedCardinality(idx_t _estimated_cardinality) {
	estimated_cardinality = _estimated_cardinality;
	has_estimated_cardinality = true;
}

// LCOV_EXCL_START
string LogicalOperator::ColumnBindingsToString(const vector<ColumnBinding> &bindings) {
	string result = "{";
	for (idx_t i = 0; i < bindings.size(); i++) {
		if (i != 0) {
			result += ", ";
		}
		result += bindings[i].ToString();
	}
	return result + "}";
}

void LogicalOperator::PrintColumnBindings() {
	Printer::Print(ColumnBindingsToString(GetColumnBindings()));
}
// LCOV_EXCL_STOP

string LogicalOperator::GetName() const {
	return LogicalOperatorToString(type);
}

InsertionOrderPreservingMap<string> LogicalOperator::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
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

void LogicalOperator::ResolveOperatorTypes() {
	types.clear();
	// first resolve child types
	for (auto &child : children) {
		child->ResolveOperatorTypes();
	}
	// now resolve the types for this operator
	ResolveTypes();
	D_ASSERT(types.size() == GetColumnBindings().size());
}

vector<ColumnBinding> LogicalOperator::GenerateColumnBindings(idx_t table_idx, idx_t column_count) {
	vector<ColumnBinding> result;
	result.reserve(column_count);
	for (idx_t i = 0; i < column_count; i++) {
		result.emplace_back(table_idx, i);
	}
	return result;
}

vector<LogicalType> LogicalOperator::MapTypes(const vector<LogicalType> &types, const vector<idx_t> &projection_map) {
	if (projection_map.empty()) {
		return types;
	} else {
		vector<LogicalType> result_types;
		result_types.reserve(projection_map.size());
		for (auto index : projection_map) {
			result_types.push_back(types[index]);
		}
		return result_types;
	}
}

vector<ColumnBinding> LogicalOperator::MapBindings(const vector<ColumnBinding> &bindings,
                                                   const vector<idx_t> &projection_map) {
	if (projection_map.empty()) {
		return bindings;
	} else {
		vector<ColumnBinding> result_bindings;
		result_bindings.reserve(projection_map.size());
		for (auto index : projection_map) {
			D_ASSERT(index < bindings.size());
			result_bindings.push_back(bindings[index]);
		}
		return result_bindings;
	}
}

string LogicalOperator::ToString(ExplainFormat format) const {
	auto renderer = TreeRenderer::CreateRenderer(format);
	duckdb::stringstream ss;
	auto tree = RenderTree::CreateRenderTree(*this);
	renderer->ToStream(*tree, ss);
	return ss.str();
}

void LogicalOperator::Verify(ClientContext &context) {
#ifdef DEBUG
	// verify expressions
	for (idx_t expr_idx = 0; expr_idx < expressions.size(); expr_idx++) {
		auto str = expressions[expr_idx]->ToString();
		// verify that we can (correctly) copy this expression
		auto copy = expressions[expr_idx]->Copy();
		auto original_hash = expressions[expr_idx]->Hash();
		auto copy_hash = copy->Hash();
		// copy should be identical to original
		D_ASSERT(expressions[expr_idx]->ToString() == copy->ToString());
		D_ASSERT(original_hash == copy_hash);
		D_ASSERT(Expression::Equals(expressions[expr_idx], copy));

		for (idx_t other_idx = 0; other_idx < expr_idx; other_idx++) {
			// comparison with other expressions
			auto other_hash = expressions[other_idx]->Hash();
			bool expr_equal = Expression::Equals(expressions[expr_idx], expressions[other_idx]);
			if (original_hash != other_hash) {
				// if the hashes are not equal the expressions should not be equal either
				D_ASSERT(!expr_equal);
			}
		}
		D_ASSERT(!str.empty());

		// verify that serialization + deserialization round-trips correctly
		if (expressions[expr_idx]->HasParameter()) {
			continue;
		}
		MemoryStream stream(Allocator::Get(context));
		// We are serializing a query plan
		try {
			BinarySerializer::Serialize(*expressions[expr_idx], stream);
		} catch (NotImplementedException &ex) {
			// ignore for now (FIXME)
			continue;
		}
		// Rewind the stream
		stream.Rewind();

		bound_parameter_map_t parameters;
		auto deserialized_expression = BinaryDeserializer::Deserialize<Expression>(stream, context, parameters);

		// FIXME: expressions might not be equal yet because of statistics propagation
		continue;
		D_ASSERT(Expression::Equals(expressions[expr_idx], deserialized_expression));
		D_ASSERT(expressions[expr_idx]->Hash() == deserialized_expression->Hash());
	}
	D_ASSERT(!ToString().empty());
	for (auto &child : children) {
		child->Verify(context);
	}
#endif
}

void LogicalOperator::AddChild(unique_ptr<LogicalOperator> child) {
	D_ASSERT(child);
	children.push_back(std::move(child));
}

idx_t LogicalOperator::EstimateCardinality(ClientContext &context) {
	// simple estimator, just take the max of the children
	if (has_estimated_cardinality) {
		return estimated_cardinality;
	}
	idx_t max_cardinality = 0;
	for (auto &child : children) {
		max_cardinality = MaxValue(child->EstimateCardinality(context), max_cardinality);
	}
	has_estimated_cardinality = true;
	estimated_cardinality = max_cardinality;
	return estimated_cardinality;
}

void LogicalOperator::Print() {
	Printer::Print(ToString());
}

vector<idx_t> LogicalOperator::GetTableIndex() const {
	return vector<idx_t> {};
}

unique_ptr<LogicalOperator> LogicalOperator::Copy(ClientContext &context) const {
	MemoryStream stream(Allocator::Get(context));
	SerializationOptions options;
	options.serialization_compatibility = SerializationCompatibility::Latest();
	BinarySerializer serializer(stream, options);
	try {
		serializer.Begin();
		this->Serialize(serializer);
		serializer.End();
	} catch (NotImplementedException &ex) {
		ErrorData error(ex);
		throw NotImplementedException("Logical Operator Copy requires the logical operator and all of its children to "
		                              "be serializable: " +
		                              error.RawMessage());
	}
	stream.Rewind();
	bound_parameter_map_t parameters;
	auto op_copy = BinaryDeserializer::Deserialize<LogicalOperator>(stream, context, parameters);
	return op_copy;
}

void LogicalOperator::resolveMLIRValue(MLIRTranslationContext &translationContext, MLIRTranslationContext::ResolverScope &scope) {
	std::cout << "[LogicalOperator](resolveMLIRValue) :: " << LogicalOperatorToString(type) << std::endl;
	for (const auto &child : children) {
		child->resolveMLIRValue(translationContext, scope);
	}
}

string LogicalOperator::resolveTableIndex(idx_t table_index) {
	string tableName = "";
	if (!children.empty()) {
		for (const auto &child : children) {
			tableName = child->resolveTableIndex(table_index);
			if (!tableName.empty()) {
				return tableName;
			}
		}
	}
	return tableName;
}

mlir::Value LogicalOperator::materializeInput = nullptr;

void LogicalOperator::setMaterializeInput(mlir::Value value) {
	materializeInput = value;
}

void LogicalOperator::materializeMLIRValue(MLIRTranslationContext &translationContext, MLIRTranslationContext::ResolverScope &scope) {
	if (this->type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
		// CTE's children[0] is the body definition; children[1] is the consumer (main query).
		return this->children[1]->materializeMLIRValue(translationContext, scope);
	}
	if (this->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return this->children[0]->materializeMLIRValue(translationContext, scope);
	}
	std::cout << "[LogicalOperator](materializeMLIRValue) :: " << LogicalOperatorToString(type) << std::endl;
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

	auto columnBindings = this->GetColumnBindings();
	std::cout << "[LogicalProjection](materializeMLIRValue) :: Column Bindings for projection: " << ColumnBindingsToString(columnBindings) << std::endl;
	int exprIndex = 0;
	for (auto& binding : columnBindings) {
		auto& mlirAttrInfo = this->resolveColumnBindingToAttributeInfo(binding);
		string columnNameWithTable = mlirAttrInfo.table_name; // table_name.column_name
		string columnName = mlirAttrInfo.col_name;
		auto columnAttr = mlirAttrInfo.column;
		std::cout << "[LogicalProjection](materializeMLIRValue) :: Resolving column binding for " << binding.ToString()
			<< " -> " << columnNameWithTable << std::endl;
		auto expr = this->expressions[exprIndex].get();
		string exprName = expr->GetName();
		std::cout << "[LogicalProjection](materializeMLIRValue) :: Expression for column " << columnNameWithTable << " is " << exprName << std::endl;
		names.push_back(builder.getStringAttr(exprName));
		members.push_back(memberManager.createMember(columnName, columnAttr->type));
		attrs.push_back(attrManager.createRef(columnAttr));
		++exprIndex;
	}

	auto localTableType = subop::LocalTableType::get(
		builder.getContext(),
		subop::StateMembersAttr::get(builder.getContext(), members),
		builder.getArrayAttr(names)
	);

	mlir::Value childValue = LogicalOperator::materializeInput;
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
