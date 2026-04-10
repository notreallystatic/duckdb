#include "duckdb/planner/operator/logical_get.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/function_serialization.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/main/client_context.hpp"

#include <iostream>

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/frontend/SQL/Parser.h"
#include "lingodb/runtime/Session.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

#include "lingodb/execution/Frontend.h"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

mlir::Type convertDuckDBTypeToNullableType(const LogicalType &type, mlir::MLIRContext *context) {
	auto baseType = getMLIRTypeFromDuckDBLogicalType(type, context);
	return lingodb::compiler::dialect::db::NullableType::get(context, baseType);
}

LogicalGet::LogicalGet() : LogicalOperator(LogicalOperatorType::LOGICAL_GET) {
}

MLIRAttributeInfo& LogicalGet::resolveColumnBindingToAttributeInfo(ColumnBinding& binding) {
	auto table_index = binding.table_index;
	if (table_index != this->table_index) {
		std::cout << "[LogicalGet](resolveColumnBindingToAttributeInfo) :: Table index " << table_index << " does not match LogicalGet's table index " << this->table_index << std::endl;
		throw std::runtime_error("Table index does not match LogicalGet's table index");
	}
	auto& col_ids = GetColumnIds();
	auto catalog_index = col_ids[binding.column_index].GetPrimaryIndex();
	return mlirAttributeInfos[catalog_index];
}

string LogicalGet::getTableName() {
	auto table_entry = GetTable();
	string table_name;

	if (table_entry) {
		// Physical table from catalog
		table_name = table_entry->name;
	} else {
		// Table function (e.g., Arrow file scan)
		// Check parameters for file path
		if (!parameters.empty()) {
			// First parameter is usually the file path for file-based scans
			table_name = parameters[0].ToString();
			// Extract just the filename if needed
			size_t last_slash = table_name.find_last_of("/\\");
			if (last_slash != string::npos) {
				string filename = table_name.substr(last_slash + 1);
				table_name = filename;

				// Remove extension if needed
				size_t last_dot = filename.find_last_of(".");
				if (last_dot != string::npos) {
					table_name = filename.substr(0, last_dot);
				}
			}
		}
	}
	return table_name;
}

void LogicalGet::Walk(int depth) {
	string indent = string(depth * 4, ' ');
	std::cout << indent << "[LogicalGet](Walk) :: " << GetName() << std::endl;

	std::cout << "[LogicalGet](Walk) Table Index :: " << table_index << std::endl;
	// Print column bindings
	auto column_bindings = GetColumnBindings();
	std::cout << indent << "[LogicalGet](Walk) Column Bindings :: " << ColumnBindingsToString(column_bindings) << std::endl;
	// Print column names
	std::cout << indent << "[LogicalGet](Walk) Column Names :: ";
	for (const auto &name : names) {
		std::cout << name << ", ";
	}
	std::cout << std::endl;
	for (auto x : input_table_names) {
		std::cout << indent << "[LogicalGet](Walk) Input Table Name :: " << x << std::endl;
	}

	std::cout << indent << "[LogicalGet](Walk) Table Filters :: " << std::endl;
	table_filters.print();
	// std::cout << std::endl;
	// std::cout << indent << "[LogicalGet](Walk) Expressions :: " << std::endl;

	auto table_entry = GetTable();
	string table_name;

	if (table_entry) {
		// Physical table from catalog
		table_name = table_entry->name;
		std::cout << indent << "Table Name (from catalog): " << table_name << std::endl;
	} else {
		// Table function (e.g., Arrow file scan)
		std::cout << indent << "Scanning via table function: " << function.name << std::endl;

		// Check parameters for file path
		if (!parameters.empty()) {
			// First parameter is usually the file path for file-based scans
			table_name = parameters[0].ToString();
			std::cout << indent << "File Path: " << table_name << std::endl;

			// Extract just the filename if needed
			size_t last_slash = table_name.find_last_of("/\\");
			if (last_slash != string::npos) {
				string filename = table_name.substr(last_slash + 1);
				std::cout << indent << "File Name: " << filename << std::endl;

				// Remove extension if needed
				size_t last_dot = filename.find_last_of(".");
				if (last_dot != string::npos) {
					string table_name_no_ext = filename.substr(0, last_dot);
					std::cout << indent << "Table Name (no ext): " << table_name_no_ext << std::endl;
				}
			}
		}

		// Also check named parameters
		for (const auto &named_param : named_parameters) {
			std::cout << indent << "Named param: " << named_param.first << " = " << named_param.second.ToString()
			          << std::endl;
		}
	}
	for (const auto &ex : expressions) {
		printExpression(ex, depth + 1);
	}
	std::cout << std::endl;
}

LogicalGet::LogicalGet(idx_t table_index, TableFunction function, unique_ptr<FunctionData> bind_data,
                       vector<LogicalType> returned_types, vector<string> returned_names,
                       virtual_column_map_t virtual_columns_p)
    : LogicalOperator(LogicalOperatorType::LOGICAL_GET), table_index(table_index), function(std::move(function)),
      bind_data(std::move(bind_data)), returned_types(std::move(returned_types)), names(std::move(returned_names)),
      virtual_columns(std::move(virtual_columns_p)), extra_info() {
}

optional_ptr<TableCatalogEntry> LogicalGet::GetTable() const {
	if (!function.get_bind_info) {
		return nullptr;
	}
	return function.get_bind_info(bind_data.get()).table;
}

InsertionOrderPreservingMap<string> LogicalGet::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;

	string filters_info;
	bool first_item = true;
	for (auto &kv : table_filters.filters) {
		auto &column_index = kv.first;
		auto &filter = kv.second;
		if (column_index < names.size()) {
			if (!first_item) {
				filters_info += "\n";
			}
			first_item = false;
			filters_info += filter->ToString(names[column_index]);
		}
	}
	result["Filters"] = filters_info;

	if (extra_info.sample_options) {
		result["Sample Method"] = "System: " + extra_info.sample_options->sample_size.ToString() + "%";
	}

	if (!extra_info.file_filters.empty()) {
		result["File Filters"] = extra_info.file_filters;
		if (extra_info.filtered_files.IsValid() && extra_info.total_files.IsValid()) {
			result["Scanning Files"] = StringUtil::Format("%llu/%llu", extra_info.filtered_files.GetIndex(),
			                                              extra_info.total_files.GetIndex());
		}
	}

	if (function.to_string) {
		TableFunctionToStringInput input(function, bind_data.get());
		auto to_string_result = function.to_string(input);
		for (const auto &it : to_string_result) {
			result[it.first] = it.second;
		}
	}
	SetParamsEstimatedCardinality(result);
	return result;
}

void LogicalGet::SetColumnIds(vector<ColumnIndex> &&column_ids) {
	this->column_ids = std::move(column_ids);
}

void LogicalGet::AddColumnId(column_t column_id) {
	column_ids.emplace_back(column_id);
}

void LogicalGet::ClearColumnIds() {
	column_ids.clear();
}

const vector<ColumnIndex> &LogicalGet::GetColumnIds() const {
	return column_ids;
}

vector<ColumnIndex> &LogicalGet::GetMutableColumnIds() {
	return column_ids;
}

vector<ColumnBinding> LogicalGet::GetColumnBindings() {
	if (column_ids.empty()) {
		return {ColumnBinding(table_index, 0)};
	}
	vector<ColumnBinding> result;
	if (projection_ids.empty()) {
		for (idx_t col_idx = 0; col_idx < column_ids.size(); col_idx++) {
			result.emplace_back(table_index, col_idx);
		}
	} else {
		for (auto proj_id : projection_ids) {
			result.emplace_back(table_index, proj_id);
		}
	}
	if (!projected_input.empty()) {
		if (children.size() != 1) {
			throw InternalException("LogicalGet::project_input can only be set for table-in-out functions");
		}
		auto child_bindings = children[0]->GetColumnBindings();
		for (auto entry : projected_input) {
			D_ASSERT(entry < child_bindings.size());
			result.emplace_back(child_bindings[entry]);
		}
	}
	return result;
}

const LogicalType &LogicalGet::GetColumnType(const ColumnIndex &index) const {
	if (index.IsVirtualColumn()) {
		auto entry = virtual_columns.find(index.GetPrimaryIndex());
		if (entry == virtual_columns.end()) {
			throw InternalException("Failed to find referenced virtual column %d", index.GetPrimaryIndex());
		}
		return entry->second.type;
	}
	return returned_types[index.GetPrimaryIndex()];
}

const string &LogicalGet::GetColumnName(const ColumnIndex &index) const {
	if (index.IsVirtualColumn()) {
		auto entry = virtual_columns.find(index.GetPrimaryIndex());
		if (entry == virtual_columns.end()) {
			throw InternalException("Failed to find referenced virtual column %d", index.GetPrimaryIndex());
		}
		return entry->second.name;
	}
	return names[index.GetPrimaryIndex()];
}

column_t LogicalGet::GetAnyColumn() const {
	auto entry = virtual_columns.find(COLUMN_IDENTIFIER_EMPTY);
	if (entry != virtual_columns.end()) {
		// return the empty column if the projection supports it
		return COLUMN_IDENTIFIER_EMPTY;
	}
	entry = virtual_columns.find(COLUMN_IDENTIFIER_ROW_ID);
	if (entry != virtual_columns.end()) {
		// return the rowid column if the projection supports it
		return COLUMN_IDENTIFIER_ROW_ID;
	}
	// otherwise return the first column
	return 0;
}

void LogicalGet::ResolveTypes() {
	if (column_ids.empty()) {
		// no projection - we need to push a column
		column_ids.emplace_back(GetAnyColumn());
	}
	types.clear();
	if (projection_ids.empty()) {
		for (auto &index : column_ids) {
			types.push_back(GetColumnType(index));
		}
	} else {
		for (auto &proj_index : projection_ids) {
			auto &index = column_ids[proj_index];
			types.push_back(GetColumnType(index));
		}
	}
	if (!projected_input.empty()) {
		if (children.size() != 1) {
			throw InternalException("LogicalGet::project_input can only be set for table-in-out functions");
		}
		for (auto entry : projected_input) {
			D_ASSERT(entry < children[0]->types.size());
			types.push_back(children[0]->types[entry]);
		}
	}
}

idx_t LogicalGet::EstimateCardinality(ClientContext &context) {
	// join order optimizer does better cardinality estimation.
	if (has_estimated_cardinality) {
		return estimated_cardinality;
	}
	if (function.cardinality) {
		auto node_stats = function.cardinality(context, bind_data.get());
		if (node_stats && node_stats->has_estimated_cardinality) {
			return node_stats->estimated_cardinality;
		}
	}
	if (!children.empty()) {
		return children[0]->EstimateCardinality(context);
	}
	return 1;
}

void LogicalGet::Serialize(Serializer &serializer) const {
	LogicalOperator::Serialize(serializer);
	serializer.WriteProperty(200, "table_index", table_index);
	serializer.WriteProperty(201, "returned_types", returned_types);
	serializer.WriteProperty(202, "names", names);
	/* [Deleted] (vector<column_t>) "column_ids" */
	serializer.WriteProperty(204, "projection_ids", projection_ids);
	serializer.WriteProperty(205, "table_filters", table_filters);
	FunctionSerializer::Serialize(serializer, function, bind_data.get());
	if (!function.serialize) {
		D_ASSERT(!function.serialize);
		// no serialize method: serialize input values and named_parameters for rebinding purposes
		serializer.WriteProperty(206, "parameters", parameters);
		serializer.WriteProperty(207, "named_parameters", named_parameters);
		serializer.WriteProperty(208, "input_table_types", input_table_types);
		serializer.WriteProperty(209, "input_table_names", input_table_names);
	}
	serializer.WriteProperty(210, "projected_input", projected_input);
	serializer.WritePropertyWithDefault(211, "column_indexes", column_ids);
	serializer.WritePropertyWithDefault(212, "extra_info", extra_info, ExtraOperatorInfo {});
	serializer.WritePropertyWithDefault<optional_idx>(213, "ordinality_idx", ordinality_idx);
}

unique_ptr<LogicalOperator> LogicalGet::Deserialize(Deserializer &deserializer) {
	vector<column_t> legacy_column_ids;

	auto result = unique_ptr<LogicalGet>(new LogicalGet());
	deserializer.ReadProperty(200, "table_index", result->table_index);
	deserializer.ReadProperty(201, "returned_types", result->returned_types);
	deserializer.ReadProperty(202, "names", result->names);
	deserializer.ReadPropertyWithDefault(203, "column_ids", legacy_column_ids);
	deserializer.ReadProperty(204, "projection_ids", result->projection_ids);
	deserializer.ReadProperty(205, "table_filters", result->table_filters);
	auto entry = FunctionSerializer::DeserializeBase<TableFunction, TableFunctionCatalogEntry>(
	    deserializer, CatalogType::TABLE_FUNCTION_ENTRY);
	result->function = entry.first;
	auto &function = result->function;
	auto has_serialize = entry.second;
	unique_ptr<FunctionData> bind_data;
	if (!has_serialize) {
		deserializer.ReadProperty(206, "parameters", result->parameters);
		deserializer.ReadProperty(207, "named_parameters", result->named_parameters);
		deserializer.ReadProperty(208, "input_table_types", result->input_table_types);
		deserializer.ReadProperty(209, "input_table_names", result->input_table_names);
	} else {
		bind_data = FunctionSerializer::FunctionDeserialize(deserializer, function);
	}
	deserializer.ReadProperty(210, "projected_input", result->projected_input);
	deserializer.ReadPropertyWithDefault(211, "column_indexes", result->column_ids);
	result->extra_info = deserializer.ReadPropertyWithExplicitDefault<ExtraOperatorInfo>(212, "extra_info", {});
	deserializer.ReadPropertyWithDefault<optional_idx>(213, "ordinality_idx", result->ordinality_idx);
	if (!legacy_column_ids.empty()) {
		if (!result->column_ids.empty()) {
			throw SerializationException(
			    "LogicalGet::Deserialize - either column_ids or column_indexes should be set - not both");
		}
		for (auto &col_id : legacy_column_ids) {
			result->column_ids.emplace_back(col_id);
		}
	}
	auto &context = deserializer.Get<ClientContext &>();
	virtual_column_map_t virtual_columns;
	if (!has_serialize) {
		TableFunctionRef empty_ref;
		TableFunctionBindInput input(result->parameters, result->named_parameters, result->input_table_types,
		                             result->input_table_names, function.function_info.get(), nullptr, result->function,
		                             empty_ref);

		vector<LogicalType> bind_return_types;
		vector<string> bind_names;
		if (!function.bind) {
			throw InternalException("Table function \"%s\" has neither bind nor (de)serialize", function.name);
		}
		bind_data = function.bind(context, input, bind_return_types, bind_names);
		if (result->ordinality_idx.IsValid()) {
			auto ordinality_pos = bind_return_types.begin() + NumericCast<int64_t>(result->ordinality_idx.GetIndex());
			bind_return_types.emplace(ordinality_pos, LogicalType::BIGINT);
		}
		if (function.get_virtual_columns) {
			virtual_columns = function.get_virtual_columns(context, bind_data.get());
		}
		for (auto &col_id : result->column_ids) {
			if (col_id.IsVirtualColumn()) {
				auto idx = col_id.GetPrimaryIndex();
				auto ventry = virtual_columns.find(idx);
				if (ventry == virtual_columns.end()) {
					throw SerializationException(
					    "Table function deserialization failure - could not find virtual column with id %d", idx);
				}
			} else {
				auto idx = col_id.GetPrimaryIndex();
				auto &ret_type = result->returned_types[idx];
				auto &col_name = result->names[idx];
				if (bind_return_types[idx] != ret_type) {
					throw SerializationException(
					    "Table function deserialization failure in function \"%s\" - column with "
					    "name %s was serialized with type %s, but now has type %s",
					    function.name, col_name, ret_type, bind_return_types[idx]);
				}
			}
		}
		result->returned_types = std::move(bind_return_types);
	} else if (function.get_virtual_columns) {
		virtual_columns = function.get_virtual_columns(context, bind_data.get());
	}
	result->virtual_columns = std::move(virtual_columns);
	result->bind_data = std::move(bind_data);
	return std::move(result);
}

vector<idx_t> LogicalGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalGet::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return StringUtil::Upper(function.name) + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return StringUtil::Upper(function.name);
}

// relalg.basetable  {rows = 0x4156E48FC0000000 : f64, table_identifier = "lineitem"} columns: {l_comment => @lineitem::@l_comment({type = !db.string}), l_commitdate => @lineitem::@l_commitdate({type = !db.date<day>}), l_discount => @lineitem::@l_discount({type = !db.decimal<12, 2>}), l_extendedprice => @lineitem::@l_extendedprice({type = !db.decimal<12, 2>}), l_linenumber => @lineitem::@l_linenumber({type = i32}), l_linestatus => @lineitem::@l_linestatus({type = !db.char<1>}), l_orderkey => @lineitem::@l_orderkey({type = i32}), l_partkey => @lineitem::@l_partkey({type = i32}), l_quantity => @lineitem::@l_quantity({type = !db.decimal<12, 2>}), l_receiptdate => @lineitem::@l_receiptdate({type = !db.date<day>}), l_returnflag => @lineitem::@l_returnflag({type = !db.char<1>}), l_shipdate => @lineitem::@l_shipdate({type = !db.date<day>}), l_shipinstruct => @lineitem::@l_shipinstruct({type = !db.char<25>}), l_shipmode => @lineitem::@l_shipmode({type = !db.char<10>}), l_suppkey => @lineitem::@l_suppkey({type = i32}), l_tax => @lineitem::@l_tax({type = !db.decimal<12, 2>})}
// For the time being, we just want to set the mlirValue to the base table op.
/**
 * NOTE:
 * 	- This works fine for the most part.
 * - Some of the i32 types are being converted to i64, need to see if that will cause an issue or not.
 */
void LogicalGet::resolveMLIRValue(MLIRTranslationContext &translationContext, MLIRTranslationContext::ResolverScope &scope) {
	std::cout << "[LogicalGet](resolveMLIRValue) :: " << GetName() << std::endl;
	const string table_name = getTableName();
	const string scope_name = table_name;

	std::cout << "[LogicalGet](resolveMLIRValue) :: Resolving MLIR value for table: " << table_name << " table_index :: " << table_index << std::endl;

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	D_ASSERT(mlirContainerInstance.getContextPtr() != nullptr);

	auto &mlirContext = mlirContainerInstance.getContext();
	auto &builder = mlirContainerInstance.getBuilder();
	auto module = mlirContainerInstance.getModuleOp();
	lingodb::compiler::dialect::tuples::ColumnManager& attrManager =
		module.getContext()
		->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()
		->getColumnManager();

	std::vector<mlir::NamedAttribute> columns;
	auto table_entry = GetTable();
	unordered_map<string, LogicalType> catalog_types_by_name;
	if (table_entry) {
		for (auto &col : table_entry->GetColumns().Logical()) {
			catalog_types_by_name[col.Name()] = col.Type();
		}
	}

	auto tpch_char_width = [&](const string &table, const string &col) -> idx_t {
		if (table == "lineitem") {
			if (col == "l_returnflag" || col == "l_linestatus") {
				return 1;
			}
			if (col == "l_shipinstruct") {
				return 25;
			}
			if (col == "l_shipmode") {
				return 10;
			}
		}
		if (table == "orders") {
			if (col == "o_orderstatus") {
				return 1;
			}
			if (col == "o_orderpriority" || col == "o_clerk") {
				return 15;
			}
		}
		if (table == "customer") {
			if (col == "c_mktsegment") {
				return 10;
			}
			if (col == "c_phone") {
				return 15;
			}
		}
		if (table == "supplier") {
			if (col == "s_name") {
				return 25;
			}
			if (col == "s_phone") {
				return 15;
			}
		}
		if (table == "nation" && col == "n_name") {
			return 25;
		}
		if (table == "region" && col == "r_name") {
			return 25;
		}
		if (table == "part") {
			if (col == "p_mfgr" || col == "p_type") {
				return 25;
			}
			if (col == "p_brand" || col == "p_container") {
				return 10;
			}
		}
		return 0;
	};

	auto add_column_to_basetable = [&](const string &col_name, const LogicalType &logical_type) {
		mlir::Type col_type;
		if (logical_type.id() == LogicalTypeId::VARCHAR) {
			auto width = tpch_char_width(table_name, col_name);
			if (width == 0) {
				// Keep varlen-like columns in a bounded fixed-size representation for now;
				// this avoids backend crashes on db.string materialization in compiled mode.
				width = 256;
			}
			col_type = lingodb::compiler::dialect::db::CharType::get(&mlirContext, width);
		} else {
			col_type = getMLIRTypeFromDuckDBLogicalType(logical_type, &mlirContext);
		}
		auto attrDef = attrManager.createDef(scope_name, col_name);
		attrDef.getColumn().type = col_type;
		columns.push_back(builder.getNamedAttr(col_name, attrDef));
		translationContext.mapAttribute(scope, col_name, &attrDef.getColumn());
		translationContext.mapAttribute(scope, table_name + "." + col_name, &attrDef.getColumn());

		auto mlirAttrInfo = MLIRAttributeInfo{table_name, col_name, &attrDef.getColumn()};
		mlirAttributeInfos.push_back(mlirAttrInfo);
	};

	if (auto table_entry = GetTable()) {
		for (auto &col : table_entry->GetColumns().Logical()) {
			add_column_to_basetable(col.Name(), col.Type());
		}
	} else {
		// Fallback for non-catalog scans: use currently projected columns.
		for (auto &col : column_ids) {
			auto col_name = GetColumnName(col);
			auto logical_type = GetColumnType(col);
			add_column_to_basetable(col_name, logical_type);
		}
	}



	this->mlirValue = builder.create<lingodb::compiler::dialect::relalg::BaseTableOp>(
	    builder.getUnknownLoc(),
	    lingodb::compiler::dialect::tuples::TupleStreamType::get(builder.getContext()), table_name,
	    builder.getDictionaryAttr(columns));

	// Print the mlie value so far
	std::cout << "[LogicalGet](resolveMLIRValue) BaseTableOp MLIR Value :: ";
	std::cout.flush();
	this->mlirValue.print(llvm::outs());
	std::cout << std::endl;
}

string LogicalGet::resolveTableIndex(idx_t table_index) {
	if (table_index  == this->table_index) {
		return getTableName();
	}
	return "";
}

} // namespace duckdb
