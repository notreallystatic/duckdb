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

#include "duckdb/mlir_util/mlir_util.hpp"

namespace duckdb {

mlir::Type convertDuckDBTypeToMLIRType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		return mlir::IntegerType::get(&MLIRContainer::context, 1);
	case LogicalTypeId::TINYINT:
		return mlir::IntegerType::get(&MLIRContainer::context, 8);
	case LogicalTypeId::SMALLINT:
		return mlir::IntegerType::get(&MLIRContainer::context, 16);
	case LogicalTypeId::INTEGER:
		return mlir::IntegerType::get(&MLIRContainer::context, 32);
	case LogicalTypeId::BIGINT:
		return mlir::IntegerType::get(&MLIRContainer::context, 64);
	case LogicalTypeId::HUGEINT:
		return mlir::IntegerType::get(&MLIRContainer::context, 128);
	case LogicalTypeId::FLOAT:
		return mlir::Float32Type::get(&MLIRContainer::context);
	case LogicalTypeId::DOUBLE:
		return mlir::Float64Type::get(&MLIRContainer::context);
	case LogicalTypeId::VARCHAR:
		return lingodb::compiler::dialect::db::StringType::get(&MLIRContainer::context);
	default:
		throw InternalException("Unsupported type for MLIR conversion");
	}
}

mlir::Type convertDuckDBTypeToNullableType(const LogicalType &type) {
	auto baseType = convertDuckDBTypeToMLIRType(type);
	return lingodb::compiler::dialect::db::NullableType::get(&MLIRContainer::context, baseType);
}

LogicalGet::LogicalGet() : LogicalOperator(LogicalOperatorType::LOGICAL_GET) {
}

void LogicalGet::Walk(ClientContext &context) {
	auto table_catalog = GetTable();
	if (!table_catalog) {
		std::cout << "LogicalGet operator with no associated table." << std::endl;
		std::cout << "Table index :: " << table_index << std::endl;
		// print the table name
		std::cout << "Table name :: " << GetName() << std::endl;
		const string table_name = "demo_table";

		auto &builder = duckdb::MLIRContainer::builder;
		auto loc = builder.getUnknownLoc();
		auto &module = duckdb::MLIRContainer::moduleOp;
		std::string scopeName = table_name;
		std::vector<mlir::NamedAttribute> columns;
		auto &mlirContext = duckdb::MLIRContainer::context;
		// TranslationContext translationContext;
		// auto scope = translationContext.createResolverScope();

		lingodb::compiler::dialect::tuples::ColumnManager &attrManager =
		    module.getContext()
		        ->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()
		        ->getColumnManager();

		for (auto &col : column_ids) {
			auto attrDef = attrManager.createDef(scopeName, GetColumnName(col));
			attrDef.getColumn().type = convertDuckDBTypeToMLIRType(GetColumnType(col));
			columns.push_back(builder.getNamedAttr(GetColumnName(col), attrDef));
			// mlirContext.mapAttribute()
		}

		// // Create a main function to hold the query logic
		// auto funcOp = builder.create<mlir::func::FuncOp>(loc, "main", builder.getFunctionType({}, {}));
		// auto entryBlock = funcOp.addEntryBlock();
		// builder.setInsertionPointToStart(entryBlock);

		// // --- Generate the Relational Algebra MLIR ---

		// // 1. Get the table and column metadata (simulated from LogicalGet)
		// mlir::StringAttr tableIdentifier = builder.getStringAttr(table_name);

		// lingodb::compiler::dialect::tuples::ColumnManager &attrManager;

		// mlir::SmallVector<mlir::NamedAttribute> columnsAttrs;
		// mlir::SmallVector<mlir::Type> resultTypes; // <-- changed from Attribute to Type
		// mlir::SmallVector<mlir::NamedAttribute> basetable_attrs;

		// for (auto &col : column_ids) {
		// 	mlir::Type colType = convertDuckDBTypeToNullableType(GetColumnType(col));
		// 	basetable_attrs.push_back(
		// 	    MLIRContainer::builder.getNamedAttr(GetColumnName(col), mlir::TypeAttr::get(colType)));
		// 	resultTypes.push_back(colType); // <-- changed from TypeAttr to Type
		// }
		// // 1. Create the Query op
		// auto queryOp = builder.create<lingodb::compiler::dialect::relalg::QueryOp>(loc,
		// mlir::TypeRange(resultTypes)); auto queryRegion = &queryOp.getRegion();
		// builder.setInsertionPointToStart(&queryRegion->front());

		// std::cout << "Module dump :: \n";
		// module.dump();
		// std::cout << "QueryOp dump :: \n";
		// queryOp.dump();

		// // 2. Create the BaseTableOp
		// auto basetable_dict = builder.getDictionaryAttr(basetable_attrs);
		// auto baseTableOp = builder.create<lingodb::compiler::dialect::relalg::BaseTableOp>(
		//     loc, mlir::TypeRange(resultTypes), tableIdentifier, basetable_dict);

		// // 3. Create the MaterializeOp and QueryReturnOp
		// mlir::SmallVector<mlir::NamedAttribute> remapping;
		// for (auto &col : column_ids) {
		// 	remapping.push_back(builder.getNamedAttr(builder.getStringAttr(GetColumnName(col)),
		// 	                                         builder.getStringAttr(GetColumnName(col))));
		// }

		// // Convert NamedAttribute vector to Attribute vector for getArrayAttr
		// mlir::SmallVector<mlir::Attribute> remapping_attrs;
		// for (auto &named_attr : remapping) {
		// 	remapping_attrs.push_back(named_attr.getValue());
		// }

		// auto materializeOp = builder.create<lingodb::compiler::dialect::relalg::MaterializeOp>(
		//     loc, mlir::TypeRange(resultTypes), baseTableOp.getResult(), builder.getArrayAttr(remapping_attrs));

		// builder.create<lingodb::compiler::dialect::relalg::QueryReturnOp>(loc, materializeOp.getResult());

		// builder.setInsertionPointToEnd(MLIRContainer::moduleOp.getBody());
		// builder.create<mlir::func::ReturnOp>(loc);
	} else {
		auto table_name = table_catalog ? table_catalog->name : "<unknown table>";

		const auto &column_list = table_catalog->GetColumns();

		for (auto &col : column_list.Logical()) {
			std::cout << " - " << col.Name() << " (" << col.Type().ToString() << ")" << std::endl;
		}

		std::cout << "Walking LogicalGet operator" << std::endl;
	}
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

} // namespace duckdb
