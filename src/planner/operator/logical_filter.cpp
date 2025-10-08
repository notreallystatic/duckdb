#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/parser/expression_util.hpp"
#include "duckdb/planner/expression/list.hpp"

#include <iostream>

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/frontend/SQL/Parser.h"
#include "lingodb/runtime/Session.h"

#include "lingodb/execution/Frontend.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

#include <optional>

namespace duckdb {

mlir::Value translateExpression(unique_ptr<Expression> &expr, mlir::OpBuilder &predBuilder);
void AddMLIRForExpression(unique_ptr<Expression> &expr, MLIRTranslationContext &translationContext, int depth);

void LogicalFilter::AddMLIRSpecific(ClientContext &context, LogicalOperatorType operator_to_process,
                                    unique_ptr<LogicalOperator> &og_tree, MLIRTranslationContext &translationContext,
                                    int depth) {
	string indent = string(depth * 4, ' ');
	std::cout << indent << "[LogicalFilter](AddMLIRSpecific) :: " << LogicalOperatorToString(type) << std::endl;
	for (auto &expr : this->expressions) {
		AddMLIRForExpression(expr, translationContext, depth + 1);
	}
}

mlir::Value translateExpression(unique_ptr<Expression> &expr, MLIRTranslationContext &translationContext,
                                mlir::OpBuilder &predBuilder) {

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	// auto attrManager = mlirContainerInstance.getAttrManager();
	auto moduleOp = mlirContainerInstance.getModuleOp();
	lingodb::compiler::dialect::tuples::ColumnManager &attrManager =
	    moduleOp->getContext()
	        ->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()
	        ->getColumnManager();

	auto loc = predBuilder.getUnknownLoc();

	auto expressionClass = expr->GetExpressionClass();

	std::cout << "[translateExpression] :: " << ExpressionTypeToString(expr->GetExpressionType()) << " "
	          << ExpressionClassToString(expressionClass) << std::endl;

	switch (expressionClass) {
	case ExpressionClass::BOUND_COMPARISON: {
		auto &bound_comparison = (BoundComparisonExpression &)*expr;
		auto left = translateExpression(bound_comparison.left, translationContext, predBuilder);
		auto right = translateExpression(bound_comparison.right, translationContext, predBuilder);
		auto dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::gt; // FIXME: Hard coded for now
		auto ct = lingodb::compiler::frontend::sql::SQLTypeInference::toCommonBaseTypes(predBuilder, {left, right});
		return predBuilder.create<lingodb::compiler::dialect::db::CmpOp>(loc, dbPred, ct[0], ct[1]);
	}
	case ExpressionClass::BOUND_COLUMN_REF: {
		auto &expressionObj = expr->Cast<BoundColumnRefExpression>();
		auto column_name = expr->ToString();
		auto *columnAttr = translationContext.getAttribute(column_name);
		if (columnAttr == nullptr) {
			std::cout << "[translateExpression] Column not found in resolver :: " << column_name << std::endl;
			std::cout.flush();
		} else {
			std::cout << "[translateExpression] Column found in resolver :: " << column_name << std::endl;
			std::cout.flush();
		}
		auto currentTuple = translationContext.getCurrentTuple();
		currentTuple.dump();
		std::cout << "\n";
		std::cout.flush();
		return predBuilder.create<lingodb::compiler::dialect::tuples::GetColumnOp>(
		    loc, columnAttr->type, attrManager.createRef(columnAttr), translationContext.getCurrentTuple());
		break;
	}
	case ExpressionClass::BOUND_CONSTANT: {
		auto &expressionObj = expr->Cast<BoundConstantExpression>();
		switch (expressionObj.value.type().id()) {
		case LogicalTypeId::INTEGER: {
			return predBuilder.create<lingodb::compiler::dialect::db::ConstantOp>(
			    loc, predBuilder.getI32Type(), predBuilder.getI32IntegerAttr(expressionObj.value.GetValue<int32_t>()));
			break;
		}
		default: {
			std::cout << "[translateExpression] Unhandled constant type :: " << expressionObj.value.type().ToString()
			          << std::endl;
			break;
		}
		}
		break;
	}
	default: {
		std::cout << "[translateExpression] Unhandled expression class :: " << ExpressionClassToString(expressionClass)
		          << std::endl;
	}
	}
	return mlir::Value();
}

void AddMLIRForExpression(unique_ptr<Expression> &expr, MLIRTranslationContext &translationContext, int depth) {
	string indent = string(depth * 4, ' ');
	std::cout << indent << "[Expression](AddMLIRForExpression) :: " << ExpressionTypeToString(expr->GetExpressionType())
	          << std::endl;
	auto expressionClass = expr->GetExpressionClass();
	std::cout << indent << "[Expression](AddMLIRForExpression) Class :: " << ExpressionClassToString(expressionClass)
	          << std::endl;

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto &builder = mlirContainerInstance.getBuilder();
	auto loc = builder.getUnknownLoc();
	auto module = mlirContainerInstance.getModuleOp();

	auto *block = new mlir::Block();
	mlir::OpBuilder predBuilder(builder.getContext());
	block->addArgument(lingodb::compiler::dialect::tuples::TupleType::get(builder.getContext()), loc);
	// TODO: translationContext.createTupleScope() Have to add this, skipping for now
	auto tupleScope = translationContext.createTupleScope();
	translationContext.setCurrentTuple(block->getArgument(0));
	predBuilder.setInsertionPointToStart(block);
	mlir::Value result_expr = translateExpression(expr, translationContext, predBuilder);
	predBuilder.create<lingodb::compiler::dialect::tuples::ReturnOp>(loc, result_expr);

	mlirContainerInstance.setPredBlock(block);
}

void LogicalFilter::Walk(int depth) {
	string indent = string(depth * 4, ' ');
	std::cout << indent << "[LogicalFilter](Walk) type :: " << LogicalOperatorToString(type) << std::endl;
	std::cout << indent << "[LogicalFilter](Walk) Expressions :: " << std::endl;

	for (const auto &ex : this->expressions) {
		printExpression(ex, depth + 1);
	}
	std::cout << std::endl;

	for (const auto &child : children) {
		child->Walk(depth + 1);
	}
}

LogicalFilter::LogicalFilter(unique_ptr<Expression> expression) : LogicalOperator(LogicalOperatorType::LOGICAL_FILTER) {
	expressions.push_back(std::move(expression));
	SplitPredicates(expressions);
}

LogicalFilter::LogicalFilter() : LogicalOperator(LogicalOperatorType::LOGICAL_FILTER) {
}

void LogicalFilter::ResolveTypes() {
	types = MapTypes(children[0]->types, projection_map);
}

vector<ColumnBinding> LogicalFilter::GetColumnBindings() {
	return MapBindings(children[0]->GetColumnBindings(), projection_map);
}

// Split the predicates separated by AND statements
// These are the predicates that are safe to push down because all of them MUST
// be true
bool LogicalFilter::SplitPredicates(vector<unique_ptr<Expression>> &expressions) {
	bool found_conjunction = false;
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (expressions[i]->GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
			auto &conjunction = expressions[i]->Cast<BoundConjunctionExpression>();
			found_conjunction = true;
			// AND expression, append the other children
			for (idx_t k = 1; k < conjunction.children.size(); k++) {
				expressions.push_back(std::move(conjunction.children[k]));
			}
			// replace this expression with the first child of the conjunction
			expressions[i] = std::move(conjunction.children[0]);
			// we move back by one so the right child is checked again
			// in case it is an AND expression as well
			i--;
		}
	}
	return found_conjunction;
}

} // namespace duckdb
