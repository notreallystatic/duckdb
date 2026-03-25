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

bool LogicalFilter::found_conjunction_and = false; // Initialize static variable

mlir::Value translateExpression(unique_ptr<Expression> &expr, mlir::OpBuilder &predBuilder);
void AddMLIRForExpression(unique_ptr<Expression> &expr, MLIRTranslationContext &translationContext, int depth);

void LogicalFilter::AddMLIRSpecific(ClientContext &context, LogicalOperatorType operator_to_process,
                                    unique_ptr<LogicalOperator> &og_tree, MLIRTranslationContext &translationContext,
                                    int depth) {
	// std::cout << "AND conjunction status :: " << (LogicalFilter::found_conjunction_and ? "true" : "false") <<
	// std::endl; string indent = string(depth * 4, ' '); std::cout << indent << "[LogicalFilter](AddMLIRSpecific) :: "
	// << LogicalOperatorToString(type) << std::endl;
	for (auto &expr : this->expressions) {
		AddMLIRForExpression(expr, translationContext, depth + 1);
	}
}

mlir::Value translateExpression(unique_ptr<Expression>& expr, MLIRTranslationContext& translationContext,
	mlir::OpBuilder& predBuilder) {

	auto& mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	// auto attrManager = mlirContainerInstance.getAttrManager();
	auto moduleOp = mlirContainerInstance.getModuleOp();
	lingodb::compiler::dialect::tuples::ColumnManager& attrManager =
		moduleOp->getContext()
		->getLoadedDialect<lingodb::compiler::dialect::tuples::TupleStreamDialect>()
		->getColumnManager();

	auto loc = predBuilder.getUnknownLoc();

	auto expressionClass = expr->GetExpressionClass();

	// std::cout << "[translateExpression] :: " << ExpressionTypeToString(expr->GetExpressionType()) << " "
	//           << ExpressionClassToString(expressionClass) << std::endl;

	switch (expressionClass) {
	case ExpressionClass::BOUND_CONJUNCTION: {
		auto& bound_conjunction = (BoundConjunctionExpression&)*expr;
		// std::cout << "[translateExpression] BOUND_CONJUNCTION with children size :: "
		//           << bound_conjunction.children.size() << std::endl;
		// std::cout << "[translateExpression] BOUND_CONJUNCTION type :: "
		//           << ExpressionTypeToString(bound_conjunction.type) << std::endl;
		std::vector<mlir::Value> childExprs;
		for (auto& child : bound_conjunction.children) {
			childExprs.push_back(translateExpression(child, translationContext, predBuilder));
		}
		switch (bound_conjunction.type) {
		case ExpressionType::CONJUNCTION_AND: {
			return predBuilder.create<lingodb::compiler::dialect::db::AndOp>(loc, childExprs);
		case ExpressionType::CONJUNCTION_OR: {
			return predBuilder.create<lingodb::compiler::dialect::db::OrOp>(loc, childExprs);
		}
		default:
			// std::cout << "[translateExpression] Unhandled conjunction type :: "
			//           << ExpressionTypeToString(bound_conjunction.type) << std::endl;
			throw std::runtime_error("Unhandled conjunction type");
		} break;
		}
	}
	case ExpressionClass::BOUND_COMPARISON: {
		auto& bound_comparison = (BoundComparisonExpression&)*expr;
		auto left = translateExpression(bound_comparison.left, translationContext, predBuilder);
		auto right = translateExpression(bound_comparison.right, translationContext, predBuilder);
		lingodb::compiler::dialect::db::DBCmpPredicate dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::eq;
		auto comparison_type = expr->GetExpressionType();
		switch (comparison_type) {
		case ExpressionType::COMPARE_EQUAL:
			dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::eq;
			break;
		case ExpressionType::COMPARE_NOTEQUAL:
			dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::neq;
			break;
		case ExpressionType::COMPARE_LESSTHAN:
			dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::lt;
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::lt;
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::gt;
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			dbPred = lingodb::compiler::dialect::db::DBCmpPredicate::gte;
			break;
		default:
			// std::cout << "[translateExpression] Unhandled comparison type :: "
			//   << ExpressionTypeToString(comparison_type) << std::endl;
			throw std::runtime_error("Unhandled comparison type");
		}
		auto ct = lingodb::compiler::frontend::sql::SQLTypeInference::toCommonBaseTypes(predBuilder, { left, right });
		return predBuilder.create<lingodb::compiler::dialect::db::CmpOp>(loc, dbPred, ct[0], ct[1]);
	}
	case ExpressionClass::BOUND_COLUMN_REF: {
		auto& expressionObj = expr->Cast<BoundColumnRefExpression>();
		auto column_name = expr->ToString();
		auto* columnAttr = translationContext.getAttribute(column_name);
		// if (columnAttr == nullptr) {
		// 	std::cout << "[translateExpression] Column not found in resolver :: " << column_name << std::endl;
		// 	std::cout.flush();
		// } else {
		// 	std::cout << "[translateExpression] Column found in resolver :: " << column_name << std::endl;
		// 	std::cout.flush();
		// }
		auto currentTuple = translationContext.getCurrentTuple();
		// currentTuple.dump();
		// std::cout << "\n";
		// std::cout.flush();
		return predBuilder.create<lingodb::compiler::dialect::tuples::GetColumnOp>(
			loc, columnAttr->type, attrManager.createRef(columnAttr), translationContext.getCurrentTuple());
		break;
	}
	case ExpressionClass::BOUND_CONSTANT: {
		auto& expressionObj = expr->Cast<BoundConstantExpression>();
		switch (expressionObj.value.type().id()) {
		case LogicalTypeId::INTEGER: {
			return predBuilder.create<lingodb::compiler::dialect::db::ConstantOp>(
				loc, predBuilder.getI32Type(), predBuilder.getI32IntegerAttr(expressionObj.value.GetValue<int32_t>()));
			break;
		}
		case LogicalTypeId::VARCHAR:
		case LogicalTypeId::CHAR: {
			auto strVal = expressionObj.value.GetValue<string>();
			auto strType = lingodb::compiler::dialect::db::CharType::get(predBuilder.getContext(), strVal.size());
			return predBuilder.create<lingodb::compiler::dialect::db::ConstantOp>(loc, strType,
				predBuilder.getStringAttr(strVal));
			break;
		}
		case LogicalTypeId::FLOAT: { // TODO: Testing pending
			string expressionValue = expressionObj.value.ToString();
			auto floatVal = expressionObj.value.GetValue<float>();
			// get the integer part and decimal part of the floatVal
			auto intPart = static_cast<unsigned long>(floatVal);
			auto decimalPart = floatVal - intPart;
			// convert the decimal part to an integer by multiplying it with 10^6
			auto decimalIntPart = static_cast<unsigned long>(decimalPart * 10000000);
			return predBuilder.create<lingodb::compiler::dialect::db::ConstantOp>(
				loc,
				lingodb::compiler::dialect::db::DecimalType::get(predBuilder.getContext(), intPart, decimalIntPart),
				predBuilder.getStringAttr(expressionValue));
		}
		default: {
			// std::cout << "[translateExpression] Unhandled constant type :: " << expressionObj.value.type().ToString()
			//   << std::endl;
			break;
		}
		}
		break;
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto& bound_op = (BoundOperatorExpression&)*expr;
		if (bound_op.children.size() != 1) {
			// std::cout << "[translateExpression] BOUND_OPERATOR with unknown children size :: "
			//   << ExpressionTypeToString(expr->GetExpressionType()) << std::endl;
			throw std::runtime_error("BOUND_OPERATOR with unhandled children size");
		}
		auto exprResult = translateExpression(bound_op.children[0], translationContext, predBuilder);
		if (mlir::isa<lingodb::compiler::dialect::db::NullableType>(exprResult.getType())) {
			auto isNull = predBuilder.create<lingodb::compiler::dialect::db::IsNullOp>(loc, exprResult);
			if (bound_op.type == ExpressionType::OPERATOR_IS_NOT_NULL) {
				return predBuilder.create<lingodb::compiler::dialect::db::NotOp>(loc, isNull);
			}
			else if (bound_op.type == ExpressionType::OPERATOR_IS_NULL) {
				return isNull;
			}
			else {
				// std::cout << "[translateExpression] Unhandled bound operator type :: "
				//   << ExpressionTypeToString(bound_op.type) << std::endl;
				throw std::runtime_error("Unhandled bound operator type");
			}
		}
		else {
			return predBuilder.create<lingodb::compiler::dialect::db::ConstantOp>(
				loc, predBuilder.getI1Type(),
				predBuilder.getIntegerAttr(predBuilder.getI1Type(),
					bound_op.type == ExpressionType::OPERATOR_IS_NOT_NULL));
		}
	}
	default: {
		std::cout << "[translateExpression] Unhandled expression class :: " <<
			ExpressionClassToString(expressionClass)
			<< std::endl;
	}
	}
	return mlir::Value();
}

void AddMLIRForExpression(unique_ptr<Expression> &expr, MLIRTranslationContext &translationContext, int depth) {
	// string indent = string(depth * 4, ' ');
	// std::cout << indent << "[Expression](AddMLIRForExpression) :: " <<
	// ExpressionTypeToString(expr->GetExpressionType())
	//           << std::endl;
	// auto expressionClass = expr->GetExpressionClass();
	// std::cout << indent << "[Expression](AddMLIRForExpression) Class :: " << ExpressionClassToString(expressionClass)
	//           << std::endl;

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
	LogicalFilter::found_conjunction_and = SplitPredicates(expressions);
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
	return false;
	// bool found_conjunction = false;
	// for (idx_t i = 0; i < expressions.size(); i++) {
	// 	if (expressions[i]->GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
	// 		auto &conjunction = expressions[i]->Cast<BoundConjunctionExpression>();
	// 		found_conjunction = true;
	// 		// AND expression, append the other children
	// 		for (idx_t k = 1; k < conjunction.children.size(); k++) {
	// 			expressions.push_back(std::move(conjunction.children[k]));
	// 		}
	// 		// replace this expression with the first child of the conjunction
	// 		expressions[i] = std::move(conjunction.children[0]);
	// 		// we move back by one so the right child is checked again
	// 		// in case it is an AND expression as well
	// 		i--;
	// 	}
	// }
	// return LogicalFilter::found_conjunction_and = found_conjunction;
}

/**
 * Create the relalg.selection operation here for the filter operator.
 * Reference:
 * def SelectionOp : RelAlg_Op<"selection",
        [Pure,Operator,PredicateOperator,TupleLamdaOperator,
         UnaryOperator,DeclareOpInterfaceMethods<ColumnFoldable>]> {
    let summary = "selection operation";
    let description = [{
        Filter tuple stream, the region returns `1` iff the value should be
        contained in the output stream.
    }];

    let arguments = (ins TupleStream:$rel);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$predicate);
    let assemblyFormat = [{ $rel custom<CustRegion>($predicate) attr-dict-with-keyword }];
    let extraClassDeclaration = [{
        mlir::LogicalResult foldColumns(dialect::relalg::ColumnFoldInfo& columnInfo);
        lingodb::compiler::dialect::relalg::FunctionalDependencies getFDs();
    }];
}
 * In the region, we will have operations that compute the predicate for the filter op.
 * We call the resolve Value on children and use that value as the input to the predicate operations.
 */
void LogicalFilter::resolveMLIRValue(MLIRTranslationContext &translationContext, MLIRTranslationContext::ResolverScope &scope) {
	for (auto &child : children) {
		child->resolveMLIRValue(translationContext, scope);
	}

	if (children.size() != 1) {
		std::cout << "LogicalFilter should have exactly one child, but found :: " << children.size() << std::endl;
		return;
	}
	auto inputValue = children[0]->getMLIRValue();
	std::cout << "LogicalFilter :: Resolving MLIR Value for filter with inputValue :: " << std::endl;
	inputValue.print(llvm::outs());
	std::cout << std::endl;

	auto &mlirContainerInstance = lingodb::execution::MLIRContainer::getInstance();
	auto &builder = mlirContainerInstance.getBuilder();
	auto loc = builder.getUnknownLoc();

	auto *predBlock = new mlir::Block();
	mlir::OpBuilder predBuilder(builder.getContext());
	predBlock->addArgument(lingodb::compiler::dialect::tuples::TupleType::get(builder.getContext()), loc);
	auto tupleScope = translationContext.createTupleScope();
	translationContext.setCurrentTuple(predBlock->getArgument(0));
	predBuilder.setInsertionPointToStart(predBlock);

	// Add the expression and return it as a result.
	mlir::Value resultExpr = expressions[0]->translateExpression(translationContext, predBuilder);
	predBuilder.create<lingodb::compiler::dialect::tuples::ReturnOp>(loc, resultExpr);
	auto selectionOp = builder.create<lingodb::compiler::dialect::relalg::SelectionOp>(
		loc,
		lingodb::compiler::dialect::tuples::TupleStreamType::get(builder.getContext()),
		inputValue
	);
	selectionOp.getPredicate().push_back(predBlock);

	this->mlirValue = selectionOp.getResult();

	std::cout << "LogicalFilter MLIR Value :: " << std::endl;
	auto op = selectionOp.getOperation();
	if (op && op->getBlock()) {
		op->getBlock()->print(llvm::outs());
		llvm::outs() << "\n";
	}

	std::cout.flush();
	this->mlirValue.print(llvm::outs());
	std::cout << std::endl;
}

} // namespace duckdb
