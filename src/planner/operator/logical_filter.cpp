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

MLIRAttributeInfo& LogicalFilter::resolveColumnBindingToAttributeInfo(ColumnBinding& binding) {
	if (children.empty()) {
		throw std::runtime_error("No children to resolve column binding to attribute info for binding " + binding.ToString());
	}
	return children[0]->resolveColumnBindingToAttributeInfo(binding);
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
	mlir::Value resultExpr = expressions[0]->translateExpression(translationContext, predBuilder, this);
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
