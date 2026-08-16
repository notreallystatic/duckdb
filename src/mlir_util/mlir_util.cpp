#include "duckdb/mlir_util/mlir_util.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"

#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/scheduler/Scheduler.h"

#include <iostream>

#include <stack>
#include <unordered_set>
#include <vector>

namespace duckdb {

class ConciseTimingPrinter : public lingodb::execution::TimingProcessor {
	double compilation;
	double execution;

public:
	ConciseTimingPrinter() : compilation(0.0), execution(0.0) {
	}
	void addTiming(const std::unordered_map<std::string, double> &timing) override {
		for (auto [name, t] : timing) {
			if (name == "executionTime") {
				execution = t;
			} else {
				compilation += t;
			}
		}
	}
	void process() override {
		std::cerr << " compilation: " << compilation << " [ms] execution: " << execution << " [ms]" << std::endl;
	}
};

void runMLIR() {
	// std::cout << "Running MLIR  module now\n";
	// std::cout.flush();

	// moduleOp->dump();

	// The LingoDB session (loaded catalog + Arrow table data), the eval runtime, and
	// the scheduler are expensive to build but identical for every query in this
	// process, so they are created once on the first call and reused thereafter. Only
	// the per-query config/executer/module below is rebuilt each time. This keeps the
	// cold-load cost off every query but the first -- run all queries in a single
	// process (one mode) to pay it once. Local statics are initialized thread-safely
	// and lazily on first use (C++11).
	//
	// The LingoDB database directory may be overridden via the second CLI argument
	// (e.g. `duckdb db.duckdb my_lingodb_dir/`), forwarded through the LINGODB_DB_DIR
	// environment variable. Falls back to "bench_db" when no directory is supplied.
	static std::shared_ptr<lingodb::runtime::Session> session = [] {
		bool eagerLoading = std::getenv("LINGODB_BACKEND_ONLY");
		const char *lingodbDbDir = std::getenv("LINGODB_DB_DIR");
		auto s = lingodb::runtime::Session::createSession(lingodbDbDir ? lingodbDbDir : "bench_db", eagerLoading);
		lingodb::compiler::support::eval::init();
		return s;
	}();
	// Hold one scheduler handle for the whole process. startScheduler() returns a
	// handle to the already-running scheduler on later calls, and the scheduler stops
	// only when the last handle is destroyed -- so keeping it static avoids the
	// per-query start/stop.
	static std::unique_ptr<lingodb::scheduler::SchedulerHandle> scheduler = lingodb::scheduler::startScheduler();

	lingodb::execution::ExecutionMode runMode = lingodb::execution::getExecutionMode();
	// std::cout << "Execution mode: " << static_cast<int>(runMode) << "\n";
	// std::cout.flush();
	auto queryExecutionConfig = lingodb::execution::createQueryExecutionConfig(runMode, false);
	queryExecutionConfig->timingProcessor = std::make_unique<ConciseTimingPrinter>();

	auto executer = lingodb::execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *session);
	executer->fromGlobalContext(true);
	lingodb::scheduler::awaitEntryTask(std::make_unique<lingodb::execution::QueryExecutionTask>(std::move(executer)));
}

bool MLIRStringInfo::isEqual(std::string a, std::string b) {
	return a == b;
}
std::string MLIRStringInfo::getEmptyKey() {
	return "";
}
std::string MLIRStringInfo::getTombstoneKey() {
	return "-";
}

size_t MLIRStringInfo::getHashValue(std::string str) {
	return std::hash<std::string> {}(str);
}

TupleScope::TupleScope(MLIRTranslationContext *context) : context(context) {
	context->currTuple.push(context->currTuple.top());
}

TupleScope::~TupleScope() {
	context->currTuple.pop();
}

DefineScope::DefineScope(MLIRTranslationContext &context) : context(context) {
	context.definedAttributes.push({});
}

DefineScope::~DefineScope() {
	context.definedAttributes.pop();
}

MLIRTranslationContext::MLIRTranslationContext() : currTuple(), resolver() {
	currTuple.push(mlir::Value());
	definedAttributes.push({});
}
mlir::Value MLIRTranslationContext::getCurrentTuple() {
	return currTuple.top();
}
void MLIRTranslationContext::setCurrentTuple(mlir::Value v) {
	currTuple.top() = v;
}
void MLIRTranslationContext::mapAttribute(ResolverScope &scope, std::string name,
                                          const lingodb::compiler::dialect::tuples::Column *attr) {
	definedAttributes.top().push_back({name, attr});
	resolver.insertIntoScope(&scope, std::move(name), attr);
}
const lingodb::compiler::dialect::tuples::Column *MLIRTranslationContext::getAttribute(std::string name) {
	const auto *res = resolver.lookup(name);
	if (!res) {
		// error("could not resolve '" + name + "'");
		throw std::runtime_error("could not resolve '" + name + "'");
	}
	return res;
}

TupleScope MLIRTranslationContext::createTupleScope() {
	return TupleScope(this);
}
MLIRTranslationContext::ResolverScope MLIRTranslationContext::createResolverScope() {
	return ResolverScope(resolver);
}

DefineScope MLIRTranslationContext::createDefineScope() {
	return DefineScope(*this);
}
const std::vector<std::pair<std::string, const lingodb::compiler::dialect::tuples::Column *>> &
MLIRTranslationContext::getAllDefinedColumns() {
	return definedAttributes.top();
}
void MLIRTranslationContext::removeFromDefinedColumns(const lingodb::compiler::dialect::tuples::Column *col) {
	auto &currDefinedColumns = definedAttributes.top();
	auto start = currDefinedColumns.begin();
	auto end = currDefinedColumns.end();
	auto position = std::find_if(start, end, [&](auto el) { return el.second == col; });
	if (position != currDefinedColumns.end()) {
		currDefinedColumns.erase(position);
	}
}

void MLIRTranslationContext::replace(ResolverScope &scope, const lingodb::compiler::dialect::tuples::Column *col,
                                     const lingodb::compiler::dialect::tuples::Column *col2) {
	auto &currDefinedColumns = definedAttributes.top();
	auto start = currDefinedColumns.begin();
	auto end = currDefinedColumns.end();
	std::vector<std::string> toReplace;
	while (start != end) {
		auto position = std::find_if(start, end, [&](auto el) { return el.second == col; });
		if (position != currDefinedColumns.end()) {
			start = position + 1;
			toReplace.push_back(position->first);
		} else {
			start = end;
		}
	}
	for (auto s : toReplace) {
		mapAttribute(scope, s, col2);
	}
}

} // namespace duckdb
