// Current-only pass orchestration. No model, historical plan, or route fallback.
constexpr StringLiteral kCorePlanAttr = "tt.hbv.l.core_plan";
constexpr StringLiteral kCoreCompilerCommit = "7c56a5e40f7fd928dfd5c72902d5def0097db73a";

std::optional<lcore::Plan> readClosedCorePlan(FuncOp entry, std::string &reason) {
  auto raw = entry->getAttrOfType<StringAttr>(kCorePlanAttr);
  if (!raw) {
    reason = "ordinary closed plan missing or mistyped";
    return std::nullopt;
  }
  return lcore::parseBundle(raw.getValue(), kCoreCompilerCommit, reason);
}

class CoreFactsPass : public impl::TritonHBVLoopFactsBase<CoreFactsPass> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << llvm::json::Value(collectOrdinaryPlanningFacts(module));
    stream.flush();
    module->setAttr(kCoreFactsAttr, StringAttr::get(module.getContext(), text));
  }
};

class CoreDecisionPass : public impl::TritonHBVLoopDecisionBase<CoreDecisionPass> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    if (!module->hasAttr(kBundleAttr)) return;
    auto entry = findEntryPoint(module);
    auto raw = module->getAttrOfType<StringAttr>(kBundleAttr);
    std::string reason;
    auto plan = raw ? lcore::parseBundle(raw.getValue(), kCoreCompilerCommit, reason)
                    : std::nullopt;
    if (failed(entry) || !plan || (*entry)->hasAttr(kCorePlanAttr)) {
      reportFailure(module, reason.empty() ? "ordinary plan owner is missing or already closed" : reason);
      return signalPassFailure();
    }
    auto result = plan->pipeline ? closeOrdinaryPipeline(*entry, *plan, reason)
                                : closeOrdinaryUnroll(*entry, *plan, reason);
    if (failed(result)) {
      reportFailure(module, reason);
      return signalPassFailure();
    }
    (*entry)->setAttr(kCorePlanAttr, raw);
    module->removeAttr(kBundleAttr);
  }
};

// A missing plan is a default path only when no selected-route state survives.
bool hasCoreSelectedState(ModuleOp module) {
  bool found = module->hasAttr(kBundleAttr);
  module.walk([&](Operation *op) {
    found |= op->hasAttr(kCorePlanAttr) || op->hasAttr(kRouteAttr) ||
             op->hasAttr(kSubjectAttr) || op->hasAttr(kRoleAttr) ||
             op->hasAttr(kRealizedAttr);
  });
  return found;
}

class CoreMaterializePass : public impl::TritonHBVLoopMaterializeBase<CoreMaterializePass> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    if (!hasCoreSelectedState(module)) return;
    auto entry = findEntryPoint(module);
    std::string reason;
    auto plan = succeeded(entry) ? readClosedCorePlan(*entry, reason) : std::nullopt;
    if (module->hasAttr(kBundleAttr) || !plan) {
      reportFailure(module, "ordinary materialization requires closed function plan: " + reason);
      return signalPassFailure();
    }
    if ((*entry)->hasAttr(kRealizedAttr) ||
        failed(materializeOrdinaryRoute(*entry, *plan, reason))) {
      reportFailure(module, reason.empty() ? "ordinary route already materialized" : reason);
      return signalPassFailure();
    }
  }
};

class CoreValidatePass : public impl::TritonHBVValidateLoopPlanBase<CoreValidatePass> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    if (!hasCoreSelectedState(module)) return;
    auto entry = findEntryPoint(module);
    std::string reason;
    auto plan = succeeded(entry) ? readClosedCorePlan(*entry, reason) : std::nullopt;
    if (module->hasAttr(kBundleAttr) || !plan) {
      reportFailure(module, "ordinary validation requires closed function plan: " + reason);
      return signalPassFailure();
    }
    if (failed(validateOrdinaryRoute(*entry, *plan, reason))) {
      reportFailure(module, reason);
      return signalPassFailure();
    }
  }
};
