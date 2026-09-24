// Current-only pass orchestration. No model, historical plan, or route fallback.
constexpr StringLiteral kCorePlanAttr = "tt.hbv.l.core_plan";

std::optional<lcore::Plan> readClosedCorePlan(FuncOp entry, std::string &reason) {
  auto raw = entry->getAttr(kCorePlanAttr);
  if (!raw) {
    reason = "ordinary closed plan missing or mistyped";
    return std::nullopt;
  }
  return lcore::decodeDecision(raw, reason);
}

class CoreFactsPass : public impl::TritonHBVLoopFactsBase<CoreFactsPass> {
public:
  void runOnOperation() override {
    getOperation().emitError(
        "IR report transport was removed; use query_l_planning_facts or queryPlanningFacts");
    signalPassFailure();
  }
};

class CoreDecisionPass : public impl::TritonHBVLoopDecisionBase<CoreDecisionPass> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    if (!module->hasAttr(kBundleAttr)) return;
    auto entry = findEntryPoint(module);
    auto raw = module->getAttr(kBundleAttr);
    std::string reason;
    auto plan = raw ? lcore::decodeDecision(raw, reason)
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

// These are planning inputs, not backend code-generation attributes. Python
// retains the launch mapping in PreparedIR before reaching this boundary.
void clearConsumedModulePlanningAttributes(ModuleOp module) {
  for (StringRef name : {StringRef(lcore::kCoreFactsAttr),
                        StringRef(lcore::kNativeDefaultStagesAttr),
                        StringRef(lcore::kBridgeDiscoveryAttr),
                        StringRef(lcore::kBridgeRuntimeScalarsAttr),
                        StringRef(lcore::kBridgeFactorAttr),
                        StringRef(lcore::kBridgeRequestedDivisorsAttr),
                        StringRef(lcore::kBridgeGridDivisorAttr),
                        StringRef(lcore::kBridgeGridDivisorYAttr),
                        StringRef(lcore::kBridgeGridDivisorZAttr)})
    module->removeAttr(name);
}

class CoreValidatePass : public impl::TritonHBVValidateLoopPlanBase<CoreValidatePass> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    if (!hasCoreSelectedState(module)) {
      clearConsumedModulePlanningAttributes(module);
      return;
    }
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
    // Planning facts describe the pre-transform cut, not this transformed IR.
    // The closed plan's last consumer is validation above. Neither is a
    // backend code-generation input. Retain compact realization/lineage facts
    // needed by later artifact verification; never strip attributes by prefix.
    clearConsumedModulePlanningAttributes(module);
    (*entry)->removeAttr(kCorePlanAttr);
  }
};
