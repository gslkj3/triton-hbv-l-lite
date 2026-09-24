// Current ordinary planning-cut facts. Historical pattern labels and Bridge
// origin are not route authority. This does not contain predictions or timing.
llvm::json::Object collectOrdinaryPlanningFacts(ModuleOp module) {
  llvm::json::Object facts;
  facts["schema"] = "l.core.planning-facts.v1";
  auto entry = findEntryPoint(module);
  if (failed(entry)) {
    facts["extractable"] = false;
    facts["reason"] = "entry_not_unique";
    return facts;
  }
  facts["extractable"] = true;
  auto functions = reachablePlanningFunctions(*entry);
  auto locators = planningLoopLocators(functions);
  auto stagesAttr = module->getAttrOfType<IntegerAttr>(kNativeDefaultStagesAttr);
  int64_t stages = stagesAttr ? stagesAttr.getInt() : 0;
  StaticWorkInventory work;
  work.observe(*entry, functions, locators);
  work.append(facts);
  llvm::json::Array census;
  PipelineSummary pipeline;
  for (FuncOp function : functions)
    function.walk([&](scf::ForOp loop) {
      census.push_back(ordinaryLoopCensus(loop, *entry, locators, stages));
      pipeline.observe(loop, stages);
    });
  facts["loop_census"] = std::move(census);
  pipeline.append(facts);
  appendPackingOpportunityFacts(*entry, facts);
  appendProgramDependencyFacts(*entry, facts);
  appendStructuredLoopTopologyFacts(functions, locators, facts);
  appendStructuredWorkFacts(*entry, facts);
  appendMemoryAccessFacts(*entry, facts);
  appendNativePointerAxisFacts(*entry, facts, locators);
  appendSiblingStateFacts(*entry, facts);
  return facts;
}
