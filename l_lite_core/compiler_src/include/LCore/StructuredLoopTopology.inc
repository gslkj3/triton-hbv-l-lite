// Direct-body SSA observations, not a backend schedule or latency model.
void appendStructuredLoopTopologyFacts(
    ArrayRef<FuncOp> reachableFunctions,
    const llvm::DenseMap<Operation *, std::string> &planningCutLoopLocators,
    llvm::json::Object &facts) {
  int64_t structuredLoopCarriedStateCount = 0;
  int64_t structuredLoopCarriedDependencyEdgeCount = 0;
  int64_t structuredLoopDirectBodyOperationCount = 0;
  int64_t structuredLoopDirectBodySSACriticalPathLowerBound = 0;
  llvm::json::Array structuredLoopCarriedDependencyRows;
  llvm::json::Array structuredLoopCarriedUpdates;
  llvm::json::Array structuredLoopUpdateSlices;
  llvm::json::Array structuredLoopCarriedDependencyRowCardinalityHistograms;
  llvm::json::Array structuredLoopOrderedDependencyDistanceHistograms;
  llvm::json::Array structuredLoopDirectBodyDepthHistograms;
  llvm::json::Array structuredLoopYieldDependencyDepths;
  llvm::json::Array structuredLoopOpaqueRegionOperationCounts;
  // A structured loop is not an unknown dependency graph merely because it
  // owns a region.  Preserve a conservative, rule-defined projection of the
  // directly visible scf.for topology.  Nested region operations remain one
  // opaque SSA node, so the reported path is a lower bound rather than an
  // invented backend schedule.
  for (FuncOp function : reachableFunctions)
    function.walk([&](scf::ForOp loop) {
      auto regionArgs = loop.getRegionIterArgs();
      structuredLoopCarriedStateCount += regionArgs.size();
      auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
      // Observation only: preserve typed state/update identity rather than
      // interpreting every carried pointer as a numerical recurrence.
      llvm::json::Array typedUpdates;
      for (auto [index, state] : llvm::enumerate(regionArgs)) {
        Value updated = yield.getOperand(index);
        Operation *definition = updated.getDefiningOp();
        std::string stateType;
        llvm::raw_string_ostream typeStream(stateType);
        state.getType().print(typeStream);
        typeStream.flush();
        llvm::json::Object update;
        update["state_index"] = static_cast<int64_t>(index);
        update["state_type"] = stateType;
        update["yield_defining_operation"] =
            definition ? definition->getName().getStringRef().str() : "";
        update["definition_in_direct_body"] =
            definition && definition->getBlock() == loop.getBody();
        update["definition_owns_regions"] =
            definition && definition->getNumRegions() != 0;
        update["same_state_forwarded"] = updated == state;
        typedUpdates.push_back(std::move(update));
      }
      llvm::json::Object typedLoop;
      typedLoop["locator"] = planningCutLoopLocators.lookup(loop.getOperation());
      typedLoop["states"] = std::move(typedUpdates);
      typedLoop["scope"] = "typed_carried_value_and_immediate_update_not_latency";
      structuredLoopCarriedUpdates.push_back(std::move(typedLoop));
      // Backward SSA slices stop at the direct body boundary. Region-owned
      // results are explicit opaque nodes, never evidence of no dependence.
      llvm::json::Array updateSlices;
      for (auto [index, state] : llvm::enumerate(regionArgs)) {
        llvm::SmallPtrSet<Operation *, 32> slice;
        bool complete = true;
        std::function<void(Value)> visit = [&](Value value) {
          Operation *op = value.getDefiningOp();
          if (!op || op->getBlock() != loop.getBody() ||
              !slice.insert(op).second)
            return;
          if (op->getNumRegions()) {
            complete = false;
            return;
          }
          for (Value operand : op->getOperands())
            visit(operand);
        };
        visit(yield.getOperand(index));
        llvm::json::Array nodes;
        int64_t ordinal = 0;
        for (Operation &op : loop.getBody()->without_terminator()) {
          int64_t current = ordinal++;
          if (!slice.contains(&op))
            continue;
          llvm::json::Object node;
          node["direct_body_ordinal"] = current;
          node["operation"] = op.getName().getStringRef().str();
          node["opaque_region"] = op.getNumRegions() != 0;
          Value dependencyState = state;
          bool depends = llvm::any_of(op.getResults(), [&](Value result) {
            return valueDependsOn(result, dependencyState);
          });
          node["self_dependency_lower_bound"] = depends;
          nodes.push_back(std::move(node));
        }
        llvm::json::Object row;
        row["state_index"] = static_cast<int64_t>(index);
        row["direct_body_slice_complete"] = complete;
        row["nodes"] = std::move(nodes);
        updateSlices.push_back(std::move(row));
      }
      llvm::json::Object slicedLoop;
      slicedLoop["locator"] = planningCutLoopLocators.lookup(loop.getOperation());
      slicedLoop["states"] = std::move(updateSlices);
      structuredLoopUpdateSlices.push_back(std::move(slicedLoop));
      llvm::json::Array loopDependencyRows;
      SmallVector<int64_t> rowCardinalityHistogram(
          regionArgs.size() + 1, 0);
      SmallVector<int64_t> orderedDistanceHistogram(
          regionArgs.empty() ? 0 : 2 * regionArgs.size() - 1, 0);
      for (auto [yieldIndex, yielded] :
           llvm::enumerate(yield.getOperands())) {
        llvm::json::Array dependencyRow;
        int64_t rowCardinality = 0;
        for (auto [stateIndex, state] : llvm::enumerate(regionArgs)) {
          bool dependent = valueDependsOn(yielded, state);
          dependencyRow.push_back(dependent);
          if (!dependent)
            continue;
          ++structuredLoopCarriedDependencyEdgeCount;
          ++rowCardinality;
          int64_t distance = static_cast<int64_t>(stateIndex) -
                             static_cast<int64_t>(yieldIndex);
          size_t bucket = static_cast<size_t>(
              distance + static_cast<int64_t>(regionArgs.size()) - 1);
          ++orderedDistanceHistogram[bucket];
        }
        ++rowCardinalityHistogram[static_cast<size_t>(rowCardinality)];
        loopDependencyRows.push_back(std::move(dependencyRow));
      }

      llvm::DenseMap<Operation *, int64_t> directDepth;
      SmallVector<int64_t> directBodyDepthHistogram(1, 0);
      int64_t loopDepth = 0;
      int64_t opaqueRegionOperationCount = 0;
      for (Operation &operation : loop.getBody()->without_terminator()) {
        ++structuredLoopDirectBodyOperationCount;
        opaqueRegionOperationCount += operation.getNumRegions() != 0;
        int64_t depth = 1;
        for (Value operand : operation.getOperands()) {
          Operation *definition = operand.getDefiningOp();
          if (!definition || definition->getBlock() != loop.getBody())
            continue;
          auto found = directDepth.find(definition);
          if (found != directDepth.end())
            depth = std::max(depth, found->second + 1);
        }
        directDepth[&operation] = depth;
        if (directBodyDepthHistogram.size() <= static_cast<size_t>(depth))
          directBodyDepthHistogram.resize(static_cast<size_t>(depth) + 1, 0);
        ++directBodyDepthHistogram[static_cast<size_t>(depth)];
        loopDepth = std::max(loopDepth, depth);
      }
      structuredLoopDirectBodySSACriticalPathLowerBound = std::max(
          structuredLoopDirectBodySSACriticalPathLowerBound, loopDepth);
      llvm::json::Array yieldDepths;
      for (Value yielded : yield.getOperands()) {
        int64_t depth = 0;
        if (Operation *definition = yielded.getDefiningOp()) {
          auto found = directDepth.find(definition);
          if (found != directDepth.end())
            depth = found->second;
        }
        yieldDepths.push_back(depth);
      }
      auto toJSON = [](ArrayRef<int64_t> values) {
        llvm::json::Array array;
        for (int64_t value : values)
          array.push_back(value);
        return array;
      };
      structuredLoopCarriedDependencyRows.push_back(
          std::move(loopDependencyRows));
      structuredLoopCarriedDependencyRowCardinalityHistograms.push_back(
          toJSON(rowCardinalityHistogram));
      structuredLoopOrderedDependencyDistanceHistograms.push_back(
          toJSON(orderedDistanceHistogram));
      structuredLoopDirectBodyDepthHistograms.push_back(
          toJSON(directBodyDepthHistogram));
      structuredLoopYieldDependencyDepths.push_back(std::move(yieldDepths));
      structuredLoopOpaqueRegionOperationCounts.push_back(
          opaqueRegionOperationCount);
    });
  facts["structured_loop_carried_state_count"] =
      structuredLoopCarriedStateCount;
  facts["structured_loop_carried_dependency_edge_count"] =
      structuredLoopCarriedDependencyEdgeCount;
  facts["structured_loop_direct_body_operation_count"] =
      structuredLoopDirectBodyOperationCount;
  facts["structured_loop_direct_body_ssa_critical_path_lower_bound"] =
      structuredLoopDirectBodySSACriticalPathLowerBound;
  facts["structured_loop_carried_dependency_rows"] =
      std::move(structuredLoopCarriedDependencyRows);
  facts["structured_loop_carried_updates"] =
      std::move(structuredLoopCarriedUpdates);
  facts["structured_loop_update_slices"] =
      std::move(structuredLoopUpdateSlices);
  facts["structured_loop_carried_dependency_row_cardinality_histogram"] =
      std::move(structuredLoopCarriedDependencyRowCardinalityHistograms);
  facts["structured_loop_ordered_dependency_distance_histogram"] =
      std::move(structuredLoopOrderedDependencyDistanceHistograms);
  facts["structured_loop_direct_body_depth_histogram"] =
      std::move(structuredLoopDirectBodyDepthHistograms);
  facts["structured_loop_yield_dependency_depths"] =
      std::move(structuredLoopYieldDependencyDepths);
  facts["structured_loop_opaque_region_operation_count"] =
      std::move(structuredLoopOpaqueRegionOperationCounts);
  facts["structured_loop_dependency_topology_available"] = true;
  // Availability means the projection exists, not that region captures or
  // backend scheduling have been resolved. A false edge is not independence.
  facts["structured_loop_dependency_rows_scope"] =
      "operand_reachability_lower_bound_not_independence_proof";
}
