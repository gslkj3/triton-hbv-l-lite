// Entry-level launch-axis observations; no candidate selection or timing.
void appendProgramDependencyFacts(FuncOp entry, llvm::json::Object &facts) {
  std::array<int64_t, 8> operationCountByProgramDependencyMask{};
  std::array<int64_t, 8> typeVolumeByProgramDependencyMask{};
  std::array<int64_t, 8> hoistableOperationCountByProgramDependencyMask{};
  std::array<bool, 3> programAxisIndependence{};
  std::array<std::string, 3> programAxisIndependenceReasons{};
  bool programDependencyGraphComplete = true;
  int64_t programSSACriticalPathSteps = 0;
  // Preserve the complete structural distribution of operation dependence
  // on the three launch axes.  The eight masks are not a fitted feature
  // basis: they are a lossless count projection of direct IR dependence and
  // let a later candidate transform derive invariant replication for any
  // grouped-axis subset without using a grid, shape, or operator identity.
  std::array<SmallVector<GetProgramIdOp>, 3> programIdsByAxis;
  entry.walk([&](GetProgramIdOp pid) {
    programIdsByAxis[pid.getAxisAsInt()].push_back(pid);
  });
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (programIdsByAxis[axis].size() != 1) {
      programAxisIndependenceReasons[axis] =
          programIdsByAxis[axis].empty()
              ? "program_axis_unused"
              : "program_axis_id_not_unique";
      continue;
    }
    LoopDependenceCertificate axisCertificate =
        certifyBridgeProgramIndependence(
            entry, programIdsByAxis[axis].front());
    programAxisIndependence[axis] = axisCertificate.safe;
    programAxisIndependenceReasons[axis] =
        axisCertificate.safe ? axisCertificate.kind
                             : axisCertificate.reason;
  }
  llvm::DenseMap<Operation *, int64_t> operationDependencyDepth;
  entry.walk([&](Operation *operation) {
    if (operation == entry.getOperation() ||
        operation->hasTrait<OpTrait::IsTerminator>() ||
        isa<GetProgramIdOp>(operation))
      return;
    unsigned mask = 0;
    for (unsigned axis = 0; axis < 3; ++axis) {
      bool dependent = llvm::any_of(
          operation->getOperands(), [&](Value operand) {
            return llvm::any_of(
                programIdsByAxis[axis], [&](GetProgramIdOp pid) {
                  return valueDependsOn(operand, pid.getResult());
                });
          });
      if (dependent)
        mask |= 1u << axis;
    }
    int64_t typeVolume = 1;
    for (Value operand : operation->getOperands())
      typeVolume = std::max(
          typeVolume, staticElementCount(operand.getType()));
    for (Value result : operation->getResults())
      typeVolume = std::max(
          typeVolume, staticElementCount(result.getType()));
    ++operationCountByProgramDependencyMask[mask];
    typeVolumeByProgramDependencyMask[mask] += typeVolume;
    if (operation->getNumRegions() == 0 &&
        isMemoryEffectFree(operation))
      ++hoistableOperationCountByProgramDependencyMask[mask];
    if (operation->getNumRegions() != 0)
      programDependencyGraphComplete = false;
    int64_t depth = 1;
    for (Value operand : operation->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (!definition)
        continue;
      auto found = operationDependencyDepth.find(definition);
      if (found == operationDependencyDepth.end()) {
        // A definition outside the ordered, regionless entry projection
        // prevents an exact SSA critical-path claim.
        if (!isa<GetProgramIdOp>(definition) &&
            !isa<arith::ConstantOp>(definition))
          programDependencyGraphComplete = false;
        continue;
      }
      depth = std::max(depth, found->second + 1);
    }
    operationDependencyDepth[operation] = depth;
    programSSACriticalPathSteps =
        std::max(programSSACriticalPathSteps, depth);
  });
  llvm::json::Array dependencyOperationCounts;
  llvm::json::Array dependencyTypeVolumes;
  llvm::json::Array dependencyHoistableCounts;
  for (unsigned mask = 0; mask < 8; ++mask) {
    dependencyOperationCounts.push_back(
        operationCountByProgramDependencyMask[mask]);
    dependencyTypeVolumes.push_back(
        typeVolumeByProgramDependencyMask[mask]);
    dependencyHoistableCounts.push_back(
        hoistableOperationCountByProgramDependencyMask[mask]);
  }
  facts["program_dependency_mask_operation_counts"] =
      std::move(dependencyOperationCounts);
  facts["program_dependency_mask_type_volumes"] =
      std::move(dependencyTypeVolumes);
  facts["program_dependency_mask_hoistable_operation_counts"] =
      std::move(dependencyHoistableCounts);
  facts["program_dependency_graph_complete"] =
      programDependencyGraphComplete;
  facts["program_ssa_critical_path_steps"] =
      programDependencyGraphComplete ? programSSACriticalPathSteps : 0;
  llvm::json::Array axisIndependence;
  llvm::json::Array axisIndependenceReasons;
  for (unsigned axis = 0; axis < 3; ++axis) {
    axisIndependence.push_back(programAxisIndependence[axis]);
    axisIndependenceReasons.push_back(
        programAxisIndependenceReasons[axis]);
  }
  facts["program_axis_independence"] = std::move(axisIndependence);
  facts["program_axis_independence_reasons"] =
      std::move(axisIndependenceReasons);
}
