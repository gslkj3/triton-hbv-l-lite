// Structural observation identity, not legality or execution multiplicity.
// The entry is first; each reachable function definition is visited once.
SmallVector<FuncOp> reachablePlanningFunctions(FuncOp entry) {
  SmallVector<FuncOp> functions{entry};
  llvm::SmallPtrSet<Operation *, 8> seen;
  seen.insert(entry.getOperation());
  for (size_t index = 0; index < functions.size(); ++index) {
    FuncOp function = functions[index];
    function.walk([&](CallOp call) {
      auto callee = dyn_cast_or_null<FuncOp>(call.resolveCallable());
      if (callee && seen.insert(callee.getOperation()).second)
        functions.push_back(callee);
    });
  }
  return functions;
}

llvm::DenseMap<Operation *, std::string>
planningLoopLocators(ArrayRef<FuncOp> functions) {
  llvm::DenseMap<Operation *, std::string> locators;
  int64_t ordinal = 0;
  for (FuncOp function : functions)
    function.walk([&](scf::ForOp loop) {
      locators[loop.getOperation()] =
          "planning-cut.loop." + std::to_string(ordinal++);
    });
  return locators;
}
