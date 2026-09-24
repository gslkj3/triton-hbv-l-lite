// Common entry and diagnostic support. Included in the compiler implementation
// namespace; depends only on native MLIR/Triton IR and LCore attributes.
std::optional<StringRef> structuralArgumentName(BlockArgument argument) {
  Location location = argument.getLoc();
  while (true) {
    if (auto name = dyn_cast<NameLoc>(location))
      return name.getName().getValue();
    if (auto callsite = dyn_cast<CallSiteLoc>(location)) {
      location = callsite.getCallee();
      continue;
    }
    break;
  }
  return std::nullopt;
}

void reportFailure(ModuleOp module, const Twine &reason) {
  module->setAttr("tt.hbv.materialization_failure",
                  StringAttr::get(module.getContext(), reason.str()));
  module.emitError() << "[HBV_MATERIALIZATION] " << reason;
}

FailureOr<FuncOp> findEntryPoint(ModuleOp module) {
  SmallVector<FuncOp> entries;
  for (FuncOp func : module.getOps<FuncOp>())
    if (func.isPublic())
      entries.push_back(func);
  if (entries.size() != 1)
    return failure();
  return entries.front();
}
