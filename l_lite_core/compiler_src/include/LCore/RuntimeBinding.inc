// Current named launch binding. Validate a complete replacement before writing
// any new facts; rejection invalidates prior owned facts and stops the pipeline.
void clearBridgeRuntimeFacts(FuncOp entry) {
  for (unsigned i = 0; i < entry.getNumArguments(); ++i)
    entry.removeArgAttr(i, kBridgeBoundScalarAttr);
  entry.walk([&](GetProgramIdOp pid) { pid->removeAttr(kBridgeAxisExtentAttr); });
  entry->removeAttr(lcore::kBridgeRuntimeBindingAppliedAttr);
}

LogicalResult bindBridgeRuntimeFacts(FuncOp entry, ModuleOp module,
                                     std::string &reason) {
  Attribute attribute = module->getAttr(kBridgeRuntimeScalarsAttr);
  if (!attribute) {
    if (entry->hasAttr(lcore::kBridgeRuntimeBindingAppliedAttr))
      clearBridgeRuntimeFacts(entry);
    return success();
  }
  auto reject = [&](StringRef message) {
    clearBridgeRuntimeFacts(entry);
    reason = message.str();
    return failure();
  };
  auto raw = dyn_cast<StringAttr>(attribute);
  if (!raw) return reject("runtime_scalar_binding_malformed");
  auto parsed = llvm::json::parse(raw.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return reject("runtime_scalar_binding_malformed");
  }
  auto *object = parsed->getAsObject();
  auto *values = object ? object->getObject("values_by_name") : nullptr;
  auto *grid = object ? object->getArray("grid") : nullptr;
  if (!object || object->getString("schema") != "triton.loop-bridge.runtime-scalars.v2" ||
      !values || !grid || grid->empty() || grid->size() > 3)
    return reject("runtime_scalar_binding_malformed");
  std::array<int64_t, 3> extents{1, 1, 1};
  for (unsigned axis = 0; axis < grid->size(); ++axis) {
    auto extent = (*grid)[axis].getAsInteger();
    if (!extent || *extent < 1 || *extent > INT32_MAX)
      return reject("runtime_scalar_grid_out_of_domain");
    extents[axis] = *extent;
  }
  for (const auto &item : *values)
    if (StringRef(item.first).empty() || !item.second.getAsInteger())
      return reject("runtime_scalar_value_out_of_domain");

  SmallVector<std::pair<unsigned, int64_t>> bindings;
  llvm::SmallVector<StringRef> matchedNames;
  for (BlockArgument argument : entry.getArguments()) {
    auto name = structuralArgumentName(argument);
    auto *value = name ? values->get(*name) : nullptr;
    if (!value) continue; // Specialized-away source arguments may be absent.
    if (llvm::is_contained(matchedNames, *name))
      return reject("runtime_scalar_name_ambiguous");
    matchedNames.push_back(*name);
    int64_t integer = *value->getAsInteger();
    auto type = dyn_cast<IntegerType>(argument.getType());
    unsigned width = isa<IndexType>(argument.getType()) ? 64 : type ? type.getWidth() : 0;
    if (!width || width > 64 ||
        (width == 1 ? (integer != 0 && integer != 1) : !llvm::isIntN(width, integer)))
      return reject("runtime_scalar_argument_type_or_range_mismatch");
    bindings.emplace_back(argument.getArgNumber(), integer);
  }
  clearBridgeRuntimeFacts(entry);
  auto integerType = IntegerType::get(module.getContext(), 64);
  for (auto [index, integer] : bindings)
    entry.setArgAttr(index, kBridgeBoundScalarAttr, IntegerAttr::get(integerType, integer));
  entry.walk([&](GetProgramIdOp pid) {
    pid->setAttr(kBridgeAxisExtentAttr,
                 IntegerAttr::get(integerType, extents[pid.getAxisAsInt()]));
  });
  entry->setAttr(lcore::kBridgeRuntimeBindingAppliedAttr,
                  UnitAttr::get(module.getContext()));
  return success();
}
