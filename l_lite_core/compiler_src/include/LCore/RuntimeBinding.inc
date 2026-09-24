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
  auto object = dyn_cast<DictionaryAttr>(attribute);
  auto values = object ? object.getAs<DictionaryAttr>("values_by_name") : DictionaryAttr();
  auto grid = object ? object.getAs<DenseIntElementsAttr>("grid") : DenseIntElementsAttr();
  auto schema = object ? object.getAs<StringAttr>("schema") : StringAttr();
  if (!object || object.size() != 3 || !schema ||
      schema.getValue() != "triton.loop-bridge.runtime-scalars.v2" ||
      !values || !grid || grid.getType().getRank() != 1 ||
      grid.getType().getElementTypeBitWidth() != 32 ||
      grid.getNumElements() < 1 || grid.getNumElements() > 3)
    return reject("runtime_scalar_binding_malformed");
  std::array<int64_t, 3> extents{1, 1, 1};
  for (auto [axis, value] : llvm::enumerate(grid.getValues<APInt>())) {
    int64_t extent = value.getSExtValue();
    if (extent < 1 || extent > INT32_MAX)
      return reject("runtime_scalar_grid_out_of_domain");
    extents[axis] = extent;
  }
  for (const auto &item : values) {
    auto integer = dyn_cast<IntegerAttr>(item.getValue());
    if (item.getName().strref().empty() || !integer ||
        integer.getType().getIntOrFloatBitWidth() != 64)
      return reject("runtime_scalar_value_out_of_domain");
  }

  SmallVector<std::pair<unsigned, int64_t>> bindings;
  llvm::SmallVector<StringRef> matchedNames;
  for (BlockArgument argument : entry.getArguments()) {
    auto name = structuralArgumentName(argument);
    auto value = name ? values.getAs<IntegerAttr>(*name) : IntegerAttr();
    if (!value) continue; // Specialized-away source arguments may be absent.
    if (llvm::is_contained(matchedNames, *name))
      return reject("runtime_scalar_name_ambiguous");
    matchedNames.push_back(*name);
    int64_t integer = value.getInt();
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
