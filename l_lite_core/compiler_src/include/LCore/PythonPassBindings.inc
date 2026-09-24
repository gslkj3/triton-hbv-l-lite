// Include inside the native TTIR pybind module initializer, with
// LCore/Passes.h and ADD_PASS_WRAPPER_0 in scope.
// These are compiler entry points, not six optimization mechanisms.
// Prediction/ranking stays in Python; decision closes an already selected plan.
auto lFactsToPython = [](llvm::json::Object input) -> py::object {
  // Convert values directly; no JSON text enters the module or pass pipeline.
  auto convert = [](const auto &self, const llvm::json::Value &value) -> py::object {
    if (auto object = value.getAsObject()) {
      py::dict result;
      for (const auto &item : *object)
        result[py::str(item.first.str())] = self(self, item.second);
      return result;
    }
    if (auto array = value.getAsArray()) {
      py::list result;
      for (const auto &item : *array) result.append(self(self, item));
      return result;
    }
    if (auto string = value.getAsString()) return py::str(string->str());
    if (auto boolean = value.getAsBoolean()) return py::bool_(*boolean);
    if (auto integer = value.getAsInteger()) return py::int_(*integer);
    if (auto number = value.getAsNumber()) return py::float_(*number);
    return py::none();
  };
  llvm::json::Value facts(std::move(input));
  return convert(convert, facts);
};
m.def("query_l_planning_facts", [lFactsToPython](mlir::ModuleOp module) {
  return lFactsToPython(mlir::triton::lcore::queryPlanningFacts(module));
});
m.def("query_l_bridge_discovery", [lFactsToPython](mlir::ModuleOp module) {
  return lFactsToPython(mlir::triton::lcore::queryBridgeDiscovery(module));
});
m.def("has_l_decision", [](mlir::ModuleOp module) {
  return module->hasAttr("tt.hbv.plan_bundle");
});
m.def("run_with_l_backend_observations",
      [lFactsToPython](py::object manager, mlir::ModuleOp module,
                       const std::string &stage) {
  llvm::json::Array records;
  bool wrongModule = false;
  mlir::triton::gpu::BackendObservationSink sink =
      [&](mlir::ModuleOp observed, llvm::json::Object record) {
        if (observed != module) { wrongModule = true; return; }
        records.push_back(std::move(record));
      };
  {
    // The normal pass manager owns execution, verification and diagnostics.
    // Stack unwinding restores the sink even when its Python run() throws.
    mlir::triton::gpu::ScopedBackendObservations capture(sink);
    manager.attr("run")(module, stage);
  }
  if (wrongModule) throw py::value_error("backend observation belongs to a different module");
  llvm::json::Object result;
  result["schema"] = "l.core.backend-observations.v1";
  result["stage"] = stage;
  result["records"] = std::move(records);
  return lFactsToPython(std::move(result));
});
ADD_PASS_WRAPPER_0("add_hbv_loop_decision", mlir::triton::lcore::createDecisionPass);
m.def("make_l_decision_attribute", [](mlir::ModuleOp module, py::dict bundle) -> mlir::Attribute {
  // Validate the complete external audit contract before projecting only the
  // materialization fields into IR. This is a value conversion, not JSON I/O.
  auto convert = [](const auto &self, py::handle value) -> llvm::json::Value {
    if (value.is_none()) return nullptr;
    if (py::isinstance<py::bool_>(value)) return py::cast<bool>(value);
    if (py::isinstance<py::int_>(value)) return py::cast<int64_t>(value);
    if (py::isinstance<py::str>(value)) return py::cast<std::string>(value);
    if (py::isinstance<py::dict>(value)) {
      llvm::json::Object result;
      for (auto item : py::reinterpret_borrow<py::dict>(value)) {
        if (!py::isinstance<py::str>(item.first))
          throw py::value_error("decision mapping keys must be strings");
        result[py::cast<std::string>(item.first)] = self(self, item.second);
      }
      return result;
    }
    if (py::isinstance<py::list>(value)) {
      llvm::json::Array result;
      for (auto item : py::reinterpret_borrow<py::list>(value))
        result.push_back(self(self, item));
      return result;
    }
    throw py::value_error("decision contract contains an unsupported value type");
  };
  auto input = convert(convert, bundle);
  std::string reason;
  auto attribute = mlir::triton::lcore::makeDecisionAttribute(module, *input.getAsObject(), reason);
  if (!attribute) throw py::value_error(reason);
  return attribute;
});
ADD_PASS_WRAPPER_0("add_hbv_loop_facts", mlir::triton::lcore::createFactsPass);
ADD_PASS_WRAPPER_0("add_loop_bridge_discover", mlir::triton::lcore::createBridgeDiscoverPass);
ADD_PASS_WRAPPER_0("add_loop_bridge_program_coarsening",
                   mlir::triton::lcore::createBridgeConstructionPass);
ADD_PASS_WRAPPER_0("add_hbv_loop_materialize", mlir::triton::lcore::createMaterializePass);
ADD_PASS_WRAPPER_0("add_hbv_validate_loop_plan", mlir::triton::lcore::createValidatePass);
