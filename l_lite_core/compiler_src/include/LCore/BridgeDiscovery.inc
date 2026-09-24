// Bridge fact binding and discovery; no profitability or downstream route choice.
#include "LCore/RuntimeBinding.inc"
class LoopBridgeDiscoverPass
    : public impl::TritonLoopBridgeDiscoverBase<LoopBridgeDiscoverPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto entry = findEntryPoint(module);
    if (failed(entry)) {
      module->setAttr(
          kBridgeDiscoveryAttr,
          StringAttr::get(module.getContext(),
                          R"({"schema":"triton.loop-bridge.discovery.v1","extractable":false,"reason":"entry_not_unique"})"));
      return;
    }

    std::string bindingFailure;
    if (failed(bindBridgeRuntimeFacts(*entry, module, bindingFailure))) {
      llvm::json::Object facts;
      facts["schema"] = "triton.loop-bridge.discovery.v1";
      facts["extractable"] = false;
      facts["reason"] = bindingFailure;
      std::string serialized;
      llvm::raw_string_ostream stream(serialized);
      stream << llvm::json::Value(std::move(facts));
      stream.flush();
      module->setAttr(kBridgeDiscoveryAttr,
                      StringAttr::get(module.getContext(), serialized));
      reportFailure(module, bindingFailure);
      return signalPassFailure();
    }

    SmallVector<GetProgramIdOp> xPids;
    (*entry).walk([&](GetProgramIdOp pid) {
      if (pid.getAxisAsInt() == 0)
        xPids.push_back(pid);
    });
    LoopDependenceCertificate certificate;
    if (xPids.size() == 1)
      certificate = certifyBridgeProgramIndependence(*entry, xPids.front());
    else
      certificate = {false, "", "axis-zero program ID is not unique"};
    auto directPartition = xPids.size() == 1
                               ? findDirectPidPartition(*entry, xPids.front())
                               : std::nullopt;
    llvm::json::Object facts;
    facts["schema"] = "triton.loop-bridge.discovery.v1";
    facts["extractable"] = true;
    facts["construction_legal"] = certificate.safe;
    facts["dependence_certificate"] = certificate.kind;
    facts["rejection_reason"] = certificate.reason;
    facts["axis_zero_program_id_count"] =
        static_cast<int64_t>(xPids.size());
    facts["partition_recurrence_legal"] = directPartition.has_value();
    facts["partition_recurrence_certificate"] =
        directPartition
            ? "consecutive_pid_signed_quotient_remainder_recurrence_v1"
            : "";
    std::string serialized;
    llvm::raw_string_ostream stream(serialized);
    stream << llvm::json::Value(std::move(facts));
    stream.flush();
    module->setAttr(kBridgeDiscoveryAttr,
                    StringAttr::get(module.getContext(), serialized));
  }
};
