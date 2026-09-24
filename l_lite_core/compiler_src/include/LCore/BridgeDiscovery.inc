// Bind compile-time runtime facts; discovery reports are queried out-of-band.
#include "LCore/RuntimeBinding.inc"

llvm::json::Object collectBridgeDiscovery(ModuleOp module) {
  llvm::json::Object facts;
  facts["schema"] = "triton.loop-bridge.discovery.v1";
  auto entry = findEntryPoint(module);
  if (failed(entry)) {
    facts["extractable"] = false;
    facts["reason"] = "entry_not_unique";
    return facts;
  }
  SmallVector<GetProgramIdOp> xPids;
  (*entry).walk([&](GetProgramIdOp pid) {
    if (pid.getAxisAsInt() == 0) xPids.push_back(pid);
  });
  LoopDependenceCertificate certificate =
      xPids.size() == 1
          ? certifyBridgeProgramIndependence(*entry, xPids.front())
          : LoopDependenceCertificate{false, "", "axis-zero program ID is not unique"};
  auto partition = xPids.size() == 1
      ? findDirectPidPartition(*entry, xPids.front()) : std::nullopt;
  facts["extractable"] = true;
  facts["construction_legal"] = certificate.safe;
  facts["dependence_certificate"] = certificate.kind;
  facts["rejection_reason"] = certificate.reason;
  facts["axis_zero_program_id_count"] = static_cast<int64_t>(xPids.size());
  facts["partition_recurrence_legal"] = partition.has_value();
  facts["partition_recurrence_certificate"] = partition
      ? "consecutive_pid_signed_quotient_remainder_recurrence_v1" : "";
  return facts;
}

class LoopBridgeDiscoverPass
    : public impl::TritonLoopBridgeDiscoverBase<LoopBridgeDiscoverPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Discard stale reports from saved IR; reports are never authority.
    module->removeAttr(kBridgeDiscoveryAttr);
    auto entry = findEntryPoint(module);
    if (failed(entry)) return;
    std::string reason;
    if (failed(bindBridgeRuntimeFacts(*entry, module, reason))) {
      reportFailure(module, reason);
      return signalPassFailure();
    }
  }
};
