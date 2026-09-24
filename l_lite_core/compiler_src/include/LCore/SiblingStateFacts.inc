// Read-only sibling graph facts. No plan parser or profitability policy.
void appendSiblingStateFacts(FuncOp entry, llvm::json::Object &facts) {
  int64_t stateAxisNormalizationCount = 0;
  int64_t stateAxisMinimumCardinality = 0;
  int64_t stateAxisMaximumCardinality = 0;
  int64_t stateAxisMaximumPaddedCardinality = 0;
  int64_t stateAxisMinimumWidth = 0;
  int64_t stateAxisMaximumWidth = 0;
  int64_t stateAxisMinimumElementBytes = 0;
  int64_t stateAxisMaximumElementBytes = 0;
  SmallVector<StateAxisNormalizationRegion, 2> stateAxisSubjects;
  stateAxisSubjects = collectStateAxisNormalizations(entry);
  for (const StateAxisNormalizationRegion &subject : stateAxisSubjects) {
    ++stateAxisNormalizationCount;
    if (stateAxisNormalizationCount == 1) {
      stateAxisMinimumCardinality = subject.stateCardinality;
      stateAxisMinimumWidth = subject.stateWidth;
      stateAxisMinimumElementBytes = subject.elementBytes;
    } else {
      stateAxisMinimumCardinality = std::min(
          stateAxisMinimumCardinality, subject.stateCardinality);
      stateAxisMinimumWidth = std::min(
          stateAxisMinimumWidth, subject.stateWidth);
      stateAxisMinimumElementBytes = std::min(
          stateAxisMinimumElementBytes, subject.elementBytes);
    }
    stateAxisMaximumCardinality = std::max(
        stateAxisMaximumCardinality, subject.stateCardinality);
    stateAxisMaximumPaddedCardinality = std::max(
        stateAxisMaximumPaddedCardinality,
        subject.paddedStateCardinality);
    stateAxisMaximumWidth = std::max(
        stateAxisMaximumWidth, subject.stateWidth);
    stateAxisMaximumElementBytes = std::max(
        stateAxisMaximumElementBytes, subject.elementBytes);
  }
  facts["state_axis_sibling_group_count"] =
      stateAxisNormalizationCount;
  facts["state_axis_sibling_group_minimum_cardinality"] =
      stateAxisMinimumCardinality;
  facts["state_axis_sibling_group_maximum_cardinality"] =
      stateAxisMaximumCardinality;
  facts["state_axis_sibling_group_maximum_padded_cardinality"] =
      stateAxisMaximumPaddedCardinality;
  facts["state_axis_sibling_group_minimum_width"] =
      stateAxisMinimumWidth;
  facts["state_axis_sibling_group_maximum_width"] =
      stateAxisMaximumWidth;
  facts["state_axis_sibling_group_minimum_element_bytes"] =
      stateAxisMinimumElementBytes;
  facts["state_axis_sibling_group_maximum_element_bytes"] =
      stateAxisMaximumElementBytes;
  facts["state_axis_sibling_group_uniform_shape"] =
      stateAxisNormalizationCount > 0 &&
      stateAxisMinimumCardinality == stateAxisMaximumCardinality &&
      stateAxisMinimumWidth == stateAxisMaximumWidth &&
      stateAxisMinimumElementBytes == stateAxisMaximumElementBytes;
  facts["state_axis_graph_schema"] =
      "hbv.loop.state-axis-sibling-graph.v1";
  facts["state_axis_backend_adapter_ref"] =
      stateAxisNormalizationCount > 0
          ? "hbv.loop.state-axis.backend.ttir.v1"
          : "";
  llvm::json::Array stateAxisGroups;
  for (auto [ordinal, subject] : llvm::enumerate(stateAxisSubjects)) {
    llvm::json::Object group;
    group["ordinal"] = static_cast<int64_t>(ordinal);
    group["state_cardinality"] = subject.stateCardinality;
    group["padded_state_cardinality"] =
        subject.paddedStateCardinality;
    group["state_width"] = subject.stateWidth;
    group["element_bytes"] = subject.elementBytes;
    group["lane_operand_index"] = subject.laneOperandIndex;
    group["packable_node_count"] = subject.packableNodeCount;
    group["graph_signature"] = subject.graphSignature;
    llvm::json::Array capabilities;
    for (const std::string &capability : subject.operationCapabilities)
      capabilities.push_back(capability);
    group["operation_capabilities"] = std::move(capabilities);
    stateAxisGroups.push_back(std::move(group));
  }
  facts["state_axis_sibling_groups"] = std::move(stateAxisGroups);
  facts["state_axis_sibling_group_certificate"] =
      stateAxisNormalizationCount > 0
          ? "provider_closed_sibling_state_operation_graph_v1"
          : "";
}
