struct StateAxisNormalizationRegion {
  SmallVector<Value> laneValues;
  SmallVector<Value> outputs;
  SmallVector<Operation *> finalLaneOperations;
  Value columnDenominator;
  int64_t laneOperandIndex = -1;
  std::string graphSignature;
  std::set<std::string> operationCapabilities;
  int64_t packableNodeCount = 0;
  int64_t stateCardinality = 0;
  int64_t paddedStateCardinality = 0;
  int64_t stateWidth = 0;
  int64_t elementBytes = 0;
};
