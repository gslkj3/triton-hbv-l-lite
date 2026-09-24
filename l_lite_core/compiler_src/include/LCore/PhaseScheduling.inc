// Dependency-constrained source-group scheduling, independent of loop origin.
// Shared attribute declarations are supplied by the compiler integration unit.
LogicalResult materializeOperationNeutralPhase(
    FuncOp func, std::optional<StringRef> subject = std::nullopt,
    bool clearLineage = true,
    ArrayRef<std::pair<unsigned, unsigned>> guardedDisjointPairs = {}) {
  llvm::DenseMap<Block *, SmallVector<Operation *>> byBlock;
  SmallVector<Block *> blockOrderSeen;
  func.walk([&](Operation *operation) {
    if (!operation->hasAttr(kOperationGroupRefAttr))
      return;
    auto roleSubject = operation->getAttrOfType<StringAttr>(kRoleSubjectAttr);
    if (subject && (!roleSubject || roleSubject.getValue() != *subject))
      return;
    Block *block = operation->getBlock();
    if (!byBlock.count(block))
      blockOrderSeen.push_back(block);
    byBlock[block].push_back(operation);
  });
  if (byBlock.empty())
    return failure();

  int64_t totalGroups = 0;
  int64_t totalSourceGroupInstances = 0;
  int64_t totalReordered = 0;
  int64_t totalTaggedGroupsReordered = 0;
  bool scheduledAny = false;
  for (Block *block : blockOrderSeen) {
    SmallVector<Operation *> tagged = std::move(byBlock[block]);
    SmallVector<Operation *> tail;
    llvm::copy_if(tagged, std::back_inserter(tail), [](Operation *operation) {
      auto partition = operation->getAttrOfType<StringAttr>(
          kUnrollPartitionLineageAttr);
      return partition && partition.getValue() == "tail";
    });
    llvm::erase_if(tagged, [](Operation *operation) {
      auto partition = operation->getAttrOfType<StringAttr>(
          kUnrollPartitionLineageAttr);
      return partition && partition.getValue() == "tail";
    });
    for (Operation *operation : tail) {
      operation->removeAttr(kRoleAttr);
      operation->removeAttr(kRoleSubjectAttr);
      operation->removeAttr(kRoleIndexAttr);
      operation->removeAttr(kOperationGroupRefAttr);
      operation->removeAttr(kUnrollPartitionLineageAttr);
    }
    if (tagged.empty())
      continue;
    totalSourceGroupInstances += static_cast<int64_t>(tagged.size());

    llvm::SmallPtrSet<Operation *, 32> taggedSet(tagged.begin(), tagged.end());
    Operation *first = nullptr;
    Operation *last = nullptr;
    for (Operation &operation : *block)
      if (taggedSet.contains(&operation)) {
        if (!first)
          first = &operation;
        last = &operation;
      }
    if (!first || !last)
      return failure();

    SmallVector<Operation *> nodes;
    for (Operation *operation = first;; operation = operation->getNextNode()) {
      nodes.push_back(operation);
      if (operation == last)
        break;
      if (!operation->getNextNode())
        return failure();
    }
    llvm::DenseMap<Operation *, unsigned> nodeIndex;
    for (auto [index, operation] : llvm::enumerate(nodes))
      nodeIndex[operation] = index;
    SmallVector<SmallVector<unsigned>> successors(nodes.size());
    SmallVector<unsigned> indegree(nodes.size(), 0);
    std::set<std::pair<unsigned, unsigned>> edges;
    auto addEdge = [&](Operation *from, Operation *to) {
      auto fromPosition = nodeIndex.find(from);
      auto toPosition = nodeIndex.find(to);
      if (fromPosition == nodeIndex.end() || toPosition == nodeIndex.end() ||
          fromPosition->second == toPosition->second)
        return;
      unsigned source = fromPosition->second;
      unsigned target = toPosition->second;
      if (!edges.insert({source, target}).second)
        return;
      successors[source].push_back(target);
      ++indegree[target];
    };
    for (Operation *root : nodes)
      root->walk([&](Operation *nested) {
        for (Value operand : nested->getOperands()) {
          Operation *definition = operand.getDefiningOp();
          if (definition)
            addEdge(definition, root);
        }
      });
    Operation *previousBarrier = nullptr;
    SmallVector<Operation *> effectPrefix;
    for (Operation *operation : nodes)
      if (!isMemoryEffectFree(operation) || !isSpeculatable(operation)) {
        // Without a bound guard, retain the original exact effect chain.
        // With a guard, compare every earlier effect; skipping only the last
        // predecessor could lose a dependency across an independent access.
        if (!guardedDisjointPairs.empty()) {
          for (Operation *earlier : effectPrefix)
            if (!guardedMemoryPairCommutes(earlier, operation, func,
                                          guardedDisjointPairs))
              addEdge(earlier, operation);
          effectPrefix.push_back(operation);
        } else if (previousBarrier)
          addEdge(previousBarrier, operation);
        previousBarrier = operation;
      }

    std::map<std::string, unsigned> groupRanks;
    unsigned nextGroupRank = 0;
    for (Operation *operation : nodes) {
      auto group = operation->getAttrOfType<StringAttr>(
          kOperationGroupRefAttr);
      if (group && !groupRanks.count(group.getValue().str()))
        groupRanks[group.getValue().str()] = nextGroupRank++;
    }
    totalGroups += nextGroupRank;
    const unsigned noPriority = nodes.size() + nextGroupRank + 1;
    SmallVector<unsigned> priority(nodes.size(), noPriority);
    for (auto [index, operation] : llvm::enumerate(nodes))
      if (auto group = operation->getAttrOfType<StringAttr>(
              kOperationGroupRefAttr))
        priority[index] = groupRanks[group.getValue().str()];
    // A pure source-tagged recurrence update can also unlock an earlier-ranked
    // group in the next iteration. Treat it like untagged scaffolding for
    // scheduling priority; a lineage tag must not prevent legal anticipation.
    // SSA and observable-effect edges above remain mandatory. This changes
    // ready-node preference only, never legality or the original effect chain.
    // Repeated relaxation is deliberate:
    // the graph is small and this remains correct even if the original block
    // order is not itself a topological numbering for nested captures.
    for (size_t iteration = 0; iteration < nodes.size(); ++iteration) {
      bool changed = false;
      for (size_t source = 0; source < nodes.size(); ++source)
        for (unsigned target : successors[source])
          if ((!nodes[source]->hasAttr(kOperationGroupRefAttr) ||
               (isMemoryEffectFree(nodes[source]) && isSpeculatable(nodes[source]))) &&
              priority[target] < priority[source]) {
            priority[source] = priority[target];
            changed = true;
          }
      if (!changed)
        break;
    }

    SmallVector<unsigned> ready;
    for (size_t index = 0; index < nodes.size(); ++index)
      if (indegree[index] == 0)
        ready.push_back(index);
    SmallVector<Operation *> schedule;
    while (!ready.empty()) {
      auto selected = llvm::min_element(
          ready, [&](unsigned left, unsigned right) {
            return std::tie(priority[left], left) <
                   std::tie(priority[right], right);
          });
      unsigned index = *selected;
      ready.erase(selected);
      schedule.push_back(nodes[index]);
      for (unsigned successor : successors[index])
        if (--indegree[successor] == 0)
          ready.push_back(successor);
    }
    if (schedule.size() != nodes.size())
      return failure();

    Operation *anchor = first->getPrevNode();
    for (Operation *operation : schedule) {
      bool alreadyAdjacent = anchor ? anchor->getNextNode() == operation
                                    : operation == &block->front();
      if (!alreadyAdjacent) {
        if (anchor)
          operation->moveAfter(anchor);
        else
          operation->moveBefore(&block->front());
        ++totalReordered;
        // Count the rule-owned source-group instance separately from
        // induction/address/predicate scaffolding in the same dependency
        // closure.  Strong phase capacity is expressed in group instances;
        // the historical totalReordered counter is expressed in arbitrary
        // operations and therefore cannot certify that causal edge.
        totalTaggedGroupsReordered += taggedSet.contains(operation);
      }
      anchor = operation;
    }
    if (clearLineage)
      for (Operation *operation : tagged) {
        operation->removeAttr(kRoleAttr);
        operation->removeAttr(kRoleSubjectAttr);
        operation->removeAttr(kRoleIndexAttr);
        operation->removeAttr(kOperationGroupRefAttr);
        operation->removeAttr(kUnrollPartitionLineageAttr);
      }
    scheduledAny = true;
  }
  if (!scheduledAny || totalGroups == 0)
    return failure();
  auto previousGroups =
      func->getAttrOfType<IntegerAttr>(kPhaseOperationGroupCountAttr);
  auto previousSourceGroups =
      func->getAttrOfType<IntegerAttr>(kPhaseSourceOperationGroupCountAttr);
  auto previousReordered =
      func->getAttrOfType<IntegerAttr>(kPhaseOperationReorderedCountAttr);
  auto previousGroupReordered = func->getAttrOfType<IntegerAttr>(
      kPhaseOperationGroupReorderedCountAttr);
  func->setAttr(
      kPhaseOperationGroupCountAttr,
      IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                       totalGroups +
                           (previousGroups ? previousGroups.getInt() : 0)));
  // Count replicated, non-tail tagged group instances at the planning cut.
  // totalGroups above is the distinct source-identity count and therefore is
  // not in the same unit as the moved-instance counter.
  func->setAttr(
      kPhaseSourceOperationGroupCountAttr,
      IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                       totalSourceGroupInstances +
                           (previousSourceGroups
                                ? previousSourceGroups.getInt()
                                : 0)));
  func->setAttr(
      kPhaseOperationReorderedCountAttr,
      IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                       totalReordered +
                           (previousReordered
                                ? previousReordered.getInt()
                                : 0)));
  func->setAttr(
      kPhaseOperationGroupReorderedCountAttr,
      IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                       totalTaggedGroupsReordered +
                           (previousGroupReordered
                                ? previousGroupReordered.getInt()
                                : 0)));
  return success();
}
