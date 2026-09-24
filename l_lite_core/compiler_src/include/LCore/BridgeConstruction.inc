// Independent Bridge construction implementation.
// Shared IR analysis helpers are being migrated separately; this fragment is
// currently included inside the L compiler implementation namespace.
#include "LCore/BridgeJointIndependence.inc"
class LoopBridgeProgramCoarseningPass
    : public impl::TritonLoopBridgeProgramCoarseningBase<
          LoopBridgeProgramCoarseningPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto factorAttr = module->getAttrOfType<IntegerAttr>(kBridgeFactorAttr);
    if (!factorAttr || factorAttr.getInt() == 1)
      return;
    int64_t factor = factorAttr.getInt();
    if (factor < 2 || factor > std::numeric_limits<int32_t>::max() ||
        !llvm::isPowerOf2_64(factor)) {
      reportFailure(module, "Bridge factor must be a positive power of two representable by the i32 loop bound");
      return signalPassFailure();
    }
    SmallVector<int64_t, 3> requestedDivisors{factor, 1, 1};
    if (auto attribute = module->getAttr(kBridgeRequestedDivisorsAttr)) {
      auto array = dyn_cast<DenseIntElementsAttr>(attribute);
      if (!array || array.getType().getRank() != 1 ||
          array.getNumElements() != 3 ||
          array.getType().getElementTypeBitWidth() != 32) {
        reportFailure(module, "Bridge divisor vector is malformed");
        return signalPassFailure();
      }
      requestedDivisors.clear();
      for (const APInt &value : array.getValues<APInt>()) {
        int64_t divisor = value.getSExtValue();
        if (divisor < 1 || !llvm::isPowerOf2_64(divisor)) {
          reportFailure(module, "Bridge divisor vector is not positive power-of-two");
          return signalPassFailure();
        }
        requestedDivisors.push_back(divisor);
      }
    }
    // Bridge is configured only by its typed factor/divisor inputs. A route
    // decision belongs to the later analysis cut, never to construction.
    // Reject the old JSON envelope (including bridge-only envelopes), rather
    // than keeping a hidden alternate construction protocol alive.
    if (module->hasAttr(kBundleAttr)) {
      reportFailure(module, "Bridge requires typed construction inputs, not a route plan bundle");
      return signalPassFailure();
    }
    auto entry = findEntryPoint(module);
    if (failed(entry)) {
      reportFailure(module, "Bridge requires one public kernel");
      return signalPassFailure();
    }
    // Capture identity before Bridge inserts loops or normalizes the CFG.
    // Native cloning copies operation attributes, and moving a region retains
    // them. This is provenance only: it does not grant a route capability or
    // imply that a helper-contained source loop is visible to the pipeliner.
    SmallVector<scf::ForOp> sourceLoops;
    (*entry).walk([&](scf::ForOp loop) { sourceLoops.push_back(loop); });
    if (llvm::any_of(sourceLoops, [](scf::ForOp loop) {
          return loop->hasAttr(kBridgeSourceLoopAttr);
        })) {
      reportFailure(module, "Bridge source loop lineage is already present");
      return signalPassFailure();
    }
    Builder lineageBuilder(module.getContext());
    for (auto [ordinal, loop] : llvm::enumerate(sourceLoops)) {
      int64_t depth = 0;
      for (Operation *parent = loop->getParentOp(); parent;
           parent = parent->getParentOp())
        depth += isa<scf::ForOp>(parent);
      loop->setAttr(kBridgeSourceLoopAttr, lineageBuilder.getDictionaryAttr({
          lineageBuilder.getNamedAttr(
              "locator", lineageBuilder.getStringAttr(
                             "planning-cut.loop." + std::to_string(ordinal))),
          lineageBuilder.getNamedAttr(
              "nesting_depth", lineageBuilder.getI64IntegerAttr(depth)),
      }));
    }
    // Bridge owns only program-to-loop construction.  Route-local
    // transformations (including exact-prefix vectorization) must be selected
    // and materialized by their own PlanContract.  Performing one here would
    // make Bridge admission depend on an unrelated loop-body fingerprint and
    // would destroy the causal boundary between the two interventions.
    auto setGridDivisor = [&](unsigned axis, int64_t divisor,
                              OpBuilder &builder) {
      StringRef attribute = axis == 0 ? StringRef(kBridgeGridDivisorAttr)
                            : axis == 1 ? StringRef(kBridgeGridDivisorYAttr)
                                        : StringRef(kBridgeGridDivisorZAttr);
      module->setAttr(attribute, builder.getI32IntegerAttr(divisor));
    };
    SmallVector<GetProgramIdOp> sourceProgramIds;
    (*entry).walk([&](GetProgramIdOp pid) {
      sourceProgramIds.push_back(pid);
    });
    std::array<std::optional<int64_t>, 3> sourceExtents;
    for (GetProgramIdOp pid : sourceProgramIds)
      if (auto extent = pid->getAttrOfType<IntegerAttr>(kBridgeAxisExtentAttr))
        sourceExtents[pid.getAxisAsInt()] = extent.getInt();
    auto propagatePhysicalExtent = [&](GetProgramIdOp pid) {
      unsigned axis = pid.getAxisAsInt();
      auto logical = sourceExtents[axis];
      int64_t divisor = requestedDivisors[axis];
      pid->removeAttr(kBridgeAxisExtentAttr);
      // Only the exact existing grid-division contract is projected here.
      // Nondivisible launches require separate tail/launch evidence.
      if (logical && *logical > 0 && *logical % divisor == 0)
        pid->setAttr(kBridgeAxisExtentAttr,
                     IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                      *logical / divisor));
    };
    SmallVector<GetProgramIdOp> allProgramIds;
    for (GetProgramIdOp pid : sourceProgramIds)
      if (requestedDivisors[pid.getAxisAsInt()] > 1)
        allProgramIds.push_back(pid);
    for (unsigned axis = 0; axis < requestedDivisors.size(); ++axis)
      if (requestedDivisors[axis] > 1 &&
          llvm::none_of(sourceProgramIds, [&](GetProgramIdOp pid) {
            return pid.getAxisAsInt() == axis;
          })) {
        reportFailure(module,
                      "Bridge divisor names an unused program axis");
        return signalPassFailure();
      }
    llvm::sort(allProgramIds, [](GetProgramIdOp lhs, GetProgramIdOp rhs) {
      return lhs.getAxisAsInt() < rhs.getAxisAsInt();
    });
    bool uniqueProgramAxes = !allProgramIds.empty();
    for (auto pair : llvm::zip(allProgramIds, llvm::drop_begin(allProgramIds)))
      uniqueProgramAxes &=
          std::get<0>(pair).getAxisAsInt() !=
          std::get<1>(pair).getAxisAsInt();
    bool genericAxisGrouping =
        uniqueProgramAxes &&
        (allProgramIds.size() > 1 ||
         allProgramIds.front().getAxisAsInt() != 0);
    int64_t groupedProgramCount = 1;
    for (GetProgramIdOp pid : allProgramIds) {
      int64_t divisor = requestedDivisors[pid.getAxisAsInt()];
      if (groupedProgramCount > std::numeric_limits<int64_t>::max() /
                                    divisor) {
        reportFailure(module, "multi-axis Bridge factor product overflows i64");
        return signalPassFailure();
      }
      groupedProgramCount *= divisor;
    }
    if (groupedProgramCount != factor) {
      reportFailure(module,
                    "Bridge divisor product disagrees with grouped-program factor");
      return signalPassFailure();
    }

    // Use one helper-call materialization for every generic axis vector.  The
    // former flat-body clone path inserted rewritten PID expressions beside
    // still-live ungrouped PID expressions and could violate SSA dominance.
    // A private helper makes the complete program body the unit of cloning:
    // grouped PIDs are explicit arguments, while ungrouped PIDs and all body
    // dependencies retain their original dominance inside the helper.  This
    // is a structural rule for every generic axis vector, not a case branch.
    if (genericAxisGrouping) {
      if (allProgramIds.size() > 1 &&
          !provesBridgeJointIndependence(*entry, sourceProgramIds)) {
        reportFailure(module, "joint multi-axis store independence is not proven");
        return signalPassFailure();
      }
      SmallVector<unsigned> groupedAxes;
      for (GetProgramIdOp pid : allProgramIds)
        groupedAxes.push_back(pid.getAxisAsInt());
      for (GetProgramIdOp pid : allProgramIds) {
        LoopDependenceCertificate axisCertificate =
            certifyBridgeProgramIndependence(*entry, pid);
        if (!axisCertificate.safe) {
          reportFailure(
              module,
              Twine("multi-axis CFG independence proof failed: ") +
                  axisCertificate.reason);
          return signalPassFailure();
        }
      }
      OpBuilder moduleBuilder(module.getContext());
      FuncOp helper = *entry;
      std::string publicName = helper.getName().str();
      FunctionType publicType = helper.getFunctionType();
      NamedAttrList publicAttrs(helper->getAttrs());
      helper.setName(publicName + "__loop_bridge_multiaxis_body");
      helper.setVisibility(SymbolTable::Visibility::Private);
      SmallVector<Type> helperInputs(helper.getArgumentTypes());
      SmallVector<DictionaryAttr> helperArgAttrs;
      for (unsigned index = 0; index < helper.getNumArguments(); ++index) {
        auto attrs = helper.getArgAttrDict(index);
        helperArgAttrs.push_back(
            attrs ? attrs : DictionaryAttr::get(module.getContext()));
      }
      for (size_t index = 0; index < allProgramIds.size(); ++index) {
        helperInputs.push_back(moduleBuilder.getI32Type());
        helperArgAttrs.push_back(DictionaryAttr::get(module.getContext()));
      }
      helper.setFunctionType(FunctionType::get(
          module.getContext(), helperInputs, helper.getResultTypes()));
      SmallVector<BlockArgument> virtualPidArguments;
      for (size_t index = 0; index < allProgramIds.size(); ++index)
        virtualPidArguments.push_back(
            helper.getBody().front().addArgument(
                moduleBuilder.getI32Type(), helper.getLoc()));
      helper.setAllArgAttrs(helperArgAttrs);
      SmallVector<GetProgramIdOp> helperPids;
      helper.walk([&](GetProgramIdOp pid) {
        if (requestedDivisors[pid.getAxisAsInt()] > 1)
          helperPids.push_back(pid);
      });
      llvm::sort(helperPids, [](GetProgramIdOp lhs, GetProgramIdOp rhs) {
        return lhs.getAxisAsInt() < rhs.getAxisAsInt();
      });
      if (helperPids.size() != virtualPidArguments.size()) {
        reportFailure(module, "multi-axis CFG helper lost a program ID");
        return signalPassFailure();
      }
      for (auto [pid, argument] : llvm::zip(helperPids, virtualPidArguments)) {
        pid.getResult().replaceAllUsesWith(argument);
        pid.erase();
      }
      moduleBuilder.setInsertionPoint(helper);
      FuncOp publicEntry = FuncOp::create(
          moduleBuilder, helper.getLoc(), publicName, publicType);
      publicEntry->setAttrs(publicAttrs);
      Block *publicBlock = publicEntry.addEntryBlock();
      OpBuilder builder = OpBuilder::atBlockBegin(publicBlock);
      Location loc = publicEntry.getLoc();
      SmallVector<Value> basePids;
      for (unsigned axis : groupedAxes) {
        int64_t divisor = requestedDivisors[axis];
        auto physical = GetProgramIdOp::create(
            builder, loc, axis);
        propagatePhysicalExtent(physical);
        Value factorValue = arith::ConstantIntOp::create(
            builder, loc, divisor, 32);
        basePids.push_back(arith::MulIOp::create(
            builder, loc, physical, factorValue));
      }
      // Construct one ordinary loop, rather than expanding virtual programs
      // here. Its ordinal is decoded into the requested mixed-radix axes.
      Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
      Value upper = arith::ConstantIntOp::create(builder, loc, groupedProgramCount, 32);
      Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
      auto loop = scf::ForOp::create(builder, loc, zero, upper, one);
      loop->setAttr(kBridgeSubjectAttr, builder.getUnitAttr());
      loop->setAttr(kBridgeFactorAttr, builder.getI32IntegerAttr(factor));
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(loop.getBody());
      Value remainder = loop.getInductionVar();
      SmallVector<Value> callArguments(publicBlock->getArguments());
      for (auto [axis, base] : llvm::zip(groupedAxes, basePids)) {
        Value divisor = arith::ConstantIntOp::create(bodyBuilder, loc, requestedDivisors[axis], 32);
        Value digit = arith::RemUIOp::create(bodyBuilder, loc, remainder, divisor);
        remainder = arith::DivUIOp::create(bodyBuilder, loc, remainder, divisor);
        Value virtualPid = arith::AddIOp::create(bodyBuilder, loc, base, digit);
        callArguments.push_back(virtualPid);
      }
      auto call = CallOp::create(bodyBuilder, loc, helper.getName(), TypeRange{}, callArguments);
      call->setAttr(kBridgeRoleAttr, bodyBuilder.getStringAttr("multi_axis_helper_call"));
      ReturnOp::create(builder, loc);
      publicEntry->setAttr(
          kBridgeOriginAttr,
          builder.getStringAttr("bridge_multiaxis_constructed"));
      publicEntry->setAttr(kBridgeFactorAttr,
                           builder.getI32IntegerAttr(factor));
      publicEntry->setAttr(kBridgeCardinalityAttr,
                           builder.getI32IntegerAttr(groupedProgramCount));
      publicEntry->setAttr(
          kDependenceAttr,
          builder.getStringAttr("multi_axis_program_disjoint_v1"));
      for (unsigned axis : groupedAxes)
        setGridDivisor(axis, requestedDivisors[axis], builder);
      return;
    }
    // A helper call is a faithful fallback for arbitrary CFG, but it hides
    // the program body's memory stages from every route materializer.  First
    // normalize the one CFG family for which moving the continuation is
    // independently certifiable.  Nonmatching CFG remains intact and is
    // reported below as a hidden-body route capability, not guessed through.
    if ((*entry).getBody().getBlocks().size() != 1) {
      BridgeCFGNormalizationResult normalization =
          normalizeSingleEarlyVoidReturnCFG(*entry);
      if (!normalization.normalized)
        (*entry)->setAttr(
            kBridgeCFGPredicationRejectionAttr,
            StringAttr::get(module.getContext(), normalization.reason));
    }
    if ((*entry).getBody().getBlocks().size() != 1) {
      SmallVector<GetProgramIdOp> sourceXPids;
      (*entry).walk([&](GetProgramIdOp pid) {
        if (pid.getAxisAsInt() == 0)
          sourceXPids.push_back(pid);
      });
      if (sourceXPids.size() != 1) {
        reportFailure(module,
                      "CFG Bridge requires one axis-zero program ID");
        return signalPassFailure();
      }
      LoopDependenceCertificate certificate =
          certifyBridgeProgramIndependence(*entry, sourceXPids.front());
      if (!certificate.safe) {
        reportFailure(module,
                      Twine("CFG Bridge independence proof failed: ") +
                          certificate.reason);
        return signalPassFailure();
      }

      OpBuilder moduleBuilder(module.getContext());
      FuncOp helper = *entry;
      std::string publicName = helper.getName().str();
      FunctionType publicType = helper.getFunctionType();
      NamedAttrList publicAttrs(helper->getAttrs());
      helper.setName(publicName + "__loop_bridge_body");
      helper.setVisibility(SymbolTable::Visibility::Private);
      SmallVector<Type> helperInputs(helper.getArgumentTypes());
      SmallVector<DictionaryAttr> helperArgAttrs;
      helperArgAttrs.reserve(helper.getNumArguments() + 1);
      for (unsigned i = 0; i < helper.getNumArguments(); ++i) {
        auto attrs = helper.getArgAttrDict(i);
        helperArgAttrs.push_back(
            attrs ? attrs : DictionaryAttr::get(module.getContext()));
      }
      helperInputs.push_back(moduleBuilder.getI32Type());
      helperArgAttrs.push_back(DictionaryAttr::get(module.getContext()));

      helper.setFunctionType(FunctionType::get(
          module.getContext(), helperInputs, helper.getResultTypes()));
      BlockArgument virtualPidArgument = helper.getBody().front().addArgument(
          moduleBuilder.getI32Type(), (*entry).getLoc());

      helper.setAllArgAttrs(helperArgAttrs);
      SmallVector<GetProgramIdOp> helperXPids;
      helper.walk([&](GetProgramIdOp pid) {
        if (pid.getAxisAsInt() == 0)
          helperXPids.push_back(pid);
      });
      if (helperXPids.size() != 1) {
        reportFailure(module,
                      "CFG Bridge helper lost its unique axis-zero PID");
        return signalPassFailure();
      }
      helperXPids.front().getResult().replaceAllUsesWith(
          virtualPidArgument);
      helperXPids.front().erase();

      moduleBuilder.setInsertionPoint(helper);
      FuncOp publicEntry = FuncOp::create(
          moduleBuilder, helper.getLoc(), publicName, publicType);
      publicEntry->setAttrs(publicAttrs);
      Block *publicBlock = publicEntry.addEntryBlock();
      OpBuilder builder = OpBuilder::atBlockBegin(publicBlock);
      Location loc = publicEntry.getLoc();
      auto physicalPid = GetProgramIdOp::create(builder, loc, 0);
      propagatePhysicalExtent(physicalPid);
      Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
      Value upper = arith::ConstantIntOp::create(
          builder, loc, factor, 32);
      Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
      Value factorValue = arith::ConstantIntOp::create(
          builder, loc, factor, 32);
      Value basePid = arith::MulIOp::create(
          builder, loc, physicalPid, factorValue);

      auto loop = scf::ForOp::create(
          builder, loc, zero, upper, one);
      loop->setAttr(kBridgeSubjectAttr, builder.getUnitAttr());
      loop->setAttr(kBridgeFactorAttr,
                    builder.getI32IntegerAttr(factor));
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(loop.getBody());
      Value virtualPid = arith::AddIOp::create(
          bodyBuilder, loc, basePid, loop.getInductionVar());
      SmallVector<Value> callArguments(publicBlock->getArguments());
      callArguments.push_back(virtualPid);
      CallOp::create(bodyBuilder, loc, helper.getName(), TypeRange{},
                     callArguments);
      ReturnOp::create(builder, loc);
      publicEntry->setAttr(kBridgeOriginAttr,
                           builder.getStringAttr("bridge_constructed"));
      publicEntry->setAttr(kBridgeFactorAttr,
                           builder.getI32IntegerAttr(factor));
      publicEntry->setAttr(kDependenceAttr,
                           builder.getStringAttr(certificate.kind));
      module->setAttr(kBridgeGridDivisorAttr,
                      builder.getI32IntegerAttr(factor));
      return;
    }
    Block &block = (*entry).getBody().front();
    SmallVector<GetProgramIdOp> xProgramIds;
    bool hasNestedTopLevelControl = false;
    for (Operation &op : block.without_terminator()) {
      if (auto pid = dyn_cast<GetProgramIdOp>(op)) {
        if (pid.getAxisAsInt() == 0)
          xProgramIds.push_back(pid);
      }
      // Other launch axes are invariant with respect to the constructed
      // axis-zero loop. Existing structured loops are cloned as part of the
      // already certified program body; Bridge does not rewrite their
      // internal dependence structure. Triton reductions and scans both own
      // region-local combiner blocks; they are dataflow aggregations inside
      // one logical program, not cross-program control flow.
      if (op.getNumRegions() != 0 &&
          !isa<ReduceOp, ScanOp, scf::IfOp, scf::ForOp,
               scf::WhileOp>(op))
        hasNestedTopLevelControl = true;
    }
    if (xProgramIds.size() != 1 || hasNestedTopLevelControl) {
      reportFailure(module,
                    "Bridge requires one axis-zero program ID and closed structured control");
      return signalPassFailure();
    }
    GetProgramIdOp physicalPid = xProgramIds.front();
    auto returnOp = dyn_cast<ReturnOp>(block.getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 0) {
      reportFailure(module, "Bridge requires a void Triton kernel return");
      return signalPassFailure();
    }
    SmallVector<Operation *> bodyOps;
    for (Operation *op = physicalPid->getNextNode(); op && op != returnOp;
         op = op->getNextNode())
      bodyOps.push_back(op);
    if (bodyOps.empty()) {
      reportFailure(module, "Bridge found no program body after axis-zero program ID");
      return signalPassFailure();
    }
    LoopDependenceCertificate certificate =
        certifyBridgeProgramIndependence(*entry, physicalPid);
    if (!certificate.safe) {
      reportFailure(module,
                    Twine("Bridge program-independence proof failed: ") +
                        certificate.reason);
      return signalPassFailure();
    }

    auto directPartition = findDirectPidPartition(*entry, physicalPid);
    // The optional quotient/remainder recurrence needs its divisor before
    // the new loop. A body-local definition will be cloned and erased below;
    // referencing it from the preheader would violate dominance and lifetime.
    // Preserve ordinary body cloning in that case, rather than reject Bridge
    // or speculate/hoist a computation without an independent safety proof.
    if (directPartition) {
      DominanceInfo dominance;
      if (!dominance.dominates(directPartition->extent, bodyOps.front()))
        directPartition.reset();
    }
    OpBuilder builder(bodyOps.front());
    Location loc = physicalPid.getLoc();
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    Value upper = arith::ConstantIntOp::create(builder, loc, factor, 32);
    Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
    Value factorValue = arith::ConstantIntOp::create(builder, loc, factor, 32);
    Value basePid = arith::MulIOp::create(
        builder, loc, physicalPid.getResult(), factorValue);

    SmallVector<Value> initialPartition;
    if (directPartition) {
      initialPartition.push_back(arith::DivSIOp::create(
          builder, loc, basePid, directPartition->extent));
      initialPartition.push_back(arith::RemSIOp::create(
          builder, loc, basePid, directPartition->extent));
    }
    auto loop = scf::ForOp::create(
        builder, loc, zero, upper, one, initialPartition);
    loop->setAttr(kBridgeSubjectAttr, builder.getUnitAttr());
    loop->setAttr(kBridgeFactorAttr, builder.getI32IntegerAttr(factor));

    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(loop.getBody());
    Value virtualPid = arith::AddIOp::create(
        bodyBuilder, loc, basePid, loop.getInductionVar());
    IRMapping mapping;
    mapping.map(physicalPid.getResult(), virtualPid);
    if (directPartition) {
      mapping.map(directPartition->quotient.getResult(),
                  loop.getRegionIterArgs()[0]);
      mapping.map(directPartition->remainder.getResult(),
                  loop.getRegionIterArgs()[1]);
    }
    for (Operation *op : bodyOps) {
      if (directPartition &&
          (op == directPartition->quotient.getOperation() ||
           op == directPartition->remainder.getOperation()))
        continue;
      bodyBuilder.clone(*op, mapping);
    }
    if (directPartition) {
      Value quotient = loop.getRegionIterArgs()[0];
      Value remainder = loop.getRegionIterArgs()[1];
      Value nextRaw = arith::AddIOp::create(
          bodyBuilder, loc, remainder, one);
      Value negativeExtent = arith::SubIOp::create(
          bodyBuilder, loc, zero, directPartition->extent);
      Value extentPositive = arith::CmpIOp::create(
          bodyBuilder, loc, arith::CmpIPredicate::sgt,
          directPartition->extent, zero);
      Value absoluteExtent = arith::SelectOp::create(
          bodyBuilder, loc, extentPositive, directPartition->extent,
          negativeExtent);
      Value crossesPartition = arith::CmpIOp::create(
          bodyBuilder, loc, arith::CmpIPredicate::eq, nextRaw,
          absoluteExtent);
      Value nextRemainder = arith::SelectOp::create(
          bodyBuilder, loc, crossesPartition, zero, nextRaw);
      Value negativeOne = arith::ConstantIntOp::create(
          bodyBuilder, loc, -1, 32);
      Value quotientStep = arith::SelectOp::create(
          bodyBuilder, loc, extentPositive, one, negativeOne);
      Value steppedQuotient = arith::AddIOp::create(
          bodyBuilder, loc, quotient, quotientStep);
      Value nextQuotient = arith::SelectOp::create(
          bodyBuilder, loc, crossesPartition, steppedQuotient, quotient);
      if (!loop.getBody()->empty()) {
        if (auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->back()))
          yield->setOperands(ValueRange{nextQuotient, nextRemainder});
        else
          scf::YieldOp::create(bodyBuilder, loc,
                               ValueRange{nextQuotient, nextRemainder});
      }
      (*entry)->setAttr(
          kBridgePartitionRecurrenceAttr,
          builder.getStringAttr(
              "consecutive_pid_signed_quotient_remainder_recurrence_v1"));
    }
    for (Operation *op : llvm::reverse(bodyOps))
      op->erase();

    (*entry)->setAttr(kBridgeOriginAttr,
                      builder.getStringAttr("bridge_constructed"));
    (*entry)->setAttr(kBridgeFactorAttr, builder.getI32IntegerAttr(factor));
    (*entry)->setAttr(kDependenceAttr,
                      builder.getStringAttr(certificate.kind));
    module->setAttr(kBridgeGridDivisorAttr,
                    builder.getI32IntegerAttr(factor));
    propagatePhysicalExtent(physicalPid);
  }
};
