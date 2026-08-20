#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include <limits>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPUNROLL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class LoopUnrollPass : public impl::TritonLoopUnrollBase<LoopUnrollPass> {

  int getUnrollFactorOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise set the
    // factor to 1 to suppress the unrolling.
    if (auto factor =
            forOp->getAttrOfType<IntegerAttr>(loopUnrollFactorAttrName))
      return factor.getInt();
    return 1;
  }

  const char *loopUnrollFactorAttrName = "tt.loop_unroll_factor";
  const char *pipelineStagesAttrName = "tt.num_stages";
  const char *hbvMainTailAttrName = "tt.hbv.l.main_tail";
  const char *hbvPartitionLineageAttrName =
      "tt.hbv.l.unroll_partition_lineage";
  const char *hbvRoleAttrName = "tt.hbv.l.role";
  const char *hbvRoleSubjectAttrName = "tt.hbv.l.role_subject";
  const char *hbvSourceExactTripCountAttrName =
      "tt.hbv.l.source_exact_trip_count";

public:
  void runOnOperation() override {
    LDBG("Loop unroll pass");
    SmallVector<scf::ForOp, 4> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with unroll factor <= 1.
      if (getUnrollFactorOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    auto ctx = getOperation()->getContext();
    for (auto loop : loops) {
      auto unrollFactor = getUnrollFactorOrDefault(loop);
      auto hbvMainTail =
          loop->getAttrOfType<StringAttr>(hbvMainTailAttrName);
      bool hbvFactorRequested =
          hbvMainTail &&
          (hbvMainTail.getValue() == "native_factor_requested" ||
           hbvMainTail.getValue() == "native_dynamic_requested");
      SmallVector<StringAttr> roleSubjects;
      SmallVector<std::pair<StringAttr, int64_t>> sourceRoleCounts;
      std::optional<int64_t> exactTripCount;
      if (auto exact = loop->getAttrOfType<IntegerAttr>(
              hbvSourceExactTripCountAttrName))
        exactTripCount = exact.getInt();
      APInt lowerValue, upperValue, stepValue;
      if (!exactTripCount &&
          matchPattern(loop.getLowerBound(), m_ConstantInt(&lowerValue)) &&
          matchPattern(loop.getUpperBound(), m_ConstantInt(&upperValue)) &&
          matchPattern(loop.getStep(), m_ConstantInt(&stepValue))) {
        int64_t lower = lowerValue.getSExtValue();
        int64_t upper = upperValue.getSExtValue();
        int64_t step = stepValue.getSExtValue();
        if (step > 0 && upper >= lower) {
          __int128 delta = static_cast<__int128>(upper) - lower;
          __int128 trip = (delta + step - 1) / step;
          if (trip <= std::numeric_limits<int64_t>::max())
            exactTripCount = static_cast<int64_t>(trip);
        }
      }
      loop->removeAttr(hbvSourceExactTripCountAttrName);
      Operation *lineageScope = loop->getParentOp();
      while (lineageScope->getParentOp() &&
             !isa<ModuleOp>(lineageScope->getParentOp()))
        lineageScope = lineageScope->getParentOp();
      if (hbvFactorRequested) {
        loop.walk([&](Operation *operation) {
          auto subject = operation->getAttrOfType<StringAttr>(
              hbvRoleSubjectAttrName);
          if (!subject)
            return;
          auto existing = llvm::find_if(
              sourceRoleCounts,
              [&](const auto &entry) { return entry.first == subject; });
          if (existing == sourceRoleCounts.end()) {
            roleSubjects.push_back(subject);
            sourceRoleCounts.emplace_back(subject, 1);
          } else {
            ++existing->second;
          }
        });
      }
      loop->removeAttr(loopUnrollFactorAttrName);
      LDBG("Unrolling loop by " << unrollFactor << " times\n" << loop);
      auto resultLoops = loopUnrollByFactor(loop, unrollFactor);
      if (succeeded(resultLoops) && hbvFactorRequested) {
        if (resultLoops->mainLoopOp) {
          (*resultLoops->mainLoopOp)
              ->setAttr(hbvMainTailAttrName,
                        StringAttr::get(ctx, "main"));
        }
        if (resultLoops->epilogueLoopOp) {
          (*resultLoops->epilogueLoopOp)
              ->setAttr(hbvMainTailAttrName,
                        StringAttr::get(ctx, "tail"));
        }
        auto isWithin = [](Operation *operation,
                           std::optional<scf::ForOp> partition) {
          if (!partition)
            return false;
          for (Operation *parent = operation; parent;
               parent = parent->getParentOp())
            if (parent == partition->getOperation())
              return true;
          return false;
        };
        // `loopUnrollByFactor` may materialize a one-iteration static
        // epilogue directly into the parent block and return no epilogue loop
        // handle.  The cloned role-subject lineage survives that rewrite, so
        // classify every clone relative to the returned main/epilogue loops.
        // Clones outside a surviving main loop are its ordered tail.  If no
        // main loop survives, a fully consumed exact-factor body is the main
        // partition; an explicit epilogue handle still wins as tail.
        SmallVector<std::pair<StringAttr, SmallVector<Operation *>>>
            clonedRoles;
        for (StringAttr subject : roleSubjects)
          clonedRoles.emplace_back(subject, SmallVector<Operation *>{});
        lineageScope->walk([&](Operation *operation) {
          if (!operation->hasAttr(hbvRoleAttrName))
            return;
          auto subject = operation->getAttrOfType<StringAttr>(
              hbvRoleSubjectAttrName);
          if (!subject || !llvm::is_contained(roleSubjects, subject))
            return;
          auto entry = llvm::find_if(
              clonedRoles,
              [&](const auto &item) { return item.first == subject; });
          entry->second.push_back(operation);
        });
        for (auto &[subject, operations] : clonedRoles) {
          StringAttr currentSubject = subject;
          auto source = llvm::find_if(
              sourceRoleCounts,
              [&](const auto &item) {
                return item.first == currentSubject;
              });
          int64_t remainder =
              exactTripCount ? *exactTripCount % unrollFactor : 0;
          int64_t tailRoleCount = remainder * source->second;
          bool handleFreeStaticPartition =
              !resultLoops->mainLoopOp && !resultLoops->epilogueLoopOp &&
              exactTripCount && remainder > 0 &&
              static_cast<int64_t>(operations.size()) ==
                  *exactTripCount * source->second;
          for (auto [ordinal, operation] : llvm::enumerate(operations)) {
            StringRef partition =
                isWithin(operation, resultLoops->mainLoopOp) ? "main"
                : isWithin(operation, resultLoops->epilogueLoopOp) ? "tail"
                : resultLoops->mainLoopOp ? "tail"
                : handleFreeStaticPartition &&
                          static_cast<int64_t>(ordinal) >=
                              static_cast<int64_t>(operations.size()) -
                                  tailRoleCount
                    ? "tail"
                    : "main";
            operation->setAttr(
                hbvPartitionLineageAttrName,
                StringAttr::get(ctx, partition));
          }
        }
      }
      // Do not pipeline the epilog loop.
      if (succeeded(resultLoops) && resultLoops->epilogueLoopOp) {
        (*resultLoops->epilogueLoopOp)
            ->setAttr(pipelineStagesAttrName,
                      mlir::IntegerAttr::get(IntegerType::get(ctx, 32), 1));
      }
    }
  }
};

} // namespace mlir::triton
