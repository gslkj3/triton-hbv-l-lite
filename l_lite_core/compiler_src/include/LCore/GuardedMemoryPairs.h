#pragma once
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>
#include <utility>

namespace mlir::triton::lcore {
// Narrow pointer forwarding only: never infer provenance through an arbitrary
// operation that happens to have a pointer operand (e.g. pointer loads).
inline std::optional<unsigned> guardedEntryPointerArgument(Value pointer,
                                                           FuncOp entry) {
  for (unsigned depth = 0; pointer && depth < 128; ++depth) {
    if (auto arg = dyn_cast<BlockArgument>(pointer)) {
      if (arg.getOwner() == &entry.getBody().front() &&
          isa<PointerType>(arg.getType()))
        return arg.getArgNumber();
      return std::nullopt;
    }
    auto *op = pointer.getDefiningOp();
    if (!op) return std::nullopt;
    if (auto add = dyn_cast<AddPtrOp>(op)) pointer = add.getPtr();
    else if (isa<SplatOp, BroadcastOp, ExpandDimsOp, ReshapeOp, TransOp>(op) &&
             op->getNumOperands() == 1)
      pointer = op->getOperand(0);
    else return std::nullopt;
  }
  return std::nullopt;
}

// Preconditions belong to a guarded call contract: each listed argument pair
// has disjoint live allocations AND all relevant accesses stay inside them.
// This does not prove SSA independence or allow speculation/region crossing.
inline bool guardedMemoryPairCommutes(
    Operation *left, Operation *right, FuncOp entry,
    ArrayRef<std::pair<unsigned, unsigned>> pairs) {
  auto ordinaryPointer = [](Operation *op) -> Value {
    if (auto load = dyn_cast<LoadOp>(op))
      return load.getIsVolatile() ? Value{} : load.getPtr();
    if (auto store = dyn_cast<StoreOp>(op)) return store.getPtr();
    return {};
  };
  Value a = ordinaryPointer(left), b = ordinaryPointer(right);
  if (!a || !b || left->getBlock() != right->getBlock()) return false;
  auto ar = guardedEntryPointerArgument(a, entry);
  auto br = guardedEntryPointerArgument(b, entry);
  if (!ar || !br || *ar == *br) return false;
  for (auto [x, y] : pairs)
    if ((*ar == x && *br == y) || (*ar == y && *br == x)) return true;
  return false;
}
} // namespace mlir::triton::lcore
