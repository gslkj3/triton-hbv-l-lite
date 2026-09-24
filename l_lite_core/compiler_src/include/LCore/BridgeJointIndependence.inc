// Joint launch-coordinate proof. Per-axis injectivity is insufficient:
// x+y is injective with either coordinate fixed, but not jointly injective.
// This deliberately proves a sufficient affine rule, not observed behavior.
struct BridgeJointFootprint {
  std::array<int64_t, 3> strides{};
  int64_t low = 0, high = 0;
};

std::optional<BridgeJointFootprint> bridgeJointOffset(
    Value value, const std::array<int64_t, 3> &extents, unsigned &budget) {
  if (!budget) return std::nullopt;
  --budget;
  BridgeJointFootprint result;
  Operation *op = value.getDefiningOp();
  if (auto pid = dyn_cast_or_null<GetProgramIdOp>(op)) {
    if (extents[pid.getAxisAsInt()] < 1) return std::nullopt;
    result.strides[pid.getAxisAsInt()] = 1;
  } else if (auto constant = splatInteger(value)) {
    result.low = result.high = *constant;
  } else if (auto range = dyn_cast_or_null<MakeRangeOp>(op)) {
    result.low = range.getStart();
    result.high = range.getEnd() - 1;
    if (result.high < result.low) return std::nullopt;
  } else if (op) {
    StringRef name = op->getName().getStringRef();
    if (name == "tt.splat" || name == "tt.broadcast" ||
        name == "tt.expand_dims" || name == "tt.reshape" ||
        name == "tt.trans" || name == "arith.extsi") {
      auto input = bridgeJointOffset(op->getOperand(0), extents, budget);
      if (!input) return std::nullopt;
      result = *input;
    } else if (name == "arith.addi" || name == "arith.subi") {
      auto lhs = bridgeJointOffset(op->getOperand(0), extents, budget);
      auto rhs = bridgeJointOffset(op->getOperand(1), extents, budget);
      if (!lhs || !rhs) return std::nullopt;
      bool subtract = name == "arith.subi";
      for (unsigned axis = 0; axis < 3; ++axis) {
        auto stride = checkedI64(static_cast<__int128>(lhs->strides[axis]) +
            (subtract ? -static_cast<__int128>(rhs->strides[axis]) : rhs->strides[axis]));
        if (!stride) return std::nullopt;
        result.strides[axis] = *stride;
      }
      auto low = checkedI64(static_cast<__int128>(lhs->low) +
                           (subtract ? -static_cast<__int128>(rhs->high) : rhs->low));
      auto high = checkedI64(static_cast<__int128>(lhs->high) +
                            (subtract ? -static_cast<__int128>(rhs->low) : rhs->high));
      if (!low || !high) return std::nullopt;
      result.low = *low; result.high = *high;
    } else if (name == "arith.muli") {
      auto scale = splatInteger(op->getOperand(1));
      Value input = op->getOperand(0);
      if (!scale) { scale = splatInteger(input); input = op->getOperand(1); }
      if (!scale) return std::nullopt;
      auto term = bridgeJointOffset(input, extents, budget);
      if (!term) return std::nullopt;
      for (unsigned axis = 0; axis < 3; ++axis) {
        auto stride = checkedI64(static_cast<__int128>(term->strides[axis]) * *scale);
        if (!stride) return std::nullopt;
        result.strides[axis] = *stride;
      }
      auto a = checkedI64(static_cast<__int128>(term->low) * *scale);
      auto b = checkedI64(static_cast<__int128>(term->high) * *scale);
      if (!a || !b) return std::nullopt;
      result.low = std::min(*a, *b); result.high = std::max(*a, *b);
    } else return std::nullopt;
  } else return std::nullopt;
  // Prove every integer intermediate, not just the final offset, cannot wrap.
  auto type = dyn_cast<IntegerType>(getElementTypeOrSelf(value.getType()));
  if (!type || type.getWidth() > 64) return std::nullopt;
  __int128 low = result.low, high = result.high;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (!result.strides[axis]) continue;
    if (extents[axis] < 1) return std::nullopt;
    __int128 delta = static_cast<__int128>(result.strides[axis]) * (extents[axis]-1);
    low += std::min<__int128>(0, delta);
    high += std::max<__int128>(0, delta);
  }
  __int128 limit = static_cast<__int128>(1) << (type.getWidth()-1);
  if (low < -limit || high >= limit) return std::nullopt;
  return result;
}

bool provesBridgeJointIndependence(FuncOp entry,
                                  ArrayRef<GetProgramIdOp> pids) {
  std::array<int64_t, 3> extents{};
  for (GetProgramIdOp pid : pids) {
    auto extent = pid->getAttrOfType<IntegerAttr>(kBridgeAxisExtentAttr);
    if (!extent || extent.getInt() < 1) return false;
    unsigned axis = pid.getAxisAsInt();
    if (extents[axis] && extents[axis] != extent.getInt()) return false;
    extents[axis] = extent.getInt();
  }
  llvm::DenseMap<Value, BridgeJointFootprint> outputs;
  bool valid = true;
  entry.walk([&](StoreOp store) {
    if (!valid) return;
    auto root = uniquePointerRoot(store.getPtr());
    Value pointer = store.getPtr();
    BridgeJointFootprint total;
    unsigned budget = 4096;
    while (root && pointer != *root) {
      Operation *op = pointer.getDefiningOp();
      if (!op || !budget) { valid = false; return; }
      --budget;
      if (auto add = dyn_cast<AddPtrOp>(op)) {
        auto term = bridgeJointOffset(add.getOffset(), extents, budget);
        if (!term) { valid = false; return; }
        for (unsigned axis = 0; axis < 3; ++axis) {
          auto sum = checkedI64(static_cast<__int128>(total.strides[axis]) + term->strides[axis]);
          if (!sum) { valid = false; return; }
          total.strides[axis] = *sum;
        }
        auto low = checkedI64(static_cast<__int128>(total.low) + term->low);
        auto high = checkedI64(static_cast<__int128>(total.high) + term->high);
        if (!low || !high) { valid = false; return; }
        total.low = *low; total.high = *high;
        pointer = add.getPtr();
      } else {
        StringRef name = op->getName().getStringRef();
        if (name != "tt.splat" && name != "tt.broadcast" &&
            name != "tt.expand_dims" && name != "tt.reshape" && name != "tt.trans") {
          valid = false; return;
        }
        pointer = op->getOperand(0);
      }
    }
    if (!root) { valid = false; return; }
    auto [it, inserted] = outputs.try_emplace(*root, total);
    if (!inserted) {
      if (it->second.strides != total.strides) { valid = false; return; }
      it->second.low = std::min(it->second.low, total.low);
      it->second.high = std::max(it->second.high, total.high);
    }
  });
  if (!valid) return false;
  for (auto &output : outputs) {
    auto &footprint = output.second;
    SmallVector<std::pair<__int128, int64_t>> axes;
    for (unsigned axis = 0; axis < 3; ++axis) {
      if (extents[axis] <= 1) continue;
      __int128 stride = footprint.strides[axis];
      axes.emplace_back(stride < 0 ? -stride : stride, extents[axis]);
    }
    llvm::sort(axes);
    __int128 span = static_cast<__int128>(footprint.high) - footprint.low;
    for (auto [stride, extent] : axes) {
      if (stride <= span) return false;
      span += stride * (extent-1);
    }
  }
  return true;
}
