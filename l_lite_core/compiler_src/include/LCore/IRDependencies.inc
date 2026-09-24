// Structural SSA reachability and pointer-root analysis, not alias disproof.
bool valueDependsOn(Value value, Value target,
                    llvm::SmallPtrSetImpl<Value> &visited) {
  if (value == target)
    return true;
  if (!visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOn(operand, target, visited);
  });
}

bool valueDependsOn(Value value, Value target) {
  llvm::SmallPtrSet<Value, 32> visited;
  return valueDependsOn(value, target, visited);
}

bool isPointerLike(Type type) {
  if (isa<PointerType>(type))
    return true;
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && isa<PointerType>(shaped.getElementType());
}

void collectPointerRoots(Value value, llvm::SmallPtrSetImpl<Value> &roots,
                         llvm::SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  if (isa<BlockArgument>(value)) {
    if (isPointerLike(value.getType()))
      roots.insert(value);
    return;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return;
  for (Value operand : definition->getOperands())
    if (isPointerLike(operand.getType()))
      collectPointerRoots(operand, roots, visited);
}

std::optional<Value> uniquePointerRoot(Value pointer) {
  llvm::SmallPtrSet<Value, 4> roots;
  llvm::SmallPtrSet<Value, 32> visited;
  collectPointerRoots(pointer, roots, visited);
  if (roots.size() != 1)
    return std::nullopt;
  return *roots.begin();
}
