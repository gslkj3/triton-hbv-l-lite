// Selected-member operation lineage shared by unroll materialization and
// validation. Role names classify existing annotations; they do not require
// any particular load/compute/reduction mixture for route admission.
bool collectRoles(FuncOp func, SmallVectorImpl<Operation *> &loads,
                  SmallVectorImpl<Operation *> &computes,
                  SmallVectorImpl<Operation *> &reduces,
                  std::optional<StringRef> subject = std::nullopt) {
  bool unknown = false;
  func.walk([&](Operation *op) {
    auto role = op->getAttrOfType<StringAttr>(kRoleAttr);
    if (!role)
      return;
    auto roleSubject = op->getAttrOfType<StringAttr>(kRoleSubjectAttr);
    if (subject && (!roleSubject || roleSubject.getValue() != *subject))
      return;
    if (role.getValue() == "load")
      loads.push_back(op);
    else if (role.getValue() == "compute")
      computes.push_back(op);
    else if (role.getValue() == "reduce")
      reduces.push_back(op);
    else
      unknown = true;
  });
  return !unknown;
}

void clearRoles(ArrayRef<Operation *> operations) {
  for (Operation *op : operations) {
    op->removeAttr(kRoleAttr);
    op->removeAttr(kRoleSubjectAttr);
    op->removeAttr(kRoleIndexAttr);
    op->removeAttr(kUnrollPartitionLineageAttr);
  }
}

std::optional<StringRef>
roleBlockPartition(ArrayRef<Operation *> operations, Block *block) {
  std::optional<StringRef> result;
  for (Operation *operation : operations) {
    if (operation->getBlock() != block)
      continue;
    auto value = operation->getAttrOfType<StringAttr>(
        kUnrollPartitionLineageAttr);
    if (!value)
      continue;
    if (value.getValue() != "main" && value.getValue() != "tail")
      return std::nullopt;
    if (result && *result != value.getValue())
      return std::nullopt;
    result = value.getValue();
  }
  return result;
}

SmallVector<std::string> collectRoleSubjects(FuncOp func) {
  std::set<std::string> seen;
  SmallVector<std::string> result;
  bool hasUnscoped = false;
  func.walk([&](Operation *op) {
    if (!op->hasAttr(kRoleAttr))
      return;
    auto subject = op->getAttrOfType<StringAttr>(kRoleSubjectAttr);
    if (!subject) {
      hasUnscoped = true;
      return;
    }
    if (seen.insert(subject.getValue().str()).second)
      result.push_back(subject.getValue().str());
  });
  if (hasUnscoped && result.empty())
    result.push_back("");
  llvm::sort(result);
  return result;
}
