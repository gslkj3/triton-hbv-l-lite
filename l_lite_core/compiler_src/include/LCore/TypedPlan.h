#pragma once
#include "LCore/Plan.h"
#include "mlir/IR/Builders.h"

namespace mlir::triton::lcore {
inline constexpr llvm::StringLiteral kDecisionCompiler =
    "7c56a5e40f7fd928dfd5c72902d5def0097db73a";

inline DictionaryAttr encodeDecision(const Plan &plan, MLIRContext *context) {
  Builder b(context);
  auto str = [&](llvm::StringRef key, llvm::StringRef value) {
    return b.getNamedAttr(key, b.getStringAttr(value));
  };
  auto num = [&](llvm::StringRef key, int64_t value) {
    return b.getNamedAttr(key, b.getI64IntegerAttr(value));
  };
  SmallVector<Attribute> members, pairs;
  for (const auto &m : plan.members)
    members.push_back(b.getDictionaryAttr({
        str("locator", m.locator), str("ref", m.reference),
        str("capability", m.capability), str("runtime", m.runtimeCertificate),
        str("parent", m.parent), str("context", m.context),
        num("depth", m.depth), num("trip", m.trip), num("factor", m.factor)}));
  for (auto [a, c] : plan.guardedDisjointPairs)
    pairs.push_back(b.getArrayAttr({b.getI64IntegerAttr(a), b.getI64IntegerAttr(c)}));
  return b.getDictionaryAttr({
      str("schema", "l.core.decision.v1"), str("compiler", kDecisionCompiler),
      str("route", plan.route), str("decision", plan.decision),
      str("subject", plan.subject),
      str("scope", plan.nested ? "nested" : plan.independent ? "independent" : "focal"),
      num("stages", plan.stageCount),
      b.getNamedAttr("members", b.getArrayAttr(members)),
      str("guard_ref", plan.memoryGuardRef),
      b.getNamedAttr("guard_pairs", b.getArrayAttr(pairs))});
}

// Recheck the compact protocol even if a client bypasses the audited factory.
// Semantic legality is additionally proved against actual IR by route closing.
inline std::optional<Plan> decodeDecision(Attribute value, std::string &reason) {
  auto reject = [&](llvm::StringRef message) -> std::optional<Plan> {
    reason = message.str();
    return std::nullopt;
  };
  auto root = dyn_cast_or_null<DictionaryAttr>(value);
  if (!root || root.size() != 10)
    return reject("typed L decision envelope required");
  auto string = [](DictionaryAttr d, llvm::StringRef key) -> std::optional<std::string> {
    auto s = d.getAs<StringAttr>(key);
    return s ? std::optional<std::string>(s.getValue().str()) : std::nullopt;
  };
  auto integer = [](DictionaryAttr d, llvm::StringRef key) -> std::optional<int64_t> {
    auto a = d.getAs<IntegerAttr>(key);
    if (!a || !a.getType().isSignlessInteger(64)) return std::nullopt;
    return a.getInt();
  };
  auto route = string(root, "route"), decision = string(root, "decision");
  auto subject = string(root, "subject"), scope = string(root, "scope");
  auto stages = integer(root, "stages");
  auto members = root.getAs<ArrayAttr>("members");
  auto guard = string(root, "guard_ref");
  auto pairs = root.getAs<ArrayAttr>("guard_pairs");
  if (string(root,"schema") != "l.core.decision.v1" ||
      string(root,"compiler") != kDecisionCompiler.str() ||
      !route || !decision || decision->empty() || !subject || subject->empty() ||
      !scope || !stages || !members || members.empty() || !guard || !pairs)
    return reject("typed L decision fields are malformed");
  Plan p;
  p.route = *route; p.decision = *decision; p.subject = *subject;
  p.pipeline = *route == "l.nvidia.software_pipeline.v1";
  bool phase = *route == "l.ttir.full_unroll_phase_major.v1";
  bool vector = *route == "l.ttir.full_unroll_logical_group.v1";
  if (!p.pipeline && !phase && !vector)
    return reject("typed L decision route is unknown");
  p.adapterVersion = p.pipeline ? 13 : phase ? 11 : 12;
  p.nested = *scope == "nested"; p.independent = *scope == "independent";
  p.stageCount = *stages;
  if ((!p.nested && !p.independent && *scope != "focal") ||
      (p.nested && p.pipeline) ||
      (p.independent ? members.size() < 2 : members.size() != 1) ||
      (p.pipeline ? *stages < 2 : *stages != 0))
    return reject("typed L decision scope or stage count is invalid");
  int64_t previous = -1;
  std::set<std::string> refs;
  for (auto value : members) {
    auto m = dyn_cast<DictionaryAttr>(value);
    if (!m || m.size() != 9) return reject("typed L member fields are malformed");
    auto locator = string(m,"locator"), ref = string(m,"ref");
    auto capability = string(m,"capability"), runtime = string(m,"runtime");
    auto parent = string(m,"parent"), context = string(m,"context");
    auto depth = integer(m,"depth"), trip = integer(m,"trip"), factor = integer(m,"factor");
    llvm::StringRef suffix = locator ? llvm::StringRef(*locator) : llvm::StringRef();
    int64_t ordinal = -1;
    bool validLocator = suffix.consume_front("planning-cut.loop.") &&
        !suffix.empty() && !suffix.getAsInteger(10,ordinal) && ordinal > previous;
    if (!validLocator || !ref || ref->empty() || !refs.insert(*ref).second ||
        !capability || capability->empty() || !runtime || !parent || !context ||
        !depth || *depth < 0 || !trip || *trip < 0 || !factor || *factor < 2 ||
        ((*trip > 0) == !runtime->empty()) ||
        (p.pipeline ? *factor != *stages :
          ((*factor & (*factor-1)) || (*trip > 0 && *factor > *trip))) ||
        (p.nested && (parent->empty() || *context != "nested_inner_local_equivalence_v2" || *depth < 1)) ||
        (p.pipeline && (!context->empty() || ((*depth > 0) != !parent->empty()))) ||
        (!p.nested && !p.pipeline && (!parent->empty() || !context->empty() || *depth != 0)))
      return reject("typed L member semantics are not closed");
    p.members.push_back({*locator,*ref,*capability,*runtime,*parent,*context,*depth,*trip,*factor});
    if (!p.memberSignature.empty()) p.memberSignature += ";";
    p.memberSignature += *locator + "=" + std::to_string(*factor);
    previous = ordinal;
  }
  if (guard->empty() != pairs.empty() ||
      (!guard->empty() && (!phase || !llvm::StringRef(*guard).starts_with("l.core.live-memory."))))
    return reject("typed L memory obligation is invalid");
  p.memoryGuardRef = *guard;
  std::pair<unsigned,unsigned> previousPair{0,0};
  for (auto value : pairs) {
    auto pair = dyn_cast<ArrayAttr>(value);
    if (!pair || pair.size() != 2) return reject("typed L memory pair is malformed");
    auto a = dyn_cast<IntegerAttr>(pair[0]), b = dyn_cast<IntegerAttr>(pair[1]);
    if (!a || !b || !a.getType().isSignlessInteger(64) || !b.getType().isSignlessInteger(64) ||
        a.getInt() < 0 || a.getInt() >= b.getInt() || b.getInt() > UINT32_MAX)
      return reject("typed L memory pair is out of range");
    std::pair<unsigned,unsigned> parsed{static_cast<unsigned>(a.getInt()),static_cast<unsigned>(b.getInt())};
    if (!p.guardedDisjointPairs.empty() && parsed <= previousPair)
      return reject("typed L memory pairs are not canonical");
    p.guardedDisjointPairs.push_back(parsed); previousPair = parsed;
  }
  return p;
}
} // namespace mlir::triton::lcore
