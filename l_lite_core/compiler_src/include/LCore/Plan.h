#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mlir::triton::lcore {

struct Member {
  std::string locator, reference, capability, runtimeCertificate, parent, context;
  int64_t depth, trip, factor;
};

struct Plan {
  std::string route, decision, subject, memberSignature;
  int64_t adapterVersion, stageCount = 0;
  bool pipeline = false;
  bool nested = false, independent = false;
  std::vector<Member> members;
  std::string memoryGuardRef;
  std::vector<std::pair<unsigned, unsigned>> guardedDisjointPairs;
};

inline bool keys(const llvm::json::Object &o,
                 std::initializer_list<llvm::StringRef> expected) {
  if (o.size() != expected.size())
    return false;
  for (auto key : expected)
    if (!o.get(key))
      return false;
  return true;
}

inline std::optional<Plan> parse(const llvm::json::Object &contract,
                               llvm::StringRef compilerCommit,
                               std::string &reason) {
  auto reject = [&](llvm::StringRef message) -> std::optional<Plan> {
    reason = message.str();
    return std::nullopt;
  };
  if (!keys(contract, {"candidate_parameters", "decision_ref", "dynamic_guard_assumptions",
                       "dynamic_materialization_bindings", "fallback_binding", "minimal_provenance",
                       "project_kind", "requested_feedback_fields", "route_ref", "schema_version",
                       "subject_locator"}) || contract.getInteger("schema_version") != 1 ||
      contract.getString("project_kind") != "loop")
    return reject("ordinary L contract envelope is malformed");
  auto route = contract.getString("route_ref");
  auto decision = contract.getString("decision_ref");
  const auto *p = contract.getObject("candidate_parameters");
  if (!p || !route || !decision || decision->empty())
    return reject("ordinary L route or parameters missing");
  bool pipeline = *route == "l.nvidia.software_pipeline.v1";
  bool phase = *route == "l.ttir.full_unroll_phase_major.v1";
  bool vector = *route == "l.ttir.full_unroll_logical_group.v1";
  if (!pipeline && !phase && !vector)
    return reject("route is not one of the three ordinary loop choices");
  int64_t version = pipeline ? 13 : phase ? 11 : 12;
  bool parameterKeys = pipeline
      ? keys(*p, {"subject_ref", "provider_ref", "members", "shared_stage_count", "kind",
                  "subject_policy", "provider_schema", "stage_scope", "adapter_version"})
      : vector ? keys(*p, {"subject_ref", "subject_policy", "provider_ref", "members", "kind",
                          "provider_schema", "packing_policy", "adapter_version"})
               : (keys(*p, {"subject_ref", "subject_policy", "provider_ref", "members", "kind",
                          "provider_schema", "adapter_version"}) ||
                  keys(*p, {"subject_ref", "subject_policy", "provider_ref", "members", "kind",
                          "provider_schema", "adapter_version", "guarded_memory"}));
  auto subject = p->getString("subject_ref"), provider = p->getString("provider_ref");
  auto policy = p->getString("subject_policy");
  bool nested = policy == "provider_bound_nested_inner_dimension_scf_for";
  bool focal = policy == "provider_bound_focal_existing_scf_for";
  bool independent = policy == "provider_bound_independent_existing_scf_for_set";
  const auto *members = p->getArray("members");
  auto stages = p->getInteger("shared_stage_count");
  llvm::StringRef schema = pipeline ? "hbv.loop-provider.bound-pipeline-subject-set.v1"
                          : vector ? "hbv.loop-provider.bound-logical-subject-set.v2"
                                   : "hbv.loop-provider.bound-subject-set.v1";
  if (!parameterKeys || p->getInteger("adapter_version") != version || p->getString("kind") != route ||
      p->getString("provider_schema") != schema || !subject || subject->empty() || !provider || provider->empty() ||
      !members || members->empty() || (!nested && !focal && !independent) || (nested && pipeline) ||
      ((nested || focal) && members->size() != 1) || (independent && members->size() < 2) ||
      (pipeline && (!stages || *stages < 2 || p->getString("stage_scope") != "whole_kernel_shared_option")) ||
      (vector && p->getString("packing_policy") != "registered_exact_operation_capability"))
    return reject("ordinary L subject set is not closed");
  Plan result;
  if (const auto *guard = p->getObject("guarded_memory")) {
    auto ref = guard->getString("guard_ref");
    const auto *pairs = guard->getArray("disjoint_argument_pairs");
    if (!phase || !keys(*guard, {"schema", "guard_ref", "disjoint_argument_pairs"}) ||
        guard->getString("schema") != "l.core.guarded-memory.v1" || !ref ||
        !ref->starts_with("l.core.live-memory.") || !pairs || pairs->empty())
      return reject("conditional memory contract malformed");
    std::pair<unsigned,unsigned> previousPair{0,0};
    for (const auto &value : *pairs) {
      const auto *pair = value.getAsArray();
      if (!pair || pair->size()!=2) return reject("conditional argument pair malformed");
      auto a=(*pair)[0].getAsInteger(), b=(*pair)[1].getAsInteger();
      if (!a || !b || *a<0 || *a>=*b || *b>UINT32_MAX)
        return reject("conditional argument pair out of range");
      std::pair<unsigned,unsigned> parsed{static_cast<unsigned>(*a),static_cast<unsigned>(*b)};
      if (!result.guardedDisjointPairs.empty() && parsed<=previousPair)
        return reject("conditional argument pairs are not canonical");
      result.guardedDisjointPairs.push_back(parsed); previousPair=parsed;
    }
    result.memoryGuardRef=ref->str();
  } else if (p->get("guarded_memory")) return reject("conditional memory contract is not an object");
  result.route = route->str(); result.decision = decision->str(); result.subject = subject->str();
  result.adapterVersion = version; result.pipeline = pipeline;
  result.nested = nested; result.independent = independent;
  result.stageCount = pipeline ? *stages : 0;
  int64_t previous = -1;
  std::set<std::string> references;
  for (const auto &value : *members) {
    const auto *m = value.getAsObject();
    if (!m || !keys(*m, {"exact_static_trip_count", "factor_admission_ref", "member_ref",
                        "nested_context_certificate_ref", "nesting_depth", "parent_loop_locator",
                        "provider_loop_locator", "route_capability_certificate_ref", "route_factor",
                        "route_factor_kind", "runtime_main_tail_certificate_ref", "schema"}))
      return reject("ordinary L member fields are malformed");
    auto locator = m->getString("provider_loop_locator"), ref = m->getString("member_ref");
    auto capability = m->getString("route_capability_certificate_ref"), admission = m->getString("factor_admission_ref");
    auto runtime = m->getString("runtime_main_tail_certificate_ref"), parent = m->getString("parent_loop_locator");
    auto context = m->getString("nested_context_certificate_ref");
    auto depth = m->getInteger("nesting_depth"), trip = m->getInteger("exact_static_trip_count");
    auto factor = m->getInteger("route_factor");
    llvm::StringRef suffix = locator.value_or("");
    int64_t ordinal = -1;
    bool locatorValid = suffix.consume_front("planning-cut.loop.") && !suffix.empty() &&
                        !suffix.getAsInteger(10, ordinal) && ordinal > previous;
    bool exact = trip && *trip > 0;
    bool unknownTrip = m->get("exact_static_trip_count")->kind() == llvm::json::Value::Null;
    bool runtimeSubject = runtime && !runtime->empty();
    bool parentProof = parent && !parent->empty() && context &&
        *context == "nested_inner_local_equivalence_v2";
    if (!locatorValid || !ref || ref->empty() || !references.insert(ref->str()).second ||
        !capability || capability->empty() || !admission || admission->empty() || !runtime || !parent || !context ||
        !depth || *depth < 0 || !factor || *factor < 2 || (!exact && !unknownTrip) || exact == runtimeSubject ||
        (!pipeline && ((*factor & (*factor-1)) || (exact && *factor > *trip))) ||
        (pipeline && *factor != *stages) ||
        m->getString("schema") != "hbv.loop.provider-bound-route-member.v1" ||
        m->getString("route_factor_kind") != (pipeline ? "pipeline_stage_count" : phase ? "phase_reorder_grouping_width" : "logical_vector_grouping_width") ||
        (nested && (!parentProof || *depth < 1)) ||
        (pipeline && (!context->empty() || ((*depth > 0) != !parent->empty()))) ||
        (!nested && !pipeline && (!parent->empty() || !context->empty() || *depth != 0)))
      return reject("ordinary L member semantics are not closed");
    result.members.push_back({locator->str(), ref->str(), capability->str(), runtime->str(),
                              parent->str(), context->str(), *depth, exact ? *trip : 0, *factor});
    if (!result.memberSignature.empty()) result.memberSignature += ";";
    result.memberSignature += locator->str()+"="+std::to_string(*factor);
    previous = ordinal;
  }
  const auto *bindings = contract.getArray("dynamic_materialization_bindings");
  if (!bindings || bindings->size() != (pipeline ? result.members.size()+1 : result.members.size()*2))
    return reject("ordinary L materialization binding count differs from members");
  auto bindingMatches = [&](size_t i, llvm::StringRef role, llvm::StringRef owner,
                            llvm::StringRef key, llvm::StringRef value) {
    const auto *b = (*bindings)[i].getAsObject();
    return b && keys(*b, {"semantic_role", "native_owner_or_binding_kind", "native_key_or_adapter_id",
                          "typed_value_or_typed_reference", "required", "binding_schema_version"}) &&
           b->getString("semantic_role") == role && b->getString("native_owner_or_binding_kind") == owner &&
           b->getString("native_key_or_adapter_id") == key && b->getString("typed_value_or_typed_reference") == value &&
           b->getBoolean("required") == true && b->getInteger("binding_schema_version") == 1;
  };
  if (pipeline && !bindingMatches(0, "loop_pipeline_stage_count", "native_triton_pipeline_option",
                                  "tt.num_stages@whole_kernel", "stage_count:"+std::to_string(*stages)))
    return reject("pipeline stage binding differs from its members");
  for (size_t i=0; i<result.members.size(); ++i) {
    const auto &m = result.members[i];
    if (pipeline) {
      if (!bindingMatches(i+1, "loop_pipeline_member", "native_triton_software_pipeline",
                          "hbv.loop.pipeline.provider_bound.v13@"+m.locator, m.reference))
        return reject("pipeline member binding differs from its plan");
    } else {
      std::string adapter = phase ? "hbv.loop.phase_major.provider_bound.v11@" : "hbv.loop.logical_group.provider_bound.v12@";
      if (!bindingMatches(i*2, "loop_full_unroll_member", "native_triton_loop_unroll",
                          "tt.loop_unroll_factor@"+m.locator, "factor:"+std::to_string(m.factor)+";member:"+m.reference) ||
          !bindingMatches(i*2+1, "loop_route_member", "triton_hbv_loop_adapter", adapter+m.locator, m.reference))
        return reject("unroll member binding differs from its plan");
    }
  }
  const auto *locator = contract.getObject("subject_locator"), *fallback = contract.getObject("fallback_binding");
  const auto *provenance = contract.getObject("minimal_provenance");
  if (!locator || !keys(*locator, {"anchor_or_marker_ref", "kind", "subject_ref"}) ||
      locator->getString("kind") != "hbv_typed_marker" || locator->getString("subject_ref") != subject ||
      locator->getString("anchor_or_marker_ref") != "tt.hbv.l.subject" ||
      !fallback || !keys(*fallback, {"max_original_route_retries", "original_route_ref", "retry_loop_guard"}) ||
      fallback->getInteger("max_original_route_retries") != 1 || fallback->getString("original_route_ref") != "l.original.default" ||
      fallback->getString("retry_loop_guard") != "hbv_disable_decision" ||
      !provenance || !keys(*provenance, {"adapter_version", "compiler_commit", "producer_schema", "source_ref"}) ||
      provenance->getInteger("adapter_version") != version || provenance->getString("compiler_commit") != compilerCommit ||
      provenance->getString("producer_schema") != "hbv.plan_contract.v1" || !provenance->getString("source_ref"))
    return reject("ordinary L provenance, subject or fallback binding is malformed");
  const auto *guards = contract.getArray("dynamic_guard_assumptions");
  const auto *feedback = contract.getArray("requested_feedback_fields");
  std::vector<llvm::StringRef> guardIds = {"l.bundle.schema", "l.target.binding",
      pipeline ? "l.subject.provider_bound_pipeline_set" : nested ? "l.subject.provider_bound_nested_inner" : focal ? "l.subject.provider_bound_focal" : "l.subject.provider_bound_independent_set",
      "l.subject.structure", "l.effects_dependencies", "l.parameters.closed", "l.route.mutual_exclusion",
      pipeline ? "l.pipeline.member_lineage" : "l.unroll.lineage", "l.route.postcondition", "l.ir.verify", "l.observation.correspondence"};
  std::vector<llvm::StringRef> fieldIds = {"loop.route.realized", "loop.route.parameters",
      pipeline ? "loop.pipeline.member_artifact_lineage" : "loop.route.postcondition",
      pipeline ? "loop.route.postcondition" : "codegen.instruction_family_counts"};
  if (!result.memoryGuardRef.empty()) guardIds.push_back("l.live_memory");
  if (!guards || guards->size() != guardIds.size() || !feedback || feedback->size() != fieldIds.size())
    return reject("ordinary L validation obligations are incomplete");
  for (size_t i=0; i<guardIds.size(); ++i) {
    const auto *g = (*guards)[i].getAsObject();
    if (!g || !keys(*g, {"guard_id", "required", "responsible_stage", "verifier_or_legality_binding"}) ||
        g->getString("guard_id") != guardIds[i] || g->getBoolean("required") != true ||
        !g->getString("responsible_stage") || !g->getString("verifier_or_legality_binding"))
      return reject("ordinary L validation guard is malformed");
    if (guardIds[i]=="l.live_memory" &&
        (g->getString("responsible_stage")!="every_launch_before_conditional_kernel" ||
         g->getString("verifier_or_legality_binding")!=result.memoryGuardRef))
      return reject("conditional memory launch obligation is unbound");
  }
  for (size_t i=0; i<fieldIds.size(); ++i) {
    const auto *f = (*feedback)[i].getAsObject();
    if (!f || !keys(*f, {"availability_stage", "evidence_sink", "field_id", "required", "source_kind"}) ||
        f->getString("field_id") != fieldIds[i] || f->getBoolean("required") != true ||
        !f->getString("availability_stage") || !f->getString("evidence_sink") || !f->getString("source_kind"))
      return reject("ordinary L feedback obligation is malformed");
  }
  return result;
}
// Standalone current envelope. Absence of a plan is handled by the driver;
// an explicitly supplied malformed or historical plan is never a no-op.
inline std::optional<Plan> parseBundle(const llvm::json::Object &bundle,
                                     llvm::StringRef compilerCommit,
                                     std::string &reason) {
  auto identity = bundle.getString("bundle_id");
  const auto *contract = bundle.getObject("contract");
  if (!keys(bundle, {"bundle_id", "contract", "schema_version"}) ||
      bundle.getInteger("schema_version") != 1 || !identity || identity->empty() ||
      !contract) {
    reason = "ordinary L bundle envelope is malformed";
    return std::nullopt;
  }
  return parse(*contract, compilerCommit, reason);
}

inline std::optional<Plan> parseBundle(llvm::StringRef payload,
                                     llvm::StringRef compilerCommit,
                                     std::string &reason) {
  auto value = llvm::json::parse(payload);
  if (!value) {
    llvm::consumeError(value.takeError());
    reason = "ordinary L bundle is not valid JSON";
    return std::nullopt;
  }
  const auto *bundle = value->getAsObject();
  if (!bundle) {
    reason = "ordinary L bundle is not an object";
    return std::nullopt;
  }
  return parseBundle(*bundle, compilerCommit, reason);
}
} // namespace mlir::triton::lcore
