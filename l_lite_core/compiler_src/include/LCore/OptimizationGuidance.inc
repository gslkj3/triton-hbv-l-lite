// A validated L plan opts the selected ordinary loop into route exploration.
// Preserve the developer's old optimization hints as provenance, not as a
// capability veto. Never remove unrelated attributes or semantic constraints.
void recordReplacedLoopGuidance(FuncOp entry, scf::ForOp loop, StringRef locator) {
  Builder builder(entry.getContext());
  SmallVector<NamedAttribute> fields;
  fields.push_back(builder.getNamedAttr("loop", builder.getStringAttr(locator)));
  for (StringRef name : {StringRef("tt.num_stages"), StringRef("tt.loop_unroll_factor")})
    if (Attribute value = loop->getAttr(name))
      fields.push_back(builder.getNamedAttr(name, value));
  if (fields.size() == 1) return;
  SmallVector<Attribute> records;
  if (auto prior = entry->getAttrOfType<ArrayAttr>("tt.hbv.l.replaced_loop_guidance"))
    records.append(prior.begin(), prior.end());
  records.push_back(builder.getDictionaryAttr(fields));
  entry->setAttr("tt.hbv.l.replaced_loop_guidance", builder.getArrayAttr(records));
}
