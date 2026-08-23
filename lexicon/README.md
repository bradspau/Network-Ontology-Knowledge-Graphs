# Reference Lexicon (auto-drafted)

One `<module>.lexicon.ttl` file per YANG module, drafted by `draft_lexicon.py`
from `yang4owl.py`'s own ontology output. Each entry is a SKOS-style
reference-lexicon record (stable ID, `skos:prefLabel`, `skos:definition`,
`lex:canonicalExample`, `lex:entityClass`) for a container/list, `identity`,
`grouping`, or enumeration `typedef` -- the "significant concepts" a module
carries, not every leaf/property.

This is **Phase 1** bootstrapping only: pure extraction from what
`yang4owl.py` already emits as `rdfs:label`/`rdfs:comment`. It does not
change `yang4owl.py`'s own pipeline or output, and it does not yet feed
back into the ontology (that's the Overlay Rule Engine, a separate,
not-yet-built phase).

## Regenerating

```bash
python3 yang4owl.py --yang-dir <source-tree> --modules <module.yang> \
    --base-uri <base-uri> --output /tmp/ontology.ttl
python3 draft_lexicon.py /tmp/ontology.ttl --base-uri <base-uri> --out-dir lexicon
```

Multiple ontology `.ttl` files (e.g. one per source tree) can be passed to
a single `draft_lexicon.py` invocation; entries are grouped by the module
segment of each class's own URI, not by which file they came from.

## `lex:needsCuration true`

An entry is flagged when its definition is empty (no YANG `description`
was available) or when it merely restates the label (e.g. `coaxial-cable`
-> "Coaxial cable.") -- i.e. it fails the "disambiguate, don't decorate"
test. **Never treat a flagged entry as ground truth for binding or
alignment work** until a human has:

1. Written a real disambiguating definition (what distinguishes this
   concept from its near-neighbors?).
2. Filled in `lex:canonicalExample` -- often more disambiguating than the
   definition itself, and always left blank by the drafting tool since it
   must never be guessed.

Un-flagged entries still deserve a skim: a present, non-restating
definition is a floor, not a guarantee of quality.

## `lex:entityClass`

Currently just the structural origin of the entry (`StructuralKind`,
`IdentityKind`, `GroupingKind`, `EnumeratedKind`) -- not a real domain
classification (node-like, termination-point-like, link, service, ...).
Refining this into genuinely useful shallow classes is itself a curation
task, not something the drafting tool should guess at.
