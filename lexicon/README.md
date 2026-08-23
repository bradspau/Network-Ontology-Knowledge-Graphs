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

## Known limitation: one entry per URI path, not per concept

`draft_lexicon.py` makes one entry per unique container/list *path*, not one
per underlying *concept*. On flatter corpora (the IETF modules this was
originally drafted against) that distinction rarely matters. It matters a
lot on `tapi-*`: TAPI roots nearly everything under one `context` container
and reuses common local names (e.g. `access-port`) at many different
nesting depths, each carrying its own genuinely distinct YANG description.
The result is `tapi-common.lexicon.ttl` alone holding **2,571** entries --
not fabricated, not identical duplicates (verified: 7 separate
`access-port` entries, each with different real description text), but
7 different *relationships to* what a reference lexicon should probably
treat as one *concept*, per the reference-lexicons draft's identity-first
framing. Left as-is deliberately for now (correct but verbose) rather than
guessing at a deduplication heuristic; revisit if/when this actually blocks
binding or alignment work (`lexicon-69m.5`).
