# Reference Lexicon (auto-drafted)

One `<module>.lexicon.ttl` file per YANG module, drafted by `draft_lexicon.py`
from `yang4owl.py`'s own ontology output. Each entry is a SKOS-style
reference-lexicon record (stable ID, `skos:prefLabel`, `skos:definition`
and/or `skos:scopeNote`, `lex:canonicalExample`, `lex:entityClass`) for a
container/list, `identity`, `grouping`, or enumeration `typedef` -- the
"significant concepts" a module carries, not every leaf/property.

This is **Phase 1** bootstrapping only: pure extraction from what
`yang4owl.py` already emits as `rdfs:label`/`rdfs:comment`. It does not
change `yang4owl.py`'s own pipeline or output, and it does not yet feed
back into the ontology (that's the Overlay Rule Engine, a separate,
not-yet-built phase).

## Regenerating

```bash
python3 yang4owl.py --yang-dir simap-yang --modules simap-yang.yang \
    --base-uri http://example.org/ontology --output /tmp/simap.ttl
python3 yang4owl.py --yang-dir yang-ivy --modules ietf-network-inventory@2026-05-27.yang \
    --base-uri http://example.org/ontology --output /tmp/ivy.ttl
python3 yang4owl.py --yang-dir ietf-teas-yang --modules ietf-te-topology@2020-08-06.yang \
    --base-uri http://example.org/ontology --output /tmp/teas.ttl
python3 yang4owl.py --yang-dir tapi-yang --modules tapi-common.yang \
    --base-uri http://example.org/ontology --output /tmp/tapi.ttl
python3 draft_lexicon.py /tmp/simap.ttl /tmp/ivy.ttl /tmp/teas.ttl /tmp/tapi.ttl \
    --base-uri http://example.org/ontology --out-dir lexicon
```

`http://example.org/ontology` is the base URI baked into the entire
committed corpus; using any other value silently produces a disjoint set
of entries. `yang4owl.py` loads **every** `*.yang` file in `--yang-dir`
regardless of `--modules` (`load_all_modules`) -- `--modules` only needs to
name one real file per tree to act as the entry point. The four trees
above (`simap-yang`, `yang-ivy`, `ietf-teas-yang`, `tapi-yang`) are the
complete input set for the committed corpus.

The four trees may be passed to a single `draft_lexicon.py` invocation
together, as shown, **or** one at a time across separate invocations --
the write path is merge-aware (see below), so either order produces the
same committed result.

## Merge-aware writes

`draft_lexicon.py`'s write path reads any existing `lexicon/<module>.lexicon.ttl`
before writing and unions entries into it, keyed on concept identity
(`(module, local_name)`) with occurrence-level union on
`prov:wasDerivedFrom`. Running the tool against a single source tree adds
its entries to a shared-module file rather than replacing it -- two
different source trees can both contribute entries to the *same* module
file (e.g. `ietf-network.lexicon.ttl` gets entries from plain RFC 8345
usage in `simap-yang`/`yang-ivy` **and** from RFC 8795's deep augmentation
of `ietf-network`'s `node` in `ietf-teas-yang`), and a single-tree run no
longer clobbers the other tree's contribution.

This was hit for real while drafting the `ietf-teas-yang` lexicon: running
`draft_lexicon.py` against only the new TEAS ontology briefly clobbered
`ietf-network.lexicon.ttl` and `ietf-network-topology.lexicon.ttl` down to
just the TEAS-only view before it was caught via `git diff`. Fixed under
`lexicon-69m.11` -- the write path is now merge-aware rather than relying
on remembering to check `git diff` every time.

Each individual module file write is atomic (temp file plus `os.replace`),
so an interrupted run leaves the previous file intact. Stale occurrences
are never pruned (D-08): if an existing entry's `prov:wasDerivedFrom` URI
is no longer produced by any graph in the current run, it is retained as-is
and announced with a `STALE:` line on stdout rather than silently deleted --
see "Known limitations" below.

## `lex:needsCuration true`

An entry is flagged when its definition and scope notes are both empty (no
real YANG `description` text was available across any occurrence) or when
every one of them merely restates the label (e.g. `coaxial-cable` ->
"Coaxial cable.") -- i.e. it fails the "disambiguate, don't decorate"
test. **Never treat a flagged entry as ground truth for binding or
alignment work** until a human has:

1. Written a real disambiguating definition (what distinguishes this
   concept from its near-neighbors?).
2. Filled in `lex:canonicalExample` -- often more disambiguating than the
   definition itself, and always left blank by the drafting tool since it
   must never be guessed.

Un-flagged entries still deserve a skim: present, non-restating text is a
floor, not a guarantee of quality.

## `lex:entityClass`

Currently just the structural origin of the entry (`StructuralKind`,
`IdentityKind`, `GroupingKind`, `EnumeratedKind`) -- not a real domain
classification (node-like, termination-point-like, link, service, ...).
Refining this into genuinely useful shallow classes is itself a curation
task, not something the drafting tool should guess at (see `lexicon-69m.12`
below). When a merged concept's occurrences carry mixed YANG kinds, one
value is resolved by precedence order `IdentityKind > GroupingKind >
EnumeratedKind > StructuralKind` -- a rendering convention for the merge
only, with no bearing on the classification problem itself.

## Concept-level entries, not one entry per URI path

`draft_lexicon.py` makes one entry per `(module, local_name)` *concept*,
not one per unique container/list *path*. TAPI roots nearly everything
under one `context` container and reuses common local names (e.g.
`access-port`) at many different nesting depths -- previously, each
distinct path produced its own entry, and `tapi-common.lexicon.ttl` alone
held 2,571 of them. Fixed under `lexicon-69m.10`: `tapi-common.lexicon.ttl`
now holds **670** entries (corpus-wide, 5,214 -> roughly 2,308), with every
distinct real description text preserved as its own `skos:scopeNote`
(never silently collapsed into one description), `skos:definition` present
only when there is exactly one distinct real text across all occurrences
and omitted otherwise, all contributing source URIs listed as multiple
`prov:wasDerivedFrom` objects on the merged entry, and entries, their scope
notes, and their provenance lists all sorted -- so output is byte-stable
across regeneration runs. This also closes, as a structural consequence of
the new `(module, local_name)` key, a pre-existing silent subject-URI
collision bug found alongside `lexicon-69m.10`: two genuinely different
occurrences sharing a slug used to silently merge their triples on Turtle
parse (last-write-wins for single-valued predicates) -- the new key groups
them deliberately instead, with nothing silently lost.

## Known limitations

The corpus repair above closes `lexicon-69m.10` and `lexicon-69m.11`, but
it is not a clean sweep. These residuals are real and deliberately left
open:

1. **Concurrent invocations are unsupported.** The write path is now a
   read-modify-write cycle with no locking. Each individual file write is
   atomic, so an *interrupted* run leaves the previous file intact -- but
   two `draft_lexicon.py` processes running against the same `--out-dir`
   at once can still lose one's occurrences. Run it once at a time.
2. **`structural_corroboration()` sees only one source path per merged
   concept.** `align_lexicons.py:601` reads `prov:wasDerivedFrom` via
   `graph.value()`, which returns a single object. A merged concept can
   carry many; the emitted one is the sorted-first URI, so it's
   deterministic, but the containment-path signal has reduced fidelity for
   merged concepts. Not a wrong-verdict risk -- `structural_corroboration()`
   returns `None` rather than `0.0` when data is thin, and callers treat
   `None` as "no signal" -- but it is a real precision loss and a candidate
   for a later phase.
3. **Concept identity is `(module, local_name)`, a heuristic rather than a
   proof.** Two genuinely unrelated concepts sharing a module and a leaf
   name would merge into one entry. Every distinct text still survives as
   its own `skos:scopeNote`, so nothing is lost -- but the result would be
   a granularity error. The worst observed case in this corpus
   (`connection-end-point`, 30 occurrences, 7 distinct texts) was checked
   by hand and is a true single concept; the rest of the corpus was not
   exhaustively checked.
4. **`lex:canonicalExample` remains empty corpus-wide, by design.** The
   drafting tool never fabricates one (see recommendation #2 in
   `docs/reference-lexicons.md`). Filling them is human curation, tracked
   as `PROV-02` in `.planning/REQUIREMENTS.md` for v2.
5. **`lex:entityClass` is still structural origin, not a domain
   classification, and is still ineffective as a blocking signal.** That
   is `lexicon-69m.12`, explicitly out of scope for this repair.
6. **Stale occurrences accumulate rather than being pruned** (D-08). An
   occurrence whose source URI no current input regenerates is kept and
   announced with a `STALE:` line on stdout. Nothing deletes it; the
   audit's `occurrences` column (see below) is where growth becomes
   visible over time.

## Curation audit

`audit_lexicon_curation.py` scans every `lexicon/*.lexicon.ttl` file
read-only and reports, per side (TAPI vs. IETF), how many entries carry
real evidence versus placeholder evidence -- closing `lexicon-69m.13`'s
open question about escalation volume for Phase 5. Regenerate it with:

```bash
python3 audit_lexicon_curation.py --lexicon-dir lexicon --out lexicon/CURATION-AUDIT.md
```

The committed `CURATION-AUDIT.md` is tool output, never hand-edited. Use
its `thin_evidence` column, not `lex:needsCuration` alone, to plan review
volume against -- `needsCuration` alone can undercount thin evidence
depending on the exact placeholder shape in the corpus at the time.
