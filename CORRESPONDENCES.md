# Correspondences (`correspondences.ttl`)

## What this artifact is

The committed output of `align_lexicons.py`'s two-stage label + definition/
example matcher: one SKOS correspondence per confirmed TAPI↔IETF/TEAS pair,
carrying confidence, evidence and lexicon version inline so a correspondence
can be judged in isolation, without the run transcript that produced it.

This file currently covers the same fixture-scale corpus Phases 1-3 built
and validated against — the eleven-entry OTN worked example
(`docs/reference-lexicons.md` §6), not the full TAPI/IETF corpus. Phase 5 is
where the full-corpus run happens; do not mistake a fixture-scale
`correspondences.ttl` for a complete alignment.

## Producing one

```bash
python3 align_lexicons.py --emit-correspondences correspondences.ttl
```

`--emit-correspondences` takes an optional path argument:

- Omitted entirely — no file is written. A run without the flag behaves
  exactly as before this artifact existed (OUT-01's compatibility
  guarantee).
- Given with no value — writes to `correspondences.ttl` next to
  `align_lexicons.py`.
- Given with a value — writes to that path.

The artifact is written only when the flag is given, and only when the run
completes without stopping early (a `--max-calls` budget stop writes no
file — a partial artifact would misstate what the run established).

## Before you run: the lexicon must be committed and clean

Every run — with or without `--emit-correspondences` — resolves
`lex:lexiconVersion` as the git commit hash of the last commit that touched
`lexicon/`, and refuses to run at all if that directory has uncommitted or
untracked changes:

```bash
git status --porcelain -- lexicon
```

If that command prints anything, the run will refuse before making any
LLM call. This is deliberate: a recorded version hash that does not
describe the bytes actually matched against would make the artifact
uncitable — the whole point of `lex:lexiconVersion` is that a reader can
check out that exact commit and see what was matched. Two ways out:

- Commit the change: `git add lexicon/ && git commit -m "..."`.
- Or stash it if it isn't meant to be kept yet: `git stash`.

The consequence is fail-closed rather than fail-open: if `lexicon/` is
outside a git repository at all, no version can be resolved, and the tool
cannot be run against it — there is no bypass flag.

## Reading a correspondence

Every confirmed pair renders as a plain base triple plus an RDF-star
annotation block quoting it:

```turtle
lex:tapi-topology-link skos:exactMatch lex:ietf-network-link .

<<lex:tapi-topology-link skos:exactMatch lex:ietf-network-link>>
    lex:confidenceTier "high" ;
    lex:evidenceQuote "A link represents a physical or logical connection." ;
    lex:lexiconVersion "3f2a9c1..." ;
    lex:model "claude-..." ;
    lex:decidedBy "confirmation-pass" ;
    lex:decidingSignal "definition-text" ;
    lex:labelDefinitionAgreement "true"^^<http://www.w3.org/2001/XMLSchema#boolean> ;
    lex:structuralCorroboration "0.20"^^<http://www.w3.org/2001/XMLSchema#decimal> ;
    lex:validatorRan "true"^^<http://www.w3.org/2001/XMLSchema#boolean> ;
    lex:validatorAgrees "true"^^<http://www.w3.org/2001/XMLSchema#boolean> ;
    lex:validatorCounterArgument "The strongest case against this verdict still fails." ;
    lex:escalated "false"^^<http://www.w3.org/2001/XMLSchema#boolean> .
```

Each predicate, in the fixed order every annotation block emits them:

1. `lex:confidenceTier` — the composed confidence tier (`high`/`medium`/
   `low`) the pair was confirmed at.
2. `lex:evidenceQuote` — the exact quote from the confirmation pass that
   grounded the verdict.
3. `lex:lexiconVersion` — the same commit hash named above, repeated on
   every annotation (not centralized) so a single correspondence triple,
   quoted or pasted in isolation, keeps its version provenance.
4. `lex:model` — the Anthropic model identifier used for this run's calls.
5. `lex:decidedBy` — which pass decided the verdict (`confirmation-pass`
   or `evidence-gate`).
6. `lex:decidingSignal` — the specific signal that carried the decision.
7. `lex:labelDefinitionAgreement` — whether the label pass and the
   definition/example pass agreed.
8. `lex:structuralCorroboration` — a containment-path corroboration score,
   **omitted entirely** when no such signal was available (never rendered
   as `0.0`, which would claim a signal that does not exist).
9. `lex:validatorRan` — always present; states whether the validator
   self-check ran at all.
10. `lex:validatorAgrees` — whether the validator agreed with the proposed
    verdict, **omitted when the validator did not run**.
11. `lex:validatorCounterArgument` — the strongest case the validator built
    against the verdict, **omitted when the validator did not run**.
12. `lex:escalated` — whether the tier was capped because the validator
    disagreed.

Because `lex:validatorRan` is always present, an absent
`lex:validatorAgrees` is never ambiguous — it always means "the validator
did not run for this pair", never "the validator ran and its answer was
lost".

An escalated correspondence (`lex:escalated "true"`) is still a
correspondence: the tier was capped by the validator's disagreement, not
the pair dropped from the artifact. Only `reject` and `insufficient_evidence`
verdicts are excluded from the artifact entirely — they stay in the run's
stdout gap report instead.

## Review verdicts: what a correspondence's presence does not tell you

Phase 5 adds a reviewer's adjudication as a second layer of RDF-star
annotations on the same `<<...>>` block documented above. Four predicates,
appended immediately after the twelve pipeline predicates, in this fixed
order:

| Predicate | Value | Present when |
|---|---|---|
| `lex:reviewVerdict` | one of `"accepted"`, `"rejected"`, `"uncertain"` | the reviewer recorded a verdict for this correspondence |
| `lex:reviewReason` | the reviewer's free-text reason | the reviewer recorded a non-empty reason |
| `lex:reviewRederived` | `"true"`/`"false"`, typed `xsd:boolean` | the correspondence is `high` tier **and** was reviewed — never present on a `medium`/`low` tier row, since only `high`-tier correspondences carry a re-derivation requirement (SC4) |
| `lex:rederivedFrom` | the reviewer's independent source citation, drawn from the source YANG module's own description text | a citation was supplied |

Extending the annotation block already shown above (a real `high`-tier,
`skos:exactMatch` block) with a reviewed, accepted, re-derived verdict —
the twelve pipeline predicates are unchanged, the four review predicates
are new:

```turtle
<<lex:tapi-topology-link skos:exactMatch lex:ietf-network-link>>
    lex:confidenceTier "high" ;
    lex:evidenceQuote "A link represents a physical or logical connection." ;
    lex:lexiconVersion "3f2a9c1..." ;
    lex:model "claude-..." ;
    lex:decidedBy "confirmation-pass" ;
    lex:decidingSignal "definition-text" ;
    lex:labelDefinitionAgreement "true"^^<http://www.w3.org/2001/XMLSchema#boolean> ;
    lex:structuralCorroboration "0.20"^^<http://www.w3.org/2001/XMLSchema#decimal> ;
    lex:validatorRan "true"^^<http://www.w3.org/2001/XMLSchema#boolean> ;
    lex:validatorAgrees "true"^^<http://www.w3.org/2001/XMLSchema#boolean> ;
    lex:validatorCounterArgument "The strongest case against this verdict still fails." ;
    lex:escalated "false"^^<http://www.w3.org/2001/XMLSchema#boolean> ;
    lex:reviewVerdict "accepted" ;
    lex:reviewReason "Both descriptions agree a link is a directed connection between two termination points; cardinality and directionality match on both sides." ;
    lex:reviewRederived "true"^^<http://www.w3.org/2001/XMLSchema#boolean> ;
    lex:rederivedFrom "tapi-topology.yang's own 'link' description, re-read independently of the evidenceQuote above, cross-checked against ietf-network.yang's 'link' description." .
```

**A `skos:exactMatch` or `skos:closeMatch` triple's presence in the base
section does NOT mean the correspondence was accepted.** A rejected or
uncertain correspondence deliberately keeps its base triple (D-16) —
annotated with its verdict, not deleted — because this project's normative
source requires that where confidence can't be established, that MUST be
surfaced rather than hidden (`docs/reference-lexicons.md` §4.1's
surface-don't-force MUST). A rejected pair with its reasoning still
visible is stronger evidence for the working group than a silent deletion,
and it lets a reader audit the matcher's failures as well as its
successes.

An annotation block can be in exactly one of three states, and they are
not interchangeable:

1. **Accepted** — `lex:reviewVerdict "accepted"` is present in the block.
2. **Rejected or uncertain** — `lex:reviewVerdict "rejected"` or
   `lex:reviewVerdict "uncertain"` is present. The base
   `skos:exactMatch`/`skos:closeMatch` triple is still there, by design
   (D-16) — its presence alone never distinguishes this state from state 1.
3. **Not yet reviewed** — no `lex:reviewVerdict` predicate at all appears
   in the block. This is a real, distinct third state, not a synonym for
   either of the above.

A downstream query that filters on triple existence alone, rather than on
`lex:reviewVerdict`'s value, will read a rejection as an endorsement.
Filter on the verdict, never on the bare triple's presence.

## Reviewed gaps

A reviewer's adjudication about a TAPI entry the matcher left without a
confirmed correspondent (a gap) persists as its own resource type — never
as a `skos:exactMatch`/`skos:closeMatch` triple, because a gap is not a
correspondence (D-10; Phase 4 D-02):

```turtle
lex:gap-tapi-topology-node-rule-group a lex:ReviewedGap ;
    lex:gapSubject lex:tapi-topology-node-rule-group ;
    lex:gapReason "insufficient-evidence" ;
    lex:reviewVerdict "accepted" ;
    lex:reviewReason "Source YANG description for this entry is a single restated sentence with no distinguishing detail -- no correspondent could be established from it, and none should be assumed." .
```

Four predicates: `lex:gapSubject` (the TAPI entry the gap is about, always
present), `lex:gapReason` (the matcher's own gap-reason code, always
present), `lex:reviewVerdict` (the reviewer's `accepted`/`rejected`/
`uncertain` adjudication of *the gap itself* — see `REVIEW-PROTOCOL.md`
for what these three words mean on a gap row, which is not the same
meaning they carry on a correspondence row), and `lex:reviewReason`
(present when the reviewer supplied a non-empty reason).

These resources live in the plain-Turtle **base** section — above the
`RDF-star annotations` separator comment, alongside the artifact resource
and the `skos:exactMatch`/`closeMatch` triples — not inside a `<<...>>`
annotation block. That placement matters for loading: unlike the
annotation blocks documented above (which `rdflib` 7.6.0 cannot parse — see
"Loading the file" below), a `lex:ReviewedGap` resource **is** ordinary
Turtle and **does** parse with `rdflib`, along with the rest of the base
section.

See `REVIEW-PROTOCOL.md` for the reviewer-facing half of this workflow:
what the worklist looks like, what the three verdict words mean in each
context, and what the tool enforces versus what only a human reviewer can
guarantee. This file documents the artifact for its *consumer*; that file
documents the process for its *producer*.

## The scope statement

Every artifact carries a machine-readable scope statement on the artifact
resource, not just a Turtle comment:

```turtle
lex:correspondence-artifact a lex:CorrespondenceArtifact ;
    lex:scopeLevel "type-level-only" ;
    lex:lexiconVersion "3f2a9c1..." ;
    lex:model "claude-..." ;
    rdfs:comment "These correspondences assert type-level identity only between TAPI and IETF/TEAS reference-lexicon concepts. They do not license instance co-reference -- matching specific physical nodes, ports, or services across systems is a separate, later problem (D-09; PROJECT.md Out of Scope)." .
```

These are triples, not a header comment, precisely so a SPARQL query can
read the scope programmatically — a downstream consumer querying this file
directly, without ever reading a comment, can still discover that a
correspondence is type-level-only. A correspondence in this file asserts
that two lexicon *concepts* correspond; it never licenses treating two
specific physical nodes, ports, or services as the same instance.

## Loading the file

The base section — the artifact resource and the plain
`skos:exactMatch`/`skos:closeMatch` triples — is ordinary Turtle any
SKOS-aware tool can read. The annotation blocks below the
`RDF-star annotations` separator comment are Turtle* (RDF-star) syntax.

Verified this session: `rdflib` 7.6.0 parses the base section cleanly but
raises `BadSyntax` on a `<<...>>` annotation block, and registers no
`turtle-star` parser plugin. A consumer that only needs the correspondences
themselves should parse the base section alone (everything above the
separator line); a consumer that needs the confidence/evidence signals
needs a Turtle*-capable store (e.g. Stardog, Jena RIOT).

## Reproducibility and the double-run check

Two changes together make a run reproducible given identical inputs:

1. A fixed `temperature=0` on every `client.messages.parse()` call.
2. Deterministic, sorted output order — the writer never depends on
   `rdflib.Graph` iteration order for any section.

Neither alone is sufficient; both are required for the byte-identical
claim.

The writer-level half of that claim (same `PairResult` objects in, same
bytes out) is covered by fast, mocked-client tests that run on every suite
execution with no API spend. The real, end-to-end, LLM-inclusive half —
does the actual pipeline, calling the actual model, produce byte-identical
output across two real runs? — is **not** covered by an automated test,
because each run makes real, billed confirmation and validator calls. It
is verified once, manually, as phase-completion evidence:

```bash
# Record the inputs both runs must share, before running either:
git -C lexicon log -1 --format=%H -- .
# ... note args.model too, e.g. the --model value or its default

python3 align_lexicons.py --emit-correspondences /tmp/run1.ttl
python3 align_lexicons.py --emit-correspondences /tmp/run2.ttl
diff /tmp/run1.ttl /tmp/run2.ttl
```

An empty `diff` output confirms byte-identity. This is a deliberate
one-time check, not a CI step, precisely because of the billed-call cost —
see `.planning/phases/04-correspondence-artifact/04-DETERMINISM-EVIDENCE.md`
for the procedure in full and the section where the result is recorded.
