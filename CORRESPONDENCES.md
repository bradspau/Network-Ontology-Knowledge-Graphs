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
git -C lexicon rev-parse HEAD
# ... note args.model too, e.g. the --model value or its default

python3 align_lexicons.py --emit-correspondences /tmp/run1.ttl
python3 align_lexicons.py --emit-correspondences /tmp/run2.ttl
diff /tmp/run1.ttl /tmp/run2.ttl
```

An empty `diff` output confirms byte-identity. This is a deliberate
one-time check, not a CI step, precisely because of the billed-call cost —
see `.planning/phases/04-correspondence-artifact/04-DETERMINISM-EVIDENCE.md`
for the procedure in full and the section where the result is recorded.
