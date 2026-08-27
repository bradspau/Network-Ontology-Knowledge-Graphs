# Review Protocol

## Who this is for

`CORRESPONDENCES.md` documents the artifact for its *consumer* — what a
reviewed correspondence looks like once it is committed. This file
documents the process for its *producer*: the reviewer walking a worklist,
column by column, deciding what to type where.

## The two-pass workflow

Reviewing is two passes over two commands, never a live interactive tool
(D-05):

1. **Generate the worklist.**
   ```bash
   python3 align_lexicons.py --full-corpus --emit-correspondences --emit-worklist
   ```
   `--emit-worklist` writes a Markdown table (`review-worklist.md` next to
   `align_lexicons.py`, by default) with one row per confirmed
   correspondence and one row per gap. `--emit-correspondences` writes the
   `correspondences.ttl` the worklist's row identifiers point back into —
   generate both from the same run, together, since the worklist and the
   artifact must describe the same run to be applied to each other later.

2. **Edit the worklist directly.** Open it in any Markdown-table-capable
   editor. Only four columns are yours to fill: `verdict`, `reason`,
   `re_derived`, `rederivation_citation`. Every other column is
   generator-written and display-only — read it, but do not edit it. An
   edit to a display-only column has no effect: the parser below never
   reads it for anything except display; it recovers a row's real
   identity (which TAPI entry, which IETF entry, which predicate) from the
   `row_id` column's own encoding, never from the columns a human might
   retype.

3. **Read the completed worklist back.**
   ```bash
   python3 align_lexicons.py --apply-review review-worklist.md --correspondences-path correspondences.ttl
   ```
   `--apply-review` parses the file you edited, validates every row, and —
   if the whole file is clean — splices your verdicts into
   `correspondences.ttl` in place. It makes no Anthropic API call and
   constructs no client; this pass costs nothing and can be re-run freely
   once the worklist is fixed.

## Worklist column reference

The sixteen columns, in the order they appear, exactly as the tool
generates them:

| Column | Who writes it | What it holds |
|---|---|---|
| `row_id` | generator | The row's real identity, colon-encoded. Never hand-edit — the parser trusts only this column for identity, never the display columns below it. |
| `kind` | generator | `correspondence` or `gap`. |
| `tier` | generator | `high` / `medium` / `low` for a correspondence row; `-` for a gap row. |
| `escalated` | generator | `Y` if the validator disagreed with the confirmed verdict, `N` otherwise; `-` for a gap row. |
| `gap_reason` | generator | One of the four gap-reason codes for a gap row; `-` for a correspondence row. |
| `evidence_strength` | generator | 0-3, how many corroborating signals support the row. |
| `tapi_lex_id` | generator | The TAPI-side lexicon entry id. |
| `tapi_label` | generator | The TAPI-side `skos:prefLabel`. |
| `ietf_lex_id` | generator | The IETF-side lexicon entry id; `-` for a gap row. |
| `ietf_label` | generator | The IETF-side `skos:prefLabel`; `-` for a gap row. |
| `predicate` | generator | `skos:exactMatch` or `skos:closeMatch`; `-` for a gap row. |
| `evidence_quote` | generator | The matcher's own quoted evidence (correspondence row) or a label/structural-score summary (gap row). Read this to understand the matcher's reasoning — never copy it into `rederivation_citation`. |
| `verdict` | **you** | Blank means unreviewed. One of `accept`, `reject`, `uncertain` (case-insensitive). Never defaulted or inferred from a blank cell. |
| `reason` | **you** | Free text explaining your verdict. |
| `re_derived` | **you**, `high`-tier correspondence rows only | `N` to start; set to `Y` only after you have independently re-derived the correspondence yourself. `-` on every other row, and not editable there. |
| `rederivation_citation` | **you**, `high`-tier correspondence rows only | Your independent source citation. `-` on every other row, and not editable there. |

## What the three verdict words mean

The same three words — `accept`, `reject`, `uncertain` — appear in the
`verdict` column on both a correspondence row and a gap row, but they mean
different things depending on which kind of row you are looking at. Read
the `kind` column first.

### On a correspondence row

- **`accept`** — you agree the TAPI entry and the IETF entry correspond,
  as proposed. The base `skos:exactMatch`/`skos:closeMatch` triple stays
  in `correspondences.ttl`, now annotated `lex:reviewVerdict "accepted"`.
- **`reject`** — you disagree; the two entries do not correspond, despite
  what the matcher proposed. The base triple **still stays** in
  `correspondences.ttl` (D-16) — annotated `lex:reviewVerdict "rejected"`
  rather than deleted, so the matcher's mistake remains visible and
  auditable rather than silently erased.
- **`uncertain`** — you cannot settle it either way from the evidence
  available. The base triple stays, annotated `lex:reviewVerdict
  "uncertain"`. Treated identically to `reject` for the keep-the-triple
  rule — the only difference from `reject` is the word itself, recording
  that you did not reach a negative conclusion, just an unresolved one.

### On a gap row

A gap row has no base triple to keep or annotate — a gap means the
matcher found *no* confirmed correspondent for a TAPI entry. Here the
three words judge the *gap itself*, not a proposed pairing:

- **`accept`** — you agree the gap is genuine: no IETF/TEAS correspondent
  exists for this TAPI entry, and the matcher was right not to force one.
- **`reject`** — you believe a correspondent *does* exist and the matcher
  missed it. This is not itself a mechanism for adding the missed
  correspondence to `correspondences.ttl` — recording `reject` here is a
  finding for a human to act on (add a new candidate pairing to a future
  run, or self-extend per `docs/reference-lexicons.md` §4.4), not an
  automatic correction.
- **`uncertain`** — unsettled; you cannot yet say whether the gap is
  genuine or a miss.

Your verdict, plus an optional `reason`, persists as a `lex:ReviewedGap`
resource in `correspondences.ttl`'s plain-Turtle base section — see
`CORRESPONDENCES.md`'s "Reviewed gaps" section for the exact shape.

## Cell escaping

A `reason` or `rederivation_citation` cell you type must stay inside one
Markdown table cell — a literal pipe character would be read as a new
column boundary, and a literal newline would be read as the end of the
row. Two escape sequences, applied when you type the cell:

- A literal pipe `|` → `&#124;`
- A literal newline → `<br>`

Worked example. Suppose your reason is, verbatim, across two lines:

```
Definition agrees | but the canonical example diverges
needs a second look before accepting
```

Type it into the `reason` cell as:

```
Definition agrees &#124; but the canonical example diverges<br>needs a second look before accepting
```

The tool applies the inverse substitution when it reads the cell back —
`&#124;` becomes `|` again and `<br>` becomes a real newline — so what you
see after `--apply-review` splices your reason into `correspondences.ttl`
is your original two-line text with the real pipe character restored, not
the escaped form.

## The high-tier re-derivation requirement

Every `high`-tier correspondence row requires an independent re-derivation
from the source YANG text before you may accept it — this is not optional
and not a formality. `high` is the tier most likely to be rubber-stamped
on trust precisely because the matcher itself was most confident; that is
exactly the risk this requirement exists to close (SC4, D-13).

**Where a citation must come from:** the source YANG module's own
description text — the same kind of text the matcher itself read, but
read again, independently, by you. Open the actual `.yang` file (or the
lexicon entry drafted from it) and quote what it actually says.

**Copying the matcher's own evidence quote is refused.** The
`evidence_quote` column shows you the matcher's reasoning so you can
evaluate whether it was sound — it is not a citation you may reuse.
`write_reviewed_correspondences()`'s distinctness check compares your
`rederivation_citation`, stripped, against the artifact's own recorded
`lex:evidenceQuote` for that exact correspondence; a byte-identical match
is refused, at write time, with every offending row named in one error.
Restating the matcher's own words proves nothing was independently
re-derived.

To accept a `high`-tier row: set `re_derived` to `Y` and
`rederivation_citation` to your own independent quote. Both are required
together — one without the other is refused (see below).

## What the tool enforces mechanically

The parser and the writer check all of the following before anything is
written, and refuse the whole operation — never a partial write — if any
check fails:

1. **The permitted verdict words** — `verdict` must be blank or exactly
   one of `accept`, `reject`, `uncertain` (case-insensitive); anything
   else is refused.
2. **Unique row identifiers** — a `row_id` that appears twice in the
   worklist is refused.
3. **The column count** — every data row must have exactly sixteen cells;
   a row with the wrong count is refused.
4. **The high-tier acceptance gate** — accepting a `high`-tier
   correspondence row without `re_derived = Y` *and* a non-empty
   `rederivation_citation` is refused (SC4).
5. **The citation-distinctness check** — a `rederivation_citation`
   byte-identical (after stripping) to the matcher's own recorded
   `lex:evidenceQuote` is refused.
6. **Whole-file refusal on any defect** — every defect in the file is
   collected and reported together in one error; nothing is written if
   even one row is malformed.
7. **Refusal to apply a worklist twice** — if the target
   `correspondences.ttl` already carries a review verdict for a row (or
   already contains a `lex:ReviewedGap` resource, for a gap-row
   application), the whole pass is refused rather than silently
   re-applying or overwriting.
8. **Refusal to apply a worklist to a different run's artifact** — the
   worklist's own recorded `lexicon_version`/`model` (from its header
   block) must match the target artifact's own recorded values; a
   mismatch is refused before any annotation block is touched.
9. **A row naming a correspondence absent from the target file** — a
   `row_id` the target `correspondences.ttl` has no matching block for is
   refused, not silently skipped.

## What no tool can check

Exactly one thing, and it is the one thing that matters most: **whether a
citation is a truthful quotation of what the source YANG text actually
says.** The tool can require that a citation exists, that it is non-empty,
and that it is not merely a copy of the matcher's own evidence — none of
that proves the citation is *accurate*. That is entirely the reviewer's
own guarantee. The mechanical gate makes an absent or lazy citation
impossible to hide; it cannot make a fabricated one impossible to write.
This is precisely where the tool's guarantee ends and your judgement
begins — do not mistake passing the mechanical gate for having verified
the correspondence.

## When the application pass refuses

`--apply-review` writes nothing on any defect — a refusal is always safe,
never a partial application. When it refuses:

1. It reports every defect it found, together, in one error — not just
   the first one it encountered.
2. Fix every reported defect in the worklist file.
3. Re-run `--apply-review` with the same command. Since nothing was
   written on the failed attempt, this is a plain retry, not a repair of
   a half-applied state.

## A note on volume

The real corpus this worklist walks is not small: 1,777 TAPI entries and
558 IETF/TEAS entries across 34 lexicon files
(`yang4owl/lexicon/CURATION-AUDIT.md`), of which 611 TAPI entries (34%)
and 14 IETF entries (2.5%) are already flagged `thin_evidence` in that
audit — source text too sparse to strongly ground a match either way.
Expect the low-tier and gap rows to be where your attention matters most:
a `high`-tier row already cleared two confirmation signals and a
validator self-check before it reached you, while a `low`-tier row or a
gap is exactly where the matcher itself was least sure. This is why the
worklist orders gaps first, then low tier before high tier, and ranks
`insufficient-evidence` gaps last among gaps (D-09, D-12) — the rows most
likely to reward careful review are the rows you reach first.
