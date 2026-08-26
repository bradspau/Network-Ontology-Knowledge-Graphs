#!/usr/bin/env python3
"""
Audit yang4owl/lexicon/*.lexicon.ttl for how much of each side's corpus
carries real evidence versus placeholder evidence (ROADMAP SC4, D-09,
lexicon-69m.13).

Standalone, rerunnable, and strictly READ-ONLY over every *.lexicon.ttl
file it audits: it parses each one with rdflib and writes exactly one
path -- the --out target. It never modifies a .lexicon.ttl file (T-03-07).

Why this exists: the existing lex:needsCuration signal drastically
under-counts thin evidence on the TAPI side, because is_restatement()
(imported from draft_lexicon.py, not reimplemented) only flags an empty or
label-echoing definition -- the literal placeholder string "none" is
neither, so it silently passes as "real content" (03-RESEARCH.md Pitfall
1: 1 flagged vs. 1,057 "none"-valued entries in one file alone). This tool
treats a missing skos:definition and a skos:definition equal to the
literal "none" as two distinct placeholder categories, both counted
independently of lex:needsCuration, so the report reads correctly against
both the pre-repair corpus shape ("none" definitions) and the post-repair
shape plan 03-03 produces (an omitted skos:definition triple).

Usage:
    python3 audit_lexicon_curation.py --lexicon-dir lexicon --out lexicon/CURATION-AUDIT.md
    python3 audit_lexicon_curation.py --lexicon-dir lexicon --out -
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, Namespace, RDF

from draft_lexicon import is_restatement

LEX = Namespace("http://example.org/ontology/lexicon-vocab#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
PROV = Namespace("http://www.w3.org/ns/prov#")

# The exact literal align_lexicons.py's NULL_EVIDENCE_LITERAL
# (align_lexicons.py:163) and draft_lexicon.py's constant of the same name
# treat as a null-evidence sentinel -- all three copies must stay equal.
NULL_EVIDENCE_LITERAL = "none"

# Column order matches this plan's <report_contract> table exactly, and is
# the single source of truth for both the per-side and per-file table
# shapes render_report() emits.
COLUMNS = (
    "entries",
    "occurrences",
    "definition_absent",
    "definition_null",
    "definition_restates",
    "scope_notes",
    "entries_with_scope_notes",
    "needs_curation",
    "canonical_example_present",
    "thin_evidence",
)


def side_for(filename: str) -> str:
    """Side-assignment rule (report_contract): a file whose name begins
    'tapi-' is the TAPI side; every other *.lexicon.ttl file -- the
    ietf-* modules plus simap-yang and iana-hardware -- is the IETF side."""
    return "tapi" if filename.startswith("tapi-") else "ietf"


def audit_entry(graph, subject) -> dict:
    """Classify one lex:ReferenceEntry subject per report_contract's
    column table. thin_evidence is the escalation-volume number: no real
    definition (absent, the "none" sentinel, or a label restatement), no
    skos:scopeNote, and no non-blank lex:canonicalExample."""
    pref_label = graph.value(subject, SKOS.prefLabel)
    pref_label = str(pref_label) if pref_label is not None else ""

    definition = graph.value(subject, SKOS.definition)
    definition_absent = definition is None
    definition_text = str(definition) if definition is not None else ""
    definition_null = (not definition_absent) and definition_text == NULL_EVIDENCE_LITERAL
    definition_restates = (
        not definition_absent
        and not definition_null
        and is_restatement(pref_label, definition_text)
    )

    scope_note_count = len(list(graph.objects(subject, SKOS.scopeNote)))

    raw_needs_curation = graph.value(subject, LEX.needsCuration)
    # T-03-09/WR-01 (align_lexicons.py:586-595): an rdflib Literal is a str
    # subclass with no __bool__ override for typed literals, so
    # bool(Literal("false", datatype=XSD.boolean)) is True. .toPython()
    # converts a typed xsd:boolean literal to a real Python bool first, so
    # an explicit lex:needsCuration false is honored rather than coerced.
    needs_curation = bool(raw_needs_curation.toPython()) if raw_needs_curation is not None else False

    canonical = graph.value(subject, LEX.canonicalExample)
    canonical_present = canonical is not None and str(canonical).strip() != ""

    no_real_definition = definition_absent or definition_null or definition_restates
    thin_evidence = no_real_definition and scope_note_count == 0 and not canonical_present

    return {
        "entries": 1,
        "occurrences": len(list(graph.objects(subject, PROV.wasDerivedFrom))),
        "definition_absent": int(definition_absent),
        "definition_null": int(definition_null),
        "definition_restates": int(definition_restates),
        "scope_notes": scope_note_count,
        "entries_with_scope_notes": int(scope_note_count > 0),
        "needs_curation": int(needs_curation),
        "canonical_example_present": int(canonical_present),
        "thin_evidence": int(thin_evidence),
    }


def audit_file(path: Path) -> dict:
    """Parse one lexicon/<module>.lexicon.ttl file read-only and return
    aggregated per-file counts across every lex:ReferenceEntry subject, per
    report_contract's column set. Never writes to path."""
    graph = Graph()
    graph.parse(str(path), format="turtle")
    totals = {c: 0 for c in COLUMNS}
    for subject in graph.subjects(RDF.type, LEX.ReferenceEntry):
        stats = audit_entry(graph, subject)
        for c in COLUMNS:
            totals[c] += stats[c]
    return totals


def _table_row(label: str, stats: dict, extra_cells=()) -> str:
    cells = [label, *extra_cells] + [str(stats.get(c, 0)) for c in COLUMNS]
    return "| " + " | ".join(cells) + " |"


def render_report(by_side: dict, by_file: dict) -> str:
    """Render the Markdown report per report_contract: a per-side summary
    table, a per-file table sorted by filename, and a "How to read this"
    section stating the side-assignment rule, the canonicalExample
    0%-by-design baseline, and thin_evidence (not needs_curation) as the
    escalation-volume number Phase 5 should use."""
    lines = [
        "# Lexicon Curation Audit",
        "",
        f"{len(by_file)} `*.lexicon.ttl` file(s) audited.",
        "",
        "**Side assignment rule:** a file whose name begins `tapi-` is the "
        "TAPI side; every other `*.lexicon.ttl` file (the `ietf-*` modules "
        "plus `simap-yang` and `iana-hardware`) is the IETF side.",
        "",
        "## Per-side summary",
        "",
        "| side | " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * (len(COLUMNS) + 1),
    ]
    for side in sorted(by_side):
        lines.append(_table_row(side, by_side[side]))

    lines += [
        "",
        "## Per-file detail",
        "",
        "| file | side | " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * (len(COLUMNS) + 2),
    ]
    for fname in sorted(by_file):
        lines.append(_table_row(fname, by_file[fname], extra_cells=(side_for(fname),)))

    lines += [
        "",
        "## How to read this",
        "",
        "- `canonical_example_present` reads 0 across the whole corpus by "
        "design, not as a defect: `draft_lexicon.py` never fabricates a "
        "canonical example (`docs/reference-lexicons.md` recommendation "
        "#2, \"disambiguate, don't decorate\") -- canonical-example "
        "curation is a deferred v2 concern (REQUIREMENTS.md PROV-02).",
        "- `needs_curation` alone understates thin evidence (verified: 1 "
        "flagged vs. 1,057 `\"none\"`-valued entries in one file alone, "
        "lexicon-69m.13) -- use `thin_evidence` as the escalation-volume "
        "number for Phase 5 planning instead.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit lexicon/*.lexicon.ttl for per-side real-vs-placeholder evidence counts (SC4, D-09).",
    )
    ap.add_argument("--lexicon-dir", default="lexicon", help="Directory containing *.lexicon.ttl files")
    ap.add_argument(
        "--out",
        default="lexicon/CURATION-AUDIT.md",
        help="Report output path, or '-' for stdout",
    )
    args = ap.parse_args()

    lexicon_dir = Path(args.lexicon_dir)

    by_file = {}
    by_side = defaultdict(lambda: {c: 0 for c in COLUMNS})

    for ttl_path in sorted(lexicon_dir.glob("*.lexicon.ttl")):
        stats = audit_file(ttl_path)
        by_file[ttl_path.name] = stats
        side = side_for(ttl_path.name)
        for c in COLUMNS:
            by_side[side][c] += stats[c]
        # Progress goes to stderr, not stdout -- so `--out -` emits a clean
        # Markdown report on stdout with nothing else interleaved.
        print(f"Audited {ttl_path.name}: {stats['entries']} entries", file=sys.stderr)

    report = render_report(dict(by_side), by_file)

    if args.out == "-":
        print(report)
    else:
        out_path = Path(args.out)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote report to {out_path}", file=sys.stderr)

    print(f"Total: {len(by_file)} file(s) audited", file=sys.stderr)


if __name__ == "__main__":
    main()
