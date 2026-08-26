#!/usr/bin/env python3
"""
Draft SKOS-style reference lexicon entries from an already-generated
yang4owl.py ontology, per docs/lexicon-overlay-design.md section 2.1
(in the companion lexicon-design project, not this repo).

Bootstraps one lexicon/<module>.lexicon.ttl file per source YANG module,
seeded from the rdfs:label / rdfs:comment already emitted for containers,
lists, identities, groupings, and enumeration-typedef classes. This is a
separate, read-only post-processing pass over yang4owl.py's own TTL
output -- it does not touch yang4owl.py's pipeline or output at all.

Deliberately does NOT fabricate canonical examples or fine-grained entity
classes. Entries whose description is empty, or merely restates the
label, are flagged with lex:needsCuration true. lex:canonicalExample is
always left as a visible, empty placeholder for a human to fill in --
per docs/reference-lexicons.md recommendation #2 ("disambiguate, don't
decorate"), a canonical example is often the single highest-leverage
field and should never be guessed.

Usage:
    python3 yang4owl.py --yang-dir yang-ivy --modules ietf-hardware@2018-03-13.yang \\
        --base-uri http://example.org/ontology --output /tmp/ivy.ttl
    python3 draft_lexicon.py /tmp/ivy.ttl --base-uri http://example.org/ontology --out-dir lexicon
"""
import argparse
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, Namespace, RDF, RDFS
from rdflib.namespace import OWL

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
PROV = Namespace("http://www.w3.org/ns/prov#")

# URI path segments that denote a "kind" other than a plain container/list.
NAMESPACED_KINDS = {"identity", "grouping", "types", "typedef", "rpc", "notification", "module"}

# Kinds excluded from lexicon drafting in this phase (see module docstring):
# typedef/rpc/module have no owl:Class of their own (or aren't domain concepts);
# notification/rpc are operations/events, not entities -- deferred, not dropped.
SKIP_KINDS = {"typedef", "rpc", "notification", "module"}

KIND_ENTITY_CLASS = {
    "identity": "IdentityKind",
    "grouping": "GroupingKind",
    "types": "EnumeratedKind",
    "container-or-list": "StructuralKind",
}

# The exact literal align_lexicons.py's NULL_EVIDENCE_LITERAL (align_lexicons.py:163)
# treats as a null-evidence sentinel -- the value yang4owl.py's comment-capture
# logic writes where the source YANG carried no description at a use-site.
# Both constants must never diverge.
NULL_EVIDENCE_LITERAL = "none"

# Fixed precedence used to resolve a single lex:entityClass when a concept's
# occurrences mix YANG kinds -- a definitional occurrence (identity/grouping)
# is more informative than a purely structural/enumerated one.
ENTITY_CLASS_PRECEDENCE = ("IdentityKind", "GroupingKind", "EnumeratedKind", "StructuralKind")


def classify_uri(uri: str, base_uri: str):
    """Return (kind, module, local_name) for a class URI, or None if it's
    outside base_uri entirely (shouldn't happen for this tool's inputs)."""
    if not uri.startswith(base_uri):
        return None
    rest = uri[len(base_uri):].strip("/")
    if not rest:
        return None
    parts = rest.split("/")
    if parts[0] in NAMESPACED_KINDS:
        kind = parts[0]
        module = parts[1] if len(parts) > 1 else None
        local_name = parts[-1]
    else:
        kind = "container-or-list"
        module = parts[0]
        local_name = parts[-1]
    return kind, module, local_name


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug or "entry"


def is_restatement(label: str, definition: str) -> bool:
    """True if definition is empty or just echoes the label -- i.e. it
    fails the "disambiguate, don't decorate" test from
    docs/reference-lexicons.md recommendation #2."""
    if not definition or not definition.strip():
        return True
    norm_label = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
    norm_def = re.sub(r"[^a-z0-9]+", " ", definition.lower()).strip()
    if norm_def == norm_label:
        return True
    for filler in ("the ", "a ", "represents ", "represents the "):
        if norm_def == (filler + norm_label).strip():
            return True
    return False


def escape_ttl(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def collect_occurrences(graph, base_uri):
    """Group OWL classes into concepts keyed by (module, local_name) -- the
    same tail classify_uri() already computes -- instead of by unique
    containment-path URI. Returns dict[(module, local_name)] -> list of
    occurrence dicts, each carrying uri/label/definition/extra_notes/kind."""
    concepts = defaultdict(list)
    for s in graph.subjects(RDF.type, OWL.Class):
        classification = classify_uri(str(s), base_uri)
        if classification is None:
            continue
        kind, module, local_name = classification
        if kind in SKIP_KINDS or module is None:
            continue

        label = graph.value(s, RDFS.label)
        label = str(label) if label else local_name
        comments = [str(c) for c in graph.objects(s, RDFS.comment)]
        definition = comments[0] if comments else ""
        extra_notes = comments[1:]

        concepts[(module, local_name)].append(
            {
                "uri": str(s),
                "label": label,
                "definition": definition,
                "extra_notes": extra_notes,
                "kind": kind,
            }
        )
    return dict(concepts)


def distinct_texts(values):
    """Sorted list of distinct non-blank texts, excluding any value exactly
    equal to NULL_EVIDENCE_LITERAL."""
    result = set()
    for v in values:
        if not v or not v.strip():
            continue
        if v == NULL_EVIDENCE_LITERAL:
            continue
        result.add(v)
    return sorted(result)


def resolve_entity_class(kinds):
    """Resolve exactly one lex:entityClass value for a concept whose
    occurrences may mix YANG kinds, by fixed precedence
    IdentityKind > GroupingKind > EnumeratedKind > StructuralKind."""
    present = {KIND_ENTITY_CLASS.get(k, "StructuralKind") for k in kinds}
    for candidate in ENTITY_CLASS_PRECEDENCE:
        if candidate in present:
            return candidate
    return "StructuralKind"


def ttl_comment(text: str) -> str:
    """A single-line Turtle comment -- every CR and LF in text is replaced
    with a space so a provenance comment can never break out of its line."""
    return "# " + text.replace("\r", " ").replace("\n", " ")


def render_concept(module, local_name, occurrences):
    """Render one concept's Turtle lines per the plan's render_contract: one
    skos:definition when exactly one distinct real text exists across all
    occurrences, otherwise one skos:scopeNote per distinct real text; every
    occurrence URI on one sorted prov:wasDerivedFrom list; exactly one
    lex:entityClass by fixed precedence; a single-line provenance comment
    naming the sorted-first source URI immediately above each emitted text."""
    slug = slugify(local_name)
    subject = f"lex:{module}-{slug}"

    labels = sorted({occ["label"] for occ in occurrences if occ["label"]})
    pref_label = labels[0] if labels else local_name

    text_sources = defaultdict(list)
    for occ in occurrences:
        d = occ["definition"]
        if d and d.strip() and d != NULL_EVIDENCE_LITERAL:
            text_sources[d].append(occ["uri"])
        for note in occ["extra_notes"]:
            if note and note.strip() and note != NULL_EVIDENCE_LITERAL:
                text_sources[note].append(occ["uri"])

    definition_texts = distinct_texts(occ["definition"] for occ in occurrences)
    note_pool = [note for occ in occurrences for note in occ["extra_notes"]]
    note_texts = [t for t in distinct_texts(note_pool) if t not in definition_texts]

    entity_class = resolve_entity_class(occ["kind"] for occ in occurrences)

    texts_union = definition_texts + note_texts
    needs_curation = (not texts_union) or all(is_restatement(pref_label, t) for t in texts_union)

    uris = sorted({occ["uri"] for occ in occurrences})

    def source_comment(text):
        source_uri = sorted(text_sources[text])[0]
        return "    " + ttl_comment(f"source: {source_uri}")

    lines = [
        subject,
        "    a lex:ReferenceEntry ;",
        f'    skos:prefLabel "{escape_ttl(pref_label)}" ;',
    ]

    if len(definition_texts) == 1:
        text = definition_texts[0]
        lines.append(source_comment(text))
        lines.append(f'    skos:definition "{escape_ttl(text)}" ;')
        for note in note_texts:
            lines.append(source_comment(note))
            lines.append(f'    skos:scopeNote "{escape_ttl(note)}" ;')
    else:
        for text in sorted(definition_texts + note_texts):
            lines.append(source_comment(text))
            lines.append(f'    skos:scopeNote "{escape_ttl(text)}" ;')

    lines.append('    lex:canonicalExample "" ;')
    lines.append(f"    lex:entityClass lex:{entity_class} ;")
    if needs_curation:
        lines.append("    lex:needsCuration true ;")

    if len(uris) == 1:
        lines.append(f"    prov:wasDerivedFrom <{uris[0]}> .")
    else:
        joined = " ,\n                        ".join(f"<{u}>" for u in uris)
        lines.append(f"    prov:wasDerivedFrom {joined} .")

    lines.append("")
    return lines


def _unescape_ttl(s: str) -> str:
    """Exact inverse of escape_ttl(): \\\\ -> \\, \\" -> ", \\n -> a real
    newline. Safe because escape_ttl() doubles backslashes before adding
    any new backslash-escapes, so every two-character `\\\\` in escaped
    text always represents exactly one original backslash, never the start
    of an ambiguous longer escape."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
            if nxt == '"':
                out.append('"')
                i += 2
                continue
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


# Matches render_concept()'s own "# source: <uri>" provenance comment
# immediately followed by the skos:definition/scopeNote line it documents --
# the exact shape render_concept() writes, never anything hand-authored
# (T-03-03: a corrupt/hand-edited file is out of this tool's trust boundary).
_SOURCE_TEXT_RE = re.compile(
    r'#\s*source:\s*(?P<uri>\S+)\s*\n'
    r'\s*skos:(?:definition|scopeNote)\s+"(?P<text>(?:[^"\\]|\\.)*)"'
)


def merge_existing_lexicon(path, base_uri):
    """Parse an existing lexicon/<module>.lexicon.ttl back into the same
    dict[(module, local_name)] -> list[occurrence] shape collect_occurrences()
    produces, so it can be unioned with a fresh run's freshly-collected
    concepts before rendering (LEX-01).

    render_concept()'s "# source: <uri>" comment is a Turtle comment, not a
    triple, so rdflib's own parse discards which occurrence contributed
    which text. That per-occurrence attribution is recovered with a
    lightweight regex scan of the file's own raw text instead (safe here
    because this function only ever reads a file this same tool wrote --
    see the docstring's trust-boundary note): for each occurrence URI, the
    lexicographically smallest text it was ever named "# source:" for
    becomes its reconstructed `definition` (comments[0]) and the rest
    become its `extra_notes` (comments[1:]) -- an occurrence never named as
    a source for anything (e.g. a pure text-duplicate of another, smaller,
    occurrence URI) reconstructs as blank. This choice does not need to
    exactly recover history; it only needs to be deterministic and to give
    every occurrence URI credit for exactly the text(s) it actually
    contributed, so that re-deriving definition_texts from the reconstructed
    occurrences reproduces the same distinct-count the original render used
    -- which is what makes a sequential multi-invocation merge byte-identical
    to a single combined invocation."""
    raw = path.read_text(encoding="utf-8")
    source_to_texts = defaultdict(list)
    for m in _SOURCE_TEXT_RE.finditer(raw):
        source_to_texts[m.group("uri")].append(_unescape_ttl(m.group("text")))

    graph = Graph()
    graph.parse(str(path), format="turtle")
    lex_ns = Namespace(f"{base_uri}/lexicon-vocab#")

    concepts = defaultdict(list)
    for subject in graph.subjects(RDF.type, lex_ns.ReferenceEntry):
        uris = sorted(str(u) for u in graph.objects(subject, PROV.wasDerivedFrom))
        if not uris:
            continue

        label = graph.value(subject, SKOS.prefLabel)
        label = str(label) if label is not None else ""

        module = local_name = None
        kinds_by_uri = {}
        for uri in uris:
            classification = classify_uri(uri, base_uri)
            if classification is None:
                continue
            kind, m, ln = classification
            kinds_by_uri[uri] = kind
            module, local_name = m, ln
        if module is None or local_name is None:
            continue

        occurrences = []
        for uri in uris:
            attributed = sorted(source_to_texts.get(uri, []))
            definition_text = attributed[0] if attributed else ""
            extra = attributed[1:]
            occurrences.append(
                {
                    "uri": uri,
                    "label": label,
                    "definition": definition_text,
                    "extra_notes": extra,
                    "kind": kinds_by_uri.get(uri, "container-or-list"),
                }
            )
        concepts[(module, local_name)].extend(occurrences)

    return dict(concepts)


def write_atomic(path: Path, content: str) -> None:
    """Write content to path via a sibling temp file + os.replace(), so an
    interrupted or failing run leaves the file at path unchanged rather
    than truncated (T-03-04). No try/except: if the write itself fails, the
    temp file may be left behind for inspection but path is never touched
    until os.replace() succeeds -- matching this tool's existing fail-loud
    convention (no try/except anywhere else in this file)."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_name, str(path))


def main():
    ap = argparse.ArgumentParser(
        description="Draft reference-lexicon entries from yang4owl.py ontology output.",
    )
    ap.add_argument("ontology_files", nargs="+", help="One or more generated ontology .ttl files")
    ap.add_argument("--base-uri", required=True, help="Base URI used when generating the ontology files")
    ap.add_argument("--out-dir", default="lexicon", help="Directory to write <module>.lexicon.ttl files into")
    args = ap.parse_args()

    base_uri = args.base_uri.rstrip("/")

    g = Graph()
    for f in args.ontology_files:
        g.parse(f, format="turtle")

    lex_ns = Namespace(f"{base_uri}/lexicon-vocab#")

    concepts = collect_occurrences(g, base_uri)

    by_module = defaultdict(dict)
    for (module, local_name), occurrences in concepts.items():
        by_module[module][local_name] = occurrences

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_entries = 0
    total_flagged = 0

    for module, local_concepts in sorted(by_module.items()):
        out_path = out_dir / f"{module}.lexicon.ttl"
        lines = [
            f"@prefix lex: <{lex_ns}> .",
            "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
            "@prefix prov: <http://www.w3.org/ns/prov#> .",
            "",
            f"# Reference lexicon for module: {module}",
            "# Auto-drafted by draft_lexicon.py from yang4owl.py output. Do not hand-edit",
            "# without removing lex:needsCuration once an entry has been reviewed --",
            "# it exists so curated and uncurated entries stay distinguishable.",
            "#",
            "# Entries flagged lex:needsCuration true have an empty or label-restating",
            "# definition and/or a placeholder canonical example -- see",
            "# docs/reference-lexicons.md recommendation #2 before relying on them.",
            "",
        ]
        # LEX-01: merge-aware by default -- read any existing output file
        # back into the same shape and union it into this run's freshly
        # collected concepts before rendering, so a single-source-tree run
        # never clobbers another source tree's contributions to a
        # shared-module file. Unconditional; no --force/--overwrite flag
        # (D-07).
        existing_concepts = {}
        if out_path.exists():
            for (m, ln), occurrences in merge_existing_lexicon(out_path, base_uri).items():
                if m == module:
                    existing_concepts[ln] = occurrences

        merged_concepts = {}
        for local_name in set(local_concepts) | set(existing_concepts):
            fresh_by_uri = {occ["uri"]: occ for occ in local_concepts.get(local_name, [])}
            existing_by_uri = {occ["uri"]: occ for occ in existing_concepts.get(local_name, [])}

            merged_occs = []
            for uri in sorted(set(fresh_by_uri) | set(existing_by_uri)):
                fresh_occ = fresh_by_uri.get(uri)
                existing_occ = existing_by_uri.get(uri)
                if fresh_occ is not None and existing_occ is not None:
                    # Present in both this run's fresh data and the prior
                    # file -- union each side's texts for this exact URI
                    # rather than letting one side silently shadow the
                    # other. Different source trees can carry a
                    # differently-complete rdfs:comment set for the
                    # identical occurrence URI (e.g. a tree without a TE
                    # augmentation sees fewer rdfs:comment values on the
                    # same base container than a tree that includes it),
                    # and D-08's never-silently-drop-real-content floor
                    # applies at the text level, not just the
                    # occurrence-URI level.
                    #
                    # The already-committed file's role split (which text
                    # was comments[0] vs comments[1:]) is always kept as the
                    # base, never overridden by this run's fresh view: this
                    # mirrors how parsing several ontology files sharing one
                    # class URI into a single combined graph behaves
                    # (rdflib unions rdfs:comment triples in first-parsed
                    # order, so whichever source was seen first keeps the
                    # comments[0] slot) -- keeping "whatever the file
                    # already recorded" authoritative for role, and only
                    # adding a fresh run's genuinely new texts as
                    # additional extra_notes, is what makes a sequential
                    # multi-invocation merge byte-identical to a single
                    # combined invocation regardless of which tree is
                    # processed first.
                    final_definition = existing_occ["definition"] or fresh_occ["definition"]
                    combined_extra = list(existing_occ["extra_notes"])
                    known = set(combined_extra)
                    if final_definition:
                        known.add(final_definition)
                    for t in [fresh_occ["definition"]] + list(fresh_occ["extra_notes"]):
                        if t and t not in known:
                            combined_extra.append(t)
                            known.add(t)

                    merged_occs.append(
                        {
                            "uri": uri,
                            "label": fresh_occ["label"] or existing_occ["label"],
                            "definition": final_definition,
                            "extra_notes": combined_extra,
                            "kind": fresh_occ["kind"],
                        }
                    )
                elif fresh_occ is not None:
                    merged_occs.append(fresh_occ)
                else:
                    # D-08: never silently delete an occurrence the current
                    # run's inputs cannot regenerate -- retain it and
                    # surface it on stdout instead.
                    print(
                        f"STALE: {uri} retained in {module}.lexicon.ttl "
                        f"(no current input regenerates it)"
                    )
                    merged_occs.append(existing_occ)

            merged_concepts[local_name] = merged_occs

        # Tie-break on local_name itself, not just its slug: two distinct
        # local_names can slugify identically (e.g. an IdentityKind written
        # in YANG's own UPPER_SNAKE_CASE convention and a GroupingKind
        # written in lower-kebab-case both collapse to the same slug via
        # slugify()'s case-folding). Sorting by slug alone leaves that tie
        # order undefined, which the caller's non-deterministic module/set
        # iteration order would then leak into the byte-stability contract.
        seen_subjects = set()
        for local_name in sorted(merged_concepts.keys(), key=lambda ln: (slugify(ln), ln)):
            occurrences = merged_concepts[local_name]
            concept_lines = render_concept(module, local_name, occurrences)
            subject = concept_lines[0]
            if subject in seen_subjects:
                # Rare: two distinct local_names slugify to the identical
                # string (e.g. YANG's own UPPER_SNAKE_CASE identity-naming
                # convention and a lower-kebab-case grouping/container name
                # collapse under slugify()'s case-folding). Grouping itself
                # stays exact on (module, local_name) -- only the rendered
                # subject slug needs a deterministic disambiguating suffix,
                # so two genuinely different concepts never share one
                # lex:ReferenceEntry subject (D-06).
                suffix = 2
                candidate = f"{subject}-{suffix}"
                while candidate in seen_subjects:
                    suffix += 1
                    candidate = f"{subject}-{suffix}"
                subject = candidate
                concept_lines[0] = subject
            seen_subjects.add(subject)
            lines.extend(concept_lines)
            total_entries += 1
            if "    lex:needsCuration true ;" in concept_lines:
                total_flagged += 1
        write_atomic(out_path, "\n".join(lines))
        print(f"Wrote {len(merged_concepts)} entries to {out_path}")

    print(f"\nTotal: {total_entries} entries across {len(by_module)} module(s); {total_flagged} flagged lex:needsCuration")


if __name__ == "__main__":
    main()
