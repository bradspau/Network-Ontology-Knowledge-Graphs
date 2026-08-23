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
import re
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, Namespace, RDF, RDFS
from rdflib.namespace import OWL

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

    by_module = defaultdict(list)
    seen_slugs = defaultdict(set)

    for s in g.subjects(RDF.type, OWL.Class):
        classification = classify_uri(str(s), base_uri)
        if classification is None:
            continue
        kind, module, local_name = classification
        if kind in SKIP_KINDS or module is None:
            continue

        label = g.value(s, RDFS.label)
        label = str(label) if label else local_name
        comments = [str(c) for c in g.objects(s, RDFS.comment)]
        definition = comments[0] if comments else ""
        extra_notes = comments[1:]

        needs_curation = is_restatement(label, definition)

        slug = slugify(local_name)
        if slug in seen_slugs[module]:
            uri_tail = str(s).rstrip("/").split("/")
            parent_hint = uri_tail[-2] if len(uri_tail) > 1 else kind
            slug = f"{slug}-{slugify(parent_hint)}"
        seen_slugs[module].add(slug)

        by_module[module].append(
            {
                "uri": str(s),
                "slug": slug,
                "label": label,
                "definition": definition,
                "extra_notes": extra_notes,
                "entity_class": KIND_ENTITY_CLASS.get(kind, "StructuralKind"),
                "needs_curation": needs_curation,
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_entries = 0
    total_flagged = 0

    for module, entries in sorted(by_module.items()):
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
        for e in sorted(entries, key=lambda x: x["slug"]):
            eid = f"lex:{module}-{e['slug']}"
            def_text = e["definition"] if e["definition"].strip() else "TODO: no YANG description was available to seed this definition."
            lines.append(eid)
            lines.append("    a lex:ReferenceEntry ;")
            lines.append(f'    skos:prefLabel "{escape_ttl(e["label"])}" ;')
            lines.append(f'    skos:definition "{escape_ttl(def_text)}" ;')
            for note in e["extra_notes"]:
                lines.append(f'    skos:scopeNote "{escape_ttl(note)}" ;')
            lines.append('    lex:canonicalExample "" ;')
            lines.append(f"    lex:entityClass lex:{e['entity_class']} ;")
            if e["needs_curation"]:
                lines.append("    lex:needsCuration true ;")
            lines.append(f'    prov:wasDerivedFrom <{e["uri"]}> .')
            lines.append("")
            total_entries += 1
            if e["needs_curation"]:
                total_flagged += 1
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {len(entries)} entries to {out_path}")

    print(f"\nTotal: {total_entries} entries across {len(by_module)} module(s); {total_flagged} flagged lex:needsCuration")


if __name__ == "__main__":
    main()
