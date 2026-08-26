"""
Unit tests for draft_lexicon.py's concept-level grouping (LEX-02), merge-aware
write path (LEX-01), and the determinism/edge-case contract pinned in
.planning/phases/03-lexicon-quality-repair/03-01-PLAN.md's <render_contract>.

Ontology fixtures are minimal synthetic Turtle (owl:Class + rdfs:label +
rdfs:comment), built with real URIs and real description text copied
verbatim from yang4owl/lexicon/tapi-common.lexicon.ttl's access-port and
node-edge-point families, and from yang4owl/lexicon/ietf-network.lexicon.ttl's
node/te-node-augmentation pair -- draft_lexicon.py consumes ontology output,
not lexicon output, so a synthetic ontology fixture carrying real content is
required (there is no "real ontology .ttl" checked into the repo to read
directly). Written .lexicon.ttl output is parsed back with rdflib.Graph()
and asserted on directly, in the style of tests/test_align_lexicons.py.

Mirrors tests/conftest.py's conventions: relies on conftest's sys.path
insertion for import resolution and lazily imports draft_lexicon inside each
test body.
"""
import sys

import pytest
from rdflib import RDF, Graph, Namespace

BASE_URI = "http://example.org/ontology"
LEX = Namespace(f"{BASE_URI}/lexicon-vocab#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
PROV = Namespace("http://www.w3.org/ns/prov#")

# ---------------------------------------------------------------------------
# Real corpus fixtures -- verbatim URIs and description text copied from
# yang4owl/lexicon/tapi-common.lexicon.ttl (access-port: lines 46-124,
# node-edge-point: lines 12161-12360) and yang4owl/lexicon/ietf-network.lexicon.ttl
# (node: line 921-929, node-statistics/te augmentation: line 940-946), read
# directly this session.
# ---------------------------------------------------------------------------

# Seven real distinct occurrence URIs sharing local_name "access-port" in
# tapi-common -- the exact worked example lexicon-69m.10 investigated.
ACCESS_PORT_DEF_SIP = (
    "Reference to the AccessPort.\n"
    "CONDITION: Mandatory where the SIP is directly supported by an access port."
)
ACCESS_PORT_DEF_NEP = (
    "Reference to the AccessPort.\n"
    "CONDITION: Mandatory where the NEP is directly supported by an access port."
)
ACCESS_PORT_DEF_DEVICE = (
    "Access ports of the device.\nCONDITION: Mandatory where access ports are present."
)
ACCESS_PORT_DEF_SPAN = (
    "The access ports that bound the physical span.\n"
    "This allows for simple point to point cases as well as multi-point cases "
    "and cases where the physical span has only one fully defined end."
)

ACCESS_PORT_OCCURRENCES = [
    (
        f"{BASE_URI}/tapi-common/context/service-interface-point/access-port-supports-sip/access-port",
        "access-port",
        [ACCESS_PORT_DEF_SIP],
    ),
    (
        f"{BASE_URI}/tapi-common/context/topology-context/topology/node/owned-node-edge-point/access-port-supports-nep/access-port",
        "access-port",
        [ACCESS_PORT_DEF_NEP],
    ),
    (
        f"{BASE_URI}/tapi-common/context/notification-context/event-notification/device/access-port",
        "access-port",
        [ACCESS_PORT_DEF_DEVICE],
    ),
    (
        f"{BASE_URI}/tapi-common/context/physical-context/device/access-port",
        "access-port",
        [ACCESS_PORT_DEF_DEVICE],
    ),
    (
        f"{BASE_URI}/tapi-common/context/notification-context/event-notification/access-port",
        "access-port",
        ["none"],
    ),
    (
        f"{BASE_URI}/tapi-common/context/notification-context/event-notification/physical-span/access-port",
        "access-port",
        [ACCESS_PORT_DEF_SPAN],
    ),
    (
        f"{BASE_URI}/tapi-common/context/physical-context/physical-span/access-port",
        "access-port",
        [ACCESS_PORT_DEF_SPAN],
    ),
]

# A real single-occurrence, single-real-text concept from the same family
# (distinct local_name from "access-port") -- the "today's shape" case.
ACCESS_PORT_SUPPORTS_NEP_URI = (
    f"{BASE_URI}/tapi-common/context/topology-context/topology/node/owned-node-edge-point/access-port-supports-nep"
)
ACCESS_PORT_SUPPORTS_NEP_DEF = (
    "This augment allows NEP to refer to its AccessPorts despite TapiTopology "
    "model does not import TapiEquipment model."
)

# A real single-occurrence concept whose only text is the "none" sentinel.
ACCESS_PORT_SUPPORTS_SIP_URI = (
    f"{BASE_URI}/tapi-common/context/service-interface-point/access-port-supports-sip"
)

# Three real distinct texts shared across a subset of the real 25-occurrence
# node-edge-point family (tapi-common.lexicon.ttl:12161-12360); the rest of
# that real family is "none".
NODE_EDGE_POINT_TEXT_A = "The supporting NodeEdgePoint (NEP) instance."
NODE_EDGE_POINT_TEXT_B = "The NEPs connected by the Link."
NODE_EDGE_POINT_TEXT_C = (
    "NEPs and their client CEPs that the rules apply to. This reference is "
    "optional, while the reverse reference is mandatory (NEP refers to NRGs)."
)

NODE_EDGE_POINT_OCCURRENCES = [
    (
        f"{BASE_URI}/tapi-common/context/connectivity-context/connectivity-service/internal-point/node-edge-point",
        "node-edge-point",
        [NODE_EDGE_POINT_TEXT_A],
    ),
    (
        f"{BASE_URI}/tapi-common/context/notification-context/event-notification/connectivity-service/internal-point/node-edge-point",
        "node-edge-point",
        [NODE_EDGE_POINT_TEXT_A],
    ),
    (
        f"{BASE_URI}/tapi-common/context/notification-context/event-notification/link/node-edge-point",
        "node-edge-point",
        [NODE_EDGE_POINT_TEXT_B],
    ),
    (
        f"{BASE_URI}/tapi-common/context/notification-context/event-notification/topology/link/node-edge-point",
        "node-edge-point",
        [NODE_EDGE_POINT_TEXT_B],
    ),
    (
        f"{BASE_URI}/tapi-common/context/topology-context/topology/link/node-edge-point",
        "node-edge-point",
        [NODE_EDGE_POINT_TEXT_B],
    ),
    (
        f"{BASE_URI}/tapi-common/context/notification-context/event-notification/node-rule-group/node-edge-point",
        "node-edge-point",
        [NODE_EDGE_POINT_TEXT_C],
    ),
    (
        f"{BASE_URI}/tapi-common/context/notification-context/event-notification/node/node-rule-group/node-edge-point",
        "node-edge-point",
        [NODE_EDGE_POINT_TEXT_C],
    ),
    (
        f"{BASE_URI}/tapi-common/context/topology-context/topology/node/node-rule-group/node-edge-point",
        "node-edge-point",
        [NODE_EDGE_POINT_TEXT_C],
    ),
    (
        f"{BASE_URI}/tapi-common/context/notification-context/event-notification/node-edge-point",
        "node-edge-point",
        ["none"],
    ),
    (
        f"{BASE_URI}/tapi-common/context/notification-context/event-notification/path/node-edge-point",
        "node-edge-point",
        ["none"],
    ),
    (
        f"{BASE_URI}/tapi-common/context/path-computation-context/path/node-edge-point",
        "node-edge-point",
        ["none"],
    ),
]

# Real ietf-network "node" occurrences -- the base RFC 8345 path and an
# RFC 8795 TE-augmentation path under the same module, the exact shared-module
# pair lexicon-69m.11's incident recorded (ietf-network.lexicon.ttl:921-929,
# 940-946).
IETF_NETWORK_NODE_BASE_URI = f"{BASE_URI}/ietf-network/networks/network/node"
IETF_NETWORK_NODE_BASE_DEF = (
    "Augments termination points that terminate links.\n"
    "Termination points can ultimately be mapped to interfaces."
)
IETF_NETWORK_NODE_BASE_NOTES = [
    "The inventory of nodes of this network.",
    "Configuration parameters for TE at the node level.",
]
IETF_NETWORK_NODE_TE_URI = f"{BASE_URI}/ietf-network/networks/network/node/te/statistics/node"
IETF_NETWORK_NODE_TE_DEF = "Contains statistics attributes at the TE node level."


# ---------------------------------------------------------------------------
# Fixture-building helpers
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _class_block(uri: str, label: str, comments: list) -> str:
    parts = ["    a owl:Class", f'    rdfs:label "{_esc(label)}"']
    for c in comments:
        parts.append(f'    rdfs:comment "{_esc(c)}"')
    return f"<{uri}>\n" + " ;\n".join(parts) + " .\n\n"


def write_ontology(path, classes) -> None:
    """classes: iterable of (uri, label, comments) tuples."""
    header = (
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n"
    )
    body = "".join(_class_block(u, l, c) for u, l, c in classes)
    path.write_text(header + body, encoding="utf-8")


def run_draft_lexicon(monkeypatch, ontology_path, out_dir, base_uri=BASE_URI):
    import draft_lexicon  # lazy import, mirrors conftest.py's align_lexicons pattern

    monkeypatch.setattr(
        sys,
        "argv",
        ["draft_lexicon.py", str(ontology_path), "--base-uri", base_uri, "--out-dir", str(out_dir)],
    )
    draft_lexicon.main()


# ---------------------------------------------------------------------------
# Task 1: end-to-end "one concept, many occurrences" tests
# ---------------------------------------------------------------------------


def test_access_port_family_collapses_to_one_concept(tmp_path, monkeypatch):
    import draft_lexicon  # lazy import

    ontology_path = tmp_path / "tapi.ttl"
    write_ontology(ontology_path, ACCESS_PORT_OCCURRENCES)

    # Unit-level: collect_occurrences() groups all seven real occurrences
    # under exactly one (module, local_name) key.
    graph = Graph()
    graph.parse(str(ontology_path), format="turtle")
    concepts = draft_lexicon.collect_occurrences(graph, BASE_URI)
    assert list(concepts.keys()) == [("tapi-common", "access-port")]
    assert len(concepts[("tapi-common", "access-port")]) == len(ACCESS_PORT_OCCURRENCES)

    # End-to-end: exactly one lex:tapi-common-access-port subject is written.
    out_dir = tmp_path / "out"
    run_draft_lexicon(monkeypatch, ontology_path, out_dir)

    out_graph = Graph()
    out_graph.parse(str(out_dir / "tapi-common.lexicon.ttl"), format="turtle")
    subject = LEX["tapi-common-access-port"]
    assert (subject, RDF.type, LEX.ReferenceEntry) in out_graph
    type_triples = list(out_graph.triples((subject, RDF.type, LEX.ReferenceEntry)))
    assert len(type_triples) == 1

    prov_uris = {str(u) for u in out_graph.objects(subject, PROV.wasDerivedFrom)}
    assert prov_uris == {u for u, _, _ in ACCESS_PORT_OCCURRENCES}


def test_every_distinct_real_text_survives_as_evidence(tmp_path, monkeypatch):
    ontology_path = tmp_path / "tapi.ttl"
    write_ontology(ontology_path, NODE_EDGE_POINT_OCCURRENCES)
    out_dir = tmp_path / "out"
    run_draft_lexicon(monkeypatch, ontology_path, out_dir)

    out_graph = Graph()
    out_graph.parse(str(out_dir / "tapi-common.lexicon.ttl"), format="turtle")
    subject = LEX["tapi-common-node-edge-point"]

    scope_notes = {str(v) for v in out_graph.objects(subject, SKOS.scopeNote)}
    assert scope_notes == {NODE_EDGE_POINT_TEXT_A, NODE_EDGE_POINT_TEXT_B, NODE_EDGE_POINT_TEXT_C}
    assert len(scope_notes) == 3
    # 2+ distinct real texts -> skos:definition is omitted entirely, per the
    # render_contract, not filled with a placeholder.
    assert out_graph.value(subject, SKOS.definition) is None


def test_null_evidence_literal_never_becomes_a_scope_note(tmp_path, monkeypatch):
    ontology_path = tmp_path / "tapi.ttl"
    write_ontology(ontology_path, ACCESS_PORT_OCCURRENCES)
    out_dir = tmp_path / "out"
    run_draft_lexicon(monkeypatch, ontology_path, out_dir)

    raw = (out_dir / "tapi-common.lexicon.ttl").read_text(encoding="utf-8")
    assert 'skos:scopeNote "none"' not in raw
    assert 'skos:definition "none"' not in raw


def test_no_duplicate_subject_uris_in_output(tmp_path, monkeypatch):
    ontology_path = tmp_path / "tapi.ttl"
    write_ontology(ontology_path, ACCESS_PORT_OCCURRENCES + NODE_EDGE_POINT_OCCURRENCES)
    out_dir = tmp_path / "out"
    run_draft_lexicon(monkeypatch, ontology_path, out_dir)

    raw = (out_dir / "tapi-common.lexicon.ttl").read_text(encoding="utf-8")
    subject_lines = [line for line in raw.splitlines() if line.startswith("lex:")]
    assert len(subject_lines) == len(set(subject_lines))
    assert set(subject_lines) == {"lex:tapi-common-access-port", "lex:tapi-common-node-edge-point"}


def test_single_occurrence_entry_shape_is_preserved(tmp_path, monkeypatch):
    ontology_path = tmp_path / "tapi.ttl"
    write_ontology(
        ontology_path,
        [(ACCESS_PORT_SUPPORTS_NEP_URI, "access-port-supports-nep", [ACCESS_PORT_SUPPORTS_NEP_DEF])],
    )
    out_dir = tmp_path / "out"
    run_draft_lexicon(monkeypatch, ontology_path, out_dir)

    out_graph = Graph()
    out_graph.parse(str(out_dir / "tapi-common.lexicon.ttl"), format="turtle")
    subject = LEX["tapi-common-access-port-supports-nep"]

    assert str(out_graph.value(subject, SKOS.definition)) == ACCESS_PORT_SUPPORTS_NEP_DEF
    assert list(out_graph.objects(subject, SKOS.scopeNote)) == []
    assert (subject, LEX.needsCuration, None) not in out_graph or not bool(
        out_graph.value(subject, LEX.needsCuration)
    )


def test_concept_with_no_real_text_has_no_definition(tmp_path, monkeypatch):
    ontology_path = tmp_path / "tapi.ttl"
    write_ontology(
        ontology_path,
        [(ACCESS_PORT_SUPPORTS_SIP_URI, "access-port-supports-sip", ["none"])],
    )
    out_dir = tmp_path / "out"
    run_draft_lexicon(monkeypatch, ontology_path, out_dir)

    out_graph = Graph()
    out_graph.parse(str(out_dir / "tapi-common.lexicon.ttl"), format="turtle")
    subject = LEX["tapi-common-access-port-supports-sip"]

    assert out_graph.value(subject, SKOS.definition) is None
    assert list(out_graph.objects(subject, SKOS.scopeNote)) == []
    needs_curation = out_graph.value(subject, LEX.needsCuration)
    assert needs_curation is not None and bool(needs_curation.toPython())

    # Task 3 extension: the read-side contract that motivates omitting the
    # placeholder -- align_lexicons.py's evidence gate must read this entry
    # as evidence-free.
    import align_lexicons  # lazy import -- see conftest.py's convention

    entry = align_lexicons.load_fixture_entries(
        out_dir,
        [
            align_lexicons.FixtureRef(
                source="tapi",
                file="tapi-common.lexicon.ttl",
                lex_id="tapi-common-access-port-supports-sip",
            )
        ],
    )[0]
    assert entry.definition is None
    assert entry.scope_notes == []
    assert entry.has_evidence is False


def test_entity_class_precedence_prefers_definitional_kind(tmp_path, monkeypatch):
    ontology_path = tmp_path / "mixed.ttl"
    grouping_uri = f"{BASE_URI}/grouping/tapi-common/widget-thing"
    structural_uri = f"{BASE_URI}/tapi-common/context/widget-thing"
    write_ontology(
        ontology_path,
        [
            (grouping_uri, "widget-thing", ["A reusable grouping describing the widget thing."]),
            (structural_uri, "widget-thing", ["A structural reference to the widget thing."]),
        ],
    )
    out_dir = tmp_path / "out"
    run_draft_lexicon(monkeypatch, ontology_path, out_dir)

    out_graph = Graph()
    out_graph.parse(str(out_dir / "tapi-common.lexicon.ttl"), format="turtle")
    subject = LEX["tapi-common-widget-thing"]
    assert out_graph.value(subject, LEX.entityClass) == LEX.GroupingKind


# ---------------------------------------------------------------------------
# Task 2: merge-aware, atomic write tests
# ---------------------------------------------------------------------------


def test_second_source_tree_run_does_not_clobber_first(tmp_path, monkeypatch):
    """The exact real shared-module pair lexicon-69m.11's incident recorded:
    ietf-network's base RFC 8345 node path (tree A) and an RFC 8795 TE
    augmentation path under the same module (tree B), run separately."""
    out_dir = tmp_path / "out"

    tree_a = tmp_path / "tree_a.ttl"
    write_ontology(
        tree_a,
        [(IETF_NETWORK_NODE_BASE_URI, "node", [IETF_NETWORK_NODE_BASE_DEF] + IETF_NETWORK_NODE_BASE_NOTES)],
    )
    run_draft_lexicon(monkeypatch, tree_a, out_dir)

    tree_b = tmp_path / "tree_b.ttl"
    write_ontology(tree_b, [(IETF_NETWORK_NODE_TE_URI, "node", [IETF_NETWORK_NODE_TE_DEF])])
    run_draft_lexicon(monkeypatch, tree_b, out_dir)

    out_graph = Graph()
    out_graph.parse(str(out_dir / "ietf-network.lexicon.ttl"), format="turtle")
    subject = LEX["ietf-network-node"]

    prov_uris = {str(u) for u in out_graph.objects(subject, PROV.wasDerivedFrom)}
    assert prov_uris == {IETF_NETWORK_NODE_BASE_URI, IETF_NETWORK_NODE_TE_URI}

    scope_notes = {str(v) for v in out_graph.objects(subject, SKOS.scopeNote)}
    assert IETF_NETWORK_NODE_BASE_DEF in scope_notes
    assert IETF_NETWORK_NODE_TE_DEF in scope_notes
    for note in IETF_NETWORK_NODE_BASE_NOTES:
        assert note in scope_notes


def test_second_source_tree_run_order_independent(tmp_path, monkeypatch):
    """Running A then B produces a file byte-identical to running B then A,
    and to running both together in one invocation."""
    tree_a = tmp_path / "tree_a.ttl"
    write_ontology(
        tree_a,
        [(IETF_NETWORK_NODE_BASE_URI, "node", [IETF_NETWORK_NODE_BASE_DEF] + IETF_NETWORK_NODE_BASE_NOTES)],
    )
    tree_b = tmp_path / "tree_b.ttl"
    write_ontology(tree_b, [(IETF_NETWORK_NODE_TE_URI, "node", [IETF_NETWORK_NODE_TE_DEF])])

    out_ab = tmp_path / "out_ab"
    run_draft_lexicon(monkeypatch, tree_a, out_ab)
    run_draft_lexicon(monkeypatch, tree_b, out_ab)

    out_ba = tmp_path / "out_ba"
    run_draft_lexicon(monkeypatch, tree_b, out_ba)
    run_draft_lexicon(monkeypatch, tree_a, out_ba)

    out_combined = tmp_path / "out_combined"
    combined = tmp_path / "combined.ttl"
    write_ontology(
        combined,
        [
            (IETF_NETWORK_NODE_BASE_URI, "node", [IETF_NETWORK_NODE_BASE_DEF] + IETF_NETWORK_NODE_BASE_NOTES),
            (IETF_NETWORK_NODE_TE_URI, "node", [IETF_NETWORK_NODE_TE_DEF]),
        ],
    )
    run_draft_lexicon(monkeypatch, combined, out_combined)

    bytes_ab = (out_ab / "ietf-network.lexicon.ttl").read_bytes()
    bytes_ba = (out_ba / "ietf-network.lexicon.ttl").read_bytes()
    bytes_combined = (out_combined / "ietf-network.lexicon.ttl").read_bytes()
    assert bytes_ab == bytes_ba == bytes_combined


def test_stale_occurrence_is_reported_and_retained(tmp_path, monkeypatch, capsys):
    """An occurrence URI present in a prior run's output but absent from
    every graph in the current run is retained and reported on stdout,
    never deleted (D-08)."""
    out_dir = tmp_path / "out"

    both = tmp_path / "both.ttl"
    write_ontology(
        both,
        [
            (IETF_NETWORK_NODE_BASE_URI, "node", [IETF_NETWORK_NODE_BASE_DEF]),
            (IETF_NETWORK_NODE_TE_URI, "node", [IETF_NETWORK_NODE_TE_DEF]),
        ],
    )
    run_draft_lexicon(monkeypatch, both, out_dir)
    capsys.readouterr()

    # Second run's input no longer produces the TE-augmentation occurrence
    # (as if that subtree were refactored away) -- it must not be deleted.
    only_base = tmp_path / "only_base.ttl"
    write_ontology(only_base, [(IETF_NETWORK_NODE_BASE_URI, "node", [IETF_NETWORK_NODE_BASE_DEF])])
    run_draft_lexicon(monkeypatch, only_base, out_dir)
    captured = capsys.readouterr()

    assert any(
        line.startswith("STALE: ") and IETF_NETWORK_NODE_TE_URI in line
        for line in captured.out.splitlines()
    )

    out_graph = Graph()
    out_graph.parse(str(out_dir / "ietf-network.lexicon.ttl"), format="turtle")
    subject = LEX["ietf-network-node"]
    prov_uris = {str(u) for u in out_graph.objects(subject, PROV.wasDerivedFrom)}
    assert IETF_NETWORK_NODE_TE_URI in prov_uris
    assert IETF_NETWORK_NODE_BASE_URI in prov_uris
    scope_notes = {str(v) for v in out_graph.objects(subject, SKOS.scopeNote)}
    assert IETF_NETWORK_NODE_TE_DEF in scope_notes


def test_merge_is_idempotent_across_repeat_runs(tmp_path, monkeypatch):
    """Running the same ontology twice does not duplicate any occurrence URI
    and produces a byte-identical file the second time."""
    ontology_path = tmp_path / "tapi.ttl"
    write_ontology(ontology_path, ACCESS_PORT_OCCURRENCES)
    out_dir = tmp_path / "out"

    run_draft_lexicon(monkeypatch, ontology_path, out_dir)
    first_bytes = (out_dir / "tapi-common.lexicon.ttl").read_bytes()

    run_draft_lexicon(monkeypatch, ontology_path, out_dir)
    second_bytes = (out_dir / "tapi-common.lexicon.ttl").read_bytes()

    assert first_bytes == second_bytes

    out_graph = Graph()
    out_graph.parse(str(out_dir / "tapi-common.lexicon.ttl"), format="turtle")
    subject = LEX["tapi-common-access-port"]
    prov_uris = [str(u) for u in out_graph.objects(subject, PROV.wasDerivedFrom)]
    assert len(prov_uris) == len(set(prov_uris))
    assert len(prov_uris) == len(ACCESS_PORT_OCCURRENCES)


def test_interrupted_write_leaves_previous_file_intact(tmp_path, monkeypatch):
    """If rendering raises partway through, the pre-existing lexicon file on
    disk is unchanged rather than truncated."""
    import draft_lexicon  # lazy import

    ontology_path = tmp_path / "tapi.ttl"
    write_ontology(ontology_path, ACCESS_PORT_OCCURRENCES + NODE_EDGE_POINT_OCCURRENCES)
    out_dir = tmp_path / "out"

    run_draft_lexicon(monkeypatch, ontology_path, out_dir)
    pre_run_bytes = (out_dir / "tapi-common.lexicon.ttl").read_bytes()

    real_render_concept = draft_lexicon.render_concept
    call_count = {"n": 0}

    def _raise_on_second_call(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated rendering failure")
        return real_render_concept(*args, **kwargs)

    monkeypatch.setattr(draft_lexicon, "render_concept", _raise_on_second_call)

    with pytest.raises(RuntimeError):
        run_draft_lexicon(monkeypatch, ontology_path, out_dir)

    post_failure_bytes = (out_dir / "tapi-common.lexicon.ttl").read_bytes()
    assert post_failure_bytes == pre_run_bytes
