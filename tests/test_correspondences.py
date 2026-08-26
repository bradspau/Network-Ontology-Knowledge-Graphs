"""
Tests for the correspondence-artifact writer (Phase 4, OUT-01/OUT-02):
CorrespondenceTriple, correspondences_from_results(),
render_correspondences_ttl(), write_correspondences_ttl(),
resolve_lexicon_version(), and assert_lexicon_clean().

Task 1 (this file's first block) wires the tracer -- one confirmed pair
through the whole new path (filter -> schema construction -> literal
rendering -> sorted Turtle* text -> file on disk via the --emit-correspondences
CLI flag). Task 2 adds the dirty-lexicon-tree hard-stop tests. Task 3 adds
the fixed-temperature and determinism proofs. Tests are grouped by task with
a `# ── Task N ──` banner, in plan-authored order.

Every test that needs a PairResult/ConfidenceBreakdown/Candidate builds them
directly from the real fixture_entries fixture rather than driving the whole
matcher pipeline (this plan's own <action> instruction), except the
main()-level tests, which drive main() itself with a scripted or recording
client double per conftest.py's convention. No test in this module requires
ANTHROPIC_API_KEY.
"""
import subprocess
import sys

import pytest
from rdflib import Graph, Literal

import align_lexicons

FAKE_VERSION = "a" * 40
FAKE_MODEL = "claude-test-model"


# ── Shared builders ─────────────────────────────────────────────────────


def _confidence(**overrides):
    """A fully-populated ConfidenceBreakdown by default (every optional
    field non-None) so tests exercising the fixed predicate order see every
    row; individual tests override specific fields (often to None) to prove
    the omission behavior."""
    fields = dict(
        label_definition_agreement=True,
        structural_corroboration=0.2,
        validator_ran=True,
        validator_agrees=True,
        validator_counter_argument="The strongest case against this verdict still fails.",
        escalated=False,
        tier="high",
    )
    fields.update(overrides)
    return align_lexicons.ConfidenceBreakdown(**fields)


def _pair_result(tapi_entry, ietf_entry, verdict, **overrides):
    candidate = align_lexicons.Candidate(
        tapi=tapi_entry, ietf=ietf_entry, label_score=100.0, origin="label-pass"
    )
    is_confirmed = verdict in align_lexicons.CONFIRMED_VERDICTS
    fields = dict(
        candidate=candidate,
        verdict=verdict,
        rationale="test rationale",
        evidence_quote="test evidence quote",
        decided_by="confirmation-pass" if is_confirmed else "evidence-gate",
        confidence=_confidence() if is_confirmed else None,
        deciding_signal="definition-text" if is_confirmed else None,
    )
    fields.update(overrides)
    return align_lexicons.PairResult(**fields)


# ── Task 1: tracer -- verdict filter, schema invariant, literal rendering,
# sorted Turtle*, file on disk via the CLI flag ──────────────────────────


def test_confirmed_verdicts_become_base_triples(fixture_entries):
    _, _, by_lex_id = fixture_entries
    exact = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match"
    )
    close = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
    )
    rejected = _pair_result(
        by_lex_id["tapi-topology-node-edge-point"],
        by_lex_id["ietf-network-tunnel-termination-point"],
        "reject",
    )
    insufficient = _pair_result(
        by_lex_id["tapi-common-service-interface-point"],
        by_lex_id["ietf-network-termination-point"],
        "insufficient_evidence",
    )

    triples = align_lexicons.correspondences_from_results(
        [exact, close, rejected, insufficient], FAKE_VERSION, FAKE_MODEL
    )
    assert len(triples) == 2
    assert {t.predicate for t in triples} == {"skos:exactMatch", "skos:closeMatch"}

    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    assert "lex:tapi-topology-node skos:exactMatch lex:ietf-network-node ." in text
    assert (
        "lex:tapi-topology-node-rule-group skos:closeMatch "
        "lex:ietf-network-connectivity-matrix ." in text
    )
    base_section = text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]
    assert base_section.count(" skos:exactMatch ") == 1
    assert base_section.count(" skos:closeMatch ") == 1


def test_rejected_and_insufficient_verdicts_are_never_emitted(fixture_entries):
    _, _, by_lex_id = fixture_entries
    rejected = _pair_result(
        by_lex_id["tapi-topology-node-edge-point"],
        by_lex_id["ietf-network-tunnel-termination-point"],
        "reject",
    )
    insufficient = _pair_result(
        by_lex_id["tapi-common-service-interface-point"],
        by_lex_id["ietf-network-termination-point"],
        "insufficient_evidence",
    )
    triples = align_lexicons.correspondences_from_results(
        [rejected, insufficient], FAKE_VERSION, FAKE_MODEL
    )
    assert triples == []
    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    assert "tapi-topology-node-edge-point" not in text
    assert "tapi-common-service-interface-point" not in text


def test_escalated_confirmed_pair_is_emitted_with_escalation_visible(fixture_entries):
    _, _, by_lex_id = fixture_entries
    escalated_confidence = _confidence(validator_agrees=False, tier="medium", escalated=True)
    result = _pair_result(
        by_lex_id["tapi-topology-node"],
        by_lex_id["ietf-network-node"],
        "confirm_exact_match",
        confidence=escalated_confidence,
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    assert len(triples) == 1
    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    assert 'lex:escalated "true"^^<http://www.w3.org/2001/XMLSchema#boolean>' in text
    assert 'lex:validatorAgrees "false"^^<http://www.w3.org/2001/XMLSchema#boolean>' in text


def test_annotation_carries_every_confidence_field_in_fixed_order(fixture_entries):
    _, _, by_lex_id = fixture_entries
    full = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match"
    )
    triples = align_lexicons.correspondences_from_results([full], FAKE_VERSION, FAKE_MODEL)
    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    annotation_section = text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[1]

    positions = [annotation_section.index(pred) for pred in align_lexicons.CORRESPONDENCE_ANNOTATION_ORDER]
    assert positions == sorted(positions)
    for pred in align_lexicons.CORRESPONDENCE_ANNOTATION_ORDER:
        assert pred in annotation_section

    sparse_confidence = _confidence(
        structural_corroboration=None,
        validator_ran=False,
        validator_agrees=None,
        validator_counter_argument=None,
    )
    sparse = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
        confidence=sparse_confidence,
    )
    sparse_triples = align_lexicons.correspondences_from_results([sparse], FAKE_VERSION, FAKE_MODEL)
    sparse_text = align_lexicons.render_correspondences_ttl(sparse_triples, FAKE_VERSION, FAKE_MODEL)
    sparse_annotation = sparse_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[1]
    assert "lex:structuralCorroboration" not in sparse_annotation
    assert "lex:validatorAgrees" not in sparse_annotation
    assert "lex:validatorCounterArgument" not in sparse_annotation
    remaining = [p for p in align_lexicons.CORRESPONDENCE_ANNOTATION_ORDER if p in sparse_annotation]
    remaining_positions = [sparse_annotation.index(p) for p in remaining]
    assert remaining_positions == sorted(remaining_positions)


def test_lexicon_version_repeats_on_every_annotation(fixture_entries):
    _, _, by_lex_id = fixture_entries
    exact = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match"
    )
    close = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
    )
    triples = align_lexicons.correspondences_from_results([exact, close], FAKE_VERSION, FAKE_MODEL)
    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    annotation_section = text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[1]
    assert annotation_section.count(FAKE_VERSION) >= len(triples)


def test_artifact_resource_states_type_level_scope_as_triples(fixture_entries):
    _, _, by_lex_id = fixture_entries
    exact = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match"
    )
    triples = align_lexicons.correspondences_from_results([exact], FAKE_VERSION, FAKE_MODEL)
    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    base_section = text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]

    graph = Graph()
    graph.parse(data=base_section, format="turtle")

    LEX = align_lexicons.LEX
    subject = LEX["correspondence-artifact"]
    assert (subject, LEX.scopeLevel, Literal("type-level-only")) in graph

    match_triples = list(graph.triples((None, align_lexicons.SKOS.exactMatch, None)))
    assert len(match_triples) == 1


def test_correspondence_triple_rejects_missing_confidence():
    with pytest.raises(ValueError, match="CorrespondenceTriple invariant violated"):
        align_lexicons.CorrespondenceTriple(
            tapi_lex_id="tapi-x",
            ietf_lex_id="ietf-y",
            predicate="skos:exactMatch",
            confidence=None,
            evidence_quote="quote",
            decided_by="confirmation-pass",
            deciding_signal="definition-text",
            lexicon_version=FAKE_VERSION,
            model=FAKE_MODEL,
        )


def test_correspondence_triple_rejects_empty_evidence_quote():
    with pytest.raises(ValueError, match="CorrespondenceTriple invariant violated"):
        align_lexicons.CorrespondenceTriple(
            tapi_lex_id="tapi-x",
            ietf_lex_id="ietf-y",
            predicate="skos:exactMatch",
            confidence=_confidence(),
            evidence_quote="   ",
            decided_by="confirmation-pass",
            deciding_signal="definition-text",
            lexicon_version=FAKE_VERSION,
            model=FAKE_MODEL,
        )


def test_correspondence_triple_rejects_missing_lexicon_version():
    with pytest.raises(ValueError, match="CorrespondenceTriple invariant violated"):
        align_lexicons.CorrespondenceTriple(
            tapi_lex_id="tapi-x",
            ietf_lex_id="ietf-y",
            predicate="skos:exactMatch",
            confidence=_confidence(),
            evidence_quote="quote",
            decided_by="confirmation-pass",
            deciding_signal="definition-text",
            lexicon_version="",
            model=FAKE_MODEL,
        )


def test_correspondence_triple_rejects_unconfirmed_verdict(fixture_entries):
    _, _, by_lex_id = fixture_entries
    rejected = _pair_result(
        by_lex_id["tapi-topology-node-edge-point"],
        by_lex_id["ietf-network-tunnel-termination-point"],
        "reject",
    )
    with pytest.raises(ValueError):
        align_lexicons.CorrespondenceTriple.from_pair_result(rejected, FAKE_VERSION, FAKE_MODEL)


def test_base_section_parses_as_turtle(fixture_entries):
    _, _, by_lex_id = fixture_entries
    exact = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match"
    )
    close = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
    )
    triples = align_lexicons.correspondences_from_results([exact, close], FAKE_VERSION, FAKE_MODEL)
    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    base_section = text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]

    graph = Graph()
    graph.parse(data=base_section, format="turtle")
    match_count = len(list(graph.triples((None, align_lexicons.SKOS.exactMatch, None)))) + len(
        list(graph.triples((None, align_lexicons.SKOS.closeMatch, None)))
    )
    assert match_count == 2


def test_writer_refuses_to_write_into_the_lexicon_directory(fixture_entries, lexicon_dir, tmp_path):
    _, _, by_lex_id = fixture_entries
    exact = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match"
    )
    triples = align_lexicons.correspondences_from_results([exact], FAKE_VERSION, FAKE_MODEL)

    with pytest.raises(ValueError):
        align_lexicons.write_correspondences_ttl(
            lexicon_dir / "correspondences.ttl", triples, FAKE_VERSION, FAKE_MODEL, lexicon_dir
        )

    with pytest.raises(ValueError):
        align_lexicons.write_correspondences_ttl(
            tmp_path / "tapi-topology.lexicon.ttl", triples, FAKE_VERSION, FAKE_MODEL, lexicon_dir
        )

    good_path = tmp_path / "correspondences.ttl"
    align_lexicons.write_correspondences_ttl(
        good_path, triples, FAKE_VERSION, FAKE_MODEL, lexicon_dir
    )
    assert good_path.exists()
    assert not (lexicon_dir / "correspondences.ttl").exists()


def test_resolve_lexicon_version_returns_the_containing_repos_hash(lexicon_dir):
    version = align_lexicons.resolve_lexicon_version(lexicon_dir)
    assert len(version) == 40
    assert all(c in "0123456789abcdef" for c in version)

    expected = subprocess.run(
        ["git", "-C", str(lexicon_dir), "log", "-1", "--format=%H", "--", "."],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert version == expected


def test_main_emit_flag_writes_correspondences_file(recording_client, monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "correspondences.ttl"
    monkeypatch.setattr(
        sys, "argv", ["align_lexicons.py", "--emit-correspondences", str(output_path)]
    )
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)

    align_lexicons.main()

    assert output_path.exists()
    text = output_path.read_text()
    assert "correspondence-artifact" in text
    captured = capsys.readouterr()
    assert str(output_path) in captured.out


def test_main_without_emit_flag_writes_no_file(recording_client, monkeypatch, tmp_path):
    fake_default = tmp_path / "correspondences.ttl"
    monkeypatch.setattr(align_lexicons, "DEFAULT_CORRESPONDENCES_PATH", fake_default)
    monkeypatch.setattr(sys, "argv", ["align_lexicons.py"])
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)

    align_lexicons.main()

    assert not fake_default.exists()
