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


# ── Task 2: refuse to run against a lexicon tree whose recorded version
# would be a lie ──────────────────────────────────────────────────────────


def _make_throwaway_repo(tmp_path):
    """A real, hermetic git repository built with subprocess -- never
    asserting against the state of the real yang4owl/lexicon/, which must
    stay clean for the pre-existing main()-level test_align_lexicons.py
    suite. Local user.name/user.email are configured so the commit succeeds
    on any machine."""
    repo_dir = tmp_path / "repo"
    lexicon_dir = repo_dir / "lexicon"
    lexicon_dir.mkdir(parents=True)
    (lexicon_dir / "sample.lexicon.ttl").write_text("# sample lexicon fixture\n")
    (repo_dir / "other.py").write_text("# a file outside the lexicon dir\n")

    def _run(*args):
        subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True)

    _run("init", "-q")
    _run("config", "user.email", "test@example.com")
    _run("config", "user.name", "Test")
    _run("add", "-A")
    _run("commit", "-q", "-m", "initial commit")
    return repo_dir, lexicon_dir


def test_assert_lexicon_clean_returns_when_clean(tmp_path):
    _, lexicon_dir = _make_throwaway_repo(tmp_path)
    align_lexicons.assert_lexicon_clean(lexicon_dir)  # must not raise


def test_assert_lexicon_clean_raises_on_modified_tracked_file(tmp_path):
    _, lexicon_dir = _make_throwaway_repo(tmp_path)
    (lexicon_dir / "sample.lexicon.ttl").write_text("# modified\n")
    with pytest.raises(align_lexicons.DirtyLexiconError, match="sample.lexicon.ttl"):
        align_lexicons.assert_lexicon_clean(lexicon_dir)


def test_assert_lexicon_clean_raises_on_untracked_file(tmp_path):
    _, lexicon_dir = _make_throwaway_repo(tmp_path)
    (lexicon_dir / "new.lexicon.ttl").write_text("# untracked new entry\n")
    with pytest.raises(align_lexicons.DirtyLexiconError, match="new.lexicon.ttl"):
        align_lexicons.assert_lexicon_clean(lexicon_dir)


def test_dirty_file_outside_the_lexicon_directory_does_not_stop_the_run(tmp_path):
    repo_dir, lexicon_dir = _make_throwaway_repo(tmp_path)
    (repo_dir / "other.py").write_text("# modified outside the lexicon dir\n")
    align_lexicons.assert_lexicon_clean(lexicon_dir)  # must not raise


def test_resolve_lexicon_version_raises_outside_a_repository(tmp_path):
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    with pytest.raises(align_lexicons.LexiconVersionUnavailable):
        align_lexicons.resolve_lexicon_version(outside)


def test_assert_lexicon_clean_raises_outside_a_repository(tmp_path):
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    with pytest.raises(align_lexicons.LexiconVersionUnavailable):
        align_lexicons.assert_lexicon_clean(outside)


def test_dirty_lexicon_tree_hard_stops_before_any_client_call(tmp_path, recording_client, monkeypatch):
    _, lexicon_dir = _make_throwaway_repo(tmp_path)
    (lexicon_dir / "sample.lexicon.ttl").write_text("# dirtied before the run\n")

    monkeypatch.setattr(sys, "argv", ["align_lexicons.py", "--lexicon-dir", str(lexicon_dir)])
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)

    with pytest.raises(align_lexicons.DirtyLexiconError):
        align_lexicons.main()

    assert recording_client.calls == [], (
        "the dirty-tree stop must precede any billed client.messages.parse() call"
    )


# ── Task 3: fixed temperature on every call, fixed byte order in every file


class _SystemRejectingRecordingMessages:
    """Rejects the system= kwarg once, then succeeds on the retry that
    omits it -- proves the WR-03 fallback call also carries an explicit
    temperature (D-07), mirroring test_align_lexicons.py's
    _SystemRejectingMessages double."""

    def __init__(self, verdict):
        self._verdict = verdict
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if "system" in kwargs:
            raise TypeError("parse() got an unexpected keyword argument 'system'")
        return _FakeParsedResponse(self._verdict)


class _FakeParsedResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _SystemRejectingRecordingClient:
    def __init__(self, verdict):
        self.messages = _SystemRejectingRecordingMessages(verdict)


def test_every_model_call_passes_an_explicit_temperature(recording_client, monkeypatch, fixture_entries):
    monkeypatch.setattr(sys, "argv", ["align_lexicons.py"])
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)

    align_lexicons.main()

    assert recording_client.calls, "expected at least one confirmation call in a full fixture run"
    assert all(
        call.get("temperature") == align_lexicons.LLM_TEMPERATURE for call in recording_client.calls
    )

    # WR-03 fallback paths (confirm_pair + validate_pair) must also carry an
    # explicit temperature -- exercised directly, since triggering the
    # fallback requires a client that rejects the system= kwarg.
    _, _, by_lex_id = fixture_entries
    candidate = align_lexicons.Candidate(
        tapi=by_lex_id["tapi-topology-node"],
        ietf=by_lex_id["ietf-network-node"],
        label_score=100.0,
        origin="label-pass",
    )
    match_verdict = align_lexicons.MatchVerdict(
        verdict="confirm_exact_match", rationale="r", evidence_quote="q"
    )
    confirm_client = _SystemRejectingRecordingClient(match_verdict)
    align_lexicons.confirm_pair(confirm_client, candidate)
    assert len(confirm_client.messages.calls) == 2
    assert confirm_client.messages.calls[1]["temperature"] == align_lexicons.LLM_TEMPERATURE

    validator_verdict = align_lexicons.ValidatorVerdict(agrees=True, counter_argument="c")
    validate_client = _SystemRejectingRecordingClient(validator_verdict)
    align_lexicons.validate_pair(validate_client, candidate, match_verdict)
    assert len(validate_client.messages.calls) == 2
    assert validate_client.messages.calls[1]["temperature"] == align_lexicons.LLM_TEMPERATURE


def test_two_renders_of_identical_results_are_byte_identical(fixture_entries):
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
    first = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    second = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    assert first == second


def test_render_order_is_independent_of_input_order(fixture_entries):
    _, _, by_lex_id = fixture_entries
    exact = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match"
    )
    close = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
    )
    forward = align_lexicons.correspondences_from_results([exact, close], FAKE_VERSION, FAKE_MODEL)
    reordered = align_lexicons.correspondences_from_results([close, exact], FAKE_VERSION, FAKE_MODEL)
    forward_text = align_lexicons.render_correspondences_ttl(forward, FAKE_VERSION, FAKE_MODEL)
    reordered_text = align_lexicons.render_correspondences_ttl(reordered, FAKE_VERSION, FAKE_MODEL)
    assert forward_text == reordered_text

    # Directly reversing the ALREADY-SORTED triples list also renders
    # identically -- proving the sort inside render_correspondences_ttl
    # itself (not just correspondences_from_results' own sort) determines
    # output order.
    manually_reversed_text = align_lexicons.render_correspondences_ttl(
        list(reversed(forward)), FAKE_VERSION, FAKE_MODEL
    )
    assert forward_text == manually_reversed_text


def test_multiline_evidence_quote_stays_contiguous_regardless_of_order(fixture_entries):
    _, _, by_lex_id = fixture_entries
    multiline = _pair_result(
        by_lex_id["tapi-topology-node"],
        by_lex_id["ietf-network-node"],
        "confirm_exact_match",
        evidence_quote="line one\nline two\nline three",
    )
    other = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
    )
    forward = align_lexicons.correspondences_from_results([multiline, other], FAKE_VERSION, FAKE_MODEL)
    reordered = align_lexicons.correspondences_from_results([other, multiline], FAKE_VERSION, FAKE_MODEL)
    forward_text = align_lexicons.render_correspondences_ttl(forward, FAKE_VERSION, FAKE_MODEL)
    reordered_text = align_lexicons.render_correspondences_ttl(reordered, FAKE_VERSION, FAKE_MODEL)
    assert forward_text == reordered_text
    assert "line one\nline two\nline three" in forward_text


def test_adversarial_literal_cannot_inject_triples(fixture_entries):
    _, _, by_lex_id = fixture_entries
    payload = (
        'has "quotes", a backslash \\, a carriage return\r, a newline\n, '
        'an embedded triple-quote """ run, and a fragment shaped like an '
        'injection: <<lex:evil skos:exactMatch lex:evil2>> lex:pwned "yes" . '
        '# lex:tapi-topology-node skos:exactMatch lex:ietf-network-node .'
    )
    result = _pair_result(
        by_lex_id["tapi-topology-node"],
        by_lex_id["ietf-network-node"],
        "confirm_exact_match",
        evidence_quote=payload,
        confidence=_confidence(validator_counter_argument=payload),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    base_section = text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]

    graph = Graph()
    graph.parse(data=base_section, format="turtle")
    match_triples = list(graph.triples((None, align_lexicons.SKOS.exactMatch, None)))
    assert len(match_triples) == 1
    subj, _, obj = match_triples[0]
    assert str(subj) == "http://example.org/ontology/lexicon-vocab#tapi-topology-node"
    assert str(obj) == "http://example.org/ontology/lexicon-vocab#ietf-network-node"


def test_artifact_contains_no_absolute_path_or_environment_value(fixture_entries, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret-value-should-never-leak")
    _, _, by_lex_id = fixture_entries
    exact = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match"
    )
    triples = align_lexicons.correspondences_from_results([exact], FAKE_VERSION, FAKE_MODEL)
    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)

    assert "sk-super-secret-value-should-never-leak" not in text
    assert "ANTHROPIC_API_KEY" not in text
    assert "/Users/" not in text
    assert "/home/" not in text


def test_run_stopped_early_writes_no_artifact(recording_client, monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "correspondences.ttl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["align_lexicons.py", "--max-calls", "1", "--emit-correspondences", str(output_path)],
    )
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)

    with pytest.raises(SystemExit) as exc_info:
        align_lexicons.main()
    assert exc_info.value.code == 1

    assert not output_path.exists()
    captured = capsys.readouterr()
    assert "=== Run summary ===" in captured.out


# ── Plan 04-02 Task 1: the artifact reproduces the drafts' own §6 OTN
# worked example (docs/reference-lexicons.md lines 160-182) ─────────────────


def _otn_lex_id(refs, target_lex_id: str) -> str:
    """Looks up a FixtureRef by lex_id and raises loudly (StopIteration) if
    the fixture is ever renamed, rather than silently building an unmapped
    scripted-verdict key -- mirrors test_align_lexicons.py's
    test_confirmation_rejects_false_cognate lookup idiom. Reading the id
    back off the real FIXTURE_TAPI/FIXTURE_IETF module constants (rather
    than typing a second copy of the literal) is what makes a rename fail
    here instead of silently unmapping a pair."""
    return next(ref.lex_id for ref in refs if ref.lex_id == target_lex_id)


def _otn_worked_example_ids() -> dict:
    """The §6-row lex ids this plan's <otn_worked_example_contract> table
    names, resolved from FIXTURE_TAPI/FIXTURE_IETF themselves."""
    tapi, ietf = align_lexicons.FIXTURE_TAPI, align_lexicons.FIXTURE_IETF
    return {
        "node_t": _otn_lex_id(tapi, "tapi-topology-node"),
        "node_i": _otn_lex_id(ietf, "ietf-network-node"),
        "nep_t": _otn_lex_id(tapi, "tapi-topology-node-edge-point"),
        "term_i": _otn_lex_id(ietf, "ietf-network-termination-point"),
        "link_t": _otn_lex_id(tapi, "tapi-topology-link"),
        "link_i": _otn_lex_id(ietf, "ietf-network-link"),
        "nrg_t": _otn_lex_id(tapi, "tapi-topology-node-rule-group"),
        "cm_i": _otn_lex_id(ietf, "ietf-network-connectivity-matrix"),
        "sip_t": _otn_lex_id(tapi, "tapi-common-service-interface-point"),
        "ttp_i": _otn_lex_id(ietf, "ietf-network-tunnel-termination-point"),
    }


def _otn_worked_example_verdicts(ids: dict) -> dict:
    """§6's own outcomes: the link pair confirms exact, the node/NEP/
    node-rule-group pairs confirm close, and the false-cognate pair (node-
    edge-point vs tunnel-termination-point -- the real label_pass()
    candidate this fixture actually proposes at the default threshold,
    verified against the corpus) explicitly rejects. Every confirmed
    verdict carries a non-empty evidence_quote -- PairResult.__post_init__
    refuses to construct a confirmed verdict without one.

    The first entry is not a §6 row at all: 'tapi-topology-node' is a
    literal string prefix of 'tapi-topology-node-rule-group', so
    scripted_client's substring-containment lookup would otherwise resolve
    node-rule-group's real misses-recovery candidate against
    ietf-network-node (node-rule-group has zero label-pass candidates of
    its own, so recovery evaluates it against every IETF entry, including
    plain 'node') to the node_t/node_i verdict below -- a fifth,
    unintended close-match. Scripting that exact pair explicitly, and
    first in dict-iteration order, intercepts it before the vaguer
    node_t/node_i key can (verified empirically against the real fixture:
    omitting this entry emits 5 correspondences, not 4)."""
    return {
        (ids["nrg_t"], ids["node_i"]): align_lexicons.MatchVerdict(
            verdict="reject",
            rationale=(
                "Not a §6 row -- an explicit override for the "
                "'tapi-topology-node' / 'tapi-topology-node-rule-group' "
                "substring collision in scripted_client's lookup; see this "
                "function's docstring."
            ),
            evidence_quote="",
        ),
        (ids["link_t"], ids["link_i"]): align_lexicons.MatchVerdict(
            verdict="confirm_exact_match",
            rationale="TAPI Link and IETF link are the same optical-domain link concept.",
            evidence_quote="A link represents a physical or logical connection.",
        ),
        (ids["node_t"], ids["node_i"]): align_lexicons.MatchVerdict(
            verdict="confirm_close_match",
            rationale="TAPI node and IETF node/te-node correspond, per §6's own worked example.",
            evidence_quote="A node represents a set of managed resources.",
        ),
        (ids["nep_t"], ids["term_i"]): align_lexicons.MatchVerdict(
            verdict="confirm_close_match",
            rationale="TAPI node-edge-point and IETF termination-point correspond.",
            evidence_quote="A node edge point represents a point of termination on a node.",
        ),
        (ids["nrg_t"], ids["cm_i"]): align_lexicons.MatchVerdict(
            verdict="confirm_close_match",
            rationale=(
                "Named too differently for the label stage to pair -- "
                "recovered from their definitions (§6)."
            ),
            evidence_quote="A node rule group represents a set of constraints on connectivity.",
        ),
        (ids["nep_t"], ids["ttp_i"]): align_lexicons.MatchVerdict(
            verdict="reject",
            rationale=(
                "The tunnel-termination-point name-matches node-edge-point but "
                "is the head of a tunnel, not a link attachment -- a false "
                "cognate (§6)."
            ),
            evidence_quote="A tunnel termination point can terminate a tunnel.",
        ),
    }


def _run_otn_worked_example(scripted_client, monkeypatch, tmp_path, capsys):
    """Drives main() once with a client scripted to §6's verdicts and the
    emit flag pointed at a tmp_path file, returning (output_text,
    captured_stdout, ids) so each of this task's five tests asserts on the
    same single run without repeating the drive boilerplate."""
    ids = _otn_worked_example_ids()
    client = scripted_client(_otn_worked_example_verdicts(ids))

    output_path = tmp_path / "correspondences.ttl"
    monkeypatch.setattr(
        sys, "argv", ["align_lexicons.py", "--emit-correspondences", str(output_path)]
    )
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: client)

    align_lexicons.main()

    captured = capsys.readouterr()
    output_text = output_path.read_text()
    return output_text, captured.out, ids


def test_otn_worked_example_emits_exactly_four_correspondences(
    scripted_client, monkeypatch, tmp_path, capsys
):
    output_text, _, _ = _run_otn_worked_example(scripted_client, monkeypatch, tmp_path, capsys)
    base_section = output_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]
    annotation_section = output_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[1]

    graph = Graph()
    graph.parse(data=base_section, format="turtle")
    match_triples = list(graph.triples((None, align_lexicons.SKOS.exactMatch, None))) + list(
        graph.triples((None, align_lexicons.SKOS.closeMatch, None)))
    assert len(match_triples) == 4

    # Every emitted correspondence carries a confidence tier -- none is
    # tier-less.
    assert annotation_section.count("lex:confidenceTier") == 4


def test_link_pair_is_an_exact_match_and_the_other_three_are_close(
    scripted_client, monkeypatch, tmp_path, capsys
):
    output_text, _, ids = _run_otn_worked_example(scripted_client, monkeypatch, tmp_path, capsys)
    base_section = output_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]

    graph = Graph()
    graph.parse(data=base_section, format="turtle")

    exact_triples = list(graph.triples((None, align_lexicons.SKOS.exactMatch, None)))
    close_triples = list(graph.triples((None, align_lexicons.SKOS.closeMatch, None)))
    assert len(exact_triples) == 1
    assert len(close_triples) == 3

    LEX = align_lexicons.LEX
    link_subj, _, link_obj = exact_triples[0]
    assert link_subj == LEX[ids["link_t"]]
    assert link_obj == LEX[ids["link_i"]]

    close_pairs = {(subj, obj) for subj, _, obj in close_triples}
    assert close_pairs == {
        (LEX[ids["node_t"]], LEX[ids["node_i"]]),
        (LEX[ids["nep_t"]], LEX[ids["term_i"]]),
        (LEX[ids["nrg_t"]], LEX[ids["cm_i"]]),
    }


def test_gap_and_false_cognate_are_absent_from_the_artifact(
    scripted_client, monkeypatch, tmp_path, capsys
):
    output_text, _, ids = _run_otn_worked_example(scripted_client, monkeypatch, tmp_path, capsys)
    # Checked over the whole artifact text, including the annotation
    # section below the separator -- an annotation block naming a rejected
    # or gap entry would be just as wrong as a base triple naming it.
    assert ids["sip_t"] not in output_text
    assert ids["ttp_i"] not in output_text


def test_gap_and_false_cognate_are_present_in_the_gap_report(
    scripted_client, monkeypatch, tmp_path, capsys
):
    """Both entries are surfaced in the same run's stdout rather than
    silently lost. SIP is a TAPI entry -- collect_gap_records()'s own
    primary key -- so it appears inside the printed '=== Gap report ==='
    block itself. TTP is IETF-only; collect_gap_records() iterates TAPI
    entries only, so no GapRecord can ever name it -- it surfaces instead
    via its rejected pair-transcript, printed earlier in the same run's
    stdout."""
    _, stdout, ids = _run_otn_worked_example(scripted_client, monkeypatch, tmp_path, capsys)
    assert "=== Gap report ===" in stdout
    gap_section = stdout.split("=== Gap report ===")[1].split("=== Run summary ===")[0]
    assert ids["sip_t"] in gap_section
    assert ids["ttp_i"] in stdout


def _build_two_reviewed_pairs(fixture_entries, tmp_path):
    """Shared setup for Phase 5 Plan 05-01 Task 2: two confirmed
    correspondences written as a correspondences.ttl on disk, plus their
    worklist rows, ready for a test to mark one row's verdict and drive
    write_reviewed_correspondences()."""
    _, _, by_lex_id = fixture_entries
    exact = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match"
    )
    close = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
        confidence=_confidence(tier="medium"),
    )
    results = [exact, close]
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))
    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])
    return corr_path, rows, triples


def _mark_row(worklist_text: str, row_id: str, *, verdict: str, reason: str = "") -> str:
    """Same hand-edit helper as test_review_worklist.py's own
    _mark_row_verdict -- duplicated locally rather than imported so this
    file has no test-module-to-test-module dependency."""
    escaped_reason = align_lexicons.escape_worklist_cell(reason)
    lines = worklist_text.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            out.append(line)
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) != len(align_lexicons.WORKLIST_COLUMNS) or cells[0] != row_id:
            out.append(line)
            continue
        cells[align_lexicons.WORKLIST_COLUMNS.index("verdict")] = verdict
        cells[align_lexicons.WORKLIST_COLUMNS.index("reason")] = escaped_reason
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + ("\n" if worklist_text.endswith("\n") else "")


# ── Phase 5 Plan 05-01 Task 2: keep-the-triple, idempotency and provenance
# guards on the review pass ─────────────────────────────────────────────


def test_rejected_correspondence_keeps_its_base_triple(fixture_entries, tmp_path, lexicon_dir):
    corr_path, rows, triples = _build_two_reviewed_pairs(fixture_entries, tmp_path)
    reviewed_row = rows[0]
    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked = _mark_row(worklist_text, reviewed_row.row_id, verdict="reject", reason="Reviewer disagrees.")
    records, version, model = align_lexicons.parse_review_worklist(marked)

    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )

    reviewed_text = corr_path.read_text()
    base_section = reviewed_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]
    graph = Graph()
    graph.parse(data=base_section, format="turtle")
    match_triples = list(graph.triples((None, align_lexicons.SKOS.exactMatch, None))) + list(
        graph.triples((None, align_lexicons.SKOS.closeMatch, None))
    )
    assert len(match_triples) == 2  # both base triples still present

    annotation_section = reviewed_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[1]
    assert 'lex:reviewVerdict "rejected"' in annotation_section


def test_uncertain_correspondence_keeps_its_base_triple(fixture_entries, tmp_path, lexicon_dir):
    corr_path, rows, triples = _build_two_reviewed_pairs(fixture_entries, tmp_path)
    reviewed_row = rows[0]
    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked = _mark_row(worklist_text, reviewed_row.row_id, verdict="uncertain", reason="Not sure.")
    records, version, model = align_lexicons.parse_review_worklist(marked)

    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )

    reviewed_text = corr_path.read_text()
    base_section = reviewed_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]
    graph = Graph()
    graph.parse(data=base_section, format="turtle")
    match_triples = list(graph.triples((None, align_lexicons.SKOS.exactMatch, None))) + list(
        graph.triples((None, align_lexicons.SKOS.closeMatch, None))
    )
    assert len(match_triples) == 2

    annotation_section = reviewed_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[1]
    assert 'lex:reviewVerdict "uncertain"' in annotation_section


def test_base_section_still_parses_after_review_splice(fixture_entries, tmp_path, lexicon_dir):
    corr_path, rows, triples = _build_two_reviewed_pairs(fixture_entries, tmp_path)
    original_base = corr_path.read_text().split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]
    original_graph = Graph()
    original_graph.parse(data=original_base, format="turtle")
    original_count = len(
        list(original_graph.triples((None, align_lexicons.SKOS.exactMatch, None)))
    ) + len(list(original_graph.triples((None, align_lexicons.SKOS.closeMatch, None))))

    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked = _mark_row(worklist_text, rows[0].row_id, verdict="reject", reason="x")
    records, version, model = align_lexicons.parse_review_worklist(marked)
    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )

    reviewed_text = corr_path.read_text()
    # Same diff-confined-to-annotations property, asserted directly: the
    # base section (everything above the separator) is byte-identical.
    reviewed_base = reviewed_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]
    assert reviewed_base == original_base

    graph = Graph()
    graph.parse(data=reviewed_base, format="turtle")
    reviewed_count = len(
        list(graph.triples((None, align_lexicons.SKOS.exactMatch, None)))
    ) + len(list(graph.triples((None, align_lexicons.SKOS.closeMatch, None))))
    assert reviewed_count == original_count


def test_applying_the_same_worklist_twice_is_refused(fixture_entries, tmp_path, lexicon_dir):
    corr_path, rows, triples = _build_two_reviewed_pairs(fixture_entries, tmp_path)
    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked = _mark_row(worklist_text, rows[0].row_id, verdict="accept", reason="ok")
    records, version, model = align_lexicons.parse_review_worklist(marked)

    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )
    bytes_after_first = corr_path.read_bytes()

    with pytest.raises(align_lexicons.AlreadyReviewedError, match=rows[0].row_id):
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )

    assert corr_path.read_bytes() == bytes_after_first


def test_multiline_evidence_quote_and_validator_argument_survive_review_splice(
    fixture_entries, tmp_path, lexicon_dir
):
    """CR-01 regression (05-REVIEW.md): render_correspondences_ttl()
    legitimately renders a multi-line evidence_quote/validator_counter_argument
    as a raw embedded newline inside a triple-quoted N3 literal (rdflib's own
    Literal(...).n3() behavior -- see
    test_multiline_evidence_quote_stays_contiguous_regardless_of_order). The
    Phase 5 splice path (_locate_annotation_block/apply_review_to_correspondences)
    must locate this block's real boundary correctly even though a naive
    line-based "does this line end with a period" scan mistakes the embedded
    newline for real Turtle-star block structure, silently splicing the
    review predicates INSIDE the still-open literal instead of after it."""
    _, _, by_lex_id = fixture_entries
    multiline_quote = "First sentence ends here.\nSecond line continues the quote and matters too."
    multiline_counter_argument = "The strongest case still fails.\nSee the second line too."
    multiline_pair = _pair_result(
        by_lex_id["tapi-topology-node"],
        by_lex_id["ietf-network-node"],
        "confirm_exact_match",
        evidence_quote=multiline_quote,
        confidence=_confidence(validator_counter_argument=multiline_counter_argument),
    )
    other_pair = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
        confidence=_confidence(tier="medium"),
    )
    triples = align_lexicons.correspondences_from_results(
        [multiline_pair, other_pair], FAKE_VERSION, FAKE_MODEL
    )
    corr_path = tmp_path / "correspondences.ttl"
    original_text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    corr_path.write_text(original_text)
    # Confirms the fixture reproduces the raw-newline literal shape CR-01
    # depends on before the splice is even attempted.
    assert multiline_quote in original_text

    multiline_triple = next(t for t in triples if t.tapi_lex_id == "tapi-topology-node")
    other_triple = next(t for t in triples if t.tapi_lex_id == "tapi-topology-node-rule-group")

    records = [
        align_lexicons.ReviewRecord(
            row_id=f"C:{multiline_triple.tapi_lex_id}:{multiline_triple.ietf_lex_id}:{multiline_triple.predicate}",
            kind="correspondence",
            tapi_lex_id=multiline_triple.tapi_lex_id,
            ietf_lex_id=multiline_triple.ietf_lex_id,
            predicate=multiline_triple.predicate,
            verdict="reject",
            reason="Reviewer disagrees despite the multi-line evidence.",
        ),
        align_lexicons.ReviewRecord(
            row_id=f"C:{other_triple.tapi_lex_id}:{other_triple.ietf_lex_id}:{other_triple.predicate}",
            kind="correspondence",
            tapi_lex_id=other_triple.tapi_lex_id,
            ietf_lex_id=other_triple.ietf_lex_id,
            predicate=other_triple.predicate,
            verdict="accept",
            reason="ok",
        ),
    ]

    align_lexicons.write_reviewed_correspondences(corr_path, records, lexicon_dir)

    reviewed_text = corr_path.read_text()

    # The multi-line literals survive completely intact -- not torn apart,
    # not truncated by the splice's boundary detection.
    assert multiline_quote in reviewed_text
    assert multiline_counter_argument in reviewed_text
    assert (
        'lex:evidenceQuote """First sentence ends here.\n'
        'Second line continues the quote and matters too.""" ;' in reviewed_text
    )

    # Each block's review predicates landed in that block's OWN annotation
    # section -- after the closing """, never spliced inside the still-open
    # multi-line literal, and never leaking into the other block.
    multiline_header = (
        f"<<lex:{multiline_triple.tapi_lex_id} {multiline_triple.predicate} "
        f"lex:{multiline_triple.ietf_lex_id}>>"
    )
    other_header = (
        f"<<lex:{other_triple.tapi_lex_id} {other_triple.predicate} lex:{other_triple.ietf_lex_id}>>"
    )
    multiline_block = reviewed_text.split(multiline_header, 1)[1].split("<<lex:", 1)[0]
    other_block = reviewed_text.split(other_header, 1)[1]
    assert 'lex:reviewVerdict "rejected"' in multiline_block
    assert 'lex:reviewVerdict "accepted"' in other_block
    assert 'lex:reviewVerdict "accepted"' not in multiline_block
    # The block still terminates in a single period -- proof the terminator
    # line was correctly relocated to the block's real end, not truncated
    # onto an internal line of the multi-line literal (05-REVIEW.md's
    # demonstrated corruption shape).
    assert multiline_block.rstrip().endswith('.')

    # The base section (SKOS match triples) is completely untouched and
    # still parses -- the corruption CR-01 describes would break this.
    base_section = reviewed_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]
    graph = Graph()
    graph.parse(data=base_section, format="turtle")
    match_triples = list(graph.triples((None, align_lexicons.SKOS.exactMatch, None))) + list(
        graph.triples((None, align_lexicons.SKOS.closeMatch, None))
    )
    assert len(match_triples) == 2


def test_rederivation_citation_matching_multiline_evidence_quote_is_refused(
    fixture_entries, tmp_path, lexicon_dir
):
    """CR-01's second failure mode: the SC4 distinctness check calls
    _read_block_evidence_quote(), which used to parse only the single
    physical line it found -- crashing with an unhandled rdflib
    AssertionError on a multi-line evidenceQuote instead of ever reaching
    this comparison. A correct fix must reconstruct the FULL multi-line
    canonical quote so a citation that merely restates it is still caught,
    not just avoid crashing."""
    _, _, by_lex_id = fixture_entries
    multiline_quote = "First sentence ends here.\nSecond line continues the quote and matters too."
    pair = _pair_result(
        by_lex_id["tapi-topology-node"],
        by_lex_id["ietf-network-node"],
        "confirm_exact_match",
        evidence_quote=multiline_quote,
    )
    triples = align_lexicons.correspondences_from_results([pair], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))
    original_bytes = corr_path.read_bytes()
    triple = triples[0]

    record = align_lexicons.ReviewRecord(
        row_id=f"C:{triple.tapi_lex_id}:{triple.ietf_lex_id}:{triple.predicate}",
        kind="correspondence",
        tapi_lex_id=triple.tapi_lex_id,
        ietf_lex_id=triple.ietf_lex_id,
        predicate=triple.predicate,
        verdict="accept",
        reason="ok",
        re_derived=True,
        rederivation_citation=multiline_quote,  # byte-identical to the matcher's own quote
    )

    with pytest.raises(align_lexicons.MalformedWorklistError, match="byte-identical"):
        align_lexicons.write_reviewed_correspondences(corr_path, [record], lexicon_dir)

    assert corr_path.read_bytes() == original_bytes


def test_blank_line_inside_evidence_quote_is_not_mistaken_for_block_boundary(
    fixture_entries, tmp_path, lexicon_dir
):
    """CR-01 edge case beyond the review's own repro: normalize_evidence_text()
    keeps evidence_quote UNCHANGED, including a genuine blank-line paragraph
    break the model quoted verbatim. The block-boundary detector must tell
    that literal blank line apart from the real blank line
    render_correspondences_ttl() emits AFTER every block -- both are
    byte-identical "" lines; only N3 string-literal state distinguishes
    them."""
    _, _, by_lex_id = fixture_entries
    paragraph_quote = "Paragraph one.\n\nParagraph two after a blank line."
    pair = _pair_result(
        by_lex_id["tapi-topology-node"],
        by_lex_id["ietf-network-node"],
        "confirm_exact_match",
        evidence_quote=paragraph_quote,
    )
    triples = align_lexicons.correspondences_from_results([pair], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))
    triple = triples[0]

    record = align_lexicons.ReviewRecord(
        row_id=f"C:{triple.tapi_lex_id}:{triple.ietf_lex_id}:{triple.predicate}",
        kind="correspondence",
        tapi_lex_id=triple.tapi_lex_id,
        ietf_lex_id=triple.ietf_lex_id,
        predicate=triple.predicate,
        verdict="accept",
        reason="looks good",
    )

    align_lexicons.write_reviewed_correspondences(corr_path, [record], lexicon_dir)
    reviewed_text = corr_path.read_text()

    assert paragraph_quote in reviewed_text
    assert 'lex:reviewVerdict "accepted"' in reviewed_text
    assert 'lex:reviewReason "looks good"' in reviewed_text
    header = f"<<lex:{triple.tapi_lex_id} {triple.predicate} lex:{triple.ietf_lex_id}>>"
    block = reviewed_text.split(header, 1)[1]
    assert block.strip().endswith(
        'lex:reviewVerdict "accepted" ;\n    lex:reviewReason "looks good" .'
    )


def test_worklist_from_a_different_lexicon_version_is_refused(fixture_entries, tmp_path, lexicon_dir):
    corr_path, rows, triples = _build_two_reviewed_pairs(fixture_entries, tmp_path)
    original_bytes = corr_path.read_bytes()
    worklist_text = align_lexicons.render_review_worklist(rows, "c" * 40, FAKE_MODEL)
    marked = _mark_row(worklist_text, rows[0].row_id, verdict="accept", reason="ok")
    records, version, model = align_lexicons.parse_review_worklist(marked)
    assert version == "c" * 40

    with pytest.raises(align_lexicons.WorklistProvenanceMismatch, match=FAKE_VERSION):
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )

    assert corr_path.read_bytes() == original_bytes

    # Same holds for a differing model value.
    worklist_text_model = align_lexicons.render_review_worklist(rows, FAKE_VERSION, "claude-other-model")
    marked_model = _mark_row(worklist_text_model, rows[0].row_id, verdict="accept", reason="ok")
    records2, version2, model2 = align_lexicons.parse_review_worklist(marked_model)
    with pytest.raises(align_lexicons.WorklistProvenanceMismatch, match=FAKE_MODEL):
        align_lexicons.write_reviewed_correspondences(
            corr_path, records2, lexicon_dir, worklist_lexicon_version=version2, worklist_model=model2
        )
    assert corr_path.read_bytes() == original_bytes


def test_adversarial_reviewer_reason_cannot_inject_triples(fixture_entries, tmp_path, lexicon_dir):
    corr_path, rows, triples = _build_two_reviewed_pairs(fixture_entries, tmp_path)
    payload = (
        'has "quotes", a backslash \\, an embedded triple-quote """ run, '
        'and a fragment shaped like an injection: <<lex:evil skos:exactMatch '
        'lex:evil2>> lex:pwned "yes" . # lex:tapi-topology-node skos:exactMatch lex:ietf-network-node .'
    )
    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked = _mark_row(worklist_text, rows[0].row_id, verdict="reject", reason=payload)
    records, version, model = align_lexicons.parse_review_worklist(marked)

    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )

    reviewed_text = corr_path.read_text()
    base_section = reviewed_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]
    graph = Graph()
    graph.parse(data=base_section, format="turtle")
    match_triples = list(graph.triples((None, align_lexicons.SKOS.exactMatch, None)))
    assert len(match_triples) == 1
    subj, _, obj = match_triples[0]
    assert str(subj) == "http://example.org/ontology/lexicon-vocab#tapi-topology-node"
    assert str(obj) == "http://example.org/ontology/lexicon-vocab#ietf-network-node"


def test_review_write_refuses_a_path_inside_the_lexicon_directory(fixture_entries, lexicon_dir, tmp_path):
    corr_path, rows, triples = _build_two_reviewed_pairs(fixture_entries, tmp_path)
    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked = _mark_row(worklist_text, rows[0].row_id, verdict="accept", reason="ok")
    records, version, model = align_lexicons.parse_review_worklist(marked)

    inside_lexicon = lexicon_dir / "correspondences.ttl"
    with pytest.raises(ValueError):
        align_lexicons.write_reviewed_correspondences(
            inside_lexicon, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )
    assert not inside_lexicon.exists()

    lexicon_shaped = tmp_path / "tapi-topology.lexicon.ttl"
    with pytest.raises(ValueError):
        align_lexicons.write_reviewed_correspondences(
            lexicon_shaped, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )
    assert not lexicon_shaped.exists()


def test_run_summary_still_reports_confirmed_escalated_and_gap_counts_together(
    scripted_client, monkeypatch, tmp_path, capsys
):
    """Holds ROADMAP SC4 against Phase 4's changes to main() -- complements
    (does not replace) test_align_lexicons.py's pre-existing
    test_summary_reports_all_counts / test_summary_reports_validator_and_
    escalation_counts, which already hold this property for Phase 1/2's
    main()."""
    _, stdout, _ = _run_otn_worked_example(scripted_client, monkeypatch, tmp_path, capsys)
    assert "=== Run summary ===" in stdout
    for verdict in align_lexicons.ALL_VERDICTS:
        assert f"{verdict}:" in stdout
    assert "escalated pairs:" in stdout
    for reason in align_lexicons.ALL_GAP_REASONS:
        assert f"{reason}:" in stdout
