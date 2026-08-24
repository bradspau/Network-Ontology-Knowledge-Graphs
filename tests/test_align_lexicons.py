"""
Tests for align_lexicons.py.

test_confirmation_rejects_false_cognate is the tracer's end-to-end check
(Phase 1, Plan 01, Task 2): it loads the two entries from the REAL lexicon
files via the lexicon_dir fixture, drives the full pipeline with the
scripted_client fixture, and asserts on both the recorded call and the
printed transcript. Only the anthropic network boundary is substituted --
parsing, normalization, scoring, and printing all run for real. No test in
this module requires ANTHROPIC_API_KEY.

test_handles_sparse_evidence_gracefully and test_evidence_is_stable_and_
deterministic (Phase 1, Plan 02, Task 1) exercise the full eleven-entry
curated OTN fixture via the fixture_entries pytest fixture.
"""
import pytest

import align_lexicons


def test_confirmation_rejects_false_cognate(lexicon_dir, scripted_client, capsys):
    tapi_entries = align_lexicons.load_fixture_entries(lexicon_dir, align_lexicons.FIXTURE_TAPI)
    ietf_entries = align_lexicons.load_fixture_entries(lexicon_dir, align_lexicons.FIXTURE_IETF)
    # The fixture now holds all 11 OTN entries (Plan 02) -- look the false-
    # cognate pair up by lex_id rather than assuming it's still entry [0].
    tapi_entry = next(e for e in tapi_entries if e.lex_id == "tapi-topology-node-edge-point")
    ietf_entry = next(e for e in ietf_entries if e.lex_id == "ietf-network-tunnel-termination-point-te")

    candidate = align_lexicons.Candidate(
        tapi=tapi_entry,
        ietf=ietf_entry,
        label_score=align_lexicons.label_score(tapi_entry.pref_label, ietf_entry.pref_label),
        origin="label-pass",
    )

    reject_verdict = align_lexicons.MatchVerdict(
        verdict="reject",
        rationale=(
            "The TAPI node-edge-point is the ingress-egress edge-port functions "
            "at a Node's boundary, not the head of a tunnel; the IETF entry's own "
            "definition text confirms the concepts differ."
        ),
        evidence_quote="A termination point can terminate a tunnel.",
    )
    client = scripted_client(
        {(tapi_entry.lex_id, ietf_entry.lex_id): reject_verdict}
    )

    assert align_lexicons.evidence_gate(candidate) is None  # both sides have real evidence

    verdict = align_lexicons.confirm_pair(client, candidate)
    result = align_lexicons.PairResult(
        candidate=candidate,
        verdict=verdict.verdict,
        rationale=verdict.rationale,
        evidence_quote=verdict.evidence_quote,
        decided_by="confirmation-pass",
    )
    align_lexicons.print_pair_transcript(result)

    captured = capsys.readouterr()

    # Exactly one confirmation call for the one pair.
    recorded_calls = client.calls
    assert len(recorded_calls) == 1

    prompt_text = "\n".join(
        str(v) for call in recorded_calls for v in _flatten(call)
    )
    assert "A termination point can terminate a tunnel." in prompt_text
    assert "ingress-egress edge-port functions" in prompt_text
    # The mechanical label restatement must have been discarded by
    # normalize_evidence_text, not forwarded as evidence.
    assert "Grouping definition:" not in prompt_text

    assert result.verdict == "reject"
    assert result.decided_by == "confirmation-pass"

    assert "A termination point can terminate a tunnel." in captured.out
    assert captured.out.count("(none available)") >= 3


def _flatten(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _flatten(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _flatten(v)


# ── Task 1: full eleven-entry fixture, deterministic evidence ──────────────


def test_handles_sparse_evidence_gracefully(fixture_entries):
    tapi_entries, ietf_entries, by_lex_id = fixture_entries

    d03 = by_lex_id["tapi-common-node-edge-point-event-notification"]
    assert d03.definition is None
    assert d03.scope_notes == []
    assert d03.canonical_example is None
    assert d03.has_evidence is False

    all_entries = tapi_entries + ietf_entries
    assert len(all_entries) == 11
    for entry in all_entries:
        assert entry.canonical_example is None


def test_evidence_is_stable_and_deterministic(lexicon_dir, fixture_entries):
    tapi_entries, ietf_entries, by_lex_id = fixture_entries

    ietf_node = by_lex_id["ietf-network-node"]
    assert len(ietf_node.scope_notes) == 2
    assert ietf_node.scope_notes == sorted(ietf_node.scope_notes)

    tapi_node = by_lex_id["tapi-topology-node"]
    assert tapi_node.definition is None
    assert any("abstract representation of the forwarding capabilities" in note for note in tapi_node.scope_notes)

    tapi_link = by_lex_id["tapi-topology-link"]
    assert tapi_link.definition is not None
    assert "effective adjacency" in tapi_link.definition
    assert tapi_link.scope_notes == []

    # Two consecutive loads over the real files produce equal results.
    tapi_again = align_lexicons.load_fixture_entries(lexicon_dir, align_lexicons.FIXTURE_TAPI)
    ietf_again = align_lexicons.load_fixture_entries(lexicon_dir, align_lexicons.FIXTURE_IETF)
    assert tapi_again == tapi_entries
    assert ietf_again == ietf_entries


# ── Task 2: blocked candidate generation ────────────────────────────────────


def test_label_tokens_and_blocking():
    assert align_lexicons.label_tokens("node-edge-point") == {"node", "edge", "point"}
    assert align_lexicons.label_tokens("Node_Edge_Point") == align_lexicons.label_tokens("node-edge-point")
    assert align_lexicons.label_tokens("") == set()


def test_label_pass_proposes_otn_candidates(fixture_entries):
    tapi_entries, ietf_entries, _ = fixture_entries
    candidates = align_lexicons.label_pass(tapi_entries, ietf_entries, align_lexicons.DEFAULT_LABEL_THRESHOLD)

    pairs = {(c.tapi.lex_id, c.ietf.lex_id) for c in candidates}
    assert ("tapi-topology-node", "ietf-network-node") in pairs
    assert ("tapi-topology-node-edge-point", "ietf-network-termination-point") in pairs

    for c in candidates:
        assert isinstance(c.label_score, float)
        assert c.origin == "label-pass"

    assert candidates == sorted(candidates, key=lambda c: (c.tapi.lex_id, c.ietf.lex_id))


def test_label_pass_is_bounded(fixture_entries):
    tapi_entries, ietf_entries, _ = fixture_entries

    candidates = align_lexicons.label_pass(tapi_entries, ietf_entries, align_lexicons.DEFAULT_LABEL_THRESHOLD)
    assert len(candidates) < len(tapi_entries) * len(ietf_entries)

    pairs = {(c.tapi.lex_id, c.ietf.lex_id) for c in candidates}
    assert ("tapi-topology-node-rule-group", "ietf-network-connectivity-matrix") not in pairs

    # block_candidates itself does not raise on an empty-token-set label,
    # and node-edge-point/termination-point are blocked together via "point"
    # while link/connectivity-matrix share no token.
    blocked_pairs = {
        (a.lex_id, b.lex_id)
        for a, b in align_lexicons.block_candidates(tapi_entries, ietf_entries)
    }
    assert ("tapi-topology-node-edge-point", "ietf-network-termination-point") in blocked_pairs
    assert ("tapi-topology-node-rule-group", "ietf-network-connectivity-matrix") not in blocked_pairs
    assert ("tapi-topology-link", "ietf-network-connectivity-matrix") not in blocked_pairs
    assert align_lexicons.block_candidates([], []) == []


def test_label_pass_threshold_boundary_is_inclusive(fixture_entries):
    tapi_entries, ietf_entries, _ = fixture_entries
    all_candidates = align_lexicons.label_pass(tapi_entries, ietf_entries, 0.0)
    assert all_candidates, "expected at least one blocked candidate at threshold=0"

    observed_score = all_candidates[0].label_score
    at_threshold = align_lexicons.label_pass(tapi_entries, ietf_entries, observed_score)
    at_threshold_pairs = {(c.tapi.lex_id, c.ietf.lex_id) for c in at_threshold}
    assert (all_candidates[0].tapi.lex_id, all_candidates[0].ietf.lex_id) in at_threshold_pairs


def test_no_entity_class_equality_blocking():
    import re

    src = open(align_lexicons.__file__).read()
    assert not re.search(r"entityClass\s*==", src)


# ── Task 3 (Plan 03): confirm every candidate, never on label alone ────────


def test_no_match_without_confirmation(fixture_entries, scripted_client):
    """The load-bearing assertion: even when the model is scripted to
    confirm EVERYTHING, an entry with no evidence (D-03) cannot be
    confirmed, because evidence_gate runs before the model is ever
    consulted."""
    tapi_entries, ietf_entries, _ = fixture_entries
    candidates = align_lexicons.label_pass(tapi_entries, ietf_entries, align_lexicons.DEFAULT_LABEL_THRESHOLD)
    assert candidates, "expected at least one label-pass candidate for this fixture"

    confirm_everything = align_lexicons.MatchVerdict(
        verdict="confirm_exact_match",
        rationale="Scripted confirm for every candidate pair -- test_no_match_without_confirmation.",
        evidence_quote="scripted evidence quote text",
    )
    client = scripted_client(
        {(c.tapi.lex_id, c.ietf.lex_id): confirm_everything for c in candidates}
    )

    results = align_lexicons.run_confirmation_stage(client, candidates, max_calls=len(candidates))
    assert len(results) == len(candidates)

    for result in results:
        if result.verdict in ("confirm_exact_match", "confirm_close_match"):
            assert result.decided_by == "confirmation-pass"
            assert result.evidence_quote.strip() != ""

    d03_lex_id = "tapi-common-node-edge-point-event-notification"
    d03_results = [r for r in results if r.candidate.tapi.lex_id == d03_lex_id]
    assert d03_results, "expected the D-03 entry to appear among label-pass candidates"
    for d03_result in d03_results:
        assert d03_result.verdict == "insufficient_evidence"
        assert d03_result.decided_by == "evidence-gate"

    recorded_calls_text = "\n".join(str(v) for call in client.calls for v in _flatten(call))
    assert d03_lex_id not in recorded_calls_text


def test_pair_result_rejects_confirmed_verdict_from_evidence_gate(fixture_entries):
    tapi_entries, ietf_entries, _ = fixture_entries
    candidate = align_lexicons.Candidate(
        tapi=tapi_entries[0], ietf=ietf_entries[0], label_score=100.0, origin="label-pass"
    )
    with pytest.raises(ValueError):
        align_lexicons.PairResult(
            candidate=candidate,
            verdict="confirm_exact_match",
            rationale="should never construct -- decided_by is evidence-gate",
            evidence_quote="a quote that exists but must not matter here",
            decided_by="evidence-gate",
        )


def test_run_confirmation_stage_respects_max_calls_cap(fixture_entries, scripted_client):
    tapi_entries, ietf_entries, _ = fixture_entries
    candidates = align_lexicons.label_pass(tapi_entries, ietf_entries, align_lexicons.DEFAULT_LABEL_THRESHOLD)
    assert any(
        align_lexicons.evidence_gate(c) is None for c in candidates
    ), "expected at least one candidate that needs a real confirmation call"

    client = scripted_client({})
    with pytest.raises(align_lexicons.CallBudgetExceeded):
        align_lexicons.run_confirmation_stage(client, candidates, max_calls=0)
