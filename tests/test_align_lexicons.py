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
import sys

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


def test_call_budget_exceeded_carries_partial_results(fixture_entries, scripted_client):
    """CR-03 regression: a CallBudgetExceeded raised mid-run must carry
    everything already computed out with it (.partial_results/.calls_used),
    not lose it with the stack frame -- main() needs this to still print a
    transcript/summary for the partial run instead of exiting silently."""
    tapi_entries, ietf_entries, _ = fixture_entries
    candidates = align_lexicons.label_pass(tapi_entries, ietf_entries, align_lexicons.DEFAULT_LABEL_THRESHOLD)
    real_call_candidates = [c for c in candidates if align_lexicons.evidence_gate(c) is None]
    assert len(real_call_candidates) >= 2, "need at least 2 real-call candidates to prove a partial stop"

    client = scripted_client({})
    with pytest.raises(align_lexicons.CallBudgetExceeded) as exc_info:
        align_lexicons.run_confirmation_stage(client, candidates, max_calls=1)

    exc = exc_info.value
    partial = exc.partial_results
    assert isinstance(partial, list)
    assert exc.calls_used == 1
    confirmation_pass_results = [r for r in partial if r.decided_by == "confirmation-pass"]
    assert len(confirmation_pass_results) == 1, (
        "exactly the one call the budget allowed should be recorded in partial_results"
    )
    assert len(partial) < len(candidates), "partial_results must stop before the full candidate list"


def test_confirmed_verdict_with_empty_evidence_quote_downgrades(fixture_entries, scripted_client):
    """CR-03 regression: PairResult's own invariant rejects a confirmed
    verdict with an empty evidence_quote -- a structurally plausible live-
    model response (MatchVerdict.evidence_quote is typed str, not
    constrained non-empty). run_confirmation_stage must downgrade this to a
    visible insufficient_evidence result rather than letting the ValueError
    crash the whole run before any transcript/summary prints."""
    tapi_entries, ietf_entries, _ = fixture_entries
    candidates = align_lexicons.label_pass(tapi_entries, ietf_entries, align_lexicons.DEFAULT_LABEL_THRESHOLD)
    real_call_candidates = [c for c in candidates if align_lexicons.evidence_gate(c) is None]
    assert real_call_candidates, "need at least one real-call candidate"
    target = real_call_candidates[0]

    malformed_verdict = align_lexicons.MatchVerdict(
        verdict="confirm_exact_match",
        rationale="Claims a match but forgot to quote evidence.",
        evidence_quote="",
    )
    client = scripted_client({(target.tapi.lex_id, target.ietf.lex_id): malformed_verdict})

    results = align_lexicons.run_confirmation_stage(client, candidates, max_calls=len(candidates))

    downgraded = next(
        r for r in results
        if r.candidate.tapi.lex_id == target.tapi.lex_id and r.candidate.ietf.lex_id == target.ietf.lex_id
    )
    assert downgraded.verdict == "insufficient_evidence"
    assert downgraded.decided_by == "confirmation-pass"
    assert downgraded.evidence_quote == ""


# ── Task 2 (Plan 03): recover the correspondences the label stage missed ───


def test_recovers_node_rule_group_correspondent(fixture_entries, scripted_client):
    tapi_entries, ietf_entries, by_lex_id = fixture_entries
    candidates = align_lexicons.label_pass(tapi_entries, ietf_entries, align_lexicons.DEFAULT_LABEL_THRESHOLD)

    node_rule_group = by_lex_id["tapi-topology-node-rule-group"]
    connectivity_matrix = by_lex_id["ietf-network-connectivity-matrix"]
    assert (node_rule_group.lex_id, connectivity_matrix.lex_id) not in {
        (c.tapi.lex_id, c.ietf.lex_id) for c in candidates
    }, "node-rule-group <-> connectivity-matrix must NOT be a label-pass candidate -- that's the point of recovery"

    confirm_verdict = align_lexicons.MatchVerdict(
        verdict="confirm_close_match",
        rationale=(
            "Both describe a node's internal switching limitations across TE "
            "links, despite sharing no label token."
        ),
        evidence_quote="Represents a node's switching limitations",
    )
    # Every OTHER pair (label-pass and recovery alike) falls back to reject --
    # this is the "rejects every label-pass pair" client the plan specifies.
    client = scripted_client(
        {(node_rule_group.lex_id, connectivity_matrix.lex_id): confirm_verdict}
    )

    label_results = align_lexicons.run_confirmation_stage(client, candidates, max_calls=len(candidates))
    assert all(r.verdict not in ("confirm_exact_match", "confirm_close_match") for r in label_results), (
        "every label-pass candidate should have been rejected by the fallback client"
    )
    calls_used = sum(1 for r in label_results if r.decided_by == "confirmation-pass")

    recovery_results = align_lexicons.recover_misses(
        client, tapi_entries, ietf_entries, label_results, max_calls=1000, calls_used=calls_used
    )

    recovered = next(
        r
        for r in recovery_results
        if r.candidate.tapi.lex_id == node_rule_group.lex_id
        and r.candidate.ietf.lex_id == connectivity_matrix.lex_id
    )
    assert recovered.candidate.origin == "misses-recovery"
    assert recovered.verdict == "confirm_close_match"
    assert recovered.decided_by == "confirmation-pass"
    assert recovered.evidence_quote.strip() != ""

    # The scope-gap case: rejected everywhere, ends unmatched -- not forced.
    service_interface_point = by_lex_id["tapi-common-service-interface-point-tapi-common"]
    all_results = label_results + recovery_results
    sip_results = [r for r in all_results if r.candidate.tapi.lex_id == service_interface_point.lex_id]
    assert sip_results, "expected service-interface-point to be evaluated at least once"
    assert not any(r.verdict in ("confirm_exact_match", "confirm_close_match") for r in sip_results)

    # The D-03 entry is never sent to the model by the recovery pass either.
    d03_lex_id = "tapi-common-node-edge-point-event-notification"
    recorded_calls_text = "\n".join(str(v) for call in client.calls for v in _flatten(call))
    assert d03_lex_id not in recorded_calls_text


def test_recovery_candidates_are_sorted_and_share_call_budget(fixture_entries, scripted_client):
    import re

    tapi_entries, ietf_entries, by_lex_id = fixture_entries

    # No prior confirmed correspondents and nothing already paired -- every
    # TAPI x IETF pair is a recovery candidate this time, generated and
    # evaluated in deterministic (tapi.lex_id, ietf.lex_id) order.
    client = scripted_client({})  # everything falls back to reject
    recovery_results = align_lexicons.recover_misses(
        client, tapi_entries, ietf_entries, [], max_calls=1000, calls_used=0
    )
    assert len(recovery_results) == len(tapi_entries) * len(ietf_entries)
    assert all(r.candidate.origin == "misses-recovery" for r in recovery_results)

    expected_pairs = sorted((t.lex_id, i.lex_id) for t in tapi_entries for i in ietf_entries)
    # Results themselves must be in generated (sorted) order.
    assert [(r.candidate.tapi.lex_id, r.candidate.ietf.lex_id) for r in recovery_results] == expected_pairs

    # Only pairs where BOTH sides have evidence reach confirm_pair -- extract
    # the exact lex_id tokens recorded in each call's prompt text (not a raw
    # substring search, since some lex_ids are prefixes of others).
    expected_call_pairs = [
        (t, i) for (t, i) in expected_pairs if by_lex_id[t].has_evidence and by_lex_id[i].has_evidence
    ]
    recorded_call_pairs = []
    for call in client.calls:
        text = "\n".join(str(v) for v in _flatten(call))
        ids = re.findall(r"lex_id: (\S+)", text)
        assert len(ids) == 2
        recorded_call_pairs.append((ids[0], ids[1]))

    assert recorded_call_pairs == expected_call_pairs


def test_recover_misses_respects_shared_call_cap(fixture_entries, scripted_client):
    tapi_entries, ietf_entries, _ = fixture_entries
    client = scripted_client({})
    with pytest.raises(align_lexicons.CallBudgetExceeded):
        align_lexicons.recover_misses(
            client, tapi_entries, ietf_entries, [], max_calls=0, calls_used=0
        )


# ── Task 3 (Plan 03): the full auditable transcript and run summary ────────


def test_full_fixture_run_no_crash(recording_client, capsys, monkeypatch):
    """The smoke test: a full run over the real, un-curated lexicon files
    completes, prints exactly one transcript block per PairResult, and
    raises nothing (direct proof of ROADMAP SC1)."""
    monkeypatch.setattr(sys, "argv", ["align_lexicons.py"])
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)

    block_count = {"n": 0}
    original_print_pair_transcript = align_lexicons.print_pair_transcript

    def _counting_print_pair_transcript(result):
        block_count["n"] += 1
        original_print_pair_transcript(result)

    monkeypatch.setattr(align_lexicons, "print_pair_transcript", _counting_print_pair_transcript)

    align_lexicons.main()

    captured = capsys.readouterr()
    assert block_count["n"] > 0
    assert captured.out.count("candidate origin:") == block_count["n"]
    assert "=== Run summary ===" in captured.out

    # The D-03 entry's block(s) show the unavailable marker on its
    # definition, scope-note, and canonical-example lines -- no field is
    # ever omitted, and nothing is fabricated in its place.
    blocks = captured.out.split("\n\n")
    d03_blocks = [
        b for b in blocks if "tapi-common-node-edge-point-event-notification" in b
    ]
    assert d03_blocks, "expected at least one transcript block for the D-03 entry"
    for block in d03_blocks:
        assert block.count("(none available)") >= 3


def test_main_prints_partial_run_on_budget_exceeded(recording_client, capsys, monkeypatch):
    """CR-03 regression at the main() level: a --max-calls cap tight enough
    to trigger CallBudgetExceeded mid-run must still print a transcript
    block per result processed so far and the run summary -- not exit with
    a raw traceback and zero output. Exits non-zero (the cap is still a
    deliberate hard stop, ROADMAP SC5/threat T-01-04), but only after
    everything computed prints."""
    monkeypatch.setattr(sys, "argv", ["align_lexicons.py", "--max-calls", "1"])
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)

    with pytest.raises(SystemExit) as exc_info:
        align_lexicons.main()
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "candidate origin:" in captured.out, "partial run must still print at least one transcript block"
    assert "=== Run summary ===" in captured.out, "partial run must still print the run summary"
    assert "STOPPED EARLY" in captured.err


def test_run_does_not_modify_lexicon_files(lexicon_dir, recording_client, monkeypatch, capsys):
    """Mechanical proof of the read-only-corpus prohibition (T-01-05): every
    file under lexicon_dir is byte-identical before and after a full run."""
    import hashlib

    def _hashes():
        return {
            str(p.relative_to(lexicon_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(lexicon_dir.rglob("*"))
            if p.is_file()
        }

    before = _hashes()
    assert before, "expected at least one file under lexicon_dir to hash"

    monkeypatch.setattr(sys, "argv", ["align_lexicons.py"])
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)
    align_lexicons.main()
    capsys.readouterr()  # drain stdout; not under test here

    after = _hashes()
    assert after == before


def test_summary_reports_all_counts(recording_client, capsys, monkeypatch):
    """A run that confirms nothing (recording_client's default verdict is
    reject) must still report everything -- no success-only path."""
    fresh_summary = align_lexicons.RunSummary(
        lexicon_dir=align_lexicons.DEFAULT_LEXICON_DIR,
        model=align_lexicons.DEFAULT_MODEL,
        label_threshold=align_lexicons.DEFAULT_LABEL_THRESHOLD,
        max_calls=align_lexicons.DEFAULT_MAX_CALLS,
        tapi_entry_count=0,
        ietf_entry_count=0,
        candidates_proposed=0,
        recovery_pairs_evaluated=0,
        confirmation_calls_made=0,
    )
    # Immediately after construction, before any result is recorded, all
    # four verdict keys exist at zero.
    assert set(fresh_summary.verdict_counts.keys()) == set(align_lexicons.ALL_VERDICTS)
    assert all(count == 0 for count in fresh_summary.verdict_counts.values())

    monkeypatch.setattr(sys, "argv", ["align_lexicons.py"])
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)
    align_lexicons.main()

    captured = capsys.readouterr()
    assert "=== Run summary ===" in captured.out
    for verdict in align_lexicons.ALL_VERDICTS:
        assert f"{verdict}:" in captured.out
    # An all-reject run still confirms nothing -- the zero-valued confirm
    # counts must be visibly present, not silently omitted.
    assert "confirm_exact_match: 0" in captured.out
    assert "confirm_close_match: 0" in captured.out

    assert "candidates proposed" in captured.out
    assert "recovery pairs evaluated" in captured.out
    assert "confirmation calls made" in captured.out
