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
