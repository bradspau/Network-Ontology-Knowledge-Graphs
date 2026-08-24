"""
Tests for align_lexicons.py.

test_confirmation_rejects_false_cognate is the tracer's end-to-end check
(Phase 1, Plan 01, Task 2): it loads the two entries from the REAL lexicon
files via the lexicon_dir fixture, drives the full pipeline with the
scripted_client fixture, and asserts on both the recorded call and the
printed transcript. Only the anthropic network boundary is substituted --
parsing, normalization, scoring, and printing all run for real. No test in
this module requires ANTHROPIC_API_KEY.
"""
import align_lexicons


def test_confirmation_rejects_false_cognate(lexicon_dir, scripted_client, capsys):
    tapi_entries = align_lexicons.load_fixture_entries(lexicon_dir, align_lexicons.FIXTURE_TAPI)
    ietf_entries = align_lexicons.load_fixture_entries(lexicon_dir, align_lexicons.FIXTURE_IETF)
    tapi_entry = tapi_entries[0]
    ietf_entry = ietf_entries[0]

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
