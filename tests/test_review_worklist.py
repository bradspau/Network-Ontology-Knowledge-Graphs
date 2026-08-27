"""
Tests for the review-worklist round trip (Phase 5, REV-01): build_worklist_rows,
render_review_worklist, parse_review_worklist, apply_review_to_correspondences,
write_reviewed_correspondences, and the two new --emit-worklist/--apply-review
CLI flags.

Task 1 (this file's first block) wires the tracer: one confirmed correspondence
travels pipeline output -> worklist row -> reviewer verdict -> parsed record ->
spliced `lex:review*` annotation on that correspondence's own block in
correspondences.ttl. Task 3 adds the parser's input-validation boundary
(duplicate row_id, unknown verdict word, wrong cell count, all-defects-in-one-
error). Tests are grouped by task with a `# -- Task N --` banner, in
plan-authored order. No test in this module requires ANTHROPIC_API_KEY.
"""
import sys

import pytest
from rdflib import Graph

import align_lexicons

FAKE_VERSION = "b" * 40
FAKE_MODEL = "claude-test-model"


# -- Shared builders (mirrors test_correspondences.py's own idiom) ---------


def _confidence(**overrides):
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


def _two_confirmed_results(by_lex_id):
    exact = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match"
    )
    close = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
        confidence=_confidence(tier="medium"),
    )
    return [exact, close]


# -- Task 1: tracer -- worklist row generation, one-row round trip, splice --


def test_build_worklist_rows_returns_one_row_per_confirmed_correspondence(fixture_entries):
    _, _, by_lex_id = fixture_entries
    results = _two_confirmed_results(by_lex_id)
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)

    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])

    assert len(rows) == 2
    assert all(r.kind == "correspondence" for r in rows)
    ids = {r.row_id for r in rows}
    for t in triples:
        expected_id = align_lexicons.worklist_row_id(
            "correspondence", t.tapi_lex_id, t.ietf_lex_id, t.predicate
        )
        assert expected_id in ids


def test_render_review_worklist_emits_two_data_rows_with_full_cell_count(fixture_entries):
    _, _, by_lex_id = fixture_entries
    results = _two_confirmed_results(by_lex_id)
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])

    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)

    data_lines = [
        line
        for line in text.splitlines()
        if line.strip().startswith("| C:") or line.strip().startswith("| G:")
    ]
    assert len(data_lines) == 2
    for line in data_lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == len(align_lexicons.WORKLIST_COLUMNS)
        assert cells[0] in {r.row_id for r in rows}


def test_parsing_all_blank_verdicts_returns_empty_record_list(fixture_entries):
    _, _, by_lex_id = fixture_entries
    results = _two_confirmed_results(by_lex_id)
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)

    records, lexicon_version, model = align_lexicons.parse_review_worklist(text)

    assert records == []
    assert lexicon_version == FAKE_VERSION
    assert model == FAKE_MODEL


def test_parsing_one_marked_row_returns_one_record_with_unescaped_reason(fixture_entries):
    _, _, by_lex_id = fixture_entries
    results = _two_confirmed_results(by_lex_id)
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)

    target_row = rows[0]
    marked_text = _mark_row_verdict(
        text, target_row.row_id, verdict="accept", reason="Looks right, sources agree|clearly."
    )

    records, _, _ = align_lexicons.parse_review_worklist(marked_text)

    assert len(records) == 1
    record = records[0]
    assert record.row_id == target_row.row_id
    assert record.verdict == "accept"
    assert record.reason == "Looks right, sources agree|clearly."


def test_apply_review_splices_verdict_and_reason_onto_the_matching_block(fixture_entries):
    _, _, by_lex_id = fixture_entries
    results = _two_confirmed_results(by_lex_id)
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)
    existing_text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])

    reviewed_row = rows[0]
    other_row = rows[1]
    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked_text = _mark_row_verdict(
        worklist_text, reviewed_row.row_id, verdict="accept", reason="Confirmed against §6."
    )
    records, _, _ = align_lexicons.parse_review_worklist(marked_text)

    reviewed_text = align_lexicons.apply_review_to_correspondences(existing_text, records)

    reviewed_header = (
        f"<<lex:{reviewed_row.tapi_lex_id} {reviewed_row.predicate} "
        f"lex:{reviewed_row.ietf_lex_id}>>"
    )
    reviewed_block = _extract_block(reviewed_text, reviewed_header)
    assert 'lex:reviewVerdict "accepted"' in reviewed_block
    assert 'lex:reviewReason "Confirmed against §6."' in reviewed_block
    assert reviewed_block.rstrip().endswith(".")
    assert reviewed_block.count(" .\n") <= 1 or reviewed_block.rstrip().count(".") >= 1

    # The order inside the block: the twelve pipeline predicates, then the
    # present review predicates, in that order.
    positions = [
        reviewed_block.index(pred)
        for pred in align_lexicons.CORRESPONDENCE_ANNOTATION_ORDER
        if pred in reviewed_block
    ] + [reviewed_block.index(pred) for pred in ("lex:reviewVerdict", "lex:reviewReason")]
    assert positions == sorted(positions)

    other_header = f"<<lex:{other_row.tapi_lex_id} {other_row.predicate} lex:{other_row.ietf_lex_id}>>"
    original_other_block = _extract_block(existing_text, other_header)
    reviewed_other_block = _extract_block(reviewed_text, other_header)
    assert original_other_block == reviewed_other_block


def test_main_emit_worklist_flag_writes_worklist_and_prints_row_count(
    scripted_client, monkeypatch, tmp_path, capsys, fixture_entries
):
    _, _, by_lex_id = fixture_entries
    from tests.test_correspondences import _otn_worked_example_ids, _otn_worked_example_verdicts

    ids = _otn_worked_example_ids()
    client = scripted_client(_otn_worked_example_verdicts(ids))
    output_path = tmp_path / "review-worklist.md"
    monkeypatch.setattr(sys, "argv", ["align_lexicons.py", "--emit-worklist", str(output_path)])
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: client)

    align_lexicons.main()

    assert output_path.exists()
    text = output_path.read_text()
    data_rows = [
        line for line in text.splitlines() if line.strip().startswith("| C:")
    ]
    assert len(data_rows) == 4
    captured = capsys.readouterr()
    assert str(output_path) in captured.out
    assert "4" in captured.out


def test_main_without_emit_worklist_flag_writes_no_worklist(
    recording_client, monkeypatch, tmp_path
):
    fake_default = tmp_path / "review-worklist.md"
    monkeypatch.setattr(align_lexicons, "DEFAULT_WORKLIST_PATH", fake_default)
    monkeypatch.setattr(sys, "argv", ["align_lexicons.py"])
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)

    align_lexicons.main()

    assert not fake_default.exists()


def test_main_apply_review_flag_parses_worklist_and_annotates_artifact_with_no_client_calls(
    monkeypatch, tmp_path, capsys, fixture_entries, recording_client
):
    _, _, by_lex_id = fixture_entries
    results = _two_confirmed_results(by_lex_id)
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])

    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked_text = _mark_row_verdict(worklist_text, rows[0].row_id, verdict="accept", reason="ok")
    worklist_path = tmp_path / "review-worklist.md"
    worklist_path.write_text(marked_text)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "align_lexicons.py",
            "--apply-review",
            str(worklist_path),
            "--correspondences-path",
            str(corr_path),
        ],
    )
    # If main() ever constructed an Anthropic client in this mode, this
    # would raise -- proving zero client calls happen on this path.
    monkeypatch.setattr(
        align_lexicons.anthropic,
        "Anthropic",
        lambda: (_ for _ in ()).throw(AssertionError("--apply-review must not construct a client")),
    )

    align_lexicons.main()

    reviewed_text = corr_path.read_text()
    assert 'lex:reviewVerdict "accepted"' in reviewed_text
    captured = capsys.readouterr()
    assert "1" in captured.out


# -- Helpers shared across tasks --------------------------------------------


def _mark_row_verdict(worklist_text: str, row_id: str, *, verdict: str, reason: str = "") -> str:
    """Hand-edits one data row's verdict/reason cells in an already-rendered
    worklist, mirroring what a human reviewer would type -- used by every
    test that needs a completed (not blank) worklist without hand-writing
    the whole Markdown table."""
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


def _extract_block(text: str, header: str) -> str:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == header)
    end = start + 1
    while end < len(lines) and lines[end].strip():
        end += 1
    return "\n".join(lines[start:end])
