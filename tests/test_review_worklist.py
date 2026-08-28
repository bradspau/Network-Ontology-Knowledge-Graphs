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


# -- Task 3: worklist parser input-validation boundary ----------------------


def _rendered_worklist(fixture_entries):
    _, _, by_lex_id = fixture_entries
    results = _two_confirmed_results(by_lex_id)
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    return text, rows, triples


def _find_data_line_index(text: str, row_id: str) -> int:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.strip("|").split("|")[0].strip() == row_id:
            return i
    raise AssertionError(f"row {row_id!r} not found in worklist text")


def test_duplicate_row_id_is_a_malformed_worklist_error(fixture_entries):
    text, rows, _ = _rendered_worklist(fixture_entries)
    marked = _mark_row_verdict(text, rows[0].row_id, verdict="accept", reason="first")

    # Duplicate the first data row verbatim (same row_id, second verdict).
    lines = marked.splitlines()
    idx = _find_data_line_index(marked, rows[0].row_id)
    duplicated = lines[:idx + 1] + [lines[idx]] + lines[idx + 1:]
    duplicated_text = "\n".join(duplicated) + "\n"

    with pytest.raises(align_lexicons.MalformedWorklistError, match=rows[0].row_id):
        align_lexicons.parse_review_worklist(duplicated_text)


def test_unknown_verdict_word_is_a_malformed_worklist_error(fixture_entries):
    text, rows, _ = _rendered_worklist(fixture_entries)
    marked = _mark_row_verdict(text, rows[0].row_id, verdict="maybe", reason="bad word")

    with pytest.raises(align_lexicons.MalformedWorklistError, match="maybe"):
        align_lexicons.parse_review_worklist(marked)


def test_verdict_word_case_is_normalized_before_the_check(fixture_entries):
    text, rows, _ = _rendered_worklist(fixture_entries)
    marked = _mark_row_verdict(text, rows[0].row_id, verdict="ACCEPT", reason="capitalized")

    records, _, _ = align_lexicons.parse_review_worklist(marked)

    assert len(records) == 1
    assert records[0].verdict == "accept"


def test_missing_column_is_a_malformed_worklist_error(fixture_entries):
    text, rows, _ = _rendered_worklist(fixture_entries)
    marked = _mark_row_verdict(text, rows[0].row_id, verdict="accept", reason="x")
    idx = _find_data_line_index(marked, rows[0].row_id)
    lines = marked.splitlines()
    stripped = lines[idx].strip()
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    truncated_cells = cells[:-1]  # drop the last column
    lines[idx] = "| " + " | ".join(truncated_cells) + " |"
    truncated_text = "\n".join(lines) + "\n"

    with pytest.raises(align_lexicons.MalformedWorklistError):
        align_lexicons.parse_review_worklist(truncated_text)


def test_every_malformed_row_is_reported_in_one_error(fixture_entries):
    text, rows, _ = _rendered_worklist(fixture_entries)
    marked = _mark_row_verdict(text, rows[0].row_id, verdict="not-a-real-verdict", reason="bad")
    lines = marked.splitlines()

    # Defect 2: truncate the second row's cell count.
    idx1 = _find_data_line_index(marked, rows[1].row_id)
    stripped1 = lines[idx1].strip()
    cells1 = [c.strip() for c in stripped1.strip("|").split("|")][:-1]
    lines[idx1] = "| " + " | ".join(cells1) + " |"

    # Defect 3: duplicate the first row's (already-defective) line.
    idx0 = _find_data_line_index(marked, rows[0].row_id)
    lines = lines[: idx0 + 1] + [lines[idx0]] + lines[idx0 + 1 :]

    broken_text = "\n".join(lines) + "\n"

    with pytest.raises(align_lexicons.MalformedWorklistError) as exc_info:
        align_lexicons.parse_review_worklist(broken_text)

    message = str(exc_info.value)
    assert "not-a-real-verdict" in message
    assert message.count("\n- ") >= 2  # at least the "N defect(s)" header plus 2 bullet lines


def test_malformed_worklist_writes_nothing(fixture_entries, tmp_path, lexicon_dir):
    text, rows, triples = _rendered_worklist(fixture_entries)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))
    original_bytes = corr_path.read_bytes()

    marked = _mark_row_verdict(text, rows[0].row_id, verdict="not-a-real-verdict", reason="bad")

    with pytest.raises(align_lexicons.MalformedWorklistError):
        records, version, model = align_lexicons.parse_review_worklist(marked)
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )

    assert corr_path.read_bytes() == original_bytes


def test_row_id_absent_from_target_raises_rather_than_silently_skipped(
    fixture_entries, tmp_path, lexicon_dir
):
    text, rows, triples = _rendered_worklist(fixture_entries)
    # Write a target correspondences.ttl containing only the SECOND triple --
    # the first row's block genuinely does not exist in this target file.
    only_second = [t for t in triples if t.tapi_lex_id == rows[1].tapi_lex_id]
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(
        align_lexicons.render_correspondences_ttl(only_second, FAKE_VERSION, FAKE_MODEL)
    )
    original_bytes = corr_path.read_bytes()

    marked = _mark_row_verdict(text, rows[0].row_id, verdict="accept", reason="x")
    records, version, model = align_lexicons.parse_review_worklist(marked)

    with pytest.raises(align_lexicons.MalformedWorklistError, match=rows[0].row_id):
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )

    assert corr_path.read_bytes() == original_bytes


def test_escaped_pipe_and_newline_round_trip_through_a_reason_cell():
    original = "line one|line two\nline three"
    escaped = align_lexicons.escape_worklist_cell(original)
    assert "\n" not in escaped
    assert "|" not in escaped
    assert align_lexicons.unescape_worklist_cell(escaped) == original


def test_empty_worklist_still_emits_header_and_explicit_no_rows_line():
    text = align_lexicons.render_review_worklist([], FAKE_VERSION, FAKE_MODEL)

    assert "| " + " | ".join(align_lexicons.WORKLIST_COLUMNS) + " |" in text
    assert any(line.strip().startswith("_") for line in text.splitlines())

    records, lexicon_version, model = align_lexicons.parse_review_worklist(text)
    assert records == []
    assert lexicon_version == FAKE_VERSION
    assert model == FAKE_MODEL


def test_correspondence_rows_are_order_independent_and_rank_key_sorted(fixture_entries):
    """Plan 05-02 supersedes Plan 01's naive lex-id-only ordering
    assumption: row order is now driven by the full rank key
    (<ranking_contract>), which orders by tier before lex ids -- a
    medium-tier row sorts ahead of a high-tier row regardless of lex id.
    The order-independence property this test originally proved (two runs
    over reordered input produce identical row order) still holds and is
    still the thing worth proving; only the "expected order" derivation
    changes to reflect the real ranking contract instead of a bare lex-id
    sort."""
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
    forward = align_lexicons.correspondences_from_results([exact, close], FAKE_VERSION, FAKE_MODEL)
    reordered = align_lexicons.correspondences_from_results([close, exact], FAKE_VERSION, FAKE_MODEL)

    forward_rows = align_lexicons.build_worklist_rows(
        forward, [exact, close], gap_records=[]
    )
    reordered_rows = align_lexicons.build_worklist_rows(
        reordered, [close, exact], gap_records=[]
    )

    assert [r.row_id for r in forward_rows] == [r.row_id for r in reordered_rows]
    expected_order = sorted(forward_rows, key=align_lexicons.worklist_rank_key)
    assert [r.row_id for r in forward_rows] == [r.row_id for r in expected_order]
    # The medium-tier row ranks ahead of the high-tier row even though its
    # tapi_lex_id sorts lexicographically after it -- proves tier, not lex
    # id, is the dominant ranking component.
    assert forward_rows[0].tier == "medium"
    assert forward_rows[1].tier == "high"


def test_worklist_contains_no_absolute_path_or_environment_value(fixture_entries, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret-value-should-never-leak")
    text, _, _ = _rendered_worklist(fixture_entries)

    assert "sk-super-secret-value-should-never-leak" not in text
    assert "ANTHROPIC_API_KEY" not in text
    assert "/Users/" not in text
    assert "/home/" not in text


# -- Plan 05-02 Task 1: ranking -- low tier first, escalations ahead -------


def test_tier_rank_is_derived_from_confidence_tiers():
    expected = {tier: i for i, tier in enumerate(reversed(align_lexicons.CONFIDENCE_TIERS))}
    assert align_lexicons.TIER_RANK == expected
    assert (
        align_lexicons.TIER_RANK["low"]
        < align_lexicons.TIER_RANK["medium"]
        < align_lexicons.TIER_RANK["high"]
    )


def test_gap_reason_rank_is_derived_from_all_gap_reasons():
    expected_order = [
        r for r in align_lexicons.ALL_GAP_REASONS if r != "insufficient-evidence"
    ] + ["insufficient-evidence"]
    expected = {reason: i for i, reason in enumerate(expected_order)}
    assert align_lexicons.GAP_REASON_RANK == expected
    assert sorted(align_lexicons.GAP_REASON_RANK) == sorted(align_lexicons.ALL_GAP_REASONS)
    assert (
        max(align_lexicons.GAP_REASON_RANK, key=align_lexicons.GAP_REASON_RANK.get)
        == "insufficient-evidence"
    )


def test_evidence_strength_counts_three_signals():
    full = _confidence(
        label_definition_agreement=True,
        structural_corroboration=align_lexicons.STRUCTURAL_SIGNAL_FLOOR,
        validator_ran=True,
        validator_agrees=True,
        escalated=False,
        tier="high",
    )
    assert align_lexicons.evidence_strength(full) == 3

    none_signals = align_lexicons.ConfidenceBreakdown(
        label_definition_agreement=False,
        structural_corroboration=None,
        validator_ran=False,
        validator_agrees=None,
        validator_counter_argument=None,
        escalated=False,
        tier="low",
    )
    assert align_lexicons.evidence_strength(none_signals) == 0
    assert align_lexicons.evidence_strength(None) == 0


def test_rows_run_low_then_medium_then_high(fixture_entries):
    _, _, by_lex_id = fixture_entries
    high = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="high"),
    )
    medium = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
        confidence=_confidence(tier="medium"),
    )
    low = _pair_result(
        by_lex_id["tapi-topology-link"], by_lex_id["ietf-network-link"], "confirm_close_match",
        confidence=_confidence(tier="low", structural_corroboration=None),
    )
    results = [high, medium, low]
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)

    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])

    assert [r.tier for r in rows] == ["low", "medium", "high"]


def test_escalated_correspondence_precedes_its_uncontested_peer(fixture_entries):
    _, _, by_lex_id = fixture_entries
    uncontested = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="medium"),
    )
    escalated = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
        confidence=_confidence(tier="medium", validator_agrees=False, escalated=True),
    )
    results = [uncontested, escalated]
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)

    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])

    assert [r.tapi_lex_id for r in rows] == [
        escalated.candidate.tapi.lex_id,
        uncontested.candidate.tapi.lex_id,
    ]


def test_rank_key_contains_no_float(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="high", structural_corroboration=0.1429),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])

    key = align_lexicons.worklist_rank_key(rows[0])

    def _flatten(value):
        if isinstance(value, tuple):
            for v in value:
                yield from _flatten(v)
        else:
            yield value

    assert all(not isinstance(c, float) for c in _flatten(key))


def test_identical_rank_prefix_is_separated_by_lex_id_tie_break(fixture_entries):
    _, _, by_lex_id = fixture_entries
    a = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="high"),
    )
    b = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"],
        by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match",
        confidence=_confidence(tier="high"),
    )
    results = [a, b]
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)

    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])

    assert len(rows) == 2
    assert rows[0].tapi_lex_id < rows[1].tapi_lex_id


def test_two_renders_of_identical_rows_are_byte_identical(fixture_entries):
    _, _, by_lex_id = fixture_entries
    results = _two_confirmed_results(by_lex_id)
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, results, gap_records=[])

    text1 = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    text2 = align_lexicons.render_review_worklist(list(rows), FAKE_VERSION, FAKE_MODEL)

    assert text1 == text2


def test_empty_tier_and_reason_buckets_render_explicit_zeros(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="high"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])

    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)

    assert "- low: 0" in text
    assert "- medium: 0" in text
    assert "- high: 1" in text
    for reason in align_lexicons.ALL_GAP_REASONS:
        assert f"- {reason}: 0" in text


# -- Plan 05-02 Task 2: gap rows share the worklist; reviewed gaps persist --
# -- as plain Turtle --------------------------------------------------------


def _gap_record(entry, gap_reason="structural", evaluated_against=None, best_label_score=10.0,
                 best_structural_score=None):
    return align_lexicons.GapRecord(
        entry=entry,
        gap_reason=gap_reason,
        best_label_score=best_label_score,
        best_structural_score=best_structural_score,
        evaluated_against=evaluated_against if evaluated_against is not None else ["ietf-network-node"],
        deciding_signals=["definition-text"],
    )


def test_gap_rows_precede_every_correspondence_row(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="low"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    gap = _gap_record(by_lex_id["tapi-topology-link"])

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[gap])

    assert rows[0].kind == "gap"
    assert rows[1].kind == "correspondence"


def test_insufficient_evidence_gaps_rank_last_among_gaps(fixture_entries):
    _, _, by_lex_id = fixture_entries
    structural_gap = _gap_record(by_lex_id["tapi-topology-link"], gap_reason="structural")
    insufficient_gap = _gap_record(
        by_lex_id["tapi-topology-node-rule-group"],
        gap_reason="insufficient-evidence",
        evaluated_against=[],
    )

    rows = align_lexicons.build_worklist_rows([], [], gap_records=[structural_gap, insufficient_gap])

    assert [r.gap_reason for r in rows] == ["structural", "insufficient-evidence"]


def test_row_count_equals_correspondences_plus_gaps(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="medium"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    gap = _gap_record(by_lex_id["tapi-topology-link"])

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[gap])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)

    assert len(rows) == len(triples) + 1
    data_lines = [
        line for line in text.splitlines()
        if line.strip().startswith("| C:") or line.strip().startswith("| G:")
    ]
    assert len(data_lines) == len(rows)


def test_gap_row_shape(fixture_entries):
    _, _, by_lex_id = fixture_entries
    entry = by_lex_id["tapi-topology-link"]
    gap = _gap_record(entry, gap_reason="ontological-content")

    rows = align_lexicons.build_worklist_rows([], [], gap_records=[gap])

    row = rows[0]
    assert row.kind == "gap"
    assert row.gap_reason == "ontological-content"
    assert row.tier == ""
    assert row.escalated is None
    assert row.ietf_lex_id is None
    assert row.ietf_label is None
    assert row.predicate is None
    assert row.row_id == align_lexicons.worklist_row_id("gap", entry.lex_id)


def test_main_emit_worklist_row_count_equals_correspondences_plus_gaps(
    scripted_client, monkeypatch, tmp_path, fixture_entries
):
    from tests.test_correspondences import _otn_worked_example_ids, _otn_worked_example_verdicts

    ids = _otn_worked_example_ids()
    client = scripted_client(_otn_worked_example_verdicts(ids))
    output_path = tmp_path / "review-worklist.md"
    monkeypatch.setattr(sys, "argv", ["align_lexicons.py", "--emit-worklist", str(output_path)])
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: client)

    align_lexicons.main()

    text = output_path.read_text()
    correspondence_rows = [l for l in text.splitlines() if l.strip().startswith("| C:")]
    gap_rows = [l for l in text.splitlines() if l.strip().startswith("| G:")]
    # FIXTURE_TAPI has 6 entries; the OTN-worked-example client script
    # confirms 4 of them, leaving exactly 2 as gaps.
    assert len(correspondence_rows) == 4
    assert len(gap_rows) == 2


def test_reviewed_gap_is_written_as_plain_turtle_in_the_base_section(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="high"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    existing_text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    gap_entry = by_lex_id["tapi-topology-link"]
    gap = _gap_record(gap_entry, gap_reason="ontological-content")

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[gap])
    gap_row = next(r for r in rows if r.kind == "gap")
    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked = _mark_row_verdict(
        worklist_text, gap_row.row_id, verdict="accept", reason="Genuinely no counterpart."
    )

    records, _, _ = align_lexicons.parse_review_worklist(marked)
    reviewed_text = align_lexicons.apply_review_to_correspondences(existing_text, records)

    assert reviewed_text.index("lex:ReviewedGap") < reviewed_text.index(
        align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR
    )
    base_section = reviewed_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]
    assert 'lex:gapReason "ontological-content"' in base_section
    assert 'lex:reviewVerdict "accepted"' in base_section


def test_reviewed_gap_is_never_a_skos_match_triple(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="high"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    existing_text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    gap_entry = by_lex_id["tapi-topology-link"]
    gap = _gap_record(gap_entry)
    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[gap])
    gap_row = next(r for r in rows if r.kind == "gap")
    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked = _mark_row_verdict(worklist_text, gap_row.row_id, verdict="reject", reason="probably missed")

    records, _, _ = align_lexicons.parse_review_worklist(marked)
    reviewed_text = align_lexicons.apply_review_to_correspondences(existing_text, records)

    gap_subject = f"{align_lexicons.REVIEWED_GAP_SUBJECT_PREFIX}{gap_entry.lex_id}"
    for line in reviewed_text.splitlines():
        if gap_subject in line:
            assert "skos:exactMatch" not in line
            assert "skos:closeMatch" not in line


def test_base_section_parses_after_reviewed_gap_splice(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="high"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    existing_text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    gap_entry = by_lex_id["tapi-topology-link"]
    gap = _gap_record(gap_entry)
    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[gap])
    gap_row = next(r for r in rows if r.kind == "gap")
    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked = _mark_row_verdict(worklist_text, gap_row.row_id, verdict="uncertain", reason="need a second look")

    records, _, _ = align_lexicons.parse_review_worklist(marked)
    reviewed_text = align_lexicons.apply_review_to_correspondences(existing_text, records)

    base_section = reviewed_text.split(align_lexicons.CORRESPONDENCE_ANNOTATION_SEPARATOR)[0]
    graph = Graph()
    graph.parse(data=base_section, format="turtle")
    assert any(str(p).endswith("gapReason") for _, p, _ in graph)


def test_applying_a_worklist_to_an_already_reviewed_gap_file_is_refused(
    fixture_entries, tmp_path, lexicon_dir
):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="high"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))
    gap_entry = by_lex_id["tapi-topology-link"]
    gap = _gap_record(gap_entry)
    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[gap])
    gap_row = next(r for r in rows if r.kind == "gap")
    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked = _mark_row_verdict(worklist_text, gap_row.row_id, verdict="accept", reason="genuine gap")

    records, version, model = align_lexicons.parse_review_worklist(marked)
    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )

    with pytest.raises(align_lexicons.AlreadyReviewedError):
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )


# -- Plan 05-02 Task 3: independent re-derivation columns, and the SC4 gate -


def _high_tier_result(by_lex_id):
    return _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="high"),
        evidence_quote="A node represents a set of managed resources.",
    )


def _mark_rederivation(worklist_text, row_id, *, verdict, reason="", re_derived="", citation=""):
    escaped_citation = align_lexicons.escape_worklist_cell(citation)
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
        cells[align_lexicons.WORKLIST_COLUMNS.index("re_derived")] = re_derived
        cells[align_lexicons.WORKLIST_COLUMNS.index("rederivation_citation")] = escaped_citation
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + ("\n" if worklist_text.endswith("\n") else "")


def _tamper_tier_cell(worklist_text, row_id, tier):
    """Mirrors _mark_rederivation()'s structure exactly but touches ONLY the
    `tier` display cell -- used to prove a hand-edit to that one cell is the
    entire tamper under test (05-06 gap closure, CR-02 reproduction)."""
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
        cells[align_lexicons.WORKLIST_COLUMNS.index("tier")] = tier
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + ("\n" if worklist_text.endswith("\n") else "")


def test_high_tier_row_carries_rederivation_columns(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])

    row = rows[0]
    assert row.re_derived == "N"
    assert row.rederivation_citation == ""
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    data_line = next(l for l in text.splitlines() if l.strip().startswith(f"| {row.row_id}"))
    cells = [c.strip() for c in data_line.strip().strip("|").split("|")]
    assert cells[align_lexicons.WORKLIST_COLUMNS.index("re_derived")] == "N"
    assert cells[align_lexicons.WORKLIST_COLUMNS.index("rederivation_citation")] == ""


def test_non_high_tier_row_has_no_rederivation_columns(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="medium"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])

    row = rows[0]
    assert row.re_derived is None
    assert row.rederivation_citation is None
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    data_line = next(l for l in text.splitlines() if l.strip().startswith(f"| {row.row_id}"))
    cells = [c.strip() for c in data_line.strip().strip("|").split("|")]
    assert cells[align_lexicons.WORKLIST_COLUMNS.index("re_derived")] == align_lexicons.WORKLIST_EMPTY_CELL
    assert (
        cells[align_lexicons.WORKLIST_COLUMNS.index("rederivation_citation")]
        == align_lexicons.WORKLIST_EMPTY_CELL
    )


def test_high_tier_accept_without_rederivation_is_refused(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]

    marked = _mark_rederivation(text, row.row_id, verdict="accept", reason="ok", re_derived="N", citation="")
    with pytest.raises(align_lexicons.MalformedWorklistError, match=row.row_id):
        align_lexicons.parse_review_worklist(marked)

    marked2 = _mark_rederivation(
        text, row.row_id, verdict="accept", reason="ok", re_derived="Y", citation=""
    )
    with pytest.raises(align_lexicons.MalformedWorklistError, match=row.row_id):
        align_lexicons.parse_review_worklist(marked2)


def test_high_tier_accept_with_citation_parses_cleanly(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]

    marked = _mark_rederivation(
        text, row.row_id, verdict="accept", reason="ok", re_derived="Y",
        citation="ietf-network.yang line 42: 'node' container definition.",
    )

    records, _, _ = align_lexicons.parse_review_worklist(marked)

    assert len(records) == 1
    assert records[0].re_derived is True
    assert records[0].rederivation_citation == "ietf-network.yang line 42: 'node' container definition."


def test_medium_tier_accept_with_empty_rederivation_cells_parses_cleanly(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="medium"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]

    marked = _mark_row_verdict(text, row.row_id, verdict="accept", reason="ok")

    records, _, _ = align_lexicons.parse_review_worklist(marked)

    assert len(records) == 1
    assert records[0].re_derived is None
    assert records[0].rederivation_citation is None


def test_high_tier_reject_with_no_citation_parses_cleanly(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]

    marked = _mark_row_verdict(text, row.row_id, verdict="reject", reason="not convinced")

    records, _, _ = align_lexicons.parse_review_worklist(marked)

    assert len(records) == 1
    assert records[0].verdict == "reject"


def test_high_tier_accept_with_citation_equal_to_evidence_quote_is_refused(
    fixture_entries, tmp_path, lexicon_dir
):
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    existing_text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    corr_path.write_text(existing_text)
    original_bytes = corr_path.read_bytes()

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]

    marked = _mark_rederivation(
        text, row.row_id, verdict="accept", reason="ok", re_derived="Y",
        citation=result.evidence_quote,
    )

    records, version, model = align_lexicons.parse_review_worklist(marked)
    with pytest.raises(align_lexicons.MalformedWorklistError, match=row.row_id):
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )
    assert corr_path.read_bytes() == original_bytes


def test_high_tier_acceptance_splices_rederivation_predicates(fixture_entries, tmp_path, lexicon_dir):
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]
    marked = _mark_rederivation(
        text, row.row_id, verdict="accept", reason="ok", re_derived="Y",
        citation="ietf-network.yang: node container, independently checked against source.",
    )
    records, version, model = align_lexicons.parse_review_worklist(marked)

    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )

    reviewed_text = corr_path.read_text()
    assert 'lex:reviewRederived "true"^^<http://www.w3.org/2001/XMLSchema#boolean>' in reviewed_text
    assert (
        'lex:rederivedFrom "ietf-network.yang: node container, independently checked against source."'
        in reviewed_text
    )


def test_no_citation_omits_rederived_from_predicate(fixture_entries, tmp_path, lexicon_dir):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="medium"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]
    marked = _mark_row_verdict(text, row.row_id, verdict="accept", reason="ok")
    records, version, model = align_lexicons.parse_review_worklist(marked)

    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )

    reviewed_text = corr_path.read_text()
    assert "lex:rederivedFrom" not in reviewed_text


# -- Plan 05-06 gap closure Task 1: SC4 tier authority moves from the -------
# -- worklist's display cell to the target artifact's own recorded ---------
# -- lex:confidenceTier (CR-02 reproduction, 05-VERIFICATION.md) -----------


def test_read_block_confidence_tier_reads_a_high_tier_block(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    lines = text.splitlines()
    triple = triples[0]

    header_idx, terminator_idx = align_lexicons._locate_annotation_block(
        lines, triple.tapi_lex_id, triple.predicate, triple.ietf_lex_id
    )

    assert align_lexicons._read_block_confidence_tier(lines, header_idx, terminator_idx) == "high"


def test_read_block_confidence_tier_reads_a_medium_tier_block(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="medium"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    lines = text.splitlines()
    triple = triples[0]

    header_idx, terminator_idx = align_lexicons._locate_annotation_block(
        lines, triple.tapi_lex_id, triple.predicate, triple.ietf_lex_id
    )

    assert align_lexicons._read_block_confidence_tier(lines, header_idx, terminator_idx) == "medium"


def test_read_block_confidence_tier_returns_none_when_the_predicate_is_absent(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    # Simulate a hand-edited/corrupted artifact whose block never recorded a
    # tier at all -- _annotation_fields() always emits this predicate for a
    # generator-produced artifact, so its absence proves a hand edit.
    stripped_text = "\n".join(
        line for line in text.splitlines() if "lex:confidenceTier" not in line
    )
    lines = stripped_text.splitlines()
    triple = triples[0]

    header_idx, terminator_idx = align_lexicons._locate_annotation_block(
        lines, triple.tapi_lex_id, triple.predicate, triple.ietf_lex_id
    )

    assert align_lexicons._read_block_confidence_tier(lines, header_idx, terminator_idx) is None


def test_read_block_confidence_tier_survives_a_preceding_multiline_evidence_quote():
    """CR-01 case: _read_block_confidence_tier() must walk the block via
    _iter_block_statements() (the N3-string-aware scanner), not a naive
    per-line scan, so a multi-line lex:evidenceQuote earlier in the block
    cannot truncate the walk before lex:confidenceTier is reached."""
    lines = [
        "<<lex:tapi-topology-node skos:exactMatch lex:ietf-network-node>>",
        '    lex:evidenceQuote """First sentence ends here.',
        'Second line continues the quote and matters too.""" ;',
        '    lex:confidenceTier "high" ;',
        '    lex:lexiconVersion "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" .',
        "",
    ]

    header_idx, terminator_idx = align_lexicons._locate_annotation_block(
        lines, "tapi-topology-node", "skos:exactMatch", "ietf-network-node"
    )

    assert align_lexicons._read_block_confidence_tier(lines, header_idx, terminator_idx) == "high"


def test_review_record_without_worklist_tier_defaults_to_none():
    record = align_lexicons.ReviewRecord(
        row_id="x",
        kind="correspondence",
        tapi_lex_id="t",
        ietf_lex_id="i",
        predicate="skos:exactMatch",
        verdict="accept",
    )

    assert record.worklist_tier is None


def test_parse_review_worklist_sets_worklist_tier_on_correspondence_and_gap_rows(fixture_entries):
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="medium"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    gap = _gap_record(by_lex_id["tapi-topology-link"])
    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[gap])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)

    corr_row = next(r for r in rows if r.kind == "correspondence")
    gap_row = next(r for r in rows if r.kind == "gap")
    marked = _mark_row_verdict(text, corr_row.row_id, verdict="accept", reason="ok")
    marked = _mark_row_verdict(marked, gap_row.row_id, verdict="accept", reason="genuine gap")

    records, _, _ = align_lexicons.parse_review_worklist(marked)

    corr_record = next(r for r in records if r.kind == "correspondence")
    gap_record = next(r for r in records if r.kind == "gap")
    assert corr_record.worklist_tier == "medium"
    assert gap_record.worklist_tier == align_lexicons.WORKLIST_EMPTY_CELL


def test_tampered_tier_cell_cannot_disable_the_high_tier_gate(fixture_entries, tmp_path, lexicon_dir):
    """05-VERIFICATION.md missing item 2 / CR-02's exact reproduction: a
    genuinely high-tier correspondence, accepted after only the worklist's
    tier cell was edited to a non-high value. parse_review_worklist()
    (Check A) is correctly bypassed by the tamper -- that IS the defect
    being contained here, not fixed at parse time. The binding refusal must
    come from write_reviewed_correspondences() (Check B), which reads the
    target artifact's own lex:confidenceTier instead."""
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    assert rows[0].tier == "high"  # proves the row is genuinely high-tier
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]

    marked = _mark_rederivation(
        text, row.row_id, verdict="accept", reason="looks fine", re_derived="N", citation=""
    )
    tampered = _tamper_tier_cell(marked, row.row_id, tier="medium")

    records, version, model = align_lexicons.parse_review_worklist(tampered)
    assert len(records) == 1  # Check A defeated by the tamper -- documents the defect, not the fix

    with pytest.raises(align_lexicons.MalformedWorklistError, match=row.row_id):
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )


def test_tampered_tier_worklist_writes_nothing(fixture_entries, tmp_path, lexicon_dir):
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))
    original_bytes = corr_path.read_bytes()

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]
    marked = _mark_rederivation(
        text, row.row_id, verdict="accept", reason="looks fine", re_derived="N", citation=""
    )
    tampered = _tamper_tier_cell(marked, row.row_id, tier="medium")
    records, version, model = align_lexicons.parse_review_worklist(tampered)

    with pytest.raises(align_lexicons.MalformedWorklistError):
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )

    assert corr_path.read_bytes() == original_bytes


def test_medium_tier_block_still_accepts_with_empty_rederivation_cells(fixture_entries, tmp_path, lexicon_dir):
    """Non-regression control: the new artifact-sourced gate adds no
    refusal on the non-high path when the worklist cell is untouched."""
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="medium"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]
    marked = _mark_row_verdict(text, row.row_id, verdict="accept", reason="ok")  # tier cell untouched
    records, version, model = align_lexicons.parse_review_worklist(marked)

    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )

    reviewed_text = corr_path.read_text()
    assert 'lex:reviewVerdict "accepted"' in reviewed_text


# -- Plan 05-06 gap closure Task 2: a worklist/artifact tier disagreement --
# -- is tamper-evident, and an unreadable tier fails closed ----------------


def test_tier_cell_mismatching_the_artifact_is_its_own_defect(fixture_entries, tmp_path, lexicon_dir):
    """A row accepted WITH a valid re-derivation (Y + a distinct citation)
    is still refused when the worklist's tier cell disagrees with the
    artifact -- the disagreement is its own defect, independent of whether
    the SC4 re-derivation columns were filled in correctly."""
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]
    marked = _mark_rederivation(
        text, row.row_id, verdict="accept", reason="ok", re_derived="Y",
        citation="ietf-network.yang: node container, independently checked against source.",
    )
    tampered = _tamper_tier_cell(marked, row.row_id, tier="medium")
    records, version, model = align_lexicons.parse_review_worklist(tampered)

    with pytest.raises(align_lexicons.MalformedWorklistError, match=row.row_id) as exc_info:
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )
    message = str(exc_info.value)
    assert "SC4 re-derivation refusal" not in message
    assert "mismatch" in message.lower()


def test_tier_cell_tampered_upward_reports_mismatch_without_the_rederivation_defect(
    fixture_entries, tmp_path, lexicon_dir
):
    """A genuinely medium-tier row whose cell was tampered UP to 'high',
    accepted with the rendered empty re-derivation cells, produces only the
    mismatch defect -- the SC4 refusal must not also fire, since the
    artifact's own tier is medium, not high."""
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="medium"),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]
    # A well-formed (Y + non-empty distinct citation) rederivation is
    # required so the tampered-to-'high' cell still passes
    # parse_review_worklist()'s own Check A syntactically -- the point of
    # this test is that the write-time mismatch check catches the
    # disagreement Check A structurally cannot see (it only validates the
    # cell's own internal shape, never cross-checks it against the
    # artifact).
    marked = _mark_rederivation(
        text, row.row_id, verdict="accept", reason="ok", re_derived="Y",
        citation="a citation that is well-formed but describes a medium-tier row.",
    )
    tampered = _tamper_tier_cell(marked, row.row_id, tier="high")
    records, version, model = align_lexicons.parse_review_worklist(tampered)

    with pytest.raises(align_lexicons.MalformedWorklistError, match=row.row_id) as exc_info:
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )
    message = str(exc_info.value)
    assert "SC4 re-derivation refusal" not in message
    assert "mismatch" in message.lower()


@pytest.mark.parametrize("tier", align_lexicons.CONFIDENCE_TIERS)
def test_untampered_tier_cell_produces_no_defect_for_every_tier(
    fixture_entries, tmp_path, lexicon_dir, tier
):
    """Boundary test: the threshold value (high) and both values one step
    either side (medium, low) all write successfully when the worklist's
    tier cell agrees with the artifact -- the mismatch/SC4 checks add no
    refusal on the honest path for any tier."""
    _, _, by_lex_id = fixture_entries
    result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier=tier),
    )
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]
    if tier == "high":
        marked = _mark_rederivation(
            text, row.row_id, verdict="accept", reason="ok", re_derived="Y",
            citation="ietf-network.yang: node container, independently checked against source.",
        )
    else:
        marked = _mark_row_verdict(text, row.row_id, verdict="accept", reason="ok")
    records, version, model = align_lexicons.parse_review_worklist(marked)

    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )

    assert 'lex:reviewVerdict "accepted"' in corr_path.read_text()


def test_blank_tier_cell_is_a_mismatch(fixture_entries, tmp_path, lexicon_dir):
    _, _, by_lex_id = fixture_entries
    for blank_value in ("", align_lexicons.WORKLIST_EMPTY_CELL):
        result = _high_tier_result(by_lex_id)
        triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
        corr_path = tmp_path / f"correspondences-{blank_value!r}.ttl"
        corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

        rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
        text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
        row = rows[0]
        marked = _mark_rederivation(
            text, row.row_id, verdict="accept", reason="ok", re_derived="Y",
            citation="ietf-network.yang: node container, independently checked against source.",
        )
        tampered = _tamper_tier_cell(marked, row.row_id, tier=blank_value)
        records, version, model = align_lexicons.parse_review_worklist(tampered)

        with pytest.raises(align_lexicons.MalformedWorklistError, match=row.row_id):
            align_lexicons.write_reviewed_correspondences(
                corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
            )


def test_tier_comparison_is_case_sensitive(fixture_entries, tmp_path, lexicon_dir):
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]
    marked = _mark_rederivation(
        text, row.row_id, verdict="accept", reason="ok", re_derived="Y",
        citation="ietf-network.yang: node container, independently checked against source.",
    )
    tampered = _tamper_tier_cell(marked, row.row_id, tier="High")
    records, version, model = align_lexicons.parse_review_worklist(tampered)

    with pytest.raises(align_lexicons.MalformedWorklistError, match=row.row_id):
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )


def test_block_without_confidence_tier_is_a_defect_not_a_skipped_gate(
    fixture_entries, tmp_path, lexicon_dir
):
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    original_text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)
    stripped_text = "\n".join(
        line for line in original_text.splitlines() if "lex:confidenceTier" not in line
    )
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(stripped_text)
    original_bytes = corr_path.read_bytes()

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]
    # Untouched cell, well-formed rederivation -- passes Check A cleanly.
    # The write-time refusal here must come from the block's tier being
    # unreadable, not from an invalid rederivation.
    marked = _mark_rederivation(
        text, row.row_id, verdict="accept", reason="ok", re_derived="Y",
        citation="ietf-network.yang: node container, independently checked against source.",
    )
    records, version, model = align_lexicons.parse_review_worklist(marked)

    with pytest.raises(align_lexicons.MalformedWorklistError, match=row.row_id):
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )

    assert corr_path.read_bytes() == original_bytes


def test_every_tier_defect_is_reported_in_one_error(fixture_entries, tmp_path, lexicon_dir):
    """Three rows, three independent defect kinds, one raised error naming
    all three. The SC4-refusal-only row is built with a direct ReviewRecord
    construction (worklist_tier=None) rather than through the worklist
    parser: an untampered 'high' cell paired with an invalid re-derivation
    is already refused by parse_review_worklist()'s own parse-time Check A
    (unchanged, existing behavior) before write_reviewed_correspondences()
    is ever reached -- so the only way to observe the write-time SC4
    refusal in isolation (without an accompanying mismatch defect) is a
    record whose worklist_tier was never populated, exactly as a record
    built outside the worklist round trip would look."""
    _, _, by_lex_id = fixture_entries
    mismatch_result = _pair_result(
        by_lex_id["tapi-topology-node"], by_lex_id["ietf-network-node"], "confirm_exact_match",
        confidence=_confidence(tier="high"),
    )
    unreadable_result = _pair_result(
        by_lex_id["tapi-topology-node-edge-point"], by_lex_id["ietf-network-termination-point"],
        "confirm_exact_match", confidence=_confidence(tier="high"),
    )
    sc4_result = _pair_result(
        by_lex_id["tapi-topology-node-rule-group"], by_lex_id["ietf-network-connectivity-matrix"],
        "confirm_close_match", confidence=_confidence(tier="high"),
    )
    results = [mismatch_result, unreadable_result, sc4_result]
    triples = align_lexicons.correspondences_from_results(results, FAKE_VERSION, FAKE_MODEL)
    original_text = align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL)

    unreadable_triple = next(t for t in triples if t.tapi_lex_id == "tapi-topology-node-edge-point")
    lines = original_text.splitlines()
    header_idx, terminator_idx = align_lexicons._locate_annotation_block(
        lines, unreadable_triple.tapi_lex_id, unreadable_triple.predicate, unreadable_triple.ietf_lex_id
    )
    tier_line_idx = next(
        i for i in range(header_idx + 1, terminator_idx + 1) if "lex:confidenceTier" in lines[i]
    )
    del lines[tier_line_idx]
    stripped_text = "\n".join(lines)

    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(stripped_text)
    original_bytes = corr_path.read_bytes()

    # Only the mismatch and unreadable rows go through the real worklist
    # round trip -- both survive parse_review_worklist()'s own Check A.
    worklist_results = [mismatch_result, unreadable_result]
    rows = align_lexicons.build_worklist_rows(
        align_lexicons.correspondences_from_results(worklist_results, FAKE_VERSION, FAKE_MODEL),
        worklist_results,
        gap_records=[],
    )
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)

    mismatch_row = next(r for r in rows if r.tapi_lex_id == "tapi-topology-node")
    unreadable_row = next(r for r in rows if r.tapi_lex_id == "tapi-topology-node-edge-point")

    marked = _mark_rederivation(
        text, mismatch_row.row_id, verdict="accept", reason="ok", re_derived="Y",
        citation="ietf-network.yang: node container, independently checked against source.",
    )
    marked = _tamper_tier_cell(marked, mismatch_row.row_id, tier="medium")
    # Well-formed rederivation on the unreadable-tier row too -- Check A
    # must pass so this row survives to the write-time pass, where the
    # block's own missing lex:confidenceTier is what refuses it.
    marked = _mark_rederivation(
        marked, unreadable_row.row_id, verdict="accept", reason="ok", re_derived="Y",
        citation="ietf-network.yang: termination-point definition, independently checked.",
    )

    parsed_records, version, model = align_lexicons.parse_review_worklist(marked)

    sc4_triple = next(t for t in triples if t.tapi_lex_id == "tapi-topology-node-rule-group")
    sc4_row_id = align_lexicons.worklist_row_id(
        "correspondence", sc4_triple.tapi_lex_id, sc4_triple.ietf_lex_id, sc4_triple.predicate
    )
    sc4_record = align_lexicons.ReviewRecord(
        row_id=sc4_row_id,
        kind="correspondence",
        tapi_lex_id=sc4_triple.tapi_lex_id,
        ietf_lex_id=sc4_triple.ietf_lex_id,
        predicate=sc4_triple.predicate,
        verdict="accept",
        reason="ok",
        re_derived=False,
        rederivation_citation="",
        # worklist_tier intentionally omitted (defaults to None): this
        # record was never routed through parse_review_worklist(), so no
        # tier cell exists to compare -- exercising the SC4 refusal in
        # isolation from the mismatch check.
    )

    records = parsed_records + [sc4_record]

    with pytest.raises(align_lexicons.MalformedWorklistError) as exc_info:
        align_lexicons.write_reviewed_correspondences(
            corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
        )
    message = str(exc_info.value)
    assert mismatch_row.row_id in message
    assert unreadable_row.row_id in message
    assert sc4_row_id in message
    assert corr_path.read_bytes() == original_bytes


def test_gap_row_tier_cell_is_never_a_tier_defect(fixture_entries, tmp_path, lexicon_dir):
    _, _, by_lex_id = fixture_entries
    gap_entry = by_lex_id["tapi-topology-link"]
    triples: List[align_lexicons.CorrespondenceTriple] = []
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

    gap = _gap_record(gap_entry)
    rows = align_lexicons.build_worklist_rows(triples, [], gap_records=[gap])
    gap_row = next(r for r in rows if r.kind == "gap")
    assert gap_row.tier == ""  # rendered as WORKLIST_EMPTY_CELL ("-") on the worklist itself
    worklist_text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    marked = _mark_row_verdict(worklist_text, gap_row.row_id, verdict="accept", reason="genuine gap")

    records, version, model = align_lexicons.parse_review_worklist(marked)
    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )

    assert "lex:ReviewedGap" in corr_path.read_text()


def test_high_tier_reject_still_writes_without_a_citation(fixture_entries, tmp_path, lexicon_dir):
    """Both the SC4 refusal and the mismatch defect are scoped to
    verdict == 'accept' -- a reject/uncertain verdict on a high-tier
    correspondence with no citation still writes successfully."""
    _, _, by_lex_id = fixture_entries
    result = _high_tier_result(by_lex_id)
    triples = align_lexicons.correspondences_from_results([result], FAKE_VERSION, FAKE_MODEL)
    corr_path = tmp_path / "correspondences.ttl"
    corr_path.write_text(align_lexicons.render_correspondences_ttl(triples, FAKE_VERSION, FAKE_MODEL))

    rows = align_lexicons.build_worklist_rows(triples, [result], gap_records=[])
    text = align_lexicons.render_review_worklist(rows, FAKE_VERSION, FAKE_MODEL)
    row = rows[0]
    marked = _mark_row_verdict(text, row.row_id, verdict="reject", reason="not convinced")
    records, version, model = align_lexicons.parse_review_worklist(marked)

    align_lexicons.write_reviewed_correspondences(
        corr_path, records, lexicon_dir, worklist_lexicon_version=version, worklist_model=model
    )

    assert 'lex:reviewVerdict "rejected"' in corr_path.read_text()


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
