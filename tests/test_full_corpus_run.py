"""
Tests for align_lexicons.py's full-corpus loading mode (Plan 05-03).

D-04's opt-in full-corpus loader (load_all_entries()/side_for_lexicon_file())
is a sibling of load_fixture_entries(), never a replacement -- the no-flag
default run must keep loading the same 11-entry FIXTURE_TAPI/FIXTURE_IETF
fixture, byte-for-byte, so every Phase 1-4 regression test keeps its exact
meaning.

full_corpus is module-scoped: parsing all 34 real committed lexicon files
takes roughly a second of rdflib work, and a per-test full load would repeat
that for every test in this module for no benefit. It cannot depend on
conftest.py's function-scoped lexicon_dir fixture (a module-scoped fixture
cannot request a function-scoped one), so LEXICON_DIR is computed the same
way conftest.py's own lexicon_dir fixture computes it.

Task 1's field-normalization/skip-warning tests use small synthetic
tmp_path corpora, mirroring test_align_lexicons.py's own WR-01 synthetic-
file pattern (test_needs_curation_false_literal_parses_as_false), rather
than depending on which real corpus entries happen to carry a given
evidence shape.

Task 2's tests drive main() via a patched sys.argv/anthropic.Anthropic(),
mirroring test_align_lexicons.py's own main()-invoking idiom
(test_full_fixture_run_no_crash and neighbors).
"""
import sys
from pathlib import Path

import pytest

import align_lexicons

LEXICON_DIR = Path(__file__).resolve().parents[1] / "lexicon"


@pytest.fixture(scope="module")
def full_corpus():
    return align_lexicons.load_all_entries(LEXICON_DIR)


# ── Task 1: enumerate the whole corpus ────────────────────────────────────


def test_all_thirty_four_lexicon_files_are_read():
    files = sorted(LEXICON_DIR.glob("*.lexicon.ttl"))
    assert len(files) == 34


def test_entry_counts_match_the_committed_curation_audit(full_corpus):
    """Pinned regression against yang4owl/lexicon/CURATION-AUDIT.md's
    committed per-side summary table -- 1,777 TAPI / 558 IETF entries
    across the full 34-file corpus, with no module pre-excluded (D-01)."""
    tapi_entries, ietf_entries = full_corpus
    assert len(tapi_entries) == 1777
    assert len(ietf_entries) == 558


def test_side_assignment_matches_the_committed_rule(full_corpus):
    """The committed rule (audit_lexicon_curation.py's side_for(),
    CURATION-AUDIT.md line 5): a file whose name begins 'tapi-' is the TAPI
    side; every other *.lexicon.ttl file -- the ietf-* modules plus the two
    non-IETF-named files simap-yang and iana-hardware -- is the IETF side,
    never excluded."""
    tapi_entries, ietf_entries = full_corpus
    assert tapi_entries, "expected at least one TAPI entry"
    assert ietf_entries, "expected at least one IETF entry"
    assert all(e.source == "tapi" for e in tapi_entries)
    assert all(e.source == "ietf" for e in ietf_entries)

    assert align_lexicons.side_for_lexicon_file("tapi-topology.lexicon.ttl") == "tapi"
    assert align_lexicons.side_for_lexicon_file("simap-yang.lexicon.ttl") == "ietf"
    assert align_lexicons.side_for_lexicon_file("iana-hardware.lexicon.ttl") == "ietf"


def test_entries_carry_normalized_evidence_fields(tmp_path):
    """The full-corpus loader must apply load_fixture_entries()'s own
    normalization byte-for-byte (shared via _entry_from_subject()): a
    null-evidence-placeholder definition and a mechanical label-restatement
    definition both come back None, and an entry carrying two distinct
    scope notes comes back with both, sorted -- not an arbitrary one."""
    ttl_content = """\
@prefix lex: <http://example.org/ontology/lexicon-vocab#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

lex:full-corpus-null-evidence
    a lex:ReferenceEntry ;
    skos:prefLabel "null evidence entry" ;
    skos:definition "none" .

lex:full-corpus-restated
    a lex:ReferenceEntry ;
    skos:prefLabel "restated entry" ;
    skos:definition "Grouping definition: restated entry" .

lex:full-corpus-two-scope-notes
    a lex:ReferenceEntry ;
    skos:prefLabel "two scope note entry" ;
    skos:scopeNote "Zebra note." ;
    skos:scopeNote "Alpha note." .
"""
    (tmp_path / "tapi-synthetic.lexicon.ttl").write_text(ttl_content)

    tapi_entries, ietf_entries = align_lexicons.load_all_entries(tmp_path)
    assert ietf_entries == []
    by_lex_id = {e.lex_id: e for e in tapi_entries}
    assert set(by_lex_id) == {
        "full-corpus-null-evidence",
        "full-corpus-restated",
        "full-corpus-two-scope-notes",
    }

    assert by_lex_id["full-corpus-null-evidence"].definition is None
    assert by_lex_id["full-corpus-restated"].definition is None
    assert by_lex_id["full-corpus-two-scope-notes"].scope_notes == [
        "Alpha note.",
        "Zebra note.",
    ]


def test_entry_with_no_preferred_label_is_skipped_with_a_warning(tmp_path, capsys):
    """An entry with no usable skos:prefLabel is skipped with a visible
    warning naming it, exactly as load_fixture_entries() already does --
    never silently dropped."""
    ttl_content = """\
@prefix lex: <http://example.org/ontology/lexicon-vocab#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

lex:full-corpus-no-label
    a lex:ReferenceEntry ;
    skos:definition "Has a definition but no usable prefLabel." .
"""
    (tmp_path / "ietf-synthetic.lexicon.ttl").write_text(ttl_content)

    tapi_entries, ietf_entries = align_lexicons.load_all_entries(tmp_path)
    assert tapi_entries == []
    assert ietf_entries == []

    captured = capsys.readouterr()
    assert "full-corpus-no-label" in captured.out
    assert "no usable skos:prefLabel" in captured.out


def test_two_loads_produce_identically_ordered_entries():
    """Files are enumerated with sorted() and each file's own entries are
    sorted by lex_id before being extended onto the side list, so two loads
    over the same committed corpus produce identically ordered lists."""
    first_tapi, first_ietf = align_lexicons.load_all_entries(LEXICON_DIR)
    second_tapi, second_ietf = align_lexicons.load_all_entries(LEXICON_DIR)
    assert [e.lex_id for e in first_tapi] == [e.lex_id for e in second_tapi]
    assert [e.lex_id for e in first_ietf] == [e.lex_id for e in second_ietf]


def test_evidence_gate_volume_uses_has_evidence_not_thin_evidence(full_corpus):
    """The escalation-volume number this plan reports must come from
    LexiconEntry.has_evidence -- the exact predicate evidence_gate() checks
    -- never from CURATION-AUDIT.md's thin_evidence column (611 TAPI / 14
    IETF, a different, narrower, also-legitimate metric per RESEARCH.md
    Pitfall 3). The number NOT being asserted here is that thin_evidence
    pair; has_evidence-false counts are 782 TAPI / 5 IETF."""
    tapi_entries, ietf_entries = full_corpus
    tapi_without_evidence = sum(1 for e in tapi_entries if not e.has_evidence)
    ietf_without_evidence = sum(1 for e in ietf_entries if not e.has_evidence)
    assert tapi_without_evidence == 782
    assert ietf_without_evidence == 5


def test_signal_floors_are_unchanged_by_this_phase():
    """MUST NOT re-fit or tune STRUCTURAL_SIGNAL_FLOOR, LABEL_SIGNAL_FLOOR,
    or DEFAULT_LABEL_THRESHOLD against full-corpus observations -- all
    three are fitted to the locked 11-entry fixture."""
    assert align_lexicons.STRUCTURAL_SIGNAL_FLOOR == 0.15
    assert align_lexicons.LABEL_SIGNAL_FLOOR == 60.0
    assert align_lexicons.DEFAULT_LABEL_THRESHOLD == 45.0


# ── Task 2: one opt-in flag selects the full corpus ───────────────────────


def test_full_corpus_flag_selects_the_full_loader(recording_client, capsys, monkeypatch):
    """main() invoked with --full-corpus loads through load_all_entries() --
    asserted by the run summary reporting the full-corpus entry counts and
    the run-header line naming the loading mode. A tight --max-calls makes
    a CallBudgetExceeded stop an acceptable, expected outcome at this scale
    (7,342 label-pass candidates against the real corpus) -- the assertion
    is on what printed, not on completing the run."""
    monkeypatch.setattr(
        sys, "argv", ["align_lexicons.py", "--full-corpus", "--max-calls", "0"]
    )
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)

    try:
        align_lexicons.main()
    except SystemExit:
        pass

    captured = capsys.readouterr()
    assert "loading_mode=full-corpus" in captured.out
    assert "tapi entries loaded: 1777" in captured.out
    assert "ietf entries loaded: 558" in captured.out


def test_default_run_still_loads_the_eleven_entry_fixture(recording_client, capsys, monkeypatch):
    """main() invoked without --full-corpus loads through
    load_fixture_entries() exactly as every Phase 1-4 run did -- asserted
    by the run summary reporting the committed fixture list lengths (D-04)."""
    monkeypatch.setattr(sys, "argv", ["align_lexicons.py"])
    monkeypatch.setattr(align_lexicons.anthropic, "Anthropic", lambda: recording_client)

    align_lexicons.main()

    captured = capsys.readouterr()
    assert "loading_mode=fixture" in captured.out
    assert "tapi entries loaded: 6" in captured.out
    assert "ietf entries loaded: 5" in captured.out


# ── Plan 05-04 Task 1: measure where the known true-positive correspondent
# ranks under each recovery signal at full-corpus scale, and pin the
# shortlist sizes to that measurement rather than a guess (D-17) ──────────

NODE_RULE_GROUP_LEX_ID = "tapi-topology-node-rule-group"
CONNECTIVITY_MATRIX_LEX_ID = "ietf-network-connectivity-matrix"


def _by_lex_id(entries, lex_id):
    return next(e for e in entries if e.lex_id == lex_id)


def _make_entry(lex_id, pref_label, source="ietf", source_path=None):
    """Synthetic LexiconEntry factory for the boundary/tie/sentinel tests
    below -- mirrors test_align_lexicons.py's own direct-construction
    pattern (e.g. test_canonical_example_only_entry_reaches_confirmation_
    with_its_text_in_prompt)."""
    return align_lexicons.LexiconEntry(
        source=source,
        lex_id=lex_id,
        pref_label=pref_label,
        definition=None,
        scope_notes=[],
        canonical_example=None,
        needs_curation=False,
        source_path=source_path,
    )


def _label_rank(tapi_entry, ietf_entries, target_lex_id):
    """Rank position (1-indexed) of target_lex_id when ietf_entries is
    sorted by label_score(tapi_entry, ietf) descending, ietf.lex_id
    ascending as tie-break -- exactly the ranking <bounding_contract>
    specifies for the label shortlist."""
    ranked = sorted(
        ietf_entries,
        key=lambda e: (-align_lexicons.label_score(tapi_entry.pref_label, e.pref_label), e.lex_id),
    )
    return next(i for i, e in enumerate(ranked, start=1) if e.lex_id == target_lex_id)


def _structural_rank(tapi_entry, ietf_entries, target_lex_id):
    """Rank position (1-indexed) of target_lex_id when ietf_entries is
    sorted by structural_corroboration(tapi_entry, ietf) descending,
    ietf.lex_id ascending as tie-break, with a None score mapped to
    RECOVERY_NO_STRUCTURAL_SIGNAL_RANK -- structural_corroboration()'s own
    no-signal-is-not-zero contract, never coerced to 0.0."""
    def _key(e):
        score = align_lexicons.structural_corroboration(tapi_entry, e)
        rank_value = score if score is not None else align_lexicons.RECOVERY_NO_STRUCTURAL_SIGNAL_RANK
        return (-rank_value, e.lex_id)

    ranked = sorted(ietf_entries, key=_key)
    return next(i for i, e in enumerate(ranked, start=1) if e.lex_id == target_lex_id)


def test_known_true_positive_label_rank_at_full_corpus(full_corpus):
    """Where does ietf-network-connectivity-matrix actually rank against
    tapi-topology-node-rule-group by label_score, among all 558 real IETF
    entries? RESEARCH.md Pitfall 1's own caution (low structural 0.0625,
    moderate label 23.53, both hand-computed against the 5-entry fixture
    only) says nothing about the full-corpus RANK position, which is what
    actually determines whether a top-K label shortlist can retain it --
    this test measures that directly rather than guessing."""
    tapi_entries, ietf_entries = full_corpus
    node_rule_group = _by_lex_id(tapi_entries, NODE_RULE_GROUP_LEX_ID)

    rank = _label_rank(node_rule_group, ietf_entries, CONNECTIVITY_MATRIX_LEX_ID)
    print(f"measured label rank of {CONNECTIVITY_MATRIX_LEX_ID}: {rank} of {len(ietf_entries)}")

    assert 1 <= rank <= len(ietf_entries), (
        f"expected a finite rank position among {len(ietf_entries)} IETF "
        f"entries; measured rank={rank}"
    )
    rank_again = _label_rank(node_rule_group, ietf_entries, CONNECTIVITY_MATRIX_LEX_ID)
    assert rank_again == rank, "label rank must be reproducible across ranking passes"


def test_known_true_positive_structural_rank_at_full_corpus(full_corpus):
    """Same measurement, structural_corroboration signal. This test and the
    label-rank test above are the ONLY place either shortlist size is
    justified -- see the sizing comment above RECOVERY_LABEL_SHORTLIST/
    RECOVERY_STRUCTURAL_SHORTLIST in align_lexicons.py."""
    tapi_entries, ietf_entries = full_corpus
    node_rule_group = _by_lex_id(tapi_entries, NODE_RULE_GROUP_LEX_ID)

    rank = _structural_rank(node_rule_group, ietf_entries, CONNECTIVITY_MATRIX_LEX_ID)
    print(f"measured structural rank of {CONNECTIVITY_MATRIX_LEX_ID}: {rank} of {len(ietf_entries)}")

    assert 1 <= rank <= len(ietf_entries), (
        f"expected a finite rank position among {len(ietf_entries)} IETF "
        f"entries; measured rank={rank}"
    )
    rank_again = _structural_rank(node_rule_group, ietf_entries, CONNECTIVITY_MATRIX_LEX_ID)
    assert rank_again == rank, "structural rank must be reproducible across ranking passes"


def test_configured_shortlists_retain_the_known_true_positive(full_corpus):
    """The union guarantee stated as one truth: at least one of the two
    CONFIGURED shortlist sizes retains ietf-network-connectivity-matrix
    when ranked against tapi-topology-node-rule-group at full-corpus
    scale.

    The measured label rank (425 of 558) sits in the "no plausible
    correspondent by name" cluster LABEL_SIGNAL_FLOOR's own comment already
    documents for this exact pair (label_score=23.53): retaining it via a
    label-ranked shortlist alone would require a cap of ~550 (~98.6% of the
    558-entry IETF corpus), which would leave the recovery pass
    functionally unbounded for every OTHER unresolved entry too --
    defeating this plan's entire purpose. RECOVERY_LABEL_SHORTLIST is
    therefore deliberately NOT sized to also retain this specific pair (see
    the recorded deviation in 05-04-SUMMARY.md); retention is carried
    entirely by RECOVERY_STRUCTURAL_SHORTLIST, consistent with
    <bounding_contract>'s own "two independent chances, not two
    requirements" design rationale."""
    tapi_entries, ietf_entries = full_corpus
    node_rule_group = _by_lex_id(tapi_entries, NODE_RULE_GROUP_LEX_ID)

    label_rank = _label_rank(node_rule_group, ietf_entries, CONNECTIVITY_MATRIX_LEX_ID)
    structural_rank = _structural_rank(node_rule_group, ietf_entries, CONNECTIVITY_MATRIX_LEX_ID)

    retained_by_label = label_rank <= align_lexicons.RECOVERY_LABEL_SHORTLIST
    retained_by_structural = structural_rank <= align_lexicons.RECOVERY_STRUCTURAL_SHORTLIST

    assert retained_by_label or retained_by_structural, (
        f"expected at least one shortlist to retain the known true positive "
        f"(label_rank={label_rank} vs RECOVERY_LABEL_SHORTLIST="
        f"{align_lexicons.RECOVERY_LABEL_SHORTLIST}, structural_rank={structural_rank} "
        f"vs RECOVERY_STRUCTURAL_SHORTLIST={align_lexicons.RECOVERY_STRUCTURAL_SHORTLIST})"
    )
    assert retained_by_structural, (
        "retention is carried by the structural shortlist by design -- the "
        f"structural rank ({structural_rank}) must fit within "
        f"RECOVERY_STRUCTURAL_SHORTLIST ({align_lexicons.RECOVERY_STRUCTURAL_SHORTLIST})"
    )


def test_recovery_candidates_per_entry_is_the_sum_of_both_shortlists():
    assert align_lexicons.RECOVERY_CANDIDATES_PER_ENTRY == (
        align_lexicons.RECOVERY_LABEL_SHORTLIST + align_lexicons.RECOVERY_STRUCTURAL_SHORTLIST
    )


def test_no_structural_signal_sentinel_is_below_every_computed_score():
    """structural_corroboration() returns a raw token-overlap ratio always
    in [0.0, 1.0] when it returns a value at all -- the sentinel a None
    score maps to must sit strictly below that whole range."""
    assert align_lexicons.RECOVERY_NO_STRUCTURAL_SIGNAL_RANK < 0.0


def test_signal_floors_and_label_threshold_unchanged_by_recovery_bounding():
    """MUST NOT re-fit STRUCTURAL_SIGNAL_FLOOR/LABEL_SIGNAL_FLOOR/
    DEFAULT_LABEL_THRESHOLD while sizing the recovery shortlists (Task 1
    prohibition, mirrors test_signal_floors_are_unchanged_by_this_phase
    above for Plan 03's own scope)."""
    assert align_lexicons.STRUCTURAL_SIGNAL_FLOOR == 0.15
    assert align_lexicons.LABEL_SIGNAL_FLOOR == 60.0
    assert align_lexicons.DEFAULT_LABEL_THRESHOLD == 45.0


# ── Plan 05-04 Task 2: replace the recovery cross product with the bounded,
# evidence-ranked shortlist (D-17) ─────────────────────────────────────────


def test_shortlist_keeps_all_when_fewer_eligible_than_the_cap():
    """An entry with fewer eligible IETF entries than either shortlist size
    keeps all of them, deduplicated, with no error -- the default
    RECOVERY_LABEL_SHORTLIST/RECOVERY_STRUCTURAL_SHORTLIST are both far
    larger than 3."""
    tapi_entry = _make_entry("synthetic-tapi", "node rule group", source="tapi")
    eligible = [
        _make_entry("ietf-one", "alpha"),
        _make_entry("ietf-two", "beta"),
        _make_entry("ietf-three", "gamma"),
    ]
    shortlist = align_lexicons.recovery_shortlist(tapi_entry, eligible)
    assert {e.lex_id for e in shortlist} == {"ietf-one", "ietf-two", "ietf-three"}


def test_shortlist_boundary_drops_the_next_ranked_candidate(monkeypatch):
    """Given more eligible entries than the cap, the shortlist returns at
    most the cap's worth, and the entry ranked immediately past the
    boundary is absent -- determined by score, never by input list order."""
    monkeypatch.setattr(align_lexicons, "RECOVERY_LABEL_SHORTLIST", 2)
    monkeypatch.setattr(align_lexicons, "RECOVERY_STRUCTURAL_SHORTLIST", 0)
    tapi_entry = _make_entry("synthetic-tapi", "node rule group", source="tapi")

    exact = _make_entry("ietf-zzz-exact", "node rule group")  # label_score 100.0
    middle = _make_entry("ietf-aaa-middle", "group of something else entirely")  # 50.0
    weakest = _make_entry("ietf-mmm-weakest", "completely unrelated phrase")  # 38.1
    eligible = [weakest, middle, exact]  # deliberately unsorted input order

    shortlist = align_lexicons.recovery_shortlist(tapi_entry, eligible)
    kept_ids = {e.lex_id for e in shortlist}
    assert kept_ids == {"ietf-zzz-exact", "ietf-aaa-middle"}, (
        f"expected the top-2 by label score kept (weakest dropped); got {kept_ids}"
    )


def test_shortlist_ties_are_broken_by_ietf_lex_id(monkeypatch):
    """Two eligible entries tied on the ranking signal at the boundary are
    ordered by ietf.lex_id, so which one is kept is reproducible and
    independent of input list order."""
    monkeypatch.setattr(align_lexicons, "RECOVERY_LABEL_SHORTLIST", 1)
    monkeypatch.setattr(align_lexicons, "RECOVERY_STRUCTURAL_SHORTLIST", 0)
    tapi_entry = _make_entry("synthetic-tapi", "node rule group", source="tapi")

    tie_b = _make_entry("ietf-b-tie", "node rule group")
    tie_a = _make_entry("ietf-a-tie", "node rule")
    assert align_lexicons.label_score(tapi_entry.pref_label, tie_a.pref_label) == (
        align_lexicons.label_score(tapi_entry.pref_label, tie_b.pref_label)
    ), "both candidates must genuinely tie for this test to prove anything"

    shortlist = align_lexicons.recovery_shortlist(tapi_entry, [tie_b, tie_a])
    assert {e.lex_id for e in shortlist} == {"ietf-a-tie"}, (
        "tied on label_score -- the lower lex_id must win the single slot"
    )

    shortlist_reordered = align_lexicons.recovery_shortlist(tapi_entry, [tie_a, tie_b])
    assert {e.lex_id for e in shortlist_reordered} == {"ietf-a-tie"}, (
        "the tie-break must be independent of input list order"
    )


def test_missing_structural_signal_never_outranks_a_computed_zero(monkeypatch):
    """An eligible entry whose structural score is None must never appear
    in the structural shortlist ahead of one whose score was computed as
    zero (structural_corroboration()'s own no-signal-is-not-zero
    contract)."""
    monkeypatch.setattr(align_lexicons, "RECOVERY_LABEL_SHORTLIST", 0)
    monkeypatch.setattr(align_lexicons, "RECOVERY_STRUCTURAL_SHORTLIST", 1)
    tapi_entry = _make_entry(
        "synthetic-tapi",
        "x",
        source="tapi",
        source_path="http://example.org/ontology/tapi-topology/topology-context/node-rule-group",
    )
    zero_signal = _make_entry(
        "ietf-zero-signal",
        "x",
        source_path="http://example.org/ontology/ietf-network/network/alpha-beta-gamma",
    )
    no_signal = _make_entry("ietf-no-signal", "x", source_path=None)

    assert align_lexicons.structural_corroboration(tapi_entry, zero_signal) == 0.0
    assert align_lexicons.structural_corroboration(tapi_entry, no_signal) is None

    shortlist = align_lexicons.recovery_shortlist(tapi_entry, [no_signal, zero_signal])
    kept_ids = {e.lex_id for e in shortlist}
    assert kept_ids == {"ietf-zero-signal"}, (
        f"a computed zero must outrank a missing (None) structural signal; got {kept_ids}"
    )


def test_bounded_recovery_is_linear_not_quadratic_at_full_corpus(full_corpus):
    """D-17/T-05-08: the bounded recovery pass must generate substantially
    fewer candidates than the unbounded cross product at full-corpus
    scale, computed directly here rather than asserted as a constant.

    This does NOT reach the "at least an order of magnitude" aspiration
    <bounding_contract> and this plan's Task 2 acceptance criteria
    describe -- see the recorded finding in 05-04-SUMMARY.md. Retaining
    the known true positive requires RECOVERY_STRUCTURAL_SHORTLIST >= 200
    (>25% headroom over its measured full-corpus rank of 153 of 558), and
    200 alone is already ~36% of the 558-entry IETF corpus -- mathematically
    incompatible with a <=10% (order-of-magnitude) per-entry reduction
    target for ANY RECOVERY_LABEL_SHORTLIST value, since retention is a
    hard prerequisite the tests above enforce. The real, measured
    reduction achieved by the union of both shortlists (deduplicated) is
    reported in this test's own assertion message."""
    tapi_entries, ietf_entries = full_corpus

    unbounded_total = 0
    bounded_total = 0
    for tapi_entry in tapi_entries:
        unbounded_total += len(ietf_entries)
        bounded_total += len(align_lexicons.recovery_shortlist(tapi_entry, ietf_entries))

    ratio = bounded_total / unbounded_total
    message = (
        f"bounded={bounded_total} unbounded={unbounded_total} ratio={ratio:.4f} "
        f"(reduction factor {unbounded_total / bounded_total:.2f}x)"
    )
    print(message)
    assert bounded_total < unbounded_total, f"expected strictly fewer candidates -- {message}"
    assert ratio <= 0.5, (
        f"expected at least a real, measured 2x reduction -- {message}. See "
        "05-04-SUMMARY.md for why the plan's order-of-magnitude aspiration is "
        "not reachable while also retaining the known true positive."
    )


def test_recovery_candidates_are_still_sorted_by_lex_ids(fixture_entries, scripted_client):
    """recover_misses() must still sort its generated candidates by
    (tapi.lex_id, ietf.lex_id) before evaluating -- the bounding change
    replaces only candidate GENERATION, never this trailing sort."""
    tapi_entries, ietf_entries, _ = fixture_entries
    client = scripted_client({})
    recovery_results, _ = align_lexicons.recover_misses(
        client, tapi_entries, ietf_entries, [], max_calls=1000, calls_used=0
    )
    pairs = [(r.candidate.tapi.lex_id, r.candidate.ietf.lex_id) for r in recovery_results]
    assert pairs == sorted(pairs), "recovery results must stay sorted by (tapi.lex_id, ietf.lex_id)"


def test_recovery_return_shape_and_budget_threading_unchanged(fixture_entries, scripted_client):
    """recover_misses() must still return a two-element (results,
    calls_used) tuple, seeding calls_used from the caller-supplied
    baseline rather than resetting it."""
    tapi_entries, ietf_entries, _ = fixture_entries
    client = scripted_client({})
    results, calls_used = align_lexicons.recover_misses(
        client, tapi_entries, ietf_entries, [], max_calls=1000, calls_used=7
    )
    assert isinstance(results, list)
    assert isinstance(calls_used, int)
    assert calls_used >= 7, "calls_used must be seeded from the caller-supplied baseline, never reset"
