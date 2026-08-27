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
