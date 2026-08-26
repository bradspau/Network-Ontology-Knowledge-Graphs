"""
Tests for yang4owl/audit_lexicon_curation.py (SC4, D-09, lexicon-69m.13).

Every numeric assertion below is made against synthetic Turtle fixtures
built in tmp_path, never against a count read from yang4owl/lexicon/ --
Plan 03-03 regenerates that corpus in the next wave, changing entry counts
and placeholder shapes (CROSS_PLAN_HAZARD in 03-02-PLAN.md). The one
exception is test_real_lexicon_dir_produces_a_row_per_file and
test_audit_does_not_modify_any_lexicon_file, which assert only shape
invariants that hold before and after regeneration.

Uses the same @prefix lex:/skos:/prov: header the real lexicon/*.lexicon.ttl
files carry (yang4owl/lexicon/tapi-topology.lexicon.ttl:1-3), and a lazy
`import audit_lexicon_curation` inside each test body, mirroring
conftest.py's own lazy-import convention for align_lexicons.
"""
import sys

TTL_HEADER = """@prefix lex: <http://example.org/ontology/lexicon-vocab#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
"""

# The four-entry fixture from <behavior>: one real definition, one
# skos:definition "none" (the sentinel), one with no skos:definition triple
# at all, and one whose definition restates its skos:prefLabel.
FOUR_ENTRY_TTL = (
    TTL_HEADER
    + """
lex:tapi-common-real-entry
    a lex:ReferenceEntry ;
    skos:prefLabel "real-entry" ;
    skos:definition "A concrete definition text that is not a placeholder." ;
    lex:canonicalExample "" ;
    lex:entityClass lex:GroupingKind ;
    prov:wasDerivedFrom <http://example.org/ontology/grouping/tapi-common/real-entry> .

lex:tapi-common-null-entry
    a lex:ReferenceEntry ;
    skos:prefLabel "null-entry" ;
    skos:definition "none" ;
    lex:canonicalExample "" ;
    lex:entityClass lex:GroupingKind ;
    prov:wasDerivedFrom <http://example.org/ontology/grouping/tapi-common/null-entry> .

lex:tapi-common-absent-entry
    a lex:ReferenceEntry ;
    skos:prefLabel "absent-entry" ;
    lex:canonicalExample "" ;
    lex:entityClass lex:GroupingKind ;
    prov:wasDerivedFrom <http://example.org/ontology/grouping/tapi-common/absent-entry> .

lex:tapi-common-restating-entry
    a lex:ReferenceEntry ;
    skos:prefLabel "restating-entry" ;
    skos:definition "Restating entry" ;
    lex:canonicalExample "" ;
    lex:entityClass lex:GroupingKind ;
    prov:wasDerivedFrom <http://example.org/ontology/grouping/tapi-common/restating-entry> .
"""
)


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _extract_cells(report_text, row_label):
    """Return the cell values of the first Markdown table row whose first
    cell is exactly row_label, e.g. a filename ('tapi-common.lexicon.ttl')
    or a side name ('tapi'). Deliberately simple line-prefix matching --
    render_report() never interpolates untrusted text into a table cell
    (T-03-08), so filenames/side names never collide with a longer row's
    prefix by construction (a dash immediately follows any name that is a
    prefix of another, never a space)."""
    for line in report_text.splitlines():
        if line.startswith(f"| {row_label} |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return cells[1:]
    raise AssertionError(f"no row found for {row_label!r} in report:\n{report_text}")


# --- Task 1: single-file end-to-end (parse -> classify -> count -> render -> write) ---


def test_counts_null_evidence_literal_as_placeholder(tmp_path):
    import audit_lexicon_curation as auditmod

    path = _write(tmp_path, "tapi-common.lexicon.ttl", FOUR_ENTRY_TTL)
    stats = auditmod.audit_file(path)

    assert stats["entries"] == 4
    assert stats["definition_null"] == 1


def test_counts_absent_definition_as_placeholder(tmp_path):
    import audit_lexicon_curation as auditmod

    path = _write(tmp_path, "tapi-common.lexicon.ttl", FOUR_ENTRY_TTL)
    stats = auditmod.audit_file(path)

    assert stats["definition_absent"] == 1


def test_counts_label_restatement_as_placeholder(tmp_path):
    import audit_lexicon_curation as auditmod

    path = _write(tmp_path, "tapi-common.lexicon.ttl", FOUR_ENTRY_TTL)
    stats = auditmod.audit_file(path)

    assert stats["definition_restates"] == 1
    # entries 4: only real-entry has a real definition, so thin_evidence
    # counts the other three -- the exact undercount that made
    # lexicon-69m.13 unanswerable via lex:needsCuration alone.
    assert stats["thin_evidence"] == 3


def test_needs_curation_alone_undercounts_thin_evidence(tmp_path):
    import audit_lexicon_curation as auditmod

    ttl = (
        TTL_HEADER
        + """
lex:ietf-network-hidden-placeholder
    a lex:ReferenceEntry ;
    skos:prefLabel "hidden-placeholder" ;
    skos:definition "none" ;
    lex:canonicalExample "" ;
    lex:entityClass lex:GroupingKind ;
    lex:needsCuration false ;
    prov:wasDerivedFrom <http://example.org/ontology/grouping/ietf-network/hidden-placeholder> .
"""
    )
    path = _write(tmp_path, "ietf-network.lexicon.ttl", ttl)
    stats = auditmod.audit_file(path)

    # lex:needsCuration is explicitly false, yet the "none" sentinel
    # definition still correctly counts as thin_evidence -- proving
    # needs_curation alone would report 0 escalation candidates here.
    assert stats["needs_curation"] == 0
    assert stats["thin_evidence"] == 1


def test_thin_evidence_excludes_entries_with_scope_notes(tmp_path):
    import audit_lexicon_curation as auditmod

    ttl = (
        TTL_HEADER
        + """
lex:tapi-common-has-scope-note
    a lex:ReferenceEntry ;
    skos:prefLabel "has-scope-note" ;
    skos:definition "none" ;
    skos:scopeNote "Some real evidence living only in the scope note." ;
    lex:canonicalExample "" ;
    lex:entityClass lex:GroupingKind ;
    prov:wasDerivedFrom <http://example.org/ontology/grouping/tapi-common/has-scope-note> .
"""
    )
    path = _write(tmp_path, "tapi-common.lexicon.ttl", ttl)
    stats = auditmod.audit_file(path)

    assert stats["scope_notes"] == 1
    assert stats["entries_with_scope_notes"] == 1
    assert stats["thin_evidence"] == 0


def test_canonical_example_baseline_is_reported_as_expected(tmp_path):
    import audit_lexicon_curation as auditmod

    path = _write(tmp_path, "tapi-common.lexicon.ttl", FOUR_ENTRY_TTL)
    stats = auditmod.audit_file(path)
    by_file = {path.name: stats}
    by_side = {"tapi": dict(stats)}

    report = auditmod.render_report(by_side, by_file)

    assert path.name in report
    assert "canonical" in report.lower()
    assert "by design" in report.lower()
    assert "thin_evidence" in report


# --- Task 2: widen to the whole directory with a per-side split ---


def test_side_assignment_splits_tapi_from_everything_else():
    import audit_lexicon_curation as auditmod

    assert auditmod.side_for("tapi-common.lexicon.ttl") == "tapi"
    assert auditmod.side_for("tapi-topology.lexicon.ttl") == "tapi"
    assert auditmod.side_for("ietf-network.lexicon.ttl") == "ietf"
    assert auditmod.side_for("simap-yang.lexicon.ttl") == "ietf"
    assert auditmod.side_for("iana-hardware.lexicon.ttl") == "ietf"


def test_report_per_side_totals_equal_sum_of_file_rows(tmp_path, monkeypatch, capsys):
    import audit_lexicon_curation as auditmod

    tapi_common_ttl = (
        TTL_HEADER
        + """
lex:tapi-common-real-entry
    a lex:ReferenceEntry ;
    skos:prefLabel "real-entry" ;
    skos:definition "A concrete definition." ;
    skos:scopeNote "Extra scope evidence." ;
    lex:canonicalExample "" ;
    lex:entityClass lex:GroupingKind ;
    prov:wasDerivedFrom <http://example.org/ontology/grouping/tapi-common/real-entry> .

lex:tapi-common-null-entry
    a lex:ReferenceEntry ;
    skos:prefLabel "null-entry" ;
    skos:definition "none" ;
    lex:canonicalExample "" ;
    lex:entityClass lex:GroupingKind ;
    prov:wasDerivedFrom <http://example.org/ontology/grouping/tapi-common/null-entry> .
"""
    )
    tapi_topology_ttl = (
        TTL_HEADER
        + """
lex:tapi-topology-other-entry
    a lex:ReferenceEntry ;
    skos:prefLabel "other-entry" ;
    skos:definition "none" ;
    lex:canonicalExample "" ;
    lex:entityClass lex:GroupingKind ;
    prov:wasDerivedFrom <http://example.org/ontology/grouping/tapi-topology/other-entry> .
"""
    )
    ietf_ttl = (
        TTL_HEADER
        + """
lex:ietf-network-real-entry
    a lex:ReferenceEntry ;
    skos:prefLabel "real-entry" ;
    skos:definition "A real IETF definition." ;
    lex:canonicalExample "" ;
    lex:entityClass lex:GroupingKind ;
    prov:wasDerivedFrom <http://example.org/ontology/grouping/ietf-network/real-entry> .
"""
    )
    _write(tmp_path, "tapi-common.lexicon.ttl", tapi_common_ttl)
    _write(tmp_path, "tapi-topology.lexicon.ttl", tapi_topology_ttl)
    _write(tmp_path, "ietf-network.lexicon.ttl", ietf_ttl)

    monkeypatch.setattr(
        sys, "argv", ["audit_lexicon_curation.py", "--lexicon-dir", str(tmp_path), "--out", "-"]
    )
    auditmod.main()
    report = capsys.readouterr().out

    tapi_common_cells = [int(v) for v in _extract_cells(report, "tapi-common.lexicon.ttl")[1:]]
    tapi_topology_cells = [int(v) for v in _extract_cells(report, "tapi-topology.lexicon.ttl")[1:]]
    tapi_side_cells = [int(v) for v in _extract_cells(report, "tapi")]
    assert tapi_side_cells == [a + b for a, b in zip(tapi_common_cells, tapi_topology_cells)]

    ietf_cells = [int(v) for v in _extract_cells(report, "ietf-network.lexicon.ttl")[1:]]
    ietf_side_cells = [int(v) for v in _extract_cells(report, "ietf")]
    assert ietf_side_cells == ietf_cells


def test_real_lexicon_dir_produces_a_row_per_file(lexicon_dir, monkeypatch, capsys):
    import audit_lexicon_curation as auditmod

    ttl_files = sorted(lexicon_dir.glob("*.lexicon.ttl"))
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in ttl_files}

    monkeypatch.setattr(
        sys, "argv", ["audit_lexicon_curation.py", "--lexicon-dir", str(lexicon_dir), "--out", "-"]
    )
    auditmod.main()
    report = capsys.readouterr().out

    after = {
        p.name: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(lexicon_dir.glob("*.lexicon.ttl"))
    }
    assert before == after

    totals_by_side = {"tapi": None, "ietf": None}
    for p in ttl_files:
        cells = [int(v) for v in _extract_cells(report, p.name)[1:]]
        assert all(v >= 0 for v in cells)
        side = auditmod.side_for(p.name)
        if totals_by_side[side] is None:
            totals_by_side[side] = cells
        else:
            totals_by_side[side] = [a + b for a, b in zip(totals_by_side[side], cells)]

    for side, expected in totals_by_side.items():
        if expected is None:
            continue
        side_cells = [int(v) for v in _extract_cells(report, side)]
        assert side_cells == expected


def test_audit_does_not_modify_any_lexicon_file(lexicon_dir, tmp_path, monkeypatch):
    import audit_lexicon_curation as auditmod

    ttl_files = sorted(lexicon_dir.glob("*.lexicon.ttl"))
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in ttl_files}

    out_path = tmp_path / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_lexicon_curation.py", "--lexicon-dir", str(lexicon_dir), "--out", str(out_path)],
    )
    auditmod.main()

    after = {
        p.name: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(lexicon_dir.glob("*.lexicon.ttl"))
    }
    assert before == after
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").strip() != ""
    # Plan 03-04 commits the report against the repaired corpus -- this
    # plan must not drop one into the real lexicon/ directory.
    assert not (lexicon_dir / "CURATION-AUDIT.md").exists()
