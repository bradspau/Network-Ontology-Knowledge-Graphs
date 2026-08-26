# Lexicon Curation Audit

34 `*.lexicon.ttl` file(s) audited.

**Side assignment rule:** a file whose name begins `tapi-` is the TAPI side; every other `*.lexicon.ttl` file (the `ietf-*` modules plus `simap-yang` and `iana-hardware`) is the IETF side.

## Per-side summary

| side | entries | occurrences | definition_absent | definition_null | definition_restates | scope_notes | entries_with_scope_notes | needs_curation | canonical_example_present | thin_evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| ietf | 558 | 859 | 13 | 0 | 9 | 137 | 121 | 14 | 0 | 14 |
| tapi | 1777 | 4355 | 684 | 0 | 2 | 452 | 321 | 611 | 0 | 611 |

## Per-file detail

| file | side | entries | occurrences | definition_absent | definition_null | definition_restates | scope_notes | entries_with_scope_notes | needs_curation | canonical_example_present | thin_evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iana-hardware.lexicon.ttl | ietf | 15 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ietf-geo-location.lexicon.ttl | ietf | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| ietf-hardware.lexicon.ttl | ietf | 11 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ietf-inet-types.lexicon.ttl | ietf | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ietf-l2-topology-state.lexicon.ttl | ietf | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ietf-l2-topology.lexicon.ttl | ietf | 16 | 17 | 1 | 0 | 0 | 7 | 6 | 0 | 0 | 0 |
| ietf-network-inventory-topology.lexicon.ttl | ietf | 5 | 5 | 0 | 0 | 0 | 3 | 3 | 0 | 0 | 0 |
| ietf-network-inventory.lexicon.ttl | ietf | 33 | 38 | 1 | 0 | 0 | 10 | 8 | 0 | 0 | 0 |
| ietf-network-state.lexicon.ttl | ietf | 22 | 39 | 1 | 0 | 0 | 11 | 6 | 0 | 0 | 0 |
| ietf-network-topology.lexicon.ttl | ietf | 6 | 6 | 0 | 0 | 0 | 2 | 2 | 0 | 0 | 0 |
| ietf-network.lexicon.ttl | ietf | 82 | 342 | 9 | 0 | 0 | 22 | 14 | 4 | 0 | 4 |
| ietf-ni-location-txt.lexicon.ttl | ietf | 4 | 4 | 0 | 0 | 0 | 4 | 4 | 0 | 0 | 0 |
| ietf-nwi-passive-inventory.lexicon.ttl | ietf | 47 | 47 | 0 | 0 | 5 | 9 | 9 | 5 | 0 | 5 |
| ietf-power-and-energy.lexicon.ttl | ietf | 22 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ietf-routing-types.lexicon.ttl | ietf | 13 | 13 | 0 | 0 | 1 | 2 | 2 | 1 | 0 | 1 |
| ietf-te-topology-state.lexicon.ttl | ietf | 10 | 28 | 0 | 0 | 0 | 2 | 2 | 0 | 0 | 0 |
| ietf-te-topology.lexicon.ttl | ietf | 35 | 35 | 0 | 0 | 0 | 34 | 34 | 0 | 0 | 0 |
| ietf-te-types.lexicon.ttl | ietf | 214 | 214 | 0 | 0 | 3 | 29 | 29 | 3 | 0 | 3 |
| simap-yang.lexicon.ttl | ietf | 20 | 20 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 1 |
| tapi-common.lexicon.ttl | tapi | 670 | 2571 | 360 | 0 | 1 | 143 | 59 | 317 | 0 | 317 |
| tapi-connectivity.lexicon.ttl | tapi | 60 | 60 | 3 | 0 | 0 | 19 | 19 | 3 | 0 | 3 |
| tapi-digital-otn.lexicon.ttl | tapi | 81 | 81 | 29 | 0 | 0 | 18 | 18 | 29 | 0 | 29 |
| tapi-dsr.lexicon.ttl | tapi | 40 | 40 | 40 | 0 | 0 | 0 | 0 | 40 | 0 | 40 |
| tapi-equipment.lexicon.ttl | tapi | 64 | 64 | 15 | 0 | 0 | 27 | 27 | 15 | 0 | 15 |
| tapi-eth.lexicon.ttl | tapi | 162 | 162 | 40 | 0 | 0 | 37 | 37 | 40 | 0 | 40 |
| tapi-fm.lexicon.ttl | tapi | 38 | 38 | 18 | 0 | 0 | 8 | 8 | 18 | 0 | 18 |
| tapi-gnmi-streaming.lexicon.ttl | tapi | 20 | 20 | 3 | 0 | 0 | 12 | 12 | 3 | 0 | 3 |
| tapi-notification.lexicon.ttl | tapi | 18 | 18 | 2 | 0 | 0 | 8 | 8 | 2 | 0 | 2 |
| tapi-oam.lexicon.ttl | tapi | 74 | 74 | 22 | 0 | 0 | 16 | 16 | 22 | 0 | 22 |
| tapi-path-computation.lexicon.ttl | tapi | 39 | 39 | 9 | 0 | 0 | 9 | 9 | 9 | 0 | 9 |
| tapi-photonic-media.lexicon.ttl | tapi | 157 | 157 | 38 | 0 | 0 | 35 | 35 | 38 | 0 | 38 |
| tapi-streaming.lexicon.ttl | tapi | 286 | 963 | 96 | 0 | 0 | 93 | 46 | 65 | 0 | 65 |
| tapi-topology.lexicon.ttl | tapi | 59 | 59 | 6 | 0 | 1 | 24 | 24 | 7 | 0 | 7 |
| tapi-virtual-network.lexicon.ttl | tapi | 9 | 9 | 3 | 0 | 0 | 3 | 3 | 3 | 0 | 3 |

## How to read this

- `canonical_example_present` reads 0 across the whole corpus by design, not as a defect: `draft_lexicon.py` never fabricates a canonical example (`docs/reference-lexicons.md` recommendation #2, "disambiguate, don't decorate") -- canonical-example curation is a deferred v2 concern (REQUIREMENTS.md PROV-02).
- `needs_curation` alone understates thin evidence (verified: 1 flagged vs. 1,057 `"none"`-valued entries in one file alone, lexicon-69m.13) -- use `thin_evidence` as the escalation-volume number for Phase 5 planning instead.
