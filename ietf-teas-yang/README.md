# IETF TEAS YANG (vendored)

The IETF TEAS-side pairing used in the `reference-lexicons` draft's TAPI-vs-IETF
worked example (RFC 8795 over RFC 8345), vendored here so it can be aligned
against `tapi-yang/` for `lexicon-69m.5`.

Sourced from [YangModels/yang](https://github.com/YangModels/yang)
(`standard/ietf/RFC/`), the community-maintained mirror of published IETF/IANA
YANG modules:

| File | RFC |
|---|---|
| `ietf-te-topology@2020-08-06.yang` | RFC 8795 -- YANG Data Model for TE Topologies |
| `ietf-te-topology-state@2020-08-06.yang` | RFC 8795 (companion state module) |
| `ietf-te-types@2020-06-10.yang` | RFC 8776 -- Common YANG Data Types for TE |
| `ietf-routing-types@2017-12-04.yang` | RFC 8294 -- Common YANG Data Types for Routing |
| `ietf-network@2018-02-26.yang` | RFC 8345 -- copied from `simap-yang/`, same revision already used there |
| `ietf-network-topology@2018-02-26.yang` | RFC 8345 -- copied from `yang-ivy/`, same revision |
| `ietf-yang-types@2013-07-15.yang` | RFC 6991 -- copied from `yang-ivy/`, same revision |
| `ietf-inet-types@2013-07-15.yang` | RFC 6991 -- copied from `yang-ivy/`, same revision |

The last four are duplicated here rather than referenced from `simap-yang/`/
`yang-ivy/` because `yang4owl.py` requires every import to resolve within a
single `--yang-dir`; using the exact same revision already present elsewhere
in this repo avoids introducing a second, divergent copy of a shared RFC 8345
model.

IETF YANG modules are covered by the IETF Trust's usual license terms for
code embedded in RFCs (see the module text's own `Copyright (c) 2020 IETF
Trust...` boilerplate, or the RFC itself at
<https://www.rfc-editor.org/rfc/rfc8795>).
