# TAPI YANG (vendored)

YANG models from the ONMI (Linux Foundation Open Network Models and
Interfaces project) TAPI repository, vendored here as a second, independent
YANG source for lexicon-drafting and (eventually) cross-lexicon alignment
work (`lexicon-69m.5`).

- **Source**: https://github.com/Open-Network-Models-and-Interfaces-ONMI/TAPI
- **Commit**: `fb341384afc5201e7211023e36db73dce05f54c5` (2025-03-20), `develop` branch
- **Path in source repo**: `YANG/`
- **License**: Apache License 2.0 (see the source repo's `LICENSE`)
- **Version**: targets TAPI v2.6.0 tooling guidelines per the modules'
  `description` statements; generated from UML via the EAGLE UML2YANG tool

Only the `.yang` files were copied (not the `.tree` files, UML, OAS, or
protobuf artifacts from the source repo, which aren't needed here). All
imports are self-contained within this directory -- no external `ietf-*`
dependencies.

See `../lexicon/README.md` for the known granularity caveat specific to
lexicons drafted from this source (TAPI's single deeply-nested `context`
tree produces many path-distinct-but-conceptually-related entries, most
visibly in `tapi-common.lexicon.ttl`).
