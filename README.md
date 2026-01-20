Mechanism for converting yang to owl for network semantic modelin

Objective
- Investigate the options for automated conversion of yang data model to RDFS/owl 
- Build upon previous IETF KG Design team work and hackathons
- Extend/Enhance IETF124 Simap turtle and extend with linkages to IETF IVY Inventory Project with Inventory, Passive, location yang data models

Back Ground
- https://github.com/Huawei-IOAM/ietf-knowledge-graphs/tree/main - IETF124 hackathon creating semantic relationship between simap-rdfs-schema.tll and Noria ontology
- https://github.com/Huawei-IOAM/yang2rdf - IETF xxx yang to rdf ABox (data) tool
- https://gitlab.eurecom.fr/huawei/yang2rdf - yang to owl TBOX (schema) tool utilising KG-Morph/RMLMapper

Current Content
- eurecom/yang2rdf/mapping/mapping-owl-inventory.tll - very much draft RMLMapping for IETF Network Inventory (requires alot more work)
- simap-rdfs-schema.ttl - IETF124 ttl utilised for reference on either tool for validation of output
- simap-yang - all yang data models for simap
- yang-ivy - all yang models for IETF network inventory, passive, location
- yang2owl.py - yang2owl in python
- simap-ontology-python.ttl - current output of yang2owl for comparison against the IETF 124 simap-rdfs-schema

Python yang2owl 
- execution  yang2oel.py --yang-dir <directory of the yang files> --modules <yang model to create ttl for> --base-uri <owl base uri> --output <turtle file> --verbose 

