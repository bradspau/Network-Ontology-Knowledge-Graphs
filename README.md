Mechanism for converting yang to owl for network semantic modeling

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
- yang4owl.py - yang4owl in python
- simap-ontology-python.ttl - current output of yang2owl for comparison against the IETF 124 simap-rdfs-schema

Python yang2owl 
- execution  yang4owl.py --yang-dir <directory of the yang files> --modules <yang model to create ttl for> --base-uri <owl base uri> --output <turtle file> --verbose 

    Current include capabilities
    - import
    - augment
    - contraints
    - datatype restrictions
    - individual enumeration
    - grouping
    - leafref
    - identityref
    - provenance to the yang models

    The script loads one or more YANG modules with pyang, walks their schema tree, and emits an OWL/RDF ontology (Turtle) describing the model, including constraints and metadata.
    ​

    High-level flow
        The main function parses CLI arguments (YANG directory, main module name, base URI, output path, verbosity), sets up logging, and instantiates YANGToOWL with the chosen directory and base URI, then calls convert().
        ​

        YANGToOWL.convert() loads all YANG modules, initializes helper resolvers, processes modules into OWL classes and properties, then writes the resulting RDF graph to a Turtle file and logs counts of generated triples, constraints, enumerations, etc.
    ​

    Key helper components
        YANGDependencyResolver uses pyang’s Context and FileRepository to load the main module and any other .yang files in the directory, exposing them in self.modules.
    ​

        YANGTypeResolver maps YANG built-in types and typedef chains to XSD datatypes and cooperates with YANGConstraintExtractor to collect range, length, and pattern constraints from type statements.
    ​

        IdentityResolver scans all modules for identity statements and records their base identities and descriptions, enabling later creation of an OWL class hierarchy for identities.
        ​

        EnhancedLeafrefResolver resolves leafref XPaths (absolute, relative, and current()-based), normalizes paths, and maps them to class_paths so that leafrefs become OWL object properties with correct domain/range.
        ​

        GroupingResolver plus RefineResolver and GroupingContextTracker collect grouping definitions, expand uses (including nested ones), and apply refine statements, treating groupings as abstract OWL classes and expanding their members into concrete locations.
    ​

    YANG → OWL mapping
        The central class YANGToOWL maintains an rdflib Graph, an ex namespace (from the base URI), and registries such as class_paths (normalized YANG path → OWL class URI) and various counters.
        ​

        _normalize_path() adds module-qualified prefixes and strips prefixes like nw:/nt:/st: to produce consistent, fully-qualified paths such as /ietf-network/networks/network, keyed by the current module name.
        ​

        For top-level and nested container/list/leaf nodes, _process_module(), _process_container(), _process_list(), and _process_leaf() create OWL classes (for containers/lists) and datatype or object properties (for leaves), attach labels/descriptions, and register the normalized path in class_paths.
        ​

        identityref leaves become OWL object properties whose range is the identity class; leafref leaves become object properties linked to the referenced container/list class via the resolved path; other leaves become datatype properties with appropriate XSD range.
        ​

        augment statements are normalized to a target path (e.g. re-anchored under ietf-network for /networks/...), the target class is stubbed if needed, and the augment’s children are processed as if physically present under the target.
    ​

    Additional features
        Groupings are first turned into abstract OWL classes, then all uses statements (in modules and within augments) are expanded so that grouping members are materialized and optionally constrained (e.g. mandatory → minCardinality 1).
        ​

        _process_containers_for_properties() adds synthetic containment properties (hasChildName) between parent and child classes for each one-level path extension in class_paths.
        ​

        _generate_cardinality_constraints() sets a default minCardinality 0 on every OWL object property, then refine/uses logic can add stricter constraints.
        ​

        _process_xsd_constraints() and _add_constraint_triples() traverse typedefs and leaves to emit XSD-based constraint triples (min/max inclusive, min/max length, patterns) for elements, while _create_owl_datatype_restrictions() builds OWL datatype restriction datatypes and connects typedef classes to them.
        ​

        _process_enumerations() and _create_enumeration_class() turn YANG enumeration typedefs into OWL classes plus NamedIndividuals for each enum value, including labels and optional descriptions.
        ​

        _add_prov_metadata() annotates every class, datatype property, and object property with PROV wasDerivedFrom URIs that encode the originating YANG path and element type.
    ​

  Updated v4.6 to v4.7
  Design decisions made on the conversion to align the semantics with performance and use with commercial reasoners and w3c RL profile. In short, avoiding the full description logic.

    1. Yang union allows a leaf to be one of several types. OWL only supports one type as data property. Challenge is that owl unionof is typically not supported in commercial reasoners due to it being out of polynomial time reasoning. Inclusion of owl unionof probably would imly a user to refactor the turtle. 

    2. Mappng of yag choices and cases. In yang this makes data nodes mutually excusive. Challenge map these to owl class disjointness.

    3. Identityref to ObjectProperty Trasition. Script currently recognises identityref and tags the leaf as ObjectProperty and points it to a specific URI. Treting the identity as a flat value rater than navigable class. _process_identitie processes identites as rdfs:subClassOf but when processing the leaf links to the base identity inhibiting a reasoner to understand sub-identity is a valid value. The rdfs:range of the ObjectProperty should point to individuals that are members of the identity class (or subclass).

    4. Yang instance-identifier is a path that points to specific instance in the data tree.  ie. foreign key. Challenge convert instance-identifier to Functional Object Properties.

    5. Yang must and when statement contain complex xpath logic that owl would struggle to express. Challenge to update the SHACL generation.

    