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

Python yang4owl 
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

  
    -----------------------------------------
    Because I forget...

    YANG to OWL/RDF Processing Map
    YANG Construct      OWL/RDF Treatment               Reasoning Category          Domain (rdfs:domain)        Range (rdfs:range)
    container           owl:Class                       Class Logic                 N/A                         N/A
    list                owl:Class                       Class Logic                 N/A                         N/A
    leaf (Standard)     owl:DatatypeProperty            Data Assertions             Parent Class URI            XSD Type (e.g., xsd:string)
    leaf (identityref)  owl:ObjectProperty              Semantic Relationship       Parent Class URI            Base Identity Class URI 
    leaf (leafref)      owl:ObjectProperty              Semantic Relationship       Parent Class URI            Target Class URI
    leaf (union)        owl:ObjectProperty              Logic Profile Compatibility Parent Class URI            Created Union Parent Class 
    leaf (instance-id)  owl:ObjectProperty              Meta-referencing            Parent Class URI            N/A (Tagged with metadata) 
    identity            owl:Class & NamedIndividual     Individual Punning          N/A                         N/A
    identity (base)     rdfs:subClassOf                 Transitive Hierarchy        Specific Identity URI       Base Identity URI 
    grouping            Abstract owl:Class              Template Modeling           N/A                         N/A
    uses                Local Property Generation       Schema Flattening           Target Class URI            Resolved Property Range 
    choice/case         owl:disjointWith                Mutual Exclusivity          Case-holding Class          Opposing Case Class 
    typedef             sh:NodeShape                    Constraint Validation       N/A                         N/A
    enum (Definition)   owl:Class & NamedIndividual     Categorical Hierarchy       N/A                         N/A
    rpc                 owl:Class                       Functional Modeling         N/A                         N/A 
    notification        owl:Class                       Functional Modeling         N/A                         N/A
    must/when           sh:condition/sh:deactivated     Conditional Logic           Property/Class URI          N/A (XPath Literal) 
    Child Containment   has[ChildName] (ObjectProperty) Structural Integrity    Parent Class URI            Child Class URI 

-----------------------------------
Queries on the plant data

Find all cables connected to an enclosure 
    PREFIX ex:       <http://www.huawei.com/ontology/>
    PREFIX inv-cab:  <http://www.huawei.com/instances/cable/>
    PREFIX inv-dev:  <http://www.huawei.com/instances/passive-device/>

    SELECT ?cable ?cableId ?end ?endDeviceId ?length
    WHERE {
    # Match cables where either a-end or z-end device-id = "enclosure_166"
    ?cable  a                <http://www.huawei.com/ontology/ietf-network-inventory/network-inventory/cable> ;
            ex:id            ?cableId ;
            ex:length        ?length .

    { ?cable ex:hasAEnd ?endNode . BIND("a-end" AS ?end) }
    UNION
    { ?cable ex:hasZEnd ?endNode . BIND("z-end" AS ?end) }

    ?endNode ex:device-id "enclosure_166" .
    ?endNode ex:device-id ?endDeviceId .
    }
    ORDER BY ?cableId

    Output 
    http://www.huawei.com/instances/cable/cable_106	"cable_106"	"a-end"	"enclosure_166"	102.1
    http://www.huawei.com/instances/cable/cable_114	"cable_114"	"z-end"	"enclosure_166"	60.0
    http://www.huawei.com/instances/cable/cable_124	"cable_124"	"a-end"	"enclosure_166"	78.07
    http://www.huawei.com/instances/cable/cable_126	"cable_126"	"a-end"	"enclosure_166"	97.1
    http://www.huawei.com/instances/cable/cable_186	"cable_186"	"a-end"	"enclosure_166"	58.9
    http://www.huawei.com/instances/cable/cable_296	"cable_296"	"a-end"	"enclosure_166"	102.1
    http://www.huawei.com/instances/cable/cable_399	"cable_399"	"a-end"	"enclosure_166"	88.07
    http://www.huawei.com/instances/cable/cable_406	"cable_406"	"a-end"	"enclosure_166"	78.07
    http://www.huawei.com/instances/cable/cable_441	"cable_441"	"a-end"	"enclosure_166"	97.1

FInd all cables at enclousure and provide device at the other end
    PREFIX ex:       <http://www.huawei.com/ontology/>
    PREFIX inv-cab:  <http://www.huawei.com/instances/cable/>
    PREFIX inv-dev:  <http://www.huawei.com/instances/passive-device/>

    SELECT ?cable ?cableId ?end ?otherDeviceId ?length
    FROM <n2>
    WHERE {
    ?cable ex:id ?cableId ; ex:length ?length .

    { ?cable ex:hasAEnd ?thisEnd ; ex:hasZEnd ?otherEnd . BIND("a-end" AS ?end) }
    UNION
    { ?cable ex:hasZEnd ?thisEnd ; ex:hasAEnd ?otherEnd . BIND("z-end" AS ?end) }

    ?thisEnd  ex:device-id "enclosure_166" .
    ?otherEnd ex:device-id ?otherDeviceId .
    }
    ORDER BY ?cableId


    Output
    http://www.huawei.com/instances/cable/cable_106	"cable_106"	"a-end"	"ATB_3469"	102.1
    http://www.huawei.com/instances/cable/cable_114	"cable_114"	"z-end"	"enclosure_106"	60.0
    http://www.huawei.com/instances/cable/cable_124	"cable_124"	"a-end"	"ATB_202"	78.07
    http://www.huawei.com/instances/cable/cable_126	"cable_126"	"a-end"	"ATB_3874"	97.1
    http://www.huawei.com/instances/cable/cable_186	"cable_186"	"a-end"	"ATB_3661"	58.9
    http://www.huawei.com/instances/cable/cable_296	"cable_296"	"a-end"	"ATB_3854"	102.1
    http://www.huawei.com/instances/cable/cable_399	"cable_399"	"a-end"	"ATB_2051"	88.07
    http://www.huawei.com/instances/cable/cable_406	"cable_406"	"a-end"	"ATB_3554"	78.07
    http://www.huawei.com/instances/cable/cable_441	"cable_441"	"a-end"	"ATB_1448"	97.1