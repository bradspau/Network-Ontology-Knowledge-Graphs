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

  Updated v4.6 to v4.7
  Design decisions made on the conversion to align the semantics with performance and use with commercial reasoners and w3c RL profile. In short, avoiding the full description logic.

    1. Yang union allows a leaf to be one of several types. OWL only supports one type as data property. Challenge is that owl unionof is typically not supported in commercial reasoners due to it being out of polynomial time reasoning. Inclusion of owl unionof probably would imply a user would refactor the turtle. 

    2. Mapping of yag choices and cases. In yang this makes data nodes mutually excusive. Challenge map these to owl class disjointness.

    3. Identityref to ObjectProperty Trasition. Script currently recognises identityref and tags the leaf as ObjectProperty and points it to a specific URI. Treting the identity as a flat value rater than navigable class. _process_identitie processes identites as rdfs:subClassOf but when processing the leaf links to the base identity inhibiting a reasoner to understand sub-identity is a valid value. The rdfs:range of the ObjectProperty should point to individuals that are members of the identity class (or subclass).

    4. Yang instance-identifier is a path that points to specific instance in the data tree.  ie. foreign key. Challenge convert instance-identifier to Functional Object Properties.

    5. Yang must and when statement contain complex xpath logic that owl would struggle to express. Challenge to update the SHACL generation.

    
    IETF-ni-location.ttl

    Running the script upon the ietf-ni-location.yang file subsumes all the associated yang files referenced by that module before producing the output. If running the script on ietf-ni-location.yang there for produces the ontology ietf-ni-location-python.ttl which includes the iana-hardware, ietf-hardware, ietf-network-topology and ietf-nwi-passive-topology. 

    I probably could have, should have, made these separate ontologies but is what it is. 


(Still testing these as I am having dificulty creating variants)
What is at a location? (works)
PREFIX ex: <http://www.huawei.com/ontology/>
PREFIX hw: <http://www.huawei.com/ontology/ietf-hardware/hardware/component/>
PREFIX nwi: <http://www.huawei.com/ontology/ietf-network-inventory-txt/network-inventory/network-elements/network-element/>
PREFIX nil: <http://www.huawei.com/ontology/ietf-ni-location/locations/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>


SELECT ?location ?id ?name ?type
WHERE {
  ?location a nil:location .
  ?location ex:id ?id .
  ?location ex:name ?name .
  ?location ex:type ?type .
}

Chassis Query (works)
    Query to retrieve the unique identifier of every component classified as a chassis, its name, the specific network element and its physical location. 

PREFIX ex: <http://www.huawei.com/ontology/>
PREFIX hw: <http://www.huawei.com/ontology/ietf-hardware/hardware/component/>
PREFIX nwi: <http://www.huawei.com/ontology/ietf-network-inventory-txt/network-inventory/network-elements/network-element/>
PREFIX nil: <http://www.huawei.com/ontology/ietf-ni-location/locations/location/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX components: <http://www.huawei.com/ontology/ietf-network-inventory-txt/network-inventory/network-elements/network-element/components/>
PREFIX identity: <http://www.huawei.com/ontology/identity/>
PREFIX loc: <http://www.huawei.com/ontology/ietf-ni-location/locations/>

SELECT ?componentID ?componentName ?networkElementID ?locationID
WHERE {
# 1. Find components classified specifically as a 'chassis'
?component a components:component ;
            ex:class identity:chassis ;
            ex:component-id ?componentID .
# 2. Get the optional human-readable name
    OPTIONAL { ?component ex:name ?componentName . }
# 3. Trace which Network Element (NE) this component belongs to
    ?ne components:hasComponent ?component ;
        ex:ne-id ?networkElementID ;   
        loc:location ?locationID .}
ORDER BY ?locationID ?networkElementID

Locate Chassis in location (works)

PREFIX ex: <http://www.huawei.com/ontology/>
PREFIX hw: <http://www.huawei.com/ontology/ietf-hardware/hardware/component/>
PREFIX nwi: <http://www.huawei.com/ontology/ietf-network-inventory-txt/network-inventory/network-elements/network-element/>
PREFIX nil: <http://www.huawei.com/ontology/ietf-ni-location/locations/location/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX components: <http://www.huawei.com/ontology/ietf-network-inventory-txt/network-inventory/network-elements/network-element/components/>
PREFIX identity: <http://www.huawei.com/ontology/identity/>
PREFIX loc: <http://www.huawei.com/ontology/ietf-ni-location/locations/>

   SELECT ?componentID ?componentName ?networkElementID ?locationID
    WHERE {
    # 1. Find components classified specifically as a 'chassis'
    ?component a components:component ;
                ex:class identity:chassis ;
                ex:component-id ?componentID .
    
    # 2. Get the optional human-readable name
    OPTIONAL { ?component ex:name ?componentName . }

    # 3. Trace which Network Element (NE) this component belongs to
    ?ne components:hasComponent ?component ;
        ex:ne-id ?networkElementID .

    # 4. Link the Network Element to its physical Location ID
    ?ne loc:location ?locationID .
    }
    ORDER BY ?locationID ?networkElementID

Locate chassis in rack and row <broken - should be fixed in 4.7.1 as the container/leaf in the group was not dealt with properly. Just need to test).

I've some thing missing in the schema...have to review where and why it is not present.

Power Energy Query 
    PREFIX ex: <http://www.huawei.com/ontology/>
    PREFIX eo-p: <http://www.huawei.com/ontology/ietf-power-and-energy-txt/energy-objects/power-entry/>
    PREFIX id: <http://www.huawei.com/ontology/identity/>

    SELECT ?chassisName ?locationID ?rackID ?row ?col ?currentPower
    WHERE {
    # Link Chassis to Power
    ?powerEntry a <http://www.huawei.com/ontology/ietf-power-and-energy-txt/energy-objects/power-entry> ;
                eo-p:source-component-id ?chassis ;
                eo-p:eo-power ?currentPower .
    
    ?chassis ex:name ?chassisName ;
            ex:component-id ?chID .

    # Link Chassis to Rack
    ?rack a <http://www.huawei.com/ontology/ietf-ni-location/locations/racks/rack> ;
            ex:id ?rackID ;
            <http://www.huawei.com/ontology/ietf-ni-location/locations/location/ni-location-ref> ?locationID ;
            <http://www.huawei.com/ontology/ietf-ni-location/locations/racks/rack/rack-location/row-number> ?row ;
            <http://www.huawei.com/ontology/ietf-ni-location/locations/racks/rack/rack-location/column-number> ?col ;
            <http://www.huawei.com/ontology/ietf-ni-location/locations/racks/rack/contained-chassis> ?container .
            
    ?container ex:component-ref ?chID .
    }
    ORDER BY DESC(?currentPower)




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