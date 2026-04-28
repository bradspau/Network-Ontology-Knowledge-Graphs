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

##################
The following document provides a comprehensive technical explanation of the `yang4owl.py` translation engine. It details how specific YANG constructs are addressed—including a deep dive into identity mapping and augmentation handling—as well as the advanced adaptations implemented to enable a high-performance semantic layer.

---

Technical Review: YANG to OWL Translation Engine

1. YANG Construct Addressal Mapping
The script performs a structural mapping of YANG primitives into equivalent OWL/RDF counterparts while ensuring semantic integrity.

* `container` and `list` Statements: These are translated into `owl:Class` definitions. The script creates a hierarchical class structure reflecting the nested nature of the original YANG tree.
* `leaf` and `leaf-list` Statements: Mapping is based on the data type. Standard types (string, boolean, decimal) become `owl:DatatypeProperty`. Boolean literals are optimized as bare `true` or `false` values for native interpretation by Stardog as `xsd:boolean`.
* `leafref` Statements: Recognized as relational keys and converted into `owl:ObjectProperty`, enabling native graph traversal between instances.
* `identity` and `identityref` Mapping Logic:
    * Identities as Classes: Every YANG `identity` is primarily mapped as an `owl:Class`. If an identity has a `base` statement, the script creates an `rdfs:subClassOf` relationship.
    * Identities as Individuals: Identities are also instantiated as `owl:NamedIndividual` of their respective classes to support value-based assignment in `identityref` leaves.
    * `identityref` Conversion: When a leaf is of type `identityref`, the script generates an `owl:ObjectProperty` rather than a string. The `rdfs:range` is set to the `owl:Class` corresponding to the `base` identity.
* `choice` and `case` Statements: Treated as logical branches, generating disjoint classes to adhere to the exclusive nature of a YANG choice.
* `grouping` and `uses` Statements: Function as reusable templates. The script performs "NESTED GROUPING RESOLUTION" and "GROUPING EXPANSION WITH REFINE" to ensure the TBOX reflects the final applied configuration.
* `augment` Statements: Handled by extending the target schema and resolving all cross-file dependencies through the following mechanisms:
    * Target Path Resolution: Identifies the absolute target path across module boundaries and resolves dependencies to locate the specific OWL class representing the target.
    * Logical Property Injection: Injects new nodes as properties of the target class. Sub-containers/lists become new classes linked via `owl:ObjectProperty`; leaves are added as `owl:DatatypeProperty` or `owl:ObjectProperty` with the target class as their domain.
    * Monolithic Integration: Processes all augmented modules in a single run to correctly map URIs to the augmenting module’s namespace and expand any groupings within the augment block.
    * SHACL Extension: Any constraints in the augmentation (e.g., `mandatory true`) are added as new SHACL shapes targeting the extended properties.

---

2. Semantic Linking and Optimization Adaptations
Beyond basic mapping, the script implements post-translation optimizations to transform rigid trees into traversable knowledge graphs.

#A. Structural Flattening (Choice/Case Removal)
The script removes `case` statements as intermediate classes, attaching properties directly to the parent container instance. This reduces the number of "hops" required for graph traversal, significantly increasing query performance.

#B. Enhanced Identity Management
Utilizes URI-based identity mapping instead of string matching. For instance, a port's type is mapped directly to an IRI (e.g., `<.../nwiPassId/active-device>`). This supports multi-level discovery via hierarchical inference; a query for "all physical network elements" will return all sub-classes defined from IANA hardware identities.

#C. SHACL Isolation and Validation
Because OWL uses an "open-world" assumption, the script isolates strict YANG "closed-world" constraints into a separate SHACL graph. This allows Stardog’s Integrity Constraint Validation (ICV) engine to enforce rules without polluting the logical consistency of the ontology.

#D. Cable and Fiber Path Optimization
Specialized logic flattens cable and fiber representations, mapping reified structures like `cable -> a-end -> device-type` into clear logical endpoints[cite: 1638]. This enables the use of native URI references (`ex:device-ref`) and port-level linkages (`ex:port-ref`) to ensure unbroken path traces.

#E. Hardware Hierarchy and State Management
Maintains deep hardware hierarchies (Chassis -> Module -> Port) using `ex:parent` references[cite: 1640]. Operational states are separated into reified state containers (`hwComp:hasState`), allowing complex reasoning, such as finding all ports affected by a specific chassis failure.

#######

    -----------------------------------------
    Because I forget...

    YANG to OWL/RDF Processing Map
    YANG Construct      OWL/RDF Treatment               Reasoning Category          Domain (rdfs:domain)        Range (rdfs:range)
    container           owl:Class                       Class Logic                 N/A                         N/A
    list                owl:Class                       Class Logic                 N/A                         N/A
    leaf (Standard)     owl:DatatypeProperty            Data Assertions             Parent Class URI            XSD Type (e.g., xsd:boolean)
    leaf (identityref)  owl:ObjectProperty              Semantic Relationship       Parent Class URI            Base Identity Class URI 
    leaf (leafref)      owl:ObjectProperty              Semantic Relationship       Parent Class URI            Target Class URI
    leaf (union)        owl:ObjectProperty              Logic Profile Compatibility Parent Class URI            Created Union Parent Class 
    leaf (instance-id)  owl:ObjectProperty              Meta-referencing            Parent Class URI            N/A (Tagged with metadata) 
    identity            owl:Class & NamedIndividual     Individual Punning          N/A                         N/A
    identity (base)     rdfs:subClassOf                 Transitive Hierarchy        Specific Identity URI       Base Identity URI 
    grouping            Abstract owl:Class              Template Modeling           N/A                         N/A
    uses                Nested Grouping Resolution      Schema Flattening           Target Class URI            Resolved Property Range 
    choice/case         Structural Flattening           Query Optimisation          Parent Container URI        Resolved Property Range 
    typedef             sh:NodeShape                    Constraint Validation       N/A                         N/A
    enum (Definition)   owl:Class & NamedIndividual     Categorical Hierarchy       N/A                         N/A
    rpc                 owl:Class                       Functional Modeling         N/A                         N/A 
    notification        owl:Class                       Functional Modeling         N/A                         N/A
    must/when           sh:condition/sh:deactivated     Conditional Logic           Property/Class URI          N/A (SHACL Filter) 
    augment             logical Property Injection      Monolithic integration      Augmented Target Class      Injected Node Class Type       
    Child Containment   has[ChildName] (ObjectProperty) Structural Integrity        Parent Class URI            Child Class URI 

Python yang4owl 
- execution  yang4owl.py --yang-dir <directory of the yang files> --modules <yang model to create ttl for> --base-uri <owl base uri> --output <turtle file> --verbose 

 YANG to OWL/RDF Processing Map (Updated)

 Current include capabilities
    Monolithic Augmentation: Cross-module target path resolution and dependency expansion. 

    Structural Flattening: Elimination of choice/case intermediate nodes to optimize SPARQL query depth. 

    Dual Identity Mapping: OWL Class + NamedIndividual (Punning) for hierarchical reasoning. 

    SHACL Isolation: Decoupling of "Closed-World" constraints (must/when/mandatory) from the "Open-World" OWL TBOX. 

    Enhanced Semantic Linking: Native URI-based resolution for leafref and identityref (removing string dead-ends). 

    Hardware Lineage: Parent-child traversal via ex:parent and pdev:hasPassivePort. 
    
    Provenance: PROV-O metadata mapping back to originating YANG paths. 

  
 High-Level Flow
 The main function orchestrates the translation by parsing CLI arguments and instantiates the YANGToOWL engine. The core convert() method initializes recursive schema walking, resolves cross-module dependencies, applies semantic patches (such as structural flattening), and emits a monolithic Turtle file. It simultaneously generates a secondary SHACL graph to house validation constraints.
  
 Key Helper Components
 •	YANGDependencyResolver: Utilizes pyang’s context to ingest a directory of modules, resolving import and augment paths across the entire library to ensure a complete schema.
 •	IdentityResolver: Maps YANG identities to a dual-layer OWL structure. Every identity is a Class (for taxonomy) and a NamedIndividual (for assignment), enabling reasoning over hardware classes (e.g., iana-hardware).
 •	EnhancedLeafrefResolver: Normalizes relative and current()-based XPaths into absolute class paths, converting them into URI-native owl:ObjectProperties with correct domain/range.
 •	Grouping & Refine Resolver: Handles "Grouping Expansion with Refine," injecting and modifying members during the expansion phase to ensure concrete path resolution in the TBOX.
  
 YANG → OWL Mapping & Semantic Optimization
 •	Normalized Path Registry: The engine maintains class_paths, a registry of module-qualified YANG paths to OWL Class URIs. It strips legacy prefixes (nw:, nt:) to ensure consistent global identifiers.
 •	Flattened Class Structure: For container and list nodes, the script generates owl:Class definitions. Critically, choice and case levels are discarded; their children are attached directly to the parent container, reducing graph depth for faster SPARQL traversals.
 •	Property Mapping: * Standard leaves become owl:DatatypeProperty with optimized bare literals for types like xsd:boolean.
 o	identityref and leafref become owl:ObjectProperty, pointing to URIs instead of string literals.
 •	Cross-Module Augmentation: Augments are re-anchored to the target path. The engine stubs target classes across modules and injects children as if physically present, ensuring a seamless graph even when spanning multiple IETF models.
  
 Hardware & Topology Logic
 •	Active Hardware Lineage: Utilizes ex:parent (ObjectProperty) to map child components (Ports/Modules) back to their parent Chassis, allowing hierarchical failure analysis.
 •	Passive Device Connectivity: Implements an explicit device-to-port link via pdev:hasPassivePort to close the structural gap between passive devices and their physical interfaces.
 •	Reified Path Tracing: For topology (Cables), the script generates reified a-end and z-end instances, utilizing ex:ne-ref (Active), ex:device-ref (Passive), and ex:port-ref to create an unbroken semantic chain across domains.
  
 Constraint & Metadata Handling
 •	SHACL Constraint Isolation: Instead of polluting the TBOX with restrictive logic, must, when, and mandatory statements are translated into sh:NodeShape and sh:property constraints. This allows for strict validation in Stardog while keeping the TBOX flexible for reasoning.
 •	XSD & Datatype Restrictions: Translates YANG range, length, and pattern restrictions into owl:withRestrictions facets, connected to specific typedef classes.
 •	PROV Metadata: Every generated entity is annotated with prov:wasDerivedFrom, providing a direct audit trail back to the specific YANG module, line number, and element type.
 
  
  


-----------------------------------
Queries on the plant data

Find all cables connected to an enclosure 
    PREFIX ex:      <http://www.huawei.com/ontology/>
    PREFIX inv-cab: <http://www.huawei.com/instances/cable/>
    PREFIX inv-dev: <http://www.huawei.com/instances/passive-device/>
    PREFIX nwi:     <http://www.huawei.com/ontology/ietf-network-inventory/network-inventory/>

    SELECT ?cable ?cableId ?end ?endDevice ?endDeviceId ?length
    WHERE {
        ?cable  a         nwi:cable ;
                ex:id     ?cableId ;
                ex:length ?length .

        { ?cable ex:hasAEnd ?endNode . BIND("a-end" AS ?end) }
        UNION
        { ?cable ex:hasZEnd ?endNode . BIND("z-end" AS ?end) }

        ?endNode ex:device-ref inv-dev:enclosure_166 .
        BIND(inv-dev:enclosure_166 AS ?endDevice)

        OPTIONAL { inv-dev:enclosure_166 ex:id ?endDeviceId }
    }
    ORDER BY ?cableId


    Output 
    inv-cab:cable_114	"cable_114"	"z-end"	inv-dev:enclosure_166	"enclosure_166"	60.0
    inv-cab:cable_124	"cable_124"	"a-end"	inv-dev:enclosure_166	"enclosure_166"	78.07
    inv-cab:cable_126	"cable_126"	"a-end"	inv-dev:enclosure_166	"enclosure_166"	97.1
    inv-cab:cable_186	"cable_186"	"a-end"	inv-dev:enclosure_166	"enclosure_166"	58.9
    inv-cab:cable_296	"cable_296"	"a-end"	inv-dev:enclosure_166	"enclosure_166"	102.1
    inv-cab:cable_406	"cable_406"	"a-end"	inv-dev:enclosure_166	"enclosure_166"	78.07

Find all cables at enclousure and provide device at the other end
    PREFIX ex:      <http://www.huawei.com/ontology/>
    PREFIX inv-cab: <http://www.huawei.com/instances/cable/>
    PREFIX inv-dev: <http://www.huawei.com/instances/passive-device/>

    SELECT ?cable ?cableId ?end ?otherDevice ?length
    WHERE {
        ?cable ex:id ?cableId ;
            ex:length ?length .

        { ?cable ex:hasAEnd ?thisEnd ; ex:hasZEnd ?otherEnd . BIND("a-end" AS ?end) }
        UNION
        { ?cable ex:hasZEnd ?thisEnd ; ex:hasAEnd ?otherEnd . BIND("z-end" AS ?end) }

        ?thisEnd  ex:device-ref inv-dev:enclosure_166 .
        ?otherEnd ex:device-ref ?otherDevice .
    }
    ORDER BY ?cableId



    Output
    inv-cab:cable_114	"cable_114"	"z-end"	inv-dev:enclosure_106	60.0
    inv-cab:cable_124	"cable_124"	"a-end"	inv-dev:ATB_202	78.07
    inv-cab:cable_126	"cable_126"	"a-end"	inv-dev:ATB_3874	97.1
    inv-cab:cable_186	"cable_186"	"a-end"	inv-dev:ATB_3661	58.9
    inv-cab:cable_296	"cable_296"	"a-end"	inv-dev:ATB_3854	102.1
    inv-cab:cable_406	"cable_406"	"a-end"	inv-dev:ATB_3554	78.07

Query all upstream cables and devices from ONT_1. Unfortaunetly cannot tell where the splitter are in the enclosures and hence what ONT are on the same feed as the OLT....bit of a shame.

ONT_1 (port_Uplink_1)
  └── cable_ONT_1_to_ATB3874 ── ATB_3874
        └── cable_126 ── enclosure_166 (FAT)
              ├── cable_114 ──┐
              ├── cable_145 ──┤
              └── cable_156 ──┤ enclosure_106 (FDT)
              └── cable_263 ──┘
                    ├── enclosure_81 (FAT) → drop cables → ATB_2531, ATB_2377, ATB_3457, ATB_2210, ATB_2190, ATB_2857, ATB_472
                    ├── enclosure_28 (FAT) → drop cables → ATB_503, ATB_2534, ATB_1468, ATB_1278, ATB_1222, ATB_273
                    └── enclosure_60 (FAT) → drop cables → ATB_2344, ATB_2777, ATB_1289, ATB_3458, ATB_1522, ATB_747

    Query

    PREFIX ex:        <http://www.huawei.com/ontology/>
    PREFIX inv-cab:   <http://www.huawei.com/instances/cable/>
    PREFIX inv-dev:   <http://www.huawei.com/instances/passive-device/>
    PREFIX nwi:       <http://www.huawei.com/ontology/ietf-network-inventory/network-inventory/>
    PREFIX rdf:       <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT DISTINCT
        ?cable
        ?cableId
        ?cableRole
        ?cableLength
        ?aEndDevice
        ?aEndDeviceId
        ?aEndPort
        ?zEndDevice
        ?zEndPort
    WHERE {
        {
            BIND(inv-cab:cable_ONT_1_to_ATB3874 AS ?cable)
        }
        UNION
        {
            inv-cab:cable_ONT_1_to_ATB3874
                ex:hasZEnd /
                ex:device-ref /
                ^ex:device-ref /
                ^( ex:hasAEnd | ex:hasZEnd ) /
                (
                    ( ex:hasAEnd | ex:hasZEnd ) /
                    ex:device-ref /
                    ^ex:device-ref /
                    ^( ex:hasAEnd | ex:hasZEnd )
                )*
            ?cable .
        }

        ?cable ex:id ?cableId .

        OPTIONAL { ?cable ex:length     ?cableLength }
        OPTIONAL { ?cable ex:cable-role ?cableRole   }

        OPTIONAL {
            ?cable ex:hasAEnd ?aEnd .
            OPTIONAL { ?aEnd ex:device-ref    ?aEndDevice   }
            OPTIONAL { ?aEnd ex:device-id     ?aEndDeviceId }
            OPTIONAL { ?aEnd ex:component-ref ?aEndPort     }
        }

        OPTIONAL {
            ?cable ex:hasZEnd ?zEnd .
            OPTIONAL { ?zEnd ex:device-ref ?zEndDevice }
            OPTIONAL { ?zEnd ex:port-ref   ?zEndPort   }
        }
    }
    ORDER BY ?cableId

    Output 
    inv-cab:cable_114	"cable_114"	nwiPassId:access-cable	60.0	inv-dev:enclosure_106			inv-dev:enclosure_166	
    inv-cab:cable_118	"cable_118"	nwiPassId:drop-cable	30.77	inv-dev:enclosure_28			inv-dev:ATB_503	
    inv-cab:cable_124	"cable_124"	nwiPassId:drop-cable	78.07	inv-dev:enclosure_166			inv-dev:ATB_202	
    inv-cab:cable_126	"cable_126"	nwiPassId:drop-cable	97.1	inv-dev:enclosure_166			inv-dev:ATB_3874	
    inv-cab:cable_134	"cable_134"	nwiPassId:drop-cable	61.3	inv-dev:enclosure_81			inv-dev:ATB_2531	
    inv-cab:cable_135	"cable_135"	nwiPassId:drop-cable	54.7	inv-dev:enclosure_81			inv-dev:ATB_2377	
    inv-cab:cable_145	"cable_145"	nwiPassId:access-cable	45.0	inv-dev:enclosure_106			inv-dev:enclosure_81	
    inv-cab:cable_152	"cable_152"	nwiPassId:drop-cable	33.3	inv-dev:enclosure_60			inv-dev:ATB_2344	
    inv-cab:cable_156	"cable_156"	nwiPassId:access-cable	97.0	inv-dev:enclosure_106			inv-dev:enclosure_28	
    inv-cab:cable_171	"cable_171"	nwiPassId:drop-cable	59.7	inv-dev:enclosure_81			inv-dev:ATB_3457	
    inv-cab:cable_184	"cable_184"	nwiPassId:drop-cable	30.77	inv-dev:enclosure_28			inv-dev:ATB_2534	
    inv-cab:cable_186	"cable_186"	nwiPassId:drop-cable	58.9	inv-dev:enclosure_166			inv-dev:ATB_3661	
    inv-cab:cable_204	"cable_204"	nwiPassId:drop-cable	59.7	inv-dev:enclosure_81			inv-dev:ATB_2210	
    inv-cab:cable_259	"cable_259"	nwiPassId:drop-cable	30.77	inv-dev:enclosure_28			inv-dev:ATB_1468	
    inv-cab:cable_263	"cable_263"	nwiPassId:access-cable	139.0	inv-dev:enclosure_106			inv-dev:enclosure_60	
    inv-cab:cable_265	"cable_265"	nwiPassId:drop-cable	66.3	inv-dev:enclosure_81			inv-dev:ATB_2190	
    inv-cab:cable_283	"cable_283"	nwiPassId:drop-cable	79.61	inv-dev:enclosure_60			inv-dev:ATB_2777	
    inv-cab:cable_296	"cable_296"	nwiPassId:drop-cable	102.1	inv-dev:enclosure_166			inv-dev:ATB_3854	
    inv-cab:cable_404	"cable_404"	nwiPassId:drop-cable	64.32	inv-dev:enclosure_28			inv-dev:ATB_1278	
    inv-cab:cable_406	"cable_406"	nwiPassId:drop-cable	78.07	inv-dev:enclosure_166			inv-dev:ATB_3554	
    inv-cab:cable_409	"cable_409"	nwiPassId:drop-cable	79.61	inv-dev:enclosure_60			inv-dev:ATB_1289	
    inv-cab:cable_418	"cable_418"	nwiPassId:drop-cable	66.3	inv-dev:enclosure_81			inv-dev:ATB_2857	
    inv-cab:cable_46	"cable_46"	nwiPassId:drop-cable	33.3	inv-dev:enclosure_60			inv-dev:ATB_3458	
    inv-cab:cable_54	"cable_54"	nwiPassId:drop-cable	79.61	inv-dev:enclosure_60			inv-dev:ATB_1522	
    inv-cab:cable_67	"cable_67"	nwiPassId:drop-cable	64.32	inv-dev:enclosure_28			inv-dev:ATB_1222	
    inv-cab:cable_81	"cable_81"	nwiPassId:drop-cable	30.77	inv-dev:enclosure_28			inv-dev:ATB_273	
    inv-cab:cable_82	"cable_82"	nwiPassId:drop-cable	59.7	inv-dev:enclosure_81			inv-dev:ATB_472	
    inv-cab:cable_84	"cable_84"	nwiPassId:drop-cable	79.61	inv-dev:enclosure_60			inv-dev:ATB_747	
    inv-cab:cable_ONT_1_to_ATB3874	"cable_ONT_1_to_ATB3874"	nwiPassId:access			"ONT_1"	inv-ne:ONT_1_port_Uplink_1	inv-dev:ATB_3874	
