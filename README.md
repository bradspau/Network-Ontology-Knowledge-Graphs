# YANG → OWL Knowledge Graph: OSP Fibre Access Network Topology

> **Translating IETF YANG passive network inventory models into a semantically enriched OWL/RDF knowledge graph, enabling single-line SPARQL traversal from ONT devices to ODF endpoints across a fibre access network.**

> **Data contained within the project is real valid data from a network build. Rather han made up or synthisised data. The issues and problems associated with modelling with YANG are there for real world issues.**

---

## Current YANG model constraints
There are a number of issues with the existing YANG data models that inhibited being able to trace a ONT to an actual OLT port in the experiment.
1.   The Passive inventory model is a device connectivity model. This assumes that cables terminate at the enclosures.
2. The Passive Inventory model does not model fibres.
3. The passive inventory model assumes that there are defined in/out ports on passive devices.
4. New style passive inventory enclosures do not have defined in/out ports. They have defined physical ports that support multiple cables. There for they have physical ports and logical ports for cables. This is not enabled in the passive inventory model. 
5. The passive and active inventory models utilise different device ids definitions.
6. Internal or juniper cables are not definied in the YANG models. eg. ODF jumper cables, ATB to NTD cables. 
7. Modern enclosures are like a ODF underground and host splice trays and splitters and hence have components. Modelling does not permit components for passive devices only active devices.
8. Modern enclosures enable splicing meaning that the physical topology and logical topologyy are different entities. As fibres are not modelled the logical topology cannot be defined.
9. As fibres are not defines capacity the logical and physical capacity of the enclosure cannot be defined. Fulfilment where fibre is required to be understood that a logical path upstream there for cannot be determined. Meaning that passive topology needs to be addressed outside of the YANG models for both fillment and assurance for services on devices.

---

## YANG Constraint work arounds
1. Due to modern enclosures not having in/out port definitions the modelling creates a defined cable hierarchy to be able to transit upstream.
2. Cable role of 'internal-cable' is defined and used for ATB to NTD cables.
3. Enclosure physical ports are modelled separately to the passive devices

---

## Background

- https://github.com/Huawei-IOAM/ietf-knowledge-graphs/tree/main - IETF124 hackathon creating semantic relationship between simap-rdfs-schema.tll and Noria ontology
- https://github.com/Huawei-IOAM/yang2rdf - IETF xxx yang to rdf ABox (data) tool
- https://gitlab.eurecom.fr/huawei/yang2rdf - yang to owl TBOX (schema) tool utilising KG-Morph/RMLMapper

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [yang4owl.py — Generic YANG to OWL Conversion](#yang4owlpy--generic-yang-to-owl-conversion)
3. [The Traversal Problem: Direct YANG→OWL vs YANG](#the-traversal-problem-direct-yangowl-vs-yang)
4. [Semantic Extensions to Enable Traversal](#semantic-extensions-to-enable-traversal)
5. [RDF-star: Annotated Shortcut Triples](#rdf-star-annotated-shortcut-triples)
6. [OWL + Semantics vs YANG for AI and Grounding](#owl--semantics-vs-yang-for-ai-and-grounding)
7. [Repository Structure](#repository-structure)
8. [Usage](#usage)
9. [SPARQL Query Examples](#sparql-query-examples)
10. [Prefix Reference](#prefix-reference)

---

## Project Overview

This project converts a set of IETF YANG data models describing a fibre access network (Outside Plant / OSP) into an OWL 2 ontology and an enriched RDF instance graph. The target network topology consists of ONT (Optical Network Terminal) customer devices connected through a chain of passive equipment, Access Terminal Boxes (ATBs), optical enclosures, and cables, terminating at an ODF (Optical Distribution Frame) in a central office.

The core engineering challenge is enabling **efficient graph traversal** from any ONT upstream to its ODF endpoint, with full cable-by-cable path detail including cable type, role, and physical length. Standard YANG-to-OWL conversion leaves this traversal impossible with a single SPARQL property path expression. This project documents the problem, the semantic solution, and why the resulting enriched knowledge graph outperforms raw YANG as a data substrate for AI reasoning and network intelligence applications.

---

## yang4owl.py — Generic YANG to OWL Conversion

`yang4owl.py` is a general-purpose YANG-to-OWL converter that takes any directory of IETF YANG module files and produces a fully structured OWL 2 TBox in Turtle/RDF format, along with a companion SHACL shapes file for validation.

### How It Works

The converter uses [pyang](https://github.com/mbj4668/pyang) to parse and resolve YANG module dependencies, then walks the parsed abstract syntax tree through a multi-pass pipeline:

| Step | Description |
|------|-------------|
| 1 | Load all YANG modules from the source directory, resolving imports and includes |
| 2 | Initialise type resolvers, leafref resolvers, grouping resolvers, and context trackers |
| 3 | Register per-module namespaces as OWL ontology prefixes |
| 4 | Process YANG `grouping` definitions as OWL abstract classes |
| 5 | Walk the data model tree: modules, containers, lists, leaves, leaf-lists |
| 6 | Process `identity` hierarchies as OWL class hierarchies (each identity becomes both an `owl:Class` and a named individual — OWL 2 punning) |
| 7 | Process `augment` statements with full `uses` expansion |
| 8 | Generate container-to-container `owl:ObjectProperty` links |
| 9 | Process imported module bases |
| 10 | Generate SHACL shapes for `typedef` constraints |
| 11 | Process `enumeration` types as OWL named individuals |
| 12 | Resolve pending `leafref` targets (Pass 2, after all classes exist) |
| 13 | Apply custom TBox patches (explicit `ObjectProperty` for cable-to-device links) |
| 14 | Add RDF-star shortcut properties for upstream connectivity |

### YANG Construct Mappings

| YANG Construct | OWL Output |
|----------------|------------|
| `container` | `owl:Class` with path-based URI; parent `owl:ObjectProperty` |
| `list` | `owl:Class` (list entry); key leaves become `owl:DatatypeProperty` |
| `leaf` (scalar type) | `owl:DatatypeProperty` with mapped `xsd:` range |
| `leaf` (leafref) | `owl:ObjectProperty` with resolved range class (Pass 2) |
| `leaf` (identityref) | `owl:ObjectProperty` with identity class as range |
| `typedef` | Resolved to base XSD type; SHACL shape for constraints |
| `identity` | `owl:Class` + `owl:NamedIndividual` (OWL 2 punning); base identity becomes `rdfs:subClassOf` |
| `grouping` | Abstract `owl:Class`; inlined at each `uses` site |
| `augment` | Properties added to augmentation target class |
| `choice` / `case` | Disjoint `owl:Class` set under parent |
| `union` type | OWL subclass hierarchy |
| `enumeration` | Set of `owl:NamedIndividual` instances |
| `rpc` / `notification` | Stub `owl:Class` nodes with I/O sub-classes |
| XSD constraints (`range`, `pattern`, `length`) | SHACL `sh:NodeShape` / `sh:PropertyShape` |

YANG to OWL/RDF Processing Map

| YANG Construct | OWL/RDF Treatment | Reasoning Category | Domain (rdfs:domain) | Range (rdfs:range) |
| container | owl:Class | Class Logic | N/A | N/A |
| list | owl:Class | Class Logic | N/A | N/A |
| leaf (Standard) | owl:DatatypeProperty | Data Assertions | Parent Class URI | XSD Type (e.g., xsd:boolean) |
| leaf (identityref) | owl:ObjectProperty | Semantic Relationship | Parent Class URI | Base Identity Class URI |
| leaf (leafref) | owl:ObjectProperty | Semantic Relationship | Parent Class URI | Target Class URI |
| leaf (union) | owl:ObjectProperty | Logic Profile Compatibility | Parent Class URI | Created Union Parent Class |
| leaf (instance-id) | owl:ObjectProperty | Meta-referencing | Parent Class URI | N/A (Tagged with metadata) |
| identity | owl:Class & NamedIndividual | Individual Punning | N/A | N/A |
| identity (base) | rdfs:subClassOf | Transitive Hierarchy | Specific Identity URI | Base Identity URI |
| grouping | Abstract owl:Class | Template Modeling | N/A | N/A |
| uses | Nested Grouping Resolution | Schema Flattening | Target Class URI | Resolved Property Range |
| choice/case | Structural Flattening | Query Optimisation | Parent Container URI | Resolved Property Range |
| typedef | sh:NodeShape | Constraint Validation | N/A | N/A |
| enum (Definition) | owl:Class & Named Individual | Categorical Hierarchy | N/A | N/A |
| rpc | owl:Class | Functional Modelling | N/A | N/A |
| notification | owl:Class | Functional Modelling | N/A | N/A |
| must/when | sh:condition/sh:deactivated | Conditional Logic | Property/Class URI | N/A (SHACL Filter) |
| augment | logical Property Injection | Monolithic integration | Augmented Target Class | Injected Node Class Type |
| Child Containment | has[ChildName] (ObjectProperty) | Structural Integrity  Parent Class URI | Child Class URI |

1. YANG Construct Address Mapping

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

### OWL 2 Punning for Identity Resolution

A key design decision is the use of **OWL 2 punning** to represent YANG `identity` statements. Each identity such as `internal-cable`, `drop-cable`, or `ODF` is simultaneously:

- An `owl:Class` — so instances can be typed with `rdf:type`
- An `owl:NamedIndividual` — so it can be the *value* of an `owl:ObjectProperty`

This dual representation mirrors how YANG identities are used: as both a type taxonomy and as enumerated values on `identityref` leaves.

### Provenance Tracking

Every generated OWL property carries a `prov:wasDerivedFrom` annotation recording the exact YANG XPath path that produced it, for example:

```turtle
ex:cable-role a owl:ObjectProperty ;
    prov:wasDerivedFrom "nwi-passive:cable-role?leaf" .
```

### Running the TBox Converter

```bash
python yang4owl.py <yang_dir> <output.ttl> \
    --modules ietf-nwi-passive-inventory.yang \
    --base-uri http://www.huawei.com/ontology
```

---

## The Traversal Problem: Direct YANG→OWL vs YANG

### Traversal in Native YANG

In a native YANG data model, device connectivity is navigated by following `leafref` XPath expressions. A cable instance carries an `a-end` container and a `z-end` container; each end container contains a `ne-ref` leaf that is a `leafref` pointing to a network element's `ne-id` key leaf, or a `device-ref` leaf pointing to a passive device's identifier.

```yang
// ietf-nwi-passive-inventory.yang (simplified)
container cable {
  container a-end {
    leaf ne-ref {
      type leafref {
        path "/nwi:network-inventory/nwi:network-elements"
           + "/nwi:network-element/nwi:ne-id";
      }
    }
  }
  container z-end {
    leaf ne-ref { ... }
  }
}
```

A YANG client traversing from an ONT to its ODF would:

1. Read the ONT's `ne-id`
2. Find cables whose `z-end/ne-ref` matches that value
3. Read the cable's `a-end/ne-ref`
4. Look up the device at that identifier
5. Repeat steps 2–4 until reaching a device of type `ODF`

This works at the protocol level (NETCONF/RESTCONF) where the server enforces referential integrity, but it requires **application-layer traversal logic** — a procedural loop that resolves one hop at a time by matching string keys.

### Why Direct YANG→OWL Breaks Graph Traversal

When a YANG model is converted to OWL using a naïve or standard approach, `leafref` leaves become `owl:DatatypeProperty` assertions carrying **string literal values** — the referenced entity's key. For example:

```turtle
# What a naïve YANG→OWL converter produces:
inv-cab:cable_311_z-end  ex:ne-ref  "ONT_6" .
inv-cab:cable_311_a-end  ex:ne-ref  "ATB_71" .
```

This representation has a critical structural flaw: the string `"ONT_6"` is a literal, not a URI. **An RDF graph engine cannot follow a string literal to another node.** SPARQL property paths — the mechanism that enables `?start ex:someProperty+ ?end` transitive traversal — only traverse `owl:ObjectProperty` links between named resources (IRIs). They do not match on string equality across separate triples.

The traversal from ONT to ODF therefore requires **four intermediate hops per cable segment**:

```
?device
  → (z-end node via cab:hasZEnd)
    → (ne-ref literal string)
      → [string-match join to find the matching device IRI]
        → (next cable's z-end ne-ref string)
          ...
```

In SPARQL, expressing even two hops requires a `UNION` of explicit join patterns. For a real OSP network where an ONT may be 7 or more cable hops from its ODF, the query becomes combinatorially verbose and practically unmaintainable:

```sparql
# The kind of query required WITHOUT semantic enrichment (4-hop pattern per cable):
SELECT ?ont ?odf WHERE {
  ?ont a nwiNEs:network-element .
  ?zend1 ex:ne-ref ?ontId .
  ?ont ex:ne-id ?ontId .
  ?cable1 cab:hasZEnd ?zend1 .
  ?cable1 cab:hasAEnd ?aend1 .
  ?aend1 ex:device-ref ?dev1 .
  # ... repeat for every subsequent hop ...
  ?devN ex:device-type nwiPassId:ODF .
}
```

This query cannot be expressed with SPARQL `+` or `*` property path operators because there is no single property connecting `?device` to `?nextDevice` — the connection is mediated by four intermediate nodes and a string equality join.

**By contrast, the equivalent traversal in native YANG is also procedural** — it requires a loop of NETCONF `get` operations with XPath filters to resolve each `leafref` hop. Neither representation gives you a single declarative expression for arbitrary-depth traversal. The difference is that YANG is *designed* for this procedural access pattern, whereas OWL/RDF's power lies in property path reasoning — which the direct conversion completely fails to unlock.

---

## Semantic Extensions to Enable Traversal

To bridge the gap between the YANG data model structure and the graph traversal capabilities of OWL/SPARQL, `yang4owl.py` applies two layers of semantic enrichment beyond what standard YANG→OWL conversion produces.

### Layer 1: TBox Patch - Explicit ObjectProperties for Cable Terminations

The first layer replaces or supplements the string-valued `leafref` data properties with proper `owl:ObjectProperty` definitions. Three properties are defined that link cable end-point nodes directly to device IRIs:

```turtle
ex:device-ref  a owl:ObjectProperty ;
    rdfs:label  "device-ref" ;
    rdfs:comment "Reference to a passive device (e.g. ATB or Enclosure)." ;
    rdfs:domain  cab:a-end, cab:z-end, cabChild:a-end, cabChild:z-end ;
    rdfs:range   nwi:passive-device .

ex:ne-ref  a owl:ObjectProperty ;
    rdfs:domain  cab:a-end, cab:z-end, cabChild:a-end, cabChild:z-end ;
    rdfs:range   nwiNEs:network-element .

ex:component-ref  a owl:ObjectProperty ;
    rdfs:domain  cab:a-end, cab:z-end, cabChild:a-end, cabChild:z-end ;
    rdfs:range   nwiComp:component .
```

With these definitions in the TBox and corresponding IRI-valued triples in the ABox (generated by `csv_to_owl_v3_7_44.py`), a graph engine can now traverse from a cable end-point to the actual device resource. However, the path from device to device still requires traversing through: `cable → z-end → device` and `cable → a-end → next-device`, which is still a multi-hop property chain — not a single step.

### Layer 2: Materialised Shortcut Properties 

The second and decisive layer introduces **direct device-to-device shortcut properties** that collapse the entire 4-hop cable traversal path into a single `owl:ObjectProperty` link:

```turtle
# TBox declarations (Step 14)
ex:hasUpstreamDevice  a owl:ObjectProperty ;
    rdfs:label   "has upstream device" ;
    rdfs:comment "Materialised shortcut: device at Z-end of a cable → device at A-end (upstream). 
                  Annotated via RDF-star with ex:viaCable and ex:cableRole. 
                  Enables: ?device ex:hasUpstreamDevice+ ?odf ." .

ex:hasDownstreamDevice  a owl:ObjectProperty ;
    owl:inverseOf  ex:hasUpstreamDevice ;
    rdfs:label     "has downstream device" .
```

These properties exist in the TBox as semantic declarations. 

The abox data every cable `nwi:cable` individual, resolves the Z-end and A-end device references, and includes:

```turtle
# Direct shortcut triple (added to the base graph):
inv-ne:ONT_6   ex:hasUpstreamDevice   inv-dev:ATB_71 .
inv-dev:ATB_71 ex:hasDownstreamDevice inv-ne:ONT_6 .
```

The result: a traversal query that once required a bounded `UNION` of multiple explicit hop patterns now collapses to:

```sparql
SELECT ?ont ?odf WHERE {
  ?ont a nwiNEs:network-element .
  ?ont ex:hasUpstreamDevice+ ?odf .
  ?odf ex:device-type nwiPassId:ODF .
}
```

### Cable Role Priority (TBox Semantics)

The cable hierarchy in the OSP domain is ordered by network tier. Each cable role identity carries an `ex:rolePriority` integer annotation defined in the TBox:

```turtle
nwiPassId:internal-cable    ex:rolePriority  0 .
nwiPassId:drop-cable        ex:rolePriority  1 .
nwiPassId:access-cable      ex:rolePriority  2 .
nwiPassId:branch-cable      ex:rolePriority  3 .
nwiPassId:aggregation-cable ex:rolePriority  4 .
nwiPassId:distribution-cable ex:rolePriority 5 .
nwiPassId:trunk-cable       ex:rolePriority  6 .
```

This is a deliberate architectural decision: `ex:rolePriority` is a **definitional** property of the YANG identity term (a schema-level fact about what each cable tier means), not an empirical measurement of a network instance. It belongs in the TBox. Queries that need to order path hops by network tier must therefore join both named graphs:

```sparql
FROM <ietf:abox>
FROM <ietf:tbox>
```

### Other optimisations
#A. Structural Flattening (Choice/Case Removal)

The script removes `case` statements as intermediate classes, attaching properties directly to the parent container instance. This reduces the number of "hops" required for graph traversal, significantly increasing query performance.

#B. Enhanced Identity Management

Utilizes URI-based identity mapping instead of string matching. For instance, a port's type is mapped directly to an IRI (e.g., `<.../nwiPassId/active-device>`). This supports multi-level discovery via hierarchical inference; a query for "all physical network elements" will return all sub-classes defined from IANA hardware identities.

#C. SHACL Isolation and Validation

Because OWL uses an "open-world" assumption, the script isolates strict YANG "closed-world" constraints into a separate SHACL graph. This allows Stardog’s Integrity Constraint Validation (ICV) engine to enforce rules without polluting the logical consistency of the ontology.

#D. Cable and Fiber Path Optimisation

Specialized logic flattens cable and fiber representations, mapping reified structures like `cable -> a-end -> device-type` into clear logical endpoints[cite: 1638]. This enables the use of native URI references (`ex:device-ref`) and port-level linkages (`ex:port-ref`) to ensure unbroken path traces.

#E. Hardware Hierarchy and State Management
Maintains deep hardware hierarchies (Chassis -> Module -> Port) using 'ex:parent' references. Operational states are separated by reified state containers ('hwcomp:hasState') allowing complex reasoning, such as finding all ports affected by a specific chassis failure.

---

## RDF-star: Annotated Shortcut Triples

### The Problem with Plain Shortcuts

Materialising `ex:hasUpstreamDevice` shortcut triples gives us efficient traversal, but loses the cable context for each hop. Knowing that `ONT_6 → ATB_71` is useful, but network engineers and AI systems also need to know *which cable* connects them, its *role/tier*, and optionally its *physical length*. In standard RDF, attaching metadata to a triple requires reification (verbosely creating a proxy resource), which is cumbersome to query.

### RDF-star Solution

This project uses **RDF-star** (RDF 1.2 / Turtle\* syntax) to attach cable context directly to each shortcut triple as a **statement-level annotation**. The `ABoxConnectivityEnricher` emits the following pattern for every upstream hop:

```turtle
# Base shortcut triple:
inv-ne:ONT_6  ex:hasUpstreamDevice  inv-dev:ATB_71 .

# RDF-star annotation block (Turtle* syntax):
<<inv-ne:ONT_6  ex:hasUpstreamDevice  inv-dev:ATB_71>>
    ex:viaCable   inv-cab:cable_311 ;
    ex:cableRole  nwiPassId:internal-cable .
```

The `<<subject predicate object>>` notation in Turtle\* treats the triple itself as a subject, allowing additional properties to be asserted *about* the triple. The two annotation properties are defined in the TBox:

```turtle
ex:viaCable   a owl:ObjectProperty ;
    rdfs:comment "RDF-star annotation on ex:hasUpstreamDevice: 
                  the physical cable realising this hop." .

ex:cableRole  a owl:ObjectProperty ;
    rdfs:comment "RDF-star annotation on ex:hasUpstreamDevice: 
                  role/tier of the cable." .
```

### Querying RDF-star Annotations

In SPARQL\* (supported by Stardog and other RDF-star-capable stores), the annotation can be queried inline using the same `<<...>>` syntax:

```sparql
PREFIX inv-ne:    <http://www.huawei.com/instances/network-element/>
PREFIX ex:        <http://www.huawei.com/ontology/>

SELECT ?from ?to ?cable ?role ?priority ?length
FROM <ietf:topology:abox>
FROM <ietf:topology:tbox>
WHERE {
    inv-ne:ONT_6 ex:hasUpstreamDevice* ?from .
    ?from ex:hasUpstreamDevice ?to .
    << ?from ex:hasUpstreamDevice ?to >> ex:viaCable   ?cable .
    << ?from ex:hasUpstreamDevice ?to >> ex:cableRole  ?role .
    ?role ex:rolePriority ?priority .
    OPTIONAL { ?cable ex:length ?length . }
}
ORDER BY ?priority
```

This returns the full annotated hop-by-hop path from ONT_6 to its ODF, with cable IRI, role, tier priority, and physical length per hop — all in a single query, with no application-layer looping.

### Why Not Standard RDF Reification?

Standard RDF reification (using `rdf:Statement`, `rdf:subject`, `rdf:predicate`, `rdf:object`) achieves the same goal but requires four extra triples per annotated statement and produces queries that are verbose, harder to read, and significantly slower to execute in most triple stores. RDF-star is natively supported by Stardog and stores that implement the RDF 1.2 specification, producing cleaner syntax with equivalent expressiveness and better query performance.

### Output File Format

The enriched ABox is written as a **Turtle\*** file with the `.ttls` extension. Stardog automatically detects the Turtle\* parser from this extension. The file is a valid superset of standard Turtle — the base triples are standard RDF; only the `<<...>>` annotation blocks require an RDF-star-capable parser.

---

## OWL + Semantics vs YANG for AI and Grounding

### The Core Question

Both YANG and the enriched OWL graph represent the same underlying network reality. The question is which representation is more *useful* as a data substrate when AI systems — large language models, retrieval-augmented generation pipelines, graph neural networks, or automated reasoning engines — need to understand, query, or reason about the network.

### YANG Limitations for AI

YANG is an excellent **configuration and operational data schema language**, but it presents significant difficulties for AI consumption:

**Structural opacity.** A YANG model is a schema, not data. The actual network state lives in NETCONF `get` responses or YANG instance data files — a separate format that requires understanding the schema to interpret. An AI system must hold both the schema and the data in context simultaneously to answer questions about the network.

**Procedural traversal requirement.** As shown in the traversal problem section, navigating relationships in YANG data requires procedural loops that resolve `leafref` pointers one step at a time. There is no mechanism for expressing "find all ONTs that cannot reach an ODF" as a single declarative statement. AI systems are poor at executing procedural database traversal logic reliably.

**No native semantics.** YANG defines structure and constraints, but not meaning. It has no concept of `owl:inverseOf`, transitivity, disjointness, or class hierarchy inference. A `device-type` leaf with value `ODF` is just a string to a YANG parser; there is no mechanism to assert that ODF is a subtype of passive-device without augmenting the model.

**Opaque identity system.** YANG `identity` and `identityref` provide a lightweight taxonomy, but the hierarchy is not traversable as a graph. An AI cannot naturally ask "which device types are passive equipment?" without understanding the module's `identity` tree and writing code to walk it.

**Schema volatility.** YANG models evolve with IETF drafts. Each new version may restructure paths, rename leaves, or change leafref targets. Any AI system that has learned the schema structure must be retrained or re-prompted when the schema changes.

### OWL + Semantic Enrichment Advantages

The enriched OWL knowledge graph addresses each of these limitations directly.

**Self-describing data.** In RDF, every triple is a complete, self-contained statement with globally unique IRIs as subjects and predicates. An AI system reading `inv-ne:ONT_6 ex:hasUpstreamDevice inv-dev:ATB_71` understands the full relationship without needing a separate schema. The ontology TBox provides additional context (`rdfs:label`, `rdfs:comment`, `rdfs:domain`, `rdfs:range`) that an LLM can use directly.

**Declarative traversal.** The materialised `ex:hasUpstreamDevice+` property path enables a single SPARQL query to answer questions like "which ONTs have no path to an ODF?", "what is the total cable length from ONT_6 to its ODF?", and "which ONTs share a trunk cable?" — all without application-layer logic. This aligns with how AI systems are best used: generating declarative queries, not procedural algorithms.

**Explicit semantics.** The TBox expresses machine-interpretable facts: `ex:hasDownstreamDevice owl:inverseOf ex:hasUpstreamDevice` means a reasoning engine can infer downstream links from upstream links without them being explicitly stored. Disjoint class declarations, domain/range constraints, and cardinality restrictions allow automatic consistency checking and inference that YANG cannot express.

**Grounded identity.** Cable roles, device types, and equipment classes are OWL named individuals with stable IRIs and `rdfs:label`/`rdfs:comment` annotations. An LLM grounded on the TBox understands that `nwiPassId:trunk-cable` is the highest-tier cable in a hierarchy (`ex:rolePriority 6`), that it is a subclass of `nwiPassId:cable-role`, and that its label is `"trunk-cable"`. This grounding is richer and more navigable than an identity statement in a YANG module.

**Context-preserving edges (RDF-star).** The RDF-star annotation pattern means that every `ex:hasUpstreamDevice` link carries its full provenance — which cable, which role, what tier. An AI asked "why is ONT_6 classified as a drop-cable customer?" can retrieve the annotated path and reason over both the topology and the semantic properties of each hop. YANG has no equivalent construct for annotating individual data relationships with metadata.

**Natural language alignment.** OWL ontologies, with their `rdfs:label` and `rdfs:comment` annotations, are far closer to natural language than YANG models. An LLM generating SPARQL from a user question like "show me all customers without a working path to a distribution point" can map directly from the question's vocabulary to ontology terms. Equivalent generation against a YANG schema requires the LLM to understand YANG XPath, leafref resolution, and list key semantics simultaneously.

**Federated reasoning.** OWL graphs can be merged with other knowledge graphs (geospatial data, fault management ontologies, SLA definitions) using shared IRIs as join keys. A unified graph can then answer cross-domain questions — e.g., "which ONTs in fault zone Z have a path length exceeding 5km?" — that would require orchestrating multiple separate YANG datastores with custom join logic.

### Comparative Summary

| Capability | YANG | OWL + RDF-star |
|---|---|---|
| Schema definition | ✅ Excellent | ✅ Good (TBox) |
| Instance data | ✅ NETCONF / instance data | ✅ ABox triples |
| Traversal expression | ❌ Procedural loop | ✅ Single SPARQL property path |
| Semantic inference | ❌ None | ✅ OWL reasoner / SPARQL |
| Edge metadata | ❌ Not supported | ✅ RDF-star annotations |
| AI/LLM readability | ⚠️ Requires schema + data together | ✅ Self-describing triples |
| Grounded identity | ⚠️ Identity hierarchy only | ✅ Named individuals + labels |
| Cross-domain federation | ❌ Protocol-level only | ✅ IRI-based graph merge |
| Consistency checking | ⚠️ YANG `must` / `when` only | ✅ OWL + SHACL |
| Query language | XPath (schema), RESTCONF (data) | SPARQL 1.1 / SPARQL* |
| Standard body | IETF RFC 7950 | W3C OWL 2, RDF 1.2 |

### Verdict

For **AI grounding and automated reasoning**, the enriched OWL knowledge graph is significantly more capable than native YANG. The combination of materialised shortcut properties, RDF-star edge annotations, OWL 2 semantics, and a well-labelled TBox gives AI systems everything they need to reason declaratively over network topology: a traversable graph, self-describing terms, semantic constraints, and rich edge metadata. YANG remains the authoritative source-of-truth protocol schema for device management; OWL is the superior representation for intelligence and reasoning over the data that schema describes.

---

## Usage

### Prerequisites

```bash
pip install pyang rdflib
```

### 1. Generate the TBox

```bash
- yang4owl.py --yang-dir <directory of the yang files> --modules <yang model to create ttl for> --base-uri <owl base uri> --output <turtle file> --verbose
```
### 2. Load TBox
Load the TBOX into a graph referred to as ietf:tbox.

### 3. Load the ABox (osp-data-topology.ttl)
Load the ABox data into a graph referred to as ietf:abox.

### eg. Using Stardog Cloud Free
It is recommended to load the TBOX and ABOX data into different graphs. Due to the cable hierarchy being defined in the TBOX the queries must include the TBOX as part of the query. 

The following utilises stardog cloud as an example. Note with SD Cloud when defining the database you need to enable the edge properties (RDF*).

Below is via CLI but GUI implementation on SD Cloud is equally usable. 

```bash
stardog data add --named-graph ietf:tbox mydb ietf-osp-topology-python2.ttl
stardog data add --named-graph ietf:abox mydb data_v3_7_44.ttls
```

> **Note:** Queries using `ex:rolePriority` must include `FROM <ietf:tbox>` in addition to `FROM <ietf:abox>`, as priority values are defined in the TBox.

---

## SPARQL Query Examples

### ONT_6 with its connected enclosures, cables, cable role, cable priority
  ```sparql
  PREFIX inv-ne: <http://www.huawei.com/instances/network-element/>
  PREFIX ex: <http://www.huawei.com/ontology/>

  SELECT ?from ?to ?cable ?role ?priority
  FROM <ietf:abox>
  FROM <ietf:tbox>
  WHERE {
  inv-ne:ONT_6 ex:hasUpstreamDevice* ?from .
  ?from ex:hasUpstreamDevice ?to .
  << ?from ex:hasUpstreamDevice ?to >> ex:viaCable ?cable .
  << ?from ex:hasUpstreamDevice ?to >> ex:cableRole ?role .
  ?role ex:rolePriority ?priority .
  }
  ORDER BY ?priority
  ```

```csv
from,to,cable,role,priority

http://www.huawei.com/instances/network-element/ONT_6,http://www.huawei.com/instances/passive-device/ATB_71,http://www.huawei.com/instances/cable/cable_311,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/internal-cable,0

http://www.huawei.com/instances/passive-device/ATB_71,http://www.huawei.com/instances/passive-device/enclosure_33,http://www.huawei.com/instances/cable/cable_316,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/drop-cable,1

http://www.huawei.com/instances/passive-device/enclosure_33,http://www.huawei.com/instances/passive-device/enclosure_39,http://www.huawei.com/instances/cable/cable_30,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/access-cable,2

http://www.huawei.com/instances/passive-device/enclosure_39,http://www.huawei.com/instances/passive-device/enclosure_56,http://www.huawei.com/instances/cable/cable_53,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/branch-cable,3

http://www.huawei.com/instances/passive-device/enclosure_56,http://www.huawei.com/instances/passive-device/enclosure_65,http://www.huawei.com/instances/cable/cable_337,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/aggregation-cable,4

http://www.huawei.com/instances/passive-device/enclosure_75,http://www.huawei.com/instances/passive-device/ODF_1,http://www.huawei.com/instances/cable/cable_417,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/distribution-cable,5

http://www.huawei.com/instances/passive-device/enclosure_65,http://www.huawei.com/instances/passive-device/enclosure_75,http://www.huawei.com/instances/cable/cable_133,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/distribution-cable,5
```
### All ONTs with a path to an ODF

```sparql
PREFIX ex:       <http://www.huawei.com/ontology/>
PREFIX nwiNEs:   <http://www.huawei.com/identity/network-element/>
PREFIX nwiPassId: <http://www.huawei.com/identity/passive-equipment/>

SELECT ?ont ?odf
FROM <ietf:topology:abox>
WHERE {
    ?ont  a nwiNEs:network-element .
    ?ont  ex:hasUpstreamDevice+  ?odf .
    ?odf  ex:device-type  nwiPassId:ODF .
}
```

```csv 
ont,odf

http://www.huawei.com/instances/network-element/ONT_2,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/network-element/ONT_4,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/network-element/ONT_6,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/passive-device/enclosure_25,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/network-element/ONT_8,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/passive-device/enclosure_33,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/network-element/ONT_9,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/passive-device/enclosure_39,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/network-element/ONT_11,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/passive-device/enclosure_48,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/network-element/ONT_13,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/passive-device/enclosure_56,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/network-element/ONT_14,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/passive-device/enclosure_65,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/passive-device/enclosure_66,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/passive-device/enclosure_70,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/passive-device/enclosure_71,http://www.huawei.com/instances/passive-device/ODF_1

http://www.huawei.com/instances/passive-device/enclosure_75,http://www.huawei.com/instances/passive-device/ODF_1
```
### Annotated hop-by-hop path for a single ONT

```sparql
PREFIX inv-ne:    <http://www.huawei.com/instances/network-element/>
PREFIX ex:        <http://www.huawei.com/ontology/>

SELECT ?from ?to ?cable ?role ?priority ?length
FROM <ietf:topology:abox>
FROM <ietf:topology:tbox>
WHERE {
    inv-ne:ONT_6 ex:hasUpstreamDevice* ?from .
    ?from ex:hasUpstreamDevice ?to .
    << ?from ex:hasUpstreamDevice ?to >> ex:viaCable   ?cable .
    << ?from ex:hasUpstreamDevice ?to >> ex:cableRole  ?role .
    ?role ex:rolePriority ?priority .
    OPTIONAL { ?cable ex:length ?length . }
}
ORDER BY ?priority
```


```csv 
from,to,cable,length,role,priority

http://www.huawei.com/instances/network-element/ONT_6,http://www.huawei.com/instances/passive-device/ATB_71,http://www.huawei.com/instances/cable/cable_311,,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/internal-cable,0

http://www.huawei.com/instances/passive-device/ATB_71,http://www.huawei.com/instances/passive-device/enclosure_33,http://www.huawei.com/instances/cable/cable_316,55,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/drop-cable,1

http://www.huawei.com/instances/passive-device/enclosure_33,http://www.huawei.com/instances/passive-device/enclosure_39,http://www.huawei.com/instances/cable/cable_30,37,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/access-cable,2

http://www.huawei.com/instances/passive-device/enclosure_39,http://www.huawei.com/instances/passive-device/enclosure_56,http://www.huawei.com/instances/cable/cable_53,100,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/branch-cable,3

http://www.huawei.com/instances/passive-device/enclosure_56,http://www.huawei.com/instances/passive-device/enclosure_65,http://www.huawei.com/instances/cable/cable_337,130,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/aggregation-cable,4

http://www.huawei.com/instances/passive-device/enclosure_75,http://www.huawei.com/instances/passive-device/ODF_1,http://www.huawei.com/instances/cable/cable_417,1558,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/distribution-cable,5

http://www.huawei.com/instances/passive-device/enclosure_65,http://www.huawei.com/instances/passive-device/enclosure_75,http://www.huawei.com/instances/cable/cable_133,984,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/distribution-cable,5
```
### Total path length for ONT_6
```sparql
PREFIX inv-ne: <http://www.huawei.com/instances/network-element/>
PREFIX ex: <http://www.huawei.com/ontology/>

SELECT ?ont (SUM(COALESCE(?length,0)) AS ?totalMetres) (COUNT(?cable) AS ?hops)
FROM <ietf:abox>
FROM <ietf:tbox>
WHERE {
BIND(inv-ne:ONT_6 AS ?ont)
?ont ex:hasUpstreamDevice* ?from .
?from ex:hasUpstreamDevice ?to .
<< ?from ex:hasUpstreamDevice ?to >> ex:viaCable ?cable .
<< ?from ex:hasUpstreamDevice ?to >> ex:cableRole ?role .
OPTIONAL { ?cable ex:length ?length . }
}
GROUP BY ?ont
```

```csv
ont,totalMetres,hops
http://www.huawei.com/instances/network-element/ONT_6,2864,7
```

### ONT_6 cable path and length to ODF
```sparql 
PREFIX inv-ne: <http://www.huawei.com/instances/network-element/>
PREFIX ex: <http://www.huawei.com/ontology/>
PREFIX nwiPassId: <http://www.huawei.com/identity/passive-equipment/>

SELECT ?from ?to ?cable ?length ?role ?priority
FROM <ietf:abox>
FROM <ietf:tbox>
WHERE {
inv-ne:ONT_6 ex:hasUpstreamDevice* ?from .
?from ex:hasUpstreamDevice ?to .
<< ?from ex:hasUpstreamDevice ?to >> ex:viaCable ?cable .
<< ?from ex:hasUpstreamDevice ?to >> ex:cableRole ?role .
?role ex:rolePriority ?priority .
OPTIONAL { ?cable ex:length ?length . }
}
ORDER BY ?priority
```

```csv
from,to,cable,length,role,priority

http://www.huawei.com/instances/network-element/ONT_6,http://www.huawei.com/instances/passive-device/ATB_71,http://www.huawei.com/instances/cable/cable_311,,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/internal-cable,0

http://www.huawei.com/instances/passive-device/ATB_71,http://www.huawei.com/instances/passive-device/enclosure_33,http://www.huawei.com/instances/cable/cable_316,55,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/drop-cable,1

http://www.huawei.com/instances/passive-device/enclosure_33,http://www.huawei.com/instances/passive-device/enclosure_39,http://www.huawei.com/instances/cable/cable_30,37,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/access-cable,2

http://www.huawei.com/instances/passive-device/enclosure_39,http://www.huawei.com/instances/passive-device/enclosure_56,http://www.huawei.com/instances/cable/cable_53,100,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/branch-cable,3

http://www.huawei.com/instances/passive-device/enclosure_56,http://www.huawei.com/instances/passive-device/enclosure_65,http://www.huawei.com/instances/cable/cable_337,130,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/aggregation-cable,4

http://www.huawei.com/instances/passive-device/enclosure_75,http://www.huawei.com/instances/passive-device/ODF_1,http://www.huawei.com/instances/cable/cable_417,1558,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/distribution-cable,5

http://www.huawei.com/instances/passive-device/enclosure_65,http://www.huawei.com/instances/passive-device/enclosure_75,http://www.huawei.com/instances/cable/cable_133,984,http://www.huawei.com/ontology/identity/ietf-nwi-passive-inventory/distribution-cable,5
```

---
