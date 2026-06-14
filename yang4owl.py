#!/usr/bin/env python3

"""
YANG to OWL Ontology Converter - VERSION 4.7.32 (RDF-star Connectivity)

Release Note: Semantic Interoperability via OWL 2 Punning
This update implements OWL 2 Punning to resolve "dead-end" string traversals common in standard YANG-to-OWL conversions.
The Problem: IETF leafrefs are often treated as simple string literals, which isolates data silos and prevents graph engines from following links between entities.
The Solution: Punning allows properties to function as both owl:DatatypeProperty (for legacy string IDs) and owl:ObjectProperty (for direct URI-to-URI links).
Key Entities: This has been applied to critical "horizontal" connectors including ex:ne-ref (Topology to Network Elements), ex:class (Hardware Classification), and ex:device-ref (Cable termination points).
The Result: Your Knowledge Graph now supports end-to-end traversal in Stardog, allowing queries to "jump" from a logical cable end directly into the physical attributes of active NTDs or passive enclosures. However your graph database needs to support punning.
There is also a section for custom TBOX patch to take the standard yang to owl with its limitations and add explicit ObjectProperties rather than leafref dataproperties. This is currently implemented explicitly for cables to devices providing a direct link without needing to use string to traverse.


ALL IMPROVEMENTS IMPLEMENTED:
1. Container Object Properties 
2. Augmentation Complete Hierarchy 
3. GROUPING EXPANSION WITH REFINE 
4. Imported Module Integration 
5. Leafref Cardinality Constraints 
6. RPC/Notification Processing 
7. Comprehensive PROV Metadata 
8. XSD Constraints Extraction 
9. OWL DATATYPE RESTRICTIONS 
10. ENUMERATION TYPES AS OWL INDIVIDUALS 
11. NESTED GROUPING RESOLUTION 
12. REFINE STATEMENT PROCESSING 
13. GROUPING CONTEXT TRACKING 
14. PATH NORMALIZATION 
15. ENHANCED IdentityRef to Objectproperty 
16. ENHANCED Choices and Cases to disjoint classes 
17. Yang Union types implemented by subclasses 
18. Separated SHACL and fixed namespace collison
19. Removing Case as Classes and moved constraints to shacl.
20. Correcting other paths that hard coded the /for a make shift iri
21. Trying to flatten the cable and fibre
22. Still fixing the uri
23. Add auto generate prefixes to the ontology
24. Make prefixes static
25. add more static prefixes
26. Post yang converter to introduce objectproprties for leafrefs for cable to passive device traversal
27. RDF-star upstream connectivity materialisation pass (ABoxConnectivityEnricher)
    - Adds ex:hasUpstreamDevice / ex:hasDownstreamDevice shortcut triples for every cable (Z→A)
    - Annotates each shortcut with <<...>> ex:viaCable and ex:cableRole (Turtle* / RDF-star)
    - Enables SPARQL property path: ?device ex:hasUpstreamDevice+ ?odf
    - Eliminates bounded UNION / multi-hop workarounds for graph traversal
    - Run: python yang4owl.py --abox-enrich <abox.ttl> --abox-out <enriched.ttls>

Author: YANG-to-OWL Converter v4.7.32
Date: 2026-06-14
"""

from os import name
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
import re

try:
    from pyang import context, repository, statements
except ImportError:
    print("ERROR: pyang not found. Install with: pip install pyang")
    sys.exit(1)

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, XSD, BNode
from rdflib.namespace import OWL, PROV
from rdflib.collection import Collection

SH = Namespace("http://www.w3.org/ns/shacl#")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

CABLE_RANK_MAP = {
    "internal-cable":     0,   # unclassified drop; never blocks traversal
    "drop-cable":         1,
    "access-cable":       2,
    "branch-cable":       3,
    "aggregation-cable":  4,
    "distribution-cable": 5,
    "trunk-cable":        6,
}

def extract_module_name(filename: str) -> str:
    """Safely normalizes a module name from a filename or argument."""
    name = filename.replace('.yang', '')
    if '@' in name: name = name.split('@')[0]
    name = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', name)
    return name


class YANGConstraintExtractor:
    def __init__(self):
        self.constraints_found = 0
        self.typedef_usage = {}

    def extract_constraints(self, type_stmt: Any) -> Dict[str, Any]:
        constraints = {}
        if not hasattr(type_stmt, 'substmts'):
            return constraints
        for sub in type_stmt.substmts:
            if not hasattr(sub, 'keyword'):
                continue
            keyword = sub.keyword
            if keyword == 'range':
                constraints['range'] = self._parse_range(sub.arg if hasattr(sub, 'arg') else '')
                self.constraints_found += 1
            elif keyword == 'pattern':
                if 'patterns' not in constraints:
                    constraints['patterns'] = []
                constraints['patterns'].append(sub.arg if hasattr(sub, 'arg') else '')
                self.constraints_found += 1
            elif keyword == 'length':
                constraints['length'] = self._parse_length(sub.arg if hasattr(sub, 'arg') else '')
                self.constraints_found += 1
            elif keyword == 'type':
                base_constraints = self.extract_constraints(sub)
                if base_constraints:
                    constraints.update(base_constraints)
        return constraints

    def _parse_range(self, range_str: str) -> Dict[str, Any]:
        result = {}
        if not range_str: return result
        ranges = range_str.split('|')
        for r in ranges:
            r = r.strip()
            if '..' in r:
                parts = r.split('..')
                if len(parts) == 2:
                    try:
                        min_val = int(parts[0].strip())
                        max_val = int(parts[1].strip())
                        if 'min' not in result or min_val < result['min']: result['min'] = min_val
                        if 'max' not in result or max_val > result['max']: result['max'] = max_val
                    except ValueError:
                        pass
        return result

    def _parse_length(self, length_str: str) -> Dict[str, Any]:
        result = {}
        if not length_str: return result
        ranges = length_str.split('|')
        for r in ranges:
            r = r.strip()
            if '..' in r:
                parts = r.split('..')
                if len(parts) == 2:
                    try:
                        min_len = int(parts[0].strip())
                        max_len = int(parts[1].strip())
                        if 'minLength' not in result or min_len < result['minLength']: result['minLength'] = min_len
                        if 'maxLength' not in result or max_len > result['maxLength']: result['maxLength'] = max_len
                    except ValueError:
                        pass
        return result

class YANGTypeResolver:
    BUILTIN_TYPES = {
        'binary': XSD.hexBinary, 'bits': RDFS.Literal, 'boolean': XSD.boolean,
        'decimal64': XSD.decimal, 'empty': XSD.boolean, 'enumeration': RDFS.Literal,
        'int8': XSD.byte, 'int16': XSD.short, 'int32': XSD.int, 'int64': XSD.long,
        'string': XSD.string, 'uint8': XSD.unsignedByte, 'uint16': XSD.unsignedShort,
        'uint32': XSD.unsignedInt, 'uint64': XSD.unsignedLong,
        'inet:ip-address': XSD.string, 'yang:date-and-time': XSD.dateTime,
        'yang:counter32': XSD.unsignedInt, 'yang:counter64': XSD.unsignedLong,
        'inet:uri': XSD.anyURI,
    }

    def __init__(self):
        self.typedefs: Dict[str, Any] = {}
        self.constraint_extractor = YANGConstraintExtractor()

    def register_typedef(self, module_name: str, name: str, typedef: Any) -> None:
        self.typedefs[f"{module_name}:{name}"] = typedef

    def resolve_type(self, type_stmt: Any, current_module: str, prefix_resolver) -> URIRef:
        type_name = getattr(type_stmt, 'arg', None)
        if not type_name: return XSD.string
        
        if type_name in self.BUILTIN_TYPES: return self.BUILTIN_TYPES[type_name]
        
        clean_name = type_name.split(':')[-1]
        if clean_name in self.BUILTIN_TYPES: return self.BUILTIN_TYPES[clean_name]
        
        target_mod = current_module
        if ':' in type_name:
            prefix = type_name.split(':')[0]
            target_mod = prefix_resolver(type_stmt, prefix)
            
        typedef_key = f"{target_mod}:{clean_name}"
        if typedef_key in self.typedefs:
            typedef_stmt = self.typedefs[typedef_key]
            if hasattr(typedef_stmt, 'substmts'):
                for sub in typedef_stmt.substmts:
                    if sub.keyword == 'type':
                        return self.resolve_type(sub, target_mod, prefix_resolver)
        return XSD.string

class YANGDependencyResolver:
    def __init__(self, yang_dir: Path):
        self.yang_dir = Path(yang_dir)
        self.repo = repository.FileRepository(str(self.yang_dir))
        self.ctx = context.Context(self.repo)
        self.modules: Dict[str, Any] = {}

    def load_all_modules(self, yang_files: List[str]) -> None:
        for yang_file in yang_files:
            self.load_module(yang_file)
        for yang_file in sorted(self.yang_dir.glob("*.yang")):
            if yang_file.name not in self.modules:
                self.load_module(yang_file.name)

    def load_module(self, filename: str) -> Optional[Any]:
        if filename in self.modules: return self.modules[filename]
        filepath = self.yang_dir / filename
        if not filepath.exists(): return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            module = self.ctx.add_module(filename, text)
            if not module:
                log.error(f"Failed to parse: {filename}")
                return None
            self.modules[filename] = module
            log.info(f"✓ Loaded {filename}")
            return module
        except Exception as e:
            log.error(f"Error loading {filename}: {e}")
            return None

class IdentityResolver:
    def __init__(self, modules: Dict[str, Any]):
        self.modules = modules
        self.identity_map: Dict[str, Any] = {}
        self.identity_bases: Dict[str, List[str]] = {}
        self._collect_all_identities()

    def _collect_all_identities(self) -> None:
        for module_name, module in self.modules.items():
            mod_clean = extract_module_name(module_name)
            if not hasattr(module, 'substmts'): continue
            for stmt in module.substmts:
                if hasattr(stmt, 'keyword') and stmt.keyword == 'identity':
                    identity_name = stmt.arg
                    key = f"{mod_clean}:{identity_name}"
                    self.identity_map[key] = stmt
                    base_names = self._extract_base_identity(stmt)
                    self.identity_bases[key] = base_names

    def _extract_base_identity(self, identity_stmt: Any) -> List[str]:
        bases = []
        if not hasattr(identity_stmt, 'substmts'): return bases
        for sub in identity_stmt.substmts:
            if not hasattr(sub, 'keyword') or sub.keyword != 'base': continue
            base_ref = sub.arg if hasattr(sub, 'arg') else None
            if base_ref:
                bases.append(base_ref)
        return bases

class EnhancedLeafrefResolver:
    def __init__(self, modules: Dict[str, Any], class_paths: Dict[str, URIRef], ex: Namespace):
        self.modules = modules
        self.class_paths = class_paths
        self.ex = ex
        self.xpath_cache: Dict[str, Optional[Tuple[str, URIRef, str]]] = {}

    def is_leafref(self, type_stmt: Any) -> bool:
        return getattr(type_stmt, 'arg', None) == 'leafref'

    def extract_xpath_path(self, leafref_type: Any) -> Optional[str]:
        if not hasattr(leafref_type, 'substmts'): return None
        for sub in leafref_type.substmts:
            if hasattr(sub, 'keyword') and sub.keyword == 'path':
                return sub.arg if hasattr(sub, 'arg') else None
        return None

    def resolve_leafref_target(self, leafref_type: Any, context_path: str) -> Optional[Tuple[str, URIRef, str]]:
        xpath_path = self.extract_xpath_path(leafref_type)
        if not xpath_path: return None
        cache_key = f"{context_path}::{xpath_path}"
        if cache_key in self.xpath_cache:
            cached = self.xpath_cache[cache_key]
            if cached: return cached
            return None

        if hasattr(leafref_type, 'i_leafref_ptr') and leafref_type.i_leafref_ptr:
            target_class_node = getattr(leafref_type.i_leafref_ptr, 'parent', None)
            if target_class_node:
                target_path = self._build_path_from_node(target_class_node)
                if target_path in self.class_paths:
                    result = (target_path, self.class_paths[target_path], xpath_path)
                    self.xpath_cache[cache_key] = result
                    return result

        resolved = self._resolve_xpath_manually(xpath_path, context_path)
        if resolved: 
            result = (*resolved, xpath_path)
            self.xpath_cache[cache_key] = result
            return result
        
        self.xpath_cache[cache_key] = None
        return None

    def _build_path_from_node(self, node: Any) -> str:
        path_parts = []
        current = node
        module_name = None
        while current:
            if hasattr(current, 'keyword') and current.keyword in ('module', 'submodule'):
                module_name = current.arg if hasattr(current, 'arg') else None
                break
            if hasattr(current, 'arg') and current.arg: 
                if hasattr(current, 'keyword') and current.keyword in ('choice', 'case'):
                    pass  # skip — flatten choice/case out of path
                else:
                    path_parts.insert(0, current.arg)
            current = getattr(current, 'parent', None)
            
        if module_name and path_parts: 
            return '/' + module_name + '/' + '/'.join(path_parts)
        elif path_parts: 
            return '/' + '/'.join(path_parts)
        return '/'

    def _resolve_xpath_manually(self, xpath: str, context_path: str) -> Optional[Tuple[str, URIRef]]:
        clean_xpath = self._clean_xpath(xpath)
        if clean_xpath.startswith('/'): return self._resolve_absolute_path(clean_xpath)
        if '../' in clean_xpath: return self._resolve_relative_path(clean_xpath, context_path)
        if 'current()' in clean_xpath: return self._resolve_current_path(clean_xpath, context_path)
        return None

    def _clean_xpath(self, xpath: str) -> str:
        cleaned = re.sub(r'[a-zA-Z0-9_-]+:', '', xpath)
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        cleaned = cleaned.replace('current()', '')
        cleaned = re.sub(r'/+', '/', cleaned)
        return cleaned.strip()

    def _resolve_absolute_path(self, xpath: str) -> Optional[Tuple[str, URIRef]]:
        parts = [p for p in xpath.split('/') if p]
        
        for i in range(len(parts), 0, -1):
            candidate_path = '/' + '/'.join(parts[:i])
            if candidate_path in self.class_paths: 
                return (candidate_path, self.class_paths[candidate_path])
                
            for registered_path in self.class_paths.keys():
                if registered_path.endswith(candidate_path):
                    return (registered_path, self.class_paths[registered_path])
                    
        return None

    def _resolve_relative_path(self, xpath: str, context_path: str) -> Optional[Tuple[str, URIRef]]:
        up_count = xpath.count('../')
        context_parts = [p for p in context_path.split('/') if p]
        
        base_parts = context_parts[:-up_count] if up_count > 0 and up_count <= len(context_parts) else []
        remaining = xpath.replace('../', '')
        remaining_parts = [p for p in remaining.split('/') if p]
        full_parts = base_parts + remaining_parts
        
        for i in range(len(full_parts), 0, -1):
            candidate_path = '/' + '/'.join(full_parts[:i])
            if candidate_path in self.class_paths:
                return (candidate_path, self.class_paths[candidate_path])
                
            suffix_parts = full_parts[1:i] if len(full_parts) > 1 else full_parts[:i]
            if not suffix_parts: 
                continue
            
            candidate_suffix = '/' + '/'.join(suffix_parts)
            
            for registered_path, uri in self.class_paths.items():
                if registered_path.endswith(candidate_suffix):
                    return (registered_path, uri)
                    
        return None

    def _resolve_current_path(self, xpath: str, context_path: str) -> Optional[Tuple[str, URIRef]]:
        cleaned = xpath.replace('current()', '').strip('/')
        if cleaned.startswith('../'): return self._resolve_relative_path(cleaned, context_path)
        context_parts = [p for p in context_path.split('/') if p]
        remaining_parts = [p for p in cleaned.split('/') if p]
        full_parts = context_parts + remaining_parts
        for i in range(len(full_parts), 0, -1):
            candidate_path = '/' + '/'.join(full_parts[:i])
            if candidate_path in self.class_paths:
                return (candidate_path, self.class_paths[candidate_path])
        return None

class RefineResolver:
    def __init__(self):
        self.refines: Dict[str, Dict[str, Any]] = {}

    def extract_refines(self, uses_stmt: Any) -> Dict[str, Dict[str, Any]]:
        refines = {}
        if not hasattr(uses_stmt, 'substmts'): return refines
        for sub in uses_stmt.substmts:
            if not hasattr(sub, 'keyword'): continue
            if sub.keyword == 'refine':
                node_path = sub.arg if hasattr(sub, 'arg') else ''
                refine_props = self._extract_refine_properties(sub)
                refines[node_path] = refine_props
        return refines

    def _extract_refine_properties(self, refine_stmt: Any) -> Dict[str, Any]:
        props = {}
        if not hasattr(refine_stmt, 'substmts'): return props
        for sub in refine_stmt.substmts:
            if not hasattr(sub, 'keyword'): continue
            if sub.keyword in ('mandatory', 'min-elements', 'max-elements', 'presence', 'description'):
                props[sub.keyword] = sub.arg if hasattr(sub, 'arg') else None
        return props

class GroupingResolver:
    def __init__(self, modules: Dict[str, Any]):
        self.modules = modules
        self.groupings: Dict[str, Any] = {}
        self.refine_resolver = RefineResolver()
        self._collect_all_groupings()

    def _collect_all_groupings(self) -> None:
        for module_name, module in self.modules.items():
            mod_clean = extract_module_name(module_name)
            if hasattr(module, 'substmts'):
                for stmt in module.substmts:
                    if hasattr(stmt, 'keyword') and stmt.keyword == 'grouping':
                        group_name = stmt.arg
                        self.groupings[f"{mod_clean}:{group_name}"] = stmt
            if hasattr(module, 'i_groupings') and module.i_groupings:
                for group_name, group_stmt in module.i_groupings.items():
                    self.groupings[f"{mod_clean}:{group_name}"] = group_stmt

    def get_grouping(self, grouping_name: str, target_module: str) -> Optional[Any]:
        clean_name = grouping_name.split(':')[-1]
        key = f"{target_module}:{clean_name}"
        return self.groupings.get(key)

    def get_grouping_children(self, grouping_name: str, target_module: str) -> List[Tuple[str, Any, str]]:
        clean_name = grouping_name.split(':')[-1]
        key = f"{target_module}:{clean_name}"
        grouping = self.groupings.get(key)
        
        if not grouping or not hasattr(grouping, 'substmts'): return []
        children = []
        for sub in grouping.substmts:
            if hasattr(sub, 'keyword') and hasattr(sub, 'arg'):
                if sub.keyword in ('leaf', 'leaf-list', 'container', 'list', 'choice', 'rpc', 'notification', 'uses', 'anydata'):
                    children.append((sub.arg, sub, sub.keyword))
        return children

    def get_grouping_description(self, grouping_name: str, target_module: str) -> Optional[str]:
        clean_name = grouping_name.split(':')[-1]
        key = f"{target_module}:{clean_name}"
        grouping = self.groupings.get(key)
        
        if not grouping or not hasattr(grouping, 'substmts'): return None
        for sub in grouping.substmts:
            if hasattr(sub, 'keyword') and sub.keyword == 'description':
                return sub.arg if hasattr(sub, 'arg') else None
        return None

class GroupingContextTracker:
    def __init__(self):
        self.uses_stack: List[Tuple[str, str]] = []
        self.expanded_uses: Set[str] = set()

    def push_grouping_context(self, grouping_name: str, context_path: str) -> None:
        context_id = f"{grouping_name}@{context_path}"
        if context_id not in self.expanded_uses:
            self.uses_stack.append((grouping_name, context_path))
            self.expanded_uses.add(context_id)

    def pop_grouping_context(self) -> Optional[Tuple[str, str]]:
        if self.uses_stack: return self.uses_stack.pop()
        return None

    def is_circular_reference(self, grouping_name: str) -> bool:
        return any(name == grouping_name for name, _ in self.uses_stack)

class YANGToOWL:
    def __init__(self, yang_dir: str, base_uri: str = "http://example.org/ontology/", raw_mode: bool = False):
        self.yang_dir = Path(yang_dir)
        self.base_uri = base_uri.rstrip('/')
        self.raw_mode = raw_mode
        self.ex = Namespace(self.base_uri + '/')
        self.resolver = YANGDependencyResolver(self.yang_dir)
        self.type_resolver = YANGTypeResolver()
        
        # Dual Graph Architecture: Separate standard OWL and SHACL rules
        self.graph = Graph()
        self.shacl_graph = Graph()
        
        for g in (self.graph, self.shacl_graph):
            g.bind('ex', self.ex)
            g.bind('owl', OWL)
            g.bind('rdf', RDF)
            g.bind('rdfs', RDFS)
            g.bind('xsd', XSD)
            g.bind('prov', PROV)
            g.bind('sh', SH)
            
        self.processed: Set[str] = set()
        self.class_paths: Dict[str, URIRef] = {}
        self.module_prefixes: Dict[str, str] = {}
        self.augment_targets: Dict[str, Dict] = {}
        self.module_namespaces: Dict[str, str] = {}
        self.current_module_name: Optional[str] = None
        self.identity_resolver: Optional[IdentityResolver] = None
        self.identity_class_uris: Dict[str, URIRef] = {}
        self.leafref_resolver: Optional[EnhancedLeafrefResolver] = None
        self.grouping_resolver: Optional[GroupingResolver] = None
        self.grouping_context_tracker: Optional[GroupingContextTracker] = None
        self.grouping_class_uris: Dict[str, URIRef] = {}
        self.rpc_classes: Dict[str, URIRef] = {}
        self.feature_classes: Dict[str, URIRef] = {}
        
        self.constraint_count = 0
        self.typedef_restrictions: Dict[str, URIRef] = {}
        self.leaf_type_map: Dict[str, str] = {}
        self.enumeration_count = 0
        self.grouping_count = 0
        self.uses_count = 0
        self.leafref_resolved_count = 0
        self.leafref_unresolved_count = 0
        self.identityref_resolved_count = 0
        self.prov_paths: Dict[str, str] = {}
        self.pending_leafrefs: List[Tuple[URIRef, Any, str, Optional[URIRef], str]] = []
        self.deferred_augments: List[Tuple[str, Any]] = [] 

    def _get_target_module_from_prefix(self, stmt: Any, prefix: str) -> str:
        if self.current_module_name in self.module_prefixes and self.module_prefixes[self.current_module_name] == prefix:
            return self.current_module_name
        
        root_module = getattr(stmt, 'i_module', None)
        if not root_module and hasattr(stmt, 'top'):
            root_module = stmt.top
            
        if root_module and hasattr(root_module, 'substmts'):
            for sub in root_module.substmts:
                if sub.keyword == 'import':
                    if hasattr(sub, 'substmts'):
                        for s in sub.substmts:
                            if s.keyword == 'prefix' and hasattr(s, 'arg') and s.arg == prefix:
                                return extract_module_name(sub.arg)
                                
        return self.current_module_name

    def _is_enumeration_type(self, type_stmt: Any) -> bool:
        return getattr(type_stmt, 'arg', None) == 'enumeration'

    def _add_constraint_triples(self, uri: URIRef, constraints: Dict[str, Any]) -> None:
        self.shacl_graph.add((uri, RDF.type, SH.PropertyShape))
        
        if 'range' in constraints and isinstance(constraints['range'], dict):
            range_info = constraints['range']
            if 'min' in range_info:
                self.shacl_graph.add((uri, SH.minInclusive, Literal(range_info['min'])))
                self.constraint_count += 1
            if 'max' in range_info:
                self.shacl_graph.add((uri, SH.maxInclusive, Literal(range_info['max'])))
                self.constraint_count += 1
        if 'length' in constraints and isinstance(constraints['length'], dict):
            length_info = constraints['length']
            if 'minLength' in length_info:
                self.shacl_graph.add((uri, SH.minLength, Literal(length_info['minLength'])))
                self.constraint_count += 1
            if 'maxLength' in length_info:
                self.shacl_graph.add((uri, SH.maxLength, Literal(length_info['maxLength'])))
                self.constraint_count += 1
        if 'patterns' in constraints and isinstance(constraints['patterns'], list):
            for pattern in constraints['patterns']:
                if pattern:
                    self.shacl_graph.add((uri, SH.pattern, Literal(pattern)))
                    self.constraint_count += 1

    def _process_xpath_constraints(self, stmt: Any, uri: URIRef) -> None:
        if not hasattr(stmt, 'substmts'): return
        for sub in stmt.substmts:
            if not hasattr(sub, 'keyword'): continue
            if sub.keyword == 'must':
                xpath_expr = sub.arg if hasattr(sub, 'arg') else ""
                self.shacl_graph.add((uri, SH.condition, Literal(xpath_expr)))
                for detail in sub.substmts:
                    if detail.keyword == 'error-message':
                        self.shacl_graph.add((uri, SH.message, Literal(detail.arg)))
            elif sub.keyword == 'when':
                xpath_expr = sub.arg if hasattr(sub, 'arg') else ""
                self.shacl_graph.add((uri, SH.deactivated, Literal(xpath_expr)))
                self.shacl_graph.add((uri, RDFS.comment, Literal(f"Conditional: exists when {xpath_expr}")))

    def _get_stmt_prefix(self, stmt: Any) -> str:
        if hasattr(stmt, 'i_module') and stmt.i_module:
            if hasattr(stmt.i_module, 'i_prefix'): return stmt.i_module.i_prefix
            if hasattr(stmt.i_module, 'prefix'): return stmt.i_module.prefix
        if hasattr(stmt, 'top') and stmt.top:
            if hasattr(stmt.top, 'i_prefix'): return stmt.top.i_prefix
            prefix_stmt = stmt.top.search_one('prefix')
            if prefix_stmt: return prefix_stmt.arg
        if self.current_module_name in self.module_prefixes:
            return self.module_prefixes[self.current_module_name]
        return "ex"

    def _get_prov_segment(self, stmt: Any) -> str:
        if not hasattr(stmt, 'arg') or not hasattr(stmt, 'keyword'): return ""
        prefix = self._get_stmt_prefix(stmt)
        return f"{prefix}:{stmt.arg}?{stmt.keyword}"

    def is_leafref(self, type_stmt: Any) -> bool:
        return getattr(type_stmt, 'arg', None) == 'leafref'

    def _normalize_path(self, path: str) -> str:
        if not path: return "/"
        clean_path = re.sub(r'[a-zA-Z0-9_-]+:', '', path)
        clean_path = re.sub(r'/+', '/', clean_path)
        clean_path = '/' + clean_path.lstrip('/')
        parts = [p for p in clean_path.split('/') if p]
        
        known_modules = {extract_module_name(m) for m in self.resolver.modules.keys()}
        
        if self.current_module_name and parts:
            if parts[0] in known_modules:
                return clean_path
            return '/' + self.current_module_name + clean_path
        return clean_path

    def _get_identity_uri(self, stmt: Any, identity_name: str) -> URIRef:
        clean_name = identity_name.split(':')[-1]
        target_mod = self.current_module_name
        if ':' in identity_name:
            prefix = identity_name.split(':')[0]
            target_mod = self._get_target_module_from_prefix(stmt, prefix)
        return self.ex[f"identity/{target_mod}/{clean_name}"]

    def _get_grouping_uri(self, stmt: Any, grouping_name: str) -> URIRef:
        clean_name = grouping_name.split(':')[-1]
        target_mod = self.current_module_name
        if ':' in grouping_name:
            prefix = grouping_name.split(':')[0]
            target_mod = self._get_target_module_from_prefix(stmt, prefix)
        return self.ex[f"grouping/{target_mod}/{clean_name}"]

    def _get_typedef_module(self, typedef_stmt: Any) -> str:
        if hasattr(typedef_stmt, 'i_module') and typedef_stmt.i_module:
            return extract_module_name(typedef_stmt.i_module.arg)
            
        current = getattr(typedef_stmt, 'parent', None)
        while current:
            if hasattr(current, 'keyword') and current.keyword in ('module', 'submodule'):
                return extract_module_name(current.arg)
            current = getattr(current, 'parent', None)
            
        return self.current_module_name or 'unknown'
    
    def convert(self, main_module: str, output_file: str) -> None:
        log.info("=" * 70)
        log.info("YANG to OWL Converter v4.7.30 (Separate SHACL Output)")
        log.info("=" * 70)

        log.info("\n[Step 1] Loading YANG modules...")
        self.resolver.load_all_modules([main_module])

        log.info("[Step 2] Initializing resolvers...")
        self.identity_resolver = IdentityResolver(self.resolver.modules)
        self.leafref_resolver = EnhancedLeafrefResolver(self.resolver.modules, self.class_paths, self.ex)
        self.grouping_resolver = GroupingResolver(self.resolver.modules)
        self.grouping_context_tracker = GroupingContextTracker()

        log.info("[Step 3] Registering module namespaces...")
        self._register_module_namespaces()

        log.info("[Step 4] ⭐ Processing grouping definitions as OWL abstract classes...")
        self._process_grouping_definitions()

        log.info("[Step 5] Processing YANG data model...")
        sorted_modules = sorted(self.resolver.modules.items(), key=lambda x: x[0])
        for module_name, module in sorted_modules:
            log.info(f" Processing: {module_name}")
            self.current_module_name = extract_module_name(module_name)
            self._process_module(module, module_name)

        log.info("[Step 6] Processing identity hierarchies...")
        self._process_identities()

        log.info("[Step 7] Processing augmentations with uses expansion...")
        self._process_deferred_augmentations()

        log.info("[Step 8] Generating container object properties...")
        self._process_containers_for_properties()

        log.info("[Step 9] Processing imported module bases...")
        self._process_imported_module_bases()

        log.info("[Step 10] Creating SHACL Shapes for Typedefs...")
        self._create_shacl_typedef_shapes()

        log.info("[Step 11] Processing Enumeration Types...")
        self._process_enumerations()

        log.info("[Step 12] Resolving Pending Leafrefs (Pass 2)...")
        self._resolve_pending_leafrefs()

        # ==========================================
        # --- START OF CUSTOM TBOX PATCH --- for leafref that are strings but we want to link to deicces
        # ==========================================
        if not self.raw_mode:
            log.info("[Step 13] Applying custom TBox extensions for direct URIs...")
            
            # --- NEW ADDITION: Logical Connection Property ---
            logically_connected = self.ex.logicallyConnectedTo
            self.graph.add((logically_connected, RDF.type, OWL.ObjectProperty))
            self.graph.add((logically_connected, RDFS.label, Literal("logically connected to", datatype=XSD.string)))
            self.graph.add((logically_connected, RDFS.comment, Literal("A direct logical relationship between an edge active element (e.g., ONT) and a core active element (e.g., OLT), bypassing physical cabling.", datatype=XSD.string)))
            self.graph.add((logically_connected, RDF.type, OWL.SymmetricProperty))
            
            rank_prop = self.ex["rank"]
            self.graph.add((rank_prop, RDF.type, OWL.DatatypeProperty))
            self.graph.add((rank_prop, RDFS.label, Literal("rank")))
            self.graph.add((rank_prop, RDFS.range, XSD.integer))

            base_prefix = str(self.base_uri).rstrip("/") + "/"
            ni_base_path = f"{base_prefix}ietf-network-inventory/network-inventory/"
            
            # 1. Define namespaces to match the exact Class URIs in your TBox
            cab_ns = Namespace(f"{ni_base_path}cable/")
            cabChild_ns = Namespace(f"{ni_base_path}cable/child-cable/")
            nwi_ns = Namespace(ni_base_path)
            
            # Fix: Using parent paths for ranges to match TBox Class URIs exactly
            nwiNEs_ns = Namespace(f"{ni_base_path}network-elements/")
            nwiComp_ns = Namespace(f"{ni_base_path}network-elements/network-element/components/")
            
            # Define common domains for all cable termination points
            cable_termination_domains = [
                cab_ns["a-end"], 
                cab_ns["z-end"], 
                cabChild_ns["a-end"], 
                cabChild_ns["z-end"]
            ]

            # 2. Define ex:device-ref for Passive Devices (ATBs/Enclosures)
            device_ref = self.ex["device-ref"]
            self.graph.add((device_ref, RDF.type, OWL.ObjectProperty))
            self.graph.add((device_ref, RDFS.label, Literal("device-ref")))
            self.graph.add((device_ref, RDFS.comment, Literal("Reference to a passive device (e.g. ATB or Enclosure).")))
            self.graph.add((device_ref, RDFS.range, nwi_ns["passive-device"]))
            for dom in cable_termination_domains:
                self.graph.add((device_ref, RDFS.domain, dom))

            # 3. Define Active Reference ObjectProperties (for NTDs and Exchanges)
            # This resolves the namespace mismatch by targeting the correct range classes
            #for prop_name, range_uri in [
            #    ("ne-ref", nwiNEs_ns["network-element"]),
            #    ("component-ref", nwiComp_ns["component"])
            #]:
            #    prop = self.ex[prop_name]
            #    self.graph.add((prop, RDF.type, OWL.ObjectProperty))
            #    self.graph.add((prop, RDFS.label, Literal(prop_name)))
            #    self.graph.add((prop, RDFS.range, range_uri))
            #    for dom in cable_termination_domains:
            #        self.graph.add((prop, RDFS.domain, dom))
            # A. ne-ref: Keep strict domain constraints (only for cable ends) as both the passive inventory and active currently have component-ref 
            ne_prop = self.ex["ne-ref"]
            self.graph.add((ne_prop, RDF.type, OWL.ObjectProperty))
            self.graph.add((ne_prop, RDFS.label, Literal("ne-ref")))
            self.graph.add((ne_prop, RDFS.range, nwiNEs_ns["network-element"]))
            for dom in cable_termination_domains:
                self.graph.add((ne_prop, RDFS.domain, dom))

            # B. component-ref: Remove domain constraints (globally applicable)
            # This reflects the netork inventory draft-18 promotion of component-ref to a core grouping
            comp_prop = self.ex["component-ref"]

            # WIPE any automatically inferred domains from the base parsing phase
            self.graph.remove((comp_prop, RDFS.domain, None))

            self.graph.add((comp_prop, RDF.type, OWL.ObjectProperty))
            self.graph.add((comp_prop, RDFS.label, Literal("component-ref")))
            self.graph.add((comp_prop, RDFS.range, nwiComp_ns["component"]))
            # NOTE: The loop applying RDFS.domain to cable_termination_domains has been removed.
                
            # ==========================================
            # --- STEP 14: RDF-star shortcut properties ---
            # These enable: ?device ex:hasUpstreamDevice+ ?odf  (single SPARQL line)
            # The ABoxConnectivityEnricher materialises the actual instance triples.
            # ==========================================
            log.info("[Step 14] Adding RDF-star upstream connectivity properties to TBox...")

            has_upstream_prop = self.ex["hasUpstreamDevice"]
            self.graph.add((has_upstream_prop, RDF.type, OWL.ObjectProperty))
            self.graph.add((has_upstream_prop, RDFS.label, Literal("has upstream device")))
            self.graph.add((has_upstream_prop, RDFS.comment, Literal(
                "Materialised shortcut: device at Z-end of a cable -> device at A-end (upstream). "
                "Annotated via RDF-star with ex:viaCable and ex:cableRole. "
                "Enables: ?device ex:hasUpstreamDevice+ ?odf ."
            )))

            has_downstream_prop = self.ex["hasDownstreamDevice"]
            self.graph.add((has_downstream_prop, RDF.type, OWL.ObjectProperty))
            self.graph.add((has_downstream_prop, RDFS.label, Literal("has downstream device")))
            self.graph.add((has_downstream_prop, RDFS.comment, Literal("Inverse of ex:hasUpstreamDevice.")))
            self.graph.add((has_downstream_prop, OWL.inverseOf, has_upstream_prop))

            via_cable_prop = self.ex["viaCable"]
            self.graph.add((via_cable_prop, RDF.type, OWL.ObjectProperty))
            self.graph.add((via_cable_prop, RDFS.label, Literal("via cable")))
            self.graph.add((via_cable_prop, RDFS.comment, Literal(
                "RDF-star annotation on ex:hasUpstreamDevice: the physical cable realising this hop."
            )))

            cable_role_anno_prop = self.ex["cableRole"]
            self.graph.add((cable_role_anno_prop, RDF.type, OWL.ObjectProperty))
            self.graph.add((cable_role_anno_prop, RDFS.label, Literal("cable role")))
            self.graph.add((cable_role_anno_prop, RDFS.comment, Literal(
                "RDF-star annotation on ex:hasUpstreamDevice: role/tier of the cable. "
                "Priority ascending: internal=0, drop=1, access=2, branch=3, aggregation=4, distribution=5, trunk=6."
            )))

            role_priority_prop = self.ex["rolePriority"]
            self.graph.add((role_priority_prop, RDF.type, OWL.DatatypeProperty))
            self.graph.add((role_priority_prop, RDFS.label, Literal("role priority")))
            self.graph.add((role_priority_prop, RDFS.range, XSD.integer))
            self.graph.add((role_priority_prop, RDFS.comment, Literal(
                "Numeric priority for cable role used in upstream-ascending traversal filter. "
                "Matches CABLE_RANK_MAP: internal=0, drop=1, access=2, branch=3, aggregation=4, distribution=5, trunk=6."
            )))

            # Annotate each cable-role named individual with its numeric priority
            _nwi_pass_id_ns = Namespace(f"{str(self.base_uri).rstrip('/')}/identity/ietf-nwi-passive-inventory/")
            for role_name, priority in CABLE_RANK_MAP.items():
                role_uri = _nwi_pass_id_ns[role_name]
                self.graph.add((role_uri, role_priority_prop, Literal(priority, datatype=XSD.integer)))

            log.info(f"  ✓ Added {len(CABLE_RANK_MAP)} cable-role priority annotations.")
        else:
            log.info("[Step 13 & 14] SKIPPED custom TBox extensions (--raw mode active).")

        # ==========================================
        # --- END OF CUSTOM TBOX PATCH ---
        # ==========================================
        shacl_file = str(Path(output_file).with_suffix('.shacl'))
        
        self._bind_ontology_prefixes()
        log.info(f"\n[Output] Saving Core OWL Ontology to {output_file}...")
        self.graph.serialize(destination=output_file, format='turtle')

        self._bind_ontology_prefixes()
        log.info(f"[Output] Saving Validation SHACL Shapes to {shacl_file}...")
        self.shacl_graph.serialize(destination=shacl_file, format='turtle')
        
        log.info(f"\n" + "=" * 20 + " CONVERSION REPORT " + "=" * 20)
        log.info(f"✓ Ontology triples generated: {len(self.graph)}")
        log.info(f"✓ SHACL triples generated: {len(self.shacl_graph)}")
        log.info(f"✓ SHACL Shapes created: {len(self.typedef_restrictions)}")
        log.info(f"✓ Enumeration individuals created: {self.enumeration_count}")
        log.info(f"✓ Grouping abstract classes created: {self.grouping_count}")
        log.info(f"✓ Uses statements expanded: {self.uses_count}")
        log.info(f"✓ Leafref resolved: {self.leafref_resolved_count}")
        log.info(f"✓ Leafref unresolved: {self.leafref_unresolved_count}")
        log.info(f"✓ Identityref resolved as ObjectProperties: {self.identityref_resolved_count}")
        log.info("=" * 59 + "\n")

    def _register_module_namespaces(self) -> None:
        for module_name, module in self.resolver.modules.items():
            if hasattr(module, 'namespace'):
                ns = module.namespace
                self.module_namespaces[module_name] = ns
                prefix = module.prefix if hasattr(module, 'prefix') else module_name
                self.module_prefixes[module_name] = prefix

    def _process_grouping_definitions(self) -> None:
        if not self.grouping_resolver: return
        for key, grouping_stmt in self.grouping_resolver.groupings.items():
            mod_name, grouping_name = key.split(':', 1)
            grouping_uri = self.ex[f"grouping/{mod_name}/{grouping_name}"]
            self.graph.add((grouping_uri, RDF.type, OWL.Class))
            self.graph.add((grouping_uri, RDFS.label, Literal(grouping_name)))
            self.graph.add((grouping_uri, RDFS.comment, Literal(f"Grouping definition: {grouping_name}")))
            self.grouping_class_uris[key] = grouping_uri
            self.grouping_count += 1
            desc = self.grouping_resolver.get_grouping_description(grouping_name, mod_name)
            if desc:
                self.graph.add((grouping_uri, RDFS.comment, Literal(desc)))

    def _shape_uri_for_class(self, class_uri: URIRef) -> URIRef:
        return URIRef(str(class_uri) + "/shape")

    def _ensure_node_shape(self, class_uri: URIRef) -> URIRef:
        shape_uri = self._shape_uri_for_class(class_uri)
        if (shape_uri, RDF.type, SH.NodeShape) not in self.shacl_graph:
            self.shacl_graph.add((shape_uri, RDF.type, SH.NodeShape))
            self.shacl_graph.add((shape_uri, SH.targetClass, class_uri))
            self.shacl_graph.add((shape_uri, RDFS.label, Literal(f"Shape for {class_uri.split('/')[-1]}")))
        return shape_uri
    
    def _add_property_shape(
        self,
        class_uri: URIRef,
        prop_uri: URIRef,
        *,
        datatype: Optional[URIRef] = None,
        value_class: Optional[URIRef] = None,
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
        pattern: Optional[str] = None,
        min_inclusive: Optional[Any] = None,
        max_inclusive: Optional[Any] = None,
        node_shape: Optional[URIRef] = None,
        message: Optional[str] = None,
    ) -> None:
        shape_uri = self._ensure_node_shape(class_uri)
        ps = BNode()
        self.shacl_graph.add((shape_uri, SH.property, ps))
        self.shacl_graph.add((ps, RDF.type, SH.PropertyShape))
        self.shacl_graph.add((ps, SH.path, prop_uri))

        if datatype is not None:
            self.shacl_graph.add((ps, SH.datatype, datatype))
        if value_class is not None:
            self.shacl_graph.add((ps, SH["class"], value_class))
            self.shacl_graph.add((ps, SH.nodeKind, SH.IRI))
        if node_shape is not None:
            self.shacl_graph.add((ps, SH.node, node_shape))

        if min_count is not None:
            self.shacl_graph.add((ps, SH.minCount, Literal(int(min_count), datatype=XSD.integer)))
        if max_count is not None:
            self.shacl_graph.add((ps, SH.maxCount, Literal(int(max_count), datatype=XSD.integer)))

        if pattern is not None:
            self.shacl_graph.add((ps, SH.pattern, Literal(pattern)))
        if min_inclusive is not None:
            self.shacl_graph.add((ps, SH.minInclusive, Literal(min_inclusive)))
        if max_inclusive is not None:
            self.shacl_graph.add((ps, SH.maxInclusive, Literal(max_inclusive)))
        if message is not None:
            self.shacl_graph.add((ps, SH.message, Literal(message)))
            
    def _add_key_uniqueness_sparql(self, list_class_uri: URIRef, key_prop_uris: List[URIRef], message: str) -> None:
        shape_uri = self._ensure_node_shape(list_class_uri)
        sparql_bn = BNode()
        self.shacl_graph.add((shape_uri, SH.sparql, sparql_bn))

        where_self = "\n".join([f"  $this <{kp}> ?k{i} ." for i, kp in enumerate(key_prop_uris, start=1)])
        where_other = "\n".join([f"  ?other <{kp}> ?k{i} ." for i, kp in enumerate(key_prop_uris, start=1)])

        query = f"""
SELECT $this WHERE {{
  $this a <{list_class_uri}> .
{where_self}
  ?other a <{list_class_uri}> .
{where_other}
  FILTER(?other != $this)
}}
""".strip()

        self.shacl_graph.add((sparql_bn, RDF.type, SH.SPARQLConstraint))
        self.shacl_graph.add((sparql_bn, SH.message, Literal(message)))
        self.shacl_graph.add((sparql_bn, SH.select, Literal(query)))


    def _process_module(self, module: Any, module_name: str) -> None:
        if not hasattr(module, 'substmts'): return
        for stmt in module.substmts:
            if not hasattr(stmt, 'keyword'): continue
            keyword = stmt.keyword
            if keyword == 'typedef': self._process_typedef(stmt)
            elif keyword == 'identity': self._process_identity(stmt)
            elif keyword == 'rpc': self._process_rpc(stmt)
            elif keyword == 'notification': self._process_notification(stmt)
            elif keyword in ('container', 'list', 'leaf'):
                normalized_path = self._normalize_path(f"/{stmt.arg}")
                if keyword == 'container': self._process_container(stmt, normalized_path)
                elif keyword == 'list': self._process_list(stmt, normalized_path)
                elif keyword == 'leaf': self._process_leaf(stmt, normalized_path)
            elif keyword == 'augment':
                self.deferred_augments.append((self.current_module_name, stmt))

    def _process_typedef(self, stmt: Any) -> None:
        if hasattr(stmt, 'arg'): self.type_resolver.register_typedef(self.current_module_name, stmt.arg, stmt)

    def _process_identity(self, stmt: Any) -> None:
        if not hasattr(stmt, 'arg'): return
        name = stmt.arg
        uri = self.ex[f"identity/{self.current_module_name}/{name}"]
        prov_path = self._get_prov_segment(stmt)
        self.graph.add((uri, PROV.wasDerivedFrom, Literal(prov_path)))
        self.graph.add((uri, RDF.type, OWL.Class))
        self.graph.add((uri, RDFS.label, Literal(name)))
        self.graph.add((uri, RDF.type, OWL.NamedIndividual))

        if name in CABLE_RANK_MAP:
            rank_value = CABLE_RANK_MAP[name]
            self.graph.add((uri, self.ex.rank, Literal(rank_value, datatype=XSD.integer)))
            log.info(f"  Enriched identity {name} with rank {rank_value}")

        if hasattr(stmt, 'substmts'):
            for sub in stmt.substmts:
                if hasattr(sub, 'keyword') and sub.keyword == 'description':
                    if hasattr(sub, 'arg'):
                        self.graph.add((uri, RDFS.comment, Literal(sub.arg)))

    def _process_identities(self) -> None:
        if not self.identity_resolver: return
        for key, stmt in self.identity_resolver.identity_map.items():
            mod_name, identity_name = key.split(':', 1)
            uri = self.ex[f"identity/{mod_name}/{identity_name}"]
            base_names = self.identity_resolver.identity_bases.get(key, [])
            if base_names:
                for base_name in base_names:
                    base_uri = self._get_identity_uri(stmt, base_name)
                    self.graph.add((uri, RDFS.subClassOf, base_uri))

    def _process_choice(self, choice_stmt: Any, parent_path: str, 
                    parent_uri: URIRef, parent_prov: str = "") -> None:
        """
        YANG choice/case flattening: case alternatives are NOT emitted as
        OWL classes. All children inside a case are promoted directly to
        the parent class level. Disjoint pairs are noted via rdfs:comment only.
        """
        if not hasattr(choice_stmt, 'substmts'):
            return

        for sub in choice_stmt.substmts:
            if not hasattr(sub, 'keyword'):
                continue
            if sub.keyword not in ('case', 'container', 'leaf', 'list',
                                    'leaf-list', 'anydata', 'uses', 'choice'):
                continue

            is_case = sub.keyword == 'case'
            # Unwrap the case wrapper — get its actual content children
            children_to_process = (
                sub.substmts if (is_case and hasattr(sub, 'substmts')) else [sub]
            )

            for child in children_to_process:
                if not hasattr(child, 'keyword'):
                    continue
                keyword = child.keyword
                # Path goes DIRECTLY under parent — no case-{name} segment
                child_path = (
                    self._normalize_path(f"{parent_path}/{child.arg}")
                    if hasattr(child, 'arg') else parent_path
                )
                if keyword == 'leaf':
                    self._process_leaf(child, child_path, parent_uri, parent_prov)
                elif keyword == 'leaf-list':
                    self._process_leaf_list(child, child_path, parent_uri, parent_prov)
                elif keyword == 'container':
                    self._process_container(child, child_path, parent_uri, parent_prov)
                elif keyword == 'list':
                    self._process_list(child, child_path, parent_uri, parent_prov)
                elif keyword == 'uses':
                    self._process_uses_in_container(child, parent_path, parent_uri)
                elif keyword == 'choice':
                    self._process_choice(child, parent_path, parent_uri, parent_prov)


    def _process_container(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None, parent_prov: str = "") -> URIRef:
        if not hasattr(stmt, 'arg'): return URIRef("")
        name = stmt.arg
        full_path = path
        uri = self.ex[full_path.lstrip('/')]

        current_segment = self._get_prov_segment(stmt)
        full_prov = f"{parent_prov}/{current_segment}" if parent_prov else current_segment
        
        self.prov_paths[full_path] = full_prov
        self.graph.add((uri, PROV.wasDerivedFrom, Literal(full_prov)))
        self.graph.add((uri, RDF.type, OWL.Class))
        self.graph.add((uri, RDFS.label, Literal(name)))

        is_config = True
        status_val = None
        if hasattr(stmt, 'substmts'):
            for sub in stmt.substmts:
                if hasattr(sub, 'keyword'):
                    if sub.keyword == 'description' and hasattr(sub, 'arg'): 
                        self.graph.add((uri, RDFS.comment, Literal(sub.arg)))
                    elif sub.keyword == 'config' and getattr(sub, 'arg', '') == 'false':
                        is_config = False
                    elif sub.keyword == 'status' and hasattr(sub, 'arg'):
                        status_val = sub.arg

        if not is_config:
            self.graph.add((uri, self.ex.isStateData, Literal(True, datatype=XSD.boolean)))
            
        if status_val in ('deprecated', 'obsolete'):
            self.graph.add((uri, OWL.deprecated, Literal(True, datatype=XSD.boolean)))

        self.class_paths[full_path] = uri
        self.processed.add(full_path)

        if hasattr(stmt, 'substmts'):
            for sub in stmt.substmts:
                if not hasattr(sub, 'keyword'): continue
                keyword = sub.keyword
                normalized_child_path = self._normalize_path(f"{full_path}/{sub.arg}") if hasattr(sub, 'arg') else full_path
                
                if keyword == 'container':
                    self._process_container(sub, normalized_child_path, uri, full_prov)
                elif keyword == 'list':
                    self._process_list(sub, normalized_child_path, uri, full_prov)
                elif keyword == 'leaf':
                    self._process_leaf(sub, normalized_child_path, uri, full_prov)
                elif keyword == 'leaf-list':
                    self._process_leaf_list(sub, normalized_child_path, uri, full_prov)
                elif keyword == 'uses':
                    self._process_uses_in_container(sub, full_path, uri)
                elif keyword == 'choice':
                    self._process_choice(sub, full_path, uri, full_prov)
                    
        self._process_xpath_constraints(stmt, uri)
        return uri

    def _process_list(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None, parent_prov: str = "") -> None:
        if not hasattr(stmt, 'arg'): return
        name = stmt.arg
        full_path = path
        uri = self.ex[full_path.lstrip('/')]

        current_segment = self._get_prov_segment(stmt)
        full_prov = f"{parent_prov}/{current_segment}" if parent_prov else current_segment
        
        self.graph.add((uri, RDF.type, OWL.Class))
        self.graph.add((uri, RDFS.label, Literal(name)))
        self.graph.add((uri, PROV.wasDerivedFrom, Literal(full_prov)))

        self.class_paths[full_path] = uri
        key_names = []
        is_config = True

        if hasattr(stmt, 'substmts'):
            for sub in stmt.substmts:
                if sub.keyword == 'description' and hasattr(sub, 'arg'):
                    self.graph.add((uri, RDFS.comment, Literal(sub.arg)))
                elif sub.keyword == 'config' and getattr(sub, 'arg', '') == 'false':
                    is_config = False
                elif sub.keyword == 'key' and hasattr(sub, 'arg'):
                    key_names = sub.arg.split()

        if not is_config:
            self.graph.add((uri, self.ex.isStateData, Literal(True, datatype=XSD.boolean)))

        if parent_uri:
            prop_uri = self.ex[f"has_{name}"]
            self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
            self.graph.add((prop_uri, RDFS.label, Literal(f"has {name}")))
            self.graph.add((prop_uri, RDFS.domain, parent_uri))
            self.graph.add((prop_uri, RDFS.range, uri))

        if hasattr(stmt, 'substmts'):
            for sub in stmt.substmts:
                if not hasattr(sub, 'keyword'): continue
                normalized_child_path = f"{full_path}/{sub.arg}" if hasattr(sub, 'arg') and sub.arg else full_path
                
                if sub.keyword == 'leaf':
                    self._process_leaf(sub, normalized_child_path, uri, full_prov)
                elif sub.keyword == 'leaf-list':
                    self._process_leaf(sub, normalized_child_path, uri, full_prov, is_leaf_list=True)
                elif sub.keyword == 'container':
                    self._process_container(sub, normalized_child_path, uri, full_prov)
                elif sub.keyword == 'list':
                    self._process_list(sub, normalized_child_path, uri, full_prov)
                elif sub.keyword == 'choice':
                    self._process_choice(sub, full_path, uri, full_prov)
                elif sub.keyword == 'uses':
                    self._process_uses_in_container(sub, full_path, uri)

        if key_names:
            key_uris = [self.ex[f"{full_path.lstrip('/')}/{k}"] for k in key_names]
            
            # 1. Maintain standard OWL logic 
            current_node = BNode()
            self.graph.add((uri, OWL.hasKey, current_node))
            for i, key_uri in enumerate(key_uris):
                self.graph.add((current_node, RDF.first, key_uri))
                if i < len(key_uris) - 1:
                    next_node = BNode()
                    self.graph.add((current_node, RDF.rest, next_node))
                    current_node = next_node
                else:
                    self.graph.add((current_node, RDF.rest, RDF.nil))
            
            # 2. Add SHACL property shapes for keys (presence)
            for k, key_uri in zip(key_names, key_uris):
                self._add_property_shape(
                    uri,
                    key_uri,
                    min_count=1,
                    max_count=1,
                    message=f"List key '{k}' is required"
                )
            
            # 3. Add SHACL uniqueness constraint
            self._add_key_uniqueness_sparql(
                uri,
                key_uris,
                message=f"Duplicate key combination in list {name}"
            )


    def _process_leaf(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None, parent_prov: str = "", is_leaf_list: bool = False) -> None:
        if not hasattr(stmt, 'arg'): return
        name = stmt.arg
        full_path = path
        #uri = self.ex[full_path.lstrip('/')]
        uri = self.ex[name]  

        current_segment = self._get_prov_segment(stmt)
        full_prov = f"{parent_prov}/{current_segment}" if parent_prov else current_segment
        self.graph.add((uri, PROV.wasDerivedFrom, Literal(full_prov)))
    
        type_stmt = None
        if hasattr(stmt, 'substmts'):
            for sub in stmt.substmts:
                if hasattr(sub, 'keyword') and sub.keyword == 'type':
                    type_stmt = sub
                    break
        if not type_stmt: return

        is_union = False
        is_enum_typedef = False
        resolved_type_stmt = type_stmt
        type_name_raw = type_stmt.arg if hasattr(type_stmt, 'arg') else None
        type_name_clean = type_name_raw.split(':')[-1] if type_name_raw else None

        target_mod = self.current_module_name
        if type_name_raw and ':' in type_name_raw:
            prefix = type_name_raw.split(':')[0]
            target_mod = self._get_target_module_from_prefix(stmt, prefix)

        typedef_key = f"{target_mod}:{type_name_clean}"
        typedef_stmt = None 

        if typedef_key in self.type_resolver.typedefs:
            typedef_stmt = self.type_resolver.typedefs[typedef_key]
            if hasattr(typedef_stmt, 'substmts'):
                for sub in typedef_stmt.substmts:
                    if sub.keyword == 'type':
                        resolved_type_stmt = sub
                        if getattr(sub, 'arg', '') == 'enumeration':
                            is_enum_typedef = True
                        break

        if hasattr(resolved_type_stmt, 'arg') and resolved_type_stmt.arg == 'union':
            is_union = True

        # 1. Determine Range/Datatype URI
        range_uri = None
        if hasattr(resolved_type_stmt, 'arg') and resolved_type_stmt.arg == 'identityref':
            base_identity = None
            if hasattr(resolved_type_stmt, 'substmts'):
                for sub in resolved_type_stmt.substmts:
                    if hasattr(sub, 'keyword') and sub.keyword == 'base':
                        base_identity = sub.arg.split(':')[-1]
                        break
            self.identityref_resolved_count += 1
            self.graph.add((uri, RDF.type, OWL.ObjectProperty))
            if base_identity:
                range_uri = self._get_identity_uri(resolved_type_stmt, base_identity)
                self.graph.add((uri, RDFS.range, range_uri))

        elif self.leafref_resolver.is_leafref(resolved_type_stmt):
            # Defer to Pass 2 for leafrefs
            self.pending_leafrefs.append((uri, resolved_type_stmt, full_path, parent_uri, name))
            range_uri = None 

        elif is_union:
            # Stardog lacks support for union axioms. We bypass OWL ranges 
            # and rely entirely on SHACL sh:or lists for validation.
            self.graph.add((uri, RDF.type, OWL.DatatypeProperty))
            
            union_members = []
            if hasattr(resolved_type_stmt, 'substmts'):
                for union_sub in resolved_type_stmt.substmts:
                    if hasattr(union_sub, 'keyword') and union_sub.keyword == 'type':
                        if union_sub.arg == 'identityref':
                            base_id = None
                            for sub in union_sub.substmts:
                                if sub.keyword == 'base':
                                    base_id = sub.arg.split(':')[-1]
                                    break
                            if base_id:
                                union_members.append(('class', self._get_identity_uri(union_sub, base_id)))
                        else:
                            member_uri = self.type_resolver.resolve_type(union_sub, self.current_module_name, self._get_target_module_from_prefix)
                            if "www.w3.org/2001/XMLSchema#" in str(member_uri):
                                union_members.append(('datatype', member_uri))
                            elif member_uri == RDFS.Literal:
                                union_members.append(('literal', member_uri))
                            else:
                                union_members.append(('class', member_uri))
            range_uri = ('union', union_members)

        elif hasattr(resolved_type_stmt, 'arg') and resolved_type_stmt.arg == 'instance-identifier':
            self.graph.add((uri, RDF.type, OWL.ObjectProperty))
            self.graph.add((uri, self.ex.isInstanceIdentifier, Literal(True, datatype=XSD.boolean)))
            range_uri = XSD.anyURI

        elif is_enum_typedef:
            typedef_mod = self._get_typedef_module(typedef_stmt) if typedef_stmt else target_mod
            range_uri = self.ex[f"types/{typedef_mod}/{type_name_clean}"]
            self.graph.add((uri, RDF.type, OWL.ObjectProperty))
            self.graph.add((uri, RDFS.range, range_uri))
        else:
            range_uri = self.type_resolver.resolve_type(type_stmt, self.current_module_name, self._get_target_module_from_prefix)
            self.graph.add((uri, RDF.type, OWL.DatatypeProperty))
            self.graph.add((uri, RDFS.range, range_uri))

        self.graph.add((uri, RDFS.label, Literal(name)))
        if parent_uri: self.graph.add((uri, RDFS.domain, parent_uri))

        # 2. Metadata Extraction
        is_mandatory = False
        is_config = True
        status_val = None
        min_elements = 0
        max_elements = 0
        
        if hasattr(stmt, 'substmts'):
            for sub in stmt.substmts:
                if hasattr(sub, 'keyword'):
                    if sub.keyword == 'description' and hasattr(sub, 'arg'):
                        self.graph.add((uri, RDFS.comment, Literal(sub.arg)))
                    elif sub.keyword == 'units' and hasattr(sub, 'arg'):
                        self.graph.add((uri, self.ex.unit, Literal(sub.arg)))
                    elif sub.keyword == 'mandatory' and getattr(sub, 'arg', '') == 'true':
                        is_mandatory = True
                    elif sub.keyword == 'config' and getattr(sub, 'arg', '') == 'false':
                        is_config = False
                    elif sub.keyword == 'status' and hasattr(sub, 'arg'):
                        status_val = sub.arg
                    elif sub.keyword == 'min-elements':
                        try: min_elements = int(sub.arg)
                        except: pass
                    elif sub.keyword == 'max-elements':
                        try: 
                            if sub.arg != 'unbounded': max_elements = int(sub.arg)
                        except: pass

        if not is_config:
            self.graph.add((uri, self.ex.isStateData, Literal(True, datatype=XSD.boolean)))
            
        if status_val in ('deprecated', 'obsolete'):
            self.graph.add((uri, OWL.deprecated, Literal(True, datatype=XSD.boolean)))

        # 3. Pure SHACL Property Shape
        if parent_uri and not self.leafref_resolver.is_leafref(resolved_type_stmt):
            shape_uri = self.ex[f"shapes/{name}"]
            self.shacl_graph.add((shape_uri, RDF.type, SH.PropertyShape))
            self.shacl_graph.add((shape_uri, SH.path, uri))
            
            # SAFE Datatype, Class Range, or UNION mapped to SHACL
            if isinstance(range_uri, tuple) and range_uri[0] == 'union':
                union_members = range_uri[1]
                if union_members:
                    or_list_node = BNode()
                    self.shacl_graph.add((shape_uri, SH['or'], or_list_node))
                    
                    # Convert to SHACL RDF List
                    current_node = or_list_node
                    for i, (m_type, m_uri) in enumerate(union_members):
                        member_shape = BNode()
                        if m_type == 'datatype':
                            self.shacl_graph.add((member_shape, SH.datatype, m_uri))
                        elif m_type == 'literal':
                            self.shacl_graph.add((member_shape, SH.nodeKind, SH.Literal))
                        else:
                            self.shacl_graph.add((member_shape, SH['class'], m_uri))
                            self.shacl_graph.add((member_shape, SH.nodeKind, SH.IRI))
                        
                        self.shacl_graph.add((current_node, RDF.first, member_shape))
                        if i < len(union_members) - 1:
                            next_node = BNode()
                            self.shacl_graph.add((current_node, RDF.rest, next_node))
                            current_node = next_node
                        else:
                            self.shacl_graph.add((current_node, RDF.rest, RDF.nil))

            elif range_uri:
                if "http://www.w3.org/2001/XMLSchema#" in str(range_uri):
                    self.shacl_graph.add((shape_uri, SH.datatype, range_uri))
                elif range_uri == RDFS.Literal:
                    self.shacl_graph.add((shape_uri, SH.nodeKind, SH.Literal))
                else:
                    self.shacl_graph.add((shape_uri, SH['class'], range_uri))
                    self.shacl_graph.add((shape_uri, SH.nodeKind, SH.IRI))
            
            # Apply Cardinality
            if is_mandatory or min_elements > 0:
                self.shacl_graph.add((shape_uri, SH.minCount, Literal(max(1, min_elements))))
            if not is_leaf_list:
                self.shacl_graph.add((shape_uri, SH.maxCount, Literal(1)))
            elif max_elements > 0:
                self.shacl_graph.add((shape_uri, SH.maxCount, Literal(max_elements)))

            # Apply String/Numeric Constraints
            if resolved_type_stmt:
                extractor = YANGConstraintExtractor()
                constraints = extractor.extract_constraints(resolved_type_stmt)
                if 'range' in constraints and isinstance(constraints['range'], dict):
                    if 'min' in constraints['range']: self.shacl_graph.add((shape_uri, SH.minInclusive, Literal(constraints['range']['min'])))
                    if 'max' in constraints['range']: self.shacl_graph.add((shape_uri, SH.maxInclusive, Literal(constraints['range']['max'])))
                if 'length' in constraints and isinstance(constraints['length'], dict):
                    if 'minLength' in constraints['length']: self.shacl_graph.add((shape_uri, SH.minLength, Literal(constraints['length']['minLength'])))
                    if 'maxLength' in constraints['length']: self.shacl_graph.add((shape_uri, SH.maxLength, Literal(constraints['length']['maxLength'])))
                if 'patterns' in constraints and isinstance(constraints['patterns'], list):
                    for pattern in constraints['patterns']:
                        self.shacl_graph.add((shape_uri, SH.pattern, Literal(pattern)))
            
            parent_node_shape = self._ensure_node_shape(parent_uri)
            self.shacl_graph.add((parent_node_shape, SH.property, shape_uri))

        self._process_xpath_constraints(stmt, uri)


    def _process_leaf_list(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None, parent_prov: str = "") -> None:
        self._process_leaf(stmt, path, parent_uri, parent_prov, is_leaf_list=True)

    def _process_augment(self, stmt: Any) -> None:
        if not hasattr(stmt, 'arg'):
            return

        raw_path = stmt.arg
        parts = [p for p in raw_path.split('/') if p]
        target_module_name = self.current_module_name
        
        if parts:
            first_part = parts[0]
            if ':' in first_part:
                prefix = first_part.split(':')[0]
                target_module_name = self._get_target_module_from_prefix(stmt, prefix)

        clean_path = re.sub(r'[a-zA-Z0-9_-]+:', '', raw_path)
        clean_path = re.sub(r'/+', '/', clean_path)
        
        target_path = clean_path
        if target_module_name:
            if not clean_path.startswith('/' + target_module_name + '/'):
                target_path = '/' + target_module_name + clean_path
                
        target_path = re.sub(r'/+', '/', target_path)

        matched_path = None
        if target_path not in self.class_paths:
            for registered_path in self.class_paths:
                if registered_path == clean_path or registered_path.endswith(clean_path):
                    matched_path = registered_path
                    break
                    
        if matched_path:
            target_path = matched_path

        target_uri = self.ex[target_path.lstrip('/')]

        if target_path not in self.class_paths:
            self.graph.add((target_uri, RDF.type, OWL.Class))
            self.class_paths[target_path] = target_uri
        
        parent_prov = self.prov_paths.get(target_path, "")

        if hasattr(stmt, 'substmts'):
            for sub in stmt.substmts:
                if not hasattr(sub, 'keyword'): continue
                
                keyword = sub.keyword
                child_name = sub.arg if hasattr(sub, 'arg') else "unknown"
                child_path = f"{target_path}/{child_name}"
                
                if keyword == 'leaf':
                    self._process_leaf(sub, child_path, target_uri, parent_prov)
                elif keyword == 'uses':
                    self._process_uses_in_container(sub, target_path, target_uri)
                elif keyword in ('container', 'list'):
                    self._process_container(sub, child_path, target_uri, parent_prov)
                elif keyword == 'choice':
                    self._process_choice(sub, target_path, target_uri, parent_prov)

    def _process_rpc(self, stmt: Any) -> None:
        if not hasattr(stmt, 'arg'): return
        name = stmt.arg
        uri = self.ex[f"rpc/{self.current_module_name}/{name}"]
        self.graph.add((uri, RDF.type, OWL.Class))
        self.graph.add((uri, RDFS.label, Literal(name)))
        self.rpc_classes[name] = uri

    def _process_notification(self, stmt: Any) -> None:
        if not hasattr(stmt, 'arg'): return
        name = stmt.arg
        uri = self.ex[f"notification/{self.current_module_name}/{name}"]
        self.graph.add((uri, RDF.type, OWL.Class))
        self.graph.add((uri, RDFS.label, Literal(name)))

    def _process_uses_in_container(self, uses_stmt: Any, target_path: str, target_uri: URIRef) -> None:
        """
        Processes 'uses' statements by creating subClassOf links to groupings 
        and mapping 'refine' constraints to SHACL shapes on the target class.
        """
        if not hasattr(uses_stmt, 'arg'): return
        grouping_name = uses_stmt.arg
        
        # 1. Maintain semantic link to the grouping class
        grouping_uri = self._get_grouping_uri(uses_stmt, grouping_name)
        self.graph.add((target_uri, RDFS.subClassOf, grouping_uri))
        
        if not self.grouping_resolver: return
        refine_resolver = RefineResolver()
        refines = refine_resolver.extract_refines(uses_stmt)
        
        target_mod = self.current_module_name
        if ':' in grouping_name:
            prefix = grouping_name.split(':')[0]
            target_mod = self._get_target_module_from_prefix(uses_stmt, prefix)
            
        grouping_children = self.grouping_resolver.get_grouping_children(grouping_name, target_mod)

        if not grouping_children:
            log.warning(f" Could not resolve grouping: {grouping_name}")
            return

        # Ensure the target class has a NodeShape to hold refinements
        self._ensure_node_shape(target_uri)

        for child_name, child_stmt, keyword in grouping_children:
            if keyword == 'uses':
                self._process_uses_in_container(child_stmt, target_path, target_uri)
                continue

            child_path = f"{target_path}/{child_name}"
            child_uri = self.ex[child_path.lstrip('/')]
            refine_props = refines.get(child_name, {})
            parent_prov = self.prov_paths.get(target_path, "")

            # Determine the correct valid SHACL property URI for the constraint
            if keyword in ('container', 'list'):
                clean_child_name = child_name[5:] if child_name.startswith('case-') else child_name
                prop_name = 'has' + ''.join(word.capitalize() for word in clean_child_name.split('-'))
                property_uri = self.ex[f"{target_path.lstrip('/')}/{prop_name}"]
            else:
                property_uri = child_uri

            # A. Process the child node normally to ensure its URI and type exist
            if keyword in ('leaf', 'leaf-list'):
                self._process_leaf(child_stmt, child_path, target_uri, parent_prov, is_leaf_list=(keyword == 'leaf-list'))
            elif keyword in ('container', 'list'):
                if keyword == 'container':
                    self._process_container(child_stmt, child_path, target_uri, parent_prov)
                else:
                    self._process_list(child_stmt, child_path, target_uri, parent_prov)
            elif keyword == 'choice':
                self._process_choice(child_stmt, target_path, target_uri, parent_prov)

            # B. Apply SHACL Refinements to the correct Property Path
            min_c = None
            max_c = None

            if refine_props.get('mandatory') == 'true':
                min_c = 1
            elif refine_props.get('min-elements'):
                try:
                    min_c = int(refine_props.get('min-elements'))
                except ValueError: pass
                
            if refine_props.get('max-elements'):
                try:
                    val = refine_props.get('max-elements')
                    if val != 'unbounded':
                        max_c = int(val)
                except ValueError: pass

            # If any refinements exist, add a specific PropertyShape to the target NodeShape
            if min_c is not None or max_c is not None:
                self._add_property_shape(
                    target_uri,
                    property_uri,
                    min_count=min_c,
                    max_count=max_c,
                    message=f"Refined constraint for {child_name} in {target_uri.split('/')[-1]}"
                )
                
        self.uses_count += 1

    def _process_deferred_augmentations(self) -> None:
        for module_name, stmt in self.deferred_augments:
            self.current_module_name = module_name
            self._process_augment(stmt)

    def _process_containers_for_properties(self) -> None:
        """
        Pass 8: Generates OWL ObjectProperties for all container and list relationships 
        and attaches strict SHACL validation to enforce correct child targeting.
        """
        for path, uri in list(self.class_paths.items()):
            child_paths = [p for p in self.class_paths.keys() if p.startswith(path + '/') and p.count('/') == path.count('/') + 1]
            for child_path in child_paths:
                child_name = child_path.split('/')[-1]
                
                #if child_name.startswith('case-'):
                #    continue
                    
                prop_name = 'has' + ''.join(word.capitalize() for word in child_name.split('-'))
                prop_uri = self.ex[path.lstrip('/') + '/' + prop_name]
                child_uri = self.class_paths[child_path]

                parent_prov = self.prov_paths.get(path, "")
                if parent_prov:
                    prov_string = f"{parent_prov}/{prop_name}"
                    self.graph.add((prop_uri, PROV.wasDerivedFrom, Literal(prov_string)))

                self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
                self.graph.add((prop_uri, RDFS.label, Literal(prop_name)))
                self.graph.add((prop_uri, RDFS.domain, uri))
                self.graph.add((prop_uri, RDFS.range, child_uri))
                self.graph.add((prop_uri, RDFS.comment, Literal("Containment relation")))
                
                # Add Strict SHACL Containment Property Shape
                parent_shape = self._ensure_node_shape(uri)
                ps = BNode()
                self.shacl_graph.add((parent_shape, SH.property, ps))
                self.shacl_graph.add((ps, RDF.type, SH.PropertyShape))
                self.shacl_graph.add((ps, SH.path, prop_uri))
                self.shacl_graph.add((ps, SH['class'], child_uri))
                self.shacl_graph.add((ps, SH.nodeKind, SH.IRI))

    def _process_imported_module_bases(self) -> None:
        for module_name, module in self.resolver.modules.items():
            self.current_module_name = extract_module_name(module_name)
            if not hasattr(module, 'substmts'): continue
            for stmt in module.substmts:
                if not hasattr(stmt, 'keyword'): continue
                keyword = stmt.keyword
                if keyword == 'container':
                    normalized_path = self._normalize_path(f"/{stmt.arg}")
                    if normalized_path not in self.class_paths:
                        uri = self.ex[normalized_path.lstrip('/')]
                        prov_path = self._get_prov_segment(stmt)
                        self.graph.add((uri, PROV.wasDerivedFrom, Literal(prov_path)))
                        self.graph.add((uri, RDF.type, OWL.Class))
                        self.class_paths[normalized_path] = uri
                elif keyword == 'list':
                    normalized_path = self._normalize_path(f"/{stmt.arg}")
                    if normalized_path not in self.class_paths:
                        uri = self.ex[normalized_path.lstrip('/')]
                        self.graph.add((uri, RDF.type, OWL.Class))
                        self.class_paths[normalized_path] = uri

    def _create_shacl_typedef_shapes(self) -> None:
        constraint_extractor = YANGConstraintExtractor()
        for module_name, module in self.resolver.modules.items():
            self.current_module_name = extract_module_name(module_name)
            if not hasattr(module, 'substmts'): continue
            for stmt in module.substmts:
                if not hasattr(stmt, 'keyword'): continue
                if stmt.keyword == 'typedef' and hasattr(stmt, 'arg'):
                    typedef_name = stmt.arg
                    is_enum = False
                    if hasattr(stmt, 'substmts'):
                        for sub in stmt.substmts:
                            if sub.keyword == 'type' and self._is_enumeration_type(sub):
                                is_enum = True
                    if is_enum: continue
                    
                    is_union_typedef = False
                    if hasattr(stmt, 'substmts'):
                        for sub in stmt.substmts:
                            if sub.keyword == 'type' and hasattr(sub, 'arg') and sub.arg == 'union':
                                is_union_typedef = True
                    if is_union_typedef: continue

                    constraints = constraint_extractor.extract_constraints(stmt)
                    shape_uri = self.ex[f"typedef/{self.current_module_name}/{typedef_name}"]
                    self.shacl_graph.add((shape_uri, RDF.type, SH.NodeShape))
                    self.shacl_graph.add((shape_uri, RDFS.label, Literal(typedef_name)))
                    
                    base_type = XSD.string
                    if hasattr(stmt, 'substmts'):
                        for sub in stmt.substmts:
                            if sub.keyword == 'type':
                                base_type = self.type_resolver.resolve_type(sub, self.current_module_name, self._get_target_module_from_prefix)
                    
                    # SAFE SHACL DATATYPE BINDING
                    if "http://www.w3.org/2001/XMLSchema#" in str(base_type):
                        self.shacl_graph.add((shape_uri, SH.datatype, base_type))
                    elif base_type == RDFS.Literal:
                        self.shacl_graph.add((shape_uri, SH.nodeKind, SH.Literal))
                    else:
                        self.shacl_graph.add((shape_uri, SH['class'], base_type))
                        self.shacl_graph.add((shape_uri, SH.nodeKind, SH.IRI))

                    if constraints:
                        if 'patterns' in constraints:
                            for pattern in constraints['patterns']:
                                self.shacl_graph.add((shape_uri, SH.pattern, Literal(pattern)))
                        if 'length' in constraints:
                            l = constraints['length']
                            if 'minLength' in l: self.shacl_graph.add((shape_uri, SH.minLength, Literal(l['minLength'])))
                            if 'maxLength' in l: self.shacl_graph.add((shape_uri, SH.maxLength, Literal(l['maxLength'])))
                        if 'range' in constraints:
                            r = constraints['range']
                            if 'min' in r: self.shacl_graph.add((shape_uri, SH.minInclusive, Literal(r['min'])))
                            if 'max' in r: self.shacl_graph.add((shape_uri, SH.maxInclusive, Literal(r['max'])))
                    self.typedef_restrictions[typedef_name] = shape_uri

    def _process_enumerations(self) -> None:
        for module_name, module in self.resolver.modules.items():
            self.current_module_name = extract_module_name(module_name)
            if not hasattr(module, 'substmts'): continue
            for stmt in module.substmts:
                if not hasattr(stmt, 'keyword'): continue
                if stmt.keyword == 'typedef' and hasattr(stmt, 'arg'):
                    typedef_name = stmt.arg
                    if hasattr(stmt, 'substmts'):
                        for sub in stmt.substmts:
                            if hasattr(sub, 'keyword') and sub.keyword == 'type':
                                if self._is_enumeration_type(sub):
                                    self._create_enumeration_class(typedef_name, sub, self.current_module_name)
                                break

    def _create_enumeration_class(self, enum_type_name: str, type_stmt: Any, module_name: str) -> int:
        enum_count = 0
        enum_type_uri = self.ex[f"types/{module_name}/{enum_type_name}"]
        self.graph.add((enum_type_uri, RDF.type, OWL.Class))
        self.graph.add((enum_type_uri, RDFS.label, Literal(enum_type_name)))

        if hasattr(type_stmt, 'substmts'):
            for enum_sub in type_stmt.substmts:
                if hasattr(enum_sub, 'keyword') and enum_sub.keyword == 'enum':
                    enum_val = enum_sub.arg if hasattr(enum_sub, 'arg') else ''
                    if enum_val:
                        individual_uri = self.ex[f"types/{module_name}/{enum_type_name}/{enum_val}"]
                        self.graph.add((individual_uri, RDF.type, OWL.NamedIndividual))
                        self.graph.add((individual_uri, RDF.type, enum_type_uri))
                        self.graph.add((individual_uri, RDFS.label, Literal(enum_val)))
                        enum_count += 1
                        if hasattr(enum_sub, 'substmts'):
                            for enum_detail in enum_sub.substmts:
                                if hasattr(enum_detail, 'keyword') and enum_detail.keyword == 'description':
                                    if hasattr(enum_detail, 'arg'):
                                        self.graph.add((individual_uri, RDFS.comment, Literal(enum_detail.arg)))
        self.enumeration_count += enum_count
        return enum_count

    def _resolve_pending_leafrefs(self) -> None:
        for item in self.pending_leafrefs:
            if len(item) == 5:
                uri, leafref_type, context_path, parent_uri, name = item
                desc, full_prov, is_mand, mine, maxe, is_leaf_list = None, None, False, 0, 0, False
            else:
                uri, leafref_type, context_path, parent_uri, name, desc, full_prov, is_mand, mine, maxe, is_leaf_list = item
            
            self.graph.add((uri, RDF.type, OWL.ObjectProperty))
            self.graph.add((uri, RDFS.label, Literal(name)))
            
            if desc: self.graph.add((uri, RDFS.comment, Literal(desc)))
            if full_prov: self.graph.add((uri, PROV.wasDerivedFrom, Literal(full_prov)))
            if parent_uri: self.graph.add((uri, RDFS.domain, parent_uri))
            
            resolution_result = self.leafref_resolver.resolve_leafref_target(leafref_type, context_path)
            if resolution_result:
                matched_path, target_class_uri, xpath_path = resolution_result
                
                self.graph.add((uri, self.ex.xpathPath, Literal(xpath_path)))
                
                if target_class_uri: 
                    self.graph.add((uri, RDFS.range, target_class_uri))
                    
                    if parent_uri:
                        min_c = max(1, mine) if (is_mand or mine > 0) else None
                        max_c = 1 if not is_leaf_list else (maxe if maxe > 0 else None)
                        
                        self._add_property_shape(
                            parent_uri,
                            uri,
                            value_class=target_class_uri,
                            min_count=min_c,
                            max_count=max_c,
                            message=f"Leafref {name} must target an instance of {target_class_uri.split('/')[-1]}"
                        )
                    
                self.leafref_resolved_count += 1
            else:
                self.leafref_unresolved_count += 1
    
    def _bind_ontology_prefixes(self) -> None:
        """Bind a curated, human-readable set of prefixes derived from the
        logical structure of the ontology URI space. Groups related sub-paths
        under a single short prefix rather than auto-deriving one per unique
        namespace path."""
        base = self.base_uri.rstrip('/')

        # Curated prefix map: prefix -> namespace URI
        # Ordered from most-specific to least-specific so rdflib picks the
        # longest matching namespace when serialising.
        curated = [
            # ── Cable sub-containers ──────────────────────────────────────
            ("cabOptical",  f"{base}/ietf-network-inventory/network-inventory/cable/optical-cable/"),
            ("cabAEnd",     f"{base}/ietf-network-inventory/network-inventory/cable/a-end/"),
            ("cabZEnd",     f"{base}/ietf-network-inventory/network-inventory/cable/z-end/"),
            ("cabChAEnd",   f"{base}/ietf-network-inventory/network-inventory/cable/child-cable/a-end/"),
            ("cabChZEnd",   f"{base}/ietf-network-inventory/network-inventory/cable/child-cable/z-end/"),
            # ── NWI sub-containers ────────────────────────────────────────
            ("nwiNEs",      f"{base}/ietf-network-inventory/network-inventory/network-elements/"),
            ("nwiCompSwRev",f"{base}/ietf-network-inventory/network-inventory/network-elements/network-element/components/component/software-rev/"),
            ("nwiCompPatch",f"{base}/ietf-network-inventory/network-inventory/network-elements/network-element/components/component/software-rev/patch/"),
            ("nwiSwRevPatch",f"{base}/ietf-network-inventory/network-inventory/network-elements/network-element/software-rev/patch/"),
            ("nwiLocs",     f"{base}/ietf-network-inventory/network-inventory/locations/"),
            ("nwiRacks",    f"{base}/ietf-network-inventory/network-inventory/locations/racks/"),
            ("nwiChassis",  f"{base}/ietf-network-inventory/network-inventory/locations/racks/rack/contained-chassis/"),
            ("nwiRefFrame", f"{base}/ietf-network-inventory/network-inventory/locations/location/geo-location/reference-frame/"),
            # ── Network sub-containers ────────────────────────────────────
            ("netNetworks", f"{base}/ietf-network/networks/"),
            ("netSuppNet",  f"{base}/ietf-network/networks/network/supporting-network/"),
            ("netSuppNode", f"{base}/ietf-network/networks/network/node/supporting-node/"),
            ("netSuppLink", f"{base}/ietf-network/networks/network/link/supporting-link/"),
            ("netSuppTP",   f"{base}/ietf-network/networks/network/node/termination-point/supporting-termination-point/"),
            ("netBrkChan",  f"{base}/ietf-network/networks/network/node/termination-point/inventory-mapping-attributes/port-breakout/breakout-channel/"),
            ("netPortBrk",  f"{base}/ietf-network/networks/network/node/termination-point/inventory-mapping-attributes/port-breakout/"),
            # ── Hardware sub-containers ───────────────────────────────────
            ("hwSensorData",f"{base}/ietf-hardware/hardware/component/sensor-data/"),
            ("hwState",     f"{base}/ietf-hardware/hardware/component/state/"),
            # ── Power sub-containers ──────────────────────────────────────
            ("paePwrEntry", f"{base}/ietf-power-and-energy/energy-objects/power-entry/"),
            # ── Typedef/enumeration sub-namespaces ───────────────────────
            ("hwAdminSt",   f"{base}/types/ietf-hardware/admin-state/"),
            ("hwOperSt",    f"{base}/types/ietf-hardware/oper-state/"),
            ("hwStandbySt", f"{base}/types/ietf-hardware/standby-state/"),
            ("hwUsageSt",   f"{base}/types/ietf-hardware/usage-state/"),
            ("hwSensorSt",  f"{base}/types/ietf-hardware/sensor-status/"),
            ("hwSensorVSc", f"{base}/types/ietf-hardware/sensor-value-scale/"),
            ("hwSensorVTy", f"{base}/types/ietf-hardware/sensor-value-type/"),
            ("inetIpVer",   f"{base}/types/ietf-inet-types/ip-version/"),
            # ── SHACL shapes namespace ────────────────────────────────────
            ("shapes",      f"{base}/shapes/"),
            # ── Typedef shapes ────────────────────────────────────────────
            ("typedef",     f"{base}/typedef/"),

            # ── Top-level module roots (catch-all for module-level URIs) ──
            ("hwRoot",      f"{base}/ietf-hardware/"),
            ("nwiRoot",     f"{base}/ietf-network-inventory/"),
            ("netRoot",     f"{base}/ietf-network/"),
            ("paeRoot",     f"{base}/ietf-power-and-energy/"),
            ("ianaHwRoot",  f"{base}/iana-hardware/"),
            ("nwiPassRoot", f"{base}/ietf-nwi-passive-inventory/"),
            ("niLocRoot",   f"{base}/ietf-ni-location/"),
            # ── Identity top-level (already have sub-paths above) ─────────
            ("idRoot",      f"{base}/identity/"),
            # ── Notifications ─────────────────────────────────────────────
            ("hwNotif",     f"{base}/notification/ietf-hardware/"),
            # ── Identity hierarchies ──────────────────────────────────────
            ("ianaHw",      f"{base}/identity/iana-hardware/"),
            ("nwiPassId",   f"{base}/identity/ietf-nwi-passive-inventory/"),
            ("nwiInvId",    f"{base}/identity/ietf-network-inventory/"),
            ("hwId",        f"{base}/identity/ietf-hardware/"),
            ("paeId",       f"{base}/identity/ietf-power-and-energy/"),
            ("yangId",      f"{base}/identity/ietf-yang-types/"),
            # ── Typedef / enumeration types ───────────────────────────────
            ("hwTypes",     f"{base}/types/ietf-hardware/"),
            ("inetTypes",   f"{base}/types/ietf-inet-types/"),
            # ── Grouping abstract classes ─────────────────────────────────
            ("grpGeo",      f"{base}/grouping/ietf-geo-location/"),
            ("grpNwi",      f"{base}/grouping/ietf-network-inventory/"),
            ("grpNwiTop",   f"{base}/grouping/ietf-network-inventory-topology/"),
            ("grpNwiPass",  f"{base}/grouping/ietf-nwi-passive-inventory/"),
            ("grpNet",      f"{base}/grouping/ietf-network/"),
            ("grpNetTop",   f"{base}/grouping/ietf-network-topology/"),
            ("grpNiLoc",    f"{base}/grouping/ietf-ni-location/"),
            # ── Hardware ──────────────────────────────────────────────────
            ("hw",          f"{base}/ietf-hardware/hardware/"),
            ("hwComp",      f"{base}/ietf-hardware/hardware/component/"),
            # ── Network Inventory ─────────────────────────────────────────
            ("nwi",         f"{base}/ietf-network-inventory/network-inventory/"),
            ("nwiNE",       f"{base}/ietf-network-inventory/network-inventory/network-elements/network-element/"),
            ("nwiComps",    f"{base}/ietf-network-inventory/network-inventory/network-elements/network-element/components/"),
            ("nwiComp",     f"{base}/ietf-network-inventory/network-inventory/network-elements/network-element/components/component/"),
            ("nwiSwRev",    f"{base}/ietf-network-inventory/network-inventory/network-elements/network-element/software-rev/"),
            ("nwiLoc",      f"{base}/ietf-network-inventory/network-inventory/locations/location/"),
            ("nwiGeo",      f"{base}/ietf-network-inventory/network-inventory/locations/location/geo-location/"),
            ("nwiRack",     f"{base}/ietf-network-inventory/network-inventory/locations/racks/rack/"),
            ("cab",         f"{base}/ietf-network-inventory/network-inventory/cable/"),
            ("cabChild",    f"{base}/ietf-network-inventory/network-inventory/cable/child-cable/"),
            ("cabOptical",  f"{base}/ietf-network-inventory/network-inventory/cable/optical-cable/"),
            ("pdev",        f"{base}/ietf-network-inventory/network-inventory/passive-device/"),
            ("pdevPort",    f"{base}/ietf-network-inventory/network-inventory/passive-device/passive-port/"),
            # ── Network topology ──────────────────────────────────────────
            ("net",         f"{base}/ietf-network/networks/network/"),
            ("netNode",     f"{base}/ietf-network/networks/network/node/"),
            ("netLink",     f"{base}/ietf-network/networks/network/link/"),
            ("netTP",       f"{base}/ietf-network/networks/network/node/termination-point/"),
            ("netInvMap",   f"{base}/ietf-network/networks/network/node/termination-point/inventory-mapping-attributes/"),
            # Layer 2 Topology (RFC 8944)
            ("l2t",       f"{base}/ietf-l2-topology/"),
            ("l2t-s",     f"{base}/ietf-l2-topology-state/"),
            # Layer 2 Groupings
            ("grpL2t",    f"{base}/grouping/ietf-l2-topology/"),
            ("grpL2t-s",  f"{base}/grouping/ietf-l2-topology-state/"),
            # Layer 2 Types & Identities (for VLANs, MACs, link types, etc.)
            ("l2tTypes",  f"{base}/types/ietf-l2-topology/"),
            # Deep Augmentation Targets (RFC 8345 base paths)
            ("nw-tp",   f"{base}/ietf-network/networks/network/node/termination-point/"),
            ("nw-s-tp", f"{base}/ietf-network-state/networks/network/node/termination-point/"),
            ("nw-node",   f"{base}/ietf-network/networks/network/node/"),
            ("nw-s-node", f"{base}/ietf-network-state/networks/network/node/"),
            ("nw-link",   f"{base}/ietf-network-topology/networks/network/link/"),
            ("nw-s-link", f"{base}/ietf-network-topology-state/networks/network/link/"),
            # Layer 2 Specific Identities/Types
            ("l2tEvent",  f"{base}/types/ietf-l2-topology/l2-network-event-type/"),
            # Network Types Container
            ("nw-types", f"{base}/ietf-network/networks/network/network-types/"),
            # Cross-Module Augmentations: L2 Topology Attributes
            ("l2t-s-attr", f"{base}/ietf-l2-topology-state/ietf-network-state/networks/network/node/termination-point/l2-termination-point-attributes/"),
            ("l2t-attr",   f"{base}/ietf-l2-topology/ietf-network-state/networks/network/node/termination-point/l2-termination-point-attributes/"),
            # Cross-Module Augmentations: Inventory Mapping
            ("nwit-s-inv", f"{base}/ietf-network-inventory-topology/ietf-network-state/networks/network/node/termination-point/inventory-mapping-attributes/"),
            # Base Network Links
            ("nws-link",   f"{base}/ietf-network-state/networks/network/link/"),
            # Deep Augmentations: Inventory Port Breakouts (Config)
            ("nwit-bo",     f"{base}/ietf-network-inventory-topology/ietf-network/networks/network/node/termination-point/inventory-mapping-attributes/port-breakout/"),
            ("nwit-chan",   f"{base}/ietf-network-inventory-topology/ietf-network/networks/network/node/termination-point/inventory-mapping-attributes/port-breakout/breakout-channel/"),
            # Deep Augmentations: Inventory Port Breakouts (State)
            ("nwit-s-bo",   f"{base}/ietf-network-inventory-topology/ietf-network-state/networks/network/node/termination-point/inventory-mapping-attributes/port-breakout/"),
            ("nwit-s-chan", f"{base}/ietf-network-inventory-topology/ietf-network-state/networks/network/node/termination-point/inventory-mapping-attributes/port-breakout/breakout-channel/"),
            # Network Level Containers (Catches 'link', 'node', etc.)
            ("nw-net",   f"{base}/ietf-network/networks/network/"),
            ("nws-net",  f"{base}/ietf-network-state/networks/network/"),
            # YANG Identity / Base Type Namespaces
            ("l2tId",    f"{base}/identity/ietf-l2-topology/"),
            ("ianaIfId", f"{base}/identity/iana-if-type/"),            
            # Layer 2 Notifications / Events
            ("l2tNotif",  f"{base}/notification/ietf-l2-topology/"),
            ("l2tSNotif", f"{base}/notification/ietf-l2-topology-state/"),
            # Layer 2 Specific Identities/Types
            ("l2tDuplex", f"{base}/types/ietf-l2-topology/duplex-mode/"),
            # ── Power and Energy ──────────────────────────────────────────
            ("pae",         f"{base}/ietf-power-and-energy/energy-objects/"),
        ]

        bound = 0
        for prefix, ns_uri in curated:
            ns = Namespace(ns_uri)
            for g in (self.graph, self.shacl_graph):
                try:
                    g.bind(prefix, ns, override=False)
                    bound += 1
                except Exception:
                    pass

        log.info(f"  Bound {bound // 2} curated ontology namespace prefixes")



class YANGToHTML:
    """Generates an HTML Tree view of the parsed YANG modules."""
    def __init__(self, modules: Dict[str, Any]):
        self.modules = modules
        self.css = """
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f9f9f9; padding: 20px; }
            .module-box { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; font-size: 1.2em; }
            ul { list-style-type: none; padding-left: 20px; border-left: 1px solid #eee; }
            li { margin: 5px 0; }
            .keyword { font-weight: bold; color: #d35400; font-size: 0.9em; text-transform: uppercase; margin-right: 5px; }
            .arg { font-weight: 600; color: #2980b9; }
            .type { color: #7f8c8d; font-style: italic; font-size: 0.85em; margin-left: 10px; }
            .desc { color: #555; font-size: 0.85em; display: block; margin-left: 20px; border-left: 2px solid #eee; padding-left: 5px; margin-top: 2px; }
            .container > .keyword { color: #16a085; }
            .list > .keyword { color: #f39c12; }
            .leaf > .keyword { color: #27ae60; }
            .augment > .keyword { color: #8e44ad; }
            .grouping > .keyword { color: #2c3e50; }
        </style>
        """

    def generate(self, output_filename: str):
        html = ["<!DOCTYPE html><html><head><title>YANG Parse Tree</title>"]
        html.append(self.css)
        html.append("</head><body><h1>YANG Parse Tree</h1>")
        for name in sorted(self.modules.keys()):
            module = self.modules[name]
            html.append(f"<div class='module-box'><h2>Module: {name}</h2><ul>")
            html.append(self._render_stmts(module.substmts))
            html.append("</ul></div>")
        html.append("</body></html>")
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("".join(html))
        log.info(f"✓ HTML Tree saved to: {output_filename}")

    def _render_stmts(self, substmts):
        output = []
        if not substmts: return ""
        for stmt in substmts:
            keyword = stmt.keyword
            arg = stmt.arg if hasattr(stmt, 'arg') else ""
            if keyword in ['description', 'reference', 'organization', 'contact', 'revision']:
                continue
            css_class = keyword if keyword in ['container', 'list', 'leaf', 'augment', 'grouping'] else 'other'
            type_str = ""
            type_stmt = stmt.search_one('type')
            if type_stmt and hasattr(type_stmt, 'arg'):
                type_str = f'<span class="type">({type_stmt.arg})</span>'
            desc_str = ""
            desc_stmt = stmt.search_one('description')
            if desc_stmt and hasattr(desc_stmt, 'arg'):
                brief = desc_stmt.arg.split('\n')[0][:60]
                desc_str = f'<span class="desc">"{brief}..."</span>'
            output.append(f"<li class='{css_class}'>")
            output.append(f"<span class='keyword'>{keyword}</span>")
            output.append(f"<span class='arg'>{arg}</span>")
            output.append(type_str)
            output.append(desc_str)
            if hasattr(stmt, 'substmts') and stmt.substmts:
                if keyword not in ['leaf', 'leaf-list', 'typedef']:
                    children_html = self._render_stmts(stmt.substmts)
                    if children_html:
                        output.append(f"<ul>{children_html}</ul>")
            output.append("</li>")
        return "".join(output)


class ABoxConnectivityEnricher:
    """
    Post-processing pass that reads an existing ABox TTL file and materialises
    RDF-star upstream connectivity triples for every cable instance.

    For each cable where both a Z-end device and an A-end device can be resolved:
      1. Adds a base triple:  <z_device> ex:hasUpstreamDevice <a_device>
      2. Adds its inverse:    <a_device> ex:hasDownstreamDevice <z_device>
      3. Appends an RDF-star annotation block (Turtle* syntax):
             <<<z_device> ex:hasUpstreamDevice <a_device>>>
                 ex:viaCable  <cable> ;
                 ex:cableRole <role> .

    The output is a valid Turtle* file (.ttls) that Stardog loads natively,
    enabling the single-line traversal query:
        ?start ex:hasUpstreamDevice+ ?target .

    Usage (standalone):
        python yang4owl.py --abox-enrich data.ttl --abox-out data_enriched.ttls

    Usage (programmatic):
        enricher = ABoxConnectivityEnricher("data.ttl", "http://www.huawei.com/ontology")
        enricher.enrich("data_enriched.ttls")
    """

    def __init__(self, abox_file: str, base_uri: str = "http://www.huawei.com/ontology"):
        self.abox_file = Path(abox_file)
        self.base_uri  = base_uri.rstrip('/')
        self.ex        = Namespace(self.base_uri + '/')

        _ni_base       = f"{self.base_uri}/ietf-network-inventory/network-inventory/"
        self.CAB_NS    = Namespace(f"{_ni_base}cable/")
        self.NWI_NS    = Namespace(_ni_base)
        self.PASS_ID   = Namespace(f"{self.base_uri}/identity/ietf-nwi-passive-inventory/")

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich(self, output_file: str) -> int:
        """
        Reads self.abox_file, materialises shortcut triples, and writes a
        Turtle* file to output_file.  Returns the count of links materialised.
        """
        log.info(f"\n{'='*60}")
        log.info(f" ABoxConnectivityEnricher  v4.7.30")
        log.info(f" Input : {self.abox_file}")
        log.info(f" Output: {output_file}")
        log.info(f"{'='*60}")

        g = Graph()
        g.parse(str(self.abox_file), format='turtle')
        log.info(f" Loaded {len(g):,} triples from ABox.")

        links = self._materialise_links(g)
        self._write_turtle_star(g, links, output_file)

        log.info(f" ✓ Materialised {len(links):,} upstream device links.")
        log.info(f" ✓ Turtle* written → {output_file}\n")
        return len(links)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _materialise_links(self, g: Graph) -> list:
        """
        Iterates over every nwi:cable individual, resolves Z-end and A-end
        devices, adds base shortcut triples to g, and returns the full list
        of (z_device, a_device, cable, cable_role) tuples.
        """
        ne_ref       = self.ex["ne-ref"]
        device_ref   = self.ex["device-ref"]
        has_a_end    = self.CAB_NS["hasAEnd"]
        has_z_end    = self.CAB_NS["hasZEnd"]
        cable_type   = self.NWI_NS["cable"]
        cable_role_p = self.ex["cable-role"]
        has_upstream = self.ex["hasUpstreamDevice"]
        has_down     = self.ex["hasDownstreamDevice"]

        cables = list(g.subjects(RDF.type, cable_type))
        log.info(f" Found {len(cables):,} cable instances.")

        links = []
        skipped = 0

        for cable in cables:
            z_end_node = g.value(cable, has_z_end)
            a_end_node = g.value(cable, has_a_end)
            if not z_end_node or not a_end_node:
                skipped += 1
                continue

            z_device = g.value(z_end_node, ne_ref) or g.value(z_end_node, device_ref)
            a_device = g.value(a_end_node, ne_ref) or g.value(a_end_node, device_ref)
            if not z_device or not a_device:
                skipped += 1
                continue

            cable_role = g.value(cable, cable_role_p)   # may be None

            # Materialise base shortcut triples into the graph
            g.add((z_device, has_upstream, a_device))
            g.add((a_device, has_down,     z_device))

            links.append((z_device, a_device, cable, cable_role))

        if skipped:
            log.warning(f" {skipped} cable(s) skipped (missing end or device reference).")
        return links

    def _write_turtle_star(self, g: Graph, links: list, output_file: str) -> None:
        """
        Serialises g as Turtle, then appends the RDF-star <<...>> annotation
        blocks for every link.  The combined output is valid Turtle* (RDF-star).
        """
        # ── 1. Serialise base graph (includes the new shortcut triples) ───
        base_ttl = g.serialize(format='turtle')

        # ── 2. Build a compact-URI helper from the graph's bound prefixes ──
        prefix_map: Dict[str, str] = {}
        for pfx, ns in g.namespaces():
            prefix_map[str(ns)] = str(pfx)
        # Ensure ex: is always resolvable
        prefix_map.setdefault(self.base_uri + '/', 'ex')

        def _compact(uri: URIRef) -> str:
            s = str(uri)
            # Longest-prefix match
            best_pfx, best_ns = None, ''
            for ns_str, pfx in prefix_map.items():
                if s.startswith(ns_str) and len(ns_str) > len(best_ns):
                    best_ns, best_pfx = ns_str, pfx
            if best_pfx is not None:
                local = s[len(best_ns):]
                # Only abbreviate if local part is a valid NCName (no slashes etc.)
                if local and '/' not in local and '#' not in local:
                    return f"{best_pfx}:{local}"
            return f"<{s}>"

        # ── 3. Build RDF-star annotation blocks ───────────────────────────
        rdfstar_lines: List[str] = [
            "",
            "# " + "─" * 72,
            "# RDF-star annotations: ex:viaCable and ex:cableRole per upstream hop",
            "# Load this file in Stardog as Turtle* (content-type: text/turtle*)",
            "# " + "─" * 72,
        ]

        for z_dev, a_dev, cable, role in links:
            zc = _compact(z_dev)
            ac = _compact(a_dev)
            cc = _compact(cable)
            rdfstar_lines.append("")
            rdfstar_lines.append(f"<<{zc} ex:hasUpstreamDevice {ac}>>")
            if role:
                rc = _compact(role)
                rdfstar_lines.append(f"    ex:viaCable  {cc} ;")
                rdfstar_lines.append(f"    ex:cableRole {rc} .")
            else:
                rdfstar_lines.append(f"    ex:viaCable  {cc} .")

        # ── 4. Write combined Turtle* output ──────────────────────────────
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(base_ttl + "\n".join(rdfstar_lines) + "\n", encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Convert YANG modules to OWL RDF ontology with separated SHACL validation.\n'
            'Also supports ABox RDF-star enrichment (--abox-enrich) as a standalone pass.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── TBox / ontology generation ────────────────────────────────────────────
    parser.add_argument('yang_dir',       nargs='?', default=None, help='Directory containing YANG files')
    parser.add_argument('output_file',    nargs='?', default=None, help='Output RDF/Turtle TBox file')
    parser.add_argument('--yang-dir',     dest='yang_dir_opt',    default=None, help='Directory containing YANG files')
    parser.add_argument('--output',       dest='output_file_opt', default=None, help='Output RDF/Turtle TBox file')
    parser.add_argument('--modules',      default='simap-yang.yang', help='Main YANG module to process')
    parser.add_argument('--base-uri',     default='http://www.huawei.com/ontology', help='Base URI for ontology')
    parser.add_argument('--verbose',      action='store_true', help='Enable verbose debug logging')
    parser.add_argument('--raw',          action='store_true', dest='raw_mode', help='Skip custom TBox semantic overlays for a raw YANG-to-OWL conversion')
    parser.add_argument('--html',         dest='html_output', default=None,
                        help='Optional: output path for HTML parse-tree visualisation')

    # ── ABox RDF-star enrichment (standalone pass) ────────────────────────────
    abox_grp = parser.add_argument_group(
        'ABox RDF-star enrichment',
        'Reads an existing ABox TTL, materialises ex:hasUpstreamDevice shortcuts\n'
        'and writes RDF-star <<>> annotations (Turtle* output).  Can be run without\n'
        'the YANG conversion step by omitting yang_dir / output_file.'
    )
    abox_grp.add_argument(
        '--abox-enrich', dest='abox_enrich', default=None, metavar='ABOX.ttl',
        help='ABox TTL file to enrich with RDF-star upstream connectivity triples'
    )
    abox_grp.add_argument(
        '--abox-out', dest='abox_out', default=None, metavar='ENRICHED.ttls',
        help='Output path for the enriched Turtle* ABox (default: <input>_enriched.ttls)'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Mode A: ABox-only enrichment (no YANG conversion needed) ─────────────
    if args.abox_enrich and not (args.yang_dir or args.yang_dir_opt or args.output_file or args.output_file_opt):
        abox_in  = args.abox_enrich
        abox_out = args.abox_out or str(Path(abox_in).with_suffix('')) + '_enriched.ttls'
        enricher = ABoxConnectivityEnricher(abox_in, args.base_uri)
        enricher.enrich(abox_out)
        return

    # ── Mode B: Full YANG → TBox conversion ───────────────────────────────────
    yang_dir    = args.yang_dir_opt    or args.yang_dir    or 'simap-yang'
    output_file = args.output_file_opt or args.output_file or 'simap-ontology.ttl'
    output_path = Path(output_file)

    if output_path.is_dir():
        log.warning("Output path is a directory; writing file inside it.")
        output_file = str(output_path / 'simap-ontology.ttl')
    elif str(output_path).endswith('/'):
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = str(output_path / 'simap-ontology.ttl')
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Configuration:")
    log.info(f" YANG directory : {yang_dir}")
    log.info(f" Output file    : {output_file}")
    log.info(f" Main module    : {args.modules}")
    log.info(f" Base URI       : {args.base_uri}")
    log.info(f" Raw Mode       : {args.raw_mode}")
    log.info("")

    converter = YANGToOWL(yang_dir, args.base_uri)
    converter.convert(args.modules, output_file)

    if args.html_output:
        html_gen = YANGToHTML(converter.resolver.modules)
        html_gen.generate(args.html_output)

    # ── Mode C: TBox conversion + ABox enrichment in one run ─────────────────
    if args.abox_enrich:
        abox_in  = args.abox_enrich
        abox_out = args.abox_out or str(Path(abox_in).with_suffix('')) + '_enriched.ttls'
        enricher = ABoxConnectivityEnricher(abox_in, args.base_uri)
        enricher.enrich(abox_out)


if __name__ == "__main__":
    main()