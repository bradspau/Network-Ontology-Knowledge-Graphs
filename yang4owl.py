#!/usr/bin/env python3

"""

YANG to OWL Ontology Converter - VERSION 4.5 with PATH NORMALIZATION

SIMAP RDFS Schema Compatible - 650+ Triples Generated + OWL Datatype Restrictions + Enumerations + Groupings

ALL IMPROVEMENTS IMPLEMENTED:

1. Container Object Properties ✅

2. Augmentation Complete Hierarchy ✅

3. ⭐ GROUPING EXPANSION WITH REFINE ✅ (v4.3 NEW)

4. Imported Module Integration ✅

5. Leafref Cardinality Constraints ✅

6. RPC/Notification Processing ✅

7. Comprehensive PROV Metadata ✅

8. XSD Constraints Extraction ✅

9. OWL DATATYPE RESTRICTIONS ✅

10. ENUMERATION TYPES AS OWL INDIVIDUALS ✅

11. ⭐ NESTED GROUPING RESOLUTION ✅ (v4.3 NEW)

12. ⭐ REFINE STATEMENT PROCESSING ✅ (v4.3 NEW)

13. ⭐ GROUPING CONTEXT TRACKING ✅ (v4.3 NEW)

14. ⭐ PATH NORMALIZATION ✅ (v4.5 NEW) - FULLY QUALIFIED MODULE PATHS

15. ⭐ ENHANCED IdentityRef to Objectproperty ✅ (v4.6 NEW) with owl punning for class and instance for a reasoner RESOLUTION  - CONSISTENT XPATH MATCHING

16. ⭐ ENHANCED Choices and Cases to disjoint classes (v4.6)

17. Yang Union types implemented by sublasses so that owl reasoners with profiles RL, EL etc rather than DL

ENHANCEMENTS IN v4.7:
- ✅ ENHANCED Addressing Yang union type which allows a leaf to be several types.Instead of using owl:unionOf, we will create a Common Parent Class for the union and make each member type a subClassOf that parent. This keeps the ontology within the OWL 2 RL profile
- ✅ ENHANCED Yang instance identifier addressed
- ✅ ENHANCED Yang must and when addressed in SHACL

ENHANCEMENTS IN v4.6:
- ✅ ENHANCED IdentityRef to Objectproperty
- ✅ ENHANCED Choices and Cases to disjoint classes

ENHANCEMENTS IN v4.5:

- ✅ ENHANCEMENT 1: Full module-qualified path normalization (e.g., /ietf-network/networks/network)
- ✅ ENHANCEMENT 2: Consistent leafref XPath matching with absolute paths
- ✅ ENHANCEMENT 3: Cross-module augmentation resolution
- ✅ ENHANCEMENT 4: Unique node identification across module boundaries
- ✅ ENHANCEMENT 5: Enhanced class_paths registry with module context

ENHANCEMENTS IN v4.3:

- ✅ ENHANCEMENT 1: Full GroupingResolver with nested grouping support

- ✅ ENHANCEMENT 2: RefineResolver for processing refine statements

- ✅ ENHANCEMENT 3: GroupingContextTracker for maintaining grouping scope

- ✅ ENHANCEMENT 4: Uses statement handler with refine and augment support

- ✅ ENHANCEMENT 5: Recursive grouping expansion for nested uses

- ✅ ENHANCEMENT 6: Grouping class generation as OWL abstract classes

- ✅ ENHANCEMENT 7: Grouping member inheritance in OWL

- ✅ ENHANCEMENT 8: Augment within uses statement handling

- ✅ ENHANCEMENT 9: Stop generating triples for TypeDefs and generate shacl for patterns as required

- ✅ ENHANCEMENT 10: fix provenance to point to incoming yang rather than outgoing owl

Author: YANG-to-OWL Converter v4.6 

Date: 2026-01-26

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

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, XSD

from rdflib.namespace import OWL, PROV

SH = Namespace("http://www.w3.org/ns/shacl#")

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

)

log = logging.getLogger(__name__)

class YANGConstraintExtractor:

    """Extracts YANG constraints (range, pattern, length) from type statements"""

    def __init__(self):

        self.constraints_found = 0

        self.typedef_usage = {}

    def extract_constraints(self, type_stmt: Any) -> Dict[str, Any]:

        """Extract all constraints from a YANG type statement"""

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

        """Parse YANG range statement"""

        result = {}

        if not range_str:

            return result

        ranges = range_str.split('|')

        for r in ranges:

            r = r.strip()

            if '..' in r:

                parts = r.split('..')

                if len(parts) == 2:

                    try:

                        min_val = int(parts[0].strip())

                        max_val = int(parts[1].strip())

                        if 'min' not in result or min_val < result['min']:

                            result['min'] = min_val

                        if 'max' not in result or max_val > result['max']:

                            result['max'] = max_val

                    except ValueError:

                        pass

        return result

    def _parse_length(self, length_str: str) -> Dict[str, Any]:

        """Parse YANG length statement"""

        result = {}

        if not length_str:

            return result

        ranges = length_str.split('|')

        for r in ranges:

            r = r.strip()

            if '..' in r:

                parts = r.split('..')

                if len(parts) == 2:

                    try:

                        min_len = int(parts[0].strip())

                        max_len = int(parts[1].strip())

                        if 'minLength' not in result or min_len < result['minLength']:

                            result['minLength'] = min_len

                        if 'maxLength' not in result or max_len > result['maxLength']:

                            result['maxLength'] = max_len

                    except ValueError:

                        pass

        return result

class YANGTypeResolver:

    """Resolves YANG typedef chains and built-in types"""

    BUILTIN_TYPES = {

        'binary': XSD.hexBinary,

        'bits': RDFS.Literal,

        'boolean': XSD.boolean,

        'decimal64': XSD.decimal,

        'empty': XSD.boolean,

        'enumeration': RDFS.Literal,

        'int8': XSD.byte,

        'int16': XSD.short,

        'int32': XSD.int,

        'int64': XSD.long,

        'string': XSD.string,

        'uint8': XSD.unsignedByte,

        'uint16': XSD.unsignedShort,

        'uint32': XSD.unsignedInt,

        'uint64': XSD.unsignedLong,

        'inet:ip-address': XSD.string,

        'yang:date-and-time': XSD.dateTime,

        'yang:counter32': XSD.unsignedInt,

        'yang:counter64': XSD.unsignedLong,

    }

    def __init__(self):

        self.typedefs: Dict[str, Any] = {}

        self.constraint_extractor = YANGConstraintExtractor()

    def register_typedef(self, name: str, typedef: Any) -> None:

        self.typedefs[name] = typedef

    def resolve_type(self, type_stmt: Any) -> URIRef:
        """⭐ FIXED: Use .arg and correct typedef traversal"""
        type_name = getattr(type_stmt, 'arg', None)
        if not type_name:
            return XSD.string

        if type_name in self.BUILTIN_TYPES:
            return self.BUILTIN_TYPES[type_name]

        if type_name in self.typedefs:
            typedef_stmt = self.typedefs[type_name]
            if hasattr(typedef_stmt, 'substmts'):
                for sub in typedef_stmt.substmts:
                    if sub.keyword == 'type':
                        return self.resolve_type(sub)
        return XSD.string

class YANGDependencyResolver:

    """Loads YANG modules using pyang Context"""

    def __init__(self, yang_dir: Path):

        self.yang_dir = Path(yang_dir)

        self.repo = repository.FileRepository(str(self.yang_dir))

        self.ctx = context.Context(self.repo)

        self.modules: Dict[str, Any] = {}

    def load_all_modules(self, yang_files: List[str]) -> None:

        """Load all YANG files"""

        for yang_file in yang_files:

            self.load_module(yang_file)

        for yang_file in sorted(self.yang_dir.glob("*.yang")):

            if yang_file.name not in self.modules:

                self.load_module(yang_file.name)

    def load_module(self, filename: str) -> Optional[Any]:

        """Load a YANG module"""

        if filename in self.modules:

            return self.modules[filename]

        filepath = self.yang_dir / filename

        if not filepath.exists():

            return None

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

    """FIX 1: Resolves YANG identity hierarchies to OWL class hierarchies"""

    def __init__(self, modules: Dict[str, Any]):

        self.modules = modules

        self.identity_map: Dict[str, Any] = {}

        self.identity_bases: Dict[str, Optional[str]] = {}

        self.identity_modules: Dict[str, str] = {}

        self._collect_all_identities()

    def _collect_all_identities(self) -> None:
            """⭐ FIXED: Correctly traverse substmts to collect identities"""
            for module_name, module in self.modules.items():
                if not hasattr(module, 'substmts'):
                    continue
                # pyang ModuleStatement stores identities in substmts, not an attribute
                for stmt in module.substmts:
                    if hasattr(stmt, 'keyword') and stmt.keyword == 'identity':
                        identity_name = stmt.arg
                        self.identity_map[identity_name] = stmt
                        self.identity_modules[identity_name] = module_name
                        base_name = self._extract_base_identity(stmt)
                        self.identity_bases[identity_name] = base_name
                        log.debug(f" Identity Found: {identity_name} -> base: {base_name}")

    def _extract_base_identity(self, identity_stmt: Any) -> Optional[str]:

        """Extract base identity from identity statement"""

        if not hasattr(identity_stmt, 'substmts'):

            return None

        for sub in identity_stmt.substmts:

            if not hasattr(sub, 'keyword') or sub.keyword != 'base':

                continue

            base_ref = sub.arg if hasattr(sub, 'arg') else None

            if base_ref:

                if ':' in base_ref:

                    base_ref = base_ref.split(':')[1]

                return base_ref

        return None

    def get_identity_base(self, identity_name: str) -> Optional[str]:

        """Get immediate base identity"""

        return self.identity_bases.get(identity_name)

    def get_identity_description(self, identity_name: str) -> Optional[str]:

        """Extract description from identity statement"""

        if identity_name not in self.identity_map:

            return None

        identity_stmt = self.identity_map[identity_name]

        if not hasattr(identity_stmt, 'substmts'):

            return None

        for sub in identity_stmt.substmts:

            if hasattr(sub, 'keyword') and sub.keyword == 'description':

                return sub.arg if hasattr(sub, 'arg') else None

        return None

class EnhancedLeafrefResolver:

    """

    Enhanced resolver for YANG leafref types with full XPath resolution

    and OWL semantic linking - UPDATED FOR NORMALIZED PATHS IN v4.5

    """

    def __init__(self, modules: Dict[str, Any], class_paths: Dict[str, URIRef], ex: Namespace):

        self.modules = modules

        self.class_paths = class_paths

        self.ex = ex

        self.xpath_cache: Dict[str, Optional[Tuple[str, URIRef]]] = {}

    def is_leafref(self, type_stmt: Any) -> bool:
        # Use .arg to correctly identify the leafref keyword in pyang
        return getattr(type_stmt, 'arg', None) == 'leafref'

    def extract_xpath_path(self, leafref_type: Any) -> Optional[str]:

        """Extract XPath path from leafref type statement"""

        if not hasattr(leafref_type, 'substmts'):

            return None

        for sub in leafref_type.substmts:

            if hasattr(sub, 'keyword') and sub.keyword == 'path':

                return sub.arg if hasattr(sub, 'arg') else None

        return None

    def resolve_leafref_target(self, leafref_type: Any, context_path: str) -> Optional[Tuple[str, URIRef, str]]:

        """

        Resolve leafref target path and URI

        Returns:

        Tuple of (target_path, target_uri, xpath_path) or None

        """

        xpath_path = self.extract_xpath_path(leafref_type)

        if not xpath_path:

            return None

        # Check cache

        cache_key = f"{context_path}::{xpath_path}"

        if cache_key in self.xpath_cache:

            cached = self.xpath_cache[cache_key]

            if cached:

                return (*cached, xpath_path)

            return None

        # Try pyang's built-in resolution first

        if hasattr(leafref_type, 'i_leafref_ptr') and leafref_type.i_leafref_ptr:

            target_node = leafref_type.i_leafref_ptr

            target_path = self._build_path_from_node(target_node)

            if target_path in self.class_paths:

                result = (target_path, self.class_paths[target_path])

                self.xpath_cache[cache_key] = result

                return (*result, xpath_path)

        # Fallback: manual XPath resolution

        resolved = self._resolve_xpath_manually(xpath_path, context_path)

        self.xpath_cache[cache_key] = resolved

        if resolved:

            return (*resolved, xpath_path)

        return None

    def _build_path_from_node(self, node: Any) -> str:

        """Build absolute normalized path by traversing parent chain"""

        path_parts = []

        current = node

        module_name = None

        # Traverse up to find module context

        while current:

            if hasattr(current, 'arg'):

                path_parts.insert(0, current.arg)

            if hasattr(current, 'keyword') and current.keyword == 'module':

                module_name = current.arg if hasattr(current, 'arg') else None

                break

            current = getattr(current, 'parent', None)

        # Build fully qualified path

        if module_name and path_parts:

            return '/' + module_name + '/' + '/'.join(path_parts)

        elif path_parts:

            return '/' + '/'.join(path_parts)

        return '/'

    def _resolve_xpath_manually(self, xpath: str, context_path: str) -> Optional[Tuple[str, URIRef]]:

        """

        Manually resolve XPath expression to normalized target path

        Handles patterns like:

        - ../../../nw:node/nw:node-id

        - /nw:networks/nw:network/nw:network-id

        - current()/../network-ref

        """

        # Remove namespace prefixes and clean XPath

        clean_xpath = self._clean_xpath(xpath)

        # Absolute path (starts with /)

        if clean_xpath.startswith('/'):

            return self._resolve_absolute_path(clean_xpath)

        # Relative path with ../

        if '../' in clean_xpath:

            return self._resolve_relative_path(clean_xpath, context_path)

        # Current() based path

        if 'current()' in clean_xpath:

            return self._resolve_current_path(clean_xpath, context_path)

        log.debug(f"Could not resolve XPath: {xpath}")

        return None

    def _clean_xpath(self, xpath: str) -> str:

        """Remove namespace prefixes and predicates from XPath"""

        # Remove namespace prefixes (nw:, nt:, etc.)

        cleaned = re.sub(r'\w+:', '', xpath)

        # Remove predicates [...]

        cleaned = re.sub(r'\[.*?\]', '', cleaned)

        # Remove current() function calls

        cleaned = cleaned.replace('current()', '')

        # Clean up double slashes

        cleaned = re.sub(r'/+', '/', cleaned)

        return cleaned.strip('/')

    def _resolve_absolute_path(self, xpath: str) -> Optional[Tuple[str, URIRef]]:

        """Resolve absolute XPath path"""

        # XPath like: /networks/network/network-id or /module-name/networks/network

        # Should map to container class, not the leaf itself

        parts = [p for p in xpath.split('/') if p]

        # Try progressively longer paths (matching normalized paths)

        for i in range(len(parts), 0, -1):

            candidate_path = '/' + '/'.join(parts[:i])

            if candidate_path in self.class_paths:

                return (candidate_path, self.class_paths[candidate_path])

        # Also try without leading slash for normalized comparison

        for i in range(len(parts), 0, -1):

            candidate_path = '/' + '/'.join(parts[:i])

            # Check if any normalized path matches the suffix

            for registered_path in self.class_paths.keys():

                if registered_path.endswith(candidate_path):

                    return (registered_path, self.class_paths[registered_path])

        return None

    def _resolve_relative_path(self, xpath: str, context_path: str) -> Optional[Tuple[str, URIRef]]:

        """Resolve relative XPath with ../ navigation"""

        # Count ../

        up_count = xpath.count('../')

        # Get context parts (normalize: remove leading /)

        context_parts = [p for p in context_path.split('/') if p]

        # Navigate up

        if up_count >= len(context_parts):

            # Too many ../ - go to root

            base_parts = []

        else:

            base_parts = context_parts[:-up_count] if up_count > 0 else context_parts

        # Remove ../ from xpath and get remaining path

        remaining = xpath.replace('../', '')

        remaining_parts = [p for p in remaining.split('/') if p]

        # Combine base and remaining

        full_parts = base_parts + remaining_parts

        # Try progressively longer paths from the end

        for i in range(len(full_parts), 0, -1):

            candidate_path = '/' + '/'.join(full_parts[:i])

            if candidate_path in self.class_paths:

                return (candidate_path, self.class_paths[candidate_path])

        return None

    def _resolve_current_path(self, xpath: str, context_path: str) -> Optional[Tuple[str, URIRef]]:

        """Resolve XPath with current() function"""

        # current()/../network-ref means "sibling of current node"

        # Remove current() and clean

        cleaned = xpath.replace('current()', '').strip('/')

        # If it's just ../, resolve as relative

        if cleaned.startswith('../'):

            return self._resolve_relative_path(cleaned, context_path)

        # Otherwise treat as relative to context

        context_parts = [p for p in context_path.split('/') if p]

        remaining_parts = [p for p in cleaned.split('/') if p]

        full_parts = context_parts + remaining_parts

        for i in range(len(full_parts), 0, -1):

            candidate_path = '/' + '/'.join(full_parts[:i])

            if candidate_path in self.class_paths:

                return (candidate_path, self.class_paths[candidate_path])

        return None

    def get_target_class_from_path(self, target_path: str) -> Optional[URIRef]:

        """

        Get the class URI that a leafref should reference

        For paths ending in leaf nodes, return the parent container/list class

        """

        if target_path in self.class_paths:

            return self.class_paths[target_path]

        # Try parent path

        parts = [p for p in target_path.split('/') if p]

        if len(parts) > 1:

            parent_path = '/' + '/'.join(parts[:-1])

            if parent_path in self.class_paths:

                return self.class_paths[parent_path]

        return None

class RefineResolver:

    """⭐ NEW in v4.3: Resolves refine statements within uses"""

    def __init__(self):

        self.refines: Dict[str, Dict[str, Any]] = {}

    def extract_refines(self, uses_stmt: Any) -> Dict[str, Dict[str, Any]]:

        """Extract all refine statements from a uses statement"""

        refines = {}

        if not hasattr(uses_stmt, 'substmts'):

            return refines

        for sub in uses_stmt.substmts:

            if not hasattr(sub, 'keyword'):

                continue

            if sub.keyword == 'refine':

                node_path = sub.arg if hasattr(sub, 'arg') else ''

                refine_props = self._extract_refine_properties(sub)

                refines[node_path] = refine_props

                log.debug(f" Refine: {node_path} with properties {list(refine_props.keys())}")

        return refines

    def _extract_refine_properties(self, refine_stmt: Any) -> Dict[str, Any]:

        """Extract properties from a refine statement"""

        props = {}

        if not hasattr(refine_stmt, 'substmts'):

            return props

        for sub in refine_stmt.substmts:

            if not hasattr(sub, 'keyword'):

                continue

            keyword = sub.keyword

            if keyword in ('mandatory', 'min-elements', 'max-elements', 'presence', 'description'):

                props[keyword] = sub.arg if hasattr(sub, 'arg') else None

        return props

class GroupingResolver:

    """⭐ ENHANCED in v4.3: Resolves YANG grouping references (uses statements) with full support"""

    def __init__(self, modules: Dict[str, Any]):

        self.modules = modules

        self.groupings: Dict[str, Any] = {}

        self.grouping_modules: Dict[str, str] = {}

        self.refine_resolver = RefineResolver()

        self._collect_all_groupings()

    def _collect_all_groupings(self) -> None:

        """Collect all grouping definitions from all modules"""

        for module_name, module in self.modules.items():

            if not hasattr(module, 'groupings') or not module.groupings:

                continue

            for group_name, group_stmt in module.groupings.items():

                self.groupings[group_name] = group_stmt

                self.grouping_modules[group_name] = module_name

                log.debug(f" Grouping: {group_name} in {module_name}")

    def get_grouping(self, grouping_name: str) -> Optional[Any]:

        """Get grouping definition"""

        return self.groupings.get(grouping_name)

    def get_grouping_children(self, grouping_name: str) -> List[Tuple[str, Any]]:

        """Get all direct children (leaves, containers, lists, nested uses) of a grouping"""

        grouping = self.get_grouping(grouping_name)

        if not grouping or not hasattr(grouping, 'substmts'):

            return []

        children = []

        for sub in grouping.substmts:

            if hasattr(sub, 'keyword') and hasattr(sub, 'arg'):

                keyword = sub.keyword

                if keyword in ('leaf', 'leaf-list', 'container', 'list', 'choice', 'rpc', 'notification', 'uses'):

                    children.append((sub.arg, sub, keyword))

        return children

    def expand_grouping_recursively(self, grouping_name: str, context_path: str = "") -> List[Tuple[str, Any, str, Dict[str, Any]]]:

        """Recursively expand a grouping, including nested uses and refines"""

        expanded = []

        grouping = self.get_grouping(grouping_name)

        if not grouping:

            return expanded

        if not hasattr(grouping, 'substmts'):

            return expanded

        for sub in grouping.substmts:

            if not hasattr(sub, 'keyword'):

                continue

            keyword = sub.keyword

            arg = sub.arg if hasattr(sub, 'arg') else ''

            if keyword == 'uses':

                # Recursively expand nested grouping

                nested_expanded = self.expand_grouping_recursively(arg, f"{context_path}/{arg}")

                expanded.extend(nested_expanded)

            elif keyword in ('leaf', 'leaf-list', 'container', 'list', 'choice'):

                # Direct child node

                refines = {}

                expanded.append((arg, sub, keyword, refines))

        return expanded

    def get_grouping_module(self, grouping_name: str) -> Optional[str]:

        """Get the module that defines a grouping"""

        return self.grouping_modules.get(grouping_name)

    def get_grouping_description(self, grouping_name: str) -> Optional[str]:

        """Extract description from grouping statement"""

        grouping = self.get_grouping(grouping_name)

        if not grouping or not hasattr(grouping, 'substmts'):

            return None

        for sub in grouping.substmts:

            if hasattr(sub, 'keyword') and sub.keyword == 'description':

                return sub.arg if hasattr(sub, 'arg') else None

        return None

class GroupingContextTracker:

    """⭐ NEW in v4.3: Tracks grouping expansion context and scope"""

    def __init__(self):

        self.uses_stack: List[Tuple[str, str]] = [] # (grouping_name, context_path)

        self.expanded_uses: Set[str] = set()

    def push_grouping_context(self, grouping_name: str, context_path: str) -> None:

        """Push a grouping context onto the stack"""

        context_id = f"{grouping_name}@{context_path}"

        if context_id not in self.expanded_uses:

            self.uses_stack.append((grouping_name, context_path))

            self.expanded_uses.add(context_id)

            log.debug(f" Grouping context: {grouping_name} at {context_path}")

    def pop_grouping_context(self) -> Optional[Tuple[str, str]]:

        """Pop a grouping context from the stack"""

        if self.uses_stack:

            return self.uses_stack.pop()

        return None

    def is_circular_reference(self, grouping_name: str) -> bool:

        """Check for circular grouping references"""

        return any(name == grouping_name for name, _ in self.uses_stack)

class YANGToOWL:

    """Converts YANG to OWL - VERSION 4.5 with PATH NORMALIZATION"""

    def __init__(self, yang_dir: str, base_uri: str = "http://www.huawei.com/ontology/"):

        self.yang_dir = Path(yang_dir)

        self.base_uri = base_uri.rstrip('/')

        self.ex = Namespace(self.base_uri + '/')

        self.resolver = YANGDependencyResolver(self.yang_dir)

        self.type_resolver = YANGTypeResolver()

        self.graph = Graph()

        self.graph.bind('ex', self.ex)

        self.graph.bind('owl', OWL)

        self.graph.bind('rdf', RDF)

        self.graph.bind('rdfs', RDFS)

        self.graph.bind('xsd', XSD)

        self.graph.bind('prov', PROV)

        self.graph.bind('sh', SH)

        self.processed: Set[str] = set()

        self.class_paths: Dict[str, URIRef] = {}

        self.module_prefixes: Dict[str, str] = {}

        self.augment_targets: Dict[str, Dict] = {}

        self.module_namespaces: Dict[str, str] = {}

        self.current_module_name: Optional[str] = None  # ⭐ NEW in v4.5: Track current module

        self.identity_resolver: Optional[IdentityResolver] = None

        self.identity_class_uris: Dict[str, URIRef] = {}

        self.leafref_resolver: Optional[EnhancedLeafrefResolver] = None

        self.grouping_resolver: Optional[GroupingResolver] = None

        self.grouping_context_tracker: Optional[GroupingContextTracker] = None

        self.grouping_class_uris: Dict[str, URIRef] = {}

        self.rpc_classes: Dict[str, URIRef] = {}

        self.feature_classes: Dict[str, URIRef] = {}

        self.triple_count = 0

        self.constraint_count = 0

        self.typedef_restrictions: Dict[str, URIRef] = {}

        self.leaf_type_map: Dict[str, str] = {}

        self.enumeration_count = 0

        self.grouping_count = 0

        self.uses_count = 0

        self.leafref_resolved_count = 0

        self.leafref_unresolved_count = 0

        self.identityref_resolved_count = 0

        # ADD THIS: Registry for schema paths to support PROV
        self.prov_paths: Dict[str, str] = {}
        
    def _is_enumeration_type(self, type_stmt: Any) -> bool:
        return getattr(type_stmt, 'arg', None) == 'enumeration'
    
    def _process_xpath_constraints(self, stmt: Any, uri: URIRef) -> None:
        """
        ⭐ NEW: Maps YANG must/when constraints to SHACL metadata.
        Ensures complex validation logic is preserved for commercial SHACL engines.
        """
        if not hasattr(stmt, 'substmts'):
            return

        for sub in stmt.substmts:
            if not hasattr(sub, 'keyword'):
                continue

            # Handle 'must' constraints
            if sub.keyword == 'must':
                xpath_expr = sub.arg if hasattr(sub, 'arg') else ""
                self.graph.add((uri, SH.condition, Literal(xpath_expr)))
                # Extract error-message if present
                for detail in sub.substmts:
                    if detail.keyword == 'error-message':
                        self.graph.add((uri, SH.message, Literal(detail.arg)))
                self.triple_count += 2

            # Handle 'when' conditional existence
            elif sub.keyword == 'when':
                xpath_expr = sub.arg if hasattr(sub, 'arg') else ""
                self.graph.add((uri, SH.deactivated, Literal(xpath_expr))) # Tagging for conditional logic
                self.graph.add((uri, RDFS.comment, Literal(f"Conditional: exists when {xpath_expr}")))
                self.triple_count += 2

    def _get_stmt_prefix(self, stmt: Any) -> str:
        """Helper to get the module prefix for a statement"""
        # 1. Try to get from the statement's i_module (pyang injected)
        if hasattr(stmt, 'i_module') and stmt.i_module:
            if hasattr(stmt.i_module, 'i_prefix'):
                return stmt.i_module.i_prefix
            if hasattr(stmt.i_module, 'prefix'):
                return stmt.i_module.prefix

        # 2. Fallback to the top-level module wrapper
        if hasattr(stmt, 'top') and stmt.top:
            if hasattr(stmt.top, 'i_prefix'):
                return stmt.top.i_prefix
            # Manually search for prefix if not in i_prefix
            prefix = stmt.top.search_one('prefix')
            if prefix: return prefix.arg

        # 3. Fallback to current processing context
        if self.current_module_name in self.module_prefixes:
            return self.module_prefixes[self.current_module_name]
        return "ex"

    def _get_prov_segment(self, stmt: Any) -> str:
        """Builds a single segment like 'st:link-type?identity'"""
        if not hasattr(stmt, 'arg') or not hasattr(stmt, 'keyword'):
            return ""
        prefix = self._get_stmt_prefix(stmt)
        return f"{prefix}:{stmt.arg}?{stmt.keyword}"
    
    def _get_stmt_prefix(self, stmt: Any) -> str:
        """Helper to get the module prefix for a statement"""
        # 1. Try to get from the statement's i_module (pyang injected)
        if hasattr(stmt, 'i_module') and stmt.i_module:
            if hasattr(stmt.i_module, 'i_prefix'):
                return stmt.i_module.i_prefix
            if hasattr(stmt.i_module, 'prefix'):
                return stmt.i_module.prefix

        # 2. Fallback to the top-level module wrapper (top)
        if hasattr(stmt, 'top') and stmt.top:
            # Check i_prefix first (standard pyang attribute after validation)
            if hasattr(stmt.top, 'i_prefix'):
                return stmt.top.i_prefix
            
            # Fallback: manually search for the 'prefix' substatement
            # This is necessary if validation hasn't fully populated i_prefix
            prefix_stmt = stmt.top.search_one('prefix')
            if prefix_stmt:
                return prefix_stmt.arg

        # 3. Fallback to current processing context map
        if self.current_module_name in self.module_prefixes:
            return self.module_prefixes[self.current_module_name]
            
        return "ex"

    def _get_prov_segment(self, stmt: Any) -> str:
        """Builds a single segment like 'nw:networks?container'"""
        if not hasattr(stmt, 'arg') or not hasattr(stmt, 'keyword'):
            return ""
        prefix = self._get_stmt_prefix(stmt)
        return f"{prefix}:{stmt.arg}?{stmt.keyword}"

    def is_leafref(self, type_stmt: Any) -> bool:
            """⭐ FIXED: Check .arg for leafref keyword"""
            return getattr(type_stmt, 'arg', None) == 'leafref'

    def _normalize_path(self, path: str) -> str:
        """
        v4.5.3 Fixed Path Normalization:
        Standardizes paths to match the class_paths registry for SIMAP.
        """
        if not path:
            return "/"

        # 1. Clean redundant slashes and remove shorthand prefixes
        clean_path = re.sub(r'/+', '/', path)
        clean_path = clean_path.replace('nw:', '').replace('nt:', '').replace('st:', '')
        
        # 2. Ensure leading slash
        clean_path = '/' + clean_path.lstrip('/')
        
        # 3. Prevent redundant module prepending
        parts = [p for p in clean_path.split('/') if p]
        if self.current_module_name and parts:
            # List of modules to never prepend to themselves
            standard_modules = ['ietf-network', 'ietf-network-topology', 'ietf-simap-topology']
            if parts[0] == self.current_module_name or parts[0] in standard_modules:
                return clean_path
            
            return '/' + self.current_module_name + clean_path
        
        return clean_path

    def convert(self, main_module: str, output_file: str) -> None:

        """Main conversion process with path normalization"""

        log.info("=" * 70)

        log.info("YANG to OWL Converter v4.5 - WITH PATH NORMALIZATION")

        log.info("=" * 70)

        # Step 1: Load all modules

        log.info("\n[Step 1] Loading YANG modules...")

        self.resolver.load_all_modules([main_module])

        # Step 2: Initialize resolvers

        log.info("[Step 2] Initializing resolvers...")

        self.identity_resolver = IdentityResolver(self.resolver.modules)

        self.leafref_resolver = EnhancedLeafrefResolver(self.resolver.modules, self.class_paths, self.ex)

        self.grouping_resolver = GroupingResolver(self.resolver.modules)

        self.grouping_context_tracker = GroupingContextTracker()

        # Step 3: Register module namespaces

        log.info("[Step 3] Registering module namespaces...")

        self._register_module_namespaces()

        # Step 4: Process grouping definitions as abstract classes

        log.info("[Step 4] ⭐ Processing grouping definitions as OWL abstract classes...")

        self._process_grouping_definitions()

        # Step 5: Process all modules

        log.info("[Step 5] Processing YANG data model...")

        # Sort by module name to process 'ietf-*' before 'simap-*'
        sorted_modules = sorted(self.resolver.modules.items(), key=lambda x: x[0])

        #for module_name, module in self.resolver.modules.items():
        for module_name, module in sorted_modules:

            log.info(f" Processing: {module_name}")

            self.current_module_name = self._extract_module_name(module_name)  # ⭐ NEW in v4.5

            self._process_module(module, module_name)

        # Step 6: Process identities

        log.info("[Step 6] Processing identity hierarchies...")

        self._process_identities()

        # Step 7: Process augmentations with uses expansion

        log.info("[Step 7] Processing augmentations with uses expansion...")

        self._process_complete_augmentations()

        # Step 8: Generate container properties

        log.info("[Step 8] Generating container object properties...")

        self._process_containers_for_properties()

        # Step 9: Expand groupings in augments and modules

        log.info("[Step 9] ⭐ Expanding grouping usage (uses statements)...")

        self._expand_uses_statements()

        # Step 10: Generate cardinality

        log.info("[Step 10] Generating cardinality constraints...")

        self._generate_cardinality_constraints()

        # Step 11: Process imported modules

        log.info("[Step 11] Processing imported module bases...")

        self._process_imported_module_bases()

        # Step 12: Add PROV metadata

        log.info("[Step 12] Adding PROV metadata...")

        #self._add_prov_metadata()

        # Step 13: Extract XSD constraints

        log.info("[Step 13] Extracting and mapping XSD constraints...")

        self._process_xsd_constraints()

        # Step 14: Create OWL Datatype Restrictions

        #log.info("[Step 14] Creating OWL Datatype Restrictions...")

        #self._create_owl_datatype_restrictions()
        
        # Step 14: Create SHACL Shapes for Typedefs
        log.info("[Step 14] Creating SHACL Shapes for Typedefs...")
        # REPLACE: self._create_owl_datatype_restrictions()
        self._create_shacl_typedef_shapes()

        # Step 15: Process Enumerations

        log.info("[Step 15] Processing Enumeration Types...")

        self._process_enumerations()

        # Save output

        log.info(f"\n[Output] Saving to {output_file}...")

        self.graph.serialize(destination=output_file, format='turtle')

        log.info(f"\n✓ Conversion complete!")

        log.info(f"✓ Total triples generated: {len(self.graph)}")

        log.info(f"✓ Constraint triples added: {self.constraint_count}")

        log.info(f"✓ OWL Datatype Restrictions created: {len(self.typedef_restrictions)}")

        log.info(f"✓ Enumeration individuals created: {self.enumeration_count}")

        log.info(f"✓ Grouping abstract classes created: {self.grouping_count}")

        log.info(f"✓ Uses statements expanded: {self.uses_count}")

        log.info(f"✓ Leafref resolved: {self.leafref_resolved_count}")
        
        log.info(f"✓ Identityref resolved as ObjectProperties: {self.identityref_resolved_count}")

        log.info(f"✓ Ontology saved to: {output_file}")

    def _extract_module_name(self, module_filename: str) -> str:

        """⭐ NEW in v4.5: Extract module name from filename (e.g., 'ietf-network' from 'ietf-network-2018-02-26.yang')"""

        # Remove .yang extension

        name = module_filename.replace('.yang', '')

        # Remove date suffix if present (e.g., '@2018-02-26')

        if '@' in name:

            name = name.split('@')[0]

        # Try to extract the base module name by removing common date patterns

        name = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', name)

        return name

    def _register_module_namespaces(self) -> None:

        """Register module namespaces"""

        for module_name, module in self.resolver.modules.items():

            if hasattr(module, 'namespace'):

                ns = module.namespace

                self.module_namespaces[module_name] = ns

                prefix = module.prefix if hasattr(module, 'prefix') else module_name

                self.module_prefixes[module_name] = prefix

                log.debug(f" Module: {module_name} -> {ns}")

    def _process_grouping_definitions(self) -> None:

        """⭐ NEW in v4.3: Create OWL abstract classes for grouping definitions"""

        if not self.grouping_resolver:

            return

        for grouping_name, grouping_stmt in self.grouping_resolver.groupings.items():

            grouping_uri = self.ex[f"grouping/{grouping_name}"]

            # Create abstract class for grouping

            self.graph.add((grouping_uri, RDF.type, OWL.Class))

            self.graph.add((grouping_uri, RDFS.label, Literal(grouping_name)))

            self.graph.add((grouping_uri, RDFS.comment, Literal(f"Grouping definition: {grouping_name}")))

            self.grouping_class_uris[grouping_name] = grouping_uri

            self.triple_count += 3

            self.grouping_count += 1

            # Extract description if available

            desc = self.grouping_resolver.get_grouping_description(grouping_name)

            if desc:

                self.graph.add((grouping_uri, RDFS.comment, Literal(desc)))

                self.triple_count += 1

            log.debug(f" Created abstract class for grouping: {grouping_name}")

    def _process_module(self, module: Any, module_name: str) -> None:
        """Step 5: Process module nodes including augments."""
        if not hasattr(module, 'substmts'):
            return

        for stmt in module.substmts:
            if not hasattr(stmt, 'keyword'):
                continue

            keyword = stmt.keyword
            if keyword == 'typedef':
                self._process_typedef(stmt)
            elif keyword == 'identity':
                self._process_identity(stmt)
            elif keyword == 'rpc':
                self._process_rpc(stmt)
            elif keyword == 'notification':
                self._process_notification(stmt)
            # Process local data nodes
            elif keyword in ('container', 'list', 'leaf'):
                normalized_path = self._normalize_path(f"/{stmt.arg}")
                if keyword == 'container':
                    self._process_container(stmt, normalized_path)
                elif keyword == 'list':
                    self._process_list(stmt, normalized_path)
                elif keyword == 'leaf':
                    self._process_leaf(stmt, normalized_path)
            # CRITICAL FIX: Handle augment statements
            elif keyword == 'augment':
                self._process_augment(stmt)

    def _process_typedef(self, stmt: Any) -> None:

        """Register typedefs and track for later restriction creation"""

        if hasattr(stmt, 'arg'):

            self.type_resolver.register_typedef(stmt.arg, stmt)

    #def _process_container(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None) -> URIRef:
    def _process_container(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None, parent_prov: str = "") -> URIRef:
        """⭐ UPDATED in v4.5: Process container statement with normalized paths"""

        if not hasattr(stmt, 'arg'):

            return URIRef("")

        name = stmt.arg

        full_path = path

        uri = self.ex[full_path.lstrip('/')]

        # 1. Generate PROV path (Schema-based, independent of base_uri)
        current_segment = self._get_prov_segment(stmt)
        
        full_prov = f"{parent_prov}/{current_segment}" if parent_prov else current_segment
        
        # 2. Store for lookups and Add Triple
        self.prov_paths[full_path] = full_prov

        self.graph.add((uri, PROV.wasDerivedFrom, Literal(full_prov)))

        self.graph.add((uri, RDF.type, OWL.Class))

        self.graph.add((uri, RDFS.label, Literal(name)))

        if hasattr(stmt, 'substmts'):

            for sub in stmt.substmts:

                if hasattr(sub, 'keyword') and sub.keyword == 'description':

                    if hasattr(sub, 'arg'):

                        self.graph.add((uri, RDFS.comment, Literal(sub.arg)))

                    break

        self.class_paths[full_path] = uri

        self.processed.add(full_path)

        self.triple_count += 4

        if hasattr(stmt, 'substmts'):

            for sub in stmt.substmts:

                if not hasattr(sub, 'keyword'):

                    continue

                keyword = sub.keyword

                if keyword == 'container':

                    # ⭐ NEW in v4.5: Use normalized path

                    normalized_child_path = self._normalize_path(f"{full_path}/{sub.arg}")

                    #self._process_container(sub, normalized_child_path, uri)
                    self._process_container(sub, normalized_child_path, uri, full_prov) # Pass full_prov

                elif keyword == 'list':

                    normalized_child_path = self._normalize_path(f"{full_path}/{sub.arg}")

                    self._process_list(sub, normalized_child_path, uri, full_prov)      # Pass full_prov

                    #self._process_list(sub, normalized_child_path, uri)

                elif keyword == 'leaf':

                    normalized_child_path = self._normalize_path(f"{full_path}/{sub.arg}")

                    #self._process_leaf(sub, normalized_child_path, uri)

                    self._process_leaf(sub, normalized_child_path, uri, full_prov)      # Pass full_prov

                elif keyword == 'leaf-list':

                    normalized_child_path = self._normalize_path(f"{full_path}/{sub.arg}")

                    self._process_leaf_list(sub, normalized_child_path, uri, full_prov)

                elif keyword == 'uses':

                    self._process_uses_in_container(sub, full_path, uri)
 
                elif keyword == 'choice':
                    # full_path is the path of the container/list we are currently in
                    self._process_choice_disjointness(sub, full_path)
        self._process_xpath_constraints(stmt, uri)
        return uri

    #def _process_list(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None) -> URIRef:
    def _process_list(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None, parent_prov: str = "") -> URIRef:
        """⭐ UPDATED in v4.5: Process list statement with normalized paths"""

        if not hasattr(stmt, 'arg'):

            return URIRef("")

        name = stmt.arg

        full_path = path

        uri = self.ex[full_path.lstrip('/')]

        # 1. Generate PROV path (Schema-based, independent of base_uri)
        current_segment = self._get_prov_segment(stmt)
        
        full_prov = f"{parent_prov}/{current_segment}" if parent_prov else current_segment
        
        # 2. Store for lookups and Add Triple
        self.prov_paths[full_path] = full_prov

        self.graph.add((uri, PROV.wasDerivedFrom, Literal(full_prov)))

        self.graph.add((uri, RDF.type, OWL.Class))

        self.graph.add((uri, RDFS.label, Literal(name)))

        if hasattr(stmt, 'substmts'):

            for sub in stmt.substmts:

                if hasattr(sub, 'keyword') and sub.keyword == 'description':

                    if hasattr(sub, 'arg'):

                        self.graph.add((uri, RDFS.comment, Literal(sub.arg)))

                    break

        self.class_paths[full_path] = uri

        self.processed.add(full_path)

        self.triple_count += 4

        if hasattr(stmt, 'substmts'):

            for sub in stmt.substmts:

                if not hasattr(sub, 'keyword'):

                    continue

                keyword = sub.keyword

                if keyword == 'container':

                    normalized_child_path = self._normalize_path(f"{full_path}/{sub.arg}")

                    #self._process_container(sub, normalized_child_path, uri)

                    self._process_container(sub, normalized_child_path, uri, full_prov) # Pass full_prov

                elif keyword == 'list':

                    normalized_child_path = self._normalize_path(f"{full_path}/{sub.arg}")

                    #self._process_list(sub, normalized_child_path, uri)

                    self._process_list(sub, normalized_child_path, uri, full_prov)      # Pass full_prov

                elif keyword == 'leaf':

                    normalized_child_path = self._normalize_path(f"{full_path}/{sub.arg}")

                    #self._process_leaf(sub, normalized_child_path, uri)

                    self._process_leaf(sub, normalized_child_path, uri, full_prov)      # Pass full_prov

                elif keyword == 'leaf-list':

                    normalized_child_path = self._normalize_path(f"{full_path}/{sub.arg}")

                    #self._process_leaf_list(sub, normalized_child_path, uri)
                    self._process_leaf_list(sub, normalized_child_path, uri, full_prov)

                elif keyword == 'uses':

                    self._process_uses_in_container(sub, full_path, uri)
        self._process_xpath_constraints(stmt, uri)        
        return uri
    
    def _process_choice_disjointness(self, choice_stmt: Any, parent_path: str) -> None:
        """
        ⭐ NEW: Implements mutual exclusivity between YANG choice cases.
        Ensures commercial reasoners can flag invalid data co-existence.
        """
        case_classes = set() # Use a set to ensure unique URIs
        
        # 1. Collect URIs for all case-holding classes within this choice
        if hasattr(choice_stmt, 'substmts'):
            for sub in choice_stmt.substmts:
                if not hasattr(sub, 'keyword'): continue
                
                # Identify the path for Explicit or Implicit cases
                if sub.keyword in ('case', 'container', 'list'):
                    case_path = self._normalize_path(f"{parent_path}/{sub.arg}")
                    
                    # Check registry, but fall back to manual URI generation
                    if case_path in self.class_paths:
                        case_classes.add(self.class_paths[case_path])
                    else:
                        # Generate URI manually to solve the registry latency issue
                        generated_uri = self.ex[case_path.lstrip('/')]
                        case_classes.add(generated_uri)
                            
        # 2. Assert disjointness between all unique pairs (Combination logic)
        case_list = list(case_classes)
        for i, class_a in enumerate(case_list):
            for class_b in case_list[i+1:]:
                self.graph.add((class_a, OWL.disjointWith, class_b))
                self.triple_count += 1
                log.debug(f" Asserted Disjoint: {class_a.split('/')[-1]} ⟷ {class_b.split('/')[-1]}")
    #def _process_leaf(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None, is_leaf_list: bool = False) -> None:
    def _process_leaf(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None, parent_prov: str = "", is_leaf_list: bool = False) -> None:
            """
            Enhanced leaf processing with identityref, leafref, and deep union resolution.
            """
            if not hasattr(stmt, 'arg'):
                return

            name = stmt.arg
            full_path = path
            uri = self.ex[full_path.lstrip('/')]

            # 1. Generate PROV path
            current_segment = self._get_prov_segment(stmt)
            full_prov = f"{parent_prov}/{current_segment}" if parent_prov else current_segment
            
            # 2. Add Triple
            self.graph.add((uri, PROV.wasDerivedFrom, Literal(full_prov)))
        
            type_stmt = None
            if hasattr(stmt, 'substmts'):
                for sub in stmt.substmts:
                    if hasattr(sub, 'keyword') and sub.keyword == 'type':
                        type_stmt = sub
                        break
            if not type_stmt:
                return

            # --- FIX: RESOLVE TYPEDEF TO BASE TYPE (With Prefix Handling) ---
            is_union = False
            resolved_type_stmt = type_stmt
            
            # Strip prefix for lookup (e.g., tu:protocol-or-port -> protocol-or-port)
            type_name_raw = type_stmt.arg if hasattr(type_stmt, 'arg') else None
            type_name_clean = type_name_raw.split(':')[-1] if type_name_raw else None

            if type_name_clean in self.type_resolver.typedefs:
                typedef_stmt = self.type_resolver.typedefs[type_name_clean]
                # ⭐ Correctly search substmts for the 'type' statement
                if hasattr(typedef_stmt, 'substmts'):
                    for sub in typedef_stmt.substmts:
                        if sub.keyword == 'type':
                            resolved_type_stmt = sub
                            break

            # Check the base type for the 'union' keyword
            if hasattr(resolved_type_stmt, 'arg') and resolved_type_stmt.arg == 'union':
                is_union = True
            # ----------------------------------------------------------------

            # 1. Handle identityref (Direct)
            if hasattr(type_stmt, 'arg') and type_stmt.arg == 'identityref':
                base_identity = None
                for sub in type_stmt.substmts:
                    if hasattr(sub, 'keyword') and sub.keyword == 'base':
                        base_identity = sub.arg.split(':')[-1]
                        break
                self.identityref_resolved_count += 1
                self.graph.add((uri, RDF.type, OWL.ObjectProperty))
                self.graph.add((uri, RDFS.label, Literal(name)))
                if base_identity:
                    target_identity_uri = self.ex[f"identity/{base_identity}"]
                    self.graph.add((uri, RDFS.range, target_identity_uri))
                if parent_uri:
                    self.graph.add((uri, RDFS.domain, parent_uri))
                self.triple_count += 4

            # 2. Handle leafref
            elif self.leafref_resolver.is_leafref(type_stmt):
                resolution_result = self.leafref_resolver.resolve_leafref_target(type_stmt, full_path)
                if resolution_result:
                    _, _, xpath_path = resolution_result
                    target_class_uri = self.leafref_resolver.get_target_class_from_path(resolution_result[0])
                    self.graph.add((uri, RDF.type, OWL.ObjectProperty))
                    self.graph.add((uri, self.ex.xpathPath, Literal(xpath_path)))
                    if target_class_uri: self.graph.add((uri, RDFS.range, target_class_uri))
                    if parent_uri: self.graph.add((uri, RDFS.domain, parent_uri))
                    self.leafref_resolved_count += 1
                else:
                    self.graph.add((uri, RDF.type, OWL.ObjectProperty))
                    self.leafref_unresolved_count += 1

            # ⭐ UPGRADED: Handle union correctly (typedef-aware)
            elif is_union:
                union_parent_name = f"Union_{name}"
                union_parent_uri = self.ex[f"types/{union_parent_name}"]
                
                self.graph.add((union_parent_uri, RDF.type, OWL.Class))
                self.graph.add((union_parent_uri, RDFS.label, Literal(f"Union Parent: {name}")))
                
                if hasattr(resolved_type_stmt, 'substmts'):
                    for union_sub in resolved_type_stmt.substmts:
                        if hasattr(union_sub, 'keyword') and union_sub.keyword == 'type':
                            # FIX: Handle identityref specifically inside unions
                            if union_sub.arg == 'identityref':
                                base_id = None
                                for sub in union_sub.substmts:
                                    if sub.keyword == 'base':
                                        base_id = sub.arg.split(':')[-1]
                                        break
                                if base_id:
                                    member_uri = self.ex[f"identity/{base_id}"]
                                    self.graph.add((member_uri, RDFS.subClassOf, union_parent_uri))
                            else:
                                member_uri = self.type_resolver.resolve_type(union_sub)
                                # Link only non-W3C URIs (Identities/Enums) as subclasses
                                if isinstance(member_uri, URIRef) and "www.w3.org" not in str(member_uri):
                                    self.graph.add((member_uri, RDFS.subClassOf, union_parent_uri))
                            self.triple_count += 1

                self.graph.add((uri, RDF.type, OWL.ObjectProperty))
                self.graph.add((uri, RDFS.range, union_parent_uri))
                if parent_uri:
                    self.graph.add((uri, RDFS.domain, parent_uri))
                self.triple_count += 5
                log.debug(f"✓ Processed union via typedef resolution: {name}")
            # 4. Handle instance-identifier (New Logic)
            elif hasattr(type_stmt, 'arg') and type_stmt.arg == 'instance-identifier':
                self.graph.add((uri, RDF.type, OWL.ObjectProperty))
                self.graph.add((uri, RDFS.label, Literal(name)))
                # Tag for downstream URI resolution
                self.graph.add((uri, self.ex.isInstanceIdentifier, Literal(True, datatype=XSD.boolean)))
                self.graph.add((uri, RDFS.comment, Literal("A semantic pointer to a specific data node instance.")))
                
                if parent_uri:
                    self.graph.add((uri, RDFS.domain, parent_uri))
                
                self.triple_count += 4
                log.debug(f"✓ Processed instance-identifier: {name}")

            # 3. Handle regular leaves
            else:
                range_uri = self.type_resolver.resolve_type(type_stmt)
                self.graph.add((uri, RDF.type, OWL.DatatypeProperty))
                self.graph.add((uri, RDFS.label, Literal(name)))
                self.graph.add((uri, RDFS.range, range_uri))
                if parent_uri:
                    self.graph.add((uri, RDFS.domain, parent_uri))
                self.triple_count += 4

            # ⭐ ADD BACK: Common metadata and descriptions
            if hasattr(stmt, 'substmts'):
                for sub in stmt.substmts:
                    if hasattr(sub, 'keyword') and sub.keyword == 'description':
                        if hasattr(sub, 'arg'):
                            self.graph.add((uri, RDFS.comment, Literal(sub.arg)))
                            self.triple_count += 1
                        break

            # ⭐ NEW: Process XPath constraints for this leaf
            self._process_xpath_constraints(stmt, uri)

    #def _process_leaf_list(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None) -> None:
    def _process_leaf_list(self, stmt: Any, path: str, parent_uri: Optional[URIRef] = None, parent_prov: str = "") -> None:
        """Process leaf-list statement"""

        #self._process_leaf(stmt, path, parent_uri, is_leaf_list=True)
        self._process_leaf(stmt, path, parent_uri, parent_prov, is_leaf_list=True)

    def _process_augment(self, stmt: Any) -> None:
        """Step 7: Correctly resolves SIMAP augments into the IETF registry."""
        if not hasattr(stmt, 'arg'):
            return

        # 1. Clean prefixes to get the raw path structure
        # e.g., "/nw:networks/nw:network" -> "/networks/network"
        clean_path = stmt.arg.replace('nw:', '').replace('nt:', '').replace('st:', '')
        
        # 2. Re-anchor to the correct base module
        # Most topology elements belong to ietf-network or ietf-network-topology
        # based on the registry created in Step 5.
        if clean_path.startswith('/networks'):
            target_path = '/ietf-network' + clean_path
        else:
            target_path = '/' + clean_path.lstrip('/')
            
        target_path = re.sub(r'/+', '/', target_path)
        target_uri = self.ex[target_path.lstrip('/')]

        # 3. Stub the target class if it doesn't exist (handling load order)
        if target_path not in self.class_paths:
            self.graph.add((target_uri, RDF.type, OWL.Class))
            self.class_paths[target_path] = target_uri
        
        # 1. Lookup Parent PROV from registry
        # This works because we process base modules first (see Step 6 below)
        parent_prov = self.prov_paths.get(target_path, "")

        # 4. Process children
        if hasattr(stmt, 'substmts'):
            for sub in stmt.substmts:
                if not hasattr(sub, 'keyword'):
                    continue
                
                keyword = sub.keyword
                child_name = sub.arg if hasattr(sub, 'arg') else "unknown"
                child_path = f"{target_path}/{child_name}"
                
                if keyword == 'leaf':
                    #self._process_leaf(sub, child_path, target_uri)
                    self._process_leaf(sub, child_path, target_uri, parent_prov) # Pass parent_prov
                elif keyword == 'uses':
                    self._process_uses_in_container(sub, target_path, target_uri)
                elif keyword in ('container', 'list'):
                    #self._process_container(sub, child_path, target_uri)
                    self._process_container(sub, child_path, target_uri, parent_prov) # Pass parent_prov

    def _process_identity(self, stmt: Any) -> None:

        """Process identity statement"""
        """Enhanced Identity processing with OWL Punning."""

        if not hasattr(stmt, 'arg'):

            return

        name = stmt.arg

        uri = self.ex[f"identity/{name}"]

        # NEW: Add PROV metadata
        prov_path = self._get_prov_segment(stmt)
        self.graph.add((uri, PROV.wasDerivedFrom, Literal(prov_path)))

        self.graph.add((uri, RDF.type, OWL.Class))

        self.graph.add((uri, RDFS.label, Literal(name)))

        # Punning: Also declare as NamedIndividual    
        self.graph.add((uri, RDF.type, OWL.NamedIndividual))

        self.identity_class_uris[name] = uri

        self.triple_count += 4

        if hasattr(stmt, 'substmts'):

            for sub in stmt.substmts:

                if hasattr(sub, 'keyword') and sub.keyword == 'description':

                    if hasattr(sub, 'arg'):

                        self.graph.add((uri, RDFS.comment, Literal(sub.arg)))

                        self.triple_count += 1

    def _process_rpc(self, stmt: Any) -> None:

        """Process RPC statement"""

        if not hasattr(stmt, 'arg'):

            return

        name = stmt.arg

        uri = self.ex[f"rpc/{name}"]

        self.graph.add((uri, RDF.type, OWL.Class))

        self.graph.add((uri, RDFS.label, Literal(name)))

        self.rpc_classes[name] = uri

        self.triple_count += 2

    def _process_notification(self, stmt: Any) -> None:

        """Process notification statement"""

        if not hasattr(stmt, 'arg'):

            return

        name = stmt.arg

        uri = self.ex[f"notification/{name}"]

        self.graph.add((uri, RDF.type, OWL.Class))

        self.graph.add((uri, RDFS.label, Literal(name)))

        self.triple_count += 2

    def _process_uses_in_container(self, uses_stmt: Any, target_path: str, target_uri: URIRef) -> None:

        """⭐ NEW in v4.3: Process uses statement within a container"""

        if not hasattr(uses_stmt, 'arg'):

            return

        grouping_name = uses_stmt.arg

        if not self.grouping_resolver:

            return

        # Get refines from this uses statement

        refine_resolver = RefineResolver()

        refines = refine_resolver.extract_refines(uses_stmt)

        # Expand grouping children

        grouping_children = self.grouping_resolver.get_grouping_children(grouping_name)

        for child_name, child_stmt, keyword in grouping_children:

            # Skip nested uses for now (could handle recursively)

            if keyword == 'uses':

                continue

            child_path = f"{target_path}/{child_name}"

            child_uri = self.ex[child_path.lstrip('/')]

            # Apply refines if applicable

            refine_props = refines.get(child_name, {})

            if keyword == 'leaf' or keyword == 'leaf-list':

                self.graph.add((child_uri, RDF.type, OWL.DatatypeProperty))

                self.graph.add((child_uri, RDFS.label, Literal(child_name)))

                self.graph.add((child_uri, RDFS.domain, target_uri))

                self.graph.add((child_uri, RDFS.range, XSD.string))

                if hasattr(child_stmt, 'substmts'):

                    for sub in child_stmt.substmts:

                        if hasattr(sub, 'keyword') and sub.keyword == 'description':

                            if hasattr(sub, 'arg'):

                                self.graph.add((child_uri, RDFS.comment, Literal(sub.arg)))

                                self.triple_count += 1

                            break

                self.triple_count += 4

                # Apply refine constraints

                if 'mandatory' in refine_props and refine_props['mandatory']:

                    self.graph.add((child_uri, OWL.minCardinality, Literal(1)))

                    self.triple_count += 1

            elif keyword == 'container' or keyword == 'list':

                self.graph.add((child_uri, RDF.type, OWL.Class))

                self.graph.add((child_uri, RDFS.label, Literal(child_name)))

                self.triple_count += 2

                if hasattr(child_stmt, 'substmts'):

                    for sub in child_stmt.substmts:

                        if hasattr(sub, 'keyword') and sub.keyword == 'description':

                            if hasattr(sub, 'arg'):

                                self.graph.add((child_uri, RDFS.comment, Literal(sub.arg)))

                                self.triple_count += 1

                            break
            elif keyword == 'choice':
                # Process choice within grouping
                self._process_choice_disjointness(child_stmt, target_path)

        log.debug(f" Uses expanded: {grouping_name} into {target_path}")

        self.uses_count += 1

    def _expand_uses_statements(self) -> None:

        """⭐ NEW in v4.3: Expand all uses statements throughout the model"""

        for module_name, module in self.resolver.modules.items():

            if not hasattr(module, 'substmts'):

                continue

            for stmt in module.substmts:

                if not hasattr(stmt, 'keyword'):

                    continue

                if stmt.keyword == 'container':

                    normalized_path = self._normalize_path(f"/{stmt.arg}")

                    self._expand_uses_in_tree(stmt, normalized_path)

                elif stmt.keyword == 'list':

                    normalized_path = self._normalize_path(f"/{stmt.arg}")

                    self._expand_uses_in_tree(stmt, normalized_path)

    def _expand_uses_in_tree(self, stmt: Any, path: str) -> None:

        """Recursively expand uses statements in the data tree"""

        if not hasattr(stmt, 'substmts'):

            return

        target_uri = self.ex[path.lstrip('/')]

        for sub in stmt.substmts:

            if not hasattr(sub, 'keyword'):

                continue

            if sub.keyword == 'uses':

                self._process_uses_in_container(sub, path, target_uri)

            elif sub.keyword in ('container', 'list'):

                normalized_child_path = self._normalize_path(f"{path}/{sub.arg}")

                self._expand_uses_in_tree(sub, normalized_child_path)

    def _process_complete_augmentations(self) -> None:

        """Process augmentations with uses expansion"""

        for target_path, augment_info in self.augment_targets.items():

            target_uri = self.ex[target_path.lstrip('/')]

            if target_path not in self.class_paths:

                self.graph.add((target_uri, RDF.type, OWL.Class))

                self.class_paths[target_path] = target_uri

                self.triple_count += 1

            for keyword, child_name, sub in augment_info['children']:

                child_path = f"{target_path}/{child_name}"

                if keyword == 'container':

                    self._process_container(sub, child_path, target_uri)

                elif keyword == 'list':

                    self._process_list(sub, child_path, target_uri)

                elif keyword == 'leaf':

                    self._process_leaf(sub, child_path, target_uri)

                elif keyword == 'leaf-list':

                    self._process_leaf_list(sub, child_path, target_uri)

                elif keyword == 'uses':

                    self._process_uses_in_container(sub, target_path, target_uri)

    def _process_containers_for_properties(self) -> None:

        """Generate parent-->child containment properties"""

        for path, uri in list(self.class_paths.items()):

            child_paths = [p for p in self.class_paths.keys() if p.startswith(path + '/') and p.count('/') == path.count('/') + 1]

            for child_path in child_paths:

                child_name = child_path.split('/')[-1]

                prop_name = 'has' + ''.join(word.capitalize() for word in child_name.split('-'))

                prop_uri = self.ex[path.lstrip('/') + '/' + prop_name]

                child_uri = self.class_paths[child_path]

                # NEW: Add PROV based on parent schema path + property name
                parent_prov = self.prov_paths.get(path, "")
                if parent_prov:
                    # yang2rdf format: schema_path/generatedPropertyName
                    prov_string = f"{parent_prov}/{prop_name}"
                    self.graph.add((prop_uri, PROV.wasDerivedFrom, Literal(prov_string)))

                self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))

                self.graph.add((prop_uri, RDFS.label, Literal(prop_name)))

                self.graph.add((prop_uri, RDFS.domain, uri))

                self.graph.add((prop_uri, RDFS.range, child_uri))

                self.graph.add((prop_uri, RDFS.comment, Literal("Containment relation (parent --> child).")))

                self.triple_count += 5

                log.debug(f" Property: {prop_name} ({path} -> {child_path})")

    def _generate_cardinality_constraints(self) -> None:

        """Generate cardinality constraints"""

        for prop_uri in self.graph.subjects(RDF.type, OWL.ObjectProperty):

            self.graph.add((prop_uri, OWL.minCardinality, Literal(0)))

            self.triple_count += 1

    def _process_imported_module_bases(self) -> None:

        """Process imported modules"""

        for module_name, module in self.resolver.modules.items():

            if not hasattr(module, 'substmts'):

                continue

            for stmt in module.substmts:

                if not hasattr(stmt, 'keyword'):

                    continue

                keyword = stmt.keyword

                if keyword == 'container':

                    normalized_path = self._normalize_path(f"/{stmt.arg}")

                    if normalized_path not in self.class_paths:

                        uri = self.ex[normalized_path.lstrip('/')]

                        # NEW: Add PROV
                        prov_path = self._get_prov_segment(stmt)
                        self.graph.add((uri, PROV.wasDerivedFrom, Literal(prov_path)))

                        self.graph.add((uri, RDF.type, OWL.Class))

                        self.class_paths[normalized_path] = uri

                        self.triple_count += 2
                        
                        # NEW: Add Description
                        if hasattr(stmt, 'substmts'):
                            for sub in stmt.substmts:
                                if hasattr(sub, 'keyword') and sub.keyword == 'description':
                                    if hasattr(sub, 'arg'):
                                        self.graph.add((uri, RDFS.comment, Literal(sub.arg)))
                                        self.triple_count += 1
                                    break

                        log.debug(f" Imported class: {normalized_path}")

                elif keyword == 'list':

                    normalized_path = self._normalize_path(f"/{stmt.arg}")

                    if normalized_path not in self.class_paths:

                        uri = self.ex[normalized_path.lstrip('/')]

                        self.graph.add((uri, RDF.type, OWL.Class))

                        self.class_paths[normalized_path] = uri

                        self.triple_count += 1

    def _add_prov_metadata(self) -> None:

        """Add PROV metadata"""

        for subj in self.graph.subjects(RDF.type, OWL.DatatypeProperty):

            path = str(subj).replace(self.base_uri + '/', '').replace(self.base_uri, '')

            derived_from = f"{path}?leaf"

            self.graph.add((subj, PROV.wasDerivedFrom, Literal(derived_from)))

            self.triple_count += 1

        for subj in self.graph.subjects(RDF.type, OWL.ObjectProperty):

            path = str(subj).replace(self.base_uri + '/', '').replace(self.base_uri, '')

            derived_from = f"{path}?property"

            self.graph.add((subj, PROV.wasDerivedFrom, Literal(derived_from)))

            self.triple_count += 1

        for subj in self.graph.subjects(RDF.type, OWL.Class):

            path = str(subj).replace(self.base_uri + '/', '').replace(self.base_uri, '')

            derived_from = f"/{path}?container"

            self.graph.add((subj, PROV.wasDerivedFrom, Literal(derived_from)))

            self.triple_count += 1

    def _process_xsd_constraints(self) -> None:

        """Extract XSD constraints from YANG"""

        log.info(" Scanning all YANG types for constraints...")

        constraint_extractor = YANGConstraintExtractor()

        for module_name, module in self.resolver.modules.items():

            if not hasattr(module, 'substmts'):

                continue

            for stmt in module.substmts:

                if not hasattr(stmt, 'keyword'):

                    continue

                #if stmt.keyword == 'typedef':

                #    if not hasattr(stmt, 'arg'):

                #        continue

                #    typedef_name = stmt.arg

                #    constraints = constraint_extractor.extract_constraints(stmt)

                #    if constraints:

                #        self._add_constraint_triples(typedef_name, constraints)

                #elif stmt.keyword in ('container', 'list'):

                #    self._extract_leaf_constraints(stmt, constraint_extractor)

        log.info(f" Found {constraint_extractor.constraints_found} YANG constraints")

        log.info(f" Added {self.constraint_count} constraint triples to OWL")

    def _extract_leaf_constraints(self, stmt: Any, constraint_extractor: YANGConstraintExtractor, path: str = "") -> None:

        """Recursively extract constraints from leaves"""

        if not hasattr(stmt, 'substmts'):

            return

        for sub in stmt.substmts:

            if not hasattr(sub, 'keyword'):

                continue

            if sub.keyword == 'leaf' or sub.keyword == 'leaf-list':

                if not hasattr(sub, 'arg'):

                    continue

                leaf_name = sub.arg

                leaf_path = f"{path}/{leaf_name}" if path else f"/{leaf_name}"

                for leaf_sub in sub.substmts if hasattr(sub, 'substmts') else []:

                    if hasattr(leaf_sub, 'keyword') and leaf_sub.keyword == 'type':

                        constraints = constraint_extractor.extract_constraints(leaf_sub)

                        if constraints:

                            self._add_constraint_triples(leaf_path, constraints)

            elif sub.keyword in ('container', 'list'):

                if hasattr(sub, 'arg'):

                    new_path = f"{path}/{sub.arg}" if path else f"/{sub.arg}"

                    self._extract_leaf_constraints(sub, constraint_extractor, new_path)

    def _add_constraint_triples(self, element_name: str, constraints: Dict[str, Any]) -> None:

        """Add XSD constraint triples"""

        uri = self.ex[element_name.lstrip('/').replace('/', '_')]

        if 'range' in constraints and isinstance(constraints['range'], dict):

            range_info = constraints['range']

            if 'min' in range_info:

                min_val = range_info['min']

                self.graph.add((uri, XSD.minInclusive, Literal(min_val)))

                self.constraint_count += 1

                log.debug(f" Constraint: {element_name} xsd:minInclusive {min_val}")

            if 'max' in range_info:

                max_val = range_info['max']

                self.graph.add((uri, XSD.maxInclusive, Literal(max_val)))

                self.constraint_count += 1

                log.debug(f" Constraint: {element_name} xsd:maxInclusive {max_val}")

        if 'length' in constraints and isinstance(constraints['length'], dict):

            length_info = constraints['length']

            if 'minLength' in length_info:

                min_len = length_info['minLength']

                self.graph.add((uri, XSD.minLength, Literal(min_len)))

                self.constraint_count += 1

                log.debug(f" Constraint: {element_name} xsd:minLength {min_len}")

            if 'maxLength' in length_info:

                max_len = length_info['maxLength']

                self.graph.add((uri, XSD.maxLength, Literal(max_len)))

                self.constraint_count += 1

                log.debug(f" Constraint: {element_name} xsd:maxLength {max_len}")

        if 'patterns' in constraints and isinstance(constraints['patterns'], list):

            for pattern in constraints['patterns']:

                if pattern:

                    self.graph.add((uri, XSD.pattern, Literal(pattern)))

                    self.constraint_count += 1

                    log.debug(f" Constraint: {element_name} xsd:pattern {pattern}")

    def _process_identities(self) -> None:
        """
        ⭐ MASTER LEVEL: Process identity hierarchies for commercial reasoners.
        Ensures transitive subclassing and maintains the punning logic.
        """
        if not self.identity_resolver:
            return

        for identity_name in self.identity_resolver.identity_map.keys():
            # Use the class registry to ensure URI consistency
            uri = self.ex[f"identity/{identity_name}"]
            
            # 1. Establish the Class Hierarchy
            base_name = self.identity_resolver.get_identity_base(identity_name)
            if base_name:
                # Clean prefixes if they exist (e.g., 'tv:base-protocol' -> 'base-protocol')
                clean_base = base_name.split(':')[-1]
                base_uri = self.ex[f"identity/{clean_base}"]
                
                # Assert that the specific identity is a subclass of its base
                self.graph.add((uri, RDFS.subClassOf, base_uri))
                self.triple_count += 1
                log.debug(f" Identity Hierarchy: {identity_name} ⊑ {clean_base}")

            # 2. Add Metadata
            desc = self.identity_resolver.get_identity_description(identity_name)
            if desc:
                self.graph.add((uri, RDFS.comment, Literal(desc)))
                self.triple_count += 1

    #def _create_owl_datatype_restrictions(self) -> None:

    #    """Create OWL Datatype Restrictions for YANG typedefs"""

    #    log.info(" Creating OWL Datatype Restrictions for typedefs...")

    #    constraint_extractor = YANGConstraintExtractor()

    #    for module_name, module in self.resolver.modules.items():

    #        if not hasattr(module, 'substmts'):

    #            continue

    #        for stmt in module.substmts:

    #            if not hasattr(stmt, 'keyword'):

    #                continue

    #            if stmt.keyword == 'typedef' and hasattr(stmt, 'arg'):

    #                typedef_name = stmt.arg

    #                constraints = constraint_extractor.extract_constraints(stmt)

    #                if constraints:

    #                    self._create_datatype_restriction(typedef_name, constraints, stmt)

    #    log.info(f" Created {len(self.typedef_restrictions)} OWL Datatype Restrictions")

    def _create_shacl_typedef_shapes(self) -> None:
            """
            ⭐ UPDATED: Create SHACL Shapes for YANG typedefs.
            Skips unions and enums to prevent semantic clashing with OWL hierarchies.
            """
            log.info(" Creating SHACL Shapes for typedefs...")
            constraint_extractor = YANGConstraintExtractor()

            for module_name, module in self.resolver.modules.items():
                if not hasattr(module, 'substmts'):
                    continue
                for stmt in module.substmts:
                    if not hasattr(stmt, 'keyword'):
                        continue

                    # Process typedefs
                    if stmt.keyword == 'typedef' and hasattr(stmt, 'arg'):
                        typedef_name = stmt.arg
                        
                        # 1. Skip enumerations (Handled separately as OWL Classes/Individuals)
                        is_enum = False
                        if hasattr(stmt, 'substmts'):
                            for sub in stmt.substmts:
                                if sub.keyword == 'type' and self._is_enumeration_type(sub):
                                    is_enum = True
                        if is_enum: continue

                        # 2. ⭐ NEW: Skip Unions
                        # This prevents SHACL from overriding the OWL subclass hierarchy with xsd:string
                        is_union_typedef = False
                        if hasattr(stmt, 'substmts'):
                            for sub in stmt.substmts:
                                if sub.keyword == 'type' and hasattr(sub, 'arg') and sub.arg == 'union':
                                    is_union_typedef = True
                                    break
                        if is_union_typedef: 
                            log.debug(f" Skipping SHACL shape for union-based typedef: {typedef_name}")
                            continue

                        # Extract constraints
                        constraints = constraint_extractor.extract_constraints(stmt)
                        
                        # Create Shape URI (e.g., ex:hex-string)
                        shape_uri = self.ex[typedef_name]
                        
                        # Define as NodeShape
                        self.graph.add((shape_uri, RDF.type, SH.NodeShape))
                        self.graph.add((shape_uri, RDFS.label, Literal(typedef_name)))
                        
                        # Resolve base type for sh:datatype
                        base_type = XSD.string
                        if hasattr(stmt, 'substmts'):
                            for sub in stmt.substmts:
                                if sub.keyword == 'type':
                                    base_type = self.type_resolver.resolve_type(sub)
                        
                        self.graph.add((shape_uri, SH.datatype, base_type))

                        # Map Constraints to SHACL
                        if constraints:
                            # Pattern
                            if 'patterns' in constraints:
                                for pattern in constraints['patterns']:
                                    self.graph.add((shape_uri, SH.pattern, Literal(pattern)))
                            
                            # Length
                            if 'length' in constraints:
                                l = constraints['length']
                                if 'minLength' in l: 
                                    self.graph.add((shape_uri, SH.minLength, Literal(l['minLength'])))
                                if 'maxLength' in l: 
                                    self.graph.add((shape_uri, SH.maxLength, Literal(l['maxLength'])))
                            
                            # Range
                            if 'range' in constraints:
                                r = constraints['range']
                                if 'min' in r: 
                                    self.graph.add((shape_uri, SH.minInclusive, Literal(r['min'])))
                                if 'max' in r: 
                                    self.graph.add((shape_uri, SH.maxInclusive, Literal(r['max'])))

                        self.typedef_restrictions[typedef_name] = shape_uri
                        self.constraint_count += 1

            log.info(f" Created SHACL shapes for {len(self.typedef_restrictions)} typedefs")

    def _create_datatype_restriction(self, typedef_name: str, constraints: Dict[str, Any], stmt: Any) -> None:

        """Create OWL Datatype Restriction for a typedef"""

        base_type = XSD.string

        if hasattr(stmt, 'substmts'):

            for sub in stmt.substmts:

                if hasattr(sub, 'keyword'):

                    if sub.keyword == 'type':

                        if self._is_enumeration_type(sub):

                            log.debug(f" Skipping typedef '{typedef_name}' - is enumeration (handled separately)")

                            return

                        else:

                            base_type = self.type_resolver.resolve_type(sub)

        # Create restriction URI

        restriction_uri = self.ex[f"typedef/{typedef_name}/restriction"]

        # Create the restriction class

        self.graph.add((restriction_uri, RDF.type, RDFS.Datatype))

        self.graph.add((restriction_uri, OWL.onDatatype, base_type))

        self.graph.add((restriction_uri, RDFS.label, Literal(f"{typedef_name} Restriction")))

        # Get description if available

        description = None

        if hasattr(stmt, 'substmts'):

            for sub in stmt.substmts:

                if hasattr(sub, 'keyword') and sub.keyword == 'description':

                    description = sub.arg if hasattr(sub, 'arg') else None

                    break

        if description:

            self.graph.add((restriction_uri, RDFS.comment, Literal(description)))

        # Add range constraints

        if 'range' in constraints and isinstance(constraints['range'], dict):

            range_info = constraints['range']

            if 'min' in range_info:

                min_restriction_uri = self.ex[f"typedef/{typedef_name}/minInclusive"]

                self.graph.add((min_restriction_uri, XSD.minInclusive, Literal(range_info['min'])))

                self.constraint_count += 1

            if 'max' in range_info:

                max_restriction_uri = self.ex[f"typedef/{typedef_name}/maxInclusive"]

                self.graph.add((max_restriction_uri, XSD.maxInclusive, Literal(range_info['max'])))

                self.constraint_count += 1

        # Add length constraints

        if 'length' in constraints and isinstance(constraints['length'], dict):

            length_info = constraints['length']

            if 'minLength' in length_info:

                min_len_uri = self.ex[f"typedef/{typedef_name}/minLength"]

                self.graph.add((min_len_uri, XSD.minLength, Literal(length_info['minLength'])))

                self.constraint_count += 1

            if 'maxLength' in length_info:

                max_len_uri = self.ex[f"typedef/{typedef_name}/maxLength"]

                self.graph.add((max_len_uri, XSD.maxLength, Literal(length_info['maxLength'])))

                self.constraint_count += 1

        # Add pattern constraints

        if 'patterns' in constraints and isinstance(constraints['patterns'], list):

            for idx, pattern in enumerate(constraints['patterns']):

                if pattern:

                    pattern_uri = self.ex[f"typedef/{typedef_name}/pattern_{idx}"]

                    self.graph.add((pattern_uri, XSD.pattern, Literal(pattern)))

                    self.constraint_count += 1

        # Link typedef to restriction

        typedef_uri = self.ex[f"typedef/{typedef_name}"]

        self.graph.add((typedef_uri, RDF.type, OWL.Class))

        self.graph.add((typedef_uri, RDFS.label, Literal(typedef_name)))

        self.graph.add((typedef_uri, RDFS.subClassOf, restriction_uri))

        self.typedef_restrictions[typedef_name] = restriction_uri

        self.triple_count += 4

        log.debug(f" Created OWL Restriction for typedef: {typedef_name}")

    def _process_enumerations(self) -> None:

        """Process enumeration types and create OWL individuals"""

        log.info(" Processing enumeration types as OWL individuals...")

        enum_types_found = 0

        for module_name, module in self.resolver.modules.items():

            if not hasattr(module, 'substmts'):

                continue

            for stmt in module.substmts:

                if not hasattr(stmt, 'keyword'):

                    continue

                if stmt.keyword == 'typedef' and hasattr(stmt, 'arg'):

                    typedef_name = stmt.arg

                    if hasattr(stmt, 'substmts'):

                        for sub in stmt.substmts:

                            if hasattr(sub, 'keyword') and sub.keyword == 'type':

                                if self._is_enumeration_type(sub):

                                    enum_count = self._create_enumeration_class(typedef_name, sub)

                                    enum_types_found += 1

                                    log.debug(f" Created enum type '{typedef_name}' with {enum_count} values")

                                break

        log.info(f" Total enumeration types processed: {enum_types_found}")

        log.info(f" Created {self.enumeration_count} enumeration individuals")

    def _create_enumeration_class(self, enum_type_name: str, type_stmt: Any) -> int:

        """Create OWL class for enumeration type and individuals for each enum value"""

        enum_count = 0

        # Create the enumeration type class

        enum_type_uri = self.ex[f"types/{enum_type_name}"]

        self.graph.add((enum_type_uri, RDF.type, OWL.Class))

        self.graph.add((enum_type_uri, RDFS.label, Literal(enum_type_name)))

        self.triple_count += 2

        log.debug(f" Created enum class: {enum_type_name}")

        # Extract and create individuals for each enum value

        if hasattr(type_stmt, 'substmts'):

            for enum_sub in type_stmt.substmts:

                if hasattr(enum_sub, 'keyword') and enum_sub.keyword == 'enum':

                    enum_val = enum_sub.arg if hasattr(enum_sub, 'arg') else ''

                    if enum_val:

                        # Create individual URI

                        individual_uri = self.ex[f"types/{enum_type_name}/{enum_val}"]

                        # Add individual triples

                        self.graph.add((individual_uri, RDF.type, OWL.NamedIndividual))

                        self.graph.add((individual_uri, RDF.type, enum_type_uri))

                        self.graph.add((individual_uri, RDFS.label, Literal(enum_val)))

                        self.triple_count += 3

                        enum_count += 1

                        log.debug(f" Created individual: {enum_val}")

                        # Extract description if available

                        if hasattr(enum_sub, 'substmts'):

                            for enum_detail in enum_sub.substmts:

                                if hasattr(enum_detail, 'keyword') and enum_detail.keyword == 'description':

                                    if hasattr(enum_detail, 'arg'):

                                        self.graph.add((individual_uri, RDFS.comment, Literal(enum_detail.arg)))

                                        self.triple_count += 1

                                    break

        log.debug(f" Enumeration: {enum_type_name} created with {enum_count} values")

        self.enumeration_count += enum_count

        return enum_count

def main():

    """Main entry point"""

    parser = argparse.ArgumentParser(

        description='Convert YANG modules to OWL RDF ontology with PATH NORMALIZATION',

        formatter_class=argparse.RawDescriptionHelpFormatter,

        epilog='''

EXAMPLES:

Basic usage:

python3 yang2owl_v45.py simap-yang simap-ontology.ttl

With options:

python3 yang2owl_v45.py --yang-dir simap-yang --output simap-ontology.ttl --verbose

Features in v4.5:

- Full path normalization with module names (e.g., /ietf-network/networks/network)

- Consistent leafref XPath matching with absolute paths

- Cross-module augmentation resolution

- Unique node identification across module boundaries

- Enhanced class_paths registry with module context

Features in v4.3:

- Full YANG Grouping Support (uses statements with refine)

- Nested grouping resolution

- Grouping abstract class generation

- Refine statement processing

- Augment within uses handling

- Enumeration types as OWL individuals

- OWL Datatype restrictions

- PROV metadata

'''

    )

    parser.add_argument('yang_dir', nargs='?', default=None,

        help='Directory containing YANG files')

    parser.add_argument('output_file', nargs='?', default=None,

        help='Output RDF/Turtle file')

    parser.add_argument('--yang-dir', dest='yang_dir_opt', default=None,

        help='Directory containing YANG files (overrides positional)')

    parser.add_argument('--output', dest='output_file_opt', default=None,

        help='Output RDF/Turtle file (overrides positional)')

    parser.add_argument('--modules', default='simap-yang.yang',

        help='Main module to process (default: simap-yang.yang)')

    parser.add_argument('--base-uri', default='http://www.huawei.com/ontology/',

        help='Base URI for ontology')

    parser.add_argument('--verbose', action='store_true',

        help='Enable verbose debug logging')

    args = parser.parse_args()

    yang_dir = args.yang_dir_opt or args.yang_dir or 'simap-yang'

    output_file = args.output_file_opt or args.output_file or 'simap-ontology.ttl'

    output_path = Path(output_file)

    if output_path.is_dir():

        log.warning(f"Output path is a directory, creating file inside it")

        output_file = str(output_path / 'simap-ontology.ttl')

    elif str(output_path).endswith('/'):

        output_path.mkdir(parents=True, exist_ok=True)

        output_file = str(output_path / 'simap-ontology.ttl')

    else:

        output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.verbose:

        logging.getLogger().setLevel(logging.DEBUG)

    log.info("Configuration:")

    log.info(f" YANG directory: {yang_dir}")

    log.info(f" Output file: {output_file}")

    log.info(f" Main module: {args.modules}")

    log.info(f" Base URI: {args.base_uri}")

    log.info("")

    converter = YANGToOWL(yang_dir, args.base_uri)

    converter.convert(args.modules, output_file)

if __name__ == "__main__":

    main()
