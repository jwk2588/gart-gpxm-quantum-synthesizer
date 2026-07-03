"""
Diagram-to-Code Inversion Agent — GART v3.0.

INVERSION of code-to-diagram: Mermaid specifications -> Python code.

Three operating modes:
    1. single:      One diagram -> one Python module
    2. hive_mind:   Multiple diagrams merged, voted on conflicts
    3. agent_swarm: Parallel class generation per diagram

Pipeline:
    Mermaid Source -> Lexer -> Parser -> AST -> Analysis -> Code Generation

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token & AST data structures
# ---------------------------------------------------------------------------


@dataclass
class Token:
    """A lexical token from Mermaid source.

    Attributes:
        token_type: Token type (KEYWORD, IDENTIFIER, SYMBOL, etc.).
        value: Token text value.
        line: Source line number.
        column: Source column number.
    """

    token_type: str
    value: str
    line: int = 0
    column: int = 0


@dataclass
class ASTNode:
    """Base AST node for Mermaid parse tree.

    Attributes:
        node_type: Node type string.
        node_id: Unique node identifier.
        attributes: Node attributes dictionary.
        line_number: Source line number.
    """

    node_type: str
    node_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    line_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "attributes": self.attributes,
            "line_number": self.line_number,
        }


@dataclass
class ClassNode(ASTNode):
    """AST node representing a class definition.

    Attributes:
        class_name: Python class name.
        stereotype: Mermaid stereotype (<<module>>, <<abstract>>, etc.).
        members: Class member definitions.
        methods: Method node definitions.
        base_classes: Inheritance parents.
    """

    class_name: str = ""
    stereotype: str = ""
    members: List[Dict[str, Any]] = field(default_factory=list)
    methods: List[Dict[str, Any]] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.node_id:
            self.node_id = self.class_name


@dataclass
class MethodNode(ASTNode):
    """AST node representing a method definition.

    Attributes:
        method_name: Method name.
        visibility: Visibility modifier (+, -, #).
        parameters: Parameter list as dicts.
        return_type: Return type annotation.
        is_abstract: Whether method is abstract.
        is_static: Whether method is static.
    """

    method_name: str = ""
    visibility: str = "+"
    parameters: List[Dict[str, str]] = field(default_factory=list)
    return_type: str = "None"
    is_abstract: bool = False
    is_static: bool = False
    is_classmethod: bool = False
    is_property: bool = False


@dataclass
class Edge:
    """AST edge representing relationships between nodes.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        edge_type: Relationship type (--|>, *--, -->, etc.).
        label: Edge label text.
    """

    source: str
    target: str
    edge_type: str = ""
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "label": self.label,
        }


@dataclass
class MermaidAST:
    """Complete Abstract Syntax Tree for a Mermaid diagram.

    Attributes:
        nodes: List of AST nodes (classes).
        edges: List of relationship edges.
        diagram_type: Type of diagram (classDiagram, flowchart, etc.).
        metadata: Additional AST metadata.
    """

    nodes: List[ClassNode] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    diagram_type: str = "classDiagram"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_classes(self) -> List[ClassNode]:
        """Get all class nodes."""
        return [n for n in self.nodes if isinstance(n, ClassNode)]

    def get_relationships(self) -> List[Edge]:
        """Get all relationship edges."""
        return self.edges

    def get_node(self, node_id: str) -> Optional[ClassNode]:
        """Get a node by ID."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None


# ---------------------------------------------------------------------------
# Mermaid Lexer
# ---------------------------------------------------------------------------


class MermaidLexer:
    """Lexical analyzer for Mermaid class diagrams.

    Tokenizes Mermaid source into a list of Token objects.
    """

    TOKEN_PATTERNS: List[Tuple[str, str]] = [
        ("COMMENT", r"%%.*"),
        ("CLASS", r"class\s+"),
        ("DIAGRAM_TYPE", r"classDiagram|flowchart|stateDiagram"),
        ("DIRECTION", r"direction\s+(TB|BT|LR|RL)"),
        ("STEREOTYPE", r"<<[^>]+>>"),
        ("ARROW", r"--\|>|\*--|--\>|\.->|o--|\|<--\|"),
        ("COLON", r":"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("IDENTIFIER", r"[A-Za-z_][A-Za-z0-9_]*"),
        ("WHITESPACE", r"\s+"),
        ("STRING", r'"[^"]*"'),
        ("SYMBOL", r"[\*\|~#+\-<>\^]"),
        ("NUMBER", r"\d+\.?\d*"),
        ("UNKNOWN", r"."),
    ]

    def __init__(self) -> None:
        self._compiled_pattern = re.compile(
            "|".join(f"(?P<{name}>{pattern})" for name, pattern in self.TOKEN_PATTERNS),
            re.MULTILINE,
        )

    def tokenize(self, source: str) -> List[Token]:
        """Tokenize Mermaid source code.

        Args:
            source: Raw Mermaid source string.

        Returns:
            List of Token objects.
        """
        tokens: List[Token] = []
        for match in self._compiled_pattern.finditer(source):
            token_type = match.lastgroup or "UNKNOWN"
            value = match.group()
            if token_type == "WHITESPACE":
                continue
            tokens.append(Token(
                token_type=token_type,
                value=value,
                line=source[:match.start()].count("\n") + 1,
            ))
        return tokens


# ---------------------------------------------------------------------------
# Mermaid AST Parser
# ---------------------------------------------------------------------------


class MermaidASTParser:
    """Parser that builds an AST from tokenized Mermaid source.

    Handles class definitions, method signatures, and relationship
    declarations in classDiagram syntax.
    """

    def __init__(self) -> None:
        self.lexer = MermaidLexer()

    def parse(self, mermaid_source: str) -> MermaidAST:
        """Parse Mermaid source into an AST.

        Args:
            mermaid_source: Raw Mermaid diagram source.

        Returns:
            MermaidAST with nodes and edges.
        """
        tokens = self.lexer.tokenize(mermaid_source)
        return self._build_ast(tokens)

    def _build_ast(self, tokens: List[Token]) -> MermaidAST:
        """Build AST from token list.

        Args:
            tokens: Tokenized source.

        Returns:
            MermaidAST.
        """
        ast = MermaidAST()
        i = 0

        while i < len(tokens):
            token = tokens[i]

            # Detect diagram type
            if token.token_type == "DIAGRAM_TYPE":
                ast.diagram_type = token.value.strip()
                i += 1
                continue

            # Parse class definition
            if token.token_type == "IDENTIFIER" and i + 1 < len(tokens):
                class_name = token.value
                i = self._parse_class_or_edge(ast, tokens, i)
                continue

            # Parse relationship arrows
            if token.token_type == "IDENTIFIER" and i + 1 < len(tokens):
                peek = tokens[i + 1]
                if peek.token_type == "ARROW":
                    i = self._parse_edge(ast, tokens, i)
                    continue

            i += 1

        return ast

    def _parse_class_or_edge(self, ast: MermaidAST, tokens: List[Token], idx: int) -> int:
        """Parse a class definition starting at given index.

        Args:
            ast: The AST being built.
            tokens: Full token list.
            idx: Current index.

        Returns:
            Next index after the class/edge.
        """
        if idx >= len(tokens):
            return idx

        name = tokens[idx].value
        idx += 1

        # Check for stereotype
        stereotype = ""
        if idx < len(tokens) and tokens[idx].token_type == "STEREOTYPE":
            stereotype = tokens[idx].value.strip("<>")
            idx += 1

        # Check for brace-enclosed class body
        if idx < len(tokens) and tokens[idx].value == "{":
            idx += 1  # skip {
            node = ClassNode(
                node_type="class",
                node_id=name,
                class_name=name,
                stereotype=stereotype,
            )

            # Parse members until }
            while idx < len(tokens) and tokens[idx].value != "}":
                idx = self._parse_class_member(node, tokens, idx)

            if idx < len(tokens) and tokens[idx].value == "}":
                idx += 1  # skip }

            ast.nodes.append(node)
            return idx

        # Check for relationship arrow (single-line: A --|> B)
        if idx < len(tokens) and tokens[idx].token_type == "ARROW":
            arrow = tokens[idx].value
            idx += 1
            if idx < len(tokens) and tokens[idx].token_type == "IDENTIFIER":
                target = tokens[idx].value
                ast.edges.append(Edge(
                    source=name, target=target, edge_type=arrow,
                ))
                idx += 1
            return idx

        # Simple class declaration without body
        ast.nodes.append(ClassNode(
            node_type="class",
            node_id=name,
            class_name=name,
            stereotype=stereotype,
        ))
        return idx

    def _parse_class_member(
        self, node: ClassNode, tokens: List[Token], idx: int,
    ) -> int:
        """Parse a single class member (attribute or method).

        Args:
            node: Class node being populated.
            tokens: Full token list.
            idx: Current index.

        Returns:
            Next index.
        """
        if idx >= len(tokens):
            return idx

        # Collect tokens until semicolon or newline indicator
        member_tokens: List[Token] = []
        while idx < len(tokens) and tokens[idx].value not in ("}", ";"):
            member_tokens.append(tokens[idx])
            idx += 1

        if idx < len(tokens) and tokens[idx].value == ";":
            idx += 1

        self._process_member_tokens(node, member_tokens)
        return idx

    def _process_member_tokens(self, node: ClassNode, tokens: List[Token]) -> None:
        """Process member tokens into method or attribute definitions.

        Args:
            node: Target class node.
            tokens: Member tokens.
        """
        if not tokens:
            return

        # Check if it's a method (has parentheses)
        has_parens = any(t.token_type in ("LPAREN", "RPAREN") for t in tokens)

        if has_parens:
            self._parse_method(node, tokens)
        else:
            # Attribute
            text = " ".join(t.value for t in tokens)
            node.members.append({"text": text, "type": "attribute"})

    def _parse_method(self, node: ClassNode, tokens: List[Token]) -> None:
        """Parse method tokens into a MethodNode.

        Args:
            node: Parent class node.
            tokens: Method tokens.
        """
        visibility = "+"
        method_name = ""
        params: List[Dict[str, str]] = []
        return_type = "None"

        i = 0
        # Visibility
        if i < len(tokens) and tokens[i].value in "+-#~":
            visibility = tokens[i].value
            i += 1

        # Method name
        if i < len(tokens) and tokens[i].token_type == "IDENTIFIER":
            method_name = tokens[i].value
            i += 1

        # Parameters
        if i < len(tokens) and tokens[i].token_type == "LPAREN":
            i += 1
            param_text = ""
            while i < len(tokens) and tokens[i].token_type != "RPAREN":
                param_text += tokens[i].value
                i += 1
            if param_text:
                for p in param_text.split(","):
                    p = p.strip()
                    if p:
                        parts = p.split(":")
                        params.append({
                            "name": parts[0].strip(),
                            "type": parts[1].strip() if len(parts) > 1 else "Any",
                        })
            i += 1  # skip RPAREN

        # Return type
        if i < len(tokens) and tokens[i].value == ":":
            i += 1
            if i < len(tokens):
                return_type = tokens[i].value

        node.methods.append({
            "name": method_name,
            "visibility": visibility,
            "parameters": params,
            "return_type": return_type,
        })

    def _parse_edge(self, ast: MermaidAST, tokens: List[Token], idx: int) -> int:
        """Parse a relationship edge.

        Args:
            ast: The AST.
            tokens: Token list.
            idx: Current index (at source identifier).

        Returns:
            Next index.
        """
        source = tokens[idx].value
        idx += 1

        if idx < len(tokens) and tokens[idx].token_type == "ARROW":
            arrow = tokens[idx].value
            idx += 1

            if idx < len(tokens) and tokens[idx].token_type == "IDENTIFIER":
                target = tokens[idx].value
                ast.edges.append(Edge(
                    source=source, target=target, edge_type=arrow,
                ))
                idx += 1

        return idx


# ---------------------------------------------------------------------------
# DependencyMapper
# ---------------------------------------------------------------------------


class DependencyMapper:
    """Maps Mermaid relationships to Python code structures.

    Extracts inheritance (--|>), composition (*--), and association
    (-->) relationships for code generation.
    """

    def __init__(self) -> None:
        self.inheritance_map: Dict[str, str] = {}
        self.composition_map: Dict[str, List[str]] = {}
        self.association_map: Dict[str, List[str]] = {}

    def map_dependencies(self, ast: MermaidAST) -> Dict[str, Any]:
        """Map all relationships from AST.

        Args:
            ast: MermaidAST with edges.

        Returns:
            Dictionary with inheritance, composition, association maps.
        """
        for edge in ast.edges:
            if "|>" in edge.edge_type:
                # Inheritance: source inherits from target
                self.inheritance_map[edge.source] = edge.target
            elif "*" in edge.edge_type:
                # Composition: source contains target
                self.composition_map.setdefault(edge.source, []).append(edge.target)
            elif "--" in edge.edge_type or "->" in edge.edge_type:
                # Association
                self.association_map.setdefault(edge.source, []).append(edge.target)

        return {
            "inheritance": self.inheritance_map,
            "composition": self.composition_map,
            "association": self.association_map,
        }

    def get_parents(self, class_name: str) -> List[str]:
        """Get parent classes for a given class.

        Args:
            class_name: Class to look up.

        Returns:
            List of parent class names.
        """
        parents: List[str] = []
        current = class_name
        while current in self.inheritance_map:
            parent = self.inheritance_map[current]
            parents.append(parent)
            current = parent
        return parents

    def topological_sort(self, ast: MermaidAST) -> List[str]:
        """Topological sort of classes (parents before children).

        Args:
            ast: MermaidAST.

        Returns:
            Sorted list of class names.
        """
        self.map_dependencies(ast)
        in_degree: Dict[str, int] = {n.class_name: 0 for n in ast.nodes}

        for child, parent in self.inheritance_map.items():
            if parent in in_degree:
                in_degree[child] = in_degree.get(child, 0) + 1

        queue = [n for n, d in in_degree.items() if d == 0]
        sorted_names: List[str] = []

        while queue:
            name = queue.pop(0)
            sorted_names.append(name)
            for child, parent in self.inheritance_map.items():
                if parent == name and child in in_degree:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        # Add any remaining
        for n in ast.nodes:
            if n.class_name not in sorted_names:
                sorted_names.append(n.class_name)

        return sorted_names


# ---------------------------------------------------------------------------
# PythonCodeGenerator
# ---------------------------------------------------------------------------


class PythonCodeGenerator:
    """Generates Python code from Mermaid AST nodes.

    Supports dataclass generation, abstract classes, and method
    stubs with type hints and docstrings.
    """

    TYPE_MAP: Dict[str, str] = {
        "str": "str",
        "string": "str",
        "int": "int",
        "float": "float",
        "bool": "bool",
        "list": "List[Any]",
        "dict": "Dict[str, Any]",
        "none": "None",
        "tensor": "torch.Tensor",
        "ndarray": "np.ndarray",
    }

    def __init__(self) -> None:
        self.indent_size = 4

    def generate_class(self, node: ClassNode, dep_mapper: DependencyMapper) -> str:
        """Generate Python class code from a ClassNode.

        Args:
            node: AST class node.
            dep_mapper: Dependency mapper for inheritance.

        Returns:
            Python class source code.
        """
        lines: List[str] = []
        indent = " " * self.indent_size

        # Determine base classes
        bases: List[str] = []
        parents = dep_mapper.get_parents(node.class_name)
        for p in parents:
            if not p.startswith(("<", "\n")):
                bases.append(p)

        if node.stereotype == "abstract":
            bases.insert(0, "ABC")

        # Class declaration
        base_str = f"({', '.join(bases)})" if bases else ""
        lines.append(f"class {node.class_name}{base_str}:")

        # Docstring
        lines.append(
            f'{indent}"""{node.class_name} — generated from Mermaid specification."""'
        )

        # Stereotype comment
        if node.stereotype:
            lines.append(f"{indent}# Mermaid stereotype: {node.stereotype}")

        # Generate __init__
        lines.append("")
        lines.extend(self._generate_init(node, indent))

        # Generate methods
        for method in node.methods:
            lines.append("")
            lines.extend(self._generate_method(method, indent))

        # If no methods, add pass
        if not node.members and not node.methods:
            lines.append(f"{indent}pass")

        return "\n".join(lines)

    def _generate_init(self, node: ClassNode, indent: str) -> List[str]:
        """Generate __init__ method.

        Args:
            node: Class node.
            indent: Indentation string.

        Returns:
            Lines of __init__ code.
        """
        lines: List[str] = []
        lines.append(f"{indent}def __init__(self) -> None:")

        if node.members:
            for member in node.members:
                text = member.get("text", "")
                if ":" in text:
                    name, type_str = text.split(":", 1)
                    py_type = self.TYPE_MAP.get(type_str.strip().lower(), type_str.strip())
                    lines.append(f"{indent}{indent}self.{name.strip()}: {py_type} = None")
                else:
                    lines.append(f"{indent}{indent}self.{text.strip()} = None")
        else:
            lines.append(f"{indent}{indent}pass")

        return lines

    def _generate_method(self, method: Dict[str, Any], indent: str) -> List[str]:
        """Generate a method stub.

        Args:
            method: Method definition dictionary.
            indent: Indentation string.

        Returns:
            Lines of method code.
        """
        lines: List[str] = []
        name = method.get("name", "unknown")
        params = method.get("parameters", [])
        return_type = method.get("return_type", "None")

        # Build parameter string
        param_strs = ["self"]
        for p in params:
            p_name = p.get("name", "arg")
            p_type = self.TYPE_MAP.get(p.get("type", "").lower(), p.get("type", "Any"))
            param_strs.append(f"{p_name}: {p_type}")

        py_return = self.TYPE_MAP.get(return_type.lower(), return_type)

        lines.append(f"{indent}def {name}({', '.join(param_strs)}) -> {py_return}:")
        lines.append(f'{indent}{indent}"""{name} method."""')
        lines.append(f"{indent}{indent}pass")

        return lines

    def generate_imports(self, classes: List[ClassNode]) -> str:
        """Generate import statements.

        Args:
            classes: List of class nodes.

        Returns:
            Import statements string.
        """
        imports = [
            "from __future__ import annotations",
            "",
            "from abc import ABC, abstractmethod",
            "from dataclasses import dataclass, field",
            "from typing import Any, Dict, List, Optional, Protocol, Tuple",
            "",
        ]

        # Check if we need torch/numpy
        for node in classes:
            for method in node.methods:
                for p in method.get("parameters", []):
                    p_type = p.get("type", "").lower()
                    if p_type in ("tensor", "torch.Tensor"):
                        imports.insert(-1, "import torch")
                    if p_type in ("ndarray", "np.ndarray"):
                        imports.insert(-1, "import numpy as np")

        return "\n".join(imports)


# ---------------------------------------------------------------------------
# Main Inversion Agent
# ---------------------------------------------------------------------------


class DiagramToCodeInversionAgent:
    """Diagram-to-Code Inversion Agent.

    INVERSION of code-to-diagram: Converts Mermaid specifications
    into Python code. Supports three modes:
        - single: One diagram at a time
        - hive_mind: Merge multiple diagrams, vote on conflicts
        - agent_swarm: Parallel processing per diagram

    Attributes:
        mode: Operating mode ("single", "hive_mind", "agent_swarm").
        parser: MermaidASTParser instance.
        generator: PythonCodeGenerator instance.
        dependency_mapper: DependencyMapper instance.
    """

    def __init__(self, mode: str = "single") -> None:
        if mode not in ("single", "hive_mind", "agent_swarm"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        self.parser = MermaidASTParser()
        self.generator = PythonCodeGenerator()
        self.dependency_mapper = DependencyMapper()

    def generate_classes(self, mermaid_source: str) -> List[str]:
        """Generate Python class code from Mermaid source.

        Args:
            mermaid_source: Mermaid diagram source.

        Returns:
            List of Python class source strings.
        """
        ast = self.parser.parse(mermaid_source)
        dep_map = self.dependency_mapper.map_dependencies(ast)

        sorted_names = self.dependency_mapper.topological_sort(ast)
        name_to_node = {n.class_name: n for n in ast.nodes}

        results: List[str] = []
        for name in sorted_names:
            node = name_to_node.get(name)
            if node:
                code = self.generator.generate_class(node, self.dependency_mapper)
                results.append(code)

        return results

    def generate_inversion_output(self, diagram_paths: List[str]) -> Dict[str, str]:
        """Generate Python code from multiple diagram files.

        Mode-dependent behavior:
            single: Process first diagram only
            hive_mind: Merge all diagrams, resolve conflicts
            agent_swarm: Parallel processing per diagram

        Args:
            diagram_paths: List of file paths to Mermaid diagrams.

        Returns:
            Dict mapping filenames to Python source code.
        """
        if self.mode == "single":
            return self._mode_single(diagram_paths)
        elif self.mode == "hive_mind":
            return self._mode_hive_mind(diagram_paths)
        else:
            return self._mode_agent_swarm(diagram_paths)

    def _mode_single(self, diagram_paths: List[str]) -> Dict[str, str]:
        """Single-diagram mode.

        Args:
            diagram_paths: Diagram file paths.

        Returns:
            Dict with one generated module.
        """
        if not diagram_paths:
            return {}

        with open(diagram_paths[0], "r") as f:
            source = f.read()

        classes = self.generate_classes(source)
        imports = self.generator.generate_imports(
            self.parser.parse(source).nodes
        )

        module_name = os.path.splitext(os.path.basename(diagram_paths[0]))[0]
        code = f"{imports}\n\n" + "\n\n".join(classes)

        return {f"{module_name}.py": code}

    def _mode_hive_mind(self, diagram_paths: List[str]) -> Dict[str, str]:
        """Hive mind mode: merge all diagrams, vote on conflicts.

        Args:
            diagram_paths: Diagram file paths.

        Returns:
            Dict with merged generated modules.
        """
        all_classes: Dict[str, str] = {}
        all_nodes: List[Any] = []

        for path in diagram_paths:
            with open(path, "r") as f:
                source = f.read()
            ast = self.parser.parse(source)
            all_nodes.extend(ast.nodes)

            classes = self.generate_classes(source)
            for i, cls_code in enumerate(classes):
                if i < len(ast.nodes):
                    name = ast.nodes[i].class_name
                    if name in all_classes:
                        # Conflict: keep the longer/more detailed version
                        if len(cls_code) > len(all_classes[name]):
                            all_classes[name] = cls_code
                    else:
                        all_classes[name] = cls_code

        imports = self.generator.generate_imports(all_nodes)
        code = f"{imports}\n\n" + "\n\n".join(all_classes.values())

        return {"merged_module.py": code}

    def _mode_agent_swarm(self, diagram_paths: List[str]) -> Dict[str, str]:
        """Agent swarm mode: parallel class generation per diagram.

        Args:
            diagram_paths: Diagram file paths.

        Returns:
            Dict with one module per diagram.
        """
        results: Dict[str, str] = {}

        for path in diagram_paths:
            with open(path, "r") as f:
                source = f.read()

            classes = self.generate_classes(source)
            imports = self.generator.generate_imports(
                self.parser.parse(source).nodes
            )

            module_name = os.path.splitext(os.path.basename(path))[0]
            code = f"{imports}\n\n" + "\n\n".join(classes)
            results[f"{module_name}.py"] = code

        return results

    def parse_mermaid(self, mermaid_source: str) -> MermaidAST:
        """Parse Mermaid source to AST.

        Args:
            mermaid_source: Mermaid diagram text.

        Returns:
            MermaidAST.
        """
        return self.parser.parse(mermaid_source)

    def emit_python(self, ast: MermaidAST) -> str:
        """Generate Python code from AST.

        Args:
            ast: MermaidAST.

        Returns:
            Python source code.
        """
        self.dependency_mapper = DependencyMapper()
        dep_map = self.dependency_mapper.map_dependencies(ast)

        sorted_names = self.dependency_mapper.topological_sort(ast)
        name_to_node = {n.class_name: n for n in ast.nodes}

        classes: List[str] = []
        for name in sorted_names:
            node = name_to_node.get(name)
            if node:
                classes.append(self.generator.generate_class(node, self.dependency_mapper))

        imports = self.generator.generate_imports(ast.nodes)
        return f"{imports}\n\n" + "\n\n".join(classes)

    def validate_output(self, python_code: str) -> Dict[str, Any]:
        """Validate generated Python code.

        Args:
            python_code: Generated Python source.

        Returns:
            Validation result dictionary.
        """
        issues: List[str] = []

        # Check for syntax issues
        if python_code.count("class ") == 0:
            issues.append("No class definitions found")

        # Check for balanced parentheses/braces
        for pair in [("(", ")"), ("[", "]"), ("{", "}")]:
            if python_code.count(pair[0]) != python_code.count(pair[1]):
                issues.append(f"Unbalanced {pair[0]}{pair[1]}")

        # Check indentation
        lines = python_code.split("\n")
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(" ") and not line.startswith("from "):
                if line.strip() not in ("", "pass") and not line.startswith("class "):
                    if i > 0 and not line.startswith("import "):
                        issues.append(f"Possible indentation issue at line {i + 1}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "line_count": len(lines),
            "class_count": python_code.count("class "),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Diagram-to-Code Inversion Agent loaded successfully.")

    # Demo with inline Mermaid
    sample_mermaid = """
classDiagram
    class Animal {
        +str name
        +int age
        +make_sound() str
    }
    class Dog {
        +str breed
        +fetch() None
    }
    Animal <|-- Dog
"""
    agent = DiagramToCodeInversionAgent(mode="single")
    classes = agent.generate_classes(sample_mermaid)
    print(f"\nGenerated {len(classes)} classes:")
    for cls in classes:
        print(f"---\n{cls[:200]}...")
