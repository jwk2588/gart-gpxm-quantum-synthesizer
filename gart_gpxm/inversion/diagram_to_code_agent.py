"""
Diagram-to-Code Agent — Inversion Engine for GART v3.0.

Converts visual system diagrams into executable Python code through
a multi-stage inversion pipeline: parse, plan, generate, verify.

Components:
    - DiagramParser: Parse diagram descriptions into structured AST
    - ArchitecturePlanner: Plan code structure from parsed diagrams
    - CodeGenerator: Generate Python code from architecture plans
    - InversionVerifier: Verify generated code matches diagram
    - DiagramToCodeAgent: Main orchestrator

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InversionError(Exception):
    """Base exception for diagram-to-code inversion."""


class ParseError(InversionError):
    """Raised when diagram parsing fails."""


class GenerationError(InversionError):
    """Raised when code generation fails."""


class VerificationError(InversionError):
    """Raised when generated code verification fails."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DiagramElementType(Enum):
    """Types of elements in a system diagram."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    INTERFACE = "interface"
    DATABASE = "database"
    API = "api"
    QUEUE = "queue"
    CACHE = "cache"
    SERVICE = "service"
    CONTROLLER = "controller"
    MIDDLEWARE = "middleware"


class ConnectionType(Enum):
    """Types of connections between diagram elements."""

    CALLS = "calls"
    IMPORTS = "imports"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    USES = "uses"
    EMITS = "emits"
    LISTENS = "listens"
    READS = "reads"
    WRITES = "writes"


class CodeTarget(Enum):
    """Target code generation platforms."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class DiagramElement:
    """An element in a system diagram.

    Attributes:
        element_id: Unique identifier.
        name: Human-readable name.
        element_type: Type of element.
        properties: Key-value properties.
        methods: List of method signatures.
        x: X position in diagram.
        y: Y position in diagram.
    """

    element_id: str
    name: str
    element_type: DiagramElementType
    properties: Dict[str, Any] = field(default_factory=dict)
    methods: List[Dict[str, Any]] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0


@dataclass
class DiagramConnection:
    """A connection between two diagram elements.

    Attributes:
        source_id: Source element ID.
        target_id: Target element ID.
        connection_type: Type of connection.
        label: Optional connection label.
    """

    source_id: str
    target_id: str
    connection_type: ConnectionType
    label: str = ""


@dataclass
class DiagramAST:
    """Abstract syntax tree representation of a diagram.

    Attributes:
        elements: All diagram elements.
        connections: All connections.
        metadata: Diagram metadata.
    """

    elements: List[DiagramElement] = field(default_factory=list)
    connections: List[DiagramConnection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_element(self, element_id: str) -> Optional[DiagramElement]:
        """Get an element by ID.

        Args:
            element_id: Element to find.

        Returns:
            DiagramElement or None.
        """
        for e in self.elements:
            if e.element_id == element_id:
                return e
        return None

    def get_connections_from(
        self,
        element_id: str,
    ) -> List[DiagramConnection]:
        """Get all connections from an element.

        Args:
            element_id: Source element.

        Returns:
            List of outgoing connections.
        """
        return [c for c in self.connections if c.source_id == element_id]

    def get_connections_to(
        self,
        element_id: str,
    ) -> List[DiagramConnection]:
        """Get all connections to an element.

        Args:
            element_id: Target element.

        Returns:
            List of incoming connections.
        """
        return [c for c in self.connections if c.target_id == element_id]


@dataclass
class ArchitecturePlan:
    """Planned code architecture from a diagram.

    Attributes:
        modules: Planned modules.
        classes: Planned classes.
        interfaces: Planned interfaces.
        dependencies: Module dependencies.
    """

    modules: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    interfaces: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class GeneratedCode:
    """Generated code artifact.

    Attributes:
        filename: Target filename.
        code: Generated source code.
        language: Target language.
        element_map: Maps diagram elements to code locations.
        hash: Content hash.
    """

    filename: str
    code: str
    language: CodeTarget
    element_map: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    hash: str = ""

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = hashlib.sha256(self.code.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# DiagramParser
# ---------------------------------------------------------------------------


class DiagramParser:
    """Parse diagram descriptions into structured AST.

    Supports multiple input formats: textual DSL, JSON, and
    pseudo-UML notation.
    """

    def __init__(self) -> None:
        self._element_counter = 0

    def parse_text(self, description: str) -> DiagramAST:
        """Parse a textual diagram description.

        Format:
            module MyModule at (100, 200)
            class MyClass at (300, 200)
            MyModule -> MyClass [calls]

        Args:
            description: Textual diagram description.

        Returns:
            Parsed DiagramAST.

        Raises:
            ParseError: If parsing fails.
        """
        ast_result = DiagramAST()
        lines = [l.strip() for l in description.split("\n") if l.strip()]

        for line in lines:
            # Skip comments
            if line.startswith("#"):
                continue

            # Parse element definition
            element_match = re.match(
                r"(\w+)\s+(\w+)\s+at\s+\((\d+),\s*(\d+)\)",
                line,
            )
            if element_match:
                elem_type_str, name, x, y = element_match.groups()
                elem_type = self._parse_element_type(elem_type_str)
                elem_id = f"{name}_{self._element_counter}"
                self._element_counter += 1

                elem = DiagramElement(
                    element_id=elem_id,
                    name=name,
                    element_type=elem_type,
                    x=float(x),
                    y=float(y),
                )
                ast_result.elements.append(elem)
                continue

            # Parse connection
            conn_match = re.match(
                r"(\w+)\s*->\s*(\w+)\s*\[([^\]]+)\](?:\s*"(.+?)")?",
                line,
            )
            if conn_match:
                source, target, conn_type_str = conn_match.groups()[:3]
                label = conn_match.group(4) if conn_match.group(4) else ""
                conn_type = self._parse_connection_type(conn_type_str)

                conn = DiagramConnection(
                    source_id=source,
                    target_id=target,
                    connection_type=conn_type,
                    label=label,
                )
                ast_result.connections.append(conn)
                continue

        return ast_result

    def parse_json(self, json_str: str) -> DiagramAST:
        """Parse a JSON diagram description.

        Args:
            json_str: JSON string.

        Returns:
            Parsed DiagramAST.
        """
        data = json.loads(json_str)
        ast_result = DiagramAST(metadata=data.get("metadata", {}))

        for elem_data in data.get("elements", []):
            elem = DiagramElement(
                element_id=elem_data["id"],
                name=elem_data["name"],
                element_type=DiagramElementType(elem_data["type"]),
                properties=elem_data.get("properties", {}),
                x=elem_data.get("x", 0),
                y=elem_data.get("y", 0),
            )
            ast_result.elements.append(elem)

        for conn_data in data.get("connections", []):
            conn = DiagramConnection(
                source_id=conn_data["source"],
                target_id=conn_data["target"],
                connection_type=ConnectionType(conn_data["type"]),
                label=conn_data.get("label", ""),
            )
            ast_result.connections.append(conn)

        return ast_result

    def _parse_element_type(self, type_str: str) -> DiagramElementType:
        """Parse element type string."""
        try:
            return DiagramElementType(type_str.lower())
        except ValueError:
            return DiagramElementType.MODULE

    def _parse_connection_type(self, type_str: str) -> ConnectionType:
        """Parse connection type string."""
        try:
            return ConnectionType(type_str.lower())
        except ValueError:
            return ConnectionType.CALLS


# ---------------------------------------------------------------------------
# ArchitecturePlanner
# ---------------------------------------------------------------------------


class ArchitecturePlanner:
    """Plan code architecture from parsed diagrams.

    Converts diagram AST into a structured architecture plan
    with modules, classes, interfaces, and dependencies.
    """

    def plan(self, ast: DiagramAST) -> ArchitecturePlan:
        """Create architecture plan from diagram AST.

        Args:
            ast: Parsed diagram.

        Returns:
            Architecture plan.
        """
        plan = ArchitecturePlan()

        for elem in ast.elements:
            if elem.element_type == DiagramElementType.MODULE:
                plan.modules.append({
                    "name": elem.name,
                    "element_id": elem.element_id,
                    "properties": elem.properties,
                })
            elif elem.element_type == DiagramElementType.CLASS:
                plan.classes.append({
                    "name": elem.name,
                    "element_id": elem.element_id,
                    "methods": elem.methods,
                    "properties": elem.properties,
                })
            elif elem.element_type == DiagramElementType.INTERFACE:
                plan.interfaces.append({
                    "name": elem.name,
                    "element_id": elem.element_id,
                    "methods": elem.methods,
                })

        for conn in ast.connections:
            source_elem = ast.get_element(conn.source_id)
            target_elem = ast.get_element(conn.target_id)
            if source_elem and target_elem:
                plan.dependencies.append((source_elem.name, target_elem.name))

        return plan


# ---------------------------------------------------------------------------
# CodeGenerator
# ---------------------------------------------------------------------------


class CodeGenerator:
    """Generate Python code from architecture plans."""

    def __init__(self, target: CodeTarget = CodeTarget.PYTHON) -> None:
        self.target = target

    def generate(self, plan: ArchitecturePlan) -> List[GeneratedCode]:
        """Generate code files from architecture plan.

        Args:
            plan: Architecture plan.

        Returns:
            List of generated code files.
        """
        files: List[GeneratedCode] = []

        # Generate module files
        for module in plan.modules:
            code = self._generate_module(module, plan)
            files.append(GeneratedCode(
                filename=f"{module['name'].lower()}.py",
                code=code,
                language=self.target,
            ))

        # Generate class files
        for class_info in plan.classes:
            code = self._generate_class(class_info, plan)
            files.append(GeneratedCode(
                filename=f"{class_info['name'].lower()}.py",
                code=code,
                language=self.target,
            ))

        # Generate interface file
        if plan.interfaces:
            code = self._generate_interfaces(plan)
            files.append(GeneratedCode(
                filename="interfaces.py",
                code=code,
                language=self.target,
            ))

        return files

    def _generate_module(
        self,
        module: Dict[str, Any],
        plan: ArchitecturePlan,
    ) -> str:
        """Generate module code."""
        lines = [
            f'"""',
            f"{module['name']} module — Auto-generated from diagram.",
            f'"""',
            "",
            "from __future__ import annotations",
            "",
            "import logging",
            "",
            f"logger = logging.getLogger(__name__)",
            "",
            f"# Module: {module['name']}",
            f"# Element ID: {module['element_id']}",
            "",
        ]

        # Add dependencies
        for source, target in plan.dependencies:
            if source == module["name"]:
                lines.append(f"# Depends on: {target}")
        lines.append("")

        code = "\n".join(lines)
        return code

    def _generate_class(
        self,
        class_info: Dict[str, Any],
        plan: ArchitecturePlan,
    ) -> str:
        """Generate class code."""
        lines = [
            f'"""',
            f"{class_info['name']} — Auto-generated from diagram.",
            f'"""',
            "",
            "from __future__ import annotations",
            "",
            "from abc import ABC, abstractmethod",
            "from dataclasses import dataclass, field",
            "from typing import Any, Dict, List, Optional",
            "",
            f"class {class_info['name']}:",
            f'    """{class_info["name"]} class."""',
            "",
            "    def __init__(self) -> None:",
            "        pass",
            "",
        ]

        # Add methods
        for method in class_info.get("methods", []):
            method_name = method.get("name", "method")
            params = ", ".join(method.get("params", ["self"]))
            lines.append(f"    def {method_name}({params}) -> Any:")
            lines.append(f'        """{method.get("doc", "Method.")}"""')
            lines.append(f"        pass")
            lines.append("")

        code = "\n".join(lines)
        return code

    def _generate_interfaces(self, plan: ArchitecturePlan) -> str:
        """Generate interfaces code."""
        lines = [
            '"""',
            "Interfaces — Auto-generated from diagram.",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "from abc import ABC, abstractmethod",
            "from typing import Any, Dict, List, Optional",
            "",
        ]

        for iface in plan.interfaces:
            lines.append(f"class {iface['name']}(ABC):")
            lines.append(f'    """{iface["name"]} interface."""')
            lines.append("")
            for method in iface.get("methods", []):
                method_name = method.get("name", "method")
                params = ", ".join(method.get("params", ["self"]))
                lines.append(f"    @abstractmethod")
                lines.append(f"    def {method_name}({params}) -> Any:")
                lines.append(f'        """{method.get("doc", "Method.")}"""')
                lines.append(f"        ...")
                lines.append("")
            lines.append("")

        code = "\n".join(lines)
        return code


# ---------------------------------------------------------------------------
# InversionVerifier
# ---------------------------------------------------------------------------


class InversionVerifier:
    """Verify that generated code matches the source diagram."""

    def verify(
        self,
        ast: DiagramAST,
        generated_files: List[GeneratedCode],
    ) -> Dict[str, Any]:
        """Verify generated code against diagram.

        Args:
            ast: Source diagram AST.
            generated_files: Generated code files.

        Returns:
            Verification report.
        """
        report = {
            "valid": True,
            "elements_covered": 0,
            "elements_total": len(ast.elements),
            "files_generated": len(generated_files),
            "errors": [],
            "warnings": [],
        }

        # Check syntax of generated code
        for gen_file in generated_files:
            if gen_file.language == CodeTarget.PYTHON:
                try:
                    ast.parse(gen_file.code)
                    report["warnings"].append(
                        f"{gen_file.filename}: Syntax OK"
                    )
                except SyntaxError as e:
                    report["valid"] = False
                    report["errors"].append(
                        f"{gen_file.filename}: Syntax error - {e}"
                    )

        # Check element coverage
        for elem in ast.elements:
            covered = any(
                elem.name.lower() in f.filename or elem.name in f.code
                for f in generated_files
            )
            if covered:
                report["elements_covered"] += 1
            else:
                report["warnings"].append(
                    f"Element '{elem.name}' not covered in generated code"
                )

        return report


# ---------------------------------------------------------------------------
# DiagramToCodeAgent — Main orchestrator
# ---------------------------------------------------------------------------


class DiagramToCodeAgent:
    """Diagram-to-Code Agent for GART v3.0.

    Orchestrates the full inversion pipeline:
        1. Parse diagram description into AST
        2. Plan architecture from AST
        3. Generate code from architecture plan
        4. Verify generated code matches diagram

    Attributes:
        parser: Diagram parser.
        planner: Architecture planner.
        generator: Code generator.
        verifier: Code verifier.
    """

    def __init__(
        self,
        target: CodeTarget = CodeTarget.PYTHON,
    ) -> None:
        self.parser = DiagramParser()
        self.planner = ArchitecturePlanner()
        self.generator = CodeGenerator(target)
        self.verifier = InversionVerifier()

    def invert(
        self,
        diagram_description: str,
        description_format: str = "text",
    ) -> Dict[str, Any]:
        """Convert a diagram description into code.

        Args:
            diagram_description: Diagram description string.
            description_format: Format ("text", "json").

        Returns:
            Inversion result with generated files and report.
        """
        # Step 1: Parse
        if description_format == "json":
            ast = self.parser.parse_json(diagram_description)
        else:
            ast = self.parser.parse_text(diagram_description)

        # Step 2: Plan
        plan = self.planner.plan(ast)

        # Step 3: Generate
        files = self.generator.generate(plan)

        # Step 4: Verify
        verification = self.verifier.verify(ast, files)

        return {
            "ast": {
                "elements": len(ast.elements),
                "connections": len(ast.connections),
            },
            "plan": {
                "modules": len(plan.modules),
                "classes": len(plan.classes),
                "interfaces": len(plan.interfaces),
                "dependencies": len(plan.dependencies),
            },
            "files": [
                {
                    "filename": f.filename,
                    "language": f.language.value,
                    "lines": f.code.count("\n"),
                    "hash": f.hash,
                }
                for f in files
            ],
            "verification": verification,
            "generated_code": {f.filename: f.code for f in files},
        }

    def invert_and_write(
        self,
        diagram_description: str,
        output_dir: str,
        description_format: str = "text",
    ) -> Dict[str, Any]:
        """Convert diagram and write files to disk.

        Args:
            diagram_description: Diagram description.
            output_dir: Output directory path.
            description_format: Format ("text", "json").

        Returns:
            Inversion result.
        """
        import os

        result = self.invert(diagram_description, description_format)

        os.makedirs(output_dir, exist_ok=True)

        for filename, code in result["generated_code"].items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                f.write(code)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "parser_ready": True,
            "planner_ready": True,
            "generator_target": self.generator.target.value,
            "verifier_ready": True,
        }
