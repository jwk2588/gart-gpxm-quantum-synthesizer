"""
Reconstruction Engine — Stash-Box Recovery System for GART v3.0.

Rebuilds system state from distributed code stashes with
integrity verification and conflict resolution.

Components:
    - StashFragment: Individual code fragment
    - StashBox: Fragment storage and indexing
    - ReconstructionEngine: State rebuild engine
    - ConflictResolver: Merge conflict resolution
    - IntegrityChecker: Stash integrity verification

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ReconstructionError(Exception):
    """Base exception for reconstruction operations."""


class ConflictError(ReconstructionError):
    """Raised when conflicting fragments cannot be resolved."""


class IntegrityError(ReconstructionError):
    """Raised when stash integrity check fails."""


class FragmentNotFoundError(ReconstructionError):
    """Raised when a fragment is not found."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FragmentType(Enum):
    """Types of code fragments."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"
    CONFIG = "config"
    DATA = "data"
    INTERFACE = "interface"


class ConflictStrategy(Enum):
    """Strategies for conflict resolution."""

    NEWEST_WINS = "newest_wins"
    OLDEST_WINS = "oldest_wins"
    MERGE = "merge"
    MANUAL = "manual"
    HIGHEST_VERSION = "highest_version"


class ReconstructionStatus(Enum):
    """Status of reconstruction."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class StashFragment:
    """A single code fragment in the stash.

    Attributes:
        fragment_id: Unique identifier.
        fragment_type: Type of fragment.
        source_path: Original source path.
        content: Fragment content.
        version: Version number.
        timestamp: Creation timestamp.
        dependencies: IDs of dependent fragments.
        checksum: Content hash.
    """

    fragment_id: str
    fragment_type: FragmentType
    source_path: str
    content: str
    version: int = 1
    timestamp: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum of content."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify fragment integrity.

        Returns:
            True if content matches checksum.
        """
        return self.checksum == self._compute_checksum()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "fragment_id": self.fragment_id,
            "fragment_type": self.fragment_type.value,
            "source_path": self.source_path,
            "content": self.content,
            "version": self.version,
            "timestamp": self.timestamp,
            "dependencies": list(self.dependencies),
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StashFragment:
        """Deserialize from dictionary."""
        return cls(
            fragment_id=data["fragment_id"],
            fragment_type=FragmentType(data["fragment_type"]),
            source_path=data["source_path"],
            content=data["content"],
            version=data.get("version", 1),
            timestamp=data.get("timestamp", time.time()),
            dependencies=list(data.get("dependencies", [])),
            checksum=data.get("checksum", ""),
        )


@dataclass
class ReconstructionPlan:
    """Plan for reconstructing state from stashes.

    Attributes:
        target_state: Target state identifier.
        fragment_order: Ordered list of fragment IDs.
        expected_checksums: Expected fragment checksums.
        strategy: Conflict resolution strategy.
    """

    target_state: str
    fragment_order: List[str] = field(default_factory=list)
    expected_checksums: Dict[str, str] = field(default_factory=dict)
    strategy: ConflictStrategy = ConflictStrategy.NEWEST_WINS


@dataclass
class ReconstructionResult:
    """Result of a reconstruction operation.

    Attributes:
        status: Reconstruction status.
        fragments_used: Number of fragments used.
        fragments_total: Total fragments available.
        conflicts: Number of conflicts encountered.
        output: Reconstructed output.
        errors: List of error messages.
    """

    status: ReconstructionStatus
    fragments_used: int = 0
    fragments_total: int = 0
    conflicts: int = 0
    output: str = ""
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# StashBox
# ---------------------------------------------------------------------------


class StashBox:
    """Storage and indexing for code fragments.

    Provides CRUD operations for fragments with
    dependency tracking.
    """

    def __init__(self) -> None:
        self._fragments: Dict[str, StashFragment] = {}
        self._index_by_path: Dict[str, Set[str]] = {}
        self._index_by_type: Dict[FragmentType, Set[str]] = {}

    def store(self, fragment: StashFragment) -> None:
        """Store a fragment.

        Args:
            fragment: Fragment to store.
        """
        self._fragments[fragment.fragment_id] = fragment

        # Index by path
        if fragment.source_path not in self._index_by_path:
            self._index_by_path[fragment.source_path] = set()
        self._index_by_path[fragment.source_path].add(fragment.fragment_id)

        # Index by type
        if fragment.fragment_type not in self._index_by_type:
            self._index_by_type[fragment.fragment_type] = set()
        self._index_by_type[fragment.fragment_type].add(fragment.fragment_id)

    def get(self, fragment_id: str) -> Optional[StashFragment]:
        """Get a fragment by ID.

        Args:
            fragment_id: Fragment to retrieve.

        Returns:
            Fragment or None.
        """
        return self._fragments.get(fragment_id)

    def get_by_path(self, source_path: str) -> List[StashFragment]:
        """Get fragments by source path.

        Args:
            source_path: Source path to query.

        Returns:
            List of fragments.
        """
        ids = self._index_by_path.get(source_path, set())
        return [self._fragments[i] for i in ids if i in self._fragments]

    def get_by_type(self, fragment_type: FragmentType) -> List[StashFragment]:
        """Get fragments by type.

        Args:
            fragment_type: Type to query.

        Returns:
            List of fragments.
        """
        ids = self._index_by_type.get(fragment_type, set())
        return [self._fragments[i] for i in ids if i in self._fragments]

    def get_all_versions(self, source_path: str) -> List[StashFragment]:
        """Get all versions of a source path.

        Args:
            source_path: Source path.

        Returns:
            List sorted by version descending.
        """
        fragments = self.get_by_path(source_path)
        return sorted(fragments, key=lambda f: f.version, reverse=True)

    def remove(self, fragment_id: str) -> bool:
        """Remove a fragment.

        Args:
            fragment_id: Fragment to remove.

        Returns:
            True if removed.
        """
        fragment = self._fragments.pop(fragment_id, None)
        if fragment:
            self._index_by_path.get(fragment.source_path, set()).discard(fragment_id)
            self._index_by_type.get(fragment.fragment_type, set()).discard(fragment_id)
            return True
        return False

    def list_all(self) -> List[StashFragment]:
        """List all fragments.

        Returns:
            List of fragments.
        """
        return list(self._fragments.values())

    def count(self) -> int:
        """Get fragment count.

        Returns:
            Number of fragments.
        """
        return len(self._fragments)

    def verify_all(self) -> Dict[str, bool]:
        """Verify integrity of all fragments.

        Returns:
            Dict of fragment_id -> valid.
        """
        return {
            fid: f.verify_integrity()
            for fid, f in self._fragments.items()
        }


# ---------------------------------------------------------------------------
# ConflictResolver
# ---------------------------------------------------------------------------


class ConflictResolver:
    """Resolve conflicts between fragment versions.

    Applies strategies to select the correct version when
    multiple fragments target the same source.
    """

    def __init__(self, strategy: ConflictStrategy = ConflictStrategy.NEWEST_WINS) -> None:
        self.strategy = strategy

    def resolve(
        self,
        fragments: List[StashFragment],
    ) -> StashFragment:
        """Resolve conflict between fragments.

        Args:
            fragments: Conflicting fragments.

        Returns:
            Selected fragment.

        Raises:
            ConflictError: If cannot resolve.
        """
        if not fragments:
            raise ConflictError("No fragments to resolve")

        if len(fragments) == 1:
            return fragments[0]

        if self.strategy == ConflictStrategy.NEWEST_WINS:
            return max(fragments, key=lambda f: f.timestamp)

        elif self.strategy == ConflictStrategy.OLDEST_WINS:
            return min(fragments, key=lambda f: f.timestamp)

        elif self.strategy == ConflictStrategy.HIGHEST_VERSION:
            return max(fragments, key=lambda f: f.version)

        elif self.strategy == ConflictStrategy.MERGE:
            return self._merge_fragments(fragments)

        else:
            raise ConflictError(f"Cannot resolve with strategy {self.strategy}")

    def _merge_fragments(
        self,
        fragments: List[StashFragment],
    ) -> StashFragment:
        """Merge multiple fragments.

        Args:
            fragments: Fragments to merge.

        Returns:
            Merged fragment.
        """
        newest = max(fragments, key=lambda f: f.timestamp)

        # Combine content
        contents = [f.content for f in sorted(fragments, key=lambda f: f.version)]
        merged_content = "\n\n# --- Merged ---\n\n".join(contents)

        return StashFragment(
            fragment_id=newest.fragment_id,
            fragment_type=newest.fragment_type,
            source_path=newest.source_path,
            content=merged_content,
            version=max(f.version for f in fragments) + 1,
            dependencies=list(set(
                dep for f in fragments for dep in f.dependencies
            )),
        )


# ---------------------------------------------------------------------------
# ReconstructionEngine — Main orchestrator
# ---------------------------------------------------------------------------


class ReconstructionEngine:
    """Reconstruction Engine for GART v3.0.

    Rebuilds system state from distributed code stashes.

    Attributes:
        stash_box: Fragment storage.
        resolver: Conflict resolver.
    """

    def __init__(
        self,
        strategy: ConflictStrategy = ConflictStrategy.NEWEST_WINS,
    ) -> None:
        self.stash_box = StashBox()
        self.resolver = ConflictResolver(strategy)

    # --- Fragment management ---

    def stash(self, fragment: StashFragment) -> None:
        """Add a fragment to the stash.

        Args:
            fragment: Fragment to store.
        """
        self.stash_box.store(fragment)

    def stash_code(
        self,
        source_path: str,
        content: str,
        fragment_type: FragmentType = FragmentType.MODULE,
        dependencies: Optional[List[str]] = None,
    ) -> StashFragment:
        """Create and store a code fragment.

        Args:
            source_path: Original source path.
            content: Code content.
            fragment_type: Type of fragment.
            dependencies: Dependency fragment IDs.

        Returns:
            Created fragment.
        """
        versions = self.stash_box.get_all_versions(source_path)
        version = versions[0].version + 1 if versions else 1

        fragment = StashFragment(
            fragment_id=f"{source_path}_v{version}",
            fragment_type=fragment_type,
            source_path=source_path,
            content=content,
            version=version,
            dependencies=dependencies or [],
        )
        self.stash_box.store(fragment)
        return fragment

    # --- Reconstruction ---

    def reconstruct(
        self,
        plan: ReconstructionPlan,
    ) -> ReconstructionResult:
        """Reconstruct state from stashes.

        Args:
            plan: Reconstruction plan.

        Returns:
            Reconstruction result.
        """
        result = ReconstructionResult(
            status=ReconstructionStatus.IN_PROGRESS,
            fragments_total=len(plan.fragment_order),
        )

        output_parts = []

        for fragment_id in plan.fragment_order:
            fragment = self.stash_box.get(fragment_id)

            if not fragment:
                # Try to find by source path
                source_fragments = self.stash_box.get_by_path(fragment_id)
                if source_fragments:
                    fragment = self.resolver.resolve(source_fragments)
                else:
                    result.errors.append(f"Fragment not found: {fragment_id}")
                    continue

            # Verify integrity
            if not fragment.verify_integrity():
                result.errors.append(f"Integrity check failed: {fragment_id}")
                continue

            # Check expected checksum
            expected = plan.expected_checksums.get(fragment_id)
            if expected and fragment.checksum != expected:
                result.conflicts += 1

            output_parts.append(fragment.content)
            result.fragments_used += 1

        result.output = "\n\n".join(output_parts)
        result.status = (
            ReconstructionStatus.COMPLETED
            if result.fragments_used == result.fragments_total
            else ReconstructionStatus.PARTIAL
        )

        return result

    def reconstruct_all(self) -> ReconstructionResult:
        """Reconstruct from all available fragments.

        Returns:
            Reconstruction result.
        """
        fragments = self.stash_box.list_all()

        plan = ReconstructionPlan(
            target_state="full_reconstruction",
            fragment_order=[f.fragment_id for f in fragments],
        )

        return self.reconstruct(plan)

    # --- Verification ---

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify integrity of all stashes.

        Returns:
            Verification report.
        """
        checks = self.stash_box.verify_all()
        return {
            "total": len(checks),
            "valid": sum(1 for v in checks.values() if v),
            "invalid": sum(1 for v in checks.values() if not v),
            "details": checks,
        }

    # --- Statistics ---

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Statistics dictionary.
        """
        fragments = self.stash_box.list_all()
        return {
            "total_fragments": len(fragments),
            "by_type": {
                ft.value: len(self.stash_box.get_by_type(ft))
                for ft in FragmentType
            },
            "integrity": self.verify_integrity(),
        }
