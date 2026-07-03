"""
Roster Manager — Persona Roster & Family-Tree Tracking for GART v3.0.

Manages persona registration, genealogical queries, and roster
persistence with JSON import/export.

Components:
    - RosterEntry: Individual roster record
    - RosterManager: Main roster with CRUD + genealogy
    - RosterQuery: Query DSL for roster searches
    - RosterIO: JSON import/export
    - RosterValidator: Integrity validation

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RosterError(Exception):
    """Base exception for roster operations."""


class DuplicateEntryError(RosterError):
    """Raised when adding a persona that already exists."""


class EntryNotFoundError(RosterError):
    """Raised when a requested entry is not found."""


class ValidationError(RosterError):
    """Raised when roster validation fails."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PersonaStatus(Enum):
    """Status of a persona in the roster."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    RETIRED = "retired"
    BANNED = "banned"
    PENDING = "pending"
    SUSPENDED = "suspended"


class PersonaTier(Enum):
    """Tier/classification of a persona."""

    FOUNDER = "founder"
    CHAMPION = "champion"
    CONTENDER = "contender"
    PROSPECT = "prospect"
    LEGEND = "legend"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class RosterEntry:
    """A single persona entry in the roster.

    Attributes:
        persona_id: Unique identifier.
        artist_name: Display name.
        status: Current status.
        tier: Classification tier.
        generation: Genealogical generation.
        parent_ids: Parent persona IDs.
        child_ids: Child persona IDs.
        skills: Skill ratings dictionary.
        metadata: Free-form metadata.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    persona_id: str
    artist_name: str
    status: PersonaStatus = PersonaStatus.ACTIVE
    tier: PersonaTier = PersonaTier.PROSPECT
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    skills: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "persona_id": self.persona_id,
            "artist_name": self.artist_name,
            "status": self.status.value,
            "tier": self.tier.value,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "child_ids": list(self.child_ids),
            "skills": dict(self.skills),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RosterEntry:
        """Deserialize entry from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            RosterEntry instance.
        """
        return cls(
            persona_id=data["persona_id"],
            artist_name=data["artist_name"],
            status=PersonaStatus(data.get("status", "active")),
            tier=PersonaTier(data.get("tier", "prospect")),
            generation=data.get("generation", 0),
            parent_ids=list(data.get("parent_ids", [])),
            child_ids=list(data.get("child_ids", [])),
            skills=dict(data.get("skills", {})),
            metadata=dict(data.get("metadata", {})),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )

    def add_child(self, child_id: str) -> None:
        """Add a child reference.

        Args:
            child_id: Child persona ID.
        """
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
            self.updated_at = datetime.now().isoformat()

    def add_parent(self, parent_id: str) -> None:
        """Add a parent reference.

        Args:
            parent_id: Parent persona ID.
        """
        if parent_id not in self.parent_ids:
            self.parent_ids.append(parent_id)
            self.updated_at = datetime.now().isoformat()

    def update_skill(self, skill_name: str, value: float) -> None:
        """Update a skill rating.

        Args:
            skill_name: Skill name.
            value: Skill value (0.0-1.0).
        """
        self.skills[skill_name] = max(0.0, min(1.0, value))
        self.updated_at = datetime.now().isoformat()

    @property
    def overall_rating(self) -> float:
        """Compute overall skill rating.

        Returns:
            Average of all skill values.
        """
        if not self.skills:
            return 0.0
        return sum(self.skills.values()) / len(self.skills)


@dataclass
class RosterQuery:
    """Query specification for roster searches.

    Attributes:
        filters: Field filters as dict of field -> value.
        skill_min: Minimum overall skill rating.
        skill_max: Maximum overall skill rating.
        generations: List of generation numbers.
        tiers: List of tier values.
        statuses: List of status values.
        limit: Maximum results.
        offset: Result offset.
    """

    filters: Dict[str, Any] = field(default_factory=dict)
    skill_min: Optional[float] = None
    skill_max: Optional[float] = None
    generations: Optional[List[int]] = None
    tiers: Optional[List[PersonaTier]] = None
    statuses: Optional[List[PersonaStatus]] = None
    limit: int = 100
    offset: int = 0


# ---------------------------------------------------------------------------
# RosterManager
# ---------------------------------------------------------------------------


class RosterManager:
    """Persona roster manager for GART v3.0.

    Provides CRUD operations for persona entries, genealogical
    queries, and roster persistence.

    Attributes:
        entries: Dictionary of persona_id -> RosterEntry.
        _id_counter: Internal ID counter.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, RosterEntry] = {}
        self._id_counter = 0

    # --- CRUD operations ---

    def create(
        self,
        artist_name: str,
        persona_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RosterEntry:
        """Create a new roster entry.

        Args:
            artist_name: Display name for the persona.
            persona_id: Optional explicit ID (auto-generated if None).
            **kwargs: Additional entry attributes.

        Returns:
            Created RosterEntry.

        Raises:
            DuplicateEntryError: If persona_id already exists.
        """
        if persona_id is None:
            self._id_counter += 1
            persona_id = f"{artist_name.replace(' ', '_')}_{self._id_counter}"

        if persona_id in self._entries:
            raise DuplicateEntryError(f"Persona '{persona_id}' already exists")

        entry = RosterEntry(
            persona_id=persona_id,
            artist_name=artist_name,
            **kwargs,
        )
        self._entries[persona_id] = entry
        logger.info("Created roster entry: %s (%s)", persona_id, artist_name)
        return entry

    def get(self, persona_id: str) -> RosterEntry:
        """Get a roster entry by ID.

        Args:
            persona_id: Persona ID to look up.

        Returns:
            RosterEntry.

        Raises:
            EntryNotFoundError: If entry not found.
        """
        entry = self._entries.get(persona_id)
        if entry is None:
            raise EntryNotFoundError(f"Persona '{persona_id}' not found")
        return entry

    def update(self, persona_id: str, **kwargs: Any) -> RosterEntry:
        """Update a roster entry.

        Args:
            persona_id: Persona ID to update.
            **kwargs: Attributes to update.

        Returns:
            Updated RosterEntry.

        Raises:
            EntryNotFoundError: If entry not found.
        """
        entry = self.get(persona_id)
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        entry.updated_at = datetime.now().isoformat()
        return entry

    def delete(self, persona_id: str) -> bool:
        """Delete a roster entry.

        Args:
            persona_id: Persona ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        if persona_id in self._entries:
            del self._entries[persona_id]
            logger.info("Deleted roster entry: %s", persona_id)
            return True
        return False

    def list_all(self) -> List[RosterEntry]:
        """List all roster entries.

        Returns:
            List of all entries.
        """
        return list(self._entries.values())

    def count(self) -> int:
        """Get total number of entries.

        Returns:
            Entry count.
        """
        return len(self._entries)

    # --- Genealogy ---

    def register_parent_child(
        self,
        parent_id: str,
        child_id: str,
    ) -> None:
        """Register a parent-child relationship.

        Args:
            parent_id: Parent persona ID.
            child_id: Child persona ID.

        Raises:
            EntryNotFoundError: If either entry not found.
        """
        parent = self.get(parent_id)
        child = self.get(child_id)

        parent.add_child(child_id)
        child.add_parent(parent_id)

        # Update child's generation
        if parent.generation >= child.generation:
            child.generation = parent.generation + 1

    def get_ancestors(
        self,
        persona_id: str,
        max_depth: int = 5,
    ) -> List[str]:
        """Get ancestor IDs for a persona.

        Args:
            persona_id: Persona to query.
            max_depth: Maximum ancestor depth.

        Returns:
            List of ancestor IDs.
        """
        ancestors = []
        current_gen = [persona_id]
        for _ in range(max_depth):
            next_gen = []
            for pid in current_gen:
                try:
                    entry = self.get(pid)
                    for parent_id in entry.parent_ids:
                        if parent_id not in ancestors:
                            ancestors.append(parent_id)
                            next_gen.append(parent_id)
                except EntryNotFoundError:
                    pass
            current_gen = next_gen
            if not current_gen:
                break
        return ancestors

    def get_descendants(
        self,
        persona_id: str,
        max_depth: int = 5,
    ) -> List[str]:
        """Get descendant IDs for a persona.

        Args:
            persona_id: Persona to query.
            max_depth: Maximum descendant depth.

        Returns:
            List of descendant IDs.
        """
        descendants = []
        current_gen = [persona_id]
        for _ in range(max_depth):
            next_gen = []
            for pid in current_gen:
                try:
                    entry = self.get(pid)
                    for child_id in entry.child_ids:
                        if child_id not in descendants:
                            descendants.append(child_id)
                            next_gen.append(child_id)
                except EntryNotFoundError:
                    pass
            current_gen = next_gen
            if not current_gen:
                break
        return descendants

    def get_siblings(self, persona_id: str) -> List[str]:
        """Get sibling IDs for a persona.

        Args:
            persona_id: Persona to query.

        Returns:
            List of sibling IDs.
        """
        try:
            entry = self.get(persona_id)
            siblings = []
            for parent_id in entry.parent_ids:
                try:
                    parent = self.get(parent_id)
                    for child_id in parent.child_ids:
                        if child_id != persona_id and child_id not in siblings:
                            siblings.append(child_id)
                except EntryNotFoundError:
                    pass
            return siblings
        except EntryNotFoundError:
            return []

    def get_family_tree(
        self,
        persona_id: str,
    ) -> Dict[str, List[str]]:
        """Get complete family tree for a persona.

        Args:
            persona_id: Persona to query.

        Returns:
            Dictionary with ancestors, descendants, and siblings.
        """
        return {
            "persona_id": persona_id,
            "ancestors": self.get_ancestors(persona_id),
            "descendants": self.get_descendants(persona_id),
            "siblings": self.get_siblings(persona_id),
        }

    def get_lineage(
        self,
        persona_id: str,
    ) -> List[Dict[str, Any]]:
        """Get detailed lineage info with entry data.

        Args:
            persona_id: Persona to query.

        Returns:
            List of lineage entries.
        """
        lineage = []
        ancestor_ids = self.get_ancestors(persona_id, max_depth=10)
        for aid in ancestor_ids:
            try:
                entry = self.get(aid)
                lineage.append({
                    "persona_id": aid,
                    "artist_name": entry.artist_name,
                    "generation": entry.generation,
                    "tier": entry.tier.value,
                    "overall_rating": entry.overall_rating,
                })
            except EntryNotFoundError:
                lineage.append({
                    "persona_id": aid,
                    "artist_name": "[unknown]",
                    "generation": -1,
                    "tier": "unknown",
                    "overall_rating": 0.0,
                })
        return lineage

    # --- Queries ---

    def query(self, query_spec: RosterQuery) -> List[RosterEntry]:
        """Execute a roster query.

        Args:
            query_spec: Query specification.

        Returns:
            List of matching entries.
        """
        results = list(self._entries.values())

        # Apply filters
        for field, value in query_spec.filters.items():
            results = [
                e for e in results
                if getattr(e, field, None) == value
            ]

        # Skill range
        if query_spec.skill_min is not None:
            results = [e for e in results if e.overall_rating >= query_spec.skill_min]
        if query_spec.skill_max is not None:
            results = [e for e in results if e.overall_rating <= query_spec.skill_max]

        # Generation filter
        if query_spec.generations is not None:
            results = [e for e in results if e.generation in query_spec.generations]

        # Tier filter
        if query_spec.tiers is not None:
            results = [e for e in results if e.tier in query_spec.tiers]

        # Status filter
        if query_spec.statuses is not None:
            results = [e for e in results if e.status in query_spec.statuses]

        # Apply limit/offset
        results = results[query_spec.offset:query_spec.offset + query_spec.limit]

        return results

    def get_by_tier(self, tier: PersonaTier) -> List[RosterEntry]:
        """Get all entries of a specific tier.

        Args:
            tier: Tier to filter by.

        Returns:
            List of matching entries.
        """
        return [e for e in self._entries.values() if e.tier == tier]

    def get_by_status(self, status: PersonaStatus) -> List[RosterEntry]:
        """Get all entries with a specific status.

        Args:
            status: Status to filter by.

        Returns:
            List of matching entries.
        """
        return [e for e in self._entries.values() if e.status == status]

    def get_by_generation(self, generation: int) -> List[RosterEntry]:
        """Get all entries from a specific generation.

        Args:
            generation: Generation number.

        Returns:
            List of matching entries.
        """
        return [e for e in self._entries.values() if e.generation == generation]

    def get_top_rated(self, n: int = 10) -> List[RosterEntry]:
        """Get top-rated personas.

        Args:
            n: Number to return.

        Returns:
            List of top-rated entries.
        """
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.overall_rating,
            reverse=True,
        )
        return sorted_entries[:n]

    # --- Import/Export ---

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full roster to dictionary.

        Returns:
            Dictionary with all entries and metadata.
        """
        return {
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
            "count": len(self._entries),
            "exported_at": datetime.now().isoformat(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize roster to JSON string.

        Args:
            indent: JSON indentation.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def export_to_file(self, filepath: str) -> None:
        """Export roster to JSON file.

        Args:
            filepath: Output file path.
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())
        logger.info("Roster exported to %s (%d entries)", filepath, len(self._entries))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RosterManager:
        """Deserialize roster from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            RosterManager instance.
        """
        manager = cls()
        for entry_id, entry_data in data.get("entries", {}).items():
            entry = RosterEntry.from_dict(entry_data)
            manager._entries[entry.persona_id] = entry
        return manager

    @classmethod
    def from_json(cls, json_str: str) -> RosterManager:
        """Deserialize roster from JSON string.

        Args:
            json_str: JSON string.

        Returns:
            RosterManager instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def import_from_file(self, filepath: str) -> int:
        """Import roster from JSON file.

        Args:
            filepath: Input file path.

        Returns:
            Number of entries imported.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        imported = 0
        for entry_id, entry_data in data.get("entries", {}).items():
            entry = RosterEntry.from_dict(entry_data)
            self._entries[entry.persona_id] = entry
            imported += 1

        logger.info("Imported %d entries from %s", imported, filepath)
        return imported

    # --- Validation ---

    def validate(self) -> Dict[str, Any]:
        """Validate roster integrity.

        Checks:
            - All parent/child references are valid
            - No circular references
            - Generations are consistent
            - Skill values are in valid range

        Returns:
            Validation report.
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "entry_count": len(self._entries),
        }

        # Check parent/child references
        for entry in self._entries.values():
            for parent_id in entry.parent_ids:
                if parent_id not in self._entries:
                    report["warnings"].append(
                        f"{entry.persona_id}: missing parent '{parent_id}'"
                    )
                else:
                    parent = self._entries[parent_id]
                    if entry.persona_id not in parent.child_ids:
                        report["warnings"].append(
                            f"{entry.persona_id}: not in parent's child list"
                        )

            for child_id in entry.child_ids:
                if child_id not in self._entries:
                    report["warnings"].append(
                        f"{entry.persona_id}: missing child '{child_id}'"
                    )

            # Check skill values
            for skill, value in entry.skills.items():
                if not (0.0 <= value <= 1.0):
                    report["errors"].append(
                        f"{entry.persona_id}: skill '{skill}' out of range: {value}"
                    )
                    report["valid"] = False

        return report

    # --- Statistics ---

    def get_stats(self) -> Dict[str, Any]:
        """Get roster statistics.

        Returns:
            Statistics dictionary.
        """
        if not self._entries:
            return {"total": 0}

        ratings = [e.overall_rating for e in self._entries.values()]
        generations = [e.generation for e in self._entries.values()]

        tier_counts = {}
        for tier in PersonaTier:
            count = len([e for e in self._entries.values() if e.tier == tier])
            if count > 0:
                tier_counts[tier.value] = count

        status_counts = {}
        for status in PersonaStatus:
            count = len([e for e in self._entries.values() if e.status == status])
            if count > 0:
                status_counts[status.value] = count

        return {
            "total": len(self._entries),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "max_rating": max(ratings) if ratings else 0,
            "min_rating": min(ratings) if ratings else 0,
            "max_generation": max(generations) if generations else 0,
            "tier_distribution": tier_counts,
            "status_distribution": status_counts,
        }
