"""
Genetic Persona with Expansive Memory — GPXM Core Module.

Implements the full GeneticPersona dataclass with five-tier expansive memory:
    - SessionMemory (ST): Hours TTL, real-time
    - ProjectMemory (MT): Days TTL, per-project
    - CoreIdentityMemory (LT): Months TTL, personal facts & beliefs
    - EpisodicMemory: Events with temporal indexing
    - SemanticMemory: Structured knowledge graph

Plus VoiceDifferentiationParameters, MemoryEntry hierarchy,
consolidation pipelines, and genetic evolution operators.

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VoiceDifferentiationParameters
# ---------------------------------------------------------------------------


@dataclass
class VoiceDifferentiationParameters:
    """Parameters that distinguish one artist's voice from another.

    Attributes:
        vocabulary_tier: Vocabulary complexity tier (Street, Mixed, Literary, Technical).
        slang_density: Slang usage frequency (Low, Medium, High, Very High).
        sentence_length: Average sentence length (Short, Medium, Long, Variable).
        pause_pattern: Pause placement style (End-Heavy, Even, Staccato, Flowing).
        emotional_range: Emotional spectrum breadth (Narrow, Medium, Wide, Extreme).
        cultural_markers: Cultural/geographic identity markers.
        narrative_mode: Storytelling approach (Linear, Fragmented, Abstract, Cinematic).
        call_response: Call-and-response pattern (None, Echo, Question, Direct).
    """

    vocabulary_tier: str = "Mixed"
    slang_density: str = "Medium"
    sentence_length: str = "Medium"
    pause_pattern: str = "Even"
    emotional_range: str = "Medium"
    cultural_markers: str = ""
    narrative_mode: str = "Linear"
    call_response: str = "None"

    def compare(self, other: VoiceDifferentiationParameters) -> float:
        """Compare two voice parameter sets.

        Args:
            other: Other voice parameters to compare against.

        Returns:
            Similarity score (0.0-1.0), higher = more similar.
        """
        score = 0.0
        fields = [
            "vocabulary_tier", "slang_density", "sentence_length",
            "pause_pattern", "emotional_range", "narrative_mode",
            "call_response",
        ]
        for f in fields:
            if getattr(self, f) == getattr(other, f):
                score += 1.0 / len(fields)
        return score


# ---------------------------------------------------------------------------
# Memory Entry hierarchy
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """Base class for all memory entries.

    Attributes:
        content: The memory content.
        timestamp: When the memory was created.
        source: Source of the memory (e.g., "ingestion", "generation").
        confidence: Confidence score (0.0-1.0).
        metadata: Additional metadata.
    """

    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class SessionMemoryEntry(MemoryEntry):
    """A session-scoped memory entry (hours TTL).

    Attributes:
        session_id: Unique session identifier.
        ttl_hours: Time-to-live in hours.
    """

    session_id: str = ""
    ttl_hours: float = 4.0

    def is_expired(self) -> bool:
        """Check if this session memory has expired.

        Returns:
            True if expired.
        """
        age = datetime.now() - self.timestamp
        return age > timedelta(hours=self.ttl_hours)


@dataclass
class ProjectMemoryEntry(MemoryEntry):
    """A project-scoped memory entry (days TTL).

    Attributes:
        project_id: Project identifier.
        ttl_days: Time-to-live in days.
    """

    project_id: str = ""
    ttl_days: float = 7.0

    def is_expired(self) -> bool:
        """Check if this project memory has expired."""
        age = datetime.now() - self.timestamp
        return age > timedelta(days=self.ttl_days)


@dataclass
class CoreIdentityEntry(MemoryEntry):
    """A core identity memory entry (months TTL).

    These are long-term personal facts and beliefs.

    Attributes:
        category: Identity category (e.g., "belief", "value", "preference").
        ttl_months: Time-to-live in months.
    """

    category: str = "belief"
    ttl_months: float = 6.0

    def is_expired(self) -> bool:
        """Check if this core identity memory has expired."""
        age = datetime.now() - self.timestamp
        return age > timedelta(days=self.ttl_months * 30)


@dataclass
class EpisodicMemoryEntry(MemoryEntry):
    """An episodic memory entry (event with temporal indexing).

    Attributes:
        event_type: Type of event.
        participants: List of participant identifiers.
        location: Event location.
    """

    event_type: str = "interaction"
    participants: List[str] = field(default_factory=list)
    location: str = ""


@dataclass
class SemanticMemoryEntry(MemoryEntry):
    """A semantic memory entry (structured knowledge).

    Attributes:
        subject: Knowledge subject.
        predicate: Knowledge predicate.
        object: Knowledge object.
        certainty: Certainty score (0.0-1.0).
    """

    subject: str = ""
    predicate: str = ""
    object: str = ""
    certainty: float = 1.0

    def as_triple(self) -> Tuple[str, str, str]:
        """Return as (subject, predicate, object) triple."""
        return (self.subject, self.predicate, self.object)


# ---------------------------------------------------------------------------
# Memory tiers
# ---------------------------------------------------------------------------


@dataclass
class SessionMemory:
    """Short-term session memory (ST tier).

    Stores recent interactions with hours-level TTL.
    """

    entries: List[SessionMemoryEntry] = field(default_factory=list)
    max_entries: int = 100

    def add(self, entry: SessionMemoryEntry) -> None:
        """Add a session memory entry."""
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_recent(self, count: int = 10) -> List[SessionMemoryEntry]:
        """Get recent non-expired entries."""
        return [e for e in self.entries[-count:] if not e.is_expired()]

    def prune_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if not e.is_expired()]
        return before - len(self.entries)

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()


@dataclass
class ProjectMemory:
    """Medium-term project memory (MT tier).

    Stores project-specific context with days-level TTL.
    """

    entries: List[ProjectMemoryEntry] = field(default_factory=list)
    max_entries: int = 500

    def add(self, entry: ProjectMemoryEntry) -> None:
        """Add a project memory entry."""
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_for_project(self, project_id: str) -> List[ProjectMemoryEntry]:
        """Get entries for a specific project."""
        return [e for e in self.entries if e.project_id == project_id and not e.is_expired()]

    def prune_expired(self) -> int:
        """Remove expired entries."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if not e.is_expired()]
        return before - len(self.entries)


@dataclass
class CoreIdentityMemory:
    """Long-term core identity memory (LT tier).

    Stores fundamental personal facts and beliefs with months-level TTL.
    """

    entries: List[CoreIdentityEntry] = field(default_factory=list)

    def add(self, entry: CoreIdentityEntry) -> None:
        """Add a core identity entry."""
        self.entries.append(entry)

    def get_by_category(self, category: str) -> List[CoreIdentityEntry]:
        """Get entries by category."""
        return [e for e in self.entries if e.category == category and not e.is_expired()]

    def get_beliefs(self) -> List[CoreIdentityEntry]:
        """Get all belief entries."""
        return self.get_by_category("belief")

    def get_values(self) -> List[CoreIdentityEntry]:
        """Get all value entries."""
        return self.get_by_category("value")

    def prune_expired(self) -> int:
        """Remove expired entries."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if not e.is_expired()]
        return before - len(self.entries)


@dataclass
class EpisodicMemory:
    """Episodic memory for event-based recall.

    Stores events with temporal indexing for autobiographical recall.
    """

    entries: List[EpisodicMemoryEntry] = field(default_factory=list)

    def add(self, entry: EpisodicMemoryEntry) -> None:
        """Add an episodic memory entry."""
        self.entries.append(entry)

    def get_by_event_type(self, event_type: str) -> List[EpisodicMemoryEntry]:
        """Get entries by event type."""
        return [e for e in self.entries if e.event_type == event_type]

    def get_by_participant(self, participant: str) -> List[EpisodicMemoryEntry]:
        """Get entries involving a participant."""
        return [e for e in self.entries if participant in e.participants]

    def get_timeline(self) -> List[EpisodicMemoryEntry]:
        """Get all entries sorted by timestamp."""
        return sorted(self.entries, key=lambda e: e.timestamp)


@dataclass
class SemanticMemory:
    """Semantic memory for structured knowledge.

    Stores facts as subject-predicate-object triples.
    """

    entries: List[SemanticMemoryEntry] = field(default_factory=list)

    def add(self, entry: SemanticMemoryEntry) -> None:
        """Add a semantic memory entry."""
        self.entries.append(entry)

    def query(self, subject: str) -> List[SemanticMemoryEntry]:
        """Query entries by subject."""
        return [e for e in self.entries if e.subject.lower() == subject.lower()]

    def query_triple(self, subject: str, predicate: str) -> List[SemanticMemoryEntry]:
        """Query entries by subject and predicate."""
        return [
            e for e in self.entries
            if e.subject.lower() == subject.lower()
            and e.predicate.lower() == predicate.lower()
        ]

    def get_knowledge_graph(self) -> List[Tuple[str, str, str]]:
        """Get all entries as triples."""
        return [e.as_triple() for e in self.entries]


# ---------------------------------------------------------------------------
# ConsolidationResult
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationResult:
    """Result of a memory consolidation operation.

    Attributes:
        st_to_mt_entries: Number of session entries promoted to project.
        mt_to_lt_entries: Number of project entries promoted to core identity.
        pruned_entries: Total number of expired entries removed.
        new_lt_entries: New long-term entries created.
        summary: Human-readable consolidation summary.
    """

    st_to_mt_entries: int = 0
    mt_to_lt_entries: int = 0
    pruned_entries: int = 0
    new_lt_entries: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# ExpansiveMemoryBank
# ---------------------------------------------------------------------------


class ExpansiveMemoryBank:
    """Five-tier expansive memory bank.

    Manages the complete memory hierarchy:
        ST (Session) -> MT (Project) -> LT (Core Identity)
        + Episodic layer + Semantic layer

    Provides consolidation pipelines that promote memories up
    the hierarchy based on importance and recency.
    """

    def __init__(self) -> None:
        self.session = SessionMemory()
        self.project = ProjectMemory()
        self.core_identity = CoreIdentityMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self._consolidation_history: List[ConsolidationResult] = []

    def add_session_memory(self, content: str, source: str = "ingestion") -> None:
        """Add a session-scoped memory."""
        entry = SessionMemoryEntry(
            content=content, source=source, session_id=str(uuid.uuid4())[:8]
        )
        self.session.add(entry)

    def add_project_memory(self, content: str, project_id: str = "") -> None:
        """Add a project-scoped memory."""
        entry = ProjectMemoryEntry(content=content, project_id=project_id or "default")
        self.project.add(entry)

    def add_core_identity(self, content: str, category: str = "belief") -> None:
        """Add a core identity memory."""
        entry = CoreIdentityEntry(content=content, category=category)
        self.core_identity.add(entry)

    def add_episodic_memory(
        self, content: str, event_type: str = "interaction", participants: Optional[List[str]] = None
    ) -> None:
        """Add an episodic memory."""
        entry = EpisodicMemoryEntry(
            content=content, event_type=event_type, participants=participants or []
        )
        self.episodic.add(entry)

    def add_semantic_fact(self, subject: str, predicate: str, object: str, certainty: float = 1.0) -> None:
        """Add a semantic fact."""
        entry = SemanticMemoryEntry(
            subject=subject, predicate=predicate, object=object, certainty=certainty
        )
        self.semantic.add(entry)

    def consolidate(self) -> ConsolidationResult:
        """Run memory consolidation pipeline.

        Promotes important memories up the hierarchy:
        - High-confidence session entries -> project memory
        - Frequently accessed project entries -> core identity
        - All tiers: prune expired entries

        Returns:
            ConsolidationResult with statistics.
        """
        result = ConsolidationResult()

        # Prune expired entries from all tiers
        result.pruned_entries += self.session.prune_expired()
        result.pruned_entries += self.project.prune_expired()
        result.pruned_entries += self.core_identity.prune_expired()

        # Promote high-confidence session entries to project memory
        for entry in self.session.entries:
            if entry.confidence >= 0.8 and not entry.is_expired():
                self.project.add(ProjectMemoryEntry(
                    content=entry.content,
                    project_id=entry.session_id,
                    source=entry.source,
                    confidence=entry.confidence,
                ))
                result.st_to_mt_entries += 1

        # Promote frequently-accessed project entries to core identity
        project_content_counts: Dict[str, int] = {}
        for entry in self.project.entries:
            project_content_counts[entry.content] = project_content_counts.get(entry.content, 0) + 1

        for content, count in project_content_counts.items():
            if count >= 3:
                self.core_identity.add(CoreIdentityEntry(
                    content=content, category="consolidated_belief"
                ))
                result.mt_to_lt_entries += 1

        result.summary = (
            f"Consolidation: {result.st_to_mt_entries} ST->MT, "
            f"{result.mt_to_lt_entries} MT->LT, "
            f"{result.pruned_entries} pruned"
        )

        self._consolidation_history.append(result)
        logger.info(result.summary)
        return result

    def get_memory_summary(self) -> Dict[str, int]:
        """Get summary of memory contents.

        Returns:
            Dict with entry counts per tier.
        """
        return {
            "session": len(self.session.entries),
            "project": len(self.project.entries),
            "core_identity": len(self.core_identity.entries),
            "episodic": len(self.episodic.entries),
            "semantic": len(self.semantic.entries),
            "total": (
                len(self.session.entries) + len(self.project.entries) +
                len(self.core_identity.entries) + len(self.episodic.entries) +
                len(self.semantic.entries)
            ),
        }

    def search_all(self, query: str) -> List[MemoryEntry]:
        """Search all memory tiers for entries matching query.

        Args:
            query: Search string.

        Returns:
            Matching entries from all tiers.
        """
        results: List[MemoryEntry] = []
        query_lower = query.lower()

        for entry in self.session.entries:
            if query_lower in entry.content.lower():
                results.append(entry)
        for entry in self.project.entries:
            if query_lower in entry.content.lower():
                results.append(entry)
        for entry in self.core_identity.entries:
            if query_lower in entry.content.lower():
                results.append(entry)
        for entry in self.episodic.entries:
            if query_lower in entry.content.lower():
                results.append(entry)
        for entry in self.semantic.entries:
            if query_lower in entry.content.lower():
                results.append(entry)

        return results


# ---------------------------------------------------------------------------
# EntropicScript placeholder
# ---------------------------------------------------------------------------


@dataclass
class EntropicScript:
    """An entropic script for creative generation.

    Attributes:
        script_id: Unique identifier.
        script_type: Type of script (e.g., "verse", "hook", "bridge").
        entropy_level: Entropy level for variation (0.0-1.0).
        template: Script template string.
        constraints: Generation constraints.
    """

    script_id: str = ""
    script_type: str = "verse"
    entropy_level: float = 0.5
    template: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GeneticPersona — Main class
# ---------------------------------------------------------------------------


@dataclass
class GeneticPersona:
    """Genetic Persona with Expansive Memory — GPXM Core.

    Represents an artist persona with genetic evolution capabilities,
    expansive 5-tier memory, voice differentiation parameters, and
    entropic scripting.

    Attributes:
        persona_id: Unique persona identifier (auto-generated if empty).
        artist_name: Primary artist name.
        aliases: List of known aliases.
        genre_anchor: Primary genre.
        sub_genre_tags: Sub-genre classification tags.
        era: Musical era (e.g., "2020s").
        regional_origin: Geographic origin.
        voice_params: VoiceDifferentiationParameters.
        memory_bank: ExpansiveMemoryBank instance.
        entropic_scripts: List of active EntropicScripts.
        genetic_fitness: Current fitness score (0.0-1.0).
        entropy_level: Entropy calibration (0.0-1.0).
        collaboration_affinity: Collaboration preference (0.0-1.0).
        evolution_generation: Number of evolution generations.
        parent_ids: IDs of parent personas (if evolved).
    """

    persona_id: str = ""
    artist_name: str = ""
    aliases: List[str] = field(default_factory=list)
    genre_anchor: str = ""
    sub_genre_tags: List[str] = field(default_factory=list)
    era: str = ""
    regional_origin: str = ""
    voice_params: VoiceDifferentiationParameters = field(default_factory=VoiceDifferentiationParameters)
    memory_bank: ExpansiveMemoryBank = field(default_factory=ExpansiveMemoryBank)
    entropic_scripts: List[EntropicScript] = field(default_factory=list)
    genetic_fitness: float = 0.0
    entropy_level: float = 0.5
    collaboration_affinity: float = 0.5
    evolution_generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Auto-generate persona_id if not provided."""
        if not self.persona_id:
            self.persona_id = str(uuid.uuid4())[:12]

    # -- Evolution operators --

    def mutate(self, mutation_rate: float = 0.1) -> None:
        """Apply mutation to this persona.

        Randomly adjusts voice parameters and entropy level.

        Args:
            mutation_rate: Probability of each trait mutating (0.0-1.0).
        """
        import random

        if random.random() < mutation_rate:
            tiers = ["Street", "Mixed", "Literary", "Technical"]
            self.voice_params.vocabulary_tier = random.choice(tiers)

        if random.random() < mutation_rate:
            densities = ["Low", "Medium", "High", "Very High"]
            self.voice_params.slang_density = random.choice(densities)

        if random.random() < mutation_rate:
            self.entropy_level = max(0.0, min(1.0, self.entropy_level + random.gauss(0, 0.1)))

        if random.random() < mutation_rate:
            self.collaboration_affinity = max(0.0, min(1.0, self.collaboration_affinity + random.gauss(0, 0.1)))

        self.evolution_generation += 1
        logger.debug("Mutated %s (gen %d)", self.artist_name, self.evolution_generation)

    def crossover(self, partner: GeneticPersona) -> GeneticPersona:
        """Create offspring via crossover with another persona.

        Blends voice parameters, takes entropic scripts from both,
        and averages fitness-related traits.

        Args:
            partner: The other parent persona.

        Returns:
            New GeneticPersona offspring.
        """
        import random

        child = GeneticPersona(
            artist_name=f"{self.artist_name}x{partner.artist_name}",
            genre_anchor=self.genre_anchor if random.random() < 0.5 else partner.genre_anchor,
            sub_genre_tags=list(set(self.sub_genre_tags + partner.sub_genre_tags)),
            era=self.era if random.random() < 0.5 else partner.era,
            regional_origin=self.regional_origin if random.random() < 0.5 else partner.regional_origin,
            voice_params=self._blend_voice_params(partner),
            memory_bank=ExpansiveMemoryBank(),
            entropic_scripts=self.entropic_scripts + partner.entropic_scripts,
            genetic_fitness=(self.genetic_fitness + partner.genetic_fitness) / 2.0,
            entropy_level=(self.entropy_level + partner.entropy_level) / 2.0,
            collaboration_affinity=(self.collaboration_affinity + partner.collaboration_affinity) / 2.0,
            evolution_generation=max(self.evolution_generation, partner.evolution_generation) + 1,
            parent_ids=[self.persona_id, partner.persona_id],
        )

        return child

    def _blend_voice_params(self, partner: GeneticPersona) -> VoiceDifferentiationParameters:
        """Blend voice parameters with a partner.

        Args:
            partner: Partner persona.

        Returns:
            Blended VoiceDifferentiationParameters.
        """
        import random

        s = self.voice_params
        p = partner.voice_params

        return VoiceDifferentiationParameters(
            vocabulary_tier=s.vocabulary_tier if random.random() < 0.5 else p.vocabulary_tier,
            slang_density=s.slang_density if random.random() < 0.5 else p.slang_density,
            sentence_length=s.sentence_length if random.random() < 0.5 else p.sentence_length,
            pause_pattern=s.pause_pattern if random.random() < 0.5 else p.pause_pattern,
            emotional_range=s.emotional_range if random.random() < 0.5 else p.emotional_range,
            cultural_markers=s.cultural_markers if random.random() < 0.5 else p.cultural_markers,
            narrative_mode=s.narrative_mode if random.random() < 0.5 else p.narrative_mode,
            call_response=s.call_response if random.random() < 0.5 else p.call_response,
        )

    def calculate_fitness(self) -> float:
        """Calculate genetic fitness score.

        Based on:
        - Entropy level (optimal at 0.5)
        - Collaboration affinity (higher = better)
        - Memory richness
        - Script diversity

        Returns:
            Fitness score (0.0-1.0).
        """
        entropy_score = 1.0 - abs(self.entropy_level - 0.5) * 2.0
        collab_score = self.collaboration_affinity
        memory_score = min(1.0, self.memory_bank.get_memory_summary()["total"] / 100.0)
        script_score = min(1.0, len(self.entropic_scripts) / 5.0)

        self.genetic_fitness = (entropy_score * 0.3 + collab_score * 0.3 +
                                memory_score * 0.2 + script_score * 0.2)
        return self.genetic_fitness

    # -- Memory interface --

    def remember_session(self, content: str, source: str = "ingestion") -> None:
        """Add a session memory."""
        self.memory_bank.add_session_memory(content, source)

    def remember_project(self, content: str, project_id: str = "") -> None:
        """Add a project memory."""
        self.memory_bank.add_project_memory(content, project_id)

    def remember_identity(self, content: str, category: str = "belief") -> None:
        """Add a core identity memory."""
        self.memory_bank.add_core_identity(content, category)

    def remember_event(self, content: str, event_type: str = "interaction", participants: Optional[List[str]] = None) -> None:
        """Add an episodic memory."""
        self.memory_bank.add_episodic_memory(content, event_type, participants)

    def remember_fact(self, subject: str, predicate: str, object: str, certainty: float = 1.0) -> None:
        """Add a semantic fact."""
        self.memory_bank.add_semantic_fact(subject, predicate, object, certainty)

    def recall(self, query: str) -> List[MemoryEntry]:
        """Search all memory tiers."""
        return self.memory_bank.search_all(query)

    def consolidate_memories(self) -> ConsolidationResult:
        """Run memory consolidation."""
        return self.memory_bank.consolidate()

    # -- Script management --

    def add_script(self, script: EntropicScript) -> None:
        """Add an entropic script."""
        self.entropic_scripts.append(script)

    def get_scripts_by_type(self, script_type: str) -> List[EntropicScript]:
        """Get scripts by type."""
        return [s for s in self.entropic_scripts if s.script_type == script_type]

    # -- Utility --

    def clone(self) -> GeneticPersona:
        """Create a deep copy of this persona."""
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "persona_id": self.persona_id,
            "artist_name": self.artist_name,
            "aliases": self.aliases,
            "genre_anchor": self.genre_anchor,
            "sub_genre_tags": self.sub_genre_tags,
            "era": self.era,
            "regional_origin": self.regional_origin,
            "voice_params": {
                "vocabulary_tier": self.voice_params.vocabulary_tier,
                "slang_density": self.voice_params.slang_density,
                "emotional_range": self.voice_params.emotional_range,
            },
            "genetic_fitness": self.genetic_fitness,
            "entropy_level": self.entropy_level,
            "collaboration_affinity": self.collaboration_affinity,
            "evolution_generation": self.evolution_generation,
            "parent_ids": self.parent_ids,
            "memory_summary": self.memory_bank.get_memory_summary(),
            "script_count": len(self.entropic_scripts),
        }

    def __repr__(self) -> str:
        return (
            f"GeneticPersona(id={self.persona_id[:8]}, "
            f"name={self.artist_name}, "
            f"fitness={self.genetic_fitness:.3f}, "
            f"entropy={self.entropy_level:.3f})"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Genetic Persona module loaded successfully.")

    # Create a persona
    durk = GeneticPersona(
        artist_name="Lil Durk",
        aliases=["Durkio", "The Voice"],
        genre_anchor="Hip-Hop",
        sub_genre_tags=["melodic drill", "trap"],
        era="2020s",
        regional_origin="Chicago, IL",
        voice_params=VoiceDifferentiationParameters(
            vocabulary_tier="Street",
            slang_density="High",
            emotional_range="Extreme",
            cultural_markers="Chicago drill culture",
        ),
        entropy_level=0.45,
        collaboration_affinity=0.8,
    )

    # Add memories
    durk.remember_session("Recorded 'All My Life' session", "session")
    durk.remember_identity("Believes in loyalty above all", "value")
    durk.remember_fact("Lil Durk", "origin", "Englewood, Chicago")

    # Calculate fitness
    fitness = durk.calculate_fitness()
    print(f"\n{durk}")
    print(f"Fitness: {fitness:.3f}")
    print(f"Memory summary: {durk.memory_bank.get_memory_summary()}")

    # Create offspring via crossover
    baby = GeneticPersona(
        artist_name="Lil Baby",
        genre_anchor="Hip-Hop",
        era="2020s",
        regional_origin="Atlanta, GA",
        voice_params=VoiceDifferentiationParameters(
            vocabulary_tier="Street",
            slang_density="High",
        ),
        entropy_level=0.5,
        collaboration_affinity=0.9,
    )

    child = durk.crossover(baby)
    print(f"\nChild: {child}")
    print(f"Parents: {child.parent_ids}")
