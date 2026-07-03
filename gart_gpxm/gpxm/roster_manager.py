"""
16-Slot Capped Artist Roster Manager — GPXM.

Manages the artist pool with:
    - 16-slot mechanical cap
    - Non-redundancy filtering
    - Taxonomy packaging
    - Collaboration matrix computation
    - Cosine-similarity-based compatibility scoring

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward reference / local import shim
# ---------------------------------------------------------------------------


def _get_genetic_persona_class():
    """Lazy import to avoid circular dependencies."""
    try:
        from .genetic_persona import GeneticPersona, VoiceDifferentiationParameters
        return GeneticPersona, VoiceDifferentiationParameters
    except ImportError:
        # Fallback for standalone usage
        from dataclasses import dataclass as dc, field as f

        @dc
        class VoiceDifferentiationParameters:  # type: ignore[no-redef]
            vocabulary_tier: str = "Mixed"
            slang_density: str = "Medium"
            emotional_range: str = "Wide"
            cultural_markers: str = ""
            narrative_mode: str = "Linear"

        @dc
        class GeneticPersona:  # type: ignore[no-redef]
            persona_id: str = ""
            artist_name: str = "Unknown"
            aliases: List[str] = f(default_factory=list)
            genre_anchor: str = ""
            sub_genre_tags: List[str] = f(default_factory=list)
            era: str = ""
            regional_origin: str = ""
            voice_params: Any = f(default_factory=VoiceDifferentiationParameters)
            entropic_scripts: List[Any] = f(default_factory=list)
            entropy_level: float = 0.5
            collaboration_affinity: float = 0.5

        return GeneticPersona, VoiceDifferentiationParameters


# ---------------------------------------------------------------------------
# Stylistic Signature Extractor
# ---------------------------------------------------------------------------


def _extract_signature_vector(persona: Any) -> List[float]:
    """Extract a numeric signature vector from a persona.

    Creates a feature vector from voice params, genre, era, and
    entropy for cosine similarity computation.

    Args:
        persona: GeneticPersona object.

    Returns:
        Numeric feature vector.
    """
    voice = getattr(persona, "voice_params", None)
    if voice is None:
        return [0.5] * 10

    # Encode categorical features as numeric values
    vocab_map = {"Street": 0.0, "Mixed": 0.33, "Literary": 0.66, "Technical": 1.0}
    slang_map = {"Low": 0.0, "Medium": 0.33, "High": 0.66, "Very High": 1.0}
    emotion_map = {"Narrow": 0.0, "Medium": 0.33, "Wide": 0.66, "Extreme": 1.0}
    narrative_map = {"Linear": 0.0, "Fragmented": 0.33, "Abstract": 0.66, "Cinematic": 1.0}

    vector = [
        vocab_map.get(getattr(voice, "vocabulary_tier", "Mixed"), 0.33),
        slang_map.get(getattr(voice, "slang_density", "Medium"), 0.33),
        emotion_map.get(getattr(voice, "emotional_range", "Wide"), 0.66),
        narrative_map.get(getattr(voice, "narrative_mode", "Linear"), 0.0),
        getattr(persona, "entropy_level", 0.5),
        getattr(persona, "collaboration_affinity", 0.5),
        # Genre encoding (hash to float)
        hash(getattr(persona, "genre_anchor", "")) % 1000 / 1000.0,
        # Era encoding
        _era_to_float(getattr(persona, "era", "2020s")),
        # Regional encoding
        hash(getattr(persona, "regional_origin", "")) % 1000 / 1000.0,
        len(getattr(persona, "sub_genre_tags", [])) / 10.0,
    ]
    return vector


def _era_to_float(era: str) -> float:
    """Convert era string to numeric value.

    Args:
        era: Era string (e.g., "1990s", "2020s").

    Returns:
        Numeric era value (0.0-1.0).
    """
    era_map = {
        "1980s": 0.0, "1990s": 0.2, "2000s": 0.4,
        "2010s": 0.6, "2020s": 0.8, "2030s": 1.0,
    }
    return era_map.get(era, 0.5)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity (-1.0 to 1.0).
    """
    if len(vec_a) != len(vec_b):
        min_len = min(len(vec_a), len(vec_b))
        vec_a = vec_a[:min_len]
        vec_b = vec_b[:min_len]

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# RosterManager
# ---------------------------------------------------------------------------


class RosterError(Exception):
    """Custom exception for roster operations."""

    pass


class RosterFullError(RosterError):
    """Raised when roster is at capacity."""

    pass


class DuplicateArtistError(RosterError):
    """Raised when adding a duplicate artist."""

    pass


@dataclass
class SlotInfo:
    """Information about a roster slot.

    Attributes:
        slot_index: Slot number (0-15).
        persona: The GeneticPersona in this slot (None if empty).
        status: Slot status.
        date_added: When the persona was added.
    """

    slot_index: int
    persona: Optional[Any] = None
    status: str = "empty"
    date_added: Optional[str] = None


class RosterManager:
    """16-Slot Capped Artist Roster Manager.

    Manages a fixed-size pool of GeneticPersona objects with
    mechanical filtering, taxonomy packaging, and collaboration scoring.

    Attributes:
        MAX_SLOTS: Maximum number of roster slots (16).
    """

    MAX_SLOTS: int = 16

    def __init__(self) -> None:
        self.slots: Dict[int, Optional[Any]] = {i: None for i in range(self.MAX_SLOTS)}
        self._slot_metadata: Dict[int, Dict[str, Any]] = {
            i: {} for i in range(self.MAX_SLOTS)
        }
        self.collaboration_matrix: Dict[Tuple[str, str], float] = {}
        self._collaboration_history: List[Tuple[str, str, float]] = []

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def add_artist(self, persona: Any) -> int:
        """Add an artist persona to the roster.

        Finds the first empty slot, validates against redundancy,
        and assigns the persona.

        Args:
            persona: GeneticPersona to add.

        Returns:
            Slot index where the persona was placed.

        Raises:
            RosterFullError: If all 16 slots are occupied.
            DuplicateArtistError: If a similar persona already exists.
        """
        # Check for duplicates (mechanical filtering)
        if self._is_duplicate(persona):
            existing = self._find_similar(persona)
            raise DuplicateArtistError(
                f"Similar artist already in roster: {existing}"
            )

        # Find empty slot
        empty_slot = self._find_empty_slot()
        if empty_slot is None:
            raise RosterFullError(
                f"Roster is full (max {self.MAX_SLOTS} slots). "
                "Remove an artist before adding a new one."
            )

        self.slots[empty_slot] = persona
        self._slot_metadata[empty_slot] = {
            "date_added": str(__import__("datetime").datetime.now()),
            "artist_name": getattr(persona, "artist_name", "Unknown"),
        }

        # Recalculate collaboration matrix for new artist
        self._update_collaboration_for_artist(persona)

        logger.info(
            "Added %s to slot %d",
            getattr(persona, "artist_name", "Unknown"), empty_slot,
        )
        return empty_slot

    def remove_artist(self, slot_index: int) -> Any:
        """Remove an artist from a slot.

        Args:
            slot_index: Slot to clear.

        Returns:
            The removed persona.

        Raises:
            RosterError: If slot is already empty.
        """
        if slot_index < 0 or slot_index >= self.MAX_SLOTS:
            raise RosterError(f"Invalid slot index: {slot_index}")

        persona = self.slots[slot_index]
        if persona is None:
            raise RosterError(f"Slot {slot_index} is already empty")

        self.slots[slot_index] = None
        self._slot_metadata[slot_index] = {}

        # Remove from collaboration matrix
        pid = getattr(persona, "persona_id", str(slot_index))
        keys_to_remove = [k for k in self.collaboration_matrix if pid in k]
        for k in keys_to_remove:
            del self.collaboration_matrix[k]

        logger.info(
            "Removed %s from slot %d",
            getattr(persona, "artist_name", "Unknown"), slot_index,
        )
        return persona

    def get_artist_by_slot(self, slot_index: int) -> Optional[Any]:
        """Get the artist at a specific slot.

        Args:
            slot_index: Slot number.

        Returns:
            GeneticPersona or None.
        """
        if 0 <= slot_index < self.MAX_SLOTS:
            return self.slots[slot_index]
        return None

    def get_artist_by_name(self, artist_name: str) -> Optional[Any]:
        """Find an artist by name.

        Args:
            artist_name: Artist name to search for.

        Returns:
            GeneticPersona or None.
        """
        for slot in self.slots.values():
            if slot is not None and getattr(slot, "artist_name", "") == artist_name:
                return slot
        return None

    def get_all_artists(self) -> List[Any]:
        """Get all occupied slots.

        Returns:
            List of GeneticPersona objects.
        """
        return [s for s in self.slots.values() if s is not None]

    def get_slot_info(self) -> List[SlotInfo]:
        """Get information about all slots.

        Returns:
            List of SlotInfo objects.
        """
        result: List[SlotInfo] = []
        for i in range(self.MAX_SLOTS):
            persona = self.slots[i]
            result.append(SlotInfo(
                slot_index=i,
                persona=persona,
                status="occupied" if persona else "empty",
                date_added=self._slot_metadata[i].get("date_added"),
            ))
        return result

    # ------------------------------------------------------------------
    # Mechanical filtering
    # ------------------------------------------------------------------

    def _find_empty_slot(self) -> Optional[int]:
        """Find the first empty slot.

        Returns:
            Slot index or None if full.
        """
        for i in range(self.MAX_SLOTS):
            if self.slots[i] is None:
                return i
        return None

    def _is_duplicate(self, persona: Any) -> bool:
        """Check if a similar persona already exists.

        Args:
            persona: Persona to check.

        Returns:
            True if a duplicate/similar artist exists.
        """
        name = getattr(persona, "artist_name", "").lower()
        aliases = [a.lower() for a in getattr(persona, "aliases", [])]

        for slot in self.slots.values():
            if slot is None:
                continue
            slot_name = getattr(slot, "artist_name", "").lower()
            slot_aliases = [a.lower() for a in getattr(slot, "aliases", [])]

            if slot_name == name:
                return True
            if name in slot_aliases or slot_name in aliases:
                return True

        return False

    def _find_similar(self, persona: Any) -> str:
        """Find the name of a similar existing artist.

        Args:
            persona: Persona to compare.

        Returns:
            Name of similar artist.
        """
        name = getattr(persona, "artist_name", "Unknown")
        for slot in self.slots.values():
            if slot is not None:
                return getattr(slot, "artist_name", "Unknown")
        return "Unknown"

    # ------------------------------------------------------------------
    # Collaboration matrix
    # ------------------------------------------------------------------

    def calculate_compatibility(self, artist_a: str, artist_b: str) -> float:
        """Calculate compatibility between two artists by name.

        Uses cosine similarity of stylistic signature vectors.

        Args:
            artist_a: First artist name.
            artist_b: Second artist name.

        Returns:
            Affinity score (0.0-1.0).

        Raises:
            RosterError: If either artist is not in the roster.
        """
        persona_a = self.get_artist_by_name(artist_a)
        persona_b = self.get_artist_by_name(artist_b)

        if persona_a is None:
            raise RosterError(f"Artist not in roster: {artist_a}")
        if persona_b is None:
            raise RosterError(f"Artist not in roster: {artist_b}")

        vec_a = _extract_signature_vector(persona_a)
        vec_b = _extract_signature_vector(persona_b)

        similarity = cosine_similarity(vec_a, vec_b)
        # Normalize to 0-1
        affinity = (similarity + 1.0) / 2.0

        # Blend with collaboration affinities
        collab_a = getattr(persona_a, "collaboration_affinity", 0.5)
        collab_b = getattr(persona_b, "collaboration_affinity", 0.5)
        blended = affinity * 0.7 + ((collab_a + collab_b) / 2.0) * 0.3

        return round(max(0.0, min(1.0, blended)), 4)

    def _update_collaboration_for_artist(self, persona: Any) -> None:
        """Update collaboration scores for a newly added artist.

        Args:
            persona: The new artist.
        """
        name = getattr(persona, "artist_name", "")
        for slot in self.slots.values():
            if slot is None or slot is persona:
                continue
            other_name = getattr(slot, "artist_name", "")
            if other_name:
                try:
                    score = self.calculate_compatibility(name, other_name)
                    key = tuple(sorted([name, other_name]))
                    self.collaboration_matrix[key] = score
                except RosterError:
                    pass

    def build_full_collaboration_matrix(self) -> Dict[Tuple[str, str], float]:
        """Build complete collaboration matrix for all pairs.

        Returns:
            Dict mapping (name_a, name_b) -> affinity score.
        """
        artists = self.get_all_artists()
        names = [getattr(a, "artist_name", "") for a in artists if a]

        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                try:
                    score = self.calculate_compatibility(name_a, name_b)
                    key = tuple(sorted([name_a, name_b]))
                    self.collaboration_matrix[key] = score
                except RosterError:
                    pass

        return self.collaboration_matrix

    def get_collaboration_pairs(self, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """Get all artist pairs with affinity above threshold.

        Args:
            threshold: Minimum affinity score (default 0.5).

        Returns:
            List of (name_a, name_b, score) tuples.
        """
        return [
            (key[0], key[1], score)
            for key, score in self.collaboration_matrix.items()
            if score >= threshold
        ]

    def get_best_collaboration(self) -> Optional[Tuple[str, str, float]]:
        """Get the highest-scoring collaboration pair.

        Returns:
            (name_a, name_b, score) or None.
        """
        if not self.collaboration_matrix:
            return None
        best_key = max(self.collaboration_matrix, key=self.collaboration_matrix.get)
        return (best_key[0], best_key[1], self.collaboration_matrix[best_key])

    # ------------------------------------------------------------------
    # Taxonomy & metadata
    # ------------------------------------------------------------------

    def get_genre_distribution(self) -> Dict[str, int]:
        """Get distribution of genres in roster.

        Returns:
            Dict mapping genre to count.
        """
        dist: Dict[str, int] = {}
        for slot in self.slots.values():
            if slot is not None:
                genre = getattr(slot, "genre_anchor", "Unknown")
                dist[genre] = dist.get(genre, 0) + 1
        return dist

    def get_era_distribution(self) -> Dict[str, int]:
        """Get distribution of eras in roster.

        Returns:
            Dict mapping era to count.
        """
        dist: Dict[str, int] = {}
        for slot in self.slots.values():
            if slot is not None:
                era = getattr(slot, "era", "Unknown")
                dist[era] = dist.get(era, 0) + 1
        return dist

    def get_regional_distribution(self) -> Dict[str, int]:
        """Get distribution of regional origins.

        Returns:
            Dict mapping region to count.
        """
        dist: Dict[str, int] = {}
        for slot in self.slots.values():
            if slot is not None:
                region = getattr(slot, "regional_origin", "Unknown")
                if region:
                    dist[region] = dist.get(region, 0) + 1
        return dist

    def get_roster_summary(self) -> Dict[str, Any]:
        """Get comprehensive roster summary.

        Returns:
            Summary dictionary.
        """
        artists = self.get_all_artists()
        return {
            "total_slots": self.MAX_SLOTS,
            "occupied": len(artists),
            "available": self.MAX_SLOTS - len(artists),
            "genre_distribution": self.get_genre_distribution(),
            "era_distribution": self.get_era_distribution(),
            "regional_distribution": self.get_regional_distribution(),
            "collaboration_pairs": len(self.get_collaboration_pairs(0.5)),
            "best_collaboration": self.get_best_collaboration(),
            "artist_names": [getattr(a, "artist_name", "Unknown") for a in artists],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize roster to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "slots": {
                str(i): {
                    "artist": getattr(s, "artist_name", None) if s else None,
                    "persona_id": getattr(s, "persona_id", None) if s else None,
                }
                for i, s in self.slots.items()
            },
            "collaboration_matrix": {
                f"{k[0]}|{k[1]}": v
                for k, v in self.collaboration_matrix.items()
            },
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Roster Manager module loaded successfully.")

    roster = RosterManager()
    GeneticPersona, VoiceDifferentiationParameters = _get_genetic_persona_class()

    # Add some artists
    artists = [
        GeneticPersona(
            artist_name="Lil Durk", aliases=["Durkio"],
            genre_anchor="Hip-Hop", sub_genre_tags=["melodic drill"],
            era="2020s", regional_origin="Chicago, IL",
            voice_params=VoiceDifferentiationParameters(
                vocabulary_tier="Street", slang_density="High",
                emotional_range="Extreme", cultural_markers="Chicago",
            ),
            entropy_level=0.45, collaboration_affinity=0.7,
        ),
        GeneticPersona(
            artist_name="Lil Baby", aliases=["4PF"],
            genre_anchor="Hip-Hop", sub_genre_tags=["trap", "melodic"],
            era="2020s", regional_origin="Atlanta, GA",
            voice_params=VoiceDifferentiationParameters(
                vocabulary_tier="Street", slang_density="High",
                emotional_range="Wide", cultural_markers="Atlanta",
            ),
            entropy_level=0.5, collaboration_affinity=0.8,
        ),
        GeneticPersona(
            artist_name="Tay B", aliases=["Tay"],
            genre_anchor="Hip-Hop", sub_genre_tags=["melodic rap", "piano"],
            era="2020s", regional_origin="Kentucky",
            voice_params=VoiceDifferentiationParameters(
                vocabulary_tier="Mixed", slang_density="Medium",
                emotional_range="Wide", cultural_markers="Kentucky",
            ),
            entropy_level=0.5, collaboration_affinity=0.6,
        ),
    ]

    for artist in artists:
        slot = roster.add_artist(artist)
        print(f"  Added {artist.artist_name} -> slot {slot}")

    print(f"\nRoster: {roster.get_roster_summary()}")

    # Test collaboration scoring
    score = roster.calculate_compatibility("Lil Durk", "Lil Baby")
    print(f"\nLil Durk <-> Lil Baby compatibility: {score:.4f}")

    score2 = roster.calculate_compatibility("Lil Durk", "Tay B")
    print(f"Lil Durk <-> Tay B compatibility: {score2:.4f}")

    print(f"\nTop collaborations (threshold=0.5):")
    for a, b, s in roster.get_collaboration_pairs(0.5):
        print(f"  {a} + {b}: {s:.4f}")
