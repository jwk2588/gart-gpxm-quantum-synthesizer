"""
Entropic Script Layering System — GPXM Core.

Implements 5-layer entropic scripting:
    Layer 0: Base Rhythm (Vocabulary Matrix)
    Layer 1: Lexical (Flow Architecture / Syntax)
    Layer 2: Prosodic (Thematic Engine)
    Layer 3: Narrative (Prosodic Features)
    Layer 4: Cultural (Cultural Semantics)

Entropy control: 0.0 (strict) to 1.0 (maximum variation)

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 0: Vocabulary Matrix
# ---------------------------------------------------------------------------


@dataclass
class VocabularyMatrix:
    """Layer 0 — Vocabulary foundation.

    Defines the core vocabulary set, slang density, dialect markers,
    and register range for an artist's linguistic base.

    Attributes:
        core_terms: Primary vocabulary terms (50-200).
        slang_density: Frequency of colloquial usage.
        dialect_markers: Regional dialect indicators.
        technical_terms: Domain-specific terminology.
        register_range: Formality spectrum (formal <-> informal).
        filler_words: Common filler words for natural pauses.
    """

    core_terms: List[str] = field(default_factory=list)
    slang_density: str = "medium"  # low, medium, high, very_high
    dialect_markers: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)
    register_range: Tuple[float, float] = (0.2, 0.8)
    filler_words: List[str] = field(default_factory=lambda: ["uh", "you know", "like"])

    def get_term_sample(self, count: int = 10) -> List[str]:
        """Get a random sample of core terms.

        Args:
            count: Number of terms to sample.

        Returns:
            Sampled terms.
        """
        if not self.core_terms:
            return []
        return random.sample(self.core_terms, min(count, len(self.core_terms)))

    def merge(self, other: VocabularyMatrix, blend: float = 0.5) -> VocabularyMatrix:
        """Merge with another vocabulary matrix.

        Args:
            other: Other vocabulary matrix.
            blend: Blend ratio (0.0 = self, 1.0 = other).

        Returns:
            Merged vocabulary matrix.
        """
        self_terms = set(self.core_terms)
        other_terms = set(other.core_terms)
        shared = list(self_terms & other_terms)
        unique_self = list(self_terms - other_terms)
        unique_other = list(other_terms - other_terms)

        n_shared = int(len(shared) * (1 - blend))
        n_self = int(len(unique_self) * (1 - blend * 0.5))
        n_other = int(len(unique_other) * blend)

        merged_terms = shared[:n_shared] + unique_self[:n_self] + unique_other[:n_other]

        return VocabularyMatrix(
            core_terms=merged_terms,
            slang_density=other.slang_density if blend > 0.5 else self.slang_density,
            dialect_markers=list(set(self.dialect_markers + other.dialect_markers)),
            register_range=(
                self.register_range[0] * (1 - blend) + other.register_range[0] * blend,
                self.register_range[1] * (1 - blend) + other.register_range[1] * blend,
            ),
        )


# ---------------------------------------------------------------------------
# Layer 1: Flow Architecture (Syntax)
# ---------------------------------------------------------------------------


@dataclass
class FlowArchitecture:
    """Layer 1 — Syntactic flow patterns.

    Defines line length, rhyme schemes, enjambment, pause placement,
    cadence templates, and breath markers.

    Attributes:
        avg_line_length: Average syllables per line.
        rhyme_scheme: Primary rhyme scheme pattern.
        enjambment_frequency: How often lines run together (0.0-1.0).
        pause_placement: Where pauses occur (end-heavy, even, staccato, flowing).
        cadence_template: Named cadence pattern.
        breath_markers: Positions for breath marks.
        internal_rhyme_density: Frequency of internal rhymes (0.0-1.0).
    """

    avg_line_length: int = 16
    rhyme_scheme: str = "AABB"
    enjambment_frequency: float = 0.3
    pause_placement: str = "end-heavy"
    cadence_template: str = "standard_4_4"
    breath_markers: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    internal_rhyme_density: float = 0.2

    def get_line_variation(self) -> Tuple[int, int]:
        """Get allowed line length variation range.

        Returns:
            (min_length, max_length) in syllables.
        """
        variance = int(self.avg_line_length * 0.25)
        return (self.avg_line_length - variance, self.avg_line_length + variance)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "avg_line_length": self.avg_line_length,
            "rhyme_scheme": self.rhyme_scheme,
            "enjambment_frequency": self.enjambment_frequency,
            "pause_placement": self.pause_placement,
            "cadence_template": self.cadence_template,
            "breath_markers": self.breath_markers,
            "internal_rhyme_density": self.internal_rhyme_density,
        }


# ---------------------------------------------------------------------------
# Layer 2: Thematic Engine
# ---------------------------------------------------------------------------


@dataclass
class ThematicEngine:
    """Layer 2 — Thematic content engine.

    Defines primary/secondary themes, metaphor families, narrative
    structures, emotional arcs, and callback frequencies.

    Attributes:
        primary_themes: Ranked list of primary themes.
        secondary_themes: Secondary theme list.
        metaphor_families: Metaphor family categories.
        narrative_structures: Preferred narrative approaches.
        emotional_arc_patterns: Emotional progression templates.
        callback_frequency: How often callbacks/references occur (0.0-1.0).
        forbidden_topics: Topics to avoid.
        required_elements: Elements that must appear.
    """

    primary_themes: List[str] = field(default_factory=list)
    secondary_themes: List[str] = field(default_factory=list)
    metaphor_families: List[str] = field(default_factory=list)
    narrative_structures: List[str] = field(default_factory=lambda: ["linear"])
    emotional_arc_patterns: List[str] = field(default_factory=lambda: ["rise_fall_rise"])
    callback_frequency: float = 0.3
    forbidden_topics: List[str] = field(default_factory=list)
    required_elements: List[str] = field(default_factory=list)

    def get_theme_blend(self, partner: ThematicEngine) -> List[str]:
        """Get blended themes with a collaborator.

        Args:
            partner: Collaborator's thematic engine.

        Returns:
            Blended theme list.
        """
        shared = list(set(self.primary_themes) & set(partner.primary_themes))
        combined = list(set(self.primary_themes + partner.primary_themes))
        return shared + [t for t in combined if t not in shared]


# ---------------------------------------------------------------------------
# Layer 3: Prosodic Features
# ---------------------------------------------------------------------------


@dataclass
class ProsodicFeatures:
    """Layer 3 — Prosodic delivery features.

    Defines pitch range, intensity curves, ad-lib patterns, vocal
    effects, tempo relationships, and dynamic variation.

    Attributes:
        pitch_range_semitones: Vocal pitch range in semitones.
        intensity_curve: Intensity pattern (verse->chorus mapping).
        adlib_patterns: Named ad-lib patterns.
        adlib_frequency: How often ad-libs occur (0.0-1.0).
        vocal_effects: Preferred vocal effects.
        tempo_preference_bpm: Preferred tempo range.
        dynamic_variation: Dynamic range (0.0 = flat, 1.0 = extreme).
    """

    pitch_range_semitones: int = 12
    intensity_curve: str = "build_drop_build"
    adlib_patterns: List[str] = field(default_factory=list)
    adlib_frequency: float = 0.4
    vocal_effects: List[str] = field(default_factory=list)
    tempo_preference_bpm: Tuple[int, int] = (120, 140)
    dynamic_variation: float = 0.6

    def get_tempo_for_mood(self, mood: str) -> int:
        """Get recommended tempo for a mood.

        Args:
            mood: Mood descriptor.

        Returns:
            Recommended BPM.
        """
        mood_map: Dict[str, int] = {
            "aggressive": 140,
            "melancholic": 110,
            "energetic": 150,
            "chill": 120,
            "dark": 130,
        }
        return mood_map.get(mood, sum(self.tempo_preference_bpm) // 2)


# ---------------------------------------------------------------------------
# Layer 4: Cultural Semantics
# ---------------------------------------------------------------------------


@dataclass
class CulturalSemantics:
    """Layer 4 — Cultural integration.

    Defines geographic references, temporal markers, intertextual
    references, cultural symbols, audience addressing, and authenticity markers.

    Attributes:
        geographic_references: Location references.
        temporal_markers: Era-specific markers.
        intertextual_references: References to other works/artists.
        cultural_symbols: Cultural symbol vocabulary.
        audience_addressing: How the artist addresses listeners.
        authenticity_markers: Markers of authenticity.
        regional_slang: Region-specific slang terms.
    """

    geographic_references: List[str] = field(default_factory=list)
    temporal_markers: List[str] = field(default_factory=list)
    intertextual_references: List[str] = field(default_factory=list)
    cultural_symbols: List[str] = field(default_factory=list)
    audience_addressing: str = "direct"
    authenticity_markers: List[str] = field(default_factory=list)
    regional_slang: List[str] = field(default_factory=list)

    def get_cultural_fingerprint(self) -> str:
        """Get a cultural fingerprint string.

        Returns:
            Concatenated cultural markers.
        """
        parts = self.geographic_references[:2] + self.cultural_symbols[:3]
        return " | ".join(parts) if parts else "neutral"


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


@dataclass
class Guardrails:
    """Safety guardrails for entropic scripting.

    Defines what must never appear, what must always appear,
    and conditional behavior rules.

    Attributes:
        never_generate: List of forbidden content patterns.
        always_include: List of required elements.
        context_rules: Conditional behavior rules.
        max_entropy_ceiling: Hard ceiling for entropy (default 0.9).
        min_anchor_count: Minimum stable voice parameters.
    """

    never_generate: List[str] = field(default_factory=list)
    always_include: List[str] = field(default_factory=list)
    context_rules: Dict[str, str] = field(default_factory=dict)
    max_entropy_ceiling: float = 0.9
    min_anchor_count: int = 2

    def validate(self, content: str, entropy: float) -> Tuple[bool, List[str]]:
        """Validate content against guardrails.

        Args:
            content: Content to validate.
            entropy: Current entropy level.

        Returns:
            (is_valid, list_of_violations)
        """
        violations: List[str] = []

        # Check entropy ceiling
        if entropy > self.max_entropy_ceiling:
            violations.append(f"Entropy {entropy:.2f} exceeds ceiling {self.max_entropy_ceiling}")

        # Check forbidden content
        content_lower = content.lower()
        for forbidden in self.never_generate:
            if forbidden.lower() in content_lower:
                violations.append(f"Forbidden content detected: '{forbidden}'")

        # Check required elements
        for required in self.always_include:
            if required.lower() not in content_lower:
                violations.append(f"Missing required element: '{required}'")

        return (len(violations) == 0, violations)


# ---------------------------------------------------------------------------
# EntropicScript — 5-Layer Script Container
# ---------------------------------------------------------------------------


@dataclass
class EntropicScript:
    """Complete 5-layer entropic script for an artist persona.

    Encodes all stylistic dimensions from vocabulary through cultural
    semantics, with entropy calibration and guardrails.

    Attributes:
        artist_name: The artist this script models.
        entropy_profile: Master entropy level (0.0-1.0).
        layer_0_vocabulary: VocabularyMatrix.
        layer_1_syntax: FlowArchitecture.
        layer_2_themes: ThematicEngine.
        layer_3_prosody: ProsodicFeatures.
        layer_4_culture: CulturalSemantics.
        guardrails: Safety guardrails.
        version: Script version.
    """

    artist_name: str = ""
    entropy_profile: float = 0.5
    layer_0_vocabulary: VocabularyMatrix = field(default_factory=VocabularyMatrix)
    layer_1_syntax: FlowArchitecture = field(default_factory=FlowArchitecture)
    layer_2_themes: ThematicEngine = field(default_factory=ThematicEngine)
    layer_3_prosody: ProsodicFeatures = field(default_factory=ProsodicFeatures)
    layer_4_culture: CulturalSemantics = field(default_factory=CulturalSemantics)
    guardrails: Guardrails = field(default_factory=Guardrails)
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Validate entropy profile."""
        self.entropy_profile = max(0.0, min(1.0, self.entropy_profile))

    def get_layer_summary(self) -> Dict[str, Any]:
        """Get summary of all 5 layers.

        Returns:
            Dictionary with layer summaries.
        """
        return {
            "layer_0_vocabulary": {
                "term_count": len(self.layer_0_vocabulary.core_terms),
                "slang_density": self.layer_0_vocabulary.slang_density,
            },
            "layer_1_syntax": self.layer_1_syntax.to_dict(),
            "layer_2_themes": {
                "primary": self.layer_2_themes.primary_themes,
                "metaphor_families": self.layer_2_themes.metaphor_families,
            },
            "layer_3_prosody": {
                "pitch_range": self.layer_3_prosody.pitch_range_semitones,
                "tempo": self.layer_3_prosody.tempo_preference_bpm,
                "adlib_freq": self.layer_3_prosody.adlib_frequency,
            },
            "layer_4_culture": {
                "fingerprint": self.layer_4_culture.get_cultural_fingerprint(),
            },
            "entropy": self.entropy_profile,
            "guardrails": {
                "ceiling": self.guardrails.max_entropy_ceiling,
                "forbidden_count": len(self.guardrails.never_generate),
            },
        }

    def merge_for_collaboration(self, partner: EntropicScript) -> EntropicScript:
        """Create a blended script for collaboration.

        Args:
            partner: Collaborator's entropic script.

        Returns:
            Blended EntropicScript.
        """
        blended = copy.deepcopy(self)
        blended.artist_name = f"{self.artist_name}+{partner.artist_name}"
        blended.entropy_profile = (self.entropy_profile + partner.entropy_profile) / 2.0

        # Merge vocabulary
        blended.layer_0_vocabulary = self.layer_0_vocabulary.merge(
            partner.layer_0_vocabulary, blend=0.5
        )

        # Blend themes
        blended.layer_2_themes.primary_themes = self.layer_2_themes.get_theme_blend(
            partner.layer_2_themes
        )

        # Blend tempo
        min_tempo = min(
            self.layer_3_prosody.tempo_preference_bpm[0],
            partner.layer_3_prosody.tempo_preference_bpm[0],
        )
        max_tempo = max(
            self.layer_3_prosody.tempo_preference_bpm[1],
            partner.layer_3_prosody.tempo_preference_bpm[1],
        )
        blended.layer_3_prosody.tempo_preference_bpm = (min_tempo, max_tempo)

        # Merge cultural references
        blended.layer_4_culture.geographic_references = list(set(
            self.layer_4_culture.geographic_references +
            partner.layer_4_culture.geographic_references
        ))

        return blended

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "artist_name": self.artist_name,
            "entropy_profile": self.entropy_profile,
            "version": self.version,
            "layer_0_vocabulary": {
                "core_terms": self.layer_0_vocabulary.core_terms,
                "slang_density": self.layer_0_vocabulary.slang_density,
                "dialect_markers": self.layer_0_vocabulary.dialect_markers,
            },
            "layer_1_syntax": self.layer_1_syntax.to_dict(),
            "layer_2_themes": {
                "primary_themes": self.layer_2_themes.primary_themes,
                "forbidden_topics": self.layer_2_themes.forbidden_topics,
            },
            "layer_3_prosody": {
                "tempo_bpm": self.layer_3_prosody.tempo_preference_bpm,
                "pitch_range": self.layer_3_prosody.pitch_range_semitones,
            },
            "layer_4_culture": {
                "geographic_refs": self.layer_4_culture.geographic_references,
                "fingerprint": self.layer_4_culture.get_cultural_fingerprint(),
            },
            "guardrails": {
                "ceiling": self.guardrails.max_entropy_ceiling,
                "never": self.guardrails.never_generate,
            },
        }


# ---------------------------------------------------------------------------
# EntropyController
# ---------------------------------------------------------------------------


class EntropyLevel(Enum):
    """Discrete entropy calibration levels."""

    STRICT = (0.0, 0.2, "Strict emulation, minimal variation")
    FAITHFUL = (0.2, 0.4, "Faithful with subtle variations")
    BALANCED = (0.4, 0.6, "Balanced creativity")
    HIGH = (0.6, 0.8, "High variation, loose inspiration")
    MAXIMUM = (0.8, 1.0, "Maximum entropy, near-abstract interpretation")

    def __init__(self, low: float, high: float, description: str) -> None:
        self.low = low
        self.high = high
        self.description = description


class EntropyController:
    """Controller for entropy calibration across all 5 layers.

    Provides entropy level classification, per-layer adjustment,
    and validation against guardrails.
    """

    @staticmethod
    def calibrate(entropy: float) -> str:
        """Classify an entropy value into a calibration band.

        Args:
            entropy: Entropy value (0.0-1.0).

        Returns:
            Human-readable calibration description.

        Raises:
            ValueError: If entropy is outside [0.0, 1.0].
        """
        if not 0.0 <= entropy <= 1.0:
            raise ValueError(f"Entropy must be 0.0-1.0, got {entropy}")

        for level in EntropyLevel:
            if level.low <= entropy <= level.high:
                return f"[{level.name}] {level.description} (entropy={entropy:.2f})"

        return f"[UNKNOWN] entropy={entropy:.2f}"

    @staticmethod
    def get_level(entropy: float) -> EntropyLevel:
        """Get the EntropyLevel enum for a value.

        Args:
            entropy: Entropy value.

        Returns:
            Matching EntropyLevel.
        """
        for level in EntropyLevel:
            if level.low <= entropy <= level.high:
                return level
        return EntropyLevel.BALANCED

    @staticmethod
    def adjust_for_layer(
        base_entropy: float,
        layer_index: int,
        layer_sensitivity: float = 1.0,
    ) -> float:
        """Adjust entropy for a specific layer.

        Different layers respond differently to entropy:
        - Layer 0 (vocab): High sensitivity
        - Layer 1 (syntax): Medium sensitivity
        - Layer 2 (themes): Medium-low sensitivity
        - Layer 3 (prosody): Low sensitivity
        - Layer 4 (culture): Lowest sensitivity

        Args:
            base_entropy: Base entropy level.
            layer_index: Layer index (0-4).
            layer_sensitivity: Additional sensitivity multiplier.

        Returns:
            Adjusted entropy for the layer.
        """
        sensitivities = [1.0, 0.8, 0.6, 0.4, 0.3]
        sens = sensitivities[layer_index] * layer_sensitivity
        adjusted = base_entropy * sens
        return max(0.0, min(1.0, adjusted))

    @staticmethod
    def validate_safety(entropy: float, guardrails: Guardrails) -> Tuple[bool, List[str]]:
        """Validate entropy against safety guardrails.

        Args:
            entropy: Current entropy.
            guardrails: Safety guardrails.

        Returns:
            (is_safe, list_of_violations)
        """
        violations: List[str] = []
        if entropy > guardrails.max_entropy_ceiling:
            violations.append(
                f"Entropy {entropy:.2f} exceeds ceiling {guardrails.max_entropy_ceiling}"
            )
        if entropy < 0.0 or entropy > 1.0:
            violations.append(f"Entropy out of range: {entropy}")
        return (len(violations) == 0, violations)

    @staticmethod
    def generate_report(script: EntropicScript) -> str:
        """Generate a human-readable entropy report.

        Args:
            script: EntropicScript to report on.

        Returns:
            Formatted report string.
        """
        summary = script.get_layer_summary()
        report_lines = [
            f"Entropic Script Report: {script.artist_name}",
            f"Entropy Profile: {script.entropy_profile:.2f}",
            f"Calibration: {EntropyController.calibrate(script.entropy_profile)}",
            "",
            "Layer Summary:",
            f"  L0 Vocabulary: {summary['layer_0_vocabulary']['term_count']} terms, "
            f"slang={summary['layer_0_vocabulary']['slang_density']}",
            f"  L1 Syntax: {summary['layer_1_syntax']['rhyme_scheme']}, "
            f"avg_line={summary['layer_1_syntax']['avg_line_length']}",
            f"  L2 Themes: {summary['layer_2_themes']['primary_themes']}",
            f"  L3 Prosody: pitch={summary['layer_3_prosody']['pitch_range']}st, "
            f"tempo={summary['layer_3_prosody']['tempo']}, "
            f"adlibs={summary['layer_3_prosody']['adlib_freq']}",
            f"  L4 Culture: {summary['layer_4_culture']['fingerprint']}",
            "",
            f"Guardrails: ceiling={summary['guardrails']['ceiling']}, "
            f"forbidden={summary['guardrails']['forbidden_count']}",
        ]
        return "\n".join(report_lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Entropic Scripting module loaded successfully.")

    # Create a sample entropic script for Lil Durk
    script = EntropicScript(
        artist_name="Lil Durk",
        entropy_profile=0.45,
        layer_0_vocabulary=VocabularyMatrix(
            core_terms=["OTF", "trenches", "Drill", "The Voice", "smurk"],
            slang_density="high",
            dialect_markers=["Chicago", "Midwest"],
            register_range=(0.1, 0.7),
        ),
        layer_1_syntax=FlowArchitecture(
            avg_line_length=14,
            rhyme_scheme="AABB",
            pause_placement="end-heavy",
            enjambment_frequency=0.2,
            internal_rhyme_density=0.3,
        ),
        layer_2_themes=ThematicEngine(
            primary_themes=["street life", "loss", "loyalty", "struggle"],
            metaphor_families=["war", "family", "survival"],
            emotional_arc_patterns=["melancholic_build"],
            forbidden_topics=["snitching"],
        ),
        layer_3_prosody=ProsodicFeatures(
            pitch_range_semitones=14,
            intensity_curve="melodic_build",
            adlib_patterns=["yeah yeah", "let's go", "gang"],
            adlib_frequency=0.5,
            tempo_preference_bpm=(130, 160),
        ),
        layer_4_culture=CulturalSemantics(
            geographic_references=["Chicago", "Englewood", "O Block"],
            cultural_symbols=["OTF chain", "trench symbolism"],
            audience_addressing="direct",
            regional_slang=["thot", "finna", "jit"],
        ),
    )

    print(f"\n{EntropyController.generate_report(script)}")

    # Test collaboration merge
    partner_script = EntropicScript(
        artist_name="Lil Baby",
        entropy_profile=0.5,
        layer_0_vocabulary=VocabularyMatrix(
            core_terms=["4PF", "lil bit", "drip", "QC"],
            slang_density="high",
            dialect_markers=["Atlanta"],
        ),
        layer_3_prosody=ProsodicFeatures(
            tempo_preference_bpm=(140, 160),
        ),
        layer_4_culture=CulturalSemantics(
            geographic_references=["Atlanta", "Oakland City"],
        ),
    )

    blended = script.merge_for_collaboration(partner_script)
    print(f"\n--- Blended Script ---")
    print(f"Name: {blended.artist_name}")
    print(f"Tempo: {blended.layer_3_prosody.tempo_preference_bpm}")
    print(f"Themes: {blended.layer_2_themes.primary_themes}")
