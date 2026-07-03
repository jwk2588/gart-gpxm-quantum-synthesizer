"""
"Stash Box" by Tay B & Lil Durk — Derivative Reconstruction Engine.

Blends Tay B's melodic production with Lil Durk's drill storytelling
for faithful track reconstruction. Key elements:

    - Tay B: melodic rap, piano-heavy production, singing-rap hybrid
    - Lil Durk: melodic drill, storytelling, emotional vulnerability
    - BPM: 130-150 (Tay B) overlapping with 130-160 (Durk)
    - Cross-persona style blending with dominance rules
    - Emotional arc: intro -> verse -> hook -> bridge -> outro

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StyleBrief:
    """Style brief for track reconstruction.

    Attributes:
        primary_artist: Lead artist name.
        featured_artist: Feature artist name.
        target_bpm: Target tempo.
        target_key: Musical key.
        mood: Overall mood descriptor.
        reference_tracks: List of reference track names.
        structural_preference: Song structure preference.
    """

    primary_artist: str = "Tay B"
    featured_artist: str = "Lil Durk"
    target_bpm: int = 140
    target_key: str = "C minor"
    mood: str = "melodic_dark"
    reference_tracks: List[str] = field(default_factory=list)
    structural_preference: str = "standard"


@dataclass
class BeatPattern:
    """A reconstructed beat pattern.

    Attributes:
        bpm: Tempo.
        drum_pattern: Drum pattern description.
        melody_layers: Melodic layer descriptions.
        piano_chords: Piano chord progression.
        808_pattern: 808/bass pattern.
        transition_points: Bar numbers for transitions.
    """

    bpm: int = 140
    drum_pattern: str = "trap_standard"
    melody_layers: List[str] = field(default_factory=list)
    piano_chords: List[str] = field(default_factory=list)
    _808_pattern: str = "rolling_808s"
    transition_points: List[int] = field(default_factory=lambda: [8, 16, 24, 32])


@dataclass
class LyricalSection:
    """A section of reconstructed lyrics.

    Attributes:
        section_type: Section type (intro, verse, hook, bridge, outro).
        lyrics: Reconstructed lyrics text.
        cadence: Cadence description.
        dominant_persona: Which persona is dominant here.
        bar_count: Number of bars.
    """

    section_type: str = "verse"
    lyrics: str = ""
    cadence: str = "melodic_flow"
    dominant_persona: str = ""
    bar_count: int = 16


@dataclass
class TrackConcept:
    """Complete reconstructed track concept.

    Attributes:
        title: Track title.
        primary_artist: Lead artist.
        featured_artist: Feature artist.
        bpm: Tempo.
        key: Musical key.
        sections: Ordered track sections.
        beat_pattern: Underlying beat pattern.
        emotional_arc: Emotional progression description.
        style_blend: BlendedStyle used.
    """

    title: str = ""
    primary_artist: str = ""
    featured_artist: str = ""
    bpm: int = 140
    key: str = ""
    sections: List[LyricalSection] = field(default_factory=list)
    beat_pattern: Optional[BeatPattern] = None
    emotional_arc: str = ""
    style_blend: Optional[Any] = None


@dataclass
class BlendedStyle:
    """Result of blending two artist styles.

    Attributes:
        dominant_persona: Which persona dominates.
        shared_elements: Overlapping style elements.
        dominance_rules: Lead vs feature dynamics.
        transition_scripts: Handoff point scripts.
        blended_entropy: Combined entropy level.
    """

    dominant_persona: str = ""
    shared_elements: List[str] = field(default_factory=list)
    dominance_rules: Dict[str, Any] = field(default_factory=dict)
    transition_scripts: List[str] = field(default_factory=list)
    blended_entropy: float = 0.5


# ---------------------------------------------------------------------------
# StashBoxReconstructionEngine
# ---------------------------------------------------------------------------


class StashBoxReconstructionEngine:
    """Reconstruction engine for 'Stash Box' by Tay B & Lil Durk.

    Blends Tay B's melodic, piano-heavy production style with Lil
    Durk's melodic drill storytelling approach. Reconstructs the
    full track structure with cross-persona handoff points.

    Track Structure:
        Intro (Tay B piano + atmospheric) -> 8 bars
        Verse 1 (Lil Durk lead) -> 16 bars
        Hook (Both, Tay B melody dominant) -> 8 bars
        Verse 2 (Tay B lead) -> 16 bars
        Bridge (Lil Durk emotional) -> 8 bars
        Hook (Both, full energy) -> 8 bars
        Outro (Tay B piano fade) -> 4 bars
    """

    def __init__(self) -> None:
        self.tay_b_persona = self._load_tay_b_persona()
        self.lil_durk_persona = self._load_lil_durk_persona()
        self._current_blend: Optional[BlendedStyle] = None

    def _load_tay_b_persona(self) -> Dict[str, Any]:
        """Load Tay B persona parameters.

        Tay B: melodic rap, piano-heavy production, singing-rap hybrid
        - Entropy: 0.5 (balanced creativity)
        - BPM preference: 130-150
        - Key characteristics: melodic hooks, piano-driven beats

        Returns:
            Persona dictionary.
        """
        return {
            "artist_name": "Tay B",
            "aliases": ["Tay"],
            "genre_anchor": "Hip-Hop",
            "sub_genre_tags": ["melodic rap", "piano trap", "singing-rap"],
            "era": "2020s",
            "regional_origin": "Kentucky",
            "voice_params": {
                "vocabulary_tier": "Mixed",
                "slang_density": "Medium",
                "sentence_length": "Medium",
                "pause_pattern": "flowing",
                "emotional_range": "Wide",
                "cultural_markers": "Kentucky",
                "narrative_mode": "Linear",
                "call_response": "Direct",
            },
            "entropy_level": 0.5,
            "collaboration_affinity": 0.6,
            "bpm_range": (130, 150),
            "key_preference": ["C minor", "A minor", "G minor"],
            "signature_elements": [
                "piano_melodies",
                "singing_rap_hybrid",
                "emotional_hooks",
                "atmospheric_pads",
            ],
            "production_style": "piano_heavy_trap",
            "dominant_in_sections": ["intro", "hook", "verse_2", "outro"],
        }

    def _load_lil_durk_persona(self) -> Dict[str, Any]:
        """Load Lil Durk persona parameters.

        Lil Durk: melodic drill, storytelling, emotional vulnerability
        - Entropy: 0.45 (faithful with variation)
        - BPM preference: 130-160
        - Key characteristics: drill flows, emotional narratives

        Returns:
            Persona dictionary.
        """
        return {
            "artist_name": "Lil Durk",
            "aliases": ["Durkio", "The Voice"],
            "genre_anchor": "Hip-Hop",
            "sub_genre_tags": ["melodic drill", "Chicago drill", "trap"],
            "era": "2020s",
            "regional_origin": "Chicago, IL",
            "voice_params": {
                "vocabulary_tier": "Street",
                "slang_density": "High",
                "sentence_length": "Medium",
                "pause_pattern": "end-heavy",
                "emotional_range": "Extreme",
                "cultural_markers": "Chicago",
                "narrative_mode": "Cinematic",
                "call_response": "Direct",
            },
            "entropy_level": 0.45,
            "collaboration_affinity": 0.7,
            "bpm_range": (130, 160),
            "key_preference": ["C minor", "F minor", "D minor"],
            "signature_elements": [
                "melodic_drill_flow",
                "storytelling_verses",
                "emotional_vulnerability",
                "auto_tune_melodies",
                "street_narratives",
            ],
            "production_style": "melodic_drill",
            "dominant_in_sections": ["verse_1", "bridge"],
        }

    def reconstruct_track(self, style_brief: Optional[StyleBrief] = None) -> TrackConcept:
        """Reconstruct the 'Stash Box' track concept.

        Creates a full track reconstruction with cross-persona style
        blending, beat patterns, and lyrical sections.

        Args:
            style_brief: Optional style brief override.

        Returns:
            Complete TrackConcept.
        """
        brief = style_brief or StyleBrief()

        # Generate collaboration blend
        blend = self.generate_collaboration_blend(
            lead=self.tay_b_persona,
            feature=self.lil_durk_persona,
        )
        self._current_blend = blend

        # Determine BPM from overlap
        tay_bpm = self.tay_b_persona["bpm_range"]
        durk_bpm = self.lil_durk_persona["bpm_range"]
        overlap_low = max(tay_bpm[0], durk_bpm[0])
        overlap_high = min(tay_bpm[1], durk_bpm[1])
        bpm = brief.target_bpm if overlap_low <= brief.target_bpm <= overlap_high else overlap_low

        # Generate beat pattern
        beat = self._reconstruct_beat_pattern(bpm, blend)

        # Generate sections
        sections = self._reconstruct_sections(blend)

        # Determine emotional arc
        emotional_arc = self._build_emotional_arc(sections)

        return TrackConcept(
            title="Stash Box",
            primary_artist=brief.primary_artist,
            featured_artist=brief.featured_artist,
            bpm=bpm,
            key=brief.target_key,
            sections=sections,
            beat_pattern=beat,
            emotional_arc=emotional_arc,
            style_blend=blend,
        )

    def generate_collaboration_blend(
        self,
        lead: Dict[str, Any],
        feature: Dict[str, Any],
    ) -> BlendedStyle:
        """Generate a blended style for collaboration.

        Identifies overlapping style elements (compatibility zones),
        defines dominance rules for lead vs. feature dynamics,
        and creates transition scripts for handoff points.

        Args:
            lead: Lead artist persona dict.
            feature: Feature artist persona dict.

        Returns:
            BlendedStyle with merged parameters.
        """
        lead_elements = set(lead.get("signature_elements", []))
        feature_elements = set(feature.get("signature_elements", []))

        # Find overlapping elements (compatibility zones)
        shared = list(lead_elements & feature_elements)
        lead_unique = list(lead_elements - feature_elements)
        feature_unique = list(feature_elements - lead_elements)

        # Determine dominant persona per section
        dominance_rules: Dict[str, str] = {}
        for section in lead.get("dominant_in_sections", []):
            dominance_rules[section] = lead["artist_name"]
        for section in feature.get("dominant_in_sections", []):
            if section not in dominance_rules:
                dominance_rules[section] = feature["artist_name"]

        # Shared sections
        shared_sections = set(lead.get("dominant_in_sections", [])) & set(feature.get("dominant_in_sections", []))
        for section in shared_sections:
            dominance_rules[section] = "both"

        # Create transition scripts
        transitions: List[str] = []
        section_order = ["intro", "verse_1", "hook", "verse_2", "bridge", "hook", "outro"]
        for i in range(len(section_order) - 1):
            current = section_order[i]
            next_s = section_order[i + 1]
            current_dominant = dominance_rules.get(current, "unknown")
            next_dominant = dominance_rules.get(next_s, "unknown")

            if current_dominant != next_dominant and current_dominant != "both" and next_dominant != "both":
                transitions.append(
                    f"Transition: {current}({current_dominant}) -> {next_s}({next_dominant}): "
                    f"Fade out {current_dominant} ad-libs, bring in {next_dominant} vocal tone"
                )
            elif current_dominant == "both" or next_dominant == "both":
                transitions.append(
                    f"Transition: {current} -> {next_s}: Layered vocals, both present"
                )

        # Blend entropy
        blended_entropy = (lead.get("entropy_level", 0.5) + feature.get("entropy_level", 0.5)) / 2.0

        return BlendedStyle(
            dominant_persona=lead["artist_name"],
            shared_elements=shared,
            dominance_rules=dominance_rules,
            transition_scripts=transitions,
            blended_entropy=round(blended_entropy, 3),
        )

    def _reconstruct_beat_pattern(self, bpm: int, blend: BlendedStyle) -> BeatPattern:
        """Reconstruct the beat pattern for 'Stash Box'.

        Combines Tay B's piano-heavy melodic style with Durk's
        drill-influenced percussion.

        Args:
            bpm: Target tempo.
            blend: Blended style configuration.

        Returns:
            BeatPattern.
        """
        return BeatPattern(
            bpm=bpm,
            drum_pattern="trap_with_drill_influence",
            melody_layers=[
                "piano_riff_tay_b_style",  # Tay B's signature piano
                "atmospheric_pad",
                "sub_bass_808",
                "light_percussion",
            ],
            piano_chords=[
                "Cm7", "Abmaj7", "Ebmaj7", "Bb7",  # Minor-key emotional progression
                "Cm7", "Fm7", "Gm7", "Cm7",
            ],
            _808_pattern="rolling_melodic_808s",
            transition_points=[8, 24, 32, 48, 56, 64, 68],
        )

    def _reconstruct_sections(self, blend: BlendedStyle) -> List[LyricalSection]:
        """Reconstruct all track sections.

        Args:
            blend: Blended style configuration.

        Returns:
            List of track sections.
        """
        sections: List[LyricalSection] = []

        # Intro: Tay B piano + atmospheric
        sections.append(LyricalSection(
            section_type="intro",
            lyrics=(
                "[Piano melody fades in...]\n"
                "Yeah...\n"
                "Tay B, Lil Durk...\n"
                "Let's get it..."
            ),
            cadence="sparse_melodic",
            dominant_persona="Tay B",
            bar_count=8,
        ))

        # Verse 1: Lil Durk lead - drill storytelling
        sections.append(LyricalSection(
            section_type="verse",
            lyrics=(
                "[Lil Durk - melodic drill flow]\n"
                "Came up from the trenches, seen it all\n"
                "Every night I'm prayin' that we don't fall\n"
                "The stash box hidden where they can't find\n"
                "Memories replayin' in my mind\n"
                "Lost some brothers to this life we chose\n"
                "Pain run deep, everybody knows\n"
                "But we keep goin', that's the only way\n"
                "Turn the pain to purpose, make 'em pay"
            ),
            cadence="melodic_drill_storytelling",
            dominant_persona="Lil Durk",
            bar_count=16,
        ))

        # Hook: Both - Tay B melody dominant
        sections.append(LyricalSection(
            section_type="hook",
            lyrics=(
                "[Both - Tay B melody lead]\n"
                "In the stash box, keep it all there\n"
                "Memories and dreams, too much to bear\n"
                "Stash box, everything we hold dear\n"
                "Stash box, wish you were still here\n"
                "\n[Ad-libs: Durk] Yeah, let's go\n"
                "[Ad-libs: Tay B] Ooh, ooh"
            ),
            cadence="melodic_singing_hook",
            dominant_persona="both",
            bar_count=8,
        ))

        # Verse 2: Tay B lead - melodic rap
        sections.append(LyricalSection(
            section_type="verse",
            lyrics=(
                "[Tay B - singing-rap hybrid]\n"
                "Piano keys playin' while I'm thinkin' 'bout you\n"
                "Every melody I sing got me runnin' right through\n"
                "Kentucky to Chicago, yeah we bridged that gap\n"
                "Stash box of memories, no we can't look back\n"
                "Used to dream about it, now we livin' it\n"
                "Every hurt, every scar, that's what made us rich\n"
                "In the melody I find my peace\n"
                "Stash box secrets, let the pain release"
            ),
            cadence="singing_rap_hybrid",
            dominant_persona="Tay B",
            bar_count=16,
        ))

        # Bridge: Lil Durk emotional
        sections.append(LyricalSection(
            section_type="bridge",
            lyrics=(
                "[Lil Durk - emotional vulnerability]\n"
                "Sometimes I open up that box and cry\n"
                "Wonder why the real ones gotta die\n"
                "The Voice speakin' from a broken place\n"
                "But I know God got me in this race..."
            ),
            cadence="emotional_spoken",
            dominant_persona="Lil Durk",
            bar_count=8,
        ))

        # Hook 2: Both - full energy
        sections.append(LyricalSection(
            section_type="hook",
            lyrics=(
                "[Both - full energy]\n"
                "In the stash box, keep it all there\n"
                "Memories and dreams, too much to bear\n"
                "Stash box, everything we hold dear\n"
                "Stash box, wish you were still here\n"
                "In the stash box (yeah yeah)\n"
                "Stash box (let's go)\n"
                "Stash box (ooh)\n"
                "Stash box..."
            ),
            cadence="full_energy_melodic",
            dominant_persona="both",
            bar_count=8,
        ))

        # Outro: Tay B piano fade
        sections.append(LyricalSection(
            section_type="outro",
            lyrics=(
                "[Piano melody fades out...]\n"
                "Stash box...\n"
                "Everything we got...\n"
                "Tay B...\n"
                "Lil Durk...\n"
                "[Fade to silence]"
            ),
            cadence="piano_fade",
            dominant_persona="Tay B",
            bar_count=4,
        ))

        return sections

    def _build_emotional_arc(self, sections: List[LyricalSection]) -> str:
        """Build emotional arc description from sections.

        Args:
            sections: Track sections.

        Returns:
            Emotional arc string.
        """
        arc_points: List[str] = []
        for section in sections:
            mood_map: Dict[str, str] = {
                "intro": "contemplative",
                "verse": "introspective",
                "hook": "cathartic",
                "bridge": "vulnerable",
                "outro": "resigned",
            }
            arc_points.append(mood_map.get(section.section_type, "neutral"))

        return " -> ".join(arc_points)

    def generate_track_report(self, concept: TrackConcept) -> str:
        """Generate a human-readable track reconstruction report.

        Args:
            concept: The track concept.

        Returns:
            Formatted report string.
        """
        lines = [
            f"{'='*60}",
            f"  TRACK RECONSTRUCTION: {concept.title}",
            f"  {concept.primary_artist} ft. {concept.featured_artist}",
            f"{'='*60}",
            f"",
            f"Tempo: {concept.bpm} BPM",
            f"Key: {concept.key}",
            f"Emotional Arc: {concept.emotional_arc}",
            f"",
            f"--- Beat Pattern ---",
        ]

        if concept.beat_pattern:
            bp = concept.beat_pattern
            lines.extend([
                f"  Drum Pattern: {bp.drum_pattern}",
                f"  Melody Layers: {bp.melody_layers}",
                f"  Piano Chords: {bp.piano_chords[:4]}...",
                f"  808 Pattern: {bp._808_pattern}",
                f"",
            ])

        lines.extend([
            f"--- Track Structure ---",
            f"  Total Sections: {len(concept.sections)}",
            f"  Total Bars: {sum(s.bar_count for s in concept.sections)}",
            f"",
        ])

        for i, section in enumerate(concept.sections):
            lines.extend([
                f"  [{i+1}] {section.section_type.upper()}",
                f"      Dominant: {section.dominant_persona}",
                f"      Cadence: {section.cadence}",
                f"      Bars: {section.bar_count}",
                f"",
            ])

        if concept.style_blend:
            lines.extend([
                f"--- Style Blend ---",
                f"  Blended Entropy: {concept.style_blend.blended_entropy}",
                f"  Shared Elements: {concept.style_blend.shared_elements}",
                f"",
                f"--- Transition Scripts ---",
            ])
            for script in concept.style_blend.transition_scripts:
                lines.append(f"  {script}")

        lines.append(f"{'='*60}")

        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Stash Box Reconstruction Engine loaded successfully.")

    engine = StashBoxReconstructionEngine()
    concept = engine.reconstruct_track()

    print(f"\n{engine.generate_track_report(concept)}")

    print(f"\n--- Collaboration Blend ---")
    if concept.style_blend:
        print(f"Shared style elements: {concept.style_blend.shared_elements}")
        print(f"Blended entropy: {concept.style_blend.blended_entropy}")
        print(f"Total bars: {sum(s.bar_count for s in concept.sections)}")
