"""
DAW Sub-Agent Utilities — Digital Audio Workstation Sub-Agent for GART v3.0.

Provides signal processing, sidechain compression, dynamic BPM cadence shifting,
and master mixing evaluation for the dual-swarm AI music production system.

Components:
    - SidechainCompressor: Semantic ducking when "Reverse Poor Man's Flex" triggers
    - DynamicBPMShifter: Cadence transitions (85 BPM -> 128 BPM range)
    - MasterMixerEvaluator: EQ cut + Limiter + Reverb tail pipeline
    - VerseStem: Musical stem data structure for verse construction

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain data structures
# ---------------------------------------------------------------------------


@dataclass
class Note:
    """A single musical note.

    Attributes:
        pitch: MIDI note number (0-127).
        velocity: Note velocity/intensity (0-127).
        start_time: Start time in beats.
        duration: Duration in beats.
    """

    pitch: int
    velocity: int
    start_time: float
    duration: float

    def __post_init__(self) -> None:
        """Validate note parameters."""
        if not (0 <= self.pitch <= 127):
            raise ValueError(f"MIDI pitch must be 0-127, got {self.pitch}")
        if not (0 <= self.velocity <= 127):
            raise ValueError(f"Velocity must be 0-127, got {self.velocity}")
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")


@dataclass
class Pattern:
    """A rhythmic pattern composed of notes.

    Attributes:
        notes: List of Note objects.
        duration_beats: Total duration in beats.
    """

    notes: List[Note] = field(default_factory=list)
    duration_beats: float = 4.0

    def quantize(self, grid_size: float = 0.25) -> None:
        """Snap all note start times to a quantized grid.

        Args:
            grid_size: Grid resolution in beats (default 1/16th = 0.25).
        """
        for note in self.notes:
            note.start_time = round(note.start_time / grid_size) * grid_size

    def humanize(self, amount: float = 0.05) -> None:
        """Add random timing variation for a human feel.

        Args:
            amount: Maximum random deviation in beats.
        """
        for note in self.notes:
            jitter = np.random.uniform(-amount, amount)
            note.start_time = max(0.0, note.start_time + jitter)


@dataclass
class EQProfile:
    """Equalization profile for signal shaping.

    Attributes:
        high_pass_freq: High-pass filter cutoff in Hz.
        low_pass_freq: Low-pass filter cutoff in Hz.
        peaking_filters: List of (freq_hz, gain_db, q_factor) peaking filters.
    """

    high_pass_freq: float = 80.0
    low_pass_freq: float = 12000.0
    peaking_filters: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class ReverbConfig:
    """Reverb configuration for tail generation.

    Attributes:
        decay_time: Decay time in seconds.
        wet_level: Wet signal mix level (0.0-1.0).
        room_size: Simulated room size (0.0-1.0).
    """

    decay_time: float = 2.5
    wet_level: float = 0.15
    room_size: float = 0.6


# Signal type alias: numpy array of float32 samples
Signal = np.ndarray


# ---------------------------------------------------------------------------
# VerseStem — Musical stem for verse construction
# ---------------------------------------------------------------------------


class VerseStem:
    """A musical stem (track) for verse construction in the DAW.

    A stem represents a single instrumental or vocal track with its
    associated pattern, intent metadata, and semantic weight.

    Attributes:
        name: Human-readable stem name (e.g., "ybnba_kick", "hive_mind_synth").
        pattern: The rhythmic/melodic Pattern.
        instrument: Instrument type identifier.
        bpm: Tempo in beats per minute.
        key: Musical key (e.g., "C minor", "F# major").
        intent: Semantic intent string for this stem.
        semantic_weight: Weight value (0.0-1.0) for mixing decisions.
    """

    def __init__(
        self,
        name: str,
        pattern: Pattern,
        instrument: str,
        bpm: int = 130,
        key: str = "C minor",
        intent: str = "",
        semantic_weight: float = 1.0,
    ) -> None:
        self.name = name
        self.pattern = pattern
        self.instrument = instrument
        self.bpm = bpm
        self.key = key
        self.intent = intent
        self.semantic_weight = max(0.0, min(1.0, semantic_weight))
        self._buffer: Optional[Signal] = None

    def render(self, sample_rate: int = 44100) -> Signal:
        """Render the pattern to an audio buffer.

        Args:
            sample_rate: Audio sample rate in Hz.

        Returns:
            Numpy array of float32 audio samples.
        """
        if self._buffer is not None:
            return self._buffer

        # Simple synthesis: generate sine waves for each note
        total_duration = self.pattern.duration_beats * (60.0 / self.bpm)
        num_samples = int(total_duration * sample_rate)
        buffer = np.zeros(num_samples, dtype=np.float32)

        for note in self.pattern.notes:
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            note_start = note.start_time * (60.0 / self.bpm)
            note_dur = note.duration * (60.0 / self.bpm)
            start_sample = int(note_start * sample_rate)
            end_sample = min(int((note_start + note_dur) * sample_rate), num_samples)
            note_samples = end_sample - start_sample

            if note_samples > 0 and freq > 0:
                t = np.arange(note_samples) / sample_rate
                velocity_norm = note.velocity / 127.0
                envelope = np.exp(-t * 5.0)  # Simple decay envelope
                wave = velocity_norm * np.sin(2.0 * np.pi * freq * t) * envelope
                buffer[start_sample:end_sample] += wave

        # Normalize
        max_val = np.max(np.abs(buffer))
        if max_val > 0:
            buffer = buffer / max_val * 0.5

        self._buffer = buffer
        return buffer

    def transpose(self, semitones: int) -> None:
        """Transpose all notes in the stem by given semitones.

        Args:
            semitones: Number of semitones to transpose (positive = up).
        """
        for note in self.pattern.notes:
            note.pitch = max(0, min(127, note.pitch + semitones))
        self._buffer = None  # Invalidate cached buffer

    def change_bpm(self, new_bpm: int) -> None:
        """Change the BPM of this stem.

        Args:
            new_bpm: New beats-per-minute value.

        Raises:
            ValueError: If new_bpm is not positive.
        """
        if new_bpm <= 0:
            raise ValueError(f"BPM must be positive, got {new_bpm}")
        self.bpm = new_bpm
        self._buffer = None  # Invalidate cached buffer


# ---------------------------------------------------------------------------
# TextSequence — Text with cadence metadata
# ---------------------------------------------------------------------------


@dataclass
class TextSequence:
    """A text sequence with associated cadence and style metadata.

    Attributes:
        text: The raw text content.
        speed: Cadence speed label ("slow", "moderate", "rapid").
        style: Style identifier (e.g., "Lil_Baby_Triplets", "Mos_Def_Piano").
        bpm: Associated tempo.
    """

    text: str
    speed: str = "moderate"
    style: str = "default"
    bpm: int = 130


@dataclass
class CypherResult:
    """Result of a cypher (verse generation) operation.

    Attributes:
        text: Generated verse text.
        stems: List of associated VerseStem objects.
        quality_score: Overall quality score (0.0-1.0).
        metadata: Additional metadata dictionary.
    """

    text: str
    stems: List[VerseStem] = field(default_factory=list)
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Signal Processing Utilities
# ---------------------------------------------------------------------------


def _apply_high_pass(signal: Signal, cutoff_hz: float, sample_rate: int = 44100) -> Signal:
    """Apply a simple high-pass filter using first-order IIR.

    Args:
        signal: Input audio signal.
        cutoff_hz: Cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.

    Returns:
        Filtered signal.
    """
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)
    result = np.zeros_like(signal)
    result[0] = signal[0]
    for i in range(1, len(signal)):
        result[i] = alpha * (result[i - 1] + signal[i] - signal[i - 1])
    return result


def _apply_low_pass(signal: Signal, cutoff_hz: float, sample_rate: int = 44100) -> Signal:
    """Apply a simple low-pass filter using first-order IIR.

    Args:
        signal: Input audio signal.
        cutoff_hz: Cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.

    Returns:
        Filtered signal.
    """
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)
    result = np.zeros_like(signal)
    result[0] = signal[0]
    for i in range(1, len(signal)):
        result[i] = result[i - 1] + alpha * (signal[i] - result[i - 1])
    return result


def _apply_limiter(signal: Signal, threshold_db: float = -1.0) -> Signal:
    """Apply a brick-wall limiter to the signal.

    Args:
        signal: Input audio signal.
        threshold_db: Limiter threshold in dB (relative to full scale).

    Returns:
        Limited signal.
    """
    threshold_linear = 10.0 ** (threshold_db / 20.0)
    result = np.clip(signal, -threshold_linear, threshold_linear)
    return result


def _apply_reverb(
    signal: Signal,
    decay_time: float = 2.5,
    wet_level: float = 0.15,
    sample_rate: int = 44100,
) -> Signal:
    """Apply a simple comb-filter reverb effect.

    Args:
        signal: Input audio signal.
        decay_time: Reverb decay time in seconds.
        wet_level: Wet signal mix level (0.0-1.0).
        sample_rate: Sample rate in Hz.

    Returns:
        Signal with reverb tail applied.
    """
    delay_samples = int(0.05 * sample_rate)  # 50ms delay
    num_echoes = min(8, int(decay_time * 10))
    wet = np.zeros_like(signal)

    for i in range(num_echoes):
        delay = delay_samples * (i + 1)
        gain = (0.5 ** (i + 1)) * np.exp(-i / decay_time)
        if delay < len(signal):
            wet[delay:] += signal[:-delay] * gain

    result = signal + wet * wet_level
    max_val = np.max(np.abs(result))
    if max_val > 1.0:
        result = result / max_val * 0.99
    return result


def _measure_lufs(signal: Signal, sample_rate: int = 44100) -> float:
    """Measure integrated loudness in LUFS (simplified ITU-R BS.1770-4).

    Args:
        signal: Input audio signal.
        sample_rate: Sample rate in Hz.

    Returns:
        Approximate integrated loudness in LUFS.
    """
    # K-weighting filter (simplified)
    high_shelf = _apply_high_pass(signal, 1500.0, sample_rate)
    # Mean square with pre-filtering
    ms = np.mean(high_shelf**2)
    if ms < 1e-10:
        return -70.0
    lufs = -0.691 + 10.0 * np.log10(ms)
    return float(lufs)


# ---------------------------------------------------------------------------
# Cliche & Entropy Measurement
# ---------------------------------------------------------------------------


# Common cliche rhyme pairs in rap
_CLICHE_RHYMES: List[Tuple[str, ...]] = [
    ("love", "above", "dove", "glove"),
    ("heart", "start", "apart", "smart", "chart"),
    ("pain", "rain", "gain", "chain", "main", "vain"),
    ("fire", "desire", "higher", "wire", "tire"),
    ("night", "light", "fight", "right", "sight", "might"),
    ("world", "girl", "pearl", "twirl"),
    ("time", "rhyme", "crime", "climb", "dime"),
    ("dream", "scheme", "team", "beam", "seem"),
    ("life", "strife", "knife", "wife", "rife"),
    ("street", "beat", "heat", "feet", "meet"),
    ("money", "honey", "sunny", "funny"),
    ("hustle", "muscle", "tussle"),
]

_COMMON_SLANG = {
    "yo", "fam", "dope", "lit", "fire", "vibes", "squad",
    "hustle", "grind", "flex", "bussin", "no cap", "fr",
    "bout", "finna", "gonna", "tryna", "outta", "kinda",
    "shawty", "opp", "slatt", "brr", "skrrt", "dab",
}


def measure_cliche_rhymes(text: str) -> float:
    """Measure the density of cliche rhyme pairs in text.

    Args:
        text: The text to analyze.

    Returns:
        Cliche rhyme density (0.0-1.0), higher = more cliche.
    """
    lines = [line.strip().lower() for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return 0.0

    cliche_count = 0
    total_rhyme_checks = 0

    for i in range(len(lines)):
        words_i = lines[i].split()
        if not words_i:
            continue
        last_word_i = re.sub(r"[^a-z]", "", words_i[-1])
        for j in range(i + 1, min(i + 4, len(lines))):
            words_j = lines[j].split()
            if not words_j:
                continue
            last_word_j = re.sub(r"[^a-z]", "", words_j[-1])
            total_rhyme_checks += 1
            for rhyme_group in _CLICHE_RHYMES:
                if last_word_i in rhyme_group and last_word_j in rhyme_group:
                    cliche_count += 1

    if total_rhyme_checks == 0:
        return 0.0
    return min(1.0, cliche_count / (total_rhyme_checks * 0.3))


def measure_slang_entropy(text: str) -> float:
    """Measure the slang entropy density of text.

    Args:
        text: The text to analyze.

    Returns:
        Slang entropy score (0.0-1.0), higher = more slang-dense.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0

    slang_count = sum(1 for w in words if w in _COMMON_SLANG)
    # Also check for multi-word slang
    for slang in ("no cap", "for real", "on god", "let's go"):
        slang_count += text.lower().count(slang)

    return min(1.0, slang_count / max(len(words) * 0.1, 1))


# ---------------------------------------------------------------------------
# DAW_SubAgent_Utilities — Main class
# ---------------------------------------------------------------------------


class DAW_SubAgent_Utilities:
    """DAW Sub-Agent Utilities for the GART dual-swarm system.

    Provides sidechain compression, dynamic BPM cadence shifting,
    and master mixing evaluation — the three core signal processing
    stages of the Linguistic DAW pipeline.

    Args:
        verse_stem_1: Protagonist stem (e.g., YBNBA kick pattern).
        verse_stem_2: Antagonist stem (e.g., hive mind synth).
        bpm_range: Tuple of (min_bpm, max_bpm) for the session.
        eq_profile: EQ profile for mixing.
        reverb_config: Reverb configuration.
    """

    def __init__(
        self,
        verse_stem_1: Optional[VerseStem] = None,
        verse_stem_2: Optional[VerseStem] = None,
        bpm_range: Tuple[int, int] = (85, 128),
        eq_profile: Optional[EQProfile] = None,
        reverb_config: Optional[ReverbConfig] = None,
    ) -> None:
        self.ybnba_kick = verse_stem_1 or VerseStem(
            name="ybnba_kick",
            pattern=Pattern(),
            instrument="kick",
            bpm=130,
            intent="protagonist_drive",
            semantic_weight=1.0,
        )
        self.hive_mind_synth = verse_stem_2 or VerseStem(
            name="hive_mind_synth",
            pattern=Pattern(),
            instrument="synth_pad",
            bpm=130,
            intent="antagonist_atmosphere",
            semantic_weight=0.8,
        )
        self.bpm_range = bpm_range
        self.eq_profile = eq_profile or EQProfile()
        self.reverb_config = reverb_config or ReverbConfig()

    def sidechain_compressor_agent(self) -> str:
        """Apply sidechain compression: duck semantic weight when triggered.

        When "Reverse Poor Man's Flex" is detected in the protagonist stem's
        intent, the antagonist stem's semantic weight is reduced to 20%,
        creating dynamic contrast in the mix.

        Returns:
            Ducking confirmation message describing the action taken.
        """
        trigger_phrase = "reverse poor man's flex"
        protagonist_intent = self.ybnba_kick.intent.lower()

        if trigger_phrase in protagonist_intent:
            old_weight = self.hive_mind_synth.semantic_weight
            self.hive_mind_synth.semantic_weight = 0.2
            msg = (
                f"[SIDECAIN] Ducking triggered: '{trigger_phrase}' detected "
                f"in protagonist intent. HiveMind synth weight: "
                f"{old_weight:.2f} -> 0.20"
            )
            logger.info(msg)
            return msg

        return (
            f"[SIDECAIN] No ducking trigger detected. "
            f"Kick intent: '{self.ybnba_kick.intent}'. "
            f"HiveMind weight: {self.hive_mind_synth.semantic_weight:.2f}"
        )

    def dynamic_bpm_cadence_shifter(
        self,
        text_sequence: TextSequence,
    ) -> TextSequence:
        """Shift text cadence by modifying punctuation and injecting markers.

        Replaces commas with cadence-speed markers, injects polysyllabic
        chord markers based on target BPM, and applies style-specific
        rhythmic annotations.

        Args:
            text_sequence: The text sequence to transform.

        Returns:
            Modified TextSequence with shifted cadence.
        """
        text = text_sequence.text
        bpm = text_sequence.bpm

        # Replace commas with cadence speed annotations
        if bpm >= 128:
            speed = "rapid"
            style = "Lil_Baby_Triplets"
            # Replace commas with rapid-fire separators
            text = text.replace(", ", " ||FAST|| ")
            text = text.replace(",", " ||PAUSE|| ")
        elif bpm >= 110:
            speed = "moderate_fast"
            style = "Flow_Switch"
            text = text.replace(", ", " | ")
        elif bpm >= 95:
            speed = "moderate"
            style = "Standard_Cadence"
            text = text.replace(", ", ",\n")
        else:
            speed = "slow"
            style = "Mos_Def_Piano"
            # Slow, deliberate — add weight to each phrase
            text = text.replace(", ", " ... ")
            text = text.replace(".", " .. ")

        # Inject polysyllabic chord markers for piano-heavy styles
        if "Piano" in style or bpm >= 130:
            words = text.split()
            result_words: List[str] = []
            for i, word in enumerate(words):
                result_words.append(word)
                # Inject chord markers after every 4th word
                if (i + 1) % 4 == 0 and i < len(words) - 1:
                    result_words.append(f"[{style}_CHORD]")
            text = " ".join(result_words)

        return TextSequence(
            text=text,
            speed=speed,
            style=style,
            bpm=bpm,
        )

    def master_mixer_evaluator(self, final_cypher: str) -> str:
        """Evaluate and process the final cypher through the master mix pipeline.

        Applies three-stage processing:
            1. EQ Stage: High-pass + Low-pass to remove cliche frequency content
            2. Limiter Stage: Brick-wall limiting if slang entropy exceeds threshold
            3. Reverb Tail: Add spatial depth and crossfade

        Args:
            final_cypher: The complete cypher text to evaluate and process.

        Returns:
            Processed cypher text with mix evaluation report appended.
        """
        if not final_cypher or not final_cypher.strip():
            return "[MIXER] ERROR: Empty cypher received."

        # Stage 1: EQ Cut — measure and filter cliche content
        cliche_threshold = measure_cliche_rhymes(final_cypher)
        eq_report = f"EQ Stage: cliche_threshold={cliche_threshold:.3f}"

        if cliche_threshold > 0.15:
            # Apply "lyrical EQ cut" — rewrite cliche lines
            lines = final_cypher.split("\n")
            processed_lines: List[str] = []
            for line in lines:
                words = re.findall(r"\b\w+\b", line.lower())
                if words:
                    last_word = words[-1]
                    is_cliche = any(
                        last_word in group for group in _CLICHE_RHYMES
                    )
                    if is_cliche:
                        line = line + " [EQ_CUT_REPLACED]"
                processed_lines.append(line)
            final_cypher = "\n".join(processed_lines)
            eq_report += " | EQ CUT APPLIED (cliche > 0.15)"
        else:
            eq_report += " | CLEAN (no EQ cut needed)"

        # Stage 2: Limiter — control slang saturation
        saturation_level = measure_slang_entropy(final_cypher)
        limiter_report = f"Limiter Stage: saturation={saturation_level:.3f}"

        if saturation_level > 0.90:
            # Apply limiter — reduce slang density
            words = final_cypher.split()
            filtered_words = [
                w for w in words
                if re.sub(r"[^a-zA-Z]", "", w).lower() not in _COMMON_SLANG
                or np.random.random() > 0.5
            ]
            final_cypher = " ".join(filtered_words)
            limiter_report += " | LIMITER APPLIED (saturation > 0.90)"
        else:
            limiter_report += " | CLEAN (no limiting needed)"

        # Stage 3: Reverb Tail — add spatial markers
        reverb_report = (
            f"Reverb Stage: decay={self.reverb_config.decay_time}s, "
            f"wet={self.reverb_config.wet_level}"
        )
        final_cypher = self._apply_reverb_tail_and_crossfade(final_cypher)

        # Master mix report
        mix_report = (
            f"\n{'='*60}\n"
            f"[MASTER MIX EVALUATION]\n"
            f"{'='*60}\n"
            f"{eq_report}\n"
            f"{limiter_report}\n"
            f"{reverb_report}\n"
            f"Final quality score: {self._compute_mix_quality(cliche_threshold, saturation_level):.3f}\n"
            f"{'='*60}"
        )

        return final_cypher + mix_report

    def _apply_reverb_tail_and_crossfade(self, cypher: str) -> str:
        """Apply reverb tail and crossfade markers to the cypher.

        Adds spatial depth markers and smooth transitions.

        Args:
            cypher: The cypher text.

        Returns:
            Cypher with reverb tail and crossfade markers.
        """
        lines = cypher.split("\n")
        if len(lines) < 2:
            return cypher + "\n[REVERB_TAIL: decay=2.5s, wet=0.15]"

        result: List[str] = []
        for i, line in enumerate(lines):
            result.append(line)
            # Add crossfade marker between sections
            if i == len(lines) // 2:
                result.append("[CROSSFADE: mid-point blend]")

        result.append("[REVERB_TAIL: decay=2.5s, wet=0.15, fade_out]")
        return "\n".join(result)

    def _compute_mix_quality(self, cliche: float, saturation: float) -> float:
        """Compute overall mix quality score.

        Args:
            cliche: Cliche rhyme density (0.0-1.0).
            saturation: Slang entropy (0.0-1.0).

        Returns:
            Quality score (0.0-1.0), higher = better.
        """
        # Lower cliche and moderate saturation = best quality
        cliche_penalty = max(0.0, (cliche - 0.15) * 2.0)
        saturation_penalty = max(0.0, (saturation - 0.90) * 5.0)
        quality = 1.0 - min(1.0, cliche_penalty + saturation_penalty)
        return max(0.0, min(1.0, quality))

    def apply_eq(self, signal: Signal, eq_profile: Optional[EQProfile] = None) -> Signal:
        """Apply EQ processing to an audio signal.

        Args:
            signal: Input audio signal.
            eq_profile: EQ profile (uses self.eq_profile if None).

        Returns:
            EQ-processed signal.
        """
        profile = eq_profile or self.eq_profile
        result = _apply_high_pass(signal, profile.high_pass_freq)
        result = _apply_low_pass(result, profile.low_pass_freq)
        for freq, gain, q in profile.peaking_filters:
            # Simplified peaking filter via gain adjustment
            pass  # Placeholder for full implementation
        return result

    def apply_limiter(self, signal: Signal, threshold_db: float = -1.0) -> Signal:
        """Apply brick-wall limiter to signal.

        Args:
            signal: Input audio signal.
            threshold_db: Limiter threshold in dB.

        Returns:
            Limited signal.
        """
        return _apply_limiter(signal, threshold_db)

    def apply_reverb_tail(self, signal: Signal, tail_length: float = 2.5) -> Signal:
        """Apply reverb tail to signal.

        Args:
            signal: Input audio signal.
            tail_length: Reverb tail length in seconds.

        Returns:
            Signal with reverb tail.
        """
        return _apply_reverb(signal, tail_length, self.reverb_config.wet_level)

    def generate_ybnba_kick_pattern(self, bar_count: int = 4) -> Pattern:
        """Generate a YBNBA-style kick drum pattern.

        Creates a trap-influenced kick pattern with characteristic
        off-beat placements and rolls.

        Args:
            bar_count: Number of bars to generate.

        Returns:
            Pattern with kick drum notes.
        """
        notes: List[Note] = []
        beats_per_bar = 4
        total_beats = bar_count * beats_per_bar

        # Standard trap kick: beats 1 and 3 with occasional rolls
        for bar in range(bar_count):
            bar_offset = bar * beats_per_bar
            # Downbeat
            notes.append(Note(36, 110, bar_offset, 0.5))
            # Backbeat
            notes.append(Note(36, 100, bar_offset + 2.0, 0.5))
            # Off-beat kick for trap feel (on 3.5)
            if bar % 2 == 1:
                notes.append(Note(36, 90, bar_offset + 3.5, 0.25))

        pattern = Pattern(notes=notes, duration_beats=float(total_beats))
        pattern.humanize(amount=0.02)
        return pattern

    def generate_hive_synth_riff(self, mood: str = "dark") -> Pattern:
        """Generate a hive mind synth riff based on mood.

        Args:
            mood: Mood descriptor ("dark", "melodic", "aggressive", "ambient").

        Returns:
            Pattern with synth notes.
        """
        notes: List[Note] = []

        mood_configs: Dict[str, Dict[str, Any]] = {
            "dark": {"root": 48, "scale": [0, 3, 5, 7, 10], "velocity": 80},
            "melodic": {"root": 60, "scale": [0, 2, 4, 7, 9], "velocity": 90},
            "aggressive": {"root": 42, "scale": [0, 1, 4, 6, 7], "velocity": 110},
            "ambient": {"root": 55, "scale": [0, 4, 7, 11], "velocity": 70},
        }

        config = mood_configs.get(mood, mood_configs["dark"])
        root = config["root"]
        scale = config["scale"]
        velocity = config["velocity"]

        # Generate 8-beat riff
        for i in range(8):
            scale_idx = (i * 2) % len(scale)
            pitch = root + scale[scale_idx]
            start = float(i) * 0.5
            notes.append(Note(pitch, velocity, start, 0.4))

        return Pattern(notes=notes, duration_beats=4.0)

    def measure_semantic_weight(self, text: str) -> float:
        """Measure the semantic weight of a text passage.

        Semantic weight is a composite of vocabulary density,
        metaphor presence, and emotional intensity.

        Args:
            text: The text to analyze.

        Returns:
            Semantic weight score (0.0-1.0).
        """
        if not text.strip():
            return 0.0

        words = text.split()
        unique_words = set(w.lower() for w in words)

        # Vocabulary density
        vocab_density = len(unique_words) / max(len(words), 1)

        # Emotional intensity (capitalization)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        # Punctuation intensity
        punct_ratio = sum(1 for c in text if c in "!?.;:") / max(len(text), 1)

        weight = (vocab_density * 0.4 + caps_ratio * 0.3 + punct_ratio * 0.3)
        return min(1.0, weight)

    def duck_frequency_range(
        self,
        signal: Signal,
        freq_range: Tuple[float, float],
    ) -> Signal:
        """Duck (reduce) a specific frequency range in the signal.

        Args:
            signal: Input audio signal.
            freq_range: Tuple of (low_freq, high_freq) in Hz.

        Returns:
            Signal with reduced energy in the specified range.
        """
        low_freq, high_freq = freq_range
        # High-pass above high_freq + low-pass below low_freq = notch
        hp = _apply_high_pass(signal, high_freq)
        lp = _apply_low_pass(signal, low_freq)
        # Mix: keep outside the notch, reduce inside
        notch_reduction = 0.2
        result = hp + lp + signal * notch_reduction
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val * 0.99
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("DAW Sub-Agent Utilities module loaded successfully.")

    # Create sample stems
    kick_pattern = Pattern(
        notes=[Note(36, 100, 0.0, 0.5), Note(36, 110, 2.0, 0.5)],
        duration_beats=4.0,
    )
    synth_pattern = Pattern(
        notes=[Note(60, 80, 0.0, 2.0), Note(64, 75, 2.0, 2.0)],
        duration_beats=4.0,
    )

    kick_stem = VerseStem(
        name="ybnba_kick", pattern=kick_pattern,
        instrument="kick", bpm=130, intent="protagonist_drive",
    )
    synth_stem = VerseStem(
        name="hive_mind_synth", pattern=synth_pattern,
        instrument="synth", bpm=130, intent="antagonist_atmosphere",
    )

    daw = DAW_SubAgent_Utilities(kick_stem, synth_stem)

    # Test sidechain
    print("\n--- Sidechain Test ---")
    print(daw.sidechain_compressor_agent())

    # Test BPM shifter
    print("\n--- BPM Shifter Test ---")
    ts = TextSequence("Yeah, I'm on the grind, every day I shine", bpm=140)
    shifted = daw.dynamic_bpm_cadence_shifter(ts)
    print(f"Original: {ts.text}")
    print(f"Shifted:  {shifted.text}")
    print(f"Speed: {shifted.speed}, Style: {shifted.style}")

    # Test master mixer
    print("\n--- Master Mixer Test ---")
    test_cypher = "I'm climbing up, never giving up\nIn the rain, feeling all the pain\nFire in my heart, never apart"
    result = daw.master_mixer_evaluator(test_cypher)
    print(result)
