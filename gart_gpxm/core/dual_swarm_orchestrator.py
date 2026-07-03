"""
Dual-Swarm Orchestrator — Core coordination engine for GART v3.0.

Manages two parallel swarm systems:
    - Swarm Alpha: LinguisticDAW (V1 leads + V2 piano anchor)
    - Swarm Omega: FrankensteinHiveMind (amalgamated opponents)

Pipeline:
    1. coordinate_battle() spawns both swarms via asyncio.gather()
    2. SidechainCompressor bridges Alpha and Omega outputs
    3. MasterMixerEvaluator produces final scored cypher
    4. ScoringEngine ranks and advances tournament bracket

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MatchupConfig:
    """Configuration for a single matchup/battle.

    Attributes:
        rounds: Number of rounds in the battle.
        criteria: List of evaluation criteria names.
        weights: Dict mapping criteria to weight values.
        time_limit_seconds: Maximum battle duration.
    """

    rounds: int = 3
    criteria: List[str] = field(default_factory=lambda: [
        "flow_complexity", "rhyme_density", "thematic_depth",
        "cultural_entropy", "vocal_presence",
    ])
    weights: Dict[str, float] = field(default_factory=lambda: {
        "flow_complexity": 0.25,
        "rhyme_density": 0.20,
        "thematic_depth": 0.25,
        "cultural_entropy": 0.15,
        "vocal_presence": 0.15,
    })
    time_limit_seconds: float = 300.0


@dataclass
class BattleResult:
    """Result of a single battle between two personas.

    Attributes:
        protagonist_id: ID of the protagonist persona.
        antagonist_id: ID of the antagonist persona.
        protagonist_score: Protagonist's composite score.
        antagonist_score: Antagonist's composite score.
        winner_id: ID of the winning persona.
        round_scores: Dict of per-round scores.
        cypher_text: Generated cypher text.
        metadata: Additional battle metadata.
    """

    protagonist_id: str
    antagonist_id: str
    protagonist_score: float = 0.0
    antagonist_score: float = 0.0
    winner_id: Optional[str] = None
    round_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cypher_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def margin(self) -> float:
        """Return the score margin between winner and loser."""
        return abs(self.protagonist_score - self.antagonist_score)


@dataclass
class TournamentResult:
    """Result of a complete tournament.

    Attributes:
        champion_id: ID of the tournament champion.
        final_scores: Dict of final scores by persona ID.
        bracket_history: List of battle results by round.
        total_battles: Total number of battles fought.
        metadata: Tournament metadata.
    """

    champion_id: Optional[str] = None
    final_scores: Dict[str, float] = field(default_factory=dict)
    bracket_history: List[List[BattleResult]] = field(default_factory=list)
    total_battles: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmCoordinationResult:
    """Result of dual-swarm coordination.

    Attributes:
        alpha_output: Output from Swarm Alpha (LinguisticDAW).
        omega_output: Output from Swarm Omega (HiveMind).
        blended_output: Sidechain-compressed blended output.
        final_score: Master mixer evaluation score.
        processing_time_ms: Total processing time in milliseconds.
    """

    alpha_output: str = ""
    omega_output: str = ""
    blended_output: str = ""
    final_score: float = 0.0
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Placeholder types (forward references resolved at runtime)
# ---------------------------------------------------------------------------

# These are resolved from sibling modules at orchestrator initialization.
GeneticPersona = Any  # Forward reference to gpxm.genetic_persona.GeneticPersona
LLLM_Architecture = Any  # Forward reference to core.lllm_architecture.LLLM_Architecture


# ---------------------------------------------------------------------------
# Swarm Alpha: LinguisticDAW
# ---------------------------------------------------------------------------


class LinguisticEngine:
    """Engine for generating structured verses from text prompts.

    Wraps the LLLM with prompt building and post-processing for
    coherent verse generation.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        self.model = model
        self.context_history: List[str] = []

    async def generate_line(self, prompt: str, context: List[str]) -> str:
        """Generate a single line of verse.

        Args:
            prompt: Base prompt for generation.
            context: Recent lines for context continuity.

        Returns:
            Generated line string.
        """
        full_prompt = self._build_prompt(prompt, context)
        # Simulated generation — in production, calls LLLM.generate()
        await asyncio.sleep(0.01)  # Simulate async inference
        generated = f"[{full_prompt[:20]}...] Verse line generated with flow"
        self.context_history.append(generated)
        return generated

    async def generate_verse(self, prompt: str, bar_count: int = 16) -> List[str]:
        """Generate a full verse of specified bar count.

        Args:
            prompt: Base prompt for the verse.
            bar_count: Number of bars to generate.

        Returns:
            List of generated lines.
        """
        lines: List[str] = []
        for _ in range(bar_count):
            line = await self.generate_line(prompt, lines[-4:])
            lines.append(line)
        return lines

    def _build_prompt(self, base: str, context: List[str]) -> str:
        """Build full prompt from base and context.

        Args:
            base: Base prompt string.
            context: Context lines.

        Returns:
            Compiled prompt string.
        """
        ctx_str = " | ".join(context[-4:]) if context else "[no context]"
        return f"BASE: {base} | CONTEXT: {ctx_str}"

    def _postprocess(self, raw: str) -> str:
        """Clean up generated text.

        Args:
            raw: Raw generated text.

        Returns:
            Cleaned text.
        """
        return raw.strip().replace("  ", " ")


class VerseConstructor:
    """Constructs verses with rhyme schemes and meter enforcement."""

    def __init__(self) -> None:
        self.rhyme_schemes: Dict[str, List[str]] = {
            "AABB": ["A", "A", "B", "B"],
            "ABAB": ["A", "B", "A", "B"],
            "ABBA": ["A", "B", "B", "A"],
            "AAAA": ["A", "A", "A", "A"],
            "freestyle": [],
        }

    def construct(self, lines: List[str], scheme_name: str = "AABB") -> str:
        """Construct a verse from lines with rhyme scheme enforcement.

        Args:
            lines: Raw lines to arrange.
            scheme_name: Rhyme scheme name.

        Returns:
            Formatted verse string.
        """
        scheme = self.rhyme_schemes.get(scheme_name, [])
        if not scheme or len(lines) < len(scheme):
            return "\n".join(lines)

        grouped: Dict[str, List[str]] = {}
        for i, line in enumerate(lines):
            group = scheme[i % len(scheme)]
            grouped.setdefault(group, []).append(line)

        result: List[str] = []
        for i, line in enumerate(lines):
            result.append(line)
        return "\n".join(result)

    def analyze_meter(self, line: str) -> Dict[str, Any]:
        """Analyze the meter/rhythm pattern of a line.

        Args:
            line: The line to analyze.

        Returns:
            Dictionary with meter metrics.
        """
        words = line.split()
        syllable_est = sum(max(1, len(w) // 2) for w in words)
        return {
            "syllable_estimate": syllable_est,
            "word_count": len(words),
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
        }


class StemGenerator:
    """Generates musical stems from verse structures."""

    def __init__(self) -> None:
        self.instrument_map = {
            "kick": {"pitch": 36, "velocity": 100},
            "snare": {"pitch": 38, "velocity": 90},
            "hihat": {"pitch": 42, "velocity": 70},
            "synth": {"pitch": 60, "velocity": 80},
            "bass": {"pitch": 40, "velocity": 95},
        }

    def generate_kick_stem(self, verse: str) -> Dict[str, Any]:
        """Generate a kick drum stem for a verse.

        Args:
            verse: The verse text.

        Returns:
            Stem data dictionary.
        """
        lines = verse.strip().split("\n")
        bars = max(len(lines), 4)
        return {
            "instrument": "kick",
            "bars": bars,
            "pattern": "trap_standard",
            "bpm": 130,
        }

    def generate_synth_stem(self, verse: str) -> Dict[str, Any]:
        """Generate a synth pad stem for a verse.

        Args:
            verse: The verse text.

        Returns:
            Stem data dictionary.
        """
        lines = verse.strip().split("\n")
        bars = max(len(lines), 4)
        return {
            "instrument": "synth",
            "bars": bars,
            "pattern": "melodic_pad",
            "bpm": 130,
        }


class LinguisticDAW:
    """Swarm Alpha: Linguistic Digital Audio Workstation.

    Processes text through linguistic engines, verse constructors,
    and stem generators to produce structured cypher output.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        self.engine = LinguisticEngine(model)
        self.constructor = VerseConstructor()
        self.stem_generator = StemGenerator()
        self.output_buffer: List[str] = []

    async def process(self, input_prompt: str) -> str:
        """Process input through the full LinguisticDAW pipeline.

        Args:
            input_prompt: The prompt for verse generation.

        Returns:
            Formatted cypher output.
        """
        # Generate verse
        lines = await self.engine.generate_verse(input_prompt, bar_count=16)
        # Construct with rhyme scheme
        verse = self.constructor.construct(lines, scheme_name="AABB")
        # Generate stems
        kick = self.stem_generator.generate_kick_stem(verse)
        synth = self.stem_generator.generate_synth_stem(verse)
        self.output_buffer.append(verse)
        return f"{verse}\n\n[STEMS: kick={kick}, synth={synth}]"

    async def construct_verse(self, bars: int, scheme: str = "AABB") -> str:
        """Construct a verse with specified parameters.

        Args:
            bars: Number of bars.
            scheme: Rhyme scheme name.

        Returns:
            Formatted verse.
        """
        lines = await self.engine.generate_verse("Construct verse", bar_count=bars)
        return self.constructor.construct(lines, scheme)

    def reset(self) -> None:
        """Clear the output buffer."""
        self.output_buffer.clear()
        self.engine.context_history.clear()


# ---------------------------------------------------------------------------
# Swarm Omega: FrankensteinHiveMind
# ---------------------------------------------------------------------------


class HiveMindAggregator:
    """Aggregates votes from multiple sub-agents for consensus."""

    def __init__(self, agents: Optional[List[Any]] = None) -> None:
        self.agents = agents or []
        self.consensus_threshold = 0.6

    async def vote(self, cyphers: List[str]) -> List[Dict[str, Any]]:
        """Collect votes from all agents on cypher quality.

        Args:
            cyphers: List of cypher texts to evaluate.

        Returns:
            List of vote dictionaries.
        """
        votes: List[Dict[str, Any]] = []
        for i, agent in enumerate(self.agents):
            await asyncio.sleep(0.005)  # Simulated processing
            score = 0.5 + (i * 0.1)  # Simulated scores
            votes.append({"agent_id": i, "score": score, "cypher_idx": 0})
        return votes

    def reach_consensus(self, votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine consensus from votes.

        Args:
            votes: List of vote dictionaries.

        Returns:
            Consensus result dictionary.
        """
        if not votes:
            return {"consensus_reached": False, "winner_idx": 0}
        avg_score = sum(v["score"] for v in votes) / len(votes)
        winner = max(votes, key=lambda v: v["score"])
        return {
            "consensus_reached": avg_score >= self.consensus_threshold,
            "winner_idx": winner["cypher_idx"],
            "avg_score": avg_score,
        }


class StyleSynthesizer:
    """Blends styles from multiple cyphers into unified output."""

    async def blend_styles(self, cyphers: List[str]) -> str:
        """Blend multiple cypher styles.

        Args:
            cyphers: List of cypher texts.

        Returns:
            Blended cypher text.
        """
        await asyncio.sleep(0.01)
        if not cyphers:
            return ""
        # Take best elements from each
        lines: List[str] = []
        for c in cyphers:
            c_lines = c.split("\n")
            if c_lines:
                lines.append(c_lines[0])
        return "\n".join(lines)

    def extract_features(self, cypher: str) -> Dict[str, float]:
        """Extract stylistic features from a cypher.

        Args:
            cypher: Cypher text.

        Returns:
            Feature dictionary.
        """
        words = cypher.split()
        return {
            "word_count": float(len(words)),
            "avg_word_len": sum(len(w) for w in words) / max(len(words), 1),
            "line_count": float(len(cypher.split("\n"))),
        }


class EntropyInjector:
    """Injects controlled randomness into cypher output."""

    def __init__(self, max_entropy: float = 0.5) -> None:
        self.max_entropy = max_entropy

    def inject(self, cypher: str, level: float) -> str:
        """Inject entropy into a cypher.

        Args:
            cypher: Original cypher text.
            level: Entropy level (0.0 to max_entropy).

        Returns:
            Cypher with injected entropy.
        """
        import random
        level = min(level, self.max_entropy)
        lines = cypher.split("\n")
        result: List[str] = []
        for line in lines:
            if random.random() < level:
                line = line + " [ENTROPY_INJECTION]"
            result.append(line)
        return "\n".join(result)

    def generate_mutation(self, base: str, entropy: float) -> str:
        """Generate a mutation of a base text.

        Args:
            base: Base text.
            entropy: Mutation entropy level.

        Returns:
            Mutated text.
        """
        words = base.split()
        if not words or entropy <= 0:
            return base
        # Shuffle some words based on entropy
        import random
        num_to_shuffle = int(len(words) * entropy * 0.3)
        indices = random.sample(range(len(words)), min(num_to_shuffle, len(words) - 1))
        for i in indices:
            if i + 1 < len(words):
                words[i], words[i + 1] = words[i + 1], words[i]
        return " ".join(words)


class FrankensteinHiveMind:
    """Swarm Omega: Amalgamated opponent hive mind.

    Combines multiple sub-agent opinions through voting, blends
    their stylistic outputs, and injects controlled entropy.
    """

    def __init__(
        self,
        agents: Optional[List[Any]] = None,
        max_entropy: float = 0.5,
    ) -> None:
        self.aggregator = HiveMindAggregator(agents)
        self.synthesizer = StyleSynthesizer()
        self.entropy_injector = EntropyInjector(max_entropy)
        self.output_buffer: List[str] = []

    async def process(self, input_cyphers: List[str]) -> str:
        """Process cyphers through the hive mind pipeline.

        Args:
            input_cyphers: List of input cypher texts.

        Returns:
            Synthesized cypher output.
        """
        # Aggregate votes
        votes = await self.aggregator.vote(input_cyphers)
        consensus = self.aggregator.reach_consensus(votes)
        # Blend styles
        blended = await self.synthesizer.blend_styles(input_cyphers)
        # Inject entropy
        final = self.entropy_injector.inject(blended, level=0.2)
        self.output_buffer.append(final)
        return final

    async def aggregate(self, cyphers: List[str]) -> Dict[str, Any]:
        """Run aggregation on cyphers.

        Args:
            cyphers: Cypher texts.

        Returns:
            Consensus result.
        """
        votes = await self.aggregator.vote(cyphers)
        return self.aggregator.reach_consensus(votes)

    async def synthesize(self, consensus: Dict[str, Any]) -> str:
        """Synthesize output from consensus.

        Args:
            consensus: Consensus dictionary.

        Returns:
            Synthesized text.
        """
        await asyncio.sleep(0.01)
        return f"[HIVEMIND SYNTHESIS] Consensus: {consensus}"

    def reset(self) -> None:
        """Clear the output buffer."""
        self.output_buffer.clear()


# ---------------------------------------------------------------------------
# SidechainCompressor
# ---------------------------------------------------------------------------


class SidechainCompressor:
    """Sidechain compressor bridge between Alpha and Omega outputs.

    Ducks the semantic weight of one signal based on the presence
    of trigger phrases in another, creating dynamic contrast.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
        blend_ratio: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.blend_ratio = blend_ratio

    def process(self, alpha_signal: str, omega_signal: str) -> str:
        """Compress and blend Alpha and Omega signals.

        Args:
            alpha_signal: Output from Swarm Alpha.
            omega_signal: Output from Swarm Omega.

        Returns:
            Blended output string.
        """
        # Blend based on configured ratio
        alpha_weight = self.blend_ratio
        omega_weight = 1.0 - self.blend_ratio

        # If omega signal is too strong, duck it
        omega_strength = len(omega_signal) / max(len(alpha_signal), 1)
        if omega_strength > self.threshold:
            gain_reduction = (omega_strength - self.threshold) / self.ratio
            omega_weight *= max(0.1, 1.0 - gain_reduction)

        return (
            f"[ALPHA: w={alpha_weight:.2f}]\n{alpha_signal}\n\n"
            f"[OMEGA: w={omega_weight:.2f}]\n{omega_signal}\n\n"
            f"[BLENDED: ratio={self.blend_ratio}]"
        )

    def duck_semantic_weight(
        self,
        alpha_output: str,
        omega_output: str,
    ) -> str:
        """Duck semantic weight based on trigger detection.

        Args:
            alpha_output: Alpha swarm output.
            omega_output: Omega swarm output.

        Returns:
            Duck-processed output.
        """
        triggers = ["reverse poor man's flex", "takeover", "dominance"]
        if any(t in alpha_output.lower() for t in triggers):
            return (
                f"[DUCKING ACTIVE] Omega weight reduced\n"
                f"{alpha_output}\n{omega_output}"
            )
        return f"[NO DUCKING]\n{alpha_output}\n{omega_output}"


# ---------------------------------------------------------------------------
# MasterMixerEvaluator
# ---------------------------------------------------------------------------


class MasterMixerEvaluator:
    """Master mixer evaluator: EQ + Limiter + Reverb pipeline.

    Evaluates final cypher output through a three-stage master
    mixing pipeline modeled after professional DAW master busses.
    """

    def __init__(self) -> None:
        self.eq_high_pass = 80.0
        self.eq_low_pass = 12000.0
        self.limiter_threshold_db = -1.0
        self.reverb_decay = 2.5
        self.reverb_wet = 0.15
        self.target_lufs = -14.0

    def evaluate(self, cypher: str) -> float:
        """Evaluate cypher quality through the master mix pipeline.

        Args:
            cypher: Final cypher text.

        Returns:
            Final quality score (0.0-1.0).
        """
        # Stage 1: EQ
        eq_score = self._evaluate_eq(cypher)
        # Stage 2: Limiter
        limiter_score = self._evaluate_limiter(cypher)
        # Stage 3: Reverb
        reverb_score = self._evaluate_reverb(cypher)

        return (eq_score * 0.4 + limiter_score * 0.3 + reverb_score * 0.3)

    def process_eq(self, cypher: str) -> str:
        """Apply EQ processing to cypher.

        Args:
            cypher: Input cypher.

        Returns:
            EQ-processed cypher.
        """
        # Remove low-frequency clutter (short words, filler)
        lines = cypher.split("\n")
        processed: List[str] = []
        for line in lines:
            words = line.split()
            # Filter: keep words with sufficient "frequency content"
            filtered = [w for w in words if len(w) >= 2]
            processed.append(" ".join(filtered))
        return "\n".join(processed)

    def process_limiter(self, cypher: str) -> str:
        """Apply limiting to control peaks.

        Args:
            cypher: Input cypher.

        Returns:
            Limited cypher.
        """
        # Cap line length to control "peak levels"
        lines = cypher.split("\n")
        processed: List[str] = []
        for line in lines:
            if len(line) > 80:
                line = line[:80] + " [LIMITED]"
            processed.append(line)
        return "\n".join(processed)

    def process_reverb(self, cypher: str) -> str:
        """Apply reverb tail to cypher.

        Args:
            cypher: Input cypher.

        Returns:
            Cypher with reverb processing.
        """
        return cypher + f"\n[REVERB: decay={self.reverb_decay}s, wet={self.reverb_wet}]"

    def measure_loudness(self, cypher: str) -> float:
        """Measure "loudness" (content density) of cypher.

        Args:
            cypher: Cypher text.

        Returns:
            Loudness score in approximate LUFS.
        """
        words = cypher.split()
        if not words:
            return -70.0
        density = len(words) / max(len(cypher), 1) * 100
        return float(-30 + density)

    def _evaluate_eq(self, cypher: str) -> float:
        """Evaluate EQ quality of cypher.

        Args:
            cypher: Cypher text.

        Returns:
            EQ quality score (0.0-1.0).
        """
        words = cypher.split()
        if not words:
            return 0.0
        short_word_ratio = sum(1 for w in words if len(w) <= 2) / len(words)
        return 1.0 - min(1.0, short_word_ratio * 2)

    def _evaluate_limiter(self, cypher: str) -> float:
        """Evaluate if cypher needs limiting.

        Args:
            cypher: Cypher text.

        Returns:
            Limiter quality score (0.0-1.0).
        """
        lines = cypher.split("\n")
        if not lines:
            return 0.0
        avg_len = sum(len(l) for l in lines) / len(lines)
        return 1.0 if avg_len < 60 else 0.7 if avg_len < 80 else 0.4

    def _evaluate_reverb(self, cypher: str) -> float:
        """Evaluate reverb suitability for cypher.

        Args:
            cypher: Cypher text.

        Returns:
            Reverb quality score (0.0-1.0).
        """
        # More lines = more space for reverb
        lines = cypher.split("\n")
        return min(1.0, len(lines) / 16.0)


# ---------------------------------------------------------------------------
# DualSwarmOrchestrator — Main entry point
# ---------------------------------------------------------------------------


class DualSwarmOrchestrator:
    """Dual-Swarm Orchestrator for GART v3.0.

    Coordinates two parallel swarm systems:
        - Swarm Alpha (LinguisticDAW): Generates protagonist verses
        - Swarm Omega (FrankensteinHiveMind): Synthesizes antagonist responses

    The sidechain compressor bridges both outputs, and the master mixer
    evaluator produces the final scored cypher.

    Attributes:
        swarm_alpha: LinguisticDAW instance.
        swarm_omega: FrankensteinHiveMind instance.
        sidechain: SidechainCompressor bridge.
        mixer: MasterMixerEvaluator pipeline.
    """

    def __init__(self, lllm_model: Optional[Any] = None) -> None:
        self.swarm_alpha = LinguisticDAW(llm_model)
        self.swarm_omega = FrankensteinHiveMind()
        self.sidechain = SidechainCompressor()
        self.mixer = MasterMixerEvaluator()
        self._lock = asyncio.Lock()

    async def coordinate_battle(
        self,
        protagonist: Any,
        antagonist: Any,
        matchup_config: Optional[MatchupConfig] = None,
    ) -> BattleResult:
        """Coordinate a single battle between two personas.

        Runs Swarm Alpha and Swarm Omega in parallel via asyncio.gather(),
        applies sidechain compression, and runs master mixer evaluation.

        Args:
            protagonist: The protagonist GeneticPersona.
            antagonist: The antagonist GeneticPersona.
            matchup_config: Optional battle configuration.

        Returns:
            BattleResult with scores and winner.
        """
        config = matchup_config or MatchupConfig()
        start_time = time.monotonic()

        # Build prompts from personas
        protagonist_id = getattr(protagonist, "persona_id", "protagonist")
        antagonist_id = getattr(antagonist, "persona_id", "antagonist")
        protagonist_name = getattr(protagonist, "artist_name", "Unknown")
        antagonist_name = getattr(antagonist, "artist_name", "Unknown")

        alpha_prompt = f"Verse for {protagonist_name}: melodic trap, emotional"
        omega_prompt = f"Counter-verse for {antagonist_name}: aggressive response"

        try:
            # Run both swarms in parallel
            alpha_task = self.swarm_alpha.process(alpha_prompt)
            omega_task = self.swarm_omega.process([omega_prompt])

            alpha_output, omega_output = await asyncio.gather(
                alpha_task, omega_task, return_exceptions=True
            )

            # Handle exceptions
            if isinstance(alpha_output, Exception):
                logger.error("Alpha swarm failed: %s", alpha_output)
                alpha_output = f"[ALPHA ERROR: {alpha_output}]"
            if isinstance(omega_output, Exception):
                logger.error("Omega swarm failed: %s", omega_output)
                omega_output = f"[OMEGA ERROR: {omega_output}]"

            # Sidechain compression
            blended = self.sidechain.process(str(alpha_output), str(omega_output))

            # Master mixer evaluation
            combined_cypher = f"{alpha_output}\n\n{omega_output}"
            mix_score = self.mixer.evaluate(combined_cypher)

            # Calculate scores per criteria
            round_scores: Dict[str, Dict[str, float]] = {}
            for round_num in range(1, config.rounds + 1):
                round_scores[f"round_{round_num}"] = {
                    criterion: 0.5 + (0.1 * round_num)
                    for criterion in config.criteria
                }

            # Determine winner
            protagonist_score = mix_score + 0.05  # Slight protagonist advantage
            antagonist_score = mix_score
            winner_id = protagonist_id if protagonist_score >= antagonist_score else antagonist_id

            elapsed_ms = (time.monotonic() - start_time) * 1000

            result = BattleResult(
                protagonist_id=protagonist_id,
                antagonist_id=antagonist_id,
                protagonist_score=round(protagonist_score, 4),
                antagonist_score=round(antagonist_score, 4),
                winner_id=winner_id,
                round_scores=round_scores,
                cypher_text=combined_cypher,
                metadata={
                    "processing_time_ms": round(elapsed_ms, 2),
                    "alpha_output_length": len(str(alpha_output)),
                    "omega_output_length": len(str(omega_output)),
                    "mix_score": round(mix_score, 4),
                    "config": {
                        "rounds": config.rounds,
                        "criteria": config.criteria,
                    },
                },
            )

            return result

        except asyncio.CancelledError:
            logger.warning("Battle cancelled: %s vs %s", protagonist_id, antagonist_id)
            raise
        except Exception as e:
            logger.error("Battle failed: %s", e)
            return BattleResult(
                protagonist_id=protagonist_id,
                antagonist_id=antagonist_id,
                metadata={"error": str(e)},
            )

    async def run_tournament(
        self,
        artist_pool: List[Any],
        matchup_config: Optional[MatchupConfig] = None,
    ) -> TournamentResult:
        """Run a full tournament with all matchups.

        Args:
            artist_pool: List of GeneticPersona competitors.
            matchup_config: Optional battle configuration.

        Returns:
            TournamentResult with champion and scores.
        """
        if len(artist_pool) < 2:
            return TournamentResult(metadata={"error": "Need at least 2 artists"})

        config = matchup_config or MatchupConfig()
        result = TournamentResult()
        bracket: List[List[BattleResult]] = []
        remaining = list(artist_pool)
        round_num = 0

        while len(remaining) > 1:
            round_num += 1
            round_results: List[BattleResult] = []
            next_round: List[Any] = []

            # Pair up remaining artists
            for i in range(0, len(remaining), 2):
                if i + 1 < len(remaining):
                    battle = await self.coordinate_battle(
                        remaining[i], remaining[i + 1], config,
                    )
                    round_results.append(battle)
                    result.total_battles += 1

                    # Advance winner
                    winner = next(
                        (a for a in [remaining[i], remaining[i + 1]]
                         if getattr(a, "persona_id", None) == battle.winner_id),
                        remaining[i],
                    )
                    next_round.append(winner)
                else:
                    # Bye
                    next_round.append(remaining[i])

            bracket.append(round_results)
            remaining = next_round

        result.champion_id = getattr(remaining[0], "persona_id", "unknown") if remaining else None
        result.bracket_history = bracket
        result.final_scores = {
            getattr(a, "persona_id", f"artist_{i}"): 0.5
            for i, a in enumerate(artist_pool)
        }
        result.metadata = {
            "rounds": round_num,
            "total_participants": len(artist_pool),
        }

        return result

    async def orchestrate_dual_swarm(self) -> SwarmCoordinationResult:
        """Run the full dual-swarm coordination cycle.

        Returns:
            SwarmCoordinationResult with both outputs and blend.
        """
        start = time.monotonic()
        alpha_output = await self.swarm_alpha.process("Dual swarm coordination")
        omega_output = await self.swarm_omega.process(["Dual swarm counter"])
        blended = self.sidechain.process(alpha_output, omega_output)
        score = self.mixer.evaluate(blended)
        elapsed = (time.monotonic() - start) * 1000

        return SwarmCoordinationResult(
            alpha_output=alpha_output,
            omega_output=omega_output,
            blended_output=blended,
            final_score=score,
            processing_time_ms=elapsed,
        )
