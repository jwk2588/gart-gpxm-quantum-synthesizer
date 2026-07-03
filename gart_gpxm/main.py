"""
GART v3.0 + GPXM Quantum Linguistic Synthesizer — Main Entry Point.

Provides CLI interface for running the dual-swarm AI music production system.

Usage:
    python -m gart_gpxm.main --mode tournament --personas 16
    python -m gart_gpxm.main --mode battle --protagonist "Lil Durk" --antagonist "Tay B"
    python -m gart_gpxm.main --mode reconstruct --track "Stash Box"
    python -m gart_gpxm.main --mode invert --diagram docs/diagram_system_architecture.mmd

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GART v3.0 + GPXM Quantum Linguistic Synthesizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode tournament --personas 16
  %(prog)s --mode battle --protagonist "Lil Durk" --antagonist "Tay B"
  %(prog)s --mode reconstruct --track "Stash Box"
  %(prog)s --mode invert --diagram docs/diagram_system_architecture.mmd
  %(prog)s --mode roster --list
        """.strip(),
    )

    parser.add_argument(
        "--mode",
        choices=["tournament", "battle", "reconstruct", "invert", "roster", "resilience", "demo"],
        default="demo",
        help="Operation mode (default: demo)",
    )
    parser.add_argument("--personas", type=int, default=16, help="Number of personas (default: 16)")
    parser.add_argument("--protagonist", type=str, default="Lil Durk", help="Protagonist artist name")
    parser.add_argument("--antagonist", type=str, default="Tay B", help="Antagonist artist name")
    parser.add_argument("--track", type=str, default="Stash Box", help="Track to reconstruct")
    parser.add_argument("--diagram", type=str, help="Mermaid diagram file for inversion")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--list", action="store_true", help="List roster")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args()


async def run_tournament_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """Run tournament mode."""
    logger.info("Starting tournament mode with %d personas", args.personas)

    from gart_gpxm.core.dual_swarm_orchestrator import DualSwarmOrchestrator
    from gart_gpxm.gpxm.genetic_persona import GeneticPersona, VoiceDifferentiationParameters

    orchestrator = DualSwarmOrchestrator()

    # Create mock personas for demonstration
    personas = _create_demo_personas(args.personas)
    logger.info("Created %d demo personas", len(personas))

    result = await orchestrator.run_tournament(personas)

    logger.info("Tournament complete! Champion: %s", result.champion_id)
    logger.info("Total battles: %d", result.total_battles)

    return {
        "mode": "tournament",
        "champion": result.champion_id,
        "total_battles": result.total_battles,
        "rounds": len(result.bracket_history),
    }


async def run_battle_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """Run single battle mode."""
    logger.info("Battle: %s vs %s", args.protagonist, args.antagonist)

    from gart_gpxm.core.dual_swarm_orchestrator import (
        DualSwarmOrchestrator,
        MatchupConfig,
    )
    from gart_gpxm.gpxm.genetic_persona import GeneticPersona, VoiceDifferentiationParameters

    orchestrator = DualSwarmOrchestrator()

    protagonist = GeneticPersona(
        artist_name=args.protagonist,
        voice_params=VoiceDifferentiationParameters(
            vocabulary_tier="Street",
            slang_density="High",
            emotional_range="Extreme",
        ),
        entropy_level=0.45,
        collaboration_affinity=0.7,
    )
    antagonist = GeneticPersona(
        artist_name=args.antagonist,
        voice_params=VoiceDifferentiationParameters(
            vocabulary_tier="Mixed",
            slang_density="Medium",
            emotional_range="Wide",
        ),
        entropy_level=0.5,
        collaboration_affinity=0.6,
    )

    config = MatchupConfig(rounds=3)
    result = await orchestrator.coordinate_battle(protagonist, antagonist, config)

    logger.info("Winner: %s", result.winner_id)
    logger.info("Protagonist score: %.4f", result.protagonist_score)
    logger.info("Antagonist score: %.4f", result.antagonist_score)

    return {
        "mode": "battle",
        "protagonist": args.protagonist,
        "antagonist": args.antagonist,
        "winner": result.winner_id,
        "protagonist_score": result.protagonist_score,
        "antagonist_score": result.antagonist_score,
        "margin": result.margin,
    }


async def run_reconstruct_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """Run track reconstruction mode."""
    logger.info("Reconstructing track: %s", args.track)

    from gart_gpxm.stash_box.reconstruction_engine import (
        StashBoxReconstructionEngine,
        StyleBrief,
    )

    engine = StashBoxReconstructionEngine()
    brief = StyleBrief(
        primary_artist="Tay B",
        featured_artist="Lil Durk",
        target_bpm=140,
        target_key="C minor",
        mood="melodic_dark",
    )

    concept = engine.reconstruct_track(brief)
    report = engine.generate_track_report(concept)

    print(f"\n{report}")

    return {
        "mode": "reconstruct",
        "track": concept.title,
        "primary_artist": concept.primary_artist,
        "featured_artist": concept.featured_artist,
        "bpm": concept.bpm,
        "sections": len(concept.sections),
        "total_bars": sum(s.bar_count for s in concept.sections),
    }


async def run_invert_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """Run diagram-to-code inversion mode."""
    if not args.diagram:
        logger.error("--diagram required for invert mode")
        return {"error": "--diagram required"}

    logger.info("Inverting diagram: %s", args.diagram)

    from gart_gpxm.inversion.diagram_to_code_agent import DiagramToCodeInversionAgent

    agent = DiagramToCodeInversionAgent(mode="single")

    with open(args.diagram, "r") as f:
        mermaid_source = f.read()

    classes = agent.generate_classes(mermaid_source)
    python_code = agent.emit_python(agent.parse_mermaid(mermaid_source))
    validation = agent.validate_output(python_code)

    logger.info("Generated %d classes", len(classes))
    logger.info("Validation: valid=%s, classes=%d, lines=%d",
                validation["valid"], validation["class_count"], validation["line_count"])

    return {
        "mode": "invert",
        "diagram": args.diagram,
        "classes_generated": len(classes),
        "lines": validation["line_count"],
        "valid": validation["valid"],
    }


async def run_roster_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """Run roster management mode."""
    from gart_gpxm.gpxm.roster_manager import RosterManager
    from gart_gpxm.gpxm.genetic_persona import GeneticPersona, VoiceDifferentiationParameters

    roster = RosterManager()

    # Add demo artists
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

    summary = roster.get_roster_summary()
    print(f"\nRoster Summary:")
    print(f"  Total slots: {summary['total_slots']}")
    print(f"  Occupied: {summary['occupied']}")
    print(f"  Available: {summary['available']}")
    print(f"  Artists: {summary['artist_names']}")

    return {
        "mode": "roster",
        "summary": summary,
    }


async def run_resilience_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """Run adaptive resilience mode."""
    from gart_gpxm.resilience.adaptive_engine import (
        AdaptiveResilienceEngine,
        SystemState,
        FailureEvent,
        FailureSeverity,
    )

    engine = AdaptiveResilienceEngine()

    # Simulate system state with high entropy
    state = SystemState(
        active_components=["lllm", "daw", "orchestrator"],
        entropy_level=0.85,
        load_factor=0.6,
        error_count=2,
    )

    decision = engine.adapt(state)
    print(f"\nAdaptation Decision:")
    print(f"  Should adapt: {decision.should_adapt}")
    print(f"  Type: {decision.adaptation_type}")
    print(f"  Reason: {decision.reason}")
    print(f"  Target entropy: {decision.target_entropy:.3f}")

    # Simulate failure
    failure = FailureEvent(
        component="lllm_encoder",
        severity=FailureSeverity.HIGH,
        error_message="Encoder output NaN detected",
    )
    recovery = engine.handle_failure(failure)
    print(f"\nRecovery Action:")
    print(f"  Type: {recovery.action_type}")
    print(f"  Target: {recovery.target_component}")

    status = engine.report_status()
    print(f"\nSystem Status:")
    print(f"  Moore growth rate: {status['moore_growth_rate']:.4f}")
    print(f"  Murphy failure prob: {status['murphy_failure_prob']:.4f}")
    print(f"  Balance: {status['moore_murphy_balance']:.4f}")
    print(f"  Degradation: {status['degradation_level']}")
    print(f"  Capabilities: {status['capabilities']}")

    return {
        "mode": "resilience",
        "adapted": decision.should_adapt,
        "status": status,
    }


async def run_demo_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """Run demo mode with all systems."""
    print("=" * 60)
    print("  GART v3.0 + GPXM Quantum Linguistic Synthesizer")
    print("  Demo Mode")
    print("=" * 60)

    # Demo 1: Battle
    print("\n--- Demo 1: Dual-Swarm Battle ---")
    battle_result = await run_battle_mode(args)
    print(f"Winner: {battle_result['winner']}")

    # Demo 2: Reconstruction
    print("\n--- Demo 2: Stash Box Reconstruction ---")
    recon_result = await run_reconstruct_mode(args)
    print(f"Track: {recon_result['track']}")
    print(f"Sections: {recon_result['sections']}")
    print(f"Total bars: {recon_result['total_bars']}")

    # Demo 3: Resilience
    print("\n--- Demo 3: Adaptive Resilience ---")
    resilience_result = await run_resilience_mode(args)

    # Demo 4: Roster
    print("\n--- Demo 4: Roster Management ---")
    roster_result = await run_roster_mode(args)

    return {
        "mode": "demo",
        "battle": battle_result,
        "reconstruction": recon_result,
        "resilience": resilience_result,
        "roster": roster_result,
    }


def _create_demo_personas(count: int) -> List[Any]:
    """Create demo personas for tournament."""
    from gart_gpxm.gpxm.genetic_persona import GeneticPersona, VoiceDifferentiationParameters

    demo_data = [
        ("Lil Durk", "Chicago, IL", "melodic drill", 0.45),
        ("Lil Baby", "Atlanta, GA", "trap", 0.5),
        ("Tay B", "Kentucky", "melodic rap", 0.5),
        ("Drake", "Toronto, ON", "melodic rap", 0.4),
        ("Kendrick Lamar", "Compton, CA", "conscious", 0.35),
        ("J. Cole", "Fayetteville, NC", "conscious", 0.3),
        ("Future", "Atlanta, GA", "trap", 0.55),
        ("Metro Boomin", "Atlanta, GA", "trap", 0.6),
    ]

    personas = []
    for i in range(min(count, len(demo_data))):
        name, origin, genre, entropy = demo_data[i]
        personas.append(GeneticPersona(
            artist_name=name,
            genre_anchor="Hip-Hop",
            sub_genre_tags=[genre],
            era="2020s",
            regional_origin=origin,
            voice_params=VoiceDifferentiationParameters(
                vocabulary_tier="Street",
                slang_density="High",
                emotional_range="Wide",
            ),
            entropy_level=entropy,
            collaboration_affinity=0.5 + (i % 5) * 0.1,
        ))

    # Fill remaining with generic personas
    for i in range(len(demo_data), count):
        personas.append(GeneticPersona(
            artist_name=f"Artist_{i+1}",
            genre_anchor="Hip-Hop",
            era="2020s",
            entropy_level=0.4 + (i % 10) * 0.05,
        ))

    return personas


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start_time = time.monotonic()

    mode_handlers = {
        "tournament": run_tournament_mode,
        "battle": run_battle_mode,
        "reconstruct": run_reconstruct_mode,
        "invert": run_invert_mode,
        "roster": run_roster_mode,
        "resilience": run_resilience_mode,
        "demo": run_demo_mode,
    }

    handler = mode_handlers.get(args.mode, run_demo_mode)
    result = await handler(args)

    elapsed = time.monotonic() - start_time
    result["elapsed_seconds"] = round(elapsed, 2)

    print(f"\nCompleted in {elapsed:.2f}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Output written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
