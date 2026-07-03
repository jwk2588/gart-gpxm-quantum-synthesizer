"""
GART v3.0 — GPXM Quantum Synthesizer CLI Entry Point.

Main entry point for the Genetic Persona Xperience Manager (GPXM) system.
Provides CLI commands for persona management, tournament execution,
code inversion, and system administration.

Usage:
    python -m gart_gpxm.main [COMMAND] [OPTIONS]
    python gart_gpxm/main.py [COMMAND] [OPTIONS]

Commands:
    persona     Manage genetic personas
    tournament  Run tournament brackets
    invert      Convert diagrams to code
    monitor     System health monitoring
    stash       Stash-box reconstruction
    simulate    Run full simulation

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gart")


# ---------------------------------------------------------------------------
# Try to import core modules (graceful fallback if not installed)
# ---------------------------------------------------------------------------

try:
    from gart_gpxm.core.daw_utilities import DAW_SubAgent_Utilities, TextSequence
    from gart_gpxm.core.dual_swarm_orchestrator import DualSwarmOrchestrator
    CORE_AVAILABLE = True
except ImportError as e:
    logger.warning("Core modules not available: %s", e)
    CORE_AVAILABLE = False

try:
    from gart_gpxm.gpxm.genetic_persona import GeneticPersona, GeneType
    from gart_gpxm.gpxm.roster_manager import RosterManager, PersonaTier
    GPXM_AVAILABLE = True
except ImportError as e:
    logger.warning("GPXM modules not available: %s", e)
    GPXM_AVAILABLE = False

try:
    from gart_gpxm.tournament.tournament_engine import (
        TournamentEngine, TournamentFormat,
    )
    TOURNAMENT_AVAILABLE = True
except ImportError as e:
    logger.warning("Tournament module not available: %s", e)
    TOURNAMENT_AVAILABLE = False

try:
    from gart_gpxm.inversion.diagram_to_code_agent import DiagramToCodeAgent
    INVERSION_AVAILABLE = True
except ImportError as e:
    logger.warning("Inversion module not available: %s", e)
    INVERSION_AVAILABLE = False

try:
    from gart_gpxm.resilience.adaptive_engine import AdaptiveEngine
    RESILIENCE_AVAILABLE = True
except ImportError as e:
    logger.warning("Resilience module not available: %s", e)
    RESILIENCE_AVAILABLE = False

try:
    from gart_gpxm.stash_box.reconstruction_engine import ReconstructionEngine
    STASH_AVAILABLE = True
except ImportError as e:
    logger.warning("Stash box module not available: %s", e)
    STASH_AVAILABLE = False

try:
    from gart_gpxm.cross_platform.adapter import PlatformFactory
    CROSS_PLATFORM_AVAILABLE = True
except ImportError as e:
    logger.warning("Cross-platform module not available: %s", e)
    CROSS_PLATFORM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
   ____    _    ____  _____ 
  / ___|  / \  |  _ \|_   _|
 | |  _  / _ \ | |_) | | |  
 | |_| |/ ___ \|  _ <  | |  
  \____/_/   \_\_| \_\ |_|  
                            
  GPXM Quantum Synthesizer v3.0
  Genetic Persona Xperience Manager
"""


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


def cmd_persona(args: argparse.Namespace) -> int:
    """Handle persona management commands."""
    if not GPXM_AVAILABLE:
        logger.error("GPXM modules not available. Install gart-gpxm package.")
        return 1

    print(f"\n{'='*60}")
    print("  PERSONA MANAGEMENT")
    print(f"{'='*60}\n")

    if args.subcommand == "create":
        print(f"Creating founder persona: {args.name}")
        builder = GeneticPersona(seed=args.seed)
        
        skill_profile = {}
        if args.flow:
            skill_profile[GeneType.FLOW] = args.flow / 100.0
        if args.delivery:
            skill_profile[GeneType.DELIVERY] = args.delivery / 100.0
        if args.wordplay:
            skill_profile[GeneType.WORDPLAY] = args.wordplay / 100.0

        dna = builder.create_founder(args.name, skill_profile or None)
        print(f"\nCreated: {dna.persona_id}")
        print(f"Generation: {dna.generation}")
        print(f"Signature: {dna.signature}")
        print(f"Fitness: {dna.get_overall_fitness():.3f}")
        print(f"\nExpressed traits:")
        for trait, value in dna.get_expressed_traits().items():
            print(f"  {trait.value}: {value:.3f}")

    elif args.subcommand == "evolve":
        print("Evolving generation...")
        builder = GeneticPersona(seed=args.seed)
        
        # Create initial population
        population = []
        for i in range(args.population):
            dna = builder.create_founder(f"Founder_{i+1}")
            population.append(dna)

        print(f"Initial population: {len(population)}")
        
        for gen in range(args.generations):
            offspring = builder.evolve_generation(population, args.offspring)
            population.extend(offspring)
            print(f"  Generation {gen+1}: {len(population)} total")

        # Show best
        best = builder.select_best(population, n=1)[0]
        print(f"\nBest fitness: {best.get_overall_fitness():.3f}")
        print(f"Best signature: {best.signature}")

    elif args.subcommand == "roster":
        manager = RosterManager()
        
        # Add sample entries
        for i in range(5):
            manager.create(
                f"Artist_{i+1}",
                tier=PersonaTier.PROSPECT,
                skills={"flow": 0.7, "delivery": 0.6, "wordplay": 0.8},
            )
        
        print(f"Roster entries: {manager.count()}")
        for entry in manager.list_all():
            print(f"  {entry.persona_id}: {entry.artist_name} "
                  f"(rating: {entry.overall_rating:.2f})")

    else:
        print("Available subcommands: create, evolve, roster")

    return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    """Handle tournament commands."""
    if not TOURNAMENT_AVAILABLE:
        logger.error("Tournament module not available.")
        return 1

    print(f"\n{'='*60}")
    print("  TOURNAMENT ENGINE")
    print(f"{'='*60}\n")

    fmt = TournamentFormat(args.format)
    engine = TournamentEngine(fmt)

    # Register competitors
    names = args.competitors.split(",") if args.competitors else [
        "MC_Alpha", "MC_Beta", "MC_Gamma", "MC_Delta",
        "MC_Epsilon", "MC_Zeta", "MC_Eta", "MC_Theta",
    ]

    for i, name in enumerate(names):
        engine.register_competitor(
            f"comp_{i}", name.strip(), seed=i+1, elo_rating=1500 + random.randint(-200, 200),
        )

    print(f"Format: {fmt.value}")
    print(f"Competitors: {len(names)}")

    engine.start_tournament()

    # Simulate rounds
    round_num = 1
    while True:
        current_matches = [
            m for m in engine.bracket._matches.values()
            if m.round_number == round_num and m.status.value == "pending"
        ]

        if not current_matches:
            break

        print(f"\n--- Round {round_num} ---")
        for match in current_matches:
            # Simulate scoring
            comp_a = engine.bracket._competitors[match.competitor_a]
            comp_b = engine.bracket._competitors[match.competitor_b]

            scores = {
                match.competitor_a: {
                    "flow_complexity": random.uniform(0.5, 1.0),
                    "rhyme_density": random.uniform(0.5, 1.0),
                    "thematic_depth": random.uniform(0.5, 1.0),
                    "vocal_presence": random.uniform(0.5, 1.0),
                },
                match.competitor_b: {
                    "flow_complexity": random.uniform(0.5, 1.0),
                    "rhyme_density": random.uniform(0.5, 1.0),
                    "thematic_depth": random.uniform(0.5, 1.0),
                    "vocal_presence": random.uniform(0.5, 1.0),
                },
            }

            winner = engine.score_match(match.match_id, scores)
            winner_name = engine.bracket._competitors[winner].name if winner else "TIE"
            print(f"  {comp_a.name} vs {comp_b.name} -> {winner_name}")

        round_num += 1
        if not engine.advance():
            break

    # Results
    champion = engine.get_champion()
    print(f"\n{'='*60}")
    if champion:
        print(f"  CHAMPION: {champion.name} (Elo: {champion.elo:.0f})")
    print(f"{'='*60}")

    print("\nStandings:")
    for i, comp in enumerate(engine.get_standings()[:5], 1):
        print(f"  {i}. {comp.name} — W:{comp.wins} L:{comp.losses} Elo:{comp.elo:.0f}")

    return 0


def cmd_invert(args: argparse.Namespace) -> int:
    """Handle diagram-to-code inversion."""
    if not INVERSION_AVAILABLE:
        logger.error("Inversion module not available.")
        return 1

    print(f"\n{'='*60}")
    print("  DIAGRAM-TO-CODE INVERSION")
    print(f"{'='*60}\n")

    agent = DiagramToCodeAgent()

    if args.input:
        with open(args.input, "r") as f:
            description = f.read()
    else:
        description = """
        module Orchestrator at (100, 100)
        class TournamentEngine at (300, 100)
        class PersonaBuilder at (500, 100)
        Orchestrator -> TournamentEngine [calls]
        Orchestrator -> PersonaBuilder [calls]
        """

    fmt = "json" if args.input and args.input.endswith(".json") else "text"
    result = agent.invert(description, fmt)

    print(f"Elements: {result['ast']['elements']}")
    print(f"Connections: {result['ast']['connections']}")
    print(f"Modules planned: {result['plan']['modules']}")
    print(f"Classes planned: {result['plan']['classes']}")
    print(f"Files generated: {len(result['files'])}")

    for file_info in result["files"]:
        print(f"\n--- {file_info['filename']} ---")
        print(result["generated_code"][file_info["filename"]])

    if args.output:
        agent.invert_and_write(description, args.output, fmt)
        print(f"\nWritten to: {args.output}")

    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """Handle system monitoring."""
    print(f"\n{'='*60}")
    print("  SYSTEM MONITORING")
    print(f"{'='*60}\n")

    if RESILIENCE_AVAILABLE:
        engine = AdaptiveEngine()
        stats = engine.get_stats()
        print(json.dumps(stats, indent=2, default=str))
    else:
        print("Resilience module not available.")
        print("Platform:", sys.platform)
        print("Python:", sys.version)

    return 0


def cmd_stash(args: argparse.Namespace) -> int:
    """Handle stash-box operations."""
    if not STASH_AVAILABLE:
        logger.error("Stash box module not available.")
        return 1

    print(f"\n{'='*60}")
    print("  STASH-BOX RECONSTRUCTION")
    print(f"{'='*60}\n")

    engine = ReconstructionEngine()

    if args.input:
        with open(args.input, "r") as f:
            content = f.read()
        engine.stash_code(args.input, content)
        print(f"Stashed: {args.input}")
    else:
        # Demo stashes
        engine.stash_code("core/orchestrator.py", "class Orchestrator: pass")
        engine.stash_code("core/engine.py", "class Engine: pass")
        engine.stash_code("gpxm/persona.py", "class Persona: pass")
        print("Demo stashes created: 3 fragments")

    stats = engine.get_stats()
    print(f"\nTotal fragments: {stats['total_fragments']}")
    print(f"Integrity: {stats['integrity']['valid']}/{stats['integrity']['total']} valid")

    # Reconstruct
    result = engine.reconstruct_all()
    print(f"\nReconstruction: {result.status.value}")
    print(f"Fragments used: {result.fragments_used}/{result.fragments_total}")

    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run full system simulation."""
    print(f"\n{'='*60}")
    print("  FULL SYSTEM SIMULATION")
    print(f"{'='*60}\n")

    print("Phase 1: Persona Creation")
    if GPXM_AVAILABLE:
        builder = GeneticPersona(seed=42)
        population = []
        for i in range(4):
            dna = builder.create_founder(f"MC_{i+1}")
            population.append(dna)
            print(f"  Created: {dna.persona_id} (fitness: {dna.get_overall_fitness():.3f})")
    else:
        print("  [GPXM not available]")

    print("\nPhase 2: Tournament")
    if TOURNAMENT_AVAILABLE and GPXM_AVAILABLE:
        engine = TournamentEngine(TournamentFormat.SINGLE_ELIMINATION)
        for i, dna in enumerate(population):
            engine.register_competitor(
                dna.persona_id, dna.persona_id.split("_G0_")[0],
                seed=i+1, elo_rating=1400 + i * 50,
            )
        engine.start_tournament()
        print(f"  Competitors: {len(population)}")
        print(f"  Matches: {len(engine.bracket._matches)}")
    else:
        print("  [Tournament not available]")

    print("\nPhase 3: DAW Processing")
    if CORE_AVAILABLE:
        daw = DAW_SubAgent_Utilities()
        print(f"  Sidechain: {daw.sidechain_compressor_agent()}")
    else:
        print("  [Core not available]")

    print("\nPhase 4: System Stats")
    available = []
    if CORE_AVAILABLE:
        available.append("core")
    if GPXM_AVAILABLE:
        available.append("gpxm")
    if TOURNAMENT_AVAILABLE:
        available.append("tournament")
    if INVERSION_AVAILABLE:
        available.append("inversion")
    if RESILIENCE_AVAILABLE:
        available.append("resilience")
    if STASH_AVAILABLE:
        available.append("stash_box")
    if CROSS_PLATFORM_AVAILABLE:
        available.append("cross_platform")

    print(f"  Available modules: {', '.join(available)}")
    print(f"  Total: {len(available)}/7 modules")

    print(f"\n{'='*60}")
    print("  SIMULATION COMPLETE")
    print(f"{'='*60}")

    return 0


# ---------------------------------------------------------------------------
# Argument Parser Setup
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="gart",
        description="GART v3.0 — GPXM Quantum Synthesizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s persona create --name "Lil_Quantum"
  %(prog)s tournament --format single_elimination
  %(prog)s invert --input diagram.txt --output ./generated
  %(prog)s simulate
        """,
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 3.0.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Persona command
    persona_parser = subparsers.add_parser("persona", help="Manage genetic personas")
    persona_sub = persona_parser.add_subparsers(dest="subcommand")

    create_parser_cmd = persona_sub.add_parser("create", help="Create a founder persona")
    create_parser_cmd.add_argument("--name", required=True, help="Persona name")
    create_parser_cmd.add_argument("--seed", type=int, default=None, help="Random seed")
    create_parser_cmd.add_argument("--flow", type=float, default=0, help="Flow skill (0-100)")
    create_parser_cmd.add_argument("--delivery", type=float, default=0, help="Delivery skill")
    create_parser_cmd.add_argument("--wordplay", type=float, default=0, help="Wordplay skill")

    evolve_parser = persona_sub.add_parser("evolve", help="Evolve personas")
    evolve_parser.add_argument("--population", type=int, default=4, help="Initial population")
    evolve_parser.add_argument("--generations", type=int, default=3, help="Generations")
    evolve_parser.add_argument("--offspring", type=int, default=4, help="Offspring per gen")
    evolve_parser.add_argument("--seed", type=int, default=None, help="Random seed")

    roster_parser = persona_sub.add_parser("roster", help="Manage roster")

    # Tournament command
    tournament_parser = subparsers.add_parser("tournament", help="Run tournament")
    tournament_parser.add_argument(
        "--format", default="single_elimination",
        choices=["single_elimination", "double_elimination", "round_robin", "swiss"],
        help="Tournament format",
    )
    tournament_parser.add_argument(
        "--competitors", default="", help="Comma-separated competitor names",
    )

    # Invert command
    invert_parser = subparsers.add_parser("invert", help="Diagram-to-code inversion")
    invert_parser.add_argument("--input", help="Input diagram file")
    invert_parser.add_argument("--output", help="Output directory")

    # Monitor command
    subparsers.add_parser("monitor", help="System monitoring")

    # Stash command
    stash_parser = subparsers.add_parser("stash", help="Stash-box operations")
    stash_parser.add_argument("--input", help="File to stash")

    # Simulate command
    subparsers.add_parser("simulate", help="Full simulation")

    return parser


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point.

    Args:
        argv: Command line arguments.

    Returns:
        Exit code.
    """
    print(BANNER)

    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "persona": cmd_persona,
        "tournament": cmd_tournament,
        "invert": cmd_invert,
        "monitor": cmd_monitor,
        "stash": cmd_stash,
        "simulate": cmd_simulate,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            return handler(args)
        except Exception as e:
            logger.error("Command failed: %s", e)
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
