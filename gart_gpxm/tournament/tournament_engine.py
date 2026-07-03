"""
Tournament Engine — GART v3.0.

16-artist tournament with GOAT metrics, matchup simulation,
and dual-swarm battle orchestration.

Features:
    - YBNBA formula: (certs_total * 0.4) + (certs_in_one_day * 0.6)
    - Plaque Velocity scoring
    - Cultural Entropy density evaluation
    - Elo rating system
    - Bracket visualization

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BattleResult:
    """Result of a single battle between two artists.

    Attributes:
        artist_a_id: First artist persona ID.
        artist_b_id: Second artist persona ID.
        winner_id: ID of the winning artist.
        artist_a_score: Score for artist A.
        artist_b_score: Score for artist B.
        plaque_velocity: Plaque velocity score.
        cultural_entropy: Cultural entropy density.
        round_number: Tournament round number.
        metadata: Additional battle metadata.
    """

    artist_a_id: str
    artist_b_id: str
    winner_id: Optional[str] = None
    artist_a_score: float = 0.0
    artist_b_score: float = 0.0
    plaque_velocity: float = 0.0
    cultural_entropy: float = 0.0
    round_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeaderboardEntry:
    """Entry on the tournament leaderboard.

    Attributes:
        artist_id: Persona ID.
        artist_name: Display name.
        wins: Number of wins.
        losses: Number of losses.
        total_score: Cumulative score.
        elo_rating: Current Elo rating.
        rank: Current rank.
    """

    artist_id: str
    artist_name: str = ""
    wins: int = 0
    losses: int = 0
    total_score: float = 0.0
    elo_rating: float = 1500.0
    rank: int = 0


@dataclass
class TournamentResult:
    """Complete tournament result.

    Attributes:
        champion_id: Winning artist ID.
        champion_name: Winning artist name.
        runner_up_id: Runner-up ID.
        leaderboard: Final leaderboard entries.
        bracket_history: Battle results by round.
        total_rounds: Number of rounds played.
        metadata: Tournament metadata.
    """

    champion_id: Optional[str] = None
    champion_name: str = ""
    runner_up_id: Optional[str] = None
    leaderboard: List[LeaderboardEntry] = field(default_factory=list)
    bracket_history: List[List[BattleResult]] = field(default_factory=list)
    total_rounds: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ScoringEngine
# ---------------------------------------------------------------------------


class ScoringEngine:
    """Weighted scoring engine for battle evaluation.

    Evaluates cyphers across multiple weighted criteria:
        - flow_complexity: Rhythmic and structural complexity
        - rhyme_density: Frequency and quality of rhymes
        - thematic_depth: Narrative and thematic richness
        - cultural_entropy: Cultural reference density
        - vocal_presence: Delivery and presence score
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "flow_complexity": 0.25,
        "rhyme_density": 0.20,
        "thematic_depth": 0.25,
        "cultural_entropy": 0.15,
        "vocal_presence": 0.15,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def score(self, criteria_scores: Dict[str, float]) -> float:
        """Calculate weighted composite score.

        Args:
            criteria_scores: Dict mapping criterion names to scores.

        Returns:
            Weighted composite score (0.0-1.0).
        """
        total = 0.0
        weight_sum = 0.0
        for criterion, score in criteria_scores.items():
            weight = self.weights.get(criterion, 0.1)
            total += score * weight
            weight_sum += weight
        if weight_sum == 0:
            return 0.0
        return min(1.0, max(0.0, total / weight_sum))

    def aggregate_scores(self, scores: List[float]) -> float:
        """Aggregate multiple scores.

        Args:
            scores: List of individual scores.

        Returns:
            Aggregated score.
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def normalize(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize a score to [0, 1] range.

        Args:
            score: Raw score.
            min_val: Minimum expected value.
            max_val: Maximum expected value.

        Returns:
            Normalized score.
        """
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))

    def calculate_plaque_velocity(
        self,
        certs_total: int,
        certs_in_one_day: int,
    ) -> float:
        """Calculate plaque velocity score.

        Higher values indicate rapid achievement recognition.

        Args:
            certs_total: Total certifications.
            certs_in_one_day: Certifications achieved in one day.

        Returns:
            Plaque velocity score.
        """
        if certs_total <= 0:
            return 0.0
        ratio = certs_in_one_day / certs_total
        return min(1.0, ratio * 2.0 + certs_in_one_day / 100.0)


# ---------------------------------------------------------------------------
# EloRatingSystem
# ---------------------------------------------------------------------------


class EloRatingSystem:
    """Elo rating system for tournament participants.

    Standard chess Elo with K-factor of 32.
    Formula: R' = R + K * (S - E)
    where E = 1 / (1 + 10^((R_opponent - R) / 400))
    """

    DEFAULT_K: int = 32
    INITIAL_RATING: float = 1500.0

    def __init__(self, k_factor: int = DEFAULT_K) -> None:
        self.k_factor = k_factor
        self._ratings: Dict[str, float] = {}

    def get_rating(self, artist_id: str) -> float:
        """Get current rating for an artist.

        Args:
            artist_id: Artist persona ID.

        Returns:
            Current Elo rating.
        """
        return self._ratings.get(artist_id, self.INITIAL_RATING)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using Elo formula.

        Args:
            rating_a: Rating of player A.
            rating_b: Rating of player B.

        Returns:
            Expected score for player A (0.0-1.0).
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(
        self,
        artist_a_id: str,
        artist_b_id: str,
        result_a: float,  # 1.0 for win, 0.5 for draw, 0.0 for loss
    ) -> Tuple[float, float]:
        """Update ratings after a match.

        Args:
            artist_a_id: ID of artist A.
            artist_b_id: ID of artist B.
            result_a: Result for artist A (1.0=win, 0.5=draw, 0.0=loss).

        Returns:
            (new_rating_a, new_rating_b).
        """
        rating_a = self.get_rating(artist_a_id)
        rating_b = self.get_rating(artist_b_id)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        result_b = 1.0 - result_a

        new_a = rating_a + self.k_factor * (result_a - expected_a)
        new_b = rating_b + self.k_factor * (result_b - expected_b)

        self._ratings[artist_a_id] = new_a
        self._ratings[artist_b_id] = new_b

        return new_a, new_b

    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Get sorted leaderboard.

        Returns:
            List of (artist_id, rating) sorted by rating descending.
        """
        return sorted(
            self._ratings.items(),
            key=lambda x: x[1],
            reverse=True,
        )


# ---------------------------------------------------------------------------
# MatchupSimulator
# ---------------------------------------------------------------------------


class MatchupSimulator:
    """Simulates battles between two artist personas.

    Uses the scoring engine and optional LLLM judge to evaluate
    head-to-head matchups across multiple criteria.
    """

    def __init__(self, scoring_engine: Optional[ScoringEngine] = None) -> None:
        self.scoring_engine = scoring_engine or ScoringEngine()

    def simulate(
        self,
        artist_a: Any,
        artist_b: Any,
    ) -> BattleResult:
        """Simulate a battle between two artists.

        Args:
            artist_a: First GeneticPersona.
            artist_b: Second GeneticPersona.

        Returns:
            BattleResult with scores and winner.
        """
        a_id = getattr(artist_a, "persona_id", "a")
        b_id = getattr(artist_b, "persona_id", "b")
        a_name = getattr(artist_a, "artist_name", "Artist A")
        b_name = getattr(artist_b, "artist_name", "Artist B")

        # Score each artist across criteria
        a_scores = self._score_artist(artist_a)
        b_scores = self._score_artist(artist_b)

        a_total = self.scoring_engine.score(a_scores)
        b_total = self.scoring_engine.score(b_scores)

        # Determine winner
        winner = a_id if a_total > b_total else b_id if b_total > a_total else None

        # Calculate cultural entropy
        a_entropy = getattr(artist_a, "entropy_level", 0.5)
        b_entropy = getattr(artist_b, "entropy_level", 0.5)
        cultural_entropy = (a_entropy + b_entropy) / 2.0

        return BattleResult(
            artist_a_id=a_id,
            artist_b_id=b_id,
            winner_id=winner,
            artist_a_score=round(a_total, 4),
            artist_b_score=round(b_total, 4),
            cultural_entropy=round(cultural_entropy, 4),
            metadata={
                "artist_a_name": a_name,
                "artist_b_name": b_name,
                "a_criteria_scores": a_scores,
                "b_criteria_scores": b_scores,
            },
        )

    def _score_artist(self, artist: Any) -> Dict[str, float]:
        """Score an artist across all criteria.

        Args:
            artist: GeneticPersona.

        Returns:
            Dict of criterion -> score.
        """
        entropy = getattr(artist, "entropy_level", 0.5)
        collab = getattr(artist, "collaboration_affinity", 0.5)
        fitness = getattr(artist, "genetic_fitness", 0.0)

        return {
            "flow_complexity": 0.3 + entropy * 0.5,
            "rhyme_density": 0.2 + collab * 0.4,
            "thematic_depth": fitness if fitness > 0 else 0.4,
            "cultural_entropy": entropy,
            "vocal_presence": collab,
        }

    def break_tie(self, artist_a: Any, artist_b: Any) -> str:
        """Break a tie between two artists.

        Uses collaboration affinity as tiebreaker.

        Args:
            artist_a: First artist.
            artist_b: Second artist.

        Returns:
            ID of tiebreak winner.
        """
        a_collab = getattr(artist_a, "collaboration_affinity", 0.5)
        b_collab = getattr(artist_b, "collaboration_affinity", 0.5)
        return (
            getattr(artist_a, "persona_id", "a")
            if a_collab >= b_collab else
            getattr(artist_b, "persona_id", "b")
        )


# ---------------------------------------------------------------------------
# TournamentBracket
# ---------------------------------------------------------------------------


class TournamentBracket:
    """Tournament bracket manager.

    Handles bracket seeding, winner advancement, and bracket
    completion detection for single-elimination tournaments.
    """

    def __init__(self, participants: List[Any]) -> None:
        self.participants = list(participants)
        self.current_round = 0
        self.rounds: List[List[Tuple[Any, Any]]] = []
        self._results: List[List[BattleResult]] = []

    def seed_bracket(self) -> List[Tuple[Any, Any]]:
        """Create initial matchups (random seeding).

        Returns:
            List of (artist_a, artist_b) matchup tuples.
        """
        import random
        shuffled = self.participants.copy()
        random.shuffle(shuffled)

        matchups: List[Tuple[Any, Any]] = []
        for i in range(0, len(shuffled), 2):
            if i + 1 < len(shuffled):
                matchups.append((shuffled[i], shuffled[i + 1]))
            else:
                # Bye
                matchups.append((shuffled[i], None))

        self.rounds.append(matchups)
        return matchups

    def advance_winners(self, results: List[BattleResult]) -> List[Tuple[Any, Any]]:
        """Advance winners to next round.

        Args:
            results: Battle results from current round.

        Returns:
            Next round matchups.
        """
        self._results.append(results)
        self.current_round += 1

        winners: List[Any] = []
        for result in results:
            if result.winner_id:
                # Find winner object
                for p in self.participants:
                    if getattr(p, "persona_id", None) == result.winner_id:
                        winners.append(p)
                        break

        if len(winners) <= 1:
            return []  # Tournament complete

        # Create next round matchups
        matchups: List[Tuple[Any, Any]] = []
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                matchups.append((winners[i], winners[i + 1]))
            else:
                matchups.append((winners[i], None))

        self.rounds.append(matchups)
        return matchups

    def is_complete(self) -> bool:
        """Check if tournament is complete.

        Returns:
            True if tournament has a single winner.
        """
        if not self.rounds:
            return False
        last_round = self.rounds[-1]
        return len(last_round) == 1 and last_round[0][1] is None

    def get_current_matchups(self) -> List[Tuple[Any, Any]]:
        """Get current round matchups.

        Returns:
            List of matchup tuples.
        """
        if self.current_round < len(self.rounds):
            return self.rounds[self.current_round]
        return []

    def get_champion(self) -> Optional[Any]:
        """Get the tournament champion.

        Returns:
            Champion artist or None.
        """
        if not self.is_complete() or not self._results:
            return None
        last_result = self._results[-1][0]
        winner_id = last_result.winner_id
        for p in self.participants:
            if getattr(p, "persona_id", None) == winner_id:
                return p
        return None


# ---------------------------------------------------------------------------
# TournamentEngine
# ---------------------------------------------------------------------------


class TournamentEngine:
    """Tournament Engine for GART v3.0.

    Runs 16-artist tournaments with GOAT metrics, matchup simulation,
    Elo ratings, and bracket management.

    Attributes:
        roster: RosterManager with participants.
        scoring_engine: ScoringEngine for evaluation.
        elo_system: EloRatingSystem for ratings.
        simulator: MatchupSimulator for battles.
    """

    def __init__(self, roster: Optional[Any] = None) -> None:
        self.roster = roster
        self.scoring_engine = ScoringEngine()
        self.elo_system = EloRatingSystem()
        self.simulator = MatchupSimulator(self.scoring_engine)
        self._bracket: Optional[TournamentBracket] = None
        self._history: List[TournamentResult] = []

    @staticmethod
    def calculate_goat_metrics(
        certs_total: int,
        certs_in_one_day: int,
    ) -> float:
        """Calculate GOAT (Greatest Of All Time) metrics.

        YBNBA formula: (certs_total * 0.4) + (certs_in_one_day * 0.6)

        Args:
            certs_total: Total certifications.
            certs_in_one_day: Peak single-day certifications.

        Returns:
            GOAT metric score.
        """
        return (certs_total * 0.4) + (certs_in_one_day * 0.6)

    def run_tournament(self, participants: Optional[List[Any]] = None) -> TournamentResult:
        """Run a complete tournament.

        Args:
            participants: Optional list of participants. Uses roster if None.

        Returns:
            TournamentResult with champion and leaderboard.
        """
        if participants is None and self.roster is not None:
            participants = self.roster.get_all_artists()

        if not participants:
            return TournamentResult(metadata={"error": "No participants"})

        # Initialize bracket
        self._bracket = TournamentBracket(participants)
        matchups = self._bracket.seed_bracket()

        all_round_results: List[List[BattleResult]] = []
        round_num = 0

        while matchups:
            round_num += 1
            round_results: List[BattleResult] = []

            for artist_a, artist_b in matchups:
                if artist_b is None:
                    # Bye
                    result = BattleResult(
                        artist_a_id=getattr(artist_a, "persona_id", ""),
                        artist_b_id="bye",
                        winner_id=getattr(artist_a, "persona_id", ""),
                        artist_a_score=1.0,
                        artist_b_score=0.0,
                        round_number=round_num,
                    )
                else:
                    result = self.simulate_battle(artist_a, artist_b)
                    result.round_number = round_num

                round_results.append(result)

                # Update Elo ratings
                a_id = result.artist_a_id
                b_id = result.artist_b_id
                if result.winner_id == a_id:
                    self.elo_system.update_ratings(a_id, b_id, 1.0)
                elif result.winner_id == b_id:
                    self.elo_system.update_ratings(a_id, b_id, 0.0)
                else:
                    self.elo_system.update_ratings(a_id, b_id, 0.5)

            all_round_results.append(round_results)
            matchups = self._bracket.advance_winners(round_results)

        # Build result
        champion = self._bracket.get_champion()
        result = TournamentResult(
            champion_id=getattr(champion, "persona_id", None) if champion else None,
            champion_name=getattr(champion, "artist_name", "") if champion else "",
            bracket_history=all_round_results,
            total_rounds=round_num,
            leaderboard=self._build_leaderboard(participants),
            metadata={
                "participant_count": len(participants),
                "total_battles": sum(len(r) for r in all_round_results),
            },
        )

        self._history.append(result)
        return result

    def simulate_battle(self, artist_a: Any, artist_b: Any) -> BattleResult:
        """Simulate a single battle between two artists.

        Args:
            artist_a: First artist.
            artist_b: Second artist.

        Returns:
            BattleResult.
        """
        result = self.simulator.simulate(artist_a, artist_b)

        # Calculate plaque velocity
        a_fitness = getattr(artist_a, "genetic_fitness", 0.5)
        b_fitness = getattr(artist_b, "genetic_fitness", 0.5)
        result.plaque_velocity = self.scoring_engine.calculate_plaque_velocity(
            certs_total=int(a_fitness * 100),
            certs_in_one_day=int(max(a_fitness, b_fitness) * 50),
        )

        return result

    def _build_leaderboard(self, participants: List[Any]) -> List[LeaderboardEntry]:
        """Build final leaderboard from participants.

        Args:
            participants: All participants.

        Returns:
            Sorted leaderboard entries.
        """
        entries: List[LeaderboardEntry] = []
        ratings = self.elo_system.get_leaderboard()
        rating_dict = dict(ratings)

        for p in participants:
            pid = getattr(p, "persona_id", "")
            entries.append(LeaderboardEntry(
                artist_id=pid,
                artist_name=getattr(p, "artist_name", "Unknown"),
                elo_rating=rating_dict.get(pid, 1500.0),
                total_score=getattr(p, "genetic_fitness", 0.0),
            ))

        # Sort by Elo rating
        entries.sort(key=lambda e: e.elo_rating, reverse=True)
        for i, entry in enumerate(entries):
            entry.rank = i + 1

        return entries

    def get_leaderboard(self) -> List[LeaderboardEntry]:
        """Get current leaderboard.

        Returns:
            Leaderboard entries.
        """
        return self.elo_system.get_leaderboard()

    def get_history(self) -> List[TournamentResult]:
        """Get tournament history.

        Returns:
            List of past tournament results.
        """
        return self._history

    def export_bracket_text(self) -> str:
        """Export bracket as text representation.

        Returns:
            ASCII bracket string.
        """
        if self._bracket is None:
            return "No tournament run yet."

        lines: List[str] = ["=== TOURNAMENT BRACKET ===\n"]
        for round_num, results in enumerate(self._bracket._results):
            lines.append(f"\n--- Round {round_num + 1} ---")
            for r in results:
                a_name = r.metadata.get("artist_a_name", r.artist_a_id[:8])
                b_name = r.metadata.get("artist_b_name", r.artist_b_id[:8])
                winner = "A" if r.winner_id == r.artist_a_id else "B"
                lines.append(
                    f"  {a_name} ({r.artist_a_score:.3f}) vs "
                    f"{b_name} ({r.artist_b_score:.3f}) -> {winner}"
                )

        champion = self._bracket.get_champion()
        if champion:
            lines.append(
                f"\nCHAMPION: {getattr(champion, 'artist_name', 'Unknown')}"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Tournament Engine loaded successfully.")

    # Create mock participants
    from dataclasses import dataclass as dc, field as f

    @dc
    class MockPersona:
        persona_id: str
        artist_name: str
        entropy_level: float = 0.5
        collaboration_affinity: float = 0.5
        genetic_fitness: float = 0.5

    participants = [
        MockPersona("p1", "Lil Durk", 0.45, 0.7, 0.8),
        MockPersona("p2", "Lil Baby", 0.5, 0.8, 0.85),
        MockPersona("p3", "Tay B", 0.5, 0.6, 0.6),
        MockPersona("p4", "Drake", 0.4, 0.9, 0.9),
        MockPersona("p5", "Kendrick", 0.35, 0.7, 0.95),
        MockPersona("p6", "J. Cole", 0.3, 0.6, 0.85),
        MockPersona("p7", "Future", 0.55, 0.8, 0.75),
        MockPersona("p8", "Metro Boomin", 0.6, 0.9, 0.8),
    ]

    engine = TournamentEngine()
    result = engine.run_tournament(participants)

    print(f"\nChampion: {result.champion_name}")
    print(f"Rounds: {result.total_rounds}")
    print(f"Total battles: {result.metadata.get('total_battles', 0)}")
    print("\nLeaderboard:")
    for entry in result.leaderboard[:5]:
        print(f"  #{entry.rank} {entry.artist_name}: Elo={entry.elo_rating:.1f}")

    print(f"\n{engine.export_bracket_text()}")

    # Test GOAT metrics
    goat = TournamentEngine.calculate_goat_metrics(50, 10)
    print(f"\nGOAT Metric (50 total, 10 in one day): {goat:.1f}")
