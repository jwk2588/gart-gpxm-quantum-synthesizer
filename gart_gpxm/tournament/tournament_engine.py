"""
Tournament Engine — Competitive Bracket System for GART v3.0.

Manages tournament brackets, match scheduling, scoring,
and champion determination with Elo-based rankings.

Components:
    - TournamentEngine: Main bracket orchestrator
    - Bracket: Single/double elimination bracket
    - Match: Individual matchup with scoring
    - EloRanking: Elo rating system
    - ScoringJudge: Multi-criteria scoring

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TournamentError(Exception):
    """Base exception for tournament operations."""


class BracketError(TournamentError):
    """Raised when bracket operations fail."""


class MatchError(TournamentError):
    """Raised when match operations fail."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TournamentFormat(Enum):
    """Tournament bracket formats."""

    SINGLE_ELIMINATION = "single_elimination"
    DOUBLE_ELIMINATION = "double_elimination"
    ROUND_ROBIN = "round_robin"
    SWISS = "swiss"


class MatchStatus(Enum):
    """Status of a match."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FORFEITED = "forfeited"
    DISPUTED = "disputed"


class ScoringCriterion(Enum):
    """Scoring criteria for matches."""

    FLOW_COMPLEXITY = "flow_complexity"
    RHYME_DENSITY = "rhyme_density"
    THEMATIC_DEPTH = "thematic_depth"
    CULTURAL_ENTROPY = "cultural_entropy"
    VOCAL_PRESENCE = "vocal_presence"
    CROWD_REACTION = "crowd_reaction"
    TECHNICAL_SKILL = "technical_skill"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class Competitor:
    """A tournament competitor.

    Attributes:
        competitor_id: Unique identifier.
        name: Display name.
        seed: Initial seed ranking.
        elo: Elo rating.
        wins: Win count.
        losses: Loss count.
        metadata: Additional data.
    """

    competitor_id: str
    name: str
    seed: int = 0
    elo: float = 1500.0
    wins: int = 0
    losses: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return self.wins / total

    def expected_score(self, opponent_elo: float) -> float:
        """Calculate expected score against opponent.

        Args:
            opponent_elo: Opponent's Elo rating.

        Returns:
            Expected score (0.0-1.0).
        """
        return 1.0 / (1.0 + 10.0 ** ((opponent_elo - self.elo) / 400.0))


@dataclass
class MatchScore:
    """Score for a single competitor in a match.

    Attributes:
        competitor_id: Competitor ID.
        criteria_scores: Dict of criterion -> score.
        total_score: Computed total.
    """

    competitor_id: str
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0


@dataclass
class Match:
    """A single tournament match.

    Attributes:
        match_id: Unique identifier.
        round_number: Tournament round.
        competitor_a: First competitor.
        competitor_b: Second competitor.
        scores: Match scores.
        status: Match status.
        winner_id: Winner competitor ID.
    """

    match_id: str
    round_number: int
    competitor_a: str
    competitor_b: str
    scores: Dict[str, MatchScore] = field(default_factory=dict)
    status: MatchStatus = MatchStatus.PENDING
    winner_id: Optional[str] = None

    def set_score(
        self,
        competitor_id: str,
        criterion: str,
        score: float,
    ) -> None:
        """Set a criterion score.

        Args:
            competitor_id: Competitor to score.
            criterion: Criterion name.
            score: Score value (0.0-1.0).
        """
        if competitor_id not in self.scores:
            self.scores[competitor_id] = MatchScore(competitor_id=competitor_id)
        self.scores[competitor_id].criteria_scores[criterion] = max(0.0, min(1.0, score))

    def compute_total(self, competitor_id: str) -> float:
        """Compute total score for a competitor.

        Args:
            competitor_id: Competitor to compute.

        Returns:
            Total score.
        """
        score = self.scores.get(competitor_id)
        if not score or not score.criteria_scores:
            return 0.0
        return sum(score.criteria_scores.values()) / len(score.criteria_scores)

    def determine_winner(self) -> Optional[str]:
        """Determine winner based on scores.

        Returns:
            Winner competitor ID or None.
        """
        score_a = self.compute_total(self.competitor_a)
        score_b = self.compute_total(self.competitor_b)

        if score_a > score_b:
            return self.competitor_a
        elif score_b > score_a:
            return self.competitor_b
        return None  # Tie


# ---------------------------------------------------------------------------
# EloRanking
# ---------------------------------------------------------------------------


class EloRanking:
    """Elo rating system for competitor rankings.

    Standard Elo with configurable K-factor.
    """

    def __init__(self, k_factor: float = 32.0) -> None:
        self.k_factor = k_factor

    def update_ratings(
        self,
        winner: Competitor,
        loser: Competitor,
        draw: bool = False,
    ) -> Tuple[float, float]:
        """Update Elo ratings after a match.

        Args:
            winner: Match winner.
            loser: Match loser.
            draw: Whether match was a draw.

        Returns:
            Tuple of (winner_new_elo, loser_new_elo).
        """
        expected_winner = winner.expected_score(loser.elo)
        expected_loser = loser.expected_score(winner.elo)

        if draw:
            actual_winner = 0.5
            actual_loser = 0.5
        else:
            actual_winner = 1.0
            actual_loser = 0.0

        new_winner_elo = winner.elo + self.k_factor * (actual_winner - expected_winner)
        new_loser_elo = loser.elo + self.k_factor * (actual_loser - expected_loser)

        winner.elo = new_winner_elo
        loser.elo = new_loser_elo

        if not draw:
            winner.wins += 1
            loser.losses += 1

        return new_winner_elo, new_loser_elo


# ---------------------------------------------------------------------------
# Bracket
# ---------------------------------------------------------------------------


class Bracket:
    """Tournament bracket manager.

    Handles bracket creation, match progression, and
    champion determination.
    """

    def __init__(
        self,
        format_type: TournamentFormat = TournamentFormat.SINGLE_ELIMINATION,
    ) -> None:
        self.format_type = format_type
        self._matches: Dict[str, Match] = {}
        self._rounds: Dict[int, List[str]] = {}
        self._competitors: Dict[str, Competitor] = {}
        self._current_round = 0
        self._eliminated: Set[str] = set()

    def add_competitor(self, competitor: Competitor) -> None:
        """Add a competitor to the bracket.

        Args:
            competitor: Competitor to add.
        """
        self._competitors[competitor.competitor_id] = competitor

    def generate_bracket(self) -> None:
        """Generate initial bracket from competitors."""
        if len(self._competitors) < 2:
            raise BracketError("Need at least 2 competitors")

        competitors = sorted(
            self._competitors.values(),
            key=lambda c: c.seed,
        )

        # Pair competitors
        matches = []
        for i in range(0, len(competitors) - 1, 2):
            match = Match(
                match_id=f"R1_M{i//2 + 1}",
                round_number=1,
                competitor_a=competitors[i].competitor_id,
                competitor_b=competitors[i + 1].competitor_id,
            )
            self._matches[match.match_id] = match
            matches.append(match.match_id)

        self._rounds[1] = matches
        self._current_round = 1

    def advance_round(self) -> bool:
        """Advance to next round with winners.

        Returns:
            True if tournament continues.
        """
        current_matches = [self._matches[m] for m in self._rounds.get(self._current_round, [])]

        winners = []
        for match in current_matches:
            if match.winner_id:
                winners.append(match.winner_id)
            else:
                # Auto-resolve if no winner set
                winner = match.determine_winner()
                if winner:
                    match.winner_id = winner
                    match.status = MatchStatus.COMPLETED
                    winners.append(winner)
                else:
                    # Random winner for ties
                    winner = random.choice([match.competitor_a, match.competitor_b])
                    match.winner_id = winner
                    match.status = MatchStatus.COMPLETED
                    winners.append(winner)

        if len(winners) <= 1:
            return False

        # Create next round
        self._current_round += 1
        next_matches = []
        for i in range(0, len(winners) - 1, 2):
            match = Match(
                match_id=f"R{self._current_round}_M{i//2 + 1}",
                round_number=self._current_round,
                competitor_a=winners[i],
                competitor_b=winners[i + 1],
            )
            self._matches[match.match_id] = match
            next_matches.append(match.match_id)

        self._rounds[self._current_round] = next_matches
        return True

    def get_champion(self) -> Optional[Competitor]:
        """Get tournament champion.

        Returns:
            Champion competitor or None.
        """
        current_matches = [self._matches[m] for m in self._rounds.get(self._current_round, [])]
        if len(current_matches) == 1 and current_matches[0].winner_id:
            return self._competitors.get(current_matches[0].winner_id)
        return None

    def get_standings(self) -> List[Competitor]:
        """Get current standings.

        Returns:
            List of competitors sorted by performance.
        """
        return sorted(
            self._competitors.values(),
            key=lambda c: (c.wins, c.elo),
            reverse=True,
        )

    def get_match_history(self, competitor_id: str) -> List[Match]:
        """Get match history for a competitor.

        Args:
            competitor_id: Competitor to query.

        Returns:
            List of matches.
        """
        return [
            m for m in self._matches.values()
            if m.competitor_a == competitor_id or m.competitor_b == competitor_id
        ]


# ---------------------------------------------------------------------------
# TournamentEngine — Main orchestrator
# ---------------------------------------------------------------------------


class TournamentEngine:
    """Tournament Engine for GART v3.0.

    Orchestrates tournament brackets, Elo rankings, and
    match scheduling.

    Attributes:
        bracket: Tournament bracket.
        elo: Elo ranking system.
        format: Tournament format.
    """

    def __init__(
        self,
        format_type: TournamentFormat = TournamentFormat.SINGLE_ELIMINATION,
    ) -> None:
        self.format_type = format_type
        self.bracket = Bracket(format_type)
        self.elo = EloRanking()

    def register_competitor(
        self,
        competitor_id: str,
        name: str,
        seed: int = 0,
        elo_rating: float = 1500.0,
    ) -> Competitor:
        """Register a tournament competitor.

        Args:
            competitor_id: Unique identifier.
            name: Display name.
            seed: Initial seed.
            elo_rating: Initial Elo rating.

        Returns:
            Created competitor.
        """
        competitor = Competitor(
            competitor_id=competitor_id,
            name=name,
            seed=seed,
            elo=elo_rating,
        )
        self.bracket.add_competitor(competitor)
        return competitor

    def start_tournament(self) -> None:
        """Start the tournament."""
        self.bracket.generate_bracket()
        logger.info(
            "Tournament started: %d competitors",
            len(self.bracket._competitors),
        )

    def score_match(
        self,
        match_id: str,
        competitor_scores: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """Score a match.

        Args:
            match_id: Match to score.
            competitor_scores: Dict of competitor_id -> criteria -> score.

        Returns:
            Winner ID or None.
        """
        match = self.bracket._matches.get(match_id)
        if not match:
            return None

        for competitor_id, criteria in competitor_scores.items():
            for criterion, score in criteria.items():
                match.set_score(competitor_id, criterion, score)

        winner_id = match.determine_winner()
        if winner_id:
            match.winner_id = winner_id
            match.status = MatchStatus.COMPLETED

            # Update Elo
            a_id = match.competitor_a
            b_id = match.competitor_b
            comp_a = self.bracket._competitors.get(a_id)
            comp_b = self.bracket._competitors.get(b_id)
            if comp_a and comp_b:
                if winner_id == a_id:
                    self.elo.update_ratings(comp_a, comp_b)
                else:
                    self.elo.update_ratings(comp_b, comp_a)

        return winner_id

    def advance(self) -> bool:
        """Advance tournament to next round.

        Returns:
            True if tournament continues.
        """
        return self.bracket.advance_round()

    def get_champion(self) -> Optional[Competitor]:
        """Get tournament champion.

        Returns:
            Champion or None.
        """
        return self.bracket.get_champion()

    def get_standings(self) -> List[Competitor]:
        """Get current standings.

        Returns:
            Sorted list of competitors.
        """
        return self.bracket.get_standings()

    def get_stats(self) -> Dict[str, Any]:
        """Get tournament statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "format": self.format_type.value,
            "competitors": len(self.bracket._competitors),
            "current_round": self.bracket._current_round,
            "total_matches": len(self.bracket._matches),
            "standings": [
                {
                    "id": c.competitor_id,
                    "name": c.name,
                    "wins": c.wins,
                    "losses": c.losses,
                    "elo": round(c.elo, 1),
                }
                for c in self.bracket.get_standings()
            ],
        }
