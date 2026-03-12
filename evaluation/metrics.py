"""Evaluation metrics for text adventure agents."""

import statistics
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrialResult:
    """Result of a single evaluation trial."""
    trial_number: int
    final_score: int
    max_score: int
    moves: int
    locations_visited: int
    game_completed: bool
    error: Optional[str] = None

    @property
    def score_percentage(self) -> float:
        if self.max_score == 0:
            return 0.0
        return (self.final_score / self.max_score) * 100

    def to_dict(self) -> dict:
        return {
            "trial_number": self.trial_number,
            "final_score": self.final_score,
            "max_score": self.max_score,
            "score_percentage": round(self.score_percentage, 2),
            "moves": self.moves,
            "locations_visited": self.locations_visited,
            "game_completed": self.game_completed,
            "error": self.error,
        }


@dataclass
class EvaluationResult:
    """Aggregated results across all trials."""
    student_id: str
    game: str
    num_trials: int
    max_steps: int
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def scores(self) -> list[int]:
        return [t.final_score for t in self.trials if t.error is None]

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return statistics.mean(self.scores)

    @property
    def std_score(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        return statistics.stdev(self.scores)

    @property
    def min_score(self) -> int:
        if not self.scores:
            return 0
        return min(self.scores)

    @property
    def max_score_achieved(self) -> int:
        if not self.scores:
            return 0
        return max(self.scores)

    @property
    def successful_trials(self) -> int:
        return len([t for t in self.trials if t.error is None])

    @property
    def mean_moves(self) -> float:
        moves = [t.moves for t in self.trials if t.error is None]
        if not moves:
            return 0.0
        return statistics.mean(moves)

    @property
    def mean_locations(self) -> float:
        locs = [t.locations_visited for t in self.trials if t.error is None]
        if not locs:
            return 0.0
        return statistics.mean(locs)

    def add_trial(self, trial: TrialResult) -> None:
        self.trials.append(trial)

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "game": self.game,
            "num_trials": self.num_trials,
            "max_steps": self.max_steps,
            "successful_trials": self.successful_trials,
            "summary": {
                "mean_score": round(self.mean_score, 2),
                "std_score": round(self.std_score, 2),
                "min_score": self.min_score,
                "max_score": self.max_score_achieved,
                "mean_moves": round(self.mean_moves, 2),
                "mean_locations": round(self.mean_locations, 2),
            },
            "trials": [t.to_dict() for t in self.trials],
        }

    def summary_str(self) -> str:
        lines = [
            f"Evaluation Results: {self.student_id}",
            f"{'=' * 50}",
            f"Game: {self.game}",
            f"Trials: {self.successful_trials}/{self.num_trials} successful",
            f"Max steps per trial: {self.max_steps}",
            f"",
            f"Score Statistics:",
            f"  Mean:  {self.mean_score:.2f}",
            f"  Std:   {self.std_score:.2f}",
            f"  Min:   {self.min_score}",
            f"  Max:   {self.max_score_achieved}",
            f"",
            f"Exploration:",
            f"  Mean moves:     {self.mean_moves:.1f}",
            f"  Mean locations: {self.mean_locations:.1f}",
            f"",
            f"Per-Trial Scores: {self.scores}",
        ]
        return "\n".join(lines)
