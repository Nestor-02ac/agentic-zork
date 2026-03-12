"""Text adventure game environment wrapper around Jericho."""

from jericho import FrotzEnv
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import os


@dataclass
class GameState:
    """Represents the current state of the game."""
    observation: str
    score: int
    max_score: int
    moves: int
    done: bool
    reward: int
    inventory: list[str]
    location: str


def get_default_games_dir() -> Path:
    project_root = Path(__file__).parent.parent
    return project_root / "z-machine-games-master" / "jericho-game-suite"


def discover_games(games_dir: Optional[Path] = None) -> dict[str, Path]:
    """Discover all available Z-machine games in the games directory."""
    if games_dir is None:
        games_dir = get_default_games_dir()

    games_dir = Path(games_dir)
    if not games_dir.exists():
        return {}

    games = {}
    for ext in ["*.z3", "*.z4", "*.z5", "*.z8"]:
        for game_path in games_dir.glob(ext):
            game_name = game_path.stem.lower()
            games[game_name] = game_path

    return dict(sorted(games.items()))


def list_available_games(games_dir: Optional[Path] = None) -> list[str]:
    return list(discover_games(games_dir).keys())


class TextAdventureEnv:
    """Wrapper around Jericho's FrotzEnv for text adventure games."""

    def __init__(self, game: str = "zork1", games_dir: Optional[str] = None):
        if os.path.isfile(game):
            game_path = Path(game)
            self.game = game_path.stem
        else:
            games_path = Path(games_dir) if games_dir else None
            available_games = discover_games(games_path)

            if game.lower() not in available_games:
                available = list(available_games.keys())[:20]
                raise ValueError(
                    f"Unknown game: {game}. "
                    f"Available: {', '.join(available)}... "
                    f"({len(available_games)} total)"
                )

            game_path = available_games[game.lower()]
            self.game = game.lower()

        self.env = FrotzEnv(str(game_path))
        self.game_path = game_path
        self._last_score = 0
        self._history: list[tuple[str, str]] = []

    def reset(self) -> GameState:
        observation, info = self.env.reset()
        self._last_score = 0
        self._history = []
        return self._make_game_state(observation, info, done=False, reward=0)

    def step(self, action: str) -> GameState:
        observation, reward, done, info = self.env.step(action)

        current_score = info.get('score', 0)
        reward = current_score - self._last_score
        self._last_score = current_score

        self._history.append((action, observation))

        return self._make_game_state(observation, info, done, reward)

    def _make_game_state(self, observation: str, info: dict, done: bool, reward: int) -> GameState:
        try:
            inventory = [str(obj) for obj in self.env.get_inventory()]
        except Exception:
            inventory = []

        try:
            location = str(self.env.get_player_location())
        except Exception:
            location = "Unknown"

        return GameState(
            observation=observation,
            score=info.get('score', 0),
            max_score=self.env.get_max_score(),
            moves=info.get('moves', 0),
            done=done,
            reward=reward,
            inventory=inventory,
            location=location,
        )

    def get_history(self) -> list[tuple[str, str]]:
        return self._history.copy()

    def get_valid_actions(self) -> list[str]:
        try:
            return self.env.get_valid_actions()
        except Exception:
            return [
                "north", "south", "east", "west",
                "up", "down", "look", "inventory",
                "take all", "open mailbox", "read"
            ]

    def save_state(self):
        return self.env.get_state()

    def load_state(self, state):
        self.env.set_state(state)

    def get_walkthrough(self) -> list[str]:
        return self.env.get_walkthrough()

    def close(self):
        self.env.close()


ZorkEnvironment = TextAdventureEnv
