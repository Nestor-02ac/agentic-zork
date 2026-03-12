"""Agent runner for evaluation.

Handles spawning the MCP server subprocess, connecting the agent, and collecting results.
"""

import asyncio
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from games.zork_env import list_available_games


@dataclass
class RunConfig:
    agent_path: Path
    server_path: Path
    game: str
    max_steps: int
    seed: int
    verbose: bool = False


@dataclass
class RunResult:
    final_score: int
    max_score: int
    moves: int
    locations_visited: set[str]
    game_completed: bool
    error: Optional[str] = None
    history: list[tuple[str, str, str]] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []


def load_agent_class(agent_path: Path):
    """Dynamically load the StudentAgent class from the configured agent file."""
    spec = importlib.util.spec_from_file_location("student_agent", agent_path)
    module = importlib.util.module_from_spec(spec)

    submission_dir = agent_path.parent
    submission_root = submission_dir.parent if submission_dir.name == "src" else submission_dir

    for path in (submission_root, submission_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    spec.loader.exec_module(module)

    if not hasattr(module, "StudentAgent"):
        raise ValueError(f"Agent file {agent_path} must define a 'StudentAgent' class")

    return module.StudentAgent


async def run_agent_with_server(config: RunConfig) -> RunResult:
    """Run the agent with its MCP server."""
    if not config.agent_path.exists():
        return RunResult(
            final_score=0, max_score=0, moves=0, locations_visited=set(),
            game_completed=False, error=f"Agent file not found: {config.agent_path}"
        )

    if not config.server_path.exists():
        return RunResult(
            final_score=0, max_score=0, moves=0, locations_visited=set(),
            game_completed=False, error=f"Server file not found: {config.server_path}"
        )

    available_games = list_available_games()
    if config.game not in available_games:
        return RunResult(
            final_score=0, max_score=0, moves=0, locations_visited=set(),
            game_completed=False, error=f"Unknown game: {config.game}. Available: {available_games[:10]}..."
        )

    try:
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport

        AgentClass = load_agent_class(config.agent_path)
        agent = AgentClass()

        env = os.environ.copy()
        env["GAME"] = config.game

        transport = StdioTransport(
            command=sys.executable,
            args=[str(config.server_path)],
            env=env,
        )

        async with Client(transport) as client:
            result = await agent.run(
                client=client,
                game=config.game,
                max_steps=config.max_steps,
                seed=config.seed,
                verbose=config.verbose,
            )
            return result

    except Exception as e:
        import traceback
        return RunResult(
            final_score=0, max_score=0, moves=0, locations_visited=set(),
            game_completed=False, error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )


def run_single_trial(config: RunConfig) -> RunResult:
    """Synchronous wrapper for running a single trial."""
    return asyncio.run(run_agent_with_server(config))
