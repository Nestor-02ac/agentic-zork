#!/usr/bin/env python3
"""Evaluation script for text adventure agents.

Usage:
    python evaluation/evaluate.py -s . -g zork1 -t 3
    python evaluation/evaluate.py -s . -g lostpig -t 5 --max-steps 150 -v
"""

import argparse
import asyncio
import json
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import EvaluationResult, TrialResult
from evaluation.runner import RunConfig, run_agent_with_server
from games.zork_env import list_available_games


def generate_seeds(base_seed: int, num_trials: int) -> list[int]:
    random.seed(base_seed)
    return [random.randint(0, 2**32 - 1) for _ in range(num_trials)]


def resolve_submission_files(submission_path: Path) -> tuple[Path, Path]:
    src_dir = submission_path / "src"
    agent_path = src_dir / "agent.py"
    server_path = src_dir / "mcp_server.py"
    if agent_path.exists() and server_path.exists():
        return agent_path, server_path
    return submission_path / "agent.py", submission_path / "mcp_server.py"


async def evaluate_submission(
    submission_path: Path,
    game: str,
    num_trials: int = 5,
    max_steps: int = 100,
    base_seed: int = 42,
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate an agent across multiple trials."""
    agent_path, server_path = resolve_submission_files(submission_path)

    student_id = submission_path.name
    readme_path = submission_path / "README.md"
    if readme_path.exists():
        content = readme_path.read_text()
        for line in content.split("\n"):
            if line.startswith("# ") or "name:" in line.lower():
                student_id = line.replace("#", "").replace("name:", "").strip()[:50]
                break

    result = EvaluationResult(
        student_id=student_id,
        game=game,
        num_trials=num_trials,
        max_steps=max_steps,
    )

    seeds = generate_seeds(base_seed, num_trials)

    print(f"\nEvaluating: {student_id}")
    print(f"Game: {game}")
    print(f"Trials: {num_trials}")
    print(f"Max steps: {max_steps}")
    print(f"Seeds: {seeds}")
    print("-" * 50)

    for i, seed in enumerate(seeds):
        trial_num = i + 1
        print(f"\nTrial {trial_num}/{num_trials} (seed={seed})...")

        config = RunConfig(
            agent_path=agent_path,
            server_path=server_path,
            game=game,
            max_steps=max_steps,
            seed=seed,
            verbose=verbose,
        )

        try:
            run_result = await run_agent_with_server(config)

            trial = TrialResult(
                trial_number=trial_num,
                final_score=run_result.final_score,
                max_score=run_result.max_score,
                moves=run_result.moves,
                locations_visited=len(run_result.locations_visited),
                game_completed=run_result.game_completed,
                error=run_result.error,
            )

            if run_result.error:
                print(f"  Error: {run_result.error[:100]}...")
            else:
                print(f"  Score: {run_result.final_score}")
                print(f"  Moves: {run_result.moves}")
                print(f"  Locations: {len(run_result.locations_visited)}")

        except Exception as e:
            trial = TrialResult(
                trial_number=trial_num,
                final_score=0,
                max_score=0,
                moves=0,
                locations_visited=0,
                game_completed=False,
                error=str(e),
            )
            print(f"  Exception: {e}")

        result.add_trial(trial)

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate text adventure agent submissions")

    parser.add_argument("-s", "--submission", type=Path, required=True, help="Path to submission directory")
    parser.add_argument("-g", "--game", type=str, default="lostpig", help="Game to evaluate on (default: lostpig)")
    parser.add_argument("-t", "--trials", type=int, default=5, help="Number of trials (default: 5)")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per trial (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("-o", "--output", type=Path, help="Output file for results (JSON)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed output")
    parser.add_argument("--list-games", action="store_true", help="List available games and exit")

    args = parser.parse_args()

    if args.list_games:
        games = list_available_games()
        print(f"Available games ({len(games)}):")
        for game in games:
            print(f"  - {game}")
        return

    available_games = list_available_games()
    if args.game not in available_games:
        print(f"Error: Unknown game '{args.game}'")
        print(f"Available: {', '.join(available_games[:10])}...")
        sys.exit(1)

    submission_path = args.submission
    if not submission_path.exists():
        print(f"Error: Submission path not found: {submission_path}")
        sys.exit(1)

    result = asyncio.run(
        evaluate_submission(
            submission_path=submission_path,
            game=args.game,
            num_trials=args.trials,
            max_steps=args.max_steps,
            base_seed=args.seed,
            verbose=args.verbose,
        )
    )

    print("\n" + result.summary_str())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
