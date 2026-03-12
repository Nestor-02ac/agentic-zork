#!/usr/bin/env python3
"""Run the MCP ReAct agent on a text adventure game.

Usage:
    python run_agent.py
    python run_agent.py --game advent
    python run_agent.py --max-steps 150 -v
    python run_agent.py --list-games
"""

import argparse
import sys
import os
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from games.zork_env import list_available_games


async def run_mcp_agent(args):
    project_dir = Path(__file__).parent
    agent_file = project_dir / "agent.py"
    server_file = project_dir / "mcp_server.py"

    sys.path.insert(0, str(project_dir))
    from agent import StudentAgent
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    print(f"\nRunning agent on: {args.game}")

    agent = StudentAgent()

    env_vars = os.environ.copy()
    env_vars["GAME"] = args.game

    transport = StdioTransport(
        command=sys.executable,
        args=[str(server_file)],
        env=env_vars,
    )

    async with Client(transport) as client:
        return await agent.run(
            client=client,
            game=args.game,
            max_steps=args.max_steps,
            seed=42,
            verbose=args.verbose,
        )


def main():
    parser = argparse.ArgumentParser(description="Run the MCP ReAct agent on text adventure games")

    available_games = list_available_games()

    parser.add_argument("-g", "--game", type=str, default="lostpig",
                        help=f"Game to play (default: lostpig). {len(available_games)} games available.")
    parser.add_argument("-n", "--max-steps", type=int, default=100,
                        help="Maximum number of steps (default: 100)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed reasoning from the agent")
    parser.add_argument("--list-games", action="store_true",
                        help="List all available games and exit")

    args = parser.parse_args()

    if args.list_games:
        print(f"\nAvailable games ({len(available_games)} total):\n")
        cols = 5
        for i in range(0, len(available_games), cols):
            row = available_games[i:i+cols]
            print("  " + "  ".join(f"{g:<15}" for g in row))
        print()
        sys.exit(0)

    if args.game.lower() not in available_games:
        print(f"\nError: Unknown game '{args.game}'")
        print(f"Use --list-games to see {len(available_games)} available options.")
        sys.exit(1)

    print(f"Game: {args.game}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Verbose: {args.verbose}")

    try:
        results = asyncio.run(run_mcp_agent(args))
    except ValueError as e:
        print(f"\n[Error] {e}")
        print("\nMake sure HF_TOKEN is set in your .env file.")
        sys.exit(1)

    return results


if __name__ == "__main__":
    main()
