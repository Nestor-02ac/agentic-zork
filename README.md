# Agentic Zork

<p align="center">
  <img src="assets/terminal_demo.gif" alt="Qwen 3.5 4B playing Zork I — opening the mailbox, discovering a hidden grating, entering the white house, and collecting items" width="620">
</p>

An LLM-powered agent that plays classic text adventure games (Zork, Lost Pig, Advent, etc.) using the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) and a ReAct reasoning loop.

The agent connects to a game server over MCP's stdio transport, reasons about what to do via chain-of-thought prompting, and sends commands to Z-machine games through the [Jericho](https://github.com/microsoft/jericho) framework. It supports two modes:

- **Cloud**: Qwen2.5-72B-Instruct via the HuggingFace Inference API
- **Local**: Qwen 3.5 4B (or any Ollama model) running entirely on your machine

```
agent.py  ◄── MCP (stdio) ──►  mcp_server.py  ◄── Jericho ──►  .z5 game file
                                      │
                                _va_worker.py  (subprocess)

local_runner/run_local.py  ◄── Ollama API ──►  local LLM
        │
        └── Jericho (direct)  ──►  .z5 game file
```

## Quick Start

```bash
# Clone and enter the repo
git clone <repo-url> && cd agentic-zork

# Create a virtual environment (recommended: Python 3.11+)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Set your HuggingFace token
cp .env.example .env
# Edit .env and add your HF_TOKEN (needs access to Qwen2.5-72B-Instruct)

# Play Lost Pig (default, requires HF_TOKEN)
python run_agent.py -v

# Play Zork 1
python run_agent.py -g zork1 -v

# Evaluate over multiple trials
python evaluation/evaluate.py -s . -g lostpig -t 3 -v
```

### Local Mode (no API key needed)

Run the agent entirely on your machine with [Ollama](https://ollama.com/):

```bash
# Install Ollama and pull a model
ollama pull qwen3.5:4b

# Play Zork 1 locally (50 steps)
python local_runner/run_local.py --game zork1 --max-steps 50

# Play Lost Pig with a different model
python local_runner/run_local.py --game lostpig --model qwen3.5:4b --max-steps 30

# Save a game log as JSON
python local_runner/run_local.py --game zork1 --log assets/my_run.json
```

The local runner is a full port of the cloud agent — it includes valid actions extraction, per-room exploration tracking, loop detection, promising actions, and error recovery.

## Project Structure

```
├── agent.py              # ReAct agent (LLM loop, exploration, loop detection)
├── mcp_server.py         # MCP server exposing game tools
├── _va_worker.py         # Persistent subprocess for valid actions
├── run_agent.py          # CLI to run the agent on any game
├── requirements.txt
├── .env.example
├── games/
│   ├── __init__.py
│   └── zork_env.py       # Jericho wrapper (GameState, env interface)
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py       # Multi-trial evaluation CLI
│   ├── runner.py         # Agent/server process management
│   └── metrics.py        # Score aggregation and statistics
├── local_runner/
│   ├── run_local.py      # Full local agent (Ollama, all features)
│   └── generate_gif.py   # Generate terminal animation for README
├── assets/
│   └── terminal_demo.gif # Animated demo of gameplay
└── z-machine-games-master/
    └── jericho-game-suite/   # Z-machine game files (.z3/.z5/.z8)
```

## How It Works

### ReAct Loop

The agent follows a Reason-Act cycle at every step:

1. The LLM receives a prompt with the current observation, score, recent history, room-specific context, and valid actions.
2. It outputs a structured response: `THOUGHT:` (reasoning), `TOOL:` (which MCP tool), `ARGS:` (parameters).
3. The tool is executed via MCP and the result becomes the next observation.

This repeats for up to `max_steps` turns or until the game ends.

### MCP Server & Tools

The server (`mcp_server.py`) runs as a subprocess communicating via MCP's stdio transport. It exposes five tools:

| Tool | Description |
|---|---|
| `play_action` | Send a command to the game (e.g. `north`, `take lamp`) |
| `memory` | Current location, score, moves, and recent history |
| `get_map` | Explored locations and the connections between them |
| `inventory` | Items the player is carrying |
| `valid_actions` | Actions that are valid in the current game state |

Every `play_action` response is enriched with a `[Location: name|id]` tag from Jericho's `get_player_location()` API, giving the agent a reliable room identifier even when the observation text starts with dialogue or action results.

### Valid Actions Worker

Jericho's `get_valid_actions()` uses spaCy internally and can block for over a minute on some game states. Calling it inside the async MCP server would freeze the event loop.

We solve this with `_va_worker.py`, a persistent subprocess with a line-based protocol:

- On init: `INIT <game_name>` → worker loads the game environment.
- Each call: the server serializes the game state via `save_state()` → base64, sends `VA <state>`.
- The worker restores the state, runs `get_valid_actions()` with a 30-second `signal.alarm` timeout (safe because the worker has no async event loop), and returns results as a `|||`-delimited list.
- The server reads the response with a 35-second `select.select` timeout as a second safety net.

If the worker hangs or crashes, it is killed and a fresh one is spawned automatically on the next call. On any failure, the server falls back to a generic action list so gameplay continues uninterrupted. Results are cached and only refreshed every 3 steps.

> Note: Uses POSIX-specific mechanisms (`signal.alarm`, `select.select`), tested on Linux.

### Loop Detection

Text adventure agents often get stuck repeating the same actions. We implement two detectors:

**Single-action loop**: flags non-directional actions (e.g. `examine`, `talk`) after 2 consecutive repeats, and directional actions (e.g. `north`) after 4 repeats (since walking through rooms back and forth is sometimes legitimate).

**Two-action loop**: detects A-B-A-B patterns (e.g. `take pig` → `look` → `take pig` → `look`) that the single-action detector misses.

When a loop is detected, instead of picking a random action, the agent **re-prompts the LLM** with the banned action(s), valid alternatives, and a "try something different" instruction. Only if the LLM still picks a banned action does the agent force an alternative from the valid actions cache.

### Structured Per-Room Exploration

For each room (identified by Jericho's location ID), the agent tracks:

- **Action log**: every action tried and its short outcome.
- **Promising actions**: extracted by a secondary LLM call on room entry — items to take, containers to open, NPCs to interact with, hidden objects to check.
- **Exits tried**: which directions have been attempted from this room.
- **Steps in room**: after 8 steps in the same room, the prompt warns the agent to move on and can suggest an unexplored exit.

All of this context is injected into the dynamic prompt so the LLM knows what has been tried and what remains.

### Tool Call Validation

The LLM sometimes calls game verbs as MCP tool names (e.g. `TOOL: examine` instead of `TOOL: play_action` with `examine pig`). A validation layer catches this and reconstructs the correct `play_action` call from the verb and its arguments. It also maps invalid game verbs (`check` → `examine`, `grab` → `take`, etc.).

## Evaluation

Run multiple trials to get score statistics:

```bash
# 3 trials on Lost Pig
python evaluation/evaluate.py -s . -g lostpig -t 3

# 5 trials on Zork 1 with verbose output, save results
python evaluation/evaluate.py -s . -g zork1 -t 5 -v -o results.json

# List all available games
python evaluation/evaluate.py --list-games
```

The evaluation framework runs each trial with a different seed for reproducibility and reports mean/std/min/max scores.

### Results

| Game | Typical Score | Notes |
|---|---|---|
| Lost Pig | ~2 pts | Finds pig + coin, visits 50+ locations |
| Zork 1 | 10–40 pts | Varies with early lamp discovery, 50+ locations |

## Known Limitations

- **LLM repetition**: the model sometimes fixates on similar interactions with slightly different phrasing, which the loop detectors can't always catch.
- **Step efficiency**: some steps are spent on redundant examinations or invalid commands.
- **POSIX-only**: the valid actions worker uses `signal.alarm` and `select.select`, so it only works on Linux/macOS.

## Dependencies

- [Jericho](https://github.com/microsoft/jericho) — Z-machine game interface
- [FastMCP](https://github.com/jlowin/fastmcp) — MCP server/client framework
- [spaCy](https://spacy.io/) + `en_core_web_sm` — used by Jericho for valid action extraction
- [HuggingFace Inference API](https://huggingface.co/docs/huggingface_hub/guides/inference) — cloud LLM access (Qwen2.5-72B-Instruct)
- [Ollama](https://ollama.com/) — local LLM inference (Qwen 3.5 4B or similar)

## License

MIT
