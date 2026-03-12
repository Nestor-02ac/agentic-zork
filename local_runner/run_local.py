#!/usr/bin/env python3
"""
Local model runner for Agentic Zork — Full implementation.
Port of agent.py that uses Ollama for inference and Jericho directly.
Designed for laptop hardware with small models (3B-9B).

Usage:
    python local_runner/run_local.py -g lostpig -n 15 -v
    python local_runner/run_local.py -g zork1 -m qwen3.5:4b -n 50 -o assets/game_log.json
"""

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from games.zork_env import TextAdventureEnv, list_available_games

OLLAMA_URL = "http://localhost:11434/api/chat"

# System prompt
SYSTEM_PROMPT = """You are an expert text adventure game player. Your goal is to explore, collect treasures, and maximize your score.

At each turn you receive the game observation, score, history, and valid actions.
Reply with EXACTLY this format (no markdown, no extra text):

THOUGHT: <brief reasoning about what to do next>
ACTION: <single game command>

VALID GAME COMMANDS:
- Movement: north, south, east, west, up, down, enter, exit, northeast, northwest, southeast, southwest
- Objects: take <item>, drop <item>, open <thing>, close <thing>, examine <thing>
- Light: turn on lamp, turn off lamp
- Combat: attack <enemy> with <weapon>
- Other: inventory, look, read <thing>, wait, look under <thing>, look in <thing>
- Interaction: ask <character> about <topic>, tell <character> about <topic>, give <item> to <character>
FORBIDDEN (will NOT work): check, inspect, search, grab, use, help

EXPLORATION STRATEGY:
- When you arrive at a NEW location, interact with objects there BEFORE moving on.
- PREFER actions from the "Valid actions" list — they are guaranteed to work.
- Open containers. Examine interesting things. Pick up useful items.
- Prioritize PROMISING ACTIONS shown in the prompt — these are high-value.
- Once you've tried promising actions, move to an UNEXPLORED exit.
- Try look in/under things (objects can be hidden).
- Give items to NPCs — that often scores points.
- Do NOT stay in the same room too long if nothing new is happening.
- Do NOT repeat the same action. Prefer untried actions.
- Do NOT examine the same object more than once — check history."""

PROMISING_ACTIONS_PROMPT = """Given this text adventure game observation, list the most promising actions the player should try. Focus on:
- Items mentioned that can be taken ("take X")
- Containers or objects that can be opened/examined ("open X", "examine X")
- Creatures or NPCs to interact with ("talk to X", "ask X about Y")
- Puzzles or mechanisms hinted at
- Hidden things suggested ("look under X", "look in X")

Observation:
{observation}

List ONLY the action commands, one per line. No explanations."""


@dataclass
class LogEntry:
    step: int
    thought: str
    action: str
    observation: str
    score: int
    reward: int
    location: str


# LLM calls
def call_ollama(model: str, messages: list[dict], max_tokens: int = 250) -> str:
    """Call ollama chat API with thinking disabled."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        # Strip any residual thinking tags
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except requests.RequestException as e:
        print(f"  [Ollama error: {e}]")
        return "THOUGHT: Cannot reach model, trying basic exploration.\nACTION: look"


def call_ollama_simple(model: str, system: str, prompt: str, max_tokens: int = 150) -> str:
    """Single-shot LLM call (for promising actions extraction)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return call_ollama(model, messages, max_tokens)


# Valid actions with timeout
def get_valid_actions_safe(env: TextAdventureEnv, timeout_sec: int = 15) -> list[str]:
    """Get valid actions from Jericho with a timeout."""
    FALLBACK = ["north", "south", "east", "west", "up", "down",
                "look", "inventory", "take all", "examine"]

    def _alarm_handler(signum, frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_sec)
    try:
        actions = env.get_valid_actions()
        signal.alarm(0)
        return actions if actions else FALLBACK
    except Exception:
        signal.alarm(0)
        return FALLBACK
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# Response parsing
INVALID_VERB_MAP = {
    "check": "examine", "inspect": "examine", "search": "look",
    "grab": "take", "pick": "take", "use": "examine", "investigate": "examine",
}


def parse_response(text: str) -> tuple[str, str]:
    """Extract THOUGHT and ACTION from LLM response."""
    thought = ""
    action = "look"

    thought_match = re.search(r"THOUGHT:\s*(.+?)(?:\n|ACTION:)", text, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if action_match:
        action = action_match.group(1).strip()

    # Clean action
    action = action.lower().strip()
    action = re.sub(r"[`*_\"]", "", action)
    action = action.split("\n")[0].strip()
    action = " ".join(action.split())

    # Fix invalid verbs
    words = action.split()
    if words and words[0] in INVALID_VERB_MAP:
        words[0] = INVALID_VERB_MAP[words[0]]
        action = " ".join(words)

    return thought, action


class LocalAgent:
    def __init__(self, model: str):
        self.model = model
        self.recent_actions: list[str] = []
        self.score: int = 0
        self.valid_actions_cache: list[str] = []
        self.steps_since_va: int = 999

        # Per-room exploration
        self.current_location: str = "Unknown"
        self.room_action_log: dict[str, list[tuple[str, str]]] = {}
        self.room_promising: dict[str, list[str]] = {}
        self.steps_in_room: int = 0
        self.room_exits_tried: dict[str, set[str]] = {}
        self.max_steps_per_room: int = 8

        # Conversation history for the LLM
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.action_history: list[dict] = []

    # Location tracking

    def update_location(self, env: TextAdventureEnv, action: str, observation: str, verbose: bool):
        """Track location changes and per-room state."""
        try:
            loc = env.env.get_player_location()
            loc_id = f"{loc.name}|{loc.num}" if loc else "Unknown|0"
            loc_name = loc.name if loc else "Unknown"
        except Exception:
            loc_id = observation.split("\n")[0][:40]
            loc_name = loc_id

        is_new = loc_id != self.current_location

        # Track exits
        if self._is_directional(action) and self.current_location:
            if self.current_location not in self.room_exits_tried:
                self.room_exits_tried[self.current_location] = set()
            self.room_exits_tried[self.current_location].add(action)

        if is_new:
            if verbose:
                print(f"  [LOCATION] {self.current_location.split('|')[0]} -> {loc_name}")
            self.current_location = loc_id
            self.steps_in_room = 0
            # Extract promising actions for new room
            self._extract_promising(observation)
        else:
            self.steps_in_room += 1

        # Log action in room
        if self.current_location not in self.room_action_log:
            self.room_action_log[self.current_location] = []
        self.room_action_log[self.current_location].append((action, observation[:80]))

    def _extract_promising(self, observation: str):
        """Use LLM to find promising actions for current room."""
        clean = re.sub(r'\[.*?\]', '', observation).strip()
        if len(clean) < 20:
            return
        prompt = PROMISING_ACTIONS_PROMPT.format(observation=clean)
        try:
            response = call_ollama_simple(
                self.model,
                "You extract game actions from text. Return ONLY action commands, one per line.",
                prompt, max_tokens=100
            )
            actions = []
            for line in response.strip().split('\n'):
                line = line.strip()
                line = re.sub(r'^\d+\.?\s*', '', line).strip()
                line = line.strip('"').strip("'").strip('`').strip('-').strip().lower()
                if line and len(line) < 50 and not line.startswith('#'):
                    actions.append(line)
            self.room_promising[self.current_location] = actions[:6]
        except Exception:
            pass

    # Loop detection

    def _is_directional(self, action: str) -> bool:
        return action.lower().strip() in {
            "north", "south", "east", "west", "up", "down",
            "n", "s", "e", "w", "u", "d",
            "ne", "nw", "se", "sw", "northeast", "northwest", "southeast", "southwest",
            "enter", "exit", "go north", "go south", "go east", "go west",
        }

    def detect_loop(self) -> tuple[str, list[str]] | None:
        ra = self.recent_actions
        if len(ra) >= 2 and ra[-1] == ra[-2]:
            if self._is_directional(ra[-1]):
                if len(ra) >= 4 and all(a == ra[-1] for a in ra[-4:]):
                    return ("single-action", [ra[-1]])
            else:
                return ("single-action", [ra[-1]])
        if len(ra) >= 4:
            a, b, c, d = ra[-4], ra[-3], ra[-2], ra[-1]
            if a == c and b == d and a != b:
                return ("two-action", [a, b])
        return None

    def break_loop(self, loop_type: str, looped: list[str], observation: str) -> tuple[str, str]:
        """Re-prompt LLM to break out of a loop."""
        banned = set(a.lower() for a in looped)

        if loop_type == "single-action":
            warn = f"You have been repeating '{looped[0]}'. Do NOT use it again."
        else:
            warn = f"You have been alternating '{looped[0]}' and '{looped[1]}'. Do NOT use either."

        alternatives = [a for a in self.valid_actions_cache if a.lower() not in banned]
        alt_str = f"\nValid alternative actions: {', '.join(alternatives[:15])}" if alternatives else ""

        reprompt = f"{warn}{alt_str}\n\nCurrent situation:\n{observation}\n\nWhat completely different action will you try?"
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": reprompt},
        ]
        response = call_ollama(self.model, msgs, 200)
        thought, action = parse_response(response)

        if action.lower() in banned:
            if alternatives:
                action = alternatives[0]
            else:
                for fb in ["inventory", "look", "north", "south", "east", "west"]:
                    if fb not in banned:
                        action = fb
                        break
            thought = f"[Loop forced to: {action}]"

        return thought, action

    # Prompt building

    def build_prompt(self, observation: str, score: int, moves: int) -> str:
        parts = []

        loc_name = self.current_location.split("|")[0] if "|" in self.current_location else self.current_location
        parts.append(f"Score: {score} | Move: {moves} | Room: {loc_name} (step {self.steps_in_room} here)")

        # Recent history
        if self.action_history:
            parts.append("\nRecent actions:")
            for entry in self.action_history[-8:]:
                res_short = entry["obs"][:80] + "..." if len(entry["obs"]) > 80 else entry["obs"]
                parts.append(f"  > {entry['action']} -> {res_short}")

        # Loop warning
        loop = self.detect_loop()
        if loop:
            lt, la = loop
            if lt == "single-action":
                parts.append(f"\nWARNING: You've been repeating '{la[0]}'. TRY SOMETHING DIFFERENT!")
            else:
                parts.append(f"\nWARNING: You've been alternating '{la[0]}' and '{la[1]}'. BREAK THE PATTERN!")

        parts.append(f"\nCurrent observation:\n{observation}")

        # Actions tried in this room
        if self.current_location in self.room_action_log:
            tried = self.room_action_log[self.current_location]
            if tried:
                parts.append("\nAlready tried in this room:")
                for a, o in tried[-6:]:
                    parts.append(f"  - {a} -> {o[:40]}")

        # Promising untried actions
        if self.current_location in self.room_promising:
            tried_actions = {a for a, _ in self.room_action_log.get(self.current_location, [])}
            promising = [a for a in self.room_promising[self.current_location] if a not in tried_actions]
            if promising:
                parts.append(f"\nPROMISING (untried) actions for this room: {', '.join(promising)}")
                parts.append(">> Try one of these FIRST! <<")

        # Valid actions
        if self.valid_actions_cache:
            parts.append(f"\nValid actions: {', '.join(self.valid_actions_cache[:20])}")

        # Exploration warnings
        if self.steps_in_room >= self.max_steps_per_room - 2:
            parts.append(f"\nYou've been in {loc_name} for {self.steps_in_room} steps. Move to a new room!")

        if self.current_location in self.room_exits_tried:
            tried_exits = self.room_exits_tried[self.current_location]
            if tried_exits:
                parts.append(f"Exits already tried: {', '.join(sorted(tried_exits))}")

        parts.append("\nWhat do you do next?")
        return "\n".join(parts)


def run_game(game: str, model: str, max_steps: int, verbose: bool) -> list[LogEntry]:
    env = TextAdventureEnv(game)
    state = env.reset()
    agent = LocalAgent(model)
    log: list[LogEntry] = []

    print(f"\n{'='*60}")
    print(f"  AGENTIC ZORK - {game.upper()}")
    print(f"  Model: {model} (local via Ollama)")
    print(f"  Max steps: {max_steps}")
    print(f"{'='*60}\n")
    print(f"  {state.observation}\n")

    # Initialize location
    agent.update_location(env, "look", state.observation, verbose)

    for step in range(1, max_steps + 1):
        # Fetch valid actions every 3 steps
        if agent.steps_since_va >= 3:
            va = get_valid_actions_safe(env, timeout_sec=15)
            if va:
                agent.valid_actions_cache = va
                if verbose:
                    va_show = va[:12]
                    print(f"  [VALID ACTIONS] {', '.join(va_show)}{'...' if len(va) > 12 else ''}")
            agent.steps_since_va = 0

        # Build prompt and call LLM
        prompt = agent.build_prompt(state.observation, state.score, state.moves)
        agent.messages.append({"role": "user", "content": prompt})

        # Keep conversation manageable
        if len(agent.messages) > 25:
            agent.messages = [agent.messages[0]] + agent.messages[-24:]

        response = call_ollama(model, agent.messages)
        thought, action = parse_response(response)

        agent.messages.append({"role": "assistant", "content": response})

        # Track recent actions
        agent.recent_actions.append(action)
        if len(agent.recent_actions) > 8:
            agent.recent_actions = agent.recent_actions[-8:]

        # Loop detection & breaking
        loop = agent.detect_loop()
        if loop:
            loop_type, looped = loop
            if verbose:
                print(f"  [LOOP] {loop_type}: {looped} - re-prompting")
            thought, action = agent.break_loop(loop_type, looped, state.observation)
            agent.recent_actions[-1] = action

        agent.steps_since_va += 1

        # Execute action
        state = env.step(action)
        agent.update_location(env, action, state.observation, verbose)
        agent.score = max(agent.score, state.score)

        # Track history
        agent.action_history.append({"action": action, "obs": state.observation})
        if len(agent.action_history) > 10:
            agent.action_history = agent.action_history[-10:]

        # Get location name
        location = ""
        try:
            loc = env.env.get_player_location()
            location = loc.name if loc else ""
        except Exception:
            pass

        entry = LogEntry(
            step=step, thought=thought, action=action,
            observation=state.observation, score=state.score,
            reward=state.reward, location=location,
        )
        log.append(entry)

        if verbose:
            print(f"  [{step:2d}] T: {thought}")
            print(f"       > {action}")
            obs_lines = state.observation.strip().split('\n')
            for line in obs_lines[:4]:
                print(f"       {line}")
            if len(obs_lines) > 4:
                print(f"       ...")
            print(f"       [Score: {state.score} | Reward: {state.reward:+d}]\n")
        else:
            score_ind = f" (+{state.reward})" if state.reward > 0 else ""
            print(f"  [{step:3d}] > {action:<30s} Score: {state.score}{score_ind}")

        if state.done:
            print(f"\n  *** GAME OVER at step {step} - Final Score: {state.score} ***")
            break

    env.close()
    return log


def save_log(log: list[LogEntry], output_path: str):
    data = [
        {
            "step": e.step,
            "thought": e.thought,
            "action": e.action,
            "observation": e.observation,
            "score": e.score,
            "reward": e.reward,
            "location": e.location,
        }
        for e in log
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Log saved to {output_path} ({len(data)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Run Agentic Zork with a local model via Ollama")
    available = list_available_games()
    parser.add_argument("-g", "--game", default="zork1", help="Game to play (default: zork1)")
    parser.add_argument("-m", "--model", default="qwen3.5:4b", help="Ollama model (default: qwen3.5:4b)")
    parser.add_argument("-n", "--steps", type=int, default=50, help="Max steps (default: 50)")
    parser.add_argument("-o", "--output", default="assets/game_log.json", help="Output log file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed reasoning")
    parser.add_argument("--list-games", action="store_true", help="List available games")
    args = parser.parse_args()

    if args.list_games:
        print(f"\nAvailable games ({len(available)}):")
        for g in available:
            print(f"  {g}")
        return

    if args.game.lower() not in available:
        print(f"Unknown game: {args.game}. Use --list-games.")
        sys.exit(1)

    log = run_game(args.game, args.model, args.steps, args.verbose)
    save_log(log, args.output)

    final = log[-1] if log else None
    print(f"\n  Summary: {len(log)} steps, Final Score: {final.score if final else 0}")


if __name__ == "__main__":
    main()
