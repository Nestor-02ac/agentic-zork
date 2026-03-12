"""MCP ReAct Agent for text adventure games."""

import json
import os
import re
import asyncio
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# LLM Configuration
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

_hf_token = os.getenv("HF_TOKEN")
if not _hf_token:
    raise ValueError("HF_TOKEN not found. Set it in your .env file.")

LLM_CLIENT = InferenceClient(token=_hf_token)


def call_llm(prompt: str, system_prompt: str, seed: int, max_tokens: int = 300) -> str:
    """Call the LLM with the given prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
    )
    return response.choices[0].message.content


@dataclass
class RunResult:
    """Result of running the agent."""
    final_score: int
    max_score: int
    moves: int
    locations_visited: set[str]
    game_completed: bool
    error: Optional[str] = None
    history: list[tuple[str, str, str]] = field(default_factory=list)


# System Prompt
SYSTEM_PROMPT = """You are an expert text adventure game player. Your goal is to explore, collect treasures, and maximize your score.

AVAILABLE TOOLS (use these via MCP):
1. play_action - Execute game commands (north, take lamp, open mailbox, etc.)
2. memory - Get current game state, score, and recent history
3. get_map - See explored locations and connections
4. inventory - Check what you're carrying
5. valid_actions - Get valid actions for the current state

VALID GAME COMMANDS for play_action:
- Movement: north, south, east, west, up, down, enter, exit, northeast, northwest, southeast, southwest
- Objects: take <item>, drop <item>, open <thing>, close <thing>, examine <thing>
- Light: turn on lamp, turn off lamp
- Combat: attack <enemy> with <weapon>
- Other: inventory, look, read <thing>, wait, look under <thing>, look in <thing>
FORBIDDEN (will NOT work): check, inspect, search, grab, use, help

RESPOND IN THIS EXACT FORMAT (no markdown):
THOUGHT: <brief reasoning about what to do next>
TOOL: <tool_name>
ARGS: <JSON arguments>

EXPLORATION STRATEGY:
- When you arrive at a NEW location, interact with objects there BEFORE moving on.
- PREFER actions from the valid_actions list (a set of heuristics on the game message but garanteed to work) in odd situations. 
- Open containers. Examine interesting things BUT before check that you haven't already examined something by looking at the history.
- ALWAYS Pick up useful items (lamp, sword, etc.)
- Prioritize PROMISING ACTIONS shown in the prompt — these are high-value.
- Once you have tried the promising actions and explored the room, move to an UNEXPLORED exit.
- Track which exits you have already taken from each room.
- Use Look in/under <things> (objects can be hidden).
- Do NOT stay in the same room for too many turns if nothing new is happening.
- As an exception, you can stay in the same room for more turns if you encounter a new character and you are advance in the game (more than 30 steps made already, never at the beginning), since that is always important.
- Consider that you MUST also try to give the things you have taken to characters, that's why is so important to explore and take (and of course to give), if a trial to take something doesn't work DON'T TRY TO GET IT ANYMORE.
- DO NOT only interact with the same character you encounter at the beginning, and if someone respond with nice logical and more long responses keep thinking of him.
- If an observation mentions an item, creature, or interactive object, interact with it!
- Do NOT examine the same objects more than once, check recent history.

Do NOT repeat the same action. Prefer untried actions over tried ones.

IMPORTANT — HOW TO USE play_action:
All game commands go through play_action. Do NOT use game verbs as tool names.

CORRECT:
  TOOL: play_action
  ARGS: {"action": "examine pig"}

  TOOL: play_action
  ARGS: {"action": "take torch"}

  TOOL: play_action
  ARGS: {"action": "ask pig about water"}

  TOOL: play_action
  ARGS: {"action": "tell pig about farm"}

WRONG (these will fail or waste a turn):
  TOOL: examine
  ARGS: {"thing": "pig"}

  TOOL: take
  ARGS: {"item": "torch"}

  TOOL: talk
  ARGS: {"topics": ["WATER", "FARM"]}

The ONLY tools you can call are: play_action, memory, get_map, inventory, valid_actions.
Everything else must be a game command string inside play_action's "action" argument.
"""

PROMISING_ACTIONS_PROMPT = """Given this text adventure game observation, list the most promising actions the player should try. Focus on:
- Items mentioned that can be taken ("take X")
- Containers or objects that can be opened/examined ("open X", "examine X")
- Creatures or NPCs to interact with ("talk to X", "ask X about Y")
- Puzzles or mechanisms hinted at
- Hidden things suggested ("look under X", "look in X")

Observation:
{observation}

List ONLY the action commands, one per line. No explanations. Example:
take lamp
open mailbox
examine rug
"""


# Agent Implementation
class StudentAgent:
    """
    MCP ReAct Agent with loop detection, structured per-room exploration,
    tool call validation, and valid actions via persistent subprocess.
    """

    def __init__(self):
        self.history: list[dict] = []
        self.recent_actions: list[str] = []
        self.score: int = 0
        self.valid_actions_cache: list[str] = []
        self.steps_since_va: int = 999  # Force fetch on first step

        # Structured exploration state
        self.current_location: str | None = None  # "name|id"
        self.current_location_name: str = "Unknown"
        self.room_action_log: dict[str, list[tuple[str, str]]] = {}
        self.room_promising: dict[str, list[str]] = {}
        self.steps_in_room: int = 0
        self.room_exits_tried: dict[str, set[str]] = {}
        self.max_steps_per_room: int = 8

    async def run(
        self,
        client,
        game: str,
        max_steps: int,
        seed: int,
        verbose: bool = False,
    ) -> RunResult:
        """Run the agent for a game session."""
        locations_visited = set()
        history = []
        moves = 0

        tools = await client.list_tools()
        tool_names = [t.name for t in tools]

        # Get initial observation
        result = await client.call_tool("play_action", {"action": "look"})
        observation = self._extract_result(result)

        location = observation.split("\n")[0] if observation else "Unknown"
        locations_visited.add(location)

        if verbose:
            print(f"\n{observation}")

        # Main ReAct loop
        for step in range(1, max_steps + 1):
            # Fetch valid actions periodically (every 3 game actions)
            if self.steps_since_va >= 3 and "valid_actions" in tool_names:
                try:
                    va_result = await client.call_tool("valid_actions", {})
                    va_text = self._extract_result(va_result)
                    parsed = self._parse_valid_actions(va_text)
                    if parsed:
                        self.valid_actions_cache = parsed
                    self.steps_since_va = 0
                    if verbose:
                        print(f"[VALID ACTIONS] {va_text}")
                except Exception as e:
                    self.steps_since_va = 0
                    if verbose:
                        print(f"[VALID ACTIONS] failed: {e}")

            prompt = self._build_prompt(observation)
            response = call_llm(prompt, SYSTEM_PROMPT, seed + step)
            thought, tool_name, tool_args = self._parse_response(response, tool_names)

            if verbose:
                print(f"\n--- Step {step} ---")
                print(f"[THOUGHT] {thought}")
                print(f"[TOOL] {tool_name}({tool_args})")

            # Validate and fix common issues
            tool_name, tool_args = self._validate_tool_call(tool_name, tool_args, tool_names)

            # Loop detection — re-prompt the LLM instead of picking randomly
            if tool_name == "play_action":
                action = tool_args.get("action", "look")
                self.recent_actions.append(action)
                if len(self.recent_actions) > 8:
                    self.recent_actions = self.recent_actions[-8:]

                loop_info = self._detect_loop()
                if loop_info:
                    loop_type, looped_actions = loop_info
                    if verbose:
                        print(f"[LOOP] {loop_type} detected: {looped_actions} — re-prompting LLM")

                    new_action, new_thought, forced = self._break_loop(
                        loop_type, looped_actions, observation, tool_names, seed, step, verbose
                    )

                    tool_name = "play_action"
                    tool_args = {"action": new_action}
                    thought = new_thought
                    self.recent_actions[-1] = new_action

                self.steps_since_va += 1
                moves += 1
            elif tool_name == "valid_actions":
                self.steps_since_va = 0

            # Execute the tool
            try:
                result = await client.call_tool(tool_name, tool_args)
                observation = self._extract_result(result)

                if tool_name == "valid_actions":
                    parsed = self._parse_valid_actions(observation)
                    if parsed:
                        self.valid_actions_cache = parsed

                if verbose:
                    print(f"[RESULT] {observation}...")
            except Exception as e:
                observation = f"Error: {e}"
                self.recent_actions.append("__error__")
                if verbose:
                    print(f"[ERROR] {e}")

            location = observation.split("\n")[0] if observation else "Unknown"
            locations_visited.add(location)

            self.history.append({
                "step": step,
                "thought": thought,
                "tool": tool_name,
                "args": tool_args,
                "result": observation
            })
            if len(self.history) > 10:
                self.history = self.history[-10:]

            self._update_score(observation)
            history.append((thought, f"{tool_name}({tool_args})", observation[:100]))

            if self._is_game_over(observation):
                if verbose:
                    print("\n*** GAME OVER ***")
                break

        return RunResult(
            final_score=self.score,
            max_score=350,
            moves=moves,
            locations_visited=locations_visited,
            game_completed=self._is_game_over(observation),
            history=history,
        )

    # Location detection & room tracking

    def _parse_location_tag(self, observation: str) -> tuple[str, str] | None:
        """Parse [Location: name|id] from MCP server response."""
        m = re.search(r'\[Location:\s*([^|]+)\|(\d+)\]', observation)
        if m:
            name = m.group(1).strip()
            loc_id = f"{name}|{m.group(2)}"
            return name, loc_id
        return None

    def _detect_new_location(self, observation: str) -> bool:
        parsed = self._parse_location_tag(observation)
        if not parsed:
            return False
        _, loc_id = parsed
        return loc_id != self.current_location

    def _update_location(self, observation: str, verbose: bool = False):
        parsed = self._parse_location_tag(observation)
        if not parsed:
            return
        name, loc_id = parsed
        if loc_id != self.current_location:
            if verbose:
                print(f"[LOCATION] Moved: {self.current_location_name} -> {name}")
            self.current_location = loc_id
            self.current_location_name = name
            self.steps_in_room = 0

    def _extract_promising_actions(self, observation: str, seed: int):
        """Use LLM to extract promising actions from observation text for current room."""
        if not self.current_location:
            return
        clean_obs = re.sub(r'\[Location:[^\]]*\]', '', observation)
        clean_obs = re.sub(r'\[Score:[^\]]*\]', '', clean_obs)
        clean_obs = re.sub(r'\+\d+ points!.*', '', clean_obs).strip()

        prompt = PROMISING_ACTIONS_PROMPT.format(observation=clean_obs)
        try:
            response = call_llm(prompt, "You extract game actions from text. Return ONLY action commands, one per line.", seed, max_tokens=150)
            actions = [line.strip() for line in response.strip().split('\n') if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('-')]
            cleaned = []
            for a in actions:
                a = re.sub(r'^\d+\.?\s*', '', a).strip()
                a = a.strip('"').strip("'").strip('`').lower()
                if a and len(a) < 50:
                    cleaned.append(a)
            self.room_promising[self.current_location] = cleaned[:8]
        except Exception:
            pass

    def _pick_unexplored_exit(self) -> str | None:
        all_dirs = ["north", "south", "east", "west", "up", "down",
                    "northeast", "northwest", "southeast", "southwest"]
        tried = self.room_exits_tried.get(self.current_location, set())
        if self.valid_actions_cache:
            for d in self.valid_actions_cache:
                if self._is_directional(d) and d.lower() not in tried:
                    return d
        for d in all_dirs:
            if d not in tried:
                return d
        return None

    def _is_directional(self, action: str) -> bool:
        a = action.lower().strip()
        directions = {"north", "south", "east", "west", "up", "down",
                      "go north", "go south", "go east", "go west", "go up", "go down",
                      "n", "s", "e", "w", "u", "d",
                      "ne", "nw", "se", "sw", "northeast", "northwest", "southeast", "southwest",
                      "enter", "exit"}
        return a in directions

    # Loop detection

    def _detect_loop(self) -> tuple[str, list[str]] | None:
        """
        Detect action loops:
        - Single-action: 2 repeats for non-directional, 4 for directional
        - Two-action: A, B, A, B pattern (2 full cycles)
        """
        ra = self.recent_actions

        if len(ra) >= 2 and ra[-1] == ra[-2]:
            if self._is_directional(ra[-1]):
                if len(ra) >= 4 and ra[-3] == ra[-1] and ra[-4] == ra[-1]:
                    return ("single-action", [ra[-1]])
            else:
                return ("single-action", [ra[-1]])

        if len(ra) >= 4:
            a, b, c, d = ra[-4], ra[-3], ra[-2], ra[-1]
            if a == c and b == d and a != b:
                return ("two-action", [a, b])

        return None

    def _break_loop(
        self, loop_type: str, looped_actions: list[str],
        observation: str, tool_names: list[str],
        seed: int, step: int, verbose: bool
    ) -> tuple[str, str, bool]:
        """Re-prompt the LLM to break out of a loop."""
        banned = set(a.lower() for a in looped_actions)

        if loop_type == "single-action":
            action_str = looped_actions[0]
            avoid_prompt = (
                f"You have been repeating '{action_str}'. Do NOT use '{action_str}' again.\n"
                f"Try a completely different action.\n"
            )
        else:
            a, b = looped_actions
            avoid_prompt = (
                f"You have been alternating '{a}' and '{b}'. Do NOT use either of those again.\n"
                f"Try something completely different.\n"
            )

        if self.valid_actions_cache:
            alternatives = [a for a in self.valid_actions_cache if a.lower() not in banned]
            if alternatives:
                avoid_prompt += f"\nValid actions available: {', '.join(alternatives[:15])}\n"

        avoid_prompt += f"\nCurrent situation:\n{observation}\n"
        if verbose:
            print(f"[LOOP REPROMPT] Banned actions: {banned}")

        retry_response = call_llm(avoid_prompt, SYSTEM_PROMPT, seed + step + 1000)
        retry_thought, retry_tool, retry_args = self._parse_response(retry_response, tool_names)

        new_action = retry_args.get("action", "look") if retry_tool == "play_action" else "look"
        forced = False

        if verbose:
            print(f"[LOOP REPROMPT] LLM suggested: '{new_action}'")

        if new_action.lower() in banned:
            forced = True
            if self.valid_actions_cache:
                alternatives = [a for a in self.valid_actions_cache if a.lower() not in banned]
                if alternatives:
                    new_action = alternatives[0]
                else:
                    new_action = "inventory" if "inventory" not in banned else "look"
            else:
                new_action = "inventory" if "inventory" not in banned else "south"
            retry_thought = f"Loop break forced to: {new_action}"
            if verbose:
                print(f"[LOOP FORCED] LLM still picked banned action — forcing: '{new_action}'")

        if verbose:
            print(f"[LOOP BREAK] Final action: '{new_action}' (forced={forced})")

        return new_action, retry_thought, forced

    # Prompt construction

    def _build_prompt(self, observation: str) -> str:
        parts = []

        parts.append(f"Current Score: {self.score}")
        parts.append(f"Current Room: {self.current_location_name} (step {self.steps_in_room} in this room)")

        if self.history:
            parts.append("\nRecent actions:")
            for entry in self.history[-8:]:
                action = entry.get("args", {}).get("action", entry["tool"])
                result_short = entry["result"][:80] + "..." if len(entry["result"]) > 80 else entry["result"]
                parts.append(f"  > {action} -> {result_short}")

            loop_info = self._detect_loop()
            if loop_info:
                loop_type, looped = loop_info
                if loop_type == "single-action":
                    parts.append(f"\n[WARNING: You've been doing '{looped[0]}' repeatedly. TRY SOMETHING DIFFERENT!]")
                else:
                    parts.append(f"\n[WARNING: You've been alternating '{looped[0]}' and '{looped[1]}'. BREAK THE PATTERN!]")

        parts.append(f"\nCurrent situation:\n{observation}")

        # Actions already tried in this room
        if self.current_location and self.current_location in self.room_action_log:
            tried = self.room_action_log[self.current_location]
            if tried:
                tried_strs = [f"{a} -> {o[:40]}" for a, o in tried[-6:]]
                parts.append(f"\nAlready tried in this room:")
                for t in tried_strs:
                    parts.append(f"  - {t}")

        # Promising actions for this room (untried)
        if self.current_location and self.current_location in self.room_promising:
            promising = self.room_promising[self.current_location]
            if promising:
                parts.append(f"\nPROMISING (untried) actions for this room: {', '.join(promising)}")
                parts.append(">> Try one of these FIRST! <<")

        if self.valid_actions_cache:
            parts.append(f"\nValid actions: {', '.join(self.valid_actions_cache[:20])}")

        # Exploration bias warning
        if self.steps_in_room >= self.max_steps_per_room - 2:
            parts.append(f"\n[EXPLORATION: You've been in {self.current_location_name} for {self.steps_in_room} steps. Consider moving to a new location soon!]")

        if self.current_location and self.current_location in self.room_exits_tried:
            tried_exits = self.room_exits_tried[self.current_location]
            if tried_exits:
                parts.append(f"Exits already tried: {', '.join(sorted(tried_exits))}")

        parts.append("\nWhat do you do next?")

        return "\n".join(parts)

    # Response parsing

    def _parse_response(self, response: str, valid_tools: list[str]) -> tuple[str, str, dict]:
        thought = "No reasoning provided"
        tool_name = "play_action"
        tool_args = {"action": "look"}

        lines = response.strip().split("\n")

        for line in lines:
            line_clean = line.strip()
            line_upper = line_clean.upper()

            if line_upper.startswith("THOUGHT:"):
                thought = line_clean.split(":", 1)[1].strip()

            elif line_upper.startswith("TOOL:"):
                raw_tool = line_clean.split(":", 1)[1].strip().lower()
                raw_tool = raw_tool.replace("**", "").replace("*", "").replace("`", "")
                raw_tool = raw_tool.split()[0] if raw_tool else "play_action"
                tool_name = raw_tool

            elif line_upper.startswith("ARGS:"):
                args_part = line_clean.split(":", 1)[1].strip()
                try:
                    args_part = args_part.replace("'", '"')
                    tool_args = json.loads(args_part)
                except json.JSONDecodeError:
                    match = re.search(r'"action"\s*:\s*"([^"]+)"', args_part)
                    if match:
                        tool_args = {"action": match.group(1)}
                    else:
                        tool_args = {"action": "look"}

        return thought, tool_name, tool_args

    # Tool call validation

    def _validate_tool_call(self, tool_name: str, tool_args: dict, valid_tools: list[str]) -> tuple[str, dict]:
        """Validate and fix common tool call issues (e.g. LLM using game verbs as tool names)."""
        original_tool_name = tool_name

        if tool_name not in valid_tools:
            if tool_name in ["action", "do", "command"]:
                tool_name = "play_action"
            elif tool_name in ["map", "location"]:
                tool_name = "get_map"
            elif tool_name in ["mem", "state", "status"]:
                tool_name = "memory"
            elif tool_name in ["inv", "items"]:
                tool_name = "inventory"
            elif tool_name in ["actions", "valid", "get_valid_actions"]:
                tool_name = "valid_actions"
            else:
                tool_name = "play_action"

        if tool_name in ("valid_actions", "memory", "get_map", "inventory"):
            tool_args = {}
            return tool_name, tool_args

        if tool_name == "play_action":
            action = tool_args.get("action", "")
            if not action:
                for key in ("direction", "command", "cmd"):
                    if key in tool_args:
                        action = tool_args[key]
                        break

            # Reconstruct action from verb + argument if LLM used a game verb as tool name
            if not action and original_tool_name not in valid_tools:
                game_verbs = {
                    "examine", "take", "drop", "open", "close", "look", "read",
                    "talk", "ask", "tell", "greet", "say", "show", "give",
                    "push", "pull", "turn", "move", "attack", "eat", "drink",
                    "put", "throw", "tie", "climb", "enter", "exit", "wait",
                    "listen", "smell", "touch", "wear", "remove", "unlock", "lock",
                }
                verb = original_tool_name.lower().strip()
                if verb in game_verbs:
                    obj = ""
                    for key in ("thing", "item", "object", "target", "topic",
                                "topics", "name", "direction", "text"):
                        if key in tool_args:
                            val = tool_args[key]
                            if isinstance(val, list):
                                obj = str(val[0]) if val else ""
                            else:
                                obj = str(val)
                            break
                    if not obj:
                        for v in tool_args.values():
                            if isinstance(v, str):
                                obj = v
                                break
                            elif isinstance(v, list) and v:
                                obj = str(v[0])
                                break
                    action = f"{verb} {obj}".strip() if obj else verb

            if not action:
                action = "look"

            invalid_verb_map = {
                "check": "examine",
                "inspect": "examine",
                "search": "look",
                "grab": "take",
                "pick": "take",
                "use": "examine",
                "investigate": "examine",
            }

            words = action.lower().split()
            if words and words[0] in invalid_verb_map:
                words[0] = invalid_verb_map[words[0]]
                action = " ".join(words)

            action = action.lower().strip()
            action = action.replace("**", "").replace("*", "").replace("`", "")
            action = " ".join(action.split())

            tool_args = {"action": action}

        return tool_name, tool_args

    # Utilities

    def _extract_result(self, result) -> str:
        if hasattr(result, 'content') and result.content:
            return result.content[0].text
        if isinstance(result, list) and result:
            return result[0].text if hasattr(result[0], 'text') else str(result[0])
        return str(result)

    def _update_score(self, text: str) -> None:
        patterns = [
            r'Score:\s*(\d+)',
            r'score[:\s]+(\d+)',
            r'\[Score:\s*(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self.score = max(self.score, int(match.group(1)))

    def _is_game_over(self, text: str) -> bool:
        game_over_phrases = [
            "game over",
            "you have died",
            "you are dead",
            "*** you have died ***",
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in game_over_phrases)

    def _parse_valid_actions(self, text: str) -> list[str]:
        if "Valid actions" in text and ":" in text:
            after_colon = text.split(":", 1)[1].strip()
            if "[" in after_colon:
                after_colon = after_colon[:after_colon.index("[")].strip()
            return [a.strip() for a in after_colon.split(",") if a.strip()]
        return []


# Local testing
async def test_agent():
    from fastmcp import Client

    agent = StudentAgent()

    async with Client("mcp_server.py") as client:
        result = await agent.run(
            client=client,
            game="zork1",
            max_steps=20,
            seed=42,
            verbose=True,
        )

        print(f"\nFinal Score: {result.final_score}")
        print(f"Moves: {result.moves}")
        print(f"Locations: {len(result.locations_visited)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
