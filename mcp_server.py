"""MCP Server for text adventure games.

Exposes game interaction via MCP tools with memory, mapping,
inventory tracking, and valid actions via persistent subprocess.
"""

import sys
import os
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from games.zork_env import TextAdventureEnv, list_available_games

INITIAL_GAME = os.environ.get("GAME", "zork1")

mcp = FastMCP("Text Adventure Server")


class GameState:
    """Manages the text adventure game state and exploration data."""

    def __init__(self, game: str = "zork1"):
        self.game_name = game
        self.env = TextAdventureEnv(game)
        self.state = self.env.reset()
        self.history: list[tuple[str, str]] = []
        self.explored_locations: dict[str, set[str]] = {}
        self.current_location: str = self._extract_location(self.state.observation)
        self.va_ok: int = 0
        self.va_fail: int = 0

    def _extract_location(self, observation: str) -> str:
        lines = observation.strip().split('\n')
        return lines[0] if lines else "Unknown"

    def take_action(self, action: str) -> str:
        self.state = self.env.step(action)
        result = self.state.observation

        self.history.append((action, result))
        if len(self.history) > 50:
            self.history = self.history[-50:]

        new_location = self._extract_location(result)
        if action in ["north", "south", "east", "west", "up", "down",
                      "enter", "exit", "n", "s", "e", "w", "u", "d"]:
            if self.current_location not in self.explored_locations:
                self.explored_locations[self.current_location] = set()
            if new_location != self.current_location:
                self.explored_locations[self.current_location].add(f"{action} -> {new_location}")
        self.current_location = new_location

        return result

    def get_memory(self) -> str:
        recent = self.history[-5:] if self.history else []
        recent_str = "\n".join([f"  > {a} -> {r[:60]}..." for a, r in recent]) if recent else "  (none yet)"

        try:
            loc = self.env.env.get_player_location()
            location_str = f"{loc.name} (id={loc.num})"
        except Exception:
            location_str = self.current_location

        return f"""Current State:
- Location: {location_str}
- Score: {self.state.score} points
- Moves: {self.state.moves}
- Game: {self.game_name}
Recent Actions:
{recent_str}
Current Observation:
{self.state.observation}"""

    def get_map(self) -> str:
        if not self.explored_locations:
            return "Map: No locations explored yet. Try moving around!"

        lines = ["Explored Locations and Exits:"]
        for loc, exits in sorted(self.explored_locations.items()):
            lines.append(f"\n* {loc}")
            for exit_info in sorted(exits):
                lines.append(f"    -> {exit_info}")

        lines.append(f"\n[Current] {self.current_location}")
        return "\n".join(lines)

    def get_inventory(self) -> str:
        items = self.state.inventory if hasattr(self.state, 'inventory') and self.state.inventory else []

        if not items:
            return "Inventory: You are empty-handed."

        item_names = []
        for item in items:
            item_str = str(item)
            item_lower = item_str.lower()
            if "parent" in item_lower:
                idx = item_lower.index("parent")
                name = item_str[:idx].strip()
                if ":" in name:
                    name = name.split(":", 1)[1].strip()
                item_names.append(name)
            elif ":" in item_str:
                name = item_str.split(":")[1].strip()
                item_names.append(name)
            else:
                item_names.append(item_str)

        return f"Inventory: {', '.join(item_names)}"

    def get_valid_actions(self) -> str:
        """Get valid actions using a persistent worker subprocess."""
        import subprocess, pickle, base64

        fallback = "look, north, south, east, west, up, down, inventory, take all, open, examine"

        try:
            # Launch persistent worker if not already running
            if not hasattr(self, '_va_worker') or self._va_worker is None or self._va_worker.poll() is not None:
                worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_va_worker.py")
                self._va_worker = subprocess.Popen(
                    [sys.executable, "-u", worker_script],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                self._va_worker.stdin.write(f"INIT {self.game_name}\n")
                self._va_worker.stdin.flush()
                import select
                ready, _, _ = select.select([self._va_worker.stdout], [], [], 35)
                if ready:
                    resp = self._va_worker.stdout.readline().strip()
                    if resp != "OK":
                        raise RuntimeError(f"Worker init failed: {resp}")
                else:
                    raise RuntimeError("Worker init timed out")

            # Serialize current game state and send request
            state_bytes = self.env.save_state()
            state_b64 = base64.b64encode(pickle.dumps(state_bytes)).decode()
            self._va_worker.stdin.write(f"VA {state_b64}\n")
            self._va_worker.stdin.flush()

            import select
            ready, _, _ = select.select([self._va_worker.stdout], [], [], 35)
            if not ready:
                self._va_worker.kill()
                self._va_worker = None
                self.va_fail += 1
                return f"Valid actions timed out. Fallback: {fallback} [timeout, ok={self.va_ok}, fail={self.va_fail}]"

            resp = self._va_worker.stdout.readline().strip()

            if resp.startswith("OK "):
                actions_str = resp[3:]
                actions = [a.strip() for a in actions_str.split("|||") if a.strip()]
                if actions:
                    self.va_ok += 1
                    return f"Valid actions ({len(actions)}): {', '.join(actions)}"
                self.va_fail += 1
                return f"No valid actions found. Fallback: {fallback} [ok={self.va_ok}, fail={self.va_fail}]"
            elif resp == "TIMEOUT":
                self.va_fail += 1
                return f"Valid actions timed out. Fallback: {fallback} [worker-timeout, ok={self.va_ok}, fail={self.va_fail}]"
            else:
                self.va_fail += 1
                return f"Worker error: {resp}. Fallback: {fallback} [ok={self.va_ok}, fail={self.va_fail}]"

        except Exception as e:
            if hasattr(self, '_va_worker') and self._va_worker is not None:
                try:
                    self._va_worker.kill()
                except Exception:
                    pass
                self._va_worker = None
            self.va_fail += 1
            return f"Error: {e}. Fallback: {fallback} [ok={self.va_ok}, fail={self.va_fail}]"


# Global game state
_game_state: GameState | None = None


def get_game() -> GameState:
    global _game_state
    if _game_state is None:
        _game_state = GameState(INITIAL_GAME)
    return _game_state


# MCP Tools

@mcp.tool()
def play_action(action: str) -> str:
    """Execute a game action in the text adventure.

    Args:
        action: The command to execute (e.g., 'north', 'take lamp', 'open mailbox')
    """
    game = get_game()
    result = game.take_action(action)

    # Append reliable location tag from Jericho API
    try:
        loc = game.env.env.get_player_location()
        location_tag = f"\n[Location: {loc.name}|{loc.num}]"
    except Exception:
        location_tag = ""

    score_info = f"\n[Score: {game.state.score} | Moves: {game.state.moves}]"
    if game.state.reward > 0:
        score_info = f"\n+{game.state.reward} points! (Total: {game.state.score})"

    done_info = ""
    if game.state.done:
        done_info = "\nGAME OVER"

    return result + location_tag + score_info + done_info


@mcp.tool()
def memory() -> str:
    """Get a summary of the current game state (location, score, moves, recent actions)."""
    return get_game().get_memory()


@mcp.tool()
def get_map() -> str:
    """Get a map showing explored locations and connections."""
    return get_game().get_map()


@mcp.tool()
def inventory() -> str:
    """Check what items you are currently carrying."""
    return get_game().get_inventory()


@mcp.tool()
def valid_actions() -> str:
    """Get a list of valid actions for the current game state."""
    try:
        return get_game().get_valid_actions()
    except Exception as e:
        return f"Error: {e}. Try: look, north, south, east, west."


# Cleanup

def _cleanup():
    global _game_state
    if _game_state is not None:
        try:
            _game_state.env.close()
        except Exception:
            pass
        _game_state = None

def _force_exit_handler(signum, frame):
    _cleanup()
    os._exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _force_exit_handler)
    signal.signal(signal.SIGINT, _force_exit_handler)
    try:
        mcp.run()
    finally:
        os._exit(0)
