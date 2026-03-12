"""Persistent worker process for get_valid_actions.

Runs as a long-lived subprocess. Reads requests from stdin (one per line),
writes responses to stdout (one per line).

Protocol (line-based):
  INIT <game_name>  ->  OK
  VA <base64-state>  ->  OK action1|||action2||| ...
                     or  ERR <message>
                     or  TIMEOUT
"""
import sys
import os
import pickle
import base64
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from games.zork_env import TextAdventureEnv

    env = None

    def _timeout_handler(signum, frame):
        raise TimeoutError("get_valid_actions timed out")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            if line.startswith("INIT "):
                game_name = line[5:].strip()
                env = TextAdventureEnv(game_name)
                env.reset()
                sys.stdout.write("OK\n")
                sys.stdout.flush()

            elif line.startswith("VA "):
                if env is None:
                    sys.stdout.write("ERR no game initialized\n")
                    sys.stdout.flush()
                    continue

                state_b64 = line[3:].strip()
                state_bytes = base64.b64decode(state_b64)
                state = pickle.loads(state_bytes)
                env.load_state(state)

                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(30)
                try:
                    actions = env.get_valid_actions()
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                    if actions:
                        sys.stdout.write("OK " + "|||".join(actions) + "\n")
                    else:
                        sys.stdout.write("ERR empty\n")
                except TimeoutError:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                    sys.stdout.write("TIMEOUT\n")
                except Exception as e:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                    sys.stdout.write(f"ERR {e}\n")
                sys.stdout.flush()

            else:
                sys.stdout.write("ERR unknown command\n")
                sys.stdout.flush()

        except Exception as e:
            try:
                sys.stdout.write(f"ERR {e}\n")
                sys.stdout.flush()
            except Exception:
                pass


if __name__ == "__main__":
    main()
