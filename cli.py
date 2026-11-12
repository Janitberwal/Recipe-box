"""Command-line chat.  Run:  python cli.py   (type 'quit' to exit, 'reset' to start over)"""
import html
import re
import sys

from recipebot.agent import Agent
from recipebot.store import get_store

CHIP_RE = re.compile(r"<span class='ichip[^']*'>(.*?)</span>")
NUM_RE = re.compile(r"<span class='rnum'>(\d+)</span>")


def strip_md(s: str) -> str:
    """Turn the UI's HTML cards into readable terminal text."""
    s = re.sub(r"</(div|li|ul|ol)>", "\n", s)
    s = re.sub(r"<li[^>]*>", "  - ", s)
    s = CHIP_RE.sub(lambda m: m.group(1) + " | ", s)
    s = NUM_RE.sub(lambda m: m.group(1) + ". ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("Loading recipes…")
    agent = Agent(get_store())
    print("Recipe Box Assistant – ask me anything (quit to exit)\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit"}:
            break
        if not q:
            continue
        print("\n" + strip_md(agent.handle(q).markdown) + "\n")


if __name__ == "__main__":
    main()
