"""
Read episodes in a compact, timestamped form, and preview a possible promo clip.

Run from the project folder (after screenplay_parser.py has created episodes.json):
    python src/show_episode.py --loglines                       # one-line summary of every episode
    python src/show_episode.py 1                                # print Episode 1
    python src/show_episode.py 1 --no-camera                    # hide CAMERA lines (quicker to read)
    python src/show_episode.py 1 --from E01_L030 --to E01_L060  # preview a clip + check its length

The same format_line() is what we will later send to the LLM.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from validate import segment_span

sys.stdout.reconfigure(encoding="utf-8")  # so Windows terminals print every character

# run.py sets PROMO_DATA so each show keeps its own parsed files; by default it is data/processed.
EPISODES_JSON = Path(os.getenv("PROMO_DATA", "data/processed")) / "episodes.json"
MIN_LEN, MAX_LEN, HOOK_WINDOW = 20.0, 60.0, 3.0


def load_episodes(path: Path = EPISODES_JSON) -> dict:
    """Return {episode_number: episode_dict}."""
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python src/screenplay_parser.py")
    return {ep["episode"]: ep for ep in json.loads(path.read_text(encoding="utf-8"))}


def episode_lines(ep: dict) -> list:
    return [ln for scene in ep["scenes"] for ln in scene["lines"]]


def mmss(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:04.1f}"


def format_line(ln: dict) -> str:
    """E01_L009 00:13.5-00:14.3 LADY ANNA (Confused): Where were you?"""
    when = f"{mmss(ln['start'])}-{mmss(ln['end'])}"
    if ln["type"] == "DIALOGUE":
        who = ln["speaker"] + (f" ({ln['delivery']})" if ln["delivery"] else "")
        return f"{ln['line_id']} {when} {who}: {ln['text']}"
    return f"{ln['line_id']} {when} [{ln['type']}] {ln['text']}"


def clip_report(lines: list, start_id: str, end_id: str) -> None:
    ids = [ln["line_id"] for ln in lines]
    for lid in (start_id, end_id):
        if lid not in ids:
            raise SystemExit(f"{lid} is not a line in this episode")
    i, j = ids.index(start_id), ids.index(end_id)
    if j < i:
        raise SystemExit("--to must come after --from")
    clip = lines[i:j + 1]
    start, end = segment_span(lines, start_id, end_id)  # same rule the validator uses
    length = end - start

    print(f"\nHOOK WINDOW (first {HOOK_WINDOW:.0f}s of the clip):")
    for ln in clip:
        if ln["start"] < start + HOOK_WINDOW:
            print("  " + format_line(ln))
    print("\nLAST LINES (where the clip cuts):")
    for ln in clip[-3:]:
        print("  " + format_line(ln))

    verdict = "OK" if MIN_LEN <= length <= MAX_LEN else f"NOT OK (must be {MIN_LEN:.0f}-{MAX_LEN:.0f}s)"
    print(f"\nClip {start_id} -> {end_id}: {mmss(start)} to {mmss(end)} = {length:.1f}s  {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?", type=int, help="episode number")
    ap.add_argument("--loglines", action="store_true", help="list every episode's logline")
    ap.add_argument("--no-camera", action="store_true", help="hide CAMERA lines")
    ap.add_argument("--from", dest="start_id", help="first line of a clip, e.g. E01_L030")
    ap.add_argument("--to", dest="end_id", help="last line of a clip, e.g. E01_L060")
    args = ap.parse_args()
    episodes = load_episodes()

    if args.loglines:
        for n, ep in sorted(episodes.items()):
            print(f"Ep {n:>2}: {ep['logline']}\n")
        return
    if args.episode not in episodes:
        raise SystemExit("Give an episode number, e.g. python src/show_episode.py 1")

    ep = episodes[args.episode]
    lines = episode_lines(ep)
    print(f"EPISODE {ep['episode']}: {ep['logline']}\n")

    shown = lines
    if args.start_id and args.end_id:
        ids = [ln["line_id"] for ln in lines]
        if args.start_id in ids and args.end_id in ids:
            shown = lines[ids.index(args.start_id): ids.index(args.end_id) + 1]
    for scene in ep["scenes"]:
        printed_heading = False
        for ln in scene["lines"]:
            if ln not in shown or (args.no_camera and ln["type"] == "CAMERA"):
                continue
            if not printed_heading:
                print(f"--- SCENE {scene['scene']}: {scene['heading']} ---")
                printed_heading = True
            print(format_line(ln))

    if args.start_id and args.end_id:
        clip_report(lines, args.start_id, args.end_id)


if __name__ == "__main__":
    main()