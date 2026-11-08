"""
Screenplay parser for the AI Promo Finder.

Reads every episode .txt file in a folder and turns it into structured rows:
one row per block (CAMERA, ACTION, SOUND FX or a line of DIALOGUE), each with
its episode, scene, line ID, start time, end time and text.

Run from the project folder:
    python src/screenplay_parser.py
    python src/screenplay_parser.py --input path/to/another/show --out data/processed

Writes:
    <out>/episodes.json   everything, nested: episode -> scenes -> lines
    <out>/lines.csv       one flat row per line (easy to open in Excel)
"""

import argparse
import csv
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns. re.IGNORECASE matters: Episode 6 writes "SCENE HEADING:"
# while every other episode writes "Scene heading:".
# ---------------------------------------------------------------------------
TIME = r"(\d{1,2}):(\d{2}(?:\.\d+)?)"  # 02:18.00  -> minutes, seconds

EPISODE_NUMBER = re.compile(r"^EPISODE NUMBER:\s*Episode\s+(\d+)", re.I)
LOGLINE = re.compile(r"^EPISODE LOGLINE:\s*(.+)", re.I)
CONTINUITY = re.compile(r"^CONTINUITY NOTES:\s*(.+)", re.I)

SCENE_NUMBER = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
SCENE_HEADING = re.compile(r"^Scene heading:\s*(.+)", re.I)
SCENE_START = re.compile(r"^Scene start:\s*" + TIME, re.I)
SCENE_END = re.compile(r"^Scene end:\s*" + TIME, re.I)

# **CAMERA:**   **LADY ANNA (Confused, Breathless):**   **GUARD (O.S.) (Firm):**
BLOCK_HEADER = re.compile(r"^\*\*(?P<label>[^*(]+?)\s*(?P<notes>(?:\([^)]*\)\s*)*):\*\*\s*$")
# Start: (00:01.00) | End: (00:04.80)
BLOCK_TIMES = re.compile(r"^Start:\s*\(" + TIME + r"\)\s*\|\s*End:\s*\(" + TIME + r"\)", re.I)

NON_DIALOGUE = {"CAMERA", "ACTION", "SOUND FX"}
SKIP = re.compile(r"^(--- SCRIPT (START|END) ---|Actions and dialogue:)", re.I)


def to_seconds(minutes: str, seconds: str) -> float:
    return round(int(minutes) * 60 + float(seconds), 2)


def parse_episode(text: str, source_name: str) -> dict:
    """Parse one screenplay into a dict. Collects problems instead of crashing."""
    episode = {"episode": None, "source_file": source_name, "logline": "",
               "continuity_notes": "", "scenes": [], "problems": []}
    scene = None
    block = None  # the block whose text we are currently collecting

    def close_block():
        nonlocal block
        if block is None:
            return
        block["text"] = " ".join(block["text"]).strip()
        if block["type"] == "DIALOGUE":
            block["text"] = block["text"].strip('"“” ')
        if block["start"] is None:
            episode["problems"].append(f"No Start/End line for {block['type']} block in scene {scene['scene']}")
        else:
            scene["lines"].append(block)
        block = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or SKIP.match(line):
            if not line:
                close_block()  # a blank line ends the current block
            continue

        if m := EPISODE_NUMBER.match(line):
            episode["episode"] = int(m.group(1))
        elif m := LOGLINE.match(line):
            episode["logline"] = m.group(1).strip()
        elif m := CONTINUITY.match(line):
            episode["continuity_notes"] = m.group(1).strip()

        elif m := SCENE_NUMBER.match(line):
            close_block()
            scene = {"scene": int(m.group(1)), "heading": "", "start": None, "end": None, "lines": []}
            episode["scenes"].append(scene)
        elif scene and (m := SCENE_HEADING.match(line)):
            scene["heading"] = m.group(1).strip()
        elif scene and (m := SCENE_START.match(line)):
            scene["start"] = to_seconds(*m.groups())
        elif scene and (m := SCENE_END.match(line)):
            scene["end"] = to_seconds(*m.groups())

        elif scene and (m := BLOCK_HEADER.match(line)):
            close_block()
            label = m.group("label").strip().upper()
            notes = re.findall(r"\(([^)]*)\)", m.group("notes"))
            is_dialogue = label not in NON_DIALOGUE
            block = {
                "type": "DIALOGUE" if is_dialogue else label,
                "speaker": label if is_dialogue else "",
                "delivery": ", ".join(notes),      # e.g. "O.S., Firm"
                "start": None, "end": None, "text": [],
            }
        elif block and block["start"] is None and (m := BLOCK_TIMES.match(line)):
            g = m.groups()
            block["start"], block["end"] = to_seconds(g[0], g[1]), to_seconds(g[2], g[3])
        elif block:
            block["text"].append(line)
        elif scene is not None:
            episode["problems"].append(f"Unrecognised line: {line[:80]}")

    close_block()
    return episode


def check_episode(ep: dict) -> None:
    """Sanity checks. Anything odd is added to ep['problems']."""
    p = ep["problems"]
    if ep["episode"] is None:
        p.append("No EPISODE NUMBER line")
    if not ep["scenes"]:
        p.append("No scenes found")
    for s in ep["scenes"]:
        if s["start"] is None or s["end"] is None:
            p.append(f"Scene {s['scene']} is missing start/end time")
            continue
        if not s["lines"]:
            p.append(f"Scene {s['scene']} has no lines")
        for ln in s["lines"]:
            if ln["end"] < ln["start"]:
                p.append(f"{ln['line_id']} ends before it starts")
            if ln["start"] < s["start"] - 0.5 or ln["end"] > s["end"] + 0.5:
                p.append(f"{ln['line_id']} ({ln['start']}-{ln['end']}s) falls outside scene {s['scene']}")
    for a, b in zip(ep["scenes"], ep["scenes"][1:]):
        if a["end"] is not None and b["start"] is not None and abs(a["end"] - b["start"]) > 0.5:
            p.append(f"Gap/overlap between scene {a['scene']} end {a['end']}s and scene {b['scene']} start {b['start']}s")


def add_line_ids(ep: dict) -> None:
    """Give every line a stable ID like E01_L014 (used later by the LLM)."""
    n = 0
    for s in ep["scenes"]:
        for ln in s["lines"]:
            n += 1
            ln["line_id"] = f"E{ep['episode']:02d}_L{n:03d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse screenplay .txt files into JSON and CSV.")
    ap.add_argument("--input", default="data/raw/episodes", help="folder containing episode .txt files")
    ap.add_argument("--out", default="data/processed", help="folder to write episodes.json and lines.csv")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.out)
    files = list(in_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {in_dir.resolve()}")

    episodes = []
    for f in files:
        ep = parse_episode(f.read_text(encoding="utf-8"), f.name)
        if ep["episode"] is None:  # fall back to the number in the file name
            if m := re.search(r"(\d+)", f.stem):
                ep["episode"] = int(m.group(1))
        add_line_ids(ep)
        check_episode(ep)
        episodes.append(ep)

    episodes.sort(key=lambda e: e["episode"])  # by episode number, not file name

    # ---- report --------------------------------------------------------
    print(f"{'EP':>3} {'scenes':>6} {'lines':>5} {'dialogue':>8} {'length':>7}  status")
    total_problems = 0
    for ep in episodes:
        lines = [ln for s in ep["scenes"] for ln in s["lines"]]
        dialogue = sum(ln["type"] == "DIALOGUE" for ln in lines)
        length = max((s["end"] or 0) for s in ep["scenes"]) if ep["scenes"] else 0
        status = "OK" if not ep["problems"] else f"CHECK ({len(ep['problems'])} problems)"
        print(f"{ep['episode']:>3} {len(ep['scenes']):>6} {len(lines):>5} {dialogue:>8} {length:>6.1f}s  {status}")
        for prob in ep["problems"][:5]:
            print(f"      - {prob}")
        total_problems += len(ep["problems"])

    numbers = [e["episode"] for e in episodes]
    missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
    dupes = sorted({n for n in numbers if numbers.count(n) > 1})
    print(f"\n{len(episodes)} episodes parsed, {total_problems} problems.")
    if missing:
        print(f"Missing episode numbers: {missing}")
    if dupes:
        print(f"Duplicate episode numbers: {dupes}")

    # ---- save ----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes.json").write_text(json.dumps(episodes, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = ["line_id", "episode", "scene", "scene_heading", "type", "speaker",
              "delivery", "start", "end", "text"]
    with open(out_dir / "lines.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for ep in episodes:
            for s in ep["scenes"]:
                for ln in s["lines"]:
                    w.writerow({"episode": ep["episode"], "scene": s["scene"],
                                "scene_heading": s["heading"], **{k: ln[k] for k in fields if k in ln}})
    print(f"Saved {out_dir / 'episodes.json'} and {out_dir / 'lines.csv'}")


if __name__ == "__main__":
    main()