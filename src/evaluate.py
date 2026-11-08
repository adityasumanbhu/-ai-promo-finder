"""
Compare a run's promos with your hand-picked golden set (golden_set.csv).

Run from the project folder:
    python src/evaluate.py --run golden_flashlite

For each golden promo it reports:
  Overlap     best time overlap with an AI promo that PASSED the rules
              (shared seconds / combined seconds: 1.0 = same clip, 0.0 = nothing shared).
              Overlap >= 0.5 counts as a match.
  Cliff       did any passed AI promo end on the same cliffhanger (within 5 seconds)?
  Incl.rej    best overlap if we also count promos the validator REJECTED.
              If this is high but Overlap is low, the AI found the moment but broke a rule.
  Baseline    overlap with a dumb rule: "the last 45 seconds of the episode".
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from show_episode import episode_lines, load_episodes
from validate import segment_span

MATCH_AT = 0.5
SAME_CLIFF_SECONDS = 5.0
BASELINE_SECONDS = 45.0


def shared_seconds(a: list, b: list) -> float:
    """a and b are lists of (start, end) intervals in the same episode."""
    return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def overlap(a: list, b: list) -> float:
    inter = shared_seconds(a, b)
    total = sum(e - s for s, e in a) + sum(e - s for s, e in b) - inter
    return inter / total if total > 0 else 0.0


def load_golden(path: Path, episodes: dict) -> list:
    """Group golden_set.csv rows by promo_id; times come from the parser, not from typed values."""
    grouped = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("promo_id") or "").strip():
                grouped[row["promo_id"].strip()].append(row)
    golden = []
    for pid, rows in grouped.items():
        ep = int(rows[0]["episode"])
        lines = episode_lines(episodes[ep])
        spans = [segment_span(lines, r["start_line"].strip(), r["end_line"].strip()) for r in rows]
        golden.append({"promo_id": pid, "episode": ep, "spans": spans})
    return golden


def best_match(g: dict, candidates: list, include_rejected: bool) -> tuple:
    best_id, best = "-", 0.0
    for c in candidates:
        if c["episode"] != g["episode"] or not c["segments"] or (not c["ok"] and not include_rejected):
            continue
        score = overlap(g["spans"], [(s["start"], s["end"]) for s in c["segments"]])
        if score > best:
            best_id, best = c["promo_id"], score
    return best_id, best


def same_cliff(g: dict, candidates: list) -> str:
    g_end = g["spans"][-1][1]
    for c in candidates:
        if c["episode"] == g["episode"] and c["ok"] and abs(c["segments"][-1]["end"] - g_end) <= SAME_CLIFF_SECONDS:
            return c["promo_id"]
    return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder name inside outputs/")
    ap.add_argument("--golden", default="golden_set.csv")
    args = ap.parse_args()

    run_dir = Path("outputs") / args.run
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    candidates = json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))
    episodes = load_episodes()

    golden = [g for g in load_golden(Path(args.golden), episodes) if g["episode"] in set(info["episodes"])]
    if not golden:
        raise SystemExit("None of your golden promos are in the episodes of this run.")

    passed = sum(c["ok"] for c in candidates if c["episode"] in {g["episode"] for g in golden})
    total = sum(1 for c in candidates if c["episode"] in {g["episode"] for g in golden})
    print(f"Run: {args.run}   Model: {info['model']}   AI promos passed the rules: {passed}/{total}\n")
    print(f"{'Golden':<7}{'Ep':>3}  {'Best AI':<8} {'Overlap':>7} {'Match':<5}  {'Cliff':<7} "
          f"{'Incl.rej':>8}  {'Baseline':>8}")

    hits = cliffs = base_hits = base_cliffs = 0
    for g in golden:
        ep_len = max(s["end"] for s in episodes[g["episode"]]["scenes"])
        base = overlap(g["spans"], [(max(0.0, ep_len - BASELINE_SECONDS), ep_len)])
        bid, best = best_match(g, candidates, include_rejected=False)
        _, best_rej = best_match(g, candidates, include_rejected=True)
        cliff = same_cliff(g, candidates)

        hits += best >= MATCH_AT
        cliffs += cliff != "-"
        base_hits += base >= MATCH_AT
        base_cliffs += abs(ep_len - g["spans"][-1][1]) <= SAME_CLIFF_SECONDS
        print(f"{g['promo_id']:<7}{g['episode']:>3}  {bid:<8} {best:>7.2f} {'YES' if best >= MATCH_AT else 'no':<5}  "
              f"{cliff:<7} {best_rej:>8.2f}  {base:>8.2f}")

    n = len(golden)
    print(f"\nAI:       matched {hits}/{n} (overlap >= {MATCH_AT}),  same cliffhanger {cliffs}/{n}")
    print(f"Baseline: matched {base_hits}/{n} (last {BASELINE_SECONDS:.0f}s of episode),  same cliffhanger {base_cliffs}/{n}")


if __name__ == "__main__":
    main()