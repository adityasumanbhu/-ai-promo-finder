"""
Ask the LLM for promo candidates in each episode, then check every candidate with validate.py.

Run from the project folder (after screenplay_parser.py):
    python src/find_promos.py --episodes 1             # one episode
    python src/find_promos.py --episodes 1-5           # a range
    python src/find_promos.py --episodes all --run run_01

Length retry: the model is bad at adding up seconds. If any candidate in an episode is outside
20-60 s, we send ONE more call for that episode with the real measured lengths
(prompt: prompts/fix_length.txt). A fixed candidate replaces the original only if it now passes
every rule. Turn this off with --no-retry.

Writes to outputs/<run>/:
    raw/E01.txt         the model's answer, exactly as returned (unedited raw output)
    raw/E01_retry.txt   the model's answer to the length retry, if there was one
    candidates.json     every candidate, with timestamps and pass/fail reasons
    candidates.csv      the same, one row per candidate
    run_info.json       model, date, episodes and the exact prompt template used
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from llm import MODEL, ask_json
from show_episode import episode_lines, format_line, load_episodes
from validate import MAX_TOTAL, MIN_TOTAL, validate_promo

PROMPT_FILE = Path("prompts/find_promos.txt")
FIX_FILE = Path("prompts/fix_length.txt")


def pick_episodes(spec: str, available: list) -> list:
    """'all', '3', '1-5' or '1,3,12' -> list of episode numbers."""
    if spec == "all":
        return sorted(available)
    chosen = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            chosen += range(int(a), int(b) + 1)
        else:
            chosen.append(int(part))
    missing = [n for n in chosen if n not in available]
    if missing:
        raise SystemExit(f"Episodes not found: {missing}")
    return chosen


def build_prompt(template: str, ep: dict, total: int) -> str:
    screenplay = "\n".join(format_line(ln) for ln in episode_lines(ep))
    return (template.replace("<<EPISODE>>", str(ep["episode"]))
                    .replace("<<TOTAL>>", str(total))
                    .replace("<<LOGLINE>>", ep["logline"])
                    .replace("<<CONTINUITY>>", ep["continuity_notes"])
                    .replace("<<SCREENPLAY>>", screenplay))


def wrong_length(check: dict) -> bool:
    """True if the segments are valid but the total is outside 20-60 s."""
    return bool(check["segments"]) and not (MIN_TOTAL <= check["total_length"] <= MAX_TOTAL)


def length_feedback(number: int, promo: dict, check: dict) -> str:
    """Tell the model what its promo really measures, segment by segment."""
    total = check["total_length"]
    if total > MAX_TOTAL:
        verdict = f"too long by {total - MAX_TOTAL:.1f}s"
    else:
        verdict = f"too short by {MIN_TOTAL - total:.1f}s"
    out = [f'Promo {number} ("{promo.get("title", "")}"): {total}s in total, {verdict}']
    for k, seg in enumerate(check["segments"], 1):
        out.append(f"  segment {k}: {seg['start_line']} to {seg['end_line']} = {seg['length']}s")
    out.append("  your earlier answer: " + json.dumps(promo, ensure_ascii=False))
    return "\n".join(out)


def retry_lengths(prompt: str, fix_template: str, promos: list, checks: list,
                  lines: list, raw_file: Path) -> dict:
    """One extra call per episode for every candidate with a wrong length.

    Updates promos/checks in place when a fixed version passes every rule.
    Returns {position: retry_note} for each candidate that was retried.
    """
    bad = [k for k, c in enumerate(checks) if wrong_length(c)]
    if not bad:
        return {}
    feedback = "\n\n".join(length_feedback(n, promos[k], checks[k]) for n, k in enumerate(bad, 1))
    data, raw, _ = ask_json(prompt + "\n\n" + fix_template.replace("<<FEEDBACK>>", feedback))
    raw_file.write_text(raw, encoding="utf-8")

    fixed = data.get("promos", []) if isinstance(data, dict) else []
    notes = {}
    for pos, k in enumerate(bad):
        first_problems = checks[k]["problems"]
        new = fixed[pos] if pos < len(fixed) and isinstance(fixed[pos], dict) else None
        new_check = validate_promo(new, lines) if new else None
        if new_check and new_check["ok"]:
            promos[k], checks[k] = new, new_check
            notes[k] = {"retry": "fixed", "first_problems": first_problems}
        else:
            still = "; ".join(new_check["problems"]) if new_check else "no answer for this promo"
            notes[k] = {"retry": f"still failing ({still})", "first_problems": first_problems}
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="1", help="'all', '3', '1-5' or '1,3,12'")
    ap.add_argument("--run", default="run_01", help="name of the output folder inside outputs/")
    ap.add_argument("--no-retry", action="store_true", help="skip the length-fix call")
    args = ap.parse_args()

    episodes = load_episodes()
    chosen = pick_episodes(args.episodes, list(episodes))
    template = PROMPT_FILE.read_text(encoding="utf-8")
    fix_template = FIX_FILE.read_text(encoding="utf-8")
    run_dir = Path("outputs") / args.run
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)

    all_candidates = []
    print(f"Model: {MODEL}   Episodes: {len(chosen)}   Output: {run_dir}\n")
    for n in chosen:
        ep = episodes[n]
        lines = episode_lines(ep)
        prompt = build_prompt(template, ep, len(episodes))
        try:
            data, raw, cached = ask_json(prompt)
        except Exception as err:  # one bad episode should not stop the whole run
            print(f"Ep {n:>2}: FAILED ({type(err).__name__}: {err})")
            continue
        (run_dir / "raw" / f"E{n:02d}.txt").write_text(raw, encoding="utf-8")

        promos = [p for p in (data.get("promos", []) if isinstance(data, dict) else [])
                  if isinstance(p, dict)]
        checks = [validate_promo(p, lines) for p in promos]
        first_kept = sum(c["ok"] for c in checks)

        retried, retry_note = {}, ""
        if not args.no_retry:
            try:
                retried = retry_lengths(prompt, fix_template, promos, checks, lines,
                                        run_dir / "raw" / f"E{n:02d}_retry.txt")
            except Exception as err:  # keep the first answers if the retry call fails
                retry_note = f"; length retry FAILED ({type(err).__name__})"
            if retried:
                fixed = sum(r["retry"] == "fixed" for r in retried.values())
                retry_note = f"; length retry fixed {fixed}/{len(retried)}"

        for k, (promo, check) in enumerate(zip(promos, checks)):
            all_candidates.append({"promo_id": f"E{n:02d}_P{k + 1}", "episode": n, **promo, **check,
                                   **retried.get(k, {})})
        note = " (cached)" if cached else ""
        print(f"Ep {n:>2}: {len(promos)} proposed, {first_kept} passed the rules{retry_note}{note}")

    # ---- save -----------------------------------------------------------
    (run_dir / "candidates.json").write_text(
        json.dumps(all_candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_info.json").write_text(json.dumps({
        "model": MODEL, "run_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": chosen, "prompt_template": template, "length_retry": not args.no_retry,
        "fix_template": fix_template}, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = ["promo_id", "episode", "ok", "problems", "retry", "total_length", "segments_timed",
              "title", "hook", "cliffhanger", "why", "hook_score", "cliffhanger_score",
              "standalone_score", "spoiler_risk"]
    with open(run_dir / "candidates.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in all_candidates:
            scores = c.get("scores") or {}
            w.writerow({
                "promo_id": c["promo_id"], "episode": c["episode"], "ok": c["ok"],
                "problems": "; ".join(c["problems"]), "retry": c.get("retry", ""),
                "total_length": c["total_length"],
                "segments_timed": " + ".join(f"{s['start']:.1f}-{s['end']:.1f}s" for s in c["segments"]),
                "title": c.get("title", ""), "hook": c.get("hook", ""), "cliffhanger": c.get("cliffhanger", ""),
                "why": c.get("why", ""), "hook_score": scores.get("hook"),
                "cliffhanger_score": scores.get("cliffhanger"), "standalone_score": scores.get("standalone"),
                "spoiler_risk": c.get("spoiler_risk", ""),
            })

    ok = sum(c["ok"] for c in all_candidates)
    print(f"\n{len(all_candidates)} candidates, {ok} passed, {len(all_candidates) - ok} rejected.")
    for c in all_candidates:
        if not c["ok"]:
            print(f"  rejected {c['promo_id']}: {'; '.join(c['problems'])}")
    print(f"Saved {run_dir / 'candidates.csv'}")


if __name__ == "__main__":
    main()