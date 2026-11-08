"""
Rank the promos of a run with separate judge models, not the generator's own self-scores
(those sit at 4-5 out of 5 for almost everything, so they cannot separate good from bad).

Run from the project folder, after find_promos.py:
    python src/rank.py --run all50_v3

How it works:
1. Code: keep only the candidates that passed every rule, and drop exact copies (the generator
   sometimes returns the same segments twice in one episode).
2. AI, heats: shuffle them into heats of about 12. A judge model ranks each heat as a cold viewer
   would. It sees only the clip's script text: no episode number, no title, no generator scores.
   Each heat is ranked twice (normal and reversed order); the top 2 by average go to the final.
3. AI, final: a stronger judge model ranks all finalists in one call, twice: once in order and
   once in reverse order, because models tend to favour what they read first. Points are averaged.
4. Code: score = judge points + early-episode bonus - spoiler penalty - "confusing" penalty.
   Then the variety rules for the top 10: at most 2 promos from one episode, and no two promos that
   are near-copies of each other (80%+ of their seconds shared); the judge's favourite stays.

Writes to outputs/<run>/:
    promos_ranked.csv    every passing promo with its rank and the parts of its score
    promos_report.md     the top promos with full text (quotes the script: never commit or publish)
    raw/judge_*_a.txt / _b.txt  the judge's answers (normal / reversed order), unedited
    rank_info.json       models, weights, heats and the judge prompt used

A failed call stops the run; run it again and the finished calls come from the cache.
"""

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from evaluate import overlap
from llm import ask_json
from show_episode import episode_lines, format_line, load_episodes, mmss

JUDGE_FILE = Path("prompts/judge.txt")

# Models. Free-tier limits: Gemma 4 31B 30 RPM / 16K tokens per minute; 3.8-flash 5 RPM / 20 per day.
HEAT_MODEL = os.getenv("JUDGE_HEAT_MODEL", "gemma-4-31b-it")
FINAL_MODEL = os.getenv("JUDGE_FINAL_MODEL", "gemini-3.8-flash")
HEAT_SLEEP = 30    # one heat is about 5K tokens, so wait to stay under 16K tokens per minute
FINAL_SLEEP = 15   # stay under 5 requests per minute

HEAT_SIZE = 12     # promos per heat
ADVANCE = 2        # promos from each heat that reach the final
SEED = 7           # fixed shuffle: a re-run gets the same heats, so it hits the cache

# [YOUR CALL] weights. Judge points: final 1-2 (by place in the final), heats 0-1 (by place in the heat).
EARLY_BONUS = 0.15                                     # episode 1 gets +0.15, the last episode +0
SPOILER_PENALTY = {"low": 0.0, "medium": 0.05, "high": 0.15}   # uses the generator's spoiler_risk
CONFUSING_PENALTY = 0.2                                # the judge said a new viewer could not follow it
TOP_N = 10
MAX_PER_EPISODE = 2                                    # variety rule inside the top 10
NEAR_COPY = 0.8                                        # shared seconds / combined seconds


# ---- what the judge sees --------------------------------------------------

def cold_line(ln: dict) -> str:
    """One script line without its ID or timestamps."""
    if ln["type"] == "DIALOGUE":
        how = f" ({ln['delivery']})" if ln["delivery"] else ""
        return f"{ln['speaker']}{how}: {ln['text']}"
    return f"[{ln['type']}] {ln['text']}"


def segment_lines(promo: dict, lines: list) -> list:
    """The script lines of each segment: [[line, ...], [line, ...]]."""
    ids = [ln["line_id"] for ln in lines]
    return [lines[ids.index(s["start_line"]): ids.index(s["end_line"]) + 1] for s in promo["segments"]]


def clip_text(promo: dict, lines: list) -> str:
    return "\n[CUT]\n".join("\n".join(cold_line(ln) for ln in seg)
                            for seg in segment_lines(promo, lines))


# ---- one judge call -------------------------------------------------------

def is_true(value) -> bool:
    return value is True or str(value).strip().lower() in ("true", "yes")


def judge(promos: list, texts: dict, template: str, model: str, sleep: float, raw_file: Path) -> tuple:
    """Ask a judge to rank some promos. Returns (results best first, ids the judge left out, cached).

    Promos are shown as C01, C02, ... in the order given, so the judge never sees promo IDs.
    """
    labels = [f"C{k:02d}" for k in range(1, len(promos) + 1)]
    by_label = dict(zip(labels, promos))
    block = "\n\n".join(f"=== {lab} ===\n{texts[p['promo_id']]}" for lab, p in by_label.items())
    prompt = template.replace("<<COUNT>>", str(len(promos))).replace("<<PROMOS>>", block)
    data, raw, cached = ask_json(prompt, model=model, sleep=sleep)
    raw_file.write_text(raw, encoding="utf-8")

    order, seen = [], set()
    items = data.get("ranking", []) if isinstance(data, dict) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        lab = str(item.get("id", "")).strip().upper()
        if lab in by_label and lab not in seen:
            seen.add(lab)
            order.append((lab, str(item.get("reason", "")).strip(), is_true(item.get("confusing"))))
    left_out = [lab for lab in labels if lab not in seen]
    order += [(lab, "(the judge left this clip out)", False) for lab in left_out]

    n = len(order)
    results = [{"promo": by_label[lab], "place": i + 1, "points": 1 - i / (n - 1) if n > 1 else 1.0,
                "reason": reason, "confusing": confusing}
               for i, (lab, reason, confusing) in enumerate(order)]
    return results, [by_label[lab]["promo_id"] for lab in left_out], cached


def judge_or_exit(*args) -> tuple:
    try:
        return judge(*args)
    except Exception as err:
        raise SystemExit(f"Judge call failed ({type(err).__name__}: {err}). "
                         f"Run again: finished calls come from the cache.")


def judge_twice(promos: list, texts: dict, template: str, model: str, sleep: float, raw_prefix: Path) -> tuple:
    """Rank promos twice, in the given order and reversed, to dampen the judge's position bias.

    Returns ({promo_id: {"place": "a/b", "points": average, "reason", "confusing"}},
             ids left out, how many of the top places the two passes agree on, all cached?).
    """
    passes, left_out, all_cached = [], [], True
    for name, order in (("a", promos), ("b", promos[::-1])):
        results, missing, cached = judge_or_exit(order, texts, template, model, sleep,
                                                 raw_prefix.with_name(f"{raw_prefix.name}_{name}.txt"))
        passes.append({r["promo"]["promo_id"]: r for r in results})
        left_out += missing
        all_cached &= cached
    combined = {}
    for pid, a in passes[0].items():
        b = passes[1][pid]
        combined[pid] = {"place": f"{a['place']}/{b['place']}", "points": (a["points"] + b["points"]) / 2,
                         "reason": a["reason"], "confusing": a["confusing"] or b["confusing"]}
    return combined, left_out, passes, all_cached


def top_agreement(passes: list, n: int) -> int:
    """How many promos are in the top n of both passes."""
    tops = [{pid for pid, r in p.items() if r["place"] <= n} for p in passes]
    return len(tops[0] & tops[1])


# ---- scoring --------------------------------------------------------------

def early_bonus(episode: int, last_episode: int) -> float:
    if last_episode <= 1:
        return 0.0
    return EARLY_BONUS * (1 - (episode - 1) / (last_episode - 1))


def spans(c: dict) -> list:
    return [(s["start"], s["end"]) for s in c["segments"]]


def near_copy_of(c: dict, chosen: list) -> str:
    """promo_id of a chosen promo that c mostly repeats, or ''."""
    for other in chosen:
        if other["episode"] == c["episode"] and overlap(spans(c), spans(other)) >= NEAR_COPY:
            return other["promo_id"]
    return ""


def apply_variety(ranked: list) -> list:
    """Top TOP_N: at most MAX_PER_EPISODE per episode and no near-copies; the rest keep their order."""
    top, rest, per_episode = [], [], Counter()
    for c in ranked:
        if len(top) >= TOP_N:
            rest.append(c)
            continue
        copy_of = near_copy_of(c, top)
        if per_episode[c["episode"]] >= MAX_PER_EPISODE:
            c["variety_moved"] = f"3rd+ promo from episode {c['episode']}"
            rest.append(c)
        elif copy_of:
            c["variety_moved"] = f"near-copy of {copy_of}"
            rest.append(c)
        else:
            top.append(c)
            per_episode[c["episode"]] += 1
    return top + rest


def drop_exact_copies(cands: list) -> tuple:
    """Keep the first of any promos with identical segments. Returns (kept, {copy_id: kept_id})."""
    first, kept, copies = {}, [], {}
    for c in cands:
        key = (c["episode"], tuple((s["start_line"], s["end_line"]) for s in c["segments"]))
        if key in first:
            copies[c["promo_id"]] = first[key]
        else:
            first[key] = c["promo_id"]
            kept.append(c)
    return kept, copies


# ---- report ---------------------------------------------------------------

def timed_segments(c: dict) -> str:
    return " + ".join(f"{mmss(s['start'])}-{mmss(s['end'])}" for s in c["segments"])


def write_report(path: Path, ranked: list, episodes: dict, info: dict) -> None:
    out = [f"# Ranked promos: {info['run']}", "",
           "Confidential: quotes the screenplay. Never commit or publish this file.", "",
           f"- Promos ranked: {len(ranked)} (passed every rule) | heats: {info['heats']} on "
           f"`{info['heat_model']}` | finalists: {info['finalists']} on `{info['final_model']}` "
           f"(every heat and the final ranked twice, normal and reversed order)",
           f"- Heats where both orders sent the same {info['advance']} promos through: "
           f"{info['heats_where_passes_agree']}/{info['heats']}",
           f"- Final top-{TOP_N} agreement between the two orders: {info['final_top_overlap']}/{TOP_N}",
           f"- Weights: early-episode bonus up to {EARLY_BONUS}, spoiler penalty {SPOILER_PENALTY}, "
           f"confusing penalty {CONFUSING_PENALTY}, at most {MAX_PER_EPISODE} per episode in the top {TOP_N}",
           "", f"## Top {TOP_N}", "",
           "| Rank | Promo | Ep | Length | Segments | Score | Judge | Bonus | Spoiler | Title |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in ranked[:TOP_N]:
        out.append(f"| {c['rank']} | {c['promo_id']} | {c['episode']} | {c['total_length']}s | "
                   f"{len(c['segments'])} | {c['score']:.2f} | {c['judge_points']:.2f} | "
                   f"+{c['early_bonus']:.2f} | {c.get('spoiler_risk', '')} | {c.get('title', '')} |")

    for c in ranked[:TOP_N]:
        lines = episode_lines(episodes[c["episode"]])
        out += ["", f"### {c['rank']}. {c['promo_id']}: {c.get('title', '')}", "",
                f"- Episode {c['episode']}, {timed_segments(c)} ({c['total_length']}s, "
                f"{len(c['segments'])} segment{'s' if len(c['segments']) > 1 else ''})",
                f"- Hook: {c.get('hook', '')}",
                f"- Cliffhanger: {c.get('cliffhanger', '')}",
                f"- Why it works (generator): {c.get('why', '')}",
                f"- Judge, cold viewer: {c['judge_reason']}" + (" **[judge: confusing]**" if c["confusing"] else ""),
                "", "```"]
        for k, seg in enumerate(segment_lines(c, lines)):
            if k:
                out.append("--- CUT ---")
            out += [format_line(ln) for ln in seg]
        out.append("```")

    out += ["", f"## Ranks {TOP_N + 1}-{2 * TOP_N}", "", "| Rank | Promo | Ep | Score | Judge says |", "|---|---|---|---|---|"]
    for c in ranked[TOP_N:2 * TOP_N]:
        out.append(f"| {c['rank']} | {c['promo_id']} | {c['episode']} | {c['score']:.2f} | {c['judge_reason']} |")

    flags = [c for c in ranked if c["confusing"] and c["rank"] <= 2 * TOP_N]
    moved = [c for c in ranked if c.get("variety_moved")]
    out += ["", "## Automatic flags", ""]
    out += [f"- {c['promo_id']} (rank {c['rank']}): the judge marked it confusing for a new viewer." for c in flags]
    out += [f"- {c['promo_id']} (rank {c['rank']}): moved out of the top {TOP_N} ({c['variety_moved']})."
            for c in moved]
    out += [f"- {pid}: the judge left it out of its answer, so it was put last." for pid in info["left_out"]]
    out += [f"- {pid}: exact copy of {kept}, removed before judging." for pid, kept in info["exact_copies"].items()]
    if not (flags or moved or info["left_out"] or info["exact_copies"]):
        out.append("- none")
    out += ["", "## Where the tool picks badly", "", "[YOUR CALL: your own review of the ranking]", ""]
    path.write_text("\n".join(out), encoding="utf-8")


# ---- main -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a folder in outputs/ made by find_promos.py")
    args = ap.parse_args()

    run_dir = Path("outputs") / args.run
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    template = JUDGE_FILE.read_text(encoding="utf-8")
    episodes = load_episodes()
    last_episode = max(episodes)

    cands = [c for c in json.loads((run_dir / "candidates.json").read_text(encoding="utf-8")) if c["ok"]]
    if not cands:
        raise SystemExit("No passing candidates to rank.")
    # Shuffle and split BEFORE dropping copies, so removing a copy only changes its own heat.
    shuffled = cands[:]
    random.Random(SEED).shuffle(shuffled)
    n_heats = math.ceil(len(shuffled) / HEAT_SIZE)
    cands, exact_copies = drop_exact_copies(cands)
    keep = {c["promo_id"] for c in cands}
    heats = [[c for c in shuffled[k::n_heats] if c["promo_id"] in keep] for k in range(n_heats)]
    shuffled = [c for c in shuffled if c["promo_id"] in keep]
    if exact_copies:
        print(f"Dropped {len(exact_copies)} exact copies: "
              + ", ".join(f"{a} (= {b})" for a, b in exact_copies.items()))

    texts = {c["promo_id"]: clip_text(c, episode_lines(episodes[c["episode"]])) for c in cands}
    for c in cands:
        c.update(heat=None, heat_place=None, final_place=None, variety_moved="")
    by_id = {c["promo_id"]: c for c in cands}
    left_out = []

    # 1. heats (skipped when everything fits in the final)
    if n_heats * ADVANCE >= len(shuffled):
        n_heats, finalists, heat_agree = 0, shuffled, None
        for c in cands:
            c.update(heat_points=1.0, heat_reason="", confusing=False)
    else:   # heats were split above: even sizes, mixed episodes
        finalists, heat_agree = [], 0
        print(f"Heats: {n_heats} x ~{HEAT_SIZE} promos on {HEAT_MODEL}, each ranked twice")
        for h, heat in enumerate(heats, 1):
            combined, missing, passes, cached = judge_twice(heat, texts, template, HEAT_MODEL, HEAT_SLEEP,
                                                            run_dir / "raw" / f"judge_heat_{h:02d}")
            left_out += missing
            for pid, r in combined.items():
                by_id[pid].update(heat=h, heat_place=r["place"], heat_points=r["points"],
                                  heat_reason=r["reason"], confusing=r["confusing"])
            through = sorted(heat, key=lambda c: -c["heat_points"])[:ADVANCE]
            finalists += through
            agree = top_agreement(passes, ADVANCE)
            heat_agree += agree == ADVANCE
            print(f"  heat {h:>2}: {len(heat)} ranked, through: {', '.join(c['promo_id'] for c in through)}"
                  f" (passes agree on {agree}/{ADVANCE}){', left out ' + str(len(missing)) if missing else ''}"
                  f"{' (cached)' if cached else ''}")

    # 2. final, twice: in order and reversed
    print(f"Final: {len(finalists)} promos on {FINAL_MODEL}, ranked twice")
    combined, missing, passes, cached = judge_twice(finalists, texts, template, FINAL_MODEL, FINAL_SLEEP,
                                                    run_dir / "raw" / "judge_final")
    left_out += missing
    top_overlap = top_agreement(passes, TOP_N)
    print(f"  done{' (cached)' if cached else ''}" + (f", left out {len(missing)}" if missing else ""))
    for pid, r in combined.items():
        by_id[pid].update(final_place=r["place"], final_points=r["points"],
                          final_reason=r["reason"], confusing=r["confusing"])

    # 3. score, sort, variety rule
    for c in cands:
        in_final = c.get("final_points") is not None
        c["judge_points"] = 1 + c["final_points"] if in_final else c["heat_points"]
        c["judge_reason"] = c["final_reason"] if in_final else c["heat_reason"]
        c["early_bonus"] = early_bonus(c["episode"], last_episode)
        c["spoiler_penalty"] = SPOILER_PENALTY.get(str(c.get("spoiler_risk", "")).lower(), 0.0)
        c["confusing_penalty"] = CONFUSING_PENALTY if c["confusing"] else 0.0
        c["score"] = c["judge_points"] + c["early_bonus"] - c["spoiler_penalty"] - c["confusing_penalty"]
    ranked = apply_variety(sorted(cands, key=lambda c: -c["score"]))
    for k, c in enumerate(ranked, 1):
        c["rank"] = k

    # 4. save
    fields = ["rank", "promo_id", "episode", "score", "judge_points", "final_place", "heat", "heat_place",
              "early_bonus", "spoiler_risk", "spoiler_penalty", "confusing", "variety_moved",
              "total_length", "segments_timed", "title", "hook", "cliffhanger", "why", "judge_reason"]
    with open(run_dir / "promos_ranked.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for c in ranked:
            w.writerow({**c, "score": round(c["score"], 3), "judge_points": round(c["judge_points"], 3),
                        "early_bonus": round(c["early_bonus"], 3), "segments_timed": timed_segments(c)})

    info = {"run": args.run, "ranked_at": datetime.now().isoformat(timespec="seconds"),
            "heat_model": HEAT_MODEL, "final_model": FINAL_MODEL, "heats": n_heats,
            "heat_size": HEAT_SIZE, "advance": ADVANCE, "seed": SEED, "finalists": len(finalists),
            "final_top_overlap": top_overlap, "heats_where_passes_agree": heat_agree,
            "left_out": sorted(set(left_out)),
            "exact_copies": exact_copies, "near_copy": NEAR_COPY,
            "weights": {"early_bonus": EARLY_BONUS, "spoiler_penalty": SPOILER_PENALTY,
                        "confusing_penalty": CONFUSING_PENALTY, "top_n": TOP_N,
                        "max_per_episode": MAX_PER_EPISODE},
            "judge_template": template}
    (run_dir / "rank_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(run_dir / "promos_report.md", ranked, episodes, info)

    if heat_agree is not None:
        print(f"\nHeats where both orders sent the same {ADVANCE} promos through: {heat_agree}/{n_heats}")
    print(f"Final top-{TOP_N} agreement between the two orders: {top_overlap}/{TOP_N}")
    print(f"Top {TOP_N}: " + ", ".join(f"{c['promo_id']}" for c in ranked[:TOP_N]))
    print(f"Saved {run_dir / 'promos_ranked.csv'} and {run_dir / 'promos_report.md'}")


if __name__ == "__main__":
    main()
