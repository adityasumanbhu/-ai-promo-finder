"""
Rule checks for promo candidates. Pure code, no AI.

The LLM only ever names line IDs. This file turns those IDs into real timestamps
and rejects anything that breaks the brief's rules, so a wrong or invented answer
from the model can never reach the final list.
"""

MIN_TOTAL, MAX_TOTAL = 20.0, 60.0   # brief: a promo is 20-60 seconds
HOOK_WINDOW = 3.0                   # brief: hook in the first 3 seconds
CLIFF_WINDOW = 8.0                  # my rule: cliffhanger line must end in the last 8 seconds
MAX_SEGMENTS = 3                    # my assumption for v1


def segment_span(lines: list, start_id: str, end_id: str) -> tuple:
    """Return (start_seconds, end_seconds) for the lines start_id..end_id, inclusive.

    Camera, action and dialogue overlap, so we use the earliest start and the
    latest end of every line inside the range, not just the first and last line.
    """
    ids = [ln["line_id"] for ln in lines]
    i, j = ids.index(start_id), ids.index(end_id)
    chunk = lines[i:j + 1]
    return min(ln["start"] for ln in chunk), max(ln["end"] for ln in chunk)


def validate_promo(promo: dict, lines: list) -> dict:
    """Check one candidate. Returns {"ok", "problems", "segments", "total_length"}."""
    problems = []
    index = {ln["line_id"]: k for k, ln in enumerate(lines)}
    by_id = {ln["line_id"]: ln for ln in lines}

    segments = promo.get("segments") or []
    if not segments:
        return {"ok": False, "problems": ["no segments"], "segments": [], "total_length": 0.0}
    if len(segments) > MAX_SEGMENTS:
        problems.append(f"{len(segments)} segments (max {MAX_SEGMENTS})")

    timed = []
    for n, seg in enumerate(segments, 1):
        a, b = seg.get("start_line"), seg.get("end_line")
        if a not in index or b not in index:
            problems.append(f"segment {n}: unknown line ID ({a} / {b})")
            continue
        if index[b] < index[a]:
            problems.append(f"segment {n}: ends before it starts")
            continue
        start, end = segment_span(lines, a, b)
        timed.append({"start_line": a, "end_line": b, "start": start, "end": end,
                      "length": round(end - start, 1), "first": index[a], "last": index[b]})

    for prev, cur in zip(timed, timed[1:]):
        if cur["first"] <= prev["last"]:
            problems.append("segments overlap or are out of story order")

    total = round(sum(s["length"] for s in timed), 1)
    if timed and not (MIN_TOTAL <= total <= MAX_TOTAL):
        problems.append(f"length {total}s (must be {MIN_TOTAL:.0f}-{MAX_TOTAL:.0f}s)")

    hook = promo.get("hook_line")
    if timed:
        first, last = timed[0], timed[-1]
        if hook not in by_id or not (first["first"] <= index[hook] <= first["last"]):
            problems.append(f"hook_line {hook} is not inside the first segment")
        elif by_id[hook]["start"] >= first["start"] + HOOK_WINDOW:
            late = by_id[hook]["start"] - first["start"]
            problems.append(f"hook starts {late:.1f}s in (must be under {HOOK_WINDOW:.0f}s)")

        cliff = promo.get("cliffhanger_line")
        if cliff not in by_id or not (last["first"] <= index[cliff] <= last["last"]):
            problems.append(f"cliffhanger_line {cliff} is not inside the last segment")
        elif by_id[cliff]["end"] < last["end"] - CLIFF_WINDOW:
            early = last["end"] - by_id[cliff]["end"]
            problems.append(f"clip runs {early:.1f}s past the cliffhanger (max {CLIFF_WINDOW:.0f}s)")

    for s in timed:
        s.pop("first"), s.pop("last")
    return {"ok": not problems, "problems": problems, "segments": timed, "total_length": total}