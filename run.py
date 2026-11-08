"""
One command: screenplays in, ranked promos out.

    python run.py --input path/to/screenplay_folder --run my_show

Steps (each one is also a script you can run on its own):
    1. src/screenplay_parser.py  .txt files -> data/processed/<run>/episodes.json   (code)
    2. src/find_promos.py        candidates per episode, rule checks, length retry   (AI + code)
    3. src/rank.py               cold-viewer judge ranking, bonuses, variety rules   (AI + code)

Results in outputs/<run>/: promos_report.md, promos_ranked.csv, candidates.csv, raw/.
If a step fails (for example a busy free-tier model), run the same command again:
every finished AI call comes back from data/cache, so nothing is paid for twice.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(title: str, script: str, *args: str, env: dict) -> None:
    print(f"\n=== {title} ===", flush=True)
    result = subprocess.run([sys.executable, str(ROOT / "src" / script), *args], cwd=ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"\nStopped at '{title}'. Fix the message above, or just run the same "
                         f"command again (finished AI calls are cached).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Find and rank promos in a folder of screenplay .txt files.")
    ap.add_argument("--input", required=True, help="folder with one .txt screenplay per episode")
    ap.add_argument("--run", required=True, help="a name for this run; results go to outputs/<run>/")
    ap.add_argument("--episodes", default="all", help="'all' (default), '3', '1-5' or '1,3,12'")
    ap.add_argument("--no-retry", action="store_true", help="skip the length-fix call in find_promos")
    ap.add_argument("--skip-rank", action="store_true", help="stop after finding promos")
    args = ap.parse_args()

    if not Path(args.input).is_dir():
        raise SystemExit(f"Input folder not found: {args.input}")
    if not (ROOT / ".env").exists():
        raise SystemExit("No .env file. Copy .env.example to .env and add your API key (see README).")

    data_dir = Path("data/processed") / args.run
    env = {**os.environ, "PROMO_DATA": str(data_dir), "PYTHONIOENCODING": "utf-8"}

    step("1/3 Parse screenplays", "screenplay_parser.py",
         "--input", str(Path(args.input).resolve()), "--out", str(data_dir), env=env)
    step("2/3 Find promos", "find_promos.py", "--episodes", args.episodes, "--run", args.run,
         *(["--no-retry"] if args.no_retry else []), env=env)
    if not args.skip_rank:
        step("3/3 Rank promos", "rank.py", "--run", args.run, env=env)

    out = ROOT / "outputs" / args.run
    print(f"\nDone. Open {out / 'promos_report.md'} (top promos with full text)")
    print(f"      and {out / 'promos_ranked.csv'} (every promo, ranked).")


if __name__ == "__main__":
    main()
