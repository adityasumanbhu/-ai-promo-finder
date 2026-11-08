# AI Promo Finder

Finds and ranks 20-60 second promos in a folder of episode screenplays.
Each promo is one or more segments (episode + start/end timestamps, in story order) with a hook in
the first 3 seconds, a cliffhanger ending, and one line on why it works.

> The screenplays and everything in `data/` and `outputs/` are confidential and are not in this repo.

## Run it

**1. Install** (Python 3.10+):

```powershell
conda create -n kuku_promo python=3.11 -y
conda activate kuku_promo
pip install -r requirements.txt
```

**2. Add an API key.** Copy `.env.example` to `.env` and put your key in `LLM_API_KEY`.
A free Google AI Studio key works with the defaults.

```powershell
Copy-Item .env.example .env
```

**3. Run on a folder of screenplays** (one `.txt` file per episode):

```powershell
python run.py --input "path\to\screenplays" --run my_show
```

Open `outputs\my_show\promos_report.md` for the ranked top 10 with full text.

If a free-tier model is busy and a step stops, run the same command again. Every finished AI call
is cached in `data/cache`, so it continues where it stopped and nothing is paid for twice.

Options: `--episodes 1-5` for a quick test, `--skip-rank` to stop after finding promos,
`--no-retry` to turn off the length fix-up call.

## What you get (`outputs/<run>/`)

| File | What it is |
|---|---|
| `promos_report.md` | Top 10 with timestamps, hook, cliffhanger, why it works, the judge's view and full script text |
| `promos_ranked.csv` | Every promo that passed the rules, ranked, with each part of its score |
| `candidates.csv` / `.json` | Every candidate the model proposed, with pass/fail reasons |
| `raw/` | Every model answer, unedited (`E01.txt`, `E01_retry.txt`, `judge_*.txt`) |
| `run_info.json`, `rank_info.json` | Models, time, weights and the exact prompts used |

## How it works

| Step | Who | What |
|---|---|---|
| Parse | Code | `src/screenplay_parser.py` turns each screenplay into timed lines with IDs like `E01_L014`. Camera, action and dialogue are parallel tracks, so a segment runs from the earliest start to the latest end of its lines. |
| Find | AI | `src/find_promos.py` sends each episode to the generator model (`prompts/find_promos.txt`), which proposes about 3 promos as **line IDs**, never timestamps. |
| Check | Code | `src/validate.py` turns the IDs into real timestamps and rejects anything that breaks a rule: unknown lines, more than 3 segments, out of order, outside 20-60 s, hook not in the first 3 s, or the clip running more than 8 s past the cliffhanger. |
| Fix length | AI + code | The model is bad at adding up seconds. A promo outside 20-60 s goes back once with its measured segment lengths (`prompts/fix_length.txt`); the fix is kept only if it passes every rule. |
| Rank | AI + code | `src/rank.py` does not use the generator's self-scores (they are 4-5/5 for almost everything). A separate judge reads each promo cold, as a new viewer would (`prompts/judge.txt`): heats of about 12 on Gemma 4 31B, then a final of 24 on Gemini 3.8 Flash. Every heat and the final are ranked twice, in normal and reversed order, to dampen position bias. Code then adds an early-episode bonus and spoiler and "confusing" penalties, drops exact duplicates, and allows at most 2 promos per episode and no near-copies in the top 10. |
| Review | Human | A person watches the top 10, picks the final cut and notes where the tool picked badly. |

## Checking quality

`src/evaluate.py` compares a run with a hand-picked golden set: overlap with each golden promo,
same cliffhanger, and a "last 45 s of the episode" baseline. The golden set describes the script,
so it is shared on Drive with the raw outputs, not in this repo; put it in the project folder as
`golden_set.csv` (columns: promo_id, segment, episode, start_time, end_time, start_line, end_line,
hook, cliffhanger, notes, duration).

```powershell
python src/evaluate.py --run my_show
```

`src/show_episode.py` previews an episode or a clip, with a length check:
`python src/show_episode.py 3 --from E03_L010 --to E03_L030`.

## Scale and cost (50 episodes, about 145K input tokens)

| Step | Calls | Model | Time |
|---|---|---|---|
| Find + length retry | about 95 | gemini-3.5-flash-lite | about 25 min (paced for the free tier) |
| Rank, heats | 24 | gemma-4-31b-it | about 15 min |
| Rank, final | 2 | gemini-3.8-flash | about 1 min |

Everything fits in the Gemini API free tier, so a full run costs $0. To change models or provider,
edit `.env`; `src/llm.py` is the only file that calls a model.
