"""
The only file that talks to an AI model. To swap models, edit .env, not the code.

.env (in the project folder, never committed):
    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    LLM_API_KEY=your-key
    LLM_MODEL=gemini-2.5-flash
    LLM_SLEEP_SECONDS=7        # optional: pause between calls to stay under free-tier limits

Any provider with an OpenAI-compatible API works (Gemini, Groq, OpenAI, OpenRouter, ...).
Every answer is cached in data/cache, so re-running costs nothing and gives the same result.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIError, BadRequestError, RateLimitError

load_dotenv()
MODEL = os.getenv("LLM_MODEL", "")
BASE_URL = os.getenv("LLM_BASE_URL", "")
API_KEY = os.getenv("LLM_API_KEY", "")
SLEEP = float(os.getenv("LLM_SLEEP_SECONDS", "7"))
CACHE_DIR = Path("data/cache")

_client = None


def _client_or_exit() -> OpenAI:
    global _client
    if _client is None:
        if not (BASE_URL and API_KEY and MODEL):
            raise SystemExit("Missing settings: put LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env")
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


def extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in ```json fences or add a sentence. Pull out the object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the model's answer")
    return json.loads(text[start:end + 1])


def _call(prompt: str, temperature: float, model: str) -> str:
    client = _client_or_exit()
    kwargs = dict(model=model, temperature=temperature,
                  messages=[{"role": "user", "content": prompt}])
    attempts = 6  # free-tier servers often answer 429/503 for a minute or two
    for attempt in range(attempts):
        try:
            try:
                resp = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
            except BadRequestError:  # some providers don't support JSON mode; the prompt still asks for JSON
                resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except (RateLimitError, APIError) as err:
            if attempt == attempts - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f"    model busy or rate-limited ({type(err).__name__}); retrying in {wait}s")
            time.sleep(wait)
    return ""


def ask_json(prompt: str, temperature: float = 0.2, model: str = "", sleep: float = None) -> tuple:
    """Send a prompt, return (parsed_json, raw_text, from_cache).

    model: leave empty to use LLM_MODEL from .env (the judge in rank.py passes its own).
    sleep: seconds to wait after a real call; default LLM_SLEEP_SECONDS.
    """
    model = model or MODEL
    key = hashlib.sha256(f"{model}\n{temperature}\n{prompt}".encode("utf-8")).hexdigest()[:24]
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))["raw"]
        return extract_json(raw), raw, True

    raw = _call(prompt, temperature, model)
    data = extract_json(raw)  # only cache answers we could read
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"model": model, "raw": raw}, ensure_ascii=False), encoding="utf-8")
    time.sleep(SLEEP if sleep is None else sleep)
    return data, raw, False