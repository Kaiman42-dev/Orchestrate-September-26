"""Minimal Gemini client (REST, stdlib only) with an on-disk cache and usage log.

- The API key is read from the GEMINI_API_KEY environment variable (or a
  local, gitignored .env file at the repo root). It is never written anywhere.
- Every call is cached by a hash of (prompt, images, schema) in code/cache/,
  so re-running the pipeline is deterministic and free.
- Token usage of real calls is appended to code/cache/usage.jsonl and summed
  into evaluation/usage_report.md.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from loader import REPO_ROOT

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
USAGE_LOG = os.path.join(CACHE_DIR, "usage.jsonl")
_EXHAUSTED: set[str] = set()
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _load_env() -> None:
    path = os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
# Free-tier quotas are per model and per day; when one model is exhausted we
# fall back to the next one. The model actually used is stored with the answer.
FALLBACK_MODELS = [m for m in os.environ.get(
    "GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite,gemini-3.5-flash-lite,gemini-3.1-flash-lite-preview").split(",") if m]


def _key(prompt: str, images: list[bytes], schema: Optional[dict]) -> str:
    h = hashlib.sha256()
    for part in (prompt, json.dumps(schema, sort_keys=True)):
        h.update(part.encode("utf-8"))
    for img in images:
        h.update(hashlib.sha256(img).digest())
    return h.hexdigest()[:24]


def generate_json(prompt: str, schema: dict, images: Optional[list[tuple[str, bytes]]] = None,
                  purpose: str = "", model: Optional[str] = None, retries: int = 4) -> dict:
    """Return the parsed JSON answer. Cached; raises if no cache and no key."""
    images = images or []
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _key(prompt, [b for _, b in images], schema)
    path = os.path.join(CACHE_DIR, f"{purpose or 'call'}_{key}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))["answer"]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set and no cached answer for this call")

    parts = [{"text": prompt}] + [
        {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}} for mime, data in images]
    body = json.dumps({
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }).encode()
    data = None
    last_error = ""
    for model in [model or MODEL] + [m for m in FALLBACK_MODELS if m != (model or MODEL)]:
        if model in _EXHAUSTED:
            continue
        delay = 10.0
        for attempt in range(retries):
            req = urllib.request.Request(API.format(model=model), data=body,
                                         headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.load(resp)
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")
                last_error = f"{model}: HTTP {e.code} {msg[:200]}"
                if e.code == 429 and "PerDay" in msg:
                    _EXHAUSTED.add(model)   # daily quota: try the next model
                    break
                if e.code in (404,):
                    break
                if e.code in (429, 500, 503) and attempt < retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                break
        if data is not None:
            break
    if data is None:
        raise RuntimeError(f"Gemini call failed: {last_error}")
    text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    answer = json.loads(text)
    usage = data.get("usageMetadata", {})
    record = {
        "purpose": purpose, "model": model, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
        "thinking_tokens": usage.get("thoughtsTokenCount", 0),
    }
    json.dump({"answer": answer, "usage": record}, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(USAGE_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return answer
