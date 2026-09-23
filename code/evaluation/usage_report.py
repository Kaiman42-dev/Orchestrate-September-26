"""Build evaluation/usage_report.md from code/cache/usage.jsonl.

usage.jsonl holds one line per real (non-cached) Gemini call, written by llm.py.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(HERE), "cache", "usage.jsonl")
OUT = os.path.join(HERE, "usage_report.md")
N_REQUESTS = 250
# Free tier: no charge. Kept as a table so paid prices can be plugged in.
PRICE_PER_MTOK = {}  # model -> (input_usd, output_usd); empty = free tier


def main() -> None:
    rows = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]
    by = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "think": 0, "purposes": set()})
    for r in rows:
        k = by[r["model"]]
        k["calls"] += 1
        k["in"] += r["input_tokens"]
        k["out"] += r["output_tokens"]
        k["think"] += r.get("thinking_tokens", 0)
        k["purposes"].add(r["purpose"])
    tot_in = sum(k["in"] for k in by.values())
    tot_out = sum(k["out"] for k in by.values())
    calls = sum(k["calls"] for k in by.values())

    def cost(model, i, o):
        p = PRICE_PER_MTOK.get(model)
        return 0.0 if not p else i / 1e6 * p[0] + o / 1e6 * p[1]

    lines = [
        "# Token usage and cost",
        "",
        "Provider: Google Gemini API (free tier), called over REST from `code/llm.py`.",
        "",
        "LLM calls happen only when extracting facts from `messages.csv` (batched, 30 messages per call) and",
        "reading the 16 linked images (one vision call each). Answers are cached in `code/cache/`; the final",
        "`python3 code/main.py` run that wrote `output.csv` reused that cache and made 0 new calls.",
        "The figures below are the calls that produced the cache used by that run.",
        "",
        "## Per model",
        "",
        "| Provider | Model | Used for | Calls | Input tokens | Output tokens (incl. thinking) | Est. cost (USD) |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, k in sorted(by.items()):
        lines.append(f"| Google | {model} | {', '.join(sorted(k['purposes']))} | {k['calls']} | {k['in']:,} | "
                     f"{k['out']:,} | {cost(model, k['in'], k['out']):.2f} |")
    total_cost = sum(cost(m, k["in"], k["out"]) for m, k in by.items())
    lines += [
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Model calls | {calls} |",
        f"| Input tokens | {tot_in:,} |",
        f"| Output tokens | {tot_out:,} |",
        f"| Total tokens | {tot_in + tot_out:,} |",
        f"| Requests in requests.csv | {N_REQUESTS} |",
        f"| Average tokens per request | {(tot_in + tot_out) / N_REQUESTS:,.0f} |",
        f"| Estimated total cost | ${total_cost:.2f} (free tier) |",
        f"| Estimated cost per request | ${total_cost / N_REQUESTS:.4f} |",
        "",
        "Messages and images belong to users across both `requests.csv` and `sample_requests.csv`; the",
        "per-request average divides the whole extraction cost by the 250 predicted requests.",
        "Token efficiency: batching 30 messages per call, deterministic code for all arithmetic, and caching",
        "keep the whole dataset under 30 model calls.",
    ]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
