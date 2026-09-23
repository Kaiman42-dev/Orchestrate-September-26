# Token usage and cost

Provider: Google Gemini API (free tier), called over REST from `code/llm.py`.

LLM calls happen only when extracting facts from `messages.csv` (batched, 30 messages per call) and
reading the 16 linked images (one vision call each). Answers are cached in `code/cache/`; the final
`python3 code/main.py` run that wrote `output.csv` reused that cache and made 0 new calls.
The figures below are the calls that produced the cache used by that run.

## Per model

| Provider | Model | Used for | Calls | Input tokens | Output tokens (incl. thinking) | Est. cost (USD) |
|---|---|---|---|---|---|---|
| Google | gemini-3.1-flash-lite | image, messages | 24 | 53,317 | 21,460 | 0.00 |

## Overall

| Metric | Value |
|---|---|
| Model calls | 24 |
| Input tokens | 53,317 |
| Output tokens | 21,460 |
| Total tokens | 74,777 |
| Requests in requests.csv | 250 |
| Average tokens per request | 299 |
| Estimated total cost | $0.00 (free tier) |
| Estimated cost per request | $0.0000 |

Messages and images belong to users across both `requests.csv` and `sample_requests.csv`; the
per-request average divides the whole extraction cost by the 250 predicted requests.
Token efficiency: batching 30 messages per call, deterministic code for all arithmetic, and caching
keep the whole dataset under 30 model calls.
