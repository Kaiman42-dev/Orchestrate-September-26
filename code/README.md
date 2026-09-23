# Buy or Wait? – solution

A financial decision agent that answers "can I afford this?" for every row of
`dataset/requests.csv` and writes `output.csv` at the repository root.

## Run

```bash
python3 code/main.py                 # writes ./output.csv (250 rows)
python3 code/evaluation/main.py -v   # scores the 25 solved samples + validates output.csv
```

Python 3.10+, standard library only. No install step.

The LLM answers (messages and images) are cached in `code/cache/`, so the
commands above are deterministic and need no API key. To re-extract from
scratch, delete `code/cache/*.json`, then:

```bash
cp .env.example .env                 # set GEMINI_API_KEY (never committed)
python3 code/extract_messages.py     # 215 messages -> cache/message_facts.json
python3 code/extract_images.py       # 16 images    -> cache/image_facts.json
python3 code/main.py
```

## Architecture

The LLM only reads unstructured evidence. Every financial decision is made by
deterministic code that can be tested and explained.

| Step | File | What it does |
|---|---|---|
| Load | `loader.py` | Typed loading of all CSVs; dated FX conversion that chains through currencies (e.g. ZAR -> EUR -> USD) |
| Normalize | `normalize.py` | Each event becomes history / future debit / confirmed salary / ignored (failed, cancelled, unrealized, pending credit, duplicate). Blank amounts are filled from images, never read as zero |
| Recurrence | `recurrence.py` | Finds monthly fixed-day items and fixed-cadence variable spending (groceries, transport, dining); drops stale series and "final payroll" endings |
| Read messages | `extract_messages.py` + `llm.py` | Gemini turns each message into facts from a fixed action enum (salary change, moved pay date, confirmed income, ended or unearned income, expense change, transfer, debit still due, suspicious) |
| Read images | `extract_images.py` | Gemini vision reads the net pay / final payable total of payslips, bills and receipts |
| Apply facts | `facts.py` | Validates LLM output (enum, amount must appear in the message text, message dated before the request) and applies it to the forecast |
| Forecast | `forecast.py` | Daily balance from request date to +90 days: reserved debits, confirmed salary, projected recurring income and expenses |
| Plan | `planner.py` | Builds full / partial / exact installment / wait plans, with and without spending changes; keeps safe and eligible ones; ranks them with the problem-statement order |
| Output | `main.py` | Writes the 8 required columns |
| Evaluate | `evaluation/main.py` | Field-by-field score on `sample_requests.csv` and a contract validator for `output.csv` |

### Key rules

- `amount_safe_to_pay` = lowest projected headroom above the minimum balance over the 90 days, capped at the request.
- `earliest_date_for_full_payment` = first date where one full payment keeps the 90-day balance above the minimum, before any spending change.
- Plans must keep the balance above the minimum through the desired completion date and finish by it. Installments copy a supplied option exactly; partial payment follows the two-payment rule.
- Eligibility comes from `payment_methods_user_will_consider` (`wait` needs `full_payment`).
- Spending changes only touch flexible, non-protected series in categories the user agreed to stop or reduce; `reduce_to` uses `minimum_allowed_amount`; the latest event id of the series is referenced; up to 3 changes, the least disruptive set wins.
- Ranking: completes by the deadline, no spending changes, lowest total, earliest start, fewest payments, lowest option id.

### Security

Messages and images are untrusted. Prompts ask for facts only and map
scam or instruction-like messages to `ignore_suspicious`; `facts.py` rejects
any amount the LLM returns that is not literally in the message.

## Results on the 25 solved samples

| Field | Correct |
|---|---|
| affordability_status | 22/25 |
| recommended_payment_method | 23/25 |
| payment_plan | 22/25 |
| earliest_date_for_full_payment | 22/25 |
| spending_changes_needed | 22/25 |
| amount_safe_to_pay (within 5%) | 10/25 |

Calibration choices were made on these samples only (variable spending at its
mean amount, payroll projected monthly, plan safety through the deadline).
No request id or expected value is hardcoded.

## Models

Gemini free tier via REST. Default `gemini-3.6-flash`, with automatic fallback
to `gemini-3.1-flash-lite` when a model's daily free quota is exhausted. The
model used for every call is logged in `code/cache/usage.jsonl`; see
`evaluation/usage_report.md`.
