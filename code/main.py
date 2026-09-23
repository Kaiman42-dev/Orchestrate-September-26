"""Entry point: python3 code/main.py [--samples]

Reads dataset/, decides every request and writes output.csv at the repo root
(or output_samples.csv with --samples, for evaluation).
"""
from __future__ import annotations

import csv
import os
import sys

from loader import REPO_ROOT, load
from planner import decide, to_row

COLUMNS = ["request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
           "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation"]


def run(samples: bool = False) -> str:
    ds = load()
    reqs = ds.samples if samples else ds.requests
    out_path = os.path.join(REPO_ROOT, "output_samples.csv" if samples else "output.csv")
    rows = []
    for req in reqs:
        d = decide(ds, req)
        rows.append(to_row(d, ds.profiles[req.user_id].home_currency))
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return out_path


if __name__ == "__main__":
    print("wrote", run(samples="--samples" in sys.argv))
