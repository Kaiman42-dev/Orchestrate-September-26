"""Evaluation workflow.

    python3 code/evaluation/main.py            # score on sample_requests + validate output.csv
    python3 code/evaluation/main.py --verbose  # also print every mismatch

1. Runs the full pipeline on dataset/sample_requests.csv (25 solved examples)
   and scores each output field against the expected values.
2. Validates the root output.csv (if present) against the contract in
   problem_statement.md: columns and order, one row per request, allowed
   values, 0 <= amount_safe_to_pay <= requested_amount, plan format, partial
   and installment rules, spending changes on flexible recurring events.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from loader import REPO_ROOT, load  # noqa: E402
from main import COLUMNS  # noqa: E402
from planner import decide, to_row  # noqa: E402

STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
PLAN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}:\d+(\.\d{1,2})?$")
CHANGE_RE = re.compile(r"^(stop:event_\d+|reduce_to:event_\d+:\d+(\.\d{1,2})?)$")
AMOUNT_TOL = 0.05  # relative tolerance for amount_safe_to_pay


def _num(x: str) -> float:
    return float(x) if x not in ("", None) else 0.0


def score_samples(verbose: bool) -> dict:
    ds = load()
    fields = ["amount_safe_to_pay", "affordability_status", "recommended_payment_method",
              "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed"]
    hits = {f: 0 for f in fields}
    all_ok = 0
    for req in ds.samples:
        row = to_row(decide(ds, req), ds.profiles[req.user_id].home_currency)
        exp = req.expected
        miss = []
        for f in fields:
            if f == "amount_safe_to_pay":
                e, g = _num(exp[f]), _num(row[f])
                ok = abs(e - g) <= max(0.01, AMOUNT_TOL * max(e, 1e-9))
            else:
                ok = (exp[f] or "").strip() == (row[f] or "").strip()
            hits[f] += ok
            if not ok:
                miss.append(f"{f}: got {row[f]!r} expected {exp[f]!r}")
        all_ok += not miss
        if verbose and miss:
            print(f"- {req.request_id}")
            for m in miss:
                print(f"    {m}")
    n = len(ds.samples)
    print(f"\nSample evaluation ({n} requests, amount tolerance {AMOUNT_TOL:.0%})")
    for f in fields:
        print(f"  {f:32s} {hits[f]:3d}/{n}  ({hits[f] / n:.0%})")
    print(f"  {'all fields correct':32s} {all_ok:3d}/{n}")
    return hits


def validate_output(path: str) -> list[str]:
    ds = load()
    errors: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            errors.append(f"columns {reader.fieldnames} != {COLUMNS}")
        rows = list(reader)
    by_id = {r["request_id"]: r for r in rows}
    reqs = {r.request_id: r for r in ds.requests}
    if set(by_id) != set(reqs) or len(rows) != len(reqs):
        errors.append(f"expected {len(reqs)} rows matching requests.csv, got {len(rows)}")
    for rid, r in by_id.items():
        req = reqs.get(rid)
        if req is None:
            continue
        err = lambda m: errors.append(f"{rid}: {m}")
        safe = _num(r["amount_safe_to_pay"])
        if not (0 <= safe <= req.amount + 0.005):
            err(f"amount_safe_to_pay {safe} outside [0, {req.amount}]")
        if r["affordability_status"] not in STATUSES:
            err(f"bad status {r['affordability_status']}")
        method = r["recommended_payment_method"]
        if method not in METHODS:
            err(f"bad method {method}")
        plan = r["payment_plan"]
        payments = []
        if plan != "none":
            parts = plan.split("|")
            if not all(PLAN_RE.match(p) for p in parts):
                err(f"bad plan format {plan}")
                continue
            payments = [(date.fromisoformat(p.split(":")[0]), float(p.split(":")[1])) for p in parts]
            if [d for d, _ in payments] != sorted(d for d, _ in payments):
                err("plan not chronological")
        if (method == "not_recommended") != (plan == "none"):
            err("plan/method mismatch")
        ed = r["earliest_date_for_full_payment"]
        if r["affordability_status"] == "affordable_now" and ed != req.request_date.isoformat():
            err("affordable_now needs earliest date == request_date")
        if method == "partial_payment":
            if r["affordability_status"] != "affordable_with_plan" or not req.allows_partial:
                err("partial_payment not allowed here")
            if len(payments) != 2 or abs(sum(a for _, a in payments) - req.amount) > 0.01:
                err("partial plan must be two payments summing to the request")
        if method == "installments":
            opts = ds.options_by_request.get(rid, [])
            match = any(o.method == "installments" and o.n_payments == len(payments)
                        and o.first_date == payments[0][0]
                        and all(abs(a - o.payment_amount) < 0.01 for _, a in payments) for o in opts)
            if not match:
                err("installments do not match a supplied option")
        changes = r["spending_changes_needed"]
        if changes != "none":
            items = changes.split("|")
            if len(items) > 3 or not all(CHANGE_RE.match(c) for c in items):
                err(f"bad spending changes {changes}")
            ids = [c.split(":")[1] for c in items]
            if len(ids) != len(set(ids)):
                err("same event both stopped and reduced")
            for eid in ids:
                ev = ds.events_by_id.get(eid)
                if ev is None or ev.user_id != req.user_id or ev.flexibility == "fixed":
                    err(f"{eid} is not a flexible event of {req.user_id}")
        if not r["decision_explanation"].strip():
            err("empty explanation")
    return errors


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    score_samples(verbose)
    out = os.path.join(REPO_ROOT, "output.csv")
    if os.path.exists(out):
        errs = validate_output(out)
        print(f"\noutput.csv validation: {'OK' if not errs else f'{len(errs)} problem(s)'}")
        for e in errs[:30]:
            print("  ", e)
