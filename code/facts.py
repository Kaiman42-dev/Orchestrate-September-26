"""Apply facts extracted by the LLM (cached JSON) to the deterministic state.

The LLM only *reads*; this module decides. Every fact is validated before use:
- the action must be in the fixed enum of extract_messages.ACTIONS
- an amount must literally appear in the message text (guards against
  hallucinated numbers); dates must parse
- only messages sent on or before the request date are used
Conflict rule from the problem statement: an explicit amendment in a message
overrides the pattern learned from history.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
VARIABLE_INCOME_WORDS = ("commission", "bonus", "payout", "earnings", "invoice", "prize", "refund",
                         "contract payment", "project payment", "milestone", "retainer", "arrears")


@lru_cache(maxsize=1)
def _image_facts() -> dict:
    p = os.path.join(CACHE, "image_facts.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


@lru_cache(maxsize=1)
def _message_facts() -> dict:
    p = os.path.join(CACHE, "message_facts.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def image_amount(event_id: str) -> Optional[tuple[float, str]]:
    f = _image_facts().get(event_id)
    if not f or f.get("amount") in (None, 0):
        return None
    return float(f["amount"]), (f.get("currency") or "").upper()


def _digits(x: float) -> set[str]:
    s = f"{x:.2f}"
    whole, frac = s.split(".")
    forms = {whole, f"{int(whole):,}", f"{int(whole):,}".replace(",", ".")}
    if frac != "00":
        forms |= {f"{whole}.{frac}", f"{int(whole):,}.{frac}", f"{whole},{frac}"}
    return forms


def _amount_in_text(amount: float, text: str) -> bool:
    return any(form in text for form in _digits(amount))


def _parse(d: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(d[:10]) if d else None
    except ValueError:
        return None


def facts_for_user(ds, user_id: str, as_of: date) -> list[tuple[dict, dict]]:
    out = []
    for m in ds.messages:
        if m["user_id"] != user_id or m["sent_at"][:10] > as_of.isoformat():
            continue
        for e in _message_facts().get(m["message_id"], []):
            if e.get("amount") is not None and not _amount_in_text(float(e["amount"]), m["message_text"]):
                e = dict(e, amount=None)   # reject numbers the text does not contain
            out.append((m, e))
    return out


def _is_salary(f) -> bool:
    return f.amount > 0 and (f.category == "salary" or "salary" in f.label.lower())


def _is_variable_income(f) -> bool:
    label = (f.series.last.event.description if f.series else f.label).lower()
    return f.amount > 0 and any(w in label for w in VARIABLE_INCOME_WORDS)


def _home(ds, fc_user_currency: str, amount: float, currency: Optional[str], on: date) -> float:
    cur = (currency or fc_user_currency).upper()
    try:
        return ds.fx.convert(amount, cur, fc_user_currency, on)
    except KeyError:
        return amount


def apply_message_facts(ds, fc, ledger, as_of: date) -> None:
    from forecast import Flow  # local import to avoid a cycle

    home = ledger.profile.home_currency
    facts = facts_for_user(ds, ledger.profile.user_id, as_of)
    if not facts:
        return
    in_window = lambda d: d is not None and fc.start < d <= fc.end

    def salary_flows():
        return sorted([f for f in fc.flows if _is_salary(f)], key=lambda f: f.day)

    def main_salary_day() -> Optional[int]:
        s = salary_flows()
        return max(s, key=lambda f: f.amount).day.day if s else None

    def monthly(start: date, day: int):
        d = start
        while d <= fc.end:
            yield d
            y, m = (d.year + d.month // 12, d.month % 12 + 1)
            try:
                d = date(y, m, day)
            except ValueError:
                d = date(y, m, 28)

    # 1. income that stops or is not earned yet
    for _, e in facts:
        if e["action"] == "end_income":
            fc.flows = [f for f in fc.flows if not (f.amount > 0 and (f.series is not None or f.label == "salary continues monthly"))]
        if e["action"] == "exclude_variable_income":
            fc.flows = [f for f in fc.flows if not _is_variable_income(f)]

    # 2. salary amount changes (the stated amount is the total monthly salary)
    for m, e in facts:
        if e["action"] != "set_income_amount" or e.get("amount") is None:
            continue
        eff = _parse(e.get("date")) or fc.start
        sal = [f for f in salary_flows() if f.day >= eff]
        if not sal and not e.get("one_time"):
            continue
        targets = sal[:1] if e.get("one_time") else sal
        months = {(f.day.year, f.day.month) for f in targets}
        day = main_salary_day()
        keep = []
        for f in fc.flows:
            if _is_salary(f) and (f.day.year, f.day.month) in months:
                continue
            keep.append(f)
        for (y, mth) in sorted(months):
            d = [f.day for f in targets if (f.day.year, f.day.month) == (y, mth)]
            when = max(d, key=lambda x: x) if day is None else min(d, key=lambda x: abs(x.day - day))
            keep.append(Flow(when, _home(ds, home, float(e["amount"]), e.get("currency"), when),
                             m["message_id"], "salary per message", category="salary"))
        fc.flows = keep

    # 3. confirmed income credits and date moves
    for m, e in facts:
        act = e["action"]
        d = _parse(e.get("date"))
        if act == "move_income_date" and d:
            # the pay date moves to a new day of the month from the next payroll on
            later = [f for f in salary_flows() if f.day > fc.start]
            if later:
                moved = []
                for i, f in enumerate(later):
                    if i == 0:
                        when = d
                    else:
                        try:
                            when = date(f.day.year, f.day.month, d.day)
                        except ValueError:
                            when = date(f.day.year, f.day.month, 28)
                        when = max(when, moved[-1][0] + timedelta(days=1))
                    moved.append((when, f))
                ids = {id(f) for f in later}
                fc.flows = [f for f in fc.flows if id(f) not in ids]
                for when, f in moved:
                    if in_window(when):
                        fc.flows.append(Flow(when, f.amount, m["message_id"], "salary moved per message",
                                             category="salary"))
        elif act == "add_confirmed_income" and d and e.get("amount") is not None:
            amt = _home(ds, home, float(e["amount"]), e.get("currency"), d)
            dates = list(monthly(d, d.day)) if e.get("recurring") else [d]
            for when in dates:
                if not in_window(when):
                    continue
                if e.get("recurring"):
                    fc.flows = [f for f in fc.flows if not (_is_salary(f) and (f.day.year, f.day.month) == (when.year, when.month))]
                fc.flows.append(Flow(when, amt, m["message_id"], "confirmed income per message",
                                     category="salary" if e.get("recurring") else "income"))
        elif act == "add_one_off_income" and e.get("amount") is not None:
            when = d or next((f.day for f in salary_flows() if f.day > fc.start), None)
            if in_window(when):
                fc.flows.append(Flow(when, _home(ds, home, float(e["amount"]), e.get("currency"), when),
                                     m["message_id"], "one-off income per message", category="income"))

    # 4. expenses
    for m, e in facts:
        act = e["action"]
        if act == "scale_recurring_expense" and e.get("percent") and e.get("category"):
            cat = e["category"].lower()
            k = 1 + float(e["percent"]) / 100.0
            fc.flows = [Flow(f.day, f.amount * k, f.source, f.label, f.series, f.category)
                        if (f.amount < 0 and f.category == cat and f.series is not None) else f for f in fc.flows]
        elif act == "add_recurring_expense" and e.get("amount") is not None:
            start = _parse(e.get("date")) or fc.start + timedelta(days=1)
            for when in monthly(start, start.day):
                if in_window(when):
                    fc.flows.append(Flow(when, -_home(ds, home, float(e["amount"]), e.get("currency"), when),
                                         m["message_id"], "new recurring expense per message",
                                         category=(e.get("category") or "other")))
        elif act in ("debit_still_due", "debit_cancelled") and m["related_event_id"]:
            entry = ledger.by_id().get(m["related_event_id"])
            if entry is None or entry.amount_home is None:
                continue
            present = any(f.source == entry.event_id for f in fc.flows)
            if act == "debit_still_due" and not present and entry.amount_home < 0:
                retry = any(x.event.linked_event_id == entry.event_id and x.kind == "future_debit"
                            for x in ledger.entries)
                if not retry and (entry.kind == "future_debit" or "duplicate" in entry.note):
                    when = max(entry.cash_date or fc.start, fc.start + timedelta(days=1))
                    fc.flows.append(Flow(when, entry.amount_home, entry.event_id, "still due per message",
                                         category=entry.event.category))
            if act == "debit_cancelled":
                fc.flows = [f for f in fc.flows if f.source != entry.event_id]

    # 5. internal transfers: a matching future debit/credit pair nets out
    if any(e["action"] == "internal_transfer" for _, e in facts):
        future = [f for f in fc.flows if f.series is None]
        for d in [f for f in future if f.amount < 0]:
            match = next((c for c in future if c.amount > 0 and abs(c.amount + d.amount) < 0.01), None)
            if match:
                fc.flows = [f for f in fc.flows if f is not d and f is not match]
