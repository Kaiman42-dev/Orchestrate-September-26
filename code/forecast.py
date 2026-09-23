"""90-day cash forecast and the two capacity numbers derived from it.

    amount_safe_to_pay           max X payable on request_date such that the
                                 projected balance never drops below the
                                 minimum balance over the forecast window
    earliest_date_for_full_payment
                                 first date D where paying the full amount
                                 on D keeps the balance >= minimum from D to
                                 the end of the window

Both are computed *before* optional spending changes. Spending changes are
passed as `adjustments` when the planner tests plans that use them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from loader import Dataset
from normalize import Ledger, normalize_user
from recurrence import Series, detect

HORIZON_DAYS = 90
EPS = 1e-6


@dataclass
class Policy:
    """Knobs calibrated on sample_requests (see evaluation)."""
    project_recurring_income: bool = True    # monthly payroll with history keeps coming
    variable_amount: str = "mean"             # 'last' | 'mean' | 'max' for non-fixed series
    include_first_day_flows: bool = False    # flows dated == request_date already in balance


@dataclass
class Flow:
    day: date
    amount: float          # signed
    source: str            # event id / series key
    label: str
    series: Optional[Series] = None


@dataclass
class Forecast:
    start: date
    end: date
    opening: float
    min_balance: float
    flows: list[Flow] = field(default_factory=list)

    def balances(self, extra: list[tuple[date, float]] = ()) -> list[tuple[date, float]]:
        """End-of-day balance for every day in [start, end]."""
        by_day: dict[date, float] = {}
        for f in self.flows:
            by_day[f.day] = by_day.get(f.day, 0.0) + f.amount
        for d, a in extra:
            by_day[d] = by_day.get(d, 0.0) + a
        out, bal, d = [], self.opening, self.start
        while d <= self.end:
            bal += by_day.get(d, 0.0)
            out.append((d, bal))
            d += timedelta(days=1)
        return out

    def min_headroom(self, extra: list[tuple[date, float]] = (), since: Optional[date] = None) -> float:
        since = since or self.start
        return min(b for d, b in self.balances(extra) if d >= since) - self.min_balance

    def is_safe(self, payments: list[tuple[date, float]]) -> bool:
        return self.min_headroom([(d, -a) for d, a in payments]) >= -EPS

    def amount_safe_today(self, cap: float) -> float:
        return max(0.0, min(cap, self.min_headroom()))

    def earliest_full_payment(self, amount: float, last_day: Optional[date] = None) -> Optional[date]:
        last_day = min(last_day or self.end, self.end)
        d = self.start
        while d <= last_day:
            if self.is_safe([(d, amount)]):
                return d
            d += timedelta(days=1)
        return None


def _series_amount(s: Series, policy: Policy) -> float:
    if s.is_fixed_amount:
        return s.last_amount
    if policy.variable_amount == "zero":
        return 0.0
    return {"last": s.last_amount, "mean": s.mean_amount, "max": s.max_amount}[policy.variable_amount]


def build(ds: Dataset, user_id: str, as_of: date, policy: Policy = Policy(),
          ledger: Optional[Ledger] = None,
          adjustments: Optional[dict[str, Optional[float]]] = None) -> Forecast:
    """adjustments: series last_event_id -> new amount (None/0 = stopped)."""
    ledger = ledger or normalize_user(ds, user_id, as_of)
    prof = ledger.profile
    end = as_of + timedelta(days=HORIZON_DAYS)
    fc = Forecast(start=as_of, end=end, opening=prof.balance, min_balance=prof.min_balance)
    first = as_of if policy.include_first_day_flows else as_of + timedelta(days=1)

    def in_window(d: Optional[date]) -> bool:
        return d is not None and first <= d <= end

    for e in ledger.of_kind("future_debit", "future_credit"):
        if e.amount_home is None:
            continue  # unknown amount: resolved from images later
        day = e.cash_date if e.cash_date >= as_of else as_of
        if in_window(day) or (day == as_of):
            fc.flows.append(Flow(day if day >= first else first, e.amount_home, e.event_id, e.event.description))

    series, _ = detect(ledger)
    adjustments = adjustments or {}
    confirmed_income_months = {(f.day.year, f.day.month) for f in fc.flows if f.amount > 0}

    # A confirmed next salary with no active payroll series behind it (first
    # job, return from leave) is treated as the start of a monthly salary.
    active_salary = any(s.direction == "credit" and s.category == "salary" and s.is_active(as_of)
                        for s in series)
    if policy.project_recurring_income and not active_salary:
        for e in ledger.of_kind("future_credit"):
            if e.amount_home is None or e.event.category != "salary":
                continue
            d = e.cash_date
            while True:
                y, m = (d.year + (d.month // 12), d.month % 12 + 1)
                try:
                    d = date(y, m, e.cash_date.day)
                except ValueError:
                    d = date(y, m, 28)
                if not in_window(d):
                    break
                if (d.year, d.month) not in confirmed_income_months:
                    fc.flows.append(Flow(d, e.amount_home, e.event_id, "salary continues monthly"))
                    confirmed_income_months.add((d.year, d.month))
    for s in series:
        if s.direction == "credit" and not policy.project_recurring_income:
            continue
        if not s.is_active(as_of):
            continue
        amount = _series_amount(s, policy)
        if s.last_event_id in adjustments:
            amount = adjustments[s.last_event_id] or 0.0
        sign = 1.0 if s.direction == "credit" else -1.0
        for d in s.next_dates(as_of if policy.include_first_day_flows else as_of, end):
            if not in_window(d):
                continue
            if s.direction == "credit" and s.category == "salary" and (d.year, d.month) in confirmed_income_months:
                continue  # already represented by the confirmed salary record
            fc.flows.append(Flow(d, sign * amount, s.last_event_id, s.key, s))
    fc.flows.sort(key=lambda f: (f.day, f.source))
    return fc


if __name__ == "__main__":
    import sys
    from itertools import product
    from loader import load

    ds = load()
    reqs = ds.samples
    if len(sys.argv) > 1 and sys.argv[1] == "grid":
        for inc, var in product([False, True], ["last", "mean", "max", "zero"]):
            pol = Policy(project_recurring_income=inc, variable_amount=var)
            ok_amt = ok_date = 0
            for r in reqs:
                fc = build(ds, r.user_id, r.request_date, pol)
                amt = fc.amount_safe_today(r.amount)
                ed = fc.earliest_full_payment(r.amount)
                exp_amt = float(r.expected["amount_safe_to_pay"])
                exp_d = r.expected["earliest_date_for_full_payment"]
                ok_amt += abs(amt - exp_amt) <= max(0.01, 0.10 * exp_amt)
                ok_date += (ed.isoformat() if ed else "") == exp_d
            print(f"income={inc!s:5} var={var:4}  amount within 10% {ok_amt}/{len(reqs)}  date exact {ok_date}/{len(reqs)}")
    else:
        pol = Policy()
        for r in reqs:
            fc = build(ds, r.user_id, r.request_date, pol)
            amt = fc.amount_safe_today(r.amount)
            ed = fc.earliest_full_payment(r.amount)
            print(f"{r.request_id:11s} safe {amt:14.2f} exp {float(r.expected['amount_safe_to_pay']):14.2f} | "
                  f"date {ed} exp {r.expected['earliest_date_for_full_payment'] or '-'}")
