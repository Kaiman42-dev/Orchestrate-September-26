"""Recurrence detection over the settled history of one user.

The data contains two shapes of recurring cash flow:

1. Calendar-monthly items on a fixed day (rent on the 2nd, a loan on the 11th,
   a streaming plan on the 10th, payroll on the 15th). Descriptions are stable.
2. Variable spending that happens at a regular cadence but under changing
   descriptions (groceries every ~10 days, transport every ~14 days, dining
   every ~21 days), with varying amounts.

So we first try the whole (category, event_type, flexibility) group as one
series; if it is not regular, we split by description and test each part.
A series is only accepted when history supports it (>= MIN_OCCURRENCES and
regular spacing); everything else is a one-off and is not projected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import mean
from typing import Optional

from normalize import Ledger, LedgerEntry

MIN_OCCURRENCES = 3
DAY_TOLERANCE = 2        # allowed jitter for fixed-step cadences
MONTH_DAY_TOLERANCE = 1  # allowed jitter of the day-of-month for monthly items
ONE_OFF_TYPES = {"refund", "investment_purchase", "investment_sale"}


@dataclass
class Series:
    key: str
    category: str
    event_type: str
    flexibility: str
    direction: str                 # 'debit' or 'credit'
    cadence: str                   # 'monthly' or 'every_n_days'
    step_days: Optional[int]       # for every_n_days
    day_of_month: Optional[int]    # for monthly
    entries: list[LedgerEntry] = field(default_factory=list)

    @property
    def last(self) -> LedgerEntry:
        return self.entries[-1]

    @property
    def last_event_id(self) -> str:
        return self.last.event_id

    @property
    def amounts(self) -> list[float]:
        return [abs(e.amount_home) for e in self.entries if e.amount_home is not None]

    @property
    def last_amount(self) -> float:
        return self.amounts[-1]

    @property
    def mean_amount(self) -> float:
        return mean(self.amounts)

    @property
    def max_amount(self) -> float:
        return max(self.amounts)

    @property
    def is_fixed_amount(self) -> bool:
        a = self.amounts
        return max(a) - min(a) < 0.01

    @property
    def minimum_allowed(self) -> Optional[float]:
        return self.last.event.minimum_allowed_amount

    def next_dates(self, after: date, until: date) -> list[date]:
        """Projected occurrence dates in (after, until]."""
        out: list[date] = []
        last = self.last.cash_date
        if self.cadence == "every_n_days":
            d = last + timedelta(days=self.step_days)
            while d <= until:
                if d > after:
                    out.append(d)
                d += timedelta(days=self.step_days)
        else:
            y, m = last.year, last.month
            while True:
                m += 1
                if m > 12:
                    y, m = y + 1, 1
                d = _safe_date(y, m, self.day_of_month)
                if d > until:
                    break
                if d > after:
                    out.append(d)
        return out


def _safe_date(y: int, m: int, day: int) -> date:
    for d in (day, 30, 29, 28):
        try:
            return date(y, m, d)
        except ValueError:
            continue
    raise ValueError


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def _classify(entries: list[LedgerEntry]) -> Optional[tuple[str, Optional[int], Optional[int]]]:
    if len(entries) < MIN_OCCURRENCES:
        return None
    dates = [e.cash_date for e in entries]
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    if min(gaps) <= 0:
        return None
    # calendar monthly on (almost) the same day
    days = [d.day for d in dates]
    months = [_months_between(a, b) for a, b in zip(dates, dates[1:])]
    if all(m == 1 for m in months) and max(days) - min(days) <= MONTH_DAY_TOLERANCE:
        return "monthly", None, round(mean(days))
    # fixed step in days
    if max(gaps) - min(gaps) <= DAY_TOLERANCE:
        return "every_n_days", round(mean(gaps)), None
    return None


def _make(key: str, entries: list[LedgerEntry]) -> Optional[Series]:
    entries = sorted(entries, key=lambda e: (e.cash_date, e.event_id))
    shape = _classify(entries)
    if not shape:
        return None
    ev = entries[-1].event
    cadence, step, dom = shape
    return Series(key=key, category=ev.category, event_type=ev.event_type,
                  flexibility=ev.flexibility, direction=ev.direction,
                  cadence=cadence, step_days=step, day_of_month=dom, entries=entries)


def detect(ledger: Ledger) -> tuple[list[Series], list[LedgerEntry]]:
    """Return (recurring series, one-off settled entries)."""
    history = [e for e in ledger.of_kind("history")
               if e.amount_home is not None and e.event.event_type not in ONE_OFF_TYPES
               and not e.event.linked_event_id]
    groups: dict[tuple, list[LedgerEntry]] = {}
    for e in history:
        k = (e.event.category, e.event.event_type, e.event.flexibility, e.event.direction)
        groups.setdefault(k, []).append(e)

    series: list[Series] = []
    used: set[str] = set()
    for k, entries in sorted(groups.items()):
        s = _make("/".join(k), entries)
        if s:
            series.append(s)
            used.update(x.event_id for x in entries)
            continue
        by_desc: dict[str, list[LedgerEntry]] = {}
        for e in entries:
            by_desc.setdefault(e.event.description, []).append(e)
        for desc, part in sorted(by_desc.items()):
            s = _make("/".join(k) + "/" + desc, part)
            if s:
                series.append(s)
                used.update(x.event_id for x in part)

    one_offs = [e for e in ledger.of_kind("history") if e.event_id not in used]
    return series, one_offs


if __name__ == "__main__":
    import sys
    from collections import Counter
    from loader import load
    from normalize import normalize_user

    ds = load()
    if len(sys.argv) > 1:
        req = next(r for r in ds.samples + ds.requests if r.request_id == sys.argv[1])
        led = normalize_user(ds, req.user_id, req.request_date)
        ser, one = detect(led)
        horizon = req.request_date + timedelta(days=90)
        for s in ser:
            print(f"{s.key:60s} {s.cadence:12s} step={s.step_days} dom={s.day_of_month} n={len(s.entries)} "
                  f"last={s.last_amount:.2f} max={s.max_amount:.2f} next={len(s.next_dates(req.request_date, horizon))} "
                  f"last_id={s.last_event_id}")
        print("one-offs:", len(one), Counter(e.event.category for e in one))
    else:
        cad, oneoff_cat = Counter(), Counter()
        for req in ds.samples + ds.requests:
            ser, one = detect(normalize_user(ds, req.user_id, req.request_date))
            cad.update(f"{s.category}:{s.cadence}" for s in ser)
            oneoff_cat.update(e.event.category for e in one)
        print("series:", dict(cad.most_common(40)))
        print("one-off categories:", dict(oneoff_cat.most_common()))
