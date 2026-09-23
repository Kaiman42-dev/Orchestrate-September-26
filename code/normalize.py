"""Event normalization: turn raw financial_events rows into a clean ledger.

Each event gets a `kind` that says how the forecaster should treat it:

    history          settled cash movement, used to learn recurring patterns
    future_debit     pending/scheduled debit still to leave the account (reserved)
    future_credit    confirmed future income (only confirmed salary counts)
    ignored          never affects cash (reason in `note`)

Rules (problem_statement.md, "90-Day Safety Check" and conflict rules):
- failed and cancelled transactions are ignored
- unrealized investment valuations (non_cash) are ignored
- pending credits (refunds, reimbursements not yet settled) are ignored
- a "possible duplicate" pending charge that mirrors an earlier settled
  charge (same amount, linked) is treated as a duplicate record
- a blank amount is never zero: the event is flagged `needs_image`
- amounts are converted to the user's home currency on the event date
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from loader import Dataset, Event, Profile

INCOME_TYPES = {"income"}
NON_CASH_STATUSES = {"unrealized"}
DEAD_STATUSES = {"failed", "cancelled"}
OPEN_STATUSES = {"pending", "scheduled"}


@dataclass
class LedgerEntry:
    event: Event
    kind: str
    amount_home: Optional[float]  # signed: + credit, - debit; None if unknown
    cash_date: Optional[date]     # date the cash actually moves
    note: str = ""
    needs_image: bool = False

    @property
    def event_id(self) -> str:
        return self.event.event_id


@dataclass
class Ledger:
    profile: Profile
    entries: list[LedgerEntry] = field(default_factory=list)

    def of_kind(self, *kinds: str) -> list[LedgerEntry]:
        return [e for e in self.entries if e.kind in kinds]

    def by_id(self) -> dict[str, LedgerEntry]:
        return {e.event_id: e for e in self.entries}


def _is_duplicate(ev: Event, events_by_id: dict[str, Event]) -> bool:
    if not ev.linked_event_id or "duplicate" not in ev.description.lower():
        return False
    parent = events_by_id.get(ev.linked_event_id)
    return (
        parent is not None
        and parent.status == "settled"
        and parent.amount is not None
        and ev.amount is not None
        and abs(parent.amount - ev.amount) < 0.005
        and parent.direction == ev.direction
    )


def normalize_user(ds: Dataset, user_id: str, as_of: date) -> Ledger:
    """Build the ledger for one user, as seen on `as_of` (the request date)."""
    profile = ds.profiles[user_id]
    ledger = Ledger(profile=profile)
    for ev in sorted(ds.events_by_user.get(user_id, []), key=lambda e: (e.event_date or date.min, e.event_id)):
        cash_date = ev.settlement_date or ev.event_date
        amount_home = None
        raw_amount, raw_currency = ev.amount, ev.currency
        if raw_amount is None:
            from facts import image_amount
            got = image_amount(ev.event_id)
            if got is not None:
                raw_amount, raw_currency = got
        if raw_amount is not None:
            rate_date = ev.event_date or as_of
            value = ds.fx.convert(raw_amount, raw_currency, profile.home_currency, rate_date)
            amount_home = value if ev.direction == "credit" else -value if ev.direction == "debit" else 0.0

        def add(kind: str, note: str = ""):
            ledger.entries.append(LedgerEntry(
                event=ev, kind=kind, amount_home=amount_home, cash_date=cash_date,
                note=note, needs_image=ev.amount is None))

        if ev.status in DEAD_STATUSES:
            add("ignored", f"{ev.status} transaction")
        elif ev.direction == "non_cash" or ev.status in NON_CASH_STATUSES:
            add("ignored", "unrealized investment value")
        elif _is_duplicate(ev, ds.events_by_id):
            add("ignored", f"duplicate of {ev.linked_event_id}")
        elif ev.status in OPEN_STATUSES:
            if ev.direction == "debit":
                add("future_debit", f"{ev.status} debit reserved")
            elif ev.event_type in INCOME_TYPES and ev.category == "salary" and ev.status == "scheduled":
                add("future_credit", "confirmed salary, counted on settlement date")
            else:
                add("ignored", f"{ev.status} credit not counted until settled")
        elif ev.status == "settled":
            add("history")
        else:
            add("ignored", f"unknown status {ev.status}")
    return ledger


def summary(ledger: Ledger) -> dict:
    out: dict[str, int] = {}
    for e in ledger.entries:
        out[e.kind] = out.get(e.kind, 0) + 1
    out["needs_image"] = sum(1 for e in ledger.entries if e.needs_image)
    return out


if __name__ == "__main__":
    from collections import Counter
    from loader import load

    ds = load()
    totals: Counter = Counter()
    notes: Counter = Counter()
    for req in ds.samples + ds.requests:
        led = normalize_user(ds, req.user_id, req.request_date)
        totals.update(summary(led))
        notes.update(e.note.split(" of ")[0] for e in led.entries if e.kind == "ignored")
    print("kinds:", dict(totals))
    print("ignored reasons:", dict(notes))
