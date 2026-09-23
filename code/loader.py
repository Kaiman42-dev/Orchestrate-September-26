"""Load the Buy or Wait? dataset into plain Python structures.

Everything here is deterministic and dependency-free (stdlib only).
Amounts are kept as floats; dates as datetime.date.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")


def parse_date(value: str) -> Optional[date]:
    value = (value or "").strip()
    if not value:
        return None
    return date.fromisoformat(value[:10])


def parse_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    if value == "":
        return None
    return float(value)


def split_pipe(value: str) -> list[str]:
    value = (value or "").strip()
    return [v for v in value.split("|") if v] if value else []


def read_csv(name: str, dataset_dir: str = DATASET_DIR) -> list[dict]:
    with open(os.path.join(dataset_dir, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@dataclass
class Profile:
    user_id: str
    home_currency: str
    balance: float
    min_balance: float
    priorities: list[str]
    protect: set[str]
    reduce_ok: set[str]
    stop_ok: set[str]
    methods: set[str]
    max_installment_months: Optional[float]


@dataclass
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Optional[float]  # None = must be read from an image
    currency: str
    event_date: Optional[date]
    settlement_date: Optional[date]
    status: str
    linked_event_id: str
    flexibility: str
    minimum_allowed_amount: Optional[float]


@dataclass
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    amount: float
    deadline: date
    allows_partial: bool
    text: str
    expected: dict = field(default_factory=dict)  # only for sample_requests


@dataclass
class PaymentOption:
    option_id: str
    request_id: str
    method: str
    payment_amount: float
    n_payments: int
    first_date: date
    frequency_days: Optional[int]
    fee: float
    total: float


class FX:
    """Dated currency conversion. Uses the latest rate on or before the date
    (falling back to the earliest available one), and chains through other
    currencies when no direct pair exists (e.g. ZAR -> EUR -> USD)."""

    def __init__(self, rows: list[dict]):
        self.rates: dict[tuple[str, str], list[tuple[date, float]]] = defaultdict(list)
        for r in rows:
            d, a, b, x = parse_date(r["rate_date"]), r["from_currency"], r["to_currency"], float(r["rate"])
            self.rates[(a, b)].append((d, x))
            self.rates[(b, a)].append((d, 1.0 / x))
        for k in self.rates:
            self.rates[k].sort()

    def _direct(self, a: str, b: str, on: date) -> Optional[float]:
        series = self.rates.get((a, b))
        if not series:
            return None
        best = series[0][1]
        for d, x in series:
            if d <= on:
                best = x
            else:
                break
        return best

    def rate(self, a: str, b: str, on: date) -> float:
        if a == b:
            return 1.0
        # BFS over the currency graph
        frontier = [(a, 1.0)]
        seen = {a}
        while frontier:
            nxt = []
            for cur, acc in frontier:
                for (x, y) in self.rates:
                    if x != cur or y in seen:
                        continue
                    r = self._direct(x, y, on)
                    if r is None:
                        continue
                    if y == b:
                        return acc * r
                    seen.add(y)
                    nxt.append((y, acc * r))
            frontier = nxt
        raise KeyError(f"No FX path {a}->{b} on {on}")

    def convert(self, amount: float, a: str, b: str, on: date) -> float:
        return amount * self.rate(a, b, on)


@dataclass
class Dataset:
    profiles: dict[str, Profile]
    events_by_user: dict[str, list[Event]]
    events_by_id: dict[str, Event]
    requests: list[Request]
    samples: list[Request]
    options_by_request: dict[str, list[PaymentOption]]
    messages: list[dict]
    images: list[dict]
    fx: FX


def load(dataset_dir: str = DATASET_DIR) -> Dataset:
    profiles = {}
    for r in read_csv("financial_profiles.csv", dataset_dir):
        profiles[r["user_id"]] = Profile(
            user_id=r["user_id"],
            home_currency=r["home_currency"],
            balance=float(r["current_available_balance"]),
            min_balance=float(r["minimum_balance_to_keep"]),
            priorities=split_pipe(r["financial_priorities"]),
            protect=set(split_pipe(r["expense_categories_to_protect"])),
            reduce_ok=set(split_pipe(r["expense_categories_user_is_willing_to_reduce"])),
            stop_ok=set(split_pipe(r["expense_categories_user_is_willing_to_stop"])),
            methods=set(split_pipe(r["payment_methods_user_will_consider"])),
            max_installment_months=parse_float(r["max_installment_months"]),
        )

    events_by_user: dict[str, list[Event]] = defaultdict(list)
    events_by_id = {}
    for r in read_csv("financial_events.csv", dataset_dir):
        e = Event(
            event_id=r["event_id"], user_id=r["user_id"], event_type=r["event_type"],
            description=r["description"], category=r["category"], direction=r["direction"],
            amount=parse_float(r["amount"]), currency=r["currency"],
            event_date=parse_date(r["event_date"]), settlement_date=parse_date(r["settlement_date"]),
            status=r["status"], linked_event_id=r["linked_event_id"].strip(),
            flexibility=r["flexibility"], minimum_allowed_amount=parse_float(r["minimum_allowed_amount"]),
        )
        events_by_user[e.user_id].append(e)
        events_by_id[e.event_id] = e

    def to_request(r: dict, with_expected: bool) -> Request:
        req = Request(
            request_id=r["request_id"], user_id=r["user_id"],
            request_date=parse_date(r["request_date"]), request_type=r["request_type"],
            amount=float(r["requested_amount"]), deadline=parse_date(r["desired_completion_date"]),
            allows_partial=r["allows_partial_payment"].strip().lower() == "true",
            text=r["request_text"],
        )
        if with_expected:
            req.expected = {k: r[k] for k in (
                "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
                "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed",
                "decision_explanation")}
        return req

    requests = [to_request(r, False) for r in read_csv("requests.csv", dataset_dir)]
    samples = [to_request(r, True) for r in read_csv("sample_requests.csv", dataset_dir)]

    options: dict[str, list[PaymentOption]] = defaultdict(list)
    for r in read_csv("request_payment_options.csv", dataset_dir):
        freq = parse_float(r["payment_frequency_days"])
        options[r["request_id"]].append(PaymentOption(
            option_id=r["payment_option_id"], request_id=r["request_id"], method=r["payment_method"],
            payment_amount=float(r["payment_amount"]), n_payments=int(r["number_of_payments"]),
            first_date=parse_date(r["first_payment_date"]),
            frequency_days=int(freq) if freq is not None else None,
            fee=float(r["financing_fee"] or 0), total=float(r["total_payable_amount"]),
        ))

    return Dataset(
        profiles=profiles,
        events_by_user=dict(events_by_user),
        events_by_id=events_by_id,
        requests=requests,
        samples=samples,
        options_by_request=dict(options),
        messages=read_csv("messages.csv", dataset_dir),
        images=read_csv("images.csv", dataset_dir),
        fx=FX(read_csv("exchange_rates.csv", dataset_dir)),
    )


if __name__ == "__main__":
    ds = load()
    n_events = sum(len(v) for v in ds.events_by_user.values())
    print(f"profiles={len(ds.profiles)} events={n_events} requests={len(ds.requests)} "
          f"samples={len(ds.samples)} messages={len(ds.messages)} images={len(ds.images)}")
    print("USD->ZAR on 2024-03-03:", round(ds.fx.rate("USD", "ZAR", date(2024, 3, 3)), 4))
