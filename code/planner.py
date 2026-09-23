"""Plan generation, safety checks and ranking.

For one request we enumerate every candidate plan the rules allow:

    full_payment   pay everything on request_date
    partial        pay amount_safe_to_pay today, the rest on the earliest safe date
    installments   each supplied installment option, schedule copied exactly
    wait           pay everything on the earliest safe date

each optionally combined with spending changes (stop / reduce_to on flexible,
non-protected recurring expenses the user agreed to change). A plan is kept
only if the 90-day forecast stays above the minimum balance and the request
completes by the desired date. Surviving plans are ranked with the rules of
problem_statement.md ("Choosing Between Safe Plans").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import combinations, product
from typing import Optional

from forecast import Flow, Forecast, Policy, build
from loader import Dataset, PaymentOption, Request
from normalize import normalize_user
from recurrence import Series

MAX_CHANGES = 3
FLEXIBLE = {"reducible", "stoppable", "reducible_or_stoppable"}


@dataclass(frozen=True)
class Change:
    event_id: str
    action: str                   # 'stop' | 'reduce_to'
    new_amount: float             # 0 for stop
    saving: float                 # per occurrence
    category: str

    def render(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{fmt_amount(self.new_amount)}"


@dataclass
class Plan:
    method: str
    payments: list[tuple[date, float]]
    changes: tuple[Change, ...] = ()
    option_id: str = ""

    @property
    def total(self) -> float:
        return sum(a for _, a in self.payments)

    @property
    def start(self) -> date:
        return self.payments[0][0]

    @property
    def end(self) -> date:
        return self.payments[-1][0]

    def rank_key(self, deadline: date):
        return (
            0 if self.end <= deadline else 1,
            0 if not self.changes else 1,
            round(self.total, 2),
            self.start,
            len(self.payments),
            sum(c.saving for c in self.changes),   # least disruptive change set
            self.option_id or "",
        )


@dataclass
class Decision:
    request: Request
    amount_safe: float
    earliest: Optional[date]
    plan: Optional[Plan]
    status: str
    method: str
    base: Forecast
    notes: list[str] = field(default_factory=list)


def fmt_amount(x: float) -> str:
    x = round(x + 0.0, 2)
    return str(int(round(x))) if abs(x - round(x)) < 0.005 else f"{x:.2f}"


# ---------------------------------------------------------------- changes --

def possible_changes(fc: Forecast, profile) -> list[Change]:
    """One entry per allowed action on each flexible projected series."""
    seen: dict[str, Series] = {}
    for f in fc.flows:
        if f.series is not None and f.amount < 0:
            seen.setdefault(f.series.last_event_id, f.series)
    out: list[Change] = []
    for eid, s in sorted(seen.items()):
        if s.flexibility not in FLEXIBLE or s.category in profile.protect:
            continue
        current = -next(f.amount for f in fc.flows if f.source == eid)
        can_stop = s.flexibility in ("stoppable", "reducible_or_stoppable") and s.category in profile.stop_ok
        can_reduce = (s.flexibility in ("reducible", "reducible_or_stoppable") and s.category in profile.reduce_ok
                      and s.minimum_allowed is not None and s.minimum_allowed < current)
        if can_stop:
            out.append(Change(eid, "stop", 0.0, current, s.category))
        if can_reduce:
            out.append(Change(eid, "reduce_to", s.minimum_allowed, current - s.minimum_allowed, s.category))
    return out


def change_sets(changes: list[Change]):
    """All non-empty sets of <= MAX_CHANGES changes on distinct events."""
    for k in range(1, MAX_CHANGES + 1):
        for combo in combinations(changes, k):
            if len({c.event_id for c in combo}) == k:
                yield combo


def apply_changes(fc: Forecast, changes: tuple[Change, ...]) -> Forecast:
    if not changes:
        return fc
    by_id = {c.event_id: c for c in changes}
    flows = []
    for f in fc.flows:
        c = by_id.get(f.source) if f.series is not None else None
        flows.append(Flow(f.day, -c.new_amount if c else f.amount, f.source, f.label, f.series) if c else f)
    return Forecast(start=fc.start, end=fc.end, opening=fc.opening, min_balance=fc.min_balance, flows=flows,
                    check_until=fc.check_until)


# ------------------------------------------------------------ candidates --

def installment_plan(opt: PaymentOption) -> list[tuple[date, float]]:
    step = opt.frequency_days or 30
    return [(opt.first_date + timedelta(days=step * i), opt.payment_amount) for i in range(opt.n_payments)]


def candidate_plans(req: Request, fc: Forecast, amount_safe: float, earliest: Optional[date],
                    options: list[PaymentOption], max_months: Optional[float]) -> list[Plan]:
    today = req.request_date
    plans = [Plan("full_payment", [(today, req.amount)])]
    if earliest and earliest > today:
        plans.append(Plan("wait", [(earliest, req.amount)]))
    if req.allows_partial and 0 < amount_safe < req.amount and earliest and earliest <= req.deadline:
        plans.append(Plan("partial_payment", [(today, round(amount_safe, 2)),
                                              (earliest, round(req.amount - amount_safe, 2))]))
    for opt in options:
        if opt.method != "installments":
            continue
        if max_months and opt.n_payments > max_months:
            continue
        plans.append(Plan("installments", installment_plan(opt), option_id=opt.option_id))
    return plans


def is_safe(fc: Forecast, plan: Plan, deadline: date) -> bool:
    return plan.end <= deadline and plan.start >= fc.start and fc.is_safe(plan.payments)


# ---------------------------------------------------------------- decide --

def decide(ds: Dataset, req: Request, policy: Policy = Policy()) -> Decision:
    ledger = normalize_user(ds, req.user_id, req.request_date)
    prof = ledger.profile
    fc = build(ds, req.user_id, req.request_date, policy, ledger=ledger)
    amount_safe = round(fc.amount_safe_today(req.amount), 2)
    earliest_horizon = fc.earliest_full_payment(req.amount)
    if policy.safety_window in ("deadline", "hybrid"):
        fc.check_until = req.deadline
    earliest = fc.earliest_full_payment(req.amount, last_day=req.deadline)

    methods = prof.methods
    options = ds.options_by_request.get(req.request_id, [])
    base_plans = candidate_plans(req, fc, amount_safe, earliest, options, prof.max_installment_months)

    def eligible(p: Plan) -> bool:
        if p.method == "wait":
            return "full_payment" in methods
        return p.method in methods

    safe: list[Plan] = [p for p in base_plans if eligible(p) and is_safe(fc, p, req.deadline)]

    # plans that only work with spending changes
    if not safe:
        changes = possible_changes(fc, prof)
        for combo in change_sets(changes):
            adj = apply_changes(fc, combo)
            e2 = adj.earliest_full_payment(req.amount)
            s2 = round(adj.amount_safe_today(req.amount), 2)
            for p in candidate_plans(req, adj, s2, e2, options, prof.max_installment_months):
                if p.method == "partial_payment":
                    continue  # partial is defined on the unchanged capacity
                p.changes = combo
                if eligible(p) and is_safe(adj, p, req.deadline):
                    safe.append(p)

    best = min(safe, key=lambda p: p.rank_key(req.deadline)) if safe else None
    if best is None:
        status, method = "not_affordable", "not_recommended"
    elif best.method == "full_payment" and not best.changes:
        status, method = "affordable_now", "full_payment"
    elif best.method == "wait":
        status, method = "affordable_later", "wait"
    else:
        status, method = "affordable_with_plan", best.method
    reported = earliest_horizon if policy.safety_window == "hybrid" else earliest
    if best is not None and best.method == "wait":
        reported = best.start
    if status == "affordable_now":
        reported = req.request_date
    return Decision(req, amount_safe, reported, best, status, method, fc)


def explain(d: Decision, currency: str) -> str:
    req, p, prof_min = d.request, d.plan, d.base.min_balance
    amt = lambda x: f"{currency} {x:,.2f}".replace(".00", "")
    if p is None:
        return (f"Not recommended by {req.deadline:%d %B %Y}. No eligible payment option keeps the "
                f"{amt(prof_min)} minimum balance over the next 90 days; only {amt(d.amount_safe)} is safe today.")
    if d.method == "wait":
        return f"Wait and pay {amt(req.amount)} on {p.start:%d %B %Y}, when it becomes safe. Only {amt(d.amount_safe)} is safe today."
    if d.method == "installments":
        return (f"Use {len(p.payments)} installments of {amt(p.payments[0][1])}, starting {p.start:%d %B %Y}. "
                f"This keeps at least {amt(prof_min)} available.")
    if d.method == "partial_payment":
        return (f"Pay {amt(p.payments[0][1])} today and {amt(p.payments[1][1])} on {p.payments[1][0]:%d %B %Y}. "
                f"This keeps at least {amt(prof_min)} available.")
    text = f"Pay {amt(req.amount)} today. This leaves at least {amt(prof_min)} available over the next 90 days."
    if p.changes:
        text += " This requires " + ", ".join(
            ("stopping " if c.action == "stop" else f"reducing to {amt(c.new_amount)} ") + c.category for c in p.changes) + "."
    return text


def to_row(d: Decision, currency: str) -> dict:
    p = d.plan
    return {
        "request_id": d.request.request_id,
        "amount_safe_to_pay": fmt_amount(d.amount_safe),
        "affordability_status": d.status,
        "recommended_payment_method": d.method,
        "payment_plan": "|".join(f"{day.isoformat()}:{fmt_amount(a)}" for day, a in p.payments) if p else "none",
        "earliest_date_for_full_payment": d.earliest.isoformat() if d.earliest else "",
        "spending_changes_needed": "|".join(c.render() for c in p.changes) if p and p.changes else "none",
        "decision_explanation": explain(d, currency),
    }
