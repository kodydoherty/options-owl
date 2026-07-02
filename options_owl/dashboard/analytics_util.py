"""Pure, DB-free analytics helpers for the dashboard. Unit-testable in isolation.

These turn the raw ``trade_premium_ticks`` series (already captured every ~15s by the live
monitor) into the verbose per-trade summary the detail view renders — peak/trough, drawdown
from peak, spread stats, underlying move — WITHOUT any new capture. B1 later enriches the
timeline with per-cycle FSM state; this works off data that exists today.
"""

from __future__ import annotations

from typing import Any


def _f(v: Any) -> float | None:
    """Best-effort float; None/blank/garbage → None (never raises)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _gain_pct(premium: float | None, entry: float | None) -> float | None:
    if premium is None or entry is None or entry <= 0:
        return None
    return (premium - entry) / entry * 100.0


def summarize_ticks(
    ticks: list[dict], entry_premium: float | None, contracts: int | None = None
) -> dict:
    """Summarize a trade's premium-tick series into verbose detail-view stats.

    Args:
        ticks: rows from ``get_premium_ticks`` (captured_at, premium, bid, ask,
            underlying_price), ascending by time. Any field may be None.
        entry_premium: the trade's entry premium/contract (basis for gain%).
        contracts: contract count (for $ swings). Optional.

    Returns a dict that is always safe to render — an empty series yields ``{"n": 0}`` and
    every downstream key defaulted to None, so the template never KeyErrors.
    """
    out: dict[str, Any] = {
        "n": 0,
        "entry_premium": _f(entry_premium),
        "premium_min": None, "premium_max": None, "premium_last": None,
        "peak_premium": None, "peak_gain_pct": None, "peak_at": None, "peak_idx": None,
        "trough_premium": None, "trough_gain_pct": None,
        "drawdown_from_peak_pct": None,
        "avg_spread_pct": None, "max_spread_pct": None,
        "underlying_first": None, "underlying_last": None, "underlying_move_pct": None,
        "peak_dollars": None, "last_dollars": None,
    }
    if not ticks:
        return out

    entry = _f(entry_premium)
    prems = [(_f(t.get("premium")), t.get("captured_at"), i) for i, t in enumerate(ticks)]
    prems = [(p, at, i) for (p, at, i) in prems if p is not None and p > 0]
    out["n"] = len(ticks)
    if not prems:
        return out

    premiums = [p for (p, _at, _i) in prems]
    out["premium_min"] = min(premiums)
    out["premium_max"] = max(premiums)
    out["premium_last"] = premiums[-1]

    peak_p, peak_at, peak_idx = max(prems, key=lambda x: x[0])
    out["peak_premium"] = peak_p
    out["peak_at"] = peak_at
    out["peak_idx"] = peak_idx
    out["peak_gain_pct"] = _gain_pct(peak_p, entry)

    trough_p = min(premiums)
    out["trough_premium"] = trough_p
    out["trough_gain_pct"] = _gain_pct(trough_p, entry)

    # Worst give-back: from the running peak to the lowest subsequent premium.
    running_peak = premiums[0]
    max_dd = 0.0
    for p in premiums:
        running_peak = max(running_peak, p)
        if running_peak > 0:
            dd = (running_peak - p) / running_peak * 100.0
            max_dd = max(max_dd, dd)
    out["drawdown_from_peak_pct"] = max_dd

    # Bid/ask spread (liquidity) — only over ticks that carry both.
    spreads = []
    for t in ticks:
        bid, ask = _f(t.get("bid")), _f(t.get("ask"))
        if bid is not None and ask is not None and ask > 0 and bid > 0:
            mid = (bid + ask) / 2.0
            if mid > 0:
                spreads.append((ask - bid) / mid * 100.0)
    if spreads:
        out["avg_spread_pct"] = sum(spreads) / len(spreads)
        out["max_spread_pct"] = max(spreads)

    # Underlying drift over the hold.
    unders = [_f(t.get("underlying_price")) for t in ticks]
    unders = [u for u in unders if u is not None and u > 0]
    if unders:
        out["underlying_first"] = unders[0]
        out["underlying_last"] = unders[-1]
        if unders[0] > 0:
            out["underlying_move_pct"] = (unders[-1] - unders[0]) / unders[0] * 100.0

    if contracts:
        ct = _f(contracts) or 0
        if entry is not None:
            out["peak_dollars"] = (peak_p - entry) * ct * 100.0
            out["last_dollars"] = (premiums[-1] - entry) * ct * 100.0

    return out


# ---------------------------------------------------------------------------
# Unified timeline (events + derived entry/peak/exit milestones)
# ---------------------------------------------------------------------------

# event_type substring → tone bucket used for colour coding in the template.
_GOOD = ("approved", "filled", "webull_filled", "profit", "scaleout", "breakeven")
_BAD = ("rejected", "error", "blocked", "abandoned", "manual_close", "hardstop", "stop")
_WARN = ("dca", "add", "partial", "reconcile", "retry", "warn")


def _tone_for(event_type: str) -> str:
    et = (event_type or "").lower()
    if any(k in et for k in _BAD):
        return "bad"
    if any(k in et for k in _GOOD):
        return "good"
    if any(k in et for k in _WARN):
        return "warn"
    return "neutral"


def _iso(v: Any) -> str | None:
    if v is None:
        return None
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


def build_timeline(trade: dict, events: list[dict], tick_stats: dict | None = None) -> list[dict]:
    """Merge entry, trade_events, the peak milestone, and exit into one ordered feed.

    Each item: ``{sort, ts, kind, label, tone, details}``. ``sort`` is an ISO string used only
    for ordering (entry pinned first, exit last) so the template can iterate directly. Pure —
    no DB, no clock; safe on partial rows.
    """
    items: list[dict] = []
    tick_stats = tick_stats or {}

    ct = trade.get("contracts")
    entry_p = _f(trade.get("premium_per_contract"))
    direction = (trade.get("direction") or trade.get("option_type") or "").upper()
    opened = _iso(trade.get("opened_at"))
    items.append({
        "sort": opened or "",
        "ts": opened,
        "kind": "entry",
        "tone": "neutral",
        "label": f"Entered {ct}x {trade.get('ticker', '')} {direction}"
                 + (f" @ ${entry_p:.2f}" if entry_p is not None else ""),
        "details": None,
    })

    for ev in events:
        et = ev.get("event_type", "")
        items.append({
            "sort": _iso(ev.get("created_at")) or opened or "",
            "ts": _iso(ev.get("created_at")),
            "kind": et,
            "tone": _tone_for(et),
            "label": et.replace("_", " "),
            "details": ev.get("details"),
        })

    peak_gain = tick_stats.get("peak_gain_pct")
    peak_at = _iso(tick_stats.get("peak_at"))
    if peak_gain is not None and peak_gain > 0 and peak_at:
        peak_p = tick_stats.get("peak_premium")
        items.append({
            "sort": peak_at,
            "ts": peak_at,
            "kind": "peak",
            "tone": "good",
            "label": f"Peak ${peak_p:.2f} (+{peak_gain:.1f}%)" if peak_p is not None
                     else f"Peak +{peak_gain:.1f}%",
            "details": None,
        })

    if (trade.get("status") or "").lower() == "closed":
        pnl = _f(trade.get("pnl_dollars"))
        exit_p = _f(trade.get("exit_premium"))
        closed = _iso(trade.get("closed_at"))
        reason = trade.get("exit_reason") or "closed"
        tone = "neutral" if pnl is None else ("good" if pnl >= 0 else "bad")
        label = f"Exited @ ${exit_p:.2f} — {reason}" if exit_p is not None else f"Exited — {reason}"
        if pnl is not None:
            label += f" ({'+' if pnl >= 0 else ''}${pnl:,.0f})"
        items.append({
            "sort": closed or "~",  # closed sorts last when timestamps tie
            "ts": closed,
            "kind": "exit",
            "tone": tone,
            "label": label,
            "details": None,
        })

    items.sort(key=lambda x: x["sort"])
    return items
