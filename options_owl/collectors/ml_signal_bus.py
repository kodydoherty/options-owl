"""ML signal bus — the payload contract between the harvester (publisher) and bots (consumers).

Part of spec 2026-08-04_centralize-ml-signals: the harvester runs the ML scan ONCE and publishes each
pattern signal to Redis `owl:ml:signals`; every bot consumes the identical signal at the identical instant
(killing the per-bot scan-timing race). This module is PURE (no I/O) so it's fully unit-testable — the
Redis transport lives in db.redis_client (publish_ml_signal / subscribe_ml_signals).

Two payload kinds on the channel:
  * signal   — a real ML entry signal (the dict _run_ml_for_ticker returns) + t + type
  * heartbeat — a liveness beat every scan tick so a bot can detect a dead feed and fall back to local scan

The signal payload carries EXACTLY the kwargs _ml_signal_to_trade_signal() consumes, so the consumer
reconstructs the TradeSignal identically to the local scan path — no behavior drift.
"""
from __future__ import annotations

import time

# The whitelist of fields the consumer needs to rebuild a TradeSignal via _ml_signal_to_trade_signal().
# Keep in lockstep with that function's signature (bot_runner._ml_signal_to_trade_signal).
SIGNAL_FIELDS = (
    "ticker", "direction", "score", "premium", "strike", "expiry",
    "ml_confidence", "underlying_price",
)
_REQUIRED = ("ticker", "direction", "strike")   # minimum to be actionable

TYPE_SIGNAL = "signal"
TYPE_HEARTBEAT = "heartbeat"


def build_signal_payload(signal: dict, now: float | None = None) -> dict:
    """Wrap a raw scan signal dict into a publishable payload: whitelisted fields + type + timestamp.
    Unknown fields are dropped (forward-compat / no accidental leakage)."""
    now = time.time() if now is None else now
    out = {k: signal[k] for k in SIGNAL_FIELDS if k in signal}
    out["type"] = TYPE_SIGNAL
    out["t"] = float(now)
    return out


def build_heartbeat(now: float | None = None) -> dict:
    """A liveness beat — no ticker, just a timestamp. Published every scan tick even when no signal fires."""
    return {"type": TYPE_HEARTBEAT, "t": float(time.time() if now is None else now)}


def is_heartbeat(payload: dict) -> bool:
    return isinstance(payload, dict) and payload.get("type") == TYPE_HEARTBEAT


def is_signal(payload: dict) -> bool:
    return isinstance(payload, dict) and payload.get("type") == TYPE_SIGNAL


def payload_age_sec(payload: dict, now: float | None = None) -> float:
    """Seconds since the payload was stamped. Large/inf = stale (or missing timestamp)."""
    now = time.time() if now is None else now
    t = payload.get("t") if isinstance(payload, dict) else None
    if t is None:
        return float("inf")
    try:
        return max(0.0, float(now) - float(t))
    except (TypeError, ValueError):
        return float("inf")


def is_fresh(payload: dict, max_age_sec: float, now: float | None = None) -> bool:
    """True if the payload is a well-formed, recent (<= max_age_sec) message. A stale signal must NOT
    be acted on — the bot was slow/restarting and the transient pattern has passed."""
    return payload_age_sec(payload, now) <= float(max_age_sec)


def parse_signal(payload: dict, max_age_sec: float, now: float | None = None) -> dict | None:
    """Validate a consumed SIGNAL payload and return the kwargs for _ml_signal_to_trade_signal(),
    or None if it's a heartbeat, malformed, missing required fields, or stale. Never raises."""
    if not is_signal(payload):
        return None
    if not is_fresh(payload, max_age_sec, now):
        return None
    for key in _REQUIRED:
        v = payload.get(key)
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
    if payload.get("direction") not in ("CALL", "PUT"):
        return None
    return {k: payload[k] for k in SIGNAL_FIELDS if k in payload}
