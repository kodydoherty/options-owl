"""Supabase Shared Brain — mutual learning integration with Vince's 0DTE alert system.

Handles all reads (alert lookup, conviction, risk context) and writes (fills,
closes, execution decisions, account state) to the shared Supabase database.

All operations are fire-and-forget — Supabase failures never block trade execution.
Failed writes are queued locally and retried.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from options_owl.config.settings import Settings

# Recovery queue for failed writes (same pattern as DB recovery queue)
_RECOVERY_DIR = Path(os.getenv("JOURNAL_DIR", "journal")) / "supabase_recovery"

# Exit reason mapping: our V5 FSM reasons → Vince's allowed close_reason values
# Keys are ExitReason.value strings from options_owl/risk/exit_v5/types.py
_EXIT_REASON_MAP = {
    # EOD gates
    "eod_cutoff": "eod",
    "eod_expiry": "eod",
    # Stop-loss family
    "hard_stop": "stop_loss",
    "confirmed_stop": "stop_loss",
    "graduated_stop": "stop_loss",       # legacy name for confirmed_stop
    "checkpoint_cut": "stop_loss",
    "backstop": "stop_loss",
    "breakeven_ratchet": "stop_loss",
    "max_trade_loss": "stop_loss",
    # Target-hit / trailing family
    "profit_target": "target_hit",
    "adaptive_trail": "target_hit",
    "soft_trail": "target_hit",
    "scalp_trail": "target_hit",
    # Partial exits
    "scaleout": "partial_50",
    # Time-based exits
    "theta_exit": "time_stop",
    "theta_bleed": "time_stop",
    "theta_timer": "time_stop",
    # Momentum / discretionary
    "bid_disappearance": "momentum_fade",
    "sideways_scalp": "momentum_fade",
    # Manual / other
    "manual": "manual",
    "signal_flip": "manual",
    # Option expiry
    "expired": "expired",
}

# Pipeline rejection reason → execution_decisions reason mapping
_REJECTION_REASON_MAP = {
    "score_too_low": "low_conviction",
    "premium_cap": "low_conviction",
    "spread_gate": "spread_too_wide",
    "duplicate_ticker": "duplicate_alert_recent",
    "concurrent_limit": "position_limit",
    "correlation_cap": "sector_concentration_max",
    "daily_loss": "daily_loss_limit",
    "circuit_breaker": "daily_loss_limit",
    "blocked_ticker": "low_conviction",
    "momentum_confirm": "low_conviction",
    "capital_block": "low_buying_power",
    "gfv_block": "low_buying_power",
    "pdt_block": "low_buying_power",
    "smart_entry_blocked": "spread_too_wide",
    "dip_confirm_skipped": "low_conviction",
    "time_filter": "time_filter",
}


class SupabaseBrain:
    """Async client for the shared Supabase learning database."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._base_url = settings.SUPABASE_URL.rstrip("/")
        self._agent_id = settings.AGENT_ID or ""
        self._read_headers = {
            "apikey": settings.SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_WEBULL_JWT}",
            "Content-Type": "application/json",
        }
        self._write_headers = {
            **self._read_headers,
            "Prefer": "return=minimal",
        }
        self._client: httpx.AsyncClient | None = None
        self._webhook_url = settings.N8N_WEBHOOK_CLOSE_URL or ""

    @property
    def enabled(self) -> bool:
        return bool(
            self.settings.ENABLE_SUPABASE_BRAIN
            and self._base_url
            and self.settings.SUPABASE_ANON_KEY
            and self.settings.SUPABASE_WEBULL_JWT
            and self._agent_id
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # READ operations
    # ------------------------------------------------------------------

    async def lookup_alert(
        self, ticker: str, direction: str,
    ) -> dict | None:
        """Look up the most recent alert for a ticker+direction from Supabase.

        Returns the full alert dict (including alert_id and conviction_0_100),
        or None if lookup fails or no alert found.
        """
        if not self.enabled:
            return None

        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self._base_url}/rest/v1/v_alerts_with_conviction",
                params={
                    "ticker": f"eq.{ticker.upper()}",
                    "direction": f"eq.{direction}",
                    "order": "fire_time.desc",
                    "limit": "1",
                },
                headers=self._read_headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    alert = data[0]
                    logger.debug(
                        f"[SupabaseBrain] Alert found: {ticker} {direction} "
                        f"alert_id={alert.get('alert_id', '?')[:8]}... "
                        f"conviction={alert.get('conviction_0_100', '?')}"
                    )
                    return alert
                else:
                    logger.debug(f"[SupabaseBrain] No alert found for {ticker} {direction}")
                    return None
            else:
                logger.warning(
                    f"[SupabaseBrain] Alert lookup failed: HTTP {resp.status_code} "
                    f"for {ticker} {direction}"
                )
                return None
        except Exception as exc:
            logger.warning(f"[SupabaseBrain] Alert lookup error for {ticker}: {exc}")
            return None

    async def get_risk_context(self) -> dict | None:
        """Read today's risk context (FOMC, CPI, vol regime, etc)."""
        if not self.enabled:
            return None

        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self._base_url}/rest/v1/v_risk_context_today",
                headers=self._read_headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data[0] if data else None
            return None
        except Exception as exc:
            logger.warning(f"[SupabaseBrain] Risk context error: {exc}")
            return None

    async def get_system_health(self) -> dict | None:
        """Read system recent performance (win rate, loss streaks)."""
        if not self.enabled:
            return None

        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self._base_url}/rest/v1/v_system_recent_performance",
                headers=self._read_headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data[0] if data else None
            return None
        except Exception as exc:
            logger.warning(f"[SupabaseBrain] System health error: {exc}")
            return None

    async def get_tier_performance(self) -> list[dict]:
        """Read 30-day performance by score tier (for sizing calibration)."""
        if not self.enabled:
            return []

        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self._base_url}/rest/v1/v_tier_performance_30d",
                headers=self._read_headers,
            )
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception as exc:
            logger.warning(f"[SupabaseBrain] Tier performance error: {exc}")
            return []

    async def get_sector_activity(self) -> list[dict]:
        """Read today's sector activity (for concentration checks)."""
        if not self.enabled:
            return []

        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self._base_url}/rest/v1/v_sector_active_today",
                headers=self._read_headers,
            )
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception as exc:
            logger.warning(f"[SupabaseBrain] Sector activity error: {exc}")
            return []

    async def get_slippage_stats(self) -> list[dict]:
        """Read 30-day slippage stats by ticker (our own execution quality)."""
        if not self.enabled:
            return []

        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self._base_url}/rest/v1/v_slippage_by_ticker_30d",
                headers=self._read_headers,
            )
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception as exc:
            logger.warning(f"[SupabaseBrain] Slippage stats error: {exc}")
            return []

    # ------------------------------------------------------------------
    # WRITE operations (all fire-and-forget, never block trading)
    # ------------------------------------------------------------------

    async def _write(
        self, endpoint: str, payload: dict, context: str,
    ) -> bool:
        """POST to a Supabase REST endpoint. Returns True on success.

        On failure, queues the write for later retry.
        Automatically injects agent_id into every payload.
        """
        if not self.enabled:
            return False

        # Every write requires agent_id (migration 005)
        payload = {**payload, "agent_id": self._agent_id}

        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._base_url}/rest/v1/{endpoint}",
                json=payload,
                headers=self._write_headers,
            )
            if resp.status_code in (200, 201):
                logger.debug(f"[SupabaseBrain] {context}: OK")
                return True
            elif resp.status_code == 409:
                # Duplicate — idempotent success
                logger.debug(f"[SupabaseBrain] {context}: duplicate (409), treating as success")
                return True
            else:
                logger.warning(
                    f"[SupabaseBrain] {context}: HTTP {resp.status_code} — {resp.text[:200]}"
                )
                self._queue_for_recovery(endpoint, payload, context)
                return False
        except Exception as exc:
            logger.warning(f"[SupabaseBrain] {context}: error — {exc}")
            self._queue_for_recovery(endpoint, payload, context)
            return False

    def _queue_for_recovery(
        self, endpoint: str, payload: dict, context: str,
    ) -> None:
        """Save a failed write to disk for later retry."""
        try:
            _RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
            entry = {
                "endpoint": endpoint,
                "payload": payload,
                "context": context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_ctx = context.replace(" ", "_").replace("/", "_")[:50]
            path = _RECOVERY_DIR / f"{ts}_{safe_ctx}.json"
            path.write_text(json.dumps(entry, indent=2, default=str))
            logger.info(f"[SupabaseBrain] Queued for recovery: {path.name}")
        except Exception as exc:
            logger.error(f"[SupabaseBrain] Failed to queue recovery: {exc}")

    async def replay_recovery_queue(self) -> int:
        """Replay any queued writes. Returns count of successfully replayed."""
        if not self.enabled or not _RECOVERY_DIR.exists():
            return 0

        queue_files = sorted(_RECOVERY_DIR.glob("*.json"))
        if not queue_files:
            return 0

        replayed = 0
        for qf in queue_files:
            try:
                entry = json.loads(qf.read_text())
                client = await self._get_client()
                resp = await client.post(
                    f"{self._base_url}/rest/v1/{entry['endpoint']}",
                    json=entry["payload"],
                    headers=self._write_headers,
                )
                if resp.status_code in (200, 201, 409):
                    qf.unlink()
                    replayed += 1
                else:
                    logger.warning(
                        f"[SupabaseBrain] Recovery replay failed for {qf.name}: "
                        f"HTTP {resp.status_code}"
                    )
            except Exception as exc:
                logger.warning(f"[SupabaseBrain] Recovery replay error for {qf.name}: {exc}")

        if replayed:
            logger.info(f"[SupabaseBrain] Replayed {replayed} queued writes")
        return replayed

    # ------------------------------------------------------------------
    # High-level write methods (called from paper_trader / position_monitor)
    # ------------------------------------------------------------------

    async def record_fill(
        self,
        alert_id: str,
        broker_order_id: str,
        fill_price: float,
        fill_quantity: int,
        strike: float,
        contract_symbol: str = "",
        slippage_pct: float | None = None,
        nbbo_at_order: dict | None = None,
    ) -> bool:
        """Record a Webull fill in Supabase."""
        payload: dict[str, Any] = {
            "alert_id": alert_id,
            "broker_order_id": str(broker_order_id),
            "fill_time": datetime.now(timezone.utc).isoformat(),
            "fill_price": fill_price,
            "fill_quantity": fill_quantity,
            "strike_filled": strike,
        }
        if contract_symbol:
            payload["contract_symbol"] = contract_symbol
        if slippage_pct is not None:
            payload["slippage_pct"] = round(slippage_pct, 2)
        # NBBO at order time — enables execution quality measurement
        if nbbo_at_order:
            raw_data = {}
            if nbbo_at_order.get("bid") is not None:
                raw_data["nbbo_bid"] = round(nbbo_at_order["bid"], 4)
            if nbbo_at_order.get("ask") is not None:
                raw_data["nbbo_ask"] = round(nbbo_at_order["ask"], 4)
            if nbbo_at_order.get("mid") is not None:
                raw_data["mid_at_order_time"] = round(nbbo_at_order["mid"], 4)
            if fill_price is not None:
                raw_data["intended_price"] = round(fill_price, 4)
            if raw_data:
                payload["raw_broker_data"] = json.dumps(raw_data)

        return await self._write("fills", payload, f"fill {broker_order_id}")

    async def record_close(
        self,
        alert_id: str,
        close_price: float,
        exit_reason: str,
        pnl_pct: float | None = None,
        pnl_usd: float | None = None,
        hold_minutes: float | None = None,
        peak_premium: float | None = None,
    ) -> bool:
        """Record a position close in Supabase + fire webhook."""
        # Map our exit reason to Vince's allowed values
        mapped_reason = _EXIT_REASON_MAP.get(exit_reason, "manual")

        payload: dict[str, Any] = {
            "alert_id": alert_id,
            "close_time": datetime.now(timezone.utc).isoformat(),
            "close_price": close_price,
            "close_reason": mapped_reason,
        }
        if pnl_pct is not None:
            payload["real_pnl_pct"] = round(pnl_pct, 2)
        if pnl_usd is not None:
            payload["real_pnl_usd"] = round(pnl_usd, 2)
        if hold_minutes is not None:
            payload["hold_minutes"] = round(hold_minutes, 1)
        if peak_premium is not None:
            payload["peak_premium"] = round(peak_premium, 4)

        # Write to Supabase (source of truth)
        ok = await self._write("closes", payload, f"close {alert_id[:8]}")

        # Fire webhook (best-effort, don't retry)
        if self._webhook_url:
            try:
                client = await self._get_client()
                await client.post(
                    self._webhook_url,
                    json=payload,
                    timeout=5.0,
                )
            except Exception:
                pass  # Supabase is source of truth

        return ok

    async def record_execution_decision(
        self,
        alert_id: str,
        decision: str,
        reason: str,
        intended_contracts: int | None = None,
        actual_contracts: int | None = None,
        intended_strike: float | None = None,
        actual_strike: float | None = None,
        conviction_score: int | None = None,
        notes: str = "",
    ) -> bool:
        """Record an execution decision (REQUIRED for every alert)."""
        payload: dict[str, Any] = {
            "alert_id": alert_id,
            "decision": decision,
            "reason": reason,
        }
        if intended_contracts is not None:
            payload["intended_contracts"] = intended_contracts
        if actual_contracts is not None:
            payload["actual_contracts"] = actual_contracts
        if intended_strike is not None:
            payload["intended_strike"] = intended_strike
        if actual_strike is not None:
            payload["actual_strike"] = actual_strike
        if conviction_score is not None:
            # Clamp to 0-100 for team compatibility (our scale is 78-177)
            # Coerce to int first — Supabase JSON may return float/str
            try:
                payload["conviction_score"] = max(0, min(100, int(conviction_score)))
            except (TypeError, ValueError):
                pass  # skip if value is unparseable
        if notes:
            payload["notes"] = notes[:500]

        return await self._write(
            "execution_decisions", payload, f"decision {decision} {reason}"
        )

    async def push_account_state(
        self,
        equity_usd: float,
        cash_usd: float,
        daily_pnl_usd: float,
        open_positions: int,
        buying_power: float | None = None,
    ) -> bool:
        """Push account state snapshot to Supabase."""
        daily_pnl_pct = (daily_pnl_usd / equity_usd * 100) if equity_usd > 0 else 0

        payload: dict[str, Any] = {
            "equity_usd": round(equity_usd, 2),
            "cash_usd": round(cash_usd, 2),
            "daily_pnl_usd": round(daily_pnl_usd, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "open_positions": open_positions,
        }
        if buying_power is not None:
            payload["buying_power"] = round(buying_power, 2)

        return await self._write("account_state", payload, "account_state")

    # ------------------------------------------------------------------
    # Convenience: map our pipeline rejection to a decision record
    # ------------------------------------------------------------------

    async def record_skip(
        self,
        alert_id: str,
        failure_reasons: list[str],
        signal_score: int | None = None,
        conviction: int | None = None,
        intended_strike: float | None = None,
    ) -> bool:
        """Record a skipped trade (pipeline rejected or other filter)."""
        # Pick the most specific reason from failures
        mapped = "manual_override"
        for fr in failure_reasons:
            fr_lower = fr.lower()
            for key, val in _REJECTION_REASON_MAP.items():
                if key in fr_lower:
                    mapped = val
                    break
            else:
                continue
            break

        return await self.record_execution_decision(
            alert_id=alert_id,
            decision="skipped",
            reason=mapped,
            actual_contracts=0,
            intended_strike=intended_strike,
            conviction_score=conviction,
            notes="; ".join(failure_reasons)[:500],
        )

    async def record_executed(
        self,
        alert_id: str,
        contracts: int,
        intended_contracts: int | None = None,
        strike: float | None = None,
        conviction: int | None = None,
    ) -> bool:
        """Record a successful execution decision."""
        return await self.record_execution_decision(
            alert_id=alert_id,
            decision="executed",
            reason="executed_normal",
            intended_contracts=intended_contracts,
            actual_contracts=contracts,
            actual_strike=strike,
            conviction_score=conviction,
        )


def map_exit_reason(our_reason: str) -> str:
    """Map our V5 FSM exit reason to Vince's close_reason enum."""
    return _EXIT_REASON_MAP.get(our_reason, "manual")
