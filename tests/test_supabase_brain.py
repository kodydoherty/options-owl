"""Comprehensive tests for Supabase Shared Brain integration.

Tests the client module, error handling, recovery queue, and integration
with paper_trader entry/exit paths.  Covers unit, integration, and e2e.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from options_owl.collectors.supabase_brain import (
    SupabaseBrain,
    _EXIT_REASON_MAP,
    _REJECTION_REASON_MAP,
    map_exit_reason,
)
from options_owl.config.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    defaults = {
        "DISCORD_TOKEN": "fake",
        "DB_PATH": "journal/test.db",
        "ENABLE_SUPABASE_BRAIN": True,
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test-anon-key",
        "SUPABASE_WEBULL_JWT": "test-jwt",
        "N8N_WEBHOOK_CLOSE_URL": "https://test.n8n.cloud/webhook/test",
        "AGENT_ID": "test-agent",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_brain(**settings_overrides) -> SupabaseBrain:
    return SupabaseBrain(_make_settings(**settings_overrides))


def _mock_client(status_code=201, json_data=None, text=""):
    """Create a mock httpx client with a preconfigured response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    if json_data is not None:
        mock_resp.json.return_value = json_data
    mock_resp.text = text

    client = AsyncMock()
    client.get = AsyncMock(return_value=mock_resp)
    client.post = AsyncMock(return_value=mock_resp)
    client.is_closed = False
    return client


# ---------------------------------------------------------------------------
# Settings defaults
# ---------------------------------------------------------------------------

class TestSettingsDefaults:
    def test_supabase_defaults_in_code(self):
        """Verify the code defaults (env file may override URL/keys)."""
        s = Settings(DISCORD_TOKEN="fake")
        # ENABLE is always False by default (not set in .env)
        assert s.ENABLE_SUPABASE_BRAIN is False
        assert s.SUPABASE_ACCOUNT_STATE_INTERVAL_SEC == 300
        # URL/keys may be set from .env — just verify the types
        assert isinstance(s.SUPABASE_URL, str)
        assert isinstance(s.SUPABASE_ANON_KEY, str)
        assert isinstance(s.SUPABASE_WEBULL_JWT, str)


# ---------------------------------------------------------------------------
# SupabaseBrain.enabled property
# ---------------------------------------------------------------------------

class TestEnabled:
    def test_enabled_when_all_set(self):
        brain = _make_brain()
        assert brain.enabled is True

    def test_disabled_when_flag_off(self):
        brain = _make_brain(ENABLE_SUPABASE_BRAIN=False)
        assert brain.enabled is False

    def test_disabled_when_url_missing(self):
        brain = _make_brain(SUPABASE_URL="")
        assert brain.enabled is False

    def test_disabled_when_anon_key_missing(self):
        brain = _make_brain(SUPABASE_ANON_KEY="")
        assert brain.enabled is False

    def test_disabled_when_jwt_missing(self):
        brain = _make_brain(SUPABASE_WEBULL_JWT="")
        assert brain.enabled is False

    def test_disabled_when_agent_id_missing(self):
        brain = _make_brain(AGENT_ID="")
        assert brain.enabled is False

    def test_agent_id_stored(self):
        brain = _make_brain(AGENT_ID="owlet_kody")
        assert brain._agent_id == "owlet_kody"


# ---------------------------------------------------------------------------
# Agent ID injection
# ---------------------------------------------------------------------------

class TestAgentIdInjection:
    @pytest.mark.asyncio
    async def test_agent_id_injected_into_every_write(self):
        """Every _write() call must include agent_id in payload."""
        brain = _make_brain(AGENT_ID="owlet_kody")
        brain._client = _mock_client(status_code=201)

        await brain._write("fills", {"alert_id": "abc"}, "test")

        call_args = brain._client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["agent_id"] == "owlet_kody"

    @pytest.mark.asyncio
    async def test_agent_id_does_not_overwrite_explicit(self):
        """agent_id in _write is always set from settings, not from caller payload."""
        brain = _make_brain(AGENT_ID="owlet_adam")
        brain._client = _mock_client(status_code=201)

        # Even if caller accidentally passes agent_id, _write overwrites it
        await brain._write("fills", {"alert_id": "abc", "agent_id": "wrong"}, "test")

        call_args = brain._client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["agent_id"] == "owlet_adam"

    @pytest.mark.asyncio
    async def test_record_fill_includes_agent_id(self):
        """High-level record_fill path injects agent_id."""
        brain = _make_brain(AGENT_ID="owlet_vinny")
        brain._client = _mock_client(status_code=201)

        await brain.record_fill(
            alert_id="abc-123",
            broker_order_id="WB-001",
            fill_price=1.50,
            fill_quantity=3,
            strike=450.0,
        )

        call_args = brain._client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["agent_id"] == "owlet_vinny"

    @pytest.mark.asyncio
    async def test_record_close_includes_agent_id(self):
        """High-level record_close path injects agent_id."""
        brain = _make_brain(AGENT_ID="owlet_yank")
        brain._client = _mock_client(status_code=201)

        await brain.record_close(
            alert_id="abc-123",
            close_price=2.50,
            exit_reason="adaptive_trail",
        )

        # record_close does two posts (close + webhook) — check the first
        first_post = brain._client.post.call_args_list[0]
        payload = first_post.kwargs.get("json") or first_post[1].get("json")
        assert payload["agent_id"] == "owlet_yank"


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

class TestLookupAlert:
    @pytest.mark.asyncio
    async def test_lookup_returns_alert(self):
        brain = _make_brain()
        brain._client = _mock_client(200, [
            {"alert_id": "abc-123", "ticker": "NVDA", "conviction_0_100": 85}
        ])

        result = await brain.lookup_alert("NVDA", "bearish")
        assert result is not None
        assert result["alert_id"] == "abc-123"
        assert result["conviction_0_100"] == 85

        # Verify correct URL and params
        call_args = brain._client.get.call_args
        assert "v_alerts_with_conviction" in call_args[0][0]
        assert call_args[1]["params"]["ticker"] == "eq.NVDA"
        assert call_args[1]["params"]["direction"] == "eq.bearish"

    @pytest.mark.asyncio
    async def test_lookup_uppercases_ticker(self):
        brain = _make_brain()
        brain._client = _mock_client(200, [{"alert_id": "x", "ticker": "AAPL"}])

        await brain.lookup_alert("aapl", "bullish")
        call_args = brain._client.get.call_args
        assert call_args[1]["params"]["ticker"] == "eq.AAPL"

    @pytest.mark.asyncio
    async def test_lookup_returns_none_on_empty(self):
        brain = _make_brain()
        brain._client = _mock_client(200, [])

        result = await brain.lookup_alert("NVDA", "bearish")
        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_returns_none_on_error(self):
        brain = _make_brain()
        client = AsyncMock()
        client.get = AsyncMock(side_effect=Exception("timeout"))
        client.is_closed = False
        brain._client = client

        result = await brain.lookup_alert("NVDA", "bearish")
        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_returns_none_on_http_error(self):
        brain = _make_brain()
        brain._client = _mock_client(500)

        result = await brain.lookup_alert("NVDA", "bearish")
        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_disabled_returns_none(self):
        brain = _make_brain(ENABLE_SUPABASE_BRAIN=False)
        result = await brain.lookup_alert("NVDA", "bearish")
        assert result is None


class TestGetRiskContext:
    @pytest.mark.asyncio
    async def test_returns_context(self):
        brain = _make_brain()
        brain._client = _mock_client(200, [{"fomc_today": True, "vol_regime": "EXPANSION"}])

        result = await brain.get_risk_context()
        assert result["fomc_today"] is True

    @pytest.mark.asyncio
    async def test_returns_none_on_empty(self):
        brain = _make_brain()
        brain._client = _mock_client(200, [])

        result = await brain.get_risk_context()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        brain = _make_brain()
        client = AsyncMock()
        client.get = AsyncMock(side_effect=Exception("fail"))
        client.is_closed = False
        brain._client = client

        result = await brain.get_risk_context()
        assert result is None


class TestGetSystemHealth:
    @pytest.mark.asyncio
    async def test_returns_health(self):
        brain = _make_brain()
        brain._client = _mock_client(200, [{"wr_last_7d": 74, "losses_last_24h": 3}])

        result = await brain.get_system_health()
        assert result["wr_last_7d"] == 74

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self):
        brain = _make_brain(ENABLE_SUPABASE_BRAIN=False)
        result = await brain.get_system_health()
        assert result is None


class TestGetTierPerformance:
    @pytest.mark.asyncio
    async def test_returns_tiers(self):
        brain = _make_brain()
        tiers = [
            {"score_tier": "elite", "win_rate_pct": 92},
            {"score_tier": "strong", "win_rate_pct": 81},
        ]
        brain._client = _mock_client(200, tiers)

        result = await brain.get_tier_performance()
        assert len(result) == 2
        assert result[0]["score_tier"] == "elite"

        # Verify URL
        call_args = brain._client.get.call_args
        assert "v_tier_performance_30d" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        brain = _make_brain()
        brain._client = _mock_client(500)

        result = await brain.get_tier_performance()
        assert result == []

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        brain = _make_brain(ENABLE_SUPABASE_BRAIN=False)
        result = await brain.get_tier_performance()
        assert result == []


class TestGetSectorActivity:
    @pytest.mark.asyncio
    async def test_returns_sectors(self):
        brain = _make_brain()
        sectors = [{"sector": "tech", "alerts_today": 4, "executed_today": 3}]
        brain._client = _mock_client(200, sectors)

        result = await brain.get_sector_activity()
        assert len(result) == 1
        assert result[0]["sector"] == "tech"

        call_args = brain._client.get.call_args
        assert "v_sector_active_today" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        brain = _make_brain()
        client = AsyncMock()
        client.get = AsyncMock(side_effect=Exception("timeout"))
        client.is_closed = False
        brain._client = client

        result = await brain.get_sector_activity()
        assert result == []


class TestGetSlippageStats:
    @pytest.mark.asyncio
    async def test_returns_stats(self):
        brain = _make_brain()
        stats = [{"ticker": "PLTR", "fills": 18, "avg_slippage_pct": 18.5}]
        brain._client = _mock_client(200, stats)

        result = await brain.get_slippage_stats()
        assert len(result) == 1
        assert result[0]["ticker"] == "PLTR"

        call_args = brain._client.get.call_args
        assert "v_slippage_by_ticker_30d" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        brain = _make_brain(ENABLE_SUPABASE_BRAIN=False)
        result = await brain.get_slippage_stats()
        assert result == []


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

class TestWrite:
    @pytest.mark.asyncio
    async def test_write_success(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        ok = await brain._write("fills", {"test": 1}, "test write")
        assert ok is True

    @pytest.mark.asyncio
    async def test_write_200_is_success(self):
        brain = _make_brain()
        brain._client = _mock_client(200)

        ok = await brain._write("fills", {"test": 1}, "test write")
        assert ok is True

    @pytest.mark.asyncio
    async def test_write_duplicate_is_success(self):
        brain = _make_brain()
        brain._client = _mock_client(409)

        ok = await brain._write("fills", {"test": 1}, "duplicate write")
        assert ok is True

    @pytest.mark.asyncio
    async def test_write_failure_queues_recovery(self, tmp_path):
        brain = _make_brain()
        brain._client = _mock_client(500, text="Internal Server Error")

        with patch("options_owl.collectors.supabase_brain._RECOVERY_DIR", tmp_path):
            ok = await brain._write("fills", {"test": 1}, "failed write")
            assert ok is False
            # Verify recovery file was created
            recovery_files = list(tmp_path.glob("*.json"))
            assert len(recovery_files) == 1
            entry = json.loads(recovery_files[0].read_text())
            assert entry["endpoint"] == "fills"
            assert entry["payload"]["test"] == 1

    @pytest.mark.asyncio
    async def test_write_exception_queues_recovery(self, tmp_path):
        brain = _make_brain()
        client = AsyncMock()
        client.post = AsyncMock(side_effect=Exception("network error"))
        client.is_closed = False
        brain._client = client

        with patch("options_owl.collectors.supabase_brain._RECOVERY_DIR", tmp_path):
            ok = await brain._write("fills", {"test": 1}, "error write")
            assert ok is False
            recovery_files = list(tmp_path.glob("*.json"))
            assert len(recovery_files) == 1

    @pytest.mark.asyncio
    async def test_write_disabled_returns_false(self):
        brain = _make_brain(ENABLE_SUPABASE_BRAIN=False)
        ok = await brain._write("fills", {"test": 1}, "disabled")
        assert ok is False

    @pytest.mark.asyncio
    async def test_write_403_queues_recovery(self, tmp_path):
        """403 (permission denied) should queue for recovery."""
        brain = _make_brain()
        brain._client = _mock_client(403, text="Permission denied")

        with patch("options_owl.collectors.supabase_brain._RECOVERY_DIR", tmp_path):
            ok = await brain._write("fills", {"test": 1}, "forbidden")
            assert ok is False
            assert len(list(tmp_path.glob("*.json"))) == 1

    @pytest.mark.asyncio
    async def test_write_400_queues_recovery(self, tmp_path):
        """400 (bad payload) should still queue for recovery."""
        brain = _make_brain()
        brain._client = _mock_client(400, text="Bad Request")

        with patch("options_owl.collectors.supabase_brain._RECOVERY_DIR", tmp_path):
            ok = await brain._write("fills", {"bad": True}, "bad payload")
            assert ok is False


# ---------------------------------------------------------------------------
# High-level write methods
# ---------------------------------------------------------------------------

class TestRecordFill:
    @pytest.mark.asyncio
    async def test_record_fill_payload(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_fill(
            alert_id="abc-123",
            broker_order_id="WB-999",
            fill_price=1.42,
            fill_quantity=3,
            strike=190.0,
            slippage_pct=5.2,
        )

        call_args = brain._client.post.call_args
        payload = call_args[1]["json"]
        assert payload["alert_id"] == "abc-123"
        assert payload["broker_order_id"] == "WB-999"
        assert payload["fill_price"] == 1.42
        assert payload["fill_quantity"] == 3
        assert payload["strike_filled"] == 190.0
        assert payload["slippage_pct"] == 5.2
        assert "fill_time" in payload

        # Verify Prefer: return=minimal header
        headers = call_args[1]["headers"]
        assert headers["Prefer"] == "return=minimal"

    @pytest.mark.asyncio
    async def test_record_fill_correct_endpoint(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_fill(
            alert_id="x", broker_order_id="WB-1",
            fill_price=1.0, fill_quantity=1, strike=100.0,
        )

        url = brain._client.post.call_args[0][0]
        assert url.endswith("/rest/v1/fills")

    @pytest.mark.asyncio
    async def test_record_fill_optional_fields_omitted(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_fill(
            alert_id="x", broker_order_id="WB-1",
            fill_price=1.0, fill_quantity=1, strike=100.0,
        )

        payload = brain._client.post.call_args[1]["json"]
        assert "contract_symbol" not in payload
        assert "slippage_pct" not in payload

    @pytest.mark.asyncio
    async def test_record_fill_with_contract_symbol(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_fill(
            alert_id="x", broker_order_id="WB-1",
            fill_price=1.0, fill_quantity=1, strike=100.0,
            contract_symbol="O:AAPL260520C00190000",
        )

        payload = brain._client.post.call_args[1]["json"]
        assert payload["contract_symbol"] == "O:AAPL260520C00190000"


class TestRecordClose:
    @pytest.mark.asyncio
    async def test_record_close_maps_reason(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_close(
            alert_id="abc-123",
            close_price=1.85,
            exit_reason="adaptive_trail",
            pnl_pct=30.3,
            hold_minutes=47.5,
        )

        # Should have been called twice: Supabase + webhook
        assert brain._client.post.call_count == 2
        # First call is Supabase write
        supabase_call = brain._client.post.call_args_list[0]
        payload = supabase_call[1]["json"]
        assert payload["close_reason"] == "target_hit"  # mapped from adaptive_trail
        assert payload["real_pnl_pct"] == 30.3

    @pytest.mark.asyncio
    async def test_record_close_no_webhook(self):
        brain = _make_brain(N8N_WEBHOOK_CLOSE_URL="")
        brain._client = _mock_client(201)

        await brain.record_close(
            alert_id="abc-123",
            close_price=1.85,
            exit_reason="eod_cutoff",
        )

        # Only Supabase, no webhook
        assert brain._client.post.call_count == 1
        payload = brain._client.post.call_args[1]["json"]
        assert payload["close_reason"] == "eod"

    @pytest.mark.asyncio
    async def test_record_close_correct_endpoint(self):
        brain = _make_brain(N8N_WEBHOOK_CLOSE_URL="")
        brain._client = _mock_client(201)

        await brain.record_close(
            alert_id="x", close_price=1.0, exit_reason="manual",
        )

        url = brain._client.post.call_args[0][0]
        assert url.endswith("/rest/v1/closes")

    @pytest.mark.asyncio
    async def test_record_close_optional_fields(self):
        brain = _make_brain(N8N_WEBHOOK_CLOSE_URL="")
        brain._client = _mock_client(201)

        await brain.record_close(
            alert_id="x", close_price=2.0, exit_reason="graduated_stop",
            pnl_pct=-15.5, pnl_usd=-155.0, hold_minutes=22.3, peak_premium=2.45,
        )

        payload = brain._client.post.call_args[1]["json"]
        assert payload["real_pnl_pct"] == -15.5
        assert payload["real_pnl_usd"] == -155.0
        assert payload["hold_minutes"] == 22.3
        assert payload["peak_premium"] == 2.45

    @pytest.mark.asyncio
    async def test_webhook_failure_doesnt_affect_result(self):
        """Webhook failure should not prevent success return."""
        brain = _make_brain()

        # First call (Supabase) succeeds, second (webhook) fails
        supabase_resp = MagicMock()
        supabase_resp.status_code = 201
        webhook_exc = Exception("webhook down")

        client = AsyncMock()
        client.post = AsyncMock(side_effect=[supabase_resp, webhook_exc])
        client.is_closed = False
        brain._client = client

        ok = await brain.record_close(
            alert_id="x", close_price=1.0, exit_reason="manual",
        )
        assert ok is True  # Supabase succeeded, that's what matters


class TestRecordExecutionDecision:
    @pytest.mark.asyncio
    async def test_executed_decision(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_executed(
            alert_id="abc-123",
            contracts=3,
            intended_contracts=3,
            strike=190.0,
            conviction=85,
        )

        payload = brain._client.post.call_args[1]["json"]
        assert payload["decision"] == "executed"
        assert payload["reason"] == "executed_normal"
        assert payload["actual_contracts"] == 3
        assert payload["intended_contracts"] == 3
        assert payload["actual_strike"] == 190.0
        assert payload["conviction_score"] == 85

    @pytest.mark.asyncio
    async def test_executed_without_optional_fields(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_executed(alert_id="abc-123", contracts=2)

        payload = brain._client.post.call_args[1]["json"]
        assert payload["decision"] == "executed"
        assert payload["actual_contracts"] == 2
        assert "intended_contracts" not in payload
        assert "actual_strike" not in payload
        assert "conviction_score" not in payload

    @pytest.mark.asyncio
    async def test_skip_decision_maps_reason(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_skip(
            alert_id="abc-123",
            failure_reasons=["concurrent_limit: 4/4 positions open"],
            signal_score=90,
            conviction=75,
            intended_strike=195.0,
        )

        payload = brain._client.post.call_args[1]["json"]
        assert payload["decision"] == "skipped"
        assert payload["reason"] == "position_limit"
        assert payload["actual_contracts"] == 0
        assert payload["intended_strike"] == 195.0

    @pytest.mark.asyncio
    async def test_skip_with_multiple_failures_picks_first_match(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_skip(
            alert_id="abc-123",
            failure_reasons=[
                "score_too_low: 72 < 78",
                "spread_gate: spread 15% > 12%",
            ],
        )

        payload = brain._client.post.call_args[1]["json"]
        # Should pick first match: score_too_low → low_conviction
        assert payload["reason"] == "low_conviction"

    @pytest.mark.asyncio
    async def test_skip_unknown_reason_falls_back(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_skip(
            alert_id="abc-123",
            failure_reasons=["some_unknown_gate: failed"],
        )

        payload = brain._client.post.call_args[1]["json"]
        assert payload["reason"] == "manual_override"

    @pytest.mark.asyncio
    async def test_decision_endpoint_correct(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_execution_decision(
            alert_id="x", decision="executed", reason="executed_normal",
        )

        url = brain._client.post.call_args[0][0]
        assert url.endswith("/rest/v1/execution_decisions")

    @pytest.mark.asyncio
    async def test_decision_with_intended_and_actual_strike(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.record_execution_decision(
            alert_id="x", decision="modified_strike", reason="better_strike",
            intended_strike=195.0, actual_strike=192.5,
            intended_contracts=3, actual_contracts=3,
        )

        payload = brain._client.post.call_args[1]["json"]
        assert payload["intended_strike"] == 195.0
        assert payload["actual_strike"] == 192.5
        assert payload["intended_contracts"] == 3

    @pytest.mark.asyncio
    async def test_notes_truncated_to_500(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        long_notes = "x" * 600
        await brain.record_execution_decision(
            alert_id="x", decision="skipped", reason="test",
            notes=long_notes,
        )

        payload = brain._client.post.call_args[1]["json"]
        assert len(payload["notes"]) == 500


class TestAccountState:
    @pytest.mark.asyncio
    async def test_push_account_state(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.push_account_state(
            equity_usd=23000.0,
            cash_usd=12000.0,
            daily_pnl_usd=-500.0,
            open_positions=3,
            buying_power=10000.0,
        )

        payload = brain._client.post.call_args[1]["json"]
        assert payload["equity_usd"] == 23000.0
        assert payload["daily_pnl_pct"] == round(-500.0 / 23000.0 * 100, 2)
        assert payload["open_positions"] == 3
        assert payload["buying_power"] == 10000.0

    @pytest.mark.asyncio
    async def test_account_state_zero_equity(self):
        """Edge case: zero equity should not divide by zero."""
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.push_account_state(
            equity_usd=0.0, cash_usd=0.0, daily_pnl_usd=0.0, open_positions=0,
        )

        payload = brain._client.post.call_args[1]["json"]
        assert payload["daily_pnl_pct"] == 0

    @pytest.mark.asyncio
    async def test_account_state_endpoint_correct(self):
        brain = _make_brain()
        brain._client = _mock_client(201)

        await brain.push_account_state(
            equity_usd=1000.0, cash_usd=500.0,
            daily_pnl_usd=-10.0, open_positions=1,
        )

        url = brain._client.post.call_args[0][0]
        assert url.endswith("/rest/v1/account_state")


# ---------------------------------------------------------------------------
# Recovery queue
# ---------------------------------------------------------------------------

class TestRecoveryQueue:
    @pytest.mark.asyncio
    async def test_replay_recovery_queue(self, tmp_path):
        brain = _make_brain()

        # Create a fake recovery file
        entry = {
            "endpoint": "fills",
            "payload": {"alert_id": "test", "broker_order_id": "WB-1"},
            "context": "test fill",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        recovery_file = tmp_path / "test_recovery.json"
        recovery_file.write_text(json.dumps(entry))

        brain._client = _mock_client(201)

        with patch("options_owl.collectors.supabase_brain._RECOVERY_DIR", tmp_path):
            count = await brain.replay_recovery_queue()

        assert count == 1
        # File should be deleted after successful replay
        assert not recovery_file.exists()

    @pytest.mark.asyncio
    async def test_replay_keeps_failed(self, tmp_path):
        brain = _make_brain()

        entry = {
            "endpoint": "fills",
            "payload": {"alert_id": "test"},
            "context": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        recovery_file = tmp_path / "test_recovery.json"
        recovery_file.write_text(json.dumps(entry))

        brain._client = _mock_client(500)

        with patch("options_owl.collectors.supabase_brain._RECOVERY_DIR", tmp_path):
            count = await brain.replay_recovery_queue()

        assert count == 0
        # File should still exist
        assert recovery_file.exists()

    @pytest.mark.asyncio
    async def test_replay_409_deletes_file(self, tmp_path):
        """409 (duplicate) during replay should still delete the file."""
        brain = _make_brain()

        entry = {
            "endpoint": "fills",
            "payload": {"alert_id": "test", "broker_order_id": "WB-dup"},
            "context": "duplicate",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        recovery_file = tmp_path / "test_dup.json"
        recovery_file.write_text(json.dumps(entry))

        brain._client = _mock_client(409)

        with patch("options_owl.collectors.supabase_brain._RECOVERY_DIR", tmp_path):
            count = await brain.replay_recovery_queue()

        assert count == 1
        assert not recovery_file.exists()

    @pytest.mark.asyncio
    async def test_replay_disabled(self):
        brain = _make_brain(ENABLE_SUPABASE_BRAIN=False)
        count = await brain.replay_recovery_queue()
        assert count == 0

    @pytest.mark.asyncio
    async def test_replay_no_directory(self):
        brain = _make_brain()
        with patch("options_owl.collectors.supabase_brain._RECOVERY_DIR", Path("/nonexistent")):
            count = await brain.replay_recovery_queue()
        assert count == 0

    @pytest.mark.asyncio
    async def test_replay_empty_directory(self, tmp_path):
        brain = _make_brain()
        with patch("options_owl.collectors.supabase_brain._RECOVERY_DIR", tmp_path):
            count = await brain.replay_recovery_queue()
        assert count == 0

    @pytest.mark.asyncio
    async def test_queue_for_recovery_creates_file(self, tmp_path):
        brain = _make_brain()
        with patch("options_owl.collectors.supabase_brain._RECOVERY_DIR", tmp_path):
            brain._queue_for_recovery("fills", {"test": 1}, "test context")

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        entry = json.loads(files[0].read_text())
        assert entry["endpoint"] == "fills"
        assert entry["payload"] == {"test": 1}
        assert entry["context"] == "test context"
        assert "timestamp" in entry


# ---------------------------------------------------------------------------
# Exit reason mapping — completeness
# ---------------------------------------------------------------------------

class TestExitReasonMapping:
    def test_all_v5_exit_reasons_mapped(self):
        """Every ExitReason enum value should have a mapping."""
        from options_owl.risk.exit_v5.types import ExitReason

        for reason in ExitReason:
            if reason == ExitReason.HOLD:
                continue  # HOLD never exits
            mapped = map_exit_reason(reason.value)
            assert mapped != "manual", (
                f"ExitReason.{reason.name} (value={reason.value!r}) falls through to "
                f"'manual' — needs explicit mapping in _EXIT_REASON_MAP"
            )

    def test_all_mapped_values_in_vinces_enum(self):
        """All mapped values must be in Vince's allowed close_reason enum."""
        allowed = {
            "target_hit", "partial_50", "partial_75", "stop_loss",
            "time_stop", "manual", "eod", "expired", "momentum_fade",
        }
        for our_reason, vinces_reason in _EXIT_REASON_MAP.items():
            assert vinces_reason in allowed, (
                f"{our_reason!r} maps to {vinces_reason!r} which is not in Vince's "
                f"allowed values: {allowed}"
            )

    def test_specific_mappings(self):
        """Spot-check important mappings."""
        assert map_exit_reason("eod_cutoff") == "eod"
        assert map_exit_reason("adaptive_trail") == "target_hit"
        assert map_exit_reason("graduated_stop") == "stop_loss"
        assert map_exit_reason("confirmed_stop") == "stop_loss"
        assert map_exit_reason("hard_stop") == "stop_loss"
        assert map_exit_reason("theta_bleed") == "time_stop"
        assert map_exit_reason("theta_timer") == "time_stop"
        assert map_exit_reason("sideways_scalp") == "momentum_fade"
        assert map_exit_reason("scaleout") == "partial_50"
        assert map_exit_reason("expired") == "expired"
        assert map_exit_reason("bid_disappearance") == "momentum_fade"

    def test_unknown_reason_maps_to_manual(self):
        assert map_exit_reason("unknown_gate") == "manual"

    def test_rejection_reasons_mapped(self):
        """Pipeline rejection reasons should all map to valid decision reasons."""
        valid_reasons = {
            "low_conviction", "spread_too_wide", "duplicate_alert_recent",
            "position_limit", "sector_concentration_max", "daily_loss_limit",
            "low_buying_power", "time_filter",
        }
        for key, val in _REJECTION_REASON_MAP.items():
            assert val in valid_reasons, f"Unexpected rejection mapping: {key} → {val}"


# ---------------------------------------------------------------------------
# Headers verification
# ---------------------------------------------------------------------------

class TestHeaders:
    def test_write_headers_include_prefer(self):
        brain = _make_brain()
        assert brain._write_headers["Prefer"] == "return=minimal"

    def test_read_headers_no_prefer(self):
        brain = _make_brain()
        assert "Prefer" not in brain._read_headers

    def test_auth_headers_correct(self):
        brain = _make_brain()
        assert brain._read_headers["apikey"] == "test-anon-key"
        assert brain._read_headers["Authorization"] == "Bearer test-jwt"

    def test_write_headers_have_both_auth(self):
        """Write headers must have BOTH apikey and Authorization."""
        brain = _make_brain()
        assert "apikey" in brain._write_headers
        assert "Authorization" in brain._write_headers
        assert brain._write_headers["apikey"] != brain._write_headers["Authorization"]

    def test_content_type_json(self):
        brain = _make_brain()
        assert brain._read_headers["Content-Type"] == "application/json"
        assert brain._write_headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Integration: paper_trader has supabase attribute + DB column
# ---------------------------------------------------------------------------

class TestPaperTraderIntegration:
    @pytest.mark.asyncio
    async def test_paper_trader_has_supabase_attr(self):
        from options_owl.execution.paper_trader import PaperTrader
        settings = Settings(
            DISCORD_TOKEN="fake",
            DB_PATH="/tmp/test_sb_integration.db",
            ENABLE_SMART_ENTRY=False,
        )
        pt = PaperTrader(settings)
        assert pt.supabase is None  # default

    @pytest.mark.asyncio
    async def test_supabase_alert_id_column_exists(self, tmp_path):
        """Verify the migration adds the supabase_alert_id column."""
        import aiosqlite
        from options_owl.execution.paper_trader import init_paper_db

        db_path = str(tmp_path / "test.db")
        await init_paper_db(db_path)

        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute("PRAGMA table_info(paper_trades)")
            columns = {row[1] for row in await cursor.fetchall()}

        assert "supabase_alert_id" in columns


# ---------------------------------------------------------------------------
# Integration: close_trade sends supabase_alert_id
# ---------------------------------------------------------------------------

class TestCloseTradeSupabase:
    @pytest.mark.asyncio
    async def test_close_trade_fires_supabase_record_close(self, tmp_path):
        """When close_trade() runs with a supabase-linked trade, record_close fires."""
        import aiosqlite
        from options_owl.execution.paper_trader import PaperTrader, init_paper_db

        db_path = str(tmp_path / "close_test.db")
        settings = Settings(
            DISCORD_TOKEN="fake",
            DB_PATH=db_path,
            PORTFOLIO_SIZE=5000,
            ENABLE_SMART_ENTRY=False,
        )
        await init_paper_db(db_path)

        pt = PaperTrader(settings)
        brain = _make_brain()
        brain._client = _mock_client(201)
        pt.supabase = brain

        # Insert a fake open trade with supabase_alert_id and webull_order_id
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                "INSERT INTO paper_portfolio (strategy, starting_balance, current_balance, total_trades, wins, losses, daily_pnl, last_trade_date, created_at) "
                "VALUES ('B', 5000, 5000, 0, 0, 0, 0, '', datetime('now'))"
            )
            await conn.execute(
                "INSERT INTO paper_trades "
                "(signal_id, ticker, direction, sentiment, score, strength, bot_source, "
                "entry_price, strike, option_type, contracts, premium_per_contract, total_cost, "
                "target_1, stop_price, strategy, supabase_alert_id, webull_order_id, status, opened_at) "
                "VALUES (1, 'NVDA', 'call', 'bullish', 90, 'strong', 'atlas', "
                "130.0, 130.0, 'call', 2, 1.50, 300.0, "
                "135.0, 128.0, 'B', 'alert-uuid-123', 'WB-ORDER-1', 'open', '2026-05-18T10:00:00')"
            )
            await conn.commit()

        result = await pt.close_trade(
            trade_id=1,
            exit_price=132.0,
            exit_premium=2.00,
            reason="adaptive_trail",
        )

        assert result is not None
        assert result["pnl"] > 0

        # Give fire-and-forget task a chance to run
        await asyncio.sleep(0.05)

        # Verify Supabase record_close was called
        # The _write call goes through the client
        supabase_calls = [
            c for c in brain._client.post.call_args_list
            if "/rest/v1/closes" in str(c)
        ]
        assert len(supabase_calls) >= 1
        payload = supabase_calls[0][1]["json"]
        assert payload["alert_id"] == "alert-uuid-123"
        assert payload["close_reason"] == "target_hit"


# ---------------------------------------------------------------------------
# Concurrency safety: fire-and-forget tasks don't crash
# ---------------------------------------------------------------------------

class TestFireAndForget:
    @pytest.mark.asyncio
    async def test_write_in_create_task_doesnt_crash(self):
        """Verify that asyncio.create_task(brain.record_*()) is safe."""
        brain = _make_brain()
        brain._client = _mock_client(201)

        # Simulate what paper_trader does: create_task
        task = asyncio.create_task(brain.record_executed(
            alert_id="test-123",
            contracts=2,
        ))
        await task  # should complete without error
        assert task.done()
        assert task.exception() is None

    @pytest.mark.asyncio
    async def test_write_error_in_task_doesnt_propagate(self, tmp_path):
        """Even if write fails, the task completes (doesn't propagate)."""
        brain = _make_brain()
        client = AsyncMock()
        client.post = AsyncMock(side_effect=Exception("network down"))
        client.is_closed = False
        brain._client = client

        with patch("options_owl.collectors.supabase_brain._RECOVERY_DIR", tmp_path):
            task = asyncio.create_task(brain.record_executed(
                alert_id="test-123",
                contracts=2,
            ))
            await task
            assert task.done()
            assert task.exception() is None  # error was caught internally

    @pytest.mark.asyncio
    async def test_multiple_concurrent_tasks(self):
        """Multiple fire-and-forget tasks should all complete independently."""
        brain = _make_brain()
        brain._client = _mock_client(201)

        tasks = [
            asyncio.create_task(brain.record_executed(alert_id=f"t-{i}", contracts=1))
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        assert all(r is True for r in results)


# ---------------------------------------------------------------------------
# Source code safety: verify no uninitialized variables
# ---------------------------------------------------------------------------

class TestSourceCodeSafety:
    def test_exit_reason_map_keys_are_strings(self):
        """All keys in _EXIT_REASON_MAP should be lowercase strings."""
        for key in _EXIT_REASON_MAP:
            assert isinstance(key, str)
            assert key == key.lower(), f"Key {key!r} should be lowercase"

    def test_rejection_reason_map_keys_are_strings(self):
        """All keys in _REJECTION_REASON_MAP should be lowercase strings."""
        for key in _REJECTION_REASON_MAP:
            assert isinstance(key, str)
            assert key == key.lower(), f"Key {key!r} should be lowercase"

    def test_record_executed_signature_has_intended_contracts(self):
        """record_executed must accept intended_contracts param."""
        import inspect
        sig = inspect.signature(SupabaseBrain.record_executed)
        assert "intended_contracts" in sig.parameters

    def test_record_executed_signature_has_strike(self):
        """record_executed must accept strike param."""
        import inspect
        sig = inspect.signature(SupabaseBrain.record_executed)
        assert "strike" in sig.parameters

    def test_record_skip_signature_has_intended_strike(self):
        """record_skip must accept intended_strike param."""
        import inspect
        sig = inspect.signature(SupabaseBrain.record_skip)
        assert "intended_strike" in sig.parameters

    def test_record_execution_decision_has_strike_params(self):
        """record_execution_decision must accept intended_strike and actual_strike."""
        import inspect
        sig = inspect.signature(SupabaseBrain.record_execution_decision)
        assert "intended_strike" in sig.parameters
        assert "actual_strike" in sig.parameters


# ---------------------------------------------------------------------------
# E2E: full signal → decision → fill → close lifecycle
# ---------------------------------------------------------------------------

class TestE2ELifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle_executed(self):
        """Simulate a full trade lifecycle through Supabase brain."""
        brain = _make_brain()
        brain._client = _mock_client(201)

        alert_id = "lifecycle-test-001"

        # 1. Record execution decision
        ok = await brain.record_executed(
            alert_id=alert_id,
            contracts=3,
            intended_contracts=3,
            strike=190.0,
            conviction=85,
        )
        assert ok is True

        # 2. Record fill
        ok = await brain.record_fill(
            alert_id=alert_id,
            broker_order_id="WB-LIFECYCLE-1",
            fill_price=1.42,
            fill_quantity=3,
            strike=190.0,
            slippage_pct=5.2,
        )
        assert ok is True

        # 3. Push account state
        ok = await brain.push_account_state(
            equity_usd=8000.0,
            cash_usd=6500.0,
            daily_pnl_usd=150.0,
            open_positions=1,
            buying_power=6500.0,
        )
        assert ok is True

        # 4. Record close
        ok = await brain.record_close(
            alert_id=alert_id,
            close_price=1.85,
            exit_reason="adaptive_trail",
            pnl_pct=30.3,
            pnl_usd=129.0,
            hold_minutes=47.5,
            peak_premium=2.10,
        )
        assert ok is True

        # Verify all 5 POST calls made (4 writes + 1 webhook)
        assert brain._client.post.call_count == 5

    @pytest.mark.asyncio
    async def test_full_lifecycle_skipped(self):
        """Simulate a signal that gets skipped through Supabase brain."""
        brain = _make_brain()
        brain._client = _mock_client(201)

        alert_id = "skip-test-001"

        # Record skip decision with rich context
        ok = await brain.record_skip(
            alert_id=alert_id,
            failure_reasons=["concurrent_limit: 5/5 positions open", "premium_cap: $7.50 > $6.00"],
            signal_score=88,
            conviction=72,
            intended_strike=195.0,
        )
        assert ok is True

        payload = brain._client.post.call_args[1]["json"]
        assert payload["decision"] == "skipped"
        assert payload["reason"] == "position_limit"  # first match
        assert payload["actual_contracts"] == 0
        assert payload["intended_strike"] == 195.0
        assert "concurrent_limit" in payload["notes"]


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------

class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_get_client_creates_new(self):
        brain = _make_brain()
        assert brain._client is None
        client = await brain._get_client()
        assert client is not None
        assert brain._client is client
        await brain.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self):
        brain = _make_brain()
        c1 = await brain._get_client()
        c2 = await brain._get_client()
        assert c1 is c2
        await brain.close()

    @pytest.mark.asyncio
    async def test_close_handles_none(self):
        brain = _make_brain()
        await brain.close()  # should not raise

    def test_webhook_url_stored(self):
        brain = _make_brain()
        assert brain._webhook_url == "https://test.n8n.cloud/webhook/test"

    def test_webhook_url_empty(self):
        brain = _make_brain(N8N_WEBHOOK_CLOSE_URL="")
        assert brain._webhook_url == ""


# ---------------------------------------------------------------------------
# Conviction score clamp tests
# ---------------------------------------------------------------------------

class TestConvictionClamp:
    """Conviction scores should be clamped to 0-100 for team compatibility."""

    @pytest.mark.asyncio
    async def test_clamps_high_score(self):
        """Score 177 (our scale) should be clamped to 100."""
        brain = _make_brain()
        mock = _mock_client(status_code=201)

        with patch.object(brain, "_get_client", return_value=mock):
            await brain.record_execution_decision(
                alert_id="test-alert",
                decision="executed",
                reason="executed_normal",
                conviction_score=177,
            )

        call_json = mock.post.call_args[1]["json"]
        assert call_json["conviction_score"] == 100

    @pytest.mark.asyncio
    async def test_clamps_negative_score(self):
        """Negative score (shouldn't happen) clamped to 0."""
        brain = _make_brain()
        mock = _mock_client(status_code=201)

        with patch.object(brain, "_get_client", return_value=mock):
            await brain.record_execution_decision(
                alert_id="test-alert",
                decision="skipped",
                reason="low_conviction",
                conviction_score=-5,
            )

        call_json = mock.post.call_args[1]["json"]
        assert call_json["conviction_score"] == 0

    @pytest.mark.asyncio
    async def test_normal_score_unchanged(self):
        """Score 85 (within 0-100) passes through unchanged."""
        brain = _make_brain()
        mock = _mock_client(status_code=201)

        with patch.object(brain, "_get_client", return_value=mock):
            await brain.record_execution_decision(
                alert_id="test-alert",
                decision="executed",
                reason="executed_normal",
                conviction_score=85,
            )

        call_json = mock.post.call_args[1]["json"]
        assert call_json["conviction_score"] == 85

    @pytest.mark.asyncio
    async def test_record_skip_clamps(self):
        """record_skip flows through record_execution_decision and clamps."""
        brain = _make_brain()
        mock = _mock_client(status_code=201)

        with patch.object(brain, "_get_client", return_value=mock):
            await brain.record_skip(
                alert_id="test-alert",
                failure_reasons=["score_too_low"],
                conviction=150,
            )

        call_json = mock.post.call_args[1]["json"]
        assert call_json["conviction_score"] == 100

    @pytest.mark.asyncio
    async def test_record_executed_clamps(self):
        """record_executed flows through record_execution_decision and clamps."""
        brain = _make_brain()
        mock = _mock_client(status_code=201)

        with patch.object(brain, "_get_client", return_value=mock):
            await brain.record_executed(
                alert_id="test-alert",
                contracts=5,
                conviction=177,
            )

        call_json = mock.post.call_args[1]["json"]
        assert call_json["conviction_score"] == 100

    @pytest.mark.asyncio
    async def test_float_score_coerced_to_int(self):
        """Float conviction (from JSON) should be coerced to int before clamp."""
        brain = _make_brain()
        mock = _mock_client(status_code=201)

        with patch.object(brain, "_get_client", return_value=mock):
            await brain.record_execution_decision(
                alert_id="test-alert",
                decision="executed",
                reason="executed_normal",
                conviction_score=85.7,  # float from JSON
            )

        call_json = mock.post.call_args[1]["json"]
        assert call_json["conviction_score"] == 85
        assert isinstance(call_json["conviction_score"], int)

    @pytest.mark.asyncio
    async def test_string_score_handled(self):
        """String conviction (edge case) should not crash — just skipped."""
        brain = _make_brain()
        mock = _mock_client(status_code=201)

        with patch.object(brain, "_get_client", return_value=mock):
            await brain.record_execution_decision(
                alert_id="test-alert",
                decision="executed",
                reason="executed_normal",
                conviction_score="not_a_number",  # type: ignore[arg-type]
            )

        call_json = mock.post.call_args[1]["json"]
        assert "conviction_score" not in call_json  # skipped, not crashed


# ---------------------------------------------------------------------------
# NBBO logging in fills
# ---------------------------------------------------------------------------

class TestNBBOInFills:
    """NBBO data should be included in fill records for execution quality."""

    @pytest.mark.asyncio
    async def test_nbbo_included_in_fill(self):
        """When nbbo_at_order is provided, raw_broker_data should contain NBBO."""
        brain = _make_brain()
        mock = _mock_client(status_code=201)

        nbbo = {"bid": 1.50, "ask": 1.65, "mid": 1.575}

        with patch.object(brain, "_get_client", return_value=mock):
            await brain.record_fill(
                alert_id="test-alert",
                broker_order_id="WB123",
                fill_price=1.62,
                fill_quantity=5,
                strike=550.0,
                nbbo_at_order=nbbo,
            )

        call_json = mock.post.call_args[1]["json"]
        assert "raw_broker_data" in call_json
        raw = json.loads(call_json["raw_broker_data"])
        assert raw["nbbo_bid"] == 1.50
        assert raw["nbbo_ask"] == 1.65
        assert raw["mid_at_order_time"] == 1.575
        assert raw["intended_price"] == 1.62

    @pytest.mark.asyncio
    async def test_no_nbbo_no_raw_data(self):
        """When nbbo_at_order is None, raw_broker_data should NOT be set."""
        brain = _make_brain()
        mock = _mock_client(status_code=201)

        with patch.object(brain, "_get_client", return_value=mock):
            await brain.record_fill(
                alert_id="test-alert",
                broker_order_id="WB123",
                fill_price=1.62,
                fill_quantity=5,
                strike=550.0,
            )

        call_json = mock.post.call_args[1]["json"]
        assert "raw_broker_data" not in call_json

    @pytest.mark.asyncio
    async def test_zero_values_still_included(self):
        """Zero bid/ask/mid should still be included (is not None check, not falsy)."""
        brain = _make_brain()
        mock = _mock_client(status_code=201)

        nbbo = {"bid": 0.0, "ask": 1.65, "mid": 0}

        with patch.object(brain, "_get_client", return_value=mock):
            await brain.record_fill(
                alert_id="test-alert",
                broker_order_id="WB123",
                fill_price=1.62,
                fill_quantity=5,
                strike=550.0,
                nbbo_at_order=nbbo,
            )

        call_json = mock.post.call_args[1]["json"]
        raw = json.loads(call_json["raw_broker_data"])
        assert raw["nbbo_bid"] == 0.0  # zero is valid, should be included
        assert raw["nbbo_ask"] == 1.65
        assert raw["mid_at_order_time"] == 0  # zero is valid, should be included


# Smart entry NBBO return tested in test_ml_first_strategy.py::TestSmartEntry
# (7 tests covering all return paths with 3-tuple unpacking)


# ---------------------------------------------------------------------------
# Fire-and-forget error handling
# ---------------------------------------------------------------------------

class TestFireAndForgetHelper:
    """_fire_and_forget helper should catch and log exceptions without crashing."""

    @pytest.mark.asyncio
    async def test_successful_task(self):
        """Successful coroutine should complete normally."""
        from options_owl.execution.paper_trader import _fire_and_forget

        called = []

        async def ok_coro():
            called.append(True)

        _fire_and_forget(ok_coro())
        await asyncio.sleep(0.05)
        assert called == [True]

    @pytest.mark.asyncio
    async def test_failing_task_does_not_crash(self):
        """Failing coroutine should be caught, not propagate."""
        from options_owl.execution.paper_trader import _fire_and_forget

        async def bad_coro():
            raise ValueError("test error")

        # Should not raise — error is caught by done callback
        _fire_and_forget(bad_coro())
        await asyncio.sleep(0.05)
        # If we get here without exception, the test passes
