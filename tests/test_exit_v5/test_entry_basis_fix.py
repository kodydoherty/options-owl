"""Entry-basis correctness (2026-06-26): the FSM must track the ACTUAL Webull fill, not
the pre-fill quote. Two layers:
  - _reconcile_entry_to_fill: re-anchor a cached state to webull_entry_fill_price once known
  - evaluate() backstop: a just-opened trade can't be instantly +70%+ → re-anchor to live premium

Root cause it guards: adam QQQ #334 — FSM cached entry $1.25 (pre-fill quote) vs fill $2.68,
so a real +49% peak was dumped at +3.7% by the profit-lock anchored to a phantom +220%.
"""

from datetime import datetime

from options_owl.risk.exit_v5.monitor_bridge import V5MonitorBridge


class FakeSettings:
    pass


def _trade(trade_id=1, ticker="QQQ", premium=1.25, contracts=1,
           opened_at="2026-04-28T14:00:00", **kw):
    d = {
        "id": trade_id, "ticker": ticker, "option_type": "call",
        "premium_per_contract": premium, "contracts": contracts,
        "opened_at": opened_at, "score": 85, "entry_price": 712.0,
        "mfe_premium": None, "strike": 712.0, "status": "open",
    }
    d.update(kw)
    return d


def _now_et(hour=10, minute=0, second=0):
    return datetime(2026, 4, 28, hour, minute, second)


class TestReconcileEntryToFill:
    """Primary fix — re-anchor a cached (pre-fill) entry to the real Webull fill."""

    def test_reanchors_phantom_to_fill(self):
        bridge = V5MonitorBridge(FakeSettings())
        # Cycle 1: state created from the pre-fill quote ($1.25), no fill yet.
        t = _trade(premium=1.25)
        bridge.evaluate(t, 1.25, 712.0, _now_et(10, 0))
        assert bridge._states[1].entry_premium == 1.25
        # Cycle 2 (~5s later): the Webull fill ($2.68) has reconciled to the DB.
        t2 = _trade(premium=2.68, webull_entry_fill_price=2.68)
        bridge.evaluate(t2, 2.70, 712.0, _now_et(10, 0, 5))
        assert bridge._states[1].entry_premium == 2.68  # re-anchored to the real fill

    def test_no_change_when_already_aligned(self):
        bridge = V5MonitorBridge(FakeSettings())
        t = _trade(premium=2.68, webull_entry_fill_price=2.68)
        bridge.evaluate(t, 2.68, 712.0, _now_et(10, 0))
        bridge.evaluate(t, 2.70, 712.0, _now_et(10, 0, 5))
        assert bridge._states[1].entry_premium == 2.68

    def test_skips_dca_blended_average(self):
        bridge = V5MonitorBridge(FakeSettings())
        # DCA'd trade: premium_per_contract is the blended avg; fill is only the 1st leg.
        t = _trade(premium=1.90)
        bridge.evaluate(t, 1.90, 712.0, _now_et(10, 0))
        t2 = _trade(premium=1.90, webull_entry_fill_price=2.68, dca_total_contracts=2)
        bridge.evaluate(t2, 1.95, 712.0, _now_et(10, 0, 5))
        assert bridge._states[1].entry_premium == 1.90  # blended avg preserved

    def test_skips_when_too_late(self):
        bridge = V5MonitorBridge(FakeSettings())
        t = _trade(premium=1.25)
        bridge.evaluate(t, 1.25, 712.0, _now_et(10, 0))
        # >180s after entry — don't disrupt a running trade.
        t2 = _trade(premium=2.68, webull_entry_fill_price=2.68)
        bridge.evaluate(t2, 2.70, 712.0, _now_et(10, 5))  # 5 min later
        assert bridge._states[1].entry_premium == 1.25  # untouched

    def test_reanchors_despite_dca_total_contracts_set(self):
        """REGRESSION (adam SPY #396, 2026-07-02): dca_total_contracts is the INITIAL
        contract count — set on EVERY trade at open, NOT a DCA marker. The old guard
        (`or dca_total_contracts`) short-circuited this reconcile for every trade, so the
        FSM stayed stuck on the pre-fill phantom basis. A normal (non-DCA) trade whose
        blended == fill must STILL re-anchor to the real fill even with the field set."""
        bridge = V5MonitorBridge(FakeSettings())
        t = _trade(premium=0.52)  # pre-fill SmartEntry quote
        bridge.evaluate(t, 0.52, 712.0, _now_et(10, 0))
        assert bridge._states[1].entry_premium == 0.52
        # Real fill $1.23 reconciles; blended==fill (no real DCA); dca_total_contracts=4.
        t2 = _trade(premium=1.23, webull_entry_fill_price=1.23, dca_total_contracts=4)
        bridge.evaluate(t2, 1.30, 712.0, _now_et(10, 0, 5))
        assert bridge._states[1].entry_premium == 1.23  # corrected to the real fill
        assert bridge._states[1].entry_from_real_fill is True


class TestEvaluateBackstop:
    """Fallback net — instant implausible gain ⇒ cached entry is a phantom, re-anchor."""

    def test_reanchors_instant_phantom_gain(self):
        bridge = V5MonitorBridge(FakeSettings())
        # Created at phantom $1.25, no fill price; first observed premium $2.62 = +110%.
        t = _trade(premium=1.25)
        bridge.evaluate(t, 2.62, 712.0, _now_et(10, 0))
        assert round(bridge._states[1].entry_premium, 2) == 2.62  # re-anchored to live

    def test_ignores_legit_small_move(self):
        bridge = V5MonitorBridge(FakeSettings())
        t = _trade(premium=1.00)
        bridge.evaluate(t, 1.20, 712.0, _now_et(10, 0))  # +20% — plausible
        assert bridge._states[1].entry_premium == 1.00

    def test_one_shot_only(self):
        bridge = V5MonitorBridge(FakeSettings())
        t = _trade(premium=1.00)
        bridge.evaluate(t, 1.00, 712.0, _now_et(10, 0))  # checked, no trigger
        assert bridge._states[1].entry_anchor_checked is True
        # A later legit +100% runner must NOT be re-anchored (one-shot already spent).
        bridge.evaluate(t, 2.00, 712.0, _now_et(10, 0, 30))
        assert bridge._states[1].entry_premium == 1.00

    def test_backstop_skipped_after_60s(self):
        bridge = V5MonitorBridge(FakeSettings())
        # First eval is >60s after entry — backstop window closed, don't re-anchor.
        t = _trade(premium=1.25, opened_at="2026-04-28T14:00:00")
        bridge.evaluate(t, 2.62, 712.0, _now_et(10, 2))  # 2 min after entry
        assert bridge._states[1].entry_premium == 1.25

    def test_fallback_never_overrides_real_fill(self):
        """B3 (2026-07-02): once the basis is the AUTHORITATIVE Webull fill, the live-
        premium fallback must NEVER fire — a genuine fast +70% 0DTE move would otherwise
        re-anchor a CORRECT basis to the live premium and corrupt gain%/trail/profit-lock."""
        bridge = V5MonitorBridge(FakeSettings())
        # Fill known at open → entry authoritative from cycle 1.
        t = _trade(premium=1.00, webull_entry_fill_price=1.00)
        bridge.evaluate(t, 1.00, 712.0, _now_et(10, 0))
        assert bridge._states[1].entry_from_real_fill is True
        # A REAL +80% move within 60s must NOT re-anchor — entry stays the real fill.
        bridge.evaluate(t, 1.80, 712.0, _now_et(10, 0, 10))
        assert bridge._states[1].entry_premium == 1.00
