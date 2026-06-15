"""Gold Standard Backtest — V7 CONVEX REDESIGN (offline research, no deploy).

Implements the convex-redesign spec (specs/active/2026-06-11_convex-redesign_v1.html)
as a runnable backtest, and compares it head-to-head with the current V6/relaxed_C
baseline on IN-SAMPLE and OOS windows.

FILE OWNERSHIP / CONFLICT AVOIDANCE
-----------------------------------
scripts/backtest_gold_standard.py is OWNED by a running sizing_sweep.py job.
This file does NOT edit it. It IMPORTS the harness module and REUSES its shared
functions (model loading, feature builders, gate helpers, dip-confirm sim, the
real V5 ExitFSM + TradeState, _apply_exit_overrides). The convex changes are
layered on top in this file's own entry/sizing/exit loop, so we never touch the
shared module's globals at import time and never write its output files. All V7
outputs go to journal/v3_eval_results/v7_* paths.

THE CONVEX REDESIGN (entry / sizing / exit as ONE system)
---------------------------------------------------------
1. THIN ENTRY: drop the $6 premium cap + score floor so cheap, low-DTE, near-ATM
   calls (where runners live) are admitted. KEEP risk controls: spread gate, EOD
   cutoff, position cap, daily/weekly loss CB, GFV. anti_chase OFF, momentum ON
   (matches the relaxed_C baseline; the convex change is removing the price/score
   exclusions, not the microstructure/risk controls).
2. P(runner)-TILTED SIZING (LEAK-FREE): each CALL entry's P(runner) is looked up
   from runner_oos_predictions.csv — the WALK-FORWARD OUT-OF-SAMPLE predictions
   from scripts/runner_prediction.py (model trained only on PRIOR months, never on
   the entry's own month). Tilt: top-decile (p high) -> 1.75x budget, middle ->
   flat 0.85, bottom-30% (decile 0-2, runner rate <7% OOS) -> 0.5x. Hard ceiling
   = MAX_POSITION_PCT. Also loosens the multi-day contract cap (sizing_sweep found
   cap=2 is the dominant P&L drag; we test cap=4). REMOVES the inverted 60%
   confidence tier (we never call score_to_contracts) and the PUT down-haircut is
   kept as the documented structural 0.5x.
3. ANTI-MARTINGALE ADD: one capped add to a CONFIRMED winner (premium up >=
   +30% with underlying confirming), whole stack under one trail. NOT DCA-into-
   losers (V6 DCA is disabled in V7).
4. EXIT: no scaleout, no CALL profit ceiling, WIDENING adaptive trail, KEEP the
   breakeven ratchet + a fast stall-stop. Applied via dataclasses.replace on the
   per-ticker V5Config + a copy of the V6 settings namespace.
5. KEEP: entry=ask / exit=bid fills, parity fixes, 0DTE risk controls.

LEAK-FREE STATEMENT
-------------------
P(runner) for an entry comes ONLY from runner_oos_predictions.csv column `p`,
which is the out-of-fold prediction from an expanding-monthly walk-forward — the
model that scored month M was trained exclusively on months < M. We join on
(ticker, date, entry_min) using the SIGNAL minute snapped to the nearest sampled
entry minute (5/15/30/45/60/75/90), which is itself <= the realised entry. The
walk-forward needs >=2 months of history before it can predict, so the earliest
OOS dates (2025-09-08..2025-11-02) have NO leak-free P(runner) — those entries
fall back to FLAT 0.85 sizing (reported). No future information ever enters
sizing.

Usage:
    python scripts/backtest_gold_standard_v7.py                 # IS + OOS, V7 vs baseline
    python scripts/backtest_gold_standard_v7.py --window is     # just in-sample
    python scripts/backtest_gold_standard_v7.py --ablate        # component ablation
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Reuse the shared harness (read-only import — we never run its main()).
import scripts.backtest_gold_standard as gs  # noqa: E402
from options_owl.risk.exit_v5.config import (  # noqa: E402
    AdaptiveTier,
    V5Config,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

REPORT_DIR = PROJECT_DIR / "journal" / "v3_eval_results"
OOS_PRED_CSV = REPORT_DIR / "runner_oos_predictions.csv"

WINDOWS = {
    "is": ("2026-02-01", "2026-05-20"),
    "oos": ("2025-09-08", "2025-12-07"),
    # Second OOS window, unlocked by the Jan–Aug 2025 backfill. Starts 2025-03-15
    # so the runner walk-forward (>=2mo warmup from DATE_LO=2025-01) has leak-free
    # P(runner) coverage across it.
    "oos2": ("2025-03-15", "2025-08-31"),
}

# Sampled entry minutes in runner_oos_predictions.csv (for snapping signal minute).
PRED_ENTRY_MINUTES = np.array([5, 15, 30, 45, 60, 75, 90])

# ── P(runner) tilt thresholds (from the spec + runner_oos_predictions deciles) ──
# Deciles 0-2 in the OOS predictions have runner rates 3.9 / 4.2 / 6.5% (all <7%)
# -> bottom-30% -> 0.5x. Decile 9 (and the high tail) -> up-tilt. The CSV's own
# precomputed w_tilt uses dec>=7 -> 1.52x; we follow the spec's smoother intent:
TILT_BOTTOM_P = 0.243809   # <= decile-2 max p  -> bottom 30% -> 0.5x
TILT_TOP_P = 0.598054      # >= decile-7 min p  -> top ~30%   -> up-tilt
TILT_DOWN_MULT = 0.50
TILT_FLAT_MULT = 0.85
TILT_UP_MULT = 1.75        # within spec's 1.5-2x band

# Anti-martingale add (to confirmed winners only).
ANTIMG_TRIGGER_PCT = 30.0          # add when premium up >= +30% from entry
ANTIMG_MIN_MINUTES = 3.0           # not during grace's first moments
ANTIMG_MAX_MINUTES = 60.0          # only early enough to ride the move
ANTIMG_UND_CONFIRM_PCT = 0.10      # underlying must be >= +0.10% vs entry (CALL)
ANTIMG_ADD_FRACTION = 1.0          # add up to 1x the original contract count
ANTIMG_MAX_POSITION_PCT = 0.15     # the add still respects the position cap


# ── Leak-free P(runner) lookup ──────────────────────────────────────────────


def load_runner_predictions() -> dict[tuple[str, str, int], float]:
    """Load walk-forward OOS P(runner) keyed by (ticker, date, entry_min).

    `p` is the out-of-fold prediction (model trained only on PRIOR months).
    """
    if not OOS_PRED_CSV.exists():
        print(f"  WARNING: {OOS_PRED_CSV} missing — V7 falls back to FLAT sizing")
        return {}
    df = pd.read_csv(OOS_PRED_CSV, usecols=["ticker", "date", "entry_min", "p"])
    lut: dict[tuple[str, str, int], float] = {}
    for tk, d, em, p in df.itertuples(index=False):
        lut[(str(tk), str(d), int(em))] = float(p)
    print(f"  Loaded {len(lut):,} leak-free P(runner) predictions "
          f"({df['date'].min()}..{df['date'].max()})")
    return lut


def lookup_p_runner(lut, ticker: str, date_str: str, signal_minute: int):
    """Snap signal minute to nearest sampled entry minute, return P(runner) or None.

    None => no leak-free prediction available (early OOS warmup) => flat sizing.
    """
    if not lut:
        return None
    em = int(PRED_ENTRY_MINUTES[int(np.argmin(np.abs(PRED_ENTRY_MINUTES - signal_minute)))])
    return lut.get((ticker, date_str, em))


def runner_tilt_mult(p_runner) -> float:
    """Map P(runner) -> budget multiplier per the convex spec."""
    if p_runner is None:
        return TILT_FLAT_MULT
    if p_runner <= TILT_BOTTOM_P:
        return TILT_DOWN_MULT
    if p_runner >= TILT_TOP_P:
        return TILT_UP_MULT
    return TILT_FLAT_MULT


# ── V7 exit config (no scaleout, no CALL ceiling, widening trail) ───────────


def make_v7_v6_settings():
    """Copy the harness V6 settings and turn OFF scaleout (keep breakeven ratchet).

    Returns a SimpleNamespace usable as ExitFSM(settings=...).
    """
    from copy import copy
    s = copy(gs._V6_SETTINGS)
    s.ENABLE_V6_SCALEOUT = False          # convex: NO scaleout (the #1 right-tail leak)
    s.ENABLE_V6_BREAKEVEN_RATCHET = True  # keep — truncates only the left tail
    # 2PM tighten is the "tightening that converts convex back to capped" pattern;
    # the spec says give the tail room. Disable it in V7.
    s.ENABLE_V6_2PM_TIGHTEN = False
    return s


def v7_widen_tiers(tiers: tuple[AdaptiveTier, ...]) -> tuple[AdaptiveTier, ...]:
    """Widen adaptive trail give-back, most at the moonshot tier (let the tail run).

    Higher min_peak_gain tiers get widened more (the EV math favors width because
    the tail dominates the sum). Clamp to a sane 5-90% band.
    """
    out = []
    for t in tiers:
        if t.min_peak_gain >= 300:
            w = t.trail_width * 1.5     # moonshot: loosest
        elif t.min_peak_gain >= 100:
            w = t.trail_width * 1.3     # runner
        else:
            w = t.trail_width * 1.1     # active: only slightly wider
        out.append(AdaptiveTier(t.min_peak_gain, max(5.0, min(90.0, w))))
    return tuple(out)


def build_v7_call_config(ticker: str, fast_stall: bool = True,
                         widen: bool = True, no_ceiling: bool = True) -> V5Config:
    """Per-ticker V5Config with the convex exit changes applied via replace().

    - no CALL profit ceiling (profit_target_general_pct = 0, propagate PUT lesson)
    - widening adaptive trail (moonshot loosest)
    - KEEP breakeven ratchet (handled in V6 settings)
    - fast stall-stop: cut the body fast via theta_bleed (stale-loser timer) +
      keep the soft/scalp trail that already stall-cuts faders.
    """
    tcfg = get_ticker_config(ticker, use_per_ticker=True)
    changes: dict = {}
    if no_ceiling:
        changes["profit_target_general_pct"] = 0.0      # no CALL ceiling
        changes["profit_target_index_0dte_pct"] = 0.0   # index ceiling off too (let it run)
    if widen:
        changes["adaptive_highvol_tiers"] = v7_widen_tiers(tcfg.adaptive_highvol_tiers)
        changes["adaptive_index_tiers"] = v7_widen_tiers(tcfg.adaptive_index_tiers)
        changes["adaptive_standard_tiers"] = v7_widen_tiers(tcfg.adaptive_standard_tiers)
    if fast_stall:
        # Fast stall-stop on the BODY: cut stale 0DTE losers sooner (60min + down
        # 25% vs default 120min + down 30%) so faders are cheap. The right tail is
        # protected by the breakeven ratchet (can't go red once +20%) so this only
        # bites trades that never worked.
        changes["theta_bleed_min"] = 60.0
        changes["theta_bleed_drop_pct"] = 25.0
    return dc_replace(tcfg, **changes)


# ── V7 run loop (reuses harness helpers; own entry/sizing/exit logic) ───────


def run_v7(start_date: str, end_date: str, *, mode: str,
           pred_lut: dict, models, tickers: list[str],
           multiday_cap, antimartingale: bool,
           exit_no_ceiling: bool, exit_widen: bool, exit_fast_stall: bool,
           exit_no_scaleout: bool, thin_entry: bool, label: str) -> dict:
    """V7 backtest run.

    mode controls sizing:
      "v7"       -> P(runner)-tilted leak-free sizing
      "flat"     -> flat 0.85 (baseline-style)
    All other knobs ablate individual convex components.
    """
    (pattern_model, pattern_meta, entry_model, entry_features,
     stop_model, regime_model, signal_model, put_pattern_model, put_pattern_meta,
     put_entry_model, put_entry_features, put_entry_threshold) = models

    p_features = pattern_meta["features"]
    put_features = put_pattern_meta["features"] if put_pattern_meta else p_features
    put_threshold = put_pattern_meta.get("best_threshold", 0.40) if put_pattern_meta else 0.74
    put_model = put_pattern_model if put_pattern_model else pattern_model
    use_put_model = put_pattern_model is not None
    pattern_threshold = float(os.getenv("V7_PATTERN_THRESH", "0.74"))

    # V6/relaxed_C settings (control) vs V7 exit settings.
    if exit_no_scaleout or exit_no_ceiling or exit_widen or exit_fast_stall:
        v6_settings_base = make_v7_v6_settings()
        if not exit_no_scaleout:
            v6_settings_base.ENABLE_V6_SCALEOUT = True
            v6_settings_base.ENABLE_V6_2PM_TIGHTEN = True
    else:
        v6_settings_base = gs._V6_SETTINGS

    conn = sqlite3.connect(gs.THETADATA_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc
        WHERE ticker = 'SPY' AND substr(timestamp, 1, 10) >= ? AND substr(timestamp, 1, 10) <= ?
        ORDER BY 1
    """, (start_date, end_date)).fetchall()]

    portfolio = gs.PORTFOLIO_START
    peak_portfolio = portfolio
    max_dd = 0.0
    trades: list[dict] = []
    daily_pnls: dict[str, float] = {}
    per_ticker = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "total_contracts": 0})
    exit_reasons: defaultdict[str, int] = defaultdict(int)
    gate_blocks: defaultdict[str, int] = defaultdict(int)
    weekly_pnls: defaultdict[str, float] = defaultdict(float)
    pred_hits = 0
    pred_misses = 0
    antimg_adds = 0

    for date_str in dates:
        day_spent = 0.0
        day_realized = 0.0
        day_cb = False
        sod_balance = portfolio
        consecutive_losses = 0
        last_loss_minute = -999
        day_entered_tickers: set[str] = set()
        stock_by_minute_cache: dict[str, dict] = {}

        # Weekly loss halt (prod parity)
        try:
            _dt = datetime.strptime(date_str, "%Y-%m-%d")
            _week_key = f"{_dt.isocalendar()[0]}-W{_dt.isocalendar()[1]:02d}"
        except ValueError:
            _week_key = None
        if _week_key is not None and weekly_pnls.get(_week_key, 0.0) < 0:
            if abs(weekly_pnls[_week_key]) / max(sod_balance, 1) * 100 >= gs.WEEKLY_LOSS_HALT_PCT:
                continue

        # ── Pre-load CALL data for the day (mirrors harness SQL) ──
        ticker_data: dict[str, dict] = {}
        for ticker in tickers:
            atm = conn.execute("""
                SELECT oohlc.strike FROM option_ohlc oohlc
                JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                    AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
                WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='CALL'
                    AND og.underlying_price > 0
                GROUP BY oohlc.strike ORDER BY MIN(ABS(og.underlying_price - oohlc.strike)) LIMIT 1
            """, (ticker, date_str)).fetchone()
            if not atm:
                continue
            strike = atm[0]
            rows = conn.execute("""
                SELECT oohlc.close, COALESCE(og.underlying_price, 0),
                       COALESCE(og.implied_vol, 0), COALESCE(oq.bid, 0), COALESCE(oq.ask, 0),
                       COALESCE(og.delta, 0), COALESCE(og.theta, 0),
                       oohlc.volume, oohlc.expiration,
                       COALESCE(og.vega, 0),
                       COALESCE(oq.bid_size, 0), COALESCE(oq.ask_size, 0),
                       oohlc.timestamp
                FROM option_ohlc oohlc
                LEFT JOIN option_quotes oq ON oohlc.ticker=oq.ticker AND oohlc.expiration=oq.expiration
                    AND oohlc.strike=oq.strike AND oohlc.right=oq.right AND oohlc.timestamp=oq.timestamp
                LEFT JOIN option_greeks og ON oohlc.ticker=og.ticker AND oohlc.expiration=og.expiration
                    AND oohlc.strike=og.strike AND oohlc.right=og.right AND oohlc.timestamp=og.timestamp
                WHERE oohlc.ticker=? AND date(oohlc.timestamp)=? AND oohlc.right='CALL' AND oohlc.strike=?
                ORDER BY oohlc.timestamp
            """, (ticker, date_str, strike)).fetchall()
            if len(rows) < 30:
                continue

            closes = np.array([float(r[0]) if r[0] else np.nan for r in rows])
            underlyings = np.array([float(r[1]) if r[1] else np.nan for r in rows])
            ivs = np.array([float(r[2]) if r[2] else np.nan for r in rows])
            bids_arr = np.array([float(r[3]) if r[3] else 0 for r in rows])
            asks_arr = np.array([float(r[4]) if r[4] else 0 for r in rows])
            deltas_arr = np.array([float(r[5]) if r[5] else np.nan for r in rows])
            thetas_arr = np.array([float(r[6]) if r[6] else np.nan for r in rows])
            volumes_arr = np.array([float(r[7]) if r[7] else 0 for r in rows])
            expiry_date = rows[0][8] if rows else date_str
            vegas_arr = np.array([float(r[9]) if r[9] else np.nan for r in rows])
            bid_sizes = np.array([float(r[10]) if r[10] else 0 for r in rows])
            ask_sizes = np.array([float(r[11]) if r[11] else 0 for r in rows])
            option_minutes = [gs._ts_session_minute(r[12]) for r in rows]

            if ticker not in stock_by_minute_cache:
                stock_by_minute_cache[ticker] = gs.load_stock_by_minute(conn, ticker, date_str)
            sbm = stock_by_minute_cache[ticker]
            stock_opens, stock_closes, stock_highs, stock_lows, stock_volumes = \
                gs.align_stock_arrays(sbm, option_minutes)

            opening_price = 0
            for c in closes[:5]:
                if not np.isnan(c) and c > 0:
                    opening_price = c
                    break
            if opening_price <= 0:
                continue
            try:
                exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                day_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                dte = max(0, (exp_dt - day_dt).days)
            except (ValueError, TypeError):
                dte = 0

            ticker_data[ticker] = {
                "closes": closes, "underlyings": underlyings, "ivs": ivs,
                "bids": bids_arr, "asks": asks_arr, "deltas": deltas_arr,
                "thetas": thetas_arr, "volumes": volumes_arr, "vegas": vegas_arr,
                "bid_sizes": bid_sizes, "ask_sizes": ask_sizes,
                "expiry_date": expiry_date, "opening_price": opening_price,
                "dte": dte, "stock_closes": stock_closes,
                "stock_highs": stock_highs, "stock_lows": stock_lows,
                "stock_opens": stock_opens, "stock_volumes": stock_volumes,
                "n_rows": len(rows), "strike": strike,
            }

        open_positions: list[dict] = []
        max_data_len = max((td["n_rows"] for td in ticker_data.values()), default=0)
        day_end_minute = max_data_len

        for minute in range(gs.SCAN_START_MIN, day_end_minute, gs.SCAN_INTERVAL):
            if day_cb:
                break

            # ── Phase 1: step open FSMs forward + anti-martingale adds ──
            closed_this_tick: list[dict] = []
            for pos in open_positions:
                td = pos["ticker_data"]
                idx = minute
                if idx >= td["n_rows"]:
                    continue
                prem = td["closes"][idx]
                if np.isnan(prem) or prem <= 0:
                    continue
                bid = float(td["bids"][idx]) if not np.isnan(td["bids"][idx]) else prem
                ask = float(td["asks"][idx]) if not np.isnan(td["asks"][idx]) else prem
                underlying = float(td["underlyings"][idx]) if not np.isnan(td["underlyings"][idx]) else 0
                now = pos["entry_ts"] + timedelta(minutes=(idx - pos["entry_minute"]))
                minutes_to_close = max(0, (16 * 60) - (now.hour * 60 + now.minute))

                action = pos["fsm"].evaluate(
                    pos["state"], prem, bid, ask, now,
                    current_underlying=underlying,
                    minutes_to_close=minutes_to_close, candle_data={},
                )
                if action.should_exit:
                    exit_price = bid if bid > 0 else prem
                    if 0 < action.contracts_to_close < pos["remaining"]:
                        pos["locked_pnl"] += (exit_price - pos["effective_entry"]) * action.contracts_to_close * 100
                        pos["remaining"] -= action.contracts_to_close
                        pos["state"].contracts = pos["remaining"]
                        continue
                    elapsed = idx - pos["entry_minute"]
                    peak_gain = (pos["state"].peak_premium - pos["effective_entry"]) / pos["effective_entry"] * 100
                    trade_pnl = pos["locked_pnl"] + (exit_price - pos["effective_entry"]) * pos["remaining"] * 100
                    pos["result"] = {"pnl": trade_pnl, "reason": action.reason.value,
                                     "hold_min": elapsed, "peak_gain": peak_gain, "exit_prem": exit_price}
                    closed_this_tick.append(pos)
                    continue

                # ── ANTI-MARTINGALE: one capped add to a CONFIRMED winner ──
                if antimartingale and pos["direction"] == "call" and not pos["antimg_done"]:
                    elapsed_min = idx - pos["entry_minute"]
                    if ANTIMG_MIN_MINUTES <= elapsed_min <= ANTIMG_MAX_MINUTES:
                        gain_pct = (prem - pos["entry_premium"]) / pos["entry_premium"] * 100
                        und_entry = pos["state"].entry_underlying_price
                        und_confirm = True
                        if und_entry > 0 and underlying > 0:
                            und_confirm = (underlying / und_entry - 1) * 100 >= ANTIMG_UND_CONFIRM_PCT
                        if gain_pct >= ANTIMG_TRIGGER_PCT and und_confirm:
                            add_ask = ask if ask > 0 else prem
                            add_fill = add_ask * (1 + gs.ENTRY_SLIPPAGE_PCT / 100)
                            add_cost_per = add_fill * 100
                            # size the add: up to ADD_FRACTION x original, bounded by
                            # the position cap on the WHOLE stack + GFV.
                            max_stack_spend = portfolio * ANTIMG_MAX_POSITION_PCT
                            cur_basis = pos["effective_entry"] * pos["remaining"] * 100
                            room_ct = int(max(0.0, max_stack_spend - cur_basis) / add_cost_per) if add_cost_per > 0 else 0
                            want_ct = int(pos["contracts"] * ANTIMG_ADD_FRACTION)
                            gfv_limit = sod_balance * (1 - gs.GFV_BUFFER_PCT / 100)
                            gfv_ct = int(max(0.0, gfv_limit - day_spent) / add_cost_per) if add_cost_per > 0 else 0
                            add_ct = max(0, min(want_ct, room_ct, gfv_ct))
                            if add_ct >= 1:
                                day_spent += add_ct * add_cost_per
                                old_ct = pos["remaining"]
                                new_ct = old_ct + add_ct
                                # whole stack under one trail: blend entry UP (add was
                                # higher than entry) so the trail/peak math stays coherent.
                                blended = (pos["effective_entry"] * old_ct + add_fill * add_ct) / new_ct
                                pos["effective_entry"] = blended
                                pos["effective_contracts"] += add_ct
                                pos["remaining"] = new_ct
                                pos["antimg_done"] = True
                                pos["antimg_contracts"] = add_ct
                                pos["state"].contracts = new_ct
                                # do NOT reset peak; keep one trail on the blended stack
                                pos["state"].entry_premium = blended
                                antimg_adds += 1

            # EOD close for positions past their data
            for pos in open_positions:
                if pos in closed_this_tick:
                    continue
                td = pos["ticker_data"]
                if minute >= td["n_rows"] - 1 and "result" not in pos:
                    last_valid = pos["effective_entry"]
                    for i in range(td["n_rows"] - 1, pos["entry_minute"], -1):
                        b = td["bids"][i] if i < len(td["bids"]) else np.nan
                        if not np.isnan(b) and b > 0:
                            last_valid = float(b); break
                        if not np.isnan(td["closes"][i]) and td["closes"][i] > 0:
                            last_valid = td["closes"][i]; break
                    elapsed = td["n_rows"] - pos["entry_minute"]
                    peak_gain = (pos["state"].peak_premium - pos["effective_entry"]) / pos["effective_entry"] * 100
                    trade_pnl = pos["locked_pnl"] + (last_valid - pos["effective_entry"]) * pos["remaining"] * 100
                    pos["result"] = {"pnl": trade_pnl, "reason": "eod_data_end",
                                     "hold_min": elapsed, "peak_gain": peak_gain, "exit_prem": last_valid}
                    closed_this_tick.append(pos)

            for pos in closed_this_tick:
                result = pos["result"]
                trade_pnl = result["pnl"]
                portfolio += trade_pnl
                tk = pos["ticker"]
                per_ticker[tk]["trades"] += 1
                if trade_pnl > 0:
                    per_ticker[tk]["wins"] += 1
                per_ticker[tk]["pnl"] += trade_pnl
                per_ticker[tk]["total_contracts"] += pos["effective_contracts"]
                exit_reasons[result["reason"]] += 1
                basis = pos["effective_entry"] * pos["effective_contracts"] * 100
                trades.append({
                    "day": date_str, "ticker": tk, "minute": pos["entry_minute"],
                    "direction": pos["direction"],
                    "entry": pos["entry_premium"], "effective_entry": round(pos["effective_entry"], 2),
                    "contracts": pos["contracts"], "effective_contracts": pos["effective_contracts"],
                    "antimg_contracts": pos.get("antimg_contracts", 0),
                    "pnl": round(trade_pnl, 2), "reason": result["reason"],
                    "hold_min": result["hold_min"], "peak_gain": round(result.get("peak_gain", 0), 1),
                    "pattern_conf": round(pos["pattern_conf"], 3), "dte": int(pos["ticker_data"].get("dte", 0)),
                    "p_runner": pos.get("p_runner"), "size_mult": pos.get("size_mult"),
                    "position_dollars": round(basis, 2),
                    "pnl_pct": round(trade_pnl / basis * 100, 1) if basis > 0 else 0.0,
                })
                daily_pnls[date_str] = daily_pnls.get(date_str, 0) + trade_pnl
                if _week_key:
                    weekly_pnls[_week_key] += trade_pnl
                day_realized += trade_pnl
                if day_realized < 0 and abs(day_realized) / sod_balance * 100 >= gs.DAILY_LOSS_CB_PCT:
                    day_cb = True
                if trade_pnl <= 0:
                    consecutive_losses += 1
                    last_loss_minute = minute
                else:
                    consecutive_losses = 0
                if portfolio > peak_portfolio:
                    peak_portfolio = portfolio
                dd = (peak_portfolio - portfolio) / peak_portfolio * 100
                if dd > max_dd:
                    max_dd = dd
            open_positions = [p for p in open_positions if p not in closed_this_tick]

            # ── Phase 2: scan for new CALL entries ──
            call_scan_open = gs.OPENING_BUFFER_MIN <= minute <= gs.SCAN_END_MIN
            if not call_scan_open:
                continue
            current_open_tickers = {p["ticker"] for p in open_positions}
            current_open_dirs = [p["direction"] for p in open_positions]

            for ticker in tickers:
                if day_cb or len(open_positions) >= gs.MAX_CONCURRENT:
                    break
                if ticker in current_open_tickers or ticker in day_entered_tickers:
                    continue
                if ticker not in ticker_data:
                    continue
                td = ticker_data[ticker]
                if minute >= td["n_rows"]:
                    continue

                feat = gs.compute_pattern_features(
                    td["closes"], td["volumes"], td["ivs"], td["deltas"], td["thetas"],
                    td["underlyings"], td["bids"], td["asks"], minute, td["opening_price"])
                if feat is None:
                    continue
                X = np.array([[feat.get(f, 0) for f in p_features]], dtype=np.float32)
                pattern_conf = pattern_model.predict(X)[0]
                gate_blocks["_scanned"] += 1
                if pattern_conf < pattern_threshold:
                    gate_blocks["pattern_model"] += 1
                    continue
                score = int(pattern_conf * 100)

                # ── THIN ENTRY: score floor only applied in NON-thin (baseline) mode ──
                if not thin_entry and score < gs.MIN_SCORE:
                    gate_blocks["score_floor"] += 1
                    continue
                # TimeOfDay early-cutoff is a risk control: keep in both modes.
                if minute < gs.TOD_EARLY_CUTOFF_MIN and score < gs.TOD_EARLY_MIN_SCORE:
                    gate_blocks["tod_early"] += 1
                    continue

                # Entry-timing quality gate (kept — it's a model, not a price exclusion).
                et_feat = None
                if entry_model and entry_features:
                    et_feat = gs.compute_entry_timing_features(
                        td["closes"], td["volumes"], td["bids"], td["asks"], td["bid_sizes"],
                        td["ask_sizes"], td["ivs"], td["deltas"], td["thetas"], td["vegas"],
                        td["underlyings"], td["stock_closes"], td["stock_highs"], td["stock_lows"],
                        minute, entry_features)
                    if et_feat is not None:
                        Xe = np.array([[et_feat.get(f, 0) for f in entry_features]], dtype=np.float32)
                        if entry_model.predict(Xe)[0] < float(os.getenv("V7_ENTRY_THRESH", "0.80")):
                            gate_blocks["entry_timing"] += 1
                            continue

                entry_premium = float(td["asks"][minute]) if td["asks"][minute] > 0 else float(td["closes"][minute])
                if entry_premium <= 0 or np.isnan(entry_premium):
                    continue

                # KEEP: min premium floor (penny-lottery reject) — a risk control.
                if entry_premium < gs.MIN_PREMIUM_FLOOR:
                    gate_blocks["min_premium"] += 1
                    continue
                # THIN ENTRY: drop the $6 premium cap (it excludes runner setups).
                if not thin_entry and entry_premium > gs.PREMIUM_CAP:
                    gate_blocks["premium_cap"] += 1
                    continue

                # KEEP: spread gate (microstructure risk control) in BOTH modes.
                bid_val = float(td["bids"][minute]) if td["bids"][minute] > 0 else 0
                if bid_val > 0 and (entry_premium - bid_val) / entry_premium * 100 > gs.SPREAD_GATE_PCT:
                    gate_blocks["spread_gate"] += 1
                    continue

                # KEEP: delta gate (far-OTM/deep-ITM reject — risk control, relaxed floor 0.12)
                if "deltas" in td:
                    dv = abs(float(td["deltas"][minute])) if not np.isnan(td["deltas"][minute]) else 0
                    if dv > 0 and (dv < gs.DELTA_MIN or dv > gs.DELTA_MAX):
                        gate_blocks["delta_gate"] += 1
                        continue

                direction = "call"
                # relaxed_C: anti_chase OFF, momentum ON, consecutive_loser OFF,
                # directional_regime ON, correlation_cap ON, afternoon_danger ON, hard_cutoff ON.
                if gs.ENABLE_AFTERNOON_DANGER and gs.AFTERNOON_DANGER_START <= minute <= gs.AFTERNOON_DANGER_END:
                    gate_blocks["afternoon_danger"] += 1
                    continue
                if gs.ENABLE_HARD_CUTOFF and minute >= gs.HARD_CUTOFF_MIN:
                    gate_blocks["hard_cutoff"] += 1
                    continue
                if gs.ENABLE_CORRELATION_CAP:
                    grp = gs._group_for_ticker(ticker)
                    if grp is not None:
                        same = sum(1 for p in open_positions
                                   if p["direction"] == direction and gs._group_for_ticker(p["ticker"]) == grp)
                        if same >= gs.CORRELATION_CAP_MAX_PER_GROUP:
                            gate_blocks["correlation_cap"] += 1
                            continue
                if gs.ENABLE_MOMENTUM_CONFIRM and len(td["stock_closes"]) > 15:
                    ok, _ = gs.check_momentum_confirm(td["stock_closes"], td["stock_highs"], td["stock_lows"], minute, is_call=True)
                    if not ok:
                        gate_blocks["momentum_confirm"] += 1
                        continue
                if gs.ENABLE_DIRECTIONAL_REGIME and len(td["stock_closes"]) > 20:
                    ok, _ = gs.check_directional_regime(td["stock_closes"], minute, is_call=True)
                    if not ok:
                        gate_blocks["directional_regime"] += 1
                        continue

                # DipConfirm (entry-price optimization, kept in both modes)
                dip_entry_minute = minute
                if gs.ENABLE_DIP_CONFIRM:
                    dc_prem, dc_delay, _, dc_outcome = gs.simulate_dip_confirm(
                        td["asks"], td["bids"], td["closes"], minute, td["n_rows"])
                    if dc_outcome == "timeout_skip":
                        gate_blocks["dip_confirm_skip"] += 1
                        continue
                    if dc_prem > 0 and not np.isnan(dc_prem):
                        entry_premium = dc_prem
                        dip_entry_minute = minute + dc_delay

                entry_premium *= (1 + gs.ENTRY_SLIPPAGE_PCT / 100)
                cost_per = entry_premium * 100

                gfv_limit = sod_balance * (1 - gs.GFV_BUFFER_PCT / 100)
                gfv_remaining = gfv_limit - day_spent
                if gfv_remaining < cost_per:
                    gate_blocks["gfv_limit"] += 1
                    continue

                # ── SIZING ──
                if mode == "v7":
                    p_runner = lookup_p_runner(pred_lut, ticker, date_str, minute)
                    if p_runner is None:
                        pred_misses += 1
                    else:
                        pred_hits += 1
                    size_mult = runner_tilt_mult(p_runner)
                else:  # flat baseline-style
                    p_runner = None
                    size_mult = TILT_FLAT_MULT

                contracts = v7_size(cost_per, portfolio, size_mult,
                                    dte=int(td.get("dte", 0)), multiday_cap=multiday_cap)
                if contracts <= 0:
                    gate_blocks["sizing_rejected"] += 1
                    continue
                gfv_ct = int(gfv_remaining / cost_per) if cost_per > 0 else 1
                contracts = max(1, min(contracts, gfv_ct))
                day_spent += contracts * cost_per

                # ── Build FSM (V7 exit config or baseline) ──
                if exit_no_ceiling or exit_widen or exit_fast_stall:
                    tcfg = build_v7_call_config(ticker, fast_stall=exit_fast_stall,
                                                widen=exit_widen, no_ceiling=exit_no_ceiling)
                else:
                    tcfg = get_ticker_config(ticker, use_per_ticker=True)
                tcfg, _settings = gs._apply_exit_overrides(tcfg, v6_settings_base)
                # NOTE: V7 deliberately ignores stop_model. The 2026-06-13 stop_cal test proved
                # ML-calibrating the fixed stop is a no-op under V7 (wide trails fire first) — the
                # graduated_stop backstop stays fixed per-ticker. stop_calibration dropped from retrain.
                fsm = ExitFSM(tcfg, settings=_settings)

                entry_ts = datetime(2026, 1, 1, 9, 30) + timedelta(minutes=dip_entry_minute)
                underlying_0 = 0
                for i in range(dip_entry_minute, min(dip_entry_minute + 5, len(td["underlyings"]))):
                    u = td["underlyings"][i]
                    if not np.isnan(u) and u > 0:
                        underlying_0 = float(u); break
                state = TradeState(
                    trade_id=len(trades) + 1, ticker=ticker, option_type="call",
                    entry_premium=entry_premium, entry_time=entry_ts,
                    contracts=contracts, peak_premium=entry_premium,
                    entry_underlying_price=underlying_0,
                    dte=td["dte"], expiry_date=td["expiry_date"] or "")
                open_positions.append({
                    "ticker": ticker, "direction": direction, "fsm": fsm, "state": state,
                    "entry_minute": dip_entry_minute, "entry_ts": entry_ts,
                    "entry_premium": entry_premium, "effective_entry": entry_premium,
                    "contracts": contracts, "effective_contracts": contracts,
                    "antimg_done": False, "antimg_contracts": 0,
                    "locked_pnl": 0.0, "remaining": contracts, "ticker_data": td,
                    "pattern_conf": pattern_conf, "p_runner": p_runner, "size_mult": size_mult,
                })
                day_entered_tickers.add(ticker)
                current_open_tickers.add(ticker)
                current_open_dirs.append(direction)

        # Force-close EOD survivors
        for pos in open_positions:
            td = pos["ticker_data"]
            last_valid = pos["effective_entry"]
            for i in range(td["n_rows"] - 1, pos["entry_minute"], -1):
                b = td["bids"][i] if i < len(td["bids"]) else np.nan
                if not np.isnan(b) and b > 0:
                    last_valid = float(b); break
                if not np.isnan(td["closes"][i]) and td["closes"][i] > 0:
                    last_valid = td["closes"][i]; break
            elapsed = td["n_rows"] - pos["entry_minute"]
            peak_gain = (pos["state"].peak_premium - pos["effective_entry"]) / pos["effective_entry"] * 100
            trade_pnl = pos["locked_pnl"] + (last_valid - pos["effective_entry"]) * pos["remaining"] * 100
            portfolio += trade_pnl
            tk = pos["ticker"]
            per_ticker[tk]["trades"] += 1
            if trade_pnl > 0:
                per_ticker[tk]["wins"] += 1
            per_ticker[tk]["pnl"] += trade_pnl
            per_ticker[tk]["total_contracts"] += pos["effective_contracts"]
            exit_reasons["eod_data_end"] += 1
            basis = pos["effective_entry"] * pos["effective_contracts"] * 100
            trades.append({
                "day": date_str, "ticker": tk, "minute": pos["entry_minute"],
                "direction": pos["direction"], "entry": pos["entry_premium"],
                "effective_entry": round(pos["effective_entry"], 2),
                "contracts": pos["contracts"], "effective_contracts": pos["effective_contracts"],
                "antimg_contracts": pos.get("antimg_contracts", 0),
                "pnl": round(trade_pnl, 2), "reason": "eod_data_end",
                "hold_min": elapsed, "peak_gain": round(peak_gain, 1),
                "pattern_conf": round(pos["pattern_conf"], 3), "dte": int(pos["ticker_data"].get("dte", 0)),
                "p_runner": pos.get("p_runner"), "size_mult": pos.get("size_mult"),
                "position_dollars": round(basis, 2),
                "pnl_pct": round(trade_pnl / basis * 100, 1) if basis > 0 else 0.0,
            })
            daily_pnls[date_str] = daily_pnls.get(date_str, 0) + trade_pnl
            if portfolio > peak_portfolio:
                peak_portfolio = portfolio
            dd = (peak_portfolio - portfolio) / peak_portfolio * 100
            if dd > max_dd:
                max_dd = dd

    conn.close()
    _scanned = gate_blocks.get("_scanned", 0)
    _gb = sorted(((k, v) for k, v in gate_blocks.items() if k != "_scanned"), key=lambda x: -x[1])
    print(f"  [{label}] FUNNEL: scanned={_scanned:,} -> entries={len(trades)} "
          f"({len(trades)/max(_scanned,1)*100:.2f}% pass). Top blockers: "
          + "  ".join(f"{k}={v:,}" for k, v in _gb[:8]), flush=True)
    return _summarize(label, trades, portfolio, dates, max_dd, per_ticker,
                      exit_reasons, gate_blocks, daily_pnls, pred_hits, pred_misses, antimg_adds)


def v7_size(cost_per: float, balance: float, budget_mult: float, *,
            dte: int, multiday_cap) -> int:
    """V7 sizing: flat slot budget x P(runner) tilt, capped by MAX_POSITION_PCT.

    No score_to_contracts (so the inverted 60% confidence tier is GONE). Multi-day
    cap is a parameter (None = off; convex default loosened to 4).
    """
    if cost_per <= 0 or balance <= 0:
        return 0
    total_deployable = balance * gs.MAX_RISK_PCT
    target_per_trade = total_deployable / max(1, gs.MAX_CONCURRENT)
    scaled_target = target_per_trade * budget_mult
    raw = int(scaled_target / cost_per)
    max_by_position = int(balance * gs.MAX_POSITION_PCT / cost_per)
    if max_by_position == 0:
        return 0
    contracts = max(1, min(raw, max_by_position))
    if dte > 0 and multiday_cap is not None and multiday_cap > 0:
        premium = cost_per / 100.0
        if premium > gs.MULTI_DAY_EXPENSIVE_THRESHOLD:
            contracts = min(contracts, 1)
        else:
            contracts = min(contracts, int(multiday_cap))
    return contracts


def _summarize(label, trades, portfolio, dates, max_dd, per_ticker, exit_reasons,
               gate_blocks, daily_pnls, pred_hits, pred_misses, antimg_adds) -> dict:
    total_pnl = portfolio - gs.PORTFOLIO_START
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    wr = wins / n * 100 if n else 0
    pnl_list = [t["pnl"] for t in trades]
    gp = sum(p for p in pnl_list if p > 0)
    gl = abs(sum(p for p in pnl_list if p <= 0))
    pf = gp / gl if gl > 0 else float("inf")
    daily = list(daily_pnls.values())
    sharpe = (np.mean(daily) / np.std(daily) * np.sqrt(252)) if len(daily) > 1 and np.std(daily) > 0 else 0
    sizes = [t["position_dollars"] for t in trades]
    rets = [t["pnl_pct"] for t in trades]
    runners100 = [t for t in trades if t["pnl_pct"] >= 100.0]
    runners200 = [t for t in trades if t["pnl_pct"] >= 200.0]
    # % of peak captured (avg over winners that had a real peak)
    cap = []
    for t in trades:
        pk = t.get("peak_gain", 0)
        if pk and pk > 0:
            cap.append(max(0.0, min(1.0, t["pnl_pct"] / pk)) * 100)
    return {
        "label": label, "period": f"{dates[0]} to {dates[-1]}", "trading_days": len(dates),
        "trades": n, "wins": wins, "win_rate": round(wr, 1),
        "total_pnl": round(total_pnl, 2), "profit_factor": round(pf, 2) if pf != float("inf") else None,
        "sharpe": round(sharpe, 2), "max_drawdown_pct": round(max_dd, 1),
        "trades_per_day": round(n / len(dates), 2) if dates else 0,
        "runners_100": len(runners100), "big_runners_200": len(runners200),
        "runner_pnl": round(sum(t["pnl"] for t in runners100), 2),
        "pct_peak_captured": round(float(np.mean(cap)), 1) if cap else 0.0,
        "p95_return_pct": round(float(np.percentile(rets, 95)), 1) if rets else 0,
        "p99_return_pct": round(float(np.percentile(rets, 99)), 1) if rets else 0,
        "avg_position_dollars": round(float(np.mean(sizes)), 0) if sizes else 0,
        "p95_position_dollars": round(float(np.percentile(sizes, 95)), 0) if sizes else 0,
        "max_position_dollars": round(float(np.max(sizes)), 0) if sizes else 0,
        "largest_winner_pct": round(max(rets), 1) if rets else 0,
        "largest_winner_dollars": round(max(pnl_list), 2) if pnl_list else 0,
        "antimg_adds": antimg_adds,
        "pred_hits": pred_hits, "pred_misses": pred_misses,
        "gate_blocks": dict(gate_blocks), "exit_reasons": dict(exit_reasons),
        "per_ticker": {k: dict(v) for k, v in per_ticker.items()},
        "trade_details": trades,
    }


# ── Baseline (V6/relaxed_C) via the shared harness run_backtest ─────────────


def configure_relaxed_c(extra=None):
    """Set the shared module globals to the validated relaxed_C gate baseline.

    Mirrors sizing_sweep.py BASE_GATES:
      --puts --no-regime --gate-anti-chase off --gate-momentum off
      --gate-consecutive-loser off --delta-floor 0.12 --tod-buffer-min 5
    We do this in-process (no subprocess) but DO NOT touch run_backtest's logic.

    NOTE: V7's convex redesign is a CALL-side overhaul (runners live in cheap
    near-ATM calls). To keep the comparison apples-to-apples, the baseline is run
    CALL-ONLY here too (PUTs OFF) — PUT P&L would otherwise pad the baseline with
    a leg V7 doesn't touch. The relaxed_C gate SET is otherwise replicated exactly.
    """
    gs.ENABLE_PUTS = False
    gs.PUTS_ONLY = False
    gs.ENABLE_ANTI_CHASE = False
    gs.ENABLE_MOMENTUM_CONFIRM = False  # relaxed_C: --gate-momentum off
    gs.ENABLE_CONSECUTIVE_LOSER = False
    gs.ENABLE_CORRELATION_CAP = True
    gs.ENABLE_DIRECTIONAL_REGIME = os.getenv("V7_DIRECTIONAL_REGIME", "1") == "1"
    gs.ENABLE_PUT_BEARISH_CONFIRM = True
    gs.DELTA_MIN = 0.12
    gs.DELTA_MAX = 0.70
    gs.ENABLE_DELTA_GATE = True
    gs.ENABLE_PRICE_GATES = False
    gs.OPENING_BUFFER_MIN = 5
    gs.SIZING_MODE = "current"
    gs.MULTI_DAY_CAP = 2  # prod reality
    if extra:
        for k, v in extra.items():
            setattr(gs, k, v)


def run_baseline(start, end, models, tickers, label) -> dict:
    """Run the current V6/relaxed_C baseline via the UNMODIFIED harness run_backtest.

    NOTE: the spec's relaxed_C baseline uses momentum OFF. But the convex V7 thin
    entry above keeps momentum ON (a risk-aware confirmation, not a price gate).
    To keep the comparison apples-to-apples on the gate set, V7 here matches
    relaxed_C exactly EXCEPT for the convex changes (drop premium cap + score
    floor, P(runner) sizing, anti-martingale, convex exits). The momentum gate is
    therefore set the SAME in both arms below (we re-run V7 with momentum off to
    match). See run_v7 call in main().
    """
    (pattern_model, pattern_meta, entry_model, entry_features,
     stop_model, regime_model, signal_model, put_pattern_model, put_pattern_meta,
     put_entry_model, put_entry_features, put_entry_threshold) = models
    r = gs.run_backtest(
        pattern_model, pattern_meta, entry_model, entry_features,
        0.74, 0.80, tickers, start, end, stop_model,
        regime_model, 0.0, signal_model, put_pattern_model, put_pattern_meta,
        put_entry_model, put_entry_features, put_entry_threshold)
    # Re-shape into the same metric dict shape as _summarize for the comparison.
    trades = r["trade_details"]
    sizes = [t.get("effective_entry", t["entry"]) * t.get("effective_contracts", t["contracts"]) * 100 for t in trades]
    rets = [t.get("pnl_pct", 0) for t in trades]
    cap = []
    for t in trades:
        pk = t.get("peak_gain", 0)
        if pk and pk > 0:
            cap.append(max(0.0, min(1.0, t.get("pnl_pct", 0) / pk)) * 100)
    return {
        "label": label, "period": r["period"], "trading_days": r["trading_days"],
        "trades": r["trades"], "wins": r["wins"], "win_rate": r["win_rate"],
        "total_pnl": r["total_pnl"], "profit_factor": r["profit_factor"],
        "sharpe": r["sharpe"], "max_drawdown_pct": r["max_drawdown_pct"],
        "trades_per_day": r.get("trades_per_day", 0),
        "runners_100": r.get("runners_100", 0), "big_runners_200": r.get("big_runners_200", 0),
        "runner_pnl": r.get("runner_pnl", 0),
        "pct_peak_captured": round(float(np.mean(cap)), 1) if cap else 0.0,
        "p95_return_pct": round(float(np.percentile(rets, 95)), 1) if rets else 0,
        "p99_return_pct": round(float(np.percentile(rets, 99)), 1) if rets else 0,
        "avg_position_dollars": round(float(np.mean(sizes)), 0) if sizes else 0,
        "p95_position_dollars": round(float(np.percentile(sizes, 95)), 0) if sizes else 0,
        "max_position_dollars": round(float(np.max(sizes)), 0) if sizes else 0,
        "largest_winner_pct": r.get("largest_winner_pct", 0),
        "largest_winner_dollars": r.get("largest_winner_dollars", 0),
        "antimg_adds": 0, "pred_hits": 0, "pred_misses": 0,
        "gate_blocks": r.get("gate_blocks", {}), "exit_reasons": r.get("exit_reasons", {}),
        "per_ticker": r.get("per_ticker", {}), "trade_details": trades,
    }


# ── Reporting ───────────────────────────────────────────────────────────────

CMP_COLS = [
    ("total_pnl", "P&L $", "{:+,.0f}"),
    ("profit_factor", "PF", "{}"),
    ("win_rate", "WR%", "{}"),
    ("max_drawdown_pct", "maxDD%", "{}"),
    ("trades", "N", "{}"),
    ("trades_per_day", "N/day", "{}"),
    ("runners_100", "R100", "{}"),
    ("big_runners_200", "R200", "{}"),
    ("pct_peak_captured", "%peakCap", "{}"),
    ("p95_return_pct", "P95ret%", "{}"),
    ("p99_return_pct", "P99ret%", "{}"),
    ("avg_position_dollars", "avgSize$", "{:,.0f}"),
    ("p95_position_dollars", "p95Size$", "{:,.0f}"),
    ("largest_winner_dollars", "bigWin$", "{:+,.0f}"),
]


def print_cmp(results: list[dict]):
    hdr = f"{'config':<22}" + "".join(f"{c[1]:>11}" for c in CMP_COLS)
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        row = f"{r['label']:<22}"
        for key, _, fmt in CMP_COLS:
            v = r.get(key)
            row += f"{(fmt.format(v) if v is not None else '—'):>11}"
        print(row)


def main():
    ap = argparse.ArgumentParser(description="Gold Standard V7 Convex Redesign backtest")
    ap.add_argument("--window", choices=["is", "oos", "oos2", "both"], default="both")
    ap.add_argument("--ablate", action="store_true", help="Run component ablation")
    ap.add_argument("--multiday-cap", default="4", help="V7 multi-day cap (int or off)")
    ap.add_argument("--start", default=None, help="Override window start (smoke test)")
    ap.add_argument("--end", default=None, help="Override window end (smoke test)")
    ap.add_argument("--detail", action="store_true",
                    help="Dump per-trade CSV + per-day P&L + exit-reason report for the V7 run")
    args = ap.parse_args()

    mdc = None if str(args.multiday_cap).lower() in ("off", "none", "0") else int(args.multiday_cap)

    print("=" * 78)
    print("GOLD STANDARD V7 — CONVEX REDESIGN  (offline research, no deploy)")
    print("=" * 78)

    tickers = [t for t in gs.TICKERS if t not in gs.EXCLUDED_TICKERS]
    pred_lut = load_runner_predictions()

    print("\nLoading models...")
    models = gs.load_models(use_entry_filter=True, use_regime=False)

    windows = ["is", "oos", "oos2"] if args.window == "both" else [args.window]
    all_out: dict[str, dict] = {}

    for win in windows:
        start, end = WINDOWS[win]
        if args.start and args.end:
            start, end = args.start, args.end
        print(f"\n{'#' * 78}\n# WINDOW: {win.upper()}  {start} .. {end}\n{'#' * 78}")
        configure_relaxed_c()

        t0 = time.time()
        baseline = run_baseline(start, end, models, tickers, "baseline (V6/relaxed_C)")
        print(f"  baseline done ({time.time()-t0:.0f}s)")

        # V7 full system. relaxed_C gate set is replicated INSIDE run_v7 by reading
        # the gs.* ENABLE_* flags configure_relaxed_c() just set (momentum off, etc.).
        t0 = time.time()
        v7 = run_v7(start, end, mode="v7", pred_lut=pred_lut, models=models, tickers=tickers,
                    multiday_cap=mdc, antimartingale=True,
                    exit_no_ceiling=True, exit_widen=True, exit_fast_stall=True,
                    exit_no_scaleout=True, thin_entry=True, label="V7 convex (full)")
        print(f"  V7 full done ({time.time()-t0:.0f}s)")

        if args.detail:
            import csv as _csv
            tr = v7.get("trade_details", [])
            if tr:
                with open(REPORT_DIR / "v7_core_trades.csv", "w", newline="") as f:
                    w = _csv.DictWriter(f, fieldnames=list(tr[0].keys()))
                    w.writeheader(); w.writerows(tr)
                print(f"  Detail -> v7_core_trades.csv ({len(tr)} trades)")

        results = [baseline, v7]

        if args.ablate:
            # Ablations: turn ONE convex element off at a time, vs V7 full.
            t0 = time.time()
            abl = []
            abl.append(("V7 −thin entry (cap+floor back)",
                        dict(thin_entry=False, antimartingale=True, exit_no_ceiling=True,
                             exit_widen=True, exit_fast_stall=True, exit_no_scaleout=True)))
            abl.append(("V7 −P(runner) tilt (flat)",
                        dict(thin_entry=True, antimartingale=True, exit_no_ceiling=True,
                             exit_widen=True, exit_fast_stall=True, exit_no_scaleout=True, flat=True)))
            abl.append(("V7 −anti-martingale",
                        dict(thin_entry=True, antimartingale=False, exit_no_ceiling=True,
                             exit_widen=True, exit_fast_stall=True, exit_no_scaleout=True)))
            abl.append(("V7 −convex exits (V6 exits)",
                        dict(thin_entry=True, antimartingale=True, exit_no_ceiling=False,
                             exit_widen=False, exit_fast_stall=False, exit_no_scaleout=False)))
            abl.append(("V7 −loosened mdcap (cap2)",
                        dict(thin_entry=True, antimartingale=True, exit_no_ceiling=True,
                             exit_widen=True, exit_fast_stall=True, exit_no_scaleout=True, mdcap2=True)))
            for lbl, kw in abl:
                flat = kw.pop("flat", False)
                mdcap2 = kw.pop("mdcap2", False)
                r = run_v7(start, end, mode=("flat" if flat else "v7"), pred_lut=pred_lut,
                           models=models, tickers=tickers,
                           multiday_cap=(2 if mdcap2 else mdc), label=lbl, **kw)
                results.append(r)
            print(f"  ablations done ({time.time()-t0:.0f}s)")

        print(f"\n— {win.upper()} comparison —")
        print_cmp(results)
        all_out[win] = {r["label"]: r for r in results}

    # Persist
    out_json = REPORT_DIR / "v7_convex_backtest_raw.json"
    with open(out_json, "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nRaw results -> {out_json}")

    # Compact CSV for the spec table
    out_csv = REPORT_DIR / "v7_convex_comparison.csv"
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["window", "config"] + [c[0] for c in CMP_COLS] + ["antimg_adds", "pred_hits", "pred_misses"])
        for win, cfgs in all_out.items():
            for lbl, r in cfgs.items():
                w.writerow([win, lbl] + [r.get(c[0]) for c in CMP_COLS]
                           + [r.get("antimg_adds"), r.get("pred_hits"), r.get("pred_misses")])
    print(f"Comparison CSV -> {out_csv}")


if __name__ == "__main__":
    main()
