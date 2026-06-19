"""REGIME-CONDITIONAL EXIT test (runner-capture thread).

Question: winners give back half-to-3/4 of their peak under the fixed V7 trail (BA +46%->+12%,
GOOG +34%->+18%). A wider trail keeps more of the runner but bleeds MORE on losers. Does making the
trail REGIME-CONDITIONAL at ENTRY (no lookahead) beat the fixed V7?

Method (mirrors backtest_2yr_regime.py):
- ATM 0DTE CALL entry at 10:00 ET on the current flow CALL set, 2024/25/26.
- Exits via the REAL V7+profit-lock ExitFSM. Same EXIT_HAIRCUT as the harness.
- For each trade we also track MFE_gain = max premium peak gain % (the runner's peak). Captured-%
  = realized_ret / MFE_gain on trades whose MFE was positive (how much of the peak we keep).
- Regime flag computed ONLY from data available AT the entry timestamp (open..entry):
    * SPY trend strength = (SPY[entry]-SPY[open]) / SPY[open]
    * underlying chop = intraday realized range of the ENTRY ticker open..entry (high-low band / price)
  TRENDING (call-favorable) = SPY trend >= +0.10% AND ticker not choppy.
  CHOPPY = everything else.
- Candidate exit configs:
    FIXED       : the deployed V7 (keep 0.60 @ +30 activate, standard wide trail).
    REGIME_COND : TRENDING -> WIDER trail (V7 widen x1.3 more) + profit-lock keep 0.80 @ +40 activate
                  (let the runner run);  CHOPPY -> TIGHTER/faster lock (trail x0.7) + keep 0.50 @ +20
                  (grab profit before chop bleeds it).
- Report captured-% + PF/WR/net per YEAR and per regime bucket, FIXED vs REGIME_COND. n on everything.

Read-only.
"""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace as _replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
from options_owl.risk.exit_v5.config import AdaptiveTier  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

DB, ET, H = D.DB, D.EXIT_HAIRCUT and D.ET, D.EXIT_HAIRCUT
ET = D.ET
H = D.EXIT_HAIRCUT
ENTRY_MI = 30  # 10:00 ET

# Current flow CALL whitelist (from prompt). All ATM 0DTE calls.
CALL_TICKERS = ["TSLA", "AAPL", "AMD", "PLTR", "META", "SPY", "AMZN"]

# Regime thresholds (computed open..entry, no lookahead)
SPY_TREND_PCT = 0.10        # SPY up >= +0.10% open->entry => call-favorable trend
CHOP_RANGE_PCT = 0.80       # ticker intraday range > this % of price = choppy (whippy tape)


def _lock_settings(keep_frac: float, activate_pct: float) -> SimpleNamespace:
    return SimpleNamespace(
        ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
        ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
        ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=keep_frac,
        V7_PROFIT_LOCK_ACTIVATE_PCT=activate_pct)


# Profit-lock setting variants
LOCK_FIXED = _lock_settings(0.60, 30.0)     # deployed V7
LOCK_WIDE = _lock_settings(0.80, 40.0)      # trending: let runner run
LOCK_TIGHT = _lock_settings(0.50, 20.0)     # choppy: grab profit early


def _scale_trail(cfg, factor: float):
    """Scale ALL adaptive trail widths of a V7 cfg by factor (clamp 5..90)."""
    def _s(tiers):
        return tuple(AdaptiveTier(t.min_peak_gain, max(5.0, min(90.0, t.trail_width * factor)))
                     for t in tiers)
    return _replace(cfg,
                    adaptive_highvol_tiers=_s(cfg.adaptive_highvol_tiers),
                    adaptive_index_tiers=_s(cfg.adaptive_index_tiers),
                    adaptive_standard_tiers=_s(cfg.adaptive_standard_tiers))


def _trend_cfg(cfg):
    """TREND: let the runner run. Loosen the give-back gates that actually fire on 0DTE calls:
    soft_trail keep 0.60->0.45 (floor lower = hold longer), scalp peak 20->35 + fade 0.60->0.45
    (don't scalp early pops), widen adaptive trail x1.3."""
    cfg = _scale_trail(cfg, 1.3)
    return _replace(cfg, soft_trail_keep_pct=0.45,
                    scalp_peak_threshold_pct=35.0, scalp_fade_ratio=0.45)


def _chop_cfg(cfg):
    """CHOP: grab profit before chop bleeds it. Tighten the give-back gates:
    soft_trail keep 0.60->0.75 (floor higher = lock sooner), scalp peak 20->15 + fade 0.60->0.70
    (scalp small pops fast), tighten adaptive trail x0.7."""
    cfg = _scale_trail(cfg, 0.7)
    return _replace(cfg, soft_trail_keep_pct=0.75,
                    scalp_peak_threshold_pct=15.0, scalp_fade_ratio=0.70)


def load_0dte(tk):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT strike, right, timestamp, close FROM option_ohlc "
        "WHERE ticker=? AND right='CALL' AND expiration = substr(timestamp,1,10) ORDER BY timestamp",
        con, params=(tk,))
    con.close()
    if df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return df


def sim(pp, mp, up, cfg, lock, ets):
    """Run the FSM. Return (realized_ret_pct, mfe_gain_pct)."""
    ep = pp[0]
    fsm = ExitFSM(cfg, settings=lock)
    st = TradeState(trade_id=1, ticker="X", option_type="call", entry_premium=ep, entry_time=ets,
                    contracts=1, peak_premium=ep, entry_underlying_price=up[0], dte=0,
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last = ep
    peak = ep
    realized = None
    for k in range(1, len(pp)):
        if pp[k] is None or np.isnan(pp[k]) or pp[k] <= 0:
            continue
        last = pp[k]
        peak = max(peak, pp[k])
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        a = fsm.evaluate(st, pp[k], pp[k] * (1 - H), pp[k], now,
                         current_underlying=up[k],
                         minutes_to_close=max(0, 960 - (now.hour * 60 + now.minute)),
                         candle_data={})
        if a.should_exit:
            realized = (pp[k] * (1 - H) - ep) / ep * 100
            _LAST_REASON.append(str(getattr(a, "reason", "?")))
            break
    if realized is None:
        realized = (last * (1 - H) - ep) / ep * 100
        _LAST_REASON.append("EOD_OR_END")
    mfe = (peak - ep) / ep * 100
    return realized, mfe


_LAST_REASON = []


def pf(x):
    x = np.array(x, float)
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return g / l if l > 0 else float("inf")


def captured(rets, mfes):
    """Mean captured-% = realized/MFE over WINNING runner trades (realized>0, MFE>5%).

    This answers 'of the peak gain a winner reached, how much did we keep'. Losers are
    excluded (a loser's realized/MFE ratio is meaningless / negative). The whole-book
    cost of giving back peak shows up in PF/net, which we report separately.
    """
    caps = []
    for r, m in zip(rets, mfes):
        if m > 5.0 and r > 0:
            caps.append(r / m)
    return (np.mean(caps) * 100 if caps else float("nan")), len(caps)


def main():
    sanity = len(sys.argv) > 1 and sys.argv[1] == "--sanity"
    tickers = ["SPY", "TSLA", "META"] if sanity else CALL_TICKERS

    print("loading SPY stock for regime...", flush=True)
    spy_stock = D._stock("SPY")

    rows = []  # date, yr, tk, regime, ret_fixed, mfe_fixed, ret_cond, mfe_cond
    for tk in tickers:
        df = load_0dte(tk)
        if df.empty:
            print(f"  {tk}: no 0DTE data", flush=True)
            continue
        stock = D._stock(tk)
        cfg_base = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type="call"))
        cfg_wide = _trend_cfg(cfg_base)    # trending -> let runner run
        cfg_tight = _chop_cfg(cfg_base)    # choppy -> grab profit early
        nd = 0
        dates = sorted(df["date"].unique())
        if sanity:
            dates = [d for d in dates if d.startswith("2025-03")]
        for date in dates:
            g = df[df["date"] == date]
            if date not in stock or ENTRY_MI not in stock[date] or 0 not in stock[date]:
                continue
            sp = spy_stock.get(date, {})
            if 0 not in sp or ENTRY_MI not in sp:
                continue
            spot = stock[date][ENTRY_MI]
            spy_trend = (sp[ENTRY_MI] - sp[0]) / sp[0] * 100
            # ticker intraday range open..entry (chop proxy, no lookahead)
            pre = [stock[date][m] for m in range(0, ENTRY_MI + 1) if m in stock[date]]
            chop = (max(pre) - min(pre)) / spot * 100 if len(pre) >= 2 else 0.0
            trending = (spy_trend >= SPY_TREND_PCT) and (chop <= CHOP_RANGE_PCT)
            regime = "TREND" if trending else "CHOP"

            strikes = g["strike"].unique()
            atm = strikes[np.argmin(np.abs(strikes - spot))]
            ch = g[(g["strike"] == atm) & (g["mi"] >= ENTRY_MI)].sort_values("mi")
            if len(ch) < 5:
                continue
            pp = ch["close"].to_numpy(float)
            mp = ch["mi"].to_numpy(int)
            if np.isnan(pp[0]) or pp[0] <= 0:
                continue
            up = [stock[date].get(int(m), spot) for m in mp]
            ets = D.datetime(*map(int, date.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=int(mp[0]))

            rf, mf = sim(pp, list(mp), list(up), cfg_base, LOCK_FIXED, ets)
            if trending:
                rc, mc = sim(pp, list(mp), list(up), cfg_wide, LOCK_WIDE, ets)
                # variant: ONLY tighten chop, leave trend at fixed
                rt = rf
            else:
                rc, mc = sim(pp, list(mp), list(up), cfg_tight, LOCK_TIGHT, ets)
                rt = rc
            rows.append((date, date[:4], tk, regime, rf, mf, rc, mc, rt))
            nd += 1
        print(f"  {tk}: {nd} trades", flush=True)

    tdf = pd.DataFrame(rows, columns=["date", "yr", "tk", "regime",
                                      "ret_f", "mfe_f", "ret_c", "mfe_c", "ret_t"])
    print(f"\ntotal {len(tdf)} call trades  "
          f"(TREND {(tdf.regime=='TREND').sum()} / CHOP {(tdf.regime=='CHOP').sum()})")

    if sanity:
        print("\n--- SANITY sample (first 12 rows) ---")
        print(tdf.head(12).to_string(index=False))
        cap_f, nc = captured(tdf.ret_f, tdf.mfe_f)
        cap_c, _ = captured(tdf.ret_c, tdf.mfe_c)
        print(f"\ncaptured% (winners only) FIXED={cap_f:.0f} (n_winrunners={nc})  COND={cap_c:.0f}")
        print(f"PF FIXED={pf(tdf.ret_f):.2f}  COND={pf(tdf.ret_c):.2f}")
        div = tdf[abs(tdf.ret_f - tdf.ret_c) > 0.01]
        print(f"\ndivergent rows (config actually changed exit): {len(div)}/{len(tdf)}")
        print(div[["date", "tk", "regime", "ret_f", "mfe_f", "ret_c"]].head(10).to_string(index=False))
        from collections import Counter
        # reasons captured during the FIXED pass (every other call); just count all
        print("\nexit reasons (which gate fires):")
        for r, n in Counter(_LAST_REASON).most_common():
            print(f"  {r:<28}{n}")
        return

    # ── Per-YEAR table ────────────────────────────────────────────────────────
    print(f"\n{'='*92}")
    print("PER-YEAR  (FIXED V7  vs  REGIME-CONDITIONAL exit)   net P&L in return-% units")
    print(f"{'year':<7}{'n':>5} | {'FIXED net':>10}{'PF':>6}{'WR':>6}{'cap%':>6} | "
          f"{'COND net':>10}{'PF':>6}{'WR':>6}{'cap%':>6}")
    print("-" * 92)
    for yr in sorted(tdf["yr"].unique()) + ["ALL"]:
        sub = tdf if yr == "ALL" else tdf[tdf["yr"] == yr]
        rf, rc = sub.ret_f.to_numpy(), sub.ret_c.to_numpy()
        wf, wc = np.mean(rf > 0) * 100, np.mean(rc > 0) * 100
        cf, ncf = captured(sub.ret_f, sub.mfe_f)
        cc, ncc = captured(sub.ret_c, sub.mfe_c)
        flag = "  (low n)" if len(sub) < 30 else ""
        print(f"{yr:<7}{len(sub):>5} | {rf.sum():>+10.0f}{pf(rf):>6.2f}{wf:>6.0f}{cf:>6.0f} | "
              f"{rc.sum():>+10.0f}{pf(rc):>6.2f}{wc:>6.0f}{cc:>6.0f}{flag}")

    # CHOP-ONLY conditioning variant (leave TREND at fixed — isolate the half that helped)
    print(f"\n{'-'*92}\nVARIANT: tighten in CHOP only, leave TREND at FIXED  (ret_t)")
    print(f"{'year':<7}{'n':>5} | {'FIXED net':>10}{'PF':>6} | {'CHOP-ONLY net':>14}{'PF':>6}")
    print("-" * 60)
    for yr in sorted(tdf["yr"].unique()) + ["ALL"]:
        sub = tdf if yr == "ALL" else tdf[tdf["yr"] == yr]
        rf, rt = sub.ret_f.to_numpy(), sub.ret_t.to_numpy()
        print(f"{yr:<7}{len(sub):>5} | {rf.sum():>+10.0f}{pf(rf):>6.2f} | "
              f"{rt.sum():>+14.0f}{pf(rt):>6.2f}")

    # ── Per-REGIME-bucket table (the real question) ───────────────────────────
    print(f"\n{'='*92}")
    print("PER-REGIME BUCKET  (does conditioning capture more of the runner without bleeding losers?)")
    print(f"{'regime':<7}{'yr':<6}{'n':>5} | {'FIXED net':>10}{'PF':>6}{'cap%':>6}(nrun) | "
          f"{'COND net':>10}{'PF':>6}{'cap%':>6}(nrun)")
    print("-" * 92)
    for reg in ["TREND", "CHOP"]:
        for yr in sorted(tdf["yr"].unique()) + ["ALL"]:
            sub = tdf[(tdf.regime == reg)]
            if yr != "ALL":
                sub = sub[sub.yr == yr]
            if sub.empty:
                continue
            rf, rc = sub.ret_f.to_numpy(), sub.ret_c.to_numpy()
            cf, ncf = captured(sub.ret_f, sub.mfe_f)
            cc, ncc = captured(sub.ret_c, sub.mfe_c)
            flag = " (low n)" if len(sub) < 30 else ""
            print(f"{reg:<7}{yr:<6}{len(sub):>5} | {rf.sum():>+10.0f}{pf(rf):>6.2f}{cf:>6.0f}"
                  f"({ncf:>3}) | {rc.sum():>+10.0f}{pf(rc):>6.2f}{cc:>6.0f}({ncc:>3}){flag}")

    # mean MFE check — do losers in CHOP truly never run?
    print(f"\n{'='*92}")
    print("MFE diagnostic (mean peak gain % the trade reached, by regime/outcome on FIXED book):")
    for reg in ["TREND", "CHOP"]:
        sub = tdf[tdf.regime == reg]
        win = sub[sub.ret_f > 0]
        los = sub[sub.ret_f <= 0]
        print(f"  {reg:<6} winners n={len(win):<4} mean MFE={win.mfe_f.mean():>6.1f}%  | "
              f"losers n={len(los):<4} mean MFE={los.mfe_f.mean():>6.1f}%")


if __name__ == "__main__":
    main()
