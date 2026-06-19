"""Circuit-breaker parameter sweep — find the ROBUST-optimal "stop-for-the-day" config.

Read-only. Sweeps two distinct breaker types across MULTIPLE time periods so we judge
ROBUSTNESS (consistent across periods), not a single overfit number:

  (A) CONSECUTIVE-LOSS halt   — halt new entries for the rest of the day after N losses in a row.
  (B) DAILY-LOSS cap          — halt new entries for the rest of the day once the day's
                                cumulative P&L crosses -X% of the day's start balance.

METHODOLOGY HONESTY (read scripts comments + the final report):
  * VEHICLE 1 (PRIMARY) — 2.5yr thetadata proxy. Real option OHLC (2024-2026), ATM 0DTE
    call+put entered at 10:00 ET per ticker/day, run through the REAL V7 ExitFSM. This gives
    real per-trade P&L and a real per-DAY aggregate. It is the CORRECT tool for the DAILY-LOSS
    cap (the cap only needs the per-day cumulative). For the CONSECUTIVE-LOSS halt it is WEAKER:
    all entries fire at 10:00, so "consecutive" must use an imposed within-day ordering
    (we order by ticker then side, deterministic). We flag this limitation loudly.
  * VEHICLE 2 (SEQUENCE CROSS-CHECK) — REAL paper_trades pooled across all local bots. True
    entry times = true intraday sequence. Small n (~350, Apr-May 2026 only) but it is the ONLY
    honest test of the consecutive-loss halt's false-halt rate. Used for (A) realism + false-halt.

Periods: per-YEAR (2024 / 2025 / 2026-YTD) + rolling quarters. Splits CALLS / PUTS / COMBINED.
Primary metric = RETURN / MAX-DD (risk-adjusted). Also: PF, total P&L, maxDD, WR, breaker FIRE rate.
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import uw_ticker_discovery as D  # noqa: E402
from options_owl.risk.exit_v5.config import (  # noqa: E402
    apply_v7_wide_trail_exits,
    get_ticker_config,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

DB = D.DB
ET = D.ET
H = D.EXIT_HAIRCUT
ENTRY_MI = 30  # 10:00 ET

# Universe: liquid 0DTE names that the prod book actually trades, with full 0DTE coverage.
TICKERS = ["SPY", "QQQ", "TSLA", "NVDA", "META", "AMD", "AMZN", "GOOGL", "AAPL", "IWM"]

# Same exit settings as the deployed V7 + profit-lock book (mirrors backtest_2yr_regime.py).
LOCK = SimpleNamespace(
    ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
    ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
    ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
    V7_PROFIT_LOCK_ACTIVATE_PCT=30.0,
)

# Prod budget weighting: puts carry PUT_BUDGET_MULTIPLIER=0.5 vs calls 1.0.
CALL_W = 1.0
PUT_W = 0.5

CACHE = ROOT / "journal" / "v3_eval_results" / "_cb_sweep_returns.parquet"


# ──────────────────────────────────────────────────────────────────────────
# STAGE 1: build the per-trade return table from the 2.5yr thetadata proxy.
# ──────────────────────────────────────────────────────────────────────────
def load_0dte(tk: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT strike, right, timestamp, close FROM option_ohlc "
        "WHERE ticker=? AND expiration = substr(timestamp,1,10) ORDER BY timestamp",
        con, params=(tk,))
    con.close()
    if df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return df


def sim(pp, mp, up, cfg, otype, ets) -> float:
    """Run the real V7 FSM on a single contract's intraday path → exit return %."""
    ep = pp[0]
    fsm = ExitFSM(cfg, settings=LOCK)
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep, entry_time=ets,
                    contracts=1, peak_premium=ep, entry_underlying_price=up[0], dte=0,
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last = ep
    for k in range(1, len(pp)):
        if pp[k] is None or np.isnan(pp[k]) or pp[k] <= 0:
            continue
        last = pp[k]
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        a = fsm.evaluate(st, pp[k], pp[k] * (1 - H), pp[k], now,
                         current_underlying=up[k],
                         minutes_to_close=max(0, 960 - (now.hour * 60 + now.minute)),
                         candle_data={})
        if a.should_exit:
            return (pp[k] * (1 - H) - ep) / ep * 100
    return (last * (1 - H) - ep) / ep * 100


def build_returns() -> pd.DataFrame:
    if CACHE.exists():
        print(f"loading cached returns from {CACHE.name}", flush=True)
        return pd.read_parquet(CACHE)
    print("building per-trade returns from 2.5yr thetadata proxy (this takes a few min)...", flush=True)
    rows = []
    for tk in TICKERS:
        df = load_0dte(tk)
        if df.empty:
            print(f"  {tk}: no 0DTE data", flush=True)
            continue
        stock = D._stock(tk)
        cfg_c = apply_v7_wide_trail_exits(get_ticker_config(tk, use_per_ticker=True, option_type="call"))
        cfg_p = apply_v7_wide_trail_exits(get_ticker_config(tk, use_per_ticker=True, option_type="put"), is_put=True)
        n = 0
        for date, g in df.groupby("date"):
            if date not in stock or ENTRY_MI not in stock[date]:
                continue
            spot = stock[date][ENTRY_MI]
            strikes = g["strike"].unique()
            atm = strikes[np.argmin(np.abs(strikes - spot))]
            for side, right, cfg in (("call", "CALL", cfg_c), ("put", "PUT", cfg_p)):
                ch = g[(g["strike"] == atm) & (g["right"] == right) & (g["mi"] >= ENTRY_MI)].sort_values("mi")
                if len(ch) < 5:
                    continue
                pp = ch["close"].to_numpy(float)
                mp = ch["mi"].to_numpy(int)
                if np.isnan(pp[0]) or pp[0] <= 0:
                    continue
                up = [stock[date].get(int(m), spot) for m in mp]
                ets = datetime(*map(int, date.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=int(mp[0]))
                ret = sim(pp, list(mp), list(up), cfg, side, ets)
                rows.append((date, date[:4], tk, side, ret))
                n += 1
        print(f"  {tk}: {n} trades", flush=True)
    out = pd.DataFrame(rows, columns=["date", "yr", "tk", "side", "ret"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE)
    print(f"cached {len(out)} trades -> {CACHE.name}", flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────────
# STAGE 2: portfolio replay with circuit breakers.
# ──────────────────────────────────────────────────────────────────────────
def replay(tdf: pd.DataFrame, *, risk_pct: float, max_consec: int | None,
           daily_loss_pct: float | None, start_bal: float = 20000.0,
           side_filter: str | None = None) -> dict:
    """Sequenced portfolio replay with optional circuit breakers.

    Each (date, ticker, side) trade risks `risk_pct` of current balance (flat sizing proxy:
    P&L = stake * ret%). Within a day, trades are ordered deterministically (ticker, side) —
    the consecutive-loss counter walks that order. The daily-loss cap halts new entries once
    the day's cumulative P&L <= -daily_loss_pct% of start-of-day balance.

    Returns metrics over the filtered set. side_filter in {None,'call','put'}.
    """
    if side_filter:
        tdf = tdf[tdf["side"] == side_filter]
    bal = start_bal
    peak = start_bal
    max_dd_dollars = 0.0
    wins = losses = 0
    gross_w = gross_l = 0.0
    fires = 0          # days where a breaker bound (blocked >=1 trade)
    blocked = 0        # individual trades blocked
    n_days = 0

    for date, g in tdf.groupby("date", sort=True):
        n_days += 1
        # weight by prod budget so puts are half-size
        g = g.assign(w=np.where(g["side"] == "call", CALL_W, PUT_W))
        g = g.sort_values(["tk", "side"])  # deterministic within-day order (proxy limitation)
        day_start_bal = bal
        day_pnl = 0.0
        consec = 0
        halted = False
        day_fired = False
        for _, r in g.iterrows():
            if halted:
                blocked += 1
                day_fired = True
                continue
            stake = bal * (risk_pct / 100.0) * r["w"]
            pnl = stake * (r["ret"] / 100.0)
            bal += pnl
            day_pnl += pnl
            if pnl >= 0:
                wins += 1
                gross_w += pnl
                consec = 0
            else:
                losses += 1
                gross_l += -pnl
                consec += 1
            # drawdown tracking (mark-to-trade)
            if bal > peak:
                peak = bal
            dd = peak - bal
            if dd > max_dd_dollars:
                max_dd_dollars = dd
            # breaker checks AFTER each close → halt the REST of the day
            if max_consec is not None and consec >= max_consec:
                halted = True
            if daily_loss_pct is not None and day_pnl <= -(daily_loss_pct / 100.0) * day_start_bal:
                halted = True
        if day_fired:
            fires += 1

    total = wins + losses
    pnl_total = bal - start_bal
    pf = (gross_w / gross_l) if gross_l > 0 else (float("inf") if gross_w > 0 else 0.0)
    max_dd_pct = (max_dd_dollars / peak * 100) if peak > 0 else 0.0
    ret_dd = (pnl_total / max_dd_dollars) if max_dd_dollars > 0 else float("inf")
    wr = (wins / total * 100) if total else 0.0
    fire_rate = (fires / n_days * 100) if n_days else 0.0
    return dict(pnl=pnl_total, pf=pf, max_dd=max_dd_dollars, max_dd_pct=max_dd_pct,
                ret_dd=ret_dd, wr=wr, n=total, blocked=blocked, fire_rate=fire_rate,
                final_bal=bal)


# ──────────────────────────────────────────────────────────────────────────
# STAGE 3: false-halt analysis for consecutive-loss (on the proxy AND real seq).
# ──────────────────────────────────────────────────────────────────────────
def false_halt_analysis(seq: list[float], n_losses: int) -> dict:
    """Given an ordered P&L sequence, count N-loss streaks and what they'd have blocked.

    A 'false halt' streak is one where the trades that WOULD have been skipped (the rest of
    that day, but here we measure the next-K trades after the streak as a proxy on a flat
    sequence) contain winners we'd have missed. We report: streaks triggered, and of the
    trades immediately following a trigger, how many were winners (the cost of halting).
    """
    triggers = 0
    consec = 0
    missed_after = []  # pnl of the trade right after a trigger
    for i, p in enumerate(seq):
        if p < 0:
            consec += 1
        else:
            consec = 0
        if consec == n_losses:
            triggers += 1
            # the next trade (had we not halted) — proxy for "what we skip"
            if i + 1 < len(seq):
                missed_after.append(seq[i + 1])
            consec = 0  # reset so we count distinct streaks
    next_wins = sum(1 for x in missed_after if x >= 0)
    next_win_rate = (next_wins / len(missed_after) * 100) if missed_after else 0.0
    missed_pnl = sum(missed_after)
    return dict(triggers=triggers, next_win_rate=next_win_rate,
                missed_pnl=missed_pnl, n_after=len(missed_after))


# ──────────────────────────────────────────────────────────────────────────
# REAL paper_trades — pooled sequence cross-check.
# ──────────────────────────────────────────────────────────────────────────
def load_real_trades() -> pd.DataFrame:
    frames = []
    for b in ["kody", "adam", "vinny", "yank"]:
        db = ROOT / "journal" / f"owlet-{b}" / "raw_messages.db"
        if not db.exists():
            continue
        con = sqlite3.connect(str(db))
        # exit_source may not exist on older bots → tolerate
        try:
            q = ("SELECT opened_at, ticker, option_type, pnl_dollars, exit_source "
                 "FROM paper_trades WHERE status='closed'")
            df = pd.read_sql_query(q, con)
        except Exception:
            q = ("SELECT opened_at, ticker, option_type, pnl_dollars FROM paper_trades "
                 "WHERE status='closed'")
            df = pd.read_sql_query(q, con)
            df["exit_source"] = "ai"
        con.close()
        df = df[df["exit_source"].isin(["ai", None]) | df["exit_source"].isna()]
        df["bot"] = b
        frames.append(df)
    allt = pd.concat(frames, ignore_index=True)
    allt = allt.dropna(subset=["pnl_dollars", "opened_at"])
    allt["dt"] = pd.to_datetime(allt["opened_at"], utc=True, format="ISO8601")
    allt["date"] = allt["dt"].dt.strftime("%Y-%m-%d")
    return allt.sort_values("dt")


def main():
    tdf = build_returns()
    print(f"\n=== PROXY DATASET: {len(tdf)} trades "
          f"({(tdf.side=='call').sum()} call / {(tdf.side=='put').sum()} put), "
          f"{tdf['date'].nunique()} days ===", flush=True)

    # periods: per-year + rolling quarters
    tdf = tdf.copy()
    tdf["q"] = pd.to_datetime(tdf["date"]).dt.to_period("Q").astype(str)
    years = ["2024", "2025", "2026"]
    quarters = sorted(tdf["q"].unique())

    RISK = 5.0  # % of balance per trade (flat-sizing proxy)

    # ── SWEEP A: consecutive-loss halt ──
    consec_vals = [None, 2, 3, 4, 5, 6]
    daily_vals = [None, 3.0, 5.0, 8.0, 10.0, 15.0, 25.0]

    def fmt(m):
        rd = "inf" if m["ret_dd"] == float("inf") else f"{m['ret_dd']:.2f}"
        pf = "inf" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
        return (f"P&L ${m['pnl']:>+10,.0f}  PF {pf:>5}  maxDD ${m['max_dd']:>9,.0f}"
                f"  Ret/DD {rd:>6}  WR {m['wr']:4.1f}%  fire {m['fire_rate']:4.1f}%  n={m['n']}")

    def run_table(title, configs, side):
        print(f"\n{'='*112}\n{title}  [side={side or 'COMBINED'}]\n{'='*112}")
        for label, mc, dl in configs:
            print(f"\n--- CONFIG: {label} ---")
            print(f"  {'period':<10}{'':2}" + "metrics")
            for per in years + ["ALL"]:
                sub = tdf if per == "ALL" else tdf[tdf["yr"] == per]
                m = replay(sub, risk_pct=RISK, max_consec=mc, daily_loss_pct=dl, side_filter=side)
                print(f"  {per:<12}{fmt(m)}")

    # baseline + each consecutive value
    consec_configs = [("baseline (no CB)", None, None)] + \
                     [(f"consec={v}", v, None) for v in consec_vals if v is not None]
    daily_configs = [("baseline (no CB)", None, None)] + \
                    [(f"daily-cap={int(v)}%", None, v) for v in daily_vals if v is not None]

    for side in [None, "call", "put"]:
        run_table("SWEEP A — CONSECUTIVE-LOSS HALT", consec_configs, side)
        run_table("SWEEP B — DAILY-LOSS CAP", daily_configs, side)

    # ── SWEEP C: best 2-way combos (combined only, by period) ──
    print(f"\n{'='*112}\nSWEEP C — 2-WAY COMBINATIONS (combined book)\n{'='*112}")
    combo = [("none", None, None),
             ("consec=3", 3, None), ("consec=4", 4, None),
             ("daily=8%", None, 8.0), ("daily=10%", None, 10.0), ("daily=15%", None, 15.0),
             ("consec=4 + daily=10%", 4, 10.0),
             ("consec=4 + daily=15%", 4, 15.0),
             ("consec=3 + daily=8%", 3, 8.0)]
    hdr = f"  {'config':<24}" + "".join(f"{p:>10}" for p in years + ["ALL"])
    for metric, mlabel in [("ret_dd", "RET/DD"), ("max_dd_pct", "MAXDD%"), ("pnl", "P&L"), ("pf", "PF")]:
        print(f"\n  -- metric: {mlabel} --")
        print(hdr)
        for label, mc, dl in combo:
            cells = []
            for per in years + ["ALL"]:
                sub = tdf if per == "ALL" else tdf[tdf["yr"] == per]
                m = replay(sub, risk_pct=RISK, max_consec=mc, daily_loss_pct=dl)
                v = m[metric]
                if v == float("inf"):
                    cells.append(f"{'inf':>10}")
                elif metric == "pnl":
                    cells.append(f"{v:>+10,.0f}")
                else:
                    cells.append(f"{v:>10.2f}")
            print(f"  {label:<24}" + "".join(cells))

    # ── SWEEP D: rolling-quarter robustness (RET/DD only, combined) ──
    print(f"\n{'='*112}\nSWEEP D — ROLLING-QUARTER ROBUSTNESS (RET/DD, combined)\n{'='*112}")
    qconfigs = [("none", None, None), ("consec=3", 3, None), ("consec=4", 4, None),
                ("daily=8%", None, 8.0), ("daily=10%", None, 10.0), ("daily=15%", None, 15.0)]
    print(f"  {'quarter':<10}" + "".join(f"{l:>14}" for l, _, _ in qconfigs))
    for q in quarters:
        sub = tdf[tdf["q"] == q]
        if len(sub) < 20:
            continue
        cells = []
        for _, mc, dl in qconfigs:
            m = replay(sub, risk_pct=RISK, max_consec=mc, daily_loss_pct=dl)
            rd = m["ret_dd"]
            cells.append(f"{'inf':>14}" if rd == float("inf") else f"{rd:>14.2f}")
        print(f"  {q:<10}" + "".join(cells))

    # ── FALSE-HALT analysis (proxy ordering + real sequence) ──
    print(f"\n{'='*112}\nFALSE-HALT ANALYSIS — consecutive-loss halt\n{'='*112}")
    # proxy: build the within-day-ordered flat sequence
    proxy_seq = []
    for _, g in tdf.sort_values("date").groupby("date", sort=True):
        proxy_seq.extend(g.sort_values(["tk", "side"])["ret"].tolist())
    print(f"\n  PROXY sequence (n={len(proxy_seq)}, imposed within-day order — WEAK for consec):")
    print(f"  overall proxy win-rate: {sum(1 for x in proxy_seq if x>=0)/len(proxy_seq)*100:.1f}%")
    print(f"  {'N-loss':<8}{'triggers':>10}{'next-trade WR%':>18}{'missed P&L (units)':>22}")
    for n in [2, 3, 4, 5, 6]:
        fa = false_halt_analysis(proxy_seq, n)
        print(f"  {n:<8}{fa['triggers']:>10}{fa['next_win_rate']:>18.1f}{fa['missed_pnl']:>+22.1f}")

    # real pooled sequence
    try:
        real = load_real_trades()
        rseq = real["pnl_dollars"].tolist()
        print(f"\n  REAL pooled paper_trades (n={len(rseq)}, TRUE entry-time order — the honest test):")
        print(f"  real win-rate: {sum(1 for x in rseq if x>=0)/len(rseq)*100:.1f}%, "
              f"date range {real['date'].min()}..{real['date'].max()}")
        print(f"  {'N-loss':<8}{'triggers':>10}{'next-trade WR%':>18}{'missed P&L ($)':>20}")
        for n in [2, 3, 4, 5, 6]:
            fa = false_halt_analysis(rseq, n)
            print(f"  {n:<8}{fa['triggers']:>10}{fa['next_win_rate']:>18.1f}{fa['missed_pnl']:>+20.0f}")
    except Exception as e:
        print(f"  [real-trade cross-check failed: {e}]")

    print("\nDONE. See report for verdicts.")


if __name__ == "__main__":
    main()
