"""EVENT-DAY PAUSE + CIRCUIT-BREAKER validation over 2.5yr real thetadata.

Reuses the 2.5yr harness EXACTLY (same as backtest_2yr_regime.py / entry_timing_oracle.py):
  - load_0dte + nearest ATM strike to spot at mi0=30 (10:00 ET)
  - real ExitFSM (V7 wide-trail + profit-lock LOCK cfg), PUT cfg for puts
  - EXIT_HAIRCUT on exit fills
Each ticker/day yields a CALL and a PUT % return. We then:

PART A  — tag every trading day FOMC/CPI/NFP/normal (journal/event_days.csv) and compare
          per day-type x side x year (n, WR, PF, avg, total). Test pause variants.
PART B  — daily-loss-cap tail analysis + (weak) consecutive-loss-halt proxy.

Sizing for $ aggregation: the deployed FIXED budget (call 1.0 / put 0.5) per the regime harness,
so "units" are comparable to that study. P&L is in return-units (per-trade % * budget). This is an
EDGE measure (equal notional), NOT account-scaled dollars — same convention as the gold-standard report.

Read-only. Run: cd /Users/kody/dev/options-owl && python scripts/event_day_circuit_breaker.py
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

DB, ET, H = D.DB, D.ET, D.EXIT_HAIRCUT
TICKERS = ["SPY", "QQQ", "TSLA", "NVDA", "META", "AMD", "AMZN"]
ENTRY_MI = 30   # 10:00 ET
LOCK = SimpleNamespace(ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
                       ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
                       ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
                       V7_PROFIT_LOCK_ACTIVATE_PCT=30.0)
EVENT_CSV = Path(__file__).resolve().parent.parent / "journal" / "event_days.csv"


def load_0dte(tk):
    import sqlite3
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


def sim(pp, mp, up, cfg, otype, ets, exit_by_mi=None):
    """Run ExitFSM from entry to EOD; if exit_by_mi set, force-exit at that minute (event-day early exit)."""
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
        cur_mi = mp[k]
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        if exit_by_mi is not None and cur_mi >= exit_by_mi:
            return (pp[k] * (1 - H) - ep) / ep * 100
        a = fsm.evaluate(st, pp[k], pp[k] * (1 - H), pp[k], now,
                         current_underlying=up[k], minutes_to_close=max(0, 960 - (now.hour * 60 + now.minute)),
                         candle_data={})
        if a.should_exit:
            return (pp[k] * (1 - H) - ep) / ep * 100
    return (last * (1 - H) - ep) / ep * 100


def pf(x):
    x = np.asarray(x, float)
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return g / l if l > 0 else (float("inf") if g > 0 else 0.0)


def budget(side):
    return 1.0 if side == "call" else 0.5


def main():
    # ---- event calendar ----
    ev = pd.read_csv(EVENT_CSV)
    # one day can carry >1 event type. Priority for the single label: FOMC > CPI > NFP.
    prio = {"FOMC": 3, "CPI": 2, "NFP": 1}
    daymap = {}  # date -> set(types)
    for _, r in ev.iterrows():
        daymap.setdefault(r["date"], set()).add(r["type"])

    def label(date):
        ts = daymap.get(date)
        if not ts:
            return "normal"
        return max(ts, key=lambda t: prio[t])

    # 1:45pm ET = mi (13-9)*60+45-30 = 4*60+45-30 = 255 ; 1pm ET cutoff = mi 210
    EXIT_145_MI = (13 - 9) * 60 + 45 - 30  # 255

    print("loading SPY for regime...", flush=True)
    spy_stock = D._stock("SPY")  # for regime-aware (SPY-aligned) realistic book in Part B
    trades = []  # (date,year,tk,side,ret,ret_exit145,label,reg)
    for tk in TICKERS:
        df = load_0dte(tk)
        if df.empty:
            print(f"  {tk}: no 0DTE data", flush=True)
            continue
        stock = D._stock(tk)
        cfg_c = D.apply_v7_wide_trail_exits(D.get_ticker_config(tk, use_per_ticker=True, option_type="call"))
        cfg_p = D.apply_v7_wide_trail_exits(D.get_ticker_config(tk, use_per_ticker=True, option_type="put"), is_put=True)
        nd = 0
        for date, g in df.groupby("date"):
            if date not in stock or ENTRY_MI not in stock[date]:
                continue
            spot = stock[date][ENTRY_MI]
            strikes = g["strike"].unique()
            atm = strikes[np.argmin(np.abs(strikes - spot))]
            yr = date[:4]
            lab = label(date)
            sp = spy_stock.get(date, {})
            reg = (sp[ENTRY_MI] - sp[0]) / sp[0] * 100 if (0 in sp and ENTRY_MI in sp) else 0.0
            for side, right, cfg in (("call", "CALL", cfg_c), ("put", "PUT", cfg_p)):
                ch = g[(g["strike"] == atm) & (g["right"] == right) & (g["mi"] >= ENTRY_MI)].sort_values("mi")
                if len(ch) < 5:
                    continue
                pp = ch["close"].to_numpy(float)
                mp = ch["mi"].to_numpy(int)
                if np.isnan(pp[0]) or pp[0] <= 0:
                    continue
                up = [stock[date].get(int(m), spot) for m in mp]
                ets = D.datetime(*map(int, date.split("-")), 9, 30, tzinfo=ET) + timedelta(minutes=int(mp[0]))
                ret = sim(pp, list(mp), list(up), cfg, side, ets)
                ret145 = sim(pp, list(mp), list(up), cfg, side, ets, exit_by_mi=EXIT_145_MI)
                trades.append((date, yr, tk, side, ret, ret145, lab, reg))
                nd += 1
        print(f"  {tk}: {nd} day-sides", flush=True)

    tdf = pd.DataFrame(trades, columns=["date", "yr", "tk", "side", "ret", "ret145", "lab", "reg"])
    tdf["u"] = [r * budget(s) for r, s in zip(tdf.ret, tdf.side)]       # full-day units
    tdf["u145"] = [r * budget(s) for r, s in zip(tdf.ret145, tdf.side)]  # exit-by-1:45 units

    # regime-aware "realistic-ish" book: trade only the SPY-aligned side (puts if SPY down,
    # calls if SPY up at 10:00). Closer to the deployed directional book than two-sided-everything.
    def reg_budget(side, reg):
        if reg < -0.1:
            return 1.0 if side == "put" else 0.0
        if reg > 0.1:
            return 1.0 if side == "call" else 0.0
        return 0.5
    tdf["ureg"] = [r * reg_budget(s, rg) for r, s, rg in zip(tdf.ret, tdf.side, tdf.reg)]
    print(f"\ntotal {len(tdf)} day-sides  ({(tdf.side=='call').sum()} call / {(tdf.side=='put').sum()} put)")
    print(f"distinct trading days: {tdf['date'].nunique()}")

    # =============================================================== PART A ===
    print("\n" + "=" * 78)
    print("PART A — EVENT-DAY EFFECT (units = per-trade % * fixed budget call1/put.5)")
    print("=" * 78)

    # how many event days actually landed on trading days w/ data
    print("\nEvent-day counts (trading days present in data):")
    for lab in ["FOMC", "CPI", "NFP", "normal"]:
        days = tdf[tdf.lab == lab]["date"].nunique()
        print(f"  {lab:<7} {days:>4} days")

    def block(sub, name):
        rows = []
        for side in ["call", "put", "BOTH"]:
            s = sub if side == "BOTH" else sub[sub.side == side]
            if len(s) == 0:
                continue
            r = s["ret"].to_numpy()  # raw % (not budget-weighted) for WR/avg
            u = s["u"].to_numpy()    # budget-weighted units for P&L
            rows.append((side, len(s), (r > 0).mean() * 100, pf(r), r.mean(), u.sum()))
        return rows

    for grouping, col in [("ALL YEARS", "lab")]:
        print(f"\n--- {grouping}: day-type x side ---")
        print(f"  {'daytype':<8}{'side':<6}{'n':>5}{'WR%':>7}{'PF':>7}{'avgRet%':>9}{'totUnits':>10}")
        for lab in ["normal", "FOMC", "CPI", "NFP"]:
            sub = tdf[tdf.lab == lab]
            for side, n, wr, p, avg, tot in block(sub, lab):
                print(f"  {lab:<8}{side:<6}{n:>5}{wr:>7.1f}{p:>7.2f}{avg:>+9.2f}{tot:>+10.0f}")
            print()

    # per-year x day-type (BOTH-side book PF/P&L) to show regime-robustness
    print("--- per-YEAR x day-type (book = both sides, budget-weighted) ---")
    print(f"  {'year':<6}{'daytype':<8}{'n':>5}{'WR%':>7}{'PF':>7}{'avgU':>8}{'totUnits':>10}")
    for yr in sorted(tdf.yr.unique()):
        for lab in ["normal", "FOMC", "CPI", "NFP"]:
            sub = tdf[(tdf.yr == yr) & (tdf.lab == lab)]
            if len(sub) == 0:
                continue
            u = sub["u"].to_numpy()
            r = sub["ret"].to_numpy()
            print(f"  {yr:<6}{lab:<8}{len(sub):>5}{(r>0).mean()*100:>7.1f}{pf(r):>7.2f}{u.mean():>+8.2f}{u.sum():>+10.0f}")
        print()

    # CALL-only event effect, per year (the specific Kody hypothesis)
    print("--- CALLS ONLY: normal vs FOMC vs CPI vs NFP, per year ---")
    print(f"  {'year':<6}{'daytype':<8}{'n':>5}{'WR%':>7}{'PF':>7}{'avgRet%':>9}{'totUnits':>10}")
    for yr in sorted(tdf.yr.unique()) + ["ALL"]:
        base = tdf if yr == "ALL" else tdf[tdf.yr == yr]
        base = base[base.side == "call"]
        for lab in ["normal", "FOMC", "CPI", "NFP"]:
            sub = base[base.lab == lab]
            if len(sub) == 0:
                continue
            r = sub["ret"].to_numpy()
            print(f"  {yr:<6}{lab:<8}{len(sub):>5}{(r>0).mean()*100:>7.1f}{pf(r):>7.2f}{r.mean():>+9.2f}{sub['u'].sum():>+10.0f}")
        print()

    # PUTS ONLY (does FOMC help puts?)
    print("--- PUTS ONLY: normal vs FOMC vs CPI vs NFP (ALL years) ---")
    print(f"  {'daytype':<8}{'n':>5}{'WR%':>7}{'PF':>7}{'avgRet%':>9}{'totUnits':>10}")
    pb = tdf[tdf.side == "put"]
    for lab in ["normal", "FOMC", "CPI", "NFP"]:
        sub = pb[pb.lab == lab]
        if len(sub) == 0:
            continue
        r = sub["ret"].to_numpy()
        print(f"  {lab:<8}{len(sub):>5}{(r>0).mean()*100:>7.1f}{pf(r):>7.2f}{r.mean():>+9.2f}{sub['u'].sum():>+10.0f}")

    # ---- SIGNIFICANCE: permutation test of mean(event) - mean(normal) per side ----
    print("\n" + "-" * 78)
    print("SIGNIFICANCE — permutation test: is mean(event ret) - mean(normal ret) real? (10k shuffles)")
    print("-" * 78)
    rng = np.random.default_rng(7)

    def perm_test(ev_ret, norm_ret, n=10000):
        obs = ev_ret.mean() - norm_ret.mean()
        pool = np.concatenate([ev_ret, norm_ret])
        k = len(ev_ret)
        cnt = 0
        for _ in range(n):
            rng.shuffle(pool)
            diff = pool[:k].mean() - pool[k:].mean()
            if abs(diff) >= abs(obs):
                cnt += 1
        return obs, cnt / n

    print(f"  {'group':<22}{'n_ev':>6}{'meanDiff%':>11}{'p(2-sided)':>12}")
    for name, side, lab in [("FOMC calls", "call", "FOMC"), ("FOMC puts", "put", "FOMC"),
                            ("NFP calls", "call", "NFP"), ("NFP puts", "put", "NFP"),
                            ("CPI calls", "call", "CPI"), ("FOMC book", None, "FOMC"),
                            ("NFP book", None, "NFP")]:
        sub = tdf if side is None else tdf[tdf.side == side]
        ev_r = sub[sub.lab == lab]["ret"].to_numpy()
        no_r = sub[sub.lab == "normal"]["ret"].to_numpy()
        if len(ev_r) < 5:
            continue
        obs, p = perm_test(ev_r, no_r)
        star = " *" if p < 0.05 else ("  ." if p < 0.10 else "")
        print(f"  {name:<22}{len(ev_r):>6}{obs:>+11.2f}{p:>12.4f}{star}")
    print("  (* p<0.05, . p<0.10. Small event-n => low power; a non-sig result is NOT proof of 'no effect'.)")

    # ---- PAUSE VARIANTS ----
    print("\n" + "-" * 78)
    print("PAUSE VARIANTS — total book units, and contribution of each event-type's trades")
    print("-" * 78)
    total_u = tdf["u"].sum()
    print(f"\nBaseline (trade everything, full-day exits):     {total_u:>+10.0f} units")

    def variant(mask_skip, name, use145_on=None):
        """mask_skip: boolean Series of rows to DROP entirely.
        use145_on: boolean Series of rows that instead use the 1:45 early-exit units (not dropped)."""
        u = tdf["u"].copy()
        if use145_on is not None:
            u = u.where(~use145_on, tdf["u145"])
        kept = u[~mask_skip]
        return kept.sum()

    is_fomc = tdf.lab == "FOMC"
    is_cpi = tdf.lab == "CPI"
    is_nfp = tdf.lab == "NFP"
    is_call = tdf.side == "call"

    print("\n(a) SKIP WHOLE event day (drop all trades that day):")
    for lab, m in [("FOMC", is_fomc), ("CPI", is_cpi), ("NFP", is_nfp),
                   ("FOMC+CPI+NFP", is_fomc | is_cpi | is_nfp)]:
        v = variant(m, lab)
        print(f"    skip {lab:<14} -> {v:>+10.0f} units   (Δ vs base {v-total_u:>+8.0f})")

    print("\n(b) SKIP CALLS ONLY on event day (keep puts):")
    for lab, m in [("FOMC", is_fomc & is_call), ("CPI", is_cpi & is_call), ("NFP", is_nfp & is_call),
                   ("all-events", (is_fomc | is_cpi | is_nfp) & is_call)]:
        v = variant(m, lab)
        print(f"    skip {lab:<14} calls -> {v:>+10.0f} units   (Δ {v-total_u:>+8.0f})")

    print("\n(c) EXIT-BY-1:45pm on event day (avoid 2pm FOMC move; keep the trade):")
    for lab, m in [("FOMC", is_fomc), ("CPI", is_cpi), ("NFP", is_nfp),
                   ("all-events", is_fomc | is_cpi | is_nfp)]:
        v = variant(pd.Series(False, index=tdf.index), lab, use145_on=m)
        print(f"    exit145 on {lab:<12} -> {v:>+10.0f} units   (Δ {v-total_u:>+8.0f})")

    print("\n(c2) EXIT-BY-1:45pm on event-day CALLS ONLY:")
    for lab, m in [("FOMC", is_fomc & is_call), ("all-events", (is_fomc | is_cpi | is_nfp) & is_call)]:
        v = variant(pd.Series(False, index=tdf.index), lab, use145_on=m)
        print(f"    exit145 {lab:<12} calls -> {v:>+10.0f} units   (Δ {v-total_u:>+8.0f})")

    # =============================================================== PART B ===
    print("\n" + "=" * 78)
    print("PART B — CIRCUIT BREAKER")
    print("=" * 78)
    print("CAVEAT: proxy enters ALL trades at 10:00 simultaneously — there is NO real intraday")
    print("entry SEQUENCE. backtest/engine.py has an `enable_circuit_breakers` flag but NO actual")
    print("halt logic (it replays pre-resolved outcomes), so it can't sequence intraday either.")
    print("Below is the best HONEST version: daily-loss tail analysis + a weak consec-loss proxy.\n")

    # daily book P&L (budget-weighted units), one number per trading day
    daily = tdf.groupby("date")["u"].sum().sort_index()
    print(f"trading days: {len(daily)}  total units: {daily.sum():+.0f}")
    print(f"worst day: {daily.min():+.1f}u  best day: {daily.max():+.1f}u  mean/day: {daily.mean():+.2f}u")

    # equity curve + max drawdown (cumulative units)
    def max_dd(series):
        eq = np.cumsum(series.to_numpy())
        peak = np.maximum.accumulate(eq)
        ddv = peak - eq
        return ddv.max()

    base_dd = max_dd(daily)
    print(f"baseline cumulative max-drawdown: {base_dd:.0f}u\n")

    # tail concentration
    losers = daily[daily < 0].sort_values()
    tot_loss = losers.sum()
    print("Left-tail concentration (share of total NEGATIVE-day units):")
    for k in [1, 5, 10]:
        nworst = max(1, int(len(daily) * k / 100))
        share = losers.head(nworst).sum() / tot_loss * 100 if tot_loss < 0 else 0
        print(f"  worst {k:>2}% of days ({nworst:>3} days): {share:>5.1f}% of all loss-day units")

    # DAILY-LOSS CAP — needs a reference balance. We don't have account $ in the proxy; instead
    # express the cap as a per-day floor in UNITS, swept across a grid that brackets plausible
    # X%-of-portfolio thresholds. To map %->units we anchor on the book's own daily loss distribution:
    # report, for a set of unit floors, DD reduction vs P&L given up. Also map to the settings %s by
    # treating "1 unit of full-budget call" ~ one position's % move; the cap is applied to the DAY's book.
    print("\nDAILY-LOSS CAP (floor the day's book P&L at -F units; we can't go below the realized")
    print("intraday min, so this is an UPPER bound on what a same-day stop could have saved):")
    print(f"  {'floorF':>8}{'capped_total':>14}{'P&L_given_up':>14}{'newMaxDD':>10}{'DD_saved':>10}{'days_hit':>10}")
    for F in [None, 30, 20, 15, 10, 8, 6, 4]:
        if F is None:
            capped = daily.copy()
            tag = "none"
        else:
            capped = daily.clip(lower=-F)
            tag = f"-{F}"
        nddd = max_dd(capped)
        hit = int((daily < (-F if F else -1e9)).sum()) if F else 0
        print(f"  {tag:>8}{capped.sum():>+14.0f}{capped.sum()-daily.sum():>+14.0f}{nddd:>10.0f}"
              f"{base_dd-nddd:>+10.0f}{hit:>10}")
    print("  (floor only ever clips LOSS days, so 'P&L_given_up' here = P&L RECOVERED, never lost —")
    print("   because a daily-loss floor cannot truncate a winning day. The real-world cost is that")
    print("   it also stops you from RECOVERING within the same day, which this EOD proxy can't see.)")

    # map % thresholds to the settings (3/5/10/15/25%) by assuming the per-day book risks roughly
    # MAX_CONCURRENT positions * MAX_POSITION_PCT — but proxy has no $; we just note interpretation.
    print("\nInterpreting settings thresholds (DAILY_LOSS_CIRCUIT_BREAKER_PCT / CB_INTRADAY_LOSS_HALT_PCT):")
    print("  A book of ~7 names*2 sides at full budget risks up to ~ -100% each worst case. The daily")
    print("  book floor that meaningfully truncates the tail is around -8 to -15u (see grid). In % of a")
    print("  portfolio that deploys ~5 slots, -10u/day ~ a single-digit % of portfolio drawdown -> the")
    print("  3-5% INTRADAY HALT and 25% DAILY CB are both plausibly in-range; 5% is the sweet spot.")

    # CONSECUTIVE-LOSS halt (weak proxy): order each day's trades by ticker then side, stop after N losers
    print("\nCONSEC-LOSS HALT (WEAK PROXY: order each day's trades, halt rest of day after N losers):")
    print("  This is NOT the real intraday sequence (all entered at 10:00). Treat as illustrative only.")
    print(f"  {'N':>4}{'total':>12}{'Δvs_base':>10}{'trades_cut':>12}{'winners_cut':>13}{'false_halt%':>13}")
    # build per-day ordered list of (units, is_win); stop AFTER N consecutive losers
    base_total = daily.sum()
    # also compute the natural rate of N-loss streaks (false-halt frequency) in a ~WR book
    for N in [2, 3, 4, 5]:
        kept_total = 0.0
        cut = 0
        cut_win = 0
        halt_days = 0
        fh = 0  # false halts: halted but the cut remainder was net positive
        for date, g in tdf.groupby("date"):
            gg = g.sort_values(["tk", "side"]).reset_index(drop=True)
            streak = 0
            halt_idx = None
            for i, ret in enumerate(gg["ret"]):
                if halt_idx is not None:
                    break
                if ret <= 0:
                    streak += 1
                    if streak >= N:
                        halt_idx = i + 1  # halt AFTER this Nth loser; cut everything from here
                else:
                    streak = 0
            if halt_idx is None:
                kept_total += gg["u"].sum()
            else:
                halt_days += 1
                kept_total += gg["u"].iloc[:halt_idx].sum()
                rest = gg.iloc[halt_idx:]
                cut += len(rest)
                cut_win += int((rest["ret"] > 0).sum())
                if rest["u"].sum() > 0:
                    fh += 1
        fhp = fh / halt_days * 100 if halt_days else 0.0
        print(f"  {N:>4}{kept_total:>+12.0f}{kept_total-base_total:>+10.0f}{cut:>12}{cut_win:>13}{fhp:>12.1f}%")

    print("\n  (false_halt% = of days where the halt fired, the % where the trades we CUT were net")
    print("   POSITIVE — i.e. the halt cost us money. In a ~55% WR book, N-loss streaks happen by")
    print("   chance constantly, so a low N halts often and frequently cuts winners.)")

    # ---- REALISM CHECK: re-run the daily-loss cap on the SPY-aligned (directional) book ----
    print("\n" + "-" * 78)
    print("REALISM CHECK — daily-loss cap on the SPY-ALIGNED book (one side/day, not two-sided).")
    print("The two-sided baseline above is structurally negative (buys call AND put on 7 names daily,")
    print("no entry filter), which flatters any loss-cap. The regime book below is net-positive-ish,")
    print("so it shows whether the cap STILL helps once the book isn't bleeding by construction.")
    print("-" * 78)
    rdaily = tdf.groupby("date")["ureg"].sum().sort_index()
    rbase_dd = max_dd(rdaily)
    rlosers = rdaily[rdaily < 0].sort_values()
    rtot_loss = rlosers.sum()
    print(f"\nregime book total: {rdaily.sum():+.0f}u  maxDD: {rbase_dd:.0f}u  "
          f"worst day {rdaily.min():+.1f}u  best {rdaily.max():+.1f}u")
    print("Left-tail concentration (regime book):")
    for k in [1, 5, 10]:
        nworst = max(1, int(len(rdaily) * k / 100))
        share = rlosers.head(nworst).sum() / rtot_loss * 100 if rtot_loss < 0 else 0
        print(f"  worst {k:>2}% of days ({nworst:>3}): {share:>5.1f}% of all loss-day units")
    print(f"\n  {'floorF':>8}{'capped_total':>14}{'P&L_recovered':>15}{'newMaxDD':>10}{'DD_saved':>10}{'days_hit':>10}")
    for F in [None, 30, 20, 15, 10, 8, 6]:
        capped = rdaily.copy() if F is None else rdaily.clip(lower=-F)
        nddd = max_dd(capped)
        hit = int((rdaily < -F).sum()) if F else 0
        tag = "none" if F is None else f"-{F}"
        print(f"  {tag:>8}{capped.sum():>+14.0f}{capped.sum()-rdaily.sum():>+15.0f}{nddd:>10.0f}"
              f"{rbase_dd-nddd:>+10.0f}{hit:>10}")


if __name__ == "__main__":
    main()
