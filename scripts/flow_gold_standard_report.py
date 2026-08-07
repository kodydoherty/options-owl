"""FULL gold-standard flow report (last ~90 days) — end-to-end with ALL deployed changes:
V7 wide-trail exits, new tickers (MU put, ORCL/INTC call), call-whitelist trim, SPY puts under
gating, and Stage D conviction sizing (uses the PRODUCTION flow_conviction_mult). Captures every
trade with date/ticker/side/cluster/premium/conviction-mult/return/exit-reason, and reports
per-day P&L, cumulative equity + maxDD, PF/WR, per-ticker, new-ticker + SPY contribution, and
flat-vs-conviction. Writes markdown + CSV. Read-only.
"""
from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
from options_owl.risk.exit_v5.config import INDEX_TICKERS  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402
from options_owl.bot_runner import select_flow_strike  # noqa: E402  PROD's exact strike selector
from options_owl.risk.vinny_strategy import flow_conviction_mult  # noqa: E402

CLUSTER_WIN = 30
SLEEVE = 750.0
HAIRCUT = 0.03            # exit slippage: sell at close × (1 - HAIRCUT)
# Entry-fill REALISM (2026-07-28) — the "paper is a fantasy" fix, flow side.
# This harness booked entries at the option CLOSE at the signal minute (pp[0]) — no spread, no run-up.
# Measured live-vs-paper: flow entries fill ~19.6% ABOVE that (fast sweeps run up between signal and fill).
# Fix: fill at the close FLOW_ENTRY_DELAY bars later (captures the run-up from the data), never cheaper
# than the signal close (no dip discount), plus a spread-cross (flow thetadata has no bid/ask). Then walk
# exits from the fill bar forward. Tunable — calibrate the total entry gap against the ~19.6% measured.
FLOW_ENTRY_RUNUP = __import__("os").getenv("FLOW_ENTRY_RUNUP", "1") != "0"   # A/B toggle
FLOW_ENTRY_DELAY = 1      # bars after the signal a live flow order realistically fills
FLOW_ENTRY_SPREAD_SLIP = 0.02   # half-spread cross on top of the run-up (measured full spread ~4%)


def _flow_executable_entry(pp, d0):
    """The price a live flow order realistically FILLS at: the close d0 bars after the signal (run-up),
    never below the signal close (no fantasy dip discount), plus a spread-cross. Returns (entry_basis, d0)."""
    base = float(pp[0])
    if FLOW_ENTRY_RUNUP and 0 <= d0 < len(pp) and pp[d0] > 0 and not np.isnan(pp[d0]):
        px = max(base, float(pp[d0]))
    else:
        px, d0 = base, 0
    return px * (1 + FLOW_ENTRY_SPREAD_SLIP), d0
PUT_UNIV = D.CUR_PUT | {"SPY"}
CALL_UNIV = D.CUR_CALL
# Mirror PROD's deployed OTM-strike layer (ENABLE_FLOW_OTM_STRIKE) so this gold-standard report
# == what prod actually trades. Validated combos trade a cheaper ~$2 OTM strike; all else ATM.
OTM_CALL = {"AMD", "INTC", "META", "SPY"}
OTM_PUT = {"TSLA"}
OTM_TARGET = 2.0
OUT_MD = D.ROOT / "journal" / "v3_eval_results" / "flow_gold_standard_report.md"
OUT_CSV = D.ROOT / "journal" / "v3_eval_results" / "flow_gold_standard_trades.csv"


def _sim_reason(pp, mp, up, ep, ets, cfg, dte, otype):
    fsm = ExitFSM(cfg, settings=D._S())
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep, entry_time=ets,
                    contracts=1, peak_premium=ep, entry_underlying_price=up[0], dte=dte,
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last = ep
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        last = prem
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        mtc = max(0, 960 - (now.hour * 60 + now.minute))
        act = fsm.evaluate(st, prem, prem * (1 - HAIRCUT), prem, now,
                           current_underlying=up[k], minutes_to_close=mtc, candle_data={})
        if act.should_exit:
            reason = getattr(getattr(act, "reason", None), "value", None) or str(getattr(act, "reason", "exit"))
            return (prem * (1 - HAIRCUT) - ep) / ep * 100, reason
    return (last * (1 - HAIRCUT) - ep) / ep * 100, "expiry/eod"


def fetch(is_put, wl):
    hdr = {"Authorization": f"Bearer {D.KEY}", "Accept": "application/json",
           "UW-CLIENT-API-ID": "100001",  # UW now requires this + a browser UA (2026-07-10)
           "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
    rows, older = [], None
    for _ in range(260):
        p = {"limit": 200, "is_put": "true" if is_put else "false", "min_premium": D.MIN_PREM}
        if older:
            p["older_than"] = older
        r = None
        for a in range(5):
            try:
                r = requests.get(D.BASE, headers=hdr, params=p, timeout=30); break
            except requests.exceptions.RequestException:
                time.sleep(2 * (a + 1))
        if r is None or r.status_code != 200:
            break
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        older = min(x["created_at"] for x in data)
        if older < D.START:
            break
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    want = "put" if is_put else "call"
    df = df[(df["type"] == want) & df["ticker"].isin(wl)].copy()
    df["prem"] = df["total_premium"].astype(float)
    df["ask_frac"] = df["total_ask_side_prem"].astype(float) / df["prem"].clip(lower=1)
    df = df[(df["ask_frac"] >= 0.6) & df["has_sweep"].astype(bool)]
    ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(D.ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return df[df["mi"].between(0, 375)].sort_values(["ticker", "date", "mi"])


def collect(is_put, wl):
    otype = "put" if is_put else "call"
    right = "PUT" if is_put else "CALL"
    raw = fetch(is_put, wl)
    # Broad-market proxy for the flow-CALL market-direction filter (2026-07-08): SPY's %
    # change from its own day-open at each trade's entry minute. A flow call bought while
    # SPY is red (spy_chg < 0) is counter-trend to the tape (the TSLA-flow-call-into-a-
    # bearish-SPY case). Loaded once; falls back to neutral (0) when SPY data is missing.
    spy_stock = D._stock("SPY")
    out = []
    for tk in sorted(raw["ticker"].unique()):
        stock, opts = D._stock(tk), D._opts(tk, right)
        cfg = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
        is_idx = tk in INDEX_TICKERS
        for d, g in raw[raw["ticker"] == tk].groupby("date"):
            mis = g["mi"].to_numpy()
            seen = set()
            for _, ev in g.iterrows():
                mb = (int(ev["mi"]) // 5) * 5
                if mb in seen or d not in stock or mb not in stock[d]:
                    continue
                seen.add(mb)
                csize = int(np.sum(np.abs(mis - ev["mi"]) <= CLUSTER_WIN))
                spot = stock[d][mb]
                oday = opts[(opts["date"] == d) & (opts["mi"] == mb)]
                if oday.empty:
                    continue
                dte0 = oday["dte"].min()
                same = oday[oday["dte"] == dte0]
                # PROD parity: use the deployed select_flow_strike (ATM default, OTM for combos).
                pseudo_chain = [{"strike": float(r.strike), "mid": float(r.close)}
                                for r in same.itertuples()]
                use_otm = tk in (OTM_PUT if is_put else OTM_CALL)
                strike, _mode = select_flow_strike(pseudo_chain, spot, is_put, use_otm, OTM_TARGET)
                if not strike:
                    continue
                ch = opts[(opts["date"] == d) & (opts["strike"] == strike) & (opts["dte"] == dte0)]
                ch = ch[ch["mi"] >= mb].sort_values("mi")
                if len(ch) < 5:
                    continue
                pp = ch["close"].values.astype(float)
                mp = ch["mi"].values.astype(int)
                up = [stock[d].get(int(m), spot) for m in mp]
                if np.isnan(pp[0]) or pp[0] <= 0:
                    continue
                ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=D.ET) + timedelta(minutes=mb)
                # Executable entry: fill at the run-up price + spread FLOW_ENTRY_DELAY bars after the
                # signal, then walk exits from the fill bar forward (not the stale signal close pp[0]).
                d0 = min(FLOW_ENTRY_DELAY, len(pp) - 1)
                entry_exec, d0 = _flow_executable_entry(pp, d0)
                ret, reason = _sim_reason(pp[d0:], mp[d0:], up[d0:], entry_exec, ets, cfg, int(dte0), otype)
                mult = flow_conviction_mult(csize, ev["prem"], ev["ask_frac"], is_idx, None)[0]
                # ── Greeks at entry (2026-07-15 vega/IV-crush test) — back IV out of the option price via
                # BS bisection, then vega. T = calendar years incl. intraday remaining (0DTE → tiny T → vega≈0,
                # which is the point: 0DTE has no vega risk, only pricier multi-day flow does). Best-effort.
                iv = vega = delta_g = 0.0
                try:
                    from options_owl.risk.greeks import calc_iv_from_premium, calc_vega, calc_delta
                    T = max(1e-6, (int(dte0) + (390 - mb) / 390.0) / 365.0)
                    _iv = calc_iv_from_premium(float(pp[0]), float(spot), float(strike), T, 0.04, otype)
                    if _iv:
                        iv = round(_iv, 4)
                        vega = round(calc_vega(float(spot), float(strike), T, 0.04, _iv), 4)
                        delta_g = round(calc_delta(float(spot), float(strike), T, 0.04, _iv, otype), 4)
                except Exception:
                    pass
                # Market-direction proxy: the UNDERLYING's % change from the day open at entry.
                # A put bought while its underlying is rallying (mkt_chg > 0) is counter-trend
                # (the today SPY-put -54% case). Stored so a filter can be swept in-memory.
                day_open = stock[d].get(min(stock[d]), spot)
                mkt_chg = round((spot - day_open) / day_open * 100, 2) if day_open else 0.0
                # SPY-broad direction at this trade's entry minute (see spy_stock note above)
                spy_chg = 0.0
                if d in spy_stock and mb in spy_stock[d]:
                    spy_open = spy_stock[d].get(min(spy_stock[d]))
                    spy_now = spy_stock[d][mb]
                    if spy_open:
                        spy_chg = round((spy_now - spy_open) / spy_open * 100, 2)
                out.append({"date": d, "ticker": tk, "side": otype, "cluster": csize,
                            "mi": int(ev["mi"]),
                            "premium": ev["prem"], "ask_frac": round(ev["ask_frac"], 2),
                            # entry_prem = the OPTION's per-share entry price (pp[0]); ×100 = cost/contract.
                            # Needed for integer-contract sizing in the portfolio-size sweep (the whale's
                            # total "premium" above is a different thing). 2026-07-15.
                            "entry_prem": round(float(entry_exec), 2), "dte": int(dte0),
                            # run-up %: how far the option ran between the whale's signal (pp[0]) and our
                            # honest fill (entry_exec). This is the slippage tax — a fill-gate skips signals
                            # that already ran away (>N%), the cheap-OTM sweeps that bled at honest fills.
                            "runup_pct": round((float(entry_exec) / float(pp[0]) - 1) * 100, 1),
                            "strike": float(strike), "spot": round(float(spot), 2),
                            "iv": iv, "vega": vega, "delta": delta_g,
                            "conv_mult": round(mult, 2), "ret_pct": round(ret, 1),
                            "exit_reason": reason, "mkt_chg": mkt_chg, "spy_chg": spy_chg,
                            # UW alert-quality tags (2026-07-10 flow-quality test): whether the
                            # sweep is one leg of a spread (NOT directional), opening vs closing,
                            # floor trade, and which UW rule fired. Carried so a filter can be swept.
                            "has_multileg": bool(ev.get("has_multileg", False)),
                            "all_opening": bool(ev.get("all_opening_trades", False)),
                            "has_floor": bool(ev.get("has_floor", False)),
                            "alert_rule": str(ev.get("alert_rule") or "")})
    return out


def _pf(p):
    g = p[p > 0].sum(); l = -p[p < 0].sum()
    return g / l if l > 0 else float("inf")


def _dd(byday):
    eq = peak = dd = 0.0
    for d in sorted(byday):
        eq += byday[d]; peak = max(peak, eq); dd = min(dd, eq - peak)
    return dd


def main():
    df = pd.DataFrame(collect(True, PUT_UNIV) + collect(False, CALL_UNIV))
    if df.empty:
        print("no trades"); return
    df.to_csv(OUT_CSV, index=False)
    NEW = {"MU", "ORCL", "INTC"}
    df["flat_pnl"] = df["ret_pct"] / 100 * SLEEVE
    df["conv_pnl"] = df["ret_pct"] / 100 * (SLEEVE * df["conv_mult"] / df["conv_mult"].mean())
    days = sorted(df["date"].unique())
    L = []
    L.append(f"# Flow Gold-Standard Report — {days[0]} → {days[-1]} ({len(days)} trading days)\n")
    L.append("Mirrors DEPLOYED PROD: V7 wide-trail exits, gate 0.62, new tickers (MU put, ORCL/INTC "
             "call), call-whitelist trim, SPY puts, nearest-DTE strikes via the prod `select_flow_strike` "
             "(OTM for AMD/INTC/META/SPY calls + TSLA puts, ATM elsewhere), Stage D conviction sizing.\n")
    L.append("> **Sizing:** the **CONVICTION** column is the prod reference (account-scaled — conviction "
             "mult vs the mean, the deployed `score_to_contracts` behavior). **FLAT $750** is the "
             "apples-to-apples per-trade EDGE measure only; prod is never fixed-$750.\n")
    L.append("## Headline")
    for lbl, col in [("FLAT $750", "flat_pnl"), ("CONVICTION (same capital)", "conv_pnl")]:
        p = df[col]
        byday = p.groupby(df["date"]).sum().to_dict()
        L.append(f"- **{lbl}**: P&L ${p.sum():+,.0f} | PF {_pf(p):.2f} | WR {(p>0).mean()*100:.0f}% | "
                 f"maxDD ${_dd(byday):+,.0f} | {len(df)} trades")
    L.append(f"\nConviction sizing lift: **${df['conv_pnl'].sum()-df['flat_pnl'].sum():+,.0f}** "
             f"(PF {_pf(df['flat_pnl']):.2f} → {_pf(df['conv_pnl']):.2f}) on equal capital.\n")

    # ── FILL-GATED FLOW (2026-08-06) ── strip the slippage-fragile subset that bled at honest fills:
    # skip signals that already ran up > FLOW_MAX_RUNUP% (we don't chase what moved), cheap options
    # < FLOW_MIN_ENTRY_PREM (cheap-OTM sweeps take the biggest run-up tax), and low-|delta| (deep-OTM
    # lottery). Keep the fill-robust core. Env-configurable so the gate can be swept.
    import os as _os
    mx = float(_os.getenv("FLOW_MAX_RUNUP_PCT", "10"))
    mp = float(_os.getenv("FLOW_MIN_ENTRY_PREM", "1.0"))
    md = float(_os.getenv("FLOW_MIN_DELTA", "0.40"))
    gated = df[(df["runup_pct"] <= mx) & (df["entry_prem"] >= mp) & (df["delta"].abs() >= md)].copy()
    dropped = len(df) - len(gated)
    L.append("## FILL-GATED FLOW (the honest-fill recovery test)")
    L.append(f"Gate: run-up ≤ {mx:.0f}%, entry premium ≥ ${mp:.2f}, |delta| ≥ {md:.2f}. "
             f"Kept **{len(gated)}/{len(df)}** trades (dropped {dropped} slippage-fragile).\n")
    L.append("| book | flat P&L | PF | WR | trades |")
    L.append("|---|---|---|---|---|")
    L.append(f"| ALL flow (honest) | ${df['flat_pnl'].sum():+,.0f} | {_pf(df['flat_pnl']):.2f} | "
             f"{(df['flat_pnl']>0).mean()*100:.0f}% | {len(df)} |")
    if len(gated):
        gf = gated["ret_pct"] / 100 * SLEEVE
        L.append(f"| **FILL-GATED** | **${gf.sum():+,.0f}** | **{_pf(gf):.2f}** | "
                 f"{(gf>0).mean()*100:.0f}% | {len(gated)} |")
        L.append(f"\n**Recovery: ${gf.sum()-df['flat_pnl'].sum():+,.0f}** vs all-flow — the fill-gate "
                 f"{'RECOVERS real edge' if gf.sum()>0 and gf.sum()>df['flat_pnl'].sum() else 'does not save it'}.\n")
        # what the gate dropped (should be net-negative = the bleeders)
        drop = df[~df.index.isin(gated.index)]
        L.append(f"- Dropped subset flat P&L: ${drop['flat_pnl'].sum():+,.0f} over {len(drop)} trades "
                 f"(the slippage-fragile bleeders the gate removes)\n")

    L.append("## Per-ticker (conviction-sized)")
    L.append("| ticker | side | n | PF | total $ | new? |")
    L.append("|---|---|---|---|---|---|")
    for (tk, sd), g in df.groupby(["ticker", "side"]):
        L.append(f"| {tk} | {sd} | {len(g)} | {_pf(g['conv_pnl']):.2f} | ${g['conv_pnl'].sum():+,.0f} | "
                 f"{'✅' if tk in NEW else ''} |")
    L.append(f"\n- **New tickers (MU/ORCL/INTC) contribution:** ${df[df.ticker.isin(NEW)]['conv_pnl'].sum():+,.0f}")
    spy = df[df.ticker == "SPY"]
    L.append(f"- **SPY puts (gated) contribution:** ${spy[spy.side=='put']['conv_pnl'].sum():+,.0f} (n={len(spy[spy.side=='put'])})\n")

    L.append("## Exit-reason breakdown")
    L.append("| reason | n | total $ |")
    L.append("|---|---|---|")
    for reason, g in sorted(df.groupby("exit_reason"), key=lambda x: -x[1]["conv_pnl"].sum()):
        L.append(f"| {reason} | {len(g)} | ${g['conv_pnl'].sum():+,.0f} |")

    L.append("\n## Per-day P&L (conviction-sized)")
    L.append("| date | trades | day P&L | cum P&L |")
    L.append("|---|---|---|---|")
    cum = 0.0
    for d in days:
        g = df[df.date == d]; cum += g["conv_pnl"].sum()
        L.append(f"| {d} | {len(g)} | ${g['conv_pnl'].sum():+,.0f} | ${cum:+,.0f} |")
    L.append(f"\nFull per-trade detail: `{OUT_CSV.name}` ({len(df)} rows).")

    OUT_MD.write_text("\n".join(L))
    print("\n".join(L[:18]))
    print(f"\nFull report -> {OUT_MD}")


if __name__ == "__main__":
    main()
