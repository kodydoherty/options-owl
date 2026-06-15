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
HAIRCUT = 0.03
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
    hdr = {"Authorization": f"Bearer {D.KEY}", "Accept": "application/json"}
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
                ret, reason = _sim_reason(pp, mp, up, pp[0], ets, cfg, int(dte0), otype)
                mult = flow_conviction_mult(csize, ev["prem"], ev["ask_frac"], is_idx, None)[0]
                out.append({"date": d, "ticker": tk, "side": otype, "cluster": csize,
                            "premium": ev["prem"], "ask_frac": round(ev["ask_frac"], 2),
                            "conv_mult": round(mult, 2), "ret_pct": round(ret, 1), "exit_reason": reason})
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
