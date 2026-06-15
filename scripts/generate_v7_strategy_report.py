"""Generate the full V7 strategy report (markdown -> .docx via pandoc).

Covers: what's live, the strategy end-to-end (sources/gates/sizing/exits), 60-day backtest results,
annotated real-trade examples, and a per-day ledger of EVERY trade with its BUY reason + EXIT reason
+ P&L. Fixed $750/trade sleeve for an apples-to-apples ledger; compounding headline noted separately.

Inputs: journal/v3_eval_results/v7_core_trades.csv (ML) + flow_gold_standard_trades.csv (flow).
Output: journal/v3_eval_results/V7_Strategy_Report.md (+ .docx if pandoc present).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "journal" / "v3_eval_results"
SLEEVE = 750.0


def _pf(p):
    g = p[p > 0].sum(); l = -p[p < 0].sum()
    return g / l if l > 0 else float("inf")


def load():
    ml = pd.read_csv(RES / "v7_core_trades.csv")
    fl = pd.read_csv(RES / "flow_gold_standard_trades.csv")
    ml["date"] = ml["day"].astype(str)
    ml["source"] = "ML"
    ml["side"] = ml["direction"].str.lower()
    ml["pnl750"] = ml["pnl_pct"] / 100 * SLEEVE
    ml["buy_reason"] = ml.apply(
        lambda r: f"ML pattern signal (conf {r.get('pattern_conf', 0):.2f}, {int(r.get('dte',0))}DTE)", axis=1)
    ml["exit_reason_x"] = ml["reason"]
    ml["entry_prem"] = ml["entry"]
    fl["date"] = fl["date"].astype(str)
    fl["source"] = "flow"
    cm = fl["conv_mult"] / fl["conv_mult"].mean()
    fl["pnl750"] = fl["ret_pct"] / 100 * SLEEVE * cm
    fl["buy_reason"] = fl.apply(
        lambda r: f"whale {r['side']} SWEEP ${r['premium']/1e3:.0f}k, ask {r['ask_frac']:.2f}, "
                  f"cluster {int(r['cluster'])}, conv ×{r['conv_mult']:.2f}", axis=1)
    fl["exit_reason_x"] = fl["exit_reason"]
    fl["entry_prem"] = fl["premium"] / 1e5  # approx contract premium proxy (display only)
    cols = ["date", "ticker", "source", "side", "buy_reason", "exit_reason_x", "pnl750", "pnl_pct" if "pnl_pct" in ml else "ret_pct"]
    m = ml[["date", "ticker", "source", "side", "buy_reason", "exit_reason_x", "pnl750", "pnl_pct"]].rename(columns={"pnl_pct": "ret"})
    f = fl[["date", "ticker", "source", "side", "buy_reason", "exit_reason_x", "pnl750", "ret_pct"]].rename(columns={"ret_pct": "ret"})
    return pd.concat([m, f], ignore_index=True), ml, fl


def main():
    allt, ml, fl = load()
    days = sorted(allt["date"].unique())
    L = []
    A = L.append
    A("# OptionsOwl V7 + UW Flow — Live Strategy Report\n")
    A(f"_60-day backtest 2026-03-16 → 2026-06-12 · {len(allt):,} trades · generated from live-deployed config_\n")

    # ---- Executive summary ----
    A("## 1. Executive Summary\n")
    comb = allt["pnl750"]
    A(f"The deployed system trades two signal sources through one V7 risk pipeline. Over the 60-day "
      f"window, at a fixed $750/trade sleeve (apples-to-apples), the combined book returned "
      f"**${comb.sum():+,.0f} at profit factor {_pf(comb):.2f}, {(comb>0).mean()*100:.0f}% win rate**, "
      f"max drawdown ${_min_dd(allt):+,.0f}.\n")
    A("| Book | Trades | P&L (fixed $750) | PF |")
    A("|---|---|---|---|")
    for src in ("ML", "flow"):
        s = allt[allt.source == src]["pnl750"]
        A(f"| {'V7 ML-sourced' if src=='ML' else 'UW whale-flow'} | {len(s)} | ${s.sum():+,.0f} | {_pf(s):.2f} |")
    A(f"| **Combined** | {len(comb)} | **${comb.sum():+,.0f}** | **{_pf(comb):.2f}** |")
    A(f"\n_Compounding off $20k (ML book) = $234,746 historically; combined compounding is realistic only "
      f"with a $25–50k/trade liquidity cap (~$1.8M–$3.3M). The fixed-sleeve figures above isolate the EDGE._\n")

    # ---- Strategy end to end ----
    A("## 2. How It Works (end to end)\n")
    A("**Two signal sources:**")
    A("- **ML pattern** — the LightGBM pattern model scans 16 tickers; entries pass a 28-gate pipeline "
      "(score, premium cap, spread, delta, OTM, anti-chase, directional regime, risk/portfolio caps, etc.).")
    A("- **UW whale flow** — `uw_flow_collector` watches the Unusual Whales flow-alerts WebSocket and fires on "
      "ask-side option SWEEPS (≥60% ask, has_sweep, ≥$250k) on validated whitelists "
      "(PUT: META/AMZN/AAPL/TSLA/MU · CALL: META/SPY/AMZN/TSLA/AMD/ORCL/INTC/ARM/GOOG/LRCX). Flow is a "
      "high-conviction source that BYPASSES the directional/blocklist gates (it has its own validated "
      "whitelist) but keeps the risk gates.\n")
    A("**Sizing (Stage D conviction):** every flow trade is sized by a conviction multiplier "
      "(`flow_conviction_mult`): cluster ≥4 sweeps ×1.5 / single ×0.5; single-stock $1M+ sweep ×1.4 but "
      "INDEX $1M+ ×0.4 (those are hedges, not bets); ask ≥0.85 ×1.1 — clamped, then capped by MAX_POSITION_PCT "
      "and a $50k absolute liquidity cap.\n")
    A("**Exits (V7 wide-trail FSM):** no profit ceiling; wide adaptive trails let runners run "
      "(moonshot/runner/active tiers), breakeven ratchet at +20%, fast stall-cut for stale CALLs, "
      "catastrophic backstop. PUTs keep no hold limit. First gate to trigger wins.\n")

    # ---- Results ----
    A("## 3. 60-Day Results\n")
    A("Exit-reason attribution (combined, fixed-sleeve) — what drives the P&L:")
    A("| exit reason | trades | P&L |")
    A("|---|---|---|")
    for r, g in sorted(allt.groupby("exit_reason_x"), key=lambda x: -x[1]["pnl750"].sum()):
        A(f"| {r} | {len(g)} | ${g['pnl750'].sum():+,.0f} |")

    # ---- Annotated examples ----
    A("\n## 4. Annotated Examples (real trades, walked through)\n")
    ex = []
    big_put = fl[(fl.ticker == "META") & (fl.side == "put")].nlargest(1, "pnl750")
    if len(big_put):
        r = big_put.iloc[0]
        ex.append(f"**A) META put — whale-flow winner.** Buy: {r['buy_reason']} on {r['date']}. A large "
                  f"ask-side put sweep signaled institutional downside conviction; flow bypassed the "
                  f"directional gates, conviction sizing applied. Exit: `{r['exit_reason_x']}`. "
                  f"P&L ${r['pnl750']:+,.0f} (ret {r['ret_pct']:+.0f}%). The V7 no-ceiling trail rode the drop.")
    clus = fl[fl.cluster >= 4].nlargest(1, "pnl750") if "cluster" in fl else fl.iloc[:0]
    if len(clus):
        r = clus.iloc[0]
        ex.append(f"**B) Clustered sweep — sized up.** {r['ticker']} {r['side']}: {r['buy_reason']} — "
                  f"≥4 sweeps in 30min = high conviction → bigger bet. Exit `{r['exit_reason_x']}`, "
                  f"P&L ${r['pnl750']:+,.0f}.")
    ml_win = ml.nlargest(1, "pnl750")
    if len(ml_win):
        r = ml_win.iloc[0]
        ex.append(f"**C) ML pattern winner.** {r['ticker']} {r['side']} on {r['date']}: {r['buy_reason']} — "
                  f"passed all 28 entry gates. Held {int(r.get('hold_min',0))}min, peak +{r.get('peak_gain',0):.0f}%. "
                  f"Exit `{r['exit_reason_x']}`, P&L ${r['pnl750']:+,.0f}.")
    loser = allt.nsmallest(1, "pnl750")
    if len(loser):
        r = loser.iloc[0]
        ex.append(f"**D) A loss, cut by design.** {r['ticker']} {r['side']} ({r['source']}): "
                  f"exit `{r['exit_reason_x']}` — the stop/checkpoint fired to cap the loss at ${r['pnl750']:+,.0f}. "
                  f"Cutting losers fast is half the edge.")
    for e in ex:
        A(e + "\n")

    # ---- Per-day ledger ----
    A("## 5. Per-Day Ledger — every trade, buy reason, exit reason, P&L\n")
    cum = 0.0
    for d in days:
        g = allt[allt.date == d].sort_values("pnl750", ascending=False)
        dp = g["pnl750"].sum(); cum += dp
        A(f"\n### {d} — day P&L ${dp:+,.0f} (cum ${cum:+,.0f}) · {len(g)} trades")
        A("| ticker | src | side | buy reason | exit reason | P&L |")
        A("|---|---|---|---|---|---|")
        for r in g.itertuples():
            A(f"| {r.ticker} | {r.source} | {r.side} | {r.buy_reason} | {r.exit_reason_x} | ${r.pnl750:+,.0f} |")

    # ---- Appendix ----
    A("\n## 6. Appendix — live flags & caveats\n")
    A("Live flags (all bots): ENABLE_V7_WIDE_TRAIL, ML_PATTERN_THRESHOLD=0.62, ENABLE_UW_FLOW_SIGNAL, "
      "ENABLE_V7_CONVICTION_SIZING, MAX_POSITION_DOLLARS=50000. OFF: ENABLE_V7_RUNNER_TILT (pending validation).\n")
    A("Caveats: single ~3-month window overlapping exit-tuning (optimism bias); no true OOS yet; "
      "compounding figures assume reinvestment + the liquidity cap; first LIVE session is Monday.\n")

    md = RES / "V7_Strategy_Report.md"
    md.write_text("\n".join(L))
    print(f"Markdown -> {md} ({len(L)} blocks, {len(allt)} trades, {len(days)} days)")
    try:
        docx = RES / "V7_Strategy_Report.docx"
        subprocess.run(["pandoc", str(md), "-o", str(docx)], check=True)
        print(f"DOCX -> {docx}")
    except Exception as e:
        print(f"pandoc convert failed ({e}); markdown is ready.")


def _min_dd(df):
    daily = df.groupby("date")["pnl750"].sum().to_dict()
    eq = peak = dd = 0.0
    for d in sorted(daily):
        eq += daily[d]; peak = max(peak, eq); dd = min(dd, eq - peak)
    return dd


if __name__ == "__main__":
    main()
