"""EXPERIMENT (2026-07-29, throwaway): can the whale flow signal be monetized in a
LOWER-FILL-TAX expression than cheap-ATM/OTM-0DTE? Tests, on the SAME whale signals at
HONEST fills, alternative EXPRESSIONS of each signal:
  - baseline: prod select_flow_strike (ATM default, OTM combos), nearest DTE
  - d55/d65/d75: strike closest to target |delta| (deeper ITM = higher premium, tighter
    %-spread, less run-up) — clamped to the deepest strike the thetadata actually has
  - maxdte: ATM strike at the FURTHEST available DTE (<=4 in this DB) — weak proxy for
    the wider-DTE hypothesis (true 7/14-30 DTE is NOT in thetadata; SPY is 0DTE only)
Honest fills stay ON (FLOW_ENTRY_RUNUP). Reuses the harness's _flow_executable_entry +
_sim_reason + select_flow_strike for exact parity. Read-only. NVDA added to test the dead name.
"""
from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
import flow_gold_standard_report as H  # noqa: E402  reuse fetch/sim/selector for parity
from options_owl.risk.exit_v5.config import INDEX_TICKERS  # noqa: E402
from options_owl.risk.greeks import calc_iv_from_premium, calc_delta  # noqa: E402

R = 0.04
DELTA_TARGETS = [0.55, 0.65, 0.75]
CACHE = Path("/private/tmp/claude-501/-Users-kody-dev-options-owl/"
             "ab4377c1-219f-4050-8390-59ecad2d6e56/scratchpad/flow_sig_cache.pkl")
OUT = D.ROOT / "journal" / "v3_eval_results" / "flow_expression_experiment.csv"

# Extend universe: add NVDA both sides (the classic "dead name" to test at higher delta)
PUT_UNIV = H.PUT_UNIV | {"NVDA"}
CALL_UNIV = H.CALL_UNIV | {"NVDA"}


def _strike_delta(prem, spot, strike, T, otype):
    try:
        iv = calc_iv_from_premium(float(prem), float(spot), float(strike), T, R, otype)
        if not iv:
            return None
        return calc_delta(float(spot), float(strike), T, R, iv, otype)
    except Exception:
        return None


def _fetch_all():
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    put_raw = H.fetch(True, PUT_UNIV)
    call_raw = H.fetch(False, CALL_UNIV)
    with open(CACHE, "wb") as f:
        pickle.dump((put_raw, call_raw), f)
    return put_raw, call_raw


def run(is_put, raw):
    otype = "put" if is_put else "call"
    right = "PUT" if is_put else "CALL"
    out = []
    for tk in sorted(raw["ticker"].unique()):
        stock, opts = D._stock(tk), D._opts(tk, right)
        cfg = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
        for d, g in raw[raw["ticker"] == tk].groupby("date"):
            seen = set()
            for _, ev in g.iterrows():
                mb = (int(ev["mi"]) // 5) * 5
                if mb in seen or d not in stock or mb not in stock[d]:
                    continue
                seen.add(mb)
                spot = stock[d][mb]
                oday = opts[(opts["date"] == d) & (opts["mi"] == mb)]
                if oday.empty:
                    continue
                dte0 = int(oday["dte"].min())
                dtemax = int(oday["dte"].max())
                same = oday[oday["dte"] == dte0]
                T0 = max(1e-6, (dte0 + (390 - mb) / 390.0) / 365.0)
                cands = []  # (strike, prem0, delta) at nearest DTE
                for r in same.itertuples():
                    prem0 = float(r.close)
                    if prem0 <= 0 or np.isnan(prem0):
                        continue
                    cands.append((float(r.strike), prem0,
                                  _strike_delta(prem0, spot, float(r.strike), T0, otype)))
                if not cands:
                    continue
                pseudo = [{"strike": s, "mid": p} for s, p, _ in cands]
                use_otm = tk in (H.OTM_PUT if is_put else H.OTM_CALL)
                base_strike, _ = H.select_flow_strike(pseudo, spot, is_put, use_otm, H.OTM_TARGET)

                # expression -> (strike, dte)
                sel = {}
                if base_strike:
                    sel["baseline"] = (base_strike, dte0)
                valid = [(s, p, dl) for s, p, dl in cands if dl is not None]
                for tgt in DELTA_TARGETS:
                    if valid:
                        best = min(valid, key=lambda x: abs(abs(x[2]) - tgt))
                        sel[f"d{int(tgt * 100)}"] = (best[0], dte0)
                # maxdte: ATM strike at furthest available DTE (proxy for wider DTE)
                if dtemax > dte0:
                    farday = oday[oday["dte"] == dtemax]
                    if not farday.empty:
                        far_atm = float(farday.assign(dist=(farday["strike"] - spot).abs())
                                        .sort_values("dist").iloc[0]["strike"])
                        sel["maxdte"] = (far_atm, dtemax)

                ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=D.ET) + timedelta(minutes=mb)
                for name, (strike, dte_use) in sel.items():
                    ch = opts[(opts["date"] == d) & (opts["strike"] == strike) & (opts["dte"] == dte_use)]
                    ch = ch[ch["mi"] >= mb].sort_values("mi")
                    if len(ch) < 5:
                        continue
                    pp = ch["close"].values.astype(float)
                    mp = ch["mi"].values.astype(int)
                    up = [stock[d].get(int(m), spot) for m in mp]
                    if np.isnan(pp[0]) or pp[0] <= 0:
                        continue
                    d0 = min(H.FLOW_ENTRY_DELAY, len(pp) - 1)
                    entry_exec, d0 = H._flow_executable_entry(pp, d0)
                    ret, reason = H._sim_reason(pp[d0:], mp[d0:], up[d0:], entry_exec, ets,
                                                cfg, int(dte_use), otype)
                    # achieved entry delta of the chosen strike
                    Tu = max(1e-6, (dte_use + (390 - mb) / 390.0) / 365.0)
                    adl = _strike_delta(pp[0], spot, strike, Tu, otype)
                    out.append({"expr": name, "date": d, "ticker": tk, "side": otype,
                                "dte": dte_use, "strike": strike, "spot": round(spot, 2),
                                "entry_prem": round(float(entry_exec), 3),
                                "achieved_delta": round(adl, 3) if adl is not None else None,
                                "ret_pct": round(ret, 2), "exit_reason": reason,
                                "signal_prem": float(ev["prem"])})
    return out


def _pf(p):
    g = p[p > 0].sum(); l = -p[p < 0].sum()
    return g / l if l > 0 else float("inf")


def _agg(df, label):
    df = df.copy()
    df["flat"] = df.ret_pct / 100 * H.SLEEVE
    # per-contract $ edge = ret% * cost/contract
    df["perc"] = df.ret_pct / 100 * df.entry_prem * 100
    rows = []
    for expr in ["baseline", "d55", "d65", "d75", "maxdte"]:
        e = df[df.expr == expr]
        if e.empty:
            continue
        rows.append({"scope": label, "expr": expr, "n": len(e),
                     "pnl_flat": round(e.flat.sum()),
                     "pf": round(_pf(e.flat), 2), "wr": round((e.flat > 0).mean() * 100),
                     "mean_ret%": round(e.ret_pct.mean(), 1),
                     "perc_edge": round(e.perc.mean()),
                     "avg_delta": round(e.achieved_delta.abs().mean(), 2),
                     "avg_prem": round(e.entry_prem.mean(), 2)})
    return rows


def main():
    put_raw, call_raw = _fetch_all()
    print(f"signals: {len(put_raw)} put, {len(call_raw)} call", flush=True)
    recs = run(True, put_raw) + run(False, call_raw)
    df = pd.DataFrame(recs)
    df.to_csv(OUT, index=False)
    print(f"\n{len(df)} expression-rows -> {OUT}\n")

    summ = _agg(df, "BOOK (all)")
    for tk in ["SPY", "ARM", "TSLA", "NVDA", "META", "AMZN", "AMD", "INTC", "MU"]:
        summ += _agg(df[df.ticker == tk], tk)
    # SPY/ARM split by side too (survivors are SPY call, ARM call, AMZN put)
    for tk, sd in [("SPY", "call"), ("SPY", "put"), ("ARM", "call"), ("AMZN", "put")]:
        summ += _agg(df[(df.ticker == tk) & (df.side == sd)], f"{tk}-{sd}")
    S = pd.DataFrame(summ)
    pd.set_option("display.width", 200, "display.max_rows", 300)
    print(S.to_string(index=False))
    S.to_csv(D.ROOT / "journal" / "v3_eval_results" / "flow_expression_summary.csv", index=False)


if __name__ == "__main__":
    main()
