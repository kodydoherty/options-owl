"""Exit/risk parameter SWEEP on the flow book (2026-07-10).

The lever this session kept coming back to is EXIT/RISK, not entry. This sweeps the live exit stack
against the prod-faithful baseline to see whether the deployed thresholds are optimal or leaving money
on the table. Efficient design: collect the raw option price paths ONCE (the expensive 35GB-thetadata
step), then re-run the FSM in-memory across a parameter grid (cheap). Every variant is scored flat-$750
on the SAME trade set, so differences are pure exit-policy, not trade selection.

CRITICAL fidelity note: the flow report's `_S` shim leaves the V7 exit stack OFF (no profit-lock, no
-25% hardstops, no stall-cut). Prod has them ON. So the BASELINE here sets the real prod values (see
ProdS) — the sweep asks "given prod's stack, does any single knob beat it?" A variant ships only if it
adds book P&L without wrecking drawdown, and ideally is monotone/sane (not a lone overfit point).

Usage: python scripts/exit_risk_sweep.py [--family exit|stops|lock|trail|all]
Bounded by thetadata (entries ≤2026-07-01).
"""
import argparse
import pickle
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import uw_ticker_discovery as D  # noqa: E402
import flow_gold_standard_report as R  # noqa: E402
from options_owl.bot_runner import select_flow_strike  # noqa: E402

BASE = 750.0
HC = R.HAIRCUT
CACHE = Path("/private/tmp/claude-501/-Users-kody-dev-options-owl/"
             "ab4377c1-219f-4050-8390-59ecad2d6e56/scratchpad/flow_raw_paths.pkl")


def _fetch_resilient(is_put, wl):
    """R.fetch but tolerant of UW 503 'overload' (retry with backoff) instead of bailing to empty."""
    hdr = {"Authorization": f"Bearer {D.KEY}", "Accept": "application/json",
           "UW-CLIENT-API-ID": "100001",
           "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
    rows, older = [], None
    for _ in range(260):
        p = {"limit": 200, "is_put": "true" if is_put else "false", "min_premium": D.MIN_PREM}
        if older:
            p["older_than"] = older
        r = None
        for a in range(7):
            try:
                r = requests.get(D.BASE, headers=hdr, params=p, timeout=30)
                if r.status_code == 200:
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(30, 3 * (a + 1)))  # backoff on overload
                    r = None
                    continue
                break  # other non-200 → give up this page
            except requests.exceptions.RequestException:
                time.sleep(3 * (a + 1))
        if r is None or r.status_code != 200:
            break
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        older = min(x["created_at"] for x in data)
        if older < D.START:
            break
        time.sleep(0.5)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    want = "put" if is_put else "call"
    df = df[(df["type"] == want) & df["ticker"].isin(wl)].copy()
    df["ask_frac"] = df["total_ask_side_prem"].astype(float) / df["total_premium"].astype(float).clip(lower=1)
    df = df[(df["ask_frac"] >= 0.6) & df["has_sweep"].astype(bool)]
    ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(D.ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return df[df["mi"].between(0, 375)].sort_values(["ticker", "date", "mi"])


class ProdS:
    """Prod-faithful settings baseline (mirrors the live droplet exit stack)."""
    ENABLE_V6_SCALEOUT = False
    ENABLE_V6_2PM_TIGHTEN = False
    ENABLE_V6_BREAKEVEN_RATCHET = True
    V6_BREAKEVEN_TRIGGER_PCT = 20.0
    ENABLE_V7_PROFIT_LOCK = True
    V7_PROFIT_LOCK_KEEP_FRAC = 0.8
    V7_PROFIT_LOCK_ACTIVATE_PCT = 25.0
    V7_PROFIT_LOCK_PUTS = True
    ENABLE_0DTE_PREMIUM_HARDSTOP = True
    PREMIUM_HARDSTOP_0DTE_PCT = 25.0
    ENABLE_MULTIDAY_CALL_HARDSTOP = True
    MULTIDAY_CALL_HARDSTOP_PCT = 25.0
    ENABLE_MULTIDAY_PUT_HARDSTOP = True
    MULTIDAY_PUT_HARDSTOP_PCT = 25.0
    ENABLE_STALL_CUT = True
    STALL_CUT_MIN_MINUTES = 30.0
    STALL_CUT_LOSS_PCT = 30.0
    STALL_CUT_PEAK_PCT = 10.0
    ENABLE_EOD_CLOSE_ALL = True


def _mk_settings(**over):
    s = ProdS()
    for k, v in over.items():
        setattr(s, k, v)
    return s


def collect_raw():
    """Collect raw price paths for the flow book (calls+puts) — the expensive step, done once + cached."""
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            trades = pickle.load(f)
        print(f"  (loaded {len(trades)} cached raw paths from {CACHE.name})")
        return trades
    trades = []
    for is_put, wl in ((True, R.PUT_UNIV), (False, R.CALL_UNIV)):
        otype = "put" if is_put else "call"
        right = "PUT" if is_put else "CALL"
        raw = _fetch_resilient(is_put, wl)
        if raw.empty:
            print(f"  WARNING: empty {otype} fetch (UW API overloaded) — skipping")
            continue
        for tk in sorted(raw["ticker"].unique()):
            stock, opts = D._stock(tk), D._opts(tk, right)
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
                    same = oday[oday["dte"] == dte0]
                    use_otm = tk in (R.OTM_PUT if is_put else R.OTM_CALL)
                    strike, _ = select_flow_strike(
                        [{"strike": float(r.strike), "mid": float(r.close)} for r in same.itertuples()],
                        spot, is_put, use_otm, R.OTM_TARGET)
                    if not strike:
                        continue
                    ch = opts[(opts["date"] == d) & (opts["strike"] == strike) & (opts["dte"] == dte0)]
                    ch = ch[ch["mi"] >= mb].sort_values("mi")
                    if len(ch) < 5:
                        continue
                    pp = ch["close"].values.astype(float)
                    mp = ch["mi"].values.astype(int)
                    if np.isnan(pp[0]) or pp[0] <= 0:
                        continue
                    up = [stock[d].get(int(m), spot) for m in mp]
                    ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=D.ET) + timedelta(minutes=mb)
                    trades.append({"tk": tk, "otype": otype, "dte0": dte0, "pp": pp, "mp": mp,
                                   "up": up, "ep": float(pp[0]), "ets": ets, "date": d, "mi": int(ev["mi"])})
    if trades:
        with open(CACHE, "wb") as f:
            pickle.dump(trades, f)
        print(f"  (cached {len(trades)} raw paths → {CACHE.name})")
    return trades


# cache the per-(ticker,otype) base config so we don't rebuild it per trade per variant
_CFG = {}


def _base_cfg(tk, otype):
    k = (tk, otype)
    if k not in _CFG:
        _CFG[k] = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=(otype == "put"))
    return _CFG[k]


def _scale_trail(cfg, factor):
    """Return a cfg copy with every adaptive-tier trail_width and soft/scalp widths scaled."""
    def widen(tiers):
        return tuple(replace(t, trail_width=round(t.trail_width * factor, 1)) for t in tiers)
    return replace(
        cfg,
        adaptive_highvol_tiers=widen(cfg.adaptive_highvol_tiers),
        adaptive_index_tiers=widen(cfg.adaptive_index_tiers),
        adaptive_standard_tiers=widen(cfg.adaptive_standard_tiers),
    )


def resim(trades, settings, cfg_fn=None):
    """Re-run the FSM over cached paths with the given settings (+ optional cfg mutator). Returns recs."""
    recs = []
    for t in trades:
        cfg = _base_cfg(t["tk"], t["otype"])
        if cfg_fn:
            cfg = cfg_fn(cfg)
        fsm = R.ExitFSM(cfg, settings=settings)
        st = R.TradeState(trade_id=1, ticker=t["tk"], option_type=t["otype"], entry_premium=t["ep"],
                          entry_time=t["ets"], contracts=1, peak_premium=t["ep"],
                          entry_underlying_price=t["up"][0], dte=t["dte0"],
                          expiry_date=t["ets"].strftime("%Y-%m-%d"))
        pp, mp, up, ep = t["pp"], t["mp"], t["up"], t["ep"]
        last, ret = ep, None
        for k in range(1, len(pp)):
            prem = pp[k]
            if prem is None or np.isnan(prem) or prem <= 0:
                continue
            last = prem
            now = t["ets"] + timedelta(minutes=int(mp[k] - mp[0]))
            mtc = max(0, 960 - (now.hour * 60 + now.minute))
            act = fsm.evaluate(st, prem, prem * (1 - HC), prem, now,
                               current_underlying=up[k], minutes_to_close=mtc, candle_data={})
            if act.should_exit:
                ret = (prem * (1 - HC) - ep) / ep * 100
                break
        if ret is None:
            ret = (last * (1 - HC) - ep) / ep * 100
        recs.append({"ret": ret, "date": t["date"], "mi": t["mi"], "otype": t["otype"]})
    return recs


def stats(recs):
    rets = [r["ret"] for r in recs]
    pnl = sum(BASE * r / 100 for r in rets)
    g = sum(BASE * r / 100 for r in rets if r > 0)
    ll = -sum(BASE * r / 100 for r in rets if r < 0)
    pf = g / ll if ll > 0 else float("inf")
    wr = sum(1 for r in rets if r > 0) / len(rets) * 100 if rets else 0
    # max drawdown on day/minute-ordered cumulative flat P&L
    order = sorted(recs, key=lambda r: (r["date"], r["mi"]))
    cum = peak = 0.0
    dd = 0.0
    for r in order:
        cum += BASE * r["ret"] / 100
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return pnl, pf, wr, dd


def variants(family):
    """Yield (label, settings, cfg_fn). Baseline first."""
    yield ("BASELINE (prod)", _mk_settings(), None)
    if family in ("lock", "all"):
        for kf in (0.5, 0.6, 0.7, 0.9, 0.95):
            yield (f"profit-lock keep={kf}", _mk_settings(V7_PROFIT_LOCK_KEEP_FRAC=kf), None)
        for ap in (15, 20, 30, 40):
            yield (f"profit-lock arm=+{ap}%", _mk_settings(V7_PROFIT_LOCK_ACTIVATE_PCT=float(ap)), None)
        yield ("profit-lock OFF", _mk_settings(ENABLE_V7_PROFIT_LOCK=False), None)
        yield ("profit-lock PUTS off", _mk_settings(V7_PROFIT_LOCK_PUTS=False), None)
    if family in ("stops", "all"):
        for hs in (15, 20, 30, 35):
            yield (f"0DTE hardstop={hs}%", _mk_settings(PREMIUM_HARDSTOP_0DTE_PCT=float(hs)), None)
        yield ("0DTE hardstop OFF", _mk_settings(ENABLE_0DTE_PREMIUM_HARDSTOP=False), None)
        for hs in (20, 30, 35):
            yield (f"MD-call hardstop={hs}%", _mk_settings(MULTIDAY_CALL_HARDSTOP_PCT=float(hs)), None)
        yield ("MD-call hardstop OFF", _mk_settings(ENABLE_MULTIDAY_CALL_HARDSTOP=False), None)
        for hs in (20, 30, 40):
            yield (f"MD-put hardstop={hs}%", _mk_settings(MULTIDAY_PUT_HARDSTOP_PCT=float(hs)), None)
        yield ("MD-put hardstop OFF", _mk_settings(ENABLE_MULTIDAY_PUT_HARDSTOP=False), None)
    if family in ("exit", "all"):
        yield ("stall-cut OFF", _mk_settings(ENABLE_STALL_CUT=False), None)
        for lp in (20, 40):
            yield (f"stall-cut loss={lp}%", _mk_settings(STALL_CUT_LOSS_PCT=float(lp)), None)
        yield ("breakeven-ratchet OFF", _mk_settings(ENABLE_V6_BREAKEVEN_RATCHET=False), None)
        for tr in (15, 25):
            yield (f"BE-ratchet trigger=+{tr}%", _mk_settings(V6_BREAKEVEN_TRIGGER_PCT=float(tr)), None)
    if family in ("trail", "all"):
        for f in (0.7, 0.8, 0.9, 1.1, 1.25):
            yield (f"trail width ×{f}", _mk_settings(), (lambda c, ff=f: _scale_trail(c, ff)))
    if family in ("soft", "all"):
        # soft_trail (18% of exits) + checkpoint_cut (10%) are top gates — sweep their cfg fields.
        for kp in (0.4, 0.5, 0.7, 0.8, 0.9):
            yield (f"soft-trail keep={kp}", _mk_settings(),
                   (lambda c, k=kp: replace(c, soft_trail_keep_pct=k)))
        for cp in (10.0, 20.0, 25.0, 35.0):
            yield (f"checkpoint drop={cp:.0f}%", _mk_settings(),
                   (lambda c, x=cp: replace(c, checkpoint_drop_pct=x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all", choices=["exit", "stops", "lock", "trail", "soft", "all"])
    args = ap.parse_args()

    print("Collecting raw flow price paths (one-time thetadata load)…")
    trades = collect_raw()
    days = sorted({t["date"] for t in trades})
    print(f"{len(trades)} flow trades over {len(days)} days ({days[0]}..{days[-1]})\n")

    base = None
    rows = []
    for label, settings, cfg_fn in variants(args.family):
        recs = resim(trades, settings, cfg_fn)
        pnl, pf, wr, dd = stats(recs)
        if base is None:
            base = pnl
        rows.append((label, pnl, pf, wr, dd, pnl - base))

    print(f"{'variant':<26}{'P&L':>10}{'PF':>7}{'WR':>6}{'maxDD':>10}{'Δ vs prod':>12}")
    print("-" * 71)
    # baseline first, then the rest ranked by Δ
    b = rows[0]
    print(f"{b[0]:<26}{b[1]:>+10,.0f}{b[2]:>7.2f}{b[3]:>5.0f}%{b[4]:>10,.0f}{'—':>12}")
    for label, pnl, pf, wr, dd, delta in sorted(rows[1:], key=lambda r: -r[5]):
        flag = "  ✓" if delta > 300 and pf >= b[2] else ""
        print(f"{label:<26}{pnl:>+10,.0f}{pf:>7.2f}{wr:>5.0f}%{dd:>10,.0f}{delta:>+12,.0f}{flag}")

    print("\n✓ = beats prod by >$300 at no worse PF. Flat-$750, same trades — pure exit-policy delta.")
    print("A single lone winner among a monotone-losing neighborhood is likely overfit; trust trends.")


if __name__ == "__main__":
    main()
