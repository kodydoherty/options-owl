"""Does P(runner) SEPARATE the outcomes of REAL trades we actually placed?

Honest gate before building runner-based sizing. Scores every real entry (REAL_ML,
REAL_DISCORD, REAL_FLOW) with the runner model(s) using the PRODUCTION serve-time feature
builder (compute_option_features_from_live -> predict_entry_confidence, the signal_ml_v2 path
in flow_runner.py) AND the standalone ml_v3 runner_v1 model (its own 18-feature schema), then
measures AUC / quartile monotonicity / Spearman of P(runner) vs the REALIZED runner outcome,
and the PF/P&L lift of P(runner)-proportional sizing vs flat.

REAL DATA ONLY. No-lookahead: every feature uses only thetadata bars at/before the entry minute;
the realized outcome (forward peak + ExitFSM sim) uses bars AFTER entry.

Read-only on all DBs. Run locally: python scripts/runner_separation_realdata.py [--validate]
"""

from __future__ import annotations

import csv
import sqlite3
from collections import Counter
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

ET = ZoneInfo("America/New_York")
THETA_DB = str(PROJECT / "journal" / "thetadata_options.db")
UW_DB = str(PROJECT / "journal" / "uw_historical.db")
SIGNALS_CSV = str(PROJECT / "journal" / "real_signals_kody.csv")

from options_owl.sourcing.scoring.ml_gates.signal_model import (  # noqa: E402
    compute_option_features_from_live,
    predict_entry_confidence,
)
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402
from options_owl.risk.exit_v5.config import V5Config  # noqa: E402

# Exit sim config — mirror entry_timing_oracle.LOCK (V7 wide-trail + profit-lock)
LOCK = SimpleNamespace(
    ENABLE_V6_SCALEOUT=False, ENABLE_V6_2PM_TIGHTEN=False,
    ENABLE_V6_BREAKEVEN_RATCHET=True, V6_BREAKEVEN_TRIGGER_PCT=20.0,
    ENABLE_V7_PROFIT_LOCK=True, V7_PROFIT_LOCK_KEEP_FRAC=0.6,
    V7_PROFIT_LOCK_ACTIVATE_PCT=30.0,
)
HAIRCUT = 0.02  # exit haircut, same as oracle harness default region

RUNNER_PCT = 50.0      # task definition: realized peak >= +50%
RUNNER_PCT_100 = 100.0  # also report at the meta's 100% label
RUNNER_PCT_30 = 30.0    # statistically-usable "good trade" threshold (22% base rate)

CALL_WL = {"META", "SPY", "AMZN", "TSLA", "AMD", "ORCL", "INTC", "ARM", "GOOG", "LRCX"}
PUT_WL = {"META", "AMZN", "AAPL", "TSLA", "MU", "SPY"}

ENTRY_MINUTES_RUNNER_V1 = None  # we use the actual entry minute, not the scan grid


def connect(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=15)
    con.execute("PRAGMA busy_timeout=5000")
    return con


# ------------------------------------------------------------------ contract series loader
def load_contract_day(con, ticker, right, exp, day, target_strike, tol_pct=2.0):
    """Return (strike_used, rows) for the contract nearest target_strike at this exp/day.

    rows: list of dicts oldest->newest with mi, close, high, volume, delta, theta, vega, iv,
    underlying_price, bid, ask, bid_size, ask_size — pulled from thetadata for the WHOLE day.
    Returns (None, None) if expiry/day not captured or nearest strike beyond tol_pct.
    """
    avail = [r[0] for r in con.execute(
        "SELECT DISTINCT strike FROM option_ohlc WHERE ticker=? AND right=? AND expiration=? "
        "AND substr(timestamp,1,10)=?", (ticker, right, exp, day)).fetchall()]
    if not avail:
        return None, None, "expiry_day_missing"
    if target_strike in avail:
        strike = target_strike
        why = "exact"
    else:
        strike = min(avail, key=lambda s: abs(s - target_strike))
        dev = abs(strike - target_strike) / max(target_strike, 1e-9) * 100
        if dev > tol_pct:
            return None, None, f"nearest_strike_{dev:.1f}pct_off"
        why = f"nearest_{dev:.2f}pct"

    q = """
    SELECT o.timestamp ts, o.close, o.high, o.volume,
           g.delta, g.theta, g.vega, g.implied_vol, g.underlying_price,
           q.bid, q.ask, q.bid_size, q.ask_size
    FROM option_ohlc o
    LEFT JOIN option_greeks g
      ON o.ticker=g.ticker AND o.expiration=g.expiration AND o.strike=g.strike
         AND o.right=g.right AND o.timestamp=g.timestamp
    LEFT JOIN option_quotes q
      ON o.ticker=q.ticker AND o.expiration=q.expiration AND o.strike=q.strike
         AND o.right=q.right AND o.timestamp=q.timestamp
    WHERE o.ticker=? AND o.right=? AND o.expiration=? AND o.strike=?
      AND substr(o.timestamp,1,10)=?
    ORDER BY o.timestamp
    """
    rows = []
    for ts, close, high, vol, delta, theta, vega, iv, up, bid, ask, bsz, asz in con.execute(
            q, (ticker, right, exp, strike, day)):
        hh = int(ts[11:13]); mm = int(ts[14:16])
        mi = (hh - 9) * 60 + (mm - 30)
        rows.append({
            "mi": mi, "close": close, "high": high, "volume": vol or 0,
            "delta": delta, "theta": theta, "vega": vega, "iv": iv,
            "underlying_price": up, "bid": bid, "ask": ask,
            "bid_size": bsz or 0, "ask_size": asz or 0,
        })
    rows = [r for r in rows if r["mi"] >= 0]
    if not rows:
        return None, None, "no_bars"
    return strike, rows, why


def load_stock_day(con, ticker, day):
    """{mi: close} for the underlying's regular-hours minute bars."""
    out = {}
    for ts, close in con.execute(
            "SELECT timestamp, close FROM stock_ohlc WHERE ticker=? AND substr(timestamp,1,10)=? "
            "ORDER BY timestamp", (ticker, day)):
        hh = int(ts[11:13]); mm = int(ts[14:16])
        mi = (hh - 9) * 60 + (mm - 30)
        if mi >= 0:
            out[mi] = close
    return out


def prior_day_stats(con, ticker, day):
    """(gap_pct_placeholder, prior_range_pct) — prior day fully closed, serve-safe."""
    rows = con.execute(
        "SELECT substr(timestamp,1,10) d, min(low) lo, max(high) hi, "
        "  (SELECT close FROM stock_ohlc s2 WHERE s2.ticker=stock_ohlc.ticker "
        "    AND substr(s2.timestamp,1,10)=substr(stock_ohlc.timestamp,1,10) ORDER BY s2.timestamp DESC LIMIT 1) cl "
        "FROM stock_ohlc WHERE ticker=? AND substr(timestamp,1,10)<? "
        "GROUP BY d ORDER BY d DESC LIMIT 1", (ticker, day)).fetchone()
    if not rows or not rows[3]:
        return None, 0.0
    _, lo, hi, cl = rows
    pr = (hi - lo) / cl * 100 if cl else 0.0
    return cl, pr


# ------------------------------------------------------------------ feature builders
def build_serve_features(rows, entry_idx, stock_day, is_call):
    """signal_ml_v2 serve-path features via compute_option_features_from_live.

    Uses ONLY rows[:entry_idx+1] (no lookahead). Returns None if greeks missing at entry.
    """
    er = rows[entry_idx]
    if er["delta"] is None or er["iv"] is None or er["close"] is None or er["close"] <= 0:
        return None
    bid = er["bid"] if er["bid"] is not None else 0.0
    ask = er["ask"] if er["ask"] is not None else 0.0
    premium = er["close"]
    entry_mi = er["mi"]

    # trailing option premium/volume histories EXCLUDING current (oldest->newest)
    hist = rows[:entry_idx]
    premium_history = [h["close"] for h in hist if h["close"] and h["close"] > 0]
    volume_history = [int(h["volume"]) for h in hist]
    # underlying history INCLUDING current (from stock bars up to entry minute)
    und_minutes = sorted(m for m in stock_day if m <= entry_mi)
    underlying_history = [stock_day[m] for m in und_minutes]
    underlying_price = (er["underlying_price"] if er["underlying_price"]
                        else (underlying_history[-1] if underlying_history else 0))
    if not underlying_price or underlying_price <= 0:
        return None

    feats = compute_option_features_from_live(
        ticker="X", premium=premium, bid=bid, ask=ask,
        iv=er["iv"], delta=er["delta"], theta=er["theta"] or 0, vega=er["vega"] or 0,
        volume=int(er["volume"]), underlying_price=underlying_price,
        minutes_since_open=entry_mi, is_call=is_call,
        premium_history=premium_history, volume_history=volume_history,
        underlying_history=underlying_history,
        bid_size=er["bid_size"], ask_size=er["ask_size"],
    )
    return feats


def build_runner_v1_features(rows, entry_idx, stock_day, ticker, target_strike,
                             dte, gap_pct, prior_range, day):
    """ml_v3 runner_v1's 18-feature schema (scripts/runner_prediction.py)."""
    er = rows[entry_idx]
    if er["delta"] is None or er["delta"] <= 0 or er["close"] is None or er["close"] <= 0:
        return None
    entry_prem = er["close"]
    und_now = er["underlying_price"]
    if not und_now or und_now <= 0:
        # fall back to stock series
        und_minutes = sorted(m for m in stock_day if m <= er["mi"])
        und_now = stock_day[und_minutes[-1]] if und_minutes else 0
    if not und_now or und_now <= 0:
        return None
    iv = er["iv"] if er["iv"] is not None else 0.0
    vega = er["vega"] if er["vega"] is not None else 0.0
    theta = er["theta"] if er["theta"] is not None else 0.0
    bid = er["bid"] if er["bid"] is not None else 0.0
    ask = er["ask"] if er["ask"] is not None else 0.0
    spread_pct = (ask - bid) / ask * 100 if ask > 0 else 0.0
    moneyness = target_strike / und_now if und_now > 0 else 1.0
    entry_mi = er["mi"]

    day_open_und = None
    if 0 in stock_day:
        day_open_und = stock_day[0]
    else:
        ms = sorted(stock_day.keys())
        day_open_und = stock_day[ms[0]] if ms else und_now
    und_move_pct = (und_now / day_open_und - 1) * 100 if day_open_und else 0.0

    # last-5-min underlying slope
    recent = [stock_day[m] for m in sorted(stock_day) if entry_mi - 5 < m <= entry_mi]
    und_slope_5 = (recent[-1] / recent[0] - 1) * 100 if len(recent) >= 2 and recent[0] > 0 else 0.0
    # underlying rvol last 15m
    r15 = [stock_day[m] for m in sorted(stock_day) if entry_mi - 15 < m <= entry_mi]
    if len(r15) >= 5:
        arr = np.array(r15)
        und_rvol_15 = float(np.std(np.diff(arr) / arr[:-1]) * 100)
    else:
        und_rvol_15 = 0.0
    # option volume last 5 min for this contract
    opt_vol_5 = float(sum(r["volume"] for r in rows if entry_mi - 5 < r["mi"] <= entry_mi))

    try:
        dow = date(*[int(x) for x in day.split("-")]).weekday()
    except Exception:
        dow = 0

    return {
        "entry_premium": entry_prem,
        "log_premium": float(np.log(entry_prem)),
        "delta": er["delta"],
        "iv": iv, "vega": vega, "theta": theta,
        "moneyness": moneyness,
        "spread_pct": spread_pct,
        "und_move_pct": und_move_pct,
        "und_slope_5": und_slope_5,
        "und_rvol_15": und_rvol_15,
        "opt_vol_5": opt_vol_5,
        "gap_pct": gap_pct if gap_pct is not None else 0.0,
        "prior_range_pct": prior_range,
        "dte": dte,
        "entry_min": entry_mi,
        "ticker": ticker,
        "day_of_week": dow,
    }


# ------------------------------------------------------------------ runner_v1 scorer
_RUNNER_V1 = None
_RUNNER_V1_META = None


def load_runner_v1():
    global _RUNNER_V1, _RUNNER_V1_META
    if _RUNNER_V1 is not None:
        return _RUNNER_V1, _RUNNER_V1_META
    import json
    import lightgbm as lgb
    import pandas as pd  # noqa
    p = PROJECT / "journal" / "models" / "ml_v3" / "runner_v1.lgb"
    _RUNNER_V1 = lgb.Booster(model_file=str(p))
    _RUNNER_V1_META = json.load(open(PROJECT / "journal" / "models" / "ml_v3" / "runner_v1_meta.json"))
    return _RUNNER_V1, _RUNNER_V1_META


def score_runner_v1(feat):
    import pandas as pd
    model, meta = load_runner_v1()
    cols = meta["features"]
    row = {c: feat.get(c) for c in cols}
    df = pd.DataFrame([row], columns=cols)
    for c in meta["cat_features"]:
        df[c] = df[c].astype("category")
    return float(model.predict(df)[0])


# ------------------------------------------------------------------ realized outcome
def realized_outcome(rows, entry_idx, otype, day):
    """Run the real ExitFSM from entry to EOD (return %) + forward peak gain %."""
    ep = rows[entry_idx]["close"]
    fut = rows[entry_idx + 1:]
    # forward peak (high preferred, else close), excluding last 15min for parity with training
    last_mi = rows[-1]["mi"]
    peak = ep
    for r in fut:
        if r["mi"] > last_mi - 15:
            continue
        cand = max([v for v in (r["high"], r["close"]) if v and v > 0], default=0)
        if cand > peak:
            peak = cand
    peak_gain = (peak / ep - 1) * 100

    # ExitFSM sim
    cfg = V5Config()
    fsm = ExitFSM(cfg, settings=LOCK)
    entry_mi = rows[entry_idx]["mi"]
    ets = datetime(*[int(x) for x in day.split("-")], 9, 30, tzinfo=ET) + timedelta(minutes=entry_mi)
    up0 = rows[entry_idx]["underlying_price"] or 0
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep, entry_time=ets,
                    contracts=1, peak_premium=ep, entry_underlying_price=up0, dte=0,
                    expiry_date=day)
    realized = None
    last = ep
    for r in fut:
        p = r["close"]
        if p is None or p <= 0:
            continue
        last = p
        now = ets + timedelta(minutes=int(r["mi"] - entry_mi))
        up = r["underlying_price"] or up0
        a = fsm.evaluate(st, p, p * (1 - HAIRCUT), p, now,
                         current_underlying=up,
                         minutes_to_close=max(0, 960 - (now.hour * 60 + now.minute)),
                         candle_data={})
        if a.should_exit:
            realized = (p * (1 - HAIRCUT) - ep) / ep * 100
            break
    if realized is None:
        realized = (last * (1 - HAIRCUT) - ep) / ep * 100
    return realized, peak_gain


# ------------------------------------------------------------------ entry-set loaders
def load_real_signals():
    out = []
    for r in csv.DictReader(open(SIGNALS_CSV)):
        try:
            oa = datetime.fromisoformat(r["opened_at"]).replace(tzinfo=timezone.utc).astimezone(ET)
        except Exception:
            continue
        out.append({
            "arm": r["arm"], "ticker": r["ticker"].upper(),
            "otype": r["option_type"].lower(), "strike": float(r["strike"]),
            "expiry": r["expiry_date"], "entry_dt": oa,
            "rec_mfe": float(r["mfe_pnl_pct"]) if r.get("mfe_pnl_pct") else None,
            "rec_pnl": float(r["pnl_pct"]) if r.get("pnl_pct") else None,
        })
    return out


def load_real_flow():
    con = connect(UW_DB)
    rows = con.execute("""SELECT ticker,created_at,type,strike,expiry,total_premium,total_ask_side_prem
        FROM flow_alerts WHERE has_sweep=1 AND total_premium>=250000""").fetchall()
    con.close()
    out = []
    for tk, ca, ty, strike, expiry, tp, ask in rows:
        if not tp or tp <= 0:
            continue
        if ask / tp < 0.6:
            continue
        tk = tk.upper()
        if ty == "call" and tk not in CALL_WL:
            continue
        if ty == "put" and tk not in PUT_WL:
            continue
        try:
            dt = datetime.fromisoformat(ca.replace("Z", "+00:00")).astimezone(ET)
        except Exception:
            continue
        out.append({
            "arm": "REAL_FLOW", "ticker": tk, "otype": ty,
            "strike": float(strike) if strike else 0, "expiry": expiry,
            "entry_dt": dt, "rec_mfe": None, "rec_pnl": None,
        })
    return out


# ------------------------------------------------------------------ entry minute resolution
def resolve_entry_idx(rows, entry_mi):
    """First bar at mi >= entry_mi with a valid close (>=0.05)."""
    for i, r in enumerate(rows):
        if r["mi"] >= entry_mi and r["close"] and r["close"] > 0.05:
            return i
    return None


# ------------------------------------------------------------------ main scoring loop
def process(entries, theta, validate=False, validate_n=5):
    results = []
    skips = []
    printed = 0
    for e in entries:
        ticker = e["ticker"]; otype = e["otype"]; right = otype.upper()
        exp = e["expiry"]; strike = e["strike"]
        day = e["entry_dt"].strftime("%Y-%m-%d")
        entry_mi = (e["entry_dt"].hour - 9) * 60 + e["entry_dt"].minute - 30

        if strike <= 0:
            skips.append((e["arm"], ticker, "no_strike")); continue

        strike_used, rows, why = load_contract_day(theta, ticker, right, exp, day, strike)
        if rows is None:
            skips.append((e["arm"], ticker, why)); continue

        entry_idx = resolve_entry_idx(rows, entry_mi)
        if entry_idx is None or entry_idx >= len(rows) - 2:
            skips.append((e["arm"], ticker, "no_entry_bar_or_eod")); continue

        stock_day = load_stock_day(theta, ticker, day)
        if not stock_day:
            skips.append((e["arm"], ticker, "no_stock_bars")); continue

        # DTE + prior-day stats for runner_v1
        try:
            ey, em_, ed = exp.split("-"); dy, dm, dd = day.split("-")
            dte = (date(int(ey), int(em_), int(ed)) - date(int(dy), int(dm), int(dd))).days
        except Exception:
            dte = 0
        prior_close, prior_range = prior_day_stats(theta, ticker, day)
        day_open_und = stock_day.get(min(stock_day.keys()))
        gap_pct = ((day_open_und / prior_close - 1) * 100) if (prior_close and day_open_und) else 0.0

        is_call = otype == "call"
        serve_feats = build_serve_features(rows, entry_idx, stock_day, is_call)
        if serve_feats is None:
            skips.append((e["arm"], ticker, "no_greeks_at_entry")); continue
        rv1_feats = build_runner_v1_features(rows, entry_idx, stock_day, ticker, strike_used,
                                             dte, gap_pct, prior_range, day)

        # Score serve path (signal_ml_v2 per-ticker/side -> generic)
        res = predict_entry_confidence(ticker, serve_feats, right)
        p_serve = res.get("runner_score")
        serve_src = res.get("model_source")
        if p_serve is None or serve_src == "none":
            p_serve = None

        # Score runner_v1 (ml_v3)
        p_rv1 = score_runner_v1(rv1_feats) if rv1_feats is not None else None

        recon_realized, recon_peak = realized_outcome(rows, entry_idx, otype, day)

        # OUTCOME SOURCE: prefer the RECORDED mfe/pnl (ground truth of what the bot actually
        # achieved) for REAL_ML/REAL_DISCORD. thetadata minute-`high` reconstruction massively
        # over-states peaks on illiquid 0DTE contracts (single-tick spikes), so it is only a
        # cross-check. REAL_FLOW has no recorded outcome -> falls back to reconstruction.
        if e["rec_mfe"] is not None and e["rec_pnl"] is not None:
            peak_gain = e["rec_mfe"]
            realized = e["rec_pnl"]
            outcome_src = "recorded"
        else:
            peak_gain = recon_peak
            realized = recon_realized
            outcome_src = "reconstructed"

        rec = {
            "arm": e["arm"], "ticker": ticker, "otype": otype, "day": day,
            "entry_mi": rows[entry_idx]["mi"], "strike_match": why,
            "p_serve": p_serve, "serve_src": serve_src, "p_rv1": p_rv1,
            "realized": realized, "peak_gain": peak_gain, "outcome_src": outcome_src,
            "is_runner30": 1 if peak_gain >= RUNNER_PCT_30 else 0,
            "is_runner50": 1 if peak_gain >= RUNNER_PCT else 0,
            "is_runner100": 1 if peak_gain >= RUNNER_PCT_100 else 0,
            "recon_realized": recon_realized, "recon_peak": recon_peak,
            "rec_mfe": e["rec_mfe"], "rec_pnl": e["rec_pnl"],
        }
        results.append(rec)

        if validate and printed < validate_n:
            printed += 1
            print(f"\n--- VALIDATE row {printed}: {e['arm']} {ticker} {otype} K={strike}"
                  f"({why}) {day} entry_mi={rec['entry_mi']} ---")
            print(f"  entry_prem={rows[entry_idx]['close']:.3f} delta={serve_feats['delta']:.3f} "
                  f"iv={serve_feats['iv']:.3f} vega={serve_feats['vega']:.3f} theta={serve_feats['theta']:.3f}")
            print(f"  und_change_5m={serve_feats['underlying_change_5m']:.2f} "
                  f"vol_ratio={serve_feats['volume_ratio']:.2f} spread_pct={serve_feats['spread_pct']:.2f}")
            if rv1_feats:
                print(f"  rv1: moneyness={rv1_feats['moneyness']:.4f} und_move={rv1_feats['und_move_pct']:.2f} "
                      f"dte={rv1_feats['dte']} gap={rv1_feats['gap_pct']:.2f}")
            print(f"  P(runner) serve={p_serve if p_serve is None else round(p_serve,3)} (src={serve_src}) "
                  f"| rv1={p_rv1 if p_rv1 is None else round(p_rv1,3)}")
            print(f"  OUTCOME realized={realized:.1f}% peak_gain={peak_gain:.1f}% "
                  f"runner50={rec['is_runner50']} | recorded mfe={e['rec_mfe']} pnl={e['rec_pnl']}")

    return results, skips


# ------------------------------------------------------------------ analysis
def auc(scores, labels):
    s = np.array(scores, dtype=float); y = np.array(labels, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # rank-based AUC
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    from scipy.stats import rankdata
    ranks = rankdata(allv)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def spearman(a, b):
    from scipy.stats import rankdata
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    if len(a) < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def quartile_table(recs, pkey, label="is_runner50"):
    vals = [(r[pkey], r) for r in recs if r.get(pkey) is not None]
    if len(vals) < 8:
        return f"  (n={len(vals)} too few for quartiles)"
    vals.sort(key=lambda x: x[0])
    n = len(vals)
    out = []
    for qi in range(4):
        lo = qi * n // 4
        hi = (qi + 1) * n // 4 if qi < 3 else n
        chunk = [r for _, r in vals[lo:hi]]
        if not chunk:
            continue
        runrate = np.mean([c[label] for c in chunk]) * 100
        wr = np.mean([1 if c["realized"] > 0 else 0 for c in chunk]) * 100
        avgret = np.mean([c["realized"] for c in chunk])
        avgpeak = np.mean([c["peak_gain"] for c in chunk])
        prange = f"{vals[lo][0]:.3f}-{vals[hi-1][0]:.3f}"
        out.append(f"  Q{qi+1} (p={prange}, n={len(chunk)}): "
                   f"runner%={runrate:4.0f}  WR={wr:4.0f}%  avgRet={avgret:+6.1f}%  avgPeak={avgpeak:6.1f}%")
    return "\n".join(out)


def sizing_lift(recs, pkey):
    """P(runner)-proportional sizing vs flat-1-contract. Quartile multipliers 0.5/0.85/1.15/1.5."""
    vals = [r for r in recs if r.get(pkey) is not None]
    if len(vals) < 8:
        return None
    vals_sorted = sorted(vals, key=lambda r: r[pkey])
    n = len(vals_sorted)
    mults = {}
    qm = [0.5, 0.85, 1.15, 1.5]
    for qi in range(4):
        lo = qi * n // 4
        hi = (qi + 1) * n // 4 if qi < 3 else n
        for r in vals_sorted[lo:hi]:
            mults[id(r)] = qm[qi]
    # flat
    flat_pnl = sum(r["realized"] for r in vals)
    flat_win = sum(r["realized"] for r in vals if r["realized"] > 0)
    flat_loss = -sum(r["realized"] for r in vals if r["realized"] < 0)
    flat_pf = flat_win / flat_loss if flat_loss > 0 else float("inf")
    # sized
    sz_pnl = sum(r["realized"] * mults[id(r)] for r in vals)
    sz_win = sum(r["realized"] * mults[id(r)] for r in vals if r["realized"] > 0)
    sz_loss = -sum(r["realized"] * mults[id(r)] for r in vals if r["realized"] < 0)
    sz_pf = sz_win / sz_loss if sz_loss > 0 else float("inf")
    # capital deployed (sum of multipliers) — normalize P&L per unit capital for a fair compare
    flat_cap = float(n)
    sz_cap = sum(mults.values())
    return {
        "n": n, "flat_pnl": flat_pnl, "flat_pf": flat_pf,
        "sized_pnl": sz_pnl, "sized_pf": sz_pf,
        "flat_per_unit": flat_pnl / flat_cap if flat_cap else 0,
        "sized_per_unit": sz_pnl / sz_cap if sz_cap else 0,
    }


def report_arm(name, recs):
    print(f"\n{'='*70}\nARM: {name}  (n={len(recs)})\n{'='*70}")
    if not recs:
        print("  no matched rows"); return
    osrc = Counter(r["outcome_src"] for r in recs)
    base30 = np.mean([r["is_runner30"] for r in recs]) * 100
    base50 = np.mean([r["is_runner50"] for r in recs]) * 100
    base100 = np.mean([r["is_runner100"] for r in recs]) * 100
    wr = np.mean([1 if r["realized"] > 0 else 0 for r in recs]) * 100
    print(f"  outcome_src={dict(osrc)}  base runner@30%={base30:.0f}% @50%={base50:.0f}% "
          f"@100%={base100:.0f}%  WR={wr:.0f}%  avgRealized={np.mean([r['realized'] for r in recs]):+.1f}%")
    for pkey, lbl in [("p_serve", "SERVE (signal_ml_v2 — PRODUCTION path)"),
                      ("p_rv1", "runner_v1 (ml_v3 standalone)")]:
        sc = [r[pkey] for r in recs if r.get(pkey) is not None]
        if len(sc) < 4:
            print(f"\n  [{lbl}] n={len(sc)} too few"); continue
        sub = [r for r in recs if r.get(pkey) is not None]
        y30 = [r["is_runner30"] for r in sub]
        y50 = [r["is_runner50"] for r in sub]
        ret = [r["realized"] for r in sub]
        a30 = auc(sc, y30); a50 = auc(sc, y50)
        sp = spearman(sc, ret)
        npos30 = sum(y30); npos50 = sum(y50)
        print(f"\n  [{lbl}] n={len(sc)}  P(runner) spread: "
              f"min={min(sc):.3f} med={np.median(sc):.3f} max={max(sc):.3f} std={np.std(sc):.3f}")
        print(f"    AUC@30%={a30 if a30 is None else round(a30,3)} (pos={npos30})  "
              f"AUC@50%={a50 if a50 is None else round(a50,3)} (pos={npos50})  "
              f"Spearman(P vs realized)={sp if sp is None else round(sp,3)}")
        print("    Quartiles (low P -> high P), runner%=@30%:")
        print(quartile_table(sub, pkey, label="is_runner30"))
        lift = sizing_lift(recs, pkey)
        if lift:
            print(f"    SIZING (0.5/0.85/1.15/1.5 by quartile) vs flat-1ct:")
            print(f"      flat:  PF={lift['flat_pf']:.2f}  totRet={lift['flat_pnl']:+.0f}%  "
                  f"per-unit-capital={lift['flat_per_unit']:+.2f}%")
            print(f"      sized: PF={lift['sized_pf']:.2f}  totRet={lift['sized_pnl']:+.0f}%  "
                  f"per-unit-capital={lift['sized_per_unit']:+.2f}%  "
                  f"(PF {lift['sized_pf']-lift['flat_pf']:+.2f})")


def main():
    validate = "--validate" in sys.argv
    theta = connect(THETA_DB)

    real = load_real_signals()
    flow = load_real_flow()
    all_entries = real + flow

    print(f"Loaded {len(real)} real signals + {len(flow)} flow = {len(all_entries)} entries")
    if validate:
        print("\n### VALIDATION MODE — first 5 matched rows ###")
        # validate across a mix
        sample = real[:3] + flow[:2]
        process(sample, theta, validate=True, validate_n=5)
        print("\n### END VALIDATION — re-run without --validate for full report ###")
        return

    results, skips = process(all_entries, theta)
    theta.close()

    # skip accounting
    from collections import Counter
    print(f"\n{'='*70}\nMATCH / SKIP ACCOUNTING\n{'='*70}")
    arms = ["REAL_ML", "REAL_DISCORD", "REAL_FLOW"]
    for arm in arms:
        m = sum(1 for r in results if r["arm"] == arm)
        sk = Counter(reason for a, t, reason in skips if a == arm)
        total = m + sum(sk.values())
        print(f"  {arm}: matched {m}/{total}  skips={dict(sk)}")
    print(f"  strike-match quality: {Counter(r['strike_match'].split('_')[0] for r in results)}")
    print(f"  serve model sources: {Counter(r['serve_src'] for r in results)}")

    for arm in arms:
        report_arm(arm, [r for r in results if r["arm"] == arm])
    report_arm("POOLED (all arms)", results)


if __name__ == "__main__":
    main()
