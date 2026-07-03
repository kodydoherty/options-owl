"""Full-stack backtest: merge the ML book (calls+puts) and the UW flow book onto ONE shared
account with production sizing, so the combined number is realistic (shared capital, not two
standalone $23k accounts added together).

Inputs:
  - ML trades:   journal/v3_eval_results/gold_standard_raw.json  (the 'trade_log' emitted by
                 backtest_gold_standard.py --puts ... — run that FIRST for the same window).
  - Flow trades: fetched live from the UW flow-alerts API (back to START) + priced from
                 thetadata, via flow_gold_standard_report.collect() (PROD strike + conviction).

Sizing (prod parity, per trade):
  base   = balance * MAX_PORTFOLIO_RISK_PCT/100 / MAX_CONCURRENT * FLAT_MULT
  mult   = conf_linear(pattern_conf)   for ML     (0.4..1.8 over conf 0.74..0.95)
         = flow_conviction_mult         for flow   (already in the trade's conv_mult)
  budget = min(base*mult, balance*MAX_POSITION_PCT/100, MAX_POSITION_DOLLARS)
  pnl    = budget * ret_pct/100        (ret_pct is return-on-capital)
Concurrency: <= MAX_CONCURRENT new positions deployed per (date, 5-min bucket); rarely binds.

Usage:
  python scripts/backtest_full_stack.py                 # window = ML trade_log's span
  python scripts/backtest_full_stack.py --flow-cache /tmp/flow.json   # reuse a fetched flow set
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "options_owl"))

ML_JSON = ROOT / "journal" / "v3_eval_results" / "gold_standard_raw.json"

# ── prod sizing constants (mirror settings.py / docker-compose kody) ──
PORTFOLIO_START = 23000.0
MAX_PORTFOLIO_RISK_PCT = 75.0
MAX_CONCURRENT = 5
FLAT_MULT = 0.85
MAX_POSITION_PCT = 15.0
MAX_POSITION_DOLLARS = 50000.0
CONF_LO, CONF_HI = 0.74, 0.95          # conf_ref
MULT_LO, MULT_HI = 0.4, 1.8            # conf_budget bounds


def conf_linear_mult(conf: float) -> float:
    if conf is None:
        return 1.0
    frac = (conf - CONF_LO) / (CONF_HI - CONF_LO)
    m = MULT_LO + frac * (MULT_HI - MULT_LO)
    return max(MULT_LO, min(MULT_HI, m))


FLAT_SLEEVE = 750.0  # team-standard per-trade edge measure (trustworthy; no compounding fantasy)


def _period_dates(period):
    """All calendar dates in 'YYYY-MM-DD to YYYY-MM-DD' — window is the whole period, not ML-trade-days."""
    from datetime import datetime, timedelta
    a, b = period.split(" to ")
    d0 = datetime.strptime(a, "%Y-%m-%d"); d1 = datetime.strptime(b, "%Y-%m-%d")
    out, d = set(), d0
    while d <= d1:
        if d.weekday() < 5:
            out.add(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def load_ml_trades():
    r = json.loads(ML_JSON.read_text())
    period = r.get("period", "?")
    out = []
    for t in r.get("trade_log", []):
        entry = t.get("effective_entry") or t.get("entry") or 0.0
        contracts = t.get("effective_contracts") or t.get("contracts") or 0
        deployed = entry * contracts * 100
        pnl = t.get("pnl", 0.0)
        ret_pct = (pnl / deployed * 100) if deployed > 0 else 0.0
        out.append({
            "book": "ML",
            "date": t["day"],
            "mi": t.get("signal_minute", t.get("minute", 0)),
            "ticker": t["ticker"],
            "side": t.get("direction", "call"),
            "ret_pct": ret_pct,
            "mult": conf_linear_mult(t.get("pattern_conf")),
        })
    return out, period, r


def load_flow_trades(window_dates, cache_path=None):
    if cache_path and Path(cache_path).exists():
        raw = json.loads(Path(cache_path).read_text())
        print(f"  flow: loaded {len(raw)} cached trades from {cache_path}")
    else:
        import flow_gold_standard_report as F
        print("  flow: fetching UW alerts (back to START) + pricing from thetadata — a few min...")
        raw = F.collect(True, F.PUT_UNIV) + F.collect(False, F.CALL_UNIV)
        if cache_path:
            Path(cache_path).write_text(json.dumps(raw))
            print(f"  flow: cached {len(raw)} trades -> {cache_path}")
    out = []
    for t in raw:
        if t["date"] not in window_dates:
            continue
        out.append({
            "book": "FLOW",
            "date": t["date"],
            "mi": t.get("mb", 200),          # 5-min bucket; flow lacks exact minute
            "ticker": t["ticker"],
            "side": t["side"],
            "ret_pct": t["ret_pct"],
            "mult": t.get("conv_mult", 1.0),
        })
    return out


def flat_edge(trades):
    """Team-standard edge: flat $750/trade, NO compounding, NO concurrency cap. The trustworthy number."""
    daily = defaultdict(float)
    per_book = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0, "gw": 0.0, "gl": 0.0})
    wins = losses = 0; gw = gl = 0.0
    for t in trades:
        pnl = FLAT_SLEEVE * t["ret_pct"] / 100
        daily[t["date"]] += pnl
        b = per_book[t["book"]]; b["n"] += 1; b["pnl"] += pnl
        if pnl > 0:
            wins += 1; gw += pnl; b["wins"] += 1; b["gw"] += pnl
        else:
            losses += 1; gl += -pnl; b["gl"] += -pnl
    n = wins + losses
    # maxDD on the cumulative daily edge curve
    cum = peak = 0.0; max_dd = 0.0
    for d in sorted(daily):
        cum += daily[d]; peak = max(peak, cum); max_dd = max(max_dd, peak - cum)
    return {"total_pnl": sum(daily.values()), "trades": n,
            "win_rate": (wins / n * 100) if n else 0.0,
            "pf": (gw / gl) if gl > 0 else float("inf"), "max_dd": max_dd,
            "per_book": dict(per_book), "daily": dict(daily)}


def simulate(trades):
    """One shared account, chronological, prod sizing + per-bucket concurrency cap."""
    trades = sorted(trades, key=lambda x: (x["date"], x["mi"]))
    balance = PORTFOLIO_START
    peak = balance
    max_dd = 0.0
    per_book = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0, "gross_win": 0.0, "gross_loss": 0.0})
    daily = defaultdict(float)
    wins = losses = 0
    gross_win = gross_loss = 0.0
    # concurrency cap per (date, bucket)
    bucket_count = defaultdict(int)
    for t in trades:
        key = (t["date"], t["mi"])
        if bucket_count[key] >= MAX_CONCURRENT:
            continue
        bucket_count[key] += 1
        base = balance * MAX_PORTFOLIO_RISK_PCT / 100 / MAX_CONCURRENT * FLAT_MULT
        budget = min(base * t["mult"], balance * MAX_POSITION_PCT / 100, MAX_POSITION_DOLLARS)
        pnl = budget * t["ret_pct"] / 100
        balance += pnl
        daily[t["date"]] += pnl
        b = per_book[t["book"]]
        b["n"] += 1
        b["pnl"] += pnl
        if pnl > 0:
            wins += 1; gross_win += pnl; b["wins"] += 1; b["gross_win"] += pnl
        else:
            losses += 1; gross_loss += -pnl; b["gross_loss"] += -pnl
        peak = max(peak, balance)
        max_dd = max(max_dd, (peak - balance) / peak * 100)
    n = wins + losses
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    return {
        "final": balance, "total_pnl": balance - PORTFOLIO_START,
        "return_pct": (balance - PORTFOLIO_START) / PORTFOLIO_START * 100,
        "trades": n, "win_rate": (wins / n * 100) if n else 0.0,
        "pf": pf, "max_dd": max_dd, "per_book": dict(per_book), "daily": dict(daily),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-cache", default=None, help="JSON path to cache/reuse fetched flow trades")
    args = ap.parse_args()

    print("=" * 72)
    print("FULL-STACK BACKTEST — ML (calls+puts) + UW flow on ONE shared $23k account")
    print("=" * 72)

    ml, period, r = load_ml_trades()
    window_dates = _period_dates(period)            # WHOLE period, not just ML-trade-days
    print(f"  ML book:   {len(ml)} trades | period {period}")
    print(f"  window:    {min(window_dates)} .. {max(window_dates)} ({len(window_dates)} trading days)")

    flow = load_flow_trades(window_dates, args.flow_cache)
    print(f"  flow book: {len(flow)} trades (in window)")

    # ── PRIMARY: flat $750/trade edge (trustworthy, additive, no compounding) ──
    ml_e, flow_e, comb_e = flat_edge(ml), flat_edge(flow), flat_edge(ml + flow)

    def eline(label, s):
        print(f"  {label:<28} {s['trades']:>4}tr  {s['win_rate']:>5.1f}%WR  "
              f"${s['total_pnl']:>+9,.0f}  PF {s['pf']:>4.2f}  maxDD ${s['max_dd']:>7,.0f}")

    print("\n  PRIMARY — flat $750/trade edge (the trustworthy measure):")
    print("  " + "-" * 68)
    eline("ML (calls+puts)", ml_e)
    eline("FLOW (whale sweeps)", flow_e)
    print("  " + "-" * 68)
    eline("COMBINED full stack", comb_e)
    print("  " + "-" * 68)

    print("\n  Worst 5 days (combined, flat $750):")
    for d, p in sorted(comb_e["daily"].items(), key=lambda x: x[1])[:5]:
        print(f"    {d}  ${p:>+9,.0f}")
    print("  Best 5 days (combined, flat $750):")
    for d, p in sorted(comb_e["daily"].items(), key=lambda x: -x[1])[:5]:
        print(f"    {d}  ${p:>+9,.0f}")

    # ── SECONDARY: compounded shared $23k account (OPTIMISTIC — liquidity/fills not modeled) ──
    comb_c = simulate(ml + flow)
    print("\n  SECONDARY — compounded shared $23k account (OPTIMISTIC, liquidity-blind):")
    print(f"    ${PORTFOLIO_START:,.0f} -> ${comb_c['final']:,.0f}  ({comb_c['return_pct']:+.1f}%)  "
          f"PF {comb_c['pf']:.2f}  DD {comb_c['max_dd']:.1f}%  ({comb_c['trades']} trades after concurrency cap)")
    print("    (compounding + real fills overstate this — see memory 'account-projection-reconciliation'.)")

    print("\n  NOTE: ML book on the AUTHORITATIVE compounded harness (prod conf_linear sizing) over this")
    print(f"        window = ${r['total_pnl']:+,.0f} ({r['return_pct']:+.1f}%, PF {r['profit_factor']}). The")
    print("        flat-$750 ML number above is the same trades on the edge basis (apples-to-apples w/ flow).")

    out = ROOT / "journal" / "v3_eval_results" / "full_stack_combined.json"
    out.write_text(json.dumps({"period": period,
                               "flat750": {"ml": _clean(ml_e), "flow": _clean(flow_e), "combined": _clean(comb_e)},
                               "compounded_optimistic": _clean(comb_c),
                               "ml_authoritative_compounded": {"total_pnl": r["total_pnl"],
                                                               "return_pct": r["return_pct"],
                                                               "pf": r["profit_factor"]}}, indent=2))
    print(f"\n  Saved -> {out}")


def _clean(s):
    return {k: v for k, v in s.items() if k not in ("per_book", "daily")}


if __name__ == "__main__":
    main()
