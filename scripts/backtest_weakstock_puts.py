"""Does buying PUTS on genuinely-weak stocks in a CONFIRMED DOWNTREND make money? (Kody's thesis:
'shit stocks going down'). Tests SMR/OKLO/ASTS/UUUU/JOBY/UEC over ~18mo from thetadata.

Entry rule: on each day the stock is in a confirmed downtrend (close < 20-day SMA AND close < close
5 days ago), buy an ATM put with the nearest expiry in [14, 45] DTE (multi-day — capture the trend,
avoid 0DTE theta). One open position per ticker at a time (non-overlapping). Exit via several rules.
Reports per-ticker + overall so we see if the downtrend edge is real or the bounces kill it.

    python scripts/backtest_weakstock_puts.py
"""
import sqlite3
import statistics as S
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "journal" / "thetadata_options.db"
TICKERS = ["SMR", "OKLO", "ASTS", "UUUU", "JOBY", "UEC"]
MIN_DTE, MAX_DTE = 14, 45


def _daily_stock(c, tk):
    rows = c.execute(
        "SELECT substr(timestamp,1,10) d, close FROM stock_ohlc WHERE ticker=? AND close>0 "
        "ORDER BY timestamp", (tk,)).fetchall()
    out = {}
    for d, cl in rows:
        out[d] = float(cl)  # last close of the day wins
    return sorted(out.items())


def _put_path(c, tk, entry_d, underlying):
    """Nearest ATM put, expiry DTE in [14,45], daily close series from entry onward."""
    exps = [r[0] for r in c.execute(
        "SELECT DISTINCT expiration FROM option_ohlc WHERE ticker=? AND right='PUT' "
        "AND expiration > ? ORDER BY expiration", (tk, entry_d)).fetchall()]
    ed = datetime.fromisoformat(entry_d)
    exp = next((e for e in exps if MIN_DTE <= (datetime.fromisoformat(e[:10]) - ed).days <= MAX_DTE), None)
    if not exp:
        return None
    strikes = [r[0] for r in c.execute(
        "SELECT DISTINCT strike FROM option_ohlc WHERE ticker=? AND right='PUT' AND expiration=?",
        (tk, exp)).fetchall()]
    if not strikes:
        return None
    atm = min(strikes, key=lambda s: abs(float(s) - underlying))
    bars = c.execute(
        "SELECT substr(timestamp,1,10) d, close FROM option_ohlc WHERE ticker=? AND right='PUT' "
        "AND expiration=? AND strike=? AND close>0 ORDER BY timestamp", (tk, exp, atm)).fetchall()
    daily = {}
    for d, cl in bars:
        if d >= entry_d:
            daily[d] = float(cl)
    path = sorted(daily.items())
    return path if len(path) >= 2 else None


def _sim(path, rule):
    e = path[0][1]
    if e <= 0:
        return None
    peak = e
    for i, (_, cl) in enumerate(path[1:], start=1):
        peak = max(peak, cl)
        ret = (cl / e - 1) * 100
        ddp = (cl / peak - 1) * 100
        if rule == "trail_40" and ddp <= -40:
            return ret, i
        if rule == "target_50_stop_50":
            if ret <= -50:
                return -50.0, i
            if ret >= 50:
                return 50.0, i
        if rule == "time_15d" and i >= 15:
            return ret, i
    return (path[-1][1] / e - 1) * 100, len(path) - 1


def main():
    c = sqlite3.connect(str(DB))
    have = [t for t in TICKERS if c.execute(
        "SELECT 1 FROM stock_ohlc WHERE ticker=? LIMIT 1", (t,)).fetchone()]
    print(f"tickers with data: {have}\n")
    trades = defaultdict(list)   # rule -> list of (tk, ret, hold)
    per_tkr = defaultdict(lambda: defaultdict(list))
    for tk in have:
        stock = _daily_stock(c, tk)
        if len(stock) < 25:
            continue
        closes = [v for _, v in stock]
        open_until = None
        for i in range(20, len(stock)):
            d, px = stock[i]
            if open_until and d <= open_until:
                continue
            sma20 = S.mean(closes[i - 20:i])
            downtrend = px < sma20 and px < closes[i - 5]
            if not downtrend:
                continue
            path = _put_path(c, tk, d, px)
            if not path:
                continue
            open_until = path[-1][0]  # block re-entry until this put's last bar (non-overlapping)
            for rule in ("trail_40", "target_50_stop_50", "time_15d"):
                r = _sim(path, rule)
                if r:
                    trades[rule].append((tk, r[0], r[1]))
                    per_tkr[rule][tk].append(r[0])
    c.close()

    def agg(rets):
        if not rets:
            return "n=0"
        w = [x for x in rets if x > 0]
        gl = abs(sum(x for x in rets if x <= 0))
        pf = sum(w) / gl if gl > 0 else float("inf")
        return f"n={len(rets):<4} WR={100*len(w)/len(rets):3.0f}%  mean={S.mean(rets):+6.1f}%  PF={pf:4.2f}"

    print("=" * 78)
    print("WEAK-STOCK DOWNTREND PUTS — does buying puts on falling names pay?")
    print("=" * 78)
    for rule in ("trail_40", "target_50_stop_50", "time_15d"):
        allr = [r for _, r, _ in trades[rule]]
        holds = [h for _, _, h in trades[rule]]
        print(f"\n── {rule}  (avg hold {S.mean(holds):.0f}d) ──")
        print(f"   ALL   {agg(allr)}")
        for tk in have:
            pr = per_tkr[rule].get(tk, [])
            if pr:
                print(f"   {tk:<6}{agg(pr)}")


if __name__ == "__main__":
    main()
