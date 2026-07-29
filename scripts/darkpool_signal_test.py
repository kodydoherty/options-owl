"""UW DARKPOOL directional-signal test (2026-07-10) — UW data opportunity #2.

We forward-collect darkpool prints but never trade them. Hypothesis: aggressive darkpool prints
(executed AT/ABOVE the ask = institutional accumulation, AT/BELOW the bid = distribution) predict
the underlying's direction, so we could buy a same-day option in that direction.

Design (no lookahead): for each ticker-day, sum premium-weighted direction of MORNING prints
(before ENTRY_ET), where dir=+1 if price>=nbbo_ask, -1 if price<=nbbo_bid, 0 if a mid-cross. If the
net signal is strong enough, simulate an option (call if net>0 else put) entered at ENTRY_ET,
nearest-DTE ATM, run through the SAME prod V7 exits + thetadata prices the flow backtest uses.
Flat-$ edge (BASE/trade). Held to the bar: the signal-direction trades must be net +EV, and the
sign must actually predict (bullish-signal days shouldn't lose).

Caveat: darkpool volume is CLOSE-heavy (most prints report at/after 4pm), so the morning signal is
thin — treat a positive result as directional, and a negative/flat result as "not exploitable intraday".

Usage: python scripts/darkpool_signal_test.py [--entry-et 12.0] [--min-net-prem 2000000]
"""
import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import uw_ticker_discovery as D  # noqa: E402
import flow_gold_standard_report as R  # noqa: E402
from options_owl.bot_runner import select_flow_strike  # noqa: E402
from options_owl.risk.exit_v5.config import INDEX_TICKERS  # noqa: E402

BASE = 750.0
DB = str(Path(__file__).resolve().parent.parent / "journal" / "uw_historical.db")
TICKERS = ["SPY", "QQQ", "TSLA", "NVDA", "META", "AAPL", "AMZN", "GOOGL", "MSFT", "AMD",
           "MSTR", "PLTR", "AVGO", "IWM"]


def load_signals(entry_et, min_net_prem):
    """Return [(ticker, date, entry_mi, direction, net_prem)] from morning darkpool prints."""
    conn = sqlite3.connect(DB)
    entry_min = int((entry_et - 9.5) * 60)  # minutes after 9:30 ET
    rows = conn.execute(
        "SELECT ticker, substr(executed_at,1,10) d, "
        "       (CAST(substr(executed_at,12,2) AS INT)-4)*60 + CAST(substr(executed_at,15,2) AS INT) - 570 mi, "
        "       price, premium, nbbo_bid, nbbo_ask "
        "FROM darkpool WHERE ticker IN (%s)" % ",".join("?" * len(TICKERS)),
        TICKERS,
    ).fetchall()
    conn.close()
    agg = defaultdict(float)
    for tk, d, mi, price, prem, bid, ask in rows:
        if mi is None or price is None or prem is None or not (0 <= mi < entry_min):
            continue  # only MORNING prints, before the entry cutoff (no lookahead)
        if ask and price >= ask:
            agg[(tk, d)] += prem       # aggressive buy (accumulation)
        elif bid and price <= bid:
            agg[(tk, d)] -= prem       # aggressive sell (distribution)
        # mid-cross prints carry no direction
    out = []
    for (tk, d), net in agg.items():
        if abs(net) >= min_net_prem:
            out.append((tk, d, entry_min, 1 if net > 0 else -1, net))
    return out


def sim_one(tk, d, entry_mi, is_put):
    """Simulate one option trade via the flow backtest's prod machinery. Returns ret_pct or None."""
    right = "PUT" if is_put else "CALL"
    otype = "put" if is_put else "call"
    stock, opts = D._stock(tk), D._opts(tk, right)
    if d not in stock:
        return None
    mb = (entry_mi // 5) * 5
    if mb not in stock[d]:
        return None
    spot = stock[d][mb]
    oday = opts[(opts["date"] == d) & (opts["mi"] == mb)]
    if oday.empty:
        return None
    dte0 = oday["dte"].min()
    same = oday[oday["dte"] == dte0]
    pseudo = [{"strike": float(r.strike), "mid": float(r.close)} for r in same.itertuples()]
    strike, _ = select_flow_strike(pseudo, spot, is_put, False, 2.0)
    if not strike:
        return None
    ch = opts[(opts["date"] == d) & (opts["strike"] == strike) & (opts["dte"] == dte0)]
    ch = ch[ch["mi"] >= mb].sort_values("mi")
    if len(ch) < 5:
        return None
    pp = ch["close"].values.astype(float)
    mp = ch["mi"].values.astype(int)
    up = [stock[d].get(int(m), spot) for m in mp]
    if np.isnan(pp[0]) or pp[0] <= 0:
        return None
    cfg = D.apply_v7_wide_trail_exits(
        D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
    ets = datetime(*map(int, d.split("-")), 9, 30) + timedelta(minutes=mb)
    ret, reason = R._sim_reason(pp, mp, up, pp[0], ets, cfg, int(dte0), otype)
    return ret


def pnl(rs):
    return sum(BASE * r / 100 for r in rs)


def pf(rs):
    g = sum(BASE * r / 100 for r in rs if r > 0); l = -sum(BASE * r / 100 for r in rs if r < 0)
    return g / l if l > 0 else float("inf")


def wr(rs):
    return (sum(1 for r in rs if r > 0) / len(rs) * 100) if rs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-et", type=float, default=12.0, help="ET hour to enter (morning signal cutoff)")
    ap.add_argument("--min-net-prem", type=float, default=2_000_000, help="min |net darkpool premium|")
    args = ap.parse_args()

    sigs = load_signals(args.entry_et, args.min_net_prem)
    print(f"Darkpool morning signals (entry {args.entry_et} ET, |net prem|>=${args.min_net_prem:,.0f}): "
          f"{len(sigs)} ticker-days\n")
    if not sigs:
        print("No signals — try a lower --min-net-prem.")
        return

    # SIGNAL-DIRECTION trades (buy the darkpool's direction) vs the CONTRARIAN (buy the opposite).
    signal_rets, contra_rets, bull_rets, bear_rets = [], [], [], []
    for tk, d, emi, direction, net in sigs:
        is_put_signal = direction < 0            # net distribution -> buy puts
        r = sim_one(tk, d, emi, is_put_signal)
        if r is None:
            continue
        signal_rets.append(r)
        (bull_rets if direction > 0 else bear_rets).append(r)
        rc = sim_one(tk, d, emi, not is_put_signal)  # opposite direction
        if rc is not None:
            contra_rets.append(rc)

    print(f"=== simulated {len(signal_rets)} of {len(sigs)} (rest lacked thetadata option prices) ===")
    print(f"  {'SIGNAL-direction (trade the print)':<36}{len(signal_rets):>4} tr  "
          f"${pnl(signal_rets):>+8,.0f}  PF {pf(signal_rets):>5.2f}  WR {wr(signal_rets):>3.0f}%")
    print(f"  {'CONTRARIAN (trade the opposite)':<36}{len(contra_rets):>4} tr  "
          f"${pnl(contra_rets):>+8,.0f}  PF {pf(contra_rets):>5.2f}  WR {wr(contra_rets):>3.0f}%")
    print(f"  {'  bullish-signal days (calls)':<36}{len(bull_rets):>4} tr  "
          f"${pnl(bull_rets):>+8,.0f}  PF {pf(bull_rets):>5.2f}  WR {wr(bull_rets):>3.0f}%")
    print(f"  {'  bearish-signal days (puts)':<36}{len(bear_rets):>4} tr  "
          f"${pnl(bear_rets):>+8,.0f}  PF {pf(bear_rets):>5.2f}  WR {wr(bear_rets):>3.0f}%")
    print("\nVERDICT: signal-direction must be clearly +EV AND beat the contrarian for darkpool to")
    print("be a usable intraday direction signal. If contrarian wins or both ~flat = not exploitable.")


if __name__ == "__main__":
    main()
