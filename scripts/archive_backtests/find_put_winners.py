#!/usr/bin/env python3
"""
Scan 3+ years of ThetaData for PUT trades that hit +25%+ profit.
Find patterns: time of day, day of week, stock conditions, ticker behavior.

Optimized: batch-loads all data per ticker+day to minimize DB queries.
"""

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "journal" / "thetadata_options.db"

# Entry time slots to scan (ET, as HH:MM)
TIME_SLOTS = [
    "09:30", "09:35", "09:40", "09:45", "09:50", "09:55",
    "10:00", "10:15", "10:30", "10:45",
    "11:00", "11:30",
    "12:00", "12:30",
    "13:00", "13:30",
    "14:00", "14:30",
]

PROFIT_THRESHOLDS = [15, 20, 25, 30, 35, 50, 75, 100]
MAX_HOLD = 120  # minutes

TICKERS = ["SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN", "GOOGL", "AMD", "MSTR", "PLTR", "AVGO", "MSFT", "IWM"]


def main():
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA cache_size = -500000")  # 500MB cache
    conn.execute("PRAGMA mmap_size = 2147483648")  # 2GB mmap

    # Get all trading days
    trading_days = [r[0] for r in conn.execute(
        "SELECT DISTINCT date(timestamp) FROM stock_ohlc WHERE ticker='SPY' ORDER BY date(timestamp)"
    ).fetchall()]
    total_days = len(trading_days)
    print(f"Total trading days: {total_days} ({trading_days[0]} to {trading_days[-1]})")
    print(f"Tickers: {len(TICKERS)}, Time slots: {len(TIME_SLOTS)}")
    print(f"Max hold: {MAX_HOLD}min")
    sys.stdout.flush()

    all_trades = []
    stats = {"total_scans": 0, "valid": 0, "no_stock": 0, "no_strike": 0, "no_bars": 0}

    for ticker_idx, ticker in enumerate(TICKERS):
        print(f"\n{'='*60}")
        print(f"[{ticker_idx+1}/{len(TICKERS)}] {ticker}")
        print(f"{'='*60}")
        sys.stdout.flush()

        ticker_trades = 0
        ticker_winners = 0

        # Get prior closes for gap calc (load all at once)
        prior_closes = {}
        for i, day in enumerate(trading_days):
            if i > 0:
                # Use previous day's last stock bar
                row = conn.execute(
                    "SELECT close FROM stock_ohlc WHERE ticker=? AND date(timestamp)=? ORDER BY timestamp DESC LIMIT 1",
                    (ticker, trading_days[i-1])
                ).fetchone()
                if row:
                    prior_closes[day] = row[0]

        for day_idx, date_str in enumerate(trading_days):
            if day_idx % 100 == 0:
                print(f"  {ticker} [{day_idx+1}/{total_days}] {date_str} — {ticker_trades}t, {ticker_winners}w")
                sys.stdout.flush()

            # Batch load: all stock bars for this day
            stock_rows = conn.execute(
                "SELECT time(timestamp), open, close FROM stock_ohlc WHERE ticker=? AND date(timestamp)=? ORDER BY timestamp",
                (ticker, date_str)
            ).fetchall()
            if not stock_rows:
                stats["no_stock"] += len(TIME_SLOTS)
                continue

            # Index stock by time
            stock_by_time = {}
            for t, o, c in stock_rows:
                # Normalize time (strip timezone, keep HH:MM:SS)
                t_clean = t[:8] if len(t) > 8 else t
                stock_by_time[t_clean[:5]] = (o, c)  # HH:MM -> (open, close)

            open_price = stock_rows[0][1]  # first bar open
            prior_close = prior_closes.get(date_str)
            gap_pct = ((open_price - prior_close) / prior_close * 100) if prior_close and prior_close > 0 else 0

            dow = datetime.strptime(date_str, "%Y-%m-%d")
            day_of_week = dow.strftime("%A")
            dow_num = dow.weekday()

            # Batch load: all PUT option bars for this day (0DTE only)
            opt_rows = conn.execute(
                """SELECT strike, time(timestamp), open, high, low, close, volume
                   FROM option_ohlc
                   WHERE ticker=? AND right='PUT' AND expiration=? AND date(timestamp)=?
                   ORDER BY strike, timestamp""",
                (ticker, date_str, date_str)
            ).fetchall()

            if not opt_rows:
                stats["no_strike"] += len(TIME_SLOTS)
                continue

            # Index option bars: strike -> [(time, o, h, l, c, vol), ...]
            opt_by_strike = defaultdict(list)
            available_strikes = set()
            for strike, t, o, h, l, c, vol in opt_rows:
                t_clean = t[:5]  # HH:MM
                opt_by_strike[strike].append((t_clean, o, h, l, c, vol))
                available_strikes.add(strike)

            strikes_list = sorted(available_strikes)
            if not strikes_list:
                continue

            for slot in TIME_SLOTS:
                stats["total_scans"] += 1

                # Get stock price at entry
                stock_data = stock_by_time.get(slot)
                if not stock_data:
                    stats["no_stock"] += 1
                    continue
                stock_price = stock_data[1]  # close
                if not stock_price or stock_price <= 0:
                    stock_price = stock_data[0]
                if not stock_price or stock_price <= 0:
                    stats["no_stock"] += 1
                    continue

                # Find ATM strike
                best_strike = min(strikes_list, key=lambda s: abs(s - stock_price))
                if abs(best_strike - stock_price) / stock_price > 0.02:
                    stats["no_strike"] += 1
                    continue  # No strike within 2% — skip

                # Get option bars from entry forward
                bars = opt_by_strike[best_strike]
                # Find index of entry slot
                entry_idx = None
                for i, (t, o, h, l, c, vol) in enumerate(bars):
                    if t == slot:
                        entry_idx = i
                        break

                if entry_idx is None or entry_idx >= len(bars) - 1:
                    stats["no_bars"] += 1
                    continue

                # Entry premium (close of entry bar + 3% slippage)
                entry_bar = bars[entry_idx]
                entry_premium = entry_bar[4]  # close
                if not entry_premium or entry_premium <= 0:
                    entry_premium = entry_bar[1]  # open
                if not entry_premium or entry_premium <= 0:
                    stats["no_bars"] += 1
                    continue

                entry_premium *= 1.03  # slippage

                # Simulate trade
                peak_pct = 0
                min_pct = 0
                hold_to_peak = 0
                thresholds_hit = {}

                for j in range(entry_idx + 1, min(entry_idx + 1 + MAX_HOLD, len(bars))):
                    t, o, h, l, c, vol = bars[j]
                    if not h or h <= 0:
                        continue

                    pct_high = (h - entry_premium) / entry_premium * 100
                    pct_low = (l - entry_premium) / entry_premium * 100 if l and l > 0 else 0

                    if pct_high > peak_pct:
                        peak_pct = pct_high
                        hold_to_peak = j - entry_idx

                    if pct_low < min_pct:
                        min_pct = pct_low

                    for thresh in PROFIT_THRESHOLDS:
                        if thresh not in thresholds_hit and pct_high >= thresh:
                            thresholds_hit[thresh] = j - entry_idx  # minutes

                # Compute pre-entry move
                pct_from_open = (stock_price - open_price) / open_price * 100 if open_price > 0 else 0

                pre_entry_move = 0
                if slot != "09:30" and "09:30" in stock_by_time:
                    open_close = stock_by_time["09:30"][1]
                    if open_close and open_close > 0:
                        pre_entry_move = (stock_price - open_close) / open_close * 100

                stats["valid"] += 1
                ticker_trades += 1
                if 25 in thresholds_hit:
                    ticker_winners += 1

                all_trades.append({
                    "ticker": ticker,
                    "date": date_str,
                    "entry_time": slot,
                    "strike": best_strike,
                    "stock_price": stock_price,
                    "entry_premium": entry_premium,
                    "peak_pct": peak_pct,
                    "min_pct": min_pct,
                    "hold_to_peak": hold_to_peak,
                    "thresholds_hit": thresholds_hit,
                    "pct_from_open": pct_from_open,
                    "gap_pct": gap_pct,
                    "pre_entry_move_pct": pre_entry_move,
                    "day_of_week": day_of_week,
                    "dow_num": dow_num,
                    "open_price": open_price,
                })

        print(f"  {ticker} DONE: {ticker_trades}t, {ticker_winners} hit +25% ({ticker_winners/max(ticker_trades,1)*100:.1f}%)")
        sys.stdout.flush()

    conn.close()

    # ============================================================
    # ANALYSIS
    # ============================================================
    print("\n" + "=" * 80)
    print("PUT WINNER PATTERN ANALYSIS — 3+ YEARS OF DATA")
    print("=" * 80)
    print(f"  Period: {trading_days[0]} to {trading_days[-1]} ({total_days} days)")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Scans: {stats['total_scans']:,}, Valid trades: {stats['valid']:,}")
    print(f"  Skipped: no_stock={stats['no_stock']:,}, no_strike={stats['no_strike']:,}, no_bars={stats['no_bars']:,}")

    if not all_trades:
        print("NO TRADES")
        return

    winners_25 = [t for t in all_trades if 25 in t["thresholds_hit"]]

    # === THRESHOLD HIT RATES ===
    print("\n" + "-" * 80)
    print("PROFIT THRESHOLD HIT RATES")
    print("-" * 80)
    print(f"  {'Threshold':>10} | {'Count':>8} | {'Rate':>7} | {'Avg Time':>10}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}")
    for thresh in PROFIT_THRESHOLDS:
        hits = [t for t in all_trades if thresh in t["thresholds_hit"]]
        if hits:
            avg_t = sum(t["thresholds_hit"][thresh] for t in hits) / len(hits)
            print(f"  {f'+{thresh}%':>10} | {len(hits):>8,} | {len(hits)/len(all_trades)*100:>6.1f}% | {avg_t:>9.1f}m")
        else:
            print(f"  {f'+{thresh}%':>10} | {0:>8} | {0:>6.1f}% | {'N/A':>10}")

    # === BY TIME OF DAY ===
    print("\n" + "-" * 80)
    print("BY TIME OF DAY (+25% hit rate)")
    print("-" * 80)
    by_time = defaultdict(lambda: {"n": 0, "w": 0, "peaks": []})
    for t in all_trades:
        by_time[t["entry_time"]]["n"] += 1
        by_time[t["entry_time"]]["peaks"].append(t["peak_pct"])
        if 25 in t["thresholds_hit"]:
            by_time[t["entry_time"]]["w"] += 1

    print(f"  {'Slot':>8} | {'N':>7} | {'+25%':>7} | {'Rate':>6} | {'AvgPeak':>8} | {'MedPeak':>8}")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
    for slot in TIME_SLOTS:
        d = by_time[slot]
        if d["n"]:
            peaks = sorted(d["peaks"])
            med = peaks[len(peaks)//2]
            print(f"  {slot:>8} | {d['n']:>7,} | {d['w']:>7,} | {d['w']/d['n']*100:>5.1f}% | {sum(d['peaks'])/d['n']:>7.1f}% | {med:>7.1f}%")

    # === BY TICKER ===
    print("\n" + "-" * 80)
    print("BY TICKER (+25% hit rate)")
    print("-" * 80)
    by_tk = defaultdict(lambda: {"n": 0, "w": 0, "peaks": [], "times": []})
    for t in all_trades:
        by_tk[t["ticker"]]["n"] += 1
        by_tk[t["ticker"]]["peaks"].append(t["peak_pct"])
        if 25 in t["thresholds_hit"]:
            by_tk[t["ticker"]]["w"] += 1
            by_tk[t["ticker"]]["times"].append(t["thresholds_hit"][25])

    print(f"  {'Ticker':>6} | {'N':>7} | {'+25%':>6} | {'Rate':>6} | {'AvgPeak':>8} | {'AvgTime':>8}")
    print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
    for tk in sorted(by_tk.keys(), key=lambda x: by_tk[x]["w"]/max(by_tk[x]["n"],1), reverse=True):
        d = by_tk[tk]
        at = sum(d["times"])/len(d["times"]) if d["times"] else 0
        print(f"  {tk:>6} | {d['n']:>7,} | {d['w']:>6,} | {d['w']/d['n']*100:>5.1f}% | {sum(d['peaks'])/d['n']:>7.1f}% | {at:>7.1f}m")

    # === BY DAY OF WEEK ===
    print("\n" + "-" * 80)
    print("BY DAY OF WEEK")
    print("-" * 80)
    by_dow = defaultdict(lambda: {"n": 0, "w": 0})
    for t in all_trades:
        by_dow[t["day_of_week"]]["n"] += 1
        if 25 in t["thresholds_hit"]:
            by_dow[t["day_of_week"]]["w"] += 1
    for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        d = by_dow[dow]
        if d["n"]:
            print(f"  {dow:>12}: {d['w']:>6,} / {d['n']:>7,} = {d['w']/d['n']*100:.1f}%")

    # === BY STOCK DIRECTION FROM OPEN ===
    print("\n" + "-" * 80)
    print("BY STOCK MOVE FROM OPEN (the key directional signal)")
    print("-" * 80)
    buckets = [
        ("Down >2%", -999, -2.0),
        ("Down 1.5-2%", -2.0, -1.5),
        ("Down 1.0-1.5%", -1.5, -1.0),
        ("Down 0.5-1.0%", -1.0, -0.5),
        ("Down 0.25-0.5%", -0.5, -0.25),
        ("Down 0-0.25%", -0.25, 0),
        ("Up 0-0.25%", 0, 0.25),
        ("Up 0.25-0.5%", 0.25, 0.5),
        ("Up 0.5-1.0%", 0.5, 1.0),
        ("Up 1.0-1.5%", 1.0, 1.5),
        ("Up 1.5-2%", 1.5, 2.0),
        ("Up >2%", 2.0, 999),
    ]
    print(f"  {'Direction':>18} | {'N':>7} | {'+25%':>7} | {'Rate':>6} | {'AvgPeak':>8}")
    print(f"  {'-'*18}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}")
    for label, lo, hi in buckets:
        trades = [t for t in all_trades if lo <= t["pct_from_open"] < hi]
        wins = [t for t in trades if 25 in t["thresholds_hit"]]
        if trades:
            ap = sum(t["peak_pct"] for t in trades) / len(trades)
            print(f"  {label:>18} | {len(trades):>7,} | {len(wins):>7,} | {len(wins)/len(trades)*100:>5.1f}% | {ap:>7.1f}%")

    # === BY GAP ===
    print("\n" + "-" * 80)
    print("BY OVERNIGHT GAP")
    print("-" * 80)
    gap_buckets = [
        ("Gap Down >1.5%", -999, -1.5),
        ("Gap Down 1-1.5%", -1.5, -1.0),
        ("Gap Down 0.5-1%", -1.0, -0.5),
        ("Gap Down 0-0.5%", -0.5, 0),
        ("Gap Up 0-0.5%", 0, 0.5),
        ("Gap Up 0.5-1%", 0.5, 1.0),
        ("Gap Up 1-1.5%", 1.0, 1.5),
        ("Gap Up >1.5%", 1.5, 999),
    ]
    print(f"  {'Gap':>18} | {'N':>7} | {'+25%':>7} | {'Rate':>6} | {'AvgPeak':>8}")
    print(f"  {'-'*18}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}")
    for label, lo, hi in gap_buckets:
        trades = [t for t in all_trades if lo <= t["gap_pct"] < hi]
        wins = [t for t in trades if 25 in t["thresholds_hit"]]
        if trades:
            ap = sum(t["peak_pct"] for t in trades) / len(trades)
            print(f"  {label:>18} | {len(trades):>7,} | {len(wins):>7,} | {len(wins)/len(trades)*100:>5.1f}% | {ap:>7.1f}%")

    # === BY PRE-ENTRY MOMENTUM ===
    print("\n" + "-" * 80)
    print("BY PRE-ENTRY MOMENTUM (stock move from open to entry, excludes 9:30)")
    print("-" * 80)
    mom_buckets = [
        ("Dumping >1.5%", -999, -1.5),
        ("Dropping 1-1.5%", -1.5, -1.0),
        ("Down 0.5-1%", -1.0, -0.5),
        ("Down 0.25-0.5%", -0.5, -0.25),
        ("Flat", -0.25, 0.25),
        ("Up 0.25-0.5%", 0.25, 0.5),
        ("Up 0.5-1%", 0.5, 1.0),
        ("Rallying 1-1.5%", 1.0, 1.5),
        ("Ripping >1.5%", 1.5, 999),
    ]
    after_open = [t for t in all_trades if t["entry_time"] != "09:30"]
    print(f"  {'Momentum':>22} | {'N':>7} | {'+25%':>7} | {'Rate':>6} | {'AvgPeak':>8}")
    print(f"  {'-'*22}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}")
    for label, lo, hi in mom_buckets:
        trades = [t for t in after_open if lo <= t["pre_entry_move_pct"] < hi]
        wins = [t for t in trades if 25 in t["thresholds_hit"]]
        if trades:
            ap = sum(t["peak_pct"] for t in trades) / len(trades)
            print(f"  {label:>22} | {len(trades):>7,} | {len(wins):>7,} | {len(wins)/len(trades)*100:>5.1f}% | {ap:>7.1f}%")

    # === CROSS: Time × Direction ===
    print("\n" + "-" * 80)
    print("TOP COMBOS: Time × Direction (min 50 trades, by +25% rate)")
    print("-" * 80)
    combos = defaultdict(lambda: {"n": 0, "w": 0, "peaks": []})
    for t in all_trades:
        pfo = t["pct_from_open"]
        d = "DOWN>0.5%" if pfo <= -0.5 else "DOWN<0.5%" if pfo <= 0 else "UP<0.5%" if pfo <= 0.5 else "UP>0.5%"
        k = f"{t['entry_time']} × {d}"
        combos[k]["n"] += 1
        combos[k]["peaks"].append(t["peak_pct"])
        if 25 in t["thresholds_hit"]:
            combos[k]["w"] += 1

    sc = sorted([(k,v) for k,v in combos.items() if v["n"]>=50], key=lambda x: x[1]["w"]/x[1]["n"], reverse=True)
    print(f"  {'Combo':>28} | {'N':>6} | {'+25%':>5} | {'Rate':>6} | {'AvgPeak':>8}")
    print(f"  {'-'*28}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*8}")
    for k,v in sc[:20]:
        print(f"  {k:>28} | {v['n']:>6,} | {v['w']:>5,} | {v['w']/v['n']*100:>5.1f}% | {sum(v['peaks'])/v['n']:>7.1f}%")

    # === CROSS: Ticker × Direction ===
    print("\n" + "-" * 80)
    print("TOP COMBOS: Ticker × Direction (min 50 trades)")
    print("-" * 80)
    combos2 = defaultdict(lambda: {"n": 0, "w": 0, "peaks": []})
    for t in all_trades:
        pfo = t["pct_from_open"]
        d = "DOWN>0.5%" if pfo <= -0.5 else "DOWN<0.5%" if pfo <= 0 else "UP<0.5%" if pfo <= 0.5 else "UP>0.5%"
        k = f"{t['ticker']:>6} × {d}"
        combos2[k]["n"] += 1
        combos2[k]["peaks"].append(t["peak_pct"])
        if 25 in t["thresholds_hit"]:
            combos2[k]["w"] += 1

    sc2 = sorted([(k,v) for k,v in combos2.items() if v["n"]>=50], key=lambda x: x[1]["w"]/x[1]["n"], reverse=True)
    print(f"  {'Combo':>22} | {'N':>6} | {'+25%':>5} | {'Rate':>6} | {'AvgPeak':>8}")
    print(f"  {'-'*22}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*8}")
    for k,v in sc2[:20]:
        print(f"  {k:>22} | {v['n']:>6,} | {v['w']:>5,} | {v['w']/v['n']*100:>5.1f}% | {sum(v['peaks'])/v['n']:>7.1f}%")

    # === TRIPLE: Ticker × Time × Direction ===
    print("\n" + "-" * 80)
    print("TOP 30 TRIPLE: Ticker × Time × Direction (min 15 trades)")
    print("-" * 80)
    combos3 = defaultdict(lambda: {"n": 0, "w": 0, "peaks": [], "times": []})
    for t in all_trades:
        pfo = t["pct_from_open"]
        d = "DOWN>0.5%" if pfo <= -0.5 else "DOWN<0.5%" if pfo <= 0 else "UP<0.5%" if pfo <= 0.5 else "UP>0.5%"
        k = f"{t['ticker']:>6} {t['entry_time']} {d}"
        combos3[k]["n"] += 1
        combos3[k]["peaks"].append(t["peak_pct"])
        if 25 in t["thresholds_hit"]:
            combos3[k]["w"] += 1
            combos3[k]["times"].append(t["thresholds_hit"][25])

    sc3 = sorted([(k,v) for k,v in combos3.items() if v["n"]>=15], key=lambda x: x[1]["w"]/x[1]["n"], reverse=True)
    print(f"  {'Combo':>32} | {'N':>4} | {'+25%':>4} | {'Rate':>6} | {'AvgPeak':>8} | {'AvgT':>6}")
    print(f"  {'-'*32}-+-{'-'*4}-+-{'-'*4}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}")
    for k,v in sc3[:30]:
        at = sum(v["times"])/len(v["times"]) if v["times"] else 0
        print(f"  {k:>32} | {v['n']:>4} | {v['w']:>4} | {v['w']/v['n']*100:>5.1f}% | {sum(v['peaks'])/v['n']:>7.1f}% | {at:>5.1f}m")

    # === WINNER PROFILE ===
    print("\n" + "=" * 80)
    print(f"WINNER PROFILE — {len(winners_25):,} trades that hit +25%")
    print("=" * 80)

    if winners_25:
        print(f"  Overall rate: {len(winners_25)/len(all_trades)*100:.1f}%")
        print(f"  Avg time to +25%: {sum(t['thresholds_hit'][25] for t in winners_25)/len(winners_25):.1f}m")
        print(f"  Avg peak: {sum(t['peak_pct'] for t in winners_25)/len(winners_25):.1f}%")

        # Direction
        down_big = len([t for t in winners_25 if t["pct_from_open"] <= -0.5])
        down_sm = len([t for t in winners_25 if -0.5 < t["pct_from_open"] <= 0])
        up_sm = len([t for t in winners_25 if 0 < t["pct_from_open"] <= 0.5])
        up_big = len([t for t in winners_25 if t["pct_from_open"] > 0.5])
        n = len(winners_25)
        print(f"\n  Stock direction at entry:")
        print(f"    Down >0.5%:  {down_big:>6} ({down_big/n*100:.1f}%)")
        print(f"    Down <0.5%:  {down_sm:>6} ({down_sm/n*100:.1f}%)")
        print(f"    Up <0.5%:    {up_sm:>6} ({up_sm/n*100:.1f}%)")
        print(f"    Up >0.5%:    {up_big:>6} ({up_big/n*100:.1f}%)")

        # Compare to losers
        losers = [t for t in all_trades if 25 not in t["thresholds_hit"]]
        l_down_big = len([t for t in losers if t["pct_from_open"] <= -0.5])
        l_down_sm = len([t for t in losers if -0.5 < t["pct_from_open"] <= 0])
        l_up_sm = len([t for t in losers if 0 < t["pct_from_open"] <= 0.5])
        l_up_big = len([t for t in losers if t["pct_from_open"] > 0.5])
        ln = len(losers) if losers else 1
        print(f"\n  Direction: Winners vs Non-winners")
        print(f"    {'':>15} {'Winners':>10} {'Non-win':>10} {'Lift':>8}")
        print(f"    {'Down >0.5%':>15} {down_big/n*100:>9.1f}% {l_down_big/ln*100:>9.1f}% {(down_big/n)/(l_down_big/ln+0.001)*100-100:>+7.0f}%")
        print(f"    {'Down <0.5%':>15} {down_sm/n*100:>9.1f}% {l_down_sm/ln*100:>9.1f}% {(down_sm/n)/(l_down_sm/ln+0.001)*100-100:>+7.0f}%")
        print(f"    {'Up <0.5%':>15} {up_sm/n*100:>9.1f}% {l_up_sm/ln*100:>9.1f}% {(up_sm/n)/(l_up_sm/ln+0.001)*100-100:>+7.0f}%")
        print(f"    {'Up >0.5%':>15} {up_big/n*100:>9.1f}% {l_up_big/ln*100:>9.1f}% {(up_big/n)/(l_up_big/ln+0.001)*100-100:>+7.0f}%")

        # Gap
        gd = len([t for t in winners_25 if t["gap_pct"] < -0.5])
        gf = len([t for t in winners_25 if -0.5 <= t["gap_pct"] <= 0.5])
        gu = len([t for t in winners_25 if t["gap_pct"] > 0.5])
        print(f"\n  Gap: Down={gd} ({gd/n*100:.1f}%), Flat={gf} ({gf/n*100:.1f}%), Up={gu} ({gu/n*100:.1f}%)")

        # DOW
        print(f"  Day of week:")
        for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            c = len([t for t in winners_25 if t["day_of_week"] == dow])
            print(f"    {dow:>12}: {c:>5} ({c/n*100:.1f}%)")

    # === PREMIUM RANGE ===
    print("\n" + "-" * 80)
    print("BY ENTRY PREMIUM")
    print("-" * 80)
    prem_bk = [
        ("$0.05-0.20", 0.05, 0.20),
        ("$0.20-0.50", 0.20, 0.50),
        ("$0.50-1.00", 0.50, 1.00),
        ("$1.00-2.00", 1.00, 2.00),
        ("$2.00-4.00", 2.00, 4.00),
        ("$4.00-8.00", 4.00, 8.00),
        ("$8.00+", 8.00, 999),
    ]
    print(f"  {'Premium':>12} | {'N':>7} | {'+25%':>7} | {'Rate':>6} | {'AvgPeak':>8}")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}")
    for label, lo, hi in prem_bk:
        trades = [t for t in all_trades if lo <= t["entry_premium"] < hi]
        wins = [t for t in trades if 25 in t["thresholds_hit"]]
        if trades:
            print(f"  {label:>12} | {len(trades):>7,} | {len(wins):>7,} | {len(wins)/len(trades)*100:>5.1f}% | {sum(t['peak_pct'] for t in trades)/len(trades):>7.1f}%")

    # === YEARLY BREAKDOWN ===
    print("\n" + "-" * 80)
    print("BY YEAR (market regime matters)")
    print("-" * 80)
    by_year = defaultdict(lambda: {"n": 0, "w": 0, "peaks": []})
    for t in all_trades:
        yr = t["date"][:4]
        by_year[yr]["n"] += 1
        by_year[yr]["peaks"].append(t["peak_pct"])
        if 25 in t["thresholds_hit"]:
            by_year[yr]["w"] += 1
    for yr in sorted(by_year.keys()):
        d = by_year[yr]
        print(f"  {yr}: {d['w']:>6,} / {d['n']:>7,} = {d['w']/d['n']*100:.1f}% hit +25%, avg peak {sum(d['peaks'])/d['n']:.1f}%")

    # === MONTHLY BREAKDOWN ===
    print("\n" + "-" * 80)
    print("BY MONTH (seasonal patterns)")
    print("-" * 80)
    by_month = defaultdict(lambda: {"n": 0, "w": 0})
    for t in all_trades:
        m = t["date"][5:7]
        by_month[m]["n"] += 1
        if 25 in t["thresholds_hit"]:
            by_month[m]["w"] += 1
    months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m, name in zip(months, mnames):
        d = by_month[m]
        if d["n"]:
            print(f"  {name}: {d['w']:>5,} / {d['n']:>6,} = {d['w']/d['n']*100:.1f}%")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
