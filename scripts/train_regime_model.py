"""Train an ML regime classifier (BULLISH / BEARISH / CHOPPY).

Uses SPY 1-min candles aggregated to 5-min bars. Produces a LightGBM model
that predicts intraday regime at multiple checkpoints throughout the day.

Strategy: Use FEWER features focused on things that actually predict direction,
regularize heavily, and validate with strict walk-forward splits.

Usage:
    python scripts/train_regime_model.py
    python scripts/train_regime_model.py --threshold 0.60
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_owl.risk.regime_detector import _adx, _ema, _rsi

DB_PATH = Path(__file__).resolve().parent.parent / "journal" / "thetadata_options.db"
MODEL_DIR = Path(__file__).resolve().parent.parent / "journal" / "models" / "ml_v3"

LABEL_BULLISH = 0
LABEL_BEARISH = 1
LABEL_CHOPPY = 2
LABEL_NAMES = {0: "BULLISH", 1: "BEARISH", 2: "CHOPPY"}

CHECKPOINTS = [15, 30, 60, 90, 120, 180]


def aggregate_5m(rows: list) -> list[dict]:
    candles = []
    batch = []
    for r in rows:
        batch.append(r)
        if len(batch) == 5:
            candles.append({
                "open": float(batch[0][1]),
                "high": max(float(b[2]) for b in batch),
                "low": min(float(b[3]) for b in batch),
                "close": float(batch[-1][4]),
                "volume": sum(float(b[5] or 0) for b in batch),
                "vwap": float(batch[-1][6] or 0),
            })
            batch = []
    return candles


def compute_features(
    today: list[dict],
    prev_days: list[list[dict]],
    checkpoint_bars: int,
) -> dict[str, float] | None:
    """Compute features. prev_days is list of prior days' candles (newest last)."""
    if checkpoint_bars > len(today) or checkpoint_bars < 1:
        return None

    current = today[:checkpoint_bars]
    day_open = today[0]["open"]
    if day_open <= 0:
        return None

    f = {}

    # --- Intraday features (what's happening right now) ---
    close = current[-1]["close"]
    hi = max(c["high"] for c in current)
    lo = min(c["low"] for c in current)

    f["return_from_open"] = ((close - day_open) / day_open) * 100
    f["intraday_range"] = ((hi - lo) / day_open) * 100

    # Current price position in today's range (0=low, 1=high)
    f["price_position"] = (close - lo) / (hi - lo) if hi > lo else 0.5

    # Opening drive
    f["opening_drive"] = ((current[0]["close"] - current[0]["open"]) / day_open) * 100

    # Green bar ratio
    green = sum(1 for c in current if c["close"] >= c["open"])
    f["green_ratio"] = green / len(current)

    # Price vs VWAP
    vwap = current[-1]["vwap"]
    f["vs_vwap"] = ((close - vwap) / vwap) * 100 if vwap > 0 else 0

    # Max drawdown and runup from open
    running_high = day_open
    max_dd = 0.0
    for c in current:
        running_high = max(running_high, c["high"])
        dd = (c["low"] - running_high) / running_high * 100
        max_dd = min(max_dd, dd)
    f["max_drawdown"] = max_dd

    running_low = day_open
    max_ru = 0.0
    for c in current:
        running_low = min(running_low, c["low"])
        ru = (c["high"] - running_low) / running_low * 100
        max_ru = max(max_ru, ru)
    f["max_runup"] = max_ru

    # Momentum: trend in last 3 bars
    if len(current) >= 3:
        f["recent_momentum"] = ((current[-1]["close"] - current[-3]["open"]) / current[-3]["open"]) * 100
    else:
        f["recent_momentum"] = f["return_from_open"]

    # Technical indicators (use prior-day lookback for warmup)
    lookback = []
    for pd in prev_days[-2:]:  # last 2 prior days
        lookback.extend(pd[-15:])  # last 15 bars each
    combined = lookback + current
    if len(combined) >= 22:
        closes = [c["close"] for c in combined]
        highs = [c["high"] for c in combined]
        lows = [c["low"] for c in combined]
        ema9 = _ema(closes, 9)
        ema21 = _ema(closes, 21)
        f["ema_spread"] = ((ema9 - ema21) / ema21) * 100 if ema21 > 0 else 0
        f["rsi"] = _rsi(closes, 14)
        f["adx"] = _adx(highs, lows, closes, 14)
    else:
        f["ema_spread"] = 0
        f["rsi"] = 50
        f["adx"] = 0

    # --- Context features (multi-day) ---
    if prev_days:
        prev = prev_days[-1]
        prev_open = prev[0]["open"]
        prev_close = prev[-1]["close"]
        f["gap_pct"] = ((day_open - prev_close) / prev_close) * 100 if prev_close > 0 else 0
        f["prev_return"] = ((prev_close - prev_open) / prev_open) * 100 if prev_open > 0 else 0

        # 3-day trend
        if len(prev_days) >= 3:
            d3_open = prev_days[-3][0]["open"]
            d3_close = prev_days[-1][-1]["close"]
            f["trend_3d"] = ((d3_close - d3_open) / d3_open) * 100 if d3_open > 0 else 0
        else:
            f["trend_3d"] = f["prev_return"]

        # 5-day trend
        if len(prev_days) >= 5:
            d5_open = prev_days[-5][0]["open"]
            d5_close = prev_days[-1][-1]["close"]
            f["trend_5d"] = ((d5_close - d5_open) / d5_open) * 100 if d5_open > 0 else 0
        else:
            f["trend_5d"] = f.get("trend_3d", 0)

        # Recent volatility (avg daily range over last 5 days)
        ranges = []
        for pd in prev_days[-5:]:
            ph = max(c["high"] for c in pd)
            pl = min(c["low"] for c in pd)
            po = pd[0]["open"]
            if po > 0:
                ranges.append((ph - pl) / po * 100)
        f["recent_vol"] = sum(ranges) / len(ranges) if ranges else 0
    else:
        f["gap_pct"] = 0
        f["prev_return"] = 0
        f["trend_3d"] = 0
        f["trend_5d"] = 0
        f["recent_vol"] = 0

    # Session progress (so model knows how much data it has)
    f["session_pct"] = checkpoint_bars / 78

    return f


def load_all_days(db_path: str) -> dict[str, list[dict]]:
    conn = sqlite3.connect(db_path)
    days = [r[0] for r in conn.execute(
        "SELECT DISTINCT date(timestamp) as d FROM stock_ohlc WHERE ticker='SPY' ORDER BY d"
    ).fetchall()]

    all_candles = {}
    for d in days:
        rows = conn.execute(
            "SELECT timestamp, open, high, low, close, volume, vwap "
            "FROM stock_ohlc WHERE ticker='SPY' AND date(timestamp)=? ORDER BY timestamp",
            (d,),
        ).fetchall()
        candles = aggregate_5m(rows)
        if len(candles) >= 10:
            all_candles[d] = candles
    conn.close()
    return all_candles


def label_day(candles: list[dict], threshold: float = 0.3) -> int:
    ret = ((candles[-1]["close"] - candles[0]["open"]) / candles[0]["open"]) * 100
    if ret > threshold:
        return LABEL_BULLISH
    elif ret < -threshold:
        return LABEL_BEARISH
    return LABEL_CHOPPY


def build_dataset(all_candles: dict[str, list[dict]]):
    days = sorted(all_candles.keys())
    feature_names = None
    X_rows, y_rows, dates_out, cp_out = [], [], [], []

    for i, date_str in enumerate(days):
        today = all_candles[date_str]
        prev_days = [all_candles[days[j]] for j in range(max(0, i - 5), i)]
        label = label_day(today)

        for cp_min in CHECKPOINTS:
            cp_bars = cp_min // 5
            feats = compute_features(today, prev_days, cp_bars)
            if feats is None:
                continue
            if feature_names is None:
                feature_names = sorted(feats.keys())
            X_rows.append([feats.get(f, 0.0) for f in feature_names])
            y_rows.append(label)
            dates_out.append(date_str)
            cp_out.append(cp_min)

    return np.array(X_rows, np.float32), np.array(y_rows, np.int32), feature_names, dates_out, cp_out


def train_model(X, y, feat_names):
    import lightgbm as lgb

    train_data = lgb.Dataset(X, label=y, feature_name=feat_names)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 15,       # much smaller trees
        "learning_rate": 0.03,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 50,  # more regularization
        "lambda_l1": 1.0,        # strong L1
        "lambda_l2": 1.0,        # strong L2
        "max_depth": 4,           # shallow trees
        "verbose": -1,
        "seed": 42,
    }
    return lgb.train(params, train_data, num_boost_round=200,
                     valid_sets=[train_data], callbacks=[lgb.log_evaluation(100)])


def evaluate(model, X, y, dates, cps, feat_names, threshold, all_candles=None, label=""):
    probs = model.predict(X)
    preds = np.argmax(probs, axis=1)
    max_conf = np.max(probs, axis=1)
    thresholded = np.where(max_conf >= threshold, preds, LABEL_CHOPPY)

    cp_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "choppy": 0, "total": 0,
                                     "false_bear": 0, "bear_pred": 0,
                                     "false_bull": 0, "bull_pred": 0})

    for i in range(len(y)):
        cp = cps[i]
        s = cp_stats[cp]
        s["total"] += 1
        pred = thresholded[i]
        actual = y[i]
        if pred == LABEL_CHOPPY:
            s["choppy"] += 1
        elif pred == actual:
            s["correct"] += 1
        else:
            s["wrong"] += 1
        if pred == LABEL_BEARISH:
            s["bear_pred"] += 1
            if actual != LABEL_BEARISH:
                s["false_bear"] += 1
        if pred == LABEL_BULLISH:
            s["bull_pred"] += 1
            if actual != LABEL_BULLISH:
                s["false_bull"] += 1

    print(f"\n{'='*85}")
    print(f"{label} (threshold={threshold})")
    print(f"{'='*85}")
    print(f"{'CP':<8} {'Total':>6} {'Corr':>6} {'Wrong':>6} {'Chop':>6} "
          f"{'DirAcc':>8} {'FBear':>8} {'FBull':>8}")
    print("-" * 72)
    for cp in sorted(cp_stats.keys()):
        s = cp_stats[cp]
        d = s["correct"] + s["wrong"]
        da = s["correct"] / d * 100 if d > 0 else 0
        fb = s["false_bear"] / s["bear_pred"] * 100 if s["bear_pred"] > 0 else 0
        fbu = s["false_bull"] / s["bull_pred"] * 100 if s["bull_pred"] > 0 else 0
        print(f"{cp}min".ljust(8) + f"{s['total']:>6} {s['correct']:>6} {s['wrong']:>6} "
              f"{s['choppy']:>6} {da:>7.1f}% {fb:>7.1f}% {fbu:>7.1f}%")

    # Feature importance
    imp = model.feature_importance(importance_type="gain")
    top = sorted(zip(feat_names, imp), key=lambda x: -x[1])[:10]
    print(f"\nTop features: {', '.join(f'{n}({v:.0f})' for n, v in top)}")

    # Big move days
    if all_candles:
        print(f"\n--- Big Move Days ---")
        seen = set()
        for i in range(len(y)):
            d = dates[i]
            if d in seen:
                continue
            candles = all_candles.get(d)
            if not candles:
                continue
            ret = ((candles[-1]["close"] - candles[0]["open"]) / candles[0]["open"]) * 100
            if abs(ret) < 1.0:
                continue
            seen.add(d)
            # Collect predictions for this day across checkpoints
            day_preds = []
            for j in range(len(y)):
                if dates[j] == d:
                    day_preds.append((cps[j], LABEL_NAMES[thresholded[j]][:4],
                                     f"{max_conf[j]:.2f}"))
            pred_str = " ".join(f"{cp}m:{p}" for cp, p, _ in sorted(day_preds)[:4])
            actual_str = LABEL_NAMES[y[i]][:4]
            print(f"  {d} {ret:>+6.2f}% [{actual_str}] → {pred_str}")

    return cp_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    args = parser.parse_args()

    print(f"Loading SPY data...")
    all_candles = load_all_days(args.db)
    days = sorted(all_candles.keys())
    print(f"{len(all_candles)} days ({days[0]} to {days[-1]})")

    labels = [label_day(all_candles[d]) for d in days]
    bull = labels.count(LABEL_BULLISH)
    bear = labels.count(LABEL_BEARISH)
    chop = labels.count(LABEL_CHOPPY)
    print(f"BULL={bull} ({bull/len(labels)*100:.0f}%) BEAR={bear} ({bear/len(labels)*100:.0f}%) "
          f"CHOP={chop} ({chop/len(labels)*100:.0f}%)")

    # Walk-forward: 5 folds
    fold_size = len(days) // 6
    print(f"\n{'='*85}")
    print(f"WALK-FORWARD VALIDATION (5 folds, {fold_size} days each)")
    print(f"{'='*85}")

    wf_agg = defaultdict(lambda: {"correct": 0, "wrong": 0, "choppy": 0})

    for fold in range(5):
        train_end = (fold + 1) * fold_size
        test_end = min(train_end + fold_size, len(days))
        train_set = {d: all_candles[d] for d in days[:train_end]}
        test_set = {d: all_candles[d] for d in days[train_end:test_end]}

        X_tr, y_tr, fn, _, _ = build_dataset(train_set)
        X_te, y_te, _, te_dates, te_cps = build_dataset(test_set)

        if len(X_tr) < 100 or len(X_te) < 20:
            continue

        m = train_model(X_tr, y_tr, fn)
        probs = m.predict(X_te)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        thresholded = np.where(confs >= args.threshold, preds, LABEL_CHOPPY)

        c = w = ch = 0
        for i in range(len(y_te)):
            cp = te_cps[i]
            p = thresholded[i]
            if p == LABEL_CHOPPY:
                ch += 1
                wf_agg[cp]["choppy"] += 1
            elif p == y_te[i]:
                c += 1
                wf_agg[cp]["correct"] += 1
            else:
                w += 1
                wf_agg[cp]["wrong"] += 1

        d = c + w
        print(f"  Fold {fold+1}: train={len(train_set)}d test={len(test_set)}d "
              f"dir_acc={c/d*100:.1f}% ({c}/{d}) choppy={ch}")

    print(f"\n{'CP':<8} {'DirAcc':>10}")
    print("-" * 20)
    for cp in sorted(wf_agg.keys()):
        s = wf_agg[cp]
        d = s["correct"] + s["wrong"]
        print(f"{cp}min".ljust(8) + f"{s['correct']/d*100:.1f}%" if d > 0 else "n/a")

    # Train final model: 80/20 split
    split = int(len(days) * 0.8)
    train_set = {d: all_candles[d] for d in days[:split]}
    test_set = {d: all_candles[d] for d in days[split:]}

    X_tr, y_tr, fn, _, _ = build_dataset(train_set)
    X_te, y_te, _, te_d, te_c = build_dataset(test_set)

    final_model = train_model(X_tr, y_tr, fn)
    evaluate(final_model, X_te, y_te, te_d, te_c, fn, args.threshold, all_candles,
             f"HOLDOUT ({len(test_set)} days, {days[split]} to {days[-1]})")

    # Compare with rule-based at same checkpoints
    print(f"\n{'='*85}")
    print("COMPARISON: ML vs Rule-Based")
    print(f"{'='*85}")

    from options_owl.risk.regime_detector import RegimeState
    rule_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "choppy": 0})

    for d in days[split:]:
        today = all_candles[d]
        prev = [all_candles[days[j]] for j in range(max(0, days.index(d) - 2), days.index(d))]
        label = label_day(today)
        actual = "UP" if label == 0 else ("DOWN" if label == 1 else "FLAT")

        for cp_min in CHECKPOINTS:
            cp_bars = cp_min // 5
            if cp_bars > len(today):
                continue
            lookback = []
            for pd in prev[-2:]:
                lookback.extend(pd[-15:])
            combined = lookback + today[:cp_bars]
            if len(combined) < 22:
                continue

            closes = [c["close"] for c in combined]
            highs = [c["high"] for c in combined]
            lows = [c["low"] for c in combined]

            adx = _adx(highs, lows, closes, 14)
            if adx < 20:
                rule_stats[cp_min]["choppy"] += 1
                continue

            price = closes[-1]
            vwap = combined[-1]["vwap"]
            ema9 = _ema(closes, 9)
            ema21 = _ema(closes, 21)
            rsi = _rsi(closes, 14)

            b = s = 0
            if price > 0 and vwap > 0:
                if price > vwap: b += 1
                else: s += 1
            if ema9 > ema21: b += 1
            else: s += 1
            if rsi > 50: b += 1
            elif rsi < 50: s += 1

            if b >= 2:
                regime = "BULL"
            elif s >= 2:
                regime = "BEAR"
            else:
                regime = "CHOP"

            if regime == "CHOP":
                rule_stats[cp_min]["choppy"] += 1
            elif (regime == "BULL" and actual == "UP") or (regime == "BEAR" and actual == "DOWN"):
                rule_stats[cp_min]["correct"] += 1
            elif actual == "FLAT":
                rule_stats[cp_min]["choppy"] += 1
            else:
                rule_stats[cp_min]["wrong"] += 1

    print(f"\n{'CP':<8} {'ML DirAcc':>12} {'Rule DirAcc':>14}")
    print("-" * 38)

    # Get ML stats from the evaluate call
    probs = final_model.predict(X_te)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    thr = np.where(confs >= args.threshold, preds, LABEL_CHOPPY)
    ml_stats = defaultdict(lambda: {"correct": 0, "wrong": 0})
    for i in range(len(y_te)):
        if thr[i] == LABEL_CHOPPY:
            continue
        if thr[i] == y_te[i]:
            ml_stats[te_c[i]]["correct"] += 1
        else:
            ml_stats[te_c[i]]["wrong"] += 1

    for cp in sorted(set(list(rule_stats.keys()) + list(ml_stats.keys()))):
        ms = ml_stats[cp]
        rs = rule_stats[cp]
        md = ms["correct"] + ms["wrong"]
        rd = rs["correct"] + rs["wrong"]
        ml_acc = f"{ms['correct']/md*100:.1f}%" if md > 0 else "n/a"
        ru_acc = f"{rs['correct']/rd*100:.1f}%" if rd > 0 else "n/a"
        print(f"{cp}min".ljust(8) + f"{ml_acc:>12} {ru_acc:>14}")

    # Save final production model (train on all data)
    print(f"\n--- Training production model on ALL {len(all_candles)} days ---")
    X_all, y_all, fn_all, _, _ = build_dataset(all_candles)
    prod_model = train_model(X_all, y_all, fn_all)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "regime_v1.lgb"
    prod_model.save_model(str(model_path))

    meta = {
        "model_type": "regime_classifier",
        "version": "v1",
        "created": datetime.now().isoformat(),
        "training_days": len(all_candles),
        "date_range": f"{days[0]} to {days[-1]}",
        "features": fn_all,
        "n_features": len(fn_all),
        "checkpoints_minutes": CHECKPOINTS,
        "labels": LABEL_NAMES,
        "confidence_threshold": args.threshold,
        "label_distribution": {"bullish": bull, "bearish": bear, "choppy": chop},
    }
    meta_path = MODEL_DIR / "regime_v1_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model: {model_path}")
    print(f"Meta:  {meta_path}")


if __name__ == "__main__":
    main()
