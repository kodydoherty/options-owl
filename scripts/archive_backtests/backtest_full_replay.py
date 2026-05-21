"""Full replay backtest: replay all 36 trades through current v2.1 exit logic.

Pulls premium curves from the harvester DB (5-second snapshots) and candle
data from Polygon. Simulates every exit gate including:
  - ENRG (multi-TF candle voting during grace period when negative)
  - Candle exhaustion (volume_peak with RSI/OBV/patterns)
  - Adaptive 3-stage trail, dollar trail, profit retrace, decel
  - Hard stop at 30%, grace period 20 min

Outputs daily P&L comparison: actual vs simulated with current strategy.

Usage:
    python scripts/backtest_full_replay.py

    # Run from droplet (has harvester DB):
    ssh root@droplet "cd /root/options-owl && python scripts/backtest_full_replay.py"
"""

import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Trade data (from owlet-kody, parent trades only)
# ---------------------------------------------------------------------------

TRADES = [
    {"id":1,"ticker":"IWM","option_type":"call","strike":263.0,"premium":0.3015,"exit_premium":0.34825,"contracts":16,"pnl_dollars":74.80,"pnl_pct":15.51,"exit_reason":"velocity_exit","mfe_premium":0.41,"opened_at":"2026-04-13T14:36:12","closed_at":"2026-04-13T14:37:30","score":100,"expiry":"2026-04-13"},
    {"id":2,"ticker":"SPY","option_type":"call","strike":680.0,"premium":0.804,"exit_premium":1.3034,"contracts":6,"pnl_dollars":299.67,"pnl_pct":62.12,"exit_reason":"velocity_exit","mfe_premium":1.50,"opened_at":"2026-04-13T14:51:13","closed_at":"2026-04-13T15:00:19","score":94,"expiry":"2026-04-13"},
    {"id":3,"ticker":"QQQ","option_type":"call","strike":613.0,"premium":0.6432,"exit_premium":0.6866,"contracts":8,"pnl_dollars":34.68,"pnl_pct":6.74,"exit_reason":"velocity_exit","mfe_premium":0.81,"opened_at":"2026-04-13T15:00:31","closed_at":"2026-04-13T15:02:38","score":100,"expiry":"2026-04-13"},
    {"id":4,"ticker":"AMZN","option_type":"call","strike":237.5,"premium":0.7537,"exit_premium":0.9453,"contracts":7,"pnl_dollars":134.05,"pnl_pct":25.41,"exit_reason":"velocity_exit","mfe_premium":1.13,"opened_at":"2026-04-13T15:24:19","closed_at":"2026-04-13T15:24:40","score":79,"expiry":"2026-04-13"},
    {"id":5,"ticker":"META","option_type":"call","strike":627.5,"premium":1.7085,"exit_premium":2.5074,"contracts":3,"pnl_dollars":239.67,"pnl_pct":46.76,"exit_reason":"velocity_exit","mfe_premium":3.03,"opened_at":"2026-04-13T15:39:19","closed_at":"2026-04-13T15:42:15","score":100,"expiry":"2026-04-13"},
    {"id":6,"ticker":"MSTR","option_type":"call","strike":139.0,"premium":3.6682,"exit_premium":3.8265,"contracts":1,"pnl_dollars":15.83,"pnl_pct":4.31,"exit_reason":"t1_hit","mfe_premium":3.846,"opened_at":"2026-04-15T14:51:54","closed_at":"2026-04-15T14:55:58","score":100,"expiry":"2026-04-15"},
    {"id":7,"ticker":"GOOGL","option_type":"call","strike":335.0,"premium":0.8643,"exit_premium":0.3383,"contracts":4,"pnl_dollars":-210.40,"pnl_pct":-60.86,"exit_reason":"stop_hit","mfe_premium":0.8643,"opened_at":"2026-04-15T14:54:53","closed_at":"2026-04-15T15:04:03","score":100,"expiry":"2026-04-15"},
    {"id":8,"ticker":"QQQ","option_type":"call","strike":632.0,"premium":1.2663,"exit_premium":1.383,"contracts":2,"pnl_dollars":23.35,"pnl_pct":9.22,"exit_reason":"velocity_exit","mfe_premium":1.65,"opened_at":"2026-04-15T15:01:10","closed_at":"2026-04-15T15:06:22","score":100,"expiry":"2026-04-15"},
    {"id":10,"ticker":"AAPL","option_type":"call","strike":262.5,"premium":0.4221,"exit_premium":1.2338,"contracts":8,"pnl_dollars":649.36,"pnl_pct":192.30,"exit_reason":"velocity_exit","mfe_premium":1.43,"opened_at":"2026-04-15T15:15:55","closed_at":"2026-04-15T15:18:04","score":100,"expiry":"2026-04-15"},
    {"id":11,"ticker":"SPY","option_type":"call","strike":698.0,"premium":1.1256,"exit_premium":1.1343,"contracts":2,"pnl_dollars":14.00,"pnl_pct":6.54,"exit_reason":"velocity_exit","mfe_premium":1.33,"opened_at":"2026-04-15T18:33:48","closed_at":"2026-04-15T18:37:58","score":100,"expiry":"2026-04-15"},
    {"id":12,"ticker":"SPY","option_type":"call","strike":700.0,"premium":0.4322,"exit_premium":0.2587,"contracts":7,"pnl_dollars":-35.00,"pnl_pct":-12.50,"exit_reason":"setup_failed","mfe_premium":0.52,"opened_at":"2026-04-15T19:24:48","closed_at":"2026-04-15T19:39:58","score":99,"expiry":"2026-04-15"},
    {"id":13,"ticker":"SPY","option_type":"call","strike":709.0,"premium":1.618,"exit_premium":2.577,"contracts":1,"pnl_dollars":93.00,"pnl_pct":61.18,"exit_reason":"t1_hit","mfe_premium":2.59,"opened_at":"2026-04-17T14:25:12","closed_at":"2026-04-17T14:43:05","score":93,"expiry":"2026-04-17"},
    {"id":14,"ticker":"AMZN","option_type":"call","strike":255.0,"premium":1.0754,"exit_premium":0.786,"contracts":2,"pnl_dollars":-40.00,"pnl_pct":-20.20,"exit_reason":"trailing_stop","mfe_premium":1.58,"opened_at":"2026-04-17T14:34:11","closed_at":"2026-04-17T14:55:46","score":100,"expiry":"2026-04-17"},
    {"id":15,"ticker":"IWM","option_type":"call","strike":277.0,"premium":0.8241,"exit_premium":0.4776,"contracts":4,"pnl_dollars":-116.00,"pnl_pct":-37.66,"exit_reason":"setup_failed","mfe_premium":0.92,"opened_at":"2026-04-17T14:43:00","closed_at":"2026-04-17T14:58:08","score":100,"expiry":"2026-04-17"},
    {"id":17,"ticker":"QQQ","option_type":"call","strike":649.0,"premium":1.5376,"exit_premium":1.4925,"contracts":2,"pnl_dollars":-92.00,"pnl_pct":-31.29,"exit_reason":"setup_failed","mfe_premium":1.73,"opened_at":"2026-04-17T14:46:02","closed_at":"2026-04-17T15:01:16","score":100,"expiry":"2026-04-17"},
    {"id":18,"ticker":"IWM","option_type":"call","strike":280.0,"premium":0.6231,"exit_premium":0.2388,"contracts":5,"pnl_dollars":-170.00,"pnl_pct":-58.62,"exit_reason":"stop_hit","mfe_premium":0.6231,"opened_at":"2026-04-21T14:31:08","closed_at":"2026-04-21T14:47:15","score":100,"expiry":"2026-04-21"},
    {"id":19,"ticker":"MSFT","option_type":"call","strike":427.5,"premium":3.2361,"exit_premium":2.3269,"contracts":1,"pnl_dollars":-90.92,"pnl_pct":-28.10,"exit_reason":"no_momentum","mfe_premium":3.3586,"opened_at":"2026-04-21T14:42:59","closed_at":"2026-04-21T15:43:13","score":100,"expiry":"2026-04-21"},
    {"id":20,"ticker":"NVDA","option_type":"call","strike":202.5,"premium":1.3769,"exit_premium":1.0018,"contracts":2,"pnl_dollars":-75.01,"pnl_pct":-27.24,"exit_reason":"no_momentum","mfe_premium":1.3769,"opened_at":"2026-04-21T14:46:01","closed_at":"2026-04-21T15:46:12","score":100,"expiry":"2026-04-21"},
    {"id":21,"ticker":"QQQ","option_type":"call","strike":647.0,"premium":1.5175,"exit_premium":1.7811,"contracts":2,"pnl_dollars":70.00,"pnl_pct":24.31,"exit_reason":"dollar_trail","mfe_premium":1.83,"opened_at":"2026-04-21T15:10:01","closed_at":"2026-04-21T15:20:25","score":100,"expiry":"2026-04-21"},
    {"id":22,"ticker":"NVDA","option_type":"call","strike":202.5,"premium":1.005,"exit_premium":0.3905,"contracts":3,"pnl_dollars":-184.34,"pnl_pct":-61.14,"exit_reason":"stop_hit","mfe_premium":1.005,"opened_at":"2026-04-21T15:57:48","closed_at":"2026-04-21T16:44:14","score":88,"expiry":"2026-04-21"},
    {"id":23,"ticker":"AMZN","option_type":"call","strike":252.5,"premium":2.221,"exit_premium":1.6478,"contracts":1,"pnl_dollars":-57.33,"pnl_pct":-25.81,"exit_reason":"no_momentum","mfe_premium":2.3911,"opened_at":"2026-04-21T16:06:21","closed_at":"2026-04-21T17:06:23","score":100,"expiry":"2026-04-21"},
    {"id":24,"ticker":"SPY","option_type":"put","strike":705.0,"premium":2.0602,"exit_premium":0.408,"contracts":1,"pnl_dollars":-165.23,"pnl_pct":-80.20,"exit_reason":"stop_hit","mfe_premium":2.0602,"opened_at":"2026-04-22T13:27:41","closed_at":"2026-04-22T13:47:47","score":100,"expiry":"2026-04-22"},
    {"id":25,"ticker":"AVGO","option_type":"call","strike":412.5,"premium":2.8642,"exit_premium":3.184,"contracts":1,"pnl_dollars":51.00,"pnl_pct":18.96,"exit_reason":"t1_hit","mfe_premium":3.20,"opened_at":"2026-04-22T15:00:42","closed_at":"2026-04-22T15:15:25","score":100,"expiry":"2026-04-22"},
    {"id":26,"ticker":"MSTR","option_type":"call","strike":180.0,"premium":5.6782,"exit_premium":5.9334,"contracts":1,"pnl_dollars":25.52,"pnl_pct":4.49,"exit_reason":"t1_hit","mfe_premium":5.9633,"opened_at":"2026-04-22T15:03:41","closed_at":"2026-04-22T15:11:23","score":100,"expiry":"2026-04-22"},
    {"id":27,"ticker":"NVDA","option_type":"call","strike":202.5,"premium":0.2915,"exit_premium":0.1194,"contracts":10,"pnl_dollars":-170.00,"pnl_pct":-58.62,"exit_reason":"no_momentum","mfe_premium":0.35,"opened_at":"2026-04-22T15:18:39","closed_at":"2026-04-22T16:18:43","score":100,"expiry":"2026-04-22"},
    {"id":28,"ticker":"AMZN","option_type":"call","strike":252.5,"premium":0.6231,"exit_premium":0.9254,"contracts":5,"pnl_dollars":185.00,"pnl_pct":64.91,"exit_reason":"dollar_trail","mfe_premium":0.94,"opened_at":"2026-04-22T15:51:40","closed_at":"2026-04-22T16:23:36","score":100,"expiry":"2026-04-22"},
    {"id":29,"ticker":"PLTR","option_type":"call","strike":150.0,"premium":3.3969,"exit_premium":3.5317,"contracts":1,"pnl_dollars":13.48,"pnl_pct":3.97,"exit_reason":"t1_hit","mfe_premium":3.5494,"opened_at":"2026-04-22T16:15:42","closed_at":"2026-04-22T16:28:53","score":100,"expiry":"2026-04-22"},
    {"id":30,"ticker":"AVGO","option_type":"call","strike":420.0,"premium":2.0,"exit_premium":0.7263,"contracts":1,"pnl_dollars":-114.00,"pnl_pct":-60.96,"exit_reason":"stop_hit","mfe_premium":2.0,"opened_at":"2026-04-22T16:24:40","closed_at":"2026-04-22T16:55:21","score":100,"expiry":"2026-04-22"},
    {"id":31,"ticker":"AMZN","option_type":"call","strike":257.5,"premium":1.618,"exit_premium":1.9781,"contracts":2,"pnl_dollars":72.01,"pnl_pct":22.25,"exit_reason":"t1_hit","mfe_premium":1.988,"opened_at":"2026-04-23T14:33:26","closed_at":"2026-04-23T14:35:31","score":100,"expiry":"2026-04-23"},
    {"id":32,"ticker":"SPY","option_type":"call","strike":711.0,"premium":1.2663,"exit_premium":1.5124,"contracts":2,"pnl_dollars":49.22,"pnl_pct":19.43,"exit_reason":"profit_retrace","mfe_premium":1.67,"opened_at":"2026-04-23T14:45:26","closed_at":"2026-04-23T14:59:27","score":100,"expiry":"2026-04-23"},
    {"id":33,"ticker":"QQQ","option_type":"call","strike":655.0,"premium":1.6381,"exit_premium":1.8607,"contracts":2,"pnl_dollars":44.50,"pnl_pct":13.58,"exit_reason":"profit_retrace","mfe_premium":2.02,"opened_at":"2026-04-23T14:57:25","closed_at":"2026-04-23T15:12:10","score":100,"expiry":"2026-04-23"},
    {"id":34,"ticker":"AMD","option_type":"call","strike":307.5,"premium":4.4923,"exit_premium":4.7734,"contracts":1,"pnl_dollars":28.10,"pnl_pct":6.26,"exit_reason":"t1_hit","mfe_premium":4.7974,"opened_at":"2026-04-23T15:00:28","closed_at":"2026-04-23T15:08:53","score":100,"expiry":"2026-04-23"},
    {"id":35,"ticker":"AVGO","option_type":"call","strike":430.0,"premium":4.1908,"exit_premium":3.7769,"contracts":1,"pnl_dollars":-41.40,"pnl_pct":-9.88,"exit_reason":"no_momentum","mfe_premium":4.4409,"opened_at":"2026-04-23T15:15:26","closed_at":"2026-04-23T16:15:29","score":100,"expiry":"2026-04-23"},
    {"id":36,"ticker":"AAPL","option_type":"call","strike":275.0,"premium":1.397,"exit_premium":0.987,"contracts":2,"pnl_dollars":-81.99,"pnl_pct":-29.35,"exit_reason":"no_momentum","mfe_premium":1.397,"opened_at":"2026-04-23T15:33:28","closed_at":"2026-04-23T16:33:28","score":100,"expiry":"2026-04-23"},
    {"id":37,"ticker":"PLTR","option_type":"put","strike":145.0,"premium":1.5477,"exit_premium":1.6941,"contracts":2,"pnl_dollars":29.29,"pnl_pct":9.46,"exit_reason":"t1_hit","mfe_premium":1.7026,"opened_at":"2026-04-23T15:39:26","closed_at":"2026-04-23T15:40:24","score":95,"expiry":"2026-04-23"},
]


# ---------------------------------------------------------------------------
# Harvester premium curve loader
# ---------------------------------------------------------------------------

import sqlite3

HARVESTER_DB = os.environ.get(
    "HARVESTER_DB",
    "journal/options_data.db",
)

# Trade timestamps are ET (naive). Harvester DB is UTC with tz offset.
# EDT offset: ET + 4 hours = UTC
ET_TO_UTC_HOURS = 4


def contract_ticker(ticker: str, expiry: str, option_type: str, strike: float) -> str:
    """Build Polygon-style option contract ticker: O:SPY260413C00680000"""
    exp = expiry.replace("-", "")[2:]  # 2026-04-13 → 260413
    ot = "C" if option_type == "call" else "P"
    strike_int = int(strike * 1000)
    return f"O:{ticker}{exp}{ot}{strike_int:08d}"


def load_premium_curve(ct: str, opened_at: str, closed_at: str) -> list[tuple[float, float, float]]:
    """Load (timestamp_epoch, midpoint, underlying_price) from harvester.

    Trade times are ET (naive). DB timestamps are UTC with tz offset.
    Returns snapshots from 1 min before open to 60 min after close (or EOD).
    """
    if not os.path.exists(HARVESTER_DB):
        return []

    # Convert ET naive → UTC naive for string comparison with DB
    open_et = datetime.fromisoformat(opened_at)
    close_et = datetime.fromisoformat(closed_at)
    open_utc = open_et + timedelta(hours=ET_TO_UTC_HOURS)
    close_utc = close_et + timedelta(hours=ET_TO_UTC_HOURS)

    from_utc = open_utc - timedelta(minutes=1)
    to_utc = min(close_utc + timedelta(minutes=60),
                 open_utc.replace(hour=23, minute=59, second=59))

    conn = sqlite3.connect(HARVESTER_DB)
    rows = conn.execute(
        "SELECT captured_at, midpoint, underlying_price "
        "FROM harvest_snapshots "
        "WHERE contract_ticker = ? AND captured_at >= ? AND captured_at <= ? "
        "ORDER BY captured_at",
        (ct, from_utc.strftime("%Y-%m-%dT%H:%M"), to_utc.strftime("%Y-%m-%dT%H:%M")),
    ).fetchall()
    conn.close()

    result = []
    for ts_str, mid, und in rows:
        if mid and mid > 0:
            # Parse UTC timestamp, convert to ET epoch for gate logic
            ts_utc = datetime.fromisoformat(ts_str.replace("+00:00", ""))
            ts_et = ts_utc - timedelta(hours=ET_TO_UTC_HOURS)
            result.append((ts_et.timestamp(), float(mid), float(und or 0)))
    return result


# ---------------------------------------------------------------------------
# Candle data loader (from Polygon, with caching)
# ---------------------------------------------------------------------------

CANDLE_DIR = "journal/candle_cache"

def get_api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        try:
            with open(".env") as f:
                for line in f:
                    if line.startswith("POLYGON_API_KEY="):
                        key = line.strip().split("=", 1)[1].strip('"').strip("'")
                        break
        except FileNotFoundError:
            pass
    return key


def fetch_polygon_candles(api_key: str, ticker: str, mult: int, span: str, date: str) -> list[dict]:
    cache_file = os.path.join(CANDLE_DIR, f"{ticker}_{date}_{mult}{span}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    if not api_key:
        return []
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{mult}/{span}"
        f"/{date}/{date}?adjusted=true&sort=asc&limit=5000&apiKey={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        bars = data.get("results", [])
    except Exception:
        bars = []
    os.makedirs(CANDLE_DIR, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(bars, f)
    return bars


def build_candle_data(api_key: str, ticker: str, date: str, cutoff_ms: int) -> dict:
    """Build candle_data dict with indicators for ENRG/exhaustion at a point in time."""
    from options_owl.collectors.candle_cache import (
        CandleBar, calc_atr, calc_obv, calc_rsi,
        calc_volume_trend, detect_candle_pattern,
    )

    TFS = [(5, "minute", "5m"), (15, "minute", "15m"), (30, "minute", "30m"),
           (1, "hour", "1h"), (4, "hour", "4h")]

    indicators = {}
    for mult, span, label in TFS:
        raw = fetch_polygon_candles(api_key, ticker, mult, span, date)
        filtered = [b for b in raw if b.get("t", 0) <= cutoff_ms]
        bars = [CandleBar(timestamp=b.get("t",0), open=float(b.get("o",0)),
                          high=float(b.get("h",0)), low=float(b.get("l",0)),
                          close=float(b.get("c",0)), volume=float(b.get("v",0)),
                          vwap=float(b.get("vw",0))) for b in filtered]
        if bars:
            indicators[label] = {
                "atr": calc_atr(bars), "rsi": calc_rsi(bars), "obv": calc_obv(bars),
                "pattern": detect_candle_pattern(bars), "volume_trend": calc_volume_trend(bars),
            }
        else:
            indicators[label] = {"atr": None, "rsi": None, "obv": None,
                                 "pattern": None, "volume_trend": None}
    return {"indicators": indicators}


# ---------------------------------------------------------------------------
# Exit simulation (simplified replay of current v2.1 gates)
# ---------------------------------------------------------------------------

from options_owl.collectors.candle_cache import check_exhaustion, evaluate_enrg
from options_owl.risk.vinny_strategy import evaluate_adaptive_trail


# Settings matching production
HARD_STOP_PCT = 50.0   # Production: PREMIUM_STOP_PCT=50
GRACE_MINUTES = 20.0   # Production: STOP_GRACE_PERIOD_MINUTES=20
ADAPTIVE_ACTIVATION = 35.0
ADAPTIVE_ACTIVE_WIDTH = 35.0
RUNNER_THRESHOLD = 150.0
RUNNER_WIDTH = 45.0
MOONSHOT_THRESHOLD = 400.0
MOONSHOT_WIDTH = 30.0
PROFIT_RETRACE_PCT = 35.0
PROFIT_RETRACE_MIN = 10.0
DOLLAR_TRAIL_ACTIVATION = 40.0
NO_MOMENTUM_MINUTES = 45.0
VOLUME_PEAK_TIGHTEN = 0.7
VOLUME_PEAK_MIN_GAIN = 35.0
ENRG_WIDEN_PCT = 15.0
THETA_BLEED_MINUTES = 45.0
THETA_BLEED_LOSS_PCT = 30.0


@dataclass
class SimResult:
    trade_id: int
    ticker: str
    date: str
    actual_pnl: float
    actual_reason: str
    sim_pnl: float
    sim_reason: str
    sim_exit_prem: float
    contracts: int
    has_curve: bool


def _enrg_no_4h_override(candle_data: dict, direction: str) -> tuple[str, str]:
    """evaluate_enrg but with 4h extreme override disabled (fix 1).

    Strips the 4h pattern before calling evaluate_enrg so the extreme
    override can only fire on 1h.
    """
    import copy
    patched = copy.deepcopy(candle_data)
    if "4h" in patched.get("indicators", {}):
        patched["indicators"]["4h"]["pattern"] = None
    return evaluate_enrg(patched, direction)


INTERIM_STOP_MINUTES = 10.0   # Fix 2: interim check at 10 min
INTERIM_STOP_PCT = 40.0       # Fix 2: exit if down 40%+ at 10 min


def simulate_trade(trade: dict, api_key: str,
                   fix_no_4h_override: bool = False,
                   fix_interim_stop: bool = False) -> SimResult:
    """Replay one trade through current exit logic using harvester premium curve.

    fix_no_4h_override: if True, ENRG extreme override only fires on 1h (not 4h)
    fix_interim_stop:   if True, add interim stop check at 10 min (-40%+ → exit)
    """
    ticker = trade["ticker"]
    opt_type = trade["option_type"]
    entry_prem = trade["premium"]
    contracts = trade["contracts"]
    opened_at = trade["opened_at"]
    closed_at = trade["closed_at"]
    expiry = trade["expiry"]
    date = opened_at[:10]

    ct = contract_ticker(ticker, expiry, opt_type, trade["strike"])
    curve = load_premium_curve(ct, opened_at, closed_at)

    if not curve:
        # No harvester data — use actual result
        return SimResult(
            trade["id"], ticker, date, trade["pnl_dollars"], trade["exit_reason"],
            trade["pnl_dollars"], trade["exit_reason"], trade["exit_premium"],
            contracts, has_curve=False,
        )

    # opened_at is ET naive — get ET epoch (curve timestamps are also ET epoch)
    open_ts = datetime.fromisoformat(opened_at).timestamp()

    # State
    peak_prem = entry_prem
    enrg_done = False
    enrg_action = None
    stop_widened = False
    vol_peak_tighten = False
    interim_checked = False

    # Load candle data once for ENRG (at ~5 min in)
    enrg_check_ms = int((open_ts + 300) * 1000)
    candle_data = build_candle_data(api_key, ticker, date, enrg_check_ms)

    # Load candle data for exhaustion (at ~halfway through trade)
    mid_ts = open_ts + (datetime.fromisoformat(closed_at).timestamp() - open_ts) / 2
    exhaust_candle_data = build_candle_data(api_key, ticker, date, int(mid_ts * 1000))

    sim_exit_prem = None
    sim_reason = None

    for ts, mid, underlying in curve:
        if ts < open_ts:
            continue

        elapsed_min = (ts - open_ts) / 60
        current_prem = mid
        peak_prem = max(peak_prem, current_prem)
        gain_pct = (current_prem - entry_prem) / entry_prem * 100
        peak_gain_pct = (peak_prem - entry_prem) / entry_prem * 100
        drop_from_entry_pct = (entry_prem - current_prem) / entry_prem * 100

        # --- Gate 0: ENRG (during grace period, when negative) ---
        if not enrg_done and elapsed_min >= 2 and elapsed_min < GRACE_MINUTES:
            if current_prem < entry_prem and candle_data.get("indicators"):
                if fix_no_4h_override:
                    enrg_action, enrg_reason = _enrg_no_4h_override(candle_data, opt_type)
                else:
                    enrg_action, enrg_reason = evaluate_enrg(candle_data, opt_type)
                enrg_done = True
                if enrg_action == "IMMEDIATE_EXIT":
                    sim_exit_prem = current_prem
                    sim_reason = f"enrg_exit ({enrg_reason})"
                    break
                elif enrg_action == "HOLD":
                    stop_widened = True

        # --- Fix 2: Interim stop at 10 min (during grace, heavy loss) ---
        if fix_interim_stop and not interim_checked and elapsed_min >= INTERIM_STOP_MINUTES:
            interim_checked = True
            if drop_from_entry_pct >= INTERIM_STOP_PCT:
                sim_exit_prem = current_prem
                sim_reason = f"interim_stop ({elapsed_min:.0f}min, -{drop_from_entry_pct:.1f}%)"
                break

        # --- Gate 1: Hard stop (with ENRG widening) ---
        if elapsed_min >= GRACE_MINUTES:
            threshold = HARD_STOP_PCT
            if stop_widened:
                threshold = threshold * (1 + ENRG_WIDEN_PCT / 100)
            if drop_from_entry_pct >= threshold:
                sim_exit_prem = current_prem
                sim_reason = f"stop_hit (-{drop_from_entry_pct:.1f}%)"
                break

        # --- Volume peak exhaustion (tighten trail) ---
        if not vol_peak_tighten and peak_gain_pct >= VOLUME_PEAK_MIN_GAIN:
            exhausted, _ = check_exhaustion(
                exhaust_candle_data, opt_type, peak_gain_pct, VOLUME_PEAK_MIN_GAIN
            )
            if exhausted:
                vol_peak_tighten = True

        # --- Profit retrace (dormant zone: 10-35%) ---
        if PROFIT_RETRACE_MIN <= peak_gain_pct < ADAPTIVE_ACTIVATION:
            profit_at_peak = peak_prem - entry_prem
            profit_now = current_prem - entry_prem
            if profit_at_peak > 0:
                retrace_pct = ((profit_at_peak - profit_now) / profit_at_peak) * 100
                if retrace_pct >= PROFIT_RETRACE_PCT:
                    sim_exit_prem = current_prem
                    sim_reason = f"profit_retrace ({retrace_pct:.0f}% of +{peak_gain_pct:.0f}%)"
                    break

        # --- Adaptive trail (35%+ activation) ---
        if peak_gain_pct >= ADAPTIVE_ACTIVATION:
            tighten = VOLUME_PEAK_TIGHTEN if vol_peak_tighten else 1.0
            result = evaluate_adaptive_trail(
                entry_premium=entry_prem,
                current_premium=current_prem,
                peak_premium=peak_prem,
                activation_pct=ADAPTIVE_ACTIVATION,
                active_width=ADAPTIVE_ACTIVE_WIDTH * tighten,
                runner_threshold=RUNNER_THRESHOLD,
                runner_width=RUNNER_WIDTH * tighten,
                moonshot_threshold=MOONSHOT_THRESHOLD,
                moonshot_width=MOONSHOT_WIDTH * tighten,
            )
            if result.should_exit:
                sim_exit_prem = current_prem
                tighten_note = " [vol-peak]" if vol_peak_tighten else ""
                sim_reason = f"adaptive_trail{tighten_note} ({result.reason})"
                break

        # --- Dollar trail (40% activation) ---
        if gain_pct >= DOLLAR_TRAIL_ACTIVATION:
            # Simplified: check if dropped $-steps from peak
            cost = entry_prem * 100
            profit = (current_prem - entry_prem) * 100
            peak_profit = (peak_prem - entry_prem) * 100
            if peak_profit > 0:
                step = cost * 0.20 if peak_profit < cost * 0.25 else cost * 0.10
                if step > 0:
                    trail_floor = peak_profit - step
                    if profit <= trail_floor and trail_floor > 0:
                        sim_exit_prem = current_prem
                        sim_reason = f"dollar_trail (profit ${profit:.0f} < floor ${trail_floor:.0f})"
                        break

        # --- Theta bleed (45 min + losing 30%+) ---
        if elapsed_min >= THETA_BLEED_MINUTES and drop_from_entry_pct >= THETA_BLEED_LOSS_PCT:
            sim_exit_prem = current_prem
            sim_reason = f"theta_bleed ({elapsed_min:.0f}min, -{drop_from_entry_pct:.1f}%)"
            break

        # --- No momentum (45 min + not up 5%) ---
        if elapsed_min >= NO_MOMENTUM_MINUTES and gain_pct < 5.0:
            sim_exit_prem = current_prem
            sim_reason = f"no_momentum ({elapsed_min:.0f}min, {gain_pct:+.1f}%)"
            break

        # --- EOD cutoff (3:45 PM ET) --- timestamps are ET epoch
        dt = datetime.fromtimestamp(ts)
        if (dt.hour > 15) or (dt.hour == 15 and dt.minute >= 45):
            sim_exit_prem = current_prem
            sim_reason = "eod_cutoff"
            break

    # If no exit triggered, use last price in curve
    if sim_exit_prem is None and curve:
        sim_exit_prem = curve[-1][1]
        sim_reason = "curve_end"
    elif sim_exit_prem is None:
        sim_exit_prem = trade["exit_premium"]
        sim_reason = trade["exit_reason"]

    sim_pnl = (sim_exit_prem - entry_prem) * 100 * contracts

    return SimResult(
        trade["id"], ticker, date, trade["pnl_dollars"], trade["exit_reason"],
        sim_pnl, sim_reason, sim_exit_prem, contracts, has_curve=True,
    )


def run_variant(trades, api_key, label, fix_no_4h=False, fix_interim=False, verbose=False):
    """Run all trades through one variant and return results."""
    results = []
    for trade in trades:
        r = simulate_trade(trade, api_key, fix_no_4h_override=fix_no_4h, fix_interim_stop=fix_interim)
        results.append(r)
    return results


def summarize(results, label):
    """Return (total_pnl, wins, losses, daily_dict, reason_dict) for a variant."""
    total = sum(r.sim_pnl for r in results)
    wins = sum(1 for r in results if r.sim_pnl > 0)
    losses = sum(1 for r in results if r.sim_pnl <= 0)
    days = {}
    for r in results:
        if r.date not in days:
            days[r.date] = 0.0
        days[r.date] += r.sim_pnl
    reasons = {}
    for r in results:
        base = r.sim_reason.split(" ")[0].split("(")[0]
        if base not in reasons:
            reasons[base] = {"count": 0, "pnl": 0.0}
        reasons[base]["count"] += 1
        reasons[base]["pnl"] += r.sim_pnl
    return total, wins, losses, days, reasons


def main():
    api_key = get_api_key()

    print("=" * 95)
    print("VARIANT COMPARISON — Testing ENRG & Interim Stop Fixes")
    print("=" * 95)
    print(f"Trades: {len(TRADES)} | Testing 4 variants:")
    print(f"  A) BASELINE  — current v2.1 (ENRG extreme on 1h+4h, 20-min grace, no interim)")
    print(f"  B) FIX 1     — disable ENRG extreme override on 4h (only trust 1h)")
    print(f"  C) FIX 2     — add interim stop at 10 min if down 40%+")
    print(f"  D) BOTH      — fix 1 + fix 2 combined")
    print()

    # --- Run all 4 variants ---
    VARIANTS = [
        ("A) BASELINE", False, False),
        ("B) NO 4H OVERRIDE", True, False),
        ("C) INTERIM STOP", False, True),
        ("D) BOTH FIXES", True, True),
    ]

    all_results = {}
    for label, f1, f2 in VARIANTS:
        print(f"Running {label}...", flush=True)
        all_results[label] = run_variant(TRADES, api_key, label, fix_no_4h=f1, fix_interim=f2)

    # --- Daily P&L comparison table ---
    print(f"\n{'=' * 95}")
    print("DAILY P&L BY VARIANT")
    print(f"{'=' * 95}")

    actual_total = sum(t["pnl_dollars"] for t in TRADES)
    summaries = {label: summarize(res, label) for label, res in all_results.items()}
    day_set = sorted(set(r.date for r in all_results[VARIANTS[0][0]]))

    header = f"{'Date':<12} {'Actual':>10}"
    for label, _, _ in VARIANTS:
        short = label.split(")")[0] + ")"
        header += f" {short:>14}"
    print(header)
    print("-" * (12 + 10 + 14 * len(VARIANTS) + len(VARIANTS)))

    for day in day_set:
        actual_day = sum(t["pnl_dollars"] for t in TRADES if t["opened_at"][:10] == day)
        line = f"{day:<12} {actual_day:>+10.0f}"
        for label, _, _ in VARIANTS:
            day_pnl = summaries[label][3].get(day, 0)
            line += f" {day_pnl:>+14.0f}"
        print(line)

    print("-" * (12 + 10 + 14 * len(VARIANTS) + len(VARIANTS)))
    line = f"{'TOTAL':<12} {actual_total:>+10.0f}"
    for label, _, _ in VARIANTS:
        line += f" {summaries[label][0]:>+14.0f}"
    print(line)

    line = f"{'W/L':<12} {'':>10}"
    for label, _, _ in VARIANTS:
        w, l = summaries[label][1], summaries[label][2]
        line += f" {f'{w}W/{l}L':>14}"
    print(line)

    # --- Trades that changed between variants ---
    print(f"\n{'=' * 95}")
    print("TRADE-BY-TRADE DIFFERENCES (only trades with curves that changed)")
    print(f"{'=' * 95}")

    baseline = all_results[VARIANTS[0][0]]
    for vlabel, _, _ in VARIANTS[1:]:
        variant = all_results[vlabel]
        diffs = []
        for b, v in zip(baseline, variant):
            if b.has_curve and abs(b.sim_pnl - v.sim_pnl) > 0.01:
                diffs.append((b, v))
        if not diffs:
            print(f"\n{vlabel}: No changes from baseline")
            continue

        print(f"\n{vlabel}: {len(diffs)} trades changed")
        print(f"  {'#':>3} {'Ticker':<6} {'Date':<12} {'Baseline':>10} {'Variant':>10} {'Delta':>10} {'Baseline Exit':<30} {'Variant Exit':<30}")
        print(f"  {'-'*112}")
        total_delta = 0
        for b, v in diffs:
            d = v.sim_pnl - b.sim_pnl
            total_delta += d
            b_reason = b.sim_reason[:28]
            v_reason = v.sim_reason[:28]
            print(f"  {b.trade_id:>3} {b.ticker:<6} {b.date:<12} {b.sim_pnl:>+10.2f} {v.sim_pnl:>+10.2f} {d:>+10.2f} {b_reason:<30} {v_reason:<30}")
        print(f"  {'':>3} {'':>6} {'NET DELTA':<12} {'':>10} {'':>10} {total_delta:>+10.2f}")

    # --- Exit reason comparison ---
    print(f"\n{'=' * 95}")
    print("EXIT REASON BREAKDOWN BY VARIANT")
    print(f"{'=' * 95}")

    all_reasons = set()
    for label, _, _ in VARIANTS:
        all_reasons.update(summaries[label][4].keys())

    header = f"{'Reason':<22}"
    for label, _, _ in VARIANTS:
        short = label.split(")")[0] + ")"
        header += f" {short:>16}"
    print(header)
    print("-" * (22 + 16 * len(VARIANTS) + len(VARIANTS)))

    for reason in sorted(all_reasons):
        line = f"{reason:<22}"
        for label, _, _ in VARIANTS:
            stats = summaries[label][4].get(reason, {"count": 0, "pnl": 0})
            line += f" {stats['count']:>3}x {stats['pnl']:>+10.0f}"
        print(line)

    # --- Final verdict ---
    print(f"\n{'=' * 95}")
    print("SUMMARY")
    print(f"{'=' * 95}")
    print(f"{'Variant':<25} {'Total P&L':>12} {'vs Baseline':>12} {'vs Actual':>12} {'Win Rate':>10}")
    print("-" * 75)
    base_pnl = summaries[VARIANTS[0][0]][0]
    for label, _, _ in VARIANTS:
        total, wins, losses, _, _ = summaries[label]
        vs_base = total - base_pnl
        vs_actual = total - actual_total
        wr = wins / (wins + losses) * 100 if (wins + losses) else 0
        vs_base_str = f"{vs_base:>+12.0f}" if label != VARIANTS[0][0] else f"{'—':>12}"
        print(f"{label:<25} {total:>+12.0f} {vs_base_str} {vs_actual:>+12.0f} {wr:>9.0f}%")
    print(f"\nActual historical P&L: ${actual_total:+,.2f}")

    # --- Portfolio-scaled replay ---
    # Recalculate contracts using $8K portfolio with current sizing rules
    PORTFOLIO = 8000.0
    MAX_RISK_PCT = 0.75
    MAX_CONCURRENT = 5
    MAX_POS_PCT = 0.15
    TARGET_PER_SLOT = PORTFOLIO * MAX_RISK_PCT / MAX_CONCURRENT
    POS_CAP_DOLLARS = PORTFOLIO * MAX_POS_PCT

    def score_budget_mult(score):
        if score >= 95: return 1.00
        if score >= 90: return 0.75
        if score >= 85: return 0.50
        if score >= 78: return 0.25
        return 0

    print(f"\n{'=' * 95}")
    print(f"PORTFOLIO-SCALED REPLAY — $8,000 Portfolio")
    print(f"{'=' * 95}")
    print(f"Sizing: ${PORTFOLIO:,.0f} × {MAX_RISK_PCT:.0%} / {MAX_CONCURRENT} = ${TARGET_PER_SLOT:,.0f}/slot")
    print(f"Position cap: {MAX_POS_PCT:.0%} = ${POS_CAP_DOLLARS:,.0f} | Score mult: 95→100%, 90→75%, 85→50%, 78→25%")
    print()

    baseline = all_results[VARIANTS[0][0]]  # Use baseline sim results

    print(f"{'#':>3} {'Ticker':<6} {'Date':<12} {'Score':>5} {'Prem':>7} {'Old#':>4} {'New#':>4}"
          f" {'Old P&L':>10} {'New P&L':>10} {'Exit Reason':<25}")
    print("-" * 100)

    scaled_daily = {}
    old_daily = {}
    cumulative = 0.0

    for i, (trade, result) in enumerate(zip(TRADES, baseline)):
        score = trade["score"]
        premium = trade["premium"]
        cost_per = premium * 100
        old_contracts = trade["contracts"]

        # New sizing — budget multiplier scales with portfolio
        mult = score_budget_mult(score)
        scaled_target = TARGET_PER_SLOT * mult
        raw = int(scaled_target / cost_per) if cost_per > 0 else 0
        pos_cap = int(POS_CAP_DOLLARS / cost_per) if cost_per > 0 else 0
        new_contracts = max(1, min(raw, pos_cap)) if mult > 0 else 0

        # Scale P&L: sim used old contracts, rescale to new
        if old_contracts > 0 and result.has_curve:
            pnl_per_contract = result.sim_pnl / old_contracts
            new_pnl = pnl_per_contract * new_contracts
        elif old_contracts > 0:
            pnl_per_contract = trade["pnl_dollars"] / old_contracts
            new_pnl = pnl_per_contract * new_contracts
        else:
            new_pnl = 0

        old_pnl = result.sim_pnl if result.has_curve else trade["pnl_dollars"]
        date = trade["opened_at"][:10]

        scaled_daily[date] = scaled_daily.get(date, 0) + new_pnl
        old_daily[date] = old_daily.get(date, 0) + old_pnl

        reason = result.sim_reason[:23] if result.has_curve else trade["exit_reason"][:23]
        print(f"{trade['id']:>3} {trade['ticker']:<6} {date:<12} {score:>5} ${premium:>5.2f} {old_contracts:>4} {new_contracts:>4}"
              f" {old_pnl:>+10.2f} {new_pnl:>+10.2f} {reason:<25}")

    # Daily summary
    print(f"\n{'=' * 95}")
    print(f"DAILY P&L — $8K Portfolio-Scaled")
    print(f"{'=' * 95}")

    days = sorted(scaled_daily.keys())
    cumulative = 0.0
    total_old = 0.0
    total_new = 0.0

    bar_max = max(abs(v) for v in scaled_daily.values())
    bar_scale = 40.0 / bar_max if bar_max > 0 else 1

    print(f"{'Date':<12} {'Sim (old #)':>12} {'Sim ($8K #)':>12} {'Cumulative':>12}  Chart")
    print("-" * 85)

    for day in days:
        old_pnl = old_daily[day]
        new_pnl = scaled_daily[day]
        cumulative += new_pnl
        total_old += old_pnl
        total_new += new_pnl

        bar_len = int(abs(new_pnl) * bar_scale)
        if new_pnl >= 0:
            bar = "  " + "█" * bar_len
        else:
            bar = "█" * bar_len + "  "
            bar = bar.rjust(42)

        print(f"{day:<12} {old_pnl:>+12,.0f} {new_pnl:>+12,.0f} {cumulative:>+12,.0f}  {bar}")

    print("-" * 85)
    print(f"{'TOTAL':<12} {total_old:>+12,.0f} {total_new:>+12,.0f} {cumulative:>+12,.0f}")

    roi = (total_new / PORTFOLIO) * 100
    print(f"\nStarting portfolio:  ${PORTFOLIO:>10,.0f}")
    print(f"Ending portfolio:    ${PORTFOLIO + total_new:>10,.0f}")
    print(f"Total return:        {roi:>+10.1f}%")


if __name__ == "__main__":
    main()
