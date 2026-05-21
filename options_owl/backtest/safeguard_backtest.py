"""Safeguard backtest — test proposed exit improvements against historical 1-min option bars.

Replays paper trades through minute-by-minute premium data and compares
different safeguard configurations to measure MFE capture improvement.

Safeguards tested:
1. Breakeven stop — after +X% gain, move stop to entry price (never go red)
2. Profit lock ratchet — after +50%, lock in +20%; after +100%, lock in +50%; etc.
3. Velocity exit — if premium drops >Y% in Z minutes, exit immediately
4. Adaptive trail tightening — tighten trail faster based on time held
5. Max hold time — hard cap at N minutes for 0DTE
6. Combined "Vinny+" — best combination of above
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BarData:
    """Single 1-minute option price bar."""
    timestamp: int  # unix epoch
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class TradeSetup:
    """A trade to replay through historical bars."""
    trade_id: int
    ticker: str
    direction: str  # call/put
    entry_premium: float
    entry_time: str
    contracts: int
    total_cost: float
    target_1: float | None
    target_2: float | None
    stop_price: float | None
    actual_exit_premium: float
    actual_exit_reason: str
    actual_pnl_pct: float
    actual_mfe_pnl_pct: float


@dataclass
class SafeguardConfig:
    """Configuration for a safeguard strategy."""
    name: str

    # Breakeven stop: after this gain%, move stop to entry
    breakeven_activation_pct: float | None = None  # e.g., 30.0

    # Profit lock ratchet: list of (gain_threshold, lock_pct) pairs
    # e.g., [(50, 20), (100, 50), (200, 100)] means:
    #   after +50%, lock in +20%; after +100%, lock in +50%
    profit_locks: list[tuple[float, float]] = field(default_factory=list)

    # Velocity exit: if premium drops this % in this many minutes, exit
    velocity_drop_pct: float | None = None  # e.g., 15.0
    velocity_window_minutes: int = 5  # look-back window

    # Phase trails (override Vinny defaults)
    phase_trails: dict[int, float] | None = None

    # Adaptive time tightening: after N minutes, multiply trail by factor
    time_tighten_after_minutes: float | None = None
    time_tighten_factor: float = 0.6  # multiply trail by this after time

    # Max hold time (hard exit)
    max_hold_minutes: float | None = None

    # Partial profit at MFE thresholds (close X% at Y% gain)
    partial_at_pct: float | None = None  # e.g., close 50% when up 100%
    partial_close_pct: float = 50.0


# Default Vinny phase trails
VINNY_TRAILS = {0: 25.0, 1: 20.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0, 6: 8.0}


@dataclass
class SimResult:
    """Result of simulating a single trade with a safeguard config."""
    trade_id: int
    ticker: str
    config_name: str
    entry_premium: float
    exit_premium: float
    exit_reason: str
    exit_minute: int  # minutes after entry
    pnl_pct: float
    mfe_pnl_pct: float
    mfe_capture_pct: float  # pnl / mfe (how much of max gain we captured)
    bars_replayed: int


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def _get_phase(premium: float, entry_premium: float, targets: list[float | None]) -> int:
    """Determine current phase based on premium gain vs target levels.

    Targets are premium-based thresholds derived from underlying targets.
    For simplicity, we use gain % thresholds: T1=+30%, T2=+60%, T3=+100%,
    T4=+150%, T5=+200%.
    """
    gain_pct = (premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
    # Phase thresholds based on premium gain
    thresholds = [30, 60, 100, 150, 200]
    phase = 0
    for i, t in enumerate(thresholds):
        if gain_pct >= t:
            phase = i + 1
    return min(phase, 6)


def simulate_trade(
    bars: list[BarData],
    setup: TradeSetup,
    config: SafeguardConfig,
) -> SimResult:
    """Replay a trade through minute bars with the given safeguard config.

    Returns the simulated exit result.
    """
    if not bars:
        return SimResult(
            trade_id=setup.trade_id, ticker=setup.ticker,
            config_name=config.name, entry_premium=setup.entry_premium,
            exit_premium=setup.entry_premium, exit_reason="no_bars",
            exit_minute=0, pnl_pct=0, mfe_pnl_pct=0, mfe_capture_pct=0,
            bars_replayed=0,
        )

    entry = setup.entry_premium
    peak = entry
    mfe_pnl_pct = 0.0
    phase = 0

    trails = config.phase_trails or VINNY_TRAILS

    # For velocity tracking
    recent_closes: list[float] = []
    velocity_window = config.velocity_window_minutes

    # For partial tracking

    exit_premium = bars[-1].close  # default: hold to end
    exit_reason = "eod_expiry"
    exit_minute = len(bars)

    for i, bar in enumerate(bars):
        current = bar.close
        minutes_held = i + 1

        # Update peak / MFE
        if current > peak:
            peak = current

        current_gain = (current - entry) / entry * 100 if entry > 0 else 0
        peak_gain = (peak - entry) / entry * 100 if entry > 0 else 0

        if peak_gain > mfe_pnl_pct:
            mfe_pnl_pct = peak_gain

        # Update phase
        phase = _get_phase(current, entry, [])

        # --- Safeguard checks (in priority order) ---

        # 1. Hard stop loss (50% of premium)
        if current_gain <= -50:
            exit_premium = current
            exit_reason = "stop_loss"
            exit_minute = minutes_held
            break

        # 2. Breakeven stop
        if config.breakeven_activation_pct is not None:
            if peak_gain >= config.breakeven_activation_pct and current <= entry:
                exit_premium = current
                exit_reason = "breakeven_stop"
                exit_minute = minutes_held
                break

        # 3. Profit lock ratchet
        if config.profit_locks:
            # Find highest applicable lock
            lock_floor = None
            for threshold, lock_pct in sorted(config.profit_locks, reverse=True):
                if peak_gain >= threshold:
                    lock_floor = lock_pct
                    break
            if lock_floor is not None:
                min_exit_pct = lock_floor
                if current_gain <= min_exit_pct:
                    exit_premium = current
                    exit_reason = f"profit_lock_{lock_floor:.0f}pct"
                    exit_minute = minutes_held
                    break

        # 4. Velocity exit
        recent_closes.append(current)
        if len(recent_closes) > velocity_window:
            recent_closes.pop(0)

        if config.velocity_drop_pct is not None and len(recent_closes) >= velocity_window:
            window_high = max(recent_closes)
            if window_high > 0:
                drop_pct = (window_high - current) / window_high * 100
                if drop_pct >= config.velocity_drop_pct:
                    exit_premium = current
                    exit_reason = "velocity_exit"
                    exit_minute = minutes_held
                    break

        # 5. Phase trailing stop (with optional time tightening)
        trail_pct = trails.get(phase, trails.get(0, 25.0))

        # Adaptive tightening based on time
        if config.time_tighten_after_minutes is not None:
            if minutes_held > config.time_tighten_after_minutes:
                trail_pct *= config.time_tighten_factor

        if peak > 0:
            drop_from_peak = (peak - current) / peak * 100
            if drop_from_peak >= trail_pct:
                exit_premium = current
                exit_reason = f"phase_trail_p{phase}"
                exit_minute = minutes_held
                break

        # 6. Max hold time
        if config.max_hold_minutes is not None:
            if minutes_held >= config.max_hold_minutes:
                exit_premium = current
                exit_reason = "max_hold"
                exit_minute = minutes_held
                break

    # Compute final metrics
    pnl_pct = (exit_premium - entry) / entry * 100 if entry > 0 else 0
    mfe_capture = (pnl_pct / mfe_pnl_pct * 100) if mfe_pnl_pct > 0 else (0 if pnl_pct <= 0 else 100)

    return SimResult(
        trade_id=setup.trade_id,
        ticker=setup.ticker,
        config_name=config.name,
        entry_premium=entry,
        exit_premium=exit_premium,
        exit_reason=exit_reason,
        exit_minute=exit_minute,
        pnl_pct=pnl_pct,
        mfe_pnl_pct=mfe_pnl_pct,
        mfe_capture_pct=mfe_capture,
        bars_replayed=len(bars),
    )


# ---------------------------------------------------------------------------
# Safeguard configurations to test
# ---------------------------------------------------------------------------


def get_safeguard_configs() -> list[SafeguardConfig]:
    """Return all safeguard configurations to backtest."""
    configs = [
        # Baseline: current Vinny trails, no extras
        SafeguardConfig(name="A_baseline_vinny"),

        # B: Breakeven stop after +30% gain
        SafeguardConfig(
            name="B_breakeven_30",
            breakeven_activation_pct=30.0,
        ),

        # C: Breakeven stop after +50% gain (less aggressive)
        SafeguardConfig(
            name="C_breakeven_50",
            breakeven_activation_pct=50.0,
        ),

        # D: Profit lock ratchet
        SafeguardConfig(
            name="D_profit_lock",
            profit_locks=[(50, 20), (100, 50), (150, 80), (200, 120)],
        ),

        # E: Velocity exit (15% drop in 5 min)
        SafeguardConfig(
            name="E_velocity_15pct_5min",
            velocity_drop_pct=15.0,
            velocity_window_minutes=5,
        ),

        # F: Velocity exit (10% drop in 3 min — tighter)
        SafeguardConfig(
            name="F_velocity_10pct_3min",
            velocity_drop_pct=10.0,
            velocity_window_minutes=3,
        ),

        # G: Time-adaptive tightening (after 60 min, trail × 0.6)
        SafeguardConfig(
            name="G_time_tighten_60min",
            time_tighten_after_minutes=60.0,
            time_tighten_factor=0.6,
        ),

        # H: Tighter base trails
        SafeguardConfig(
            name="H_tight_trails",
            phase_trails={0: 20.0, 1: 15.0, 2: 12.0, 3: 10.0, 4: 8.0, 5: 7.0, 6: 5.0},
        ),

        # I: Max hold 90 min
        SafeguardConfig(
            name="I_max_hold_90min",
            max_hold_minutes=90.0,
        ),

        # J: Max hold 120 min
        SafeguardConfig(
            name="J_max_hold_120min",
            max_hold_minutes=120.0,
        ),

        # K: Combined "Vinny+" — breakeven + profit lock + velocity + time tighten
        SafeguardConfig(
            name="K_vinny_plus",
            breakeven_activation_pct=40.0,
            profit_locks=[(80, 30), (150, 70), (250, 150)],
            velocity_drop_pct=12.0,
            velocity_window_minutes=4,
            time_tighten_after_minutes=60.0,
            time_tighten_factor=0.7,
        ),

        # L: Aggressive combined — tighter everything
        SafeguardConfig(
            name="L_aggressive",
            breakeven_activation_pct=25.0,
            profit_locks=[(40, 15), (80, 40), (120, 70), (200, 120)],
            velocity_drop_pct=10.0,
            velocity_window_minutes=3,
            phase_trails={0: 20.0, 1: 15.0, 2: 12.0, 3: 10.0, 4: 8.0, 5: 6.0, 6: 5.0},
            time_tighten_after_minutes=45.0,
            time_tighten_factor=0.6,
        ),

        # M: Conservative — wider trails, only breakeven + slow tighten
        SafeguardConfig(
            name="M_conservative",
            breakeven_activation_pct=60.0,
            profit_locks=[(100, 40), (200, 100)],
            time_tighten_after_minutes=90.0,
            time_tighten_factor=0.75,
        ),
    ]
    return configs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_option_bars(
    db_path: str,
    contract_ticker: str,
) -> list[BarData]:
    """Load 1-min bars for a specific option contract from historical DB."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM option_bars WHERE contract_ticker = ? ORDER BY timestamp",
        (contract_ticker,),
    )
    bars = [
        BarData(
            timestamp=row[0], open=row[1], high=row[2],
            low=row[3], close=row[4], volume=row[5] or 0,
        )
        for row in cursor.fetchall()
    ]
    conn.close()
    return bars


def load_all_bars_for_date(
    db_path: str,
    date: str,
    ticker: str,
    option_type: str = "call",
) -> tuple[list[BarData], str]:
    """Load ATM bars for a ticker on a given date.

    Returns (bars, contract_ticker).
    """
    conn = sqlite3.connect(db_path)
    # Find the ATM contract for this date
    col = "atm_call_ticker" if option_type == "call" else "atm_put_ticker"
    cursor = conn.execute(
        f"SELECT {col} FROM trading_days WHERE date = ? AND ticker = ?",
        (date, ticker),
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        conn.close()
        return [], ""

    contract_ticker = row[0]
    bars = load_option_bars(db_path, contract_ticker)
    conn.close()
    return bars, contract_ticker


def load_paper_trade_setups(db_path: str) -> list[TradeSetup]:
    """Load closed root trades from the paper trading DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("""
        SELECT id, ticker, direction, premium_per_contract, opened_at,
               contracts, total_cost, target_1, target_2, stop_price,
               exit_premium, exit_reason, pnl_pct, mfe_pnl_pct
        FROM paper_trades
        WHERE status = 'closed' AND parent_trade_id IS NULL
        ORDER BY opened_at
    """)
    setups = []
    for row in cursor.fetchall():
        setups.append(TradeSetup(
            trade_id=row["id"],
            ticker=row["ticker"],
            direction=row["direction"],
            entry_premium=row["premium_per_contract"],
            entry_time=row["opened_at"],
            contracts=row["contracts"],
            total_cost=row["total_cost"],
            target_1=row["target_1"],
            target_2=row["target_2"],
            stop_price=row["stop_price"],
            actual_exit_premium=row["exit_premium"] or 0,
            actual_exit_reason=row["exit_reason"] or "unknown",
            actual_pnl_pct=row["pnl_pct"] or 0,
            actual_mfe_pnl_pct=row["mfe_pnl_pct"] or 0,
        ))
    conn.close()
    return setups


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------


def run_safeguard_backtest(
    historical_db: str,
    paper_db: str,
    configs: list[SafeguardConfig] | None = None,
) -> dict[str, list[SimResult]]:
    """Run the safeguard backtest across all configs and all historical ATM bars.

    Uses historical option bars (not paper trades) for broad signal replay.
    Returns {config_name: [SimResult, ...]}.
    """
    if configs is None:
        configs = get_safeguard_configs()

    # Load all trading days from historical DB
    conn = sqlite3.connect(historical_db)
    cursor = conn.execute(
        "SELECT date, ticker, atm_call_ticker, atm_put_ticker, "
        "atm_strike, open_price FROM trading_days "
        "WHERE call_bars > 30 ORDER BY date"
    )
    trading_days = cursor.fetchall()
    conn.close()

    results: dict[str, list[SimResult]] = {c.name: [] for c in configs}
    len(trading_days)

    for idx, (date, ticker, call_contract, put_contract, atm_strike, open_price) in enumerate(trading_days):
        # Use ATM call bars (most liquid, typical 0DTE play)
        if not call_contract:
            continue

        bars = load_option_bars(historical_db, call_contract)
        if len(bars) < 30:
            continue

        # Simulate entry at bar 15 (~15 min after open, realistic entry timing)
        entry_bar_idx = 15
        if entry_bar_idx >= len(bars):
            continue

        entry_premium = bars[entry_bar_idx].close
        if entry_premium <= 0.01:
            continue

        # Create a synthetic trade setup
        setup = TradeSetup(
            trade_id=idx,
            ticker=ticker,
            direction="call",
            entry_premium=entry_premium,
            entry_time=date,
            contracts=1,
            total_cost=entry_premium * 100,
            target_1=None,
            target_2=None,
            stop_price=None,
            actual_exit_premium=bars[-1].close,
            actual_exit_reason="eod",
            actual_pnl_pct=(bars[-1].close - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0,
            actual_mfe_pnl_pct=0,
        )

        # Run through each config using bars from entry onward
        trade_bars = bars[entry_bar_idx + 1:]  # bars after entry
        for config in configs:
            result = simulate_trade(trade_bars, setup, config)
            results[config.name].append(result)

    return results


def run_paper_trade_replay(
    historical_db: str,
    paper_db: str,
    configs: list[SafeguardConfig] | None = None,
) -> dict[str, list[SimResult]]:
    """Replay actual paper trades through historical bars with different safeguards.

    This is a more targeted backtest using the real trades that were executed.
    """
    if configs is None:
        configs = get_safeguard_configs()

    setups = load_paper_trade_setups(paper_db)
    results: dict[str, list[SimResult]] = {c.name: [] for c in configs}

    conn = sqlite3.connect(historical_db)
    # Get available tickers in historical DB
    cursor = conn.execute("SELECT DISTINCT ticker FROM trading_days")
    available_tickers = {row[0] for row in cursor.fetchall()}
    conn.close()

    for setup in setups:
        ticker = setup.ticker
        if ticker not in available_tickers:
            continue

        # Find the trading date from entry_time
        try:
            dt = datetime.fromisoformat(setup.entry_time)
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        # Load ATM bars for this date
        option_type = "call" if setup.direction == "call" else "put"
        bars, contract = load_all_bars_for_date(
            historical_db, date_str, ticker, option_type,
        )
        if not bars or len(bars) < 10:
            continue

        # Scale bars to match entry premium (historical bars have different absolute values)
        # Use ratio scaling: first bar close → entry premium
        scale_bar = bars[0]
        if scale_bar.close <= 0:
            continue
        scale_factor = setup.entry_premium / scale_bar.close

        scaled_bars = [
            BarData(
                timestamp=b.timestamp,
                open=b.open * scale_factor,
                high=b.high * scale_factor,
                low=b.low * scale_factor,
                close=b.close * scale_factor,
                volume=b.volume,
            )
            for b in bars
        ]

        for config in configs:
            result = simulate_trade(scaled_bars, setup, config)
            results[config.name].append(result)

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def format_comparison_report(
    results: dict[str, list[SimResult]],
    title: str = "Safeguard Backtest Results",
) -> str:
    """Format a comparison report across all safeguard configurations."""
    lines = [
        "=" * 80,
        f"  {title}",
        "=" * 80,
        "",
    ]

    # Summary table
    header = (
        f"{'Config':<25} {'Trades':>6} {'Win%':>6} {'AvgPnL':>8} "
        f"{'AvgMFE':>8} {'MFEcap':>7} {'BestPnL':>8} {'WorstPnL':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    config_stats = {}

    for config_name, trades in sorted(results.items()):
        if not trades:
            continue

        total = len(trades)
        wins = sum(1 for t in trades if t.pnl_pct > 0)
        win_rate = wins / total * 100 if total > 0 else 0

        avg_pnl = sum(t.pnl_pct for t in trades) / total
        avg_mfe = sum(t.mfe_pnl_pct for t in trades) / total
        avg_capture = sum(t.mfe_capture_pct for t in trades) / total

        best = max(t.pnl_pct for t in trades)
        worst = min(t.pnl_pct for t in trades)

        total_pnl = sum(t.pnl_pct for t in trades)

        config_stats[config_name] = {
            "total": total, "wins": wins, "win_rate": win_rate,
            "avg_pnl": avg_pnl, "avg_mfe": avg_mfe, "avg_capture": avg_capture,
            "best": best, "worst": worst, "total_pnl": total_pnl,
        }

        lines.append(
            f"{config_name:<25} {total:>6} {win_rate:>5.1f}% {avg_pnl:>+7.1f}% "
            f"{avg_mfe:>+7.1f}% {avg_capture:>6.1f}% {best:>+7.1f}% {worst:>+8.1f}%"
        )

    lines.append("")

    # Find the best config by total PnL
    if config_stats:
        best_config = max(config_stats, key=lambda k: config_stats[k]["total_pnl"])
        baseline_pnl = config_stats.get("A_baseline_vinny", {}).get("total_pnl", 0)
        best_pnl = config_stats[best_config]["total_pnl"]

        lines.append(f"  Best config: {best_config}")
        lines.append(f"  Best total PnL:  {best_pnl:+.1f}%")
        lines.append(f"  Baseline PnL:    {baseline_pnl:+.1f}%")
        if baseline_pnl != 0:
            improvement = (best_pnl - baseline_pnl) / abs(baseline_pnl) * 100
            lines.append(f"  Improvement:     {improvement:+.1f}%")

        lines.append("")
        lines.append("  MFE Capture Ranking (higher = better at locking in gains):")
        ranked = sorted(config_stats.items(), key=lambda x: x[1]["avg_capture"], reverse=True)
        for i, (name, stats) in enumerate(ranked, 1):
            lines.append(f"    {i}. {name:<25} {stats['avg_capture']:.1f}%")

    lines.append("")
    lines.append("=" * 80)

    # Detail per-trade breakdown for top 3 configs
    ranked_by_pnl = sorted(config_stats.items(), key=lambda x: x[1]["total_pnl"], reverse=True)[:3]
    for config_name, _ in ranked_by_pnl:
        trades = results[config_name]
        lines.append("")
        lines.append(f"  --- {config_name} Trade Details ---")
        lines.append(f"  {'Ticker':<6} {'Entry':>7} {'Exit':>7} {'PnL%':>7} {'MFE%':>7} {'Cap%':>6} {'Min':>4} {'Reason':<20}")

        for t in trades:
            lines.append(
                f"  {t.ticker:<6} ${t.entry_premium:>5.2f} ${t.exit_premium:>5.2f} "
                f"{t.pnl_pct:>+6.1f}% {t.mfe_pnl_pct:>+6.1f}% {t.mfe_capture_pct:>5.1f}% "
                f"{t.exit_minute:>4} {t.exit_reason:<20}"
            )

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_momentum_filtered_backtest(
    historical_db: str,
    configs: list[SafeguardConfig] | None = None,
) -> dict[str, list[SimResult]]:
    """Run backtest with momentum-filtered entry — simulates Vinny's high-conviction signals.

    Only enters when the first 10 bars show upward momentum (premium up >5%
    from open), mimicking how Discord signals tend to fire on momentum moves.
    This more accurately represents the ~90% win rate signal quality.
    """
    if configs is None:
        configs = get_safeguard_configs()

    conn = sqlite3.connect(historical_db)
    cursor = conn.execute(
        "SELECT date, ticker, atm_call_ticker, atm_put_ticker, "
        "atm_strike, open_price FROM trading_days "
        "WHERE call_bars > 60 ORDER BY date"
    )
    trading_days = cursor.fetchall()
    conn.close()

    results: dict[str, list[SimResult]] = {c.name: [] for c in configs}
    entered = 0
    skipped = 0

    for idx, (date, ticker, call_contract, put_contract, atm_strike, open_price) in enumerate(trading_days):
        if not call_contract:
            continue

        bars = load_option_bars(historical_db, call_contract)
        if len(bars) < 60:
            continue

        # Check first 10 bars for momentum (simulates signal quality filter)
        # Entry at bar 10, only if premium rose >5% from bar 0
        open_premium = bars[0].close
        if open_premium <= 0.01:
            continue

        check_premium = bars[10].close
        early_gain = (check_premium - open_premium) / open_premium * 100

        # Only enter on momentum days (simulates high-score signal)
        if early_gain < 5.0:
            skipped += 1
            continue

        entered += 1
        entry_premium = bars[10].close

        setup = TradeSetup(
            trade_id=idx,
            ticker=ticker,
            direction="call",
            entry_premium=entry_premium,
            entry_time=date,
            contracts=1,
            total_cost=entry_premium * 100,
            target_1=None,
            target_2=None,
            stop_price=None,
            actual_exit_premium=bars[-1].close,
            actual_exit_reason="eod",
            actual_pnl_pct=(bars[-1].close - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0,
            actual_mfe_pnl_pct=0,
        )

        trade_bars = bars[11:]
        for config in configs:
            result = simulate_trade(trade_bars, setup, config)
            results[config.name].append(result)

    return results


def main():
    """Run the full safeguard backtest and print results."""
    project_root = Path(__file__).resolve().parent.parent.parent
    historical_db = str(project_root / "journal" / "historical_0dte.db")
    paper_db = str(project_root / "journal" / "raw_messages.db")

    configs = get_safeguard_configs()

    # 1. Random-entry broad backtest (all days)
    print("=" * 80)
    print("  TEST 1: ALL-DAY ENTRIES (random entry at bar 15)")
    print("=" * 80)
    results = run_safeguard_backtest(historical_db, paper_db, configs)
    total_trades = sum(len(v) for v in results.values())
    print(f"  {total_trades} trade-config combos across {len(results.get('A_baseline_vinny', []))} days")
    report = format_comparison_report(results, "All-Day Entry Backtest (1,815 days × 13 configs)")
    print(report)

    # 2. Momentum-filtered entries (simulates high-conviction signals)
    print("\n")
    print("=" * 80)
    print("  TEST 2: MOMENTUM-FILTERED ENTRIES (early +5% gain = signal quality)")
    print("=" * 80)
    momentum_results = run_momentum_filtered_backtest(historical_db, configs)
    mom_trades = len(momentum_results.get("A_baseline_vinny", []))
    print(f"  {mom_trades} momentum-qualified days")
    mom_report = format_comparison_report(
        momentum_results,
        f"Momentum-Filtered Backtest ({mom_trades} qualifying days × 13 configs)",
    )
    print(mom_report)

    # 3. Paper trade replay
    print("\n")
    print("=" * 80)
    print("  TEST 3: PAPER TRADE REPLAY (actual OptionsOwl trades)")
    print("=" * 80)
    replay_results = run_paper_trade_replay(historical_db, paper_db, configs)
    replay_total = sum(len(v) for v in replay_results.values())
    if replay_total > 0:
        replay_report = format_comparison_report(replay_results, "Paper Trade Replay Backtest")
        print(replay_report)
    else:
        print("  No matching paper trades found in historical data.")

    return results, momentum_results, replay_results


if __name__ == "__main__":
    main()
