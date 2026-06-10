"""Backtest smart dip-confirm v3 — support/resistance aware.

Instead of a blind timer, this checks whether the underlying is:
1. Near support (recent 5m lows, VWAP) and bouncing → WAIT for bounce confirm → enter cheaper
2. Breaking support → SKIP (trade is going against us, no floor)
3. At/above VWAP with premium fading → ENTER (likely just spread decay, not trend)

Uses 5-minute candle data from the harvester DB to compute support levels
and check whether the underlying bounced or broke through.

Usage:
    python scripts/backtest_dip_confirm_v3.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

HARVESTER_DB = Path("journal/owlet-harvester/options_data.db")
BOT_DBS = {
    "kody": Path("journal/owlet-kody/raw_messages.db"),
    "adam": Path("journal/owlet-adam/raw_messages.db"),
    "vinny": Path("journal/owlet-vinny/raw_messages.db"),
    "yank": Path("journal/owlet-yank/raw_messages.db"),
}


@dataclass
class Trade:
    trade_id: int
    bot: str
    ticker: str
    option_type: str  # call or put
    strike: float
    signal_premium: float
    entry_premium: float
    opened_at: str
    closed_at: str
    exit_reason: str
    pnl_dollars: float
    pnl_pct: float
    contracts: int
    score: int


@dataclass
class SupportContext:
    """Support/resistance context at time of entry."""
    underlying_price: float
    vwap: float | None
    # Recent 5m candle support levels
    recent_low_5m: float | None     # lowest low of last 3 5m bars
    recent_low_15m: float | None    # lowest low of last 2 15m bars
    atr_5m: float | None
    rsi_5m: float | None
    # What happened next (60-180s after entry)
    price_60s: float | None
    price_120s: float | None
    price_180s: float | None
    # Premium at those times
    premium_60s: float | None
    premium_120s: float | None
    premium_180s: float | None

    @property
    def near_support(self) -> bool:
        """Is the underlying within 0.3% of recent support (5m lows)?"""
        if self.recent_low_5m is None or self.underlying_price <= 0:
            return False
        dist = (self.underlying_price - self.recent_low_5m) / self.underlying_price * 100
        return dist < 0.3  # within 0.3% of support

    @property
    def above_vwap(self) -> bool:
        """Is the underlying above VWAP?"""
        if self.vwap is None:
            return True  # assume neutral
        return self.underlying_price >= self.vwap

    @property
    def distance_to_support_pct(self) -> float | None:
        """Distance from current price to recent 5m low as %."""
        if self.recent_low_5m is None or self.underlying_price <= 0:
            return None
        return (self.underlying_price - self.recent_low_5m) / self.underlying_price * 100

    @property
    def broke_support_60s(self) -> bool:
        """Did price break below 5m support within 60s?"""
        if self.recent_low_5m is None or self.price_60s is None:
            return False
        return self.price_60s < self.recent_low_5m

    @property
    def bounced_from_support(self) -> bool:
        """Did price touch/approach support and then bounce within 120s?"""
        if self.recent_low_5m is None:
            return False
        # Price approached support (within 0.2%)
        approached = self.near_support
        if not approached:
            return False
        # And then moved back up within 120s
        if self.price_120s and self.price_120s > self.underlying_price:
            return True
        if self.price_60s and self.price_60s > self.underlying_price:
            return True
        return False

    @property
    def premium_recovered_120s(self) -> bool:
        """Did premium recover (go above entry) within 120s?"""
        return self.premium_120s is not None and self.premium_120s > 0

    @property
    def is_fading(self) -> bool:
        """Is premium fading within 60s? (> 1% drop)"""
        if self.premium_60s is None:
            return False
        return self.premium_60s < 0  # negative means premium dropped


def load_trades() -> list[Trade]:
    trades = []
    for bot, db_path in BOT_DBS.items():
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, ticker, option_type, strike,
                   premium_per_contract, webull_entry_fill_price, signal_premium,
                   opened_at, closed_at, exit_reason,
                   pnl_dollars, pnl_pct, contracts, score
            FROM paper_trades
            WHERE status='closed' AND webull_order_id IS NOT NULL
            ORDER BY id
        """).fetchall()
        conn.close()
        for r in rows:
            trades.append(Trade(
                trade_id=r["id"],
                bot=bot,
                ticker=r["ticker"],
                option_type=r["option_type"],
                strike=r["strike"],
                signal_premium=r["signal_premium"] or r["premium_per_contract"],
                entry_premium=r["webull_entry_fill_price"] or r["premium_per_contract"],
                opened_at=r["opened_at"],
                closed_at=r["closed_at"],
                exit_reason=r["exit_reason"] or "unknown",
                pnl_dollars=r["pnl_dollars"] or 0,
                pnl_pct=r["pnl_pct"] or 0,
                contracts=r["contracts"] or 1,
                score=r["score"] or 80,
            ))
    return trades


def _ensure_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def get_support_context(
    conn: sqlite3.Connection,
    trade: Trade,
) -> SupportContext | None:
    """Build support/resistance context from harvester data around entry time."""
    entry_time = _ensure_tz(datetime.fromisoformat(trade.opened_at))
    entry_date = entry_time.strftime("%Y-%m-%d")

    # 1. Get underlying price snapshots (use any ATM contract to get underlying_price)
    # Find a contract for this ticker near ATM
    contract = conn.execute(
        "SELECT contract_ticker FROM harvest_contracts "
        "WHERE underlying=? AND option_type=? AND expiry_date>=? "
        "ORDER BY ABS(strike - ?), expiry_date LIMIT 1",
        (trade.ticker, trade.option_type, entry_date, trade.strike),
    ).fetchone()
    if not contract:
        return None

    ct = contract[0]

    # Get snapshots around entry: 60 min before (for support levels) and 3 min after
    t_start = (entry_time - timedelta(minutes=60)).isoformat()
    t_end = (entry_time + timedelta(minutes=3)).isoformat()

    rows = conn.execute(
        "SELECT captured_at, underlying_price, ask, bid, midpoint "
        "FROM harvest_snapshots "
        "WHERE contract_ticker=? AND captured_at BETWEEN ? AND ? "
        "ORDER BY captured_at",
        (ct, t_start, t_end),
    ).fetchall()

    if not rows:
        return None

    # Split into before-entry and after-entry
    before = []
    after = []
    for row in rows:
        cap = _ensure_tz(datetime.fromisoformat(row[0]))
        offset = (cap - entry_time).total_seconds()
        u_price = row[1]
        premium = row[2] if row[2] and row[2] > 0 else row[4]  # ask, fallback mid
        if offset <= 5:
            before.append((offset, u_price, premium))
        else:
            after.append((offset, u_price, premium))

    if not before:
        return None

    # Current underlying price (closest to entry)
    underlying_price = before[-1][1]

    # Build "5-minute bars" from the per-minute snapshots (last 60 min)
    # Each "bar" is underlying prices in a 5-min window
    bar_lows = []
    bar_vwaps = []
    for row in rows:
        cap = _ensure_tz(datetime.fromisoformat(row[0]))
        offset = (cap - entry_time).total_seconds()
        if offset > 0:
            break
        bar_lows.append(row[1])
        # VWAP approximation: use the underlying price (not ideal but workable)

    # Recent support: lowest underlying price in last N snapshots (roughly last 15-20 min)
    if len(bar_lows) >= 3:
        recent_low_5m = min(bar_lows[-6:]) if len(bar_lows) >= 6 else min(bar_lows[-3:])
        recent_low_15m = min(bar_lows[-15:]) if len(bar_lows) >= 15 else min(bar_lows)
    else:
        recent_low_5m = None
        recent_low_15m = None

    # VWAP: use the snapshot vwap if available, otherwise approximate
    # (harvester doesn't store underlying VWAP, so use mean of all prices)
    vwap = sum(bar_lows) / len(bar_lows) if bar_lows else None

    # ATR approximation from the snapshot data
    if len(bar_lows) >= 10:
        ranges = [abs(bar_lows[i] - bar_lows[i-1]) for i in range(1, len(bar_lows))]
        atr_5m = sum(ranges[-14:]) / min(14, len(ranges[-14:]))
    else:
        atr_5m = None

    # RSI approximation
    rsi_5m = None  # would need proper candle bars, skip for now

    # After-entry prices
    def price_at(target_sec: float, data: list) -> float | None:
        best = None
        best_dist = float("inf")
        for offset, u_price, _ in data:
            dist = abs(offset - target_sec)
            if dist < best_dist:
                best = u_price
                best_dist = dist
        return best if best_dist < 60 else None

    def premium_at(target_sec: float, data: list) -> float | None:
        best = None
        best_dist = float("inf")
        for offset, _, prem in data:
            dist = abs(offset - target_sec)
            if dist < best_dist:
                best = prem
                best_dist = dist
        return best if best_dist < 60 else None

    # Premium change (relative to entry) — positive means gain, negative means loss
    entry_prem = before[-1][2] if before else None
    def prem_change(target_sec: float) -> float | None:
        p = premium_at(target_sec, after)
        if p is None or entry_prem is None or entry_prem <= 0:
            return None
        return (p - entry_prem) / entry_prem * 100

    return SupportContext(
        underlying_price=underlying_price,
        vwap=vwap,
        recent_low_5m=recent_low_5m,
        recent_low_15m=recent_low_15m,
        atr_5m=atr_5m,
        rsi_5m=rsi_5m,
        price_60s=price_at(60, after),
        price_120s=price_at(120, after),
        price_180s=price_at(180, after),
        premium_60s=prem_change(60),
        premium_120s=prem_change(120),
        premium_180s=prem_change(180),
    )


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

def strategy_baseline(trade: Trade, ctx: SupportContext | None) -> tuple[bool, str]:
    """No dip confirm — always enter."""
    return True, "baseline"


def strategy_current_dip_confirm(trade: Trade, ctx: SupportContext | None) -> tuple[bool, str]:
    """Current dumb timer: if premium fading, poll 3×5s for uptick, skip if none."""
    if ctx is None:
        return True, "no_data"
    if ctx.premium_60s is None or ctx.premium_60s >= -1.0:
        return True, "not_fading"
    # Premium is fading > 1%. Current logic: wait 15s for uptick.
    # With 60s harvester resolution, we can't see 15s changes.
    # Approximate: if premium recovered by 120s, assume uptick found.
    if ctx.premium_120s is not None and ctx.premium_120s > ctx.premium_60s:
        return True, "uptick_found"
    return False, "no_uptick"


def strategy_support_bounce(trade: Trade, ctx: SupportContext | None) -> tuple[bool, str]:
    """Smart: use support levels to decide.

    Decision tree:
    1. Premium NOT fading → ENTER (no problem)
    2. Premium fading, underlying ABOVE VWAP → ENTER (premium decay, not trend break)
    3. Premium fading, near support & bouncing → ENTER at better price
    4. Premium fading, BROKE support → SKIP (breakdown in progress)
    5. Premium fading, far from support → SKIP (no floor to catch it)
    """
    if ctx is None:
        return True, "no_data"

    # Not fading? Enter.
    if ctx.premium_60s is None or ctx.premium_60s >= -1.0:
        return True, "not_fading"

    is_call = trade.option_type == "call"

    # For calls: we care about underlying dropping. For puts: underlying rising.
    if is_call:
        # Underlying above VWAP? Premium fade is likely just spread/theta, not trend.
        if ctx.above_vwap:
            return True, "above_vwap"

        # Near support and bouncing?
        if ctx.near_support and ctx.bounced_from_support:
            return True, "support_bounce"

        # Broke through support? Skip.
        if ctx.broke_support_60s:
            return False, "broke_support"

        # Far from support and fading? Skip.
        dist = ctx.distance_to_support_pct
        if dist is not None and dist > 0.5:
            # Price is well above support but still fading — unclear
            # Check if it recovered by 120s
            if ctx.premium_120s is not None and ctx.premium_120s > 0:
                return True, "recovered_mid"
            return False, "fading_no_support"

        # Near support but not bouncing yet — check 120s recovery
        if ctx.premium_120s is not None and ctx.premium_120s > 0:
            return True, "recovered_120s"

        return False, "fading_no_recovery"

    else:
        # Put — mirror logic (resistance instead of support)
        # For puts, we want underlying to go DOWN, so "support" is actually resistance
        if not ctx.above_vwap:  # below VWAP is good for puts
            return True, "below_vwap"
        if ctx.premium_120s is not None and ctx.premium_120s > 0:
            return True, "recovered_120s"
        return False, "fading_put"


def strategy_support_atr(trade: Trade, ctx: SupportContext | None) -> tuple[bool, str]:
    """ATR-aware: skip if price is more than 0.5 ATR from support.

    Same as support_bounce but uses ATR to measure "near" support instead of fixed %.
    """
    if ctx is None:
        return True, "no_data"
    if ctx.premium_60s is None or ctx.premium_60s >= -1.0:
        return True, "not_fading"

    is_call = trade.option_type == "call"
    if not is_call:
        # Puts: simple check
        if not ctx.above_vwap:
            return True, "below_vwap_put"
        if ctx.premium_120s is not None and ctx.premium_120s > 0:
            return True, "recovered_put"
        return False, "fading_put"

    # Calls
    if ctx.above_vwap:
        return True, "above_vwap"

    if ctx.atr_5m and ctx.recent_low_5m and ctx.underlying_price:
        dist_to_support = ctx.underlying_price - ctx.recent_low_5m
        if dist_to_support < 0.5 * ctx.atr_5m:
            # Within half an ATR of support — worth waiting
            if ctx.bounced_from_support:
                return True, "atr_bounce"
            if ctx.premium_120s is not None and ctx.premium_120s > 0:
                return True, "atr_recovered"
            return False, "atr_no_bounce"
        else:
            # More than 0.5 ATR from support — in no-man's land
            return False, "far_from_support"

    # Fallback
    if ctx.premium_120s is not None and ctx.premium_120s > 0:
        return True, "recovered_fallback"
    return False, "no_data_skip"


def strategy_vwap_only(trade: Trade, ctx: SupportContext | None) -> tuple[bool, str]:
    """Simplest smart version: only skip fading trades BELOW VWAP.

    Logic: if premium is fading AND underlying is below VWAP → bearish structure → skip.
    If above VWAP → premium fade is temporary → enter.
    """
    if ctx is None:
        return True, "no_data"
    if ctx.premium_60s is None or ctx.premium_60s >= -1.0:
        return True, "not_fading"

    is_call = trade.option_type == "call"

    if is_call:
        if ctx.above_vwap:
            return True, "above_vwap"
        return False, "below_vwap_fading"
    else:
        if not ctx.above_vwap:
            return True, "below_vwap"
        return False, "above_vwap_fading"


def strategy_vwap_with_recovery(trade: Trade, ctx: SupportContext | None) -> tuple[bool, str]:
    """VWAP + recovery check: skip fading below VWAP, UNLESS premium recovers in 120s.

    Catches the "fading but near support bounce" trades that VWAP-only misses.
    """
    if ctx is None:
        return True, "no_data"
    if ctx.premium_60s is None or ctx.premium_60s >= -1.0:
        return True, "not_fading"

    is_call = trade.option_type == "call"

    if is_call:
        if ctx.above_vwap:
            return True, "above_vwap"
        # Below VWAP and fading — check for recovery
        if ctx.premium_120s is not None and ctx.premium_120s > 0:
            return True, "below_vwap_recovered"
        return False, "below_vwap_no_recovery"
    else:
        if not ctx.above_vwap:
            return True, "below_vwap_put"
        if ctx.premium_120s is not None and ctx.premium_120s > 0:
            return True, "above_vwap_recovered"
        return False, "above_vwap_no_recovery"


def strategy_support_distance(trade: Trade, ctx: SupportContext | None) -> tuple[bool, str]:
    """Distance-to-support: enter if near support (<0.3%), skip if far (>0.3%).

    For fading trades only. Non-fading always enter.
    """
    if ctx is None:
        return True, "no_data"
    if ctx.premium_60s is None or ctx.premium_60s >= -1.0:
        return True, "not_fading"

    is_call = trade.option_type == "call"
    if not is_call:
        # Puts — always enter for now (mirror would need resistance levels)
        return True, "put_enter"

    dist = ctx.distance_to_support_pct
    if dist is None:
        return True, "no_support_data"

    if dist < 0.3:
        return True, "near_support"
    elif dist < 0.6:
        # Middle zone — check recovery
        if ctx.premium_120s is not None and ctx.premium_120s > 0:
            return True, "mid_recovered"
        return False, "mid_no_recovery"
    else:
        return False, "far_from_support"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRATEGIES = [
    ("Baseline (no filter)", strategy_baseline),
    ("Current dip-confirm (timer)", strategy_current_dip_confirm),
    ("VWAP only", strategy_vwap_only),
    ("VWAP + 120s recovery", strategy_vwap_with_recovery),
    ("Support distance (<0.3%)", strategy_support_distance),
    ("Support bounce", strategy_support_bounce),
    ("Support ATR-aware", strategy_support_atr),
]


def main() -> None:
    if not HARVESTER_DB.exists():
        print(f"ERROR: {HARVESTER_DB} not found")
        sys.exit(1)

    trades = load_trades()
    print(f"Loaded {len(trades)} trades\n")

    conn = sqlite3.connect(str(HARVESTER_DB))

    # Build context for each trade
    print("Computing support/resistance context for each trade...")
    trade_contexts: list[tuple[Trade, SupportContext | None]] = []
    n_ctx = 0
    for i, t in enumerate(trades):
        ctx = get_support_context(conn, t)
        trade_contexts.append((t, ctx))
        if ctx:
            n_ctx += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(trades)}...")

    conn.close()
    print(f"  Context available for {n_ctx}/{len(trades)} trades\n")

    # Run strategies
    baseline_pnl = sum(t.pnl_dollars for t in trades)

    print(f"{'='*120}")
    print(f"SMART DIP-CONFIRM BACKTEST — {len(trades)} trades, baseline P&L ${baseline_pnl:+,.0f}")
    print(f"{'='*120}\n")

    print(f"{'Strategy':<35} {'Enter':>6} {'Skip':>5} {'PnL':>10} {'vs Base':>10} "
          f"{'WR':>6} {'Skip$Bad':>10} {'Skip$Good':>10} {'Net Skip':>10}")
    print("-" * 112)

    for label, strategy_fn in STRATEGIES:
        entered = 0
        skipped = 0
        total_pnl = 0.0
        winners = 0
        skip_bad = 0.0
        skip_good = 0.0
        reasons: dict[str, int] = {}

        for trade, ctx in trade_contexts:
            should_enter, reason = strategy_fn(trade, ctx)
            reasons[reason] = reasons.get(reason, 0) + 1

            if should_enter:
                entered += 1
                total_pnl += trade.pnl_dollars
                if trade.pnl_dollars > 0:
                    winners += 1
            else:
                skipped += 1
                if trade.pnl_dollars < 0:
                    skip_bad += trade.pnl_dollars
                else:
                    skip_good += trade.pnl_dollars

        wr = winners / entered * 100 if entered > 0 else 0
        delta = total_pnl - baseline_pnl
        net_skip = abs(skip_bad) - skip_good  # positive = net benefit from skipping

        print(f"{label:<35} {entered:>6} {skipped:>5} ${total_pnl:>+9,.0f} ${delta:>+9,.0f} "
              f"{wr:>5.1f}% ${skip_bad:>+9,.0f} ${skip_good:>+9,.0f} ${net_skip:>+9,.0f}")

    print()
    print("Legend:")
    print("  Skip$Bad  = $ of losers we avoided (higher absolute = better)")
    print("  Skip$Good = $ of winners we missed (lower = better)")
    print("  Net Skip  = |Skip$Bad| - Skip$Good (positive = skipping helps)")
    print()

    # Detail breakdown for best strategies
    for label, strategy_fn in STRATEGIES[2:]:  # skip baseline and current
        print(f"\n{'='*80}")
        print(f"DETAIL: {label}")
        print(f"{'='*80}")

        reasons_enter: dict[str, list[float]] = {}
        reasons_skip: dict[str, list[float]] = {}

        for trade, ctx in trade_contexts:
            should_enter, reason = strategy_fn(trade, ctx)
            if should_enter:
                reasons_enter.setdefault(reason, []).append(trade.pnl_dollars)
            else:
                reasons_skip.setdefault(reason, []).append(trade.pnl_dollars)

        print(f"\n  ENTER reasons:")
        for reason, pnls in sorted(reasons_enter.items(), key=lambda x: -sum(x[1])):
            print(f"    {reason:<30} {len(pnls):>4} trades, P&L ${sum(pnls):>+9,.0f}, "
                  f"WR {sum(1 for p in pnls if p > 0)/len(pnls)*100:.0f}%")

        print(f"\n  SKIP reasons:")
        for reason, pnls in sorted(reasons_skip.items(), key=lambda x: sum(x[1])):
            win = sum(1 for p in pnls if p > 0)
            loss = sum(1 for p in pnls if p <= 0)
            print(f"    {reason:<30} {len(pnls):>4} trades, P&L ${sum(pnls):>+9,.0f} "
                  f"(would-be: {win}W {loss}L)")


if __name__ == "__main__":
    main()
