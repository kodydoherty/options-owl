"""Backtesting engine — replay historical signals against a simulated portfolio."""

from __future__ import annotations

import math

import aiosqlite
from pydantic import BaseModel


class BacktestConfig(BaseModel):
    """Configuration for a backtest run."""

    starting_balance: float = 5000.0
    max_position_pct: float = 5.0
    max_concurrent: int = 3
    min_score: int = 75

    start_date: str | None = None  # YYYY-MM-DD
    end_date: str | None = None

    enable_kelly: bool = False
    enable_circuit_breakers: bool = False
    enable_spreads: bool = False


class TradeDetail(BaseModel):
    """Record of a single simulated trade."""

    signal_id: int
    ticker: str
    direction: str
    score: int
    bot_source: str
    contracts: int
    entry_premium: float
    exit_premium: float
    total_cost: float
    proceeds: float
    pnl_dollars: float
    pnl_pct: float
    outcome: str
    balance_after: float
    opened_at: str
    closed_at: str


class BacktestResult(BaseModel):
    """Aggregated results from a backtest run."""

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0

    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0

    max_drawdown_pct: float = 0.0
    max_drawdown_dollars: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0

    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0

    best_trade: float = 0.0
    worst_trade: float = 0.0

    daily_pnl_series: list[tuple[str, float]] = []
    trades: list[TradeDetail] = []


class BacktestEngine:
    """Replays historical signals through a simulated portfolio."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.balance = config.starting_balance
        self.peak_balance = config.starting_balance
        self.max_drawdown_dollars = 0.0
        self.max_drawdown_pct = 0.0

        self._trades: list[TradeDetail] = []
        self._open_positions: list[dict] = []  # tracks concurrent open trades
        self._daily_pnl: dict[str, float] = {}  # date -> cumulative pnl
        self._trade_returns: list[float] = []  # per-trade pnl_pct for metrics

    def run(self, signals: list[dict]) -> BacktestResult:
        """Simulate trading through a list of historical signals sequentially.

        Each signal dict is expected to have fields from trade_signals joined
        with signal_outcomes (outcome, pnl_atm_est or pnl_underlying_pct, etc.).
        """
        for sig in signals:
            self._process_signal(sig)

        return self.compute_metrics()

    def _process_signal(self, sig: dict) -> None:
        """Evaluate and potentially trade a single signal."""
        score = sig.get("score", 0)
        if score < self.config.min_score:
            return

        # Check concurrent position limit
        if len(self._open_positions) >= self.config.max_concurrent:
            return

        # Need a premium to size the trade
        premium = sig.get("atm_premium") or sig.get("premium_per_contract")
        if not premium or premium <= 0:
            return

        # Position sizing
        max_position = self.balance * (self.config.max_position_pct / 100)
        cost_per_contract = premium * 100
        if cost_per_contract <= 0:
            return

        contracts = max(1, int(max_position / cost_per_contract))
        total_cost = contracts * cost_per_contract

        if total_cost > self.balance:
            contracts = max(1, int(self.balance / cost_per_contract))
            total_cost = contracts * cost_per_contract

        if total_cost > self.balance:
            return

        # Determine exit premium from outcome data
        pnl_pct = self._resolve_pnl_pct(sig)
        exit_premium = premium * (1 + pnl_pct / 100)
        if exit_premium < 0:
            exit_premium = 0.0

        proceeds = exit_premium * contracts * 100
        pnl_dollars = proceeds - total_cost

        # Update balance
        self.balance -= total_cost
        self.balance += proceeds

        # Track drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown_dollars = self.peak_balance - self.balance
        drawdown_pct = (drawdown_dollars / self.peak_balance * 100) if self.peak_balance > 0 else 0.0
        if drawdown_dollars > self.max_drawdown_dollars:
            self.max_drawdown_dollars = drawdown_dollars
        if drawdown_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = drawdown_pct

        # Determine outcome label
        outcome = sig.get("outcome", "unknown")

        # Dates
        opened_at = sig.get("created_at", sig.get("opened_at", ""))
        closed_at = sig.get("resolved_at", sig.get("closed_at", opened_at))

        trade = TradeDetail(
            signal_id=sig.get("signal_id", sig.get("id", 0)),
            ticker=sig.get("ticker", ""),
            direction=sig.get("direction", ""),
            score=score,
            bot_source=sig.get("bot_source", ""),
            contracts=contracts,
            entry_premium=premium,
            exit_premium=exit_premium,
            total_cost=total_cost,
            proceeds=proceeds,
            pnl_dollars=pnl_dollars,
            pnl_pct=pnl_pct,
            outcome=outcome,
            balance_after=self.balance,
            opened_at=opened_at,
            closed_at=closed_at,
        )
        self._trades.append(trade)
        self._trade_returns.append(pnl_pct)

        # Track daily PnL
        date_key = closed_at[:10] if closed_at and len(closed_at) >= 10 else "unknown"
        cum_pnl = self.balance - self.config.starting_balance
        self._daily_pnl[date_key] = cum_pnl

    @staticmethod
    def _resolve_pnl_pct(sig: dict) -> float:
        """Extract the PnL percentage from signal outcome data."""
        # Prefer explicit pnl_pct from paper_trades
        if sig.get("pnl_pct") is not None:
            return float(sig["pnl_pct"])

        # Estimate from ATM premium PnL
        if sig.get("pnl_atm_est") is not None:
            return float(sig["pnl_atm_est"])

        # Fall back to underlying PnL (rough approximation)
        if sig.get("pnl_underlying_pct") is not None:
            return float(sig["pnl_underlying_pct"])

        # Use outcome heuristics
        outcome = sig.get("outcome", "unknown")
        if outcome == "t1_hit":
            t1_pct = sig.get("target_1_pct")
            return float(t1_pct) if t1_pct else 25.0
        elif outcome == "t2_hit":
            t2_pct = sig.get("target_2_pct")
            return float(t2_pct) if t2_pct else 50.0
        elif outcome == "stop_hit":
            stop_pct = sig.get("stop_pct")
            return -abs(float(stop_pct)) if stop_pct else -30.0
        elif outcome == "expired":
            return -100.0

        return 0.0

    def compute_metrics(self) -> BacktestResult:
        """Compute aggregate performance metrics from collected trades."""
        total = len(self._trades)
        wins = sum(1 for t in self._trades if t.pnl_dollars >= 0)
        losses = total - wins

        win_rate = (wins / total * 100) if total > 0 else 0.0
        total_pnl = self.balance - self.config.starting_balance
        total_pnl_pct = (total_pnl / self.config.starting_balance * 100) if self.config.starting_balance > 0 else 0.0

        win_pcts = [t.pnl_pct for t in self._trades if t.pnl_dollars >= 0]
        loss_pcts = [t.pnl_pct for t in self._trades if t.pnl_dollars < 0]

        avg_win_pct = (sum(win_pcts) / len(win_pcts)) if win_pcts else 0.0
        avg_loss_pct = (sum(loss_pcts) / len(loss_pcts)) if loss_pcts else 0.0

        best_trade = max((t.pnl_dollars for t in self._trades), default=0.0)
        worst_trade = min((t.pnl_dollars for t in self._trades), default=0.0)

        # Profit factor = gross_wins / gross_losses
        gross_wins = sum(t.pnl_dollars for t in self._trades if t.pnl_dollars > 0)
        gross_losses = abs(sum(t.pnl_dollars for t in self._trades if t.pnl_dollars < 0))
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

        # Sharpe ratio (using trade returns, annualized assuming ~252 trading days)
        sharpe_ratio = self._compute_sharpe(self._trade_returns)
        sortino_ratio = self._compute_sortino(self._trade_returns)

        daily_pnl_series = sorted(self._daily_pnl.items())

        return BacktestResult(
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown_pct=self.max_drawdown_pct,
            max_drawdown_dollars=self.max_drawdown_dollars,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            best_trade=best_trade,
            worst_trade=worst_trade,
            daily_pnl_series=daily_pnl_series,
            trades=[t for t in self._trades],
        )

    @staticmethod
    def _compute_sharpe(returns: list[float], risk_free: float = 0.0) -> float:
        """Compute Sharpe ratio from a list of per-trade return percentages."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        return (mean - risk_free) / std

    @staticmethod
    def _compute_sortino(returns: list[float], risk_free: float = 0.0) -> float:
        """Compute Sortino ratio (penalises only downside deviation)."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        downside = [r for r in returns if r < risk_free]
        if not downside:
            return float("inf") if mean > risk_free else 0.0
        downside_var = sum((r - risk_free) ** 2 for r in downside) / len(downside)
        downside_std = math.sqrt(downside_var)
        if downside_std == 0:
            return 0.0
        return (mean - risk_free) / downside_std

    def format_report(self) -> str:
        """Return a human-readable backtest report."""
        r = self.compute_metrics()
        lines = [
            "=" * 60,
            "  OPTIONS OWL — BACKTEST REPORT",
            "=" * 60,
            "",
            f"  Starting Balance:  ${self.config.starting_balance:,.2f}",
            f"  Final Balance:     ${self.balance:,.2f}",
            f"  Total PnL:         ${r.total_pnl:+,.2f} ({r.total_pnl_pct:+.1f}%)",
            "",
            f"  Total Trades:      {r.total_trades}",
            f"  Wins / Losses:     {r.wins} / {r.losses}",
            f"  Win Rate:          {r.win_rate:.1f}%",
            "",
            f"  Avg Win:           {r.avg_win_pct:+.1f}%",
            f"  Avg Loss:          {r.avg_loss_pct:+.1f}%",
            f"  Best Trade:        ${r.best_trade:+,.2f}",
            f"  Worst Trade:       ${r.worst_trade:+,.2f}",
            "",
            f"  Max Drawdown:      ${r.max_drawdown_dollars:,.2f} ({r.max_drawdown_pct:.1f}%)",
            f"  Sharpe Ratio:      {r.sharpe_ratio:.2f}",
            f"  Sortino Ratio:     {r.sortino_ratio:.2f}",
            f"  Profit Factor:     {r.profit_factor:.2f}",
            "",
            f"  Config: min_score={self.config.min_score}, "
            f"max_pos={self.config.max_position_pct}%, "
            f"max_concurrent={self.config.max_concurrent}",
            "=" * 60,
        ]
        return "\n".join(lines)


async def load_historical_signals(
    db_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Load closed paper trades joined with signal data from the DB for replay."""
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row

        query = (
            "SELECT pt.*, ts.atm_premium, ts.target_1_pct, ts.target_2_pct, "
            "ts.stop_pct, ts.score as ts_score, "
            "so.outcome, so.pnl_atm_est, so.pnl_underlying_pct "
            "FROM paper_trades pt "
            "LEFT JOIN trade_signals ts ON pt.signal_id = ts.id "
            "LEFT JOIN signal_outcomes so ON pt.signal_id = so.signal_id "
            "WHERE pt.status = 'closed' "
        )
        params: list[str] = []

        if start_date:
            query += "AND pt.opened_at >= ? "
            params.append(start_date)
        if end_date:
            query += "AND pt.opened_at <= ? "
            params.append(end_date + "T23:59:59")

        query += "ORDER BY pt.opened_at"

        cursor = await conn.execute(query, params)
        rows = [dict(r) for r in await cursor.fetchall()]

        # Ensure score is present (prefer trade_signals score)
        for r in rows:
            if r.get("ts_score") is not None:
                r["score"] = r["ts_score"]

        return rows
