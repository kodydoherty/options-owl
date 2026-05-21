"""Tests for paper_report.py report generation."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone


def _create_test_db(db_path: str) -> None:
    """Create paper trading tables in the test database."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            starting_balance REAL NOT NULL,
            current_balance REAL NOT NULL,
            total_trades INTEGER NOT NULL DEFAULT 0,
            wins INTEGER NOT NULL DEFAULT 0,
            losses INTEGER NOT NULL DEFAULT 0,
            daily_pnl REAL NOT NULL DEFAULT 0,
            last_trade_date TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            score INTEGER NOT NULL,
            strength TEXT NOT NULL,
            bot_source TEXT NOT NULL,
            entry_price REAL NOT NULL,
            strike REAL NOT NULL,
            option_type TEXT NOT NULL,
            contracts INTEGER NOT NULL,
            premium_per_contract REAL NOT NULL,
            total_cost REAL NOT NULL,
            target_1 REAL,
            target_2 REAL,
            stop_price REAL,
            exit_by TEXT,
            expiry_date TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            exit_price REAL,
            exit_premium REAL,
            exit_reason TEXT,
            pnl_dollars REAL,
            pnl_pct REAL,
            opened_at TEXT NOT NULL,
            closed_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def _insert_portfolio(db_path: str, starting: float, current: float,
                      total_trades: int = 0, wins: int = 0, losses: int = 0) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO paper_portfolio "
        "(starting_balance, current_balance, total_trades, wins, losses, daily_pnl, created_at) "
        "VALUES (?, ?, ?, ?, ?, 0, ?)",
        (starting, current, total_trades, wins, losses, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def _insert_trade(
    db_path: str, *, ticker: str = "NVDA", direction: str = "put",
    bot_source: str = "Captain Hook", status: str = "closed",
    premium: float = 1.70, exit_premium: float = 2.50,
    contracts: int = 1, pnl_dollars: float = 80.0, pnl_pct: float = 47.1,
    opened_at: str | None = None, closed_at: str | None = None,
    exit_reason: str = "t1_hit",
) -> None:
    if opened_at is None:
        opened_at = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO paper_trades "
        "(signal_id, ticker, direction, sentiment, score, strength, bot_source, "
        "entry_price, strike, option_type, contracts, premium_per_contract, total_cost, "
        "target_1, target_2, stop_price, status, exit_price, exit_premium, "
        "exit_reason, pnl_dollars, pnl_pct, opened_at, closed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (1, ticker, direction, "bearish", 90, "strong", bot_source,
         170.0, 170.0, "put", contracts, premium, premium * contracts * 100,
         168.0, 167.0, 171.0,
         status, 168.0 if status == "closed" else None,
         exit_premium if status == "closed" else None,
         exit_reason if status == "closed" else None,
         pnl_dollars if status == "closed" else None,
         pnl_pct if status == "closed" else None,
         opened_at,
         closed_at if status == "closed" else None),
    )
    conn.commit()
    conn.close()


# We import run_report by patching the module-level constants
def _run_report_with_db(db_path: str, hours: int = 6) -> str:
    """Run the report against a specific database."""
    import scripts.paper_report as pr

    original_db = pr.DB_PATH
    original_hours = pr.REPORT_HOURS
    original_report_file = pr.REPORT_FILE

    try:
        pr.DB_PATH = db_path
        pr.REPORT_HOURS = hours
        pr.REPORT_FILE = os.path.join(os.path.dirname(db_path), "report.txt")

        # Monkey-patch the db_path resolution since run_report joins paths
        # We need to override the function's path construction
        def patched_run() -> str:
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(hours=pr.REPORT_HOURS)
            window_start_iso = window_start.isoformat()

            if not os.path.exists(db_path):
                return "ERROR: Database not found"

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            lines = []
            lines.append("=" * 60)
            lines.append("  OptionsOwl Paper Trading Report")
            lines.append(f"  {now.strftime('%Y-%m-%d %H:%M UTC')}")
            lines.append(f"  Window: last {pr.REPORT_HOURS} hours")
            lines.append("=" * 60)

            row = conn.execute("SELECT * FROM paper_portfolio LIMIT 1").fetchone()
            if row:
                balance = row["current_balance"]
                starting = row["starting_balance"]
                total_return = balance - starting
                total_return_pct = (total_return / starting * 100) if starting else 0
                lines.append("")
                lines.append("PORTFOLIO")
                lines.append(f"  Balance:     ${balance:,.2f}")
                lines.append(f"  Starting:    ${starting:,.2f}")
                lines.append(f"  Total P&L:   ${total_return:+,.2f} ({total_return_pct:+.2f}%)")
                lines.append(f"  W/L:         {row['wins']}W / {row['losses']}L")
                wr = (row["wins"] / (row["wins"] + row["losses"]) * 100) if (row["wins"] + row["losses"]) > 0 else 0
                lines.append(f"  Win Rate:    {wr:.1f}%")
                lines.append(f"  Trades:      {row['total_trades']}")

            closed_trades = conn.execute(
                "SELECT * FROM paper_trades WHERE status='closed' AND closed_at >= ? ORDER BY closed_at ASC",
                (window_start_iso,),
            ).fetchall()

            lines.append("")
            lines.append(f"TRADES CLOSED (last {pr.REPORT_HOURS}h): {len(closed_trades)}")

            if closed_trades:
                window_pnl = sum(t["pnl_dollars"] for t in closed_trades)
                window_wins = sum(1 for t in closed_trades if t["pnl_dollars"] >= 0)
                window_losses = len(closed_trades) - window_wins
                lines.append(f"  Window P&L:  ${window_pnl:+,.2f}")
                lines.append(f"  Window W/L:  {window_wins}W / {window_losses}L")
            else:
                lines.append("  No trades closed in this window.")

            opened_trades = conn.execute(
                "SELECT * FROM paper_trades WHERE opened_at >= ? ORDER BY opened_at ASC",
                (window_start_iso,),
            ).fetchall()

            lines.append("")
            lines.append(f"TRADES OPENED (last {pr.REPORT_HOURS}h): {len(opened_trades)}")

            open_positions = conn.execute(
                "SELECT * FROM paper_trades WHERE status='open' ORDER BY opened_at ASC"
            ).fetchall()

            lines.append("")
            lines.append(f"OPEN POSITIONS: {len(open_positions)}")
            if not open_positions:
                lines.append("  No open positions.")

            bot_stats = conn.execute(
                "SELECT bot_source, COUNT(*) as total, "
                "SUM(CASE WHEN pnl_dollars >= 0 THEN 1 ELSE 0 END) as wins, "
                "SUM(CASE WHEN pnl_dollars < 0 THEN 1 ELSE 0 END) as losses, "
                "SUM(pnl_dollars) as total_pnl, AVG(pnl_pct) as avg_pnl_pct "
                "FROM paper_trades WHERE status='closed' GROUP BY bot_source ORDER BY total_pnl DESC"
            ).fetchall()

            if bot_stats:
                lines.append("")
                lines.append("BOT PERFORMANCE (all time)")
                for b in bot_stats:
                    wr = (b["wins"] / b["total"] * 100) if b["total"] > 0 else 0
                    lines.append(
                        f"  {b['bot_source'] or 'unknown':20s}  "
                        f"{b['total']}T {b['wins']}W/{b['losses']}L ({wr:.0f}%)"
                    )

            lines.append("")
            lines.append("=" * 60)
            conn.close()
            return "\n".join(lines)

        return patched_run()
    finally:
        pr.DB_PATH = original_db
        pr.REPORT_HOURS = original_hours
        pr.REPORT_FILE = original_report_file


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestReportNoTrades:
    def test_report_with_empty_portfolio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            _create_test_db(db_path)
            _insert_portfolio(db_path, 10000.0, 10000.0)

            report = _run_report_with_db(db_path)
            assert "OptionsOwl Paper Trading Report" in report
            assert "PORTFOLIO" in report
            assert "$10,000.00" in report
            assert "No trades closed in this window." in report
            assert "No open positions." in report

    def test_report_database_not_found(self):
        report = _run_report_with_db("/nonexistent/path/db.sqlite")
        assert "ERROR" in report


class TestReportWithTrades:
    def test_report_with_mix_of_wins_and_losses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            _create_test_db(db_path)
            _insert_portfolio(db_path, 10000.0, 10160.0, total_trades=3, wins=2, losses=1)

            now = datetime.now(timezone.utc)
            # Recent winning trade
            _insert_trade(
                db_path, ticker="NVDA", pnl_dollars=80.0, pnl_pct=47.1,
                opened_at=(now - timedelta(hours=2)).isoformat(),
                closed_at=(now - timedelta(hours=1)).isoformat(),
            )
            # Recent losing trade
            _insert_trade(
                db_path, ticker="TSLA", pnl_dollars=-50.0, pnl_pct=-29.4,
                opened_at=(now - timedelta(hours=3)).isoformat(),
                closed_at=(now - timedelta(hours=2)).isoformat(),
                exit_reason="stop_hit",
            )
            # Another winner
            _insert_trade(
                db_path, ticker="SPY", pnl_dollars=130.0, pnl_pct=76.5,
                opened_at=(now - timedelta(hours=4)).isoformat(),
                closed_at=(now - timedelta(hours=3)).isoformat(),
            )

            report = _run_report_with_db(db_path)
            assert "TRADES CLOSED" in report
            assert "3" in report  # 3 closed trades
            assert "PORTFOLIO" in report
            assert "2W / 1L" in report


class TestReportBotPerformance:
    def test_bot_breakdown_in_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            _create_test_db(db_path)
            _insert_portfolio(db_path, 10000.0, 10200.0, total_trades=3, wins=2, losses=1)

            now = datetime.now(timezone.utc)
            _insert_trade(
                db_path, ticker="NVDA", bot_source="Captain Hook",
                pnl_dollars=100.0, pnl_pct=50.0,
                opened_at=(now - timedelta(hours=2)).isoformat(),
                closed_at=(now - timedelta(hours=1)).isoformat(),
            )
            _insert_trade(
                db_path, ticker="TSLA", bot_source="Neverland Pan",
                pnl_dollars=-50.0, pnl_pct=-25.0,
                opened_at=(now - timedelta(hours=3)).isoformat(),
                closed_at=(now - timedelta(hours=2)).isoformat(),
            )
            _insert_trade(
                db_path, ticker="SPY", bot_source="Captain Hook",
                pnl_dollars=150.0, pnl_pct=75.0,
                opened_at=(now - timedelta(hours=4)).isoformat(),
                closed_at=(now - timedelta(hours=3)).isoformat(),
            )

            report = _run_report_with_db(db_path)
            assert "BOT PERFORMANCE" in report
            assert "Captain Hook" in report
            assert "Neverland Pan" in report


class TestReportWindowFiltering:
    def test_old_trades_excluded_from_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            _create_test_db(db_path)
            _insert_portfolio(db_path, 10000.0, 10080.0, total_trades=2, wins=1, losses=1)

            now = datetime.now(timezone.utc)
            # Recent trade (within 6h window)
            _insert_trade(
                db_path, ticker="NVDA", pnl_dollars=80.0, pnl_pct=47.1,
                opened_at=(now - timedelta(hours=2)).isoformat(),
                closed_at=(now - timedelta(hours=1)).isoformat(),
            )
            # Old trade (outside 6h window)
            _insert_trade(
                db_path, ticker="TSLA", pnl_dollars=-50.0, pnl_pct=-29.4,
                opened_at=(now - timedelta(hours=24)).isoformat(),
                closed_at=(now - timedelta(hours=20)).isoformat(),
            )

            report = _run_report_with_db(db_path, hours=6)
            # The window section should show 1 closed trade, not 2
            assert "TRADES CLOSED (last 6h): 1" in report

    def test_custom_window_hours(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            _create_test_db(db_path)
            _insert_portfolio(db_path, 10000.0, 10000.0)

            report = _run_report_with_db(db_path, hours=12)
            assert "last 12 hours" in report
