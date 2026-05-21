"""Aggregate signal outcomes into per-bot performance reports."""

from __future__ import annotations

from loguru import logger

from options_owl.journal import db
from options_owl.models.signals import BotPerformanceReport, BotSource, TradeOutcome


def _is_win(outcome: str) -> bool:
    return outcome in (TradeOutcome.T1_HIT, TradeOutcome.T2_HIT)


async def compute_bot_performance(
    db_path: str,
    bot_source: BotSource,
) -> BotPerformanceReport:
    """Compute performance stats for a single bot."""
    all_signals = await db.get_signals_by_bot(db_path, bot_source.value)
    outcomes = await db.get_outcomes_by_bot(db_path, bot_source.value)

    total = len(all_signals)
    resolved = len(outcomes)
    wins = sum(1 for o in outcomes if _is_win(o["outcome"]))
    losses = resolved - wins
    win_rate = (wins / resolved * 100) if resolved > 0 else 0.0

    pnls = [o["pnl_underlying_pct"] for o in outcomes]
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0

    atm_pnls = [o["pnl_atm_est"] for o in outcomes if o["pnl_atm_est"] is not None]
    avg_atm = sum(atm_pnls) / len(atm_pnls) if atm_pnls else None

    best = max(pnls) if pnls else 0.0
    worst = min(pnls) if pnls else 0.0

    scores = [s["score"] for s in all_signals]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Elite-only stats
    elite_outcomes = [o for o in outcomes if o.get("is_elite")]
    elite_wr = None
    if elite_outcomes:
        elite_wins = sum(1 for o in elite_outcomes if _is_win(o["outcome"]))
        elite_wr = elite_wins / len(elite_outcomes) * 100

    # Strong-only stats
    strong_outcomes = [o for o in outcomes if o.get("strength") == "strong"]
    strong_wr = None
    if strong_outcomes:
        strong_wins = sum(1 for o in strong_outcomes if _is_win(o["outcome"]))
        strong_wr = strong_wins / len(strong_outcomes) * 100

    # Get latest Smee data for comparison
    smee_reports = await db.get_smee_performance(db_path)
    smee_wr = smee_reports[0]["win_rate_pct"] if smee_reports else None
    smee_pnl = smee_reports[0]["avg_pnl_pct"] if smee_reports else None

    return BotPerformanceReport(
        bot_source=bot_source,
        total_signals=total,
        resolved_signals=resolved,
        wins=wins,
        losses=losses,
        win_rate_pct=round(win_rate, 1),
        avg_pnl_pct=round(avg_pnl, 4),
        avg_pnl_atm=round(avg_atm, 2) if avg_atm is not None else None,
        best_trade_pnl=round(best, 4),
        worst_trade_pnl=round(worst, 4),
        avg_score=round(avg_score, 1),
        elite_win_rate_pct=round(elite_wr, 1) if elite_wr is not None else None,
        strong_win_rate_pct=round(strong_wr, 1) if strong_wr is not None else None,
        smee_reported_win_rate=smee_wr,
        smee_reported_avg_pnl=smee_pnl,
    )


async def compute_all_bots_performance(
    db_path: str,
) -> list[BotPerformanceReport]:
    """Compute performance for all trading bots."""
    trading_bots = [BotSource.CAPTAIN_HOOK, BotSource.NEVERLAND_PAN, BotSource.TINKER]
    reports = []
    for bot in trading_bots:
        report = await compute_bot_performance(db_path, bot)
        if report.total_signals > 0:
            reports.append(report)
    return reports


def format_report(reports: list[BotPerformanceReport]) -> str:
    """Format performance reports as a readable string."""
    if not reports:
        return "No data yet. Collect signals and run backfill first."

    lines = ["=" * 60, "  OPTIONS OWL — PERFORMANCE REPORT", "=" * 60, ""]

    for r in reports:
        lines.append(f"Bot: {r.bot_source.value}")
        lines.append(f"  Signals: {r.total_signals} total, {r.resolved_signals} resolved")
        lines.append(f"  Record:  {r.wins}W / {r.losses}L ({r.win_rate_pct:.1f}%)")
        lines.append(f"  PnL:     avg {r.avg_pnl_pct:+.2f}% | best {r.best_trade_pnl:+.2f}% | worst {r.worst_trade_pnl:+.2f}%")
        if r.avg_pnl_atm is not None:
            lines.append(f"  ATM Est: avg {r.avg_pnl_atm:+.2f}%")
        lines.append(f"  Avg Score: {r.avg_score:.0f}/100")
        if r.elite_win_rate_pct is not None:
            lines.append(f"  Elite 💎:  {r.elite_win_rate_pct:.1f}% win rate")
        if r.strong_win_rate_pct is not None:
            lines.append(f"  Strong 🟢: {r.strong_win_rate_pct:.1f}% win rate")
        lines.append("")

    # Smee comparison
    smee = next((r for r in reports if r.smee_reported_win_rate is not None), None)
    if smee:
        lines.append("--- Smee Comparison ---")
        lines.append(f"  Smee reported: {smee.smee_reported_win_rate:.0f}% WR, {smee.smee_reported_avg_pnl:+.2f}% avg PnL")
        total_wins = sum(r.wins for r in reports)
        total_resolved = sum(r.resolved_signals for r in reports)
        our_wr = total_wins / total_resolved * 100 if total_resolved else 0
        our_pnl = sum(r.avg_pnl_pct * r.resolved_signals for r in reports) / total_resolved if total_resolved else 0
        lines.append(f"  Our tracking: {our_wr:.0f}% WR, {our_pnl:+.2f}% avg PnL")
        lines.append("")

    lines.append("=" * 60)
    logger.info("\n".join(lines))
    return "\n".join(lines)
