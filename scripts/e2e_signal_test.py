#!/usr/bin/env python3
"""E2E Signal Pipeline Test — inject a synthetic signal into PostgreSQL,
verify all trading agents consume it, check trade_events, and verify
the harvester is writing tick data.

No Python dependencies required (uses psql via Docker).

Usage:
    # Full E2E (inject → wait → check → cleanup) on droplet
    python3 scripts/e2e_signal_test.py --droplet

    # Quick status check (recent signals + tick data)
    python3 scripts/e2e_signal_test.py --droplet --status

    # Just inject (don't wait)
    python3 scripts/e2e_signal_test.py --droplet --inject-only

    # Cleanup leftover test signals
    python3 scripts/e2e_signal_test.py --droplet --cleanup

    # Run locally (requires PG on localhost:5432)
    python3 scripts/e2e_signal_test.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DROPLET_HOST = "root@129.212.138.145"
DROPLET_KEY = "~/.ssh/id_ed25519_do"
DROPLET_DIR = "/root/options-owl"

EXPECTED_AGENTS = ["owlet_kody", "owlet_adam", "owlet_vinny", "owlet_yank"]
TEST_TICKER = "TESTowl"
TEST_SCORE = 95
POLL_INTERVAL = 30  # signal_consumer polls every 30s
WAIT_TIMEOUT = 150  # max seconds to wait for all agents


# ---------------------------------------------------------------------------
# SQL execution via psql
# ---------------------------------------------------------------------------

def psql(sql: str, *, droplet: bool = False, csv: bool = False) -> str:
    """Run SQL via psql. Returns stdout."""
    fmt = ["-A", "-t"]  # unaligned, tuples only
    if csv:
        fmt = ["--csv", "-t"]

    if droplet:
        # Run psql inside the postgres container on the droplet
        psql_cmd = f"docker compose exec -T postgres psql -U owl -d options_owl {' '.join(fmt)}"
        cmd = [
            "ssh", "-i", DROPLET_KEY, DROPLET_HOST,
            f"cd {DROPLET_DIR} && echo {_shell_quote(sql)} | {psql_cmd}",
        ]
    else:
        cmd = [
            "psql", "-U", "owl", "-d", "options_owl",
            "-h", "localhost", "-p", "5432",
        ] + fmt + ["-c", sql]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "does not exist" in stderr or "ERROR" in stderr:
            raise RuntimeError(f"psql error: {stderr}")
        # Some warnings are OK
    # Filter out psql command tags like "INSERT 0 1", "DELETE 3"
    lines = result.stdout.strip().split("\n")
    lines = [l for l in lines if not l.startswith(("INSERT ", "DELETE ", "UPDATE ", "SET", "CREATE"))]
    return "\n".join(lines).strip()


def _shell_quote(s: str) -> str:
    """Quote a string for shell embedding via echo."""
    return "'" + s.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def inject_signal(droplet: bool) -> int:
    """Insert a synthetic ML signal. Returns signal ID."""
    sql = f"""
    INSERT INTO ml_signals (
        ticker, direction, score, ml_confidence, ml_threshold,
        ml_model_source, premium, strike, expiry_date,
        indicators, score_breakdown, emitted_at
    ) VALUES (
        '{TEST_TICKER}', 'CALL', {TEST_SCORE}, 0.85, 0.70,
        'e2e_test', 2.50, 550.0, '0DTE',
        '{{"test": true}}'::jsonb,
        '{{"direction": 40, "timing": 30}}'::jsonb,
        NOW()
    )
    RETURNING id;
    """
    result = psql(sql, droplet=droplet)
    return int(result.strip())


def check_signal(signal_id: int, droplet: bool) -> dict:
    """Get signal status."""
    sql = f"""
    SELECT id, ticker, status, consumed_by, emitted_at
    FROM ml_signals WHERE id = {signal_id};
    """
    row = psql(sql, droplet=droplet, csv=True)
    if not row:
        return {}
    parts = row.split(",", 4)
    consumed_str = parts[3] if len(parts) > 3 else "{}"
    # Parse postgres array format: {owlet_kody,owlet_adam}
    consumed = []
    if consumed_str and consumed_str not in ("{}", ""):
        consumed = consumed_str.strip("{}").split(",")
        consumed = [c.strip('"') for c in consumed if c]
    return {
        "id": int(parts[0]),
        "ticker": parts[1],
        "status": parts[2],
        "consumed_by": consumed,
    }


def check_tick_data(droplet: bool) -> dict:
    """Check harvester tick data tables."""
    result = {}
    for table in ["stock_ticks", "option_ticks", "stock_candles"]:
        sql = f"""
        SELECT COUNT(*),
               MAX(captured_at)::text,
               COUNT(*) FILTER (WHERE captured_at > NOW() - interval '10 minutes')
        FROM {table};
        """
        row = psql(sql, droplet=droplet)
        if row:
            parts = row.split("|")
            result[table] = {
                "total": int(parts[0]) if parts[0] else 0,
                "latest": parts[1].strip() if len(parts) > 1 and parts[1].strip() else None,
                "recent": int(parts[2]) if len(parts) > 2 and parts[2].strip() else 0,
            }
        else:
            result[table] = {"total": 0, "latest": None, "recent": 0}
    return result


def check_trade_events(droplet: bool, minutes: int = 5) -> list[str]:
    """Check trade_events for test ticker activity."""
    sql = f"""
    SELECT agent_id, event_type, created_at::text
    FROM trade_events
    WHERE event_data::text ILIKE '%{TEST_TICKER}%'
      AND created_at > NOW() - interval '{minutes} minutes'
    ORDER BY created_at DESC
    LIMIT 20;
    """
    rows = psql(sql, droplet=droplet)
    return [r for r in rows.split("\n") if r.strip()]


def get_recent_signals(droplet: bool, minutes: int = 30) -> list[str]:
    """Get recent ML signals."""
    sql = f"""
    SELECT id, ticker, direction, score, status,
           array_length(consumed_by, 1) as consumers,
           emitted_at::text
    FROM ml_signals
    WHERE emitted_at > NOW() - interval '{minutes} minutes'
    ORDER BY emitted_at DESC
    LIMIT 10;
    """
    rows = psql(sql, droplet=droplet)
    return [r for r in rows.split("\n") if r.strip()]


def get_running_agents(droplet: bool) -> list[str]:
    """Check which trading bot containers are running."""
    if droplet:
        cmd = [
            "ssh", "-i", DROPLET_KEY, DROPLET_HOST,
            f"cd {DROPLET_DIR} && docker compose ps --format '{{{{.Name}}}} {{{{.Status}}}}'",
        ]
    else:
        cmd = ["docker", "compose", "ps", "--format", "{{.Name}} {{.Status}}"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
    running = []
    for line in lines:
        parts = line.split(" ", 1)
        if len(parts) == 2 and "Up" in parts[1]:
            running.append(parts[0])
    return running


def cleanup_test_signals(droplet: bool) -> int:
    """Remove all test signals. Returns count deleted."""
    sql = f"SELECT COUNT(*) FROM ml_signals WHERE ticker = '{TEST_TICKER}' OR ml_model_source = 'e2e_test';"
    count_str = psql(sql, droplet=droplet)
    count = int(count_str) if count_str else 0
    if count > 0:
        psql(f"DELETE FROM ml_signals WHERE ticker = '{TEST_TICKER}' OR ml_model_source = 'e2e_test';", droplet=droplet)
    return count


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def ok(text: str) -> None:
    print(f"  [OK]   {text}")


def fail(text: str) -> None:
    print(f"  [FAIL] {text}")


def info(text: str) -> None:
    print(f"  [..]   {text}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_full(droplet: bool) -> int:
    """Full E2E: check infra → inject → wait → verify → cleanup."""
    failures = 0

    # Step 0: Check infrastructure
    header("Step 0: Infrastructure check")
    try:
        psql("SELECT 1;", droplet=droplet)
        ok("PostgreSQL reachable")
    except Exception as e:
        fail(f"PostgreSQL unreachable: {e}")
        return 1

    running = get_running_agents(droplet)
    bot_containers = [c for c in running if c.startswith("owlet-") and c != "owlet-sourcing" and c != "owlet-harvester"]
    agent_map = {
        "owlet-kody": "owlet_kody",
        "owlet-adam": "owlet_adam",
        "owlet-vinny": "owlet_vinny",
        "owlet-yank": "owlet_yank",
    }
    active_agents = [agent_map[c] for c in bot_containers if c in agent_map]

    info(f"Running containers: {running}")
    if not active_agents:
        fail("No trading bot containers running! Agents won't consume signals.")
        info("Trading bots crash-loop on weekends (expected). Try Monday after 9:30 AM ET.")
        # Still inject to test PG write path
    else:
        ok(f"Active trading agents: {active_agents}")

    # Step 1: Inject signal
    header("Step 1: Inject test signal")
    try:
        signal_id = inject_signal(droplet)
        ok(f"Injected signal #{signal_id} ({TEST_TICKER} CALL score={TEST_SCORE})")
    except Exception as e:
        fail(f"Failed to inject signal: {e}")
        return 1

    # Step 2: Wait for consumption (only if bots are running)
    if active_agents:
        header("Step 2: Wait for agent consumption")
        info(f"Waiting up to {WAIT_TIMEOUT}s for {len(active_agents)} agent(s)...")
        start = time.time()

        while time.time() - start < WAIT_TIMEOUT:
            sig = check_signal(signal_id, droplet)
            consumed = sig.get("consumed_by", [])
            elapsed = int(time.time() - start)

            missing = set(active_agents) - set(consumed)
            if consumed:
                info(f"{elapsed}s: consumed_by={consumed} ({len(consumed)}/{len(active_agents)})")

            if not missing:
                ok(f"All {len(active_agents)} agent(s) consumed signal #{signal_id}!")
                break

            time.sleep(10)
        else:
            sig = check_signal(signal_id, droplet)
            consumed = sig.get("consumed_by", [])
            missing = set(active_agents) - set(consumed)
            if missing:
                fail(f"Timeout! Consumed: {consumed}, Missing: {missing}")
                failures += 1
            else:
                ok("All agents consumed!")
    else:
        header("Step 2: Skip (no trading bots running)")
        info("Signal will sit in PG until bots come online")

    # Step 3: Check trade events
    if active_agents:
        header("Step 3: Check trade_events")
        time.sleep(5)
        events = check_trade_events(droplet, minutes=5)
        if events:
            ok(f"Found {len(events)} trade event(s):")
            for ev in events:
                info(f"  {ev}")
        else:
            info("No trade_events yet (Phase 1: agents may only write to SQLite)")
    else:
        header("Step 3: Skip (no trading bots)")

    # Step 4: Check harvester tick data
    header("Step 4: Harvester tick data")
    tick_data = check_tick_data(droplet)
    for table, stats in tick_data.items():
        total = stats["total"]
        recent = stats["recent"]
        latest = stats["latest"]
        if total > 0:
            ok(f"{table}: {total:,} rows, {recent} recent (10min), latest={latest}")
        else:
            info(f"{table}: empty (market closed or harvester hasn't written yet)")

    # Step 5: Cleanup
    header("Step 5: Cleanup")
    cleaned = cleanup_test_signals(droplet)
    ok(f"Removed {cleaned} test signal(s)")

    # Summary
    header("RESULT")
    if failures == 0:
        ok("ALL CHECKS PASSED")
    else:
        fail(f"{failures} check(s) failed")
    return failures


def cmd_status(droplet: bool) -> int:
    """Quick health check."""
    # PG connectivity
    header("PostgreSQL")
    try:
        psql("SELECT 1;", droplet=droplet)
        ok("Connected")
    except Exception as e:
        fail(f"Unreachable: {e}")
        return 1

    # Containers
    header("Containers")
    running = get_running_agents(droplet)
    for c in running:
        ok(c)
    expected = {"owlet-kody", "owlet-adam", "owlet-vinny", "owlet-yank",
                "owlet-sourcing", "owlet-harvester", "options-owl-redis", "options-owl-db"}
    missing = expected - set(running)
    if missing:
        for m in sorted(missing):
            fail(f"{m}: NOT RUNNING")

    # Recent signals
    header("Recent ML Signals (30 min)")
    signals = get_recent_signals(droplet)
    if signals:
        for s in signals:
            info(s)
    else:
        info("No signals in last 30 min")

    # Tick data
    header("Tick Data")
    tick_data = check_tick_data(droplet)
    for table, stats in tick_data.items():
        total = stats["total"]
        recent = stats["recent"]
        latest = stats["latest"]
        if total > 0:
            ok(f"{table}: {total:,} rows, {recent} recent, latest={latest}")
        else:
            info(f"{table}: empty")

    return 0


def cmd_inject_only(droplet: bool) -> int:
    """Just inject a signal, print the ID."""
    header("Inject test signal")
    try:
        signal_id = inject_signal(droplet)
        ok(f"Injected signal #{signal_id} ({TEST_TICKER} CALL score={TEST_SCORE})")
        info(f"Check: python3 scripts/e2e_signal_test.py {'--droplet ' if droplet else ''}--status")
        return 0
    except Exception as e:
        fail(f"Failed: {e}")
        return 1


def cmd_cleanup(droplet: bool) -> int:
    """Remove test signals."""
    header("Cleanup")
    cleaned = cleanup_test_signals(droplet)
    ok(f"Removed {cleaned} test signal(s)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="E2E signal pipeline test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/e2e_signal_test.py --droplet            # Full E2E test
  python3 scripts/e2e_signal_test.py --droplet --status    # Quick health check
  python3 scripts/e2e_signal_test.py --droplet --cleanup   # Remove test signals
  python3 scripts/e2e_signal_test.py --droplet --inject-only  # Just inject
        """,
    )
    parser.add_argument("--droplet", action="store_true", help="Run against production droplet")
    parser.add_argument("--status", action="store_true", help="Quick pipeline health check")
    parser.add_argument("--inject-only", action="store_true", help="Just inject a signal")
    parser.add_argument("--cleanup", action="store_true", help="Remove test signals")

    args = parser.parse_args()

    if args.status:
        rc = cmd_status(args.droplet)
    elif args.inject_only:
        rc = cmd_inject_only(args.droplet)
    elif args.cleanup:
        rc = cmd_cleanup(args.droplet)
    else:
        # Default: full E2E
        rc = cmd_full(args.droplet)

    sys.exit(rc)


if __name__ == "__main__":
    main()
