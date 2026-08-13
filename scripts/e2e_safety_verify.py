"""End-to-end verification of the 2026-08-13 safety work, run against REAL config.

WHY THIS EXISTS
---------------
Three things shipped to live money today: the phantom-quantity fixes, the invariant
monitors, and concurrency-aware sizing at slots=3. Unit tests cover each in isolation.
This checks the things unit tests structurally cannot:

  * that the sizing change cannot over-deploy at the balances kody and dennis actually
    carry -- the whole safety property of the clamp, exercised with real numbers rather
    than the round ones in the unit test;
  * that sizing up ~2.6x does not push a single position past the position caps;
  * that the guards are all still present and wired (a later refactor can silently drop
    one without failing any test that does not specifically assert it);
  * that the DEPLOYED container is running this code, not a stale image.

Read the failure text, not just the exit code: each check states what breaking it would
mean in production.

Usage:
  python scripts/e2e_safety_verify.py                 # local checks
  python scripts/e2e_safety_verify.py --with-droplet  # also verify what is deployed
"""

from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))


# ---------------------------------------------------------------- sizing safety
def verify_sizing_cannot_overdeploy() -> None:
    """The clamp is the reason sizing up is safe. Prove it at REAL balances.

    Opens positions one after another until sizing refuses, and asserts the running
    total never exceeds the risk cap. If this fails, concurrency sizing can deploy more
    capital than MAX_PORTFOLIO_RISK_PCT allows -- the exact failure the clamp prevents.
    """
    from options_owl.risk.vinny_strategy import score_to_contracts

    for label, balance in (("kody", 18771.74), ("dennis", 6926.28)):
        # cost_per_contract is DOLLARS PER CONTRACT, premium already x100
        # ($0.35 premium = $35/contract). Passing the raw premium here double-counts
        # the multiplier and makes the caps look breached when they are not.
        for cost in (35.0, 110.0, 300.0, 600.0):       # cheap 0DTE .. expensive multiday
            deployable = balance * 0.75                 # MAX_PORTFOLIO_RISK_PCT=75
            deployed = 0.0
            opened = 0
            for _ in range(40):                         # far more than MAX_CONCURRENT
                n = score_to_contracts(
                    95, cost_per_contract=cost, balance=balance,
                    max_position_pct=15.0, max_concurrent=8,
                    max_portfolio_risk_pct=75.0, ml_confidence=0.85,
                    concurrency_slots=3, deployed_dollars=deployed,
                )
                if n <= 0:
                    break
                deployed += n * cost
                opened += 1
            over = deployed > deployable + 1e-6
            check(
                f"no over-deploy [{label} ${cost:.0f}/ct]",
                not over,
                f"deployed ${deployed:,.0f} of ${deployable:,.0f} cap across {opened} legs"
                + (" -- BREACHED" if over else ""),
            )


def verify_single_position_cap_still_binds() -> None:
    """Sizing up must not let ONE position exceed MAX_POSITION_PCT.

    slots=3 raises the per-trade budget ~2.6x. If the position cap stopped binding, a
    single trade could take a far larger share of the account than intended -- which is
    how one bad fill becomes an account-level event.
    """
    from options_owl.risk.vinny_strategy import score_to_contracts

    balance, pct = 18771.74, 15.0
    for cost in (35.0, 110.0, 300.0):
        n = score_to_contracts(
            95, cost_per_contract=cost, balance=balance,
            max_position_pct=pct, max_concurrent=8, max_portfolio_risk_pct=75.0,
            ml_confidence=0.95, concurrency_slots=3, deployed_dollars=0.0,
        )
        spend = n * cost
        cap = balance * pct / 100
        check(
            f"position cap binds [${cost:.0f}/ct]",
            spend <= cap + 1e-6,
            f"{n} contracts = ${spend:,.0f} vs cap ${cap:,.0f}",
        )


def verify_slots_off_is_unchanged() -> None:
    """Paper bots run slots=0. That path must be byte-identical to before the change."""
    from options_owl.risk.vinny_strategy import score_to_contracts

    a = score_to_contracts(95, cost_per_contract=110.0, balance=10000.0,
                           max_position_pct=15.0, max_concurrent=8,
                           max_portfolio_risk_pct=75.0, ml_confidence=0.85)
    b = score_to_contracts(95, cost_per_contract=110.0, balance=10000.0,
                           max_position_pct=15.0, max_concurrent=8,
                           max_portfolio_risk_pct=75.0, ml_confidence=0.85,
                           concurrency_slots=0, deployed_dollars=0.0)
    check("slots=0 is a no-op", a == b, f"{a} vs {b} contracts")


# ---------------------------------------------------------------- guards present
def verify_guards_wired() -> None:
    """Each guard, and what its absence would cost."""
    from options_owl.execution import position_monitor as pm
    from options_owl.execution.paper_trader import PaperTrader, SellOutcome
    from options_owl.execution.webull_executor import WebullExecutor

    rec = inspect.getsource(pm._reconcile_positions)
    check("quantity invariant in reconcile", "broker_qty" in rec and "db_qty" in rec,
          "without it a phantom quantity is invisible until an exit is blocked")
    check("heal shrinks only", "broker_qty < db_qty" in rec,
          "growing a record to match a larger broker position could create a naked short")

    fin = inspect.getsource(pm._finalize_full_close_inner)
    check("quantity-mismatch self-heal", "QUANTITY_MISMATCH" in fin,
          "a permanent oversell rejection would retry forever (the 760-attempt loop)")
    check("escalation never goes silent", "transient_count % 25 == 0" in fin,
          "failure #1000 would be quieter than #20")
    check("no silent alert gating", "and discord_client:" not in fin,
          "alerts vanish when no client is configured, as they did today")

    check("loud alert helper", hasattr(pm, "_alert_or_shout"),
          "undeliverable alerts must announce themselves")
    check("permanent oversell classified", hasattr(SellOutcome, "QUANTITY_MISMATCH"),
          "falls through to TRANSIENT_ERROR and retries forever")

    buy = inspect.getsource(WebullExecutor._place_buy_with_escalation)
    check("size-down persists true qty", buy.count("requested_contracts") >= 2,
          "FILLED-DURING-CANCEL reports a sized-down fill as full -> phantom quantity")

    close = inspect.getsource(PaperTrader.close_webull_position)
    check("oversell error recognised", "must_be_close_than_sell_short" in close.lower(),
          "the broker's permanent rejection is treated as transient")

    sc = Path("scripts/safety_check.py").read_text()
    check("safety check is market-hours aware", "_market_is_open" in sc,
          "fires false alarms every 5 min overnight -> alert fatigue")


def run_unit_suites() -> None:
    """The targeted regression suites for today's incidents."""
    files = [
        "tests/test_quantity_mismatch_selfheal.py",
        "tests/test_position_invariant_monitors.py",
        "tests/test_concurrency_aware_sizing.py",
        "tests/test_runner_v1_floor_gate.py",
        "tests/test_stale_exit_price_and_regime_backoff.py",
    ]
    r = subprocess.run([sys.executable, "-m", "pytest", *files, "-q"],
                       capture_output=True, text=True)
    tail = (r.stdout or "").strip().splitlines()[-1] if r.stdout else "no output"
    check("incident regression suites", r.returncode == 0, tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-droplet", action="store_true")
    args = ap.parse_args()

    verify_sizing_cannot_overdeploy()
    verify_single_position_cap_still_binds()
    verify_slots_off_is_unchanged()
    verify_guards_wired()
    run_unit_suites()

    if args.with_droplet:
        host = "root@129.212.138.145"
        key = str(Path.home() / ".ssh" / "id_ed25519_do")
        for bot, slots, ent in (("kody", "3", None), ("dennis", "3", "0.70")):
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=20", "-i", key, host,
                 f"docker exec owlet-{bot} printenv CONCURRENCY_SIZING_SLOTS; "
                 f"docker exec owlet-{bot} printenv PAPER_TRADE; "
                 f"docker exec owlet-{bot} grep -c 'QUANTITY_MISMATCH' "
                 f"/app/options_owl/execution/position_monitor.py"],
                capture_output=True, text=True)
            out = (r.stdout or "").split()
            got_slots = out[0] if out else "?"
            paper = out[1] if len(out) > 1 else "?"
            has_fix = out[2] if len(out) > 2 else "0"
            check(f"deployed slots [{bot}]", got_slots == slots, f"got {got_slots}")
            check(f"deployed is LIVE [{bot}]", paper == "false", f"PAPER_TRADE={paper}")
            check(f"deployed has the fix [{bot}]", has_fix not in ("0", "?"),
                  f"QUANTITY_MISMATCH occurrences in container: {has_fix}")

    print("\nE2E SAFETY VERIFICATION\n" + "=" * 74)
    for status, name, detail in results:
        mark = "  ok " if status == PASS else "  ** "
        print(f"{mark}{name:<40} {detail}")
    bad = [r for r in results if r[0] == FAIL]
    print("=" * 74)
    print(f"{len(results) - len(bad)}/{len(results)} passed"
          + ("" if not bad else f"   {len(bad)} FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
