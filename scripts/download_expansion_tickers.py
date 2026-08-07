#!/usr/bin/env python3
"""Parallel thetadata downloader for the 20 universe-expansion candidates.

See specs/active/2026-08-05_ticker-universe-expansion-20.md. Downloads full-history option
data (WITH greeks — greeks-less runs go hollow, see thetadata-greeks-required memory) for each
candidate into the shared thetadata_options.db. Tickers are processed HIGHEST-CONFIDENCE-FIRST
so we get backtestable signal on the best candidates soonest.

Robustness:
  * Concurrency capped (default 4) — respects the thetadata terminal's rate limits AND keeps
    WAL write-contention low (the writes are a low duty-cycle vs the network fetches).
  * Each ticker's download is retried up to 3× on failure — the underlying downloader is
    idempotent (skips dates already in download_log), so a retry cleanly RESUMES, never redoes.
  * Per-ticker logs in journal/thetadata_expansion_logs/<TICKER>.log; live status to stdout.

Usage:
  python scripts/download_expansion_tickers.py                  # all 20, concurrency 4
  python scripts/download_expansion_tickers.py --concurrency 3  # gentler on the terminal
  python scripts/download_expansion_tickers.py --only TQQQ,SOXL # subset
  python scripts/download_expansion_tickers.py --start 2024-01-01
"""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_DIR / "journal" / "thetadata_expansion_logs"
STATUS_PATH = LOG_DIR / "_status.json"


def _load_status() -> dict:
    """Persistent cross-run progress: {ticker: {status, attempts, sec, round, ts}}.
    Lets a re-run (after a crash/sleep/terminal outage) skip already-COMPLETED tickers
    instantly instead of re-scanning them."""
    try:
        return json.loads(STATUS_PATH.read_text())
    except Exception:
        return {}


def _save_status(status: dict) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(STATUS_PATH)  # atomic — never leaves a half-written status file
    except Exception as exc:
        print(f"  (status save failed: {exc})", flush=True)

# Ordered HIGHEST-CONFIDENCE-FIRST (mirror tapes we already win on → speculative last).
# (ticker, tier-note) — see the spec for full rationale.
CANDIDATES = [
    ("TQQQ", "3x Nasdaq — we win QQQ, most liquid lev ETF"),
    ("SOXL", "3x semis — we win NVDA/AMD/semis"),
    ("SPXL", "3x S&P — we win SPY"),
    ("TNA",  "3x Russell — we win IWM"),
    ("SMH",  "semis ETF — re-test, semis winners"),
    ("QCOM", "semis single, MWF like our book"),
    ("CRM",  "megacap software, MWF like our book"),
    ("UBER", "high-beta single, MWF"),
    ("XLF",  "financials sector, liquid"),
    ("GLD",  "gold — liquid macro hedge (wins on crashes)"),
    ("TLT",  "20yr treasury — rates/Fed days"),
    ("SQQQ", "-3x Nasdaq — down-day vehicle"),
    ("IBIT", "Bitcoin ETF — liquid crypto, no MSTR blowup"),
    ("HOOD", "high-beta retail flow"),
    ("XLE",  "energy sector"),
    ("KRE",  "regional banks — bank-stress spikes"),
    ("XBI",  "biotech — catalyst-driven"),
    ("SLV",  "silver — re-test on honest harness"),
    ("UNG",  "natgas — extreme vol, size small"),
    ("MARA", "crypto miner — extreme vol/risk"),
]

MAX_RETRIES = 3


def download_one(ticker: str, note: str, start: str, otm: int, otm_below: int) -> dict:
    """Run the thetadata downloader for one ticker, retrying (idempotent resume) on failure."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{ticker}.log"
    cmd = [
        sys.executable, str(PROJECT_DIR / "scripts" / "download_thetadata.py"),
        "--ticker", ticker, "--start", start,
        "--otm", str(otm), "--otm-below", str(otm_below),
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        with open(log_path, "a") as lf:
            lf.write(f"\n===== attempt {attempt}/{MAX_RETRIES}  cmd: {' '.join(cmd)} =====\n")
            lf.flush()
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        if rc == 0:
            return {"ticker": ticker, "status": "OK", "attempts": attempt, "sec": round(elapsed)}
        print(f"  [{ticker}] attempt {attempt} FAILED (rc={rc}, {elapsed:.0f}s) — "
              f"{'retrying (resumes)' if attempt < MAX_RETRIES else 'giving up'}", flush=True)
        time.sleep(5)
    return {"ticker": ticker, "status": "FAILED", "attempts": MAX_RETRIES, "sec": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=4, help="parallel downloads (default 4)")
    ap.add_argument("--start", type=str, default="2024-01-01", help="history start (default 2024-01-01)")
    ap.add_argument("--otm", type=int, default=4, help="OTM strikes above ATM")
    ap.add_argument("--otm-below", type=int, default=8, help="OTM strikes below ATM (crash-day put coverage)")
    ap.add_argument("--only", type=str, default=None, help="comma-separated subset of tickers")
    ap.add_argument("--rounds", type=int, default=6,
                    help="self-heal rounds: re-attack still-failed tickers across terminal outages (default 6)")
    ap.add_argument("--round-delay", type=int, default=120, help="seconds between self-heal rounds")
    ap.add_argument("--fresh", action="store_true", help="ignore saved status and re-attempt everything")
    args = ap.parse_args()

    todo = CANDIDATES
    if args.only:
        want = {t.strip().upper() for t in args.only.split(",") if t.strip()}
        todo = [(t, n) for (t, n) in CANDIDATES if t in want]

    status = {} if args.fresh else _load_status()
    already = {t for t, s in status.items() if s.get("status") == "OK"}
    if already:
        print(f"Resuming — {len(already)} already complete, skipping: {', '.join(sorted(already))}", flush=True)

    print(f"Downloading {len(todo)} tickers, concurrency={args.concurrency}, start={args.start}, "
          f"rounds={args.rounds}", flush=True)
    print(f"Order (highest-confidence-first): {', '.join(t for t, _ in todo)}\n", flush=True)

    # Self-heal rounds: each round runs every not-yet-OK ticker; a ticker that fails all its
    # in-run retries is picked up again next round (after the underlying thetadata terminal has
    # had time to recover). One invocation drives the whole job to completion across outages.
    for rnd in range(1, args.rounds + 1):
        pending = [(t, n) for (t, n) in todo if status.get(t, {}).get("status") != "OK"]
        if not pending:
            break
        print(f"\n########## ROUND {rnd}/{args.rounds} — {len(pending)} pending: "
              f"{', '.join(t for t, _ in pending)} ##########", flush=True)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(download_one, t, n, args.start, args.otm, args.otm_below): t
                    for (t, n) in pending}
            for fut in concurrent.futures.as_completed(futs):
                r = fut.result()
                r["round"] = rnd
                r["ts"] = int(time.time())
                status[r["ticker"]] = r
                _save_status(status)                       # persist after EVERY ticker
                done += 1
                nok = sum(1 for s in status.values() if s.get("status") == "OK")
                print(f"[r{rnd} {done}/{len(pending)} | total OK {nok}/{len(todo)}] "
                      f"{r['ticker']:5s} {r['status']:6s} ({r['attempts']} try, {r['sec']}s)", flush=True)
        still_bad = [t for (t, _) in todo if status.get(t, {}).get("status") != "OK"]
        if still_bad and rnd < args.rounds:
            print(f"  round {rnd} left {len(still_bad)} failed ({', '.join(still_bad)}) — "
                  f"waiting {args.round_delay}s before round {rnd + 1}", flush=True)
            time.sleep(args.round_delay)

    ok = sorted(t for (t, _) in todo if status.get(t, {}).get("status") == "OK")
    bad = sorted(t for (t, _) in todo if status.get(t, {}).get("status") != "OK")
    print(f"\n===== DONE: {len(ok)}/{len(todo)} OK =====", flush=True)
    print(f"OK:     {', '.join(ok)}", flush=True)
    if bad:
        print(f"FAILED: {', '.join(bad)}  (just re-run the script — resumes from status + download_log)", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
