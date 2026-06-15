"""UW flow UNIVERSE ranking — which tickers get the most QUALIFYING whale sweeps market-wide
(ask-side >=60% + sweep + >=$250k), so we know which high-conviction names we're currently
BLIND to (no thetadata option premiums => can't backtest them yet). Flags coverage gaps.
Read-only. Recent window only (UW caps history ~3mo); ~80 pages/side = ranking is stable.
"""
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
KEY = next((ln.split("=", 1)[1].strip() for ln in (ROOT / ".env").read_text().splitlines()
            if ln.startswith("UNUSUAL_WHALES_API_KEY=")), "")
BASE = "https://api.unusualwhales.com/api/option-trades/flow-alerts"
HAVE = {"AAPL", "SPY", "GOOGL", "IWM", "MSTR", "AMZN", "AMD", "PLTR",
        "QQQ", "MSFT", "AVGO", "TSLA", "META", "NVDA"}  # thetadata coverage
MIN_PREM = 250_000


def tally(is_put, pages=80):
    hdr = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}
    c = Counter()
    older = None
    got = 0
    for _ in range(pages):
        p = {"limit": 200, "is_put": "true" if is_put else "false", "min_premium": MIN_PREM}
        if older:
            p["older_than"] = older
        r = requests.get(BASE, headers=hdr, params=p, timeout=20)
        if r.status_code != 200:
            print(f"  API {r.status_code} (is_put={is_put}) — stop at {got} rows"); break
        d = r.json().get("data", [])
        if not d:
            break
        for x in d:
            tot = float(x.get("total_premium") or 0)
            askf = float(x.get("total_ask_side_prem") or 0) / tot if tot > 0 else 0
            if tot >= MIN_PREM and askf >= 0.6 and bool(x.get("has_sweep")):
                c[str(x.get("ticker", "")).upper()] += 1
        got += len(d)
        older = min(x["created_at"] for x in d)
        time.sleep(0.4)
    return c, got


def main():
    for is_put, label in [(True, "PUT"), (False, "CALL")]:
        print(f"\n=== {label} qualifying whale sweeps — top 30 tickers market-wide ===")
        c, got = tally(is_put)
        print(f"(scanned {got} alerts; {sum(c.values())} qualified across {len(c)} tickers)")
        print(f"{'rank':<5}{'tk':<7}{'sweeps':>7}  coverage")
        for i, (tk, n) in enumerate(c.most_common(30), 1):
            cov = "have" if tk in HAVE else "  ** NO DATA — candidate to add **"
            print(f"{i:<5}{tk:<7}{n:>7}  {cov}")
        gaps = [tk for tk, _ in c.most_common(30) if tk not in HAVE]
        if gaps:
            print(f"  >> High-flow {label} names we CAN'T backtest yet (add to thetadata): {', '.join(gaps)}")


if __name__ == "__main__":
    main()
