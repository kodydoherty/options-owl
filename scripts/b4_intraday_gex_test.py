"""B4: does the INTRADAY SPY gamma regime (at entry) separate flow outcomes? Daily GEX was refuted;
this tests the 1-min /spot-exposures signal — negative dealer gamma = trending/amplifying (good for
directional 0DTE), positive = pinning/mean-reverting. Point-in-time (gamma as of entry minute, no
lookahead). Reuses the cached flow trades + caches SPY GEX. Read-only.
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, "scripts")
import uw_ticker_discovery as D  # noqa: E402

_spec = importlib.util.spec_from_file_location("b2", "scripts/b2_pointintime_cached.py")
b2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b2)

H = {"Authorization": f"Bearer {D.KEY}", "Accept": "application/json"}
SE = "https://api.unusualwhales.com/api/stock/SPY/spot-exposures"
CACHE = Path("/tmp/b4_spy_gex.pkl")


def gex_timelines(days):
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    tl = {}
    for d in days:
        try:
            r = requests.get(SE, headers=H, params={"date": d}, timeout=15)
            ticks = r.json().get("data", []) if r.status_code == 200 else []
        except requests.exceptions.RequestException:
            ticks = []
        out = []
        for t in ticks:
            ts = str(t.get("time", ""))  # UTC ISO
            try:
                mi = (int(ts[11:13]) - 13) * 60 + int(ts[14:16]) - 30  # 13:30 UTC = 9:30 ET open
                out.append((mi, float(t.get("gamma_per_one_percent_move_oi") or 0)))
            except (ValueError, IndexError):
                pass
        tl[d] = sorted(out)
        time.sleep(0.3)
    CACHE.write_bytes(pickle.dumps(tl))
    return tl


def gamma_at(timeline, mi):
    g = None
    for m, gg in timeline:
        if m <= mi:
            g = gg
        else:
            break
    return g


def main():
    print("building flow trades (cached) + SPY intraday GEX...", flush=True)
    df = b2.flow_trades()
    tl = gex_timelines(sorted(df["date"].unique()))
    df["gamma"] = df.apply(lambda r: gamma_at(tl.get(r["date"], []), r["mb"]), axis=1)
    df = df.dropna(subset=["gamma"])
    df["trending"] = df.gamma < 0
    print(f"\n=== B4 intraday SPY gamma regime at entry (no lookahead) — {len(df)} trades ===")
    print(f"  TRENDING (dealer γ<0): {b2._stat(df[df.trending]['ret'])}")
    print(f"  PINNING  (γ>=0)      : {b2._stat(df[~df.trending]['ret'])}")
    print("  by side:")
    for side in ("call", "put"):
        s = df[df.side == side]
        print(f"    {side} trending: {b2._stat(s[s.trending]['ret'])}")
        print(f"    {side} pinning : {b2._stat(s[~s.trending]['ret'])}")
    print("\nIf one regime clearly beats the other (esp. per side), the intraday GEX regime is a gate/tilt.")


if __name__ == "__main__":
    main()
