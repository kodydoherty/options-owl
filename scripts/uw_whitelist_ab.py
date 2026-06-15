"""Split-half validation of the CALL whitelist recommendation (UW retains only ~3mo of flow
history, so a true 2nd window is impossible — this splits the one window into early/late halves).
Compares OLD vs NEW call whitelist total return in BOTH halves. NEW wins both => validated.
Reads journal/v3_eval_results/uw_ticker_discovery_monthly.csv (written by uw_ticker_discovery.py).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M = pd.read_csv(ROOT / "journal" / "v3_eval_results" / "uw_ticker_discovery_monthly.csv")

OLD_CALL = ["TSLA", "AAPL", "AMD", "AVGO", "PLTR"]
NEW_CALL = ["TSLA", "AAPL", "AMD", "PLTR", "META", "SPY", "AMZN"]  # +META/SPY/AMZN, -AVGO
H1 = {"2026-03", "2026-04"}   # early half
H2 = {"2026-05", "2026-06"}   # late half


def wl_total(side, wl, months):
    s = M[(M["side"] == side) & (M["tk"].isin(wl)) & (M["month"].isin(months))]
    return s["total"].sum(), int(s["n"].sum())


print("=== CALL whitelist split-half A/B (total return %, $750-equiv sleeve per trade) ===")
print(f"{'whitelist':<10}{'H1(Mar-Apr)':>16}{'H2(May-Jun)':>16}{'FULL':>14}")
for name, wl in [("OLD", OLD_CALL), ("NEW", NEW_CALL)]:
    h1, n1 = wl_total("call", wl, H1)
    h2, n2 = wl_total("call", wl, H2)
    full = h1 + h2
    print(f"{name:<10}{h1:>+11.0f}(n{n1:<3}){h2:>+11.0f}(n{n2:<3}){full:>+14.0f}")

oh1 = wl_total("call", OLD_CALL, H1)[0]; oh2 = wl_total("call", OLD_CALL, H2)[0]
nh1 = wl_total("call", NEW_CALL, H1)[0]; nh2 = wl_total("call", NEW_CALL, H2)[0]
print("\n=== per-candidate (the change) — must hold sign in BOTH halves ===")
print(f"{'tk':<6}{'role':<8}{'H1':>10}{'H2':>10}")
for tk, role in [("META", "ADD"), ("SPY", "ADD"), ("AMZN", "ADD"), ("AVGO", "DROP")]:
    h1 = M[(M.side == "call") & (M.tk == tk) & (M.month.isin(H1))]["total"].sum()
    h2 = M[(M.side == "call") & (M.tk == tk) & (M.month.isin(H2))]["total"].sum()
    print(f"{tk:<6}{role:<8}{h1:>+10.0f}{h2:>+10.0f}")

both = nh1 > oh1 and nh2 > oh2
adds_ok = all(M[(M.side == "call") & (M.tk == tk) & (M.month.isin(H1))]["total"].sum() > 0
              and M[(M.side == "call") & (M.tk == tk) & (M.month.isin(H2))]["total"].sum() > 0
              for tk in ["META", "SPY", "AMZN"])
avgo_bad = all(M[(M.side == "call") & (M.tk == "AVGO") & (M.month.isin(h))]["total"].sum() <= 0
               for h in (H1, H2))
print(f"\nNEW beats OLD in both halves: {both}")
print(f"All 3 adds positive in both halves: {adds_ok}")
print(f"AVGO negative in both halves (drop justified): {avgo_bad}")
print("VERDICT:", "VALIDATED — deploy new call whitelist" if (both and avgo_bad)
      else "MIXED — review before deploying")
sys.exit(0)
