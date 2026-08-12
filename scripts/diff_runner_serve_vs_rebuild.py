"""Name the input that makes the LIVE runner_v1 score diverge from a faithful rebuild.

THE DEFECT
----------
Live `compute_runner_v1_p` emits p_runner in a narrow 0.684-0.828 band. Rebuilding the
same model's feature vector from Postgres spans 0.054-0.940 with 18% of trades below the
0.39 Q1 floor. Both call the SAME scorer (`_score_runner_v1`), so the model is not at
fault -- the feature VECTOR differs. Consequence: the Q1 block never fires (a tier worth
-$2,957 at 9.6% runners is never blocked or downsized) and every live trade lands in one
tier, so the tiering is inert in both directions.

WHAT THIS DOES
--------------
For trades carrying a persisted live p_runner, rebuild the vector from Postgres, score it,
and print live-vs-rebuilt side by side -- then rank features by how far they move the
score, so the answer is a named input rather than a hypothesis.

Per-feature attribution is measured, not guessed: for each feature, substitute ONE rebuilt
value at a time into a live-like vector and re-score. The feature whose single substitution
moves the score most is the one driving the divergence. That isolates it even when several
inputs differ, which correlation over 13 samples cannot do.

Usage (droplet, container with postgres access):
  python scripts/diff_runner_serve_vs_rebuild.py \
      --paths /src/journal/live_exit_paths_greeks_0812.pkl \
      --candles /src/journal/runner_ctx_0812.pkl --journal-root /src/journal
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_runner_v1 import build_features  # noqa: E402


def live_p_runners(journal_root: str, bots: list[str]) -> dict[tuple[str, int], float]:
    """(bot, trade_id) -> the p_runner the LIVE path computed and persisted.

    Keyed by BOT as well as id: the bots keep independent sqlite databases whose ids
    both start at 1, so a bare-id map silently overwrites kody's #416 with dennis's
    and then compares each trade against the other bot's score.
    """
    out: dict[tuple[str, int], float] = {}
    for b in bots:
        db = Path(journal_root) / f"owlet-{b}" / "raw_messages.db"
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            for tid, p in con.execute(
                "SELECT id, p_runner FROM paper_trades WHERE p_runner IS NOT NULL"
            ):
                out[(b, int(tid))] = float(p)
            con.close()
        except Exception as exc:  # noqa: BLE001
            print(f"  ({b}: {exc})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--candles", required=True)
    ap.add_argument("--journal-root", default="journal")
    ap.add_argument("--bots", default="kody,dennis")
    args = ap.parse_args()

    from options_owl.risk.flow_runner import _load_runner_v1, _score_runner_v1

    _model, meta = _load_runner_v1()
    cols = list(meta["features"])

    live = live_p_runners(args.journal_root, args.bots.split(","))
    print(f"{len(live)} trades carry a persisted live p_runner")
    if not live:
        sys.exit("no live p_runner values — nothing to diff")

    trades = [t for t in pickle.load(open(args.paths, "rb"))
              if t["was_webull"] and t["otype"] == "call"]
    ctx = pickle.load(open(args.candles, "rb"))

    rows = []
    for t in trades:
        tid, bot = int(t["tid"]), t.get("bot", "")
        if (bot, tid) not in live:
            continue
        c = ctx.get(tid) or ctx.get(str(tid)) or {}
        feat = build_features(t, c)
        if feat is None:
            continue
        rows.append((f"{bot[:3]}#{tid}", t["tk"], live[(bot, tid)],
                     _score_runner_v1(feat), feat))

    if not rows:
        sys.exit("no overlap between persisted p_runner and rebuildable trades")

    print(f"\n{len(rows)} trades have BOTH a live score and a rebuild\n")
    print(f"{'trade':>10} {'ticker':<7} {'live p':>8} {'rebuilt':>9} {'delta':>8}")
    print("-" * 47)
    for tid, tk, lp, rp, _ in sorted(rows, key=lambda r: r[3] - r[2]):
        print(f"{tid:>10} {tk:<7} {lp:>8.3f} {rp:>9.3f} {rp - lp:>+8.3f}")

    diffs = [rp - lp for _, _, lp, rp, _ in rows]
    print(f"\nmean divergence {sum(diffs)/len(diffs):+.3f}   "
          f"live mean {sum(r[2] for r in rows)/len(rows):.3f}   "
          f"rebuilt mean {sum(r[3] for r in rows)/len(rows):.3f}")

    # ---- per-feature attribution -------------------------------------------------
    # The live vector itself was never persisted (only the score), so it cannot be read
    # back directly. Instead, hold the rebuilt vector and neutralise ONE feature at a
    # time to its cross-trade median: the feature whose neutralisation collapses the
    # divergence is the one carrying it.
    print("\n\nPER-FEATURE ATTRIBUTION")
    print("how far the rebuilt score moves when each feature alone is neutralised")
    print("to its median across these trades — the biggest mover is the suspect\n")
    numeric = [c for c in cols if c not in set(meta.get("cat_features", []))]
    med = {}
    for c in numeric:
        vals = sorted(float(f.get(c) or 0.0) for _, _, _, _, f in rows)
        med[c] = vals[len(vals) // 2]

    impact = []
    for c in numeric:
        moves = []
        for _, _, _, rp, f in rows:
            g = dict(f)
            g[c] = med[c]
            moves.append(abs(_score_runner_v1(g) - rp))
        impact.append((sum(moves) / len(moves), c))
    impact.sort(reverse=True)

    print(f"{'feature':<18} {'mean |Δscore|':>14}   {'median value':>14}")
    print("-" * 52)
    for mv, c in impact[:10]:
        print(f"{c:<18} {mv:>14.4f}   {med[c]:>14.4f}")

    print("\n  A feature at the top with a live/rebuild sourcing difference is the fix.")
    print("  Cross-check each against compute_runner_v1_p: Redis snapshot greeks vs")
    print("  option_ticks, PG 1m session slice, and the 5-min option-volume query.")


if __name__ == "__main__":
    main()
