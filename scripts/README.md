# scripts/ — What to Use

## Canonical Tools (source of truth)

| Script | Purpose |
|---|---|
| `backtest_gold_standard.py` | **THE canonical production-parity backtest.** Imports the real `ExitFSM` from `options_owl/risk/exit_v5/` so exits behave exactly like production. Use this for all strategy evaluation. Any number that doesn't come from this script should be treated as unverified. |
| `trade-pnl.py` | **The canonical P&L tool.** Correctly groups DCA + scaleout trade families and uses the best available fill prices (Webull > paper). Never use raw `SUM(pnl_dollars)` from the DB. |
| `backtest_put_scalp.py` | PUT exit-config backtest — feeds `PUT_SCALP_CONFIG` (no profit ceiling, CALL-style trailing). Kept because PUT config changes are validated here. |

## Model Training

All `train_*.py` scripts are the model trainers (e.g. `train_ml_models_v3.py`,
`train_put_pattern.py`, `train_pattern_entry.py`, `train_peak_detector*.py`, ...).
Trained models land in `journal/models/` — note `journal/models/` is NOT synced by
`rebuild.sh`; SCP new models to the droplet manually.

`backtest_ladder_report.py` remains here only because `train_peak_detector.py`
imports `load_signals`/`load_ticks`/`size_contracts` from it.

## Operational / Deploy

`rebuild.sh` (the only sanctioned deploy path), `restart-staggered.sh`,
`trade-log.sh`, `babysit.sh` / `setup-babysit.sh`, `deploy.sh`, `provision.sh`,
`harvester-sync.sh`, `reconcile_local.py`, `export_pg_to_thetadata.py`,
`backfill_*.py`, `download_*.py`, `check_*.py`, `fix_*.py`, `webull_*.py`,
`show_trades.py`, `paper_report.py`, `agent_summary.py`, etc.

## archive_backtests/

Superseded one-off backtest / replay / sweep experiments. They were kept for
historical reference only. **Do not trust their numbers** — they predate the
current V5/V6 production config and most do not use the real `ExitFSM`. For any
current evaluation, use `backtest_gold_standard.py`.
