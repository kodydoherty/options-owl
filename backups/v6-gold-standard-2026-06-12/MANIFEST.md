# V6 Gold-Standard Backup — frozen 2026-06-12

Full snapshot of the **V6 production logic** (the live, deployed system) and the
canonical **gold-standard backtest** harness, taken before V7 (convex redesign)
deployment prep. This is the rollback/reference baseline. **Read-only — never edit
files here; they are a frozen reference.**

## Provenance
- **Git commit:** `f9497c2a727d03e9082143360ad8a8978ff72d12`
- **Branch:** `fix/owl-review-bugfixes`
- **Date frozen:** 2026-06-12
- **Models in use at freeze:** `journal/models/ml_v3/` (meta JSONs copied to `models_meta/`).
  Note: models were retrained on extended 2025 data on 2026-06-11/12 — the meta here
  reflects the **post-retrain** models. The pre-retrain binaries are backed up separately
  at `journal/models/ml_v3_backup_20260611_124224/`.

## What's inside
| Dir | Contents | Role in V6 |
|---|---|---|
| `risk/` | full `options_owl/risk/` tree | exit FSM (`exit_v5/`), entry pipeline, sizing (`vinny_strategy.py`), manager, regime, greeks |
| `execution/` | position_monitor, paper_trader, webull_executor, alerts | the live execution path that runs the V6 logic |
| `config/` | `settings.py` | all `ENABLE_V6_*` flags + risk params (code defaults) |
| `gold_standard/` | `backtest_gold_standard.py` | canonical V6 backtest harness (the "gold standard") |
| `models_meta/` | `*_meta.json` | feature lists + thresholds for every model V6 loads |

## Live production V6 configuration (captured from .env / docker-compose at freeze)
```
EXIT_ENGINE=v5                      # V5 FSM (carries the V6 enhancements)
ENABLE_V6_PER_TICKER_CONFIG=true
ENABLE_V6_BREAKEVEN_RATCHET=true    # once +20%, stop floor = entry price
ENABLE_V6_2PM_TIGHTEN=true          # tighten adaptive trails 30% after 2PM ET
ENABLE_V6_SCALEOUT=true             # sell 1/3 at +20% (one-shot)
ENABLE_V6_SPREAD_GATE=true
ENABLE_V6_EARLY_POP_GATE=true
ENABLE_V6_SIDEWAYS_SCALP=true
ENABLE_V6_DCA=true                  # auto-double on 15–35% premium dip
ENABLE_V6_PREMIUM_CAP=false         # NOTE: disabled in production
MAX_CONCURRENT=8
MAX_PORTFOLIO_RISK_PCT=75–80        # (kody 75; some bots 80)
MAX_POSITION_PCT=15–20
```

## V6 exit gate priority (first match wins — reference)
1. eod_cutoff (0DTE, 15min pre-close) · 2. bid_disappearance (30s) · **5min grace
(backstop still fires: 0DTE −65% / multi-day −75%)** · 3. profit_target (index 0DTE 30%)
· 3.5 breakeven_ratchet · 3.7 scaleout · 4. scalp_trail · 5. checkpoint_cut (0DTE) ·
6. graduated_stop · 7. soft_trail · 8. adaptive_trail (category-aware, primary) ·
9. theta_exit. Full detail in repo `CLAUDE.md`.

## How to restore V6 (if V7 must be rolled back)
```bash
# Code: V6 lives at git f9497c2. To restore a single module:
git checkout f9497c2 -- options_owl/risk/<file>.py
# Or restore from this backup directly:
cp backups/v6-gold-standard-2026-06-12/risk/<file>.py options_owl/risk/<file>.py

# Models (pre-retrain V6 binaries):
cp -R journal/models/ml_v3_backup_20260611_124224/. journal/models/ml_v3/

# Flags: set ENABLE_V7_* = false in docker-compose.yml, then `docker compose up -d`
```

## Gold-standard reference result (V6, per CLAUDE.md)
V5/V6 FSM backtested **$21,685 over 161 trades**. This is the bar V7 must beat on
**risk-adjusted return across multiple OOS windows** before deployment.
