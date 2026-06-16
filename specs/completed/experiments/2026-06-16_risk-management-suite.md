# Risk-Management Suite — Profit-Lock, Take-Profit Cap, Fleet Staggering (2026-06-16)

**Status:** COMPLETED — all deployed live to all 5 bots, verified in containers.
**Category:** experiments / risk-management

## Context
After the first live anti-martingale adds (TSLA/AMZN/SPY) and a bug-polluted two weeks (flow didn't
trade until 2026-06-15; webull rejects; GEX error; scan timeouts — all fixed prior), we audited the
strategy's risk profile via backtest and shipped a suite of risk controls. The live last-2-weeks
losses (−$926 call / −$921 put on kody) were the BUGS, not the strategy — the clean backtest of the
deployed config is PF 1.47 / 60% WR, profitable every week.

## Changes shipped (all flag-gated, validated by backtest)

| # | Change | Flag | Verdict |
|---|---|---|---|
| 1 | CALL profit-lock (keep 60% of peak gain once +30%) | `ENABLE_V7_PROFIT_LOCK=true` | KEEP — +7% call P&L, +3pts WR, consistent per-month; the runner-capture win. PUTs exempt (ride slow crashes). |
| 2 | Anti-martingale adds | `ENABLE_ANTIMARTINGALE_ADD=false` | OFF — flat-+EV but DILUTES book PF (1.47→1.18) + 2× drawdown → wealth-reducer when compounding ($2.1M→$0.6M). Separate-leg + safety code stays dormant. |
| 3 | Take-profit sizing cap (freeze sizing at $50k, bank excess) | `MAX_SIZING_BALANCE=50000` | DEPLOYED — biggest risk lever; turns ~−47% DD into ~−9%. |
| 4 | Fleet staggering K=2 (each signal → 2 of 5 bots, priority round-robin) | `ENABLE_FLEET_STAGGER=true`, `FLEET_OVERLAP=2` | DEPLOYED — decorrelates the fleet (corr 0.95→0.22, fleet DD 5× smaller). Priority kody=0/adam=1/dennis=2/yank=3/vinny=4. |

## Backtest evidence (60d, cached flow + ML, measured PG liquidity/spreads)

**Take-profit cap (single account):** capping turns −47% DD → −9% (ret/DD 1.5→9.9) for ~25% less P&L.

**Fleet staggering (5 bots, $44.4k total capital, $50k cap each):**
| scheme | fleet P&L | fleet DD | corr | worst day | ret/DD |
|---|---|---|---|---|---|
| K=5 (identical, old) | $254k | −$102k | 0.95 | −$51k | 2.5 |
| K=3 | $180k | −$49k | 0.44 | −$23k | 3.7 |
| **K=2 (deployed)** | $126k | **−$21k** | **0.22** | −$11k | **6.0** |

## DEPLOYED config — per-account 60-day projection (K=2 + $50k cap)
| bot (rank) | start | P&L | end | maxDD | ret% |
|---|---|---|---|---|---|
| **kody (r0)** | **$23,000** | **+$74,076** | **$97,076** | **−$15,385** | **+322%** |
| adam (r1) | $4,685 | +$16,202 | $20,887 | −$5,272 | +346% |
| dennis (r2) | $10,000 | +$19,594 | $29,594 | −$5,849 | +196% |
| yank (r3) | $3,600 | +$6,332 | $9,932 | −$2,696 | +176% |
| vinny (r4) | $3,123 | +$10,068 | $13,191 | −$5,513 | +322% |

vs old endless-compound kody: ~$237k end but −$140k (−47%) drawdown. The suite trades upside for a
vastly smoother ride (kody DD −$15k / −16%).

## Honest caveats
- $ are ILLUSTRATIVE — the backtest fills at cached mid/close; real 0DTE fills (esp. exits in fast
  moves) are worse, and there's no edge-decay/outage modeling. Treat as an optimistic ceiling; the
  trustworthy edge is the flat-$750 measure (+$52.6k/60d). Rankings/drawdown-shape are robust.
- Liquidity is NOT the brake at these account sizes (SPY trades 192k+ contracts/day); the brakes are
  fill quality, drawdown tolerance, and live edge decay.
- Biggest remaining lever: widen the ticker universe (we watch ~12-32 of hundreds) — each needs validation.

## Scripts
`backtest_add_handling.py`, `backtest_add_validate.py` (profit-lock + adds), `backtest_v7_antimg_compound.py`
(compound), `backtest_current_strat.py` (flat edge), `backtest_realistic_account.py` (liquidity-grounded
+ take-profit cap), `backtest_fleet_stagger.py` (staggering + per-account projection).

## Forward test
Now that the bugs are fixed and the risk suite is live, the real proof is forward: live should converge
toward PF ~1.47 / 60% WR, with the fleet's bad days decorrelated (no synchronized −$51k crater).
