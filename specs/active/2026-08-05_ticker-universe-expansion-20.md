# Ticker Universe Expansion — 20 New 0DTE Candidates

**Status:** ACTIVE (research → download → backtest → promote winners)
**Created:** 2026-08-05
**Goal:** Find 20 new tickers to trade 0DTE like the current book, download thetadata, backtest on the honest-fill gold-standard harness, promote winners to `TICKERS` + `HARVEST_UNIVERSE`.

## Market-structure facts (verified Aug 2026)
- **Only 6 products have TRUE daily M–F 0DTE:** SPX, SPY, XSP, NDX, QQQ, IWM. We already run SPY/QQQ/IWM.
- **SPX / XSP / NDX = the deepest 0DTE in existence** (SPX+SPY ≈ 70% of all 0DTE volume) BUT they are **cash-settled index options — thetadata does NOT carry them** (equities/ETFs only). → **live-only scaling track**, backtest by SPY-proxy (XSP ≈ SPX/10 ≈ SPY×10). This is the capacity unlock from [[capacity-scaling-to-100k-2026-07-30]], not part of the 20 testable below.
- **No single stock has Tue/Thu expirations — every single name is MWF at best.** Our current names already trade at max cadence. The untapped daily/near-daily liquidity is almost entirely in **ETFs**, which our book barely touches.
- Current book = ~22 names, ALL tech/mega-cap single stocks + SPY/QQQ/IWM. **Zero sector / commodity / rate / crypto / leveraged exposure.** That concentration is the gap these 20 fill.

## Selection criteria
1. **thetadata-downloadable** (equity or ETF — NOT index).
2. **Frequent 0DTE** (MWF minimum; SPY/QQQ/IWM-class daily already held).
3. **Liquid** (top-tier options volume / tight-ish spreads — the fill wall matters).
4. **Diversifying** (away from the all-tech single-name concentration).
5. **High enough intraday range** — 0DTE needs movement (leveraged ETFs & high-beta names favored).
6. Not already **refuted** — SLV/USO/GDX lost on the *fantasy* harness; flagged for honest-harness RE-TEST, not blind add.

## The 20 candidates

### Tier 1 — Leveraged index ETFs (highest 0DTE momentum + liquid + index-diversified) — 5
| # | Ticker | What | Why |
|---|---|---|---|
| 1 | **TQQQ** | 3× Nasdaq-100 | One of the most liquid 0DTE ETFs; 3× intraday range = momentum the strategy rides, no single-name gap risk |
| 2 | **SOXL** | 3× semis | Semis exposure amplified; very active options; rides the AI/semi tape |
| 3 | **SPXL** | 3× S&P 500 | Leveraged SPY — same edge, bigger moves |
| 4 | **TNA** | 3× Russell 2000 | Leveraged IWM; small-cap beta, big intraday swings |
| 5 | **SQQQ** | −3× Nasdaq | **Put-side / inverse** — a clean bearish vehicle for down days (calls on SQQQ = market puts) |

### Tier 2 — Crypto & retail high-vol proxies — 3
| 6 | **IBIT** | BlackRock Bitcoin ETF | MWF 0DTE, enormous volume, crypto momentum WITHOUT MSTR single-name blowup risk |
| 7 | **HOOD** | Robinhood | High-beta retail flow, liquid options, moves hard intraday |
| 8 | **MARA** | Bitcoin miner | Extreme intraday vol (crypto-levered) — high risk/high 0DTE payoff; size small |

### Tier 3 — Sector ETFs (true diversification from tech) — 5
| 9 | **XLF** | Financials | In the daily-0DTE coverage list; bank/rate tape, uncorrelated to tech |
| 10 | **XLE** | Energy | Oil/energy beta, moves on macro not earnings |
| 11 | **SMH** | Semis ETF | RE-TEST (mixed on fantasy harness); the "SPY of semis," liquid |
| 12 | **KRE** | Regional banks | Spikes hard on bank-stress episodes — event-driven 0DTE |
| 13 | **XBI** | Biotech | High vol, catalyst-driven, low tech correlation |

### Tier 4 — Commodity / rate ETFs (macro diversification) — 4
| 14 | **GLD** | Gold | Liquid, risk-off vehicle — WINS when equities crash (natural hedge diversifier) |
| 15 | **TLT** | 20yr Treasury | Rate-driven, very active on Fed/CPI/NFP days — a whole new catalyst set |
| 16 | **SLV** | Silver | RE-TEST on honest harness (lost on fantasy); high vol |
| 17 | **UNG** | Natural gas | Among the MOST volatile ETFs — big 0DTE range; size small |

### Tier 5 — Liquid single-name diversifiers (MWF 0DTE) — 3
| 18 | **CRM** | Salesforce | Mega-cap software, MWF, liquid — enterprise-tech beta distinct from our megacaps |
| 19 | **QCOM** | Qualcomm | Semis single-name, MWF, liquid |
| 20 | **UBER** | Uber | High-beta consumer tech, liquid options, distinct catalyst set |

## Download plan
thetadata download REQUIRES greeks (NOT `--ohlc-only` — greeks-less runs go hollow, see [[thetadata-greeks-required-for-backtest-2026-07-16]]). Greeks depth caps the window, so download full history:

```bash
# one ticker (repeat / parallelize per ticker)
python scripts/download_thetadata.py --ticker TQQQ --start 2024-01-01 --otm 4 --otm-below 8
```
- `--otm-below 8` for crash-day PUT coverage (leveraged/high-vol names gap hard).
- Parallelize across tickers (independent); watch thetadata rate limits.

## Test plan (per ticker, after download)
1. Backtest on the **honest-fill** gold-standard harness (fill-miss modeling ON):
   `python scripts/backtest_gold_standard.py --days 126 --tickers <T> --pattern-threshold 0.62 --no-entry-filter`
2. Promote only **honest-harness winners** (PF > ~1.5, positive at realistic fills) to `TICKERS` + `HARVEST_UNIVERSE`, CALL-side first (puts separately validated), flag-gated like `ENABLE_EXPANSION_TICKERS`.
3. Leveraged/crypto names: watch fill realism (thin OTM books) and size conservatively.

## Parallel live-only track (not backtestable on thetadata)
**SPX / XSP / NDX** — the real capacity unlock (10× depth, cash-settled, true daily). Backtest via SPY/QQQ proxy; validate fills live on a paper bot. Pursue separately from the 20 above.
