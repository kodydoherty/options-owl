"""Generate the OptionsOwl strategy write-up as a .docx (audit-grade, for external review).

Covers: edge thesis, data/signal sources, every entry gate (run order, from pipeline.py),
the sizing stack, the exit FSM (priority order + tiers + DTE + profit-lock), PUT pipeline,
risk controls, worked example trades on real historical data, backtest validation, and the
honest limitations.  Output: OptionsOwl_Strategy_Review.docx
"""
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Inches

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "OptionsOwl_Strategy_Review.docx"
ML_JSON = ROOT / "journal" / "v3_eval_results" / "gold_standard_raw.json"
FS_JSON = ROOT / "journal" / "v3_eval_results" / "full_stack_combined.json"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x55, 0x55)
RED = RGBColor(0xB0, 0x00, 0x00)
GREEN = RGBColor(0x00, 0x70, 0x30)

d = Document()
# base styles
st = d.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)


def h1(t):
    p = d.add_heading(t, level=1)
    for r in p.runs:
        r.font.color.rgb = NAVY
    return p


def h2(t):
    p = d.add_heading(t, level=2)
    for r in p.runs:
        r.font.color.rgb = NAVY
    return p


def para(t, italic=False, bold=False, color=None, size=None):
    p = d.add_paragraph()
    r = p.add_run(t)
    r.italic = italic
    r.bold = bold
    if color:
        r.font.color.rgb = color
    if size:
        r.font.size = Pt(size)
    return p


def bullet(t, bold_lead=None):
    p = d.add_paragraph(style="List Bullet")
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
        p.add_run(t)
    else:
        p.add_run(t)
    return p


def table(headers, rows, widths=None):
    t = d.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, hh in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(hh)
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        c.paragraphs[0].runs
    # shade header
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    for c in t.rows[0].cells:
        tcPr = c._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "1F3A5F")
        tcPr.append(shd)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(8.5)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    d.add_paragraph()
    return t


# ══════════════════════════════════════════════════════════════════════
# COVER
# ══════════════════════════════════════════════════════════════════════
tp = d.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run("OptionsOwl")
r.bold = True
r.font.size = Pt(30)
r.font.color.rgb = NAVY
sp = d.add_paragraph()
sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sp.add_run("0DTE Options Trading System — Strategy & Decision Logic Review")
r.font.size = Pt(14)
r.font.color.rgb = GREY
sp2 = d.add_paragraph()
sp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sp2.add_run("Entry pipeline · position sizing · exit engine · risk controls · worked examples · backtest evidence")
r.italic = True
r.font.size = Pt(10)
r.font.color.rgb = GREY
dt = d.add_paragraph()
dt.alignment = WD_ALIGN_PARAGRAPH.CENTER
dt.add_run("Prepared July 2026 · Confidential").font.size = Pt(9)
d.add_paragraph()

para("How to read this document.", bold=True)
para("This is a complete, honest description of every rule the system uses to open, size, and close "
     "a position, plus the data it uses to decide. It is written for an experienced markets reviewer. "
     "Where a rule has been tested and rejected, or where a backtest number is optimistic, that is stated "
     "plainly rather than hidden. Section 10 walks real historical trades through the gates end-to-end, and "
     "Section 11 gives the most recent out-of-sample backtest with its caveats.", italic=True, color=GREY)

d.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY / EDGE THESIS
# ══════════════════════════════════════════════════════════════════════
h1("1. Executive Summary & Edge Thesis")
para("OptionsOwl trades very-short-dated (mostly 0-DTE, some 1–4 DTE) equity and index options. It is not "
     "a directional forecaster; its edge is structural and comes from three places:")
bullet(" the system rides asymmetric convex payoffs (options) and cuts losers hard while letting winners "
       "trail with no profit ceiling. The exit engine — not entry selection — is the primary driver of P&L.",
       bold_lead="Exit asymmetry —")
bullet(" it acts on high-conviction institutional order flow (Unusual Whales ask-side sweeps ≥ $250k) within "
       "seconds, before the option fully reprices. Historically this flow source is ~4× the edge of the ML book.",
       bold_lead="Order-flow front-running —")
bullet(" three independent signal sources (Discord analyst signals, a LightGBM pattern model, and whale flow) "
       "feed one common risk pipeline, so no single source dominates and each is gated separately.",
       bold_lead="Signal diversification —")
para("")
para("Honest framing of where the edge is NOT.", bold=True, color=RED)
para("The machine-learning pattern model (the 'ML book') is a weak-to-moderate standalone edge and goes through "
     "losing regimes — notably June 2026, a chop/‘call-bleed’ tape where it lost money. The account stays "
     "profitable in those stretches because the flow book carries it. This is documented, expected, and is the "
     "single most important thing to understand about the system: profitability is flow-concentrated.", color=GREY)

# ══════════════════════════════════════════════════════════════════════
# 2. ARCHITECTURE & DATA
# ══════════════════════════════════════════════════════════════════════
h1("2. System Architecture & Data Sources")
para("A fleet of identical bots trades the same code and strategy; only account size and paper/live flags "
     "differ. A single 'harvester' process holds the market-data and flow WebSocket connections and publishes "
     "to Redis/Postgres, so the trading bots never contend for the upstream feeds.")
h2("2.1 Data sources")
table(["Source", "What it provides", "Latency / use"],
      [["Polygon.io (WS + REST)", "Real-time option quotes, greeks, underlying trades/candles", "Live pricing, freshness self-test"],
       ["Unusual Whales (WS + REST)", "Ask-side option sweep 'flow alerts' (≥$250k)", "Live flow signals + historical backtest"],
       ["Discord (Neverland Pirates)", "Human analyst trade calls", "Parsed + scored signal source"],
       ["ThetaData", "2.5 yr historical option OHLC + greeks", "Backtesting / model training"],
       ["Harvester Postgres/Redis", "Shared option snapshots, 1-min candles, GEX", "All serving features (no per-bot fetch)"]])
h2("2.2 The decision inputs (everything the system 'looks at')")
para("Every entry and exit decision can draw on the following. Missing inputs fail safe (the dependent rule "
     "no-ops rather than guessing).")
table(["Category", "Specific inputs used"],
      [["Price / trend", "1/5/15/30/60-min candles, VWAP, RSI, support levels (candle lows), day-open % change"],
       ["Option greeks", "delta, gamma, theta, IV; bid/ask spread; open interest; volume"],
       ["Cross-chain (puts)", "IV skew (put IV / call IV), put/call volume ratio"],
       ["Market regime", "SPY direction gate, directional regime model (candle-based), VIX regime, GEX (gamma exposure)"],
       ["Flow context", "sweep cluster count (30-min window), total premium, ask-side fraction, single-stock vs index"],
       ["ML confidence", "pattern-model probability, entry-timing probability, P(runner) score"],
       ["Account state", "balance, open positions, per-direction slots, daily/weekly P&L, correlation groups"]])

# ══════════════════════════════════════════════════════════════════════
# 3. SIGNAL SOURCES
# ══════════════════════════════════════════════════════════════════════
h1("3. Signal Sources (three independent origins)")
h2("3.1 ML pattern model (CALLs and PUTs)")
para("A LightGBM classifier scores each candidate contract intraday. CALLs scan 5–90 min after the open; "
     "PUTs scan all day (5–360 min) because put opportunities are decline-driven and can happen anytime. "
     "PUTs use a dedicated model (27 features incl. cross-chain IV skew, AUC ≈ 0.80–0.82, walk-forward "
     "validated) — not the call model. A candidate must clear the pattern-probability threshold (0.62) to "
     "proceed to the pipeline.")
h2("3.2 Unusual Whales flow (the primary edge)")
para("The harvester subscribes to the UW flow-alerts WebSocket and filters to whale ask-side option SWEEPS: "
     "≥ 60% traded at the ask, has_sweep = true, ≥ $250k premium, on a validated per-side ticker whitelist. "
     "A qualifying sweep emits a trade signal that is executed within seconds. Flow signals bypass the "
     "direction/regime entry gates (they carry their own validated conviction) but still face all risk gates "
     "(spread, delta, premium, EOD, position caps).")
h2("3.3 Discord analyst signals")
para("Human trade calls are parsed and scored, premium-verified against live quotes, and routed through the "
     "identical risk pipeline. (Discord has been intermittently disabled operationally; the flow + ML sources "
     "are the current live drivers.)")

# ══════════════════════════════════════════════════════════════════════
# 4. ENTRY PIPELINE (the gates, from code)
# ══════════════════════════════════════════════════════════════════════
h1("4. Entry Pipeline — The Gates (in run order)")
para("A candidate signal must pass every applicable gate below, evaluated in this exact order (source: "
     "DEFAULT_ENTRY_GATES in risk/pipeline.py). The first failure rejects the trade and is logged with a "
     "reason. Flow-sourced signals skip the direction/regime gates (0b–0d, 4, 4b) as noted.")
table(["#", "Gate", "What it enforces"],
      [["0", "BlockedTicker", "Historically unprofitable tickers are refused outright"],
       ["0a", "PutTickerExclusion", "No PUTs on net-loser names (PLTR, AMD, MSTR, AVGO)"],
       ["0c", "PutMarketDirection", "PUTs only when SPY is green / bear-mode confirmed"],
       ["0d", "PutBearishConfirm", "PUTs require VWAP breakdown + bearish candles + RSI < 45"],
       ["0b", "DirectionalRegime", "Trade direction must match the candle-based market regime"],
       ["1", "Score", "Signal score ≥ floor (78)"],
       ["2", "Premium", "A valid, live-verified option premium exists"],
       ["2b", "PremiumCap", "Reject over-priced premium (tiered cap; non-index)"],
       ["2c", "SpreadCost", "Reject wide bid-ask spreads (> ~40%)"],
       ["2d", "OTMDistance", "Reject strikes too far OTM (> 0.5% from spot)"],
       ["2e", "DeltaEntry", "Reject far-OTM / deep-ITM by delta"],
       ["3", "StopPrice", "A stop price must be assigned (flow gets a 0.5% underlying stop)"],
       ["4", "AntiChase", "Reject if the underlying already moved > 0.3% (don't chase)"],
       ["4b", "MomentumConfirm", "Reject if the underlying is fading against the trade"],
       ["5", "TimeOfDay", "Time-of-day score thresholds / cutoffs (EOD 3:55 hard stop)"],
       ["6", "ConsecutiveLoser", "Pause after a streak of losers"],
       ["7", "DailyLoss", "Daily loss limit"],
       ["8", "ConcurrentPositions", "Max simultaneous positions (5)"],
       ["8b", "DirectionSlot", "Per-direction slot cap (regime-aware)"],
       ["9", "DuplicateTicker", "No second OPEN position in the same ticker"],
       ["10", "CorrelationCap", "≤ 3 same-direction positions per correlated group"],
       ["11", "CircuitBreaker / PortfolioRisk", "Time buffers, streak/drawdown halts, portfolio risk cap (75%)"],
       ["12", "PerTradeRisk", "Per-trade risk cap (max loss %)"],
       ["13", "Liquidity", "Open-interest / volume / spread minimums"],
       ["14", "WeeklyLoss", "Weekly loss limit"],
       ["15–16", "IVFilter / VIXRegime", "IV rank / percentile and VIX regime sanity"],
       ["17", "AnalystFilter", "Bot/source performance filter"],
       ["18", "Balance", "Sufficient buying power"]])
para("Anti-chase and momentum-confirm are currently turned OFF in production (validated: they cost more "
     "entries than they save — see Section 12). They remain in the pipeline, gated by flags.", italic=True, color=GREY)

# ══════════════════════════════════════════════════════════════════════
# 5. POSITION SIZING
# ══════════════════════════════════════════════════════════════════════
h1("5. Position Sizing — The Multiplier Stack")
para("Sizing is deliberately NOT a function of the signal score (backtests showed score does not predict "
     "outcome). It starts from a flat per-slot budget and applies validated multipliers, then hard caps:")
para("base budget  =  balance × 75% (deployable) ÷ 5 (max concurrent) × 0.85 (flat)", bold=True)
table(["Layer", "Effect", "Bounds / notes"],
      [["Flat base", "Equal allocation for every qualifying trade", "score ≥ 78 only"],
       ["Flow conviction", "Bigger on clustered / high-premium single-stock sweeps; SMALLER on index $1M+ (hedges)", "clamp 0.25–2.5"],
       ["runner_v1 P(runner)", "CALLs sized up by modeled probability of being a 'runner'", "×0.7 / 0.9 / 1.1 / 1.3 by quartile"],
       ["conf_linear", "Scale by ML confidence (concentrate on high-confidence)", "0.4×–1.8× (validated bounds)"],
       ["delta haircut", "CALLs only: cheap-OTM blow-up brake", "× min(1, |delta|/0.45)"],
       ["Hard caps", "Position cannot exceed caps regardless of multipliers", "≤ 15% of portfolio AND ≤ $50,000"]])
para("The conf_linear and runner_v1 layers STACK (combined ≈ ×0.28–×2.34); the 15% / $50k caps bound the "
     "upside. Both were validated over 6 months (conf_linear: +107% P&L at LOWER drawdown vs flat). Note the "
     "honest caveat in Section 11: over the most recent single month, conf_linear was a drag.", color=GREY)

# ══════════════════════════════════════════════════════════════════════
# 6. EXIT ENGINE
# ══════════════════════════════════════════════════════════════════════
h1("6. Exit Engine — The V5/V7 Category-Aware FSM")
para("The exit engine is where the edge lives. A finite-state machine runs every 5 seconds per open position. "
     "Gates are evaluated in priority order; the FIRST to trigger closes the position. States (GRACE / "
     "DEVELOPING / TRAILING) are informational.")
h2("6.1 Gate priority (first match exits)")
table(["#", "Gate", "What it does", "Key thresholds"],
      [["1", "EOD cutoff", "0DTE: force-close before the bell", "3:45–3:55 PM ET"],
       ["2", "Bid disappearance", "No buyers for 30s → get out", "30s zero bid"],
       ["—", "5-min grace", "Skip gates below early in the trade (backstop still fires)", "5 min (TSLA/QQQ 8)"],
       ["2.55", "Multi-day hard stop", "−25% premium hard cut extended to multi-day CALL legs", "−25%"],
       ["3", "Profit target", "Index 0DTE: lock gains", "SPY/QQQ/IWM +30%"],
       ["3.5", "Breakeven ratchet", "Once +20%, stop floor = entry (cannot lose after)", "+20%"],
       ["3.6", "Profit-lock", "Once +25% peak, exit if gain < 80% of peak gain", "arm +25%, keep 80%"],
       ["3.7", "Scaleout", "Sell 1/3 at +20% (one-shot)", "+20%"],
       ["4", "Scalp trail", "Peaked then faded < 60% of peak", "DTE-aware"],
       ["5", "Checkpoint cut", "0DTE: down 30% AND underlying against 0.5%", "0DTE only"],
       ["6", "Graduated stop", "Tight if underlying against; wide backstop otherwise", "0DTE 35/65, multi 52/75"],
       ["7", "Soft trail", "15–50% band: keep 60–70% of the gain", "floor = entry + 60-70%×(peak−entry)"],
       ["8", "Adaptive trail", "PRIMARY winner exit — category-aware trailing stop", "see tiers below"],
       ["9", "Theta exit", "Cut stale losers", "0DTE 120m+−30%, multi 180m+−15%"]])
h2("6.2 Grace-period backstop (catastrophe guard)")
para("The 5-minute grace period does NOT protect catastrophic losses — a backstop fires DURING grace at −65% "
     "(0DTE) / −75% (multi-day). This closed a historical bug where a −95% trade sat untouched for 5 minutes.")
h2("6.3 Adaptive trail tiers (gate 8) — the no-ceiling winner engine")
para("Tickers are classed HIGH_VOL / INDEX / STANDARD; each gets different trail widths. There is no profit "
     "ceiling — a runner keeps running; the trail simply widens as the gain grows.")
table(["Category", "Tickers", "Active (40%+)", "Runner (150%+)", "Moonshot (400%+)"],
      [["HIGH_VOL", "MSTR AMD TSLA NVDA AVGO META COIN SMCI PLTR", "50% drop", "55% drop", "35% drop"],
       ["INDEX", "SPY QQQ IWM DIA XLF XLK", "35% drop", "40% drop", "25% drop"],
       ["STANDARD", "everything else", "35% drop", "40% drop", "25% drop"]])
h2("6.4 DTE awareness")
table(["Parameter", "0DTE", "Multi-day"],
      [["Tight stop (underlying against)", "35%", "52%"],
       ["Backstop (underlying neutral)", "65%", "75%"],
       ["Checkpoint cut", "active", "disabled"],
       ["Theta exit", "120 min + down 30%", "180 min + down 15%"]])

# ══════════════════════════════════════════════════════════════════════
# 7. PUT PIPELINE
# ══════════════════════════════════════════════════════════════════════
h1("7. PUT Trading — Separate Model, No-Ceiling Trail")
para("PUTs run a completely separate pipeline. Entry uses the dedicated put model with cross-chain features; "
     "entry is further gated by market-direction (SPY) and a bearish-confirm gate (VWAP/candles/RSI). Exits "
     "remove the profit ceiling entirely — puts ride panic drops the way calls ride momentum. Half-size "
     "budget (0.50×) reflects structurally worse odds; net-loser tickers are excluded. Backtests: removing the "
     "profit ceiling lifted 60-day put P&L from ~+$20k to ~+$63k (PF 1.57).")

# ══════════════════════════════════════════════════════════════════════
# 8. RISK CONTROLS
# ══════════════════════════════════════════════════════════════════════
h1("8. Risk Controls & Safety Systems")
table(["Control", "Behavior"],
      [["Hard premium stop", "−25% underlying-independent cut (0DTE + multi-day legs, calls & puts)"],
       ["Breakeven ratchet", "After +20%, the position cannot close below entry"],
       ["EOD cutoff", "0DTE positions force-closed before the bell; no overnight 0DTE"],
       ["Late-entry cutoff", "No new entries after 14:30 ET"],
       ["Stall cut", "Cut dead multi-day legs (>30 min, −30%, peak < 10%)"],
       ["FOMC pause", "Blocks ALL new entries on Fed-announcement days"],
       ["Data-freshness guard", "Blocks entries + alerts if the market feed goes stale during hours"],
       ["Market-holiday calendar", "Single source of truth for 'is the market open' (prevents closed-market crash-loops)"],
       ["Position caps", "≤ 15% of portfolio and ≤ $50k per trade; ≤ 5 concurrent; 75% max deployable"],
       ["Kill switch", "WEBULL_KILL_SWITCH halts all live order placement instantly"],
       ["Event-loop timeouts", "Every external I/O in the monitor loop has a 15s hard timeout (a hung call can never freeze sells)"]])

# ══════════════════════════════════════════════════════════════════════
# 9. LIFECYCLE
# ══════════════════════════════════════════════════════════════════════
h1("9. Trade Lifecycle (entry → fill → exit)")
for i, s in enumerate([
    "Signal arrives (Discord parse, ML scan, or a UW flow sweep).",
    "Smart entry verifies a live premium and resolves the nearest tradeable expiry (nearest-DTE, not the "
    "whale's far-dated contract — multi-day flow was tested and is a decisive loser).",
    "The 18-gate entry pipeline runs; first failure rejects and logs the reason.",
    "Position sizing applies the multiplier stack and hard caps.",
    "The order is placed via a fill-chasing limit ladder (fresh live-quote pricing, 4 rungs, index decisive cross).",
    "The position monitor polls every 5s and runs the exit FSM; the first triggered gate closes the trade.",
    "Every decision is written to a persisted audit table (trade_events) and structured logs.",
]):
    d.add_paragraph(f"{i+1}. {s}", style="List Number")

# ══════════════════════════════════════════════════════════════════════
# 10. WORKED EXAMPLES (real data)
# ══════════════════════════════════════════════════════════════════════
h1("10. Worked Example Trades (real historical trades through the gates)")
para("These are actual trades from the June 2026 backtest trade log, traced through the decision logic. "
     "Entry premium, contracts, hold time, exit reason, peak gain and model confidence are as recorded.", color=GREY)

r = json.loads(ML_JSON.read_text())
tl = r["trade_log"]


def find(day, tk, direction):
    for t in tl:
        if t["day"] == day and t["ticker"] == tk and t["direction"] == direction:
            return t
    return None


def example(title, t, narrative):
    h2(title)
    if t:
        table(["Field", "Value"],
              [["Date / ticker / side", f"{t['day']} · {t['ticker']} · {t['direction'].upper()} · {t.get('dte',0)}DTE"],
               ["Entry premium", f"${t['entry']:.2f}  (effective ${t.get('effective_entry',t['entry']):.2f}"
                                 f"{', after DCA' if t.get('dca') else ''})"],
               ["Contracts", t["effective_contracts"]],
               ["Model confidence", f"{t.get('pattern_conf','—')}"],
               ["Peak gain", f"+{t.get('peak_gain',0):.1f}%"],
               ["Hold time", f"{t['hold_min']} min"],
               ["Exit reason", t["reason"]],
               ["Realized P&L", f"${t['pnl']:+,.2f}"]])
    for n in narrative:
        bullet(n)
    d.add_paragraph()


example("10.1 A clean winner — profit-lock captures the gain",
        find("2026-06-16", "GOOGL", "call"),
        ["Entry: a GOOGL 1-DTE call at $2.51, model confidence 0.817 (high) → conf_linear sizes it up.",
         "The trade ran to +45.7% peak, then faded. The profit-lock gate (arm +25%, keep 80% of peak) triggered "
         "the exit before it round-tripped — banking +$592 instead of giving it back.",
         "This is the exit engine doing its job: no fixed target, ride the move, lock it when it fades."])

example("10.2 A loser cut by the hard stop — and why the DCA hurt",
        find("2026-06-29", "IWM", "call"),
        ["Entry: an IWM 0-DTE call, high confidence (0.819). The premium dipped, DCA doubled the position at a "
         "lower effective price ($0.87 → $0.78 effective, 40 contracts).",
         "The underlying kept going against it; the −25% hard stop fired at 12 min for −$1,395 — the largest loss "
         "of the window.",
         "HONEST READ: the DCA add amplified a losing 0DTE trade. This is exactly the risk of averaging down on "
         "same-day options; it is one reason the DCA layer is watched closely and sized conservatively."])

example("10.3 A put that bled to the close",
        find("2026-06-04", "TSLA", "put"),
        ["Entry: a TSLA 1-DTE put, cleared the bearish-confirm + SPY-direction gates.",
         "It never worked (peak only +9.6%) and slowly bled; with no catastrophic stop hit, it held to the EOD "
         "data end for −$128 on a single contract (half-size put budget kept the loss small).",
         "PUTs are structurally harder; the half-size budget and no-ceiling trail are designed for exactly this "
         "asymmetry — small controlled losses, occasional large panic-drop winners."])

# ══════════════════════════════════════════════════════════════════════
# 11. BACKTEST EVIDENCE
# ══════════════════════════════════════════════════════════════════════
h1("11. Backtest Evidence — Most Recent 21 Trading Days")
para("Out-of-sample, on freshly downloaded data (2026-06-03 → 07-01). The flow book is fetched live from the "
     "UW flow-alerts API and priced from ThetaData; the ML book is the production gold-standard harness with "
     "production conf_linear sizing. The headline is the flat $750/trade EDGE measure — additive across books "
     "and free of the compounding/liquidity distortions.")
fs = json.loads(FS_JSON.read_text()) if FS_JSON.exists() else {}
f7 = fs.get("flat750", {})
if f7:
    table(["Book", "Trades", "Win rate", "Edge P&L", "PF"],
          [["ML (calls+puts)", f7["ml"]["trades"], f"{f7['ml']['win_rate']:.1f}%", f"${f7['ml']['total_pnl']:+,.0f}", f"{f7['ml']['pf']:.2f}"],
           ["FLOW (whale sweeps)", f7["flow"]["trades"], f"{f7['flow']['win_rate']:.1f}%", f"${f7['flow']['total_pnl']:+,.0f}", f"{f7['flow']['pf']:.2f}"],
           ["COMBINED", f7["combined"]["trades"], f"{f7['combined']['win_rate']:.1f}%", f"${f7['combined']['total_pnl']:+,.0f}", f"{f7['combined']['pf']:.2f}"]])
para("Reading the result honestly:", bold=True)
bullet(" the full stack was positive over the window (combined edge PF ≈ 1.19), but ~94% of the edge came from "
       "FLOW, not ML. This is the flow-concentration thesis, confirmed out of sample.", bold_lead="Edge is flow-driven —")
bullet(" the same ML trades are +$738 on the flat-$750 basis but −$2,501 (−10.9%, PF 0.75) on the authoritative "
       "compounded harness with production conf_linear sizing. conf_linear bet bigger on trades that lost in the "
       "June chop. Over 6 months conf_linear is a validated winner (+107%); this is one adverse month, but it is "
       "real and shown, not hidden.", bold_lead="ML sizing was a drag this month —")
bullet(" the 565 flow 'trades' are all deduped signals, not what the live bot captures (it is capped at 5 "
       "concurrent + gated). The flat-$750 figure is edge-per-signal, not an achievable account return. A "
       "naive compounded shared-account sim shows +165%, but that is liquidity-blind and should NOT be trusted.",
       bold_lead="Capacity caveat —")

# ══════════════════════════════════════════════════════════════════════
# 12. WHAT WE TESTED AND REJECTED
# ══════════════════════════════════════════════════════════════════════
h1("12. What We Tested and Rejected (intellectual honesty)")
para("A large part of the research record is negative results. The following were tested rigorously and "
     "rejected — they are NOT in the live system:")
table(["Idea", "Verdict"],
      [["Entry timing (wait for a dip / for the underlying to 'turn')", "REJECTED — winners don't dip; waiting chases them (adverse selection). Buy on the signal."],
       ["Regime call/put budget amplification", "REJECTED — 6-month loser (put ×2.0 backfired)"],
       ["Sideways-scalp (take small profits in chop)", "REJECTED — clips would-be runners, fights the wide trail"],
       ["Multi-day flow (trade the whale's far-dated expiry)", "REJECTED — decisive loser (PF 0.69 vs nearest-DTE 1.39)"],
       ["P(runner) sizing for PUTs", "REJECTED — puts have runners but not predictable at entry (OOS AUC 0.66)"],
       ["Consecutive-loss circuit breaker (auto-halt)", "REJECTED — 60% false-halt rate, clips winners"],
       ["Anti-chase / momentum entry gates", "OFF — cost more entries than they saved"],
       ["Tweet-driven options (Elon/Trump)", "REJECTED — premium reprices in 1–2 min; our lag makes it breakeven"]])

# ══════════════════════════════════════════════════════════════════════
# 13. LIMITATIONS
# ══════════════════════════════════════════════════════════════════════
h1("13. Known Limitations")
bullet(" all edge measurement uses flat per-trade sizing; real fills, slippage on thin names, and 5-concurrent "
       "capacity limits mean live capture is a fraction of the raw signal edge.", bold_lead="Liquidity / capacity —")
bullet(" the pattern model goes through losing regimes (June 2026). The system relies on flow to carry those "
       "stretches. A prolonged flow drought would expose the ML weakness.", bold_lead="Regime dependence —")
bullet(" backtests close 0DTE at EOD and cannot perfectly model intra-second fills, DCA blends, or the exact "
       "Webull fill ladder. Numbers are directionally trustworthy, not penny-exact.", bold_lead="Backtest fidelity —")
bullet(" the flow whitelist and excluded-ticker lists are partly in-sample; they are re-validated periodically "
       "but carry overfitting risk.", bold_lead="Whitelist selection —")
para("")
para("Prepared from the live production codebase and the July 2026 out-of-sample backtest. All figures are "
     "reproducible from the committed harnesses (backtest_gold_standard.py, backtest_full_stack.py).",
     italic=True, color=GREY, size=9)

d.save(str(OUT))
print(f"Saved -> {OUT}")
print(f"Size: {OUT.stat().st_size/1024:.0f} KB")
