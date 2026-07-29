"""6-month sweep of the CALL move-from-open trigger to find the ROBUST optimal (not overfit to 2wk).
Normal call book (puts off), 126 days. Reports P&L/PF/DD per trigger value."""
import argparse, sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_gold_standard as gs  # noqa: E402
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--days",type=int,default=126)
    ap.add_argument("--pattern-threshold",type=float,default=0.62); ap.add_argument("--regime-threshold",type=float,default=0.02); ap.add_argument("--offset",type=int,default=0)
    a=ap.parse_args()
    gs.ENABLE_ANTI_CHASE=True; gs.ENABLE_MOMENTUM_CONFIRM=True; gs.ENABLE_DIRECTIONAL_REGIME=True
    gs.ENABLE_CONSECUTIVE_LOSER=True; gs.ENABLE_CORRELATION_CAP=True
    gs.ENABLE_PUTS=False; gs.PUTS_ONLY=False; gs.PORTFOLIO_START=23000
    conn=sqlite3.connect(gs.THETADATA_DB)
    ad=[r[0] for r in conn.execute("SELECT DISTINCT substr(timestamp,1,10) FROM option_ohlc WHERE ticker='SPY' ORDER BY 1 DESC").fetchall()]; conn.close()
    end=ad[min(a.offset,len(ad)-1)]; start=ad[min(a.offset+a.days-1,len(ad)-1)]
    tickers=[t for t in gs.TICKERS if t not in gs.EXCLUDED_TICKERS]
    print("="*80); print(f"CALL move-from-open TRIGGER SWEEP | {start}→{end} ({a.days}d) | call book"); print("="*80)
    m=gs.load_models(use_entry_filter=False,use_regime=True)
    rows=[]
    for trig in [None,0.0,0.25,0.5,0.75,1.0]:
        gs.CALL_DIRECTION_TRIGGER=trig
        print(f"\n{'#'*50}\n### CALL_DIRECTION_TRIGGER = {trig}\n{'#'*50}",flush=True)
        r=gs.run_backtest(m[0],m[1],m[2],m[3],a.pattern_threshold,0.80,tickers,start,end,m[4],m[5],a.regime_threshold,m[6],m[7],m[8],m[9],m[10],m[11],"none")
        rows.append((trig,r))
    print("\n\n"+"="*80); print(f"RESULTS — CALL trigger sweep {a.days}d"); print("="*80)
    print(f"{'trigger':<12}{'trades':>7}{'WR%':>6}{'P&L':>10}{'PF':>7}{'DD%':>7}")
    base=None
    for trig,r in rows:
        if base is None: base=r["total_pnl"]
        d=f"  ({r['total_pnl']-base:+.0f})" if trig is not None else ""
        print(f"{str(trig):<12}{r['trades']:>7}{r['win_rate']:>6.0f}{r['total_pnl']:>+10.0f}{r['profit_factor']:>7.2f}{r['max_drawdown_pct']:>7.1f}{d}")
if __name__=="__main__": main()
