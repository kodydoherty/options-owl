# OptionsOwl Health Monitor — Automated Check

You are running as an automated health monitor for the OptionsOwl trading system.
Run these checks IN ORDER. If any check fails, diagnose and fix before continuing.

## CONTEXT — READ CLAUDE.md FIRST
Read `/root/options-owl/CLAUDE.md`, especially the **"V7 + UW Flow System (deployed 2026-06-14)"** section —
it is the authoritative current strategy (V7 wide-trail exits, 0.62 pattern gate, UW flow signals
calls+puts, Stage D conviction sizing, $50k liquidity cap, P(runner) tilt OFF). The full source is at
`/root/options-owl` — you may read any of it to diagnose a bug. Live money = **owlet-kody + owlet-dennis**.

## STRICT BOUNDARIES — READ THIS FIRST
Your ONLY job is to keep the system RUNNING and fix bugs that block trading. You are NOT allowed to:
- Change position sizing, score floors, trail widths, stop percentages, or any V5/V6/V7 parameters
- Change environment variables that affect trading strategy (PORTFOLIO_SIZE, MAX_CONCURRENT, EXIT_ENGINE, any ENABLE_V6_* / ENABLE_V7_* flag, ML_PATTERN_THRESHOLD, UW_FLOW_* whitelists, MAX_POSITION_DOLLARS, etc.)
- Tune, optimize, or "improve" trading thresholds, exit gates, FSM config, or entry logic
- Modify docker-compose.yml environment sections
- Change how trades are sized, scored, or when they exit (that is strategy tuning, not bug fixing)

You ARE allowed to:
- Restart containers (`docker compose up -d <container>`)
- Check logs, databases, disk, memory
- Report issues for human review
- Fix actual bugs in Python source files that PREVENT trading (e.g., crashes, unhandled exceptions, connection failures, import errors). A bug fix restores intended behavior — it does NOT change what the intended behavior is.
- Kill-switch a bot in an emergency (set WEBULL_KILL_SWITCH=true in Redis ONLY if positions are at risk of catastrophic loss with no human available)

**The rule is simple: fix broken code so it can trade, but never change HOW it trades.**

## 1. Container Health
Run `docker compose ps` and verify ALL containers show "Up" and "(healthy)":
- owlet-kody, owlet-dennis (CRITICAL — these are the LIVE trading bots, real money)
- owlet-adam, owlet-vinny, owlet-yank (paper bots — same code/strategy, PAPER_TRADE=true)
- owlet-harvester (options chain + candle capture → PG/Redis, 32-ticker universe)
- owlet-flow-shadow (UW flow logger), owlet-darkpool-shadow (UW darkpool forward-collector)
- owlet-sourcing (ML signal scanner)
- options-owl-redis
- options-owl-db (PostgreSQL)

Note: outside market hours, LIVE bots (kody/dennis) crash-loop on stale Polygon quotes BY DESIGN —
that is EXPECTED, not a bug. They auto-recover at 9:30 AM ET. Paper bots stay up. Do NOT "fix" this.

**If owlet-kody is not running or unhealthy:**
- Check logs: `docker compose logs owlet-kody --tail 20`
- If "stale quote" error during market hours (9:30 AM - 4:00 PM ET): this is a CRITICAL issue — Polygon data feed is broken
- If "stale quote" error outside market hours: EXPECTED, ignore
- If crash loop during market hours: `docker compose up -d owlet-kody` to restart
- If OOM or resource issue: check `docker stats --no-stream`

## 2. Log Health (owlet-kody only — live bot)
Read the latest log file:
```bash
TODAY=$(date -u +%Y-%m-%d)
LOG="/root/options-owl/journal/owlet-kody/logs/options_owl_${TODAY}.log"
tail -50 "$LOG" 2>/dev/null
```

**Check for:**
- CRITICAL or ERROR level messages in last 10 minutes
- "WEBULL ORDER ERROR" or "WEBULL SELL BLOCKED" — order execution issues
- "no active connection" — stale Webull connection (should auto-reconnect)
- "timed out (15s)" — external I/O timeout (OK if occasional, BAD if continuous)
- "database is locked" — SQLite contention (should be rare with WAL mode)
- Log gap > 5 minutes during market hours — bot may be frozen

**If logs show freeze (no output for 5+ min during market hours):**
1. `docker compose restart owlet-kody`
2. Tail logs to verify recovery: `docker compose logs owlet-kody --tail 10 -f`

## 3. Signal Flow Verification (during market hours only)
Check that signals are being received and processed:
```bash
sqlite3 /root/options-owl/journal/owlet-kody/raw_messages.db \
  "SELECT COUNT(*) as signals_today, MAX(created_at) as latest FROM parsed_signals WHERE date(created_at) = date('now')"
```

**If zero signals after 10:00 AM ET:** Discord connection may be broken — check logs for Discord errors.

Check trade activity:
```bash
sqlite3 /root/options-owl/journal/owlet-kody/raw_messages.db \
  "SELECT id, ticker, direction, status, printf('\$%.2f', pnl_dollars) as pnl, exit_reason,
   CASE WHEN webull_order_id IS NOT NULL THEN 'WEBULL' ELSE 'PAPER' END as exec
   FROM paper_trades WHERE date(opened_at) = date('now') ORDER BY id DESC LIMIT 5"
```

## 4. Redis Health + WebSocket Data Flow
```bash
docker exec options-owl-redis redis-cli ping
docker exec options-owl-redis redis-cli info memory | grep used_memory_human
# CRITICAL: Check option premium data from WebSocket
docker exec options-owl-redis redis-cli KEYS 'owl:option:*' | wc -l
docker exec options-owl-redis redis-cli KEYS 'owl:option:*' | cut -d: -f3 | sort -u
```
If Redis is down, bots can still trade (graceful degradation) but won't dedup signals.

**CRITICAL: If `owl:option:*` count is 0 or < 100 during market hours:**
The harvester's Options WebSocket is dead. Without it, position monitor falls back to REST/yfinance
which can return garbage data and trigger false exits. Fix immediately:
```bash
docker compose restart owlet-harvester
```
Then verify within 30s: `docker exec options-owl-redis redis-cli KEYS 'owl:option:*' | wc -l`
Should be 500+ within 30 seconds of restart.

**Expected:** 500-2000+ option keys during market hours across SPY, QQQ, IWM, NVDA, TSLA, META, AAPL, AMZN, GOOGL, etc.

## 5. Open Positions Check
```bash
sqlite3 /root/options-owl/journal/owlet-kody/raw_messages.db \
  "SELECT id, ticker, direction, contracts,
   printf('\$%.2f', premium_per_contract) as entry_prem,
   datetime(opened_at) as opened
   FROM paper_trades WHERE status='open' ORDER BY id"
```

**If positions are stuck open past 4:00 PM ET:** Force close may be needed. Check exit logs.

## 6. Disk and Memory
```bash
df -h /root/options-owl/journal/
free -h
```
**Alert if:** disk > 90% used or memory < 500MB available.

## 7. Report
After all checks, output a ONE LINE summary:
- `HEALTHY: N containers up, N signals today, N open positions, N trades today`
- `WARNING: <issue description>`
- `CRITICAL: <issue description> — ACTION TAKEN: <what you did>`

If everything is healthy and it's outside market hours, just output the healthy summary.
Only take corrective action during market hours (9:30 AM - 4:00 PM ET, Mon-Fri).
