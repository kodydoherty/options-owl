# Spec 05: Agent Dashboard UI

**Priority**: High
**Effort**: Medium (2-3 days)
**Impact**: Each user can monitor, control, and debug their trading agent in real-time

## Problem

Currently, monitoring and managing trading agents requires SSH access to the droplet and running CLI commands (`docker compose logs`, `sqlite3` queries, `trade-pnl.py`). Only Kody has this access. Adam, Vinny, and Yank have zero visibility into what their agents are doing, whether trades executed, or if errors are occurring.

## Requirements

1. **Per-user authentication** — 4 users (kody, adam, vinny, yank), bcrypt-hashed passwords, JWT sessions
2. **Live trade dashboard** — open trades with real-time P&L, entry logic, exit gate status
3. **Trade history** — closed trades with full lifecycle (entry, DCA, scaleout, exit reason, P&L)
4. **Portfolio overview** — balance, daily P&L, win rate, open position count
5. **Agent controls** — toggle paper mode, restart agent, kill switch
6. **Error log viewer** — recent errors/warnings from persisted log files
7. **Live updates** — WebSocket push so the page updates without refresh

## Architecture

### Why FastAPI + Plain HTML (not Next.js/React)

- Runs on the same droplet as the agents — no separate hosting
- Python backend reads PG directly (same pool pattern as existing code)
- No build step, no npm, no node — just a Python process serving HTML + JS
- WebSocket support built into FastAPI (Starlette)
- Memory budget: ~128MB (droplet has 139GB free)

### Stack

| Layer | Tech | Why |
|---|---|---|
| **Backend** | FastAPI + uvicorn | Async Python, WebSocket native, same ecosystem |
| **Frontend** | Jinja2 templates + Tailwind CDN + vanilla JS | No build step, minimal deps |
| **Auth** | bcrypt + JWT (python-jose) | Industry standard, stateless tokens |
| **Live updates** | WebSocket (FastAPI → browser) | Real-time P&L, new trades, errors |
| **Data** | PostgreSQL (existing) + Redis (existing) | Already deployed and populated |

### Container

```yaml
# docker-compose.yml addition
owlet-dashboard:
  <<: *bot-common
  container_name: owlet-dashboard
  command: ["python", "-m", "options_owl.dashboard.app"]
  ports:
    - "0.0.0.0:8443:8443"  # HTTPS (TLS terminated by uvicorn or nginx)
  depends_on:
    postgres:
      condition: service_healthy
    redis:
      condition: service_healthy
  environment:
    - PYTHONUNBUFFERED=1
    - DATABASE_URL=postgresql://owl:${POSTGRES_PASSWORD:-owl_dev_2026}@postgres:5432/options_owl
    - REDIS_URL=redis://redis:6379/0
    - DASHBOARD_SECRET_KEY=${DASHBOARD_SECRET_KEY}  # JWT signing key
    - DASHBOARD_PORT=8443
  volumes:
    - ./journal:/app/journal:ro   # read-only access to all agent logs
  deploy:
    resources:
      limits:
        memory: 256M
  healthcheck:
    test: ["CMD", "python", "-c", "import httpx; r=httpx.get('http://localhost:8443/health'); assert r.status_code==200"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 10s
```

## Data Sources

All reads — dashboard never writes to trading tables.

| Data | Source | How |
|---|---|---|
| Open/closed trades | PG `trades` table | `SELECT * FROM trades WHERE agent_id = $1` |
| Trade events (entry/exit logic) | PG `trade_events` table | `SELECT * FROM trade_events WHERE agent_id = $1 AND trade_id = $2` |
| Portfolio state | PG `agent_state` table | `SELECT * FROM agent_state WHERE agent_id = $1` |
| Live premiums | Redis `owl:price:*`, `owl:option:*` | Real-time from harvester |
| Pending signals | PG `ml_signals` | Signals waiting to be consumed |
| Error logs | `journal/owlet-{name}/logs/options_owl_YYYY-MM-DD.log` | Grep for ERROR/WARNING |
| Agent status | Docker API or Redis heartbeat | Container up/down/healthy |

## Pages

### 1. Login (`/login`)

Simple form: username + password. On success, sets `HttpOnly` JWT cookie (24h expiry).

- bcrypt password hashes stored in a `dashboard_users` PG table
- Rate limit: 5 attempts per minute per IP
- No registration — admin seeds users manually

```sql
CREATE TABLE IF NOT EXISTS dashboard_users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,     -- kody, adam, vinny, yank
    password_hash TEXT NOT NULL,       -- bcrypt
    agent_id TEXT NOT NULL,            -- owlet_kody, owlet_adam, etc.
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 2. Dashboard Home (`/`)

Single-page view with 4 sections:

#### 2a. Portfolio Summary (top bar)

```
Portfolio: $23,000    Today P&L: +$450 (+1.9%)    Open: 3/5 slots    Win Rate: 62%
Agent: RUNNING (paper)    Last Heartbeat: 2s ago    [Restart] [Kill Switch]
```

- Pulls from `agent_state` table + Redis heartbeat
- Green/red styling based on daily P&L

#### 2b. Open Trades (main area)

Live-updating cards for each open position:

```
#245 NVDA CALL  $530 strike  3 contracts
Entry: $2.15 @ 10:32 AM    Now: $2.85 (+32.6%)    Peak: $3.10 (+44.2%)
State: TRAILING    Gate: adaptive_trail (HIGH_VOL active tier)
DCA: 1x at $1.80 (blended $1.97)
[View Events]
```

Fields per trade:
- Trade ID, ticker, direction, strike, contracts
- Entry premium, current premium (from Redis), P&L %, peak gain %
- FSM state (GRACE/DEVELOPING/TRAILING)
- Active exit gate and thresholds
- DCA history (if any)
- Time in trade

#### 2c. Recent Closed Trades (below open)

Table of last 20 closed trades:

```
| # | Ticker | Dir | Entry | Exit | P&L | Reason | Hold | Source |
|245| NVDA   | CALL| $2.15 | $3.10| +$285 (+44%)| soft_trail | 45m | ai |
|244| SPY    | PUT | $1.80 | $1.20| -$180 (-33%)| graduated_stop | 22m | ai |
```

- Sortable by date, P&L, ticker
- Click to expand full trade event timeline
- Color-coded: green for wins, red for losses

#### 2d. Error Log (collapsible bottom panel)

Last 50 ERROR/WARNING lines from today's log file:

```
14:32:15 WARNING | Polygon REST timeout for MSTR (10s)
14:31:02 ERROR   | Webull order rejected: HTTP 417 PARAM_ERR
13:45:30 WARNING | Redis publish_price failed: ConnectionError
```

- Auto-scrolls to newest
- Filter by severity (ERROR only / ERROR+WARNING)
- Grep search box

### 3. Trade Detail (`/trade/{id}`)

Full lifecycle view for one trade:

- Premium chart (from `trade_premium_ticks` — 5s resolution)
- All trade events in chronological order
- Entry pipeline gates (which passed/failed)
- Exit FSM gate evaluations
- DCA events
- Webull order IDs and fill prices

## WebSocket Protocol

Single WebSocket connection per authenticated user at `wss://host:8443/ws`.

### Server → Client Messages

```json
// Trade update (every 5s for open trades)
{
  "type": "trade_update",
  "trade_id": 245,
  "premium": 2.85,
  "pnl_pct": 32.6,
  "peak_pct": 44.2,
  "fsm_state": "TRAILING",
  "active_gate": "adaptive_trail"
}

// New trade opened
{
  "type": "trade_opened",
  "trade": { ...full trade object... }
}

// Trade closed
{
  "type": "trade_closed",
  "trade_id": 245,
  "exit_reason": "soft_trail",
  "pnl_dollars": 285.0,
  "pnl_pct": 44.2
}

// Portfolio update (every 30s)
{
  "type": "portfolio_update",
  "balance": 23450.0,
  "daily_pnl": 450.0,
  "open_count": 3,
  "win_rate": 62.1
}

// Error log line
{
  "type": "log_entry",
  "level": "ERROR",
  "timestamp": "2026-05-27T14:31:02",
  "message": "Webull order rejected: HTTP 417 PARAM_ERR"
}

// Agent status
{
  "type": "agent_status",
  "status": "running",      // running, stopped, unhealthy
  "paper_mode": true,
  "last_heartbeat": "2026-05-27T14:32:15"
}
```

### How Live Data Flows

```
Position monitor (5s loop) → writes premium to PG trade_premium_ticks
                            → publishes to Redis owl:trade_update:{agent_id}

Dashboard WS server → subscribes to Redis owl:trade_update:{agent_id}
                    → pushes to browser WebSocket

Browser JS → updates DOM (premium, P&L, FSM state)
```

For the live premium feed, the position monitor already has the data — we just need it to publish a lightweight update to Redis each cycle:

```python
# In position_monitor.py, after premium fetch (existing code):
await redis_client.publish_trade_update(agent_id, {
    "trade_id": trade.id,
    "premium": current_premium,
    "pnl_pct": pnl_pct,
    "peak_pct": peak_gain_pct,
    "fsm_state": fsm.state.value,
    "active_gate": last_gate_name,
})
```

Dashboard subscribes via Redis pub/sub — zero coupling to the trading bot process.

## Agent Controls

### Toggle Paper Mode

```
POST /api/agent/paper-mode  {enabled: true/false}
```

Writes to Redis key `owl:control:{agent_id}:paper_mode`. Trading bot checks this on next trade evaluation. Does NOT restart the container — takes effect on next signal.

### Restart Agent

```
POST /api/agent/restart
```

Calls Docker API (`docker compose restart owlet-{name}`). Dashboard container needs Docker socket mounted:

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock:ro
```

Requires confirmation dialog in UI ("Are you sure? Open trades will continue monitoring after restart.").

### Kill Switch

```
POST /api/agent/kill-switch  {enabled: true/false}
```

Writes to Redis `owl:control:{agent_id}:kill_switch`. Trading bot checks before placing any Webull order. Instant effect — no restart needed.

## Security

| Concern | Mitigation |
|---|---|
| **Password storage** | bcrypt with work factor 12 |
| **Session tokens** | JWT with HS256, HttpOnly + Secure + SameSite=Strict cookies |
| **HTTPS** | TLS via Let's Encrypt (certbot) or self-signed for initial deploy |
| **CSRF** | SameSite=Strict cookie + origin check on mutations |
| **Rate limiting** | 5 login attempts/min/IP, 60 API calls/min/user |
| **SQL injection** | asyncpg parameterized queries (already used everywhere) |
| **XSS** | Jinja2 auto-escaping, no innerHTML in JS |
| **Docker socket** | Read-only mount, only restart/stop allowed (no exec/build) |
| **Agent isolation** | Every query filtered by `agent_id` from JWT — users cannot see other agents |
| **Network** | Port 8443 exposed, all other ports (PG, Redis) remain localhost-only |

## File Structure

```
options_owl/dashboard/
├── app.py                 # FastAPI app, routes, WebSocket handler
├── auth.py                # bcrypt verify, JWT create/decode, login route
├── db.py                  # Read-only PG queries for dashboard
├── ws.py                  # WebSocket manager, Redis pub/sub bridge
├── controls.py            # Agent restart/kill switch/paper mode
├── logs.py                # Log file reader (tail + grep)
├── templates/
│   ├── base.html          # Layout, nav, Tailwind CDN
│   ├── login.html         # Login form
│   ├── dashboard.html     # Main dashboard (trades, portfolio, logs)
│   └── trade_detail.html  # Single trade deep-dive
└── static/
    └── ws.js              # WebSocket client, DOM updates
```

## Dependencies (additions to pyproject.toml)

```
python-jose[cryptography]   # JWT
bcrypt                      # Password hashing
uvicorn[standard]           # ASGI server
fastapi                     # Web framework (includes Starlette WS)
jinja2                      # Templates
python-multipart            # Form parsing (login)
```

## Implementation Phases

### Phase 1: Core (Day 1)
- FastAPI app with auth (login, JWT, bcrypt)
- Dashboard page: portfolio summary + open trades (static, manual refresh)
- Closed trades table with pagination
- Deploy as `owlet-dashboard` container

### Phase 2: Live Updates (Day 2)
- Add Redis pub/sub publish in position_monitor.py
- WebSocket endpoint in dashboard
- Browser JS: auto-update open trade cards
- Portfolio summary auto-refresh

### Phase 3: Controls + Logs (Day 3)
- Agent controls (paper toggle, restart, kill switch)
- Error log viewer with tail + search
- Trade detail page with event timeline
- Premium chart (trade_premium_ticks data)

## Open Questions

1. **Domain/SSL**: Use IP:8443 with self-signed cert initially, or set up a domain + Let's Encrypt?
2. **Mobile**: Tailwind is responsive by default — should we optimize for phone viewing?
3. **Alerts**: Should the dashboard also send Discord DMs / push notifications, or is viewing enough?
4. **Multi-agent view for Kody**: Kody manages all 4 agents — should he have an admin view showing all agents side-by-side?
