# Database Evaluation: SQLite vs PostgreSQL vs MongoDB

**Status:** Decision required before Phase 3 implementation
**Context:** SQLite WAL mode has served us well but hit concurrency limits — "database is locked" errors, the 7GB file copy freeze, and `:rw` mount requirements for WAL sidecar files across 6 Docker containers.

---

## 1. Current State (SQLite)

### What We Have

| Database | Size | Writers | Readers | Issues |
|---|---|---|---|---|
| `raw_messages.db` (per-bot, ×4) | ~50MB each | discord_collector, paper_trader | position_monitor, circuit_breaker, pipeline, supabase_brain | "database is locked" under load |
| `options_data.db` (harvester) | ~7GB | owlet-harvester | 4 trading bots + owlet-sourcing (planned) | 7GB temp copy freeze (fixed with WAL), WAL sidecar requires `:rw` mounts |
| `state.db` (sourcing, planned) | <100MB | owlet-sourcing | owlet-sourcing only | Not yet built |
| `signals.db` (sourcing, planned) | <100MB | owlet-sourcing | 4 trading bots | Phase 2 — same pattern as harvester shared DB |

### SQLite Pain Points (Experienced)

1. **Single writer lock.** WAL mode helps (concurrent reads + single write), but `busy_timeout=5000` means a write can block for 5 seconds. Under load (paper_trader + discord_collector + supabase_brain all writing), we hit "database is locked" freezes.

2. **No true concurrent writes.** Even with WAL, only ONE writer at a time. If paper_trader is mid-transaction and position_monitor needs to write, it waits or fails.

3. **7GB file copy incident.** The harvester DB grew to 7GB. The original `shutil.copy2()` approach froze the event loop. Fixed with direct WAL reads, but SQLite at 7GB+ is pushing its design limits.

4. **Docker volume sharing is fragile.** WAL mode creates `-shm` and `-wal` sidecar files. All containers need `:rw` access even for reads. One misconfigured mount = silent data corruption.

5. **No network access.** Every reader must mount the file locally. Can't scale beyond a single droplet.

6. **No built-in connection pooling.** Each `aiosqlite.connect()` opens a new file handle. Under heavy load, file descriptor limits can be hit.

### SQLite Strengths (Keep in Mind)

- Zero operational overhead (no server process, no config, no auth)
- Embedded — works in Docker with zero infrastructure
- WAL mode IS fast for read-heavy workloads (our trading bots are 95% reads)
- Battle-tested in our codebase (8 production files use `_connect_db`)
- Backups are just file copies
- No network latency (local file I/O)

---

## 2. Option A: PostgreSQL (Recommended)

### Why PostgreSQL

| Feature | SQLite | PostgreSQL |
|---|---|---|
| Concurrent writers | 1 (WAL) | Unlimited (MVCC) |
| Connection pooling | None | Built-in + PgBouncer |
| Network access | File only | TCP (any container) |
| JSON support | Basic (`json_extract`) | Native JSONB (indexed, queryable) |
| Full-text search | FTS5 (manual) | Built-in `tsvector` |
| Max DB size | ~1TB (practical ~10GB) | Unlimited |
| Replication | None | Streaming replication |
| Docker overhead | 0 | ~200MB RAM, 1 container |

### Architecture with PostgreSQL

```yaml
# docker-compose.yml addition
postgres:
  image: postgres:16-alpine
  container_name: options-owl-db
  restart: always
  environment:
    POSTGRES_DB: options_owl
    POSTGRES_USER: owl
    POSTGRES_PASSWORD: ${DB_PASSWORD}
  volumes:
    - ./journal/postgres-data:/var/lib/postgresql/data
  ports:
    - "127.0.0.1:5432:5432"  # localhost only, not exposed to internet
  deploy:
    resources:
      limits:
        memory: 512M
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U owl"]
    interval: 10s
    timeout: 5s
    retries: 3
```

All bot containers connect via `DATABASE_URL=postgresql://owl:${DB_PASSWORD}@postgres:5432/options_owl`.

### Migration Path

**Phase 1 (minimal, safe):** Only owlet-sourcing uses PostgreSQL for its new tables:
- `scoring_audit` — high-write audit table (every scan × 13 tickers = 130+ rows/hour)
- `signals` — signal output for Phase 2 direct feed
- `state` — cooldowns, counters, circuit breaker

Trading bots keep their existing SQLite databases untouched. Zero risk to live trading.

**Phase 2 (after sourcing proven):** Migrate harvester `options_data.db` to PostgreSQL:
- Fixes the 7GB single-file problem
- Enables indexed queries on candle data (currently full-table scans)
- All bots connect to the same Postgres instance — no shared file mounts

**Phase 3 (optional):** Migrate `raw_messages.db` per-bot tables to PostgreSQL:
- Eliminates all "database is locked" issues
- Enables cross-bot queries (compare P&L across all agents)
- Paper_trader + position_monitor + discord_collector all write to same DB with true concurrency

### Python Libraries

```python
# async PostgreSQL (drop-in replacement for aiosqlite pattern)
import asyncpg  # fastest async Postgres driver
# OR
from sqlalchemy.ext.asyncio import create_async_engine  # if you want ORM

# Connection pool (built into asyncpg)
pool = await asyncpg.create_pool(
    dsn="postgresql://owl:password@postgres:5432/options_owl",
    min_size=2,
    max_size=10,
)

# Usage (similar to aiosqlite context manager)
async with pool.acquire() as conn:
    rows = await conn.fetch("SELECT * FROM signals WHERE ticker = $1", "NVDA")
```

### Cost

- RAM: ~200-300MB for PostgreSQL 16 Alpine (droplet has spare capacity)
- Disk: Same as SQLite (data + WAL)
- CPU: Negligible for our write load (~100 writes/hour)
- Operational: Near-zero with Docker + healthcheck + `restart: always`

### Pros
- Solves ALL SQLite concurrency issues permanently
- True connection pooling (asyncpg pool)
- JSONB columns for score breakdowns (indexed, queryable)
- Network-accessible (all containers connect via TCP, no file mounts)
- Easy migration from SQLite (same SQL with minor syntax tweaks)
- Can add read replicas later if needed
- pgvector extension available for future ML embedding storage

### Cons
- One more container to manage (but Docker makes this trivial)
- Needs password management (add to `.env`)
- Slightly more complex backup (pg_dump vs file copy)
- Small latency overhead vs local file I/O (~1ms per query)

---

## 3. Option B: MongoDB

### Why NOT MongoDB for This Use Case

| Factor | Assessment |
|---|---|
| Data model | Our data is highly relational (trades → signals → outcomes → events). MongoDB's document model adds complexity. |
| Schema | We have well-defined schemas (pydantic models). MongoDB's schemaless nature is a liability, not a feature. |
| Joins | We do frequent joins (trade_signals ↔ signal_outcomes, paper_trades ↔ trade_events). MongoDB requires `$lookup` aggregation or denormalization. |
| Transactions | We need multi-table transactions (open trade + log event + update balance). MongoDB added multi-doc transactions but they're slower than Postgres. |
| RAM usage | MongoDB uses ~500MB-1GB minimum. Postgres Alpine uses ~200MB. |
| Query language | SQL is what we know. MongoDB's query API requires learning a new syntax. |
| Ecosystem | asyncpg is faster and more mature than motor (async MongoDB driver). |
| Time-series data | Our candle data is time-series. Postgres has `TimescaleDB` extension. MongoDB has time-series collections but they're newer and less battle-tested. |

**Bottom line:** MongoDB is designed for unstructured/semi-structured documents at massive scale. Our data is structured, relational, and moderate scale (~7GB). PostgreSQL is the right tool.

### When MongoDB Would Make Sense

- If we were storing raw N8N workflow execution logs (deeply nested JSON, variable schema)
- If we were ingesting millions of social media posts for sentiment analysis
- If we needed horizontal sharding across multiple servers
- None of these apply to our current or planned architecture

---

## 4. Option C: Keep SQLite (with improvements)

### What This Looks Like

Instead of migrating, we could address SQLite's pain points directly:

1. **Connection pool via aiosqlite-pool** — reuse connections instead of opening new ones
2. **Write serialization** — funnel all writes through a single async queue per DB
3. **Partition harvester DB** — split `options_data.db` by month (7GB → 12 × ~600MB files)
4. **Read replicas** — periodically copy DB files for read-only containers

### Assessment

| Improvement | Effort | Effectiveness |
|---|---|---|
| Connection pool | Low | Moderate — reduces file descriptor pressure |
| Write queue | Medium | High — eliminates "database is locked" |
| DB partitioning | Medium | High — solves 7GB problem |
| Read replicas | High | Moderate — adds complexity, stale data risk |

This is viable but creates more tech debt. We'd be working around SQLite's limitations instead of using a tool designed for concurrent access.

---

## 5. Recommendation

**Use PostgreSQL for all new code (owlet-sourcing). Keep SQLite for existing trading bots until Phase 2.**

### Implementation Plan

| Phase | What Changes | Risk |
|---|---|---|
| **Now** | Add `postgres` container to docker-compose.yml | Zero — doesn't affect existing bots |
| **Phase 1** | owlet-sourcing uses Postgres for state, signals, audit | Zero — new code only |
| **Phase 2** | Migrate harvester candle data to Postgres | Low — read-only migration, bots fall back to SQLite |
| **Phase 3** | Migrate bot DBs (raw_messages, paper_trades) to Postgres | Medium — requires careful testing |

### Why Not Migrate Everything Now?

The trading bots are live and making real money. Changing their DB layer during market hours is high-risk. The safe path:

1. Build owlet-sourcing on Postgres from day one
2. Prove Postgres works in production (weeks of uptime)
3. Then migrate trading bots one at a time, with rollback to SQLite

### Connection Pattern

```python
# options_owl/sourcing/db.py (new)
import asyncpg

_pool: asyncpg.Pool | None = None

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
        )
    return _pool

async def execute(query: str, *args) -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)

async def fetch(query: str, *args) -> list[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)
```

### Schema for Sourcing Tables (PostgreSQL)

```sql
-- Scoring audit (replaces SQLite version)
CREATE TABLE scoring_audit (
    id BIGSERIAL PRIMARY KEY,
    scan_time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT,
    score_total INTEGER,
    score_direction INTEGER,
    score_timing INTEGER,
    score_amplifiers INTEGER,
    score_adjustments INTEGER,
    score_calibration INTEGER,
    reasons JSONB,                    -- indexed JSON array
    filter_result TEXT,
    filter_reason TEXT,
    options_strike NUMERIC(10,2),
    options_premium NUMERIC(10,4),
    options_spread_pct NUMERIC(5,2),
    scan_duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_scan_time ON scoring_audit(scan_time);
CREATE INDEX idx_audit_ticker ON scoring_audit(ticker);
CREATE INDEX idx_audit_score ON scoring_audit(score_total);

-- Signals output (Phase 2 — trading bots read from here)
CREATE TABLE signals (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    score INTEGER NOT NULL,
    score_breakdown JSONB NOT NULL,
    strike NUMERIC(10,2),
    expiry DATE,
    premium NUMERIC(10,4),
    spread_pct NUMERIC(5,2),
    indicators JSONB,                  -- full indicator snapshot
    alpha_sources JSONB,               -- insider, congress, sentiment data
    emitted_at TIMESTAMPTZ NOT NULL,
    consumed_by JSONB DEFAULT '[]',    -- which bots have read this signal
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_signals_emitted ON signals(emitted_at);
CREATE INDEX idx_signals_ticker ON signals(ticker);

-- State: cooldowns
CREATE TABLE cooldowns (
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    last_alert_at TIMESTAMPTZ NOT NULL,
    cooldown_minutes INTEGER NOT NULL DEFAULT 90,
    PRIMARY KEY (ticker, direction)
);

-- State: circuit breaker
CREATE TABLE circuit_breaker (
    id INTEGER PRIMARY KEY DEFAULT 1,
    is_tripped BOOLEAN NOT NULL DEFAULT FALSE,
    trip_count INTEGER NOT NULL DEFAULT 0,
    tripped_at TIMESTAMPTZ,
    reset_at TIMESTAMPTZ,
    reason TEXT
);
```

---

## 6. Droplet Resource Check

Current droplet (129.212.138.145) running:
- owlet-kody, owlet-adam, owlet-vinny, owlet-yank (4 trading bots)
- owlet-harvester (data collection)
- **Planned:** owlet-sourcing + postgres

```
Current RAM usage:  ~2-3GB (5 containers)
Postgres addition:  ~200-300MB
Sourcing addition:  ~200-300MB (lighter than trading bots — no Webull SDK)
Total estimated:    ~3-4GB
Droplet RAM:        8GB (plenty of headroom)
```

---

## 7. Decision Matrix

| Criteria | SQLite (keep) | PostgreSQL | MongoDB |
|---|---|---|---|
| Solves concurrency issues | Partial (WAL) | **Full** | Full |
| Migration effort | None | Low-Medium | High |
| Operational complexity | None | Low | Medium |
| RAM overhead | 0 | ~200MB | ~500MB+ |
| Fits our data model | Good | **Best** | Worst |
| Future scalability | Poor | **Excellent** | Good |
| Team familiarity | High | High (SQL) | Low |
| Risk to live trading | None | **None (phased)** | Medium |
| Cost | $0 | $0 | $0 |

**Winner: PostgreSQL with phased migration.**
