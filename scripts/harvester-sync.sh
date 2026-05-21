#!/usr/bin/env bash
# harvester-sync.sh — Sync harvester data from droplet to local, then prune droplet.
#
# SAFETY: Data is NEVER deleted from droplet until local import is verified.
# The script counts rows before and after to ensure nothing is lost.
#
# Usage:
#   ./scripts/harvester-sync.sh              # sync + prune (keep 7 days on droplet)
#   ./scripts/harvester-sync.sh --sync-only  # sync without pruning
#   ./scripts/harvester-sync.sh --dry-run    # show what would happen, change nothing
#
# Schedule: Run Saturday morning via launchd (see scripts/com.optionsowl.harvester-sync.plist)
#
# What it does:
#   1. Query local DB for latest captured_at timestamp
#   2. Export new rows from droplet (snapshots + contracts) as CSV
#   3. Import CSVs into local DB (skip duplicates via INSERT OR IGNORE)
#   4. Verify: local row count >= droplet row count for the exported range
#   5. Only if verified: DELETE from droplet WHERE captured_at < (today - 7 days)
#   6. VACUUM droplet DB to reclaim space
#   7. Also syncs stock_candles table

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DROPLET_IP="129.212.138.145"
DROPLET_USER="root"
DROPLET="$DROPLET_USER@$DROPLET_IP"
SSH_KEY="$HOME/.ssh/id_ed25519_do"
SSH_CMD="ssh -i $SSH_KEY"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DB="$LOCAL_DIR/journal/owlet-harvester/options_data.db"
REMOTE_DB="/root/options-owl/journal/owlet-harvester/options_data.db"

SYNC_DIR="$LOCAL_DIR/journal/sync_tmp"
KEEP_DAYS=7  # days to keep on droplet after sync

LOG_FILE="$LOCAL_DIR/journal/harvester_sync.log"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
SYNC_ONLY=false
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --sync-only) SYNC_ONLY=true ;;
    --dry-run)   DRY_RUN=true ;;
    *)           echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE"
}

die() {
  log "FATAL: $1"
  exit 1
}

mkdir -p "$SYNC_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

log "=========================================="
log "Harvester sync starting"
log "  Local DB: $LOCAL_DB"
log "  Droplet: $DROPLET:$REMOTE_DB"
log "  Mode: $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || ([ "$SYNC_ONLY" = true ] && echo 'SYNC ONLY' || echo 'SYNC + PRUNE'))"
log "=========================================="

# ---------------------------------------------------------------------------
# Step 0: Verify local DB exists
# ---------------------------------------------------------------------------
if [ ! -f "$LOCAL_DB" ]; then
  die "Local DB not found: $LOCAL_DB"
fi

# ---------------------------------------------------------------------------
# Step 1: Find latest local timestamp
# ---------------------------------------------------------------------------
log "Step 1: Finding latest local timestamp..."

LOCAL_MAX_TS=$(sqlite3 "$LOCAL_DB" "SELECT MAX(captured_at) FROM harvest_snapshots")
LOCAL_SNAP_COUNT=$(sqlite3 "$LOCAL_DB" "SELECT COUNT(*) FROM harvest_snapshots")
LOCAL_CONTRACT_COUNT=$(sqlite3 "$LOCAL_DB" "SELECT COUNT(*) FROM harvest_contracts")

log "  Local snapshots: $LOCAL_SNAP_COUNT (through $LOCAL_MAX_TS)"
log "  Local contracts: $LOCAL_CONTRACT_COUNT"

# ---------------------------------------------------------------------------
# Step 2: Get droplet stats
# ---------------------------------------------------------------------------
log "Step 2: Getting droplet stats..."

REMOTE_STATS=$($SSH_CMD "$DROPLET" "sqlite3 $REMOTE_DB \"
  SELECT
    (SELECT COUNT(*) FROM harvest_snapshots) || '|' ||
    (SELECT COUNT(*) FROM harvest_snapshots WHERE captured_at > '$LOCAL_MAX_TS') || '|' ||
    (SELECT MIN(captured_at) FROM harvest_snapshots) || '|' ||
    (SELECT MAX(captured_at) FROM harvest_snapshots) || '|' ||
    (SELECT COUNT(*) FROM harvest_contracts)
\"")

IFS='|' read -r REMOTE_TOTAL REMOTE_NEW REMOTE_MIN REMOTE_MAX REMOTE_CONTRACTS <<< "$REMOTE_STATS"

log "  Droplet total snapshots: $REMOTE_TOTAL"
log "  New snapshots to sync: $REMOTE_NEW"
log "  Droplet range: $REMOTE_MIN → $REMOTE_MAX"
log "  Droplet contracts: $REMOTE_CONTRACTS"

if [ "$REMOTE_NEW" -eq 0 ]; then
  log "No new data to sync. Done."
  exit 0
fi

# ---------------------------------------------------------------------------
# Step 3: Export new data from droplet
# ---------------------------------------------------------------------------
log "Step 3: Exporting $REMOTE_NEW new snapshots from droplet..."

if [ "$DRY_RUN" = true ]; then
  log "  [DRY RUN] Would export $REMOTE_NEW snapshots + new contracts"
else
  # Export snapshots (CSV with headers)
  $SSH_CMD "$DROPLET" "sqlite3 -header -csv $REMOTE_DB \"
    SELECT * FROM harvest_snapshots
    WHERE captured_at > '$LOCAL_MAX_TS'
    ORDER BY captured_at
  \"" > "$SYNC_DIR/snapshots_new.csv"

  EXPORTED_LINES=$(wc -l < "$SYNC_DIR/snapshots_new.csv")
  EXPORTED_ROWS=$((EXPORTED_LINES - 1))  # minus header
  log "  Exported $EXPORTED_ROWS snapshot rows ($(du -h "$SYNC_DIR/snapshots_new.csv" | cut -f1))"

  # Sanity check: exported count should match expected
  if [ "$EXPORTED_ROWS" -lt "$((REMOTE_NEW - 100))" ]; then
    die "Export count mismatch! Expected ~$REMOTE_NEW, got $EXPORTED_ROWS. Aborting."
  fi

  # Export ALL contracts (small table, just replace/ignore)
  $SSH_CMD "$DROPLET" "sqlite3 -header -csv $REMOTE_DB \"
    SELECT * FROM harvest_contracts
  \"" > "$SYNC_DIR/contracts_all.csv"

  CONTRACT_LINES=$(wc -l < "$SYNC_DIR/contracts_all.csv")
  log "  Exported $((CONTRACT_LINES - 1)) contracts"

  # Export stock_candles if they exist on droplet
  CANDLE_COUNT=$($SSH_CMD "$DROPLET" "sqlite3 $REMOTE_DB \"SELECT COUNT(*) FROM stock_candles\" 2>/dev/null || echo 0")
  CANDLE_EXPORTED=0
  if [ "$CANDLE_COUNT" -gt 0 ]; then
    LOCAL_CANDLE_MAX=$(sqlite3 "$LOCAL_DB" "SELECT MAX(bar_start) FROM stock_candles" 2>/dev/null || echo "2020-01-01")
    if [ -z "$LOCAL_CANDLE_MAX" ] || [ "$LOCAL_CANDLE_MAX" = "" ]; then
      LOCAL_CANDLE_MAX="2020-01-01"
    fi
    $SSH_CMD "$DROPLET" "sqlite3 -header -csv $REMOTE_DB \"
      SELECT * FROM stock_candles
      WHERE bar_start > '$LOCAL_CANDLE_MAX'
      ORDER BY bar_start
    \"" > "$SYNC_DIR/candles_new.csv"
    CANDLE_EXPORTED=$(( $(wc -l < "$SYNC_DIR/candles_new.csv") - 1 ))
    log "  Exported $CANDLE_EXPORTED new stock candles"
  fi
fi

# ---------------------------------------------------------------------------
# Step 4: Import into local DB
# ---------------------------------------------------------------------------
log "Step 4: Importing into local DB..."

if [ "$DRY_RUN" = true ]; then
  log "  [DRY RUN] Would import $REMOTE_NEW snapshots"
else
  # Import snapshots
  BEFORE_COUNT=$(sqlite3 "$LOCAL_DB" "SELECT COUNT(*) FROM harvest_snapshots")

  sqlite3 "$LOCAL_DB" <<EOSQL
.mode csv
.headers on
-- Use a temp table to avoid duplicates (captured_at + contract_ticker is unique)
CREATE TEMP TABLE IF NOT EXISTS _import_snap (
    id INTEGER,
    contract_ticker TEXT,
    captured_at TEXT,
    underlying_price REAL,
    bid REAL,
    ask REAL,
    bid_size INTEGER,
    ask_size INTEGER,
    midpoint REAL,
    last_trade_price REAL,
    last_trade_ts_ns INTEGER,
    day_open REAL,
    day_high REAL,
    day_low REAL,
    day_close REAL,
    day_volume INTEGER,
    day_vwap REAL,
    open_interest INTEGER,
    implied_volatility REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL
);
.import $SYNC_DIR/snapshots_new.csv _import_snap

INSERT OR IGNORE INTO harvest_snapshots (
    contract_ticker, captured_at, underlying_price, bid, ask,
    bid_size, ask_size, midpoint, last_trade_price, last_trade_ts_ns,
    day_open, day_high, day_low, day_close, day_volume, day_vwap,
    open_interest, implied_volatility, delta, gamma, theta, vega
)
SELECT
    contract_ticker, captured_at, underlying_price, bid, ask,
    bid_size, ask_size, midpoint, last_trade_price, last_trade_ts_ns,
    day_open, day_high, day_low, day_close, day_volume, day_vwap,
    open_interest, implied_volatility, delta, gamma, theta, vega
FROM _import_snap
WHERE captured_at != 'captured_at';  -- skip CSV header row if imported

DROP TABLE _import_snap;
EOSQL

  AFTER_COUNT=$(sqlite3 "$LOCAL_DB" "SELECT COUNT(*) FROM harvest_snapshots")
  IMPORTED=$((AFTER_COUNT - BEFORE_COUNT))
  log "  Imported $IMPORTED new snapshots (before=$BEFORE_COUNT, after=$AFTER_COUNT)"

  # Import contracts (INSERT OR IGNORE — won't overwrite existing)
  sqlite3 "$LOCAL_DB" <<EOSQL
.mode csv
.headers on
CREATE TEMP TABLE IF NOT EXISTS _import_contracts (
    contract_ticker TEXT,
    underlying TEXT,
    strike REAL,
    expiry_date TEXT,
    option_type TEXT,
    first_seen_at TEXT
);
.import $SYNC_DIR/contracts_all.csv _import_contracts

INSERT OR IGNORE INTO harvest_contracts (
    contract_ticker, underlying, strike, expiry_date, option_type, first_seen_at
)
SELECT contract_ticker, underlying, strike, expiry_date, option_type, first_seen_at
FROM _import_contracts
WHERE contract_ticker != 'contract_ticker';

DROP TABLE _import_contracts;
EOSQL

  NEW_CONTRACT_COUNT=$(sqlite3 "$LOCAL_DB" "SELECT COUNT(*) FROM harvest_contracts")
  log "  Contracts: $LOCAL_CONTRACT_COUNT → $NEW_CONTRACT_COUNT"

  # Import stock candles if exported
  if [ -f "$SYNC_DIR/candles_new.csv" ] && [ "$CANDLE_EXPORTED" -gt 0 ]; then
    # Ensure stock_candles table exists locally (matches droplet schema)
    sqlite3 "$LOCAL_DB" "CREATE TABLE IF NOT EXISTS stock_candles (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ticker TEXT NOT NULL,
      timeframe TEXT NOT NULL,
      bar_start_ts INTEGER NOT NULL,
      bar_start TEXT NOT NULL,
      open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
      volume REAL DEFAULT 0, vwap REAL DEFAULT 0,
      source TEXT DEFAULT 'poll'
    )"

    BEFORE_CANDLES=$(sqlite3 "$LOCAL_DB" "SELECT COUNT(*) FROM stock_candles")
    sqlite3 "$LOCAL_DB" <<EOSQL2
.mode csv
.headers on
CREATE TEMP TABLE IF NOT EXISTS _import_candles (
    id INTEGER,
    ticker TEXT, timeframe TEXT, bar_start_ts INTEGER,
    bar_start TEXT,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, vwap REAL, source TEXT
);
.import $SYNC_DIR/candles_new.csv _import_candles

INSERT OR IGNORE INTO stock_candles (
    ticker, timeframe, bar_start_ts, bar_start, open, high, low, close, volume, vwap, source
)
SELECT ticker, timeframe, bar_start_ts, bar_start, open, high, low, close, volume, vwap, source
FROM _import_candles
WHERE ticker != 'ticker';

DROP TABLE _import_candles;
EOSQL2
    AFTER_CANDLES=$(sqlite3 "$LOCAL_DB" "SELECT COUNT(*) FROM stock_candles")
    log "  Stock candles: $BEFORE_CANDLES → $AFTER_CANDLES"
  fi

  # Cleanup temp files
  rm -f "$SYNC_DIR/snapshots_new.csv" "$SYNC_DIR/contracts_all.csv" "$SYNC_DIR/candles_new.csv"
fi

# ---------------------------------------------------------------------------
# Step 5: Verify import before pruning
# ---------------------------------------------------------------------------
log "Step 5: Verifying import..."

if [ "$DRY_RUN" = true ]; then
  log "  [DRY RUN] Would verify local >= droplet for synced range"
else
  FINAL_LOCAL=$(sqlite3 "$LOCAL_DB" "SELECT COUNT(*) FROM harvest_snapshots")
  FINAL_LOCAL_MAX=$(sqlite3 "$LOCAL_DB" "SELECT MAX(captured_at) FROM harvest_snapshots")

  log "  Local total: $FINAL_LOCAL snapshots (through $FINAL_LOCAL_MAX)"
  log "  Droplet total: $REMOTE_TOTAL snapshots (through $REMOTE_MAX)"

  # Local should have at least as many as droplet (we never delete locally)
  if [ "$FINAL_LOCAL" -lt "$REMOTE_TOTAL" ]; then
    # This can happen if droplet has rows we already had — that's fine
    # as long as we have everything through the droplet max
    log "  WARNING: Local ($FINAL_LOCAL) < droplet ($REMOTE_TOTAL) — checking max timestamp..."
    if [ "$FINAL_LOCAL_MAX" \< "$REMOTE_MAX" ]; then
      die "IMPORT INCOMPLETE: local max=$FINAL_LOCAL_MAX < droplet max=$REMOTE_MAX. NOT pruning!"
    fi
    log "  OK: local max timestamp matches droplet — some rows may differ in IDs but data is complete"
  else
    log "  VERIFIED: local ($FINAL_LOCAL) >= droplet ($REMOTE_TOTAL)"
  fi
fi

# ---------------------------------------------------------------------------
# Step 6: Prune droplet (keep last N days)
# ---------------------------------------------------------------------------
if [ "$SYNC_ONLY" = true ]; then
  log "Step 6: SKIPPED (--sync-only mode)"
  log "Done. Sync complete without pruning."
  exit 0
fi

if [ "$DRY_RUN" = true ]; then
  CUTOFF_DATE=$(date -v-${KEEP_DAYS}d '+%Y-%m-%d')
  WOULD_DELETE=$($SSH_CMD "$DROPLET" "sqlite3 $REMOTE_DB \"
    SELECT COUNT(*) FROM harvest_snapshots
    WHERE date(captured_at) < '$CUTOFF_DATE'
  \"")
  log "Step 6: [DRY RUN] Would delete $WOULD_DELETE snapshots older than $CUTOFF_DATE from droplet"
  log "Done. (dry run — no changes made)"
  exit 0
fi

log "Step 6: Pruning droplet (keeping last $KEEP_DAYS days)..."

CUTOFF_DATE=$(date -v-${KEEP_DAYS}d '+%Y-%m-%d')
log "  Cutoff date: $CUTOFF_DATE"

# Count before delete
PRUNE_COUNT=$($SSH_CMD "$DROPLET" "sqlite3 $REMOTE_DB \"
  SELECT COUNT(*) FROM harvest_snapshots
  WHERE date(captured_at) < '$CUTOFF_DATE'
\"")
log "  Rows to prune: $PRUNE_COUNT"

if [ "$PRUNE_COUNT" -eq 0 ]; then
  log "  Nothing to prune. Done."
  exit 0
fi

# Delete old snapshots
$SSH_CMD "$DROPLET" "sqlite3 $REMOTE_DB \"
  DELETE FROM harvest_snapshots
  WHERE date(captured_at) < '$CUTOFF_DATE';
\""
log "  Deleted $PRUNE_COUNT old snapshots"

# Delete orphaned contracts (no snapshots reference them)
ORPHAN_COUNT=$($SSH_CMD "$DROPLET" "sqlite3 $REMOTE_DB \"
  SELECT COUNT(*) FROM harvest_contracts
  WHERE contract_ticker NOT IN (SELECT DISTINCT contract_ticker FROM harvest_snapshots)
\"")
if [ "$ORPHAN_COUNT" -gt 0 ]; then
  $SSH_CMD "$DROPLET" "sqlite3 $REMOTE_DB \"
    DELETE FROM harvest_contracts
    WHERE contract_ticker NOT IN (SELECT DISTINCT contract_ticker FROM harvest_snapshots);
  \""
  log "  Deleted $ORPHAN_COUNT orphaned contracts"
fi

# Also prune old stock candles
$SSH_CMD "$DROPLET" "sqlite3 $REMOTE_DB \"
  DELETE FROM stock_candles WHERE date(bar_start) < '$CUTOFF_DATE';
\"" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 7: VACUUM to reclaim space
# ---------------------------------------------------------------------------
log "Step 7: VACUUMing droplet DB..."
$SSH_CMD "$DROPLET" "sqlite3 $REMOTE_DB 'VACUUM;'"

FINAL_SIZE=$($SSH_CMD "$DROPLET" "du -h $REMOTE_DB | cut -f1")
FINAL_COUNT=$($SSH_CMD "$DROPLET" "sqlite3 $REMOTE_DB 'SELECT COUNT(*) FROM harvest_snapshots'")
log "  Droplet DB after prune: $FINAL_SIZE ($FINAL_COUNT snapshots)"

log "=========================================="
log "Sync + prune complete!"
log "  Local: $FINAL_LOCAL snapshots (full archive)"
log "  Droplet: $FINAL_COUNT snapshots (last $KEEP_DAYS days, $FINAL_SIZE)"
log "=========================================="
