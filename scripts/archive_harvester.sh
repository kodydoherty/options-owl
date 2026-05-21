#!/bin/bash
# archive_harvester.sh — Pull harvester DB from droplet, archive locally, purge old data on droplet.
#
# Run every Friday after market close (4:30 PM ET / 1:30 PM PT) via cron:
#   30 13 * * 5 /Users/kody/dev/options-owl/scripts/archive_harvester.sh
#
# What it does:
#   1. Pulls the harvester DB (options_data.db + WAL) from the droplet
#   2. Saves to local archive dir with week timestamp: archives/harvester_YYYY-MM-DD.db
#   3. Purges data older than 5 trading days on the droplet (keeps live DB small)
#   4. VACUUMs the droplet DB to reclaim space
#
# Archive lives at: ~/dev/options-owl/archives/
# Each archive is a full standalone SQLite DB (WAL checkpointed).

set -euo pipefail

DROPLET="root@129.212.138.145"
SSH_KEY="$HOME/.ssh/id_ed25519_do"
REMOTE_DB="/root/options-owl/journal/owlet-harvester/options_data.db"
LOCAL_ARCHIVE_DIR="$HOME/dev/options-owl/archives"
TODAY=$(date +%Y-%m-%d)
ARCHIVE_FILE="$LOCAL_ARCHIVE_DIR/harvester_${TODAY}.db"
KEEP_DAYS=5  # Keep 5 trading days on the droplet

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[archive] Starting weekly harvester archive — $TODAY${NC}"

# Step 0: Create local archive dir
mkdir -p "$LOCAL_ARCHIVE_DIR"

# Step 1: Checkpoint WAL on droplet (merge WAL into main DB)
echo -e "${YELLOW}[archive] Checkpointing WAL on droplet...${NC}"
ssh -i "$SSH_KEY" "$DROPLET" "sqlite3 $REMOTE_DB 'PRAGMA wal_checkpoint(TRUNCATE);'" 2>/dev/null || true

# Step 2: Pull the DB to local archive
echo -e "${YELLOW}[archive] Downloading harvester DB...${NC}"
scp -i "$SSH_KEY" "$DROPLET:$REMOTE_DB" "$ARCHIVE_FILE"
DB_SIZE=$(du -sh "$ARCHIVE_FILE" | cut -f1)
echo -e "${GREEN}[archive] Saved: $ARCHIVE_FILE ($DB_SIZE)${NC}"

# Step 3: Verify the archive is valid
ROW_COUNT=$(sqlite3 "$ARCHIVE_FILE" "SELECT COUNT(*) FROM harvest_snapshots;" 2>/dev/null || echo "0")
CANDLE_COUNT=$(sqlite3 "$ARCHIVE_FILE" "SELECT COUNT(*) FROM stock_candles;" 2>/dev/null || echo "0")
echo -e "${GREEN}[archive] Archive contains: $ROW_COUNT option snapshots, $CANDLE_COUNT stock candles${NC}"

if [ "$ROW_COUNT" -eq 0 ] && [ "$CANDLE_COUNT" -eq 0 ]; then
    echo "ERROR: Archive is empty! Aborting purge."
    exit 1
fi

# Step 4: Purge old data on droplet (keep last N days)
echo -e "${YELLOW}[archive] Purging data older than $KEEP_DAYS days on droplet...${NC}"
CUTOFF_DATE=$(date -v-${KEEP_DAYS}d +%Y-%m-%d 2>/dev/null || date -d "-${KEEP_DAYS} days" +%Y-%m-%d)

ssh -i "$SSH_KEY" "$DROPLET" "sqlite3 $REMOTE_DB \"
    DELETE FROM harvest_snapshots WHERE date(captured_at) < '$CUTOFF_DATE';
    DELETE FROM stock_candles WHERE date(bar_start) < '$CUTOFF_DATE';
\""

# Step 5: VACUUM to reclaim space
echo -e "${YELLOW}[archive] VACUUMing droplet DB...${NC}"
ssh -i "$SSH_KEY" "$DROPLET" "sqlite3 $REMOTE_DB 'VACUUM;'"

# Step 6: Report new size
NEW_SIZE=$(ssh -i "$SSH_KEY" "$DROPLET" "du -sh $REMOTE_DB" | cut -f1)
echo -e "${GREEN}[archive] Droplet DB after purge: $NEW_SIZE${NC}"

# Step 7: List local archives
echo -e "\n${GREEN}[archive] Local archives:${NC}"
ls -lh "$LOCAL_ARCHIVE_DIR"/harvester_*.db 2>/dev/null || echo "  (none)"

echo -e "\n${GREEN}[archive] Done! Archive: $ARCHIVE_FILE${NC}"
