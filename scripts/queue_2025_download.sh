#!/bin/bash
# Queue the 2025 historical download to start after the current download finishes.
# Current download: Jan 2026 - May 2026 (PID 62744)
# Queued download: Jan 2025 - Dec 2025

echo "Waiting for ThetaData download (PID 62744) to finish..."
while kill -0 62744 2>/dev/null; do
    sleep 60
done
echo "Current download finished at $(date)"

echo ""
echo "Starting 2025 historical download (Jan 2025 - Dec 2025)..."
cd /Users/kody/dev/options-owl

nohup python scripts/download_thetadata.py \
    --start 2025-01-02 \
    --end 2025-12-31 \
    --otm 2 \
    --batch-size 5 \
    > /tmp/thetadata_download_2025.log 2>&1 &

echo "2025 download started — PID: $!"
echo "Monitor: tail -f /tmp/thetadata_download_2025.log"
