#!/bin/bash
# One-time setup for the babysit cron job on the droplet.
# Usage: ssh to droplet, then run: ./scripts/setup-babysit.sh <ANTHROPIC_API_KEY>

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <ANTHROPIC_API_KEY>"
    echo "Get a key from: https://console.anthropic.com/settings/keys"
    exit 1
fi

API_KEY="$1"

# Store API key in /etc/environment (persists across reboots, available to cron)
if grep -q ANTHROPIC_API_KEY /etc/environment 2>/dev/null; then
    sed -i "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=$API_KEY|" /etc/environment
else
    echo "ANTHROPIC_API_KEY=$API_KEY" >> /etc/environment
fi

# Also export for current session
export ANTHROPIC_API_KEY="$API_KEY"

# Test Claude Code auth
echo "Testing Claude Code auth..."
RESULT=$(claude -p "respond with just OK" --max-turns 1 --output-format text 2>&1 | head -1)
if echo "$RESULT" | grep -qi "ok"; then
    echo "Claude Code auth: OK"
else
    echo "Claude Code auth FAILED: $RESULT"
    exit 1
fi

# Create log directory
mkdir -p /root/options-owl/journal

# Lock down /etc/environment (root-only read — key never in crontab)
chmod 600 /etc/environment

# Set up cron job — every 10 min during market hours (Mon-Fri, 9:20-16:10 ET)
# ET = UTC-4 (EDT) or UTC-5 (EST). Using UTC times: 13:20-20:10 EDT
# Sources /etc/environment for ANTHROPIC_API_KEY (not inline — keeps key out of crontab)
CRON_LINE="*/10 13-20 * * 1-5 . /etc/environment && /root/options-owl/scripts/babysit.sh >> /root/options-owl/journal/babysit.log 2>&1"

# Remove old babysit cron entry if exists
crontab -l 2>/dev/null | grep -v 'babysit.sh' | crontab - 2>/dev/null || true

# Add new cron entry
(crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -

echo "Cron job installed:"
crontab -l | grep babysit

echo ""
echo "Setup complete! The babysitter will run every 10 min during market hours."
echo "Logs: /root/options-owl/journal/babysit.log"
echo "Test now: /root/options-owl/scripts/babysit.sh"
