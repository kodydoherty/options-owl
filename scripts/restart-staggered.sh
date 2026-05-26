#!/usr/bin/env bash
# Restart all trading owlets with staggered delays to avoid Webull 429 rate limits.
# Usage: ssh root@droplet "cd /root/options-owl && ./scripts/restart-staggered.sh"
#   or:  ./scripts/restart-staggered.sh  (runs via SSH from local)

set -e

DELAY=${1:-15}  # seconds between each restart (default 15)

BOTS=(owlet-kody owlet-adam owlet-vinny owlet-yank owlet-harvester)

run_on_droplet() {
    if [ -f /root/options-owl/docker-compose.yml ]; then
        # Running on droplet directly
        cd /root/options-owl
        for bot in "${BOTS[@]}"; do
            echo "Recreating $bot..."
            # Use 'up -d' (not 'restart') so config changes in docker-compose.yml
            # are applied. 'restart' only restarts existing containers with OLD config.
            docker compose up -d "$bot"
            if [ "$bot" != "${BOTS[-1]}" ]; then
                echo "  Waiting ${DELAY}s before next bot..."
                sleep "$DELAY"
            fi
        done
        echo "All bots recreated."
        docker compose ps
    else
        # Running locally — SSH to droplet
        echo "=== Staggered recreate (${DELAY}s delay) ==="
        for bot in "${BOTS[@]}"; do
            echo "Recreating $bot..."
            ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145 \
                "cd /root/options-owl && docker compose up -d $bot"
            if [ "$bot" != "${BOTS[-1]}" ]; then
                echo "  Waiting ${DELAY}s before next bot..."
                sleep "$DELAY"
            fi
        done
        echo "=== All bots recreated ==="
        ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145 \
            "cd /root/options-owl && docker compose ps"
    fi
}

run_on_droplet
