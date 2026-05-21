#!/usr/bin/env bash
set -e

echo "=== OptionsOwl Restart ==="

# Stop existing container
echo "Stopping container..."
docker compose down 2>/dev/null || true

# Clear Docker build cache for this project
echo "Clearing build cache..."
docker builder prune -f --filter "label=com.docker.compose.project=options-owl" 2>/dev/null || true
docker rmi options-owl-collector 2>/dev/null || true
docker rmi $(docker images -q --filter "dangling=true") 2>/dev/null || true

# Clear Python caches
echo "Clearing Python caches..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Rebuild from scratch (no cache)
echo "Building fresh image..."
docker compose build --no-cache

# Start
echo "Starting collector..."
docker compose up -d

echo ""
echo "=== Running! ==="
echo "Logs:    docker compose logs -f"
echo "Stop:    docker compose down"
echo "Status:  docker compose ps"
