"""Log file reader — tail + grep for error/warning lines."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def get_log_dir(agent_id: str) -> Path:
    """Get the log directory for an agent. Works inside Docker containers."""
    name = agent_id.replace("owlet_", "owlet-")
    # Inside the dashboard container, journal/ is mounted at /app/journal
    # Each agent's logs are at journal/owlet-{name}/logs/
    return Path(f"/app/journal/{name}/logs")


def get_today_log(agent_id: str) -> Path | None:
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    log_dir = get_log_dir(agent_id)
    log_file = log_dir / f"options_owl_{today}.log"
    if log_file.exists():
        return log_file
    return None


def tail_errors(
    agent_id: str,
    max_lines: int = 50,
    levels: tuple[str, ...] = ("ERROR", "WARNING"),
    search: str | None = None,
) -> list[dict]:
    """Read recent error/warning lines from today's log file."""
    log_file = get_today_log(agent_id)
    if not log_file:
        return []

    results = []
    level_pattern = "|".join(levels)
    regex = re.compile(rf"\b({level_pattern})\b", re.IGNORECASE)

    try:
        with open(log_file, "r", errors="replace") as f:
            for line in f:
                if not regex.search(line):
                    continue
                if search and search.lower() not in line.lower():
                    continue

                # Parse loguru format: "YYYY-MM-DD HH:MM:SS.mmm | LEVEL | ..."
                parts = line.split("|", 3)
                if len(parts) >= 3:
                    timestamp = parts[0].strip()
                    level = parts[1].strip()
                    message = parts[2].strip() if len(parts) == 3 else parts[3].strip()
                else:
                    timestamp = ""
                    level = "UNKNOWN"
                    message = line.strip()

                results.append({
                    "timestamp": timestamp,
                    "level": level,
                    "message": message[:500],  # cap length
                    "raw": line.strip()[:600],
                })

        # Return last N lines
        return results[-max_lines:]
    except Exception:
        return []
