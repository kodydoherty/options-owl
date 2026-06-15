"""Wrapper to run the archived Simpsons backtest with corrected DB path.

The archived script (scripts/archive_backtests/backtest_simpsons.py) computes
PROJECT_DIR as parent.parent of its own file, which resolves to scripts/ after
the file was moved into archive_backtests/. This wrapper fixes the ThetaData
DB path without modifying the archived file.

Usage: same CLI args as the archived script, e.g.
    python scripts/backtest_simpsons_replay.py --days 190
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "archive_backtests"))

import backtest_simpsons as bs  # noqa: E402

bs.THETADATA_DB = str(ROOT / "journal" / "thetadata_options.db")

if __name__ == "__main__":
    bs.main()
