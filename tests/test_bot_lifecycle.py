"""Tests for bot lifecycle: retry logic, heartbeat, logging setup."""

import time
from pathlib import Path
from unittest.mock import MagicMock

from options_owl.main import (
    configure_logging,
    write_heartbeat,
)


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class TestHeartbeat:
    def test_write_heartbeat_creates_file(self, tmp_path, monkeypatch):
        hb = tmp_path / "heartbeat"
        monkeypatch.setattr("options_owl.main.HEARTBEAT_PATH", hb)
        write_heartbeat()
        assert hb.exists()
        ts = int(hb.read_text())
        assert abs(ts - int(time.time())) < 2

    def test_heartbeat_updates_on_second_call(self, tmp_path, monkeypatch):
        hb = tmp_path / "heartbeat"
        monkeypatch.setattr("options_owl.main.HEARTBEAT_PATH", hb)
        write_heartbeat()
        first = int(hb.read_text())
        write_heartbeat()
        second = int(hb.read_text())
        assert second >= first

    def test_heartbeat_survives_bad_path(self, monkeypatch):
        """write_heartbeat should not raise even if path is broken."""
        bad = Path("/nonexistent_root_dir/heartbeat")
        monkeypatch.setattr("options_owl.main.HEARTBEAT_PATH", bad)
        # Should not raise
        write_heartbeat()


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


class TestLogging:
    def test_configure_logging_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("options_owl.main.LOG_DIR", tmp_path / "logs")
        configure_logging(verbose=False)
        assert (tmp_path / "logs").exists()

    def test_configure_logging_verbose(self, tmp_path, monkeypatch):
        monkeypatch.setattr("options_owl.main.LOG_DIR", tmp_path / "logs")
        configure_logging(verbose=True)
        assert (tmp_path / "logs").exists()


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_keyboard_interrupt_exits_cleanly(self, monkeypatch):
        """run_collector_with_retry should exit on KeyboardInterrupt."""
        from options_owl.main import run_collector_with_retry

        settings = MagicMock()
        settings.DISCORD_TOKEN = "fake"

        call_count = 0

        def fake_run(coro):
            nonlocal call_count
            coro.close()  # dispose the un-awaited coroutine (asyncio.run is mocked)
            call_count += 1
            raise KeyboardInterrupt

        monkeypatch.setattr("options_owl.main.asyncio.run", fake_run)
        monkeypatch.setattr("options_owl.main.write_heartbeat", lambda: None)
        monkeypatch.setattr("options_owl.main.LOG_DIR", Path("/tmp/test_logs"))

        # Should not raise — exits cleanly
        run_collector_with_retry(settings)
        assert call_count == 1

    def test_retries_on_exception(self, monkeypatch):
        """run_collector_with_retry should retry on generic exceptions."""
        from options_owl.main import run_collector_with_retry

        settings = MagicMock()
        settings.DISCORD_TOKEN = "fake"

        call_count = 0

        def fake_run(coro):
            nonlocal call_count
            coro.close()  # dispose the un-awaited coroutine (asyncio.run is mocked)
            call_count += 1
            if call_count < 3:
                raise ConnectionError("test")
            raise KeyboardInterrupt  # stop after 3 attempts

        monkeypatch.setattr("options_owl.main.asyncio.run", fake_run)
        monkeypatch.setattr("options_owl.main.write_heartbeat", lambda: None)
        monkeypatch.setattr("options_owl.main.time.sleep", lambda _: None)  # skip delays

        run_collector_with_retry(settings)
        assert call_count == 3
