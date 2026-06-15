"""Tests for the agent dashboard UI (Spec 05)."""

from __future__ import annotations

import inspect



# ---------------------------------------------------------------------------
# Auth module
# ---------------------------------------------------------------------------


class TestAuth:
    def test_hash_and_verify_password(self):
        from options_owl.dashboard.auth import hash_password, verify_password

        hashed = hash_password("testpass123")
        assert verify_password("testpass123", hashed)
        assert not verify_password("wrongpass", hashed)

    def test_create_and_decode_token(self):
        from options_owl.dashboard.auth import create_token, decode_token

        token = create_token("kody", "owlet_kody")
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "kody"
        assert payload["agent_id"] == "owlet_kody"

    def test_decode_invalid_token(self):
        from options_owl.dashboard.auth import decode_token

        assert decode_token("invalid.token.here") is None
        assert decode_token("") is None

    def test_decode_expired_token(self):
        from jose import jwt
        from options_owl.dashboard.auth import SECRET_KEY, ALGORITHM, decode_token
        import time

        token = jwt.encode(
            {"sub": "kody", "agent_id": "owlet_kody", "exp": int(time.time()) - 10},
            SECRET_KEY,
            algorithm=ALGORITHM,
        )
        assert decode_token(token) is None

    def test_users_table_sql(self):
        from options_owl.dashboard.auth import USERS_TABLE_SQL

        assert "dashboard_users" in USERS_TABLE_SQL
        assert "password_hash" in USERS_TABLE_SQL
        assert "agent_id" in USERS_TABLE_SQL
        assert "is_admin" in USERS_TABLE_SQL


# ---------------------------------------------------------------------------
# DB queries module
# ---------------------------------------------------------------------------


class TestDBQueries:
    def test_all_query_functions_exist(self):
        from options_owl.dashboard import db

        assert callable(db.get_open_trades)
        assert callable(db.get_closed_trades)
        assert callable(db.get_trade_by_id)
        assert callable(db.get_trade_events)
        assert callable(db.get_agent_state)
        assert callable(db.get_portfolio_stats)
        assert callable(db.get_premium_ticks)
        # Analytics queries
        assert callable(db.get_pnl_curve)
        assert callable(db.get_daily_pnl)
        assert callable(db.get_exit_distribution)
        assert callable(db.get_ticker_performance)
        assert callable(db.get_hourly_performance)
        assert callable(db.get_trade_duration_stats)

    def test_get_closed_trades_accepts_days_param(self):
        sig = inspect.signature(
            __import__(
                "options_owl.dashboard.db", fromlist=["get_closed_trades"]
            ).get_closed_trades
        )
        assert "days" in sig.parameters
        assert sig.parameters["days"].default == 7

    def test_queries_filter_by_agent_id(self):
        """All queries must filter by agent_id for user isolation."""
        from options_owl.dashboard import db

        # Every query function should use agent_id parameter
        for fn_name in [
            "get_open_trades", "get_closed_trades", "get_trade_by_id",
            "get_trade_events", "get_agent_state", "get_portfolio_stats",
        ]:
            fn = getattr(db, fn_name)
            fn_source = inspect.getsource(fn)
            assert "agent_id" in fn_source, f"{fn_name} missing agent_id filter"


# ---------------------------------------------------------------------------
# Controls module
# ---------------------------------------------------------------------------


class TestControls:
    def test_control_functions_exist(self):
        from options_owl.dashboard import controls

        assert callable(controls.get_paper_mode)
        assert callable(controls.set_paper_mode)
        assert callable(controls.get_kill_switch)
        assert callable(controls.set_kill_switch)

    def test_control_keys_use_agent_id(self):
        from options_owl.dashboard import controls

        source = inspect.getsource(controls)
        assert "owl:control:" in source
        assert "paper_mode" in source
        assert "kill_switch" in source


# ---------------------------------------------------------------------------
# Logs module
# ---------------------------------------------------------------------------


class TestLogs:
    def test_tail_errors_function(self):
        from options_owl.dashboard.logs import tail_errors

        # Should return empty list when no log file exists
        result = tail_errors("owlet_nonexistent")
        assert result == []

    def test_log_severity_filtering(self):
        sig = inspect.signature(
            __import__(
                "options_owl.dashboard.logs", fromlist=["tail_errors"]
            ).tail_errors
        )
        assert "levels" in sig.parameters
        assert "search" in sig.parameters
        assert "max_lines" in sig.parameters


# ---------------------------------------------------------------------------
# WebSocket manager
# ---------------------------------------------------------------------------


class TestWSManager:
    def test_connection_manager_exists(self):
        from options_owl.dashboard.ws import ConnectionManager, manager

        assert isinstance(manager, ConnectionManager)

    def test_manager_active_count(self):
        from options_owl.dashboard.ws import ConnectionManager

        mgr = ConnectionManager()
        assert mgr.active_count == 0


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


class TestApp:
    def test_app_creates(self):
        from options_owl.dashboard.app import app

        assert app.title == "OptionsOwl Dashboard"

    def test_routes_exist(self):
        from options_owl.dashboard.app import app

        routes = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/login" in routes
        assert "/" in routes
        assert "/health" in routes
        assert "/ws" in routes
        assert "/api/paper-mode" in routes
        assert "/api/kill-switch" in routes
        assert "/api/logs" in routes
        assert "/logout" in routes
        assert "/analytics" in routes
        assert "/api/export" in routes

    def test_trade_detail_route(self):
        from options_owl.dashboard.app import app

        routes = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/trade/{trade_id}" in routes

    def test_template_filters(self):
        from options_owl.dashboard.app import _fmt_money, _fmt_pct, _pnl_class

        assert _fmt_money(100.5) == "+$100.50"
        assert _fmt_money(-50) == "$-50.00"
        assert _fmt_money(None) == "$0.00"
        assert _fmt_pct(32.6) == "+32.6%"
        assert _fmt_pct(-10.5) == "-10.5%"
        assert _pnl_class(100) == "text-green-400"
        assert _pnl_class(-50) == "text-red-400"


# ---------------------------------------------------------------------------
# Templates exist
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_template_files_exist(self):
        from pathlib import Path

        base = Path(__file__).parent.parent / "options_owl" / "dashboard" / "templates"
        assert (base / "base.html").exists()
        assert (base / "login.html").exists()
        assert (base / "dashboard.html").exists()
        assert (base / "trade_detail.html").exists()

    def test_analytics_template_exists(self):
        from pathlib import Path

        base = Path(__file__).parent.parent / "options_owl" / "dashboard" / "templates"
        assert (base / "analytics.html").exists()

    def test_static_files_exist(self):
        from pathlib import Path

        base = Path(__file__).parent.parent / "options_owl" / "dashboard" / "static"
        assert (base / "ws.js").exists()

    def test_login_has_csrf_protection(self):
        """Login form should use SameSite=Strict cookie."""
        source = inspect.getsource(
            __import__(
                "options_owl.dashboard.app", fromlist=["login_submit"]
            ).login_submit
        )
        assert "samesite" in source.lower()

    def test_login_has_rate_limiting(self):
        source = inspect.getsource(
            __import__(
                "options_owl.dashboard.app", fromlist=["login_submit"]
            ).login_submit
        )
        assert "rate_limit" in source.lower()


# ---------------------------------------------------------------------------
# Docker compose
# ---------------------------------------------------------------------------


class TestDockerCompose:
    def test_dashboard_container_exists(self):
        from pathlib import Path

        compose = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()
        assert "owlet-dashboard:" in compose
        assert "8443:8443" in compose
        assert "DASHBOARD_SECRET_KEY" in compose

    def test_dashboard_depends_on_pg_and_redis(self):
        from pathlib import Path

        compose = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()
        # Find dashboard section
        start = compose.find("owlet-dashboard:")
        end = compose.find("\n  owlet-", start + 20)
        section = compose[start:end] if end > 0 else compose[start:]
        assert "postgres:" in section
        assert "redis:" in section

    def test_dashboard_journal_readonly(self):
        """Dashboard should have read-only access to journal."""
        from pathlib import Path

        compose = Path("/Users/kody/dev/options-owl/docker-compose.yml").read_text()
        start = compose.find("owlet-dashboard:")
        end = compose.find("\n  owlet-", start + 20)
        section = compose[start:end] if end > 0 else compose[start:]
        assert "journal:/app/journal:ro" in section


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


class TestDependencies:
    def test_dashboard_deps_in_pyproject(self):
        from pathlib import Path

        toml = Path("/Users/kody/dev/options-owl/pyproject.toml").read_text()
        assert "fastapi" in toml
        assert "uvicorn" in toml
        assert "python-jose" in toml
        assert "bcrypt" in toml
        assert "jinja2" in toml
        assert "python-multipart" in toml


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_httponly_cookie(self):
        """JWT cookie must be HttpOnly."""
        source = inspect.getsource(
            __import__(
                "options_owl.dashboard.app", fromlist=["login_submit"]
            ).login_submit
        )
        assert "httponly=True" in source

    def test_auth_middleware_exists(self):
        from options_owl.dashboard.app import AuthMiddleware
        assert AuthMiddleware is not None

    def test_public_paths_limited(self):
        from options_owl.dashboard.app import PUBLIC_PATHS
        assert "/login" in PUBLIC_PATHS
        assert "/health" in PUBLIC_PATHS
        # Dashboard home should NOT be public
        assert "/" not in PUBLIC_PATHS

    def test_ws_validates_token(self):
        """WebSocket endpoint should validate JWT before accepting."""
        source = inspect.getsource(
            __import__(
                "options_owl.dashboard.app", fromlist=["websocket_endpoint"]
            ).websocket_endpoint
        )
        assert "decode_token" in source
        assert "4001" in source  # close code for unauthorized
