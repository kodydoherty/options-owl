"""OptionsOwl Agent Dashboard — FastAPI app with auth, trades, controls, WebSocket."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import asyncpg
import uvicorn
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocket, WebSocketDisconnect

from options_owl.dashboard.auth import (
    authenticate,
    change_password,
    create_token,
    decode_token,
    ensure_users_table,
    seed_default_users,
    verify_password,
    get_user,
)
from options_owl.dashboard.controls import (
    get_agent_heartbeat,
    get_kill_switch,
    get_paper_mode,
    set_kill_switch,
    set_paper_mode,
)
from options_owl.dashboard.db import (
    get_agent_state,
    get_closed_trades,
    get_daily_pnl,
    get_exit_distribution,
    get_hourly_performance,
    get_open_trades,
    get_pnl_curve,
    get_portfolio_stats,
    get_premium_ticks,
    get_ticker_performance,
    get_trade_by_id,
    get_trade_duration_stats,
    get_trade_events,
)
from options_owl.dashboard.logs import tail_errors
from options_owl.dashboard.ws import manager, redis_subscriber

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    dsn = os.getenv(
        "DATABASE_URL",
        "postgresql://owl:owl_dev_2026@postgres:5432/options_owl",
    )
    logger.info(f"Dashboard: connecting to PG {dsn.split('@')[-1]}")
    _pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=5)
    await ensure_users_table(_pool)
    await seed_default_users(_pool)
    logger.info("Dashboard: PG pool ready, users seeded")

    # Start Redis subscriber for live updates
    asyncio.create_task(redis_subscriber(_pool))

    yield

    if _pool:
        await _pool.close()


app = FastAPI(title="OptionsOwl Dashboard", docs_url=None, redoc_url=None, lifespan=lifespan)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Connection pool
_pool: asyncpg.Pool | None = None


# ---------------------------------------------------------------------------
# Template filters
# ---------------------------------------------------------------------------


def _fmt_money(value) -> str:
    if value is None:
        return "$0.00"
    v = float(value)
    sign = "+" if v > 0 else ""
    return f"{sign}${v:,.2f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "0.0%"
    v = float(value)
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}%"


def _fmt_time(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%I:%M %p")
    return str(value)


def _fmt_date(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%m/%d %I:%M %p")
    return str(value)


def _pnl_class(value) -> str:
    if value is None:
        return ""
    return "text-green-400" if float(value) >= 0 else "text-red-400"


templates.env.filters["money"] = _fmt_money
templates.env.filters["pct"] = _fmt_pct
templates.env.filters["ftime"] = _fmt_time
templates.env.filters["fdate"] = _fmt_date
templates.env.filters["pnl_class"] = _pnl_class


def _render(name: str, context: dict, status_code: int = 200):
    """Render a template — compatible with FastAPI 0.111+ and 0.136+."""
    request = context.get("request")
    return templates.TemplateResponse(
        request=request, name=name, context=context, status_code=status_code,
    )


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

PUBLIC_PATHS = {"/login", "/health", "/favicon.ico"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)

        token = request.cookies.get("owl_token")
        if not token:
            return RedirectResponse("/login", status_code=302)

        payload = decode_token(token)
        if not payload:
            resp = RedirectResponse("/login", status_code=302)
            resp.delete_cookie("owl_token")
            return resp

        request.state.user = payload
        return await call_next(request)


app.add_middleware(AuthMiddleware)


# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory)
# ---------------------------------------------------------------------------

_login_attempts: dict[str, list[float]] = {}
LOGIN_RATE_LIMIT = 5
LOGIN_RATE_WINDOW = 60


def _check_rate_limit(ip: str) -> bool:
    import time
    now = time.time()
    attempts = _login_attempts.get(ip, [])
    attempts = [t for t in attempts if now - t < LOGIN_RATE_WINDOW]
    _login_attempts[ip] = attempts
    return len(attempts) < LOGIN_RATE_LIMIT


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "pool": _pool is not None}


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    return _render("login.html", {"request": request, "error": error})


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(ip):
        return _render(
            "login.html",
            {"request": request, "error": "Too many attempts. Try again in a minute."},
            status_code=429,
        )

    import time
    _login_attempts.setdefault(ip, []).append(time.time())

    user = await authenticate(_pool, username, password)
    if not user:
        return _render(
            "login.html",
            {"request": request, "error": "Invalid username or password"},
            status_code=401,
        )

    token = create_token(user["username"], user["agent_id"])
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        "owl_token",
        token,
        httponly=True,
        samesite="strict",
        max_age=86400,
    )
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("owl_token")
    return response


# ---------------------------------------------------------------------------
# Dashboard home
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, days: int = Query(default=7, ge=1, le=90)):
    user = request.state.user
    agent_id = user["agent_id"]

    open_trades, closed_trades, stats, agent_state, daily_pnl = await asyncio.gather(
        get_open_trades(_pool, agent_id),
        get_closed_trades(_pool, agent_id, days=days),
        get_portfolio_stats(_pool, agent_id),
        get_agent_state(_pool, agent_id),
        get_daily_pnl(_pool, agent_id, days=min(days, 14)),
    )

    paper_mode, kill_switch, heartbeat = await asyncio.gather(
        get_paper_mode(agent_id),
        get_kill_switch(agent_id),
        get_agent_heartbeat(agent_id),
    )
    errors = tail_errors(agent_id, max_lines=30)

    # Serialize daily P&L for sparkline
    sparkline = []
    for d in daily_pnl:
        sparkline.append({
            "date": d["trade_date"].isoformat() if hasattr(d["trade_date"], "isoformat") else str(d["trade_date"]),
            "pnl": float(d["daily_pnl"]),
        })

    return _render("dashboard.html", {
        "request": request,
        "user": user,
        "agent_id": agent_id,
        "open_trades": open_trades,
        "closed_trades": closed_trades,
        "stats": stats,
        "agent_state": agent_state,
        "paper_mode": paper_mode,
        "kill_switch": kill_switch,
        "errors": errors,
        "days": days,
        "sparkline": sparkline,
        "heartbeat": heartbeat,
    })


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request, days: int = Query(default=30, ge=1, le=90)):
    user = request.state.user
    agent_id = user["agent_id"]

    pnl_curve, daily_pnl, exits, tickers, hourly, duration, stats = await asyncio.gather(
        get_pnl_curve(_pool, agent_id, days=days),
        get_daily_pnl(_pool, agent_id, days=days),
        get_exit_distribution(_pool, agent_id, days=days),
        get_ticker_performance(_pool, agent_id, days=days),
        get_hourly_performance(_pool, agent_id, days=days),
        get_trade_duration_stats(_pool, agent_id, days=days),
        get_portfolio_stats(_pool, agent_id),
    )

    # Convert Decimal/date types for JSON serialization
    def _serialize(rows):
        result = []
        for row in rows:
            d = {}
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    d[k] = v.isoformat()
                elif hasattr(v, "__float__"):
                    d[k] = float(v)
                else:
                    d[k] = v
            result.append(d)
        return result

    return _render("analytics.html", {
        "request": request,
        "user": user,
        "days": days,
        "pnl_curve": _serialize(pnl_curve),
        "daily_pnl": _serialize(daily_pnl),
        "exits": _serialize(exits),
        "tickers": _serialize(tickers),
        "hourly": _serialize(hourly),
        "duration": {k: float(v) if v is not None else 0 for k, v in duration.items()},
        "stats": stats,
    })


# ---------------------------------------------------------------------------
# Trade detail
# ---------------------------------------------------------------------------


@app.get("/trade/{trade_id}", response_class=HTMLResponse)
async def trade_detail(request: Request, trade_id: int):
    user = request.state.user
    agent_id = user["agent_id"]

    trade = await get_trade_by_id(_pool, agent_id, trade_id)
    if not trade:
        return HTMLResponse("<h1>Trade not found</h1>", status_code=404)

    events = await get_trade_events(_pool, agent_id, trade.get("sqlite_id", trade_id))
    ticks = await get_premium_ticks(_pool, agent_id, trade.get("sqlite_id", trade_id))

    return _render("trade_detail.html", {
        "request": request,
        "user": user,
        "trade": trade,
        "events": events,
        "ticks": ticks,
    })


# ---------------------------------------------------------------------------
# Settings (password change)
# ---------------------------------------------------------------------------


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, msg: str = "", error: str = ""):
    return _render("settings.html", {
        "request": request,
        "user": request.state.user,
        "msg": msg,
        "error": error,
    })


@app.post("/settings/password")
async def settings_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
):
    user = request.state.user
    username = user["sub"]

    if new_password != confirm_password:
        return _render("settings.html", {
            "request": request, "user": user,
            "error": "New passwords don't match", "msg": "",
        })

    if len(new_password) < 8:
        return _render("settings.html", {
            "request": request, "user": user,
            "error": "Password must be at least 8 characters", "msg": "",
        })

    db_user = await get_user(_pool, username)
    if not db_user or not verify_password(current_password, db_user["password_hash"]):
        return _render("settings.html", {
            "request": request, "user": user,
            "error": "Current password is incorrect", "msg": "",
        })

    await change_password(_pool, username, new_password)
    return _render("settings.html", {
        "request": request, "user": user,
        "msg": "Password changed successfully", "error": "",
    })


# ---------------------------------------------------------------------------
# API: Agent controls
# ---------------------------------------------------------------------------


@app.post("/api/paper-mode")
async def api_paper_mode(request: Request):
    user = request.state.user
    body = await request.json()
    enabled = body.get("enabled", True)
    ok = await set_paper_mode(user["agent_id"], enabled)
    return JSONResponse({"ok": ok, "paper_mode": enabled})


@app.post("/api/kill-switch")
async def api_kill_switch(request: Request):
    user = request.state.user
    body = await request.json()
    enabled = body.get("enabled", True)
    ok = await set_kill_switch(user["agent_id"], enabled)
    return JSONResponse({"ok": ok, "kill_switch": enabled})


# ---------------------------------------------------------------------------
# API: Logs
# ---------------------------------------------------------------------------


@app.get("/api/logs")
async def api_logs(
    request: Request,
    level: str = Query(default="ERROR,WARNING"),
    search: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=200),
):
    user = request.state.user
    levels = tuple(level.upper().split(","))
    entries = tail_errors(
        user["agent_id"],
        max_lines=limit,
        levels=levels,
        search=search or None,
    )
    return JSONResponse(entries)


# ---------------------------------------------------------------------------
# API: CSV Export
# ---------------------------------------------------------------------------


@app.get("/api/export")
async def api_export(
    request: Request,
    days: int = Query(default=30, ge=1, le=90),
):
    import csv
    import io

    user = request.state.user
    agent_id = user["agent_id"]
    trades = await get_closed_trades(_pool, agent_id, days=days, limit=5000)

    output = io.StringIO()
    writer = csv.writer(output)
    columns = [
        "id", "ticker", "direction", "strike", "contracts",
        "premium_per_contract", "exit_premium", "pnl_dollars", "pnl_pct",
        "exit_reason", "exit_source", "hold_minutes", "score",
        "opened_at", "closed_at",
    ]
    writer.writerow(columns)
    for t in trades:
        writer.writerow([t.get(c, "") for c in columns])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=trades_{agent_id}_{days}d.csv"},
    )


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.cookies.get("owl_token")
    if not token:
        await websocket.close(code=4001)
        return

    payload = decode_token(token)
    if not payload:
        await websocket.close(code=4001)
        return

    agent_id = payload["agent_id"]
    await manager.connect(websocket, agent_id)
    logger.info(f"Dashboard WS: {payload['sub']} connected ({manager.active_count} total)")

    try:
        while True:
            # Keep connection alive, handle client pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        manager.disconnect(websocket, agent_id)
        logger.info(f"Dashboard WS: {payload['sub']} disconnected")
    except Exception:
        manager.disconnect(websocket, agent_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    port = int(os.getenv("DASHBOARD_PORT", "8443"))
    uvicorn.run(
        "options_owl.dashboard.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
