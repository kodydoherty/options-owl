"""Authentication: bcrypt password hashing, JWT token create/decode."""

from __future__ import annotations

import os
import time

import asyncpg
import bcrypt
from jose import JWTError, jwt
from loguru import logger

SECRET_KEY = os.getenv("DASHBOARD_SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_SECONDS = 86400  # 24 hours

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------


def create_token(username: str, agent_id: str) -> str:
    payload = {
        "sub": username,
        "agent_id": agent_id,
        "exp": int(time.time()) + TOKEN_EXPIRE_SECONDS,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# User DB operations
# ---------------------------------------------------------------------------

USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dashboard_users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""


async def ensure_users_table(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(USERS_TABLE_SQL)


async def get_user(pool: asyncpg.Pool, username: str) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM dashboard_users WHERE username = $1", username
        )
        return dict(row) if row else None


async def create_user(
    pool: asyncpg.Pool,
    username: str,
    password: str,
    agent_id: str,
    is_admin: bool = False,
) -> int:
    hashed = hash_password(password)
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """INSERT INTO dashboard_users (username, password_hash, agent_id, is_admin)
               VALUES ($1, $2, $3, $4) RETURNING id""",
            username, hashed, agent_id, is_admin,
        )


async def authenticate(pool: asyncpg.Pool, username: str, password: str) -> dict | None:
    user = await get_user(pool, username)
    if user and verify_password(password, user["password_hash"]):
        return user
    return None


async def change_password(
    pool: asyncpg.Pool, username: str, new_password: str
) -> bool:
    hashed = hash_password(new_password)
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE dashboard_users SET password_hash = $2 WHERE username = $1",
            username, hashed,
        )
        return result == "UPDATE 1"


async def seed_default_users(pool: asyncpg.Pool) -> None:
    """Create default users if the table is empty."""
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM dashboard_users")
        if count > 0:
            return

    defaults = [
        ("kody", "owlet_kody", True),
        ("adam", "owlet_adam", False),
        ("vinny", "owlet_vinny", False),
        ("yank", "owlet_yank", False),
    ]
    default_pw = os.getenv("DASHBOARD_DEFAULT_PASSWORD", "changeme123")
    for username, agent_id, is_admin in defaults:
        try:
            await create_user(pool, username, default_pw, agent_id, is_admin)
            logger.info(f"Dashboard: seeded user {username} ({agent_id})")
        except Exception:
            pass  # already exists
