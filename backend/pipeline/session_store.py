"""
backend/pipeline/session_store.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Session Store
Uses Redis in production; falls back to in-memory dict for
local development when Redis is unavailable.
─────────────────────────────────────────────────────────
"""

import json
import asyncio
from typing import Optional
from loguru import logger

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


SESSION_TTL_SECONDS = 1800   # 30 minutes


class SessionStore:
    """Async key-value store for KYC session state."""

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._local: dict = {}          # fallback
        self._use_redis = False

    async def connect(self, redis_url: str):
        if not REDIS_AVAILABLE:
            logger.warning("[Store] redis-py not installed — using in-memory store")
            return
        try:
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            await self._redis.ping()
            self._use_redis = True
            logger.info("[Store] Connected to Redis")
        except Exception as e:
            logger.warning(f"[Store] Redis unavailable ({e}) — using in-memory store")
            self._use_redis = False

    async def set(self, session_id: str, data: dict):
        if self._use_redis:
            await self._redis.setex(session_id, SESSION_TTL_SECONDS, json.dumps(data, default=str))
        else:
            self._local[session_id] = data

    async def get(self, session_id: str) -> Optional[dict]:
        if self._use_redis:
            raw = await self._redis.get(session_id)
            return json.loads(raw) if raw else None
        return self._local.get(session_id)

    async def delete(self, session_id: str):
        if self._use_redis:
            await self._redis.delete(session_id)
        else:
            self._local.pop(session_id, None)

    async def close(self):
        if self._redis:
            await self._redis.close()


# ── Global singleton ──────────────────────────────────────────────────────────
session_store = SessionStore()
