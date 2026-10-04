"""
Resolve `publisher_context` strings from Supabase's `public.ownership_graph`
view.

The view exposes one flat row per domain with the publisher's owner, parent
conglomerate, current political affiliation, and an optional alignment
warning. We turn that into a short, prompt-shaped paragraph the LLM can ground
its bias verdict on.

Contracts:
    - Input keys are domain strings as bb-scraper emits them (``detik.com``,
      ``kompas.com``, ``tribunnews.com``).
    - A missing or unknown domain returns an empty string, matching the
      "no known context" convention already used by the API.
    - Lookups are cached in-process forever; the ownership graph changes on
      a human cadence, not a per-request one.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import Any, Final, Optional

import asyncpg

log = logging.getLogger(__name__)

DATABASE_URL_ENV: Final[str] = "DATABASE_URL"

_QUERY: Final[str] = """
    SELECT publisher_name,
           owner_name,
           parent_conglomerate,
           political_affiliation,
           alignment_warning
    FROM public.ownership_graph
    WHERE domain = $1
    LIMIT 1
"""


class PublisherContextStore:
    """Caching, async lookup of publisher_context strings keyed by domain.

    Connection pool + cache are created lazily so the API stays up when the
    DB is unreachable — every lookup just falls back to an empty context.
    """

    def __init__(self, dsn: str | None = None):
        self._dsn = dsn or os.environ.get(DATABASE_URL_ENV)
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()
        self._cache: dict[str, str] = {}

    @property
    def configured(self) -> bool:
        return bool(self._dsn)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def context_for(self, domain: str) -> str:
        """Return a context paragraph for ``domain``, or ``""`` if unknown.

        Failures (DB unreachable, malformed DSN, missing row) are logged
        once and yield an empty context — bias analysis still proceeds.
        """
        if not domain:
            return ""
        if domain in self._cache:
            return self._cache[domain]
        if not self._dsn:
            self._cache[domain] = ""
            return ""

        pool = await self._ensure_pool()
        if pool is None:
            self._cache[domain] = ""
            return ""

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(_QUERY, domain)
        except Exception as exc:
            log.warning("ownership_graph lookup failed for %s: %s", domain, exc)
            self._cache[domain] = ""
            return ""

        context = format_context(row) if row else ""
        self._cache[domain] = context
        return context

    async def _ensure_pool(self) -> asyncpg.Pool | None:
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is not None:
                return self._pool
            try:
                # statement_cache_size=0 — Supabase pooler is pgbouncer in
                # transaction mode, which doesn't support prepared statements.
                self._pool = await asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=0,
                    max_size=3,
                    statement_cache_size=0,
                )
            except Exception as exc:
                log.warning("could not create ownership_graph pool: %s", exc)
                return None
            return self._pool


def format_context(row: Optional[Mapping[str, Any]]) -> str:
    """Render an ``ownership_graph`` row into a compact paragraph for the LLM
    prompt. Takes an ``asyncpg.Record`` or a plain dict (the public backtest
    reads the same view over REST)."""
    if row is None:
        return ""

    publisher = (row["publisher_name"] or "").strip()
    owner = (row["owner_name"] or "").strip()
    conglomerate = (row["parent_conglomerate"] or "").strip()
    affiliation = (row["political_affiliation"] or "").strip()
    warning = (row["alignment_warning"] or "").strip()

    if not publisher:
        return ""

    parts: list[str] = []
    if owner and conglomerate:
        parts.append(f"{publisher} is owned by {owner} (parent: {conglomerate}).")
    elif conglomerate:
        parts.append(f"{publisher} is part of the {conglomerate} conglomerate.")
    elif owner:
        parts.append(f"{publisher} is owned by {owner}.")
    else:
        parts.append(f"{publisher} has no recorded ownership in the graph.")

    if affiliation:
        parts.append(f"Known political alignment: {affiliation}.")
    if warning:
        parts.append(f"Alignment warning: {warning}.")

    return " ".join(parts)
