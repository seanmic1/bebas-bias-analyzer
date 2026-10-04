"""
Read published verdicts, and the inputs that produced them, from the public site.

The website reads Supabase with its publishable key, under row-level security
that lets anyone SELECT ``public.articles`` and the ``ownership_graph`` view.
This module uses the same REST API and key, so anyone can fetch a verdict the
site shows together with the exact text production scored, and re-run the
engine on it (see ``backtest.py``). Read-only. The publishable key is not a
secret: the site ships it to every browser, and RLS is what enforces access.

Rows are rebuilt into engine inputs through ``records.article_from_row`` and
publisher context through ``publishers.format_context``, the same functions
production uses, so a backtest sees what production saw.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Self

import httpx
from pydantic import ValidationError

from .publishers import format_context
from .records import article_from_row
from .schemas import ArticleVerdict, NLPFlags, StoryThreadArticle

log = logging.getLogger("bebas_bias.public_data")

# The production project behind the website. Override both to point at a
# fork's own Supabase project.
SUPABASE_URL_ENV: Final[str] = "SUPABASE_URL"
SUPABASE_KEY_ENV: Final[str] = "SUPABASE_PUBLISHABLE_KEY"
DEFAULT_SUPABASE_URL: Final[str] = "https://gayllghbwboxylrsznuk.supabase.co"
DEFAULT_PUBLISHABLE_KEY: Final[str] = "sb_publishable_G7XIBEe644OTusymow9M0w_GAt8Fj5U"

# PostgREST caps a response at 1000 rows.
MAX_PAGE: Final[int] = 1000

_ARTICLE_SELECT: Final[str] = ",".join(
    [
        "id",
        "url",
        "title",
        "publisher_name",
        "raw_text",
        "published_at",
        "created_at",
        "status",
        "bias_fingerprint",
        "reliability_index",
        "analysis_summary",
        "evidence_snippets",
        "nlp_flags",
        "analysis_model",
        "analysis_run_id",
        "analyzed_at",
        "rubric_version",
        # Fallback publisher when the article row has none, as in production.
        "story_thread_articles(story_threads(publisher_name))",
    ]
)

_CONTEXT_SELECT: Final[str] = (
    "publisher_name,owner_name,parent_conglomerate,political_affiliation,alignment_warning"
)

_UUID_RE: Final[re.Pattern[str]] = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_SITE_ARTICLE_RE: Final[re.Pattern[str]] = re.compile(
    rf"/article/({_UUID_RE.pattern})", re.IGNORECASE
)


class NotFound(LookupError):
    """No analyzed article matches the reference."""


@dataclass(frozen=True)
class PublishedAnalysis:
    """One verdict as the site shows it, plus the input that produced it."""

    #: The engine input, rebuilt from the row exactly as production does.
    article: StoryThreadArticle
    #: The verdict the site shows.
    verdict: ArticleVerdict
    #: The pre-LLM signals stored with the verdict (None on rows that predate them).
    nlp_flags: NLPFlags | None
    #: Model and rubric revision that produced the verdict (None on old rows).
    model: str | None
    rubric_version: str | None
    analyzed_at: datetime | None
    run_id: str | None


def parse_ref(ref: str) -> tuple[str, str]:
    """Map a user-supplied reference to a ``(column, value)`` filter.

    Accepts the article id (a UUID), a site link containing ``/article/<id>``,
    or the publisher's article URL.
    """
    ref = ref.strip()
    if _UUID_RE.fullmatch(ref):
        return "id", ref.lower()
    site = _SITE_ARTICLE_RE.search(ref)
    if site:
        return "id", site.group(1).lower()
    if ref.startswith(("http://", "https://")):
        return "url", ref
    raise ValueError(
        f"{ref!r} is not an article id, a site article link, or a publisher URL"
    )


def _thread_publisher(row: dict[str, Any]) -> str | None:
    for link in row.get("story_thread_articles") or []:
        thread = link.get("story_threads") or {}
        if thread.get("publisher_name"):
            return thread["publisher_name"]
    return None


def _timestamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def analysis_from_row(row: dict[str, Any]) -> PublishedAnalysis:
    """Turn one REST row into a :class:`PublishedAnalysis`. Pure."""
    article = article_from_row(
        {
            "article_id": row["id"],
            "url": row["url"],
            "publisher": row.get("publisher_name"),
            "thread_publisher": _thread_publisher(row),
            "title": row["title"],
            "body": row["raw_text"],
            "published_at": row["published_at"],
            "scraped_at": row.get("created_at"),
        }
    )
    verdict = ArticleVerdict(
        article_id=article.article_id,
        url=article.url,
        publisher=article.publisher,
        bias_fingerprint=row["bias_fingerprint"],
        reliability_index=row["reliability_index"],
        analysis_summary=row.get("analysis_summary") or "",
        evidence_snippets=row.get("evidence_snippets") or [],
    )
    flags: NLPFlags | None = None
    if row.get("nlp_flags"):
        try:
            flags = NLPFlags.model_validate(row["nlp_flags"])
        except ValidationError:
            log.warning("stored nlp_flags for %s have an older shape; ignored", row["id"])
    return PublishedAnalysis(
        article=article,
        verdict=verdict,
        nlp_flags=flags,
        model=row.get("analysis_model"),
        rubric_version=row.get("rubric_version"),
        analyzed_at=_timestamp(row.get("analyzed_at")),
        run_id=row.get("analysis_run_id"),
    )


class PublicSite:
    """Read-only client for the site's public data.

    Also satisfies the publisher-context interface of
    ``publishers.PublisherContextStore`` (``configured`` + ``context_for``), so
    the counterfactual audit can run against it without database access.
    """

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        base = (url or os.environ.get(SUPABASE_URL_ENV) or DEFAULT_SUPABASE_URL).rstrip("/")
        key = key or os.environ.get(SUPABASE_KEY_ENV) or DEFAULT_PUBLISHABLE_KEY
        self.base_url = base
        self._http = httpx.AsyncClient(
            base_url=f"{base}/rest/v1",
            headers={"apikey": key, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )
        self._contexts: dict[str, str] = {}

    @property
    def configured(self) -> bool:
        return True

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def _get(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        response = await self._http.get(path, params=params)
        response.raise_for_status()
        return response.json()

    async def analysis(self, ref: str) -> PublishedAnalysis:
        """The analyzed article ``ref`` points at. Raises :class:`NotFound`."""
        column, value = parse_ref(ref)
        rows = await self._get(
            "/articles",
            {"select": _ARTICLE_SELECT, column: f"eq.{value}", "limit": "1"},
        )
        if not rows:
            raise NotFound(f"no article matches {ref!r}")
        row = rows[0]
        if row.get("status") != "analyzed" or row.get("bias_fingerprint") is None:
            raise NotFound(f"article {row['id']} has no published verdict (status={row.get('status')!r})")
        return analysis_from_row(row)

    async def latest(
        self,
        *,
        limit: int = 20,
        publisher: str | None = None,
        since: datetime | None = None,
        rubric_version: str | None = None,
        model: str | None = None,
    ) -> list[PublishedAnalysis]:
        """The most recently analyzed articles, newest first, optionally filtered."""
        if not 1 <= limit <= MAX_PAGE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE}")
        params = {
            "select": _ARTICLE_SELECT,
            "status": "eq.analyzed",
            "bias_fingerprint": "not.is.null",
            # The same rows production would analyze at all.
            "title": "not.is.null",
            "raw_text": "not.is.null",
            "published_at": "not.is.null",
            "order": "analyzed_at.desc.nullslast",
            "limit": str(limit),
        }
        if publisher:
            params["publisher_name"] = f"eq.{publisher}"
        if since:
            params["analyzed_at"] = f"gte.{since.isoformat()}"
        if rubric_version:
            params["rubric_version"] = f"eq.{rubric_version}"
        if model:
            params["analysis_model"] = f"eq.{model}"
        return [analysis_from_row(r) for r in await self._get("/articles", params)]

    async def context_for(self, domain: str) -> str:
        """Publisher context for ``domain``, rendered as production renders it.

        ``""`` for an unknown domain, as in production. Unlike production, a
        failed lookup raises: a backtest that silently drops the context is not
        reproducing anything.
        """
        if not domain:
            return ""
        if domain not in self._contexts:
            rows = await self._get(
                "/ownership_graph",
                {"select": _CONTEXT_SELECT, "domain": f"eq.{domain}", "limit": "1"},
            )
            self._contexts[domain] = format_context(rows[0]) if rows else ""
        return self._contexts[domain]
