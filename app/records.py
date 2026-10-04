"""
How a ``public.articles`` row becomes the article the engine analyzes.

Shared by the production DB reader (``persistence.py``) and the public backtest
reader (``public_data.py``), so a backtest rebuilds exactly the input production
scored. Pure: rows are anything indexable by key (``dict`` or
``asyncpg.Record``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schemas import StoryThreadArticle


def article_from_row(row: Mapping[str, Any]) -> StoryThreadArticle:
    """Build the engine input for one article row.

    Expected keys: ``article_id``, ``url``, ``publisher``, ``thread_publisher``,
    ``title``, ``body``, ``published_at``, ``scraped_at``. Only ``publisher``,
    ``url``, ``title`` and ``body`` reach the prompt; the rest is metadata.
    """
    body = row["body"] or ""
    return StoryThreadArticle(
        article_id=row["article_id"],
        url=row["url"],
        publisher=row["publisher"] or row["thread_publisher"] or "unknown",
        headline=row["title"],
        body=body,
        author=None,  # bb-scraper does not persist the byline
        published_at=row["published_at"],
        scraped_at=row["scraped_at"] or row["published_at"],
        word_count=len(body.split()),
    )
