"""Unit tests for the publisher_context resolver.

We don't exercise asyncpg here — the goal is to lock down the formatting
contract (what string the LLM actually sees) and the caching/empty-DSN
fallback behavior.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.publishers import PublisherContextStore, format_context


def _row(**kw) -> dict[str, str | None]:
    return {
        "publisher_name": None,
        "owner_name": None,
        "parent_conglomerate": None,
        "political_affiliation": None,
        "alignment_warning": None,
        **kw,
    }


def test_format_full_row() -> None:
    row = _row(
        publisher_name="Acme News",
        owner_name="Jane Doe",
        parent_conglomerate="Globex Media",
        political_affiliation="Pro-Establishment",
    )
    out = format_context(row)
    assert "Acme News is owned by Jane Doe (parent: Globex Media)." in out
    assert "Known political alignment: Pro-Establishment." in out
    assert "Alignment warning" not in out  # no warning -> not included


def test_format_includes_warning_when_present() -> None:
    row = _row(
        publisher_name="Initech Daily",
        parent_conglomerate="Initech Group",
        political_affiliation="Editorially Independent",
        alignment_warning="Heavy sponsored content.",
    )
    out = format_context(row)
    assert out.startswith("Initech Daily is part of the Initech Group conglomerate.")
    assert "Alignment warning: Heavy sponsored content." in out


def test_format_handles_owner_only() -> None:
    row = _row(publisher_name="Umbrella Post", owner_name="Umbrella Post Group")
    out = format_context(row)
    assert out == "Umbrella Post is owned by Umbrella Post Group."


def test_format_returns_empty_for_no_publisher() -> None:
    # Without a publisher name there's nothing meaningful to anchor the
    # context paragraph on; the LLM should just get an empty string.
    assert format_context(_row()) == ""
    assert format_context(None) == ""


def test_context_for_returns_empty_when_dsn_missing() -> None:
    store = PublisherContextStore(dsn=None)
    assert store.configured is False
    out = asyncio.run(store.context_for("example.com"))
    assert out == ""


def test_context_for_caches_unknown_domain() -> None:
    """Second lookup must not retry the DB after the first miss."""
    store = PublisherContextStore(dsn=None)
    asyncio.run(store.context_for("example.com"))
    assert store._cache["example.com"] == ""


def test_context_for_returns_empty_for_blank_domain() -> None:
    store = PublisherContextStore(
        dsn="postgresql://nope:nope@127.0.0.1:1/none"
    )
    out = asyncio.run(store.context_for(""))
    assert out == ""
