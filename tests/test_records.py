"""article_from_row is the one mapping from an articles row to engine input."""

from __future__ import annotations

from datetime import datetime, timezone

from app.records import article_from_row

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _row(**kw) -> dict:
    return {
        "article_id": "a1",
        "url": "https://example.com/a1",
        "publisher": "example.com",
        "thread_publisher": "thread.example.com",
        "title": "Judul",
        "body": "satu dua  tiga",
        "published_at": _NOW,
        "scraped_at": None,
        **kw,
    }


def test_maps_prompt_fields_and_derives_metadata() -> None:
    a = article_from_row(_row())
    assert (a.publisher, a.headline, a.body) == ("example.com", "Judul", "satu dua  tiga")
    assert a.word_count == 3
    assert a.scraped_at == _NOW  # falls back to published_at
    assert a.author is None


def test_publisher_falls_back_to_thread_then_unknown() -> None:
    assert article_from_row(_row(publisher=None)).publisher == "thread.example.com"
    assert article_from_row(_row(publisher=None, thread_publisher=None)).publisher == "unknown"


def test_missing_body_is_empty() -> None:
    a = article_from_row(_row(body=None))
    assert a.body == "" and a.word_count == 0
