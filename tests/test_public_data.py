"""Unit tests for the public-site reader (app/public_data.py).

No network: PublicSite is driven through an httpx.MockTransport that records
each request, so the tests pin both what we ask PostgREST for and how a row
becomes the engine input production scored.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.public_data import NotFound, PublicSite, analysis_from_row, parse_ref

_ID = "88741137-8d33-4833-831c-87d42793c970"


def _row(**kw) -> dict:
    return {
        "id": _ID,
        "url": "https://news.example.com/berita/1",
        "title": "Judul berita",
        "publisher_name": "example.com",
        "raw_text": "Isi berita  yang  cukup panjang.",
        "published_at": "2026-10-04T08:32:37+00:00",
        "created_at": "2026-10-04T09:04:27+00:00",
        "status": "analyzed",
        "bias_fingerprint": {
            "elite_alignment": 75,
            "identity_lens": 50,
            "economic_sovereignty": 50,
            "state_vs_civil": 85,
            "sensationalism": 30,
        },
        "reliability_index": 62,
        "analysis_summary": "Ringkasan.",
        "evidence_snippets": [
            {"text": "kutipan", "axis_affected": "state_vs_civil", "explanation": "alasan"}
        ],
        "nlp_flags": {
            "article_id": _ID,
            "url": "https://news.example.com/berita/1",
            "clickbait": {"headline_sentiment": 0.0, "body_sentiment": 0.5, "clickbait_delta": -0.5},
            "charged_adjectives": ["fantastis"],
            "charged_adjective_count": 1,
        },
        "analysis_model": "gpt-6-luna",
        "analysis_run_id": "11111111-1111-1111-1111-111111111111",
        "analyzed_at": "2026-10-04T09:06:07+00:00",
        "rubric_version": "2026-06-14",
        "story_thread_articles": [{"story_threads": {"publisher_name": "thread.example.com"}}],
        **kw,
    }


class _Recorder:
    """MockTransport handler: answers by path, remembers every request."""

    def __init__(self, responses: dict[str, object], status: int = 200) -> None:
        self.responses = responses
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = self.responses.get(request.url.path.rsplit("/", 1)[-1], [])
        return httpx.Response(self.status, content=json.dumps(body))


def _site(recorder: _Recorder) -> PublicSite:
    return PublicSite(
        url="https://proj.supabase.co", key="sb_publishable_test",
        transport=httpx.MockTransport(recorder),
    )


async def _with(site: PublicSite, coro_fn):
    async with site:
        return await coro_fn(site)


# -- parse_ref ---------------------------------------------------------------
def test_parse_ref_accepts_bare_id_site_link_and_publisher_url() -> None:
    assert parse_ref(_ID.upper()) == ("id", _ID)
    assert parse_ref(f"https://site.example/article/{_ID}?tab=evidence") == ("id", _ID)
    assert parse_ref("https://news.example.com/berita/1") == (
        "url", "https://news.example.com/berita/1",
    )


def test_parse_ref_rejects_anything_else() -> None:
    with pytest.raises(ValueError):
        parse_ref("not-an-article")


# -- analysis_from_row -------------------------------------------------------
def test_row_becomes_the_input_production_scored() -> None:
    p = analysis_from_row(_row())
    a = p.article
    assert a.article_id == _ID
    assert a.publisher == "example.com"
    assert a.headline == "Judul berita"
    assert a.body == "Isi berita  yang  cukup panjang."
    assert a.word_count == 5
    assert p.verdict.bias_fingerprint.state_vs_civil == 85
    assert p.verdict.reliability_index == 62
    assert p.verdict.evidence_snippets[0].axis_affected == "state_vs_civil"
    assert p.nlp_flags is not None and p.nlp_flags.charged_adjectives == ["fantastis"]
    assert (p.model, p.rubric_version) == ("gpt-6-luna", "2026-06-14")
    assert p.analyzed_at is not None and p.analyzed_at.tzinfo is not None


def test_missing_publisher_falls_back_to_the_thread_publisher() -> None:
    assert analysis_from_row(_row(publisher_name=None)).article.publisher == "thread.example.com"
    bare = _row(publisher_name=None, story_thread_articles=[])
    assert analysis_from_row(bare).article.publisher == "unknown"


def test_old_nlp_flag_shape_is_ignored_not_fatal() -> None:
    p = analysis_from_row(_row(nlp_flags={"clickbait_delta": 0.1}))
    assert p.nlp_flags is None
    assert analysis_from_row(_row(nlp_flags=None)).nlp_flags is None


# -- PublicSite --------------------------------------------------------------
def test_analysis_queries_by_id_with_the_publishable_key() -> None:
    rec = _Recorder({"articles": [_row()]})
    p = asyncio.run(_with(_site(rec), lambda s: s.analysis(_ID)))
    assert p.article.article_id == _ID
    req = rec.requests[0]
    assert req.url.path == "/rest/v1/articles"
    assert req.url.params["id"] == f"eq.{_ID}"
    assert req.headers["apikey"] == "sb_publishable_test"


def test_analysis_by_publisher_url_filters_on_url() -> None:
    rec = _Recorder({"articles": [_row()]})
    asyncio.run(_with(_site(rec), lambda s: s.analysis("https://news.example.com/berita/1")))
    assert rec.requests[0].url.params["url"] == "eq.https://news.example.com/berita/1"


def test_unknown_or_unanalyzed_article_is_not_found() -> None:
    with pytest.raises(NotFound):
        asyncio.run(_with(_site(_Recorder({"articles": []})), lambda s: s.analysis(_ID)))
    pending = _Recorder({"articles": [_row(status="pending", bias_fingerprint=None)]})
    with pytest.raises(NotFound):
        asyncio.run(_with(_site(pending), lambda s: s.analysis(_ID)))


def test_latest_passes_filters_and_only_asks_for_scorable_rows() -> None:
    rec = _Recorder({"articles": [_row(), _row(id="22222222-2222-2222-2222-222222222222")]})
    found = asyncio.run(
        _with(_site(rec), lambda s: s.latest(limit=2, publisher="detik.com", rubric_version="r1"))
    )
    assert len(found) == 2
    params = rec.requests[0].url.params
    assert params["status"] == "eq.analyzed"
    assert params["publisher_name"] == "eq.detik.com"
    assert params["rubric_version"] == "eq.r1"
    assert params["raw_text"] == "not.is.null"
    assert params["limit"] == "2"
    assert "analysis_model" not in params


def test_latest_rejects_out_of_range_limits() -> None:
    with pytest.raises(ValueError):
        asyncio.run(_with(_site(_Recorder({})), lambda s: s.latest(limit=0)))


def test_context_is_rendered_like_production_and_cached() -> None:
    rec = _Recorder({"ownership_graph": [{
        "publisher_name": "Detik",
        "owner_name": "Chairul Tanjung",
        "parent_conglomerate": "Trans Media",
        "political_affiliation": "Pro-Establishment",
        "alignment_warning": None,
    }]})

    async def twice(site: PublicSite) -> list[str]:
        return [await site.context_for("detik.com"), await site.context_for("detik.com")]

    first, second = asyncio.run(_with(_site(rec), twice))
    assert first == (
        "Detik is owned by Chairul Tanjung (parent: Trans Media). "
        "Known political alignment: Pro-Establishment."
    )
    assert second == first
    assert len(rec.requests) == 1
    assert rec.requests[0].url.params["domain"] == "eq.detik.com"


def test_unknown_domain_has_empty_context_but_errors_raise() -> None:
    empty = asyncio.run(
        _with(_site(_Recorder({"ownership_graph": []})), lambda s: s.context_for("x.com"))
    )
    assert empty == ""
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_with(_site(_Recorder({}, status=503)), lambda s: s.context_for("x.com")))
