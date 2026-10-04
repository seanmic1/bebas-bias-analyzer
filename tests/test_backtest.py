"""Unit tests for the backtest (app/backtest.py).

No network and no NLTK data: the LLM client is a fake that returns scripted
scores, publisher context is a dict, and analyze_article is monkeypatched.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app import backtest as bt_mod
from app.backtest import Backtester, check_inputs, compare, render_markdown
from app.prompt import RUBRIC_VERSION
from app.public_data import PublishedAnalysis
from app.schemas import (
    ArticleVerdict,
    BiasFingerprint,
    ClickbaitSignal,
    NLPFlags,
    StoryThreadArticle,
)

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_AXES = {
    "elite_alignment": 50,
    "identity_lens": 50,
    "economic_sovereignty": 50,
    "state_vs_civil": 50,
    "sensationalism": 10,
}


def _flags(aid: str = "a1", *, delta: float = 0.0, charged: list[str] | None = None) -> NLPFlags:
    return NLPFlags(
        article_id=aid,
        url=f"https://example.com/{aid}",
        clickbait=ClickbaitSignal(headline_sentiment=0.0, body_sentiment=0.0, clickbait_delta=delta),
        charged_adjectives=charged or [],
        charged_adjective_count=len(charged or []),
    )


def _verdict(aid: str, reliability: int = 70, **axes: int) -> ArticleVerdict:
    return ArticleVerdict(
        article_id=aid,
        url=f"https://example.com/{aid}",
        publisher="example.com",
        bias_fingerprint=BiasFingerprint(**{**_AXES, **axes}),
        reliability_index=reliability,
        analysis_summary="",
    )


def _published(
    aid: str = "a1",
    *,
    model: str | None = "gpt-6-luna",
    rubric: str | None = RUBRIC_VERSION,
    flags: NLPFlags | None = None,
    **axes: int,
) -> PublishedAnalysis:
    article = StoryThreadArticle(
        article_id=aid,
        url=f"https://example.com/{aid}",
        publisher="example.com",
        headline="Judul",
        body="Isi.",
        published_at=_NOW,
        scraped_at=_NOW,
        word_count=1,
    )
    return PublishedAnalysis(
        article=article,
        verdict=_verdict(aid, **axes),
        nlp_flags=flags if flags is not None else _flags(aid),
        model=model,
        rubric_version=rubric,
        analyzed_at=_NOW,
        run_id=None,
    )


class _FakeClient:
    """Returns scripted state_vs_civil scores in order; other axes neutral."""

    def __init__(self, model: str, scores: list[int]) -> None:
        self.model = model
        self._scores = list(scores)
        self.calls: list[str] = []

    def analyze(self, article, nlp, publisher_context) -> ArticleVerdict:
        self.calls.append(publisher_context)
        return _verdict(article.article_id, state_vs_civil=self._scores.pop(0))


class _Contexts:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def context_for(self, domain: str) -> str:
        if self.fail:
            raise RuntimeError("site down")
        return f"context for {domain}"


@pytest.fixture(autouse=True)
def _no_nltk(monkeypatch):
    monkeypatch.setattr(bt_mod, "analyze_article", lambda a: _flags(a.article_id))


# -- compare -----------------------------------------------------------------
def test_compare_flags_only_deltas_beyond_noise_and_tolerance() -> None:
    steady = compare("state_vs_civil", 50, [70, 70, 70])
    assert (steady.rerun_mean, steady.delta, steady.noise, steady.flagged) == (70, 20, 0, True)

    small = compare("state_vs_civil", 50, [53, 53, 53])
    assert small.flagged is False  # within TOLERANCE even with zero noise

    noisy = compare("state_vs_civil", 50, [40, 80, 60])
    # delta 10, noise 26.67 -> well inside the scorer's own spread
    assert noisy.delta == 10 and noisy.flagged is False


# -- check_inputs ------------------------------------------------------------
def test_check_inputs_matches_on_signals_not_bookkeeping() -> None:
    stored = _flags("legacy-in-memory-id")  # article_id differs, signals equal
    check = check_inputs(_published(flags=stored), _flags("a1"))
    assert check.rubric_match is True
    assert check.nlp_match is True and check.nlp_diff == {}


def test_check_inputs_reports_differing_signals_and_rubric() -> None:
    p = _published(rubric="2020-01-01", flags=_flags(delta=0.5, charged=["heboh"]))
    check = check_inputs(p, _flags())
    assert check.rubric_match is False
    assert check.nlp_match is False
    assert check.nlp_diff["clickbait_delta"] == (0.5, 0.0)
    assert check.nlp_diff["charged_adjectives"] == (["heboh"], [])


def test_check_inputs_without_stored_flags_is_unknown() -> None:
    p = _published()
    p = PublishedAnalysis(**{**p.__dict__, "nlp_flags": None})
    assert check_inputs(p, _flags()).nlp_match is None


# -- Backtester --------------------------------------------------------------
def _backtester(clients: dict, **kw) -> Backtester:
    return Backtester(client_for=lambda m: clients[m], publishers=_Contexts(), **kw)


def test_rescores_with_the_published_model_and_compares() -> None:
    fake = _FakeClient("gpt-6-luna", [80, 80, 80])
    bt = _backtester({"gpt-6-luna": fake}, repeats=3)
    report = asyncio.run(bt.run([_published(state_vs_civil=50)]))

    assert fake.calls == ["context for example.com"] * 3
    [result] = report["results"]
    assert result["rerun_model"] == "gpt-6-luna"
    assert result["flagged"] == ["state_vs_civil"]
    by_metric = {c["metric"]: c for c in result["comparisons"]}
    assert by_metric["state_vs_civil"]["delta"] == 30
    assert by_metric["reliability_index"]["flagged"] is False
    assert report["articles_flagged"] == 1
    overall = {a["metric"]: a for a in report["overall"]}
    assert overall["state_vs_civil"]["mean_delta"] == 30
    assert overall["state_vs_civil"]["flagged"] == 1


def test_model_override_replaces_the_published_model() -> None:
    fake = _FakeClient("other-model", [50, 50])
    bt = _backtester({"other-model": fake}, model="other-model", repeats=2)
    report = asyncio.run(bt.run([_published(model="gpt-6-luna")]))
    assert report["results"][0]["rerun_model"] == "other-model"
    assert report["results"][0]["published_model"] == "gpt-6-luna"


def test_one_failing_article_does_not_sink_the_run() -> None:
    bt = Backtester(
        client_for=lambda m: _FakeClient("gpt-6-luna", [50, 50]),
        publishers=_Contexts(fail=True),
        repeats=2,
    )
    report = asyncio.run(bt.run([_published()]))
    assert report["articles_failed"] == 1
    assert "site down" in report["results"][0]["error"]
    assert report["overall"] == []


def test_single_repeat_is_rejected() -> None:
    with pytest.raises(ValueError):
        _backtester({}, repeats=1)


def test_report_renders_and_writes(tmp_path) -> None:
    fake = _FakeClient("gpt-6-luna", [80, 80])
    bt = _backtester({"gpt-6-luna": fake}, repeats=2, output_dir=tmp_path)
    report = asyncio.run(bt.run([_published(rubric="2020-01-01")]))
    md = render_markdown(report)
    assert "published under rubric `2020-01-01`" in md
    assert "| `state_vs_civil` | 50 | 80, 80 |" in md
    json_path, md_path = bt.write(report)
    assert json_path.exists() and md_path.read_text(encoding="utf-8") == md
