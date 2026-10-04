"""build_chat_request picks token/reasoning kwargs by model family."""

from __future__ import annotations

from datetime import datetime, timezone

from app.llm import OpenAIClient
from app.schemas import ClickbaitSignal, NLPFlags, StoryThreadArticle

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _article(aid: str) -> StoryThreadArticle:
    return StoryThreadArticle(
        article_id=aid,
        url=f"https://example.com/{aid}",
        publisher="example.com",
        headline=f"Headline {aid}",
        body="Isi artikel untuk pengujian.",
        published_at=_NOW,
        scraped_at=_NOW,
        word_count=4,
    )


def _fake_flags(aid: str) -> NLPFlags:
    return NLPFlags(
        article_id=aid,
        url=f"https://example.com/{aid}",
        clickbait=ClickbaitSignal(
            headline_sentiment=0.0, body_sentiment=0.0, clickbait_delta=0.0
        ),
    )


def _body(model: str, **kwargs) -> dict:
    client = OpenAIClient(model=model, api_key="test-key-not-used", **kwargs)
    return client.build_chat_request(_article("a1"), _fake_flags("a1"), "")


def test_reasoning_model_gets_completion_tokens_and_effort(monkeypatch) -> None:
    monkeypatch.delenv("BEBAS_BIAS_REASONING_EFFORT", raising=False)
    body = _body("gpt-6-luna")
    assert body["max_completion_tokens"] == 2048
    assert body["reasoning_effort"] == "low"
    assert "max_tokens" not in body


def test_reasoning_effort_env_override(monkeypatch) -> None:
    monkeypatch.setenv("BEBAS_BIAS_REASONING_EFFORT", "medium")
    assert _body("gpt-5.4-nano")["reasoning_effort"] == "medium"


def test_non_reasoning_model_omits_reasoning_effort() -> None:
    body = _body("gpt-4o")
    assert body["max_tokens"] == 2048
    assert "reasoning_effort" not in body
    assert "max_completion_tokens" not in body
