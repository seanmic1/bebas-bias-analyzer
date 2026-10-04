"""
Pydantic schemas mirroring the bb-scraper wire format and bb-think's response.

Source of truth for the input contract: real scraper output sample
(2026-05-26). See memory: project-storythread-contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Inputs — mirror bb-scraper exactly
# ---------------------------------------------------------------------------
class StoryThreadArticle(BaseModel):
    """One article inside a StoryThread."""

    article_id: str
    url: HttpUrl
    publisher: str
    headline: str
    body: str
    author: Optional[str] = None
    published_at: datetime
    scraped_at: datetime
    word_count: int


class StoryThread(BaseModel):
    """A cluster of articles covering the same story.

    Currently single-publisher per thread, but the schema does not enforce
    that — cross-publisher clustering is a planned scraper upgrade.
    """

    thread_id: str
    publisher: str
    representative_headline: str
    keywords: List[str] = Field(default_factory=list)
    article_count: int
    earliest_published_at: datetime
    latest_published_at: datetime
    articles: List[StoryThreadArticle]


class ScrapeRun(BaseModel):
    """Top-level envelope emitted by bb-scraper."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    articles_scraped: int
    articles_failed: int
    threads: List[StoryThread]
    errors: List[Any] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
class AnalyzeThreadRequest(BaseModel):
    """POST /analyze — single thread.

    The caller resolves `publisher_context` (e.g. from Supabase
    ownership_graph) and passes it in. Empty string means "no known context".
    """

    story: StoryThread
    publisher_context: str = ""


class AnalyzeRunRequest(BaseModel):
    """POST /analyze/run — whole scrape run.

    `publisher_contexts` is a mapping `publisher -> context string`. Threads
    whose publisher is missing from the map are analyzed with no context.
    """

    run: ScrapeRun
    publisher_contexts: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
class ClickbaitSignal(BaseModel):
    headline_sentiment: float = Field(..., description="Compound sentiment, [-1, 1]")
    body_sentiment: float = Field(..., description="Compound sentiment, [-1, 1]")
    clickbait_delta: float = Field(
        ...,
        description=(
            "abs(headline_sentiment) - abs(body_sentiment). "
            "Positive => headline more emotionally intense than body."
        ),
    )


class NLPFlags(BaseModel):
    article_id: str
    url: HttpUrl
    clickbait: ClickbaitSignal
    charged_adjectives: List[str] = Field(default_factory=list)
    charged_adjective_count: int = 0


class BiasFingerprint(BaseModel):
    """Five-axis bias fingerprint for an Indonesian news article.

    Each axis runs 0..100; 50 is the neutral baseline. See the system prompt
    for the directional meaning of each pole.
    """

    elite_alignment: int = Field(..., ge=0, le=100)
    identity_lens: int = Field(..., ge=0, le=100)
    economic_sovereignty: int = Field(..., ge=0, le=100)
    state_vs_civil: int = Field(..., ge=0, le=100)
    sensationalism: int = Field(..., ge=0, le=100)


# Axis keys the model is allowed to attribute an evidence snippet to.
BIAS_AXES: List[str] = [
    "elite_alignment",
    "identity_lens",
    "economic_sovereignty",
    "state_vs_civil",
    "sensationalism",
]


class EvidenceSnippet(BaseModel):
    text: str
    axis_affected: str
    explanation: str


class ArticleVerdict(BaseModel):
    article_id: str
    url: HttpUrl
    publisher: str
    bias_fingerprint: BiasFingerprint
    reliability_index: int = Field(..., ge=0, le=100)
    analysis_summary: str
    evidence_snippets: List[EvidenceSnippet] = Field(default_factory=list)


class ThreadAnalysis(BaseModel):
    thread_id: str
    publisher: str
    representative_headline: str
    nlp_flags: List[NLPFlags]
    verdicts: List[ArticleVerdict]


class AnalyzeThreadResponse(ThreadAnalysis):
    """Single-thread response is just a ThreadAnalysis."""


class AnalyzeRunResponse(BaseModel):
    run_id: str
    threads: List[ThreadAnalysis]
