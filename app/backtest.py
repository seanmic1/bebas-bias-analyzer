"""
Backtest published verdicts: re-run the engine on the exact input production
scored, and compare.

Production scores each article once, so a published number is a single draw
from a nondeterministic scorer. Re-scoring the same input ``repeats`` times
shows whether that number is what the engine says about the text, or an outlier.
Each metric is judged against the scorer's own spread, the same effect-vs-noise
reading the counterfactual audit uses (``audit/metrics.py``):

  ``published``   the score the site shows.
  ``rerun_mean``  mean of the re-scorings.
  ``delta``       rerun_mean - published. Signed: averaged over many articles
                  it shows drift, e.g. the current engine scoring
                  sensationalism lower than the published verdicts did.
  ``noise``       mean pairwise spread among the re-scorings.
  ``flagged``     |delta| > FLAG_RATIO x noise and |delta| > TOLERANCE.

The input goes through the production path (``nlp.analyze_article`` ->
``LLMClient.analyze`` -> ``build_chat_request`` -> ``_parse_verdict``), and the
recomputed NLP signals are checked against the stored ones before any model
call, so a different text or a changed NLP pass is caught for free.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .audit.metrics import AUDIT_METRICS, FLAG_RATIO, mean_pairwise_abs
from .llm import LLMClient
from .nlp import analyze_article
from .prompt import RUBRIC_VERSION
from .public_data import PublishedAnalysis
from .schemas import ArticleVerdict, NLPFlags

log = logging.getLogger("bebas_bias.backtest")

DEFAULT_REPEATS = 3
DEFAULT_CONCURRENCY = 4

#: Deltas this small (points on the 0-100 scale) are never flagged. With a few
#: repeats the noise estimate is often 0, and a 3-point move is not something
#: anyone reads into a score.
TOLERANCE = 5


class ContextSource(Protocol):
    async def context_for(self, domain: str) -> str: ...


def verdict_scores(verdict: ArticleVerdict) -> dict[str, int]:
    """The five axes plus reliability, keyed as in ``AUDIT_METRICS``."""
    scores = verdict.bias_fingerprint.model_dump()
    scores["reliability_index"] = verdict.reliability_index
    return scores


# ---------------------------------------------------------------------------
# Input check (free)
# ---------------------------------------------------------------------------
@dataclass
class InputCheck:
    """Whether today's engine sees the input production saw."""

    #: The stored rubric revision equals this engine's RUBRIC_VERSION.
    rubric_match: bool
    #: Recomputed NLP signals equal the stored ones. None when nothing is stored.
    nlp_match: bool | None
    #: Field -> (stored, recomputed) for every signal that differs.
    nlp_diff: dict[str, tuple[Any, Any]] = field(default_factory=dict)


def _signals(flags: NLPFlags) -> dict[str, Any]:
    # article_id/url are bookkeeping (rows analyzed through the HTTP API carry
    # the scraper's in-memory id), so only the signals the prompt sees count.
    return {
        "headline_sentiment": flags.clickbait.headline_sentiment,
        "body_sentiment": flags.clickbait.body_sentiment,
        "clickbait_delta": flags.clickbait.clickbait_delta,
        "charged_adjectives": sorted(flags.charged_adjectives),
    }


def check_inputs(published: PublishedAnalysis, flags: NLPFlags | None = None) -> InputCheck:
    """Compare the stored NLP signals and rubric revision with today's engine."""
    rubric_match = published.rubric_version == RUBRIC_VERSION
    if published.nlp_flags is None:
        return InputCheck(rubric_match=rubric_match, nlp_match=None)
    stored = _signals(published.nlp_flags)
    fresh = _signals(flags or analyze_article(published.article))
    diff = {k: (stored[k], fresh[k]) for k in stored if stored[k] != fresh[k]}
    return InputCheck(rubric_match=rubric_match, nlp_match=not diff, nlp_diff=diff)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
@dataclass
class MetricComparison:
    metric: str
    published: int
    reruns: list[int]
    rerun_mean: float
    delta: float
    noise: float
    flagged: bool


def compare(metric: str, published: int, reruns: list[int]) -> MetricComparison:
    """Judge one published score against its re-scorings."""
    mean = statistics.fmean(reruns)
    delta = mean - published
    noise = mean_pairwise_abs([float(r) for r in reruns])
    return MetricComparison(
        metric=metric,
        published=published,
        reruns=list(reruns),
        rerun_mean=round(mean, 2),
        delta=round(delta, 2),
        noise=round(noise, 2),
        flagged=abs(delta) > max(FLAG_RATIO * noise, TOLERANCE),
    )


@dataclass
class ArticleBacktest:
    article_id: str
    url: str
    publisher: str
    published_model: str | None
    rerun_model: str
    published_rubric: str | None
    analyzed_at: str | None
    publisher_context: str
    inputs: InputCheck
    comparisons: list[MetricComparison] = field(default_factory=list)
    error: str | None = None

    @property
    def flagged(self) -> list[str]:
        return [c.metric for c in self.comparisons if c.flagged]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["flagged"] = self.flagged
        return out


@dataclass
class AggregateDrift:
    """Per-metric rollup across every article backtested."""

    metric: str
    n: int
    mean_delta: float
    mean_abs_delta: float
    mean_noise: float
    flagged: int


def aggregate(results: list[ArticleBacktest]) -> list[AggregateDrift]:
    by_metric: dict[str, list[MetricComparison]] = {}
    for r in results:
        for c in r.comparisons:
            by_metric.setdefault(c.metric, []).append(c)
    out: list[AggregateDrift] = []
    for metric in AUDIT_METRICS:
        rows = by_metric.get(metric)
        if not rows:
            continue
        out.append(
            AggregateDrift(
                metric=metric,
                n=len(rows),
                mean_delta=round(statistics.fmean(c.delta for c in rows), 2),
                mean_abs_delta=round(statistics.fmean(abs(c.delta) for c in rows), 2),
                mean_noise=round(statistics.fmean(c.noise for c in rows), 2),
                flagged=sum(c.flagged for c in rows),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class Backtester:
    """Re-scores published verdicts through the production path."""

    def __init__(
        self,
        client_for: Callable[[str | None], LLMClient],
        publishers: ContextSource,
        *,
        model: str | None = None,
        repeats: int = DEFAULT_REPEATS,
        concurrency: int = DEFAULT_CONCURRENCY,
        output_dir: Path | None = None,
    ) -> None:
        """``client_for(model)`` returns the client to re-score with; ``None``
        means the default model. Unless ``model`` is given, each article is
        re-scored with the model that produced its published verdict."""
        if repeats < 2:
            raise ValueError(
                "repeats must be >= 2: a single re-scoring gives no noise floor, "
                "so a difference from the published score can't be read."
            )
        self._client_for = client_for
        self._clients: dict[str | None, LLMClient] = {}
        self.publishers = publishers
        self.model = model
        self.repeats = repeats
        self.output_dir = output_dir or Path(__file__).resolve().parents[1] / "outputs"
        self._sem = asyncio.Semaphore(concurrency)

    def client(self, published_model: str | None) -> LLMClient:
        key = self.model or published_model
        if key not in self._clients:
            self._clients[key] = self._client_for(key)
        return self._clients[key]

    async def _score_once(self, client: LLMClient, published: PublishedAnalysis, flags: NLPFlags, context: str) -> dict[str, int]:
        async with self._sem:
            verdict = await asyncio.to_thread(client.analyze, published.article, flags, context)
        return verdict_scores(verdict)

    async def backtest(self, published: PublishedAnalysis) -> ArticleBacktest:
        article = published.article
        client = self.client(published.model)
        flags = analyze_article(article)
        result = ArticleBacktest(
            article_id=article.article_id,
            url=str(article.url),
            publisher=article.publisher,
            published_model=published.model,
            rerun_model=getattr(client, "model", "unknown"),
            published_rubric=published.rubric_version,
            analyzed_at=published.analyzed_at.isoformat() if published.analyzed_at else None,
            publisher_context="",
            inputs=check_inputs(published, flags),
        )
        try:
            result.publisher_context = await self.publishers.context_for(article.publisher)
            runs = await asyncio.gather(
                *(
                    self._score_once(client, published, flags, result.publisher_context)
                    for _ in range(self.repeats)
                )
            )
        except Exception as exc:  # one bad article must not sink the run
            log.warning("backtest failed for %s: %s", article.article_id, exc)
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        stored = verdict_scores(published.verdict)
        result.comparisons = [
            compare(metric, stored[metric], [r[metric] for r in runs])
            for metric in AUDIT_METRICS
        ]
        return result

    async def run(self, analyses: list[PublishedAnalysis]) -> dict[str, Any]:
        log.info(
            "backtesting %d article(s) x %d repeat(s) = %d model call(s)",
            len(analyses), self.repeats, len(analyses) * self.repeats,
        )
        # Resolve each publisher's context once, not once per concurrent article.
        # A failure here is retried and recorded per article in backtest().
        for domain in sorted({p.article.publisher for p in analyses}):
            with contextlib.suppress(Exception):
                await self.publishers.context_for(domain)
        started = time.monotonic()
        results = list(await asyncio.gather(*(self.backtest(p) for p in analyses)))
        return self._report(results, time.monotonic() - started)

    def _report(self, results: list[ArticleBacktest], elapsed: float) -> dict[str, Any]:
        ok = [r for r in results if r.error is None]
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engine_rubric_version": RUBRIC_VERSION,
            "rerun_models": sorted({r.rerun_model for r in results}),
            "repeats": self.repeats,
            "flag_rule": f"|delta| > {FLAG_RATIO} x noise and |delta| > {TOLERANCE}",
            "articles": len(results),
            "articles_failed": len(results) - len(ok),
            "articles_flagged": sum(bool(r.flagged) for r in ok),
            "rubric_mismatches": sum(not r.inputs.rubric_match for r in results),
            "nlp_mismatches": sum(r.inputs.nlp_match is False for r in results),
            "elapsed_s": round(elapsed, 1),
            "overall": [asdict(a) for a in aggregate(ok)],
            "results": [r.to_dict() for r in results],
        }

    def write(self, report: dict[str, Any], stem: str = "backtest") -> tuple[Path, Path]:
        """Persist the report as JSON + a readable Markdown summary."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        json_path = self.output_dir / f"{stem}_{ts}.json"
        md_path = self.output_dir / f"{stem}_{ts}.md"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        return json_path, md_path


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Backtest of published verdicts",
        "",
        f"- engine rubric: `{report['engine_rubric_version']}`",
        f"- re-scored with: {', '.join(f'`{m}`' for m in report['rerun_models']) or '—'}",
        f"- repeats per article: {report['repeats']}",
        f"- articles: {report['articles']} (failed: {report['articles_failed']}, "
        f"flagged: {report['articles_flagged']})",
        f"- published under a different rubric: {report['rubric_mismatches']}",
        f"- NLP signals differ from the stored ones: {report['nlp_mismatches']}",
        f"- flag rule: {report['flag_rule']}",
        f"- generated: {report['generated_at']}",
        "",
        "`delta` = mean(re-scorings) − published score. `noise` = the scorer's",
        "own spread across the re-scorings of identical input.",
        "",
        "## Overall",
        "",
        "| metric | n | mean delta | mean abs delta | mean noise | flagged |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in report["overall"]:
        lines.append(
            f"| `{a['metric']}` | {a['n']} | {a['mean_delta']:+.2f} | "
            f"{a['mean_abs_delta']:.2f} | {a['mean_noise']:.2f} | {a['flagged']} |"
        )
    lines += ["", "## Articles", ""]
    for r in report["results"]:
        lines += _article_lines(r)
    return "\n".join(lines + ["", *_CAVEATS])


def _article_lines(r: dict[str, Any]) -> list[str]:
    head = [f"### `{r['article_id']}` ({r['publisher']})", "", r["url"], ""]
    notes = []
    if not r["inputs"]["rubric_match"]:
        notes.append(f"published under rubric `{r['published_rubric']}`")
    if r["published_model"] and r["published_model"] != r["rerun_model"]:
        notes.append(f"published by `{r['published_model']}`, re-scored with `{r['rerun_model']}`")
    if r["inputs"]["nlp_match"] is False:
        notes.append(f"NLP signals differ: {r['inputs']['nlp_diff']}")
    if notes:
        head += [f"- {n}" for n in notes] + [""]
    if r["error"]:
        return head + [f"Failed: `{r['error']}`", ""]
    rows = [
        "| metric | published | re-scorings | mean | delta | noise | flag |",
        "|---|---:|---|---:|---:|---:|:--:|",
    ]
    for c in r["comparisons"]:
        rows.append(
            f"| `{c['metric']}` | {c['published']} | {', '.join(map(str, c['reruns']))} | "
            f"{c['rerun_mean']:.1f} | {c['delta']:+.1f} | {c['noise']:.1f} | "
            f"{'⚠️' if c['flagged'] else ''} |"
        )
    return head + rows + [""]


_CAVEATS = [
    "## How to read this",
    "",
    "**A flag is a lead, not a verdict.** The published score is one draw from a",
    "nondeterministic scorer; a flag says it sits outside what the engine",
    "produces for the same input today. Before concluding anything, check:",
    "",
    "1. **Rubric revision.** A verdict published under an older rubric is not",
    "   expected to reproduce. Across many articles, `mean delta` then measures",
    "   how the rubric change moved scores.",
    "2. **Publisher context.** The ownership context is not stored with the",
    "   verdict; the backtest uses today's ownership graph. If ownership data",
    "   changed after the verdict, the input changed too.",
    "3. **Model and settings.** The re-scoring uses the published model unless",
    "   `--model` overrides it, with the engine's default reasoning effort. A",
    "   retired model can't be re-run.",
    "4. **NLP mismatch.** If the recomputed NLP signals differ from the stored",
    "   ones, the article text or the NLP code differs from what production ran,",
    "   and the comparison is not like for like.",
]
