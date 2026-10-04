"""
The counterfactual audit itself.

For each real article that names a demographic group, build the minimal pair —
identical text except for the group named — score both variants ``repeats``
times through the *unmodified* production scorer, and report the per-axis
effect against the scorer's own sampling noise.

Deliberately reuses the live path (``OpenAIClient.analyze`` ->
``build_chat_request`` -> ``_parse_verdict``, ``nlp.analyze_article``,
``PublisherContextStore.context_for``) rather than reimplementing it, so what
we measure is what production does. NLP flags are recomputed per variant
because substitution changes the text; publisher context is resolved once per
article and shared by both variants so it stays a controlled variable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..llm import LLMClient
from ..publishers import PublisherContextStore
from ..schemas import NLPFlags, StoryThreadArticle
from .groups import Contrast, Gazetteer, default_gazetteer
from .metrics import AUDIT_METRICS, AggregateEffect, AxisEffect, aggregate, score_effect
from .substitute import count_mentions, substitute

log = logging.getLogger("bebas_bias.audit")

DEFAULT_REPEATS = 3
DEFAULT_CONCURRENCY = 4


@dataclass
class ArticlePair:
    """One article rendered as a minimal pair over a contrast."""

    article_id: str
    url: str
    publisher: str
    contrast: Contrast
    variant_a: StoryThreadArticle
    variant_b: StoryThreadArticle
    mentions: int
    #: Which side the original article actually named.
    original_side: str

    @property
    def label(self) -> str:
        return self.contrast.label


def _nlp_flags(article: StoryThreadArticle) -> NLPFlags:
    """Run the production pre-LLM NLP pass on one article.

    Imported lazily rather than at module scope: `plan()` and the pair builders
    never score anything, so coverage checks should not need NLTK (or its
    downloaded corpora) present.
    """
    try:
        from ..nlp import analyze_article
    except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            f"scoring needs the app's runtime dependencies ({exc.name} is "
            "missing). Run the audit with the project environment, e.g. "
            "`./.venv/bin/python -m scripts.counterfactual_audit ...`."
        ) from exc
    return analyze_article(article)


def _retext(
    article: StoryThreadArticle, headline: str, body: str
) -> StoryThreadArticle:
    """Clone an article with new text, keeping every other field identical."""
    return article.model_copy(
        update={
            "headline": headline,
            "body": body,
            "word_count": len(body.split()),
        }
    )


def build_pairs(
    articles: list[StoryThreadArticle], gazetteer: Gazetteer | None = None
) -> list[ArticlePair]:
    """Find every (article, contrast) minimal pair we can construct.

    An article naming *both* sides of a contrast is skipped: substitution would
    collapse two distinct referents into one and the pair would no longer be
    minimal.
    """
    gaz = gazetteer or default_gazetteer()
    pairs: list[ArticlePair] = []

    for article in articles:
        text = f"{article.headline}\n{article.body}"
        for contrast in gaz.contrasts:
            n_a = count_mentions(text, contrast.a)
            n_b = count_mentions(text, contrast.b)
            if bool(n_a) == bool(n_b):
                # Neither side present, or both — no clean minimal pair.
                continue

            if n_a:
                src, dst, side, mentions = contrast.a, contrast.b, "a", n_a
            else:
                src, dst, side, mentions = contrast.b, contrast.a, "b", n_b

            swapped = _retext(
                article,
                substitute(article.headline, src, dst).text,
                substitute(article.body, src, dst).text,
            )
            variant_a, variant_b = (
                (article, swapped) if side == "a" else (swapped, article)
            )
            pairs.append(
                ArticlePair(
                    article_id=article.article_id,
                    url=str(article.url),
                    publisher=article.publisher,
                    contrast=contrast,
                    variant_a=variant_a,
                    variant_b=variant_b,
                    mentions=mentions,
                    original_side=side,
                )
            )
    return pairs


@dataclass
class PairResult:
    article_id: str
    url: str
    publisher: str
    contrast: str
    mentions: int
    scores_a: dict[str, list[int]] = field(default_factory=dict)
    scores_b: dict[str, list[int]] = field(default_factory=dict)
    effects: list[AxisEffect] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "article_id": self.article_id,
            "url": self.url,
            "publisher": self.publisher,
            "contrast": self.contrast,
            "mentions": self.mentions,
            "scores_a": self.scores_a,
            "scores_b": self.scores_b,
            "effects": [
                {**e.__dict__, "ratio": round(e.ratio, 2)} for e in self.effects
            ],
            "error": self.error,
        }


class CounterfactualAudit:
    """Runs minimal-pair scoring over real articles."""

    def __init__(
        self,
        client: LLMClient | None = None,
        publishers: PublisherContextStore | None = None,
        *,
        output_dir: Path | None = None,
        repeats: int = DEFAULT_REPEATS,
        concurrency: int = DEFAULT_CONCURRENCY,
        gazetteer: Gazetteer | None = None,
    ) -> None:
        if repeats < 2:
            raise ValueError(
                "repeats must be >= 2: with a single scoring per variant there "
                "is no noise floor and the deltas are uninterpretable."
            )
        # None is legal for plan()-only use, which never calls the model.
        self.client = client
        self.publishers = publishers
        self.output_dir = output_dir or Path(__file__).resolve().parents[2] / "outputs"
        self.repeats = repeats
        self.gazetteer = gazetteer or default_gazetteer()
        self._sem = asyncio.Semaphore(concurrency)

    # -- coverage, no LLM spend -------------------------------------------
    def plan(self, articles: list[StoryThreadArticle]) -> dict:
        """Report what a run *would* measure, without calling the model."""
        pairs = build_pairs(articles, self.gazetteer)
        by_contrast: dict[str, int] = {}
        for p in pairs:
            by_contrast[p.label] = by_contrast.get(p.label, 0) + 1
        blocked = sum(
            substitute(f"{a.headline}\n{a.body}", c.a, c.b).blocked
            for a in articles
            for c in self.gazetteer.contrasts
        )
        return {
            "articles_scanned": len(articles),
            "pairs": len(pairs),
            "mentions_blocked_by_guards": blocked,
            "llm_calls": len(pairs) * self.repeats * 2,
            "repeats": self.repeats,
            "gazetteer_version": self.gazetteer.version,
            "by_contrast": dict(
                sorted(by_contrast.items(), key=lambda kv: -kv[1])
            ),
        }

    # -- scoring ----------------------------------------------------------
    async def _context_for(self, publisher: str) -> str:
        if self.publishers is None or not self.publishers.configured:
            return ""
        return await self.publishers.context_for(publisher)

    async def _score_once(self, article: StoryThreadArticle, context: str) -> dict:
        flags = _nlp_flags(article)
        async with self._sem:
            verdict = await asyncio.to_thread(
                self.client.analyze, article, flags, context
            )
        fp = verdict.bias_fingerprint.model_dump()
        fp["reliability_index"] = verdict.reliability_index
        return fp

    async def _score_variant(
        self, article: StoryThreadArticle, context: str
    ) -> dict[str, list[int]]:
        runs = await asyncio.gather(
            *(self._score_once(article, context) for _ in range(self.repeats))
        )
        return {axis: [r[axis] for r in runs] for axis in AUDIT_METRICS}

    async def _run_pair(self, pair: ArticlePair) -> PairResult:
        result = PairResult(
            article_id=pair.article_id,
            url=pair.url,
            publisher=pair.publisher,
            contrast=pair.label,
            mentions=pair.mentions,
        )
        try:
            context = await self._context_for(pair.publisher)
            scores_a, scores_b = await asyncio.gather(
                self._score_variant(pair.variant_a, context),
                self._score_variant(pair.variant_b, context),
            )
        except Exception as exc:  # one bad article must not sink the audit
            log.warning("pair failed (%s, %s): %s", pair.article_id, pair.label, exc)
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        result.scores_a = scores_a
        result.scores_b = scores_b
        result.effects = [
            score_effect(axis, scores_a[axis], scores_b[axis])
            for axis in AUDIT_METRICS
        ]
        return result

    async def run(
        self, articles: list[StoryThreadArticle], *, limit: int | None = None
    ) -> dict:
        pairs = build_pairs(articles, self.gazetteer)
        if limit is not None:
            pairs = pairs[:limit]
        if not pairs:
            log.info("no minimal pairs found in %d article(s)", len(articles))
            return self._report([], 0.0, len(articles))

        log.info(
            "auditing %d pair(s) x %d repeat(s) x 2 variants = %d LLM call(s)",
            len(pairs), self.repeats, len(pairs) * self.repeats * 2,
        )
        started = time.monotonic()
        results = [await self._run_pair(p) for p in pairs]
        elapsed = time.monotonic() - started
        return self._report(results, elapsed, len(articles))

    # -- reporting --------------------------------------------------------
    def _report(
        self, results: list[PairResult], elapsed: float, n_articles: int
    ) -> dict:
        ok = [r for r in results if r.error is None]
        overall = aggregate([e for r in ok for e in r.effects])

        by_contrast: dict[str, list[AggregateEffect]] = {}
        grouped: dict[str, list[AxisEffect]] = {}
        for r in ok:
            grouped.setdefault(r.contrast, []).extend(r.effects)
        for contrast, effects in grouped.items():
            by_contrast[contrast] = aggregate(effects)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gazetteer_version": self.gazetteer.version,
            "model": getattr(self.client, "model", "unknown"),
            "repeats": self.repeats,
            "articles_scanned": n_articles,
            "pairs_measured": len(ok),
            "pairs_failed": len(results) - len(ok),
            "elapsed_s": round(elapsed, 1),
            "overall": [a.to_dict() for a in overall],
            "by_contrast": {
                k: [a.to_dict() for a in v] for k, v in sorted(by_contrast.items())
            },
            "pairs": [r.to_dict() for r in results],
        }

    def write(self, report: dict, stem: str = "counterfactual") -> tuple[Path, Path]:
        """Persist the report as JSON + a readable Markdown summary."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        json_path = self.output_dir / f"{stem}_{ts}.json"
        md_path = self.output_dir / f"{stem}_{ts}.md"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        return json_path, md_path


def _table(rows: list[dict]) -> list[str]:
    out = [
        "| axis | n | signed Δ | abs Δ | noise | ratio | flag |",
        "|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in rows:
        ratio = r["ratio"]
        ratio_s = "∞" if ratio == float("inf") else f"{ratio:.2f}"
        out.append(
            f"| `{r['axis']}` | {r['n']} | {r['signed_effect']:+.2f} | "
            f"{r['abs_effect']:.2f} | {r['noise']:.2f} | {ratio_s} | "
            f"{'⚠️' if r['flagged'] else ''} |"
        )
    return out


def render_markdown(report: dict) -> str:
    """Human-readable summary. `signed Δ` is the bias statistic; `ratio` is
    abs Δ over the scorer's own sampling noise — under 1.0 means nothing was
    found."""
    lines = [
        "# Counterfactual bias audit",
        "",
        f"- model: `{report['model']}`",
        f"- gazetteer: `{report['gazetteer_version']}`",
        f"- repeats per variant: {report['repeats']}",
        f"- articles scanned: {report['articles_scanned']}",
        f"- pairs measured: {report['pairs_measured']} "
        f"(failed: {report['pairs_failed']})",
        f"- generated: {report['generated_at']}",
        "",
        "`signed Δ` = mean(variant B) − mean(variant A), averaged over articles.",
        "`noise` = the scorer's within-variant spread on identical text.",
        f"`ratio` = abs Δ ÷ noise; flagged at ≥ 1.5.",
        "",
        "Each contrast fixes an arbitrary A/B orientation, so the **overall**",
    "signed Δ mixes directions and can cancel — read it for magnitude only.",
    "`by_contrast` is authoritative for direction.",
    "",
    "## Overall",
        "",
        *_table(report["overall"]),
        "",
        "## By contrast",
        "",
    ]
    for contrast, rows in report["by_contrast"].items():
        lines += [f"### `{contrast}`", "", *_table(rows), ""]
    return "\n".join(lines + _caveats(report))


_CAVEAT_TEXT = [
    "## How to read this",
    "",
    "**A flag is a lead, not a verdict.** Open the flagged pairs in the JSON and",
    "read both variants before concluding anything. Two confounds produce real",
    "score deltas that are not demographic bias:",
    "",
    "1. **World-knowledge incoherence.** An article naming a person bound to the",
    "   group — \"Sekretaris Jenderal PDIP, Hasto Kristiyanto\" — becomes",
    "   factually wrong when the party is swapped. The scorer may react to the",
    "   contradiction rather than to the party. These pairs are not filtered:",
    "   detecting them needs an entity-affiliation database we do not have.",
    "2. **Acronym and toponym collisions.** The gazetteer guards catch the known",
    "   ones (\"Jawa Barat\", \"Panglima TNI\", \"One UI\"), but a novel collision",
    "   in new text will slip through. `mentions_blocked_by_guards` in the plan",
    "   output shows the guards firing.",
    "",
    "Contrasts in a category marked non-substitutable are excluded by default",
    "for exactly this reason; `--include-fragile` opts back in.",
]


def _caveats(report: dict) -> list[str]:
    return ["", *_CAVEAT_TEXT]
