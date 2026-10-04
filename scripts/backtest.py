#!/usr/bin/env python
"""
CLI: backtest verdicts published on the site against the engine (see app/backtest.py).

    python -m scripts.backtest show <article>             # free: verdict + inputs + NLP check
    python -m scripts.backtest show <article> --request   # ... plus the exact model request
    python -m scripts.backtest run <article> [<article> ...] --repeats 3
    python -m scripts.backtest run --latest 20 --publisher detik.com

<article> is the article id, a site link containing /article/<id>, or the
publisher's article URL. `show` needs nothing but network access; `run` calls
the model and needs OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the package importable whether run as `python -m scripts.backtest` or
# `python scripts/backtest.py`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT.parent / ".env")  # the backend's repo-root .env

try:
    from app.backtest import (  # noqa: E402
        DEFAULT_CONCURRENCY,
        DEFAULT_REPEATS,
        Backtester,
        check_inputs,
        verdict_scores,
    )
    from app.llm import OpenAIClient, get_default_client  # noqa: E402
    from app.nlp import analyze_article  # noqa: E402
    from app.prompt import MAX_BODY_CHARS, RUBRIC_VERSION  # noqa: E402
    from app.public_data import NotFound, PublicSite, PublishedAnalysis  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
    raise SystemExit(
        f"Missing dependency {exc.name!r}. Install the requirements first:\n"
        f"    pip install -r {_ROOT / 'requirements.txt'}"
    ) from exc

log = logging.getLogger("bebas_bias.backtest")


def _logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger("bebas_bias")
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------
def _describe(published: PublishedAnalysis, context: str) -> str:
    a = published.article
    flags = analyze_article(a)
    check = check_inputs(published, flags)
    when = published.analyzed_at.strftime("%Y-%m-%d %H:%M UTC") if published.analyzed_at else "unknown time"
    rubric = published.rubric_version or "unrecorded"
    rubric_note = "same as this engine" if check.rubric_match else f"this engine is on {RUBRIC_VERSION}"
    body = a.body.strip()
    sent = min(len(body), MAX_BODY_CHARS) if MAX_BODY_CHARS else len(body)

    if check.nlp_match is None:
        nlp = "not stored with this verdict (older row); nothing to compare"
    elif check.nlp_match:
        nlp = "match the stored ones"
    else:
        nlp = "DIFFER from the stored ones: " + ", ".join(
            f"{k} stored={s!r} now={n!r}" for k, (s, n) in check.nlp_diff.items()
        )

    lines = [
        f"Article    {a.article_id}  ({a.publisher})",
        f"URL        {a.url}",
        f"Headline   {a.headline}",
        f"Analyzed   {when} by {published.model or 'unrecorded model'}, "
        f"rubric {rubric} ({rubric_note})",
        "",
        "Published verdict",
        *(f"  {k:<22}{v:>4}" for k, v in verdict_scores(published.verdict).items()),
        "",
        f"  {published.verdict.analysis_summary}",
        "",
        "Inputs the engine sees",
        f"  body               {len(body)} chars, {sent} sent to the model",
        f"  publisher_context  {context or '(none: publisher not in the ownership graph)'}",
        f"  NLP signals        {nlp}",
        f"    clickbait_delta={flags.clickbait.clickbait_delta} "
        f"headline_sentiment={flags.clickbait.headline_sentiment} "
        f"body_sentiment={flags.clickbait.body_sentiment} "
        f"charged_adjectives={flags.charged_adjectives}",
    ]
    return "\n".join(lines)


def _request(published: PublishedAnalysis, context: str, model: str | None) -> dict:
    # Building the request never calls the API, so no real key is needed.
    client = OpenAIClient(model=model or published.model, api_key="unused-for-request-building")
    return client.build_chat_request(published.article, analyze_article(published.article), context)


async def _show(args) -> None:
    async with PublicSite() as site:
        published = await site.analysis(args.article)
        context = await site.context_for(published.article.publisher)
    if args.json:
        out = {
            "article": published.article.model_dump(mode="json"),
            "verdict": published.verdict.model_dump(mode="json"),
            "model": published.model,
            "rubric_version": published.rubric_version,
            "engine_rubric_version": RUBRIC_VERSION,
            "publisher_context": context,
            "inputs": check_inputs(published).__dict__,
        }
        if args.request:
            out["request"] = _request(published, context, args.model)
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
        return
    print(_describe(published, context))
    if args.request:
        print("\nExact request body (OpenAI Chat Completions)\n")
        print(json.dumps(_request(published, context, args.model), indent=2, ensure_ascii=False))
    else:
        print("\nAdd --request to print the exact request body sent to the model.")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
async def _select(site: PublicSite, args) -> list[PublishedAnalysis]:
    if args.articles:
        return [await site.analysis(ref) for ref in args.articles]
    rubric = {"current": RUBRIC_VERSION, "any": None}.get(args.rubric, args.rubric)
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc) if args.since else None
    found = await site.latest(
        limit=args.latest, publisher=args.publisher, since=since, rubric_version=rubric
    )
    if not found:
        raise SystemExit("no published verdicts match those filters")
    return found


async def _run(args) -> dict:
    async with PublicSite() as site:
        analyses = await _select(site, args)
        backtester = Backtester(
            client_for=lambda model: get_default_client(model=model) if model else get_default_client(),
            publishers=site,
            model=args.model,
            repeats=args.repeats,
            concurrency=args.concurrency,
        )
        report = await backtester.run(analyses)
    json_path, md_path = backtester.write(report)
    log.info("wrote %s", json_path)
    log.info("wrote %s", md_path)
    return {k: v for k, v in report.items() if k != "results"}


def main() -> None:
    _logging()
    parser = argparse.ArgumentParser(description="Backtest published verdicts against the engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_show = sub.add_parser("show", help="a published verdict and its inputs (no model calls)")
    p_show.add_argument("article", help="article id, site link, or publisher URL")
    p_show.add_argument("--request", action="store_true", help="also print the exact model request")
    p_show.add_argument("--model", help="model to build the request for (default: the published one)")
    p_show.add_argument("--json", action="store_true", help="machine-readable output")

    p_run = sub.add_parser("run", help="re-score published verdicts and compare (calls the model)")
    p_run.add_argument("articles", nargs="*", help="article ids, site links, or publisher URLs")
    p_run.add_argument("--latest", type=int, help="instead of naming articles, take the N most recently analyzed")
    p_run.add_argument("--publisher", help="with --latest: only this publisher domain, e.g. detik.com")
    p_run.add_argument("--since", help="with --latest: only verdicts analyzed on or after this date (YYYY-MM-DD)")
    p_run.add_argument(
        "--rubric", default="current",
        help="with --latest: 'current' (default; verdicts made under this engine's rubric, so they "
             "should reproduce), 'any', or a specific rubric version",
    )
    p_run.add_argument("--model", help="re-score with this model instead of the one each verdict was published with")
    p_run.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="re-scorings per article (>=2)")
    p_run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="parallel model calls")

    args = parser.parse_args()
    if args.command == "run":
        if bool(args.articles) == bool(args.latest):
            parser.error("run: name articles or pass --latest N (not both)")
        provider = os.environ.get("BEBAS_BIAS_PROVIDER", "openai").lower()
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            parser.error("run calls the model: set OPENAI_API_KEY (environment or .env)")

    try:
        if args.command == "show":
            asyncio.run(_show(args))
        else:
            print(json.dumps(asyncio.run(_run(args)), indent=2, default=str))
    except NotFound as exc:
        raise SystemExit(str(exc)) from None
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    except LookupError as exc:  # nltk raises LookupError for missing data
        raise SystemExit(
            f"{exc}\nThe NLP signals need NLTK data. Download it once with:\n"
            "    python scripts/bootstrap_nltk.py"
        ) from None


if __name__ == "__main__":
    main()
