#!/usr/bin/env python
"""
CLI for the counterfactual bias audit (see app/audit/).

Takes real articles, swaps the demographic group they name for a same-category
counterpart, re-scores both variants through the unmodified production scorer,
and reports the per-axis effect against the scorer's own sampling noise. A fair
scorer produces effects indistinguishable from that noise.

    python -m scripts.counterfactual_audit plan                # coverage, no spend
    python -m scripts.counterfactual_audit run --limit 20       # measure 20 pairs
    python -m scripts.counterfactual_audit run --from-json f.json --limit 5
    python -m scripts.counterfactual_audit run --from-site --articles 300

Articles come from --from-json, else the database when DATABASE_URL is set,
else the public site (--from-site forces it). Publisher context comes from the
database when DATABASE_URL is set, else the public site. `run` needs
OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Make the bb-think package importable whether run as `python -m
# scripts.counterfactual_audit` or `python scripts/counterfactual_audit.py`.
_BB_THINK = Path(__file__).resolve().parents[1]
if str(_BB_THINK) not in sys.path:
    sys.path.insert(0, str(_BB_THINK))

from dotenv import load_dotenv

# A local .env, then the backend's repo-root .env one level above bb-think/
# (both no-ops inside Docker).
load_dotenv(_BB_THINK / ".env")
load_dotenv(_BB_THINK.parent / ".env")

try:
    from app.audit.counterfactual import CounterfactualAudit  # noqa: E402
    from app.audit.groups import default_gazetteer  # noqa: E402
    from app.llm import get_default_client  # noqa: E402
    from app.public_data import PublicSite  # noqa: E402
    from app.publishers import DATABASE_URL_ENV, PublisherContextStore  # noqa: E402
    from app.schemas import StoryThreadArticle  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
    # A bare `python -m scripts...` picks up whatever interpreter is on PATH,
    # which usually is not the one holding the app's dependencies. Say so
    # instead of surfacing a ten-line traceback about a transitive import.
    _venv = _BB_THINK / ".venv" / "bin" / "python"
    raise SystemExit(
        f"Missing dependency {exc.name!r}.\n"
        + (
            f"Run with the project environment:\n"
            f"    {_venv} -m scripts.counterfactual_audit ...\n"
            if _venv.exists()
            else f"Install the dependencies:\n"
                 f"    pip install -r {_BB_THINK / 'requirements.txt'}\n"
        )
    ) from exc

log = logging.getLogger("bebas_bias.audit")


def _logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger("bebas_bias")
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False


def _articles_from_json(path: Path) -> list[StoryThreadArticle]:
    """Load articles from a scraper-shaped JSON file (a bare list of articles,
    a ScrapeRun envelope, or anything with a `threads[].articles[]` path)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = [a for t in raw.get("threads", []) for a in t.get("articles", [])]
    else:
        items = raw
    return [StoryThreadArticle.model_validate(a) for a in items]


def _use_db() -> bool:
    return bool(os.environ.get(DATABASE_URL_ENV))


async def _articles_from_db(status: str) -> list[StoryThreadArticle]:
    # Imported here: the public analyzer repo ships without the DB layer.
    from app.persistence import AnalysisStore

    store = AnalysisStore()
    try:
        by_id = await store.fetch_articles_by_status(status)
    finally:
        await store.close()
    if not by_id:
        log.warning("no articles with status=%r", status)
    return list(by_id.values())


async def _articles_from_site(limit: int) -> list[StoryThreadArticle]:
    async with PublicSite() as site:
        return [p.article for p in await site.latest(limit=limit)]


def _filter_gazetteer(patterns: list[str] | None, include_fragile: bool = False):
    """Restrict the audit to contrasts whose label contains any pattern."""
    gaz = default_gazetteer(include_fragile=include_fragile)
    if not patterns:
        return gaz
    keep = [c for c in gaz.contrasts if any(p in c.label for p in patterns)]
    if not keep:
        raise SystemExit(
            f"--contrast {patterns!r} matched none of: "
            + ", ".join(c.label for c in gaz.contrasts)
        )
    gaz.contrasts = keep
    return gaz


async def _load(args) -> list[StoryThreadArticle]:
    if args.from_json:
        return _articles_from_json(Path(args.from_json))
    if args.from_site or not _use_db():
        log.info("reading the %d most recent published articles from the site", args.articles)
        return await _articles_from_site(args.articles)
    return await _articles_from_db(args.status)


async def _plan(args) -> dict:
    articles = await _load(args)
    audit = CounterfactualAudit(
        client=None,  # plan() never calls the model
        repeats=args.repeats,
        gazetteer=_filter_gazetteer(args.contrast, args.include_fragile),
    )
    return audit.plan(articles)


async def _run(args) -> dict:
    articles = await _load(args)
    publishers = PublisherContextStore() if _use_db() else PublicSite()
    try:
        audit = CounterfactualAudit(
            client=get_default_client(),
            publishers=publishers,
            output_dir=_BB_THINK / "outputs",
            repeats=args.repeats,
            concurrency=args.concurrency,
            gazetteer=_filter_gazetteer(args.contrast, args.include_fragile),
        )
        report = await audit.run(articles, limit=args.limit)
        json_path, md_path = audit.write(report)
        log.info("wrote %s", json_path)
        log.info("wrote %s", md_path)
        # The per-pair detail is large and already on disk; keep stdout scannable.
        return {k: v for k, v in report.items() if k != "pairs"}
    finally:
        await publishers.close()


def main() -> None:
    _logging()
    parser = argparse.ArgumentParser(description="Counterfactual bias audit")
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--from-json", help="read articles from a JSON file instead of the DB")
        p.add_argument(
            "--from-site", action="store_true",
            help="read published articles from the public site even when DATABASE_URL is set",
        )
        p.add_argument(
            "--articles", type=int, default=200,
            help="with the public site: how many recent published articles to scan (max 1000)",
        )
        p.add_argument("--status", default="analyzed", help="DB article status to sample")
        p.add_argument("--repeats", type=int, default=3, help="scorings per variant (>=2)")
        p.add_argument(
            "--contrast", action="append",
            help="only contrasts whose label contains this (repeatable), e.g. ethnicity",
        )
        p.add_argument(
            "--include-fragile", action="store_true",
            help=(
                "also audit categories whose members are not interchangeable in "
                "context (government_institution). Their deltas are confounded "
                "by incoherence — read them with care."
            ),
        )

    p_plan = sub.add_parser("plan", help="report coverage without calling the model")
    _common(p_plan)

    p_run = sub.add_parser("run", help="score the minimal pairs and write a report")
    _common(p_run)
    p_run.add_argument("--limit", type=int, default=20, help="max pairs to measure")
    p_run.add_argument("--concurrency", type=int, default=4, help="parallel LLM calls")

    args = parser.parse_args()
    result = asyncio.run(_plan(args) if args.command == "plan" else _run(args))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
