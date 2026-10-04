"""
LLM client abstraction for Bebas Bias.

Default provider is OpenAI. The `LLMClient` Protocol is the seam: to add or
swap another provider (Anthropic, Gemini, a local model), implement
`analyze()` and register it in `get_default_client()`. The prompt template,
NLP pre-processing, and FastAPI route never need to change.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from openai import OpenAI

from .prompt import SYSTEM_PROMPT, build_user_prompt
from .schemas import (
    BIAS_AXES,
    ArticleVerdict,
    BiasFingerprint,
    EvidenceSnippet,
    NLPFlags,
    StoryThreadArticle,
)


# ---------------------------------------------------------------------------
# Seam
# ---------------------------------------------------------------------------
class LLMClient(Protocol):
    """Minimal contract every provider adapter must satisfy."""

    def analyze(
        self,
        article: StoryThreadArticle,
        nlp: NLPFlags,
        publisher_context: str,
    ) -> ArticleVerdict: ...


# ---------------------------------------------------------------------------
# OpenAI implementation (default)
# ---------------------------------------------------------------------------
class OpenAIClient:
    """
    OpenAI Chat Completions adapter.

    Notes:
      - We force structured output with `response_format={"type": "json_object"}`.
        OpenAI requires the token "JSON" to appear somewhere in the messages;
        our SYSTEM_PROMPT already says "Output JSON ONLY" so that's satisfied.
      - Prompt caching on OpenAI is automatic for long stable prefixes — no
        explicit cache_control needed. Keeping SYSTEM_PROMPT stable across
        calls is what unlocks the cache hit.
    """

    def __init__(
        self,
        model: str | None = None,
        # 2048 fits a Bahasa Indonesia verdict with 4–5 evidence snippets even
        # on chatty newer-gen models. Indonesian uses more tokens per word
        # than English, and gpt-5/o-series models reserve a chunk for hidden
        # reasoning, so 1024 truncated the JSON mid-array in practice.
        max_tokens: int = 2048,
        # Hidden reasoning tokens bill as output, so they dominate cost on
        # reasoning models. "low" is enough for a rubric-guided verdict; set
        # BEBAS_BIAS_REASONING_EFFORT to override (e.g. "none", "medium").
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        # Per-request timeout (seconds) and SDK retry count; None keeps the SDK
        # defaults. The scheduled sync analysis bounds both so a slow call
        # can't outlive the job's own timeout.
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        client_opts: dict = {}
        if timeout is not None:
            client_opts["timeout"] = timeout
        if max_retries is not None:
            client_opts["max_retries"] = max_retries
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"), **client_opts
        )
        self.model = model or os.environ.get("BEBAS_BIAS_MODEL", "gpt-6-luna")
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort or os.environ.get(
            "BEBAS_BIAS_REASONING_EFFORT", "low"
        )

    def build_chat_request(
        self,
        article: StoryThreadArticle,
        nlp: NLPFlags,
        publisher_context: str,
    ) -> dict:
        """Build the exact Chat Completions request body for one article.

        Shared by the synchronous `analyze()` path and the offline Batch API
        path (app/batch.py) so both produce byte-identical requests — the batch
        JSONL line's `body` is literally this dict.
        """
        user_prompt = build_user_prompt(article, nlp, publisher_context)

        # GPT-5+/o-series are reasoning models: they reject `max_tokens` in
        # favour of `max_completion_tokens` and accept `reasoning_effort`;
        # gpt-4o/4.1 take neither. Our default (gpt-6-luna) takes the
        # reasoning branch. Detect by model name rather than gating on the SDK
        # version so the swap is data-driven.
        is_reasoning = self.model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
        body = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        if is_reasoning:
            body["max_completion_tokens"] = self.max_tokens
            body["reasoning_effort"] = self.reasoning_effort
        else:
            body["max_tokens"] = self.max_tokens
        return body

    def analyze(
        self,
        article: StoryThreadArticle,
        nlp: NLPFlags,
        publisher_context: str,
    ) -> ArticleVerdict:
        body = self.build_chat_request(article, nlp, publisher_context)
        response = self.client.chat.completions.create(**body)
        raw = response.choices[0].message.content or ""
        return _parse_verdict(raw, article)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_default_client(**options) -> LLMClient:
    """Resolve the LLM client from env. Add new providers by registering them
    here and giving them an adapter class above. ``options`` are passed to the
    adapter's constructor (e.g. ``timeout`` / ``max_retries``)."""
    provider = os.environ.get("BEBAS_BIAS_PROVIDER", "openai").lower()
    if provider == "openai":
        return OpenAIClient(**options)
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Add an adapter in app/llm.py and register it in get_default_client()."
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(raw: str, article: StoryThreadArticle) -> ArticleVerdict:
    """Extract the JSON object and validate it. With OpenAI's json_object mode
    the whole `raw` is already valid JSON, but we still defensively scan for
    the first {...} block in case we later swap to a provider without forced
    JSON mode."""
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise ValueError(f"LLM did not return JSON. Raw output:\n{raw}")
    payload = json.loads(match.group(0))

    fingerprint_payload = payload["bias_fingerprint"]
    fingerprint = BiasFingerprint(
        elite_alignment=int(fingerprint_payload["elite_alignment"]),
        identity_lens=int(fingerprint_payload["identity_lens"]),
        economic_sovereignty=int(fingerprint_payload["economic_sovereignty"]),
        state_vs_civil=int(fingerprint_payload["state_vs_civil"]),
        sensationalism=int(fingerprint_payload["sensationalism"]),
    )

    # Silently drop evidence snippets whose axis_affected is not one of the
    # 5 known axes rather than rejecting the whole verdict — keeps the
    # pipeline robust to model drift.
    snippets = [
        EvidenceSnippet(
            text=str(s["text"]),
            axis_affected=str(s["axis_affected"]),
            explanation=str(s["explanation"]),
        )
        for s in payload.get("evidence_snippets", [])
        if s.get("axis_affected") in BIAS_AXES
    ]

    return ArticleVerdict(
        article_id=article.article_id,
        url=article.url,
        publisher=article.publisher,
        bias_fingerprint=fingerprint,
        reliability_index=int(payload["reliability_index"]),
        analysis_summary=str(payload.get("analysis_summary", "")),
        evidence_snippets=snippets,
    )
