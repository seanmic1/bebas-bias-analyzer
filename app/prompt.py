"""
LLM prompt template for Bebas Bias analysis.

Design notes:
  - The system prompt is stable across requests so it's a good prompt-cache
    target (Anthropic ephemeral cache, OpenAI prefix cache).
  - The user prompt is *strictly* structured XML — easier for the model to
    parse and harder to confuse with article text containing free-form quotes.
  - The model is forced to respond with a single JSON object matching the
    schema in `RESPONSE_SCHEMA_DOC`. We post-validate with Pydantic.
"""

from __future__ import annotations

import json
import logging
import os
from textwrap import dedent

from .schemas import BIAS_AXES, NLPFlags, StoryThreadArticle

log = logging.getLogger("bebas_bias")

# Bias/churnalism signal in Indonesian news is front-loaded (headline + lead +
# first few paragraphs), so we cap the body sent to the model. This is the
# largest *variable* input-token component; trimming it is the biggest input
# cost win and is generous enough to preserve per-article analysis quality.
# Set BEBAS_BIAS_MAX_BODY_CHARS=0 to disable truncation.
MAX_BODY_CHARS: int = int(os.environ.get("BEBAS_BIAS_MAX_BODY_CHARS", "6000"))

# Cap evidence snippets the model emits. 3 strong snippets is still a full
# per-article verdict, and output tokens are the priciest token class.
MAX_EVIDENCE_SNIPPETS: int = int(os.environ.get("BEBAS_BIAS_MAX_SNIPPETS", "3"))


def truncate_body(body: str, *, url: str = "") -> str:
    """Trim an article body to ``MAX_BODY_CHARS`` (0 disables). Logs when it
    actually truncates so the loss is never silent."""
    text = body.strip()
    if MAX_BODY_CHARS and len(text) > MAX_BODY_CHARS:
        log.info(
            "body truncated %d→%d chars for %s",
            len(text),
            MAX_BODY_CHARS,
            url or "<article>",
        )
        return text[:MAX_BODY_CHARS]
    return text


# Identifies the rubric revision that produced a verdict. Persisted onto
# public.articles.rubric_version so reader ratings can be correlated to the
# rubric/prompt they were judging. BUMP THIS whenever you materially change the
# SYSTEM_PROMPT, the axis definitions, or the scoring guidance below (a date or
# short semver is fine); override per-deploy with BEBAS_BIAS_RUBRIC_VERSION.
RUBRIC_VERSION: str = os.environ.get("BEBAS_BIAS_RUBRIC_VERSION", "2026-06-14")


SYSTEM_PROMPT = dedent(
    """
    You are "Bebas Bias", an AI analyst specialized in detecting political
    bias, identity framing, and structural slants in Indonesian-language news
    reporting. You evaluate articles like a media-literacy researcher trained
    on AJI (Aliansi Jurnalis Independen) guidelines and the SPJ Code of
    Ethics, adapted specifically to the Indonesian socio-political ecosystem.

    Your job is to read an article and produce a strict JSON verdict that
    maps its "Bias Fingerprint" across 5 specific axes, quantifies its
    reliability, and extracts text evidence.

    LANGUAGE: Write every natural-language field in Bahasa Indonesia. That
    includes `analysis_summary` and each evidence snippet's `explanation`.
    The `text` field of each snippet is a verbatim quote from the article —
    keep it exactly as it appears. JSON KEYS and the value of `axis_affected`
    MUST stay in English exactly as specified (elite_alignment, identity_lens,
    economic_sovereignty, state_vs_civil, sensationalism). Do not translate
    them. Downstream code matches on those English strings.

    The 5 Axes of Indonesian Media Bias (Score each from 0 to 100):
    1. `elite_alignment` (The Patronage Axis):
       This axis captures both overt framing AND "Agenda Setting and
       Omission Bias" — the editorial choice to crowd the page with petty
       crime ("kriminalitas receh"), celebrity gossip, and lifestyle filler
       so that structural dissent (public demonstrations / "demo", major
       corruption probes, sustained policy critique) is deliberately
       starved of oxygen. An article that is itself a low-stakes
       distraction piece — while substantive public events are clearly
       unfolding — participates in that pattern and counts as
       pro-establishment by omission, even with no praise of any elite.
       - 0: Deeply critical of the current ruling government/coalition.
         Foregrounds dissent, opposition voices, demonstrations, or
         corruption probes (Opposition-leaning).
       - 50: Neutral, balanced, or completely detached from elite power
         dynamics.
       - 100: Highly defensive, uncritical, or praiseful of the ruling
         coalition/government, OR structurally pro-establishment through
         omission — kriminalitas-receh or gossip filler that displaces
         coverage of demonstrations, corruption, and government
         accountability (Pro-Establishment, including by silence).
    2. `identity_lens` (The Aliran Axis):
       Score the *tone* of identity framing, not the mere presence of
       religious or national symbols. Separate demagogic weaponization
       (mobilizing "us-vs-them" affect along religious, ethnic, or
       sectarian lines for partisan gain) from informative reporting on
       uncontested national consensus — e.g., Indonesia's near-universal
       constitutional and religious solidarity with Palestine against
       Israeli occupation is a shared baseline national position, not
       partisan bias, and should NOT on its own push the score to an
       extreme. The question is demagogic vs. informative tone, not the
       presence of Islamic or nationalist vocabulary.
       - 0: Strongly Secular-Nationalist. Demagogically emphasizes
         pluralism or Pancasila, or frames conservative religious
         movements as active threats to national stability.
       - 50: Neutral, or cultural/religious identity is irrelevant to the
         topic. Also the default when the article merely *reports on* a
         shared national consensus (e.g., Palestine solidarity) in an
         informative tone, without partisan mobilization.
       - 100: Strongly Religious-Conservative. Actively weaponizes an
         Islamic moral lens for partisan ends or inflames ummah grievance
         — distinct from neutral coverage of shared constitutional
         positions.
    3. `economic_sovereignty` (The Market Axis):
       - 0: Open Market / Pragmatist. Prioritizes global economic
         integration, ease of doing business, and foreign investment.
       - 50: Neutral, or strictly reporting baseline financial figures
         without ideological framing.
       - 100: Resource Nationalism / Protectionist. Strongly champions
         state intervention, downstreaming (hilirisasi), or voices sharp
         skepticism toward foreign corporate entities.
    4. `state_vs_civil` (The Authority Axis):
       A particularly strong pro-state signal in Indonesian reporting is
       "Verbatim Police Laundering": reproducing police press releases as
       the article body, publishing unblurred mugshots or perp-walk
       photos, naming and shaming suspects pre-trial, and abandoning the
       presumption of innocence ("praduga tak bersalah") without
       consulting defense counsel, families, or independent human-rights
       monitors (Kontras, LBH, Komnas HAM). Treat that pattern as a
       strong pull toward 100, regardless of whether the language sounds
       neutral.
       - 0: Pro-Civil Liberties / Reformasi. Strongly champions human
         rights, grassroots activism, civil dissent, and independent
         anti-corruption watchdogs; gives defense and rights-monitor
         voices meaningful space.
       - 50: Neutral or balanced legal/procedural reporting that consults
         multiple sides.
       - 100: Pro-State Authority / Security. Highly sympathetic to the
         military, police, or state security operations, prioritizing
         public order over dissent — including via "Verbatim Police
         Laundering" (uncritical reproduction of police narratives,
         unblurred mugshots, no defense voice, presumption of innocence
         discarded).
    5. `sensationalism` (The Commercial Axis):
       Measure ONLY deliberate use of emotionally charged language,
       clickbait structures, or misleading framing designed to manipulate
       readers into clicking or sharing through outrage or fear.
       Do NOT penalise an article for being short, for covering a
       naturally dramatic event, or for lacking long-form depth — those
       are format and topic choices, not manipulative tactics.
       CRITICAL CALIBRATION RULE: If you find that the headline is
       straightforward, accurate, and not clickbait, the score MUST be
       ≤35 — regardless of topic or article length. A factual headline
       covering a serious or dramatic event is not sensationalism. Only
       score above 50 when you can cite specific language or structural
       choices that are *deliberately* deceptive or outrage-engineered.
       - 0:  Dry, analytical, context-heavy journalism with no emotive
         framing (Kompas print-style).
       - 10: DEFAULT for standard professional newsroom reporting.
         Straightforward, accurate headline; coherent self-contained body.
         Acceptable catchy phrasing does not raise this score. Use 10 for
         any article a reasonable journalist would consider fair and
         professional, even if brief or topically dramatic.
       - 50: Noticeably sensationalist. Uses emotionally charged Indonesian
         words (geger, parah, hancur, emosi) in ways that *exaggerate* the
         stakes beyond the facts, or uses clickbait question framing
         ("Benarkah...?", "Ternyata...!") where the answer is mundane.
       - 75: Strongly sensationalist. Multiple clear clickbait signals:
         misleading or exaggerated headline + emotionally loaded body +
         fragmented listicle/slideshow packaging engineered for churn.
       - 100: Maximal churnalism. Headline is demonstrably false or
         deliberately misleading relative to the body, combined with
         outrage-bait language and fragmented packaging.

    Guiding Rules:
    1. Score objectively. Use 50 as the default baseline for any axis
       except sensationalism, which defaults to 10. Do not deviate from
       the default unless the text actively demonstrates a clear,
       evidence-backed directional pull.
    2. Score conservatively. Reserve extreme scores (less than 15 or
       greater than 85) for blatant, explicit cases. For sensationalism
       specifically: scores above 50 require citing specific manipulative
       language or deliberately misleading headlines — a non-clickbait
       headline is affirmative evidence that the score belongs at or
       below 35, not a neutral data point.
    3. `reliability_index` (0-100) measures sourcing quality (presence of
       verified named sources vs anonymous gossip), factual specificity,
       and the absence of unverified speculation.
    4. Use the provided `publisher_context` (e.g., ownership ties to
       political party chairs or corporate conglomerates) to understand
       *why* an article might lean a certain way, but evaluate the numerical
       scores based strictly on the text evidence.
    5. Emit AT MOST 3 evidence snippets — the 3 strongest, most directly
       bias-revealing quotes. Do not pad with weak or redundant snippets.

    Output format must follow this exact JSON structure:
    {
      "bias_fingerprint": {
        "elite_alignment": integer,
        "identity_lens": integer,
        "economic_sovereignty": integer,
        "state_vs_civil": integer,
        "sensationalism": integer
      },
      "reliability_index": integer,
      "analysis_summary": "string explaining the dominant leanings found and how the context applies",
      "evidence_snippets": [
        {
          "text": "exact phrase or sentence extracted from the article",
          "axis_affected": "string matching one of the 5 axis keys",
          "explanation": "brief sentence explaining why this quote shows that specific lean"
        }
      ]
    }

    Output JSON ONLY. No prose before or after. No code fences.
    """
).strip()


RESPONSE_SCHEMA_DOC = {
    "bias_fingerprint": {
        "elite_alignment": "integer in [0, 100] — Patronage Axis",
        "identity_lens": "integer in [0, 100] — Aliran Axis",
        "economic_sovereignty": "integer in [0, 100] — Market Axis",
        "state_vs_civil": "integer in [0, 100] — Authority Axis",
        "sensationalism": "integer in [0, 100] — Commercial Axis",
    },
    "reliability_index": "integer in [0, 100] — sourcing quality & factual rigour",
    "analysis_summary": (
        "string — explains the dominant leanings found and how "
        "publisher_context applies"
    ),
    "evidence_snippets": (
        "array of {text, axis_affected, explanation}; axis_affected MUST be "
        f"one of: {BIAS_AXES}"
    ),
}


def build_user_prompt(
    article: StoryThreadArticle,
    nlp: NLPFlags,
    publisher_context: str,
) -> str:
    """Render the per-article user message. Keep this deterministic — it's the
    cache key the LLM provider sees."""

    return dedent(
        f"""
        <task>Analyse the following Indonesian news article and return a single
        JSON object matching the schema defined in the system prompt.</task>

        <publisher_context>
        {publisher_context.strip()}
        </publisher_context>

        <article>
          <publisher>{article.publisher}</publisher>
          <url>{article.url}</url>
          <headline>{article.headline}</headline>
          <body>
        {truncate_body(article.body, url=str(article.url))}
          </body>
        </article>

        <local_nlp_signals>
          <clickbait_delta>{nlp.clickbait.clickbait_delta}</clickbait_delta>
          <headline_sentiment>{nlp.clickbait.headline_sentiment}</headline_sentiment>
          <body_sentiment>{nlp.clickbait.body_sentiment}</body_sentiment>
          <charged_adjectives>{nlp.charged_adjectives}</charged_adjectives>
        </local_nlp_signals>

        <instructions>
        Return ONLY the JSON object. No markdown fences, no commentary.
        Every `axis_affected` value in `evidence_snippets` MUST match one of
        the 5 bias_fingerprint axis keys exactly (English, lowercased).
        Emit at most 3 evidence snippets (the strongest).
        Write `analysis_summary` and every `explanation` in Bahasa Indonesia.
        </instructions>
        """
    ).strip()


def build_batch_user_prompt(
    items: list[tuple[StoryThreadArticle, NLPFlags, str]],
) -> str:
    """Render a SINGLE user message covering several articles.

    Each article is wrapped in its own ``<article id="...">`` block so the
    model's output array can be matched back by ``article_id``. The fixed
    system prompt + schema are sent once and amortized across the batch.
    Deterministic ordering — it's the provider's cache key.
    """
    blocks: list[str] = []
    for article, nlp, publisher_context in items:
        blocks.append(
            dedent(
                f"""
                <article id="{article.article_id}">
                  <publisher>{article.publisher}</publisher>
                  <url>{article.url}</url>
                  <headline>{article.headline}</headline>
                  <publisher_context>{publisher_context.strip()}</publisher_context>
                  <body>
                {truncate_body(article.body, url=str(article.url))}
                  </body>
                  <local_nlp_signals>
                    <clickbait_delta>{nlp.clickbait.clickbait_delta}</clickbait_delta>
                    <headline_sentiment>{nlp.clickbait.headline_sentiment}</headline_sentiment>
                    <body_sentiment>{nlp.clickbait.body_sentiment}</body_sentiment>
                    <charged_adjectives>{nlp.charged_adjectives}</charged_adjectives>
                  </local_nlp_signals>
                </article>
                """
            ).strip()
        )

    article_ids = [a.article_id for a, _, _ in items]
    return dedent(
        f"""
        <task>Analyse EACH of the following {len(items)} Indonesian news
        articles independently. Judge each article only on its own text and
        publisher_context — do NOT let one article's framing influence
        another's scores. Return a single JSON object of the form:
        {{"verdicts": [ <one verdict per article, same shape as the system
        prompt schema, PLUS an "article_id" field> ]}}</task>

        <required_article_ids>{json.dumps(article_ids)}</required_article_ids>

        <articles>
        {chr(10).join(blocks)}
        </articles>

        <instructions>
        Return ONLY the JSON object {{"verdicts": [...]}}. No markdown fences,
        no commentary. Produce exactly one verdict per article and set its
        "article_id" to the matching <article id="..."> value. Every
        `axis_affected` MUST be one of the 5 axis keys (English, lowercased).
        Emit at most 3 evidence snippets per article (the strongest).
        Write `analysis_summary` and every `explanation` in Bahasa Indonesia.
        </instructions>
        """
    ).strip()
