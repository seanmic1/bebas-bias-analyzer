"""
Pre-LLM NLP extraction for Bebas Bias.

Two responsibilities:
  1. Clickbait Delta — compare headline vs. body sentiment intensity.
  2. Charged-adjective flagging — surface emotionally loaded Indonesian terms.

VADER (nltk) is English-tuned, so for the Indonesian sentiment pass we use a
hybrid: VADER on a transliterated/loanword pass *plus* a lexicon-driven score
from a local Indonesian polarity dictionary. The IndoBERT hook below is a
placeholder — when an IndoBERT classifier becomes available, swap
`_indonesian_sentiment` to call it instead. The public function signatures stay
stable, so callers (and the LLM prompt) do not need to change.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List, Set

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

from .schemas import ClickbaitSignal, NLPFlags, StoryThreadArticle


# ---------------------------------------------------------------------------
# Indonesian lexicons
# ---------------------------------------------------------------------------
# Charged adjectives / nouns commonly used in clickbait or polemical reporting.
# This is a curated seed list; in production load from a JSON file managed by
# the editorial team. Lowercased; matching is case-insensitive on word boundary.
CHARGED_ADJECTIVES: Set[str] = {
    # shock / sensation
    "geger", "heboh", "viral", "gempar", "menggemparkan",
    # destruction / disaster framing
    "parah", "hancur", "ambruk", "kacau", "porak-poranda",
    # moral panic
    "biadab", "keji", "sadis", "brutal", "tragis",
    # outrage triggers
    "murka", "marah", "geram", "berang", "naik-pitam",
    # superlatives that signal hype
    "luar-biasa", "fantastis", "mencengangkan", "mengejutkan", "menggila",
    # delegitimizing
    "antek", "kadrun", "cebong", "rezim", "oligarki",
}

# Light Indonesian polarity lexicon for sentiment scoring. Positive (+1) /
# Negative (-1). Trim and expand from InSet or SentiWordNet-ID in production.
ID_POLARITY: dict[str, int] = {
    # positive
    "baik": 1, "bagus": 1, "hebat": 1, "sukses": 1, "untung": 1,
    "damai": 1, "indah": 1, "berhasil": 1, "naik": 1, "tumbuh": 1,
    # negative
    "buruk": -1, "jelek": -1, "gagal": -1, "rugi": -1, "korup": -1,
    "korupsi": -1, "krisis": -1, "anjlok": -1, "turun": -1, "bohong": -1,
    "hoaks": -1, "palsu": -1, "kriminal": -1, "tewas": -1, "mati": -1,
    # the charged ones lean negative too, so seed them
    "parah": -1, "hancur": -1, "kacau": -1, "biadab": -1, "keji": -1,
    "sadis": -1, "brutal": -1, "tragis": -1, "murka": -1, "marah": -1,
}

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ\-']+", re.UNICODE)


@lru_cache(maxsize=1)
def _vader() -> SentimentIntensityAnalyzer:
    """VADER is lazy-loaded so the FastAPI worker boot stays fast."""
    return SentimentIntensityAnalyzer()


def _tokenize(text: str) -> List[str]:
    try:
        return [t.lower() for t in word_tokenize(text)]
    except LookupError:
        # nltk 'punkt' not downloaded — fall back to regex.
        return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _indonesian_sentiment(text: str) -> float:
    """
    Lexicon-driven compound sentiment for Indonesian text in [-1, 1].

    PLACEHOLDER for an IndoBERT classifier — keep the signature
    `(text: str) -> float in [-1, 1]` when swapping the implementation.
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    scored = [ID_POLARITY[t] for t in tokens if t in ID_POLARITY]
    if not scored:
        # Fall back to VADER on the raw text — captures English loanwords
        # ("viral", "drama") that show up in Indonesian online prose.
        return _vader().polarity_scores(text)["compound"]
    # Normalize: average polarity, scaled by lexical density so a long article
    # with one charged word doesn't read as fully polarized.
    density = len(scored) / max(len(tokens), 1)
    avg = sum(scored) / len(scored)
    return max(-1.0, min(1.0, avg * (0.5 + density)))


def compute_clickbait_delta(headline: str, body: str) -> ClickbaitSignal:
    """
    Clickbait Delta = |sentiment(headline)| - |sentiment(body)|.

    Positive values mean the headline is more emotionally intense than the
    body it represents — a classic clickbait signature.
    """
    h = _indonesian_sentiment(headline)
    b = _indonesian_sentiment(body)
    return ClickbaitSignal(
        headline_sentiment=round(h, 4),
        body_sentiment=round(b, 4),
        clickbait_delta=round(abs(h) - abs(b), 4),
    )


def find_charged_adjectives(
    text: str, lexicon: Iterable[str] = CHARGED_ADJECTIVES
) -> List[str]:
    """Return charged terms (deduped, in order of first appearance)."""
    found: List[str] = []
    seen: Set[str] = set()
    lower = text.lower()
    for term in lexicon:
        # Word-boundary match so "parah" doesn't fire on "parahyangan".
        if re.search(rf"\b{re.escape(term)}\b", lower) and term not in seen:
            found.append(term)
            seen.add(term)
    return found


def analyze_article(article: StoryThreadArticle) -> NLPFlags:
    """Run the full pre-LLM NLP pass on a single article."""
    clickbait = compute_clickbait_delta(article.headline, article.body)
    # Charged adjectives are most diagnostic in headlines but we scan both.
    combined = f"{article.headline}\n{article.body}"
    charged = find_charged_adjectives(combined)
    return NLPFlags(
        article_id=article.article_id,
        url=article.url,
        clickbait=clickbait,
        charged_adjectives=charged,
        charged_adjective_count=len(charged),
    )
