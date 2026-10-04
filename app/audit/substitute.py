"""
Group-name substitution inside Indonesian article text.

The audit rests entirely on this being *surgical*. If a swap also mangles
grammar or rewrites an unrelated place name, any score delta we measure is
confounded by incoherence rather than bias, and the result is worthless.

Four guards, all declared in the gazetteer rather than hardcoded here:

  * ``require_prefix`` — only rewrite a name carrying an explicit ethnonym
    marker ("orang Jawa", "suku Dayak"), so "Jawa Barat" is never touched.
  * ``not_preceded_by`` — block fixed title collocations. "Panglima TNI" is an
    office; rewriting it to "Panglima Komnas HAM" produces a body that names a
    post which does not exist.
  * ``not_followed_by`` — the mirror guard, for compound toponyms.
  * ``case_sensitive`` — short acronyms (UI, TNI) match exactly, so they never
    fire inside unrelated words or English text.

Capitalisation of the replacement follows the token it replaces, so
"orang jawa" -> "orang papua" and "orang Jawa" -> "orang Papua".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .groups import Group


@dataclass(frozen=True)
class Substitution:
    """Result of rewriting one group's mentions in a piece of text."""

    text: str
    count: int
    #: Mentions that matched the name but were rejected by a guard. Surfaced so
    #: a badly-tuned gazetteer shows up as "found nothing" rather than silence.
    blocked: int = 0


def _match_case(template: str, replacement: str) -> str:
    """Echo ``template``'s capitalisation onto ``replacement``.

    Only for like-for-like shapes: a lowercase ethnonym stays lowercase, an
    acronym swapped for another acronym stays upper. A multi-word proper name
    always keeps its canonical casing — "UI" -> "Universitas Cenderawasih",
    never "UNIVERSITAS CENDERAWASIH".
    """
    single_token = " " not in replacement
    if template.islower() and single_token:
        return replacement.lower()
    if template.isupper() and replacement.isupper():
        return replacement.upper()
    return replacement


@lru_cache(maxsize=512)
def _pattern(group: Group) -> re.Pattern[str]:
    """Compile the mention pattern for ``group``.

    The preceding word is always captured as ``pre`` so the prefix and
    title-collocation guards can be applied in Python — a negative lookbehind
    cannot be used because the blocked terms differ in length.
    """
    name = re.escape(group.name).replace(r"\ ", r"\s+")
    parts = [r"(?:(?P<pre>[\w'-]+)\s+)?", rf"(?P<name>\b{name}\b)"]

    if group.not_followed_by:
        blocked = "|".join(re.escape(t) for t in group.not_followed_by)
        parts.append(rf"(?!\s+(?:{blocked})\b)")

    flags = 0 if group.case_sensitive else re.IGNORECASE
    return re.compile("".join(parts), flags)


def _allowed(match: re.Match[str], group: Group) -> bool:
    """Does this mention survive the prefix / collocation guards?"""
    pre = (match.group("pre") or "").lower()
    if group.require_prefix and pre not in {p.lower() for p in group.require_prefix}:
        return False
    if pre and pre in {t.lower() for t in group.not_preceded_by}:
        return False
    return True


def count_mentions(text: str, group: Group) -> int:
    """How many times ``group`` is referred to in ``text``, guards applied."""
    return sum(1 for m in _pattern(group).finditer(text or "") if _allowed(m, group))


def substitute(text: str, src: Group, dst: Group) -> Substitution:
    """Rewrite every guard-approved mention of ``src`` in ``text`` as ``dst``."""
    swapped = 0
    blocked = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal swapped, blocked
        if not _allowed(m, src):
            blocked += 1
            return m.group(0)
        swapped += 1
        matched = m.group("name")
        head = m.group(0)[: m.start("name") - m.start()]
        return head + _match_case(matched, dst.name)

    return Substitution(
        text=_pattern(src).sub(_sub, text or ""), count=swapped, blocked=blocked
    )
