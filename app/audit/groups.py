"""
Gazetteer loading for the counterfactual audit.

Data lives in ``data/groups.id.json`` so the editorial team owns the taxonomy
without touching Python — the arrangement `nlp.py` recommends for its own
lexicons but never got. Taxonomy categories mirror IndoBias
(arXiv:2606.01260): ethnicity, religion, political_party,
government_institution, university.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

_DATA = Path(__file__).parent / "data" / "groups.id.json"


@dataclass(frozen=True)
class Group:
    """One demographic group the scorer might see named in an article."""

    category: str
    name: str
    case_sensitive: bool = False
    not_followed_by: tuple[str, ...] = ()
    #: Words that, when they immediately precede the name, mark a fixed title
    #: collocation that must not be rewritten ("Panglima TNI", "Mabes Polri").
    not_preceded_by: tuple[str, ...] = ()
    #: Tokens that must immediately precede the name for it to count as a
    #: reference to this group (empty = bare mentions count). Used to keep
    #: ethnonyms ("orang Jawa") apart from toponyms ("Jawa Barat").
    require_prefix: tuple[str, ...] = ()


@dataclass(frozen=True)
class Contrast:
    """An ordered pair of same-category groups to swap between."""

    category: str
    a: Group
    b: Group
    #: False for categories whose members are not interchangeable in context —
    #: excluded from the audit unless explicitly opted into. See the
    #: `substitutable` note in the gazetteer JSON.
    substitutable: bool = True

    @property
    def label(self) -> str:
        return f"{self.category}:{self.a.name}->{self.b.name}"


@dataclass
class Gazetteer:
    groups: dict[str, dict[str, Group]] = field(default_factory=dict)
    contrasts: list[Contrast] = field(default_factory=list)
    version: str = "unknown"

    def all_groups(self) -> Iterator[Group]:
        for members in self.groups.values():
            yield from members.values()

    def get(self, category: str, name: str) -> Group:
        return self.groups[category][name]


def load_gazetteer(
    path: Path | None = None, *, include_fragile: bool = False
) -> Gazetteer:
    """Parse the JSON gazetteer. Raises on unknown contrast members so a typo
    fails loudly at load time rather than silently shrinking the audit.

    Categories marked ``"substitutable": false`` are skipped unless
    ``include_fragile`` — their members are not interchangeable in context, so
    swapping them measures incoherence rather than bias.
    """
    raw = json.loads((path or _DATA).read_text(encoding="utf-8"))
    gaz = Gazetteer(version=raw.get("version", "unknown"))

    for category, spec in raw["categories"].items():
        prefixes = tuple(spec.get("require_prefix", ()))
        substitutable = bool(spec.get("substitutable", True))
        members: dict[str, Group] = {}
        for name, opts in spec["members"].items():
            members[name] = Group(
                category=category,
                name=name,
                case_sensitive=bool(opts.get("case_sensitive", False)),
                not_followed_by=tuple(opts.get("not_followed_by", ())),
                not_preceded_by=tuple(opts.get("not_preceded_by", ())),
                require_prefix=prefixes,
            )
        gaz.groups[category] = members

        for a_name, b_name in spec.get("contrasts", []):
            for n in (a_name, b_name):
                if n not in members:
                    raise ValueError(
                        f"contrast member {n!r} is not a member of {category!r}"
                    )
            if not substitutable and not include_fragile:
                continue
            gaz.contrasts.append(
                Contrast(
                    category=category,
                    a=members[a_name],
                    b=members[b_name],
                    substitutable=substitutable,
                )
            )

    return gaz


@lru_cache(maxsize=2)
def default_gazetteer(*, include_fragile: bool = False) -> Gazetteer:
    return load_gazetteer(include_fragile=include_fragile)
