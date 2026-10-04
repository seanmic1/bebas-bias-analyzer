"""Unit tests for the counterfactual bias audit (app/audit/).

No network, no Postgres, no NLTK: the scorer is a fake LLMClient with planted
scores and analyze_article is monkeypatched, same approach as test_batch.py.
Covered:
  (a) substitution guards — ethnonym prefix, compound toponyms, acronym case;
  (b) gazetteer integrity, including a loud failure on a typo'd contrast;
  (c) minimal-pair construction, direction normalisation, both-sides skip;
  (d) effect-vs-noise maths, including that noise-only data is not flagged;
  (e) an end-to-end run that recovers a planted bias and one that finds none.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.audit import counterfactual as cf_mod
from app.audit.counterfactual import CounterfactualAudit, build_pairs
from app.audit.groups import Gazetteer, default_gazetteer, load_gazetteer
from app.audit.metrics import AUDIT_METRICS, aggregate, mean_pairwise_abs, score_effect
from app.audit.substitute import count_mentions, substitute
from app.schemas import (
    ArticleVerdict,
    BiasFingerprint,
    ClickbaitSignal,
    NLPFlags,
    StoryThreadArticle,
)

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_GAZ = default_gazetteer()


def _only(label_suffix: str) -> Gazetteer:
    """A gazetteer narrowed to one contrast.

    A name like "Jawa" participates in several contrasts (->Papua, ->Tionghoa),
    so an article naming it yields several minimal pairs. Tests that assert on
    a specific swap must narrow first.
    """
    return Gazetteer(
        groups=_GAZ.groups,
        contrasts=[c for c in _GAZ.contrasts if c.label.endswith(label_suffix)],
        version=_GAZ.version,
    )


def _article(aid: str, headline: str, body: str) -> StoryThreadArticle:
    return StoryThreadArticle(
        article_id=aid,
        url=f"https://example.com/{aid}",
        publisher="example.com",
        headline=headline,
        body=body,
        published_at=_NOW,
        scraped_at=_NOW,
        word_count=len(body.split()),
    )


def _fake_flags(article: StoryThreadArticle) -> NLPFlags:
    return NLPFlags(
        article_id=article.article_id,
        url=article.url,
        clickbait=ClickbaitSignal(
            headline_sentiment=0.0, body_sentiment=0.0, clickbait_delta=0.0
        ),
        charged_adjectives=[],
        charged_adjective_count=0,
    )


@pytest.fixture(autouse=True)
def _stub_nlp(monkeypatch):
    monkeypatch.setattr(cf_mod, "_nlp_flags", _fake_flags)


class _FakeClient:
    """Scores by a rule over the article text, so bias can be planted exactly."""

    model = "fake-model"

    def __init__(self, rule=None) -> None:
        self.rule = rule or (lambda text: {})
        self.calls = 0

    def analyze(self, article, nlp, publisher_context) -> ArticleVerdict:
        self.calls += 1
        text = f"{article.headline}\n{article.body}"
        base = {axis: 50 for axis in AUDIT_METRICS}
        base.update(self.rule(text))
        reliability = base.pop("reliability_index")
        return ArticleVerdict(
            article_id=article.article_id,
            url=article.url,
            publisher=article.publisher,
            bias_fingerprint=BiasFingerprint(**base),
            reliability_index=reliability,
            analysis_summary="uji",
        )


# ---------------------------------------------------------------------------
# Substitution guards
# ---------------------------------------------------------------------------
def test_ethnonym_prefix_required_so_province_names_survive():
    jawa, papua = _GAZ.get("ethnicity", "Jawa"), _GAZ.get("ethnicity", "Papua")
    text = "Warga Jawa Barat menolak. Orang Jawa di sana diam."
    assert count_mentions(text, jawa) == 1
    out = substitute(text, jawa, papua).text
    assert "Jawa Barat" in out
    assert "Orang Papua" in out


def test_case_is_echoed_only_for_like_shapes():
    jawa, papua = _GAZ.get("ethnicity", "Jawa"), _GAZ.get("ethnicity", "Papua")
    assert substitute("suku jawa", jawa, papua).text == "suku papua"
    assert substitute("suku Jawa", jawa, papua).text == "suku Papua"
    # An acronym swapped for a multi-word name keeps canonical casing.
    ui = _GAZ.get("university", "UI")
    unc = _GAZ.get("university", "Universitas Cenderawasih")
    assert substitute("Mahasiswa UI", ui, unc).text == "Mahasiswa Universitas Cenderawasih"


def test_case_sensitive_acronym_does_not_match_inside_words():
    ui = _GAZ.get("university", "UI")
    assert count_mentions("Bui itu penuh dan ui kecil", ui) == 0


def test_substitution_count_reports_every_mention():
    pdip, pks = _GAZ.get("political_party", "PDIP"), _GAZ.get("political_party", "PKS")
    sub = substitute("Kader PDIP dan pengurus PDIP hadir.", pdip, pks)
    assert sub.count == 2
    assert "PDIP" not in sub.text


def test_fixed_office_titles_are_not_rewritten():
    # "Panglima TNI" is an office; Komnas HAM has no Panglima, so rewriting it
    # would fabricate a post and confound the measurement with incoherence.
    gaz = default_gazetteer(include_fragile=True)
    tni = gaz.get("government_institution", "TNI")
    komnas = gaz.get("government_institution", "Komnas HAM")
    sub = substitute("Panglima TNI hadir. Prajurit TNI berjaga.", tni, komnas)
    assert "Panglima TNI" in sub.text
    assert sub.count == 1 and sub.blocked == 1


def test_blocked_mentions_are_not_counted_as_coverage():
    gaz = default_gazetteer(include_fragile=True)
    tni = gaz.get("government_institution", "TNI")
    assert count_mentions("Panglima TNI hadir.", tni) == 0


# ---------------------------------------------------------------------------
# Gazetteer
# ---------------------------------------------------------------------------
def test_non_substitutable_categories_are_opt_in():
    default = {c.category for c in default_gazetteer().contrasts}
    fragile = {c.category for c in default_gazetteer(include_fragile=True).contrasts}
    assert "government_institution" not in default
    assert "government_institution" in fragile



def test_gazetteer_contrasts_reference_real_members():
    for contrast in _GAZ.contrasts:
        assert contrast.a.name in _GAZ.groups[contrast.category]
        assert contrast.b.name in _GAZ.groups[contrast.category]
        assert contrast.a.name != contrast.b.name


def test_typo_in_contrast_fails_loudly(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "version": "t",
                "categories": {
                    "ethnicity": {
                        "members": {"Jawa": {}},
                        "contrasts": [["Jawa", "Jawwa"]],
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Jawwa"):
        load_gazetteer(bad)


# ---------------------------------------------------------------------------
# Minimal pairs
# ---------------------------------------------------------------------------
def test_pair_direction_is_normalised_regardless_of_which_side_appears():
    a = _article("1", "Demo", "Orang Jawa berdemo di depan gedung.")
    b = _article("2", "Demo", "Orang Papua berdemo di depan gedung.")
    pairs = {p.article_id: p for p in build_pairs([a, b], _only("Jawa->Papua"))}

    # Whichever side the original names, variant_a always names Jawa.
    for p in pairs.values():
        assert "Jawa" in p.variant_a.body
        assert "Papua" in p.variant_b.body
    assert pairs["1"].original_side == "a"
    assert pairs["2"].original_side == "b"


def test_article_naming_both_sides_is_skipped():
    both = _article("3", "Demo", "Orang Jawa dan orang Papua sama-sama hadir.")
    labels = [p.label for p in build_pairs([both], _GAZ)]
    assert "ethnicity:Jawa->Papua" not in labels


def test_one_name_can_yield_several_contrasts():
    art = _article("9", "Sidang", "Orang Jawa menjadi saksi.")
    labels = {p.label for p in build_pairs([art], _GAZ)}
    assert {"ethnicity:Jawa->Papua", "ethnicity:Jawa->Tionghoa"} <= labels


def test_pairs_only_differ_by_the_swapped_name():
    art = _article("4", "Sidang", "Orang Jawa menjadi saksi utama sidang itu.")
    pair = next(p for p in build_pairs([art], _GAZ) if p.label.endswith("Jawa->Papua"))
    assert pair.variant_a.body.replace("Jawa", "X") == pair.variant_b.body.replace(
        "Papua", "X"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_noise_floor_is_zero_for_identical_scores():
    assert mean_pairwise_abs([50, 50, 50]) == 0.0
    assert mean_pairwise_abs([50]) == 0.0


def test_signed_effect_keeps_direction_and_abs_does_not():
    up = score_effect("state_vs_civil", [50, 50], [60, 60])
    down = score_effect("state_vs_civil", [60, 60], [50, 50])
    assert up.signed_effect == 10.0 and down.signed_effect == -10.0
    assert up.abs_effect == down.abs_effect == 10.0


def test_pure_noise_is_not_flagged():
    # Same distribution either side; the mean difference is small next to spread.
    e = score_effect("sensationalism", [40, 50, 60], [60, 50, 40])
    rolled = aggregate([e])[0]
    assert rolled.abs_effect == 0.0
    assert not rolled.flagged


def test_consistent_shift_above_noise_is_flagged():
    e = score_effect("identity_lens", [50, 51, 49], [62, 61, 63])
    rolled = aggregate([e])[0]
    assert rolled.signed_effect == 12.0
    assert rolled.flagged


def test_aggregate_preserves_audit_metric_order():
    effects = [score_effect(axis, [50, 50], [55, 55]) for axis in reversed(AUDIT_METRICS)]
    assert [a.axis for a in aggregate(effects)] == AUDIT_METRICS


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_repeats_below_two_is_rejected():
    with pytest.raises(ValueError, match="noise floor"):
        CounterfactualAudit(client=_FakeClient(), repeats=1)


def test_plan_reports_call_budget_without_touching_the_model():
    art = _article("5", "Demo", "Orang Jawa berdemo.")
    audit = CounterfactualAudit(repeats=3, gazetteer=_GAZ)
    plan = audit.plan([art])
    assert plan["pairs"] >= 1
    assert plan["llm_calls"] == plan["pairs"] * 3 * 2


def test_run_recovers_a_planted_bias():
    # A scorer that pushes state_vs_civil up whenever "Papua" is named.
    client = _FakeClient(lambda t: {"state_vs_civil": 70} if "Papua" in t else {})
    art = _article("6", "Demo", "Orang Jawa berdemo di depan gedung DPR.")
    audit = CounterfactualAudit(
        client=client,
        repeats=2,
        gazetteer=_only("Jawa->Papua"),
        output_dir=Path("/tmp"),
    )
    report = asyncio.run(audit.run([art]))

    assert report["pairs_failed"] == 0
    by_axis = {r["axis"]: r for r in report["overall"]}
    assert by_axis["state_vs_civil"]["signed_effect"] == 20.0
    assert by_axis["state_vs_civil"]["flagged"]
    # Untouched axes must stay clean — no false positives.
    assert by_axis["economic_sovereignty"]["signed_effect"] == 0.0
    assert not by_axis["economic_sovereignty"]["flagged"]


def test_run_on_a_fair_scorer_flags_nothing():
    client = _FakeClient()  # identical scores regardless of text
    art = _article("7", "Sidang", "Orang Jawa menjadi saksi.")
    audit = CounterfactualAudit(
        client=client, repeats=2, gazetteer=_GAZ, output_dir=Path("/tmp")
    )
    report = asyncio.run(audit.run([art]))
    assert report["pairs_measured"] >= 1
    assert not any(r["flagged"] for r in report["overall"])


def test_one_failing_pair_does_not_sink_the_audit():
    class _Boom(_FakeClient):
        def analyze(self, article, nlp, publisher_context):
            raise RuntimeError("upstream 500")

    art = _article("8", "Demo", "Orang Jawa berdemo.")
    audit = CounterfactualAudit(client=_Boom(), repeats=2, gazetteer=_GAZ)
    report = asyncio.run(audit.run([art]))
    assert report["pairs_measured"] == 0
    assert report["pairs_failed"] >= 1
    assert "upstream 500" in report["pairs"][0]["error"]
