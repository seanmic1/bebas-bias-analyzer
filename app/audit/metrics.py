"""
Effect-vs-noise statistics for the counterfactual audit.

The whole audit turns on one methodological point: the scorer is
nondeterministic, so a raw A/B score difference means nothing on its own. A
delta of 6 points is damning if repeated scorings of the *same* text vary by 1,
and meaningless if they vary by 8.

So every variant is scored ``repeats`` times and we report three numbers per
axis:

  ``signed_effect``  mean(B) - mean(A), aggregated across articles. This is the
                     bias statistic — it answers "does swapping Jawa for Papua
                     systematically *raise* state_vs_civil?". Signed, because
                     averaging absolute values would hide cancellation and
                     manufacture an effect out of pure noise.
  ``abs_effect``     mean |mean(B) - mean(A)| per article. Magnitude regardless
                     of direction; always >= |signed_effect|.
  ``noise``          mean within-variant pairwise spread. The floor that
                     ``abs_effect`` must clear to mean anything.

``ratio = abs_effect / noise``. Below ~1 the audit found nothing.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from ..schemas import BIAS_AXES

#: Everything the audit tracks — the five bias axes plus overall reliability,
#: which is just as susceptible to a demographic prior as the axes are.
AUDIT_METRICS: list[str] = [*BIAS_AXES, "reliability_index"]

#: An abs_effect this many times the noise floor is worth a human look.
FLAG_RATIO: float = 1.5


def mean_pairwise_abs(values: list[float]) -> float:
    """Mean |x_i - x_j| over all unordered pairs — the within-variant spread.

    Returns 0.0 for fewer than two samples (no noise estimate available).
    """
    n = len(values)
    if n < 2:
        return 0.0
    diffs = [
        abs(values[i] - values[j]) for i in range(n) for j in range(i + 1, n)
    ]
    return sum(diffs) / len(diffs)


@dataclass
class AxisEffect:
    """Per-axis effect for a single article/contrast."""

    axis: str
    mean_a: float
    mean_b: float
    signed_effect: float
    abs_effect: float
    noise: float

    @property
    def ratio(self) -> float:
        return self.abs_effect / self.noise if self.noise else float("inf") if self.abs_effect else 0.0


def score_effect(axis: str, a: list[float], b: list[float]) -> AxisEffect:
    """Compare repeated scorings of variant A against variant B on one axis."""
    mean_a = statistics.fmean(a) if a else 0.0
    mean_b = statistics.fmean(b) if b else 0.0
    signed = mean_b - mean_a
    # Noise floor pools both variants — they are the same text modulo the
    # swapped name, so their sampling spread is the same quantity measured
    # twice.
    noise = statistics.fmean([mean_pairwise_abs(a), mean_pairwise_abs(b)])
    return AxisEffect(
        axis=axis,
        mean_a=round(mean_a, 2),
        mean_b=round(mean_b, 2),
        signed_effect=round(signed, 2),
        abs_effect=round(abs(signed), 2),
        noise=round(noise, 2),
    )


@dataclass
class AggregateEffect:
    """Per-axis rollup across every article measured for one contrast."""

    axis: str
    n: int
    signed_effect: float
    abs_effect: float
    noise: float
    ratio: float
    flagged: bool

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate(effects: list[AxisEffect]) -> list[AggregateEffect]:
    """Roll per-article effects up per axis, preserving AUDIT_METRICS order."""
    by_axis: dict[str, list[AxisEffect]] = {}
    for e in effects:
        by_axis.setdefault(e.axis, []).append(e)

    out: list[AggregateEffect] = []
    for axis in AUDIT_METRICS:
        rows = by_axis.get(axis)
        if not rows:
            continue
        signed = statistics.fmean([r.signed_effect for r in rows])
        abs_eff = statistics.fmean([r.abs_effect for r in rows])
        noise = statistics.fmean([r.noise for r in rows])
        ratio = abs_eff / noise if noise else (float("inf") if abs_eff else 0.0)
        out.append(
            AggregateEffect(
                axis=axis,
                n=len(rows),
                signed_effect=round(signed, 2),
                abs_effect=round(abs_eff, 2),
                noise=round(noise, 2),
                ratio=round(ratio, 2) if ratio != float("inf") else ratio,
                flagged=ratio >= FLAG_RATIO,
            )
        )
    return out
