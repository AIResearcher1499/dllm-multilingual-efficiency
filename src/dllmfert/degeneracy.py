"""Degeneracy metrics and the statistics Phase 3 is judged by.

Pure Python on purpose: the quantities that decide whether this paper has a
claim must be runnable and testable without a GPU, like the sampler.

Every cutoff here is **relative to a model's own distribution**. An absolute
char-distinct threshold calibrated on Dream (0.75) labels LLaDA's *English*
degenerate, because LLaDA is more repetitive at every language. Absolute
thresholds do not transfer across models and must not appear in this repo.
"""

from __future__ import annotations

import math
import re
from collections import Counter

SHINGLE_K = 8


def shingles(text: str, k: int = SHINGLE_K) -> list[str]:
    """Character k-shingles over whitespace-normalised text.

    Character rather than token shingles so the measure is comparable across
    tokenisers and scripts -- a token-level measure would be contaminated by
    exactly the fertility differences under study.
    """
    t = re.sub(r"\s+", " ", text.strip())
    if len(t) < k:
        return [t] if t else []
    return [t[i:i + k] for i in range(len(t) - k + 1)]


def distinctness(text: str, k: int = SHINGLE_K) -> float:
    """Fraction of shingles that are unique. 1.0 = no repetition at all."""
    s = shingles(text, k)
    if not s:
        return 1.0
    return len(Counter(s)) / len(s)


def max_repeat(text: str, k: int = SHINGLE_K) -> int:
    """Occurrences of the most repeated shingle. The competing cheap signal
    that `parallel_factor` has to beat for M1 to be worth writing up."""
    s = shingles(text, k)
    if not s:
        return 0
    return max(Counter(s).values())


def relative_cutoff(values: list[float], pct: float = 20.0) -> float:
    """Percentile of a model's own distribution. See the module docstring for
    why nothing here is allowed to be an absolute constant."""
    if not values:
        raise ValueError("no values")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (pct / 100.0) * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_ranks(x), _ranks(y))


def partial_spearman(x: list[float], y: list[float], z: list[float]) -> float:
    """Spearman(x, y) with z partialled out.

    P1 requires this: a repetition penalty changes how much text is produced,
    and `content_len` sits in the numerator of `parallel_factor`, so an
    uncontrolled correlation cannot separate the mechanism from the length.
    """
    rxy = spearman(x, y)
    if len(set(z)) <= 1:
        # A control with no variance controls for nothing; the partial
        # correlation is the plain one. Returning nan here would push a
        # perfectly clean result into the "confounded" branch.
        return rxy
    rxz, ryz = spearman(x, z), spearman(y, z)
    den = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den else float("nan")


def auc(scores: list[float], labels: list[bool]) -> float:
    """ROC AUC via the Mann-Whitney U statistic, ties counted as half.

    Returns nan when one class is absent -- with no positives there is nothing
    to detect and reporting 0.5 would read as a measured result.
    """
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    r = _ranks(scores)
    rank_pos = sum(rr for rr, l in zip(r, labels) if l)
    u = rank_pos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))
