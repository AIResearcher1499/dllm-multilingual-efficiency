"""Stage 1: how far apart are the languages on the tokenizer's own axis?

Pure functions; the tokenizer and corpus are injected so the whole stage is
testable without a download. Frozen by docs/prereg-g0.md:

    fertility_ratio(lang) = mean over parallel items of
                            tokens(item, lang) / tokens(item, en)

The ratio is per-item and then averaged, not a ratio of means: a few long
problems would otherwise dominate, and the parallel corpus exists precisely so
that each item can be its own control.
"""

from __future__ import annotations

from dllmfert import MIN_FERTILITY_SPREAD, PIVOT_LANG


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def token_counts(texts: dict[str, str], encode) -> dict[str, int]:
    return {lang: len(encode(t)) for lang, t in texts.items()}


def item_ratios(counts: dict[str, int], pivot: str = PIVOT_LANG) -> dict[str, float]:
    """Per-item token ratio against the pivot language."""
    base = counts.get(pivot)
    if not base:
        return {}
    return {lang: n / base for lang, n in counts.items()}


def fertility_ratios(items: list[dict[str, str]], encode, *, pivot: str = PIVOT_LANG) -> dict:
    """items: list of {lang: parallel_text}. Items missing the pivot are skipped."""
    per_lang: dict[str, list[float]] = {}
    raw: dict[str, list[int]] = {}
    used = 0
    for texts in items:
        counts = token_counts(texts, encode)
        ratios = item_ratios(counts, pivot)
        if not ratios:
            continue
        used += 1
        for lang, r in ratios.items():
            per_lang.setdefault(lang, []).append(r)
            raw.setdefault(lang, []).append(counts[lang])
    return {
        "n_items": used,
        "pivot": pivot,
        "fertility_ratio": {l: _mean(v) for l, v in sorted(per_lang.items())},
        "mean_tokens": {l: _mean(v) for l, v in sorted(raw.items())},
    }


def spread(ratios: dict[str, float]) -> float | None:
    """max/min of the per-language fertility ratio."""
    vals = [v for v in ratios.values() if v]
    if len(vals) < 2:
        return None
    return max(vals) / min(vals)


def stage1_verdict(result: dict) -> dict:
    """Stage-1 gate. A compressed x-axis cannot resolve a slope, whatever the
    GPU stage would show, so this runs first and costs nothing."""
    ratios = result.get("fertility_ratio", {})
    s = spread(ratios)
    if s is None:
        decision = "INCOMPLETE"
        reads = "fewer than two languages measured"
    elif s >= MIN_FERTILITY_SPREAD:
        decision = "PROCEED"
        reads = (
            f"fertility spans {s:.2f}x across languages — wide enough to "
            "regress parallel_factor against"
        )
    else:
        decision = "PARK"
        reads = (
            f"fertility spans only {s:.2f}x (< {MIN_FERTILITY_SPREAD}x). The "
            "x-axis is too compressed; the GPU stage cannot resolve a slope."
        )
    ordered = sorted(ratios.items(), key=lambda kv: kv[1] or 0)
    return {
        "stage": 1,
        "decision": decision,
        "spread": s,
        "threshold": MIN_FERTILITY_SPREAD,
        "n_langs": len(ratios),
        "lowest": ordered[:3],
        "highest": ordered[-3:],
        "reads_as": reads,
    }
