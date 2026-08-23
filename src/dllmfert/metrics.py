"""Stage 2 metrics and the frozen G0 verdict. Thresholds: docs/prereg-g0.md.

`parallel_factor` is the crux: tokens finalised per denoising step. It is what
the published speedups are literally made of, and unlike the predecessor's stop
margin it has real dynamic range (1 to >2 tokens/step in the literature).
"""

from __future__ import annotations

from dllmfert import MIN_ITEMS, MIN_LANGS, MIN_PARALLEL_SPREAD


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def parallel_factor(finalised_per_step: list[int]) -> float | None:
    """Tokens finalised per denoising step over one generation.

    Equivalent to tokens_generated / NFE, but computed from the per-step
    record so that a sampler which finalises nothing on some steps is not
    silently averaged away.
    """
    if not finalised_per_step:
        return None
    return sum(finalised_per_step) / len(finalised_per_step)


def padding_waste(canvas: int, tokens_generated: int) -> float | None:
    if not canvas:
        return None
    return max(0.0, (canvas - tokens_generated) / canvas)


def by_language(rows: list[dict], arm: str) -> dict[str, dict]:
    """Per-language aggregates for one model arm, error rows excluded."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("arm") != arm or r.get("error"):
            continue
        out.setdefault(r["lang"], []).append(r)
    return {
        lang: {
            "n": len(rs),
            "parallel_factor": _mean([r["parallel_factor"] for r in rs
                                      if r.get("parallel_factor") is not None]),
            "nfe": _mean([r["nfe"] for r in rs if r.get("nfe") is not None]),
            "wall_clock": _mean([r["wall_clock"] for r in rs
                                 if r.get("wall_clock") is not None]),
            "accuracy": _mean([bool(r.get("acc")) for r in rs]),
            "padding_waste": _mean([r["padding_waste"] for r in rs
                                    if r.get("padding_waste") is not None]),
        }
        for lang, rs in sorted(out.items())
    }


def relative_spread(values: list[float]) -> float | None:
    """(max - min) / min. The prereg's effect-size measure."""
    vals = [v for v in values if v]
    if len(vals) < 2 or min(vals) == 0:
        return None
    return (max(vals) - min(vals)) / min(vals)


def _rank_agreement(pairs: list[tuple[float, float]]) -> float | None:
    """Fraction of language pairs whose parallel_factor ordering matches the
    fertility ordering. 1.0 means perfectly monotone, 0.0 perfectly inverted.

    Used instead of a correlation coefficient because with ~8 languages a
    Pearson r is noise, whereas 'does the ranking agree' is directly the
    monotonicity the prereg asks for.
    """
    n = 0
    agree = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (f1, p1), (f2, p2) = pairs[i], pairs[j]
            if f1 == f2 or p1 == p2:
                continue
            n += 1
            if (f1 < f2) == (p1 < p2):
                agree += 1
    return agree / n if n else None


def g0_verdict(rows: list[dict], fertility: dict[str, float], arm: str) -> dict:
    """Frozen: magnitude gates, direction only labels the result."""
    agg = by_language(rows, arm)
    scored = {l: a for l, a in agg.items()
              if a["n"] >= MIN_ITEMS and a["parallel_factor"] is not None}
    if len(scored) < MIN_LANGS:
        return {
            "arm": arm,
            "decision": "INCOMPLETE",
            "n_langs_scored": len(scored),
            "need": MIN_LANGS,
            "min_items": MIN_ITEMS,
        }
    pairs = [(fertility[l], a["parallel_factor"])
             for l, a in scored.items() if l in fertility]
    if len(pairs) < MIN_LANGS:
        return {"arm": arm, "decision": "INCOMPLETE",
                "reason": "fertility missing for scored languages",
                "n_paired": len(pairs)}
    spread = relative_spread([p for _, p in pairs])
    agreement = _rank_agreement(pairs)
    monotone = agreement is not None and (agreement >= 0.75 or agreement <= 0.25)
    big = spread is not None and spread >= MIN_PARALLEL_SPREAD
    if big and monotone:
        decision = "GO"
        direction = "B" if agreement >= 0.75 else "A"
        reads = (
            "high fertility RAISES the parallel factor — direction B, the "
            "paradigm contrast against speculative decoding"
            if direction == "B" else
            "high fertility LOWERS the parallel factor — direction A, same "
            "conclusion as 2605.30580; needs a large effect to carry a paper"
        )
    elif big:
        decision, direction = "UNDECIDED", None
        reads = (f"spread {spread:.0%} is large but not monotone "
                 f"(rank agreement {agreement:.2f}) — something else varies")
    else:
        decision, direction = "PARK", None
        reads = (f"spread {spread:.0%} < {MIN_PARALLEL_SPREAD:.0%}: script "
                 "moves the parallel factor, but not by an amount that matters")
    return {
        "arm": arm,
        "decision": decision,
        "direction": direction,
        "n_langs": len(pairs),
        "parallel_spread": spread,
        "rank_agreement": agreement,
        "threshold": MIN_PARALLEL_SPREAD,
        "by_language": scored,
        "reads_as": reads,
    }


# ---------------------------------------------------------------------------
# End-to-end cost ratio. Added 2026-08-21 after preflight timing on an A100
# showed the original single metric could not carry the question.
#
# Measured there: a diffusion forward is NOT constant-time. t(L) = a + b*L,
# with a = 15.4 ms and b = 0.102 ms/token for Dream-7B -- a 6.4x larger canvas
# costs 4.4x more per step. A diffusion step processes the whole canvas, so it
# behaves like a prefill, not like an autoregressive decode step.
#
# That means the two competing mechanisms act on different quantities:
#   A (canvas cost)          -> time per forward, via canvas ∝ fertility
#   B (intra-word redundancy)-> parallel_factor, via confidence
# and parallel_factor alone tests B while ignoring A. The quantity that
# contains both, and the one a practitioner actually feels, is
#
#     cost_ratio(lang) = total dLLM seconds / total AR seconds
#
# with, for large fertility f,   cost_ratio ∝ f / parallel_factor(f).
# The exponent k in cost_ratio ∝ f^k is therefore a direct read-out of how far
# B offsets A:  k≈1 means B is absent, k≈0 means B fully compensates.
# ---------------------------------------------------------------------------

MIN_SLOPE_SE = 0.15      # k must be pinned to +-0.3 at 2 sigma
MIN_R2 = 0.5


def cost_ratio_by_language(rows: list[dict], *, dllm_arm: str, ar_arm: str,
                           mode: str = "threshold",
                           require_lang_pass: bool = True) -> dict[str, dict]:
    """Total diffusion seconds over total autoregressive seconds, per language.

    A ratio of totals, not a mean of per-item ratios: the practitioner-facing
    question is how long the whole benchmark takes, and a ratio of totals is
    not dominated by short items where a fixed overhead swamps the signal.
    Restricted to item ids present in both arms so the comparison stays paired.
    """
    def usable(r):
        if r.get("error"):
            return False
        # An item where either arm answered in the wrong language is not a
        # harder or easier version of the same job -- it is a different job.
        # Qwen scored 2/3 on Telugu by answering in English, so accuracy cannot
        # be the filter; this has to be.
        return (not require_lang_pass) or bool(r.get("lang_pass"))

    out: dict[str, dict] = {}
    langs = {r["lang"] for r in rows}
    for lang in sorted(langs):
        d = {r["id"]: r for r in rows
             if r["lang"] == lang and r["arm"] == dllm_arm
             and r["mode"] == mode and usable(r)}
        a = {r["id"]: r for r in rows
             if r["lang"] == lang and r["arm"] == ar_arm and usable(r)}
        ids = sorted(set(d) & set(a))
        if not ids:
            continue
        dt = sum(d[i]["wall_clock"] for i in ids)
        at = sum(a[i]["wall_clock"] for i in ids)
        attempted_d = sum(1 for r in rows if r["lang"] == lang
                          and r["arm"] == dllm_arm and r["mode"] == mode
                          and not r.get("error"))
        attempted_a = sum(1 for r in rows if r["lang"] == lang
                          and r["arm"] == ar_arm and not r.get("error"))
        out[lang] = {
            "n": len(ids),
            "lang_drop_rate": {
                "dllm": 1 - len(d) / attempted_d if attempted_d else None,
                "ar": 1 - len(a) / attempted_a if attempted_a else None,
            },
            "dllm_seconds": dt,
            "ar_seconds": at,
            "cost_ratio": (dt / at) if at else None,
            "mean_parallel_factor": _mean(
                [d[i]["parallel_factor"] for i in ids
                 if d[i].get("parallel_factor") is not None]),
            "mean_tokens_dllm": _mean([d[i]["tokens_generated"] for i in ids
                                       if d[i].get("tokens_generated") is not None]),
            "mean_padding": _mean([d[i]["padding_positions"] for i in ids
                                   if d[i].get("padding_positions") is not None]),
            "stop_rate": _mean([bool(d[i].get("stopped")) for i in ids]),
        }
    return out


def _ols_loglog(xs: list[float], ys: list[float]) -> dict | None:
    """Slope of log y on log x, with the standard error and R^2.

    Hand-rolled so the fertility stage keeps running on a laptop with nothing
    but a tokenizer installed.
    """
    import math

    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    n = len(pts)
    if n < 3:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    k = sxy / sxx
    c = my - k * mx
    resid = [p[1] - (c + k * p[0]) for p in pts]
    sse = sum(r * r for r in resid)
    sst = sum((p[1] - my) ** 2 for p in pts)
    se = math.sqrt(sse / (n - 2) / sxx) if n > 2 and sse > 0 else 0.0
    return {"k": k, "intercept": c, "se": se,
            "r2": (1 - sse / sst) if sst > 0 else 1.0, "n": n}


def slope_verdict(by_lang: dict[str, dict], fertility: dict[str, float]) -> dict:
    """Primary result: how does the diffusion-vs-AR cost ratio scale with script?"""
    pairs = [(fertility[l], v["cost_ratio"]) for l, v in by_lang.items()
             if l in fertility and v.get("cost_ratio")]
    fit = _ols_loglog([f for f, _ in pairs], [r for _, r in pairs])
    if fit is None:
        return {"decision": "INCOMPLETE", "reason": "fewer than three languages"}
    k, se, r2 = fit["k"], fit["se"], fit["r2"]
    precise = se <= MIN_SLOPE_SE and r2 >= MIN_R2
    if not precise:
        decision = "PARK"
        reads = (f"k = {k:.2f} +- {se:.2f}, R2 = {r2:.2f}: too imprecise to tell "
                 "mechanism A alone (k=1) from full compensation (k=0). The "
                 "measurement cannot answer the question, whatever the value.")
    elif k >= 0.75:
        decision, reads = "GO", (
            f"k = {k:.2f} +- {se:.2f}: the diffusion penalty grows about "
            "linearly with fertility. Mechanism B does not compensate; every "
            "published dLLM speedup is an artefact of English tokenisation.")
    elif k <= 0.25:
        decision, reads = "GO", (
            f"k = {k:.2f} +- {se:.2f}: script barely moves the trade-off. "
            "Intra-word redundancy compensates for the larger canvas, which "
            "contradicts the speculative-decoding result in 2605.30580.")
    else:
        decision, reads = "GO", (
            f"k = {k:.2f} +- {se:.2f}: partial compensation. Report the "
            "decomposition -- how much of the canvas cost parallel_factor buys "
            "back -- rather than a direction.")
    ratios = sorted(((v["cost_ratio"], l) for l, v in by_lang.items()
                     if v.get("cost_ratio")))
    return {
        "decision": decision,
        "k": k, "se": se, "r2": r2, "n_langs": fit["n"],
        "thresholds": {"max_se": MIN_SLOPE_SE, "min_r2": MIN_R2},
        "cheapest": ratios[:2], "dearest": ratios[-2:],
        "spread": (ratios[-1][0] / ratios[0][0]) if ratios else None,
        "crossover": [l for r, l in ratios if r <= 1.0][-1:] or None,
        "reads_as": reads,
    }


# Frozen in prereg Revision 5, before LLaDA's grid was run.
TOKENISATION_R2_MARGIN = 0.15
# The rule is a conjunction, and these are the languages it named. They are
# hardcoded because they were chosen in advance for being where the two
# tokenisers disagree most (LLaDA charges 1.50x Qwen for ru, 1.68x for th and
# 0.82x for zh); picking them from the data would defeat the whole design.
TOKENISATION_POSITIVE_RESIDUAL = ("ru", "th")
TOKENISATION_NEGATIVE_RESIDUAL = ("zh",)


def tokenisation_verdict(by_lang: dict[str, dict], own_fertility: dict[str, float],
                         other_fertility: dict[str, float]) -> dict:
    """Revision 5's discriminating test: tokenisation (T) or language (L)?

    Fertility and language are perfectly confounded inside one tokeniser. A
    second tokeniser that ranks the same languages differently breaks the
    confound: T predicts an arm's cost ratio tracks **its own** tokeniser, L
    predicts the tokeniser is irrelevant because the difficulty lives in the
    language.

    The rule is frozen and this function does not choose anything -- including
    the case the rule did not anticipate. If the two fertility tables agree too
    closely, the test has no power to separate them, and reporting a winner off
    an R^2 gap of a few thousandths would be reading noise. That is reported as
    NO_POWER, not as a result.
    """
    langs = [l for l, v in by_lang.items()
             if v.get("cost_ratio") and l in own_fertility and l in other_fertility]
    ratios = [by_lang[l]["cost_ratio"] for l in langs]
    fit_own = _ols_loglog([own_fertility[l] for l in langs], ratios)
    fit_other = _ols_loglog([other_fertility[l] for l in langs], ratios)
    if fit_own is None or fit_other is None:
        return {"decision": "INCOMPLETE", "reason": "fewer than three languages"}

    agreement = _pearson_log([own_fertility[l] for l in langs],
                             [other_fertility[l] for l in langs])
    gap = fit_own["r2"] - fit_other["r2"]
    resid = _loglog_residuals([other_fertility[l] for l in langs], ratios, langs)
    named_pos = {l: resid[l] for l in TOKENISATION_POSITIVE_RESIDUAL if l in resid}
    named_neg = {l: resid[l] for l in TOKENISATION_NEGATIVE_RESIDUAL if l in resid}
    residuals_ok = (bool(named_pos) and all(v > 0 for v in named_pos.values()))

    # The frozen rule is a conjunction: the R^2 margin ALONE does not support T.
    if gap >= TOKENISATION_R2_MARGIN and residuals_ok:
        decision = "T_SUPPORTED"
    elif gap >= TOKENISATION_R2_MARGIN:
        decision = "UNDECIDED_RESIDUALS_WRONG_WAY"
    elif fit_other["r2"] >= fit_own["r2"]:
        decision = "L_SUPPORTED"
    else:
        decision = "UNDECIDED"
    if agreement is not None and agreement >= 0.90 and decision != "T_SUPPORTED":
        decision = "NO_POWER"
    return {
        "decision": decision,
        "n_langs": len(langs),
        "r2_own_tokeniser": fit_own["r2"],
        "r2_other_tokeniser": fit_other["r2"],
        "r2_gap": gap,
        "k_own": fit_own["k"], "k_other": fit_other["k"],
        "fertility_table_agreement": agreement,
        "residuals_vs_other_tokeniser": {l: resid[l] for l in sorted(resid)},
        "named_residuals_predicted_positive": named_pos,
        "named_residuals_predicted_negative": named_neg,
        "residual_condition_met": residuals_ok,
        "note": ("the two fertility tables correlate this closely in log space; "
                 "a gap below the margin cannot separate T from L"),
    }


def _pearson_log(a: list[float], b: list[float]) -> float | None:
    import math

    pts = [(math.log(x), math.log(y)) for x, y in zip(a, b) if x > 0 and y > 0]
    n = len(pts)
    if n < 3:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = math.sqrt(sum((x - mx) ** 2 for x, _ in pts)
                    * sum((y - my) ** 2 for _, y in pts))
    return num / den if den else None


def _loglog_residuals(xs: list[float], ys: list[float],
                      keys: list[str]) -> dict[str, float]:
    """Residuals of log y on log x, keyed by language."""
    import math

    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    denom = sum((a - mx) ** 2 for a in lx)
    b = sum((a - mx) * (c - my) for a, c in zip(lx, ly)) / denom if denom else 0.0
    a0 = my - b * mx
    return {k: y - (a0 + b * x) for k, x, y in zip(keys, lx, ly)}
