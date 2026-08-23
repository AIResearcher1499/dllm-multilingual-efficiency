"""Stage-2 verdict tests. The prereg gates on MAGNITUDE; direction only labels
the result. These tests exist because the predecessor's first decision rule called
+0.007 nats a finding."""

import pytest

from dllmfert.metrics import (
    by_language,
    g0_verdict,
    padding_waste,
    parallel_factor,
    relative_spread,
)

FERT = {"en": 1.0, "de": 1.15, "ru": 1.4, "zh": 1.1, "ja": 1.9, "th": 2.6,
        "te": 3.1, "bn": 2.8}


def test_parallel_factor_counts_dead_steps():
    """A sampler that finalises nothing on some steps must be penalised, not
    averaged away — which is why this is not tokens/NFE computed post hoc."""
    assert parallel_factor([2, 2, 2, 2]) == pytest.approx(2.0)
    assert parallel_factor([4, 0, 4, 0]) == pytest.approx(2.0)
    assert parallel_factor([]) is None


def test_padding_waste():
    assert padding_waste(1000, 250) == pytest.approx(0.75)
    assert padding_waste(1000, 1200) == 0.0
    assert padding_waste(0, 10) is None


def test_relative_spread():
    assert relative_spread([1.0, 1.5]) == pytest.approx(0.5)
    assert relative_spread([2.0]) is None


def _rows(pf_by_lang, arm="dream", n=100):
    return [{"arm": arm, "lang": lang, "parallel_factor": pf, "nfe": 100,
             "wall_clock": 1.0, "acc": True, "padding_waste": 0.1, "error": None}
            for lang, pf in pf_by_lang.items() for _ in range(n)]


def test_go_direction_b_when_fertility_raises_parallelism():
    pf = {"en": 1.5, "de": 1.6, "zh": 1.55, "ru": 1.7, "ja": 2.1, "th": 2.6,
          "te": 2.9, "bn": 2.7}
    v = g0_verdict(_rows(pf), FERT, "dream")
    assert v["decision"] == "GO"
    assert v["direction"] == "B"
    assert v["parallel_spread"] > 0.2
    assert "paradigm contrast" in v["reads_as"]


def test_go_direction_a_when_fertility_lowers_parallelism():
    pf = {"en": 2.9, "de": 2.7, "zh": 2.8, "ru": 2.5, "ja": 2.0, "th": 1.6,
          "te": 1.4, "bn": 1.5}
    v = g0_verdict(_rows(pf), FERT, "dream")
    assert v["decision"] == "GO"
    assert v["direction"] == "A"
    assert "2605.30580" in v["reads_as"]


def test_park_when_the_effect_is_real_but_small():
    """Monotone but under threshold: the axis exists and does not matter."""
    pf = {"en": 2.00, "de": 2.01, "zh": 2.02, "ru": 2.04, "ja": 2.06,
          "th": 2.08, "te": 2.10, "bn": 2.09}
    v = g0_verdict(_rows(pf), FERT, "dream")
    assert v["decision"] == "PARK"
    assert v["parallel_spread"] < 0.2
    assert "not by an amount that matters" in v["reads_as"]


def test_undecided_when_large_but_not_monotone():
    pf = {"en": 1.5, "de": 3.0, "zh": 1.6, "ru": 2.9, "ja": 1.4, "th": 3.1,
          "te": 1.5, "bn": 2.8}
    v = g0_verdict(_rows(pf), FERT, "dream")
    assert v["decision"] == "UNDECIDED"
    assert v["direction"] is None


def test_incomplete_below_six_languages():
    pf = {"en": 1.5, "ja": 2.5, "th": 3.0}
    assert g0_verdict(_rows(pf), FERT, "dream")["decision"] == "INCOMPLETE"


def test_incomplete_when_a_language_has_too_few_items():
    pf = {"en": 1.5, "de": 1.6, "zh": 1.55, "ru": 1.7, "ja": 2.1, "th": 2.6}
    rows = _rows(pf, n=50)
    assert g0_verdict(rows, FERT, "dream")["decision"] == "INCOMPLETE"


def test_error_rows_are_excluded():
    rows = _rows({"en": 1.5}, n=5)
    rows += [{"arm": "dream", "lang": "en", "parallel_factor": 99.0,
              "error": "OOM"} for _ in range(5)]
    agg = by_language(rows, "dream")
    assert agg["en"]["n"] == 5
    assert agg["en"]["parallel_factor"] == pytest.approx(1.5)


def test_arms_are_kept_separate():
    rows = _rows({"en": 1.5}, arm="dream", n=3) + _rows({"en": 3.0}, arm="llada", n=3)
    assert by_language(rows, "dream")["en"]["parallel_factor"] == pytest.approx(1.5)
    assert by_language(rows, "llada")["en"]["parallel_factor"] == pytest.approx(3.0)


# --- end-to-end cost ratio -------------------------------------------------

from dllmfert.metrics import _ols_loglog, cost_ratio_by_language, slope_verdict  # noqa: E402


def _timed(lang, arm, mode, secs, n=10, pf=3.0, toks=100, pad=0, stopped=True):
    # lang_pass is set: these fixtures exercise the cost-ratio arithmetic, and
    # the language filter is on by default so a row without it is dropped.
    return [{"lang": lang, "arm": arm, "mode": mode, "id": f"{lang}-{i}",
             "wall_clock": secs, "parallel_factor": pf, "tokens_generated": toks,
             "padding_positions": pad, "stopped": stopped, "error": None,
             "lang_pass": True, "response_lang": lang}
            for i in range(n)]


def test_cost_ratio_pairs_on_shared_item_ids_only():
    rows = _timed("en", "D", "threshold", 2.0, n=3) + _timed("en", "Q", "ar", 1.0, n=5)
    out = cost_ratio_by_language(rows, dllm_arm="D", ar_arm="Q")
    assert out["en"]["n"] == 3          # not 5: only paired ids count
    assert out["en"]["cost_ratio"] == pytest.approx(2.0)


def test_cost_ratio_ignores_the_other_decoding_mode():
    rows = (_timed("en", "D", "threshold", 2.0) + _timed("en", "D", "naive", 90.0)
            + _timed("en", "Q", "ar", 1.0))
    assert cost_ratio_by_language(rows, dllm_arm="D", ar_arm="Q")["en"][
        "cost_ratio"] == pytest.approx(2.0)


def test_ols_recovers_a_known_exponent():
    xs = [1.0, 2.0, 4.0, 8.0]
    assert _ols_loglog(xs, [x ** 1.0 for x in xs])["k"] == pytest.approx(1.0)
    assert _ols_loglog(xs, [x ** 0.5 for x in xs])["k"] == pytest.approx(0.5)
    assert _ols_loglog(xs, [3.0 for _ in xs])["k"] == pytest.approx(0.0)


FERT8 = {"en": 1.0, "zh": 1.08, "de": 1.33, "ru": 1.49, "sw": 1.73,
         "th": 2.13, "bn": 4.19, "te": 6.25}


def _grid(ratio_of):
    rows = []
    for lang, f in FERT8.items():
        rows += _timed(lang, "D", "threshold", ratio_of(f))
        rows += _timed(lang, "Q", "ar", 1.0)
    return rows


def test_no_compensation_reads_as_mechanism_a():
    v = slope_verdict(cost_ratio_by_language(_grid(lambda f: f),
                                             dllm_arm="D", ar_arm="Q"), FERT8)
    assert v["decision"] == "GO"
    assert v["k"] == pytest.approx(1.0, abs=0.01)
    assert "does not compensate" in v["reads_as"]
    assert v["spread"] == pytest.approx(6.25, rel=0.01)


def test_full_compensation_reads_as_mechanism_b():
    v = slope_verdict(cost_ratio_by_language(_grid(lambda f: 1.4),
                                             dllm_arm="D", ar_arm="Q"), FERT8)
    assert v["k"] == pytest.approx(0.0, abs=0.01)
    assert "contradicts the speculative-decoding result" in v["reads_as"]


def test_partial_compensation_is_reported_as_a_decomposition():
    v = slope_verdict(cost_ratio_by_language(_grid(lambda f: f ** 0.5),
                                             dllm_arm="D", ar_arm="Q"), FERT8)
    assert v["k"] == pytest.approx(0.5, abs=0.01)
    assert "Partial compensation" in v["reads_as"] or "partial" in v["reads_as"]


def test_a_noisy_fit_parks_regardless_of_the_value():
    """The gate is precision, not effect size: a k nobody can pin down cannot
    separate mechanism A from full compensation, whatever number it prints."""
    import random

    rnd = random.Random(0)
    rows = []
    for lang, f in FERT8.items():
        rows += _timed(lang, "D", "threshold", rnd.uniform(0.3, 12.0))
        rows += _timed(lang, "Q", "ar", 1.0)
    v = slope_verdict(cost_ratio_by_language(rows, dllm_arm="D", ar_arm="Q"), FERT8)
    assert v["decision"] == "PARK"
    assert "cannot answer the question" in v["reads_as"]


def test_crossover_language_is_identified():
    v = slope_verdict(cost_ratio_by_language(_grid(lambda f: f / 1.5),
                                             dllm_arm="D", ar_arm="Q"), FERT8)
    assert v["crossover"]  # the last language still at or below parity with AR


def _lang(lang, arm, mode, secs, n=10, ok=True):
    return [{"lang": lang, "arm": arm, "mode": mode, "id": f"{lang}-{i}",
             "wall_clock": secs, "parallel_factor": 3.0, "tokens_generated": 100,
             "padding_positions": 0, "stopped": True, "error": None,
             "lang_pass": ok, "response_lang": lang if ok else "en"}
            for i in range(n)]


def test_wrong_language_rows_are_dropped_and_the_loss_is_reported():
    """An item answered in the wrong language is a different job, not a harder
    one. Accuracy cannot catch it: Qwen scored 2/3 on Telugu in English."""
    rows = (_lang("te", "D", "threshold", 4.0, n=6, ok=True)
            + _lang("te", "D", "threshold", 99.0, n=4, ok=False)
            + _lang("te", "Q", "ar", 1.0, n=10, ok=True))
    out = cost_ratio_by_language(rows, dllm_arm="D", ar_arm="Q")["te"]
    assert out["n"] == 6
    assert out["cost_ratio"] == pytest.approx(4.0)       # the 99s rows are gone
    assert out["lang_drop_rate"]["dllm"] == pytest.approx(0.4)
    assert out["lang_drop_rate"]["ar"] == pytest.approx(0.0)


def _lang_at(offset, lang, arm, mode, secs, n, ok):
    rows = _lang(lang, arm, mode, secs, n=n, ok=ok)
    for i, r in enumerate(rows):
        r["id"] = f"{lang}-{offset + i}"
    return rows


def test_the_filter_can_be_turned_off_for_an_explicit_comparison():
    rows = (_lang_at(0, "te", "D", "threshold", 4.0, 5, True)
            + _lang_at(5, "te", "D", "threshold", 6.0, 5, False)
            + _lang("te", "Q", "ar", 1.0, n=10, ok=True))
    off = cost_ratio_by_language(rows, dllm_arm="D", ar_arm="Q",
                                 require_lang_pass=False)["te"]
    assert off["n"] == 10
    assert off["cost_ratio"] == pytest.approx(5.0)
