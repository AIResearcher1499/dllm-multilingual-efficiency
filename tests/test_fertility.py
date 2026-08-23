"""Stage-1 tests. No tokenizer, no download: encode is injected."""

import pytest

from dllmfert.fertility import (
    fertility_ratios,
    item_ratios,
    spread,
    stage1_verdict,
    token_counts,
)


def char_encode(text):
    """Stand-in tokenizer: one token per character."""
    return list(text)


def test_token_counts_and_item_ratios():
    texts = {"en": "abcd", "ja": "abcdefgh", "de": "abcde"}
    counts = token_counts(texts, char_encode)
    assert counts == {"en": 4, "ja": 8, "de": 5}
    assert item_ratios(counts) == {"en": 1.0, "ja": 2.0, "de": 1.25}


def test_item_ratios_empty_without_pivot():
    assert item_ratios({"ja": 8}) == {}


def test_ratio_is_per_item_then_averaged_not_ratio_of_means():
    """One long problem must not dominate: the parallel corpus exists so each
    item can be its own control."""
    items = [
        {"en": "a" * 10, "ja": "a" * 40},   # ratio 4.0
        {"en": "a" * 990, "ja": "a" * 990},  # ratio 1.0
    ]
    r = fertility_ratios(items, char_encode)
    assert r["fertility_ratio"]["ja"] == pytest.approx(2.5)   # mean of ratios
    # A ratio of means would have given 1030/1000 = 1.03 and hidden the effect.
    ratio_of_means = 1030 / 1000
    assert r["fertility_ratio"]["ja"] != pytest.approx(ratio_of_means)


def test_items_missing_the_pivot_are_skipped():
    items = [{"ja": "aaaa"}, {"en": "ab", "ja": "abcd"}]
    r = fertility_ratios(items, char_encode)
    assert r["n_items"] == 1
    assert r["fertility_ratio"]["ja"] == pytest.approx(2.0)


def test_spread_needs_two_languages():
    assert spread({"en": 1.0}) is None
    assert spread({"en": 1.0, "th": 2.5}) == pytest.approx(2.5)


def test_stage1_proceeds_on_a_wide_axis():
    v = stage1_verdict({"fertility_ratio": {"en": 1.0, "de": 1.1, "ja": 1.9,
                                            "th": 2.6, "te": 3.1, "zh": 1.2}})
    assert v["decision"] == "PROCEED"
    assert v["spread"] == pytest.approx(3.1)
    assert v["highest"][-1][0] == "te"
    assert v["lowest"][0][0] == "en"


def test_stage1_parks_on_a_compressed_axis():
    """The gate that saves a GPU day: no slope is resolvable here."""
    v = stage1_verdict({"fertility_ratio": {"en": 1.0, "de": 1.05, "fr": 1.1,
                                            "es": 1.2, "ru": 1.3, "zh": 1.4}})
    assert v["decision"] == "PARK"
    assert v["spread"] == pytest.approx(1.4)
    assert "too compressed" in v["reads_as"]


def test_stage1_incomplete_with_one_language():
    assert stage1_verdict({"fertility_ratio": {"en": 1.0}})["decision"] == "INCOMPLETE"
