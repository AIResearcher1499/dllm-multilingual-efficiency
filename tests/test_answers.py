"""Answer scoring must survive leaving English — the entire premise of the repo
is that things break when you do."""

import pytest

from dllmfert.answers import extract_numbers, is_correct, last_number, normalize_digits


def test_native_digits_are_normalized():
    assert normalize_digits("१२३") == "123"      # Devanagari
    assert normalize_digits("১২৩") == "123"      # Bengali
    assert normalize_digits("౧౨౩") == "123"      # Telugu
    assert normalize_digits("๑๒๓") == "123"      # Thai


def test_a_model_answering_in_script_is_not_scored_wrong():
    """Without digit normalization every Telugu/Bengali answer reads as a miss
    and K3 would fire for a scoring bug rather than a model failure."""
    assert is_correct("సమాధానం ౧౮", 18) is True
    assert is_correct("উত্তর ১৮", 18) is True


def test_locale_thousands_separators():
    assert last_number("the total is 1,234") == pytest.approx(1234)
    assert last_number("insgesamt 1 234") == pytest.approx(1234)
    assert last_number("decimal 12.5") == pytest.approx(12.5)


def test_last_number_wins_because_the_answer_follows_the_reasoning():
    assert last_number("first 7 then 12 so 18") == pytest.approx(18)
    assert is_correct("first 7 then 12 so 18", 18) is True
    assert is_correct("first 7 then 12 so 18", 7) is False


def test_no_number_is_not_correct():
    assert last_number("no digits here") is None
    assert is_correct("no digits here", 18) is False


def test_negative_and_float_gold():
    assert is_correct("result -3.5", -3.5) is True
    assert extract_numbers("-3.5 and 2") == [-3.5, 2.0]


def test_bad_gold_is_false_not_an_exception():
    assert is_correct("18", None) is False
    assert is_correct("18", "n/a") is False
