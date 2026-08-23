"""Language-check tests. The confound these guard let a model score 2/3 on
Telugu by answering in English."""

import pytest

py3langid = pytest.importorskip("py3langid")

from dllmfert.language import (  # noqa: E402
    is_scorable,
    judge,
    line_pass_rate,
    response_language,
)

EN = "Janet's ducks lay sixteen eggs every single day of the week."
TE = "ప్రతిరోజూ ఆమె బాతులు పెడతాయి పదహారు గుడ్లు ఆమె ఉదయం అల్పాహారం కొరకు"
TH = "เจเน็ตมีไข่สิบหกฟองทุกวันและเธอกินสามฟองเป็นมื้อเช้าทุกเช้า"
ZH = "让我们逐步分析这个问题每天产蛋量十六个她早餐吃三个"


def test_pure_arithmetic_lines_are_not_scorable():
    """MGSM answers are full of these; they carry no language and must not
    count for or against a response."""
    assert is_scorable("16 - 3 - 4 = 9") is False
    assert is_scorable("   ") is False
    assert is_scorable(EN) is True


def test_response_language_detects_each_mgsm_script():
    assert response_language(EN) == "en"
    assert response_language(TE) == "te"
    assert response_language(TH) == "th"
    assert response_language(ZH) == "zh"


def test_the_actual_confound_is_caught():
    """Qwen answered a Telugu question in English and still scored on accuracy.
    The filter has to catch what accuracy cannot."""
    v = judge(EN, expected="te")
    assert v["response_lang"] == "en"
    assert v["lang_pass"] is False


def test_an_in_language_answer_passes():
    v = judge(TE, expected="te")
    assert v["response_lang"] == "te"
    assert v["lang_pass"] is True
    assert v["line_pass_rate"] == pytest.approx(1.0)


def test_line_pass_rate_is_a_fraction_not_a_verdict():
    mixed = f"{TE}\n{EN}\n{TE}"
    rate = line_pass_rate(mixed, "te")
    assert rate == pytest.approx(2 / 3)
    assert judge(mixed, "te", threshold=0.6)["lang_pass"] is True
    assert judge(mixed, "te", threshold=0.9)["lang_pass"] is False


def test_equations_only_response_is_unjudgeable_not_failing():
    v = judge("16 - 3 - 4 = 9\n9 * 2 = 18", expected="te")
    assert v["line_pass_rate"] is None
    assert v["lang_pass"] is False   # unjudgeable cannot count as a pass
