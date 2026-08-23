"""Which language did the model actually answer in?

The confound this exists for: Qwen answers Thai questions in Chinese and Telugu
questions in English, so its output length is flat across scripts while the
diffusion arm pays the full token cost. Any efficiency comparison across
languages is then comparing two different jobs.

This is **language confusion**, named and benchmarked in `2406.20052`
(EMNLP 2024) -- including the implicit case, "output to Hindi input should be
Hindi". The metric here follows their line-level pass rate rather than
inventing one, and the mitigation used upstream (few-shot in-language
exemplars) is theirs too.

Accuracy cannot substitute for this check: Qwen scored 2/3 on Telugu **by
answering in English**.
"""

from __future__ import annotations

import re

# A line needs some actual words before an identifier can say anything. MGSM
# answers are full of pure-arithmetic lines like "16 - 3 - 4 = 9", which carry
# no language and must not count against a response either way.
MIN_ALPHA_CHARS = 10
DEFAULT_PASS = 0.8

_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)


def is_scorable(line: str, min_alpha: int = MIN_ALPHA_CHARS) -> bool:
    return len(_ALPHA.findall(line)) >= min_alpha


def classify(text: str) -> tuple[str | None, float]:
    """(language code, confidence). None when there is nothing to classify."""
    if not is_scorable(text):
        return None, 0.0
    import py3langid

    lang, score = py3langid.classify(text)
    return lang, float(score)


def line_languages(text: str) -> list[tuple[str, str | None]]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append((line, classify(line)[0] if is_scorable(line) else None))
    return out


def line_pass_rate(text: str, expected: str) -> float | None:
    """LCB-style: fraction of scorable lines that are in the expected language.

    None when no line carries enough text to judge — which is itself worth
    recording, since a response made entirely of equations is not evidence of
    anything about language.
    """
    langs = [lg for _, lg in line_languages(text) if lg is not None]
    if not langs:
        return None
    return sum(1 for lg in langs if lg == expected) / len(langs)


def response_language(text: str) -> str | None:
    """Dominant language over the whole response."""
    return classify(text)[0]


def judge(text: str, expected: str, threshold: float = DEFAULT_PASS) -> dict:
    """Everything a row needs to be filtered on later, computed once."""
    rate = line_pass_rate(text, expected)
    return {
        "response_lang": response_language(text),
        "line_pass_rate": rate,
        "lang_pass": (rate is not None and rate >= threshold),
        "lang_threshold": threshold,
    }
