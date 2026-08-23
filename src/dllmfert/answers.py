"""MGSM answer extraction. Judge-free: MGSM gold answers are plain numbers.

Kept deliberately small. Accuracy is a control here, not the paper's claim —
its job is to catch K3 (quality collapsing in high-fertility languages), so it
must be robust across scripts rather than clever.
"""

from __future__ import annotations

import re

# Devanagari, Bengali, Telugu, Thai and Arabic-Indic digits all appear in MGSM
# outputs; a model answering in-script must not be scored as wrong.
_DIGIT_MAP = {}
for _base, _zero in (
    ("latin", 0x0030), ("arabic", 0x0660), ("devanagari", 0x0966),
    ("bengali", 0x09E6), ("telugu", 0x0C66), ("thai", 0x0E50),
):
    for _d in range(10):
        _DIGIT_MAP[chr(_zero + _d)] = str(_d)

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_digits(text: str) -> str:
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def strip_separators(text: str) -> str:
    """Thousands separators vary by locale: 1,234 / 1.234 / 1 234."""
    return re.sub(r"(?<=\d)[,  \s](?=\d{3}\b)", "", text)


def extract_numbers(text: str) -> list[float]:
    cleaned = strip_separators(normalize_digits(text))
    out = []
    for m in NUMBER_RE.finditer(cleaned):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def last_number(text: str) -> float | None:
    nums = extract_numbers(text)
    return nums[-1] if nums else None


def is_correct(text: str, gold, tol: float = 1e-6) -> bool:
    """Last number in the output vs gold. Last, not first: the answer comes
    after the reasoning."""
    try:
        target = float(str(gold).replace(",", "").strip())
    except (TypeError, ValueError):
        return False
    got = last_number(text)
    return got is not None and abs(got - target) <= tol
