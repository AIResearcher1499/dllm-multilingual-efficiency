"""MGSM loading. Kept apart from fertility.py so stage 1 stays testable
without a download."""

from __future__ import annotations

MGSM = "juletxara/mgsm"
# MGSM's advertised set. Verified at runtime rather than trusted: docs say
# "verify what MGSM actually ships, do not assume".
EXPECTED_LANGS = ("bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh")


def available_langs(candidates=EXPECTED_LANGS) -> list[str]:
    from datasets import get_dataset_config_names

    configs = set(get_dataset_config_names(MGSM))
    return [l for l in candidates if l in configs]


def load_parallel(langs: list[str], n: int, split: str = "test") -> list[dict[str, str]]:
    """n parallel items as [{lang: question}]. MGSM is aligned by row index."""
    from datasets import load_dataset

    per_lang = {}
    for lang in langs:
        ds = load_dataset(MGSM, lang, split=split)
        per_lang[lang] = [ds[i]["question"] for i in range(min(n, len(ds)))]
    size = min(len(v) for v in per_lang.values())
    return [{lang: per_lang[lang][i] for lang in langs} for i in range(size)]


def encode_fn(tokenizer):
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return encode


def load_items(lang: str, n: int, split: str = "test") -> list[dict]:
    """MGSM items for one language, as {id, question, gold}.

    Ids are prefixed by language so a row key is unique across the run, and
    index-aligned across languages so the same problem is comparable.
    """
    from datasets import load_dataset

    ds = load_dataset(MGSM, lang, split=split)
    out = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        out.append({
            "id": f"{lang}-{i}",
            "question": row["question"],
            "gold": row.get("answer_number"),
        })
    return out


def load_exemplars(lang: str, k: int = 3, split: str = "train") -> list[dict]:
    """In-language few-shot exemplars, from MGSM's own train split.

    Few-shot prompting is the mitigation `2406.20052` reports for language
    confusion, and MGSM ships eight worked examples per language with the
    reasoning already written in that language -- so the fix costs nothing and
    is the benchmark's standard protocol rather than something invented here.
    """
    from datasets import load_dataset

    ds = load_dataset(MGSM, lang, split=split)
    out = []
    for i in range(min(k, len(ds))):
        row = ds[i]
        if row.get("question") and row.get("answer"):
            out.append({"question": row["question"], "answer": row["answer"]})
    return out
