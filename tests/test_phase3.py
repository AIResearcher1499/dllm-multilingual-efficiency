"""Phase 3 decision rules, including the branches that end the project.

Every test here builds data with a known answer and asserts the frozen rule
lands where it should. A rule that cannot be shown to fire is not a rule.
"""

import pytest

from dllmfert.degeneracy import distinctness, relative_cutoff
from dllmfert.phase3 import p1_verdict, p2_verdict

PENALTIES = [1.0, 1.05, 1.1, 1.2, 1.4]


def repetitive(n_unique: int, total: int = 60) -> str:
    """Text whose char-distinctness rises with n_unique."""
    words = [f"w{i%n_unique:03d}" for i in range(total)]
    return " ".join(words)


def p1_rows(*, pf_at, len_at, uniq_at, arm="Dream-7B", n=20):
    rows = []
    for k, p in enumerate(PENALTIES):
        for i in range(n):
            rows.append({
                "arm": arm, "repetition_penalty": p, "error": None,
                "parallel_factor": pf_at(k, i),
                "tokens_generated": len_at(k, i),
                "text": repetitive(uniq_at(k)),
                "acc": True,
            })
    return rows


def test_p1_supported_when_pf_falls_as_repetition_is_suppressed():
    rows = p1_rows(pf_at=lambda k, i: 4.0 - 0.5 * k + 0.01 * (i % 3),
                   len_at=lambda k, i: 200 + (i % 5),
                   uniq_at=lambda k: 2 + 4 * k)
    v = p1_verdict(rows)["Dream-7B"]
    assert v["decision"] == "SUPPORTED"
    assert v["spearman_pf_vs_penalty"] <= -0.8
    assert v["spearman_distinct_vs_penalty"] >= 0.8


@pytest.mark.parametrize("pf_at, why", [
    (lambda k, i: 3.0 + 0.01 * (i % 3), "exactly flat"),
    (lambda k, i: 3.0 + 0.02 * k + 0.01 * (i % 3), "moves the wrong way"),
])
def test_p1_kill_fires_when_pf_ignores_the_penalty(pf_at, why):
    """The branch that ends the project. If suppressing repetition leaves the
    measured parallel_factor where it was, the artifact reading of Phase 2 is
    wrong and Phase 3 stops. An exactly flat response yields a nan correlation
    and must still fire, not fall through to UNDECIDED."""
    rows = p1_rows(pf_at=pf_at, len_at=lambda k, i: 200 + (i % 5),
                   uniq_at=lambda k: 2 + 4 * k)
    assert p1_verdict(rows)["Dream-7B"]["decision"] == "KILL_K3", why


def test_p1_reports_confounded_when_pf_only_tracks_length():
    """parallel_factor has content_len in its numerator. A penalty that merely
    shortens the output would reproduce the predicted correlation without the
    mechanism, and that must not read as support."""
    rows = p1_rows(pf_at=lambda k, i: (200 - 30 * k + (i % 5)) / 50.0 + 0.002 * (i % 7),
                   len_at=lambda k, i: 200 - 30 * k + (i % 5),
                   uniq_at=lambda k: 2 + 4 * k)
    v = p1_verdict(rows)["Dream-7B"]
    assert v["decision"] == "CONFOUNDED_WITH_LENGTH"
    assert v["spearman_pf_vs_penalty"] <= -0.8            # the naive read
    assert v["spearman_pf_vs_penalty_controlling_length"] > -0.3   # the honest one


def test_p1_needs_at_least_three_settings():
    rows = [r for r in p1_rows(pf_at=lambda k, i: 1.0, len_at=lambda k, i: 10,
                               uniq_at=lambda k: 5)
            if r["repetition_penalty"] in (1.0, 1.4)]
    assert p1_verdict(rows)["Dream-7B"]["decision"] == "INCOMPLETE"


def p2_rows(*, pf_early, rep_early, n=40, arm="Dream-7B"):
    """Half the items degenerate. `pf_early`/`rep_early` say whether each
    signal separates the classes inside the first tenth of the steps."""
    rows = []
    for i in range(n):
        bad = i % 2 == 0
        total = 40
        content = 200
        # commit steps: a degenerate item commits in bulk early when pf_early
        per_step = (8 if (bad and pf_early) else 5)
        steps, toks = [], []
        for j in range(content):
            steps.append(min(total - 1, j // per_step))
            if bad and rep_early:
                toks.append(j % 3)              # visibly repetitive from token 0
            elif bad:
                toks.append(j if j < content // 2 else j % 3)   # only late
            else:
                toks.append(j)
        rows.append({
            "arm": arm, "error": None, "commit_step": steps,
            "content_tokens": toks,
            "text": repetitive(2 if bad else 40),
        })
    return rows


def test_p2_supported_when_pf_separates_earlier_than_surface_repetition():
    v = p2_verdict(p2_rows(pf_early=True, rep_early=False))["Dream-7B"]
    assert v["decision"] == "M1_SUPPORTED"
    assert v["budgets"]["10%"]["margin"] >= 0.05


def test_p2_dropped_when_the_cheap_ngram_signal_is_at_least_as_good():
    """M1 has to beat a signal that is also nearly free. If it does not, the
    paper ships the measurement and makes no detector claim."""
    v = p2_verdict(p2_rows(pf_early=False, rep_early=True))["Dream-7B"]
    assert v["decision"] == "M1_DROPPED"


def test_p2_needs_enough_rows():
    v = p2_verdict(p2_rows(pf_early=True, rep_early=False, n=10))["Dream-7B"]
    assert v["decision"] == "INCOMPLETE"


def test_degeneracy_cutoff_is_relative_to_the_model_not_absolute():
    """The bug this rule exists to prevent: an absolute char-distinct cutoff
    calibrated on Dream (0.75) labels LLaDA's *English* degenerate, because
    LLaDA is more repetitive at every language."""
    llada_like = [0.54, 0.71, 0.74, 0.74, 0.75, 0.77, 0.82, 0.86, 0.94]
    cut = relative_cutoff(llada_like, 20.0)
    assert cut < 0.75, "a relative cutoff must not condemn most of the model"
    assert sum(d <= cut for d in llada_like) <= 2


def test_distinctness_is_bounded_and_ordered():
    assert distinctness("abcdefghijklmnop") == pytest.approx(1.0)
    assert distinctness("ab" * 50) < 0.1


def test_r2_margin_alone_does_not_support_tokenisation():
    """Revision 5's rule is a conjunction. An R^2 gap over the margin whose
    named residuals point the wrong way is UNDECIDED, not support -- the
    residual half is what makes the prediction risky rather than a curve fit."""
    from dllmfert.metrics import tokenisation_verdict

    langs = ["en", "zh", "ru", "th", "de", "fr", "es"]
    own = {"en": 1.0, "zh": 0.9, "ru": 2.2, "th": 3.6, "de": 1.6, "fr": 1.5, "es": 1.5}
    other = {"en": 1.0, "zh": 1.1, "ru": 1.5, "th": 2.1, "de": 1.3, "fr": 1.3, "es": 1.3}
    # Cost tracks `own` closely, but ru and th are pushed *below* the line that
    # `other` fits, which is the opposite of what T predicts.
    ratios = {"en": 2.0, "zh": 1.6, "ru": 1.7, "th": 1.9, "de": 3.0, "fr": 2.9,
              "es": 2.9}
    by_lang = {l: {"cost_ratio": ratios[l]} for l in langs}
    v = tokenisation_verdict(by_lang, own, other)
    assert v["residual_condition_met"] is False
    assert v["decision"] != "T_SUPPORTED"


def test_unknown_contention_is_not_reported_as_clean():
    """A box where nvidia-smi is missing must not certify its own timings. The
    distinction between None and 0 is the whole point of the field."""
    from dllmfert.provenance import timing_is_trustworthy

    assert timing_is_trustworthy({"concurrent_procs": 1}) is True
    assert timing_is_trustworthy({"concurrent_procs": 2}) is False
    assert timing_is_trustworthy({"concurrent_procs": None}) is False
    assert timing_is_trustworthy({}) is False


def test_provenance_is_cached_but_contention_is_not(monkeypatch):
    """Static fields cost one subprocess per process; contention is re-read per
    row because it is the only field that moves during a run."""
    import dllmfert.provenance as pv

    monkeypatch.setattr(pv, "_STATIC", None)
    calls = {"static": 0, "procs": 0}

    def fake_smi(q):
        calls["static"] += 1
        return ["FakeGPU"] if q == "name" else ["999.99"]

    def fake_procs():
        calls["procs"] += 1
        return calls["procs"]

    monkeypatch.setattr(pv, "_nvidia_smi", fake_smi)
    monkeypatch.setattr(pv, "_procs_on_gpu", fake_procs)
    a, b, c = pv.hardware_provenance(), pv.hardware_provenance(), pv.hardware_provenance()
    assert calls["static"] == 2, "static fields should be queried once per process"
    assert [a["concurrent_procs"], b["concurrent_procs"], c["concurrent_procs"]] == [1, 2, 3]
    assert a["gpu_name"] in ("FakeGPU", a["gpu_name"])
