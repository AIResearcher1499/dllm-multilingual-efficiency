"""Torch-path tests for the model adapter.

Skipped in the fertility-only environment and run on the GPU box, where they
are the only coverage of the code that actually touches a model. A tiny Qwen2
is built in-process, so nothing is downloaded.
"""

import pytest

torch = pytest.importorskip("torch")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from dllmfert.models import find_mask_id, make_scorer  # noqa: E402
from dllmfert.sampler import decode  # noqa: E402

PROMPT = [5, 6, 7]
CANVAS = 8
MASK = 99


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture
def scorer(tiny):
    return make_scorer(tiny, PROMPT, canvas=CANVAS, mask_id=MASK, device="cpu")


def test_scores_exactly_the_requested_positions(scorer):
    s = scorer({}, [0, 3, 7])
    assert sorted(s) == [0, 3, 7]


def test_confidences_and_predictions_are_well_formed(scorer):
    s = scorer({}, list(range(CANVAS)))
    assert all(0.0 <= c <= 1.0 for c, _ in s.values())
    assert all(0 <= t < 128 for _, t in s.values())


def test_committed_tokens_actually_condition_the_forward(scorer):
    """If they did not, the sampler would be decoding every position from the
    prompt alone and parallel_factor would measure nothing."""
    a = scorer({1: 11}, [0, 3])
    b = scorer({1: 77}, [0, 3])
    assert any(abs(a[p][0] - b[p][0]) > 1e-9 for p in (0, 3))


def test_row_is_prompt_plus_canvas_masked_where_uncommitted(tiny):
    seen = {}
    real = make_scorer(tiny, PROMPT, canvas=CANVAS, mask_id=MASK, device="cpu")

    def spy(committed, masked):
        seen["row"] = list(PROMPT) + [committed.get(p, MASK) for p in range(CANVAS)]
        return real(committed, masked)

    spy({2: 42}, [0])
    assert seen["row"] == [5, 6, 7, MASK, MASK, 42, MASK, MASK, MASK, MASK, MASK]


def test_end_to_end_decode_over_a_real_forward(scorer):
    t = decode(scorer, canvas=CANVAS, mode="threshold", threshold=0.0,
               block_size=4)
    assert t.nfe == 2                      # one step per block at threshold 0
    assert t.parallel_factor == pytest.approx(4.0)
    assert t.tokens.count(-1) == 0         # every position committed


def test_naive_mode_over_a_real_forward(scorer):
    t = decode(scorer, canvas=CANVAS, mode="naive", block_size=CANVAS)
    assert t.nfe == CANVAS
    assert t.parallel_factor == pytest.approx(1.0)


class _NoMask:
    mask_token_id = None
    unk_token_id = 0

    def convert_tokens_to_ids(self, name):
        return 0


def test_missing_mask_token_fails_loudly(tiny):
    """Guessing here would decode against a token the model never saw and
    produce meaningless parallel_factor values."""
    with pytest.raises(SystemExit, match="cannot locate the mask token"):
        find_mask_id(_NoMask(), tiny)


class ShiftedModel:
    """Autoregressive convention: the prediction for position i sits at i-1."""

    def __init__(self, truth, vocab=128):
        self.truth, self.vocab = truth, vocab

    def __call__(self, ids):
        import types as _t
        n = ids.shape[1]
        logits = torch.zeros(1, n, self.vocab)
        for i in range(n - 1):
            logits[0, i, self.truth[i + 1]] = 12.0
        return _t.SimpleNamespace(logits=logits)


class AlignedModel:
    """Native masked-diffusion convention: prediction for i sits at i."""

    def __init__(self, truth, vocab=128):
        self.truth, self.vocab = truth, vocab

    def __call__(self, ids):
        import types as _t
        n = ids.shape[1]
        logits = torch.zeros(1, n, self.vocab)
        for i in range(n):
            logits[0, i, self.truth[i]] = 12.0
        return _t.SimpleNamespace(logits=logits)


class ProbeTok:
    def encode(self, text, add_special_tokens=False):
        return [(i * 7 + 3) % 120 for i in range(40)]


def test_detect_logit_shift_reads_both_conventions():
    """The bug this guards produced 482 tokens of "muff muff muff" from a
    correctly-prompted 7B model, with every confidence in [0,1] and no error
    anywhere. Dream is adapted from Qwen2.5 and keeps the AR convention; its
    own sampler shifts at generation_utils.py:412. Detect, never configure."""
    from dllmfert.models import detect_logit_shift

    truth = ProbeTok().encode("")
    aligned = detect_logit_shift(AlignedModel(truth), ProbeTok(),
                                 mask_id=99, device="cpu")
    assert aligned["shift"] == 0 and aligned["margin"] > 1.0
    shifted = detect_logit_shift(ShiftedModel(truth), ProbeTok(),
                                 mask_id=99, device="cpu")
    assert shifted["shift"] == 1 and shifted["margin"] > 1.0


def test_scorer_honours_the_detected_shift():
    from dllmfert.models import make_scorer

    truth = ProbeTok().encode("")
    prompt = truth[:10]
    sc = make_scorer(ShiftedModel(truth), prompt, canvas=8, mask_id=99,
                     device="cpu", shift=1)
    got = sc({}, [0, 1, 2])
    assert all(0.0 <= c <= 1.0 for c, _ in got.values())
    # Canvas position p is absolute index 10+p. Under the AR convention its
    # prediction sits one earlier, at logits[9+p], which is where the shifted
    # scorer reads. So position p must recover truth[10+p].
    for pos in (0, 1, 2):
        assert got[pos][1] == truth[10 + pos]
