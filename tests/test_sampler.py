"""Sampler tests. No torch: the scorer is injected, which is the whole point of
keeping the selection logic tensor-free."""

import pytest

from dllmfert.sampler import (
    DecodeTrace,
    block_bounds,
    complete_prefix_stop,
    decode,
    select_positions,
    trim_at,
)


def test_naive_mode_commits_exactly_one():
    conf = {0: 0.99, 1: 0.98, 2: 0.97}
    assert select_positions(conf, mode="naive", threshold=0.5) == [0]


def test_threshold_mode_commits_everything_above_the_bar():
    conf = {0: 0.95, 1: 0.4, 2: 0.99, 3: 0.91}
    assert sorted(select_positions(conf, mode="threshold", threshold=0.9)) == [0, 2, 3]


def test_threshold_mode_always_commits_at_least_one():
    """Without this floor an all-uncertain block spins until the step cap and
    the run reads as a timeout rather than a slow decode."""
    conf = {5: 0.10, 6: 0.20, 7: 0.05}
    assert select_positions(conf, mode="threshold", threshold=0.9) == [6]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        select_positions({0: 1.0}, mode="greedy", threshold=0.5)


def test_block_bounds():
    assert block_bounds(64, 32) == [(0, 32), (32, 64)]
    assert block_bounds(70, 32) == [(0, 32), (32, 64), (64, 70)]
    assert block_bounds(20, 0) == [(0, 20)]
    assert block_bounds(20, 99) == [(0, 20)]


def fixed_scorer(confidence: float, token: int = 7):
    def scorer(committed, masked):
        return {p: (confidence, token) for p in masked}

    return scorer


def test_naive_decode_gives_parallel_factor_one_and_nfe_equals_canvas():
    """The K = Lg convention Apple reports as standard, reproduced exactly."""
    t = decode(fixed_scorer(0.99), canvas=16, mode="naive", block_size=8)
    assert t.nfe == 16
    assert t.parallel_factor == pytest.approx(1.0)
    assert all(n == 1 for n in t.finalised_per_step)


def test_confident_decode_finishes_each_block_in_one_step():
    t = decode(fixed_scorer(0.99), canvas=64, mode="threshold",
               threshold=0.9, block_size=32)
    assert t.finalised_per_step == [32, 32]
    assert t.nfe == 2
    assert t.parallel_factor == pytest.approx(32.0)


def test_unconfident_decode_degrades_to_one_per_step():
    """The floor means parallel_factor bottoms out at 1, never below."""
    t = decode(fixed_scorer(0.10), canvas=16, mode="threshold",
               threshold=0.9, block_size=16)
    assert t.parallel_factor == pytest.approx(1.0)
    assert t.nfe == 16


def test_parallel_factor_tracks_confidence_between_the_extremes():
    """Half the positions confident: two steps per block, factor 8."""
    def scorer(committed, masked):
        return {p: (0.99 if p % 2 == 0 else 0.1, 7) for p in masked}

    t = decode(scorer, canvas=16, mode="threshold", threshold=0.9, block_size=16)
    assert t.finalised_per_step[0] == 8
    assert t.parallel_factor > 1.0


def test_scorer_sees_previously_committed_positions():
    seen = []

    def scorer(committed, masked):
        seen.append(dict(committed))
        return {masked[0]: (0.99, 42)}

    decode(scorer, canvas=3, mode="naive", block_size=3)
    assert seen[0] == {}
    assert len(seen[1]) == 1 and list(seen[1].values()) == [42]
    assert len(seen[2]) == 2


def test_step_cap_is_recorded_not_silently_truncated():
    t = decode(fixed_scorer(0.1), canvas=100, mode="naive",
               block_size=100, max_steps=5)
    assert t.hit_step_cap is True
    assert t.nfe == 5
    assert t.tokens.count(-1) == 95  # uncommitted positions stay visible


def test_nfe_counts_one_forward_per_step():
    t = DecodeTrace(canvas=10, finalised_per_step=[4, 3, 3], content_len=10)
    assert t.nfe == 3
    assert t.parallel_factor == pytest.approx(10 / 3)


def test_parallel_factor_counts_content_not_canvas():
    """The bug this guards cost a real A100 run. A canvas sized for a
    high-fertility language is mostly padding when the answer is short, the
    confidence sampler commits padding fastest of all, and canvas is
    proportional to fertility -- so padding would impersonate the effect under
    study. Measured before the fix: Telugu 31.4 tokens/step vs English 3.0,
    while generating zero content."""
    padded = DecodeTrace(canvas=1632, finalised_per_step=[32] * 52, content_len=40)
    assert padded.parallel_factor == pytest.approx(40 / 52)
    assert padded.padding_positions == 1592
    # the old definition would have read:
    assert sum(padded.finalised_per_step) / padded.nfe == pytest.approx(32.0)


def test_complete_prefix_stop_ignores_a_stop_behind_a_hole():
    """A stop committed at 5 means nothing while position 2 is still masked:
    the model guessed an ending, it did not reach one."""
    assert complete_prefix_stop({0: 1, 1: 1, 2: 9}, 8, {9}) == 2
    assert complete_prefix_stop({0: 1, 1: 1, 5: 9}, 8, {9}) is None
    assert complete_prefix_stop({0: 1, 1: 1, 2: 1}, 8, {9}) is None


def test_trim_at_stop_token():
    assert trim_at([1, 2, 9, 3, 4], {9}) == [1, 2]
    assert trim_at([1, 2, 3], {9}) == [1, 2, 3]


def scripted_scorer(stop_at: int, stop_id: int = 9, confidence: float = 0.99):
    """High confidence everywhere; position `stop_at` decodes to a stop token."""
    def scorer(committed, masked):
        return {p: (confidence, stop_id if p >= stop_at else 7) for p in masked}
    return scorer


def test_decode_stops_at_the_first_complete_prefix_stop():
    t = decode(scripted_scorer(stop_at=40), canvas=1632, mode="threshold",
               threshold=0.9, block_size=32, stop_ids={9})
    assert t.stopped is True
    assert t.content_len == 40
    # two full blocks committed (0..63); the stop is found at the end of block 2
    assert t.nfe == 2
    assert t.parallel_factor == pytest.approx(20.0)
    assert t.padding_positions == 1592


def test_without_stop_ids_the_whole_canvas_is_filled():
    """Old behaviour, kept so the difference is visible rather than implied."""
    t = decode(scripted_scorer(stop_at=40), canvas=256, mode="threshold",
               threshold=0.9, block_size=32)
    assert t.stopped is False
    assert t.content_len == 256
    assert t.nfe == 8


def test_a_model_that_never_stops_counts_everything_as_content():
    t = decode(fixed_scorer(0.99), canvas=64, mode="threshold",
               threshold=0.9, block_size=32, stop_ids={9})
    assert t.stopped is False
    assert t.content_len == 64
    assert t.parallel_factor == pytest.approx(32.0)


def test_naive_mode_still_pins_parallel_factor_at_about_one_with_early_stop():
    t = decode(scripted_scorer(stop_at=20), canvas=256, mode="naive",
               block_size=32, stop_ids={9})
    assert t.stopped is True
    assert t.parallel_factor is not None and t.parallel_factor <= 1.0


def test_padding_grows_with_canvas_but_parallel_factor_does_not():
    """The invariant the whole design rests on: give the same short answer more
    canvas and the metric must not move."""
    factors = []
    for canvas in (256, 512, 1024, 2048):
        t = decode(scripted_scorer(stop_at=40), canvas=canvas, mode="threshold",
                   threshold=0.9, block_size=32, stop_ids={9})
        factors.append(t.parallel_factor)
        assert t.content_len == 40
    assert len(set(factors)) == 1, f"parallel_factor moved with canvas: {factors}"
