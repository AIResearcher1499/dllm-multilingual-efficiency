"""K2 tests. The check that the parked predecessor did not have: does the mechanism the
design rests on actually operate, measured before anything is built on it."""

import pytest

from dllmfert.k2 import k2_verdict, scaling_exponent


def test_scaling_exponent_reads_one_for_one_step_per_token():
    """The naive K = Lg convention Apple reports as standard."""
    assert scaling_exponent([256, 512, 1024], [256, 512, 1024]) == pytest.approx(1.0)


def test_scaling_exponent_reads_zero_for_a_flat_step_count():
    assert scaling_exponent([256, 2048], [64, 64]) == pytest.approx(0.0)


def test_scaling_exponent_needs_two_distinct_canvases():
    assert scaling_exponent([512], [128]) is None
    assert scaling_exponent([512, 512], [128, 200]) is None


def _points(pairs, n=5):
    return [{"canvas": c, "nfe": nfe, "parallel_factor": c / nfe, "error": None}
            for c, nfe in pairs for _ in range(n)]


def test_proceed_when_steps_track_the_canvas():
    v = k2_verdict(_points([(256, 256), (512, 512), (1024, 1024)]))
    assert v["decision"] == "NFE-SCALES"
    assert v["scaling_exponent"] == pytest.approx(1.0)
    assert "not terminating" in v["reads_as"]


def test_k2_fires_when_a_bigger_canvas_is_nearly_free():
    """Flat NFE is the expected outcome once early stopping works: step count
    is content/parallel_factor and content does not depend on canvas. The
    canvas cost lives in time-per-forward instead, which is measured."""
    v = k2_verdict(_points([(256, 60), (512, 62), (1024, 64), (2048, 66)]))
    assert v["decision"] == "FLAT-NFE"
    assert v["scaling_exponent"] < 0.5
    assert "EXPECTED result, not a kill" in v["reads_as"]


def test_sublinear_but_real_scaling_still_proceeds():
    v = k2_verdict(_points([(256, 100), (1024, 200)]))  # exponent 0.5
    assert v["scaling_exponent"] == pytest.approx(0.5)
    assert v["decision"] == "NFE-SCALES"


def test_incomplete_with_a_single_canvas():
    assert k2_verdict(_points([(512, 512)]))["decision"] == "INCOMPLETE"


def test_error_rows_are_excluded():
    pts = _points([(256, 256), (1024, 1024)])
    pts += [{"canvas": 4096, "nfe": None, "parallel_factor": None,
             "error": "OOM"} for _ in range(5)]
    v = k2_verdict(pts)
    assert v["canvases"] == [256, 1024]
    assert v["decision"] == "NFE-SCALES"


def test_parallel_factor_is_reported_alongside():
    v = k2_verdict(_points([(256, 128), (1024, 512)]))
    assert v["mean_parallel_factor"] == [pytest.approx(2.0), pytest.approx(2.0)]
