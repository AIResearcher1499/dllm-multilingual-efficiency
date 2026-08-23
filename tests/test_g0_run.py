"""Runner tests with an injected loader and corpus: no torch, no download, no
GPU. Covers the control flow that the GPU run cannot afford to get wrong."""

import json

import pytest

from dllmfert.g0_run import G0Config, canvas_for, load_done, run_g0, step_cap


class FakeTok:
    eos_token_id = 99
    pad_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens=True):
        return "the answer is 18"


def fake_loader(confidence=0.99, fail_langs=()):
    def loader(model_id, cfg):
        return ("MODEL", FakeTok(), "cpu", (7, 0), model_id, confidence, set(fail_langs))

    return loader


def fake_items(n=3):
    def items_fn(lang, count):
        return [{"id": f"{lang}-{i}", "question": f"q{i}", "gold": 18}
                for i in range(min(n, count))]

    return items_fn


@pytest.fixture
def patched(monkeypatch):
    """Replace the two GPU entry points; everything else is the real runner."""
    import dllmfert.g0_run as g

    def run_dllm_item(bits, item, *, cfg, lang, arm, mode, canvas):
        _, _, _, _, _, conf, fail = bits
        if lang in fail:
            raise RuntimeError("CUDA out of memory")
        per_step = 1 if mode == "naive" else max(1, int(conf * 8))
        nfe = max(1, canvas // per_step)
        return ({"nfe": nfe, "parallel_factor": float(per_step),
                 "tokens_generated": canvas // 2, "hit_step_cap": False},
                "the answer is 18", 0.5)

    def run_ar_item(bits, item, *, canvas):
        return ({"nfe": canvas, "parallel_factor": 1.0,
                 "tokens_generated": canvas // 2, "hit_step_cap": False},
                "the answer is 18", 1.0)

    monkeypatch.setattr(g, "run_dllm_item", run_dllm_item)
    monkeypatch.setattr(g, "run_ar_item", run_ar_item)


def write_fertility(tmp_path, ratios):
    p = tmp_path / "fertility.json"
    p.write_text(json.dumps({"fertility_ratio": ratios,
                             "verdict": {"decision": "PROCEED"}}))
    return p


def config(tmp_path, ratios, **kw):
    kw.setdefault("out", tmp_path / "g0.jsonl")
    kw.setdefault("fertility", write_fertility(tmp_path, ratios))
    kw.setdefault("dllm_models", ("org/Dream-7B",))
    kw.setdefault("ar_model", "org/Qwen-7B")
    kw.setdefault("n", 3)
    return G0Config(**kw)


def rows_of(cfg):
    return [json.loads(l) for l in cfg.out.read_text().splitlines() if l.strip()]


def test_canvas_scales_with_fertility_and_snaps_to_block():
    cfg = G0Config(base_canvas=256, block_size=32, canvas_mode="fertility")
    assert canvas_for(cfg, 1.0) == 256
    assert canvas_for(cfg, 2.13) == 576   # ceil(545/32)*32
    assert canvas_for(cfg, 6.25) == 1600
    assert canvas_for(cfg, 0.5) == 256    # never shrinks below the base


def test_shared_canvas_control_ignores_fertility():
    cfg = G0Config(base_canvas=256, block_size=32, canvas_mode="shared")
    assert canvas_for(cfg, 6.25) == 256


def test_unknown_canvas_mode_is_rejected():
    with pytest.raises(SystemExit, match="unknown canvas_mode"):
        canvas_for(G0Config(canvas_mode="magic"), 1.0)


def test_end_to_end_row_shape_and_cells(tmp_path, patched):
    cfg = config(tmp_path, {"en": 1.0, "th": 2.13})
    run_g0(cfg, loader=fake_loader(), items_fn=fake_items())
    rows = rows_of(cfg)
    # 2 langs x (2 diffusion modes + 1 AR) x 3 items
    assert len(rows) == 2 * 3 * 3
    assert {r["arm"] for r in rows} == {"Dream-7B", "Qwen-7B"}
    assert {r["mode"] for r in rows} == {"naive", "threshold", "ar"}
    need = {"lang", "arm", "mode", "id", "canvas", "canvas_mode", "nfe",
            "parallel_factor", "tokens_generated", "padding_waste",
            "wall_clock", "acc", "error"}
    assert need <= set(rows[0])
    assert rows[0]["acc"] is True


def test_naive_mode_pins_parallel_factor_at_one(tmp_path, patched):
    cfg = config(tmp_path, {"en": 1.0})
    run_g0(cfg, loader=fake_loader(), items_fn=fake_items())
    naive = [r for r in rows_of(cfg) if r["mode"] == "naive"]
    assert naive and all(r["parallel_factor"] == 1.0 for r in naive)


def test_padding_waste_is_recorded(tmp_path, patched):
    cfg = config(tmp_path, {"en": 1.0})
    run_g0(cfg, loader=fake_loader(), items_fn=fake_items())
    r = rows_of(cfg)[0]
    assert r["padding_waste"] == pytest.approx(0.5)


def test_resume_keys_on_the_whole_cell_not_the_item_id(tmp_path, patched):
    """The same item is decoded once per (lang, arm, mode); keying on id alone
    would silently skip most of the run."""
    cfg = config(tmp_path, {"en": 1.0, "th": 2.13})
    run_g0(cfg, loader=fake_loader(), items_fn=fake_items())
    first = len(rows_of(cfg))
    cfg2 = config(tmp_path, {"en": 1.0, "th": 2.13}, out=cfg.out,
                  fertility=cfg.fertility, resume=True)
    run_g0(cfg2, loader=fake_loader(), items_fn=fake_items())
    assert len(rows_of(cfg2)) == first  # nothing re-run
    done = load_done(cfg.out)
    assert ("en", "Dream-7B", "naive", "en-0", 1.0) in done
    assert ("en", "Dream-7B", "threshold", "en-0", 1.0) in done


def test_resume_does_not_skip_a_different_repetition_penalty(tmp_path, patched):
    """The Phase 3 sweep reuses one out file across penalty settings. If the
    resume key ignored the penalty, every setting after the first would be
    skipped and the sweep would report the penalty-1.0 numbers five times --
    a flat line that looks exactly like the kill condition firing."""
    cfg = config(tmp_path, {"en": 1.0}, ar_model=None)
    run_g0(cfg, loader=fake_loader(), items_fn=fake_items())
    first = len(rows_of(cfg))
    cfg2 = config(tmp_path, {"en": 1.0}, out=cfg.out, fertility=cfg.fertility,
                  ar_model=None, resume=True)
    cfg2.repetition_penalty = 1.2
    run_g0(cfg2, loader=fake_loader(), items_fn=fake_items())
    rows = rows_of(cfg2)
    assert len(rows) == 2 * first, "the second penalty was skipped"
    assert {r["repetition_penalty"] for r in rows} == {1.0, 1.2}


def test_sweeping_the_penalty_refuses_to_include_the_ar_arm(tmp_path, patched):
    """AR generation never sees the penalty, so a swept AR arm would write
    duplicate rows under distinct resume keys and they would later be averaged
    as independent measurements."""
    import pytest

    cfg = config(tmp_path, {"en": 1.0})
    cfg.repetition_penalty = 1.2
    with pytest.raises(SystemExit, match="diffusion arm only"):
        run_g0(cfg, loader=fake_loader(), items_fn=fake_items())


def test_item_failure_is_recorded_and_the_run_continues(tmp_path, patched):
    cfg = config(tmp_path, {"en": 1.0, "th": 2.13})
    run_g0(cfg, loader=fake_loader(fail_langs=("th",)), items_fn=fake_items())
    rows = rows_of(cfg)
    bad = [r for r in rows if r["error"]]
    assert bad and all(r["lang"] == "th" and r["arm"] == "Dream-7B" for r in bad)
    assert [r for r in rows if r["lang"] == "th" and r["arm"] == "Qwen-7B"]


def test_refuses_existing_out_without_resume(tmp_path, patched):
    cfg = config(tmp_path, {"en": 1.0})
    run_g0(cfg, loader=fake_loader(), items_fn=fake_items())
    with pytest.raises(SystemExit, match="--resume"):
        run_g0(config(tmp_path, {"en": 1.0}, out=cfg.out, fertility=cfg.fertility),
               loader=fake_loader(), items_fn=fake_items())


def test_refuses_languages_with_no_measured_fertility(tmp_path, patched):
    cfg = config(tmp_path, {"en": 1.0}, langs=("en", "te"))
    with pytest.raises(SystemExit, match="no fertility measured"):
        run_g0(cfg, loader=fake_loader(), items_fn=fake_items())


def test_languages_default_to_fertility_order(tmp_path, patched):
    cfg = config(tmp_path, {"th": 2.13, "en": 1.0, "te": 6.25})
    out = run_g0(cfg, loader=fake_loader(), items_fn=fake_items())
    assert out["langs"] == ["en", "th", "te"]


from dllmfert.g0_run import canvas_from_lengths  # noqa: E402


def test_canvas_from_lengths_uses_p95_with_headroom():
    lens = list(range(100, 200)) + [900]      # one runaway must not set the size
    c = canvas_from_lengths(lens, headroom=1.5, block=32)
    assert 250 <= c <= 350
    assert c % 32 == 0


def test_canvas_from_lengths_has_a_floor():
    assert canvas_from_lengths([3, 4, 5], floor=128) == 128
    assert canvas_from_lengths([]) == 128


def test_measured_canvas_beats_fertility_scaling(tmp_path):
    """Sizing from fertility gave Telugu 3232 tokens of canvas for a few
    hundred tokens of output, and a diffusion step costs a + b*canvas -- so the
    arithmetic error was being charged to the paradigm under test."""
    cfg = G0Config(canvas_mode="measured", canvas_table={"te": 1440},
                   base_canvas=512, block_size=32)
    assert canvas_for(cfg, 6.25, "te") == 1440
    fert = G0Config(canvas_mode="fertility", base_canvas=512, block_size=32)
    assert canvas_for(fert, 6.25, "te") == 3200   # 512 * 6.25, already on-block


def test_measured_canvas_falls_back_when_a_language_is_missing():
    cfg = G0Config(canvas_mode="measured", canvas_table={"te": 1440},
                   base_canvas=512, block_size=32)
    assert canvas_for(cfg, 2.13, "th") == 1120   # fertility fallback, not a crash


def test_step_cap_is_never_looser_than_the_canvas():
    cfg = G0Config(expected_parallel_factor=2.0, step_cap_factor=3.0)
    for canvas in (256, 512, 1440, 3232):
        assert step_cap(cfg, canvas, "threshold") <= canvas + 1
    assert step_cap(G0Config(), 512, "naive") == 513
