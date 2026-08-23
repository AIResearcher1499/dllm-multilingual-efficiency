"""Stage 2 runner — docs/prereg-g0.md. GPU. No training.

Writes one row per (lang, arm, mode, item) to data/g0.jsonl. The verdict is
computed by dllmfert.metrics and is not re-implemented here.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from dllmfert import AR_MODEL, DLLM_MODELS, MIN_ITEMS
from dllmfert.answers import is_correct
from dllmfert.sampler import NAIVE, THRESHOLD, decode, trim_at

INSTRUCTION = (
    "Solve this math problem. Reason step by step, then state the final "
    "numeric answer.\n\n"
)


def build_prompt(tokenizer, question: str, exemplars: list[dict] | None = None
                 ) -> list[int]:
    """Chat-formatted prompt ids.

    All three arms are instruction-tuned, and feeding them a bare string is
    not a small infidelity. Measured on an A100 without the template: Dream
    produced **zero** tokens in Telugu and LLaDA produced three to nine tokens
    for a grade-school word problem. Those are not model failures; they are
    prompts the models were never trained to read.
    """
    msgs = []
    for ex in exemplars or []:
        msgs.append({"role": "user", "content": INSTRUCTION + ex["question"]})
        msgs.append({"role": "assistant", "content": ex["answer"]})
    msgs.append({"role": "user", "content": INSTRUCTION + question})
    try:
        rendered = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    except (AttributeError, ValueError, TypeError):
        rendered = "\n\n".join(m["content"] for m in msgs)
    return tokenizer.encode(rendered, add_special_tokens=False)


def stop_ids_for(tokenizer) -> set[int]:
    ids = {i for i in (tokenizer.eos_token_id, tokenizer.pad_token_id)
           if i is not None}
    for name in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
            ids.add(int(tid))
    return ids


@dataclass
class G0Config:
    out: Path = Path("data/g0.jsonl")
    fertility: Path = Path("data/fertility.json")
    langs: tuple[str, ...] = ()
    dllm_models: tuple[str, ...] = DLLM_MODELS
    ar_model: str = AR_MODEL
    modes: tuple[str, ...] = (NAIVE, THRESHOLD)
    threshold: float = 0.9
    block_size: int = 32
    repetition_penalty: float = 1.0
    """Phase 3 intervention (prereg Revision 6). 1.0 reproduces Phase 2
    exactly; higher values suppress self-repetition so that its effect on the
    *measured* parallel_factor can be observed at fixed language."""
    base_canvas: int = 256
    canvas_mode: str = "measured"   # or "fertility" / "shared", both controls
    canvas_table: dict = field(default_factory=dict)
    n: int = 100
    shots: int = 3
    lang_threshold: float = 0.8
    # A generation that never terminates costs canvas steps: 3233 of them at
    # Telugu's canvas, or 18.6 minutes for one item. One Dream/te row in four
    # did exactly that. The cap is set from expected content rather than from
    # canvas, so a healthy generation is untouched and a runaway is bounded.
    step_cap_factor: float = 3.0
    expected_parallel_factor: float = 2.0
    resume: bool = False
    device: str | None = None
    mask_id: int | None = None
    keep_text: bool = False
    warmup: bool = True
    extra: dict = field(default_factory=dict)


def canvas_from_lengths(lengths: list[int], *, headroom: float = 1.5,
                       block: int = 32, floor: int = 128) -> int:
    """Canvas from an observed length distribution, the way a practitioner
    would size one: the 95th percentile plus headroom."""
    if not lengths:
        return floor
    xs = sorted(lengths)
    p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
    want = max(floor, p95 * headroom)
    return int(math.ceil(want / block) * block)


def canvas_for(cfg: G0Config, fertility: float, lang: str | None = None) -> int:
    """Canvas budget per language.

    `measured` is the honest default. Sizing from fertility alone gave Telugu a
    3232-token canvas for a few hundred tokens of output, because output length
    does **not** scale as steeply as fertility does — Qwen's Telugu runs 3.02x
    English against a fertility of 6.25x. Thai wasted 73% of its canvas. A
    diffusion step costs `a + b*canvas`, so that arithmetic error was being
    charged to the paradigm under test.

    `fertility` and `shared` are kept as controls: the first shows what naive
    scaling costs, the second what a language-blind budget costs. Both are part
    of the finding; neither should be the operating point.
    """
    if cfg.canvas_mode == "measured":
        if lang and lang in cfg.canvas_table:
            return int(cfg.canvas_table[lang])
        want = cfg.base_canvas * max(1.0, fertility)   # fall back, and say so
    elif cfg.canvas_mode == "shared":
        want = cfg.base_canvas
    elif cfg.canvas_mode == "fertility":
        want = cfg.base_canvas * max(1.0, fertility)
    else:
        raise SystemExit(f"unknown canvas_mode {cfg.canvas_mode!r}")
    block = max(1, cfg.block_size)
    return int(math.ceil(want / block) * block)


def load_done(path: Path) -> set[tuple]:
    """Resume keys on the full cell, not the item id: the same item is decoded
    once per (lang, arm, mode) and keying on id alone would skip most of them."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            # The penalty is part of the key. Without it a sweep would treat
            # rows already produced at penalty 1.0 as covering every other
            # setting and silently return the same numbers five times.
            done.add((r["lang"], r["arm"], r["mode"], r["id"],
                      r.get("repetition_penalty", 1.0)))
    return done


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_row(*, lang, arm, mode, item, canvas, canvas_mode, trace_like, text,
              wall_clock, error=None, keep_text=False, lang_expected=None,
              lang_threshold=0.8, repetition_penalty=1.0) -> dict:
    from dllmfert.metrics import padding_waste

    generated = trace_like.get("tokens_generated")
    row = {
        "lang": lang,
        "arm": arm,
        "mode": mode,
        "id": item["id"],
        "gold": item.get("gold"),
        "canvas": canvas,
        "canvas_mode": canvas_mode,
        "nfe": trace_like.get("nfe"),
        "parallel_factor": trace_like.get("parallel_factor"),
        "tokens_generated": generated,
        "padding_waste": (padding_waste(canvas, generated)
                          if generated is not None else None),
        "wall_clock": wall_clock,
        "acc": is_correct(text, item["gold"]) if text else False,
        "padding_positions": trace_like.get("padding_positions"),
        "stopped": trace_like.get("stopped"),
        "hit_step_cap": trace_like.get("hit_step_cap", False),
        "repetition_penalty": repetition_penalty,
        "error": error,
    }
    # Which box produced this row, and whether anything else was on the GPU at
    # the time. wall_clock from two different machines must never be compared,
    # and after files are merged the row itself is the only place that can say so.
    from dllmfert.provenance import hardware_provenance, timing_is_trustworthy

    prov = hardware_provenance()
    row["hw"] = prov
    row["timing_trustworthy"] = timing_is_trustworthy(prov)
    # Per-step record for the M1 detector test (prereg Revision 6 P2). Kept
    # only when text is kept: without the text there is nothing to compare the
    # signal against, and the lists are large.
    if keep_text:
        row["finalised_per_step"] = trace_like.get("finalised_per_step")
        row["commit_step"] = trace_like.get("commit_step")
        row["content_tokens"] = trace_like.get("content_tokens")
    if text and lang_expected:
        from dllmfert.language import judge

        try:
            row.update(judge(text, lang_expected, threshold=lang_threshold))
        except Exception as exc:  # noqa: BLE001 — a row must not die on langid
            row["lang_error"] = str(exc)
    if keep_text:
        # Preflight keeps the text. "0 errors" said nothing useful last time;
        # three lines of output would have shown the broken prompt instantly.
        row["text"] = text[:2000]
    return row


def step_cap(cfg: G0Config, canvas: int, mode: str) -> int:
    """Bound on denoising steps for one item.

    Naive decoding is one token per step by definition, so its budget is the
    canvas. Threshold decoding should finish in about canvas/parallel_factor
    steps; anything past a few multiples of that is a generation that is not
    going to terminate, and it is cheaper to record that than to pay for it.
    """
    if mode == NAIVE:
        return canvas + 1
    # Never looser than filling the canvas one token at a time: with a canvas
    # sized from measured content, that is already the natural bound, and the
    # cap only bites when a generation is not going to terminate.
    expected = canvas / max(1.0, cfg.expected_parallel_factor)
    return min(canvas + 1, max(64, int(cfg.step_cap_factor * expected)))


def run_dllm_item(model_bits, item, *, cfg, lang, arm, mode, canvas):
    """One diffusion decode. Returns (trace_like, text, seconds)."""
    from dllmfert.models import make_scorer

    model, tok, device, (mask_id, shift) = model_bits
    prompt_ids = build_prompt(tok, item["question"], item.get("exemplars"))
    stops = stop_ids_for(tok)
    scorer = make_scorer(model, prompt_ids, canvas=canvas, mask_id=mask_id,
                         device=device, shift=shift,
                         repetition_penalty=cfg.repetition_penalty)
    cap = step_cap(cfg, canvas, mode)
    t0 = time.perf_counter()
    trace = decode(scorer, canvas=canvas, mode=mode, threshold=cfg.threshold,
                   block_size=cfg.block_size, stop_ids=stops, max_steps=cap)
    seconds = time.perf_counter() - t0
    kept = [t for t in trace.tokens[: trace.content_len] if t >= 0]
    return (
        {"nfe": trace.nfe, "parallel_factor": trace.parallel_factor,
         "tokens_generated": trace.content_len,
         "padding_positions": trace.padding_positions,
         "stopped": trace.stopped, "hit_step_cap": trace.hit_step_cap,
         "finalised_per_step": list(trace.finalised_per_step),
         # Only the content region: trailing padding carries no signal and
         # would dominate the record for a high-fertility canvas.
         "commit_step": list(trace.commit_step[: trace.content_len]),
         # Token ids as well as the decoded text: the M1 detector reads the
         # prefix committed so far, and `text` alone cannot be cut at a step
         # because positions do not commit in order.
         "content_tokens": list(trace.tokens[: trace.content_len])},
        tok.decode(kept, skip_special_tokens=True),
        seconds,
    )


def run_ar_item(model_bits, item, *, canvas):
    from dllmfert.models import ar_generate

    model, tok, device, _bits = model_bits
    prompt_ids = build_prompt(tok, item["question"], item.get("exemplars"))
    t0 = time.perf_counter()
    new, steps = ar_generate(model, tok, prompt_ids, max_new_tokens=canvas,
                             device=device)
    seconds = time.perf_counter() - t0
    stop_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
    kept = trim_at(new, stop_ids)
    return (
        {"nfe": steps, "parallel_factor": 1.0, "tokens_generated": len(kept),
         "padding_positions": 0, "stopped": len(kept) < len(new),
         "hit_step_cap": False},
        tok.decode(kept, skip_special_tokens=True),
        seconds,
    )


def default_loader(model_id: str, cfg: G0Config):
    from dllmfert.models import detect_logit_shift, find_mask_id, load

    model, tok, device = load(model_id, cfg.device)
    mask_id = cfg.mask_id if cfg.mask_id is not None else None
    if mask_id is None:
        try:
            mask_id = find_mask_id(tok, model)
        except SystemExit:
            return model, tok, device, (-1, 0)   # AR arm uses neither
    cal = detect_logit_shift(model, tok, mask_id=mask_id, device=device)
    print(f"  logit alignment: shift={cal['shift']} margin={cal['margin']:.2f} "
          f"({cal['convention']})", flush=True)
    if cal["margin"] < 0.5:
        print("  !! alignment is ambiguous; treat this arm's output as suspect",
              flush=True)
    return model, tok, device, (mask_id, cal["shift"])


def run_g0(cfg: G0Config, *, loader=default_loader, items_fn=None) -> dict:
    if items_fn is None:
        from dllmfert.corpus import load_items

        items_fn = load_items
    fert = json.loads(Path(cfg.fertility).read_text())
    ratios = fert["fertility_ratio"]
    langs = cfg.langs or tuple(sorted(ratios, key=lambda l: ratios[l]))
    missing = [l for l in langs if l not in ratios]
    if missing:
        raise SystemExit(
            f"no fertility measured for {missing}; run `dllmfert fertility` "
            "first — the regression has no x-value for those languages"
        )
    done = load_done(cfg.out) if cfg.resume else set()
    if cfg.out.exists() and not cfg.resume:
        raise SystemExit(f"{cfg.out} exists; pass --resume or another --out")

    # An empty ar_model drops the baseline. The Phase 3 penalty sweep compares
    # the diffusion arm against itself across settings, so paying for AR
    # generation five times over would buy nothing.
    arms = [(m, "dllm") for m in cfg.dllm_models]
    if cfg.ar_model:
        arms.append((cfg.ar_model, "ar"))
    counts: dict[str, int] = {}
    if cfg.repetition_penalty != 1.0 and any(k == "ar" for _, k in arms):
        # The AR baseline never sees the penalty -- ar_generate does not take
        # it -- so sweeping it would append rows that are byte-identical in
        # content but distinct under the resume key: silent duplicates that
        # would then be averaged as if they were independent measurements.
        raise SystemExit(
            "repetition_penalty != 1.0 sweeps the diffusion arm only; "
            "pass --dllm-models without an AR arm for the Phase 3 sweep")

    for model_id, kind in arms:
        arm = model_id.split("/")[-1]
        modes = cfg.modes if kind == "dllm" else ("ar",)
        planned = [(l, m) for l in langs for m in modes
                   if any((l, arm, m, f"{l}-{i}", cfg.repetition_penalty) not in done
                          for i in range(cfg.n))]
        if not planned:
            print(f"arm {arm}: nothing to do", flush=True)
            continue
        print(f"loading {model_id}", flush=True)
        bits = loader(model_id, cfg)
        if cfg.warmup:
            # The first generation after a model loads pays for kernel
            # autotuning: measured 258 ms/token against 20 ms/token for every
            # call after it. Burn one item rather than let a 13x outlier into
            # the timings.
            warm = items_fn(langs[0], 1)
            if warm:
                try:
                    if kind == "dllm":
                        run_dllm_item(bits, warm[0], cfg=cfg, lang=langs[0],
                                      arm=arm, mode=modes[0], canvas=64)
                    else:
                        run_ar_item(bits, warm[0], canvas=32)
                    print(f"  warmup done for {arm}", flush=True)
                except Exception as exc:  # noqa: BLE001 — warmup must never fail a run
                    print(f"  warmup skipped: {exc}", flush=True)
        for lang in langs:
            canvas = canvas_for(cfg, ratios[lang], lang)
            items = items_fn(lang, cfg.n)
            if cfg.shots and kind in ("dllm", "ar"):
                from dllmfert.corpus import load_exemplars

                shots = load_exemplars(lang, cfg.shots)
                for it in items:
                    it["exemplars"] = shots
            for mode in modes:
                for item in items:
                    key = (lang, arm, mode, item["id"],
                           cfg.repetition_penalty)
                    if key in done:
                        continue
                    err, text, seconds = None, "", 0.0
                    trace_like: dict = {}
                    try:
                        if kind == "dllm":
                            trace_like, text, seconds = run_dllm_item(
                                bits, item, cfg=cfg, lang=lang, arm=arm,
                                mode=mode, canvas=canvas)
                        else:
                            trace_like, text, seconds = run_ar_item(
                                bits, item, canvas=canvas)
                    except Exception as exc:  # noqa: BLE001 — the row records it
                        err = str(exc)
                    row = build_row(lang=lang, arm=arm, mode=mode, item=item,
                                    canvas=canvas, canvas_mode=cfg.canvas_mode,
                                    trace_like=trace_like, text=text,
                                    wall_clock=seconds, error=err,
                                    keep_text=cfg.keep_text, lang_expected=lang,
                                    lang_threshold=cfg.lang_threshold,
                                    repetition_penalty=cfg.repetition_penalty)
                    append_row(cfg.out, row)
                    counts[arm] = counts.get(arm, 0) + 1
                print(f"[{arm}/{lang}/{mode}] canvas={canvas} "
                      f"done={counts.get(arm, 0)}", flush=True)
        del bits
        _empty_cache()
    return {"rows_written": counts, "langs": list(langs),
            "min_items": MIN_ITEMS, "out": str(cfg.out)}


def _empty_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — best effort between arms
        pass


def run_k2(cfg: G0Config, *, lang: str, canvases: list[int], mode: str = THRESHOLD,
           loader=None, items_fn=None) -> list[dict]:
    """Canvas sweep at a fixed language. Isolates S(L) — see dllmfert.k2.

    Deliberately not folded into run_g0: bundling a mechanism check with the
    main grid is how you end up unable to tell which variable moved.
    """
    if loader is None:
        loader = default_loader
    if items_fn is None:
        from dllmfert.corpus import load_items

        items_fn = load_items
    model_id = cfg.dllm_models[0]
    arm = model_id.split("/")[-1]
    print(f"K2 sweep: {arm} lang={lang} mode={mode} canvases={canvases}", flush=True)
    bits = loader(model_id, cfg)
    items = items_fn(lang, cfg.n)
    rows = []
    for canvas in canvases:
        for item in items:
            err, text, seconds = None, "", 0.0
            trace_like: dict = {}
            try:
                trace_like, text, seconds = run_dllm_item(
                    bits, item, cfg=cfg, lang=lang, arm=arm, mode=mode,
                    canvas=canvas)
            except Exception as exc:  # noqa: BLE001 — the row records it
                err = str(exc)
            row = build_row(lang=lang, arm=arm, mode=mode, item=item,
                            canvas=canvas, canvas_mode="k2-sweep",
                            trace_like=trace_like, text=text,
                            wall_clock=seconds, error=err)
            append_row(cfg.out, row)
            rows.append(row)
        ok = [r for r in rows if r["canvas"] == canvas and not r["error"]]
        mean_nfe = sum(r["nfe"] for r in ok) / len(ok) if ok else None
        print(f"  canvas={canvas} mean_nfe={mean_nfe}", flush=True)
    return rows
