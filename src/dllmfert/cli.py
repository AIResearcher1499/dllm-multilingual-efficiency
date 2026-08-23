"""dllmfert fertility | g0 | verdict"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dllmfert import TOKENIZER


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path}")


def cmd_fertility(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from dllmfert.corpus import available_langs, encode_fn, load_parallel
    from dllmfert.fertility import fertility_ratios, stage1_verdict

    langs = available_langs()
    print(f"MGSM configs present: {langs}", flush=True)
    items = load_parallel(langs, args.n)
    print(f"{len(items)} parallel items x {len(langs)} languages", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    result = fertility_ratios(items, encode_fn(tok))
    result["tokenizer"] = args.tokenizer
    result["langs"] = langs
    verdict = stage1_verdict(result)
    out = {**result, "verdict": verdict}
    _write(Path(args.out), out)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"decision: {verdict['decision']}")


def cmd_g0(args: argparse.Namespace) -> None:
    from dllmfert.fertility import stage1_verdict
    from dllmfert.g0_run import G0Config, run_g0

    fert_path = Path(args.fertility)
    if not fert_path.exists():
        raise SystemExit(
            f"{fert_path} missing — run `dllmfert fertility` first. "
            "docs/prereg-g0.md forbids stage 2 before stage 1 clears its gate."
        )
    stage1 = json.loads(fert_path.read_text()).get("verdict") or stage1_verdict(
        json.loads(fert_path.read_text())
    )
    if stage1["decision"] != "PROCEED" and not args.force:
        raise SystemExit(
            f"stage 1 decision is {stage1['decision']}: {stage1['reads_as']}. "
            "Pass --force only with a written deviation."
        )
    cfg = G0Config(
        out=Path(args.out),
        fertility=fert_path,
        langs=tuple(args.langs) if args.langs else (),
        dllm_models=(tuple(args.dllm_models) if args.dllm_models
                     else G0Config.dllm_models),
        modes=tuple(args.modes),
        threshold=args.threshold,
        block_size=args.block_size,
        base_canvas=args.base_canvas,
        canvas_mode=args.canvas_mode,
        canvas_table=(json.loads(Path(args.canvas_table).read_text())
                      if args.canvas_table and Path(args.canvas_table).exists()
                      else {}),
        n=args.n,
        resume=args.resume,
        mask_id=args.mask_id,
        keep_text=args.keep_text,
        shots=args.shots,
        lang_threshold=args.lang_threshold,
        repetition_penalty=args.repetition_penalty,
        **({} if args.ar_model is None else {"ar_model": args.ar_model}),
    )
    print(json.dumps(run_g0(cfg), indent=2))


def cmd_k2(args: argparse.Namespace) -> None:
    from dllmfert.g0_run import G0Config, run_k2
    from dllmfert.k2 import k2_verdict

    cfg = G0Config(out=Path(args.out), fertility=Path(args.fertility),
                   n=args.n, threshold=args.threshold,
                   block_size=args.block_size, mask_id=args.mask_id,
                   resume=args.resume)
    if args.model:
        cfg.dllm_models = (args.model,)
    rows = run_k2(cfg, lang=args.lang, canvases=args.canvases, mode=args.mode)
    v = k2_verdict(rows)
    _write(Path(args.summary), v)
    print(json.dumps(v, indent=2, ensure_ascii=False))
    print(f"decision: {v['decision']}")


def cmd_phase3(args: argparse.Namespace) -> None:
    from dllmfert.phase3 import p1_verdict, p2_verdict

    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines()
            if l.strip()]
    out = {"p1_intervention": p1_verdict(rows), "p2_detector": p2_verdict(rows)}
    _write(Path(args.out), out)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    for name, block in out.items():
        for arm, res in block.items():
            print(f"{name} / {arm}: {res['decision']}")


def cmd_verdict(args: argparse.Namespace) -> None:
    from dllmfert.metrics import g0_verdict

    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    fert = json.loads(Path(args.fertility).read_text())["fertility_ratio"]
    arms = sorted({r["arm"] for r in rows})
    out = {arm: g0_verdict(rows, fert, arm) for arm in arms}
    _write(Path(args.out), out)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(prog="dllmfert")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fertility", help="stage 1; Mac, tokenizer only")
    f.add_argument("--n", type=int, default=250)
    f.add_argument("--tokenizer", default=TOKENIZER)
    f.add_argument("--out", default="data/fertility.json")
    f.set_defaults(func=cmd_fertility)

    g = sub.add_parser("g0", help="stage 2; GPU")
    g.add_argument("--out", default="data/g0.jsonl")
    g.add_argument("--fertility", default="data/fertility.json")
    g.add_argument("--langs", nargs="+", default=None)
    g.add_argument("--dllm-models", nargs="+", default=None,
                   help="restrict the diffusion arms; LEAN needs only Dream")
    g.add_argument("--modes", nargs="+", default=["naive", "threshold"],
                   choices=["naive", "threshold"])
    g.add_argument("--threshold", type=float, default=0.9)
    g.add_argument("--ar-model", default=None,
                   help="AR baseline; pass an empty string to drop it, which "
                        "the Phase 3 penalty sweep needs.")
    g.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="Phase 3 intervention (prereg rev 6): penalise tokens "
                        "already committed in the canvas. 1.0 = off, and "
                        "reproduces Phase 2 exactly.")
    g.add_argument("--block-size", type=int, default=32)
    g.add_argument("--base-canvas", type=int, default=256)
    g.add_argument("--canvas-mode", default="measured",
                   choices=["measured", "fertility", "shared"],
                   help="measured is the operating point; the other two are controls")
    g.add_argument("--canvas-table", default="data/canvas.json",
                   help="per-language canvas sized from observed output length")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--mask-id", type=int, default=None)
    g.add_argument("--resume", action="store_true")
    g.add_argument("--shots", type=int, default=3,
                   help="in-language few-shot exemplars; the language-confusion "
                        "mitigation from 2406.20052")
    g.add_argument("--lang-threshold", type=float, default=0.8)
    g.add_argument("--keep-text", action="store_true",
                   help="store generated text in each row; use for preflight")
    g.add_argument("--force", action="store_true",
                   help="run stage 2 despite a stage-1 PARK; needs a written deviation")
    g.set_defaults(func=cmd_g0)

    k = sub.add_parser(
        "k2",
        help="does NFE scale with canvas? Run this BEFORE the g0 grid",
    )
    k.add_argument("--lang", default="en",
                   help="fixed: varying language would confound canvas with content")
    k.add_argument("--canvases", nargs="+", type=int,
                   default=[256, 512, 1024, 2048])
    k.add_argument("--mode", default="threshold", choices=["naive", "threshold"])
    k.add_argument("--model", default=None)
    k.add_argument("--n", type=int, default=5)
    k.add_argument("--threshold", type=float, default=0.9)
    k.add_argument("--block-size", type=int, default=32)
    k.add_argument("--mask-id", type=int, default=None)
    k.add_argument("--out", default="data/k2.jsonl")
    k.add_argument("--fertility", default="data/fertility.json")
    k.add_argument("--summary", default="data/k2_summary.json")
    k.add_argument("--resume", action="store_true")
    k.set_defaults(func=cmd_k2)

    v = sub.add_parser("verdict")
    p3 = sub.add_parser("phase3", help="P1/P2 verdicts against prereg rev 6")
    p3.add_argument("--rows", default="data/phase3.jsonl")
    p3.add_argument("--out", default="data/phase3_summary.json")
    p3.set_defaults(func=cmd_phase3)

    v.add_argument("--rows", default="data/g0.jsonl")
    v.add_argument("--fertility", default="data/fertility.json")
    v.add_argument("--out", default="data/g0_summary.json")
    v.set_defaults(func=cmd_verdict)

    args = p.parse_args()
    args.func(args)
