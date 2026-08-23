# Phase 3 runbook

Read `docs/prereg-g0.md` Revision 6 first. Both decision rules are frozen; this
file only says how to produce the data they consume.

## What it costs and why it is small

English only, diffusion arm only, one canvas, one threshold. The AR baseline is
dropped on purpose (`--ar-model ""`) -- P1 compares the diffusion arm against
itself across penalty settings, so paying for AR generation five times buys
nothing. `run_g0` refuses to sweep an AR arm rather than letting it produce
duplicate rows under distinct resume keys.

Estimated: 5 settings x 50 items x ~2.5 s ~= 11 min of A100 time per arm.

## Run

```bash
for R in 1.0 1.05 1.1 1.2 1.4; do
  PYTHONUNBUFFERED=1 uv run dllmfert g0 \
    --langs en --n 50 --modes threshold \
    --dllm-models GSAI-ML/LLaDA-8B-Instruct \
    --ar-model "" \
    --canvas-mode measured --canvas-table data/canvas_llada.json \
    --fertility data/fertility_llada_full.json \
    --shots 3 --keep-text --resume \
    --repetition-penalty "$R" \
    --out data/phase3.jsonl
done
uv run dllmfert phase3 --rows data/phase3.jsonl
```

Repeat with `--dllm-models Dream-org/Dream-v0-Instruct-7B` and the Qwen canvas
table; P1 asks for both arms.

`--keep-text` is required, not optional: without it there is no text to measure
distinctness on and no `commit_step`/`content_tokens` for the M1 test, and the
whole run would have to be paid for twice.

## Reading the result

`dllmfert phase3` prints one decision per arm per rule. The outcomes that end
work are as informative as the ones that do not:

- **P1 `KILL_K3`** -- suppressing repetition did not move the measured
  parallel_factor. The artifact reading of Phase 2 is wrong. Write it up and
  stop; do not go looking for a third reading.
- **P1 `CONFOUNDED_WITH_LENGTH`** -- the correlation survives only because the
  penalty shortened the output. Report as confounded, not as support.
- **P2 `M1_DROPPED`** -- parallel_factor is not an earlier signal than counting
  repeated n-grams. Ship the measurement plus M3 and make no detector claim.

Write `docs/phase3-result-<date>.md`. Do not edit `docs/prereg-g0.md`.
