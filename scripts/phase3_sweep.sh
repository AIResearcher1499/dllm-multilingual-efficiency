#!/usr/bin/env bash
# Phase 3 P1/P2 sweep. Read docs/prereg-g0.md Revision 6 first; the decision
# rules are frozen and this script only produces the data they consume.
#
# English only, diffusion arms only, one canvas, one threshold. The AR baseline
# is dropped deliberately -- P1 compares each diffusion arm against itself
# across penalty settings, so paying for AR generation five times buys nothing,
# and run_g0 refuses to sweep it anyway.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

N="${SWEEP_N:-50}"
OUT=data/phase3.jsonl
PENALTIES="${PENALTIES:-1.0 1.05 1.1 1.2 1.4}"

run_arm() {
  local model="$1" fert="$2" canvas="$3" label="$4"
  for R in $PENALTIES; do
    echo "=== $label penalty=$R ==="
    # No `set -e`: one failed cell must not abandon the settings after it, or
    # the sweep would return a truncated ladder that still looks monotone.
    uv run dllmfert g0 \
      --langs en --n "$N" --modes threshold \
      --dllm-models "$model" --ar-model "" \
      --canvas-mode measured --canvas-table "$canvas" --fertility "$fert" \
      --shots 3 --keep-text --resume \
      --repetition-penalty "$R" --out "$OUT" 2>&1 | tail -5
    echo "  rows now: $(wc -l < $OUT 2>/dev/null || echo 0)"
  done
}

run_arm GSAI-ML/LLaDA-8B-Instruct data/fertility_llada_full.json \
        data/canvas_llada.json LLaDA
run_arm Dream-org/Dream-v0-Instruct-7B data/fertility.json \
        data/canvas.json Dream

echo "=== sweep complete: $(wc -l < $OUT) rows ==="
uv run dllmfert phase3 --rows "$OUT" 2>&1 | tail -40
