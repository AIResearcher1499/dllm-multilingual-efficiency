#!/usr/bin/env bash
# Phase 4b: is Fast-dLLM's reported speed-up a function of the canvas?
# Read docs/prereg-g0.md Revision 8 first. The rule is frozen.
#
# Measures the accelerated arm at two canvases. The baseline is NOT run: their
# baseline sets steps = gen_length and the scheduler then commits exactly one
# position per step, so NFE_base(L) = L analytically. Verified by reading
# generate() and get_num_transfer_tokens, not assumed.
#
# Why steps = gen_length at both canvases: the dual-cache path caps refinement
# at steps_per_block = steps // num_blocks. With block_length 32 and
# steps = gen_length that is 32 for every L, so the cap is identical across
# canvases and (at a measured 4.4 steps/block) never binding. Setting steps any
# other way would make the two canvases incomparable.
#
# --log_samples is not optional here: the last run lost every per-item record
# to a `tail -60` in the script, which cost us the content-length distribution
# and any paired accuracy comparison.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="${HF_HOME:-/runpod-volume/hf}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/root/Fast-dLLM
VENV=/root/fdllm-venv
OUT=/root/dllm-fertility/data/canvas_runs

echo "===== 0. environment ====="
[ -d "$REPO" ] || git clone --depth 1 https://github.com/NVlabs/Fast-dLLM.git "$REPO"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv "$VENV" --python 3.11
  # cu121, pinned: the default PyPI wheel bundles the newest CUDA and refuses
  # to start on this pod's 12.4 driver. Removing this pin once already cost a run.
  uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu121
  uv pip install --python "$VENV/bin/python" -r "$REPO/v1/requirements.txt"
fi
"$VENV/bin/python" - <<'PY'
import torch, transformers
from importlib.metadata import version
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| lm_eval", version("lm_eval"))
assert torch.cuda.is_available(), "torch cannot see the GPU -- wrong CUDA build for this driver"
print("device:", torch.cuda.get_device_name(0))
PY

mkdir -p "$OUT"
cd "$REPO/v1/llada"

run_canvas () {
  local L="$1"
  echo ""
  echo "===== canvas L=$L ====="
  # steps=L on purpose; see the header. block_length fixed at 32 across canvases.
  "$VENV/bin/accelerate" launch eval_llada.py \
    --tasks gsm8k --num_fewshot 8 --limit 100 \
    --confirm_run_unsafe_code --model llada_dist \
    --batch_size 2 \
    --log_samples --output_path "$OUT/L$L" \
    --model_args "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=$L,steps=$L,block_length=32,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,is_check_greedy=False" \
    > "$OUT/L$L.log" 2>&1
  echo "exit=$?"
  # Echo the numbers into the console log too, so a failed fetch does not lose them.
  grep -E "Total number of tokens|Total time taken|Tokens per second|Total NFE" "$OUT/L$L.log" || true
  grep -E "^\|gsm8k|flexible-extract|strict-match" "$OUT/L$L.log" || true
}

# 512 first: it is the new data point and the cheaper one, so a crash costs less.
run_canvas 512
run_canvas 1024

echo ""
echo "===== canvas sweep complete ====="
du -sh "$OUT" 2>/dev/null
