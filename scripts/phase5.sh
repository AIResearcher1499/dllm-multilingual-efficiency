#!/usr/bin/env bash
# Phase 5: is Fast-dLLM's reported tokens/second anti-correlated with output
# quality across languages? Rules frozen in docs/prereg-g0.md Revision 10.
#
# SERIAL by construction: throughput is the measurement, and two cells sharing a
# box put PCIe and CPU contention straight into it. One cell at a time, always.
#
# Verified against their code before writing this:
#   - LLaDA counts tokens AFTER truncating at the stop sequence
#     (eval_llada.py:357-366); Dream counts the whole window minus EOS BEFORE
#     truncation (eval.py:325). Different conventions, and the frozen rule
#     correlates within each model separately, so this is reported, not fixed.
#   - Dream reports "Generated token num per second", LLaDA "Tokens per second".
#     Dream reports no NFE at all.
#   - Dream's use_cache swaps in generation_utils_block rather than passing a
#     kwarg; block_length defaults to 32 there, matching LLaDA's 32.
#   - Dream's max_length defaults to 2048 against LLaDA's 4096; set explicitly.
#   - Both wrappers index sequences[0], so batch must be 1.
#   - mgsm_native_cot is zero-shot: "Question: ...\nStep-by-Step Answer:".
#     Few-shot is not available at this canvas -- LLaDA's context is 4096 and a
#     high-fertility language plus a 1024 canvas would not fit the exemplars.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="${HF_HOME:-/runpod-volume/hf}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/root/Fast-dLLM
VENV=/root/fdllm-venv
OUT=/root/dllm-fertility/data/phase5
N="${P5_N:-100}"
LANGS="${P5_LANGS:-en zh es fr de ja sw ru th bn te}"

echo "===== environment ====="
[ -d "$REPO" ] || git clone --depth 1 https://github.com/NVlabs/Fast-dLLM.git "$REPO"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv "$VENV" --python 3.11
  uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu121
  uv pip install --python "$VENV/bin/python" -r "$REPO/v1/requirements.txt"
fi
"$VENV/bin/python" - <<'PY'
import torch, transformers
from importlib.metadata import version
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| lm_eval", version("lm_eval"))
assert torch.cuda.is_available(), "wrong CUDA build for this driver"
print("device:", torch.cuda.get_device_name(0))
PY
mkdir -p "$OUT"

run_llada () {
  local lg="$1" tag="llada_$1"
  echo ""; echo "===== $tag ====="
  ( cd "$REPO/v1/llada" && "$VENV/bin/accelerate" launch eval_llada.py \
      --tasks "mgsm_native_cot_$lg" --limit "$N" --confirm_run_unsafe_code \
      --model llada_dist --batch_size 1 \
      --log_samples --output_path "$OUT/$tag" \
      --model_args "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=1024,steps=1024,block_length=32,use_cache=True,threshold=0.9,show_speed=True,is_check_greedy=False" \
  ) > "$OUT/$tag.log" 2>&1
  echo "exit=$?"; grep -E "Tokens per second|Total NFE|^\|mgsm" "$OUT/$tag.log" || true
}

run_dream () {
  local lg="$1" tag="dream_$1"
  echo ""; echo "===== $tag ====="
  ( cd "$REPO/v1/dream" && "$VENV/bin/accelerate" launch eval.py \
      --tasks "mgsm_native_cot_$lg" --limit "$N" --confirm_run_unsafe_code \
      --model dream --batch_size 1 \
      --log_samples --output_path "$OUT/$tag" \
      --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,max_new_tokens=1024,diffusion_steps=1024,max_length=4096,alg=confidence_threshold,threshold=0.9,use_cache=True,apply_chat_template=True" \
  ) > "$OUT/$tag.log" 2>&1
  echo "exit=$?"; grep -E "Generated token num per second|^\|mgsm" "$OUT/$tag.log" || true
}

# English first on each arm: it validates the arm on a known quantity before the
# expensive tail. Then the rest, cheapest tokeniser first.
# P5_ARMS lets a resumed run cover only what a terminated pod left behind.
for arm in ${P5_ARMS:-llada dream}; do
  for lg in $LANGS; do "run_$arm" "$lg"; done
done

echo ""; echo "===== phase 5 complete ====="; du -sh "$OUT"
