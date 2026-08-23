#!/usr/bin/env bash
# Phase 4 Gate A: reproduce Fast-dLLM's published GSM8K accuracy through their
# published code. Read docs/prereg-g0.md Revision 7 first.
#
# Gate: |accuracy - 76.0| <= 3.0 points. NOT_REPRODUCED otherwise -> stop and
# write it up; a critique built on a configuration we could not reproduce is
# worthless.
#
# Deliberate deviations, recorded here rather than discovered later:
#   - batch_size=2, not the class default of 32. Memory here is dominated by
#     full-sequence logits: LLaDA's vocabulary is ~126k, an 8-shot GSM8K prompt
#     plus a 1024 canvas is ~2.4k positions, and batch 8 already OOMs an 80GB
#     A100 (it asked for 23.9 GiB in one allocation). Gate A is scored on
#     accuracy, which does not depend on batch size. Tokens/second does, so it
#     is reported descriptively and never compared against their 19.3.
#   - Their released eval_gsm8k.sh runs length=256, 5-shot. The 76.0 headline is
#     the 1024 / 8-shot configuration, so that is what is set here.
#   - is_check_greedy=False is their own documented recommendation (see the
#     docstring in eval_llada.py) and does not apply to generate_until tasks.
#     Listed for completeness, not because it is a departure.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="${HF_HOME:-/runpod-volume/hf}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/root/Fast-dLLM
VENV=/root/fdllm-venv

echo "===== 0. separate environment ====="
# Their requirements pin transformers==4.49.0; this project pins 4.46.2 because
# Dream's bundled code breaks above it. Installing theirs into our venv would
# silently break our own harness, so they get their own.
[ -d "$REPO" ] || git clone --depth 1 https://github.com/NVlabs/Fast-dLLM.git "$REPO"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv "$VENV" --python 3.11
  "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
  # Pin the CUDA build. The default PyPI wheel bundles the *newest* CUDA and
  # refuses to run on this pod's 12.4 driver ("NVIDIA driver is too old, found
  # version 12040"). Removing this pin once already cost a run.
  uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu121
  uv pip install --python "$VENV/bin/python" -r "$REPO/v1/requirements.txt"
fi
"$VENV/bin/python" - <<'PY'
import torch, transformers
from importlib.metadata import version
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| lm_eval", version("lm_eval"))
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch cannot see the GPU -- wrong CUDA build for this driver"
print("device:", torch.cuda.get_device_name(0))
PY

echo ""
echo "===== 0b. does their harness carry MGSM? (needed by P3, checked now) ====="
"$VENV/bin/python" - <<'PY'
try:
    from lm_eval.tasks import TaskManager
    names = TaskManager().all_tasks
    mg = sorted(n for n in names if "mgsm" in n.lower())
    print(f"mgsm tasks available: {len(mg)}")
    print(" ", " ".join(mg[:24]))
except Exception as e:
    print("could not enumerate tasks:", e)
PY

echo ""
echo "===== 1. Gate A: dual cache + parallel, GSM8K, 8-shot, gen_length 1024 ====="
cd "$REPO/v1/llada"
LIMIT="${GATEA_LIMIT:-100}"
set -x
"$VENV/bin/accelerate" launch eval_llada.py \
  --tasks gsm8k --num_fewshot 8 --limit "$LIMIT" \
  --confirm_run_unsafe_code --model llada_dist \
  --batch_size "${GATEA_BATCH:-2}" \
  --model_args model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=1024,steps=1024,block_length=32,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,is_check_greedy=False \
  2>&1 | tail -60
set +x
echo "===== Gate A complete ====="
