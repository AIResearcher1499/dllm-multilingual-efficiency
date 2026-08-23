#!/usr/bin/env bash
# Phase 4 P3: does Fast-dLLM's parallel-decoding gain survive a quality-matched
# accounting, and does it shrink more outside English?
# Rules: docs/prereg-g0.md Revision 7 (frozen) + Revision 9 (implementation).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="${HF_HOME:-/runpod-volume/hf}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/root/Fast-dLLM
VENV=/root/fdllm-venv
OUT=/root/dllm-fertility/data/p3
MODEL=GSAI-ML/LLaDA-8B-Instruct

echo "===== 0. environment ====="
[ -d "$REPO" ] || git clone --depth 1 https://github.com/NVlabs/Fast-dLLM.git "$REPO"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv "$VENV" --python 3.11
  # cu121 pinned: the default wheel bundles the newest CUDA and will not start
  # on this pod's 12.4 driver. Removing this pin once already cost a run.
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
cd "$REPO/v1/llada"

echo ""
echo "===== 0b. resolve the MGSM task names before spending anything ====="
# Hard stop on a missing task. A wrong name would burn hours before failing,
# and silently falling back to the direct-answer variant would be a different
# experiment: it emits a bare number and leaves no room for the degeneracy
# this phase exists to measure.
"$VENV/bin/python" - > /tmp/p3_tasks.txt <<'PYEOF'
from lm_eval.tasks import TaskManager
names = set(TaskManager().all_tasks)
missing = []
for lg in ("en", "te", "th", "ru"):
    hit = next((c for c in (f"mgsm_native_cot_{lg}", f"mgsm_cot_native_{lg}")
                if c in names), None)
    if hit is None:
        missing.append(lg)
    else:
        print(f"{lg} {hit}")
if missing:
    import sys
    print("MISSING " + ",".join(missing), file=sys.stderr)
    cand = sorted(n for n in names if n.startswith("mgsm"))
    print("available mgsm tasks: " + " ".join(cand[:40]), file=sys.stderr)
    sys.exit(1)
PYEOF
if [ $? -ne 0 ] || [ ! -s /tmp/p3_tasks.txt ]; then
  echo "FATAL: could not resolve mgsm native-CoT task names; nothing was run" >&2
  exit 1
fi
cat /tmp/p3_tasks.txt

# One cell = one (language, arm). Separate output dirs and logs so a failure
# late in the sweep cannot take the finished cells with it.
run_cell () {
  local task="$1" arm="$2" n="$3" extra="$4"
  local tag="${task}__${arm}"
  echo ""
  echo "===== cell $tag  (n=$n) ====="
  "$VENV/bin/accelerate" launch eval_llada.py \
    --tasks "$task" --limit "$n" \
    --confirm_run_unsafe_code --model llada_dist \
    --batch_size 1 \
    --log_samples --output_path "$OUT/$tag" \
    --model_args "model_path=$MODEL,gen_length=1024,steps=1024,block_length=32,use_cache=True,show_speed=True,is_check_greedy=False${extra}" \
    > "$OUT/$tag.log" 2>&1
  echo "exit=$?"
  grep -E "Total number of tokens|Total time taken|Tokens per second|Total NFE" "$OUT/$tag.log" || true
  grep -E "^\|mgsm" "$OUT/$tag.log" || true
  echo "per-item nfe records: $(grep -cE '^nfe: ' "$OUT/$tag.log" || echo 0)"
}

# English first (validates the pipeline on a known quantity), then Telugu (the
# most informative single cell), then Thai and Russian.
while read -r LG TASK; do
  [ -n "$TASK" ] || continue
  run_cell "$TASK" accel "${P3_N_ACC:-100}" ",threshold=0.9"
  run_cell "$TASK" base  "${P3_N_BASE:-20}" ""
done < /tmp/p3_tasks.txt

echo ""
echo "===== P3 sweep complete ====="
du -sh "$OUT"
