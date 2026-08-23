#!/usr/bin/env bash
# Bootstrap a borrowed GPU box for stage 2. Read docs/prereg-g0.md first.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU =="
nvidia-smi

echo "== env =="
# Local disk for the venv, network volume for weights. Building a venv on a
# network mount took twenty minutes; the weights are large sequential files
# that must outlive the pod.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/venv}"
export HF_HOME="${HF_HOME:-/runpod-volume/hf}"
mkdir -p "$HF_HOME"
echo "venv -> $UV_PROJECT_ENVIRONMENT   hf -> $HF_HOME"
df -h "$HF_HOME" 2>/dev/null | tail -1
# The installer drops uv in ~/.local/bin, which a non-interactive ssh session
# does not have on PATH; put it there before anything tries to call uv.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || {
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}
command -v uv >/dev/null || { echo "uv still not on PATH after install" >&2; exit 1; }
uv --version
uv sync --extra gpu
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
uv run pytest -q

echo "== stage 1 (cheap; also confirms the fertility file matches this box) =="
[ -f data/fertility.json ] || uv run dllmfert fertility

cat <<'NEXT'

Stage 2, in this order. The first two steps answer different questions and
must not be merged -- one is plumbing, the other can kill the design.

  # 1. PREFLIGHT (plumbing only): does the model load, does the mask id
  #    resolve, does decoding terminate, does scoring survive Telugu?
  PYTHONUNBUFFERED=1 uv run dllmfert g0 --langs en te --n 3 \
      --out data/preflight.jsonl

  # 2. K2 (mechanism): does NFE scale with the canvas under threshold
  #    decoding? Canvas is swept at ONE language on purpose -- varying
  #    language would move canvas and content together and the slope would
  #    mean nothing. If this fires, a larger canvas is nearly free, mechanism
  #    A cannot act, and the full grid is not worth running yet.
  PYTHONUNBUFFERED=1 uv run dllmfert k2 --lang en --canvases 256 512 1024 2048

  # 3. full grid, only after K2 says PROCEED
  PYTHONUNBUFFERED=1 uv run dllmfert g0 --resume
  uv run dllmfert verdict

Then write docs/stage2-result-<date>.md. Do not edit docs/prereg-g0.md.
NEXT
