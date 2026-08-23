#!/usr/bin/env bash
# Day one on a new GPU box. Answers, in one output, everything needed before
# any experiment is worth starting. Nothing here costs more than a few minutes.
#
#   bash scripts/box_bringup.sh 2>&1 | tee bringup.txt
#
# Send bringup.txt back. Every check reports rather than aborting, so one
# failure does not hide the state of everything after it.
set -u
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; }
note() { printf '        %s\n' "$1"; }

echo "===== 1. GPU ====="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
  note "compute processes now: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l | tr -d ' ')"
else
  bad "nvidia-smi not found"
fi

echo ""
echo "===== 2. Outbound network -- this blocks everything if it fails ====="
for host in https://github.com https://huggingface.co https://pypi.org; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$host" 2>/dev/null || echo 000)
  [ "$code" != "000" ] && ok "$host ($code)" || bad "$host unreachable"
done
note "no huggingface = no model weights = nothing runs, whatever git does"

echo ""
echo "===== 3. Disk for weights ====="
for d in "${HF_HOME:-$HOME/.cache/huggingface}" "$HOME" .; do
  df -h "$d" 2>/dev/null | tail -1 | awk -v d="$d" '{printf "  %-40s %s free of %s\n", d, $4, $2}'
done
note "each 8B checkpoint is ~16 GB; the grid wants LLaDA and Dream"

echo ""
echo "===== 4. Toolchain ====="
for c in git python3 uv; do
  if command -v "$c" >/dev/null; then ok "$c $($c --version 2>&1 | head -1)"; else bad "$c missing"; fi
done
note "no uv: curl -LsSf https://astral.sh/uv/install.sh | sh"

echo ""
echo "===== 5. Repository ====="
if [ -d .git ]; then
  ok "in a git repo at $(pwd)"
  note "HEAD $(git log --format='%h %s' -1 2>/dev/null | cut -c1-60)"
else
  bad "not in the repository -- clone it first:"
  note "git clone https://github.com/AIResearcher1499/dllm-multilingual-efficiency.git"
fi

echo ""
echo "===== 6. Environment and tests ====="
if command -v uv >/dev/null && [ -f pyproject.toml ]; then
  uv sync --extra fertility >/dev/null 2>&1 && ok "uv sync --extra fertility" || bad "uv sync failed"
  out=$(uv run pytest -q 2>&1 | tail -1)
  case "$out" in *failed*) bad "pytest: $out";; *passed*) ok "pytest: $out";; *) bad "pytest: $out";; esac
else
  note "skipped: need uv and pyproject.toml"
fi

echo ""
echo "===== 7. Torch sees the GPU ====="
if command -v uv >/dev/null; then
  uv run --extra gpu python - 2>&1 <<'PY' | sed 's/^/  /'
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i} {p.name} {p.total_memory/2**30:.0f}GiB sm{p.major}{p.minor}")
except Exception as e:
    print("FAIL:", type(e).__name__, e)
PY
fi

echo ""
echo "===== summary ====="
echo "  If section 2 shows huggingface unreachable, stop and say so -- the plan"
echo "  changes completely and weights have to be staged by hand."
echo "  Otherwise the next step is the stability check in docs/local-gpu-runbook.md."
