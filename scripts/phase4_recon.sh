#!/usr/bin/env bash
# Phase 4 stage 0: find out how Fast-dLLM actually works before paying to run
# it. Read docs/prereg-g0.md Revision 7 first.
#
# This stage answers questions that decide the rest of the phase and that no
# amount of reading the README from a laptop could settle:
#   - does their code install and import at all on this box
#   - which flags separate KV cache from parallel decoding
#   - what the tokens-per-second denominator is (P5 deliverable 1)
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
REPO=/root/Fast-dLLM

echo "===== 1. clone ====="
[ -d "$REPO" ] || git clone --depth 1 https://github.com/NVlabs/Fast-dLLM.git "$REPO"
cd "$REPO" && git log --oneline -1 && echo "tree:" && ls

echo ""
echo "===== 2. v1 layout ====="
ls -R v1 2>/dev/null | head -60

echo ""
echo "===== 3. the LLaDA eval guide (the file the README defers to) ====="
for f in v1/llada/eval.md v1/README.md v1/llada/README.md; do
  [ -f "$f" ] && { echo "--- $f ---"; head -120 "$f"; }
done

echo ""
echo "===== 4. every generate/eval flag, straight from argparse ====="
grep -rn "add_argument" v1/llada/*.py 2>/dev/null | sed 's/^ *//' | head -60

echo ""
echo "===== 5. P5 deliverable 1: what does throughput divide by? ====="
echo "--- sites that compute a rate ---"
grep -rn "tokens_per_sec\|tok/s\|throughput\|token_per_second\|/ *elapsed\|/ *(time\|time.time()" \
     v1 --include=*.py | head -40
echo ""
echo "--- how generated length is counted near those sites ---"
grep -rn "gen_length\|num_tokens\|\.numel()\|eos\|EOS\|126081\|mask_id" \
     v1/llada/generate.py 2>/dev/null | head -40

echo ""
echo "===== 6. requirements ====="
cat v1/requirements.txt 2>/dev/null | head -30
echo ""
echo "installed now:"
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)" 2>&1

echo ""
echo "===== recon complete ====="
