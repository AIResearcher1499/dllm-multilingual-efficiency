#!/usr/bin/env bash
# Run one cell twice, alone on the box, and report the drift. Do this before
# trusting any wall-clock number from a machine we have not measured on before.
#
#   bash scripts/stability_check.sh
#
# Override the cell with CMD. Any {OUT} in it is replaced by a per-repeat path,
# because dllmfert refuses to write over an existing file and appending both
# repeats into one would silently average them.
#
# Precedent for what healthy looks like: R(1024) measured on two different A100
# pods came out 141.5 and 139.3 NFE, 1.6% apart, accuracy 81% and 80%.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-logs/stability-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"

CMD="${CMD:-PYTHONUNBUFFERED=1 uv run dllmfert g0 --langs en --n 20 --modes threshold \
--dllm-models GSAI-ML/LLaDA-8B-Instruct --ar-model \"\" --keep-text --out {OUT}}"

for i in 1 2; do
  OUT="$OUTDIR/run$i.jsonl"
  echo "=== repeat $i -> $OUT ==="
  CUDA_VISIBLE_DEVICES="$GPU" bash -c "${CMD//\{OUT\}/$OUT}" > "$OUTDIR/run$i.log" 2>&1
  echo "  exit=$? $(wc -l < "$OUT" 2>/dev/null || echo 0) rows"
done

uv run python - "$OUTDIR" <<'PY'
import json, pathlib, sys, re
d = pathlib.Path(sys.argv[1])

def from_jsonl(p):
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    if not rows:
        return None
    n = len(rows)
    g = lambda k: sum(r[k] for r in rows if r.get(k) is not None)
    return {"rows": float(n), "nfe": g("nfe"), "wall_clock": g("wall_clock"),
            "tokens": g("tokens_generated"),
            "parallel_factor": g("parallel_factor") / n,
            "accuracy": sum(1.0 for r in rows if r.get("acc")) / n}

def from_log(p):
    # Fast-dLLM's own eval prints totals instead of writing rows.
    s = p.read_text(errors="ignore") if p.exists() else ""
    g = lambda pat: (float(m.group(1)) if (m := re.search(pat, s)) else None)
    out = {"nfe": g(r"Total NFE is (\d+)"), "tokens": g(r"Total number of tokens generated: (\d+)"),
           "wall_clock": g(r"Total time taken: ([\d.]+)")}
    return out if any(v is not None for v in out.values()) else None

a = from_jsonl(d / "run1.jsonl") or from_log(d / "run1.log")
b = from_jsonl(d / "run2.jsonl") or from_log(d / "run2.log")
if not a or not b:
    print("\nUNUSABLE: one of the repeats produced nothing. Read the logs in", d)
    raise SystemExit(1)

# NFE is deterministic given identical logits, so drift there means the kernels
# are not reproducing and every paired comparison on this box is suspect.
# Timings are allowed real variance.
LIMITS = {"nfe": 0.5, "rows": 0.0, "tokens": 0.5, "parallel_factor": 0.5,
          "accuracy": 0.0, "wall_clock": 5.0}
print(f"\n{'quantity':<18}{'run1':>14}{'run2':>14}{'drift':>9}{'limit':>8}")
bad = []
for k in a:
    if a.get(k) is None or b.get(k) is None:
        continue
    lim = LIMITS.get(k, 3.0)
    drift = abs(a[k] - b[k]) / a[k] * 100 if a[k] else (0.0 if a[k] == b[k] else 100.0)
    flag = "" if drift <= lim else "  <-- over"
    print(f"{k:<18}{a[k]:>14.3f}{b[k]:>14.3f}{drift:>8.2f}%{lim:>7.1f}%{flag}")
    if drift > lim:
        bad.append(f"{k} drifted {drift:.2f}% (limit {lim}%)")
print()
if bad:
    print("UNSTABLE:")
    for x in bad:
        print("  " + x)
    print("\nA box that cannot reproduce itself cannot support a timing claim.")
    raise SystemExit(1)
print("STABLE -- timings from this box are usable.")
PY
