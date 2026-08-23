# Running on the 2x A6000 box

Replaces `scripts/runpod_run.sh`, which mixed provisioning with running. On a
box we own there is nothing to provision and nothing to terminate, so the
runner is `scripts/run_local.sh` and it does one thing.

## The rule that matters most

**Never compare a number measured on one machine with a number measured on
another.** Every quantity here is one of two kinds:

| hardware-invariant | hardware-dependent |
|---|---|
| NFE, accuracy, `parallel_factor` | `wall_clock` |
| char-distinct-8, max repeat, fertility | tokens per second |
| anything computed from the generated text | anything derived from either |

The A6000 has 768 GB/s of memory bandwidth against the A100's 1935. At batch 1
a forward pass is dominated by streaming 16 GB of LLaDA weights, so **one A6000
is roughly 2.5x slower than the A100 every timing in this repo was measured on**.
Absolute timings from the RunPod phases do not transfer. Ratios mostly do, but
not exactly, because the two arms differ in sequence length and cache access.

There is a subtler consequence. A100 is SM80 and A6000 is SM86: same
architecture family, different tile heuristics in cuBLAS, so logits can differ
in the last bits. Under confidence-threshold decoding a position sitting near
0.9 can flip between committing and not, and the generation diverges from
there. **Generations are not guaranteed identical across the two machines**, so
paired item-level comparisons must stay within one box.

Every row now carries `hw` and `timing_trustworthy` from
`dllmfert.provenance`, so a merged file can still answer "which box, and was
anything else running".

## Two modes, and picking the wrong one silently corrupts the result

```bash
CELLS=cells.txt MODE=parallel GPUS=0,1 bash scripts/run_local.sh
```

- **`MODE=parallel`** -- one cell per GPU. Correct for anything scored on NFE,
  accuracy or text. A neighbour on the box cannot change a forward-pass count.
- **`MODE=serial`** -- one cell at a time on the whole box. **Required** for
  anything scored on `wall_clock` or tokens/second. Two cells contend for CPU
  and PCIe and the contention lands inside the number being measured.

`timing_trustworthy` is false on any row produced while another compute process
was on the box, so a parallel run later analysed for timing is detectable. That
is a backstop, not a substitute for choosing the mode correctly.

## Do this first, before trusting any timing

```bash
CMD='PYTHONUNBUFFERED=1 uv run dllmfert g0 --langs en --n 20 --modes threshold \
     --dllm-models GSAI-ML/LLaDA-8B-Instruct --ar-model "" --keep-text \
     --out /tmp/stab.jsonl' \
bash scripts/stability_check.sh
```

Runs the same cell twice, alone, and prints the drift. NFE must be within 0.5%
-- it is deterministic given identical logits, so drift there means the kernels
are not reproducing and every paired comparison on this box is suspect. Timings
are allowed 3%.

Precedent: `R(1024)` measured on two different A100 pods came out 141.5 and
139.3 NFE, 1.6% apart, accuracy 81% and 80%. That is what a healthy instrument
looks like.

## Environment

No `/runpod-volume`; set `HF_HOME` to somewhere with room for ~50 GB of weights.
The pins are not optional and are not stylistic:

- `transformers==4.46.2`, `torch==2.5.1`, python `<3.13` for **this** project --
  Dream's bundled code raises `KeyError: 'default'` in `ROPE_INIT_FUNCTIONS`
  above that.
- Fast-dLLM needs `transformers==4.49.0` and therefore lives in its **own**
  venv. Installing its requirements into ours silently breaks our harness.
- Match the torch CUDA build to the driver. The default PyPI wheel bundles the
  newest CUDA and refuses to start on an older driver; that cost one run.

## What to re-run, and why it is not just for the bigger n

Hardware consistency is its own reason. The paper should stand on one platform,
and mixing A100 timings with A6000 timings is exactly the error this document
exists to prevent. Free long-term compute makes a full re-run affordable, so
take it.

Order:

1. Stability check (above). Nothing else is trustworthy until it passes.
2. Our own grid, 11 languages, both models, at the largest n that fits. Several
   frozen verdicts -- `k` PARKing at `R^2 = 0.479` against a 0.50 gate, Gate A
   missing by 5.0 points with a standard error of 3.9 -- were **under-powered,
   not null**. Re-running once at larger `n` against the **same** thresholds is
   added power, not moved goalposts.
3. Whatever Revision 10 registers.

**Each frozen test is re-run exactly once at larger n.** A second re-run because
the first was not to our liking is p-hacking, and free compute is precisely the
condition under which that becomes tempting.
