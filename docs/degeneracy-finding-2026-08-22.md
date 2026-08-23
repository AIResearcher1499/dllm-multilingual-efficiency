# The parallel-decoding gain at high fertility is degeneracy

A $0 analysis of data already collected, prompted by LLaDA's preflight showing
Telugu output filling the canvas (1598 of 1600 tokens) with `parallel_factor`
9.2–9.6, near-identical across items, and 0/3 correct.

## The measurement

Per-language, over Dream's 550 grid rows, using character 8-shingles so the
measure works for scripts without spaces:

| lang | fert | pf | char-distinct-8 | max shingle repeat | tok | acc |
|---|---|---|---|---|---|---|
| en | 1.00 | 2.02 | 0.84 | 4.2 | 117 | 80% |
| zh | 1.08 | 2.13 | 0.97 | 1.9 | 124 | 78% |
| es | 1.27 | 2.12 | 0.79 | 5.7 | 147 | 70% |
| fr | 1.32 | 2.37 | 0.79 | 5.2 | 170 | 76% |
| ja | 1.32 | 1.50 | 0.96 | 2.4 | 126 | 64% |
| de | 1.33 | 2.10 | 0.82 | 5.5 | 160 | 72% |
| ru | 1.49 | 2.24 | 0.83 | 4.9 | 165 | 70% |
| sw | 1.73 | 2.15 | 0.79 | 4.1 | 144 | 24% |
| th | 2.13 | 1.49 | 0.79 | 4.6 | 196 | 38% |
| **bn** | 4.19 | **4.49** | **0.55** | **11.4** | **673** | 40% |
| **te** | 6.25 | **4.02** | **0.62** | **8.7** | **670** | 22% |

**corr(parallel_factor, char-distinct) = −0.86**
**corr(parallel_factor, max shingle repeat) = +0.90**

## What this kills

The Phase 2 write-up claimed mechanism B — intra-word redundancy raising
confidence and buying back parallelism — was real, on the strength of
`parallel_factor` roughly doubling at Bengali and Telugu.

It is not intra-word redundancy. Those two languages are exactly the two where
the model emits **four times as much text**, far more repetitively, and gets
22–40% of the answers right. Repeated text is trivially predictable, so a
confidence-threshold sampler commits it in bulk. **The apparent efficiency gain
is the model failing.**

LLaDA reproduces it independently: Telugu output filled 1598 of 1600 canvas
tokens across all three preflight items, `parallel_factor` 9.2–9.6, accuracy
0/3, the text restating the question rather than solving it.

## What it does to the headline

`k = 0.566 ± 0.110` was computed over all 11 languages, so it includes the
degenerate rows. Restricting to languages where the model works — accuracy
≥ 60%, low repetition — leaves en, zh, es, fr, ja, de, ru, a fertility range of
1.00–1.49, and `k = 0.97 ± 0.45, R² = 0.48`: **PARK**.

So the gated GO depends on rows the model got wrong, and the clean subset has
too little fertility range to resolve a slope. The slope claim does not
survive.

## What does survive

1. **Cross-lingual dLLM efficiency has never been measured.** Still true after
   ~15 searches including one in Chinese.
2. **In the working regime** (fertility ≤ 1.5), diffusion's advantage is
   clearest in English (0.69) and roughly parity elsewhere (0.94–1.30).
3. **Beyond fertility ≈ 1.7 both diffusion models degenerate**: output grows
   four-fold, repetition triples, accuracy falls to 22–40%.
4. **Three measurement traps, each demonstrated:**
   - *Language confusion.* Uncontrolled, Qwen answers Telugu in English and
     scores 2/3 on accuracy while doing a different task.
   - *Degeneracy inflating efficiency.* corr = +0.90 above.
   - *Harness choices.* Canvas sizing and prompt caching move the headline
     several-fold on identical data.

## Reading

The honest paper is no longer "diffusion efficiency scales with tokenizer
fertility". It is that **cross-lingual efficiency comparison of diffusion LMs
is dominated by measurement traps**, that all three traps push in the same
direction — flattering the diffusion arm at high fertility — and that in the
regime where the models actually work the effect is small.

Sixth correction in this project. Every one has lowered our own headline, which
is at least the right direction for corrections to run.
