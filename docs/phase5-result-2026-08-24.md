# Phase 5 — the throughput claim: SUPPORTED

Frozen in `docs/prereg-g0.md` Revision 10 before any of this data existed.
Fast-dLLM's own code, unmodified, `mgsm_native_cot_*` across all 11 MGSM
languages, two models, n=100, canvas 1024, block 32, prefix cache on, threshold
0.9, batch 1, `MODE=serial`. 22 cells. Cost about $19 across three pods.

## Verdict

Both models independently clear all three gates.

| | corr(tok/s, char-distinct-8) | bootstrap 95% CI | partial \| log fertility |
|---|---|---|---|
| gate | <= -0.50 | upper < -0.20 | <= -0.35 |
| **LLaDA-8B** | **-0.662** | [-0.951, **-0.440**] | **-0.658** |
| **Dream-7B** | **-0.760** | [-0.915, **-0.424**] | **-0.718** |

**SUPPORTED.**

The descriptive claim, reported separately as Revision 10 requires and without a
threshold: in **both** models the language with the highest reported throughput
sits in the bottom quartile of accuracy.

- LLaDA's fastest language is **Swahili at 46.79 tokens/second, 4% correct**.
- Dream's fastest is **Telugu at 30.17 tokens/second, 7% correct**.

## The tables

| LLaDA | fert | acc | distinct | tok/s | | Dream | fert | acc | distinct | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| zh | 0.89 | 53% | 0.856 | 17.25 | | en | 1.00 | 82% | 0.823 | 9.44 |
| en | 1.00 | 73% | 0.641 | 24.19 | | zh | 1.08 | 64% | 0.947 | 10.44 |
| es | 1.49 | 59% | 0.634 | 22.21 | | es | 1.27 | 70% | 0.778 | 12.73 |
| fr | 1.50 | 51% | 0.607 | 22.26 | | fr | 1.32 | 55% | 0.664 | 15.89 |
| de | 1.58 | 54% | 0.596 | 21.59 | | ja | 1.32 | 52% | 0.855 | 15.24 |
| ja | 1.68 | 53% | 0.746 | 22.04 | | de | 1.33 | 60% | 0.652 | 17.34 |
| **sw** | 1.79 | **4%** | 0.411 | **46.79** | | ru | 1.49 | 65% | 0.682 | 16.26 |
| ru | 2.23 | 51% | 0.638 | 22.59 | | sw | 1.73 | 15% | 0.890 | 14.86 |
| th | 3.57 | 6% | 0.519 | 29.40 | | th | 2.13 | 28% | 0.665 | 21.07 |
| bn | 4.84 | 2% | 0.259 | 27.11 | | bn | 4.19 | 14% | 0.710 | 26.29 |
| te | 6.00 | 1% | 0.305 | 32.07 | | **te** | 6.25 | **7%** | 0.515 | **30.17** |

## What is fragile, stated before anyone else has to find it

The **raw correlation is robust in both models.** Dropping any single language,
the worst case is -0.590 (LLaDA, without `zh`) and -0.604 (Dream, without `te`).
Both still clear -0.50.

The **partial correlation is robust in Dream and fragile in LLaDA.**

| | leave-one-out worst partial | |
|---|---|---|
| LLaDA | drop `sw` -> **-0.316** | **fails the -0.35 gate** |
| Dream | drop `de` -> -0.629 | passes |

LLaDA's fertility control therefore hinges on Swahili, the single most extreme
cell in the study: fertility only 1.79, yet the highest throughput in either
table and 4% accuracy. Swahili is precisely the point that breaks the
fertility-throughput alignment, so removing it lets fertility explain more and
the partial shrinks. The bootstrap interval on LLaDA's partial spans
[-0.915, +0.419] and includes zero; Dream's is [-0.955, -0.230] and does not.

Revision 10 required a bootstrap interval on the **raw** correlation only, and
both models meet it. The partial's intervals are reported because 11 points with
a collinear control cannot support a precise partial, not because the rule asked.

## A caution about Dream's control

`corr(tok/s, log fertility) = +0.963` on Dream. Fertility explains nearly all of
its throughput variation, so the partial is computed from a thin residual and its
denominator, `sqrt(1 - 0.963^2) = 0.27`, amplifies whatever is left. That the
result survives every leave-one-out anyway is the reason to believe it, not the
coefficient itself.

## Conventions that differ between the arms, and why they do not merge

LLaDA counts tokens **after** truncating at the stop sequence
(`eval_llada.py:357-366`); Dream counts the whole generation window minus EOS
**before** truncation (`eval.py:325`). Dream reports no NFE. Absolute throughput
is therefore not comparable across arms -- LLaDA's 17-47 against Dream's 9-30 --
and the frozen rule correlates **within** each model for exactly this reason.
Pooling the two arms into one correlation would manufacture a result out of the
convention difference.

## Limitations

- **Zero-shot.** `mgsm_native_cot` supplies no exemplars, while Fast-dLLM's own
  GSM8K numbers are 5- and 8-shot. Few-shot is not available at this canvas:
  LLaDA's context is 4096 and exemplars for a high-fertility language plus a
  1024-token canvas do not fit. A reviewer is entitled to note that zero-shot
  inflates the quality spread this claim feeds on.
- **Two models, both instruction-tuned, both about 8B.**
- **n=100 per cell, 11 cells per model.** Eleven points is enough for a
  correlation with an interval and not enough for a precise partial.

## Provenance

Collected across three pods after two launcher failures terminated pods
mid-experiment; the `llada_en` cell was measured on all three and returned NFE
11787 every time, with throughput 24.28 / 24.00 / 24.19. The instrument
reproduces exactly on forward-pass count and to about 1% on wall-clock.
