# LLaDA grid — result, 2026-08-22

Second arm of the Phase 2 grid. 11 languages x 50 MGSM items x 2 arms = 1100
rows, `data/remote/g0_llada.jsonl`, md5 verified against the pod before it was
terminated. LLaDA-8B-Instruct against Qwen2.5-7B-Instruct, threshold decoding,
canvas from measured output, 3-shot in-language exemplars.

| lang | LLaDA fert | Qwen fert | parallel_factor | char-distinct-8 | max repeat | acc | cost_ratio |
|---|---|---|---|---|---|---|---|
| zh | 0.89 | 1.08 | 2.16 | 0.94 | 2.8 | 70% | 1.54 |
| en | 1.00 | 1.00 | 2.64 | 0.74 | 7.3 | 80% | 2.33 |
| es | 1.49 | 1.27 | 2.61 | 0.74 | 7.9 | 68% | 3.39 |
| fr | 1.50 | 1.32 | 2.64 | 0.77 | 6.8 | 60% | 2.91 |
| de | 1.58 | 1.33 | 2.47 | 0.82 | 5.8 | 50% | 2.09 |
| ja | 1.68 | 1.32 | 2.38 | 0.86 | 4.6 | 46% | 2.96 |
| sw | 1.79 | 1.73 | 3.44 | 0.71 | 8.7 | 8% | 1.77 |
| ru | 2.23 | 1.49 | 3.04 | 0.75 | 7.7 | 52% | 3.95 |
| th | 3.57 | 2.13 | 5.86 | 0.54 | 12.8 | 14% | 5.33 |
| bn | 4.84 | 4.19 | 7.95 | 0.32 | 17.0 | 4% | 3.89 |
| te | 6.00 | 6.25 | 9.39 | 0.24 | 19.1 | 6% | 3.45 |

## 1. The frozen discriminating test passed (prereg Revision 5)

Both halves of the conjunction, on predictions registered at 11:40 before the
first LLaDA row existed at 14:54:

| | | frozen threshold |
|---|---|---|
| R^2 on own tokeniser | 0.479 | |
| R^2 on Qwen's tokeniser | 0.219 | |
| gap | **+0.260** | >= 0.15 |
| residual at `ru` | **+0.369** | predicted positive |
| residual at `th` | **+0.558** | predicted positive |
| residual at `zh` | **-0.479** | predicted negative |

**Decision: T_SUPPORTED.** The penalty is a property of the *tokeniser*, not of
the language being hard. The three languages the prediction named in advance
-- chosen because the two tokenisers disagree most there -- are the three
extremes of the residual table and all three move the predicted way.

This is the one causal claim in the project, and it is the one thing here that
was risky before it was run. Note the honest caveat: the two fertility tables
correlate +0.94 in log space, so the test had little power, and the margin
arrived anyway.

## 2. The primary slope still PARKs

`k = 0.425 +- 0.148`, `R^2 = 0.479` against LLaDA's own fertility. `SE` passes
(0.148 <= 0.15); `R^2` misses (0.479 < 0.50). The frozen verdict is **PARK**,
and it stays PARK. The fertility scaling law is not this paper's result.

## 3. The degeneracy finding replicated, harder

| | corr(parallel_factor, char-distinct-8) | corr(parallel_factor, max repeat) |
|---|---|---|
| Dream-7B, Qwen tokeniser | -0.86 | +0.90 |
| **LLaDA-8B, own tokeniser** | **-0.98** | **+0.97** |

Different architecture, different tokeniser, documented on Dream two hours
before LLaDA's first row arrived. The illustration is blunt: LLaDA's *fastest*
language is Telugu at **9.39 tokens per step**, with **6% accuracy** and text
that is 76% repeated shingles. Bengali is second fastest at 7.95, with 4%
accuracy.

## 4. Where 1 and 3 collide, and what we cannot say

`bn` (3.89) and `te` (3.45) have a **lower** cost ratio than `th` (5.33) while
carrying 1.4-1.7x its fertility. Degeneracy inflates their parallel_factor
enough to make the two worst languages look cheap -- so the artifact acts
*against* the tokenisation effect, and mechanism T shows up despite it.

The obvious follow-up does not work. Refitting without the degenerate languages
moves the point estimate the predicted way but destroys the precision:

| | n | k | SE | R^2 |
|---|---|---|---|---|
| all languages | 11 | +0.425 | 0.148 | 0.479 |
| non-degenerate only | 8 | +0.650 | 0.353 | 0.361 |

Dropping the three highest-fertility points from a log-log fit removes its
leverage, so this comparison cannot decide anything and is **not** evidence.
Reported here as exploratory and inconclusive, which is what it is.

## 5. Standing

- The causal claim (T) is pre-registered and passed.
- The measurement claim (degeneracy inflates the reported speedup) replicated
  across two architectures and two tokenisers.
- The fertility scaling law is refuted as a headline: PARK on both arms.
- Nothing here is a *method*. Phase 3 (`docs/prereg-g0.md` Revision 6,
  `docs/phase3-runbook.md`) asks whether the diagnosis yields one, and can
  still kill the artifact reading outright.

Related: `docs/degeneracy-finding-2026-08-22.md`, `docs/g0-result-2026-08-22.md`.
