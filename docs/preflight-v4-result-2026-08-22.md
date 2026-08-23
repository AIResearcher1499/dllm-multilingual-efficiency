# Preflight v4 — the gate passed, and the effect is visible

A100 80GB, CA-MTL-3, cache volume reused (29 GB, no re-download). 3 languages,
n=4, threshold decoding, three in-language few-shot exemplars. 24 rows.

## Language confusion: fixed, completely

**24/24 rows pass**, both arms, all three languages. `response_lang` matches
the input everywhere. Qwen now answers Telugu in Telugu and Thai in Thai; two
runs ago it answered them in English and Chinese.

Output length now tracks fertility instead of ignoring it:

| arm | en | th (fert 2.13) | te (fert 6.25) |
|---|---|---|---|
| Qwen | 219 | 254 (x1.16) | 662 (**x3.02**, was x1.01) |
| Dream | 82 | 160 (x1.95) | 994 (x12.09) |

Few-shot was the mitigation `2406.20052` reports, and MGSM already shipped the
exemplars. The fix cost nothing.

## The result

| lang | fertility | dLLM s | AR s | **cost_ratio** | parallel_factor |
|---|---|---|---|---|---|
| en | 1.00 | 16.9 | 22.7 | **0.75** | 1.79 |
| th | 2.13 | 80.5 | 25.6 | **3.15** | 1.57 |
| te | 6.25 | 459.8 | 67.0 | **6.86** | 3.31 |

**Diffusion is faster than autoregressive in English and 6.9x slower in
Telugu.** Spread 9.2x, monotone, R² = 0.93, k = 1.18 ± 0.33.

The crossover sits between English and Thai — inside the range of languages
people actually deploy in, not out at the tail.

**Both mechanisms are visible.** `parallel_factor` is highest for Telugu
(3.31 against English's 1.79), so intra-word redundancy does buy back some
parallelism — mechanism B is real. It is simply dominated: the canvas cost
grows faster than the parallelism recovers, which is what k ≈ 1 means.

## The verdict says PARK, and it is right to

`SE(k) = 0.33` against a threshold of 0.15. Three languages and four items
cannot separate k=1 from k=0, whatever the point estimate looks like.

The sensitivity check shows why. One runaway Dream/te row — 3228 tokens, 302
seconds, canvas exhausted — carries the result:

| | with it | without it |
|---|---|---|
| te cost_ratio | 6.86 | 3.66 |
| k | 1.18 | 0.82 |

One row in twelve moves k by 0.36. That is the imprecision the gate exists to
catch, and it is exactly what n=50 across 11 languages is for.

## What to watch in Phase 2

- **Runaway generations.** One Dream/te row in four never terminated. If that
  rate holds, the Telugu mean is a mixture of two behaviours and should be
  reported as such rather than averaged.
- **Accuracy at the top of the axis.** Dream 2/4 on Telugu, Qwen 1/4. Small n,
  but the efficiency comparison needs both arms to be doing the task.
- **Few-shot is not free.** Peak memory went 18.3 GB to 23.6 GB because the
  exemplars lengthen the prompt, and a diffusion step recomputes the prompt
  every time. The prompt is itself fertility-scaled, so this cost rides on the
  axis under study. Report prompt and generation time separately.
