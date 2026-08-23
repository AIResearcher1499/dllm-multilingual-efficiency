# Phase 4 P3 — the registered kill condition fired

Rules frozen in `docs/prereg-g0.md` Revision 7 (decision) and Revision 9
(implementation). Fast-dLLM's own `eval_llada.py`, unmodified, LLaDA-8B-Instruct,
`mgsm_native_cot_{en,ru,th,te}`, prefix cache on in both arms, threshold 0.9 on
the accelerated arm, canvas 1024, batch 1, n=100 accelerated / n=20 baseline.
Cost $5.4.

Baseline NFE came out to exactly `n * 1024` in all four languages, confirming the
analytical claim that let the baseline run at n=20 instead of n=100.

## Verdict: K4

| lang | m good | S_raw | S_eff | S_eff >= 0.95 S_raw |
|---|---|---|---|---|
| en | 60 | 5.77 | **6.48** | yes |
| ru | 47 | 3.42 | **3.77** | yes |
| th | 6 | 4.58 | **5.27** | yes |
| te | 1 | 4.65 | **6.94** | yes |

K4 fires in all four. Restricting to correct, non-degenerate outputs does not
shrink the speed-up -- it **raises** it, in every language. The second
condition fails too: the mean gap over {th, te} is -1.489, which is not greater
than -0.531 over {en, ru}.

The direction is the opposite of what we predicted, and the reason is
mechanical: degenerate items consume *more* forward passes, so removing them
makes the accelerated arm look faster.

## The test was designed against the wrong quantity, and that is our error

`S_raw` and `S_eff` are ratios of **time**. What Fast-dLLM puts in its headline
is **tokens per second**. Degeneracy pushes those two in opposite directions:

- more repetition -> more steps -> the time ratio **falls**
- more repetition -> disproportionately more tokens counted -> tokens/second
  **rises**

Revision 7 built its rule on the first quantity while the paper's argument is
about the second. That is a design error on our side, not a property of the
data.

## What was observed, labelled post-hoc

Not registered, and not to be presented as though it were:

| lang | fertility | accuracy | char-distinct-8 | tokens counted / item | tokens/sec |
|---|---|---|---|---|---|
| en | 1.00 | 73% | 0.641 | 324.3 | 24.28 |
| ru | 2.23 | 51% | 0.638 | 525.2 | 23.04 |
| th | 3.57 | 6% | 0.519 | 518.3 | 30.05 |
| te | 6.00 | 1% | 0.305 | 564.4 | 32.66 |

`corr(tokens/sec, char-distinct-8) = -0.932`, `corr(tokens/sec, accuracy) = -0.918`,
over four languages.

Russian is the informative cell. It carries high fertility (2.23) while the model
still works there (51% correct, distinctness 0.638, indistinguishable from
English), and it posts the **lowest** throughput and the lowest speed-up in the
table. That is the tokenisation cost with quality intact. Thai and Telugu then
reverse the trend -- higher fertility still, but throughput climbs -- and they
are the two cells where the output has degenerated.

Four points is not a correlation anybody should lean on, and the ordering is
what carries the information, not the coefficient.

## Standing limit, honoured

Revision 9 states: *if P3 returns K4 or UNDECIDED, the attack on Fast-dLLM stops
there and the paper is written around first-to-measure-outside-English. No
fourth angle.* K4 fired. **Phase 4 is closed.**

The tokens-per-second observation is a candidate for its own pre-registration
and an independent replication -- 11 languages, a second model, larger n -- on
the newly available hardware. That is new data under a new rule, not a rescue of
this one.

## Also recorded

- Fast-dLLM's GSM8K throughput accounting is sound: it truncates at the stop
  sequence before counting (`docs/fastdllm-throughput-audit-2026-08-22.md`).
- Their canvas choice is defensible: sizing it to the content costs seven
  accuracy points (`docs/phase4b-canvas-result-2026-08-23.md`).
- Three Phase 4 hypotheses, three failures, all in the same direction. The
  paper should say so plainly; a critique that reports the checks it failed is
  more credible than one that reports only the check it passed.
