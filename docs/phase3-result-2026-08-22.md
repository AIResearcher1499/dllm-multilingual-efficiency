# Phase 3 — result, 2026-08-22

500 rows, English only, 5 repetition-penalty settings x 50 MGSM items x 2 arms.
`data/remote/phase3.jsonl`, md5 verified against the pod before terminating.
Cost $1.99. Verdicts produced by `dllmfert phase3` against the rules frozen in
`docs/prereg-g0.md` Revision 6, before any of this code existed.

## P1 — the causal test: SUPPORTED on Dream, UNDECIDED on LLaDA

| | Dream-7B | LLaDA-8B | frozen threshold |
|---|---|---|---|
| rho(char-distinct, penalty) | **+1.00** | +1.00 | >= +0.80 |
| rho(parallel_factor, penalty) | **-1.00** | -0.70 | <= -0.80 |
| rho controlling content length | -0.53 | -0.45 | <= -0.30 |
| parallel_factor | 2.02 -> 1.35 (**-33.2%**) | 2.64 -> 1.57 (-40.5%) | |
| accuracy | 80% -> 56% | 80% -> 44% | |

**K3 did not fire.** Suppressing self-repetition, at fixed language, tokeniser,
canvas, threshold and items, lowers the *measured* parallel_factor. The artifact
reading of Phase 2 survives its own kill condition.

Dream's response is perfectly monotone across all five settings (2.02, 2.00,
1.93, 1.78, 1.35) and passes without argument.

### The rank statistic was the wrong choice, and that is our error

LLaDA reads UNDECIDED on a **larger** effect than the arm that passed. Its
cells are 2.64, 2.62, 2.65, 2.52, 1.57: the first three are separated by
`z <= 0.24` -- indistinguishable -- while the endpoints differ by `z = 8.76`.
Spearman over five points weights a noise-level inversion among near-ties
exactly as heavily as the real collapse, so the frozen rule cannot see a 40%
effect.

That is a defect in how we froze the rule, not in the data. **The rule is not
being changed after seeing the data**, and UNDECIDED is reported as the primary
verdict for this arm. The magnitudes are reported beside it so the reader can
see what the statistic missed. A future pre-registration on a five-point sweep
should gate on an effect size with a confidence interval, not on a rank
correlation.

### Two honest complications

1. **The intervention is not clean.** Accuracy falls with the penalty (80% ->
   56% / 44%). The penalty damages the model as well as its repetition.
2. **But the damage runs the wrong way to explain the result.** Under the
   degeneracy reading, *worse* output produces a *higher* parallel_factor. Here
   output got worse and parallel_factor fell by a third. The fall cannot be
   attributed to output quality dropping; it tracks the repetition that was
   removed. Content length is flat throughout (Dream 117 -> 116, LLaDA 233 ->
   236), and the length-controlled correlation clears its bar on both arms.

## P2 — the detector (M1): DROPPED

| budget | Dream AUC(pf) | Dream AUC(n-gram) | LLaDA AUC(pf) | LLaDA AUC(n-gram) |
|---|---|---|---|---|
| 10% of steps | 0.456 | 0.500 | 0.489 | 0.510 |
| 25% of steps | 0.516 | 0.500 | 0.547 | **0.680** |
| 50% of steps | 0.712 | 0.530 | 0.592 | **0.757** |

Verdicts: **M1_DROPPED** (LLaDA), **UNDECIDED** (Dream).

At the early budgets that matter, `parallel_factor` is close to chance, and the
trivial competing signal -- counting repeated n-grams in the text decoded so far
-- beats it outright on LLaDA. `parallel_factor` becomes informative on Dream
only at half the step budget (AUC 0.712), and a detector that fires after half
the compute is already spent saves little. That is exactly why the frozen rule
required the early budgets.

Per Revision 6: the paper ships the measurement and makes **no detector claim**.
This was registered in advance as an acceptable outcome, and it cost $2 to
establish before the method was built rather than after a reviewer asked why it
was never compared against counting n-grams.

## Standing after Phase 3

- **Causal claim (artifact reading):** survives. Supported on one arm outright;
  larger but statistically unreadable on the other for a reason we caused.
- **Tokenisation claim (T):** unchanged, pre-registered, supported
  (`docs/g0-llada-result-2026-08-22.md`).
- **Fertility scaling law:** refuted as a headline. PARK on both arms.
- **Method M1:** dropped by its own pre-registered test.
- **Remaining method candidate:** M3, per-language threshold calibration. Not
  yet run, and not yet promised.
