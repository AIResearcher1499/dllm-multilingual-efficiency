# Phase 4b — the canvas prediction, and why it failed

Frozen rule: `docs/prereg-g0.md` Revision 8. Fast-dLLM's own code, LLaDA-8B,
GSM8K 8-shot, dual cache + threshold 0.9, block 32, the same 100 items at both
canvases, `--log_samples` so the comparison is paired. Cost $2.59.

| | L=512 | L=1024 |
|---|---|---|
| NFE per item | **149.4** | **139.3** |
| `R = L / NFE` | 3.43 | 7.35 |
| accuracy (flexible-extract) | **73%** | **80%** |
| content tokens per item | 265.9 | 230.2 |

`R(512) / R(1024) = 0.466`, comfortably past the 0.60 the rule wanted. The
accuracy guard fails: 7 points, where the rule allowed 3.

## Verdict: CONFOUNDED

Per Revision 8: *"The large canvas is then buying something real and the
criticism is weaker; report it that way."*

The paired data is unambiguous about the direction. Over 100 items:

| | |
|---|---|
| correct at both canvases | 73 |
| **correct only at 512** | **0** |
| correct only at 1024 | 7 |
| wrong at both | 20 |

McNemar two-sided p = 0.016. Shrinking the canvas did not rescue a single item
and broke seven.

## Both of our predictions were wrong

**Prediction 1: `NFE_acc` is roughly flat in the canvas.** It is not. NFE is
*higher* at the smaller canvas -- 149.4 against 139.3. Halving the canvas cost
more forward passes, not fewer.

**Prediction 2: the canvas is padding you pay for.** It is not only that. In a
masked diffusion model the canvas is *part of the input*: a 512-mask prompt and
a 1024-mask prompt are different inputs, and the model conditions on them. The
canvas is a conditioning signal, not a compute budget with the answer held
fixed. This is obvious in hindsight and we did not account for it.

The seven broken items show the mechanism. At the small canvas they got
**longer and more repetitive**:

| doc | chars @512 | chars @1024 | distinct @512 | distinct @1024 | max repeat @512 |
|---|---|---|---|---|---|
| 15 | 1790 | 884 | 0.542 | 0.756 | 13 |
| 20 | 1794 | 1330 | 0.442 | 0.512 | 20 |
| 25 | 1583 | 1018 | 0.660 | 0.724 | 8 |

That is degeneracy, and the *small* canvas induced it. Our own C2 finding
predicts what follows: degenerate items commit fast, which is why the small
canvas still posts a lower NFE-per-token even as it needs more NFE overall.

## What survives and what does not

**Survives (measured, Revision 7 P5, registered threshold 0.50):** at their
headline configuration the canvas is 77% padding. 7.24 positions commit per
step but only **1.69 of them are content**. Content fraction 0.233.

**Refuted (ours):** the inference that the reported speed-up is therefore an
artifact of choosing an oversized canvas. It is not available for free. Sizing
the canvas to the content costs seven accuracy points, and their choice of 1024
is defensible on the evidence we just produced.

The paper must not say the headline is inflated by canvas choice. It may say
that most of the parallelism is spent on padding, which is a description of
where the work goes, not an accusation.

## Reproducibility of our own measurement

`R(1024)` measured twice on different pods: NFE 141.5 (Gate A) and 139.3 here,
1.6% apart; accuracy 81% and 80%. Our instrument is stable.

## Standing

This is the second Phase 4 hypothesis to die against Fast-dLLM's actual
behaviour -- the padding-in-the-denominator reading died by reading their code
at $0, this one by running it at $2.59. Both died in the same direction: their
headline is more defensible than we assumed. That pattern is itself information
about whether our critique reaches the published literature, and it should be
weighed before spending the remaining balance on P3.
