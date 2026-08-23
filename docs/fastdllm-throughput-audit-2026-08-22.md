# What Fast-dLLM's reported throughput actually counts

Prereg Revision 7, P5 deliverable 1. Answered by reading their published code,
at **zero GPU cost**. Source: `NVlabs/Fast-dLLM` at `a9b81e4`,
`v1/llada/eval_llada.py`.

## The measurement

```python
# eval_llada.py:393-397
end_time = time.time()
if self.show_speed:
    print(f"Total number of tokens generated: {num_tokens}")
    print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
```

`num_tokens` is accumulated on **two different paths**:

| benchmark | path | what is counted |
|---|---|---|
| HumanEval | `eval_llada.py:351-353` | the whole generation window minus token `126081`; **no stop-sequence truncation** |
| **GSM8K** (the headline) | `eval_llada.py:357-366` | decoded, **truncated at the first stop sequence** from the harness `until` list, re-tokenised, then counted |

## Our padding hypothesis was wrong, and it was wrong about the headline

We predicted that at generation length 1024 with ~120-token answers, the ~900
remaining positions would enter the tokens-per-second denominator, and that this
would be visible in the 27.6x headline.

**It does not.** For GSM8K, Fast-dLLM truncates at the stop sequence before
counting. The headline benchmark is measured honestly on this axis, and any
claim that their throughput is inflated by counting padding would be false.
This was checked before spending anything, which is the point of doing it first.

## What survives, and it is sharper

The truncation is conditional on a stop sequence *appearing*. Line 359:

```python
for stop_seq in stop_tokens:
    if stop_seq in generated_answer_i:
        generated_answer_i = generated_answer_i.split(stop_seq)[0]
```

If no stop sequence appears, nothing is truncated and the full window counts.
A generation that fails to terminate is exactly a degenerate one -- and in our
own grid, non-termination and repetition are the same event: LLaDA filled the
canvas on 50% of Thai items and 76% of Telugu shingles were repeats.

So the claim is not "they count padding". It is:

> **When the model fails to terminate, every repeated token it emits counts
> toward tokens per second.** Degeneracy raises the numerator of the throughput
> measure, and it raises it most in the languages where the model is worst.

This is a narrower claim than the one we set out to test, it is grounded in
their code rather than in our harness, and it is measurable: the quantity to
report per language is the **non-termination rate** under their own
configuration, not the padding fraction.

## Consequence for Phase 4

P5's registered threshold ("material if mean content fraction in English is
below 0.50") was written against the padding hypothesis and is now measuring
the wrong thing. It is **not** being retargeted after the fact. It will be
reported as registered -- content fraction, which we expect to be high and
therefore immaterial -- and the non-termination rate will be reported
**separately and explicitly labelled as a post-hoc measurement**, because that
is what it is.

P3 is unaffected: it compares `S_raw` against `S_eff` under their own ablation
switches, which the recon confirmed are exactly the five documented commands in
`v1/llada/eval.md` (baseline / prefix cache / parallel / cache+parallel /
dual cache+parallel). Holding the cache on and toggling `threshold` isolates the
parallel-decoding component, as Revision 7 requires.
