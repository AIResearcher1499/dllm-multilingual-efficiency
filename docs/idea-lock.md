# Idea lock — dLLM efficiency as a function of script

- **Locked:** 2026-08-21. Successor to a predecessor project, parked the same day.
- **Status:** G0 not run. The claim is **not** locked until stage 1 shows a
  real fertility spread and stage 2 shows the effect is large.
- **Full literature record:**
  `../literature_review/notes/topics/idea4-dllm-fertility-2026-08-21.md`
  (two novelty passes, ten searches including Chinese, five full-text reads).

## One sentence

The diffusion-vs-autoregressive efficiency trade-off is a function of
**script**, because tokenizer fertility changes both the canvas a dLLM must
pay for and the intra-word redundancy its confidence-based parallel decoding
feeds on — and the field's headline speedups were all measured in English.

## Why this cell is open

`2510.18480` went hunting for confounds in dLLM efficiency evaluation, named
three (evaluation scope, infrastructure, metric inconsistency), and **language
was not among them**. ParallelBench `2510.04767` and Apple `2510.04146` do not
mention tokenization or non-English anywhere. Ten searches, one of them in
Chinese, found no cross-lingual dLLM efficiency measurement.

It is open because the dLLM crowd is English-centric and the multilingual
tokenizer crowd does not work on diffusion. A cross, not an oversight.

## Two mechanisms, opposite signs

| | Mechanism | Effect |
|---|---|---|
| **A** | higher fertility → larger canvas → more compute per step, and `S ∝ L` | hurts dLLM, superlinearly |
| **B** | higher fertility → words split into many pieces → later pieces near-deterministic → higher confidence → more positions finalised per step | helps dLLM |

Speculative decoding resolves this one way (`2605.30580`: non-English destroys
acceptance) but by a mechanism that **does not exist here** — there is no draft
model. dLLM parallelism comes from self-confidence across positions within one
model. That is why the sign is genuinely unknown, and why B would be a result.

## Must cite, must not re-claim

| Paper | Owns |
|---|---|
| `2605.30580`, `2510.02128` | speedup is language-dependent and can invert — **for speculative decoding** |
| `2510.04146` (Apple) | AR-vs-diffusion characterisation, `K=Lg`, context-length bottleneck, English |
| `2510.18480` | the meta-critique of dLLM efficiency evaluation |
| `2510.04767` ParallelBench | parallel-decoding trade-offs, English |
| Fast-dLLM `2505.22618` | chunked KV cache + confidence-aware parallel decoding, 27.6× |
| `2606.17999` VoidPadding | padding in the canvas |
| `2601.15165` Flexibility Trap | arbitrary order harms reasoning (ICML 2026 Outstanding) |
| multilingual fertility literature | fertility itself and its cost to AR models |

## Lessons carried over from the parked predecessor

1. **Do not assert a mechanism and then measure.** v3 died asserting a graded
   stop decision that did not exist. Here, check K2 (`S` vs canvas under
   acceleration) **before** building anything on mechanism A.
2. **Write the effect-size threshold before seeing data.** v1 of the probe's
   decision rule called +0.007 nats a result. `docs/prereg-g0.md` freezes 20%.
3. **Pick a quantity with dynamic range.** The stop margin was saturated at
   ~21 nats. `parallel_factor` runs from 1 to >2 tokens/step and is literally
   what the speedup is made of.
4. **Cheap gate first.** Stage 1 costs nothing and can kill the design before
   any GPU time.
