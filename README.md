# dllm-fertility — is the diffusion-LLM speed-up an English number?

Every headline diffusion-LLM speed-up is measured in English. Tokenizer
fertility varies 6x across scripts. This repository measures what that does to
the diffusion-vs-autoregressive trade-off, and what the standard speed-up
metric is actually counting when it leaves English.

**4,900 generations · 11 languages · 3 models · pre-registered throughout.**

## What is here

Every quantitative claim was registered in `docs/prereg-g0.md` **before** the
data that tests it existed, with a kill condition attached. Several of those
kill conditions fired, including on this project's own headline hypothesis, and
the results below report them.

### Held up

- **The throughput a published accelerator reports is anti-correlated with the
  quality of what it produces.** Running Fast-dLLM's own code unmodified across
  11 languages, its reported tokens/second correlates **-0.66** with output
  distinctness on LLaDA-8B and **-0.76** on Dream-7B, and the relationship
  survives controlling for tokenizer fertility. In both models the language with
  the **highest** reported throughput sits in the **bottom quartile** of
  accuracy: Swahili at 46.8 tokens/second and 4% correct on LLaDA, Telugu at
  30.2 and 7% on Dream (`docs/phase5-result-2026-08-24.md`). Registered in
  advance, including the fertility control and the interval, and the write-up
  names the half that is fragile before a reviewer has to.
- **The cost penalty is a property of the tokenizer, not of the language being
  hard.** Fertility and difficulty are perfectly confounded inside one
  tokenizer, so the test uses two models whose tokenizers rank languages
  *differently* — LLaDA charges less for Chinese than English and 1.68x more
  for Thai than Qwen does. Registered prediction, R² gap +0.260 against a 0.15
  threshold, and the three languages named in advance all moved the predicted
  way (`docs/g0-llada-result-2026-08-22.md`).
- **The standard parallelism metric is a repetition detector.** Tokens
  finalised per denoising step correlates **-0.86** with output distinctness on
  Dream-7B and **-0.98** on LLaDA-8B — different architectures, different
  tokenizers, and the second measured after the first was written down
  (`docs/degeneracy-finding-2026-08-22.md`). Confirmed causally: suppressing
  self-repetition at fixed language, tokenizer and canvas lowers the *measured*
  speed-up by 33% (`docs/phase3-result-2026-08-22.md`).

### Refuted, including by us

- **The fertility scaling law**, which was this project's original hypothesis.
  `k = 0.43 ± 0.15, R² = 0.48` — PARK on both arms, reported as refuted.
- **`parallel_factor` as an early degeneracy detector.** Dropped by its own
  pre-registered test: counting repeated n-grams beats it outright
  (`docs/phase3-result-2026-08-22.md`).
- **Three attempts to show a published accelerator's accounting is inflated.**
  Fast-dLLM truncates at the stop sequence before counting tokens, so its GSM8K
  throughput is not padded (`docs/fastdllm-throughput-audit-2026-08-22.md`);
  its canvas choice is defensible, since sizing the canvas to the content costs
  seven accuracy points (`docs/phase4b-canvas-result-2026-08-23.md`); and a
  quality-matched accounting *raises* rather than lowers its speed-up, firing
  the registered kill condition (`docs/phase4-p3-result-2026-08-23.md`).

A critique that reports the checks it failed is worth more than one that
reports only the check it passed.

## Reproducing

```bash
uv sync --extra fertility          # base deps are empty on purpose: the
                                   # tokenizer stage must not constrain the
                                   # pinned versions the GPU stage needs
uv run pytest                      # 102 tests, no GPU needed
uv run dllmfert fertility          # tokenizer only, no GPU
uv run dllmfert g0 --resume        # the grid; see docs/local-gpu-runbook.md
uv run dllmfert verdict
uv run dllmfert phase3 --rows data/phase3.jsonl
```

The sampler's selection logic — which decides `parallel_factor` — is pure
Python and fully tested without a GPU. Torch is confined to `models.py`.

`docs/local-gpu-runbook.md` carries the rule that matters most for anyone
extending this: quantities here are either hardware-invariant (NFE, accuracy,
distinctness) or hardware-dependent (wall-clock, tokens per second), and mixing
the second kind across machines is the easiest way to publish a wrong number.
Every row records the box that produced it and whether anything else was on the
GPU at the time.

## Layout

| | |
|---|---|
| `docs/prereg-g0.md` | every registered prediction and threshold, in revision order |
| `docs/*-result-*.md` | one file per phase, reporting against those thresholds |
| `src/dllmfert/` | the harness; `sampler.py` and `degeneracy.py` are GPU-free |
| `data/` | per-item rows for every run, including the failures |
| `scripts/` | provisioning and cell runners |
