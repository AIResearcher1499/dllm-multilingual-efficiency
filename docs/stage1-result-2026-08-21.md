# Stage 1 result — PROCEED (2026-08-21)

`uv run dllmfert fertility --n 250`, Qwen2.5-7B-Instruct tokenizer, MGSM test
split. Raw: `data/fertility.json`. Cost: minutes, on a Mac, $0.

## Verdict: PROCEED. Spread 6.25×, gate was 1.5×.

All 11 MGSM configs present, 250 parallel items each.

| lang | fertility vs en | mean tokens |
|---|---|---|
| en | 1.00 | 61.1 |
| zh | 1.08 | 65.7 |
| es | 1.27 | 77.8 |
| fr | 1.32 | 80.5 |
| ja | 1.32 | 80.5 |
| de | 1.33 | 81.7 |
| ru | 1.49 | 91.4 |
| sw | 1.73 | 106.2 |
| th | 2.13 | 129.6 |
| bn | 4.19 | 257.8 |
| te | **6.25** | 383.9 |

## Why this axis is better than expected

**6.25× of dynamic range.** The prereg asked for 1.5×. Telugu costs six times
what English costs to say the same thing, and Bengali four. Any effect of
canvas size on diffusion decoding has room to show itself here; a null would
be informative rather than underpowered.

**Chinese is the control the design needed.** zh sits at 1.08 — a non-Latin
script that is *cheap* under this tokenizer, because Qwen was built for it.
That decouples two explanations that would otherwise be confounded:

- if the effect tracks **fertility**, zh behaves like English;
- if it tracks **script family** or **training-data share**, zh behaves like
  ja/th.

Without a cheap non-Latin language there would be no way to tell those apart,
and a reviewer would have said so. Note this before running stage 2, not after.

**ja is only 1.32.** The intuition that Japanese is expensive is wrong for this
tokenizer. The high-fertility end is te / bn / th, not CJK. Plan the language
budget accordingly: the interesting comparisons are en–zh (cheap, one Latin
one not), ru–sw–th (middle), and bn–te (extreme).

## What this does and does not establish

It establishes the **x-axis**, nothing else. Whether `parallel_factor` moves
along it is stage 2, and the prereg's 20% magnitude gate is untouched.

The fertility numbers themselves are not a contribution — fertility across
languages is well documented for AR models. What is new is regressing a
*diffusion decoding* quantity against it.
