# RunPod cost — measured, 2026-08-21

## The recommendation changed twice, and here is why

**First guess: A100 80GB.** Reasoning: batch-1 inference reads all 14 GB of
weights per forward, so pick the card with the most bandwidth. That is
autoregressive physics and it is wrong here.

**Second guess: RTX A6000 48GB.** Correct on capacity — the model measured
**18.3 GB** in use, so 80 GB was 4.5x more than needed — but still built on the
bandwidth model.

**Measured answer: a diffusion step is a prefill, not a decode step.** It
processes the whole canvas, so it is compute-bound:

| canvas | Dream-7B ms/forward | LLaDA-8B ms/forward |
|---|---|---|
| 256 | 41.5 | 43.0 |
| 1632 | 181.7 | 211.0 |

`t(L) = 15.4 + 0.102 L` ms. A 6.4x canvas costs 4.4x per step. Cost therefore
scales with **fertility squared** — canvas grows with it and so does the step
count — which is the effect the paper is about, showing up in the bill.

One more correction on the way: the first pass at this table used FP32
throughput for the Ampere workstation cards and produced a nonsensical 93-hour
estimate for the A6000. Tensor-core bf16 is the right figure.

## LEAN grid — Dream threshold vs Qwen AR, 11 languages

| GPU | bf16 TFLOPS | $/h | n=100 | n=50 |
|---|---|---|---|---|
| RTX A6000 48GB | 155 | 0.33 | 30.7 h, **$7–10** | 15.7 h, $3–5 |
| A40 48GB | 150 | 0.35 | 31.6 h, $7–11 | 16.2 h, $4–6 |
| **L40S 48GB** | **362** | **0.79** | **16.1 h, $9–13** | **8.4 h, $5–7** |
| A100 PCIe 80GB | 312 | 1.19 | 17.8 h, $15–21 | 9.3 h, $8–11 |
| H100 SXM 80GB | 989 | 2.69 | 9.2 h, $19–25 | 5.0 h, $10–13 |

**Take the L40S.** More bf16 throughput than an A100 at two thirds the price,
and 48 GB against a measured 18.3 GB. The A6000 is cheaper still in dollars but
costs thirty hours; the H100 is fastest but pays a premium the budget does not
need.

## Practical notes

- **48 GB is ample.** Measured peak 18.3 GB with a 1632-token canvas. A 24 GB
  card would be tight, mostly because the scorer materialises a
  `[canvas, vocab]` float32 tensor — about 1 GB at Telugu's canvas, and
  avoidable by softmaxing only the masked positions if it ever matters.
- **Spot is safe.** `--resume` keys on the full `(lang, arm, mode, id)` cell,
  so an interrupted pod loses at most one item.
- Telugu alone is 27% of the total canvas and, because cost goes as canvas
  squared, roughly half the bill.
- API key: `~/.config/runpod/api_key`, mode 600, never committed.

## Deviations this order implies

The prereg freezes both dLLM arms and both decoding modes. Running LEAN first
is **staged**, not reduced: the full grid still runs if LEAN shows signal, and
`--resume` recomputes nothing. If the paper ships on LEAN alone, that is a
deviation and belongs in the results doc.

`naive` mode is analytically `parallel_factor = 1` by construction, so
measuring it 4,400 times buys little; ~10 items per language confirms the
implementation matches theory. Also a deviation, also recorded.
