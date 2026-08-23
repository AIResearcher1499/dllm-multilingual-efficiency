# Preflight result — the fixes worked, and uncovered a fatal confound

`dllmfert g0 --langs en th te --n 3 --modes threshold --base-canvas 512
--dllm-models Dream --keep-text`, A100 80GB, CA-MTL-3. 18 rows.
Raw: `data/remote/preflight.jsonl`.

## What the three repairs bought

| | Before | After |
|---|---|---|
| Dream, English | `"muff muff muff ..."`, 0/3 correct | coherent step-by-step reasoning, **2/3 correct** |
| Dream, Thai | 1s of repeated digits | Thai reasoning, **2/3 correct** |
| `parallel_factor` | 31.38 identically, on zero content | 1.08–7.09, varying per item |
| Early stop | absent | fires on 8 of 9 rows |
| K2 | exponent 0.44, read as a kill | exponent **0.039**, the predicted healthy result |

The logit-shift detector agreed with what Dream's own `generation_utils.py:412`
does, reached independently: read the alignment off the model, and the model
starts doing arithmetic.

## The confound: the two arms answer in different languages

Script composition of the actual output:

| arm | input | output scripts | |
|---|---|---|---|
| Dream | en | Latin 100% | ok |
| Dream | th | **Thai 61%**, Latin 38% | ok |
| Dream | te | **Telugu 100%** | ok |
| Qwen | en | Latin 100% | ok |
| Qwen | th | **CJK 62%**, Latin 35%, Kana 2% | **answers Thai in Chinese** |
| Qwen | te | **Latin 100%** | **answers Telugu in English** |

Mean output length, relative to English:

| arm | en | th (fertility 2.13) | te (fertility 6.25) |
|---|---|---|---|
| Dream | 217 | 198 (x0.91) | 2221 (**x10.2**) |
| Qwen | 298 | 222 (x0.74) | 301 (**x1.01**) |

**Qwen's output length is flat across languages because it never leaves
English or Chinese.** Dream answers in-language and pays the token cost; the
autoregressive baseline sidesteps it entirely.

That breaks the comparison at its root. `cost_ratio` would rise with fertility
because one arm is doing a harder job, not because diffusion scales worse. The
paradigm effect and the response-language effect are perfectly confounded, and
the design has no way to separate them.

Dream's Telugu is its own problem: it answers in Telugu but **restates the
question** rather than solving it, running to 3219 and 3232 tokens, one row
never terminating, 0/3 correct. That is K3 firing at the top of the fertility
axis.

## Why this survived three preflights

Nothing in the row schema encodes what language the output is in. Accuracy does
not catch it — Qwen scored 2/3 on Telugu **by answering in English**. Only
`--keep-text` plus a script histogram makes it visible, and both were added
after the previous run reported "0 errors" over unusable data.

## Where that leaves the design

Not fixable by tuning. Three ways forward:

1. **Force the output language** in the instruction, verify with a script
   check, and drop items that fail it. Keeps MGSM and the reasoning workload,
   but forces Qwen into a language it evidently avoids, so quality becomes part
   of what is being measured.
2. **Change the task to translation** (FLORES-200). Output language is
   guaranteed by the task and its length follows fertility by construction.
   Cleanest control, but moves away from the benchmark the dLLM efficiency
   papers actually report.
3. **Report the confound itself.** Multilingual efficiency comparisons between
   model families silently compare different workloads. True, useful, and a
   smaller paper than the one being attempted.

Whatever is chosen, output-script detection belongs in the pipeline. Its
absence is the only reason this reached a fourth run.
