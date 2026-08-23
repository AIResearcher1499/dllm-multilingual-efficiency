# Prereg — Phase 5

## Revision 10, 2026-08-23 — the throughput claim, registered properly this time

Phase 4 closed under its own kill condition. What came out of it sideways, and
was labelled post-hoc because it was, is this: across four languages inside
Fast-dLLM's own evaluation, the tokens-per-second they report ran at -0.93
against output distinctness and -0.92 against accuracy, with Telugu -- 1%
correct, 70% repeated shingles -- posting the **highest** throughput in the
table.

That observation is not evidence yet. It is a hypothesis, and this revision
registers it before the data that could test it exists.

### Aimed at the right quantity, which Phase 4 was not

Revision 7 built its rule on `S_raw` / `S_eff`, a ratio of **times**. The
argument is about **tokens per second**. Degeneracy pushes those two in
opposite directions -- more repetition means more steps (time ratio falls) and
disproportionately more tokens counted (throughput rises) -- so P3 tested the
opposite of what the paper claims and K4 fired on a quantity nobody was
disputing. That error is the reason this revision exists.

### Design

Fast-dLLM's own code, unmodified. `mgsm_native_cot_*`, all **11 MGSM
languages**, **two models** (LLaDA-8B-Instruct and Dream-v0-Instruct-7B),
n = 100, canvas 1024, block 32, prefix cache on, threshold 0.9. **22 cells.**

Run in **`MODE=serial`**. Throughput is the measurement; two cells sharing the
box put PCIe and CPU contention directly into it. Rows carrying
`timing_trustworthy = false` are excluded and their count is reported.

Per cell, record their reported tokens/second, char-distinct-8, accuracy, and
the tokeniser fertility of that language for that model.

### Primary quantity is distinctness, not accuracy, and that is a lesson

Accuracy saturates at the floor: LLaDA scored 1% on Telugu and 6% on Thai, so
across the high-fertility half of the table accuracy has almost no variance
left to correlate with. P3 also lost its restricted estimate entirely because
one language had a single correct item. Distinctness keeps its full range
everywhere and cannot run out of good items. Accuracy is reported alongside as
the secondary.

### The confound that must be controlled, and is not optional

Tokens per second has token count in its numerator, and a high-fertility
tokeniser emits more tokens for the same content. Fertility therefore inflates
throughput mechanically, with no degeneracy involved. Any correlation that does
not survive controlling for fertility is measuring the tokeniser, not the
failure mode, and would be a restatement of our own T result rather than a new
claim.

### Decision rule

Correlations are over 11 cells per model, and 11 points is enough for a
correlation with an interval -- unlike the five-point rank statistic frozen in
Phase 3, which could not see a 40% effect because three cells sat within noise
of each other. No rank statistics here.

- **SUPPORTED** if, in **both** models independently:
  - `corr(tokens/sec, char-distinct-8) <= -0.50`, and
  - the bootstrap 95% CI for that correlation has an upper bound `< -0.20`, and
  - the **partial** correlation controlling `log(fertility)` is `<= -0.35`.
- **REFUTED** if either model gives `corr >= -0.20`, or if either model's
  partial correlation controlling fertility is `>= -0.20`. The reported
  throughput is then not tracking degeneracy and the claim is dropped from the
  paper.
- Anything else: **UNDECIDED**, reported as such.

### The vivid claim, reported separately and without a threshold

For each model: is the language with the **highest** reported tokens/second in
the **bottom quartile** of accuracy? Reported yes or no per model, with the
numbers. This is a description of the table, not a test, and it is not allowed
to substitute for the rule above.

### Standing limits

- **Each frozen test is re-run exactly once at larger n.** Free compute is the
  condition under which re-running until something passes becomes tempting, and
  this is the sentence that forbids it.
- Nothing measured on the RunPod A100s enters this analysis. The A6000 box is
  ~2.5x slower at batch 1 and is a different SM generation; timings do not
  transfer and generations are not guaranteed identical.
- The stability check in `docs/local-gpu-runbook.md` must pass on this box
  before any timing here is believed.

---

