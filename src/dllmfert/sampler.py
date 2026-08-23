"""Instrumented masked-diffusion decoding.

The quantity this whole repo exists to measure — `parallel_factor`, tokens
finalised per denoising step — is decided entirely by *which masked positions
the sampler commits at each step*. That selection logic is therefore kept in
pure Python with no tensor dependency, so it is fully testable on a laptop;
torch appears only in the model adapter that feeds it confidences.

This is a faithful reimplementation of low-confidence / confidence-threshold
block decoding (LLaDA, Fast-dLLM class), not the official kernels. **Absolute
speedups from this sampler are not comparable to published numbers.** Only
cross-language ratios are, which is all the prereg regresses on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NAIVE = "naive"
THRESHOLD = "threshold"
MODES = (NAIVE, THRESHOLD)


def select_positions(
    confidences: dict[int, float], *, mode: str, threshold: float
) -> list[int]:
    """Masked positions to commit this step.

    `naive` commits exactly one — the `K = Lg` convention Apple `2510.04146`
    reports as standard, giving parallel_factor == 1 by construction.

    `threshold` commits every position above the confidence threshold, but
    **always at least one**: without that guarantee a block whose positions are
    all uncertain would spin forever, and the run would look like a timeout
    rather than a slow decode. That floor is also why parallel_factor cannot
    drop below 1 and why the metric's dynamic range is upward.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if not confidences:
        return []
    ranked = sorted(confidences.items(), key=lambda kv: (-kv[1], kv[0]))
    if mode == NAIVE:
        return [ranked[0][0]]
    picked = [p for p, c in ranked if c >= threshold]
    return picked or [ranked[0][0]]


def block_bounds(canvas: int, block_size: int) -> list[tuple[int, int]]:
    """Left-to-right blocks. block_size <= 0 means one block over the canvas."""
    if block_size <= 0 or block_size >= canvas:
        return [(0, canvas)]
    return [(s, min(s + block_size, canvas)) for s in range(0, canvas, block_size)]


def complete_prefix_stop(committed: dict[int, int], canvas: int,
                        stop_ids: set[int]) -> int | None:
    """Index of the first stop token that has an unbroken committed prefix.

    A stop committed at position 40 means nothing while position 3 is still
    masked — the model has not finished, it has merely guessed the end. Only a
    contiguous run ending in a stop token is a finished generation.
    """
    for i in range(canvas):
        if i not in committed:
            return None
        if committed[i] in stop_ids:
            return i
    return None


@dataclass
class DecodeTrace:
    canvas: int
    finalised_per_step: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    commit_step: list[int] = field(default_factory=list)
    """Step index at which each canvas position was committed, -1 if never.

    Recorded so a prefix can be reconstructed offline at any step without
    storing a full per-step trace: a committed token is final in masked
    diffusion decoding, so (final tokens, commit step) determines the state at
    every step exactly. This is what lets us ask whether `parallel_factor`
    signals degeneracy *earlier* than the decoded text does.
    """
    hit_step_cap: bool = False
    content_len: int = 0
    stopped: bool = False

    @property
    def nfe(self) -> int:
        """Forward evaluations. One per step, since each step recomputes the
        whole sequence — the absence of a KV cache is the point."""
        return len(self.finalised_per_step)

    @property
    def parallel_factor(self) -> float | None:
        """**Content** tokens finalised per step.

        Not total positions per step. A canvas sized for a high-fertility
        language is mostly padding when the answer is short, and padding is
        exactly what a confidence sampler commits fastest — the model is
        certain it is done. Counting it inflates the metric in direct
        proportion to canvas size, and canvas is proportional to fertility, so
        the padding would masquerade as the very effect being measured.

        Measured on a real A100 before this was fixed: Telugu scored 31.4
        tokens/step against English's 3.0, identical to two decimal places
        across three different problems, while generating zero content.
        """
        if not self.finalised_per_step:
            return None
        return self.content_len / len(self.finalised_per_step)

    @property
    def padding_positions(self) -> int:
        """Canvas paid for but not used. A first-class cost of a fixed canvas."""
        return max(0, self.canvas - self.content_len)


def decode(
    scorer,
    *,
    canvas: int,
    mode: str = THRESHOLD,
    threshold: float = 0.9,
    block_size: int = 32,
    max_steps: int | None = None,
    stop_ids: set[int] | None = None,
) -> DecodeTrace:
    """Run masked decoding, recording how many positions each step commits.

    `scorer(committed, masked_positions) -> {position: (confidence, token_id)}`
    is the injected model: `committed` maps already-decided canvas positions to
    token ids, so the scorer sees the same state the real model would.

    With `stop_ids`, decoding halts as soon as a stop token has an unbroken
    committed prefix — the generation is finished and the rest of the canvas is
    padding nobody will read. Real implementations stop there too, and
    continuing would charge the remaining canvas to `parallel_factor` as though
    it were content.
    """
    cap = max_steps if max_steps is not None else canvas + 1
    trace = DecodeTrace(canvas=canvas)
    committed: dict[int, int] = {}
    step_of: dict[int, int] = {}
    stops = set(stop_ids or ())
    for start, end in block_bounds(canvas, block_size):
        while any(p not in committed for p in range(start, end)):
            if trace.nfe >= cap:
                trace.hit_step_cap = True
                break
            step_idx = trace.nfe
            masked = [p for p in range(start, end) if p not in committed]
            scored = scorer(dict(committed), masked)
            confidences = {p: c for p, (c, _) in scored.items()}
            chosen = select_positions(confidences, mode=mode, threshold=threshold)
            for p in chosen:
                committed[p] = scored[p][1]
                step_of[p] = step_idx
            trace.finalised_per_step.append(len(chosen))
            if stops:
                at = complete_prefix_stop(committed, canvas, stops)
                if at is not None:
                    trace.content_len = at
                    trace.stopped = True
                    break
        if trace.hit_step_cap or trace.stopped:
            break
    trace.tokens = [committed.get(p, -1) for p in range(canvas)]
    trace.commit_step = [step_of.get(p, -1) for p in range(canvas)]
    if not trace.stopped:
        # No stop token ever earned a complete prefix: the model filled the
        # canvas without finishing. Everything committed counts as content,
        # which is the honest reading of a generation that never terminated.
        trace.content_len = sum(1 for p in range(canvas) if p in committed)
    return trace


def prefix_at_step(tokens: list[int], commit_step: list[int], step: int,
                   *, limit: int | None = None) -> list[int]:
    """Tokens committed at or before `step`, in canvas order.

    Uncommitted positions are skipped rather than filled: a partially decoded
    canvas really does have holes, and an online detector reading the text so
    far would see exactly this. `limit` bounds the read to the content region
    so trailing padding cannot enter the prefix.
    """
    end = len(tokens) if limit is None else min(limit, len(tokens))
    return [t for p, t in enumerate(tokens[:end])
            if 0 <= commit_step[p] <= step and t >= 0]


def trim_at(tokens: list[int], stop_ids: set[int]) -> list[int]:
    """Generated content is the canvas up to the first stop token; everything
    after it is padding the model still paid for."""
    for i, t in enumerate(tokens):
        if t in stop_ids:
            return tokens[:i]
    return tokens
