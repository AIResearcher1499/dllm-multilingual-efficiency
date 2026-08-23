"""K2 — how does a larger canvas actually cost more?

**Superseded in interpretation, 2026-08-21.** K2 was written when the cost
model said a forward is constant-time, so the only way a bigger canvas could
cost anything was by needing more steps. On that model, `S` flat in `L` meant
mechanism A could not operate and the design was dead.

The A100 measurement replaced that model. A diffusion step is a prefill, and
`t(L) = 15.4 + 0.102 L` ms for Dream-7B: a 6.4x canvas costs 4.4x **per step**.
Mechanism A therefore operates through time-per-forward, which is measured and
plainly non-zero, whatever the step count does.

Worse for the original reading, the sampler now stops when a stop token has an
unbroken committed prefix. Step count is then `content / parallel_factor`, and
content does not depend on canvas — so `S` is *expected* to be flat in `L`, and
"K2 FIRES" is the predicted, healthy outcome rather than a kill.

What this check is still good for: confirming the early stop works (NFE must
not grow with canvas once the answer is complete) and measuring `b`, the
per-token slope of the forward pass, on whatever card the run uses. Read the
exponent as a description, not a verdict.
"""

from __future__ import annotations

import math

# Below this, S is close enough to flat that a larger canvas is nearly free.
MIN_SCALING_EXPONENT = 0.5


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def scaling_exponent(canvases: list[float], nfes: list[float]) -> float | None:
    """Exponent `a` in `NFE ~ canvas^a`, from the endpoints.

    1.0 means one step per token (the naive convention). 0.0 means the step
    count ignores the canvas entirely.
    """
    pairs = sorted((c, n) for c, n in zip(canvases, nfes) if c > 0 and n > 0)
    if len(pairs) < 2 or pairs[0][0] == pairs[-1][0]:
        return None
    (c_lo, n_lo), (c_hi, n_hi) = pairs[0], pairs[-1]
    if c_hi == c_lo or n_lo == 0:
        return None
    return math.log(n_hi / n_lo) / math.log(c_hi / c_lo)


def k2_verdict(points: list[dict]) -> dict:
    """points: [{canvas, nfe, parallel_factor}] at one language, one mode."""
    by_canvas: dict[int, list[dict]] = {}
    for p in points:
        if p.get("error") or p.get("nfe") is None:
            continue
        by_canvas.setdefault(int(p["canvas"]), []).append(p)
    if len(by_canvas) < 2:
        return {"decision": "INCOMPLETE", "n_canvases": len(by_canvas),
                "reads_as": "need at least two canvas sizes to see a slope"}
    canvases = sorted(by_canvas)
    nfe = [_mean([p["nfe"] for p in by_canvas[c]]) for c in canvases]
    pf = [_mean([p["parallel_factor"] for p in by_canvas[c]
                 if p.get("parallel_factor") is not None]) for c in canvases]
    a = scaling_exponent([float(c) for c in canvases], nfe)
    if a is None:
        decision, reads = "INCOMPLETE", "could not fit a slope"
    elif a >= MIN_SCALING_EXPONENT:
        decision = "NFE-SCALES"
        reads = (
            f"NFE ~ canvas^{a:.2f}: step count still grows with the canvas. "
            "With early stopping enabled this is unexpected and means "
            "generations are not terminating — check the stop rate before "
            "trusting any timing from this run."
        )
    else:
        decision = "FLAT-NFE"
        reads = (
            f"NFE ~ canvas^{a:.2f}: step count barely tracks the canvas. With "
            "early stopping this is the EXPECTED result, not a kill — step "
            "count is content/parallel_factor and content does not depend on "
            "canvas. Mechanism A acts through time-per-forward instead, which "
            "is measured directly (t = a + b*L). Confirm b > 0 before reading "
            "anything further into this."
        )
    return {
        "decision": decision,
        "scaling_exponent": a,
        "threshold": MIN_SCALING_EXPONENT,
        "canvases": canvases,
        "mean_nfe": nfe,
        "mean_parallel_factor": pf,
        "reads_as": reads,
    }
