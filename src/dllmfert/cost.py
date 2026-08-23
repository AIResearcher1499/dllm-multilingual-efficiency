"""Cost estimator for the stage-2 grid, driven by the measured fertility file.

The dominant cost is that a masked dLLM has no KV cache: every denoising step
is a full forward over prompt+canvas. At batch 1 a 7B bf16 forward is bound by
reading the weights, not by sequence length, so time per step is roughly
constant and the bill is essentially **number of forwards**.
"""

from __future__ import annotations

from dllmfert.g0_run import G0Config, canvas_for

# MEASURED on an A100, 2026-08-21. The earlier model assumed a batch-1 forward
# is bound by reading the 14 GB of weights, so time per forward was constant
# and the bill was "number of forwards". That is autoregressive physics: one
# token per step. A diffusion step processes the WHOLE canvas, so it is a
# prefill and is compute-bound above a few hundred tokens.
#
#   canvas 256  -> 41.5 ms (Dream-7B), 43.0 ms (LLaDA-8B)
#   canvas 1632 -> 181.7 ms          , 211.0 ms
#
# i.e. t(L) = a + b*L, and a 6.4x canvas costs 4.4x per step, not 1.0x.
# Getting this wrong understated the grid by roughly a factor of two and hid
# the fact that dLLM cost grows with the SQUARE of fertility: canvas scales
# with it, and so does the number of steps.
FORWARD_MS = {           # arm -> (fixed ms, ms per canvas token)
    "dream": (15.4, 0.1019),
    "llada": (11.7, 0.1221),
}
AR_MS_PER_TOKEN = 20.0   # measured Qwen2.5-7B, discarding the 258 ms first call

# name -> (dense bf16 tensor-core TFLOPS, on-demand USD/h). RunPod prices
# 2026-08-21. Throughput must be the *bf16 tensor* figure, not FP32: using
# FP32 for the Ampere workstation cards understated them fourfold and produced
# a nonsensical 93-hour estimate for the A6000.
GPUS = {
    "RTX A6000 48GB": (154.8, 0.33),
    "A40 48GB": (149.7, 0.35),
    "L40S 48GB": (362.1, 0.79),
    "A100 PCIe 80GB": (312.0, 1.19),
    "A100 SXM 80GB": (312.0, 1.39),
    "H100 SXM 80GB": (989.4, 2.69),
}
OVERHEAD = (0.85, 1.4)   # spread seen across repeats of the same canvas


def forward_ms(canvas: int, arm: str = "dream") -> float:
    a, b = FORWARD_MS.get(arm, FORWARD_MS["dream"])
    return a + b * canvas


def grid_forwards(fertility: dict[str, float], cfg: G0Config, *,
                  parallel_factor: float = 2.0,
                  langs: list[str] | None = None,
                  modes: tuple[str, ...] | None = None,
                  n_dllm_models: int = 2) -> dict:
    """Forward passes for the diffusion arms, and AR tokens for the baseline."""
    langs = langs or sorted(fertility)
    modes = modes or cfg.modes
    canvases = {l: canvas_for(cfg, fertility[l]) for l in langs}
    per_item = 0
    for canvas in canvases.values():
        for mode in modes:
            per_item += canvas if mode == "naive" else canvas / parallel_factor
    dllm_forwards = per_item * cfg.n * n_dllm_models
    ar_tokens = sum(canvases.values()) * cfg.n * 0.6  # AR stops early on MGSM
    return {
        "langs": langs,
        "canvases": canvases,
        "n_items": cfg.n,
        "n_modes": len(modes),
        "n_dllm_models": n_dllm_models,
        "total_canvas": sum(canvases.values()),
        "forwards_per_item_all_langs": per_item,
        "dllm_forwards": dllm_forwards,
        "ar_tokens": ar_tokens,
    }


def hours(grid: dict, arm: str = "dream", parallel_factor: float = 3.0,
          gpu: str | None = None) -> tuple[float, float]:
    """Wall clock, from the measured per-forward model rather than a constant.

    `gpu` is accepted for compatibility but only scales the estimate: the
    measurement was taken on an A100 PCIe, and a card with half the compute
    roughly doubles the compute-bound part.
    """
    # Only the compute-bound part of t(L) = a + b*L scales with the card. The
    # fixed term is kernel launch plus weight read and stays put.
    scale = 1.0
    if gpu and gpu in GPUS:
        scale = GPUS["A100 PCIe 80GB"][0] / GPUS[gpu][0]
    a, b = FORWARD_MS.get(arm, FORWARD_MS["dream"])
    total_s = 0.0
    for canvas in grid["canvases"].values():
        nfe = canvas / max(1e-9, parallel_factor)
        total_s += nfe * (a + b * canvas * scale) / 1000
    total_s *= grid["n_items"] * grid["n_modes"] * grid["n_dllm_models"]
    # AR decode is one token per forward: memory-bound, so it does not scale
    # with the card's compute the way the diffusion arm does.
    ar_s = grid["ar_tokens"] * AR_MS_PER_TOKEN / 1000
    lo = (total_s * OVERHEAD[0] + ar_s) / 3600
    hi = (total_s * OVERHEAD[1] + ar_s) / 3600
    return lo, hi


def dollars(grid: dict, gpu: str, setup_hours: float = 0.75,
            arm: str = "dream", parallel_factor: float = 3.0) -> dict:
    """setup_hours is dominated by ~45 GB of model downloads, which is why the
    trivial runs still cost close to an hour."""
    _, rate = GPUS[gpu]
    lo_h, hi_h = hours(grid, arm=arm, parallel_factor=parallel_factor, gpu=gpu)
    return {
        "gpu": gpu,
        "rate": rate,
        "gpu_hours": (round(lo_h, 2), round(hi_h, 2)),
        "with_setup_hours": (round(lo_h + setup_hours, 2),
                             round(hi_h + setup_hours, 2)),
        "usd": (round((lo_h + setup_hours) * rate, 2),
                round((hi_h + setup_hours) * rate, 2)),
    }
