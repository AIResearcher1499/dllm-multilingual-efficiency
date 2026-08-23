"""HF adapters. Torch lives here and nowhere else, so the sampler's selection
logic — the part that decides parallel_factor — stays testable on a laptop."""

from __future__ import annotations

MASK_TOKEN_CANDIDATES = ("<|mdm_mask|>", "<mask>", "[MASK]", "<|mask|>")


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_mask_id(tokenizer, model=None) -> int:
    """Dream and LLaDA name their mask token differently and neither exposes it
    as `mask_token_id` reliably. Fail loudly rather than decode against a token
    the model never saw."""
    if getattr(tokenizer, "mask_token_id", None) is not None:
        return int(tokenizer.mask_token_id)
    cfg = getattr(model, "config", None)
    for attr in ("mask_token_id", "mask_id"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    for name in MASK_TOKEN_CANDIDATES:
        ids = tokenizer.convert_tokens_to_ids(name)
        if ids is not None and ids != getattr(tokenizer, "unk_token_id", None):
            return int(ids)
    raise SystemExit(
        "cannot locate the mask token id for this model; pass --mask-id "
        "explicitly rather than guessing — decoding against the wrong token "
        "would silently produce meaningless parallel_factor values"
    )


def load(model_id: str, device: str | None = None):
    import torch
    import transformers
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # transformers renamed torch_dtype -> dtype in 5.x. The GPU env is pinned to
    # 4.46.2 because that is what Dream's bundled modeling code supports, so
    # send the name that version expects.
    dtype_kw = "torch_dtype" if int(transformers.__version__.split(".")[0]) < 5 else "dtype"
    kwargs = {dtype_kw: dtype, "trust_remote_code": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):  # LLaDA exposes a bare AutoModel
        model = AutoModel.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    return model, tok, device


PROBE_TEXT = (
    "The quick brown fox jumps over the lazy dog. She bought three apples and "
    "two oranges at the market, then walked home along the river before dinner."
)


def detect_logit_shift(model, tokenizer, *, mask_id: int, device: str,
                       probe_text: str = PROBE_TEXT) -> dict:
    """Which logit index holds the prediction for a masked position?

    A natively trained masked diffusion model puts it at the position itself.
    A model adapted from an autoregressive checkpoint keeps the AR convention,
    where ``logits[i]`` predicts token ``i+1``, and its own sampler shifts to
    compensate -- Dream does exactly that at ``generation_utils.py:412``:

        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

    Getting this wrong is **silent**. Confidences stay in [0,1], decoding
    terminates, nothing raises, and the output is confident nonsense: measured
    on an L40S, Dream answered a grade-school word problem with "muff muff muff
    muff ..." for 482 tokens. Nothing in the row schema could have caught it.

    So the alignment is measured rather than configured: mask a spread of
    interior positions in known text and see which reading recovers the tokens
    that are actually there.
    """
    import torch

    ids = tokenizer.encode(probe_text, add_special_tokens=False)
    if len(ids) < 12:
        raise SystemExit("probe text too short to calibrate logit alignment")
    probe_at = list(range(4, len(ids) - 1, 3))
    corrupted = list(ids)
    for p in probe_at:
        corrupted[p] = mask_id
    with torch.inference_mode():
        out = model(torch.tensor([corrupted], device=device))
        logits = out.logits if hasattr(out, "logits") else out[0]
        logprobs = torch.log_softmax(logits[0].float(), dim=-1)
    scores = {}
    for shift in (0, 1):
        total = 0.0
        for p in probe_at:
            idx = p - shift
            if idx < 0:
                continue
            total += float(logprobs[idx, ids[p]])
        scores[shift] = total / len(probe_at)
    best = max(scores, key=scores.get)
    return {
        "shift": best,
        "mean_logprob": scores,
        "margin": scores[best] - scores[1 - best],
        "convention": ("position-aligned (native masked diffusion)" if best == 0
                       else "AR-adapted: logits[i] predicts token i+1"),
    }


def make_scorer(model, prompt_ids: list[int], *, canvas: int, mask_id: int,
                device: str, shift: int = 0, repetition_penalty: float = 1.0):
    """Scorer for `sampler.decode`: rebuild the canvas, one full forward, read
    confidences at the still-masked positions.

    One forward per step over the whole sequence is not an implementation
    shortcut — it is the property that makes dLLM cost scale with canvas, and
    therefore the property this repo is measuring.

    **Masked-diffusion indexing.** `logits[offset + p]` is read as the
    distribution *for* canvas position `p`, which is what a bidirectional
    masked model produces. An autoregressive model puts the prediction for
    position `p` at `logits[offset + p - 1]` instead, so feeding one in here
    would be silently off by one and would yield meaningless confidences
    rather than an error. The AR baseline therefore goes through
    `ar_generate`, never through this function.
    """
    import torch

    base = list(prompt_ids)
    offset = len(base)

    def scorer(committed: dict[int, int], masked: list[int]):
        row = base + [committed.get(p, mask_id) for p in range(canvas)]
        ids = torch.tensor([row], device=device)
        with torch.inference_mode():
            out = model(ids)
            logits = out.logits if hasattr(out, "logits") else out[0]
            # `shift` comes from detect_logit_shift, never from a hardcoded
            # model name. Canvas position p is read at offset+p-shift.
            scores = logits[0, offset - shift:].float()
            if repetition_penalty != 1.0 and committed:
                seen = torch.tensor(sorted(set(committed.values())),
                                    device=scores.device, dtype=torch.long)
                col = scores.index_select(1, seen)
                # CTRL-style: divide positive logits, multiply negative ones,
                # so the penalty always moves mass away from a seen token.
                scores = scores.index_copy(
                    1, seen,
                    torch.where(col > 0, col / repetition_penalty,
                                col * repetition_penalty))
            probs = torch.softmax(scores, dim=-1)
            conf, pred = probs.max(dim=-1)
        return {p: (float(conf[p]), int(pred[p])) for p in masked}

    return scorer


def ar_generate(model, tokenizer, prompt_ids: list[int], *, max_new_tokens: int,
                device: str) -> tuple[list[int], int]:
    """Autoregressive baseline. Returns (generated ids, steps == tokens), since
    an AR model finalises exactly one token per forward — parallel_factor 1 by
    definition, which is what the diffusion arms are compared against."""
    import torch

    ids = torch.tensor([prompt_ids], device=device)
    with torch.inference_mode():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new = out[0, ids.shape[1]:].tolist()
    return new, len(new)
