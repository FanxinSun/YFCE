"""Perplexity harness.

Absolute values will not match a `llama-perplexity` run -- windowing and
tokenisation differ. E1 asks whether one representation beats another, so every
number comes from this same harness on the same windows and only differences
are read.

Memory note: the model occupies nearly the whole card, so the loss is computed
in slices. A 512x128256 logit tensor upcast to float32 in one piece is 262 MB,
which does not fit in the headroom that remains.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LOSS_SLICE = 128


def load_cpu(path: str):
    """Load bf16 onto the CPU and leave it there.

    Quantisation runs before the model reaches the GPU, and that ordering is
    load-bearing: once the weights are resident, ~370 MiB of VRAM remains and
    the scale search OOMs. Quantising first gives it the full card.
    """
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    m.eval()
    return m


def to_gpu(m, device: str = "cuda"):
    """Move everything except the embedding to the GPU.

    `accelerate` is unavailable so `device_map` cannot be used. Keeping
    `embed_tokens` (0.98 GiB) on the CPU is what brings a 14.96 GiB model inside
    14.60 GiB of VRAM; a forward hook carries its output across.
    """
    m.model.layers.to(device)
    m.model.norm.to(device)
    m.lm_head.to(device)
    if hasattr(m.model, "rotary_emb"):
        m.model.rotary_emb.to(device)
    m.model.embed_tokens.register_forward_hook(lambda mod, i, o: o.to(device))
    return m


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, ctx: int = 512, limit: int | None = None) -> float:
    n = ids.shape[1]
    nll, count = 0.0, 0
    for w, start in enumerate(range(0, n - 1, ctx)):
        if limit is not None and w >= limit:
            break
        end = min(start + ctx, n - 1)
        if end - start < 2:
            break
        logits = model(ids[:, start:end]).logits
        tgt = ids[:, start + 1 : end + 1].to(logits.device)
        for i in range(0, logits.shape[1], LOSS_SLICE):
            lg = logits[:, i : i + LOSS_SLICE].float()
            tg = tgt[:, i : i + LOSS_SLICE]
            nll += F.cross_entropy(
                lg.reshape(-1, lg.size(-1)), tg.reshape(-1), reduction="sum"
            ).item()
            count += tg.numel()
        del logits
    return float(torch.exp(torch.tensor(nll / count)))


def target_modules(model):
    """The 2-D projections a quantiser would touch.

    Embeddings, LM head and norms stay bf16 in every variant. Mixed-precision
    policy is a real design choice but an orthogonal one -- it moves both curves
    together, and holding it fixed isolates the representation comparison.
    """
    import torch.nn as nn

    return [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and m.weight.dim() == 2 and "lm_head" not in n
    ]
