"""Faithful Q4_K, ported from ggml.

The baseline has to be llama.cpp's real quantiser, not a min/max reimplementation.
Q4_K's quality comes almost entirely from `make_qkx2_quants` -- a 21-step search
over candidate scales, minimising weighted squared error with a magnitude-derived
importance weight -- and a baseline without it is a strawman that any candidate
beats for the wrong reason.

Ported from ggml-quants.c (`make_qkx2_quants`, `quantize_row_q4_K_impl`,
`get_scale_min_k4`), vectorised over blocks in torch. Layout, 256 weights per
superblock, 144 bytes:

    d      fp16                      super-scale for the 6-bit scales
    dmin   fp16                      super-scale for the 6-bit mins
    scales 12 bytes                  8 sub-blocks x (6-bit scale, 6-bit min)
    qs     128 bytes                 256 x 4-bit quants

Reconstruction, per 32-weight sub-block j:  w = (d * sc[j]) * q  -  (dmin * m[j])
"""

from __future__ import annotations

import torch

QK_K = 256
SUB = 32  # weights per sub-block
NSUB = QK_K // SUB  # 8
NMAX = 15  # 4-bit


def _nearest_int(x: torch.Tensor) -> torch.Tensor:
    """ggml's nearest_int is round-half-away-from-zero, not torch's round-half-even."""
    return torch.floor(torch.abs(x) + 0.5) * torch.sign(x)


def make_qkx2_quants(
    x: torch.Tensor,  # (B, SUB) float32
    weights: torch.Tensor,  # (B, SUB) importance
    nmax: int = NMAX,
    rmin: float = -1.0,
    rdelta: float = 0.1,
    nstep: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (scale, the_min, L). Vectorised over the block dimension B."""
    xmin = x.min(dim=1).values
    xmax = x.max(dim=1).values
    sum_w = weights.sum(dim=1)
    sum_x = (weights * x).sum(dim=1)

    mn = torch.clamp(xmin, max=0.0)  # ggml: if (min > 0) min = 0
    degenerate = xmax == mn

    span = (xmax - mn).clamp(min=1e-30)
    iscale = nmax / span
    scale = 1.0 / iscale

    L = torch.clamp(_nearest_int(iscale[:, None] * (x - mn[:, None])), 0, nmax)
    diff = scale[:, None] * L + mn[:, None] - x
    best_mad = (weights * diff * diff).sum(dim=1)

    best_scale = scale.clone()
    best_min = mn.clone()
    best_L = L.clone()

    for is_ in range(nstep + 1):
        iscale = (rmin + rdelta * is_ + nmax) / span
        Laux = torch.clamp(_nearest_int(iscale[:, None] * (x - mn[:, None])), 0, nmax)

        sum_l = (weights * Laux).sum(dim=1)
        sum_l2 = (weights * Laux * Laux).sum(dim=1)
        sum_xl = (weights * Laux * x).sum(dim=1)

        D = sum_w * sum_l2 - sum_l * sum_l
        ok = D > 0
        Ds = torch.where(ok, D, torch.ones_like(D))

        this_scale = (sum_w * sum_xl - sum_x * sum_l) / Ds
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / Ds

        # ggml: if (this_min > 0) { this_min = 0; this_scale = sum_xl/sum_l2; }
        pos = this_min > 0
        safe_l2 = torch.where(sum_l2 > 0, sum_l2, torch.ones_like(sum_l2))
        this_scale = torch.where(pos, sum_xl / safe_l2, this_scale)
        this_min = torch.where(pos, torch.zeros_like(this_min), this_min)

        d2 = this_scale[:, None] * Laux + this_min[:, None] - x
        mad = (weights * d2 * d2).sum(dim=1)

        take = ok & (mad < best_mad)
        best_mad = torch.where(take, mad, best_mad)
        best_scale = torch.where(take, this_scale, best_scale)
        best_min = torch.where(take, this_min, best_min)
        best_L = torch.where(take[:, None], Laux, best_L)

    best_scale = torch.where(degenerate, torch.zeros_like(best_scale), best_scale)
    best_L = torch.where(degenerate[:, None], torch.zeros_like(best_L), best_L)
    return best_scale, -best_min, best_L


def _fp16(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through fp16, as the on-disk format does."""
    return x.half().float()


def quantize_q4_k(w: torch.Tensor) -> dict:
    """Quantise a 2-D weight tensor. Returns the coded fields plus the dequantised
    tensor, so callers can do fake-quant evaluation and bit accounting together."""
    shape = w.shape
    flat = w.reshape(-1).float()
    pad = (-flat.numel()) % QK_K
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])
    blocks = flat.reshape(-1, QK_K)  # (nb, 256)
    nb = blocks.shape[0]

    # ggml importance weights, no imatrix: w_l = av_x + |x_l|, av_x = sqrt(2*sum_x2/QK_K)
    sigma2 = 2.0 * (blocks * blocks).sum(dim=1) / QK_K
    av_x = torch.sqrt(sigma2)
    sub = blocks.reshape(nb * NSUB, SUB)
    imp = av_x.repeat_interleave(NSUB)[:, None] + sub.abs()

    scales, mins, _ = make_qkx2_quants(sub, imp)
    scales = scales.reshape(nb, NSUB)
    mins = mins.reshape(nb, NSUB)

    # Quantise the scales and mins to 6 bits against a per-superblock maximum.
    max_scale = scales.max(dim=1).values
    max_min = mins.max(dim=1).values
    inv_scale = torch.where(max_scale > 0, 63.0 / max_scale, torch.zeros_like(max_scale))
    inv_min = torch.where(max_min > 0, 63.0 / max_min, torch.zeros_like(max_min))

    ls = torch.clamp(_nearest_int(inv_scale[:, None] * scales), 0, 63)
    lm = torch.clamp(_nearest_int(inv_min[:, None] * mins), 0, 63)

    d = _fp16(max_scale / 63.0)
    dmin = _fp16(max_min / 63.0)

    # Requantise the weights against the *quantised* scales, as ggml does.
    dsub = (d[:, None] * ls).reshape(nb * NSUB, 1)
    msub = (dmin[:, None] * lm).reshape(nb * NSUB, 1)
    nz = dsub != 0
    L = torch.zeros_like(sub)
    L = torch.where(
        nz,
        torch.clamp(_nearest_int((sub + msub) / torch.where(nz, dsub, torch.ones_like(dsub))), 0, NMAX),
        L,
    )

    deq = (dsub * L - msub).reshape(-1)[: shape.numel()].reshape(shape)

    return {
        "q": L.reshape(nb, QK_K).to(torch.uint8),  # 4-bit indices
        "ls": ls.to(torch.uint8),  # 6-bit sub-scales
        "lm": lm.to(torch.uint8),  # 6-bit sub-mins
        "d": d,  # fp16 super-scale
        "dmin": dmin,  # fp16 super-min
        "dequant": deq.to(w.dtype),
        "nb": nb,
        "numel": shape.numel(),
    }


def q4_k_bits_per_weight() -> float:
    """144 bytes per 256 weights, by construction."""
    return 144 * 8 / QK_K  # 4.5
