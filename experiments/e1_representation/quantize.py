"""Chunked driver for both representations.

Llama-3.1-8B's largest projection is 4096x14336 -- 1.83M sub-blocks. Quantising
that in one shot allocates several GB, and this experiment runs with ~0.6 GiB of
VRAM headroom because the model itself occupies the rest of the card. So every
tensor is processed in fixed-size chunks, and bit accounting accumulates
*histograms* rather than symbol arrays: entropy only needs counts, and counts
are O(alphabet) instead of O(weights).
"""

from __future__ import annotations

import torch

from qk import NMAX, NSUB, QK_K, SUB, _fp16, _nearest_int, make_qkx2_quants

FREQ_TABLE_BITS = 12
NCTX = 8  # sub-block magnitude classes
CHUNK_SUB = 1 << 18  # sub-blocks per chunk


def _entropy_bits_from_counts(counts: torch.Tensor) -> float:
    """Bits to code a stream with these symbol counts, plus its frequency table."""
    counts = counts.double()
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * torch.log2(p)).sum() * n) + counts.numel() * FREQ_TABLE_BITS


def _cond_entropy_bits(table: torch.Tensor) -> float:
    """`table` is (contexts, alphabet); each context codes under its own model."""
    return sum(_entropy_bits_from_counts(table[c]) for c in range(table.shape[0]))


def fit_codebook(w: torch.Tensor, levels: int, iters: int = 20, sample: int = 1 << 21):
    """Lloyd-Max on normalised values, initialised at the uniform grid.

    Initialising uniformly means training distortion can only improve on the
    uniform quantiser, so any loss on the perplexity curve is a real
    generalisation effect rather than a fitting artefact.
    """
    xs = []
    for chunk in _iter_chunks(w, levels):
        v = chunk["normalised"].reshape(-1)
        xs.append(v[:: max(1, v.numel() // (sample // 8))])
    x = torch.cat(xs)
    if x.numel() > sample:
        x = x[:sample]
    cb = torch.linspace(0, levels - 1, levels, device=x.device, dtype=torch.float32)
    for _ in range(iters):
        idx = torch.bucketize(x, (cb[1:] + cb[:-1]) / 2)
        s = torch.zeros(levels, device=x.device, dtype=torch.float64)
        c = torch.zeros(levels, device=x.device, dtype=torch.float64)
        s.scatter_add_(0, idx, x.double())
        c.scatter_add_(0, idx, torch.ones_like(x, dtype=torch.float64))
        cb = torch.where(c > 0, (s / c.clamp(min=1)).float(), cb)
        cb, _ = torch.sort(cb)
    return cb


def _iter_chunks(w: torch.Tensor, levels: int):
    """Yield per-chunk intermediate state: scales, mins, and normalised values."""
    nmax = levels - 1
    flat = w.reshape(-1).float()
    pad = (-flat.numel()) % QK_K
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])
    nb_total = flat.numel() // QK_K
    per = max(1, CHUNK_SUB // NSUB)  # superblocks per chunk

    for start in range(0, nb_total, per):
        nb = min(per, nb_total - start)
        blocks = flat[start * QK_K : (start + nb) * QK_K].reshape(nb, QK_K)

        sigma2 = 2.0 * (blocks * blocks).sum(dim=1) / QK_K
        sub = blocks.reshape(nb * NSUB, SUB)
        imp = torch.sqrt(sigma2).repeat_interleave(NSUB)[:, None] + sub.abs()

        scales, mins, _ = make_qkx2_quants(sub, imp, nmax=nmax)
        scales = scales.reshape(nb, NSUB)
        mins = mins.reshape(nb, NSUB)

        max_scale = scales.max(dim=1).values
        max_min = mins.max(dim=1).values
        inv_s = torch.where(max_scale > 0, 63.0 / max_scale, torch.zeros_like(max_scale))
        inv_m = torch.where(max_min > 0, 63.0 / max_min, torch.zeros_like(max_min))
        ls = torch.clamp(_nearest_int(inv_s[:, None] * scales), 0, 63)
        lm = torch.clamp(_nearest_int(inv_m[:, None] * mins), 0, 63)
        d = _fp16(max_scale / 63.0)
        dmin = _fp16(max_min / 63.0)

        dsub = (d[:, None] * ls).reshape(nb * NSUB, 1)
        msub = (dmin[:, None] * lm).reshape(nb * NSUB, 1)
        nz = dsub != 0
        safe = torch.where(nz, dsub, torch.ones_like(dsub))
        normalised = torch.clamp((sub + msub) / safe, 0, nmax)

        yield {
            "nb": nb, "start": start, "ls": ls, "lm": lm,
            "dsub": dsub, "msub": msub, "nz": nz, "normalised": normalised,
        }


def quantize(w: torch.Tensor, levels: int = 16, codebook: torch.Tensor | None = None) -> dict:
    """Quantise one weight tensor. `codebook=None` reproduces ggml's uniform grid."""
    shape = w.shape
    n = shape.numel()
    out = torch.empty(n + (-n) % QK_K, device=w.device, dtype=torch.float32)

    q_counts = torch.zeros(levels, dtype=torch.float64, device=w.device)
    q_cond = torch.zeros(NCTX, levels, dtype=torch.float64, device=w.device)
    ls_counts = torch.zeros(64, dtype=torch.float64, device=w.device)
    lm_counts = torch.zeros(64, dtype=torch.float64, device=w.device)
    nb_total = 0

    for ch in _iter_chunks(w, levels):
        norm, nz = ch["normalised"], ch["nz"]
        if codebook is None:
            L = torch.clamp(_nearest_int(norm), 0, levels - 1)
            recon = L
        else:
            L = torch.bucketize(norm, (codebook[1:] + codebook[:-1]) / 2).float()
            recon = codebook[L.long()]
        L = torch.where(nz, L, torch.zeros_like(L))

        deq = ch["dsub"] * recon - ch["msub"]
        s = ch["start"] * QK_K
        out[s : s + ch["nb"] * QK_K] = deq.reshape(-1)

        qi = L.reshape(-1).long()
        q_counts += torch.bincount(qi, minlength=levels).double()
        ctx = torch.clamp((ch["ls"].float() / 64.0 * NCTX).long(), 0, NCTX - 1)
        ctx = ctx.reshape(-1).repeat_interleave(SUB)
        q_cond += torch.bincount(ctx * levels + qi, minlength=NCTX * levels).double().reshape(NCTX, levels)
        ls_counts += torch.bincount(ch["ls"].reshape(-1).long(), minlength=64).double()
        lm_counts += torch.bincount(ch["lm"].reshape(-1).long(), minlength=64).double()
        nb_total += ch["nb"]

    idx_bits = (levels - 1).bit_length()
    bits = {
        "q_raw": nb_total * QK_K * idx_bits,
        "q_order0": _entropy_bits_from_counts(q_counts),
        "q_cond": _cond_entropy_bits(q_cond),
        "scales_raw": nb_total * NSUB * 12,  # 6-bit scale + 6-bit min
        "scales_order0": _entropy_bits_from_counts(ls_counts)
        + _entropy_bits_from_counts(lm_counts),
        "super_raw": nb_total * 32,  # d, dmin as fp16
        "codebook": 0 if codebook is None else levels * 16,
    }
    return {"dequant": out[:n].reshape(shape).to(w.dtype), "bits": bits, "numel": n}


def bpw(bits: dict, numel: int, variant: str) -> float:
    fixed = bits["super_raw"] + bits["codebook"]
    parts = {
        "raw": bits["q_raw"] + bits["scales_raw"],
        "A": bits["q_order0"] + bits["scales_raw"],
        "AB": bits["q_order0"] + bits["scales_order0"],
        "ABc": bits["q_cond"] + bits["scales_order0"],
    }
    return (parts[variant] + fixed) / numel
