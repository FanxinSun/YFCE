"""E1: does a purpose-designed representation beat Q4_K at iso-quality?

Three model passes, because holding two copies of an 8B model does not fit here:

  1. bf16 reference
  2. ggml uniform grid  -- faithful Q4_K, including make_qkx2_quants
  3. Lloyd-Max codebook -- same structure, non-uniform reconstruction levels

The lossless variants (entropy-coded indices, entropy-coded scale planes,
magnitude-conditioned indices) reconstruct bit-identical values to their parent,
so they share its perplexity by construction and only their bit accounting
differs. That is the point of separating them: they move bpw at fixed quality.

    python3 run.py --limit 250
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data import wikitext2_test  # noqa: E402
from ppl import load_cpu, perplexity, target_modules, to_gpu  # noqa: E402
from quantize import bpw, fit_codebook, quantize  # noqa: E402

MODEL = "/mnt/d/Models/Llama-3.1-8b-Instruct"


def quantize_model(model, levels: int, use_codebook: bool):
    """Weights live on the CPU here; each tensor visits the GPU alone.

    The largest projection is 117 MiB, so with the model still off the card the
    scale search has the whole 14.6 GiB to work in.
    """
    totals, numel = {}, 0
    mods = target_modules(model)
    for i, (name, mod) in enumerate(mods):
        wg = mod.weight.data.to("cuda", non_blocking=False)
        cb = fit_codebook(wg, levels) if use_codebook else None
        out = quantize(wg, levels=levels, codebook=cb)
        mod.weight.data = out["dequant"].to("cpu")
        numel += out["numel"]
        for k, v in out["bits"].items():
            totals[k] = totals.get(k, 0) + v
        del wg, out, cb
        torch.cuda.empty_cache()
        if (i + 1) % 40 == 0:
            print(f"    quantised {i + 1}/{len(mods)} tensors", flush=True)
    return totals, numel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--levels", type=int, default=16)
    ap.add_argument("--limit", type=int, default=250, help="eval windows of 512 tokens")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(wikitext2_test(), return_tensors="pt").input_ids
    print(f"eval: {ids.shape[1]:,} tokens, {args.limit} windows of 512\n", flush=True)

    rows = []

    def run(tag: str, use_codebook: bool | None):
        t0 = time.time()
        print(f"[{tag}] loading...", flush=True)
        m = load_cpu(args.model)
        bits = numel = None
        if use_codebook is not None:
            print(f"[{tag}] quantising (levels={args.levels})...", flush=True)
            bits, numel = quantize_model(m, args.levels, use_codebook)
        to_gpu(m)
        print(f"[{tag}] evaluating...", flush=True)
        p = perplexity(m, ids, limit=args.limit)
        print(f"[{tag}] ppl {p:.4f}   ({time.time() - t0:.0f}s)\n", flush=True)
        del m
        torch.cuda.empty_cache()
        return p, bits, numel

    ref, _, _ = run("bf16", None)
    rows.append({"variant": "bf16", "bpw": 16.0, "ppl": ref, "d_ppl": 0.0})

    p_uni, bits_uni, numel = run("uniform", False)
    p_cb, bits_cb, _ = run("codebook", True)

    for tag, p, bits in (("ggml-Q4_K", p_uni, bits_uni), ("YQ4-cb", p_cb, bits_cb)):
        for v in ("raw", "A", "AB", "ABc"):
            b = bpw(bits, numel, v)
            name = f"{tag}" if v == "raw" else f"{tag}+{v}"
            rows.append({"variant": name, "bpw": b, "ppl": p, "d_ppl": p - ref})

    print(f"{'variant':<22}{'bpw':>8}{'ppl':>10}{'d_ppl':>9}")
    for r in rows:
        print(f"{r['variant']:<22}{r['bpw']:>8.3f}{r['ppl']:>10.4f}{r['d_ppl']:>9.4f}")

    Path(args.out).write_text(
        json.dumps({"model": args.model, "levels": args.levels,
                    "windows": args.limit, "rows": rows}, indent=2)
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
