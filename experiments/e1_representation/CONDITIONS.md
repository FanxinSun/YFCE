# E1 as run

Conditions, and every place the run departs from the protocol in
`../README.md`. Recorded before the numbers came back.

## As specified

- **Model** — Llama-3.1-8B-Instruct, BF16, from `/mnt/d/Models/`. The model E1
  names, and the one lmz's published figures were measured on.
- **Eval** — wikitext-2-raw-v1 test, the full 1,285,622 characters,
  288,937 Llama tokens.
- **Baseline** — Q4_K including `make_qkx2_quants`, ggml's 21-step
  error-minimising scale search, not a min/max reimplementation.

## Departures, and why

**Baseline is a port, not the llama.cpp binary.** `llama-quantize` is not
installed and installing was ruled out. `qk.py` ports `make_qkx2_quants`,
`quantize_row_q4_K_impl` and `get_scale_min_k4` from ggml-quants.c and
vectorises them.

*Validated:* the port beats a naive min/max quantiser by **17.7% SSE** on
synthetic weights, and raw packing accounts to **exactly 4.5000 bpw**. A broken
port would show neither. This is the load-bearing check — if the search were
inert the baseline would be a strawman and any candidate would beat it for the
wrong reason.

**Fake-quant, not a packed file.** Weights are quantised and dequantised in
place; the model runs in bf16. Bit accounting is computed analytically from
symbol histograms. This measures the representation, which is what E1 asks, and
sidesteps needing a kernel for every candidate.

**Bits are exact empirical entropy, not a coded stream.** Frequency tables are
charged at 12 bits per symbol. rANS lands within 2% of measured entropy (lmz's
own suite asserts this), so these are slightly conservative rather than
idealised.

**Perplexity windows are 512 tokens, 250 of them** (128,000 tokens), not the
full 564. Absolute perplexity will not match a `llama-perplexity` run —
windowing and tokenisation differ. Only differences between variants are read,
and every variant sees identical windows.

**`embed_tokens` stays on CPU.** `accelerate` is unavailable so `device_map`
cannot be used, and the 14.96 GiB model does not otherwise fit in 14.60 GiB of
free VRAM. A forward hook moves the embedding output to the GPU. Affects speed
only, not numerics.

**Quantisation is chunked** at 2^18 sub-blocks with histogram accumulation,
because the largest projection (4096x14336) would otherwise allocate several GB
against ~0.6 GiB of headroom.

## Not done

- **No downstream task.** E1 asks for one alongside perplexity so the result is
  not purely a language-modelling artefact. Not run — no eval harness available
  without installing one. **The result is weaker for it** and should not be
  treated as settled on perplexity alone.
- **One level count (16).** The full bpw sweep across 8/16/32 levels is not run
  here; 16 is the Q4_K_M operating point and the one the gate is about.
- **Only `attn`/`mlp` projections are quantised.** Embeddings, LM head and norms
  stay bf16 in every variant, so mixed-precision policy is held fixed rather
  than being a confounder.
