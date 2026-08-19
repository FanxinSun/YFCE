# E1 results — the representation gate

**Question.** Can a purpose-designed representation beat `Q4_K` at iso-quality
by enough to justify a decode block in custom silicon?

**Answer. No.** 4.52% fewer bits at identical quality, and the lossy half of the
candidate is actively worse. Conditions and departures in `CONDITIONS.md`.

Llama-3.1-8B-Instruct, wikitext-2-raw test, 250 windows x 512 tokens, levels=16.

| variant | bpw | ppl | Δppl vs bf16 | bits vs Q4_K |
|---|---|---|---|---|
| bf16 reference | 16.000 | 9.1123 | — | — |
| **ggml Q4_K** (faithful) | **4.5000** | **9.5154** | +0.4031 | — |
| Q4_K + entropy-coded indices | 4.3621 | 9.5154 | +0.4031 | **−3.06%** |
| Q4_K + also scale planes | 4.3016 | 9.5154 | +0.4031 | **−4.41%** |
| Q4_K + also magnitude conditioning | 4.2967 | 9.5154 | +0.4031 | **−4.52%** |
| YQ4 Lloyd-Max codebook | 4.5000 | **9.5456** | +0.4333 | −0.00% |
| YQ4 codebook + all entropy coding | 4.2996 | 9.5456 | +0.4333 | −4.45% |

Perplexity is identical to four decimals across the entropy-coded rows because
they reconstruct bit-identical values. That is the design: they move bpw at
fixed quality, which is the only clean way to read a ratio win.

## The full sweep

Same model, same windows, three level counts. `entropy gain` is the lossless
reduction at *identical* perplexity; `codebook Δppl` is positive when the
Lloyd-Max codebook beats the uniform grid.

| levels | index bits | base bpw | base ppl | entropy gain | best bpw | codebook Δppl |
|---|---|---|---|---|---|---|
| 8 | 3 | 3.5000 | 11.5450 | **6.22%** | 3.2822 | +0.0569 |
| 16 | 4 | 4.5000 | 9.5154 | **4.52%** | 4.2967 | −0.0302 |
| 32 | 5 | 5.5000 | 9.2277 | **3.68%** | 5.2974 | +0.0235 |

**The entropy-coding trend is clean and has a mechanism.** Gains rise as bpw
falls, because the scale and min planes cost a fixed 0.5 bpw regardless of level
count -- 14.3% of a 3.5 bpw file but only 9.1% of a 5.5 bpw one -- and those
planes are the most compressible part of the block. Nothing about this changes
the verdict: the best point on the whole curve is 6.22%.

**The codebook has no trend and no mechanism.** It is worse at 4-bit, better at
3-bit and 5-bit, always by less than 0.06 perplexity. An earlier reading of the
3-bit point as "coarser grid leaves more room" does not survive the 5-bit point,
which should then have shown the least benefit and instead shows some. The
defensible statement is that a per-tensor Lloyd-Max codebook is worth
approximately nothing at every operating point tested, with a sign that depends
on details of the fit rather than on bpw.

The runs are deterministic -- bf16 reproduced 9.1123 across three separate
invocations -- so this is a real property of the method, not evaluation noise.

## The two findings

**1. The lossless win is real, small, and independently corroborated.**
4.52%, at exactly equal quality. lmz measured 5.1% on real Q4_K_M files by a
completely different route — coding an existing GGUF rather than quantising from
BF16. Two independent measurements landing within 0.6 points is the strongest
evidence in this experiment that the number is right.

Conditioning barely helps: order-0 gives 4.41%, adding sub-block magnitude
context gives 4.52%. That 0.11-point increment matches lmz's earlier finding
that context is nearly worthless on quantised payloads.

**2. The lossy redesign fails, and the reason is structural.**
A Lloyd-Max codebook, initialised at the uniform grid and fitted per tensor,
converges back to almost exactly the uniform grid — `[0.09, 1.05, 2.03, … 14.91]`
— and ends up 0.03 perplexity *worse* for no bit saving.

Q4_K already normalises every 32-weight sub-block by its own min and max. After
that normalisation the residual distribution is close to uniform, so there is no
shape left for a better codebook to exploit. The adaptive scaling did the work a
non-uniform codebook would have done. This reproduced on synthetic lognormal
weights before the real run, so it is not a fitting artefact.

## What it means for the chip

The decode block's whole justification was that `1/f` multiplies the memory bus.

    f = 0.955  →  effective bandwidth x1.047

A ~2.6 mm² interleaved rANS block, a format that bakes in its lane count, and a
hardware decode path — for 4.7% more tokens per second. A slightly faster DDR5
grade buys more, for nothing. **The decode block is not justified by this
result**, and it was the one piece of genuinely distinctive silicon in the design.

## The caveat that matters most

This tests one representation family: non-uniform scalar quantisation plus
entropy coding, at one operating point. That family lost. It is **not** true that
the representation axis is exhausted in general — AQLM, QuIP# and imatrix-guided
k-quants beat Q4_K_M by considerably more than 4.5%.

But note what they spend to do it. Vector quantisation needs large codebook
lookups; QuIP#-style methods need a Hadamard transform per block. **The
representations that win on ratio win by making decode more expensive** — which
is exactly the resource a mature-node fixed-function block does not have.

That tension is the real finding here, and it is worse news for the chip than the
4.52% is. Ratio and cheap-decode are not independent axes; on this evidence they
trade against each other.

## Not established

- **Perplexity only, no downstream task.** The largest weakness in this result.
- **No vector-quantised or rotation-based candidate was tried** — AQLM, QuIP#,
  or imatrix-guided quantisation. This is the family that would actually beat
  Q4_K_M, and the family whose decoders are expensive. Testing one would settle
  the ratio-versus-decode-cost tension directly instead of by inference.
- 2-bit not run. The entropy trend suggests ~8%, still far below the bar, so it
  was not worth the run.
