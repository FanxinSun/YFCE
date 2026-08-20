# Residency — the tiers, and what decides what goes where

Blueprint M3 as a contract on a commodity GPU. This file fixes the tiers, the
unit they are managed in, the placement rule, and the two scheduling
properties without which the placement is worthless. `tiers.py` is the
executable half; this is the half that says what the engine must do.

**Scope: this is the read-only, inference-side contract.** Weights are never
written, so every block is clean and eviction is a free-list operation. The
training store is the same shape with one difference that reaches everywhere —
master weights and both Adam moments are rewritten every step, so it has dirty
blocks, a write-back path and a commit ordering. That is `runtime.md`, and
where the two disagree, `runtime.md` is the one describing the actual
project.

## The unit

**One 64 KiB page-aligned block, throughout.** The unit of storage, of DMA, of
decode, of eviction and of prefetch is the same block, and it is lmz's block —
blueprint I1 already fixed this for the SoC, and the measurement in
`MEASURED.md` says it is right here too: 64 KiB random reads at depth beat a
single-threaded sequential loop, so there is no gain in laying weights out to
please the disk and no penalty for paging at block granularity.

A **tensor** is a named range of blocks. An **expert** or an **adapter** is a
set of tensors. Nothing smaller than a block is ever moved and nothing is ever
partially resident.

## The tiers

| tier | holds | measured, BF16 | space cost per model byte |
|---|---|---|---|
| **V-raw** | decoded weights in VRAM | 887 GB/s | 1 |
| **V-coded** | coded blocks in VRAM, decoded on the way to the matrix units | **948 GB/s** | 1/ratio |
| **H** | coded blocks in page-locked host RAM | 42.8 GB/s | 0 |
| **S** | coded blocks in the archive on NVMe | 9.4 GB/s | 0 |

**V-coded is faster than V-raw**, and that is not a typo. `fused/fused_gemm.cu`
decodes an rANS stream into `mma` A-fragments in registers and never writes a
decoded weight to DRAM, so it moves 1/ratio of the bytes and the decoder is
quick enough to keep up: 948 GB/s of BF16 out of 182 MB against 887 out of
268. V-raw is read-only bandwidth, which is what a weight tier needs, and not
the read+write copy figure the bandwidth probe reports.

This file was written when V-coded ran at 0.51× of V-raw and had to argue for
its space. It does not any more, and several paragraphs below are marked where
that changes them.

Three rules follow from the ladder and none of them is optional.

**Host memory is always page-locked.** 1.8× for nothing (`MEASURED.md`). A
tier backed by pageable memory is not tier H, it is a slower tier that looks
like it.

**Nothing is ever stored raw off the card.** There is no raw H or raw S tier
in the table because there is no reason for one: decode outruns both links by
33× and 150×, so a raw byte off the card is strictly worse than a coded one.
Raw exists only in VRAM, and only where the promotion pass put it.

**S is reachable without materialising.** The archive on disk is the same
archive; it is not expanded to a scratch file first. That is the "one
representation end to end" claim (G4) and the thing that separates this from
every offload path that writes a temp file.

## Placement

Fill by time saved per byte of VRAM spent, then promote with what is left.

    density(t) = ( 1/B_fallback − 1/B_t ) / vram_per_byte(t)

Fill greedily in descending density; then, while VRAM remains, move weights
from a slower on-card tier to a faster one whenever the time saved per extra
VRAM byte is positive, best first.

Both passes are still specified, but on this card the second one no longer
fires. **V-coded dominates V-raw on both axes** — 1.07× the bandwidth at
1/ratio of the space — so the fill puts everything in V-coded and there is
never a promotion worth making. The promotion pass stays in the contract
because it is a property of the solver and not of this GPU: it is what stops
the engine compressing a model that already fitted whenever the decoder is
slower than raw VRAM, which was true here until `fused/` and is true of any
weaker decoder, any slower GPU, or any format whose ratio is near 1.

**Consequence, stated plainly:** every byte of VRAM should hold coded weights,
whether or not VRAM is the binding constraint. When it was 0.51× this was a
claim about scarcity; now it is unconditional for BF16, and it is the opposite
of what every current offload system does.

## The two scheduling properties

Placement is arithmetic. These two are the engineering, and between them they
are worth more than the codec.

**Depth, not demand.** Demand paging — fault, read, resume — is a queue of one
and gets 0.34 GB/s where the disk gives 6.34 (`MEASURED.md`, 18.6×). The
engine must know the next blocks before it needs them and keep 32–64 reads
outstanding. Transformer decode makes this easy in the dense case: layer order
is known, so the fetch for layer *n+1* is issued when layer *n* starts. It is
hard in exactly one place, and that place is MoE.

**Overlap, not bandwidth.** Four streams do not move more bytes than one
(`MEASURED.md`); they exist so that a transfer runs under a kernel. The engine
issues H2D on a copy stream, decode on a compute stream, and the matrix work
on a third, with events between them — double-buffered per layer, so the
staging buffers are two layers' worth and not the model's.

## Eviction

LRU over blocks, with two exceptions that matter more than the policy:

- **Pinned by class.** Embeddings, the LM head, layer norms and anything under
  a block in size are pinned to V-raw. The reason is granularity, not speed —
  nothing smaller than a 64 KiB block can be coded — and now that V-coded is
  the faster tier, this rule costs a little rather than saving anything. It
  stays because they are small and touched every token, so paging them is
  pure loss either way.
- **Never evict to a slower tier than the archive.** Blocks are clean — they
  came from a read-only archive — so eviction is a free-list operation, never
  a write-back. Nothing in the *inference* weight path is ever dirty; in
  training, everything the optimizer touches is.

KV cache is **not** in this design. It is written every token, so it needs an
encoder rather than a decoder, and lmz has no GPU encoder. Offloading KV is a
separate problem with a separate answer and it is out of scope until the
weight path is measured.

## MoE, which is the case this is actually for

The 120B-A5B row in `tiers.py` is the one where offload already works: 60 GB
of footprint, 2.5 GB touched per token, 14.6 tok/s on a 16 GB card against
1.0 for a dense 70B. Experts are the natural block set and expert swap at page
granularity is what blueprint M3 exists for.

It is also where "depth, not demand" gets hard: routing for layer *n+1* is not
known until layer *n* has run. Three answers, in increasing order of
ambition, and the engine should be built so the choice is a policy and not an
architecture:

1. **Cache by frequency.** Expert use is not uniform; keep the hot ones in
   V-coded and take the miss on the rest. `tiers.py` assumes uniform routing,
   which understates this.
2. **Speculate from the router.** The gate's logits for layer *n+1* can be
   approximated from layer *n*'s hidden state; prefetch the top-k of that
   guess and eat a wrong one as a demand read.
3. **Prefetch on the previous token.** Expert selection is highly correlated
   between adjacent tokens; last token's experts are a good prior for this
   one's, and they are known a whole token ahead.

None of these is measured here. The engine's job is to keep the queue deep
enough that they can be tried.

## What this contract must prove — G4 at PC scale

Against stock llama.cpp and stock vLLM on the same model and machine, per
`docs/simulation.md` S7:

1. **Cold start** — archive to first token, including the conversion the
   conventional path needs and excluding it.
2. **Swap** — model A → B → A wall time; expert swap on an MoE.
3. **Resident footprint** — RSS + VRAM for the same model, coded against
   materialised.
4. **Sustained tok/s** against the `tiers.py` prediction, per tier, with the
   placement the solver chose — and the prediction is falsified if the ratio
   between tiers does not track the ladder.
5. **Bit-exactness** — the same prompt, plug-in on and off, identical tokens.
   Lossless is the whole claim of phase 1 and it is cheap to check.
