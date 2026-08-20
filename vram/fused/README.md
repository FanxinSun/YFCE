# fused — decoding into the matrix unit

`tiers.py` has a tier called **V-coded**: coded weights held in VRAM and
decoded *on the way to the matrix units*, so the only DRAM traffic is the
coded read. Its bandwidth column was an estimate taken from lmz's standalone
decoder, and `MEASURED.md` said so:

> What is not measured anywhere is a decoder sharing SMs with a GEMV, which
> is what the fused VRAM tier in `tiers.py` assumes, and it is the first
> thing this project has to measure.

This is that kernel and that measurement. One CUDA file, `fused_gemm.cu`.

## The result

**Coded weights in VRAM are now faster than raw weights in VRAM, and take
two thirds of the space.** An rANS decoder feeding `mma.sync.m16n8k16`
directly delivers 948 GB/s of BF16 against 887 GB/s for the same kernel
reading already-decoded weights — while holding 182 MB where the raw weights
need 268 MB. Real BF16 weights from a checkpoint, all resident in VRAM,
`M=8`, median of five runs:

| | ms | GB/s of BF16 | |
|---|---|---|---|
| raw BF16 in VRAM → mma (control) | 0.302 | 887 | 92% of this card's 960 GB/s peak |
| decode → VRAM (what lmz does today) | 0.599 | 448 | |
| decode → VRAM, then mma (two passes) | 0.938 | 286 | what an unfused engine must do |
| decode → shared, mma removed | 0.280 | 960 | the decoder with nothing to feed |
| **decode → shared → mma (fused)** | **0.283** | **948** | |

| | VRAM held | GB/s of BF16 | relative |
|---|---|---|---|
| raw BF16 in VRAM | 268.4 MB | 887 | 1.00× |
| coded in VRAM, fused decode | 182.3 MB | 948 | **1.07×** |

So the matrix unit costs the decoder 1%, and there is no longer a trade to
make: for weights read a token at a time, **holding BF16 raw on the card is
strictly worse than holding it coded** — slower *and* half again as large.
That inverts the placement rule in `../spec/residency.md`, which was written
for a tier that cost 0.51× of raw speed to buy its space.

The 0.51× was real. It is what this kernel measured when it was first
written, and it is what `tiers.py` predicted. What follows is the 2.05× that
closed it, which came from three changes and none of them from the codec.

## The three changes

Measured together in one sweep, each row paired with a control measured
beside it — this card's clocks drift over a sweep this long, and only the
ratio survives that:

| lanes/stream | states/lane | states/stream | fused GB/s | raw GB/s | rel |
|---|---|---|---|---|---|
| 8 | 1 | 8 — *lmz's shape* | 465 | 874 | 0.53× |
| 8 | 4 | 32 | 648 | 874 | 0.74× |
| 32 | 1 | 32 | 857 | 888 | 0.97× |
| **32** | **4** | **128** | **948** | **882** | **1.07×** |

**Lanes per stream, 8 → 32: the occupancy one, and the largest.** lmz's
format interleaves 8 rANS states in a stream, so 8 lanes share one stream and
a warp carries four streams at once — four A-tiles and four ring buffers,
4 KB of shared memory per warp, which caps the SM at 16 warps. Widening a
stream to 32 states makes one warp one stream: one tile, one ring, 1 KB per
warp. Same decode work per lane, twice the warps. Worth **1.84×**.

**States per lane, 1 → 4: the latency one.** A rANS state is a strictly
serial chain — state indexes a table, the table decides whether to
renormalise, renormalisation makes the next state — and the SM spends it
waiting. Four states per lane is four independent chains interleaved in one
instruction stream, and because they are states of the *same* stream they
share the tile, the ring and the input cursor, so the interleaving costs no
shared memory at all. It costs four ballots instead of one, summed in index
order to keep lmz's single-cursor rule. Worth **1.11×** on top.

**Prefetching the activation fragment: the one that only appeared at the
end.** With the decoder at 950 GB/s a single global load on the critical path
is expensive, and the `mma`'s B fragment was one — fetched after the tile was
decoded, then waited on. Its address depends only on *k*, never on the
decoder, so hoisting it above the decode gives it a whole tile to arrive in.
This is visible as the gap between the last two rows of the first table:
before, the decoder ran at 940 GB/s and the fused kernel at 818, so the
matrix unit cost **13%**; after, 960 and 948, so it costs **1%**.

Occupancy is still what everything rests on:

| warps/block | shared KB | ms | GB/s of BF16 |
|---|---|---|---|
| 1 | 17.5 | 0.694 | 387 |
| 2 | 19.0 | 0.527 | 510 |
| 4 | 22.0 | 0.344 | 780 |
| 8 | 28.0 | 0.309 | 868 |
| **16** | **40.0** | **0.287** | **935** |

The 16 KB rANS table is per *block* and the SM has 100 KB to hand out, so one
warp per block puts 20 KB behind a single warp. 2.4× for a launch parameter.

## What it costs the format

128 states per stream means 512 bytes of initial state at the head of each
stream. Amortised over a stream of 32768 elements it is 0.6% of the ratio:
**1.473× against 1.482×** for the same data at 32 states, and against lmz's
published **1.485×**, which pays no per-stream header at all. Nothing else
about the coding changes — a static table costs `-log2(p)` per symbol
wherever the symbol sits, so permuting the symbol order into tile order is
free, and `enc_shared_n` at 8 states is checked byte-identical against lmz's
own `lmzx_encode_shared` on every run.

The stream that won is **32768 BF16 elements — exactly the 64 KiB block**
that `../spec/residency.md` already fixes as the unit of storage, DMA,
eviction and prefetch. That was not arranged; longer streams amortise the
state header and the sweep picked it.

## What is actually fused

The A fragment of a tensor-core `mma` is 16×16 BF16 held across a warp's 32
lanes. In this kernel that fragment is produced by an entropy decoder running
in the same warp a handful of instructions earlier, and it never exists in
DRAM. The chain per tile is:

    coded bytes in VRAM
      -> cp.async ring in shared        (4-slot, lmz's V5 arrangement)
      -> 128 rANS states in registers, four per lane
      -> 8 exponent bytes packed in a register
      -> merged with 8 raw sign+mantissa bytes -> 8 BF16 in one register
      -> one uint4 store to a 16x16 shared tile
      -> ldmatrix -> mma -> accumulator

The exponent plane is never written to memory in any form: it is decoded into
a register and merged out of one. That is what makes this different from
`lmz/scratchpad/gpu/cuda/gpu_fused.cu`, which is the same decoder writing
BF16 to VRAM for somebody else to read back — measured here at 448 GB/s, and
the fused path is **2.1× faster than it** for the simple reason that it never
makes the 268 MB of stores.

Two layout choices make it work, and neither touches the coder:

**Tile order.** A stream's element order is defined as consecutive 16×16
A-tiles, row-major, k-major within a row-block. A warp decoding 256 symbols
produces exactly one A-tile, already in the order `ldmatrix` wants. No
transpose, no gather.

**Lane order.** Within a tile the order is permuted once more so that a
lane's eight *consecutive* symbols are eight *consecutive* weights. That
removes the staging buffer entirely — before this change the 256 exponent
bytes went out to shared memory and came back, which cost 7% and 1 KB of
shared per warp.

## What it checks

Every run verifies four things before reporting a number, and every number
here is from a run that passed all four:

1. **The generic coder is lmz's coder.** `enc_shared_n` takes the state count
   as a parameter; at 8 states it must emit bytes identical to the
   submodule's own `lmzx_encode_shared`, which is checked on real data.
2. **The decode is byte-identical to the checkpoint** over the whole 268 MB —
   not a sample, and not against a reference derived from the planes. This is
   checked for *every* row of the format sweep, not just the winner, because
   a format that decodes wrongly can easily decode fast.
3. **Fused output is bit-identical to the control kernel's**, which shares
   every instruction from the A fragment onward. Same accumulation order, so
   this is `memcmp`, not a tolerance.
4. **Both match an fp64 reference** computed on the host over a slice, to
   2–9e-6 relative — so a bug shared by both kernels cannot hide.

## Three shapes

| tensor | shape | MB | ratio | fused | raw | rel |
|---|---|---|---|---|---|---|
| `mlp.gate_proj` ×8 | [65536 × 2048] | 268 | 1.473× | 948 | 887 | **1.07×** |
| `mlp.down_proj` ×12 | [24576 × 8192] | 403 | 1.473× | 930 | 894 | **1.04×** |
| `self_attn.q_proj` ×12 | [24576 × 2048] | 101 | 1.472× | 828 | 860 | 0.96× |

The third is 101 MB, only 1.5× this card's L2, so its control is partly
cache-fed; read it as the weakest of the three.

One structural limit showed up here and is worth stating: a warp owns 16 rows,
so the grid is `N / (16 × warps_per_block)` blocks, and the same `down_proj`
stacked 8 layers deep instead of 12 gives 64 blocks on 84 SMs and drops to
0.90×. **The kernel needs at least one block per SM**, which for 16 warps per
block means `N ≥ 16 × 16 × 84 ≈ 21500` rows. Below that, either fewer warps
per block or a split-k reduction, and neither is written.

## Batch

The coded tier is for weights read a token at a time, and that is the only
place this wins:

| M | raw ms | fused ms | cuBLAS ms | fused/raw |
|---|---|---|---|---|
| 8 | 0.309 | 0.284 | 0.342 | **0.92×** |
| 16 | 0.332 | 0.341 | 0.314 | **1.03×** |
| 32 | 0.344 | 0.432 | 0.329 | 1.27× |
| 64 | 0.457 | 0.707 | 0.331 | 1.55× |
| 128 | 0.681 | 0.883 | 0.360 | 1.30× |

The crossover is around M=16. Below it both kernels are bound by DRAM and the
coded one moves 32% fewer bytes; above it the raw kernel is bound by the
matrix unit and the decode is pure addition. Read the large-M rows narrowly:
the control is a *fair* control — the fused kernel with the decoder removed —
and at M≤16 it is on the memory roofline, where it beats cuBLAS because a
skinny GEMM is not what a GEMM library is tiled for. It is not a competitive
GEMM at large M, where cuBLAS is 2× it because it tiles the batch properly and
this kernel holds the batch in accumulator registers.

## What is left on the table

The fused kernel moves **640 GB/s** of coded bytes against **889 GB/s** the
card will give it, so it is still decode-bound with 1.4× of the memory system
unused. A decoder 1.4× faster than this one would make the tier DRAM-bound at
about 1300 GB/s of BF16, or **1.46× raw VRAM**, and at that point the only
reason to hold a raw weight on this card would be a batch.

Two things tried and rejected, both measured: preloading the renormalisation
window into registers and distributing it by `__shfl` takes a dependent
shared load out of the chain and is worth 12% at one state per lane, but the
extra registers cost a block of occupancy and it loses 15% at four; and
telling ptxas the occupancy target to win that back just moves the cost to
spills.

## Build and run

    /usr/local/cuda-13.2/bin/nvcc -O3 -arch=sm_120 -t 12 -lcublas \
        -Xcompiler -fopenmp -lgomp -o fused_gemm fused_gemm.cu
    ./fused_gemm                                     # sweep, 8x gate_proj
    ./fused_gemm "model.layers.%d.mlp.down_proj.weight" 12
    ./fused_gemm "…gate_proj.weight" 8 <safetensors> 32,4,2048   # pin the format

The tensor argument is a `printf` format taking the layer index and the count
is how many layers to stack; the fourth argument pins `lanes,states,KS` and
skips the sweep. **The stack exists so the working set misses L2, and that
matters more than anything else in the harness:** one 33.6 MB `gate_proj` sits
entirely inside the 5080's 64 MB L2, where the raw control reads at 1500 GB/s
and the whole comparison is fiction.

Conditions: RTX 5080 (sm_120, 84 SMs, 16 GB GDDR7), CUDA 13.2, driver 595,
Ubuntu 26.04 under WSL2 — the box in `../MEASURED.md`. Each time is the best
of seven `cudaEvent` measurements, alternated with its control; run to run the
spread is about 3%, which is a consumer card driving a desktop with no way to
lock clocks. Every ratio here reproduced across five runs; the third
significant figure did not. Pin the format before quoting a number — the
sweep is 25 configurations over 268 MB each and leaves the card warm.

## What this does not measure

- **Anything off the card.** Every tier here is VRAM. The H and S tiers are
  where compression was always unambiguously free — the decoder now outruns
  PCIe by 33× — and they need a residency engine, not a kernel.
- **A whole model.** One layer shape at a time, weights already resident. No
  layer scheduling, no prefetch, no eviction.
- **Anything but BF16.** The formats people actually run — Q8_0, Q4_K_M —
  compress by 5–7%, and 1.06× of capacity is a much weaker reason to hold a
  coded tier than 1.47× is, even now that it is free. That is
  `../README.md`'s argument and this does not overturn it; what it overturns
  is the belief that the coded tier costs speed.
- **The encoder.** Read-only weights only. Training rewrites state every step
  and lmz has no GPU encoder.
