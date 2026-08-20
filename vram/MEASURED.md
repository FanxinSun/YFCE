# Measured

Every number the charter and `tiers.py` rest on, with the conditions that
produced it. Two probes, both in `probe/`, both building against nothing that
is not already on the machine.

## The box

Ryzen 7 9800X3D (8C/16T), 24 GB to WSL2 of 31 GB host, RTX 5080 (sm_120,
84 SMs, 16 GB GDDR7, 256-bit, 960 GB/s peak), CUDA 13.2, driver 595, Ubuntu
26.04 under WSL2, ext4 on a Hyper-V VHDX over an NVMe SSD.

**The slot reports PCIe 4.0 ×16**, not 5.0 — `pcie.link.gen.max` is 4 —
so the link figures below are a Gen4 machine's, and a Gen5 host would roughly
double them. That does not weaken the argument; it moves it. Every ratio in
`tiers.py` that says *compression is free here* gets **harder** to satisfy on
a faster link, and at Gen5 the decoder still clears it by 7×.

## Link and VRAM — `probe/bw.cu`

    nvcc -O3 -arch=sm_120 -o bw vram/probe/bw.cu && ./bw

Best of five 1 GiB transfers, four consecutive runs.

| | GB/s | |
|---|---|---|
| VRAM copy, read+write counted | **776** | 768–785; 81% of the 960 peak |
| H2D, page-locked host memory | **28.8** | ±0.1, run to run |
| H2D, ordinary `malloc` memory | **16.0** | 13.9–16.9 — the driver's staging buffer, and it is noisy |
| D2H, page-locked | 28.6 | |
| H2D, page-locked, 4 streams × 64 MiB | 28.8 | *identical to one copy* |

Two things worth keeping.

**Page-locking is worth 1.8× and costs nothing.** A `cudaMemcpy` from
pageable memory bounces through a driver staging buffer. Any offload path
that hands the driver a plain `malloc` — which is what a framework's default
CPU tensor is — leaves that on the floor before any codec is involved.

**Streams are not for bandwidth.** One large copy already saturates the link;
four streams over the same gigabyte deliver the same 28.8 GB/s. Streams are
how a transfer overlaps *compute*, which is what a residency engine needs
them for, and it is a different claim from "more streams, more bytes".

## Compute — `probe/gemm.cu`

    nvcc -O3 -arch=sm_120 -lcublas -o gemm vram/probe/gemm.cu && ./gemm

| square GEMM | TFLOP/s |
|---|---|
| 4096³ | 105.6 – 114.6 |
| 8192³ | 113.0 – **119.8** |
| 16384³ | 114.0 – 118.3 |

Two runs minutes apart, and the spread is clocks, not method. `train.py` uses
the **high** end, which is the conservative choice: a faster GPU needs *more*
tokens per microbatch before the offload hides, so planning against 119 and
getting 114 leaves the schedule intact.

BF16 in, **FP32 accumulate**, because that is what mixed-precision training
does. Consumer Blackwell runs FP16-accumulate at roughly twice this, and
quoting that figure would halve every token floor in `train.py` — plans that
would then not hold.

This is the number the training model turns on. The token count at which
streamed weights disappear under compute is `FLOPS / bandwidth`, so on this
box it is 119e12 / 28.8e9 = **4132 FLOP per byte**, and a microbatch has to be
big enough to supply that.

## Storage — `probe/io.c`

    cc -O2 -pthread -o io vram/probe/io.c && ./io ~/scratch.tmp

`O_DIRECT`, so no page cache and no root. 4 GiB file, 64 KiB random reads —
lmz's block size, and the granularity an expert or an adapter is paged at.

| | GB/s | IOPS | µs/read |
|---|---|---|---|
| sequential, 4 MiB, one thread | 3.59 | | |
| random 64 KiB, QD=1 | **0.34** | 5,243 | 191 |
| random 64 KiB, QD=2 | 0.65 | 9,918 | 202 |
| random 64 KiB, QD=4 | 1.17 | 17,820 | 225 |
| random 64 KiB, QD=8 | 2.22 | 33,939 | 236 |
| random 64 KiB, QD=16 | 3.88 | 59,216 | 270 |
| random 64 KiB, QD=32 | 5.35 | 81,684 | 392 |
| random 64 KiB, QD=64 | **6.34** | 96,677 | 662 |

**This is the largest single lever in the whole project and it has nothing to
do with compression.** Demand paging — fault, read, resume — is a queue of
one, and a queue of one gets 5% of what the disk can do. At depth 64 the same
64 KiB random reads beat a single-threaded *sequential* 4 MiB loop by 1.8×.
The access pattern was never the problem; the depth was.

The corollary matters for the format: 64 KiB random is not a penalty here, so
lmz's page-aligned block structure costs nothing to page against, and there is
no reason to lay weights out sequentially to please the disk.

The 191 µs at QD=1 is a VHDX under WSL2, not bare metal — a native NVMe would
be nearer 80–100. Read these as relative, per `docs/simulation.md`'s rule for
storage on this box. The *shape* — flat latency to QD=16, then queueing — is
the disk's, not the hypervisor's.

## Decode — lmz's own GPU experiment, same GPU

Not measured here; taken from `lmz/scratchpad/gpu/README.md`, which ran on
this card and verified every kernel byte-identical to lmz's CPU decoder over
936 MB and 1.87 GB of real Qwen2.5 planes.

| | GB/s |
|---|---|
| exponent plane, 8 lanes/stream, `cp.async` 4-slot | **418** |
| fused whole-BF16 (exponent rANS + raw sign/mantissa + merge) | **399** |
| lmz on this CPU, all cores, free-threaded | 2.21 |
| lmz on this CPU, one thread | 0.48 |

Those are standalone kernels decoding into VRAM. The next section is the one
that was missing.

## Fused decode into the matrix unit — `fused/fused_gemm.cu`

    /usr/local/cuda-13.2/bin/nvcc -O3 -arch=sm_120 -t 12 -lcublas \
        -Xcompiler -fopenmp -lgomp -o fused_gemm vram/fused/fused_gemm.cu

The V-coded tier assumes coded weights are decoded *on the way to* the matrix
units, so the only DRAM traffic is the coded read. That kernel now exists.
lmz's coder, unmodified in behaviour and checked byte-identical against the
submodule's own encoder on every run; the weights re-laid-out so a warp
decoding 256 symbols produces exactly one 16×16 `mma` A-tile in the order
`ldmatrix` wants. 268 MB of real BF16 weights (8 layers of `mlp.gate_proj`
from the same checkpoint), all resident in VRAM, M=8, median of five runs:

| | ms | GB/s of BF16 |
|---|---|---|
| raw BF16 in VRAM → mma — the control | 0.302 | 887 |
| decode → VRAM (the standalone kernel above, this layout) | 0.599 | 448 |
| decode → VRAM, then mma — two passes | 0.938 | 286 |
| decode → shared, mma removed | 0.280 | 960 |
| **decode → shared → mma — fused** | **0.283** | **948** |

**The matrix unit costs the decoder 1%, and the coded tier is now faster than
the raw one.** 948 GB/s of BF16 out of 182.3 MB of VRAM against 887 GB/s out
of 268.4 MB — **1.07× the speed at 0.68× the footprint**. For weights read a
token at a time there is no longer a trade: on this card, holding BF16 raw in
VRAM is strictly worse than holding it coded.

That is not where this started. The first working version of the kernel
measured **0.51×**, which is exactly what `tiers.py` assumed, and the 2.05×
that closed the gap came from three things, none of them the codec:

| lanes/stream | states/lane | fused GB/s | rel to raw |
|---|---|---|---|
| 8 | 1 — lmz's shape | 465 | 0.53× |
| 8 | 4 | 648 | 0.74× |
| 32 | 1 | 857 | 0.97× |
| **32** | **4** | **948** | **1.07×** |

- **Lanes per stream, 8 → 32 (1.84×).** With 8 states per stream a warp
  carries four streams, so four A-tiles and four ring buffers — 4 KB of shared
  per warp, capping the SM at 16 warps. One warp per stream is 1 KB.
- **States per lane, 1 → 4 (1.11×).** A rANS state is a serial chain; four
  interleaved chains per lane share the tile, the ring and the cursor, so the
  interleaving is free in shared memory. Four ballots instead of one.
- **Prefetching the `mma` B fragment.** At 950 GB/s one global load on the
  critical path costs 13%; its address depends only on k, so hoisting it above
  the decode gives it a tile to arrive in. That is what took the matrix unit
  from 13% back to 1%.

The winning stream is **32768 BF16 elements — exactly the 64 KiB block**
`spec/residency.md` already fixes as the unit. 128 states per stream costs
512 bytes of header, which is **0.6% of ratio: 1.473× against lmz's 1.485×**.

Three numbers above this section move:

- **VRAM read-only is 887 GB/s, not 776.** The copy probe counts a read and a
  write; a weight is only read. `tiers.py`'s on-card tiers were using the
  wrong one.
- **The fused decoder is 948, not 399.** The same code writing BF16 to VRAM
  gets 448 in this layout, and the 2.1× on top of it is the 268 MB of stores
  that never happen.
- **The two-pass alternative is 286**, not the ~330 a pure-DRAM model
  predicts: the decode pass is decode-bound before its write is counted, so
  the terms add rather than one hiding the other.

Three conditions matter more than the second significant figure. **The
working set must miss L2** — one 33.6 MB weight matrix sits entirely inside
the 5080's 64 MB L2, where the raw control reads at 1500 GB/s and the
comparison is fiction; the harness stacks layers to get past it. **Every
measurement is paired with a control taken beside it**, because clocks drift
several percent over a sweep this long and only the ratio survives. And
**the kernel needs at least one block per SM** — a warp owns 16 rows, so
below about 21,500 rows it starves and the same shape drops to 0.90×.

The fused tier moves **640 GB/s** of coded bytes against the 889 the card
will give it, so it is still decode-bound with 1.4× of the memory system
unused. Full write-up, including two optimisations that were measured and
rejected, in `fused/README.md`.

## Overlap — `probe/overlap.cu`

    nvcc -O3 -arch=sm_120 -lcublas -o overlap vram/probe/overlap.cu && ./overlap

The claim the whole design rests on: a step costs `max(compute, traffic)`, not
`compute + traffic`. One layer of 8-bit Adam state (800 MB moved) streaming
host ↔ device against a cuBLAS GEMM chain. Warmed to steady clocks, all
configurations interleaved within each round, median of 15.

| | ms |
|---|---|
| GEMM alone | 87 |
| optimizer alone (DMA + elementwise kernel) | 33 |
| **measured together** | **114** |
| max — traffic hides | 87 |
| sum — traffic does not hide | 120 |

**About 20% of the traffic hides. Not 100%.** `max(...)` is wrong and
`compute + 0.8 × traffic` is close to right, which is worth roughly 2× on
every token floor in `train.py`.

The control says why, and it rules out the obvious suspects. The same DMA
against a kernel that touches **no memory at all** overlaps ~**100%** — so the
copy engine works, the streams work, and WSL2's GPU virtualisation is not
interfering. What fails is specific to the GEMM: the elementwise Adam kernel
costs 1.4 ms standalone and up to 20 ms when co-resident, and the DMA costs
close to its full standalone time. **Streaming traffic evicts the tiles cuBLAS
reuses out of L2.**

That is a named problem with a known toolbox — CUDA's L2 access-policy windows
(`cudaAccessPropertyStreaming`) exist precisely to keep streaming data from
evicting persistent data — and nobody has tried it here. Until someone does,
20% is the number to plan with.

**Sustained ≠ burst, and it matters more than it looks.** GEMM alone reads 35–60
TFLOP/s in this harness against the 114–120 that `gemm.cu` reports, because
this one warms to steady state first and that one measures short bursts from
cold clocks. Training is sustained by definition, so `train.py` uses 38. It
cuts the opposite way from the overlap penalty: a slower GPU needs *fewer*
tokens to cover the same transfer, and the two corrections nearly cancel.

**What is not trustworthy here.** The GEMM baseline has 22–37% spread run to
run on this box, which is a consumer card driving a desktop under WSL2 with no
way to lock clocks. The direction and rough size of every effect above
reproduced across five independent runs; the second significant figure did
not. Anything that turns on precision needs a headless GPU with fixed clocks.

## Ratios — lmz's measured results

`lmz/docs/results.md`, real checkpoints, every round-trip verified
byte-identical.

| stored as | lossless ratio | saved |
|---|---|---|
| BF16 | 1.485× | 34.7% |
| FP8 | 1.207× | 17.1% |
| Q8_0 | 1.072× | 6.7% |
| Q4_K_M | 1.054× | 5.1% |

The last two rows are the ones that decide whether this project is worth
anything, and they are the small ones. See the README.
