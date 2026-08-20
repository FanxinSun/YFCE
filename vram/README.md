# vram — SSD and RAM as VRAM

A runtime that trains a model too large for the GPU it is on, by placing
weights, gradients and optimizer state across VRAM, host RAM and SSD, and
scheduling the traffic so that it disappears underneath the compute. CUDA and
Metal. Pure software, and nothing about the arithmetic of the model changes.

The target is **full fine-tuning and continued pretraining** — the case where
the optimizer state, not the weights, is what does not fit. Adapters are the
easy corner of the same machinery and come free; they are not the point.

This is `docs/blueprint.md` M3 — the residency manager — on hardware somebody
else built, which under D14 is the base case and not a detour.

## The one ratio

Streaming a weight costs `bytes / bandwidth`. The compute it feeds is
`2 × tokens_per_microbatch` FLOPs per byte. So the traffic hides once

    2 × tokens_per_microbatch  ≥  FLOPS / bandwidth

On this box: **38 TFLOP/s** of *sustained* BF16 (not the 119 a cold burst
reports) against a **28.8 GB/s** measured PCIe link. Naively that is ~700
tokens per microbatch — but two measured corrections apply, and the honest
answer is **~4,750 tokens per microbatch for 90% utilisation**, which is a
4096-token sequence and change.

The corrections matter more than the headline. **Only ~20% of streamed traffic
actually hides under a real GEMM** (`probe/overlap.cu`) — streaming DMA evicts
the tiles cuBLAS reuses out of L2, so `max(compute, traffic)` is wrong and
`compute + 0.8 × traffic` is close to right. And sustained GEMM is a third of
burst GEMM, which pushes the other way. They nearly cancel; what survives is
that there is **no point at which the traffic becomes free**, only a point at
which it costs less than a tenth of the step.

Inference decode reads the same weights to do one multiply-add each: **1 FLOP
per byte** of BF16. Against the same 4132 required, it is short by a factor of
four thousand, and no scheduling recovers it.

**That factor of 4000 is the whole reason this project is about training.**
The identical offload that is hopeless one side of it is nearly free on the
other.

## Where the memory actually is

`python3 vram/train.py`. GB, and the interesting column is not the first.

| model | method | weights | optimizer state | activations @2k | total |
|---|---|---|---|---|---|
| Llama-3.1-8B | full, fp32 Adam | 16.1 | **112.4** | 0.8 | 129.3 |
| Llama-3.1-8B | full, 8-bit Adam | 16.1 | 64.2 | 0.8 | 81.1 |
| Llama-3.1-8B | full, bf16 Adam | 16.1 | 32.1 | 0.8 | 49.0 |
| Llama-3.1-70B | full, 8-bit Adam | 141.2 | 564.8 | 3.2 | 709.2 |
| Llama-3.1-70B | QLoRA | 38.8 | 1.4 | 3.2 | 43.5 |

Inference's entire problem — the weights — is training's small column. An 8B
full fine-tune is **eight times the card** before a single activation is
allocated, and seven eighths of that is Adam.

Which sets the first lever, and it is not offload at all: **fp32 Adam → 8-bit
moments → bf16 with stochastic rounding is 3.5× off the state** (14 → 8 → 4
bytes per parameter) before anything crosses a bus. A runtime that does not
own the optimizer cannot pull that lever, which is one of two reasons this
has to be a runtime rather than a planner emitting configs.

## The token floors

Tokens needed before the traffic is fully hidden. Below these the GPU waits;
above them the offload is free.

| device | method | per microbatch | per optimizer step |
|---|---|---|---|
| RTX 5080, PCIe 4 | full, fp32 Adam | 4,750 | 35,625 |
| RTX 5080, PCIe 4 | full, 8-bit Adam | 4,750 | 21,375 |
| RTX 5080, PCIe 4 | full, bf16 Adam | 4,750 | 11,875 |
| RTX 5080, PCIe 5 | full, 8-bit Adam | 2,487 | 11,193 |
| M4 Max unified | full, 8-bit Adam | 24,480 (disk) | 110,160 |
| M4 Pro unified | full, 8-bit Adam | 12,240 (disk) | 55,080 |

These are floors for **90% utilisation**, not for break-even, and they carry
the 20% overlap figure — which was measured on CUDA and merely *assumed* for
Metal. The Metal rows are the least trustworthy numbers in this repository.

Per-microbatch is weights read for the forward and again for the backward.
Per-step is gradients, moments and the weight write-back, over the link if the
state fits in host RAM and **over the disk if it does not** — and for an 8B it
does not: 64 GB of 8-bit Adam state against 24 GB of host RAM puts it on NVMe,
where the floor rises from 9,297 to **42,232 tokens per step**.

That number is reachable — 2048 tokens × 21 accumulation steps — and it is the
project's central tension. Bigger steps hide more traffic; bigger microbatches
cost activation memory. Solving that trade is the planner's whole job.

## What the runtime owns, and what it must not

The single decision that makes this buildable. "Full runtime on both backends"
means owning **memory and the optimizer step**, not autodiff and not kernels.

| owned here | borrowed |
|---|---|
| the tiered parameter store, block-granular | forward and backward graphs |
| placement: what lives in VRAM, host, disk | GEMM, attention, norm kernels |
| prefetch, queue depth, copy/compute overlap | the tokenizer, data loading, the training loop's outer shell |
| the **fused streaming optimizer step** | autodiff itself |
| gradient accumulation buffers and where they live | |
| the planner that chooses all of the above | |

Autodiff and kernels come from PyTorch (CUDA and MPS) or MLX. Writing a
transformer's backward pass twice, once per backend, is a different and much
longer project, and one nobody needs.

**The fused streaming optimizer step is the piece with no substitute.** Adam
is elementwise, so parameters can be streamed through it in blocks: read
`(grad, master, m, v)`, update, write back `(master, m, v, bf16 weight)`. The
working set is one window, not the whole model — which is what lets 700 GB of
state live on a disk. And because the step has essentially zero arithmetic
intensity it can never hide behind its own compute, so it must be fired **per
layer, the moment that layer's gradients are final in the last microbatch**,
and hidden under the backward pass still running below it. Get that wrong and
every number above is fiction.

## Two backends, two different machines

| | discrete GPU | unified memory |
|---|---|---|
| tiers | VRAM ‖ PCIe ‖ host RAM ‖ NVMe | one pool ‖ NVMe |
| the scarce link | PCIe, 28.8 GB/s measured | there isn't one |
| what spills | anything past ~13 GB | anything past total RAM |
| host RAM as a tier | real, and 4.5× the disk | **does not exist as a distinct tier** |
| contention | GPU and CPU have separate memory | they share the same bandwidth |

The abstraction has to hide four tiers on one side and two on the other, and
the second is not a degenerate case of the first — on unified memory the
host-RAM tier vanishes entirely and the disk moves up to be the *only* place
to spill. A planner that models Metal as "CUDA with a very fast link" will
choose wrong every time.

## What this competes with, stated plainly

On CUDA, **DeepSpeed ZeRO-Infinity already does this and does it well.** So do
FSDP with `cpu_offload`, bitsandbytes' paged optimizers, and Unsloth's kernels.
Anyone starting here should assume the CUDA half is a re-implementation that
has to justify itself on ergonomics or on the planner, not on being first.

Off CUDA there is nothing. MLX trains on Apple silicon with no offload story;
PyTorch MPS has no ZeRO equivalent; llama.cpp fine-tunes but not at this
scale. **A 64 GB Mac can hold a 70B QLoRA today and cannot full-fine-tune an
8B, and no tool exists to close that** — the state would sit on the SSD, the
floor is 15,300 tokens per step, and the arithmetic says it works.

That is the gap, and it argues for building the planner portable and the
runtime Metal-first, with CUDA as the backend that validates it against a
mature incumbent.

## Where the codec fits, which is smaller than it looked

Compression does not change the memory footprint of anything resident and it
does not change the FLOPs. What it does is **lower the token floor**, because
the floor is `FLOPS / bandwidth` and a codec multiplies the bandwidth:

- 70B QLoRA streamed off NVMe: floor drops 3,441 → 2,294 tokens/microbatch at
  1.5×, which is activation memory handed back.
- 8-bit Adam state on NVMe: lmz measures **26.1%** on real 8-bit AdamW state,
  so the 42,232-token step floor becomes ~31,300.

Useful, bounded, and secondary. Two caveats keep it honest: the state is
*written* every step, so this needs a GPU **encoder**, which lmz does not have
— the decoder now runs at 948 GB/s straight into the tensor cores (`fused/`)
and there is still no counterpart going the other way. And gradients are
close to incompressible losslessly, so the lever does not reach the largest
per-step term at all.

## The inference half, which is where this started

The same tier machinery under inference is a different and much worse story,
and it is written up because it was measured rather than because it is the
plan: `tiers.py`, and the ladder in `MEASURED.md`. The short version is that
decode's 1 FLOP/byte cannot hide any offload, so the only lever is fewer bytes
on the link, and lossless coding delivers 6% on the quantised formats people
actually run. It produced one result worth keeping — a **coded tier inside
VRAM** — and one reopened decision, in `decisions.md` under Reversed.

That tier was an estimate until `fused/` was written, and the estimate said it
would cost 0.51× of raw VRAM speed to buy 1/ratio of the space. It is now a
kernel — an rANS decoder that produces `mma` A-fragments in registers and
never writes a decoded weight to DRAM — and the answer came out the other way:

**948 GB/s of BF16 feeding the tensor cores, against 887 GB/s for the same
kernel reading weights that were already decoded.** 1.07× the speed at 0.68×
the footprint, byte-identical to the checkpoint and bit-identical to a
raw-weight control. For weights read a token at a time, holding BF16 raw on
this card is now strictly worse than holding it coded. The matrix unit costs
the decoder 1%; what cost the other 2.05× was the shape of the decode —
lanes per stream, states per lane, and one prefetch — and none of it was the
codec. Details in `fused/README.md`, the ladder in `MEASURED.md`.

## What is not known

In the order that can kill it:

1. **Can the overlap be recovered?** Partly answered, and the answer was no:
   ~20% hides, because streaming traffic destroys the GEMM's L2 reuse
   (`MEASURED.md`). The open part is whether CUDA's L2 access-policy windows
   get it back — nobody has tried — and whether Metal behaves the same way.
   A second, untested problem sits behind it: layer *i*'s gradients are final
   only in the last microbatch's backward and its updated weight is needed at
   the start of the next forward, so the window is the backward pass, not the
   step, and gradient accumulation lengthens the step without lengthening the
   window.
2. **Is the activation model right?** `train.py` counts checkpoint boundaries
   plus one layer's working set, and says so. Attention and the MLP
   intermediate make the real figure larger, and every token floor is bought
   with activation memory, so an optimistic activation model silently breaks
   every plan in this document.
3. **Does MPS or MLX expose enough to build this on?** The runtime needs
   explicit control of allocation, streams or command queues, and events. If
   Metal's frameworks will not give up that control, the Metal-first argument
   collapses and with it the reason to build rather than configure DeepSpeed.
4. **Does the NVMe hold up under a mixed read/write step?** All storage
   figures here are reads. The optimizer step writes back as much as it reads,
   and sustained mixed traffic on a consumer SSD is a different number that
   has not been measured.

## Layout

    vram/README.md          this charter
    vram/MEASURED.md        every number with its conditions, and the probes
    vram/train.py           the training roofline: budgets, token floors, the device axis
    vram/tiers.py           the inference roofline: the tier ladder, placement
    vram/probe/gemm.cu      achievable BF16 GEMM, FP32 accumulate
    vram/probe/bw.cu        link and VRAM bandwidth
    vram/probe/io.c         storage, sequential and 64 KiB random over queue depth
    vram/fused/fused_gemm.cu the coded VRAM tier as a kernel: rANS decode into mma
    vram/spec/planner.md    the solver: placement, microbatch, accumulation, devices
    vram/spec/runtime.md    the parameter store, the streaming optimizer, the overlap
    vram/spec/backends.md   CUDA and Metal: two topologies behind one abstraction
    vram/spec/residency.md  the tier contract, read and write
    vram/spec/plugin.md     how the inference half attaches without a fork
    vram/engine/            (phase 1) the runtime
    vram/RESULTS.md         (phase 1) measured against DeepSpeed and against nothing

## Status

Charter, two rooflines and the measurements behind them, 2026-08-20. No
runtime code yet.

Phase 1 is one model, one machine, end to end: an 8B full fine-tune with
8-bit Adam on a 16 GB card, state on NVMe, the per-layer streaming optimizer,
and the measured step time against `train.py`'s prediction. If the prediction
holds, the planner is worth generalising; if it does not, the reason will be
item 1 above and it is better to find that out on one model than on six.

**The name is provisional.** `vram/` is the directory; renaming is one command
and should happen before anything is published.
