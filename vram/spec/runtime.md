# Runtime — the parameter store, the streaming optimizer, the overlap

What executes a plan. Three components and one schedule; everything else is
borrowed from the framework underneath.

The boundary, restated because it is what makes this buildable: the runtime
owns **memory and the optimizer step**. Forward and backward graphs, GEMM,
attention and norm kernels, autodiff, the data loader and the outer training
loop come from PyTorch (CUDA and MPS) or MLX. Writing a transformer's backward
pass twice, once per backend, is a different project.

## 1. The parameter store

Every tensor the model owns, addressed as a range of blocks, with a tier.

    handle = store.open(tensor_name)      -> class, dtype, shape, block range
    store.prefetch(handle, queue)          -> start moving it toward the GPU
    ptr = store.acquire(handle, queue)     -> device pointer, valid on that queue
    store.release(handle)                   -> may be evicted after this
    store.commit(handle, queue)            -> written back to its tier

**Block granularity is 64 KiB, page-aligned**, the same unit throughout, and
it is the right one by measurement rather than by inheritance: 64 KiB random
reads at queue depth 64 reach 6.34 GB/s on this box, beating a single-threaded
sequential 4 MiB loop at 3.59. The access pattern was never the cost; the
depth was.

`commit` is the half that inference did not need. Weights are read-only there
and `residency.md` could say every block was clean; here master weights and
both moments are rewritten every step, so the store has dirty blocks, a
write-back path, and an ordering obligation — a block being written must not
be evicted, and a step is not complete until every commit has landed.

## 2. The fused streaming optimizer step

The piece with no substitute, and the reason a config generator would not have
been enough.

Adam is elementwise. So parameters stream through it in windows rather than
residing:

    for each window w of P parameters:
        read   grad[w], master[w], m[w], v[w]        from wherever they live
        update m, v, master; produce the bf16 weight
        write  master[w], m[w], v[w], weight[w]      back

The working set is one window — a few tens of MB — not the model. **That is
what lets 565 GB of 8-bit Adam state for a 70B live on a disk** while the
update runs on a 16 GB card.

Two properties of that loop decide whether any of this works.

**It has no arithmetic to hide behind.** A few FLOPs per parameter against
~30 bytes of traffic: it is pure bandwidth, and unlike the forward pass it can
never cover its own transfers. It must be hidden under *someone else's*
compute, and there is only one candidate.

**So it fires per layer, during the backward pass.** A layer's gradients are
final the moment its backward completes in the last microbatch of the step.
At that instant its optimizer window is launched, and it runs underneath the
backward passes of the layers below it. Waiting until the backward finishes
and then running the whole update serialises `t_state` after `t_compute`, and
`planner.md`'s `max(...)` becomes a sum.

**This is the assumption most likely to be wrong**, and it fails at a
predictable place: near the top of the network there is little backward left
to hide under. Expect the first and last few layers to be exposed, expect the
planner to need a term for it, and measure it before believing any step time
in this repository.

**Precision is the runtime's to choose**, and it is the largest single lever
before any byte moves: fp32 Adam is 14 bytes per parameter of state, 8-bit
moments are 8, bf16 with stochastic rounding is 4. That is 3.5× off the
dominant term, which is why the optimizer could not be delegated.

## 3. Gradient accumulation

Gradients for a whole 8B are 16 GB in bf16 — they do not fit either, and they
are touched by every microbatch rather than once per step. Three placements,
in the order the planner should try them:

1. **Accumulate in place into fp32 master state.** No separate gradient
   buffer at all; each microbatch's gradient is added straight into a
   resident accumulator. Cheapest in memory, and it forces fp32 accumulation,
   which is what you want anyway.
2. **Per-layer gradient buffers, VRAM-resident, released after use.** Only
   the layers currently in flight hold gradients. Works when a single layer's
   gradient fits, which it does for everything here.
3. **Gradients in host RAM.** Last resort: it doubles the per-microbatch link
   traffic, and unlike weights that traffic is a write.

## The step schedule

    forward   layer i : acquire(w[i]) already prefetched by layer i-2
                        prefetch(w[i+2])
                        release(w[i-1])
    backward  layer i : acquire(w[i]), recompute if checkpointed
                        accumulate grad[i]
                        if last microbatch: launch optimizer window for layer i
                        prefetch(w[i-2])

Three queues — copy-in, compute, copy-out/commit — with events between them,
double-buffered by two layers. The depth is not decoration: **a demand-paged
version of this schedule gets 0.34 GB/s off the same disk that gives 6.34 at
depth 64**, an 18.6× swing that has nothing to do with any codec and is larger
than every other lever in the project.

Streams do not buy bandwidth — four concurrent copies move the same 28.8 GB/s
as one. They buy overlap, which is the entire point.

## Correctness

- **Determinism, on request.** Fixed accumulation order per parameter, fixed
  reduction order. Off by default because it costs scheduling freedom; on for
  any run whose result has to be reproduced.
- **The plug-in must be removable.** Same data, same seed, same steps, with
  the runtime and with plain in-memory training on a model small enough to fit
  both ways: the losses must match to the precision the arithmetic allows.
  This is the only cheap check that placement has not corrupted anything.
- **Checkpoint and resume are the store's job**, not the framework's. The
  state already lives in a tiered store on disk; a checkpoint is a barrier and
  a manifest, not a 700 GB copy.
- **A block that fails its checksum is a hard error.** Silently wrong weights
  after nine hours is the one outcome worse than crashing.

## Failure

Every failure is at plan time or it is a bug:

- **Out of memory belongs to the planner**, verified at step zero by
  allocating the peak resident set up front. An OOM at step 900 means the plan
  was checked on averages.
- **Disk write throughput exhausted** — the planner used a read figure and the
  step also writes. Measure both; the mixed number is the one that binds.
- **A tier budget exceeded** — never allocate past it, and never let the
  framework's allocator and this one both believe they own the same VRAM.
