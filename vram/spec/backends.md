# Backends — CUDA and Metal are not the same machine

The planner is portable. The runtime is not, and the reason is not API
differences — it is that a discrete GPU and a unified-memory machine have
different numbers of memory tiers, so the same plan is not merely faster or
slower on one, it is *shaped differently*.

## The two topologies

| | discrete GPU | unified memory |
|---|---|---|
| tiers | VRAM ‖ link ‖ host RAM ‖ NVMe | one pool ‖ NVMe |
| the scarce link | PCIe — 28.8 GB/s measured here | there is none |
| host RAM as a tier | real, and 4.5× the disk | **does not exist** |
| spilling past VRAM | costs a link crossing | costs nothing until RAM runs out, then costs a disk |
| host and device memory | separate, and copies are explicit | the same bytes; a "copy" is a pointer |
| contention | GPU and CPU have their own bandwidth | they share one pool's bandwidth |
| page-locking | worth 1.8×, measured | meaningless — nothing is paged for the GPU |

**Unified memory is not "CUDA with a very fast link".** The middle tier is
gone. On a discrete card the planner's first question is what to spill to host
RAM; on a Mac that question has no answer and the next stop is the SSD, four
and a half times slower. A planner that models one as a limit of the other
picks wrong every time — it will happily spill on Apple silicon believing it
costs a link crossing, when it costs a disk.

The compensation runs the other way: with no link there is no per-microbatch
weight-streaming floor at all, so the microbatch can be sized purely against
activation memory. The whole "2,100 tokens" column vanishes on Metal. Only
the per-step floor survives, against the disk.

## What the abstraction has to provide

Small, and deliberately below the framework:

    alloc(tier, bytes)            VRAM / host / unified; disk is a file
    copy(dst, src, queue)         async, ordered on a queue
    event(queue) / wait(event)    cross-queue ordering
    launch(kernel, queue)         the optimizer window kernel
    read_blocks(file, ranges, q)  deep-queued, into a device-reachable buffer

Everything above is one of six calls, and both backends have all six. The
divergence is in what they *mean*.

| | CUDA | Metal |
|---|---|---|
| device memory | `cudaMalloc` | `MTLHeap`, `MTLResourceStorageModePrivate` |
| host tier | `cudaMallocHost` — page-locked, 1.8× | no equivalent, and no need |
| shared | `cudaHostAllocMapped` | `MTLResourceStorageModeShared` — the normal case |
| queues | streams | `MTLCommandQueue` |
| events | `cudaEvent` | `MTLEvent` / `MTLSharedEvent` |
| disk → device | `pread` + staging, or cuFile/GDS | `pread` into a shared buffer, zero-copy from there |
| the optimizer kernel | CUDA C | Metal Shading Language |

The optimizer window kernel is the only kernel the project writes, and it has
to be written twice. It is elementwise and about a hundred lines; that is an
acceptable duplication and it is the price of not depending on either vendor's
training stack.

## Where the disk path differs, and it favours Metal

On a discrete card a block from NVMe goes disk → host → VRAM, two hops, unless
GPUDirect Storage is available — which it generally is not on consumer parts
under a consumer OS. On unified memory it goes disk → the one pool, one hop,
and the GPU can read it in place. **Apple silicon is structurally better at
the tier this project cares most about**, which is a strange result and worth
stating, because everything else about the platform is worse-supported.

## What is uncertain, and it is the third unknown in the charter

The runtime needs explicit control of allocation, queue ordering and events.
PyTorch's MPS backend and MLX both manage memory themselves and neither was
designed to have a residency layer underneath it. If they will not give up
that control — no way to pin an allocation to a tier, no way to order a copy
against a compute queue the framework owns — then the Metal-first argument
fails, and with it most of the reason to build a runtime rather than configure
DeepSpeed.

That question is answerable in a few days on borrowed hardware and it should
be answered **before** the CUDA runtime is written, not after, because it
decides whether the abstraction above is worth having at all. This box has no
Apple silicon, so it cannot be answered here.

## Order of work

1. Establish that MPS or MLX exposes the six calls. Cheap, and it gates
   everything.
2. Build the CUDA runtime first regardless — DeepSpeed exists to check it
   against, and a runtime with no reference implementation to disagree with is
   a runtime nobody can debug.
3. Port to Metal, where the reference is nothing at all, and where the result
   is the part of this project that does not already exist.
