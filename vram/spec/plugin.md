# Plug-in — what "external and optional" has to mean

**Scope: the inference half.** The project's main line is a training runtime
that owns its own memory and optimizer step (`runtime.md`, `backends.md`);
this file covers the other case, where the goal is to attach to somebody
else's inference engine without forking it.

The engine is a library with a small C ABI and one thin adapter per inference
framework. This file fixes the ABI, what each adapter can reach without a
fork, and the four things the plug-in is never allowed to do.

Framework extension points named below are the *shape* of the attachment, not
a promise about a particular release — every one of them has to be checked
against the version in hand before an adapter is written.

## The contract

**Optional means removable.** With the plug-in loaded and with it disabled,
the same prompt against the same model must produce the **same tokens, bit for
bit**. Phase 1 is lossless end to end and that is the cheapest possible check
that it stayed lossless; a run that differs is a bug, not a trade-off.

**External means no fork.** An adapter is a package the user installs
alongside their stack. Where that is impossible the spec says so rather than
quietly shipping a patched engine.

**Nothing is materialised.** No conversion step, no scratch file, no
"decompress then load". The archive on disk is what is read and it stays the
archive. This is G4's claim and it is what separates the engine from every
offload path that expands to a temp file first.

**Nothing is written.** The weight archive is opened read-only. Blocks in
every tier are clean, so eviction is a free-list operation and there is no
write-back path to get wrong.

## The fetch ABI

Deliberately small, C, and framework-blind, so an adapter can be Python, C++
or Rust. Handles are opaque; everything is stream-ordered.

    vz_archive *vz_open(const char *path);
    int  vz_lookup(vz_archive *, const char *tensor, vz_extent *out);

    vz_engine  *vz_engine_create(const vz_config *);   /* tier budgets, queue depth */
    int  vz_plan(vz_engine *, const vz_extent *, size_t n);

    int  vz_prefetch(vz_engine *, const vz_extent *, size_t n, vz_stream);
    int  vz_fetch(vz_engine *, const vz_extent *, void *dst_device, vz_stream);
    const void *vz_coded(vz_engine *, const vz_extent *, vz_stream);
    void vz_release(vz_engine *, const vz_extent *);

`vz_lookup` turns a tensor name into a block range with its dtype and shape;
it touches the index only, so opening a 70B archive must not cost a gigabyte
of RAM or eleven seconds — which today it does, and which is item 3 of lmz's
handover.

**The two fetch verbs are the whole design.** `vz_fetch` decodes into a device
buffer the caller owns: it works with any kernel, needs no cooperation from
the framework, and reaches tiers H and S. `vz_coded` returns a device pointer
to *coded* blocks and decodes nothing: it is how tier V-coded is reached, and
it requires the caller to own a kernel that decodes as it multiplies. An
adapter that only calls `vz_fetch` is a good offload engine. An adapter that
calls `vz_coded` is the thing the ladder in the README is about.

`vz_plan` runs the placement solver over a whole model's extents once, at
load. It is separate from fetching because placement is a global decision and
fetching is a per-layer one.

## What each framework can reach

| engine | tiers H and S | tier V-coded | fork? |
|---|---|---|---|
| **PyTorch / transformers** | module pre-forward hooks in the shape `accelerate` already uses for `device_map`, plus an archive-backed state-dict loader | a replacement `Linear` module owning a fused decode-GEMV | **none** |
| **vLLM** | an out-of-tree model loader | an out-of-tree linear method — the same registration point a quantisation scheme uses, which is what this effectively is | **none**, if that point holds |
| **llama.cpp / ggml** | a custom backend buffer type whose fetch decodes | a **new ggml type** with its own `vec_dot` and row dequant — which is exactly how every `Q*_K` already works | **upstream patch**: ggml's type table is compile-time |
| **anything that calls `open`/`read`/`mmap`** | `lmz mount`, today, unmodified | not reachable | **none** |

Three things are worth pulling out of that table.

**vLLM's quantisation hook is the right shape by accident.** A residency
method and a quantisation method want the same thing from the framework:
own the weight's storage, own the kernel that consumes it. Registering as one
is not a hack, it is the same abstraction.

**ggml has already solved this problem for itself.** A ggml quant type *is* a
weight format that decodes inside the kernel — the V-coded tier is a ggml type
whose decode happens to be rANS. That makes llama.cpp the most natural home
for the idea and the only one that needs an upstream change, because ggml
types are a compile-time table.

**The no-patch path exists today and is slow, and that is fine.** `lmz mount`
makes a coded model readable as ordinary files by anything, with no adapter at
all — at 2.2 GB/s of CPU decode, which is *below* this box's sequential NVMe
read of 3.59. It buys disk, not speed, exactly as lmz's own `limitations.md`
says. It is the fallback and the compatibility proof, not the product.

## Configuration, and what the user is asked

As little as possible. The engine measures the machine and solves the
placement; the user says how much of each tier it may have.

    vram_budget    bytes of VRAM the engine may use for weights
    host_budget    bytes of page-locked host RAM
    archive        path to the coded model
    queue_depth    outstanding block reads, default 32
    disable        run as if the plug-in were absent

`vram_budget` is the only one that usually needs setting, and getting it wrong
is safe in one direction and fatal in the other: too small costs throughput,
too large gets an OOM from the framework's own allocator halfway through a
run. The engine reserves its budget up front so the failure is at load.

## Failure

Every failure mode falls back to correct-and-slower, never to wrong:

- **No CUDA, no GPU decoder, unsupported arch** → CPU decode, or the mount.
- **A block that does not verify** → hard error. lmz checksums per chunk;
  a silently wrong weight is the one outcome worse than not running.
- **Budget exhausted** → spill to the next tier down and log it. Never
  allocate past the budget, and never let the framework's allocator and this
  one both believe they own the same VRAM.
- **A tensor the archive does not have** → fall through to the framework's
  own loader for that tensor. A partially covered model must still run.
