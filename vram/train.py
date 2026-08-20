"""Roofline for *training* a model that does not fit, on one small GPU.

`tiers.py` is about inference, where the weights are read-only and every byte
crosses the link once per **token**. Training inverts all three of those:
gradients and optimizer state are written, the state is touched once per
**step**, and a step is thousands of tokens of compute. That one change moves
the answer by three orders of magnitude, and it is the reason offloading is
hopeless for inference and nearly free for training.

The arithmetic in one line. Streaming weights costs `bytes/BW`; the compute
they feed is `~2 x tokens` FLOPs per byte. So the traffic hides under the
compute once

    2 x tokens_per_microbatch  >=  FLOPS / BW

On this box that is 119e12 / 28.8e9 = 4132 FLOP/byte, so **~2100 tokens per
microbatch** and the link stops mattering. Decode reads the same weights to
do one multiply-add each: 1 FLOP per byte of BF16, against the same 4132
required. That factor of ~4000 is the whole difference, and it is why the
identical offload is useless for inference and nearly free for training.

So for training the question is not "how fast is the link" but "what is the
smallest split that keeps the GPU fed", and that is a scheduling problem with
a closed-form answer. This file computes it.

No dependencies. `python3 vram/train.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1_000_000_000


@dataclass(frozen=True)
class Device:
    """A machine, described by the three numbers that decide the split."""

    name: str
    vram_gb: float  # usable for tensors, after driver and framework
    host_gb: float  # page-locked host memory available to spill into
    flops: float  # achieved BF16 GEMM, not peak
    link: float  # host -> device bytes/s; None-like (0) means unified memory
    disk: float  # sustained read from the archive/state file
    unified: bool = False

    @property
    def intensity(self) -> float:
        """FLOPs the device can do per byte the link can deliver."""
        return self.flops / self.link if self.link else float("inf")


# Measured, probe/overlap.cu. The fraction of streamed traffic that actually
# disappears under a cuBLAS GEMM. It is NOT 1.0, and that is the single most
# important correction in this file.
#
# The control in the same probe overlaps a copy against a kernel that touches
# no memory at all and recovers ~100%, so the copy engine is not the problem.
# Against a real GEMM only ~20% hides: streaming traffic evicts the tiles
# cuBLAS reuses out of L2, and the GEMM slows by roughly what the transfer
# should have cost. Attackable -- CUDA's L2 access-policy windows exist for
# exactly this -- but not free, and unproven until someone attacks it.
HIDDEN = 0.20

# Utilisation the planner aims for. A floor is not "traffic becomes free" any
# more; it is "traffic costs less than a tenth of the step".
TARGET_UTIL = 0.90

DEVICES = [
    # Measured on this box: probe/bw.cu, probe/gemm.cu, probe/io.c.
    # 38e12, not the 119e12 a burst measures. probe/gemm.cu runs short bursts
    # from cold clocks; probe/overlap.cu warms to steady state first and sees
    # 35-60. Training is sustained by definition, so the low figure is the
    # honest one -- and it cuts the other way from HIDDEN, because a slower
    # GPU needs fewer tokens to cover the same transfer.
    Device("RTX 5080 16GB / PCIe4", 13.0, 20.0, 38e12, 28.8 * GB, 6.34 * GB),
    # Same card on a Gen5 slot: the link doubles, nothing else changes.
    Device("RTX 5080 16GB / PCIe5", 13.0, 20.0, 38e12, 55.0 * GB, 6.34 * GB),
    # Unified memory: there is no link, so the only offload tier is the SSD
    # and 'vram' and 'host' are the same pool. Vendor figures, not measured
    # here -- this box has no Apple silicon.
    Device("M4 Max 64GB unified", 56.0, 56.0, 34e12, 0.0, 5.0 * GB, unified=True),
    Device("M4 Pro 24GB unified", 20.0, 20.0, 17e12, 0.0, 5.0 * GB, unified=True),
]


@dataclass(frozen=True)
class Method:
    """A way of training, priced by bytes per parameter and FLOPs per token.

    `resident` is the per-parameter cost of things that must be reachable
    every microbatch (the weights). `state` is the per-parameter cost of
    things touched once per optimizer step (gradients, master weights,
    moments) -- the distinction is the whole point, because only the second
    kind amortises against batch size.
    """

    name: str
    weight_bytes: float  # per parameter, what the forward reads
    state_bytes: float  # per parameter, gradients + master + moments
    flops_per_token: float  # multiples of P; 8 = fwd+bwd+recompute
    trains: str


METHODS = [
    # Mixed-precision Adam: bf16 weights + bf16 grads + fp32 master + m + v.
    Method("full, fp32 Adam", 2, 2 + 4 + 4 + 4, 8, "every weight"),
    # 8-bit moments (bitsandbytes-class): m and v drop from 4 bytes to 1.
    Method("full, 8-bit Adam", 2, 2 + 4 + 1 + 1, 8, "every weight"),
    # bf16 throughout with stochastic rounding: no fp32 master copy.
    Method("full, bf16 Adam", 2, 2 + 1 + 1, 8, "every weight"),
    # LoRA: base weights frozen, so no gradient or moment for them, and the
    # weight-gradient GEMM is skipped -- 6 FLOP/param/token, not 8.
    Method("LoRA, bf16 base", 2, 0.02, 6, "adapters only"),
    Method("QLoRA, 4-bit base", 0.55, 0.02, 6, "adapters only"),
]


@dataclass(frozen=True)
class Model:
    name: str
    params: float
    layers: int
    hidden: int


MODELS = [
    Model("Llama-3.1-8B", 8.03e9, 32, 4096),
    Model("Qwen-14B", 14.8e9, 48, 5120),
    Model("Llama-3.1-70B", 70.6e9, 80, 8192),
]


def activation_gb(m: Model, tokens: int) -> float:
    """Checkpointed activations: one saved tensor per layer boundary, plus
    the working set of the single layer being recomputed. Approximate, and
    deliberately on the low side -- attention and the MLP intermediate make
    the real figure larger, and a tool would measure rather than model it."""
    boundaries = m.layers * tokens * m.hidden * 2
    working = tokens * m.hidden * 2 * 16  # one layer's intermediates, bf16
    return (boundaries + working) / GB


def _penalty() -> float:
    """How much bigger a microbatch has to be than the naive answer.

    Naively the traffic is hidden once compute exceeds it. Two measured facts
    move that. Only `HIDDEN` of the traffic actually overlaps, so the exposed
    part is `(1 - HIDDEN)`; and aiming for TARGET_UTIL rather than break-even
    means compute must exceed the exposed part by `u / (1 - u)`.
    """
    return (1 - HIDDEN) * TARGET_UTIL / (1 - TARGET_UTIL)


def tokens_to_hide_weights(d: Device, meth: Method, bw: float) -> float:
    """Microbatch size at which streaming the weights hides under compute.

    Per microbatch the weights are read twice -- once for the forward, once
    for the backward -- so the requirement is
        flops_per_token x tokens  >=  2 x weight_bytes x (FLOPS / bandwidth)

    `bw` is the bandwidth of wherever the weights actually live, which is not
    always the link: weights that spill past host RAM come off the disk at a
    quarter of the link's rate, and on a unified-memory machine there is no
    link at all but there is still a disk. Both of those were wrong here at
    first, and both errors flattered the answer.
    """
    if bw is None:
        return 0.0
    return (_penalty() * 2 * meth.weight_bytes * (d.flops / bw)
            / meth.flops_per_token)


def tokens_to_hide_state(d: Device, meth: Method, bw: float) -> float:
    """Tokens per *optimizer step* at which the state traffic hides.

    State is read and written once per step, and the updated weights are
    written back, so the per-parameter traffic is 2 x state + weights."""
    per_param = 2 * meth.state_bytes + meth.weight_bytes
    return _penalty() * per_param * (d.flops / bw) / meth.flops_per_token


def shard(m: Model, meth: Method, n: int, interconnect: float) -> dict:
    """The device axis, first-order only.

    Sharding the state across `n` devices (ZeRO-3 shaped) divides what each
    one has to hold, and buys that with traffic: weights are all-gathered a
    layer at a time on the way forward and again on the way back, and
    gradients are reduce-scattered. Per device and per microbatch that is
    `(n-1)/n` of the weights in each direction; the state is never gathered,
    so it stays local and its per-device cost falls as 1/n.

    **What this does not model, and a planner must:** pipeline bubbles, the
    fact that consumer cards have no NVLink and share one PCIe root complex
    so `interconnect` is contended rather than per-device, uneven layer
    costs, and the choice between sharding and plain pipeline parallelism.
    It is here so the axis exists in the solver rather than being bolted on
    later -- it is not yet good enough to choose a topology with.
    """
    share = (n - 1) / n if n > 1 else 0.0
    return dict(
        weights_gb=m.params * meth.weight_bytes / n / GB,
        state_gb=m.params * meth.state_bytes / n / GB,
        gather_bytes=2 * share * m.params * meth.weight_bytes,
        scatter_bytes=share * m.params * meth.weight_bytes,
        interconnect=interconnect,
    )


def plan(d: Device, m: Model, meth: Method, tokens_per_mb: int = 2048):
    """Where each thing has to live, and what that costs."""
    w_gb = m.params * meth.weight_bytes / GB
    s_gb = m.params * meth.state_bytes / GB
    act_gb = activation_gb(m, tokens_per_mb)

    # Weights first: on the fast pool if they fit beside the activations,
    # else streamed from wherever they do fit.
    room = d.vram_gb - act_gb
    w_resident = room >= w_gb
    if d.unified:
        # One pool. Everything that fits in it is resident; the rest is disk,
        # and disk still has to be read every microbatch.
        where_w = "unified" if d.vram_gb >= w_gb + act_gb + s_gb else "disk"
        w_bw = None if where_w == "unified" else d.disk
        state_bw = d.disk if where_w == "disk" else None
        where_s = "unified" if state_bw is None else "disk"
    else:
        where_w = "VRAM" if w_resident else ("host" if w_gb <= d.host_gb else "disk")
        w_bw = {"VRAM": None, "host": d.link, "disk": d.disk}[where_w]
        # State lives wherever it fits, after the weights have taken theirs.
        host_left = d.host_gb - (w_gb if where_w == "host" else 0)
        if w_resident and room - w_gb >= s_gb:
            where_s, state_bw = "VRAM", None
        elif s_gb <= host_left:
            where_s, state_bw = "host", d.link
        else:
            where_s, state_bw = "disk", d.disk

    need_mb = tokens_to_hide_weights(d, meth, w_bw)
    need_step = 0.0 if state_bw is None else tokens_to_hide_state(d, meth, state_bw)

    fits = act_gb < d.vram_gb
    return dict(w_gb=w_gb, s_gb=s_gb, act_gb=act_gb, where_w=where_w,
                where_s=where_s, need_mb=need_mb, need_step=need_step,
                fits=fits, state_bw=state_bw)


def _budget() -> None:
    print("Where the memory actually goes -- GB, and note which column is big")
    print(f"{'model':<16}{'method':<20}{'weights':>9}{'state':>8}{'act@2k':>9}{'total':>9}")
    for m in MODELS:
        for meth in METHODS:
            w = m.params * meth.weight_bytes / GB
            s = m.params * meth.state_bytes / GB
            a = activation_gb(m, 2048)
            print(f"{m.name:<16}{meth.name:<20}{w:>9.1f}{s:>8.1f}{a:>9.1f}{w + s + a:>9.1f}")
        print()


def _hide() -> None:
    print("The scheduling answer: tokens needed before the offload disappears")
    print(f"{'device':<24}{'method':<20}{'per microbatch':>15}{'per step':>10}")
    for d in DEVICES:
        bw = d.link or d.disk  # unified machines offload to the disk
        for meth in METHODS:
            mb = tokens_to_hide_weights(d, meth, bw)
            st = tokens_to_hide_state(d, meth, bw)
            print(f"{d.name:<24}{meth.name:<20}{mb:>15,.0f}{st:>10,.0f}")
        print()
    print("Per microbatch = weights streamed for forward and backward. Per step\n"
          "= gradients, moments and the weight write-back, over the link if the\n"
          "state fits in host RAM and over the disk if it does not. Both are\n"
          "'above this, the GPU stops waiting'. For contrast, inference decode\n"
          "moves one token per pass of the weights, so the same sums ask for\n"
          "~4000x more tokens than exist and the offload can never be hidden.\n")


def _plans() -> None:
    print("What each machine can actually train, at 2048 tokens per microbatch")
    print(f"{'device':<24}{'model':<16}{'method':<20}{'weights':>8}{'state':>7}"
          f"{'need/mb':>9}{'need/step':>10}")
    for d in DEVICES:
        for m in MODELS:
            for meth in METHODS:
                p = plan(d, m, meth)
                if not p["fits"]:
                    continue
                note = "" if p["need_mb"] <= 2048 else "  <-- starved"
                print(f"{d.name:<24}{m.name:<16}{meth.name:<20}"
                      f"{p['where_w']:>8}{p['where_s']:>7}"
                      f"{p['need_mb']:>9,.0f}{p['need_step']:>10,.0f}{note}")
        print()
    print("'weights'/'state' is where each has to live. 'need/mb' and\n"
          "'need/step' are the token counts from the table above; anything at\n"
          "or under the 2048 assumed here runs at full GPU speed with the\n"
          "traffic entirely hidden. A row marked starved needs a longer\n"
          "sequence or a bigger microbatch, which costs activation memory --\n"
          "that trade is the planner's whole job.\n")


if __name__ == "__main__":
    _budget()
    _hide()
    _plans()
