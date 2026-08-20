"""Roofline model for coded memory tiers on a GPU that is short of VRAM.

`model/system.py` asks what a machine's memory bus is worth. This asks the
question one level down, on a machine somebody else already built: when the
weights do not fit in VRAM, what does each place you could put them actually
deliver, and what is a compressed representation worth in each place?

One measured ratio decides most of it. A GPU rANS decoder feeding the tensor
cores directly runs at 948 GB/s on this card; the PCIe link into it runs at
28.8. **The decoder is 33x faster than the link it feeds**, so on the host and
storage tiers compression is free -- every point of ratio becomes a point of
bandwidth, long before the decoder becomes the constraint. Against VRAM itself
the same decoder is **1.07x**, so on the card compression now buys capacity
and costs nothing: a coded byte in VRAM is delivered faster than a raw one.
The third fact is that a coded VRAM tier is 22x faster than the fastest tier
off the card, which is why it is worth having at all.

That middle number used to be 0.51x, and every "compression costs speed on
the card" claim in this repository descends from it. It was not the codec
that changed.

Every bandwidth here was measured on the machine this repository lives on --
`probe/bw.cu`, `probe/io.c` and `fused/fused_gemm.cu`, conditions in
`MEASURED.md`. The standalone decode rates come from lmz's own GPU experiment
on the same GPU (`lmz/scratchpad/gpu/README.md`); the fused ones are this
project's.

No dependencies. `python3 vram/tiers.py` prints the tables.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1_000_000_000  # decimal, to match how everyone quotes bandwidth

# ---------------------------------------------------------------------------
# Measured on this box, 2026-08-20. RTX 5080 (sm_120, 84 SMs, 16 GB, PCIe 4 x16
# as the slot reports it), Ryzen 7 9800X3D, WSL2 ext4 on a virtualised NVMe.
# ---------------------------------------------------------------------------

VRAM_COPY = 776.0  # device-to-device, read+write counted, 81% of 960 peak; 768-785
VRAM_READ = 887.0  # read-only, which is what a weight is: fused/fused_gemm.cu's
# control kernel over 268 MB, matched by cuBLAS on the same GEMM. This, not
# VRAM_COPY, is the right figure for a tier that only ever reads.
H2D_PINNED = 28.80  # 1 GiB cudaMemcpy from page-locked host memory; +/- 0.1
H2D_PAGEABLE = 16.00  # the same copy from ordinary malloc'd memory; 13.9-16.9 run to run
SSD_SEQ = 3.59  # O_DIRECT, 4 MiB reads, one thread
SSD_RND_QD1 = 0.34  # O_DIRECT, 64 KiB random, one outstanding request
SSD_RND_QD64 = 6.34  # O_DIRECT, 64 KiB random, 64 outstanding

# lmz/scratchpad/gpu, same GPU: the fused whole-BF16 kernel, and the
# exponent-plane kernel it was built from. Both decode into VRAM.
GPU_DECODE = 399.0
GPU_DECODE_PLANE = 418.0
CPU_DECODE = 2.21  # lmz, all cores, free-threaded -- for contrast only

# fused/fused_gemm.cu, same GPU, 268 MB of real BF16 weights past L2.
# The first is a decoder feeding mma.sync directly and never touching DRAM
# with its output; the second is the same code writing BF16 to VRAM, which is
# what any unfused consumer forces. Median of five runs, +/- 3%.
#
# GPU_DECODE_FUSED > VRAM_READ is the whole shape of the model now: a coded
# byte on the card is delivered FASTER than a raw one, so the coded tier is
# not a capacity-for-speed trade any more. The format that gets there costs
# 0.6% of ratio (1.473x against lmz's 1.485x) for its per-stream state header.
GPU_DECODE_FUSED = 948.0
GPU_DECODE_TO_VRAM = 448.0

# lmz's measured lossless ratios, by what the checkpoint is stored as.
# docs/results.md in the submodule. These are the multipliers a lossless
# codec has to offer; the point of the last table is how far short they fall.
RATIOS = {
    "BF16": 1.485,  # 34.7% saved
    "FP8": 1.207,  # 17.14%
    "Q8_0": 1.072,  # 6.7%
    "Q4_K_M": 1.054,  # 5.1%
}


@dataclass(frozen=True)
class Tier:
    """Somewhere a weight can live, and what it costs to read it from there.

    `vram_per_byte` is how much VRAM one byte of *model* occupies here: 1.0
    for raw weights in VRAM, 1/ratio for coded ones, 0 for anything off-card.
    """

    name: str
    link: float  # GB/s of coded bytes this tier can deliver
    on_card: bool
    coded: bool
    fused: bool = True

    def bandwidth(self, ratio: float) -> float:
        """Effective GB/s of *model* bytes, after decode."""
        if not self.coded:
            return self.link
        if not self.on_card:
            # Coded bytes cross the link; decode happens on the GPU and is
            # not in the way as long as it outruns ratio x link.
            return min(self.link * ratio, GPU_DECODE_FUSED)
        if self.fused:
            # Decode a tile straight into the matrix unit: the only DRAM
            # traffic is the coded read, so the decoder is the ceiling.
            # MEASURED -- fused/fused_gemm.cu, 440 GB/s of BF16 feeding
            # mma.sync, 3-4% below the same decoder with nothing to feed.
            return min(GPU_DECODE_FUSED, self.link * ratio)
        # Decode into a staging buffer: the decode pass writes raw bytes as
        # it goes and the matrix unit reads them back. The two passes add --
        # neither hides the other, because the first is decode-bound before
        # its write is counted. Predicts 297, measured 286.
        return 1.0 / (1.0 / GPU_DECODE_TO_VRAM + 1.0 / self.link)

    def vram_per_byte(self, ratio: float) -> float:
        if not self.on_card:
            return 0.0
        return 1.0 / ratio if self.coded else 1.0


def tiers(fused: bool = True) -> list[Tier]:
    return [
        Tier("VRAM, raw", VRAM_READ, True, False),
        Tier("VRAM, coded", VRAM_READ, True, True, fused),
        Tier("host RAM, pinned, coded", H2D_PINNED, False, True),
        Tier("host RAM, pinned, raw", H2D_PINNED, False, False),
        Tier("host RAM, pageable, raw", H2D_PAGEABLE, False, False),
        Tier("NVMe 64 KiB QD=64, coded", SSD_RND_QD64, False, True),
        Tier("NVMe 64 KiB QD=64, raw", SSD_RND_QD64, False, False),
        Tier("NVMe 64 KiB QD=1, raw", SSD_RND_QD1, False, False),
    ]


def place(footprint_gb: float, vram_gb: float, ratio: float,
          avail: list[Tier]) -> list[tuple[Tier, float]]:
    """Fractional knapsack: spend each VRAM byte where it saves the most time.

    Two passes, because one is not enough. Filling ranks the on-card tiers by
    time saved *per byte of VRAM spent*, and while VRAM is scarce the coded
    tier usually wins that outright -- it is slower than raw VRAM but asks for
    1/ratio of the space, and what it displaces is 27x slower than either. But
    once the whole model is on the card there may be VRAM left over, and then
    the right move is the opposite one: promote coded weights back to raw,
    paying (1 - 1/ratio) of space for (1/B_coded - 1/B_raw) of time. Without
    the second pass the solver "compresses" a model that already fits, and
    reports a slowdown as a gain.
    """
    on_card = [t for t in avail if t.on_card]
    off_card = [t for t in avail if not t.on_card]
    fallback = max(off_card, key=lambda t: t.bandwidth(ratio))
    t_slow = 1.0 / fallback.bandwidth(ratio)

    def density(t: Tier) -> float:
        return (t_slow - 1.0 / t.bandwidth(ratio)) / t.vram_per_byte(ratio)

    held: dict[str, float] = {}
    left_model, left_vram = footprint_gb, vram_gb
    for t in sorted(on_card, key=density, reverse=True):
        take = min(left_model, left_vram / t.vram_per_byte(ratio))
        if take <= 1e-9:
            continue
        held[t.name] = take
        left_model -= take
        left_vram -= take * t.vram_per_byte(ratio)

    # Promote, while there is space and a promotion that pays.
    by_name = {t.name: t for t in on_card}
    while left_vram > 1e-9:
        best, gain = None, 0.0
        for src in list(held):
            for dst in on_card:
                extra = dst.vram_per_byte(ratio) - by_name[src].vram_per_byte(ratio)
                if extra <= 1e-9 or held[src] <= 1e-9:
                    continue
                saved = 1.0 / by_name[src].bandwidth(ratio) - 1.0 / dst.bandwidth(ratio)
                if saved <= 0:
                    continue
                if saved / extra > gain:
                    best, gain = (src, dst, extra), saved / extra
        if best is None:
            break
        src, dst, extra = best
        move = min(held[src], left_vram / extra)
        held[src] -= move
        held[dst.name] = held.get(dst.name, 0.0) + move
        left_vram -= move * extra

    out = [(by_name[n], gb) for n, gb in held.items() if gb > 1e-9]
    out.sort(key=lambda p: p[0].bandwidth(ratio), reverse=True)
    if left_model > 1e-9:
        out.append((fallback, left_model))
    return out


def seconds_per_token(placement, footprint_gb: float, per_token_gb: float,
                      ratio: float) -> float:
    """Time for one token's weight traffic.

    For a dense model per_token == footprint. For an MoE only a slice is
    touched per token; assuming routing spreads uniformly over experts, the
    tier fractions of the footprint are also the tier fractions of the slice.
    """
    scale = per_token_gb / footprint_gb
    return sum(gb * scale / t.bandwidth(ratio) for t, gb in placement)


# ---------------------------------------------------------------------------


def _ladder() -> None:
    print("The ladder, measured on this box -- GB/s of model bytes delivered")
    print(f"{'tier':<28}{'link':>8}{'BF16':>8}{'FP8':>8}{'Q8_0':>8}{'Q4_K_M':>8}")
    for t in tiers():
        cols = "".join(f"{t.bandwidth(r):>8.1f}" for r in RATIOS.values())
        print(f"{t.name:<28}{t.link:>8.1f}{cols}")
    print()
    print(f"GPU rANS decode into the matrix unit      {GPU_DECODE_FUSED:>8.1f}")
    print(f"GPU rANS decode into VRAM                 {GPU_DECODE_TO_VRAM:>8.1f}")
    print(f"lmz on the CPU, all cores, free-threaded  {CPU_DECODE:>8.1f}")
    print()
    for label, val, note in [
        ("decode / host link", GPU_DECODE_FUSED / H2D_PINNED,
         "compression is free here, up to that ratio"),
        ("decode / NVMe at depth", GPU_DECODE_FUSED / SSD_RND_QD64,
         "and free here by a wider margin still"),
        ("decode / VRAM", GPU_DECODE_FUSED / VRAM_READ,
         "on the card it now buys capacity and costs nothing"),
        ("coded VRAM / best off-card", GPU_DECODE_FUSED / (H2D_PINNED * RATIOS["BF16"]),
         "why a coded VRAM tier is worth having"),
        ("pinned / pageable", H2D_PINNED / H2D_PAGEABLE,
         "free, codec or no codec, and most offload paths miss it"),
        ("QD=64 / QD=1 on NVMe", SSD_RND_QD64 / SSD_RND_QD1,
         "the largest single lever in this table"),
    ]:
        print(f"{label:<26}{val:>6.2f}x   {note}")
    print()


@dataclass(frozen=True)
class Case:
    name: str
    footprint_gb: float
    fmt: str
    per_token_gb: float = 0.0

    @property
    def touched(self) -> float:
        return self.per_token_gb or self.footprint_gb


CASES = [
    Case("Llama-3.1-8B BF16", 16.06, "BF16"),
    Case("Llama-3.1-8B Q8_0", 8.54, "Q8_0"),
    Case("Llama-3.1-70B Q4_K_M", 40.0, "Q4_K_M"),
    Case("Llama-3.1-70B Q8_0", 75.0, "Q8_0"),
    Case("120B-A5B MoE Q4", 60.0, "Q4_K_M", 2.5),
]

VRAM_FOR_WEIGHTS = 13.0  # 16 GB card, less driver, context, activations, KV


def _naive(fused: bool) -> list[Tier]:
    """What an out-of-the-box offload path does: pageable copies, shallow queue."""
    keep = {"VRAM, raw", "host RAM, pageable, raw", "NVMe 64 KiB QD=1, raw"}
    return [t for t in tiers(fused) if t.name in keep]


def _tuned(fused: bool) -> list[Tier]:
    """The same thing engineered properly, and still with no codec at all."""
    keep = {"VRAM, raw", "host RAM, pinned, raw", "NVMe 64 KiB QD=64, raw"}
    return [t for t in tiers(fused) if t.name in keep]


def _cases(fused: bool = True) -> None:
    print(f"Where each gain comes from -- {VRAM_FOR_WEIGHTS:.0f} GB of VRAM for "
          f"weights, {'fused' if fused else 'staged'} decode")
    print(f"{'model':<24}{'naive':>7}{'tuned':>7}{'coded':>7}"
          f"{'eng':>6}{'codec':>7}  placement")
    for c in CASES:
        r = RATIOS[c.fmt]
        rows = []
        for avail in (_naive(fused), _tuned(fused), tiers(fused)):
            p = place(c.footprint_gb, VRAM_FOR_WEIGHTS, r, avail)
            rows.append((p, 1.0 / seconds_per_token(p, c.footprint_gb, c.touched, r)))
        (_, tn), (_, tt), (pc, tc) = rows
        where = ", ".join(f"{gb / c.footprint_gb:.0%} {t.name}" for t, gb in pc)
        print(f"{c.name:<24}{tn:>7.1f}{tt:>7.1f}{tc:>7.1f}"
              f"{tt / tn:>5.1f}x{tc / tt:>6.2f}x  {where}")
    print("\nnaive = pageable host copies and one outstanding read: what an\n"
          "        out-of-the-box offload path does.\n"
          "tuned = the same placement, page-locked and queued to depth, no\n"
          "        codec anywhere. 'eng' is what that costs nothing to have.\n"
          "coded = the coded tiers made available too. 'codec' is what the\n"
          "        compression adds on top of an already-tuned path, which is\n"
          "        the only honest way to score it.\n"
          "tok/s counts weight traffic only. KV cache, activations and prefill\n"
          "are not in this model and do not change its ordering.\n")


def _ratio_needed() -> None:
    """What ratio would each case need for the whole model to sit on the card?"""
    coded = [t for t in tiers() if t.name == "VRAM, coded"][0]
    print("What ratio each case needs to be VRAM-resident, and what that buys")
    print(f"{'model':<24}{'need':>7}{'have':>7}{'then':>8}{'tuned':>8}{'gain':>7}")
    for c in CASES:
        need = c.footprint_gb / VRAM_FOR_WEIGHTS
        have = RATIOS[c.fmt]
        p = place(c.footprint_gb, VRAM_FOR_WEIGHTS, have, _tuned(True))
        now = 1.0 / seconds_per_token(p, c.footprint_gb, c.touched, have)
        if need <= 1.0:
            print(f"{c.name:<24}{'fits':>7}{have:>7.2f}{'-':>8}{now:>8.1f}{'-':>7}")
            continue
        if have >= need:
            # Already over the bar: report what the solver actually reaches at
            # the ratio in hand, which is better than the break-even figure
            # because the spare VRAM gets promoted back to raw.
            then = 1.0 / seconds_per_token(
                place(c.footprint_gb, VRAM_FOR_WEIGHTS, have, tiers()),
                c.footprint_gb, c.touched, have)
            mark = "*"
        else:
            then = coded.bandwidth(need) / c.touched
            mark = " "
        print(f"{c.name:<24}{need:>7.2f}{have:>7.2f}{then:>7.1f}{mark}{now:>8.1f}"
              f"{then / now:>6.1f}x")
    print("\n'need' is footprint / VRAM: the ratio at which the model stops\n"
          "crossing the link at all. 'have' is what lossless coding of that\n"
          "format actually delivers, and 'then' is what the model would run at\n"
          "if a representation met 'need'. The gap between those two columns is\n"
          "the whole open question -- see README, 'the gate this opens'.\n"
          "* = 'have' already clears 'need', so this row is the measured\n"
          "    result at the ratio in hand, not a hypothetical.\n")


if __name__ == "__main__":
    _ladder()
    _cases(fused=True)
    _cases(fused=False)
    _ratio_needed()
