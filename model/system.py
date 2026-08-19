"""Roofline model for a memory-first inference appliance.

The whole design rests on one claim, which the reference table at the bottom of
this file exists to keep honest: decode-phase throughput is set by how many
bytes cross the memory bus per token, and the memory bus is bought with pins,
not with nanometres. Apple's M4 Max and AMD's Strix Halo sit one process node
apart and deliver 1.07 and 1.00 GB/s per bit of bus respectively.

So the levers this model exposes are bus width, memory grade, and the fraction
of each weight that has to travel. Process node appears nowhere in it.

No dependencies. `python3 system.py` prints the reference configurations.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1_000_000_000  # decimal, to match how memory vendors quote bandwidth

# Measured on Strix Halo: 256 GB/s theoretical against ~215 GB/s achieved.
# Every real system lands near here; treating peak as achievable is the most
# common way these estimates go wrong.
BUS_EFFICIENCY = 0.84


@dataclass(frozen=True)
class Memory:
    """A memory configuration, described the way a board designer buys it."""

    name: str
    # 'ddr'    socketed DIMM
    # 'lpddr'  soldered
    # 'lpcamm' socketed LPDDR module (LPCAMM2): 128 bits each, so bus width is
    #          bought in module counts, and the sockets/solder choice stops
    #          being the same choice as DDR5/LPDDR5X. See decisions.md.
    kind: str
    mt_s: int  # transfers/second in MT/s
    bus_bits: int  # total data bus width across all channels
    usd_per_gb: float
    capacity_gb: int
    # Whether a 28nm-class PHY can plausibly close timing at this grade.
    mature_node_phy: bool
    node: str = "-"  # the process the *system* is built on, for the table below

    @property
    def peak_bytes_s(self) -> float:
        return self.bus_bits / 8 * self.mt_s * 1e6

    @property
    def real_bytes_s(self) -> float:
        return self.peak_bytes_s * BUS_EFFICIENCY

    @property
    def usd(self) -> float:
        return self.usd_per_gb * self.capacity_gb

    @property
    def gb_s_per_bit(self) -> float:
        """The figure of merit that shows node is not the variable."""
        return self.peak_bytes_s / GB / self.bus_bits


@dataclass(frozen=True)
class Codec:
    """A weight representation.

    `f` is the fraction of the original size that actually crosses the bus.
    `decode_bytes_s` is the *output* rate of the decoder: it must emit the full
    uncompressed stream, so this is compared against uncompressed size, not
    against the compressed bytes read.
    """

    name: str
    f: float
    decode_bytes_s: float  # float('inf') for a passthrough

    @property
    def bandwidth_ceiling(self) -> float:
        """Best possible speedup, reached only if the decoder is free."""
        return 1.0 / self.f


PASSTHROUGH = Codec("raw", 1.0, float("inf"))


@dataclass(frozen=True)
class Workload:
    name: str
    params: int
    bytes_per_param: float  # 2.0 BF16, 1.0 int8, ~0.55 for Q4_K_M incl. scales

    @property
    def weight_bytes(self) -> float:
        return self.params * self.bytes_per_param


@dataclass
class Result:
    tokens_s: float
    limiter: str
    t_bus: float
    t_decode: float
    resident_gb: float
    fits: bool


def decode_phase(work: Workload, mem: Memory, codec: Codec = PASSTHROUGH) -> Result:
    """Batch-1 autoregressive decode: every weight is read once per token.

    Arithmetic intensity is ~1-2 FLOP/byte here, so compute is omitted -- on any
    design with a real NPU it is not the limiter, and pretending otherwise is
    how NPU TOPS numbers end up in marketing decks that mean nothing. If this
    model is ever pointed at prefill, that omission stops being safe.
    """
    w = work.weight_bytes

    t_bus = w * codec.f / mem.real_bytes_s
    t_decode = w / codec.decode_bytes_s

    t_token = max(t_bus, t_decode)
    limiter = "bus" if t_bus >= t_decode else "decoder"

    # Compression buys capacity as well as bandwidth: weights stay coded in
    # DRAM and are expanded on the way to compute, never as a resident copy.
    resident_gb = w * codec.f / GB

    return Result(
        tokens_s=1.0 / t_token,
        limiter=limiter,
        t_bus=t_bus,
        t_decode=t_decode,
        resident_gb=resident_gb,
        fits=resident_gb <= mem.capacity_gb,
    )


def required_decoder_rate(work: Workload, mem: Memory, codec: Codec) -> float:
    """Decoder output rate needed for the bus to stay the limiter.

    This is the number the silicon has to hit. Below it, the decoder -- not the
    memory system you paid for -- sets throughput.
    """
    return mem.real_bytes_s / codec.f


# --- Reference configurations -------------------------------------------------
# Shipping systems, so the model can be checked against something real, plus the
# two candidate designs. Prices are order-of-magnitude for mid-2026.

REFERENCE = [
    Memory("Phone SoC (LPDDR5X-8533)", "lpddr", 8533, 64, 9.0, 16, False, "N3"),
    Memory("Strix Halo (LPDDR5X-8000)", "lpddr", 8000, 256, 9.0, 128, False, "N4"),
    # NVIDIA N1X, announced 2026-08 for autumn systems: 20 Grace cores plus a
    # Blackwell GPU on TSMC 3nm, 128 GB soldered LPDDR5X, ~300 GB/s quoted,
    # which is 256 bits at LPDDR5X-9600. A leading-edge part, one petaflop of
    # AI compute, and it stops at half the bus width of the M4 Max beside it.
    Memory("RTX Spark N1X (LPDDR5X-9600)", "lpddr", 9600, 256, 9.0, 128, False, "N3"),
    Memory("M4 Max (LPDDR5X-8533)", "lpddr", 8533, 512, 9.0, 128, False, "N3E"),
]

CANDIDATES = [
    Memory("8ch DDR5-5600 DIMM", "ddr", 5600, 512, 2.50, 128, True),
    Memory("12ch DDR5-5600 DIMM", "ddr", 5600, 768, 2.50, 192, True),
    Memory("8ch DDR5-4800 DIMM", "ddr", 4800, 512, 2.20, 128, True),
    Memory("512b LPDDR4X-4266", "lpddr", 4266, 512, 5.00, 64, True),
    Memory("512b LPDDR5-6400", "lpddr", 6400, 512, 7.00, 96, False),
    # LPCAMM2: LPDDR5X on a socketed 128-bit module, so width comes in units of
    # two channels and capacity in units of one module (64 GB today, 96 GB
    # sampling). Two modules is the shipping configuration; four is not shipping
    # anywhere and is a signal-integrity question, not a purchase.
    #
    # Priced at launch MSRP ($451.99 / 64 GB), to stay on the same pre-spike
    # basis as the DDR5 rows above -- on that basis LPCAMM2 is 2.8x DDR5 per GB
    # and the DIMM decision holds. It does not hold at August 2026 retail; see
    # the note under the table.
    Memory("2x LPCAMM2-7500 (256b)", "lpcamm", 7500, 256, 7.06, 128, False),
    Memory("2x LPCAMM2-8533 (256b)", "lpcamm", 8533, 256, 7.06, 128, False),
    Memory("4x LPCAMM2-8533 (512b)", "lpcamm", 8533, 512, 7.06, 256, False),
    Memory("4x LPCAMM2-9600 (512b)", "lpcamm", 9600, 512, 7.06, 256, False),
]

WORKLOADS = [
    Workload("8B BF16", 8_000_000_000, 2.0),
    Workload("8B int8", 8_000_000_000, 1.0),
    Workload("70B int8", 70_000_000_000, 1.0),
]


def _table() -> None:
    print("Where bandwidth actually comes from -- the claim the design rests on")
    print(f"{'system':<30}{'node':<8}{'bus':>6}{'MT/s':>8}{'GB/s':>9}")
    for m in REFERENCE:
        print(
            f"{m.name:<30}{m.node:<8}{m.bus_bits:>6}"
            f"{m.mt_s:>8}{m.peak_bytes_s / GB:>9.0f}"
        )
    print()
    print("Bandwidth is width x data rate. Node appears nowhere in it. The M4 Max")
    print("leads Strix Halo 546 to 256 on 2x the bus and 1.07x the grade, while")
    print("sitting one node behind it in nothing that matters here. LPDDR5X-8533")
    print("is a JEDEC part anyone can buy; what gates the design is whether the")
    print("PHY closes timing, which is a mixed-signal problem, not a density one.\n")

    print("Candidate memory systems, uncompressed 8B BF16 decode")
    print(f"{'config':<26}{'GB/s':>8}{'$mem':>8}{'tok/s':>8}{'$/tok/s':>10}")
    work = WORKLOADS[0]
    for m in CANDIDATES:
        r = decode_phase(work, m)
        flag = "" if r.fits else "  (model does not fit)"
        print(
            f"{m.name:<26}{m.real_bytes_s / GB:>8.0f}{m.usd:>8.0f}"
            f"{r.tokens_s:>8.1f}{m.usd / r.tokens_s:>10.0f}{flag}"
        )
    print()
    print("Memory prices above are the pre-spike basis these models were built")
    print("on -- DDR5 $2.50/GB, LPCAMM2 at its $7.06/GB launch MSRP -- which is")
    print("where 'sockets are 3x cheaper per GB' comes from. At August 2026")
    print("retail that ordering inverts: DDR5 UDIMM $12-18.50/GB against")
    print("LPCAMM2 $8.26/GB. The conclusion drawn from this column is dated, not")
    print("wrong; decisions.md carries the reversal.")
    print()


if __name__ == "__main__":
    _table()
