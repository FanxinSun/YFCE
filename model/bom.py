"""Cost model. The target is the cheapest hardware that hits a throughput
number, so cost is a first-class output of the design, not an afterthought.

The finding that drives the whole board design is in `_compare()`: for a
mains-powered appliance, socketed DDR5 DIMMs beat soldered LPDDR on both $/GB
and $/GB/s by roughly 3x. Phones solder LPDDR because they are battery- and
volume-constrained. This is neither, so it should not inherit that decision.

`python3 bom.py` prints the comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- Process economics, 28nm-class -------------------------------------------
WAFER_USD = 3_000.0
WAFER_RADIUS_MM = 150.0
DEFECT_DENSITY_PER_CM2 = 0.07  # mature node, well-characterised


def dies_per_wafer(die_mm2: float) -> float:
    """Gross die count, with the usual edge-loss correction."""
    r = WAFER_RADIUS_MM
    return math.pi * r * r / die_mm2 - math.pi * 2 * r / math.sqrt(2 * die_mm2)


def yield_murphy(die_mm2: float) -> float:
    ad = die_mm2 / 100.0 * DEFECT_DENSITY_PER_CM2
    if ad == 0:
        return 1.0
    return ((1 - math.exp(-ad)) / ad) ** 2


def die_usd(die_mm2: float) -> float:
    good = dies_per_wafer(die_mm2) * yield_murphy(die_mm2)
    return WAFER_USD / max(good, 1e-9)


# --- Floorplan ----------------------------------------------------------------
# Area estimates for a 28nm-class die. The PHY dominates, which is the honest
# cost of the wide bus the whole thesis depends on.

@dataclass
class Floorplan:
    phy_mm2_per_64bit: float = 5.5  # DDR5/LPDDR PHY, per 64-bit channel
    npu_mm2: float = 15.0  # ~20 TOPS int8 at 28nm; decode is not compute-bound
    sram_mm2_per_mb: float = 1.55  # incl. periphery, 28nm
    sram_mb: float = 16.0
    decoder_mm2: float = 3.0  # see decoder.py
    misc_mm2: float = 20.0  # CPU cores, IO, fabric, PLLs

    def total(self, bus_bits: int) -> float:
        return (
            self.phy_mm2_per_64bit * bus_bits / 64
            + self.npu_mm2
            + self.sram_mm2_per_mb * self.sram_mb
            + self.decoder_mm2
            + self.misc_mm2
        )


# --- Board and package --------------------------------------------------------

def pcb_layers(bus_bits: int, module: str) -> int:
    """Routing a wide bus is the reason this needs a real motherboard.

    The stub allowance belongs to DIMMs, not to sockets in general: an LPCAMM2
    module is compression-attached point-to-point next to the package, which is
    what lets a socketed memory system go in something portable at all.
    """
    base = 8 if bus_bits <= 128 else 10 if bus_bits <= 256 else 14
    return base + (2 if module == "dimm" else 0)


def pcb_usd(bus_bits: int, module: str, area_cm2: float = 400.0) -> float:
    return 0.011 * area_cm2 * pcb_layers(bus_bits, module)


def package_usd(bus_bits: int) -> float:
    """FCBGA, ball count driven by the memory bus plus power delivery."""
    balls = bus_bits * 2.2 + 600
    return 0.9 + balls * 0.012


@dataclass
class Config:
    name: str
    bus_bits: int
    mt_s: int
    usd_per_gb: float
    capacity_gb: int
    # 'dimm' socketed DDR5 | 'lpcamm' socketed LPDDR5X module | 'solder'
    module: str
    floorplan: Floorplan = field(default_factory=Floorplan)
    area_cm2: float = 400.0  # a motherboard; a portable is roughly half
    # PSU, thermal, chassis. A portable pays for a battery and a panel instead
    # of a PSU, and pays more: ~99 Wh at $0.5/Wh plus a small panel.
    chassis_usd: float = 45.0

    @property
    def die_mm2(self) -> float:
        return self.floorplan.total(self.bus_bits)

    @property
    def bandwidth_gb_s(self) -> float:
        return self.bus_bits / 8 * self.mt_s * 1e6 * 0.84 / 1e9

    def bom(self) -> dict[str, float]:
        parts = {
            "die": die_usd(self.die_mm2),
            "package": package_usd(self.bus_bits),
            "dram": self.usd_per_gb * self.capacity_gb,
            "pcb": pcb_usd(self.bus_bits, self.module, self.area_cm2),
            "psu/thermal/chassis": self.chassis_usd,
        }
        parts["total"] = sum(parts.values())
        return parts


CONFIGS = [
    Config("8ch DDR5-5600, 128GB", 512, 5600, 2.50, 128, "dimm"),
    Config("12ch DDR5-5600, 192GB", 768, 5600, 2.50, 192, "dimm"),
    Config("8ch DDR5-4800, 128GB", 512, 4800, 2.20, 128, "dimm"),
    Config("512b LPDDR4X-4266, 64GB", 512, 4266, 5.00, 64, "solder"),
    Config("256b LPDDR5-6400, 64GB", 256, 6400, 7.00, 64, "solder"),
    # Portable variants. LPCAMM2 at its launch MSRP, to stay on the same
    # pre-spike basis as the rows above; half the board area; a battery and a
    # panel instead of a PSU. What these do not show is the node: no 28nm
    # LPDDR5X-8533 PHY is known to exist, so a portable probably buys a
    # 16/12nm program and the NRE lines below are the wrong ones for it.
    Config("2x LPCAMM2-8533, 128GB", 256, 8533, 7.06, 128, "lpcamm",
           area_cm2=200.0, chassis_usd=145.0),
    Config("4x LPCAMM2-8533, 256GB", 512, 8533, 7.06, 256, "lpcamm",
           area_cm2=200.0, chassis_usd=145.0),
]


# --- Development cost ---------------------------------------------------------
# The marginal die is cheap. The program that produces it is not, and at the
# volumes a new entrant ships, amortised NRE dominates every other line in the
# BOM combined. This is the number that decides whether to tape out at all.

NRE = {
    "mask set (28nm)": 1_100_000,
    "DDR5 PHY IP licence": 2_500_000,
    "other IP (cores, fabric)": 1_800_000,
    "EDA licences over program": 3_500_000,
    "engineering, ~40 eng x 2.5 yr": 24_000_000,
    "emulation, verification": 2_200_000,
    "test dev, bring-up, one respin": 5_000_000,
}
NRE_TOTAL = sum(NRE.values())


def unit_cost(config: Config, volume: int, nre: float = NRE_TOTAL) -> float:
    return config.bom()["total"] + nre / volume


def nre_crossover(config: Config, nre: float = NRE_TOTAL) -> float:
    """Volume at which amortised NRE stops exceeding the bill of materials."""
    return nre / config.bom()["total"]


def _compare() -> None:
    print("Build cost at volume, and what it buys\n")
    print(f"{'config':<26}{'die':>7}{'mm2':>7}{'dram':>7}{'pcb':>6}"
          f"{'total':>8}{'GB/s':>7}{'$/GB/s':>9}")
    for c in CONFIGS:
        b = c.bom()
        print(
            f"{c.name:<26}{b['die']:>7.0f}{c.die_mm2:>7.0f}{b['dram']:>7.0f}"
            f"{b['pcb']:>6.0f}{b['total']:>8.0f}{c.bandwidth_gb_s:>7.0f}"
            f"{b['total'] / c.bandwidth_gb_s:>9.2f}"
        )

    print("\nMemory cost per unit of bandwidth, isolated:")
    print(f"{'grade':<22}{'$/GB':>8}{'GB/s per 512b':>15}{'$/GB/s (mem only)':>19}")
    for c in CONFIGS:
        if c.bus_bits != 512:
            continue
        bw_512 = 512 / 8 * c.mt_s * 1e6 * 0.84 / 1e9
        mem = c.usd_per_gb * c.capacity_gb
        print(f"{c.name.split(',')[0]:<22}{c.usd_per_gb:>8.2f}"
              f"{bw_512:>15.0f}{mem / bw_512:>19.2f}")

    print("\nDDR5 gives up latency and idle power and wins on both $/GB and")
    print("$/GB/s. For a plugged-in appliance that is the right side of the")
    print("trade, and it is only available because this is a board, not a phone.")

    ref = CONFIGS[0]
    bom_total = ref.bom()["total"]
    print(f"\n\nDevelopment cost, and what volume it needs ({ref.name})")
    print(f"{'NRE line':<34}{'$M':>8}")
    for k, v in NRE.items():
        print(f"{k:<34}{v / 1e6:>8.1f}")
    print(f"{'total':<34}{NRE_TOTAL / 1e6:>8.1f}")

    print(f"\n{'volume':>10}{'BOM':>9}{'NRE/unit':>12}{'cost/unit':>12}")
    for v in (1_000, 10_000, 100_000, 1_000_000):
        print(f"{v:>10,}{bom_total:>9.0f}{NRE_TOTAL / v:>12,.0f}"
              f"{unit_cost(ref, v):>12,.0f}")

    print(f"\nNRE stops dominating the BOM at ~{nre_crossover(ref):,.0f} units.")
    print("Below that, the development cost of the chip exceeds every physical")
    print("part in the machine put together, and an off-the-shelf board running")
    print("the same software stack is strictly cheaper. That threshold, not any")
    print("bandwidth number, is what decides whether this design gets fabricated.")


if __name__ == "__main__":
    _compare()
