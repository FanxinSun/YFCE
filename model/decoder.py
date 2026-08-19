"""Feasibility of the weight-decode block on a mature node.

This is the calculation that decides whether the chip is possible at all, so it
is worth stating the one structural result up front:

    The entropy decoder does not have to keep up with the memory bus.

A BF16 weight is 1 sign + 8 exponent + 7 mantissa. On real trained weights the
sign-and-mantissa plane measures 7.92 bits of entropy out of 8 -- genuine noise,
nothing to code. So it is not coded: it moves as raw bytes and never enters the
entropy path. Every bit of the compression comes from the exponent plane, whose
distribution is sharply peaked because trained weights cluster in magnitude.

That halves the symbol rate the decoder must sustain and lets the wide, boring
half of the datapath be a wire.

Numbers here are engineering estimates with their assumptions named, not
synthesis results. They are meant to answer "is this 3 mm2 or 300 mm2", which is
the question at this stage. `python3 decoder.py` prints the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1_000_000_000

# --- Measured inputs, from lmz on real Llama-3.1-8B BF16 ----------------------
BF16_SAVED = 0.347  # lossless, whole-checkpoint
SIGN_MANTISSA_BITS = 8  # passthrough plane
EXPONENT_BITS = 8  # entropy-coded plane

# --- 28nm-class process assumptions ------------------------------------------
GATES_PER_MM2 = 2_000_000  # random logic, routing-limited, conservative
SRAM_MM2_PER_MBIT = 0.25  # small distributed arrays incl. periphery
GATES_PER_LANE = 20_000  # rANS step: renorm, table lookup, multiply, shift
TABLE_BITS_PER_LANE = 8 * 1024  # 256-entry freq/cumfreq table, ~32b entries


def exponent_coded_bits() -> float:
    """Bits the exponent plane codes down to, implied by the measured total.

    If the 8-bit sign+mantissa plane is incompressible and the whole 16 bits
    compresses to (1 - saved), everything else came out of the exponent.
    """
    total_after = 16 * (1 - BF16_SAVED)
    return total_after - SIGN_MANTISSA_BITS


@dataclass(frozen=True)
class DecoderSpec:
    lanes: int
    clock_hz: float

    @property
    def symbols_s(self) -> float:
        """One rANS lane retires one symbol per cycle."""
        return self.lanes * self.clock_hz

    @property
    def logic_mm2(self) -> float:
        return self.lanes * GATES_PER_LANE / GATES_PER_MM2

    @property
    def sram_mm2(self) -> float:
        return self.lanes * TABLE_BITS_PER_LANE / 1e6 * SRAM_MM2_PER_MBIT

    @property
    def area_mm2(self) -> float:
        return self.logic_mm2 + self.sram_mm2


def required_symbol_rate(bus_bytes_s: float, bytes_per_param: float, f: float) -> float:
    """Exponent symbols/second needed for the bus to remain the limiter.

    One symbol per parameter. Note this is independent of model size: a bigger
    model runs proportionally slower and asks for the same symbol rate.
    """
    return bus_bytes_s / (bytes_per_param * f)


def lanes_for(symbol_rate: float, clock_hz: float) -> int:
    return int(-(-symbol_rate // clock_hz))  # ceil


def size_decoder(bus_bytes_s: float, clock_hz: float = 800e6,
                 bytes_per_param: float = 2.0, f: float = 1 - BF16_SAVED) -> DecoderSpec:
    rate = required_symbol_rate(bus_bytes_s, bytes_per_param, f)
    return DecoderSpec(lanes_for(rate, clock_hz), clock_hz)


def _sweep() -> None:
    exp_bits = exponent_coded_bits()
    print(f"Implied exponent plane: {EXPONENT_BITS} bits -> {exp_bits:.2f} bits "
          f"({exp_bits / EXPONENT_BITS:.1%} of original)")
    print("Sign+mantissa plane: uncoded passthrough, 7.92/8 bits of real entropy\n")

    print("Decode block sized against the bus it must not throttle (BF16, 800 MHz)")
    print(f"{'bus GB/s':>10}{'Gsym/s':>10}{'lanes':>8}{'logic':>9}{'sram':>8}{'total':>9}")
    for bw_gb in (129, 215, 229, 344, 459):
        spec = size_decoder(bw_gb * GB)
        rate = required_symbol_rate(bw_gb * GB, 2.0, 1 - BF16_SAVED)
        print(
            f"{bw_gb:>10}{rate / 1e9:>10.0f}{spec.lanes:>8}"
            f"{spec.logic_mm2:>8.2f}{spec.sram_mm2:>8.2f}{spec.area_mm2:>8.2f}"
        )

    print("\nSensitivity: area at 229 GB/s across clock and gate-count assumptions")
    print(f"{'clock':>10}{'lanes':>8}{'@15k':>9}{'@20k':>9}{'@30k':>9}")
    for clk in (600e6, 800e6, 1.0e9, 1.4e9):
        spec = size_decoder(229 * GB, clock_hz=clk)
        areas = [
            spec.lanes * g / GATES_PER_MM2 + spec.sram_mm2
            for g in (15_000, 20_000, 30_000)
        ]
        print(f"{clk / 1e6:>9.0f}M{spec.lanes:>8}"
              f"{areas[0]:>9.2f}{areas[1]:>9.2f}{areas[2]:>9.2f}")

    print("\nFor scale: a 512-bit LPDDR/DDR PHY at 28nm is tens of mm2, and Groq's")
    print("first-generation LPU was 725 mm2 on GlobalFoundries 14nm.")
    print("\nFormat consequence: interleaved rANS bakes the lane count into the")
    print("stream. A decoder cannot use more lanes than the encoder interleaved,")
    print("so lane count is a format parameter fixed before any tapeout, and it")
    print("has to be chosen for the widest bus the family will ever ship.")


if __name__ == "__main__":
    _sweep()
