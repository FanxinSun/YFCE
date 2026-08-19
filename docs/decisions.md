# Decision log

What is settled, what killed what, and what is still open. Entries are kept
when they turn out wrong — a route that was ruled out for a reason that later
dissolves needs its reason on record, not deleted.

## Settled

**Socketed DDR5 over soldered LPDDR.** ~3× cheaper per GB, ~1.3× cheaper per
GB/s, and the capacity is what keeps a large model resident. Costs idle power
and latency, which a mains-powered appliance can pay. `model/bom.py`
*Partly reversed 2026-08-17 — see below. What survives is "socketed"; what
does not is "DDR5", and the price ratio it rested on.*

**Bandwidth comes from bus width, not process node.** M4 Max (N3E, 512-bit,
546 GB/s) against Strix Halo (N4, 256-bit, 256 GB/s): one node apart, and the
gap is width. This is the premise the whole design rests on. `model/system.py`

**The entropy coder handles the exponent plane only.** Sign+mantissa measures
7.92 of 8 bits on real trained weights — noise, passed through raw. Halves the
symbol rate and makes the decode block ~2.6 mm² at 28 nm rather than something
that competes with the PHY for area. `model/decoder.py`

**Bigger NPU is not the answer.** Decode-phase arithmetic intensity is 1–2
FLOP/byte; the matrix units already idle. TOPS is the wrong figure of merit for
this machine and should not appear in its specification.

## Ruled out

**Losslessly compressing Q4_K_M to buy bandwidth.** Ceiling is 5.1%, and it is
an entropy bound, not an engineering gap — Q8_0's quant payload measures 7.64 of
8 bits. The quantiser already took those bytes. Only escapable by owning the
representation instead of coding what someone else's quantiser left behind.

**A software decoder in the bandwidth path.** lmz decompresses at 1.76 GiB/s on
a desktop CPU against a 229+ GB/s bus — two orders of magnitude short. The
decode path is hardware or it does not exist. Software compression remains
correct for the flash→DRAM hop, where it is faster than the storage.

**Inheriting phone memory architecture.** 64-bit LPDDR at 68 GB/s is a battery
decision. Copying it would discard the one advantage a board has.

## Answered

**Does the representation beat Q4_K at iso-perplexity? Yes, by 4.52% — which is
not enough.** E1, Llama-3.1-8B, wikitext-2, levels=16. Entropy-coding the indices
and scale planes gives 4.5000 → 4.2967 bpw at *identical* perplexity (9.5154).
A Lloyd-Max codebook is 0.03 perplexity **worse** for no bit saving, because
Q4_K's per-sub-block min/max already flattens the residual and leaves no shape
for a better codebook to exploit.

Corroborated: lmz measured 5.1% on real Q4_K_M files by an independent route.

**Consequence: the decode block is not justified.** f = 0.955 is a 1.047×
bandwidth multiplier — 4.7% more tokens/sec for a 2.6 mm² block, a hardware
decode path, and a format that bakes in its lane count. A faster DDR5 grade buys
more for nothing. That block was the only distinctive silicon in the design.

**And ratio trades against decode cost.** The representations that *do* beat
Q4_K_M materially (AQLM, QuIP#, imatrix k-quants) buy it with large codebook
lookups or per-block Hadamard transforms — the exact resource a mature-node
fixed-function decoder cannot spend. Ratio and cheap-decode are not independent
axes. See `experiments/e1_representation/RESULTS.md`.

## Open — ordered by how much they can still kill

**1. Does the picture change at 3 or 2 bits?**
E1 tested 4-bit only. Lower bpw is where the uniform grid is most clearly
suboptimal and a designed representation has the most room, so this is the one
cheap run that could still revive the chip. Until it is done, the E1 verdict is
"no at 4-bit", not "no".

**2. Is there a DDR5 or LPDDR5 PHY licensable at 28 nm?**
The mixed-signal PHY, not the logic, is what gates a mature-node design. High
data rates want the digital assist and equalisation that advanced nodes provide.
If nothing above LPDDR4X-4266 is licensable at 28 nm, the bandwidth ceiling
drops to ~229 GB/s and the compression term stops being optional. Unlike the
compute-in-memory case, DDR PHYs do *not* prefer mature nodes.

**3. Volume, now against an incumbent.** Amortised NRE crosses the bill of
materials at ~87,000 units (~$40M program against a ~$460 BOM; ~28,000 at
2026's ~$10/GB DRAM, ~103,000 if a portable forces 16/12 nm). Below that the
chip's development cost exceeds every physical part in the machine combined,
and an off-the-shelf board running the same stack is strictly cheaper. This is
the number that decides whether silicon happens, and no bandwidth result
changes it.

NVIDIA disclosed **RTX Spark** on 2026-08: 3 nm, 128 GB soldered LPDDR5X at
~300 GB/s, ~1 PFLOP, ~$2,899, in laptops and desktops from six OEMs this
autumn. It does not change any number above, and it changes the market for
them completely: the envelope this design targets is now a shipping consumer
product, and the units have to be sold against it. What is left to be better
at — width beyond 256 bits, capacity beyond 128 GB, and an operating system
nobody else is building — is in `blueprint.md` §1. What is left to be worse
at is prefill, by 50×.

**4. Lane count.** A format parameter with a hardware consequence, fixed before
tapeout, chosen for the widest bus the family will ever ship. Needs deciding
early and cannot be revised without a format break.

**5. Prefill.** Compute-bound, unaddressed, and visibly worse than a
leading-edge SoC. Either accept it in positioning or find an answer.

## Reversed

**"Socketed DDR5 over soldered LPDDR" — the half of it that said DDR5.**
2026-08-17. Two things dissolved the reason, and a third made it matter.

*The dichotomy was false.* **LPCAMM2** is LPDDR5X on a socketed,
compression-attached 128-bit module: 120 GB/s each at LPDDR5X-7500, up to
8533 at 1.05 V, 9600 sampling at 96 GB. Sockets and low power were never
actually opposed — the choice was only binary while the parts were DIMMs and
solder. Everything the settled entry claimed for sockets (capacity,
replaceability, buying width in modules) is available on LPDDR.

*The price ratio inverted.* The 3× was DDR5 at $2.50/GB against LPDDR at
$9.00. At August 2026 retail it is DDR5 UDIMM **$12–18.50/GB** against
LPCAMM2 **$8.26/GB** — the DIMM is now the expensive one. On the pre-spike
basis the original conclusion still holds (LPCAMM2 launched at $7.06/GB,
2.8× DDR5), so this is a dated conclusion rather than a wrong one, and
`model/system.py` prints both bases with the dates attached. The DRAM market
is what moved, not the argument.

*And the premise it depended on was optional.* "A mains-powered appliance is
neither battery- nor volume-constrained" is true of YFCE-1 and false of
YFCE-M, the portable variant (`blueprint.md` §1). At ~12 pJ/bit for a
terminated DIMM bus against ~5 for LPDDR5X, an 8B model at Q4 costs 432 mJ
per token on DIMMs and 180 on LPCAMM2 — 29 W against 12 W of DRAM traffic at
67 tok/s, before the ~80% standby saving that a machine which mostly listens
lives on.

**What replaces it:** the memory module is a *variant parameter*, not a
project-wide decision. DIMMs for the mains appliance while DDR5 is cheap per
GB; LPCAMM2 for anything portable, and for anything at all if the price
inversion holds. Both are socketed; both buy width in modules; neither
changes the format, the runtime or the invariant. **What does change is the
node question**: no 28 nm LPDDR5X-8533 PHY is known to exist, so the portable
variant probably buys a 16/12 nm program — E3/G2 gets heavier, and NRE goes
from ~$40M toward ~$48M (crossover ~87k → ~103k units at pre-spike DRAM).

**What is not reversed:** bandwidth still comes from width. RTX Spark is a
3 nm part with 256 bits; the M4 Max is two years older with 512 and 1.8× the
bandwidth. Four LPCAMM2 modules is 512 bits — 459 GB/s at 8533, 516 at 9600 —
which is the same argument this project started with, on different parts.
