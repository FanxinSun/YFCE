# YFCE

A memory-first inference appliance on mature-node silicon: chip, board, and
filesystem designed as one system, targeting the cheapest hardware that runs a
large model at usable speed.

## Why this can work

On-device language model decoding is memory-bound. Throughput is bandwidth
divided by bytes-per-token, the matrix units idle either way, and **neither term
is a lithography variable**:

| system | node | bus | MT/s | GB/s |
|---|---|---|---|---|
| Phone SoC | N3 | 64-bit | 8533 | 68 |
| AMD Strix Halo | N4 | 256-bit | 8000 | 256 |
| NVIDIA RTX Spark (N1X) | N3 | 256-bit | 9600 | ~300 |
| Apple M4 Max | **N3E** | 512-bit | 8533 | 546 |

M4 Max leads Strix Halo by 2.1× while sitting a node behind it. The gap is bus
width. Bandwidth is bought with pins, and bytes-per-token is a representation
choice belonging to whoever owns the format.

NVIDIA's RTX Spark, disclosed August 2026, is the same argument from the other
side: a 3 nm part with a petaflop of compute, 128 GB of unified memory, and
**256 bits** — half the bus of an Apple laptop chip two years older. The
envelope this project targets is now something you can buy; the width is still
on the table. `docs/blueprint.md` §1 says what that leaves.

Groq's first-generation LPU is the existence proof for the silicon half: a
competitive inference chip on GlobalFoundries 14 nm, winning on memory
architecture rather than density.

## Where it stands

Design-stage. Three cost and feasibility models, no RTL, no board.

**What the models say so far:**

- A 128 GB / 301 GB/s machine on 8-channel DDR5-5600 has a **~$460 bill of
  materials**, on a ~107 mm² 28 nm die that costs ~$5 in silicon — at the
  pre-spike DRAM prices these models were built on. At August 2026 prices the
  same machine is ~$1,420, because it is mostly DRAM and so is everything it
  competes with.
- The weight-decode block is **~2.6 mm² at 28 nm** — not the hard part. It is
  small because only the exponent plane is entropy-coded; sign and mantissa
  measure 7.92 of 8 bits on real weights and move as raw bytes.
- Socketed DDR5 beat soldered LPDDR by ~3× on $/GB — *on 2025 prices, and
  against the wrong alternative*. **LPCAMM2 is socketed LPDDR5X**, and at
  August 2026 retail it is the cheaper of the two per GB. Sockets survive;
  DDR5 is now a variant choice, and a portable variant (`YFCE-M`) takes the
  LPDDR side. `docs/decisions.md`, Reversed.
- **Amortised NRE crosses the BOM at ~87,000 units** (~$40M program). Below that
  the chip's development cost exceeds every physical part in the machine
  combined. This, not any bandwidth number, decides whether silicon happens.

**What is not yet known** — in the order that can kill the design:

1. Whether a purpose-designed representation beats `Q4_K_M` at iso-perplexity.
   If not, there is no ratio to justify a chip. `experiments/README.md` E1.
2. Whether a DDR5/LPDDR5 PHY is licensable at 28 nm. The mixed-signal PHY, not
   the logic, is what gates a mature-node design — and an LPDDR5X-8533 PHY, at
   any node, for the portable variant.
3. Volume — now to be sold against an RTX Spark on six OEMs' shelves.

## Layout

    docs/blueprint.md      the whole plan: the appliance and the AI PC (the model
                           as the OS, no Linux/Windows), modules, interfaces, gates
                           and forks, the PC-side campaign, then BOMs and next-plans
                           for board, FPGA and test chip
    docs/simulation.md     the PC-side campaign in detail, S0-S10, sized to this machine
    docs/build_pdf.py      typesets both into docs/YFCE-blueprint.pdf
    os/                    the AI OS: voice-first, limited touch/keyboard, models control
                           apps and system settings — charter and specs
    lmz/                   the codec and residency layer (submodule)
    docs/architecture.md   the design, its invariant, and what it deliberately omits
    docs/decisions.md      settled / ruled out / open, with reasons kept
    model/system.py        roofline: bus width, codec, model -> tokens/sec
    model/decoder.py       rANS decode block area and throughput at 28nm
    model/bom.py           build cost, development cost, and the volume crossover
    experiments/           what to measure, cheapest-to-kill first

Python 3.10+, no dependencies. Each model runs standalone:

    python3 model/system.py
    python3 model/decoder.py
    python3 model/bom.py

## Relationship to lmz

[`lmz`](lmz/) — a git submodule of this repository (`git submodule update
--init` after cloning) — is the codec and the residency layer, not the product. Its
existing block structure — 64 KiB page-aligned blocks, a one-byte read expanding
one block — is already the structure the hardware DMA path needs, because random
access imposes the same constraint on a FUSE read and a descriptor ring.

What lmz does **not** carry over is lossless coding of an existing quantised
format: that ceiling is 5.1% and it is an entropy bound, not an engineering gap.
The representation for this system has to be designed as one. See E1.

## The honest summary

The physics is right and the parts are cheap. The machine is gated on one
experiment and one number: whether a designed representation beats `Q4_K_M` at
equal quality, and whether anything here ever ships 87,000 units. NVIDIA has
since shipped the envelope — 128 GB at ~300 GB/s, on 256 bits — which settles
the premise and takes the market for it in the same move, and leaves width,
capacity and the operating system as the things left to be better at. Until
the first question is answered, the correct move is unchanged and now cheaper
to justify: E2 — prove the architecture in software on a board someone else
already built, and measure theirs while you are there.
