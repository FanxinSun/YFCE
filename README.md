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
**256 bits** — half the bus of an Apple laptop chip two years older.

Which carries a consequence this project was slow to draw. If bandwidth is
bought with pins, **nobody holds a position on bandwidth, including us**. The
envelope commoditises — $1,000–1,500 for 128 GB at 300+ GB/s by about 2028 is
the conservative reading — and a chip program started now delivers parts in
2029–30, after the curve arrives. So the hardware is assumed to get cheap, and
what is built here is the layer that does not commoditise: the residency
model, and an operating system made of models. `docs/blueprint.md` §1.

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
  combined. This, not any bandwidth number, decides whether silicon happens —
  and against a commodity curve that reaches the same envelope first, the
  current answer is that it does not. The chip is an option the plan defers,
  not a destination it works toward.

**What is not yet known** — in the order that can kill the design:

1. Whether one-representation-end-to-end beats convert-and-materialise on the
   same machine: cold start, model and expert swap, resident footprint. Since
   the bandwidth term commoditises and E1 measured only 4.5% of headroom in the
   bytes-per-token term, **residency is what the format is worth**, and this is
   now the load-bearing experiment. `docs/simulation.md` S7, and `vram/` is
   where it is being built.
2. Whether a model can be an operating system — carrying an ordinary day of
   work by voice, safely, at these token rates. `os/`, S9.
3. Whether any market condition justifies silicon before the commodity curve
   arrives. Currently none does.

## Layout

    docs/blueprint.md      the whole plan: the appliance and the AI PC (the model
                           as the OS, no Linux/Windows), modules, interfaces, gates
                           and forks, the PC-side campaign, then BOMs and next-plans
                           for board, FPGA and test chip
    docs/simulation.md     the PC-side campaign in detail, S0-S10, sized to this machine
    docs/build_pdf.py      typesets both into docs/YFCE-blueprint.pdf
    os/                    the AI OS: voice-first, limited touch/keyboard, models control
                           apps and system settings — charter and specs
    vram/                  SSD and RAM as VRAM: a runtime for training a model too
                           large for its GPU, CUDA and Metal — charter, two rooflines,
                           measured; and the inference half that started it
    lmz/                   the codec (submodule)
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
--init` after cloning) — is the codec, not the product; `vram/` is the
residency layer that consumes it. Its
existing block structure — 64 KiB page-aligned blocks, a one-byte read expanding
one block — is already the structure the hardware DMA path needs, because random
access imposes the same constraint on a FUSE read and a descriptor ring.

What lmz does **not** carry over is lossless coding of an existing quantised
format: that ceiling is 5.1% and it is an entropy bound, not an engineering gap.
The representation for this system has to be designed as one. See E1.

## Relationship to vram

[`vram/`](vram/) is the residency layer — blueprint M3 — built for hardware
somebody else made, which under D14 is the base case and not a detour. Its
main line is a **runtime for training a model too large for the GPU it is
on**: weights, gradients and optimizer state placed across VRAM, host RAM and
SSD, with the traffic scheduled to disappear under the compute. CUDA and
Metal, full fine-tuning rather than adapters.

One measured ratio decides it. Streaming a weight costs `bytes/bandwidth`; the
compute it feeds is `2 × tokens_per_microbatch` FLOPs per byte. On this box —
119 TFLOP/s BF16 measured against a 28.8 GB/s PCIe link — that is 4132 FLOP
per byte, so **~2,100 tokens per microbatch and the link stops existing**.
Inference decode manages 1 FLOP per byte against the same requirement, short
by a factor of four thousand. The identical offload is hopeless one side of
that line and nearly free on the other, which is why the project is about
training.

And training's memory is not where the appliance's is. An 8B full fine-tune is
129 GB, of which the weights are 16 and Adam is 112 — so the first lever is
not offload at all but the optimizer's own precision, 3.5× before a byte
crosses a bus. That is why it has to be a runtime and not a configuration
generator.

The honest part: on CUDA, DeepSpeed ZeRO-Infinity already does this well.
**Off CUDA there is nothing** — a 64 GB Mac can hold a 70B QLoRA today and
cannot full-fine-tune an 8B, and the arithmetic says it should be able to.
That gap is what argues for building rather than configuring.

`vram/` also carries the inference half that started it, because it was
measured: a coded tier inside VRAM, 0.51× the speed of raw weights at 1/ratio
of the space and 9.3× faster than anything off the card, and one reopened
ruling in `docs/decisions.md` under Reversed.

## The honest summary

The physics is right and the parts are cheap — and getting cheaper for
everyone, which is the whole point. The premise says bandwidth is bought with
pins; the honest consequence is that the bandwidth cannot be owned, so the
machine was never the product. What is left is what the machine was always in
service of: weights that are never materialised, models that are resident and
swap at page granularity, and an OS made of models rather than one that hosts
them. That runs on hardware other people are racing to make cheap.

The chip stays in the plan as an option with its conditions written down, in
case a representation or a residency result turns up that a commodity host
cannot reach. Until then the correct move is E2 — prove the architecture in
software on hardware someone else already built — and it is now the *whole*
move, not a step toward a tapeout.
