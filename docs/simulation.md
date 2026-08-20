# Simulation plan

The PC-side campaign from `blueprint.md` §7, stage by stage: what each stage
asks, how it is built, what it costs on this machine, and how its result is
read. It ends at "simulation complete"; the board and silicon steps that
follow are BOMs and next-plans in `blueprint.md` §8. S9 and S10 are the
AI PC's stages (blueprint §3): the model-as-OS contracts as a running
prototype, and — optionally — this PC booting into the model with no OS.

## The envelope

Everything here is sized to the machine this repository lives on:

| resource | available | consequence |
|---|---|---|
| CPU | Ryzen 7 9800X3D, 8C/16T, 96 MB L3 | Verilator, Ramulator2, Yosys run comfortably; RTL sims parallelise across seeds, not threads |
| RAM | 24 GB to WSL2 (31 GB host, `.wslconfig`), 8 GB swap | an 8B BF16 checkpoint (16 GB) fits once on the CPU; never two copies; P&R scoped to a slot cluster |
| GPU | RTX 5080, 16 GB, sm_120 | 8B-class models only; embeddings stay on CPU or use `accelerate` offload; 70B is out of reach here |
| CUDA | `/usr/local/cuda-13.2` (nvcc 13.2, driver 595) | targets sm_120; the apt `nvidia-cuda-toolkit` (12.4) cannot and must not be installed |
| Disk | ~700 GB free on the WSL ext4 disk; models on `/mnt/d/Models` (NTFS via 9P, slow) | copy working models to ext4; do all I/O measurements on ext4 |
| OS | Ubuntu 26.04 in WSL2 (WSLg gives a window), no Docker | storage timings are Hyper-V-virtualised — S7 reports relative numbers; S9's renderer draws into a WSLg window or a local browser tab |
| Firmware | UEFI PC; boots USB media | S10 can boot the runtime from a stick without touching either installed OS; QEMU + OVMF inside the env for development |
| Existing envs | `torch_env`, `exatrkx` (torch 2.12 cu130, transformers 5.8), system Python 3.14 | **read-only for this campaign** — nothing is installed into or upgraded in them |

**Isolation rule.** One new conda environment (`yfce`), one source-build
prefix (`~/opt/yfce`), one tools checkout directory (`~/src/yfce-tools`).
No `apt install` is required; where an apt package would do, the
conda-forge equivalent inside `yfce` is listed first and apt is the
alternative for the user to choose. Nothing in `/usr`, `/usr/local/cuda*`,
or the existing envs is modified.

## S0 — Environment

| component | how | size | used by |
|---|---|---|---|
| Python 3.12 env `yfce` | `conda create -n yfce python=3.12` | 0.5 GB | all |
| torch (cu13x wheel), transformers, safetensors, accelerate, numpy, scipy | pip inside `yfce`; `--index-url` for the cu130 wheel matching the driver | 6 GB | S1, S7 |
| lm-eval (`lm_eval`), `gguf` (llama.cpp's Python package), hypothesis, pytest, simpy, cocotb | pip inside `yfce` | small | S1, S2, S3, S5 |
| llama.cpp | `git clone` into `~/src/yfce-tools`, `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc`, build in-tree, **no install** | 2 GB | S1 (real `llama-quantize`, `llama-imatrix`, `llama-perplexity`), S7 (baseline) |
| lmz | submodule `lmz/` (`git submodule update --init`), `lmz/lmz-cli`; the `gpu-decoder` branch on its origin is the natural seed for S7-b | — | S2, S7 |
| Verilator, Icarus, Yosys, GTKWave | `conda install -c conda-forge verilator iverilog yosys gtkwave` (alt: apt has verilator 5.032, iverilog 12, yosys 0.52) | 1 GB | S3, S4 |
| OpenSTA | source build, `-DCMAKE_INSTALL_PREFIX=~/opt/yfce`; tcl/swig/eigen from conda-forge | 0.5 GB | S4 |
| Open libraries: Nangate45 (from OpenROAD-flow-scripts `platforms/`), ASAP7, sky130 HD, IHP sg13g2 | clone into `~/src/yfce-tools` | 3 GB | S4 |
| OpenROAD (optional, tier 3 of S4) | prebuilt Ubuntu 24.04 release unpacked under `~/opt/yfce` (`dpkg -x`), or the litex-hub conda channel — verify either works on 26.04; skip if not | 2 GB | S4 |
| CACTI 7 | source build in `~/opt/yfce` | small | S4 |
| Ramulator 2.0 (and/or DRAMsim3) | source build (C++20) in `~/opt/yfce` | small | S5 |
| Models | Llama-3.1-8B-Instruct BF16 (present), Q8_0 GGUF (present), Ministral-8B (present); generate Q4_K_M / Q3_K_M / Q2_K / IQ4_XS / IQ3_XXS / IQ2_XS GGUFs; optional second family (Qwen3-8B) | 40–60 GB on ext4 | S1, S7 |
| S9 model set | a 3B-class reflex model (Q4), whisper-small/turbo, a ~0.4B vision encoder, a small TTS, bge-m3 (present) — GGUF or safetensors | ~6 GB | S9 |
| peft + a 4-bit training backend (bitsandbytes or torchao — whichever ships a cu13x wheel at install time), datasets | pip inside `yfce` | small | S9 (QLoRA of the UI/tool-grammar adapter) |
| qemu-system-x86_64 (conda-forge) + an OVMF firmware image; gnu-efi source vendored into `~/src/yfce-tools` and built in-tree | inside the env / tools dir; nothing system-wide | 0.5 GB | S10 |

Half a day plus build time (llama.cpp CUDA build ~15 min on 8 cores).
Verify at the end: `python -c "import torch; torch.cuda.get_device_capability()"`
→ `(12, 0)`; `verilator --version`; `yosys -V`; `llama-quantize --help`.

## S1 — Representation: G1

**Question.** Does any representation that a mature-node fixed-function
block can decode at bus rate reach f ≤ 0.85 against the best cheap-decode
incumbent at the same operating point, at iso-quality on perplexity and one
downstream task (blueprint D5)?

E1 tested one family (non-uniform scalar + entropy) and it lost; its 3-bit
point (6.22%) is already in `experiments/e1_representation/RESULTS.md`.
This stage closes the 2-bit point cheaply and then tests the family that
actually beats Q4_K_M — but scores it on decode cost as well as ratio.

**E1b — 2-bit, the E1 family.** `run.py --levels 4` as it stands. Expected
~8%, far below the bar; ~1 hour; closes `decisions.md` open item 1.

**E1c — cheap-decode incumbents.** Quantise Llama-3.1-8B with llama.cpp's
own quantisers: Q4_K_M, Q3_K_M, Q2_K, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XS,
IQ2_XXS, with and without an importance matrix (`llama-imatrix` on a
wikitext-train slice). The IQ family is a lattice codebook (E8-class,
256–1024-entry 8- or 4-dim grids plus sign patterns) — decode is a table
lookup, a sign apply and a scale: cheap in hardware, and the best ratio the
incumbents have. Evaluate every variant **inside E1's harness**, not with
`llama-perplexity`: dequantise each GGUF tensor with `gguf-py`, load into
the transformers model in place of the projection weights, run
`ppl.py` on the same 250 × 512-token windows. That keeps the numbers on the
same axes as E1's table. `llama-perplexity` on the same files is a
cross-check only. Bits per weight are computed from the file, per tensor
class, not from the type's nominal figure.

**E1c-r — rotation + uniform grid.** One QuaRot-class candidate in the
harness: block-diagonal random Hadamard on input/output dimensions, GPTQ
(128 × 2k calibration tokens, ~1–2 h on the 5080 layer-by-layer), uniform
3- and 4-bit. Its decoder is the plain uniform dequant; the transform lands
on the activation side, once per token per layer, O(d log d). This is the
measurement `RESULTS.md` inferred rather than made.

**Entropy on top.** E1's entropy accounting (`quantize.py` order-0 and
conditioned) applied to the IQ and E1c-r index and scale planes, to see
whether any lossless margin survives on top of a good lossy stage.

**Decode-cost score, per variant.** Coded bytes per symbol → Gsym/s at 301
and 452 GB/s (blueprint D1); table bits; operations per weight (lookup,
sign, multiply-add, transform); slots at S = 8; projected area and power
from `model/decoder.py` (constants replaced by S4 as it lands). Reported as
a column next to bpw and perplexity — a candidate is scored on
`(quality) × (1/f)` **subject to** the D5 budget, as `experiments/README.md`
already demands.

**E1d — downstream.** lm-eval on the top three candidates and their
baselines: hellaswag, arc_challenge, winogrande, mmlu (5-shot). Fake-quant
model through lm-eval's HF backend with `accelerate` offloading the
embeddings. 8–16 GPU-hours, overnight. Then the top candidate on
Ministral-8B, so the result is not a Llama artefact.

**Budget.** ~5 min per (model, variant) perplexity point (77 s bf16, 200–270
s quantised, measured in `e1.log`); ~30 points ≈ 3 h; GPTQ 1–2 h; downstream
overnight; total ≈ 2 GPU-days plus 3–4 weeks of work.

**Outputs.** `experiments/e1c_incumbents/{CONDITIONS.md,RESULTS.md,*.json}`
in E1's format; the perplexity-vs-bpw table with a decode-cost column; the
G1 verdict against D5, and which fork it selects.

**Read.** Pass → Fork A or A′, and S3 grows the corresponding stage.
Fail → Fork B, S3 stays on the common path. Either way S5 and S7 proceed.

## S2 — Format v0 and the golden model

**Question.** What exactly is on the wire (blueprint I1), and can it be
encoded and decoded bit-exactly by a model the RTL is checked against?

**Deliverables.**

- `format/SPEC.md` — v0: container (lmz page-mapped lineage: 64 KiB coded
  blocks, 4 KiB-aligned payloads, chunk table keyed by destination offset),
  block header (format id, tensor id, offset in tensor, plane lengths, table
  ids, CRC32), plane order, table-by-reference section, rANS parameters
  (S = 8, 12-bit probabilities, 16-bit renormalisation, 32-bit states,
  4-byte flush), the descriptor record (I2), and one format id per fork
  (raw Q4_K passthrough is a valid format so Fork B uses the same
  container).
- `format/yfce/` — Python package: encoder, decoder, block index, descriptor
  generator, CRC; both stream layouts (shared interleaved as lmz today; one
  substream per state) behind a flag; C fast path by reusing `lmzcore.c`'s
  rANS for the lmz-compatible layout.
- `format/vectors/` — golden vectors: real Llama tensors of each shape class
  (attention, MLP, small); synthetic edge cases: single-symbol alphabet,
  frequency-1 symbols, empty and raw-fallback planes, maximum block, table
  change between consecutive blocks; each with expected decoded output and
  descriptors.

**Decisions settled by measurement here.** Per-state substreams vs shared
stream (overhead is ≤ 0.05% either way; choose per-state unless S3 shows the
8-way prefix-sum is free); tables by reference vs per block (per-block
tables cost ~0.8% of a 64 KiB block — most of the E1 gain — so by-reference
unless S3 shows table reload stalls dominate); 64 vs 128 KiB blocks (random
access granularity vs header overhead); which planes each fork's format id
carries.

**Tests.** Round-trip on all vectors; cross-decode against lmz for the
compatible layout; hypothesis property tests on random planes; a full 8B
archive encoded and decoded (Python rANS ~10 MB/s → ~10 min with the C
path, hours without — either is fine).

**Budget.** CPU only; 2–3 weeks.

## S3 — Decode block and DMA engine: RTL and verification

**Question.** Does a block-parallel decoder (blueprint D2) sustain one symbol
per state per cycle under realistic DMA behaviour, how many slots does the
bus need, and where does it stall?

**Language and tools.** SystemVerilog, synthesizable subset that is
Verilator-clean, Yosys `read_verilog -sv`-clean and Vivado-clean (T3 will
want it unchanged). Verilator as the primary simulator, Icarus for a
periodic second opinion on the unit level, cocotb for testbenches, GTKWave
for waves.

**Microarchitecture (parametric).**

- `slot` — S = 8 rANS state machines: 32-bit state; 12-bit slot extract;
  symbol lookup — table organisation is a parameter with three options to be
  costed in S4: `tree16` (16-entry cumulative-frequency compare tree, no
  SRAM: right for 4-bit index alphabets), `two-level-256` (16 groups × 16),
  `direct-4096` (4096-entry symbol map, 4 KiB per copy — simplest, largest);
  update `x = freq·(x >> 12) + slot − cum`; 16-bit renormalisation from the
  slot's input FIFO, per-state or via an 8-wide prefix-sum on the shared
  stream (parameter). Then the **unpack/dequant stage** (present in every
  fork): symbols → packed 4-bit indices (or lattice indices) + sub-block
  scales/mins → I3 tiles.
- `slot_array[B]` — dispatcher assigns descriptors to free slots; per-slot
  input FIFO fed by the DMA (bursts 64 B–4 KiB); output arbiter into staging.
- `dma_engine` — ring reader (I2), N-outstanding AXI read master, completion
  writer, CRC check, error path.
- `staging` — banked SRAM, write arbiter, one read port to an NPU stand-in
  (a rate-programmable sink, so back-pressure can be studied).
- Parameters: S, B, table type, stream layout, FIFO depths, AXI width,
  outstanding reads. Fork B build: entropy stage compiled out.

**Verification design.** The S2 golden model is the reference; cocotb drives
descriptors and a Python-side memory model that serves AXI reads with
programmable latency and bandwidth (coarse — real DRAM timing is S5's job),
and checks I3 output bit-exactly against S2's expected output.

Stimulus tiers: (1) unit — one slot, one block, every S2 vector; (2)
constrained-random — random tables, planes, lengths, back-pressure, DMA
latency, table changes; (3) real tensors; (4) full-model regression — every
block of an 8B archive through a B = 16 array (≈ 65 M array-cycles: under
an hour with a C++ harness, a few hours through cocotb — overnight is fine).
Coverage: Verilator line/toggle; functional points for slot occupancy, FIFO
full/empty, every renormalisation case, every table type, descriptor
error paths, back-pressure. Assertions in the RTL; the single rANS step
proven against its arithmetic definition with `yosys-smtbmc` (boolector/z3
are available) — a small formal target that pays for itself.

**Metrics.** Symbols/cycle/slot (target ≥ 7.5 of 8 sustained), stall
breakdown (input starvation, output back-pressure, table reload), latency
per block, ring depth at which throughput saturates, and the slot count for
301 and 452 GB/s at 800 MHz and 1 GHz.

**Budget.** RTL 4–6 weeks, testbench 2–3 weeks overlapping; sims from
minutes (unit) to overnight (full model); RAM small. Fork B build: 2–3 weeks
total.

**Read.** ≥ 7.5 sym/cycle/slot with realistic latency → the block-parallel
structure holds and D2 goes to `decisions.md`; slot counts feed S4's area
multiplication and S5's SRAM sizing. Below that → the slot is redesigned
before anything is synthesised.

## S4 — Synthesis, timing, area, power

**Question.** What does one slot cost, and does the array fit blueprint D5's
≤ 4 mm² / ≤ 1 W at 28 nm — as an honest bracket, since no 28 nm PDK is
available without an NDA?

**Tiers.**

1. Yosys `synth` generic → cell and gate counts per slot, per table type,
   per stream layout; `stat -liberty` against Nangate45, ASAP7, sky130 HD,
   IHP sg13g2 → area per slot in each library.
2. OpenSTA on the Yosys netlist + liberty → fmax and the critical path per
   library (post-synthesis, no wires; report with a 30% wire pessimism).
   Expected critical path: multiply-shift + table lookup + renormalisation
   select; if it does not close 800 MHz in Nangate45 it will not at 28 nm
   without pipelining, which changes the sym/cycle/slot picture — back to S3.
3. Optional: OpenROAD place-and-route of one slot and a 4-slot cluster in
   Nangate45 and ASAP7 for routed area, timing and power (`report_power`
   with a SAIF from the S3 sim). Hours; several GB of RAM; only if tiers 1–2
   leave the D5 answer ambiguous.

SRAM — table copies, FIFOs, staging — from CACTI 7 at 32 nm and 22 nm,
which replaces `decoder.py`'s 0.25 mm²/Mbit and `bom.py`'s 1.55 mm²/MB.

**Scaling.** Report every number per library plus the gate count; project to
28 nm through gate density (decoder.py's 2 M gates/mm² is conservative
against 28 nm HPC's ~2.5–3 M) and state the projection as a range. A point
value comes from a foundry PDK or C1, not from here.

**Budget.** 2–3 weeks; ≤ 16 GB RAM with the slot-cluster scoping.

**Read.** Area per slot × B, power per slot × B, fmax — against D5. This
is the second half of G1.

## S5 — Memory system and pipeline: G3, SRAM and ring sizing

**Question.** Does 8- or 12-channel DDR5 deliver ≥ 0.80 of peak under the
weight-streaming pattern with KV traffic and refresh, and how much staging
SRAM and ring depth keep the decoder fed?

**Ramulator 2.0.** DDR5-4800/5600/6400 device models, x8 devices, 1–2 ranks,
8 and 12 channels × 64-bit; controller FR-FCFS, open vs closed page, address
mapping with channel interleave at 256 B / 4 KiB / 64 KiB, 1× refresh; also
LPDDR4X-4266 (DRAMsim3 if Ramulator2 lacks it) for the G2-fails branch. Trace
frontend, traces generated in Python from the S2 block layout: sequential
64 KiB block reads striped across channels; concurrent KV read/write at 1–3%
of weight traffic (per context length); a little activation traffic. Sweep
mapping, ranks, page policy, data rate, channel count → achieved GB/s and
the efficiency that replaces `BUS_EFFICIENCY = 0.84`.

**Pipeline model.** A discrete-event model (SimPy in the env, or C++ if it
turns out slow): ring (depth D) → DMA (N outstanding, burst size) → memory
(Ramulator2-derived latency/bandwidth) → B slots at 8 sym/cycle × clock,
with a table-reload penalty on tensor change → staging SRAM (size, banks) →
NPU sink at a rate tied to GEMV at bus rate. Token time for 8B (68k blocks)
and 70B (600k blocks) — seconds to a minute per token in SimPy — with bus,
slot and SRAM utilisation; the minimum SRAM and ring depth for ≥ 0.98 of
bus-limited throughput; sensitivity to DMA latency and to the NPU sink rate.

**Method check on this PC.** Nothing here has 8 channels, so validate the
method: model this box's own dual-channel DDR5 in Ramulator2 (DIMM grade is
an input — read it from Windows, `wmic memorychip get speed`) and compare
with a STREAM-style measurement in WSL. Absolute 8-channel calibration is
T2's job.

**Budget.** Ramulator2 runs minutes each; the DES seconds to minutes; 3–4
weeks of work.

**Read.** G3 pass at ≥ 0.80. `BUS_EFFICIENCY` becomes a modelled function of
(grade, channels, mapping); M6's SRAM and I2's ring depth get numbers.

## S6 — Recalibration of models and docs

The simulations exist so that `model/*.py` and `docs/` say what was
measured. After S1–S5 and S7:

- `model/system.py` — per-configuration efficiency from S5; a symbol-rate
  helper keyed on coded bytes per symbol (blueprint D1); predictions for the
  T1/T2 platforms so the board runs can falsify it.
- `model/decoder.py` — block-parallel structure (slots × S), per-slot area
  and power from S4, SRAM from CACTI; the BF16 case kept as one row, not the
  headline.
- `model/bom.py` — DRAM $/GB as a scenario ($2.5 / 5 / 10 / 15), decoder and
  staging areas from S4, NRE crossover under each scenario.
- `docs/architecture.md` — the lane-count paragraph becomes block-parallel
  (D2) if S3/S5 confirm; symbol-rate numbers per D1.
- `docs/decisions.md` — D1–D6 in Settled/Ruled out/Reversed as the evidence
  says; open item 1 closed by E1b, open item 4 by D2.
- `experiments/README.md` and `README.md` — E1's "~175 Gsym/s in ~3 mm²"
  and the "$460 BOM" become the measured ranges.

One week.

## S7 — E2-lite: the software architecture on this PC (G4 at small scale)

**Question.** On one machine, does one-representation-end-to-end beat the
convert-and-materialise path on cold start, swap, and resident footprint —
and how far is a software decoder from the bus, on a GPU as well as a CPU?

**Setup.** Stock llama.cpp (CPU and CUDA builds) is the baseline; models on
ext4; page cache dropped between cold-start runs (needs root — or use a fresh
copy per run); the storage under WSL2 is a Hyper-V VHDX, so every number is
**relative**, same box, same run conditions.

**The YFCE-lite path.** An lmz page-mapped archive of the model — Q4_K_M
GGUF under lmz's block codec today, the S2 format when it lands — plus a
loader that demand-pages coded blocks and decodes them into the runtime's
weight buffers: CPU via `lmzcore.c` / the S2 C path; GPU via a CUDA rANS
block decoder (S7-b, about a week; it doubles as the fast stimulus
generator for S3's full-model regression). This is the "coded on flash,
coded in transit" half of the invariant, measurable here.

**S7-b has since become a project of its own: `vram/`.** The CUDA decoder was
built in lmz's scratchpad and reached 399–418 GB/s on this card, and the link
into that card measures 28.8 GB/s — so the decoder has 14× of headroom on the
path it feeds, and a half-week experiment turned into a residency engine with
four tiers and a placement solver. `vram/MEASURED.md` carries the conditions
and `vram/spec/residency.md` restates the four measurements below as what the
inference side has to prove.

**And then past S7 entirely.** The same tier machinery under *training* is a
different problem with a much better answer: a training microbatch reuses each
weight thousands of times, so at ~2,100 tokens the offload hides under the
compute and the link stops mattering, where decode is short of that by 4000×
(`decisions.md`, Answered). `vram/` has taken that as its main line — a
runtime for full fine-tuning a model too large for its GPU, on CUDA and
Metal — and `vram/README.md` is now its charter. S7 remains what it was: the
inference measurement, and the baseline the coded tier is scored against.

**Measurements**, each against stock on identical model and hardware:

1. **Cold start** — flash → first token, including the conversion the
   conventional path needs (HF → GGUF convert + quantise) and excluding it.
2. **Model swap** — A → B → A wall time; **expert swap** on a small MoE that
   fits 24 GB (OLMoE-1B-7B or Qwen1.5-MoE-A2.7B class).
3. **Resident footprint** — RSS + VRAM for the same model, coded-resident
   versus materialised.
4. **Sustained tok/s and 1/f** — CPU: coded-in-RAM decoded per token by
   `lmzcore` (expected far slower — the 1.76 GiB/s wall, already known);
   GPU: the CUDA block decoder decoding into a VRAM staging buffer per
   token, reported as GB/s decoded against the 5080's ~806 GB/s achievable.
   The expected result is that neither software path tracks 1/f. That is
   the point, and it is what T3-c exists to change.

   *Half wrong, 2026-08-20.* The expectation holds for the CPU, and for the
   GPU **against VRAM** — 399 GB/s of decode against 776 measured is 0.51×,
   so a coded tier inside VRAM costs speed and buys capacity. It fails for
   the GPU **against the link**, where 399 against 28.8 is 14× of headroom
   and 1/f is tracked exactly. The prediction was made against the wrong
   bus. `vram/`, and `decisions.md` under Reversed.

**Budget.** 3–4 weeks including S7-b; runs are minutes.

**Read.** G4 at PC scale, in relative terms; the README's
"materialised four or five times" claim becomes measured; a GPU data point
next to the CPU one for "software decode in the bandwidth path".

## S8 — Prefill and NPU sizing, and the kernel-model choice, on paper

Three or four days, no simulation. TOPS versus time-to-first-token for
512 / 2k / 8k prompts on 8B, 70B dense, and a 120B-A5B MoE; KV-cache bytes
per token versus context length as a fraction of weight traffic; vision and
speech encoders at 20 / 60 / 100 TOPS; the die and power cost of each NPU
size at 28 nm (`bom.py` floorplan). Output: one table that says, for the AI
PC, whether the kernel is a MoE on a 20 TOPS NPU or a dense model on a
60–100 TOPS one — blueprint §3 and D11 — plus the prefill paragraph for
`architecture.md`'s "what is deliberately not here".

## S9 — The AI OS on this PC: contracts and prototype (G6)

**Question.** Can the model-as-OS of blueprint §3 be written down as three
contracts — event/token bus (I7), interaction (I8: speech first, a
glanceable screen, the fixed touch/key set), tool ABI (I9: apps and
settings) — and carry an ordinary set of PC tasks **by voice** at these
token rates, with no unauthorised tool call under attack, including
attacks that arrive as sound? The charter and the first specs are in `os/`. Nothing here needs YFCE silicon; the
RTX 5080 stands in for the NPU and this PC's RAM for the DIMMs. What is
measured is the *shape* of the AI OS: tokens per interaction, latency
classes, state growth, safety — the numbers §3 asserts.

**Not bare metal, on purpose.** S9 runs as one userspace process under WSL,
so it can be built and measured quickly; S10 removes the OS underneath. The
process is written the way the runtime will be — one binary, one event
loop, static memory plan, no threads it does not own — so that S10 and the
YFCE-P runtime are the same code with different drivers.

**Build.**

- **Runtime prototype** (C, or Rust inside the env): block store over lmz
  page-mapped archives; models through llama.cpp as a library (CUDA);
  persistent KV via its state save/load; a grammar-constrained sampler
  (GBNF) that *is* the I7 demultiplexer — UI updates, tool calls, speech,
  memory writes are the only things the grammar can produce; the widget
  set and compositor rendering the I8 tree into a WSLg window or a local
  browser tab at 60 Hz, with caret/scroll/pointer handled by widgets; the
  I9 tools behind a capability guard with a trusted surface the model cannot
  draw over; an append-only log; the memory model (bge-m3, present) over
  the log and imported documents.
- **Model set that fits 16 GB:** kernel = Llama-3.1-8B Q4 (4.5 GB), reflex
  = a 3B-class Q4 (2 GB), streaming ASR (whisper-small/turbo), a wake-word
  and voice-activity model, a speaker-verification model, a ~0.4B vision
  encoder, a streaming TTS, bge-m3 — ~9–10 GB plus KV. Voice runs through
  the PC's own microphone and speakers (or a USB array, T4) via WSLg's
  audio path; echo cancellation is the prototype's, not the OS's. The kernel/reflex *split* is what is
  being tested, not the kernel's quality; the 5080's ~800 GB/s makes this
  8B "kernel" faster than the real one, so latency classes are reported in
  tokens as well as seconds and rescaled with `model/system.py`.
- **S9b — the app adapter.** QLoRA on the 8B (10–12 GB VRAM, hours) over
  synthetic data generated from the I8/I9 specs plus traces of the
  prototype's own sessions: teaches the UI language and tool grammar, and
  proves D8 — the adapter is installed as blocks and demand-paged, and
  switching it is a descriptor change. Its size (tens of MB) is the "app"
  size.

**Measure.**

1. **The task set** — 20 ordinary tasks a person does on a PC, **spoken**,
   half of them with hands busy (write and send a message, look something
   up and keep it, edit a document, sort photos, set a reminder, do
   arithmetic over a table, dictate, read a page aloud, import a file from a
   stick, change the volume / join a network / turn on do-not-disturb, …).
   Completed / not, completed hands-free / not, tool calls per task, tokens
   per step, wall time per step, ASR errors that changed an action, and
   where a person had to touch or type.
2. **Interaction cost** — the voice pipeline: wake → cue, end of utterance
   → transcript, → first spoken word (acknowledgement, answer), barge-in →
   silence, p50/p95, against `os/spec/interaction.md`'s classes; tokens per
   screen update by kind (heard/doing/need regions, list, table, diff); the
   diff-vs-full ratio; all on this PC and rescaled to 8 channels with
   `model/system.py`.
3. **State** — KV growth per hour of use; compaction policy (the model
   summarising its own context) and its cost; log and index sizes.
4. **Safety** — an injection set (≥ 200 cases: instructions hidden in
   pages, messages, images, file names, and **audio** — in played media, in
   a call, in a cloned owner voice, in another language, mid-request) run
   against every tool; count of unauthorised calls (must be zero, by
   construction of the guard, not the model's judgement) and of successful
   *authorised* misuse the trusted surface failed to make visible;
   speaker-verification false accept/reject under noise and playback.
5. **Boot and swap** — cold start to first reflex token, to kernel; app
   (adapter) swap latency; a second model set swapped in.

**Budget.** 5–6 weeks including S9b; runs are minutes; VRAM 16 GB is the
constraint (no 70B here — its numbers come from S8 and `system.py`).

**Outputs.** `os/spec/events.md`, `os/spec/models.md`, and the grammars
and signatures that complete `os/spec/interaction.md`, `tools.md` and
`trust.md`; the prototype in `os/runtime/`; the adapter in `os/adapters/`;
the task set, harness and injection set in `os/eval/`; `os/RESULTS.md` with
the five measurements against G6.

**Read.** G6 pass → the contracts freeze at v0 and D7–D10 go to
`decisions.md`; the tokens-per-update figure and latency classes feed S8's
NPU/kernel decision. Fail on the task set → the reflex/kernel split or the
UI language is redesigned before any hardware conversation about YFCE-P.
Fail on safety → the guard, not the model, is redesigned.

## S10 — Bare-metal proof on this PC: boots into the model, no OS (optional)

**Question.** Does the runtime stand up with nothing under it but firmware —
and what exactly must firmware provide? This is the "no Windows, no Linux"
claim made literal on the machine at hand, and it is what the YFCE-P boot
chain and the T2 bare-metal build inherit.

**Build.** The S9 runtime relinked as a UEFI application: UEFI is the boot
ROM here (GOP framebuffer for the compositor, SimpleText/Pointer protocols
for input, Block I/O to read archives, SNP for the network tool, timers).
CPU-only inference — no GPU driver exists without an OS — with an AVX-512
GEMV over the PC's dual-channel DDR5 (~70 GB/s achievable → 8B Q4 at ~10–15
tok/s, bandwidth-bound like everything else here). Freestanding C, `gcc`
that is already installed, gnu-efi vendored and built in-tree; developed and
tested in QEMU + OVMF inside the env; run for real from a USB stick.
Reads only from the stick; writes nothing to any internal disk; the
installed Windows and Linux are untouched and the machine returns to them on
reboot.

**Measure.** Time from power-on to first token; the list of firmware
services actually used (this list is the YFCE-P boot-ROM/runtime contract);
tok/s versus the STREAM number of the same box (S5's method check); which
S9 measurements survive with the OS removed — they should all survive, or
the prototype was leaning on Linux somewhere.

**Budget.** 4–8 weeks. Optional: it changes no gate on its own, but it is
the cheapest existence proof of D7 and it is the image T2 boots.

**Read.** Boots and answers → D7 is demonstrated on commodity hardware; the
firmware-service list goes into `blueprint.md` §3 as the runtime's contract
with the SoC. Fails on some device → that device's driver joins the staged
list in §3, not the list of things a host OS must provide.

## Order, dependencies, exit

    S0 ──┬── S1 (G1) ─────────────────┐
         ├── S2 ── S3 ── S4 ──────────┤
         ├── S5 (G3) ─────────────────┼── S6 ── simulation complete
         ├── S2 ── S7 (G4-lite) ──────┤
         └── S9 (G6) ── S10 (opt.) ───┘
              S8 anywhere; S9 feeds S8's kernel/NPU choice

S1, S5, S7 and S9 do not wait for each other; S3 starts on the common path
(DMA, unpack/dequant, staging) the day S2's descriptor record exists and
grows the entropy or lattice stage when G1 says which. About seven months
serial for one person, four with the parallel branches running; S10 is
extra and optional.

**Simulation complete means:** format v0 spec and golden model with vectors;
RTL frozen at v0 with ≥ 90% line and functional coverage and the metrics
above; PPA brackets per library with the 28 nm projection stated as a range;
Ramulator2 efficiency curves and DES sizes for SRAM and ring; models and
docs recalibrated (S6); the E2-lite report; the AI-OS contracts I7–I9 at v0
with the S9 measurements behind them and, if S10 ran, a stick that boots
this PC into the model. That package is what `blueprint.md` §8's T1–T4 and
C1 consume, and nothing in it required buying anything.
