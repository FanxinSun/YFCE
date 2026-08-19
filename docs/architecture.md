# Architecture

## The claim

On-device language model decoding is memory-bound. Every weight crosses the bus
once per token, arithmetic intensity is 1–2 FLOP/byte, and the matrix units sit
idle waiting. Throughput is therefore set by two numbers and only two:

    tokens/sec  =  memory bandwidth  /  bytes per token

Bandwidth is width × data rate. Neither is a lithography variable — the M4 Max
leads Strix Halo 546 GB/s to 256 GB/s on twice the bus while sitting a node
behind it. Bytes per token is a representation choice, and it belongs to
whoever owns the format.

So a machine built on mature-node silicon can win on both terms, provided it is
designed as one system rather than assembled from parts that each assume
somebody else owns the layer below.

## Design invariant

**Weights are never materialised uncompressed.** Coded on flash, coded in
transit, coded in DRAM, expanded only in the decode block on the way to the
matrix units, and never written back.

Everything below follows from holding that invariant, and most of the value
comes from it rather than from any single component being clever. The
conventional path materialises a model four or five times — hub format,
download, convert, quantise, load, resident copy — with a conversion at every
boundary, because the storage format, the runtime format, and the DRAM layout
belong to three different parties. Owning all three collapses that to one
representation with no conversions at all.

## Three layers

### 1. Board — bandwidth from width

Socketed DDR5 DIMMs, 8 or 12 channels, rather than soldered LPDDR. Phones
solder LPDDR because they are battery- and volume-constrained; a mains-powered
appliance is neither and should not inherit that decision. DDR5 wins ~3× on
$/GB and ~1.3× on $/GB-per-second, and the capacity it buys cheaply is what
lets a large model stay resident at all.

The cost is idle power, latency, and board area — a 512-bit bus needs ~14
layers and a package in the 1,700-ball range. That routing problem is the
reason this layer exists as a designed thing rather than a purchased one.

### 2. Silicon — a decode block, not a bigger NPU

The matrix units are not the bottleneck and making them larger changes nothing.
The die does three things that matter: drive a wide PHY, hold enough SRAM to
stage weights, and expand the coded stream at bus rate.

The decode block is small, and the reason is structural. A BF16 weight is
1 sign + 8 exponent + 7 mantissa, and on real trained weights the
sign-and-mantissa plane measures 7.92 bits of entropy out of 8 — noise. It is
not coded at all; it moves as raw bytes. Every bit of compression comes from
the exponent plane, whose distribution is sharply peaked. So the entropy
decoder sees one symbol per parameter instead of one per byte, and at 229 GB/s
that is ~220 interleaved rANS lanes, ~2.6 mm² at 28 nm.

For scale: the PHY it feeds is tens of mm², and Groq's first-generation LPU was
725 mm² on GlobalFoundries 14 nm. The decoder is not the hard part of this chip.

### 3. Filesystem — a residency manager, not a POSIX layer

On the hot path this is not a filesystem. It is the thing that keeps coded
bytes moving from flash to the decode block without ever transforming them:

- **Random access at page granularity.** A one-byte read must expand one block,
  not one tensor. This is what makes expert swapping, adapter swapping, and
  demand-paged cold start possible instead of all-or-nothing loads.
- **Page-aligned coded blocks**, so a fault fetches a self-contained decodable
  unit and the DMA path needs no fixups.
- **Descriptor-ring feed** to the decode block: `(coded address, length,
  destination, format parameters)`, DMA in, expanded stream out to staging SRAM.
- **No transform anywhere in between.** The bytes written by the encoder are
  the bytes the hardware decodes.

`lmz` already implements the first two — 64 KiB blocks, page-aligned archives,
a page-mapped reader where a one-byte read expands no more than one block. That
is not a coincidence worth congratulating; random access imposes the same
constraint whether the consumer is a FUSE read or a DMA engine. It does mean
the existing block structure is the right one to build the hardware against.

## The format constraint the hardware imposes

Interleaved rANS bakes its lane count into the stream: a decoder cannot use more
lanes than the encoder interleaved, and fewer lanes means proportionally slower.

So **lane count is a format parameter, fixed before any tapeout**, and it has to
be chosen for the widest bus the product family will ever ship — not the first
one. Getting this wrong is not a performance bug, it is a format revision.

## What is deliberately not here

- **No claim that compression beats quantisation.** It does not. Lossless coding
  of an already-quantised format is worth ~5%, which is measured and is an
  entropy bound, not an engineering gap. The representation this system uses has
  to be designed as a representation, not bolted on after someone else's
  quantiser has run. That work is open — see `decisions.md`.
- **No compute story.** Prefill and vision encoders are compute-bound and this
  architecture does nothing for them. A machine built on these principles will
  have visibly worse time-to-first-token than a leading-edge SoC and should be
  positioned accordingly.
