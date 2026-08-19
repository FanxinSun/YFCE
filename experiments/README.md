# Experiments

Ordered so that the cheapest thing that can kill the design runs first.

## E1 — The representation gate

**Question.** Can a representation designed here beat `Q4_K_M` at equal quality?

This decides whether the chip exists. Everything in `docs/architecture.md`
assumes the coded stream is both small and good; if a purpose-designed format
cannot beat the incumbent quantiser at iso-perplexity, there is no ratio to
build silicon around and the project is software on other people's boards.

**Protocol.**

- Base: Llama-3.1-8B-Instruct, BF16, the same weights lmz is already measured on.
- Baseline: llama.cpp's own `Q4_K_M` build. Not a reimplementation — its real
  quantiser, including the error-minimising scale search.
- Eval: wikitext-2 perplexity, fixed context, fixed seed, plus one downstream
  task so the number is not purely a language-modelling artefact.
- Sweep the candidate representation across bits-per-weight; plot perplexity
  against bpw for both, on the same axes.

**Read.** The candidate must sit below Q4_K_M's curve — lower perplexity at
equal bpw, or fewer bpw at equal perplexity. Ties lose: matching an incumbent
format is not a reason to build hardware.

**Second axis, and it is not optional.** Every candidate must also be decodable
at bus rate. A representation that wins on the curve but needs a decoder that
cannot sustain ~175 G symbols/s in ~3 mm² has not won anything. Score
`(quality) × (1/f)` subject to that constraint, not ratio alone.

## E2 — The architecture, on hardware that already exists

**Question.** Does one-representation-end-to-end actually beat the conventional
convert-and-materialise path, on the same machine?

A Strix Halo box is ~$2k, 128 GB, 256 GB/s — the same memory architecture the
chip would have, purchasable today. The system thesis can be proved or killed in
software before any silicon commitment.

**Measure**, against stock llama.cpp on identical hardware and model:

- cold start, flash → first token
- model swap, and expert/adapter swap under an MoE
- resident footprint for the same model
- sustained tokens/sec, and whether the gap tracks `1/f` as predicted

**Read.** `model/system.py` predicts each of these. If measurements diverge from
prediction, the model is wrong and the numbers in `docs/` need revising before
anything else proceeds.

## E3 — PHY availability

Not an experiment, a procurement question, but it gates the bandwidth ceiling
and therefore the whole cost model: is DDR5 or LPDDR5 PHY IP licensable at
28 nm, and at what data rate and price? If the answer tops out at
LPDDR4X-4266, bandwidth caps near 229 GB/s and the compression term stops being
an optimisation and becomes load-bearing.

## Not yet scheduled

- Decode block RTL and synthesis, to replace `model/decoder.py`'s estimates
  with real area and timing.
- Prefill. Compute-bound, unaddressed, and the known weak flank.
