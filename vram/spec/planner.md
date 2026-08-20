# Planner — choosing the split

The portable half of the project, and the half worth being right about. Given
a model, a method and a machine, it decides where every class of tensor lives,
how big a microbatch is, how many of them make a step, which layers are
recomputed, and — later — across how many devices the state is sharded. The
runtime executes that plan and measures it; the planner is what makes the plan
defensible before anything runs for six hours.

`train.py` is the roofline the planner optimises against.

## Inputs

| | |
|---|---|
| **model** | parameters, layers, hidden size, vocabulary; enough to size activations per token |
| **method** | bytes per parameter for weights and for state, FLOPs per parameter per token, and whether base weights get gradients |
| **device** | VRAM usable for tensors, host RAM available to page-lock, achieved FLOPS, link bandwidth (or none, on unified memory), sustained disk read **and write** |
| **constraints** | maximum tokens per optimizer step the training recipe will tolerate, and any tensor the user pins by hand |

Every device figure is measured, not looked up. `probe/` produces all four and
the planner should refuse a profile it did not measure — a plan built on a
vendor FLOPS number is off by the factor between peak and achieved, which on
this box is about two.

## Decision variables

- **placement**, per tensor class — weights, gradients, fp32 master, moment
  `m`, moment `v` — over `{VRAM, host, disk}`; on unified memory the middle
  option does not exist.
- **tokens per microbatch**, which buys hiding with activation memory.
- **accumulation steps**, which buys hiding for free but lengthens the step.
- **recompute policy** — which layers are checkpointed.
- **shards**, later: how many devices the state is split across.

## The shape of the problem, and why it is not one greedy pass

The variables are circular. A bigger microbatch lowers the fraction of time
spent waiting on streamed weights, but costs activation memory, which reduces
what can stay resident, which raises how much has to be streamed. Placement
depends on microbatch and microbatch depends on placement.

It is small enough not to matter. **Microbatch size is a one-dimensional
search over a short ladder** — 256, 512, 1024, 2048, 4096, 8192 tokens — and
inside each candidate the placement is the fractional knapsack `tiers.py`
already solves, ranked by time saved per byte of VRAM spent. Evaluate the
ladder, keep the best feasible point, report the runner-up. Six placements is
not an optimisation problem, it is a table.

What the planner must not do is average. A plan that is feasible "on average"
and exceeds VRAM on the widest layer fails at step one, so every constraint is
checked against the **worst** layer, not the mean.

## The two objectives, in order

**Feasibility first.** Resident bytes ≤ VRAM at the peak moment of the step,
which is the backward pass of the widest layer with its checkpoint boundary,
its recomputed working set, and whatever the optimizer window holds.

**Then throughput.** Maximise tokens per second:

    t_step = max( t_compute , t_weights , t_state )
    t_compute = flops_per_token × params × tokens_per_step / FLOPS
    t_weights = 2 × weight_bytes × params × microbatches / bandwidth(weights)
    t_state   = (2 × state_bytes + weight_bytes) × params / bandwidth(state)

`max` rather than sum is the claim the runtime has to earn: the three are
overlapped, not sequential. If the runtime cannot overlap them the planner is
optimistic by up to 3× and every plan in the charter is wrong — which is why
`RESULTS.md` compares predicted against measured step time before anything
else.

The token floors in `train.py` are the points where `t_weights` and `t_state`
fall under `t_compute`. Above them the plan is compute-bound and the offload
is invisible; below them the plan is honest about how much the GPU waits.

## The device axis

Designed in, not modelled yet — `shard()` in `train.py` is first-order. With
`n` devices, ZeRO-3 shaped:

    resident per device   = (weights + state) / n
    all-gather per microbatch = 2 × (n-1)/n × weights, per device
    reduce-scatter per step   = (n-1)/n × weights, per device

which trades the thing that is scarce (per-device memory) for a thing that may
be scarcer (interconnect). The trap on consumer hardware: **there is no
NVLink, and every card hangs off the same PCIe root complex**, so `n` devices
do not have `n` links — they share one, and the per-device bandwidth in the
formulas above falls as `1/n`. Adding a second card can therefore make a plan
slower, and the planner has to be able to say so.

That is also why the axis is designed in rather than bolted on: sharding
changes which tier a tensor lands in, so it belongs inside the placement
search, not after it.

## What it emits

A plan, in a form a human can argue with:

    placement       tensor class -> tier, with the bytes
    schedule        tokens/microbatch, microbatches/step, recompute policy
    prediction      t_compute, t_weights, t_state, and which one binds
    headroom        VRAM at the peak moment, and how close that was
    refused         the alternatives considered and why each lost

The last two lines are the point. A planner that emits only a winner cannot be
checked, and the failure mode of this whole design is a plan that looks fine
and thrashes — so the runner-up and the binding constraint travel with it.

On CUDA it should also emit an equivalent DeepSpeed or FSDP configuration.
Not to use, but so the same plan can be run under a mature implementation and
the runtime's step time compared against it. A runtime that loses to the
config it generated has found a bug in itself.

## What it must refuse

- A device profile it did not measure.
- A plan whose peak resident bytes exceed VRAM, even by a little — the answer
  is a smaller microbatch, not optimism.
- A plan below the token floor **without saying so**. Running starved is a
  legitimate choice when the recipe forbids a bigger step; hiding it is not.
  The plan reports the expected utilisation and the user accepts it.
