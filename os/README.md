# os — the AI OS

The operating system of the AI PC (`docs/blueprint.md` §3), as a directory:
its charter, the contracts it is built from, the prototype, and the
measurements that decide whether it works. Nothing here runs Windows or
Linux; the model set is the OS and the runtime under it is firmware.

## Charter

**Voice first.** The machine is operated by speaking to it and it answers by
speaking back, with a glanceable screen that shows what it heard, what it is
doing, and what it needs. Voice is the primary channel because it is the one
a language model is native to, and because it is the one that needs no
learning: the person says what they want and the model does it.

**Limited touch and keyboard.** Touch and keys exist, but as a small, fixed
set the runtime owns and the model cannot extend: wake / push-to-talk;
confirm and cancel (the trusted surface); a numeric pad for secrets that must
never be spoken; volume; tap and scroll on whatever the model shows;
correction of a transcript by touch; and an optional USB keyboard for text
when speaking is inappropriate. There is no desktop, no windows, no menus,
no settings screens to find. "Limited" is a design decision, not a
shortfall: everything a person cannot do by hand they do by voice, and
everything reactive (caret, scroll, playback) is a widget in the runtime, so
the model is never in the keystroke path.

**Models control the apps.** There are no applications in the conventional
sense. An app is an adapter bundle — weights, a skill document, tool grants,
UI and voice templates — installed as blocks and demand-paged by the
residency manager. The kernel model launches, uses, switches and composes
apps by tool calls; the person only ever asks for outcomes.

**Models control the system settings.** Settings are tools. Volume,
brightness, do-not-disturb, timers, network, Bluetooth, power, updates,
storage, language and voice, accessibility, privacy — all read and written
by the model through the tool ABI, in tiers: everyday settings the reflex
model may change on request; consequential ones that need a spoken
confirmation *and* the physical confirm key; and a short list that no model
touches at all — microphone and camera mute (a hardware switch), factory
reset, boot medium, and the capability grants themselves.

## How it fits the blueprint

| blueprint | here |
|---|---|
| §3 the model set as the OS image | `spec/models.md` — roles, resident set, what runs where |
| I7 event and token bus | `spec/events.md` — the tagged spans in and the grammar out |
| I8 UI language → the interaction contract | `spec/interaction.md` — voice in/out, the glanceable screen, the fixed touch/key set, latency classes |
| I9 tool ABI | `spec/tools.md` — apps and settings as tools, in tiers |
| D10 trust | `spec/trust.md` — capabilities, trusted surface, audio injection, privacy |
| S9 (simulation plan) | `runtime/` the prototype, `adapters/` the app adapters, `eval/` the task set, latency harness and injection set, `RESULTS.md` |
| S10 | `runtime/` relinked as a UEFI application: this PC boots into the model |

## Layout

    os/README.md            this charter
    os/spec/interaction.md  voice-first interaction: channels, the fixed touch/key set, latency classes
    os/spec/tools.md        apps and settings as tools; tiers; what no model may touch
    os/spec/trust.md        capabilities, trusted surface, audio injection, privacy
    os/spec/events.md       (S9) I7 as text and grammar
    os/spec/models.md       (S9) the resident set and roles
    os/runtime/             (S9) one process, one event loop; (S10) the same code, no OS under it
    os/adapters/            (S9b) the UI/tool-grammar adapter and app adapters
    os/eval/                (S9) task set, latency harness, injection set incl. audio
    os/RESULTS.md           (S9) measurements against G6

The three specs that exist today fix what `docs/blueprint.md` already
decided and what this charter adds; each ends with what is still open. The
rest is S9's work and is listed so it has a place to land.

## Status

Charter and first specs, 2026-08-17. No runtime code yet. Nothing here
requires YFCE silicon: the prototype runs on the PC this repository lives on
(RTX 5080 as the NPU, RAM as the DIMMs), and the bare-metal build runs on
any UEFI machine with a CPU that has AVX-512.
