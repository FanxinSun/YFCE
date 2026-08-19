# Interaction — voice first, a glanceable screen, a fixed touch/key set

The interaction contract (blueprint I8) with voice as the primary channel.
What is fixed here comes from `docs/blueprint.md` §3 and `os/README.md`; the
open items are S9's to settle by measurement.

## Channels

| channel | direction | owner | notes |
|---|---|---|---|
| Speech in | in | perception model (streaming ASR) + runtime (VAD, AEC, wake) | far-field mic array; wake word or push-to-talk; barge-in interrupts speech out within 100 ms |
| Speech out | out | voice model (streaming TTS) + runtime (mixer) | phrase-streamed: the first spoken word does not wait for the last token |
| Glanceable screen | out | reflex/kernel model via the UI language, rendered by runtime widgets | large type, few elements: **heard / doing / need** — the transcript, the current action, what is being asked of the person |
| Touch and keys | in | runtime | the fixed set below; nothing else exists |
| Earcons and LED | out | runtime | listening / thinking / acting / needs-you states, driven by the runtime, never by the model |
| Camera | in | perception model | on demand only, with the LED lit; never always-on |

## The fixed touch/key set

Owned by the runtime, identical on every YFCE-P, not extensible by any
model or app. This is the whole list.

| control | form | what it does |
|---|---|---|
| wake / push-to-talk | key, or wake word | opens the microphone; the LED shows it |
| confirm / cancel | **physical keys** (the trusted surface) | grants a consequential action; a sound cannot press them |
| numeric pad | on-screen, runtime-drawn | PINs and codes: secrets are never spoken and never reach the model as text |
| volume | keys | runtime-only |
| mic / camera mute | **hardware switch** | cuts the signal path; no software, no model, no runtime setting can undo it |
| tap, scroll, select | touch on what the model shows | acts on the widget; the model sees the result as an event |
| transcript correction | touch on the transcript | fixes what was heard before it is acted on |
| USB keyboard | optional | text entry when speaking is inappropriate; still no shell, no shortcuts |

## The UI language, under voice

Blueprint §3 stands: the model emits a versioned declarative tree with
incremental updates and the runtime renders it. Under voice-first it is
constrained further:

- **Screens are glanceable.** ≤ 3 regions (heard / doing / need); large
  type; a list or a table when the answer is one; an image when it is one.
- **Spoken text is the primary output; the screen mirrors it.** The model
  writes one stream; the runtime speaks it and shows it. A response is short
  by policy — a sentence or two, 15–30 tokens — and the reflex model
  acknowledges ("opening…") while the kernel works.
- **Tokens per update ≤ 40** (blueprint D9), measured in S9.
- **Voice grammars per app.** An app's intents join the sampler grammar when
  its adapter is resident; the runtime, not the model, decides which
  grammars are live.

## Latency classes (8-channel reference machine)

| moment | budget | who |
|---|---|---|
| wake → listening cue | ≤ 200 ms | runtime |
| end of utterance → transcript on screen | ≤ 300 ms | streaming ASR |
| → first spoken word of an acknowledgement | ≤ 500 ms | reflex model + TTS |
| → first spoken word of the answer | ≤ 1.5 s (MoE kernel), ≤ 3 s (dense, speculative) | kernel + TTS |
| barge-in → speech out stops | ≤ 100 ms | runtime |
| touch → widget response | ≤ 16 ms | runtime |
| a consequential action | after spoken confirm **and** the confirm key | trusted surface |

## Failure handling

- **Uncertain transcript** (low ASR confidence, or a name/number): show it,
  ask before acting; correction by touch or by voice.
- **Ambiguous request:** one clarifying question, spoken and shown; never a
  menu.
- **The model is wrong:** "undo" is a first-class spoken command; the log
  makes every action reversible where the world allows.
- **Silence and noise:** the runtime, not the model, decides when the
  microphone closes; the model never hears the room when the LED is off.

## Open — for S9

- Wake word vs push-to-talk as the default; whether both ship.
- The exact three-region layout and its token cost per update.
- Which apps get a spoken grammar and how many can be live at once.
- Whether owner recognition (speaker verification) gates everyday
  settings, or only consequential ones — see `trust.md`.
