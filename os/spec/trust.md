# Trust — capabilities, the trusted surface, audio injection, privacy

Blueprint D10: the runtime is the trusted computing base; the model is not.
Voice-first adds an attack surface — sound — and this file says what is done
about it. Fixed items come from `docs/blueprint.md` §3; open ones are S9's.

## The threat, stated plainly

Every input the model reads is a potential instruction: a page, a message,
an image, and now **any sound the microphone hears** — a voice on a call, a
video's soundtrack, a synthetic copy of the owner's voice, an inaudible or
adversarial signal. Prompt injection is this machine's malware, and audio is
its easiest delivery.

## What holds

| mechanism | what it guarantees | owner |
|---|---|---|
| capability guard | a tool call outside granted capabilities never executes — by construction of the runtime, not by the model's judgement | runtime |
| grammar-constrained sampling | the model cannot emit a malformed or unknown tool call at all | runtime (sampler) |
| trusted surface | consequential actions require the **physical confirm key** and a runtime-drawn prompt the model cannot draw over; a sound cannot press a key | runtime + hardware |
| speaker verification | consequential (and, if enabled, everyday) actions accept the owner's voice only, with liveness; unknown voices get a guest tier | perception model + runtime |
| echo reference | what the machine itself plays is subtracted from what it hears; its own speech and media cannot command it | runtime (AEC) |
| hardware mute | microphone and camera off in the signal path; the LED shows it; no software undoes it | hardware |
| logging | every tool call, grant and revocation is in the append-only log, reviewable by voice ("what did you do today?") | runtime |
| signed archives, verified boot | the runtime and the model set are what they claim; an app's blocks are signed by their publisher | boot ROM + runtime |
| local by default | no network egress without a granted destination class; the log and the user's things stay on the machine | runtime |

## Privacy of a listening machine

- The microphone is open only after wake or push-to-talk, and the LED says
  so; the model never receives audio from a closed microphone.
- Wake-word and voice-activity detection run in the runtime's always-on
  path, on-device, and keep nothing.
- Secrets — PINs, codes, keys — enter through the runtime's numeric pad and
  are never spoken, never transcribed, never in the model's context.
- Camera on demand only, LED lit.
- Nothing leaves the machine unless a network capability was granted for
  that destination class, and the log says what did.

## What is measured (S9, G6)

- An injection set of ≥ 200 cases across text, images and **audio**:
  commands in media, in calls, in the owner's cloned voice, in another
  language, in the middle of a legitimate request. Count of unauthorised
  tool calls — must be **zero** — and of authorised actions the trusted
  surface failed to make legible.
- Speaker-verification false accept/reject on the owner's voice under noise
  and playback attacks.
- Whether the confirm-key flow survives use: how often people confirm
  without reading — measured, because a surface people rubber-stamp is not
  trusted.

## Open — for S9

- Guest tier: what a non-owner voice may do (nothing? everyday settings?).
- Whether owner recognition gates everyday settings by default.
- The publisher-signing scheme for app bundles.
- Recovery when the owner's voice is unavailable (illness, noise): the
  keyboard path and the confirm key must always suffice.
