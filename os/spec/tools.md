# Tools — apps and system settings under model control

The tool ABI (blueprint I9): the "syscalls" the model set may call. Every
tool names a capability; capabilities are granted through the trusted
surface (`trust.md`), logged, and revocable. This file fixes the tool
families and the tiers; S9 writes the exact signatures and the grammar.

## Families

| family | tools | notes |
|---|---|---|
| memory | recall, store, forget | over the log, the user's things, and app state; the memory model indexes |
| residency | load / unload adapter or model, list resident, prefetch | apps are adapters; switching is a descriptor change |
| apps | invoke, hand off, compose | the kernel drives an app's skill with its grammar live; results come back as events |
| network | fetch, send, subscribe | egress is a capability per destination class; content returns as text/images, never as code |
| device | show, speak, listen, capture, import from USB, print | all through the runtime's widgets and drivers |
| settings | get, set, watch | the tiers below |
| time and power | timers, alarms, sleep, restart | |
| self | summarise context, compact KV, report status | the model managing its own state |

## Settings, in tiers

Settings live in a small typed store in the runtime; the model reads and
writes them through `settings.get / set`; the runtime validates ranges and
enforces the tier. There is no settings screen: the person asks, the model
does, the screen mirrors.

| tier | examples | what it takes |
|---|---|---|
| **everyday** — reflex may change on request | volume, brightness, do-not-disturb, timers and alarms, playback, display sleep, voice choice, language of the moment | a spoken request; shown on screen; undoable |
| **consequential** — kernel only, confirmed | join / forget a network, Bluetooth pairing, power off and sleep schedule, updates to the model set, import / export of the user's things, accounts and keys, default language, accessibility profiles, sharing anything off the machine | spoken confirmation **and** the physical confirm key; logged; speaker verification where enabled |
| **hardware only** — no model, no runtime setting | microphone and camera mute, factory reset, boot medium, the capability grants themselves | a switch or the trusted surface, operated by the person |

The model can never lower its own guard: there is no tool that changes a
capability; grants and revocations happen only through the trusted-surface
flow, which the runtime draws and the model cannot draw over.

## Apps

An app is a bundle of blocks in the archive format: an adapter (LoRA or
expert set), a skill document, tool grants, UI templates, a voice grammar.
Installing one is a consequential action (it asks for grants). Once
resident:

- the kernel invokes it by tool call; the reflex model may run its everyday
  intents directly ("next track");
- its grammar is live only while the runtime says so;
- its state is blocks in the store, indexed by the memory model;
- removing it revokes its grants and drops its blocks.

There is no window, no launcher, no foreground/background: the model
composes apps in the conversation, and the screen shows what is happening.

## Open — for S9

- Exact signatures and the GBNF grammar for every family.
- Whether `network.fetch` returns rendered content (the runtime's reader)
  or raw text for the model to render — cost in tokens either way.
- The everyday/consequential boundary for a few settings (display sleep,
  voice choice) — decided against the injection set.
- App grants at install time vs first use.
