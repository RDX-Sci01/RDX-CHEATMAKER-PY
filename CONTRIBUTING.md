# Contributing to RDX CheatMaker

Read `docs/DEVELOPMENT.md` before changing anything. It documents the
architecture and, more importantly, the **critical invariants** — rules that
each came from a real failure or a real hardware observation.

## The short version

```
inspect the actual source        (not memory, not an old summary)
  -> inspect the covering tests
  -> name which invariant your change affects
  -> make the smallest sensible change
  -> add a regression test
  -> python3 -m pytest tests -W error
  -> python3 tests/check.py
  -> update the docs
  -> commit
```

## Workflow expectations

**Modify the current implementation directly.** Do not create numbered patch
copies — that was the old workflow and it was consolidated away at 1.0.0. Git
history is the development history.

**Every behavioural change needs a test** that fails without the change. A test
that passes both before and after is a guard, not a regression test.

**Warnings are errors.** The suite runs under `-W error`. A NumPy overflow
should fail the build, not scroll past.

**Do not weaken a test to get a green result.** If a test fails, either the
change is wrong or the test encodes an assumption worth arguing about
explicitly.

## Before touching hardware

Reproduce against `tests/golden_watchpoint_event.bin` first. It is a real
captured debug event (DR6 cleared, trap RIP six bytes past the writer) and it
exercises writer resolution offline.

**A game process allows one debugger attach per launch.** A second attach
returns `CMD_ERROR 0xF0000001`. Plan what you need from a session before
spending it, and restart the game to get another.

**Use a game you can afford to restart.** Never attach to jailbreak or system
infrastructure.

## Honesty rules

These matter more than style here.

- **Never claim hardware validation that was not performed.** If a test did not
  run, say so. This project's documentation is deliberately explicit about what
  is unproven, and that is its most valuable property.
- **Do not turn a known limitation into a supported feature** in the docs.
- **Verify payload identity with `CMD_BRANDING`** before concluding anything
  about debugger behaviour. Never infer it from port number, protocol
  compatibility, command availability or capability bitmap. A long-standing
  wrong conclusion in this project came from testing MemDBG's compatibility
  shim while believing it was ps5debug-NG.
- Keep VERIFIED / INFERRED / NOT TESTED / KNOWN LIMITATION labels accurate in
  `HARDWARE_TEST_CHECKLIST.md` and `docs/DEVELOPMENT.md`.

## For AI agents

Also read `AI_CONTEXT.md`. Inspect the real source before proposing anything —
line numbers in the docs drift. Never trust a conversational summary over the
code and tests. Do not create patches for speculative improvements; if nothing
is broken, change nothing.

## Style

Match the surrounding code. Comments explain *why*, not *what* — the existing
source is consistent about this, and the reasoning it records is frequently the
only place a hard-won constraint is written down.
