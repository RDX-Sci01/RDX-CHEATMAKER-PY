# RDX CheatMaker — Release Manifest

Generated from the release tree. Figures here are measured, not estimated.

## Release

| | |
|---|---|
| Version | **1.0.0** |
| Production entry point | `RDX-CHEATMAKER-UI-final.py` |
| Implementation | the same file — entry point and implementation are one |
| Size | 18,965 lines |
| SHA-256 (first 16) | `d7478dadadc6ac4e` |

Run it with:

```bash
python3 RDX-CHEATMAKER-UI-final.py
```

## Dependencies

| Package | Status | Notes |
|---|---|---|
| `numpy>=1.21` | **required** | carries every scan, comparison and pointer-index operation; no fallback |
| `numba>=0.57` | optional | JIT for the comparison filter; speed only, identical results |
| `psutil>=5.9` | optional | host memory reporting; conservative default without it |
| `pytest` | test only | the suite is unittest-based and also runs under pytest |

Both optional packages are imported inside `try/except` and degrade cleanly.
See `requirements.txt`.

## Tests

| | |
|---|---|
| Suite | `tests/test_pointer_subsystem.py` (10,959 lines) |
| Result | `620 passed, 1 skipped, 17 subtests passed in 51.85s` |
| Mode | warnings-as-errors (`-W error`) |
| Gate | `python3 tests/check.py` — unit suite, offline rehearsal, real-terminal UI smoke |
| Offline fixture | `tests/golden_watchpoint_event.bin` (1,184 bytes, real captured debug event) |

The gate's *regression-convention* check reports that it has nothing to compare
against: it compared each build to the previous numbered patch, and those
archives were removed at consolidation.

## Hardware validation status

Platform: PS5 firmware **10.01**, payload **ps5debug-NG by OSR v1.3.0**
(protocol 1.3), title *Enter the Gungeon* (CUSA01659).

**VERIFIED** — scanning, writes, freezing, pointer chains, trainer
import/export; debugger attach against a freshly launched game; hardware write
watchpoint; writer-instruction resolution proved by effective-address
calculation; 32-byte AOB anchor relocating uniquely; executable-only scanning
(620.25 s → 4.53 s); and the full capture → relocate → verify → patch →
readback → restore chain.

**NOT TESTED** — the in-game *effect* of an instruction patch (the mechanism
works; the gameplay consequence was never observed); CheatRunner address mode;
two-reload pointer promotion.

**KNOWN LIMITATIONS** — one debugger attach per game-process lifetime; DR6
cannot be the sole watchpoint-hit discriminator; `_KLASS_NAME_OFFSETS` ordering
is correct but not structurally guaranteed; AOB relocation has no console-side
engine.

Full detail with per-item labels: `HARDWARE_TEST_CHECKLIST.md`.

## Documentation

| File | Audience |
|---|---|
| `README.md` | ordinary users — start here |
| `docs/DEVELOPMENT.md` | developers and AI agents — architecture and critical invariants |
| `docs/PS5DEBUG-NG-PROTOCOL.md` | payload protocol reference |
| `CONTRIBUTING.md` | development and testing expectations |
| `CHANGELOG.md` | milestones |
| `RELEASE_NOTES.md` | narrative release history |
| `HARDWARE_TEST_CHECKLIST.md` | what has and has not touched hardware |
| `AI_CONTEXT.md` | short index for AI agents |
