# AI_CONTEXT — RDX CheatMaker

Short index. **Read `docs/DEVELOPMENT.md` for anything substantive.**

## State

```
Release:         1.0.0             (consolidated; development stable)
Entry point:     RDX-CHEATMAKER-UI-final.py   -- IS the implementation
Implementation:  single file, 18,925 lines, numpy the only hard dependency
Tests:           tests/test_pointer_subsystem.py -- 621 tests, OK (skipped=1)
Developer docs:  docs/DEVELOPMENT.md
History:         git. Numbered patch archives were removed at consolidation.
```

**Do not reconstruct the project from historical patch files.** The current
source is authoritative. **Do not create numbered patch copies for ordinary
development** — edit the implementation, run the tests, commit.

## Before you change anything

1. Inspect the **actual source**. Line numbers below are patch140 and will drift.
2. Never trust an old summary over the code and tests.
3. Name which invariant (DEVELOPMENT.md section 5) your change affects, before making it.
4. If nothing is broken, change nothing. Do not create speculative patches.
5. Never claim hardware validation that did not happen.

## The five invariants that break everything if violated

1. **Trap RIP is never the writer.** x86 data breakpoints are trap-type; the
   reported RIP is the instruction *after* the store. No fallback, anywhere.
2. **The decoded operand must equal the watched address.** It is the only proof
   of event ownership, because the payload clears DR6.
3. **An AOB matching more than once is refused**, never guessed.
4. **Instruction anchors are captured only in executable, non-writable memory.**
5. **Verification precedes every write.** A matching signature is evidence, not
   permission.

## Where things live

| Area | Function | Line |
|---|---|---|
| Trace / watchpoint | `_trace_temporary_access` | 4617 |
| Effective address | `_decoded_effective_address` | 4592 |
| Anchor contract | `_instruction_anchor_contract` | 6690 |
| Capture pipeline | `capture_instruction_anchor` | 6731 |
| AOB capture | `capture_aob_signature` | 6569 |
| AOB relocation | `relocate_by_aob_signature` | 6648 |
| Pattern scan | `scan_first_pattern` | 6933 |
| Verification | `verify_instruction_anchor` | 6798 |
| Patch / restore | `patch_instruction_anchor` / `restore_instruction_anchor` | 6872 / 6897 |
| Artifacts | `anchor_to_json` / `anchor_from_json` | 6915 / 6920 |
| UI operation | `do_capture_instruction_anchor` | 14157 |
| Pointer path (no writer needed) | `_pointer_candidates_from_trace` | 5110 |

Constants: `_AOB_SIGNATURE_BYTES = 32` (6445), `_AOB_MIN_LITERAL_BYTES = 8`
(6446), `_ANCHOR_VERSION = 1` (6687), `_DEBUG_EVENT_SIZE = 0x4A0` (3988).

## Two traps that have already caught people

- **The gate's regression-convention check now SKIPS.** `tests/check.py` used
  to run the suite against the previous numbered patch, where new tests were
  *supposed* to fail. The archives were removed at consolidation, so it reports
  it has nothing to compare against. If you ever see it emit a `FAILED` line,
  that belongs to the comparison run, not to the current suite.
- **One debugger attach per game-process lifetime.** A second attach returns
  `CMD_ERROR 0xF0000001`. Relaunch the game before spending it.

## Replay hardware behaviour offline

`tests/golden_watchpoint_event.bin` — the real 1,184-byte captured event
(DR6 cleared, trap RIP six bytes past the writer). Use it before touching a
console.

## Commands

```
python3 -m pytest tests -W error         # 621 tests
python3 tests/check.py                   # unit + rehearsal + UI smoke gate
python3 RDX-CHEATMAKER-UI-final.py       # run the app
```

## Unproven — do not claim otherwise

Gameplay effect of a patch; CheatRunner Address Mode; two-reload pointer
promotion. See DEVELOPMENT.md section 6 for the full list with evidence labels.
