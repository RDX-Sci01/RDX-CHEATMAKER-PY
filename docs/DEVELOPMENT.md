# RDX CheatMaker — Developer & AI Handoff

Describes the **current implementation**, not the historical patch series.
Line numbers refer to `RDX-CHEATMAKER-UI-final.py` (18,925 lines).

> ## Handoff note
>
> **Do not reconstruct the project from historical patch files. The current
> production source is authoritative. Use Git history for historical changes.**
>
> **Do not create numbered patch copies for ordinary development. Modify the
> current implementation, run the tests, and commit the result. Only create a
> numbered patch/release when the project's release workflow actually requires
> one.**

**Evidence labels:** **VERIFIED** = ran on real hardware and passed.
**INFERRED** = follows from source or protocol docs, not separately proven.
**NOT TESTED** = never exercised. **KNOWN LIMITATION** = ran, and this is the
honest result.

---

## 1. Purpose and structure

RDX CheatMaker is a curses memory scanner, cheat builder and trainer exporter
for a jailbroken PlayStation 5. It finds values in a running game, edits and
freezes them, resolves stable pointer chains, identifies the machine
instruction that writes a value, and exports trainers for third-party runners.

```
RDX-CHEATMAKER-UI-final.py     the entire implementation AND the entry point
README.md                      user documentation
LICENSE                        all rights reserved (no licence granted)
RELEASE_NOTES.md               milestone summary
HARDWARE_TEST_CHECKLIST.md     what has and has not touched real hardware
requirements.txt               dependency model
AI_CONTEXT.md                  short index for AI agents
tests/
  test_pointer_subsystem.py    621 tests
  fake_console.py              protocol-speaking offline console
  ui_smoke.py                  drives the real curses app through a pty
  rehearsal.py                 end-to-end offline rehearsal
  check.py                     the acceptance gate
  golden_watchpoint_event.bin  1,184-byte captured hardware event
docs/
  DEVELOPMENT.md               this file
  PS5DEBUG-NG-PROTOCOL.md      payload protocol reference
```

**Production entry point:** `RDX-CHEATMAKER-UI-final.py`. It is a single file
that both defines the application and starts it (`if __name__ == '__main__':`
at :18905). There is no launcher indirection and no patch-numbered
implementation file.

**Runtime file resolution.** Per-user state is written *next to the script*
via `Path(__file__).with_name(...)`, so it follows the release tree:
`.rdx-preferences.json` (:1021) and `.rdx-pointer-candidates.json` (:1019).
Neither ships with the release; both are created on first run. `export_dir`
defaults to `Path.home()`.

---

## 2. Subsystem architecture

| Subsystem | Key functions | Line |
|---|---|---|
| UI / menus | `_results_more_menu`, `do_capture_instruction_anchor`, `do_trace_item_write`, `message_box`, `confirm_box`, `input_box` | 13843, 14157, 14268, 11843, 11817 |
| Console connection | `ps5_connect`, socket pool (`_MAX_CONSOLE_SOCKETS = 10`) | 2040, 5458 |
| Debugger / protocol | `CMD_DEBUG_*`, `_debug_status_name`, `_debug_parse_event` | 3981–3985, 4295, 4535 |
| Memory operations | `ps5_proc_list`, `ps5_maps`, `ps5_read`, `ps5_write`, `_get_maps_cached` | 3496, 3526, 3773, 3808, 5724 |
| Pointer scanning | `_pointer_candidates_from_trace`, `_trace_base_is_resolvable` | 5110, 5003 |
| Temporary-address tracing | `_trace_temporary_access` | 4617 |
| Watchpoints | `_debug_set_watchpoint`, `_debug_continue`, `_debug_free_watchpoint_all`, `_debug_thread_list` | 4520, 4531, 4372, 4307 |
| Writer resolution | `_debug_disasm`, `_decoded_effective_address` | 4550, 4592 |
| AOB capture | `capture_aob_signature` | 6569 |
| AOB relocation | `relocate_by_aob_signature`, `scan_first_pattern` | 6648, 6933 |
| Anchor contract | `_instruction_anchor_contract`, `capture_instruction_anchor` | 6690, 6731 |
| Verification | `verify_instruction_anchor`, `_instruction_region` | 6798, 6473 |
| Patch / restore | `patch_instruction`, `patch_instruction_anchor`, `restore_instruction`, `restore_instruction_anchor`, `nop_bytes` | 6489, 6872, 6554, 6897, 6466 |
| Artifacts | `anchor_to_json`, `anchor_from_json`, `_ANCHOR_VERSION` | 6915, 6920, 6687 |

### Debugger / protocol layer

- `CMD_DEBUG_ATTACH = 0xBDBB0001`, `DETACH = 0xBDBB0002`,
  `SET_WATCHPOINT = 0xBDBB0004`, `GET_THREAD_LIST = 0xBDBB0005`,
  `GETDBREGS = 0xBDBB000C` (:3981–3985). Attach body is a 4-byte LE pid.
- Status words: `STATUS_SUCCESS = 0x80000000` (:250),
  `CMD_ERROR = 0xF0000001` (:251), `CMD_DATA_NULL = 0xF0000003`,
  `CMD_ALREADY_DEBUG = 0xF0000004`, `CMD_INVALID_INDEX = 0xF0000005`
  (:4284–4286).
- Event packet: `_DEBUG_EVENT_SIZE = 0x4A0`, registers at `0x30`, debug
  registers at `0x420`; RIP is at +136 inside the register block.
- **The client must be listening on TCP 755 before ATTACH.** The console dials
  *outbound* to the client on that port. See `docs/PS5DEBUG-NG-PROTOCOL.md`.
- **ATTACH resumes the target.** Stop it (`_debug_continue(cmd, 1)`) before
  changing debug registers, so the tool's own activity cannot be mistaken for
  the game's access. This is why the proven workflow is stop → arm → resume.
  *(Salvaged from the pre-consolidation history; honored at :4768.)*
- `_DEBUG_TRACE_ENABLED = False` (:3996) gates non-experimental callers by
  design. The UI operations pass `experimental=True`. **This is intentional —
  do not "fix" it.**

### Memory subsystem

Reads/writes stream over the console connection. Maps are cached
(`_get_maps_cached`, :5724) with a TTL — note this is a *cache TTL*, not a
timeout, a distinction that has previously been misread.

**ps5debug map order is not guaranteed.** Anything that bisects or
`searchsorted`s over region starts must sort/merge first, or valid pointers are
silently discarded. Currently honored at three sites (:2895, :7980, :8151),
each fed from an explicitly sorted or coalesced source. *(Salvaged from
pre-consolidation history.)*

---

## 3. The instruction-anchor pipeline

```
temporary value the player can see
  -> watched address
  -> hardware write watchpoint         _debug_set_watchpoint
  -> debug event (0x4A0 bytes)         _debug_parse_event
  -> raw trap RIP                      DIAGNOSTIC ONLY
  -> backward disassembly              _debug_disasm(rip - 8, ...)
  -> decoded effective address         _decoded_effective_address
  -> writer instruction                the instruction ENDING at trap RIP
  -> executable, non-writable capture  capture_aob_signature
  -> 32-byte AOB signature
  -> unique relocation                 relocate_by_aob_signature
  -> verification                      verify_instruction_anchor
  -> patch                             patch_instruction_anchor
  -> independent readback
  -> restore                           restore_instruction_anchor
  -> verification
```

### Trap RIP versus writer RIP

x86 hardware **data** breakpoints are *trap*-type. The CPU completes the store,
**then** raises `#DB`. The RIP in the event names the **next** instruction, not
the one that wrote.

Anchoring on the trap RIP would capture a signature for the wrong instruction —
one that generally does not touch the watched address at all — and a patch
there would NOP unrelated code. **The skid is not a fixed offset**: it equals
the length of whatever instruction performed the store. RDX therefore resolves
it structurally — find the instruction where `addr + length == rip` that also
touches memory — never by subtracting a constant.

### Effective-address validation

The decoded operand is recomputed from the captured register block as
`base + index*scale + displacement` and must equal the watched address. Because
DR6 cannot be trusted (below), **this is the only thing that proves an event
belongs to this watchpoint.** It runs inside the wait loop, so a non-matching
event is resumed past rather than aborting the trace, and again after the loop
as an invariant.

### AOB capture and relocation

- `_AOB_SIGNATURE_BYTES = 32` (:6445); `_AOB_MIN_LITERAL_BYTES = 8` (:6446).
- Mask is a hex string: `FF` fixed, `00` wildcard. The hardware-captured
  signature is all-`FF`.
- **Capture refuses writable memory** (`prot & 0x2`), unmapped addresses, and
  windows too uniform to be unique (`len(set(window)) < 4`). It clamps the
  window to the region at both edges.
- **Relocation:** 0 matches → `None` (failure); exactly 1 → accept;
  >1 → `None` (ambiguous, refused). `hits[0]` (:6677) is safe **only because**
  the `hits.size > 1` rejection precedes it — at that point one match exists.
  Do not reorder those.
- `scan_first_pattern` accepts `region_scope="executable"` (filters
  `prot & 0x4`); relocation passes it. An instruction anchor can only live in
  executable memory.

### Patch / restore verification

`verify_instruction_anchor` re-proves an anchor against live memory: relocation
still unique, bytes still equal the captured instruction, region executable and
non-writable. It returns an address **only** when every check holds.
`patch_instruction_anchor` writes only after that; a failed check means **zero
writes**. `patch_instruction` additionally bounds the patch length by
`expected_original` and verifies the readback.

`restore_instruction_anchor` cannot re-relocate — the patch has destroyed the
bytes the signature describes, so its scan reports 0 matches. It deliberately
falls back to the recorded, **writer-derived** address
(`anchor["relocated"]`, then `anchor["writer"]`), never the trap RIP.

### UI integration

Two experimental operations in `_results_more_menu` (:13843), both explicitly
labelled and behind confirmation: "Trace Write → Find Pointer" (pointer chains)
and "Trace Write → Instruction Anchor" (:14157). The anchor screen displays
TRAP RIP, WRITER and STABLE ANCHOR as three distinct things and asks a second
time before patching.

---

## 4. Pointer system

Pointer chains walk backwards from a temporary heap address to a
module-relative root. `_trace_base_is_resolvable` (:5003) rejects bases that
cannot seed a permanent chain: `rip` (a code reference), `rsp`/`rbp` (stack
frames), indexed accesses, and absent bases. Promotion requires surviving **two
real game reloads**; same-session evidence is explicitly rejected.

**Pointer-chain tracing legitimately produces results with no instruction
writer.** That path never anchors on an instruction. `trace_writer` is recorded
only when present and `trace_trap_rip` is diagnostic. Three regression tests
exercise writer-less traces. Imposing the anchor contract here was a real bug,
caught and fixed during consolidation of the anchor work.

---

## 5. CRITICAL INVARIANTS

These are load-bearing. Each came from a real failure or a real hardware
observation. **Do not casually modify any of them.**

1. **Trap RIP must never automatically be treated as the writer instruction.**
2. **A resolved writer must be independently validated through its effective
   memory operand.**
3. **Instruction anchors must come from executable, non-writable memory.**
4. **AOB relocation must produce exactly one match.**
5. **Relocated bytes must equal the captured bytes before patching.**
6. **Failed verification must abort patching.**
7. **Restore must use the verified writer-derived anchor.**
8. **DR6 cannot currently be treated as the sole proof of a watchpoint hit.**
9. **Pointer-chain tracing may legitimately produce results without an
   instruction writer. Do not impose the instruction-anchor contract on
   pointer-only paths.**
10. **Never introduce a raw trap-RIP fallback into the instruction-anchor
    pipeline.**

Supporting rules: do not reorder the `hits.size > 1` rejection before
`hits[0]`; do not change the watchpoint length encoding `{1:0, 2:1, 4:3, 8:2}`
(:4772) — it is counter-intuitive and hardware-confirmed; do not assume a
second attach to the same process will succeed.

---

## 6. Hardware validation

All from PS5 firmware 10.01, **ps5debug-NG by OSR v1.3.0, protocol 1.3**,
*Enter the Gungeon* (CUSA01659), `eboot.bin`.

### VERIFIED

- **Debugger attach works against a freshly launched game.** A fresh process
  attached with `CMD_SUCCESS`.
- **Watchpoint workflow:** listener on 755 → CONNECT → ATTACH → GET_THREADS
  (44 threads) → STOP → arm (slot 3, 4-byte, `length_code=3`, `breaktype=1`
  write-only) → verify `DR7 = 0xd00004c0` → CONTINUE once → one event → resolve
  → clear → DETACH. No repeated CONTINUE was needed; the console stayed healthy.
- **The event RIP is one instruction after the storing instruction:**

```
watched address   0x00032a153f74      trap RIP   0x018f5b5b
writer            0x018f5b55          89 8B 24 01 00 00
                                      mov dword ptr [rbx+0x124], ecx
rbx 0x32a153e50 ; rbx + 0x124 = 0x00032a153f74   == watched address
ecx 0x25 = 37                                    == live value
```

- **The writer was proven through effective-address calculation**, not through
  event metadata.
- **A 32-byte AOB anchor relocated uniquely** to the same address.
- **Executable-region scanning dramatically reduced scan cost:**
  620.25 s → 4.53 s (~137x). Profiling attributed 99.8% of time to remote
  memory reads and 0.2% to matching; 99.52% of bytes read could never hold a
  match. Matching was never the bottleneck — the transport was. Coalescing was
  measured and rejected (70 executable mappings, non-adjacent, coalesce to 70).
- **capture → relocation → verification → patch → restore was hardware-tested**
  through the production layer, verification never bypassed.
- **Patch bytes were independently read back** (`909090909090`).
- **Restore was independently verified** (`898B24010000`, matching the original).

### KNOWN LIMITATION

- **A game process currently has a one-attach lifetime.** A second attach
  returns `CMD_ERROR 0xF0000001`. Relaunch the game. Budget the single attach
  before spending it.
- **DR6 may be cleared by event handling** and therefore cannot be the sole hit
  discriminator, nor identify which slot fired. It arrived as `0x0` on a
  genuine hit while `DR7` still showed the watchpoint armed.
- **`_KLASS_NAME_OFFSETS` ordering is correct but not structurally guaranteed** —
  it works by luck of ordering rather than by construction.
- **AOB relocation has no console-side engine**; it streams memory to the host.
- **Region-classifier startup dominates the now-fast scan** (~2 s of 4.5 s).

### NOT YET TESTED

- **Gameplay effect itself was not conclusively observed.** The bytes were
  written, read back and restored, but the player was not firing during the
  window, so the value held at 33 and no in-game change was seen. **The
  mechanism is proven; the gameplay consequence is not.**
- **CheatRunner Address Mode remains unverified** — exports load in all three
  formats, but the test cheats were inert, so module-relative versus absolute
  offset handling is unconfirmed.
- **Two-reload pointer promotion remains a limitation** — chains validated
  across one relocation only.

### Historical correction worth keeping

Earlier notes claimed watchpoints "do not work on ps5debug-NG", based on two
attaches that hung on `CMD_DEBUG_CONTINUE`. Those had reached **MemDBG's
ps5debug-compat shim on port 744**; payload identity was never checked at the
time. **Always confirm identity with `CMD_BRANDING`.** Never infer it from port
number, protocol compatibility, command availability or capability bitmap, and
treat `capabilities = 0xFFFFFFFF` as UNKNOWN rather than "all capabilities".

### Method provenance

The anchor technique comes from a community walkthrough using **PS4 Cheater**
and **PS4 Reaper Studio** (by Shiningami; not open source). The key insight
adopted here: the anchor is an **instruction signature**, not a pointer. RDX's
implementation is independent.

### Unexplored lead

MemDBG exposes a debugger command family (`0x0600`–`0x0619`) and a tracer
family (`0x0700`–`0x0703`) that RDX does not implement. `MEMDBG_CAP_DEBUGGER`
is bit 20 and `CAP_TRACER` bit 21; `MEMDBG_ERR_UNSUPPORTED = -6` and
`MEMDBG_ERR_STATE = -10` (the latter means "no session", and has been misread
as "unimplemented"). This is the natural route to a second debugger backend.

---

## 7. Testing architecture

`tests/test_pointer_subsystem.py` — 621 tests. `SOURCE` at the top points at
`RDX-CHEATMAKER-UI-final.py`.

1. **Unit tests** — the bulk, run with warnings as errors.
2. **Offline fixture replay** — `tests/golden_watchpoint_event.bin` is the real
   captured hardware event (DR6 cleared, trap RIP six bytes past the writer).
   `GoldenWatchpointEventTests` and `GoldenWriterResolutionTests` drive
   `_trace_temporary_access` with it, so writer resolution is regression-tested
   without a console. **Reproduce against this fixture before touching
   hardware.**
3. **Anchor pipeline** — `InstructionAnchorPipelineTests`, 18 tests covering
   contract, capture, verification, patch refusal, restore, JSON round-trip.
4. **Scanner scope** — `ExecutableScopedAobScanTests`.
5. **Hygiene** — `ProductionHygieneTests`: no console IP, no developer paths,
   IP prefill comes from preferences.
6. **UI smoke** — `tests/ui_smoke.py` drives the real curses app through a pty.
7. **Offline rehearsal** — `tests/rehearsal.py` against `tests/fake_console.py`.

```
python3 -m pytest tests -W error          # full suite
python3 tests/check.py                    # unit + rehearsal + UI smoke gate
python3 RDX-CHEATMAKER-UI-final.py        # run the app
```

`tests/check.py` also has a *regression-convention* check that ran the suite
against the previous numbered patch, where new tests were expected to fail.
With the archives removed it reports that it has nothing to compare against and
skips. **If you ever see that check emit a `FAILED` line, it belongs to the
comparison run, not to the current suite.**

Test comments referencing `patch1xx` explain *why* a given regression test
exists. They are legitimate technical provenance — leave them.

---

## 8. Release procedure

1. Modify `RDX-CHEATMAKER-UI-final.py` directly.
2. Add a regression test that fails without the change.
3. `python3 -m pytest tests -W error`
4. `python3 tests/check.py`
5. Update `RELEASE_NOTES.md` and, if hardware was involved,
   `HARDWARE_TEST_CHECKLIST.md`.
6. Commit. **Git history is the development history.**

Only produce a numbered build if a distribution workflow demands it.

---

## 9. Safe modification guidelines

```
inspect the actual source        (never work from memory or old summaries)
  -> inspect the covering tests
  -> name which invariant (section 5) the change affects
  -> make the smallest sensible change
  -> add a regression test
  -> run the strict suite
  -> replay the offline fixture
  -> run the UI smoke
  -> hardware-test when applicable  (budget the single attach)
  -> update the docs
  -> commit
```

**For AI agents specifically:** read this file and `AI_CONTEXT.md` first.
Inspect the real source before proposing anything — line numbers drift. Never
trust an old conversational summary over the code and tests; several confident
conclusions in this project's history were wrong and were caught only by
re-reading the source or re-checking hardware. Distinguish historical hardware
evidence (the addresses in section 6 were true for one process on one day —
they are fixtures, not constants) from current runtime state. Do not create
patches for speculative improvements. **Never claim hardware validation that
was not actually performed** — this documentation's honesty about what is
unproven is its most valuable property.
