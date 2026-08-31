# RDX CheatMaker — release notes

**Current release: 1.0.0**
**Production source: `RDX-CHEATMAKER-UI-final.py`** (single file, entry point
and implementation).

This file summarises the milestones that shaped the project. It is not a
changelog of every build — the project was developed as a long series of
numbered patch files, which were consolidated away at 1.0.0. **Git history is
the development history.**

---

## 1.0.0 — consolidation and final release

The project moved from a numbered-patch workflow to a normal source tree.

- `RDX-CHEATMAKER-UI-final.py` is now the implementation itself, not a launcher
  pointing at a numbered patch file. Users no longer need to know which patch
  contains the code.
- Tests, harnesses and the captured hardware fixture moved to `tests/`.
- Developer and AI-handoff documentation consolidated into
  `docs/DEVELOPMENT.md`; the payload protocol reference into
  `docs/PS5DEBUG-NG-PROTOCOL.md`.
- `README.md` rewritten for ordinary users.
- `LICENSE` and `requirements.txt` added, both stating what was already true:
  all rights reserved, and numpy as the only hard dependency.
- Archived patch files, development exports, research notes and audit passes
  removed from the release tree. Durable technical knowledge from them was
  transferred into `docs/DEVELOPMENT.md` first — notably that **ATTACH resumes
  the target** (hence stop → arm → resume) and that **ps5debug map order is not
  guaranteed**, so anything bisecting region starts must sort or merge first.
- A development console's IP address had been prefilled on the connect screen
  for first-run users; removed, with hygiene tests guarding against a private
  address or developer path ever shipping again.

No change to scanning, pointer resolution, the debugger, the watchpoint
workflow, the AOB algorithm, or patch/restore behaviour.

---

## Milestones that got the project here

**Scanning and the UI.** Exact and unknown-value scanning across ten numeric
types plus AOB patterns with wildcards; relational narrowing; three scan
engines (console-resident TurboScan, legacy console, host streaming) selected
automatically; a curses interface driven and regression-tested through a pty.

**Trainer interchange.** Import and export of `.mc4`, `.shn`, GoldHEN/etaHEN
JSON and a native `.rdx.json`, including every rejection path. Verified against
a live CheatRunner.

**The pointer subsystem.** Backwards chain search to module-relative roots, a
bounded resolver, and the rule that a chain is not "permanent" until it has
survived two real game reloads — same-session evidence is explicitly rejected,
because a chain that resolves right now is often just this session's heap
coincidence.

**Debugger integration.** The ps5debug-NG protocol: the 0x4A0 debug event, the
outbound TCP 755 interrupt channel, session teardown, and automatic recovery
from a held `CMD_ALREADY_DEBUG` session.

**Hardware watchpoints.** Arming a write watchpoint on a live value and
capturing the resulting debug event — the foundation for "what writes this
address".

**AOB anchoring.** A 32-byte signature captured around an instruction, with
uniqueness required and ambiguity refused. An early capture window could escape
its own memory region; fixed and covered by a property sweep.

**Writer resolution — the correction that mattered most.** The debug event's
RIP names the instruction *after* the store, because x86 data breakpoints are
trap-type. The resolver had always decoded backwards correctly, but it was
unreachable: every event was discarded by a gate requiring DR6 to name the
watchpoint slot, and the payload clears DR6 while handling the trap. DR6 became
a hint rather than a gate, and the decoded operand's effective address — which
must equal the watched address — became the proof of ownership. The resolved
`writer` is now published separately from the raw `rip`.

**Scanner optimization.** AOB relocation searched the whole address space to
find an instruction that can only exist in executable memory. Restricting it to
executable mappings took a process-wide scan from **620.25 s to 4.53 s**
(~137x). Profiling showed matching was never the bottleneck — 99.8% of the time
was remote memory transport, and 99.52% of the bytes read could never hold a
match.

**The instruction-anchor pipeline.** The primitives existed but had no
production callers until they were wired into one workflow: a canonical anchor
contract in which the writer is mandatory and the trap RIP is diagnostic only;
capture; unique relocation; verification against live memory; and patching that
refuses to write unless every check holds. Acquisition and patching were kept
separable, and anchors are portable JSON artifacts — which matters because a
game process allows only one debugger attach per launch.

**Patch and restore.** Guarded instruction patching: the patch length is bounded
by the captured instruction, the live bytes must still match what was captured,
the write is verified by readback, and restore requires the bytes the patch
actually applied rather than assuming NOPs.

---

## What was proven on hardware

PS5 firmware 10.01, ps5debug-NG by OSR v1.3.0, *Enter the Gungeon*. The full
chain ran end to end: capture → relocate → verify → patch → readback → restore,
with the writing instruction identified and proved by recomputing its effective
address from the captured registers.

What was **not** proven: the in-game effect of a patch, CheatRunner's address
mode, and the two-reload pointer promotion. `HARDWARE_TEST_CHECKLIST.md` and
`docs/DEVELOPMENT.md` record all of this with explicit VERIFIED / NOT TESTED /
KNOWN LIMITATION labels, and neither turns an unverified result into a claim of
support.
