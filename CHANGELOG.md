# Changelog

Milestones, not a patch-by-patch history. RDX was developed as a long series of
numbered patch files which were consolidated away at 1.0.0; **git history is the
development history**. `RELEASE_NOTES.md` carries the narrative version of the
same story.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

---

## [1.0.0] — consolidation and first release

### Added
- `LICENSE` (all rights reserved — no licence granted), `requirements.txt`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/DEVELOPMENT.md`,
  `docs/PS5DEBUG-NG-PROTOCOL.md`, `AI_CONTEXT.md`, `RELEASE_MANIFEST.md`.

### Changed
- `RDX-CHEATMAKER-UI-final.py` is now the implementation itself rather than a
  launcher pointing at a numbered patch file.
- Tests, harnesses and the captured hardware fixture moved to `tests/`.
- `README.md` rewritten for ordinary users; developer material moved to
  `docs/DEVELOPMENT.md`.

### Removed
- 122 archived patch files, development exports, research notes and upstream
  audit passes. Durable technical knowledge was transferred into
  `docs/DEVELOPMENT.md` first.

### Fixed
- A development console's IP address was prefilled on the connect screen for
  first-run users. Removed, with hygiene tests guarding against a private
  address or developer path shipping again.
- A user-facing "Cheat Module Unavailable" error showed only a raw exception;
  it now explains what happened and what to do.

---

## Earlier milestones

These predate the 1.0.0 consolidation and are summarised rather than versioned.

### Instruction anchoring
- **Writer resolution corrected.** A debug event's RIP names the instruction
  *after* the store, because x86 data breakpoints are trap-type. The backward
  decode had always been right, but was unreachable: every event was discarded
  by a gate requiring DR6 to name the watchpoint slot, and the payload clears
  DR6 while handling the trap. DR6 became a hint; the decoded operand's
  effective address became the proof of ownership.
- **The anchor pipeline was wired.** Capture, unique relocation, verification
  and guarded patching had existed as primitives with no production callers.
- **Patch and restore.** Patch length bounded by the captured instruction, live
  bytes must still match, writes verified by readback, restore requires the
  bytes actually applied rather than assuming NOPs.
- **Scanner optimization.** Restricting AOB relocation to executable mappings
  took a process-wide scan from 620.25 s to 4.53 s (~137x).

### Debugger
- ps5debug-NG protocol support: the 0x4A0 debug event, the outbound TCP 755
  interrupt channel, session teardown, and recovery from a held
  `CMD_ALREADY_DEBUG` session.
- Hardware write watchpoints on live values.

### Pointers
- Backwards chain search to module-relative roots, a bounded resolver, and the
  rule that a chain is not permanent until it survives two real game reloads.

### Scanning and interchange
- Exact and unknown-value scanning across ten numeric types, AOB patterns with
  wildcards, relational narrowing, three scan engines.
- Import and export of `.mc4`, `.shn`, GoldHEN/etaHEN JSON and native
  `.rdx.json`.
