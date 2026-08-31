# RDX CheatMaker — Hardware Test Checklist

Everything in this codebase has been protocol- and unit-tested against mocked
sockets and simulated memory. Nothing has been run against a real PS4/PS5.
This checklist exists to close that gap. Run it against `RDX-CHEATMAKER-UI-final.py`,
which is `RDX-CHEATMAKER-UI-final.py` — since the 1.0.0 consolidation the
launcher and the implementation are the same file.

## ⇒ RUN THIS FIRST (no console needed): `python3 tests/rehearsal.py`

Run `python3 tests/check.py` first — it runs every offline check in one command.

Fifteen checklist items turn out not to need hardware at all — they were asking
whether RDX's own wire code does the right thing, not what the console does.
The rehearsal answers those against a protocol-speaking fake console in about
fifteen seconds, so a red result there is found before an attach rather than
during one. Anything it reports FAIL is a bug in RDX and the session should not
start until it is green.

It deliberately cannot answer: whether the payload honours debug registers,
whether type pointers behave as assumed on a real IL2CPP title, whether
CheatRunner accepts the trainer, or any timing. Those are below.

## ⇒ BEFORE THE ATTACH: the trace return path (2026-08-30)

**The DR read-back has been blocked by networking, not by the payload.** The
attach was authorised and RDX refused it, correctly, at the pre-flight:

    _local_address_towards("192.168.0.88") -> 100.122.106.94

Confirmed against the kernel, not inferred:

    ip route get 192.168.0.88
      -> dev tailscale0 table 52 src 100.122.106.94
    ip rule
      5270: from all lookup 52        <- Tailscale, consulted first
      32766: from all lookup main
    table 52   192.168.0.0/23 dev tailscale0  +  default dev tailscale0
    table main 192.168.0.0/23 dev wlp0s20f3 src 192.168.0.41   <- shadowed

Tailscale is carrying the console's own LAN subnet, and the `default dev
tailscale0` entry means an exit node is active. The console therefore sees this
client as a CGNAT address (100.64.0.0/10) and cannot open the debug-interrupt
connection back to it on port 755.

Scans, reads, writes and TurboScan are unaffected because they are all
client-to-console. **Tracing is the only operation needing the reverse path**,
which is why three sessions of successful scanning never surfaced it.

`_trace_network_refusal` caught this before attaching. Had it not, the attach
would have stopped the game and blocked waiting for a callback that could
never arrive, leaving the target traced — the failure its docstring describes.

**Remedy, on the client, before retrying:** stop Tailscale carrying the LAN —
`tailscale set --exit-node=` to drop the exit node, or
`tailscale set --accept-routes=false`, or `tailscale down`. Then re-run the
pre-flight and confirm the dial-back address is a `192.168.x.x` one. The direct
route already exists in table `main`; it only needs to stop being shadowed.

## ⇒ ATTACH ATTEMPTED 2026-08-30 — FAILED, NO VERDICT, GAME LOST

Authorised by the user, pre-flight fully green after Tailscale was disabled
(route `dev wlp0s20f3 src 192.168.0.11`, port 755 bindable, no stuck session).
`_trace_temporary_access(..., experimental=True)` on `eboot.bin` pid 93,
target `0x251042720`.

**Outcome: no DR read-back was obtained.** The sequence never reached the
diagnostic. It failed earlier, at the resume:

    _trace_temporary_access -> _debug_continue(cmd, 0)
      -> _debug_send(0xBDBB0010) -> TimeoutError after 61.4 s

Then:

    Debug detach failed (TimeoutError). The target is most likely still traced.
    _debug_force_resume(): returned False on 3 consecutive attempts
    target address: unreadable (TimeoutError, then ConnectionError)
    ps5_proc_list: "PS5 disconnected"
    console: still pings (2.4 ms), port 744 still ACCEPTS TCP, but the
             payload serves no commands

So the game is stopped and traced, and the payload's command handler is wedged
behind the leaked debug session. Force Resume did not recover it; the port
accepting a connection is not evidence the payload is alive.

**What this establishes.** `_DEBUG_TRACE_ENABLED = False` is correct and should
stay false. The blocker is not DR honouring, which is still unmeasured — it is
the debugger *lifecycle*: this payload does not reliably answer
`CMD_DEBUG_CONTINUE` after a watchpoint arm, and RDX cannot recover from that
state. Session 1 recorded "attach works but crashed a live game"; this is the
same failure reached deliberately, with the failing call now identified.

**What it does not establish.** Nothing about debug registers. The three
verdicts in the section below remain untested.

**Recovery (needs a human at the console):** relaunch the game — a fresh
process clears a leaked trace — and reload ps5debug-NG only if that does not
help. Both are RDX's own recorded remedy, printed at the failure.

**Before any retry:** the next attempt should arm the watchpoint and read the
debug registers back *without* the continue/resume round trip, so the
diagnostic can produce a verdict even when the lifecycle is broken. Ordering
the read-back before the resume is the change that makes this measurable.

## ⇒ IDENTITY CORRECTION 2026-08-30 — PORT 744 WAS NOT ps5debug-NG

A read-only `CMD_BRANDING` (`0xBD000501`) on port 744 returns:

    "MemDBG ps5debug-compat"   "MDBG-1"

Port 744 is **MemDBG's ps5debug-compatibility shim**, not the ps5debug-NG
payload. Both listeners on this console are MemDBG: 9020 native, 744 compat.

**The branding was never checked before either attach**, so the payload
identity during those tests is unconfirmed. The conclusion recorded below —
"the payload does not serve debug registers" — is accurate about *whatever was
answering on 744*, and is very likely MemDBG's shim rather than ps5debug-NG.
It should not be read as a verdict on ps5debug-NG.

This is consistent with everything observed: the shim serves the memory command
family well (process list, maps, reads, writes, even TurboScan with engines
`0x03FF`), and its debugger family is incomplete — attach appears to succeed, a
watchpoint arms, then `CMD_DEBUG_CONTINUE` never answers and `GETDBREGS` reads
back on zero threads. A partial reimplementation failing exactly at the hard
part is a much better explanation than the mature payload being broken.

**Real ps5debug-NG has not been tested on this console.** Its documentation
claims DR0-DR3 hardware watchpoints with write granularity and lists firmware
10.00-10.60 as fully verified; this console is on **10.01**, inside that range.

**Action:** load the genuine ps5debug-NG payload and re-run the diagnostic
order. Confirm identity with `CMD_BRANDING` *before* attaching, every time.

**Lesson worth keeping:** an open port on 744 is not evidence of which payload
is behind it, and neither is the user's expectation. One read-only command
settles it in 2 ms and should precede any debugger work.

## ⇒ ANSWERED 2026-08-30 (second attach): THE PAYLOAD DOES NOT SERVE DEBUG REGISTERS

Second attach, pre-flight fully green (dial-back `192.168.0.11`, no stuck
session), `eboot.bin` pid 97, target `0x24da2ef90`. patch129/130's pre-resume
reporting did its job and produced the measurement the first attach lost:

    Watchpoint DR pre-resume (1 thread, slot 3 @ 0x24da2ef90):
      debug registers could not be read back before the resume

That is `_debug_verify_watchpoint` returning `checked == 0`. It ran — the
"pre-resume check unavailable" branch did not fire — and read back zero
threads. The resume then timed out identically to the first attempt
(`CMD_DEBUG_CONTINUE`, 61.4 s), detach failed, `_debug_force_resume` returned
False twice, and the user confirmed the game frozen.

**This resolves the blocker carried since session 1, and none of the three
hypotheses below apply.** The question was "does the payload honour debug
registers for a given store?" It cannot be asked: the payload does not report
debug registers at all, and does not answer `CMD_DEBUG_CONTINUE` after an arm.
The debug command set is non-functional on this payload/firmware once
attached, which also explains session 1's "attach works but crashed a live
game" and both of today's failures.

**Consequences, decided:**

- **Do not implement `CMD_DEBUG_SETDBREGS`.** It was gated on evidence of
  per-thread application. There is no such evidence and no way to gather it
  here — `SETDBREGS` would be written against a debugger that cannot be
  observed. This is exactly what the calibration note warned against.
- **`_DEBUG_TRACE_ENABLED` stays False.** Watchpoint tracing cannot work on
  this payload. It is not a tuning problem.
- **"Find what writes this address" is unavailable on ps5debug-NG**, and with
  it the instruction-patching workflow UPSTREAM_AUDIT_PASS8 identified as what
  the GoldHEN ecosystem actually does. That path needs a payload with a working
  debugger, not more RDX code.

  **SCOPE CORRECTION (same day, UPSTREAM_AUDIT_PASS10).** An earlier draft of
  this entry read as though instruction patching were impossible on PS5. It is
  not. A published guide sets a **write watchpoint on a PS5**, traps the
  instruction, disassembles it and NOPs it — using **PS4 Reaper Studio** as the
  debugger with **PS4 Cheater** as the scanner. The kernel pauses on the trap
  and resumes when the breakpoint is reset.

  So the platform supports it and another tool does it. What is unavailable is
  **ps5debug-NG's debugger as driven by RDX**. The blocker is "we are speaking
  to the wrong debugger", not "this cannot be done" — a different and much more
  tractable problem. Do not cite this entry as evidence that watchpoints are
  impossible on PS5.
- One difference worth recording: this time the payload stayed responsive
  (`ps5_proc_list` returned 88 processes afterwards) while reads to the traced
  process hung. The first attach wedged the whole command handler. So the
  damage is the *game*, and relaunching it is usually enough.

**Cost of the answer:** two frozen games and three console restarts across one
day. It was worth taking once; it should not be retried. The verdicts below are
retained for a different payload, not for this one.

## ⇒ RETAINED FOR A DIFFERENT PAYLOAD: the watchpoint DR read-back

patch97 added the diagnostic that makes the unresolved watchpoint testable, and
**one attach settles it**. Arm a watchpoint as before; RDX now reads DR0-DR3 and
DR7 back on every thread and logs coverage plus a verdict. Record which of the
three it prints:

**Read the sweep's caveat first.** The sweep carries an 8 s budget; if it was
cut short the verdict says *"sample only: N of M thread(s) read"* and is not a
conclusion — re-run before acting on it. Also watch for a **DR mismatch** line:
that means the arm was visible to a stopped thread and not to a running one,
i.e. the payload stages debug registers, which changes what everything below
means.

- [ ] **"set on all N readable thread(s)"** — per-thread application is ruled
      out. The cause is DR honouring on this firmware, or the store reaching the
      page through a different mapping. Do **not** implement `SETDBREGS`.
- [ ] **"set on M of N thread(s)"** — the payload applies debug registers
      per-thread. `CMD_DEBUG_SETDBREGS` (`0xBDBB000D`) is then the documented
      fix; note its response is **two** status words, and a client that reads
      one desynchronises the socket.
- [ ] **"set on no thread"** — the arm is acknowledged and discarded. A
      payload/firmware bug; report upstream with the log line.

The verdict is also appended to the `TimeoutError` a fruitless trace raises, so
it appears even when nothing fires. Until this runs, `SETDBREGS` stays
unimplemented on purpose — see the calibration note at the end of this file.

## patch97–101: new surfaces awaiting hardware

- [ ] **All-thread free-slot probe.** `_debug_free_watchpoint_all` now picks a
      slot free on every thread rather than on `threads[0]`. Confirm arming
      still succeeds on a 40-thread target and that the index chosen differs
      from the old one only when a slot really is busy elsewhere.
- [ ] **Export portability line.** Export a same-session heap cheat and confirm
      the preflight says it is session-bound; promote one through Pointer
      Project and confirm the line flips to reload-safe.
- [ ] **Changed-memory highlight.** Open the hex view on a value you can change
      in-game and confirm only the bytes that moved light up, that scrolling
      does not light up the whole window, and that `C` toggles it.
- [ ] **Pointer preview.** In the structure view on a real IL2CPP object,
      confirm `ptr` slots resolve to plausible module/heap regions and that
      obvious non-pointers read `unmapped`.
- [ ] **Type Scan on a real title.** The headline unknown: does grouping by the
      qword at offset 0 actually surface `Il2CppClass` pointers on this title?
      Record the top few groups, their instance counts, and whether the
      module+offset shown looks like a class table. If the result is dominated
      by one enormous group, the heap/static split needs revisiting.
- [ ] **Type Scan cost.** It streams the whole writable heap. Record wall time
      against the ~2 GiB title and whether the 8 M candidate cap was hit.
      patch116 rewrote the tally to bound memory (268 MB -> 62 MB on a 256 MiB
      synthetic heap); watch RSS during the real scan and note whether the
      distinct-type cap (250 k) was reached, which would mean the heap/static
      split is letting through more noise than expected.
- [ ] **Watchpoint stop duration.** patch116 moved the DR sweep after the
      resume specifically so the game is not held stopped for it. Confirm the
      pause-before-attach approach still leaves the title running normally,
      and note how long the target is stopped.
- [ ] **Live class names (patch107).** The headline item now. Run Type Scan on
      the IL2CPP title and record whether class names resolve, which offset
      the probe accepted, and whether any name looks wrong. A *wrong* name is
      worse than none: if that happens, note the title and Unity version.
- [ ] **Chained bookmark across a reload (patch108).** Bookmark a value, attach
      a chain with P, reload the game, and confirm the bookmark still resolves
      rather than going stale.
- [ ] **Salvage (patch116).** Point import at a HEN-Cheats-Collection trainer
      for a *different version* of a game you have, and record how many of its
      chains re-verify against your build.
- [ ] **Symbol overlay.** Run Il2CppDumper against the title, load `dump.cs`,
      overlay a class on an instance found by Type Scan, and confirm the field
      values look sane. This is the end-to-end test of items 7 and 8 together.

## patch87–96: new surfaces awaiting hardware

Added by the upstream UI/UX audit (notes removed at the 1.0.0
consolidation; see git history). All are unit-
and smoke-tested; none has touched a console.

- [ ] **`.mc4` Address Mode — the highest-value item here.** CheatRunner shows
      an Address Mode selector above the mod list for SHN and MC4: `Auto`
      (default), `Absolute`, `Relative`. RDX emits module-relative offsets with
      `<Section>0</Section>`, which is almost certainly what `Auto` resolves
      to — but "almost certainly" is the same confidence this document already
      flags as unvalidated. Load an exported trainer and confirm which mode
      applies the patch correctly. **Test this as its own item, not folded into
      "does `.mc4` work".**
- [ ] **`.shn` accepted by CheatRunner.** Export now writes the plaintext twin
      beside every `.mc4`. This is what makes the previous item diagnosable:
      `.shn` accepted + `.mc4` rejected isolates the AES/base64 container;
      both rejected isolates the schema. Try `.shn` first.
- [ ] **Game marker in the process picker.** Confirm the `▶` lands on the
      running title and not on a system app that also runs as `eboot.bin`.
      Confirm the probe never delays attach on a slow link.
- [ ] **Hex viewer against real mappings.** Confirm `??` appears when scrolling
      past the end of a mapping rather than an error box, and that the anchor
      row highlight tracks the address it was opened on.
- [ ] **Structure auto-dissect quality.** The pointer test checks a qword
      against the live map. On a real IL2CPP object, confirm the proposed
      fields are a usable starting point and that pointer slots really are
      pointers — an integer that happens to look like a mapped address is
      indistinguishable at this level and is expected to be mislabelled
      sometimes.
- [ ] **AOB across a mapping boundary.** patch91 merges adjacent regions
      specifically so a pattern straddling two mappings is findable. Construct
      or find such a case and confirm it is now matched. This was a real miss
      before, not only a speedup.
- [ ] **`TURBO_MIN_SURVIVORS` threshold.** Confirm that a refine below 512
      candidates on `auto` is not slower than the turbo path on real hardware,
      and that switching between them never resurrects a dropped address
      (which would mean a resident session was not closed).
- [ ] **RLE undo on a real scan.** Confirm undo restores exactly, and that
      `Clear Scan History` now reports far smaller figures than before.
- [ ] **Region settings warning.** Set a non-default min size or exclude token
      and confirm the log names the setting as the reason the scan narrowed.
      This exists so a thin scan is not mistaken for a console fault mid-session.
- [ ] **Region min-size filter.** With it set to `0x32000`, confirm the value
      you are hunting is still found — a game that keeps state in a small
      mapping would be a counter-example worth recording here.
- [ ] **MemDBG capability reporting.** The new `Target` seam is unit-tested but
      still has never met a daemon. Connect with MemDBG loaded and confirm the
      transport line logged at connect names the real capability set.

## Session log

**Session 3 — 2026-08-30, 192.168.0.88, ps5debug-NG, Enter the Gungeon
(CUSA01659) as `eboot.bin` (pid 93).** Loaded to exercise the ps5debug-only
paths that patch118 and patch120 changed and that had never run on hardware.

- [x] §0 ps5debug-NG on 744; 88 processes in 13 ms; maps 307 regions,
      4.26 GiB, 22 ms. MemDBG deliberately not loaded (9020 refused).
- [x] **patch120 success path verified.** TurboScan caps: version 1,
      engines `0x03FF`, 4 threads, required `0x15` present.
      `_turbo_worth_probing` **True before and after** a successful scan — the
      capability is not falsely latched off on a console that has it. Only the
      failure path had ever executed before today.
- [x] **TurboScan actually used**: `Turbo first scan completed in 1.01s`,
      100,353 matches over 2.15 GiB.
- [x] **patch118 stays silent when the classifier answers.** Classifier
      supported=True, 4 uncached ranges, 2,076 MiB flagged, 0.09 s. Zero
      `probed at ... MiB/s` lines, as designed.
- [x] ~~**The probe and the classifier agree on both 2 GiB mappings**~~ —
      **RETRACTED.** That was one sample per region. Re-measured with seven
      samples each, the classifier-confirmed *uncached* mapping reads faster
      than the cached one (med 5.0 vs 4.4 MiB/s) and throughput varies
      threefold within a single mapping. Read-throughput does not distinguish
      them on this transport; see patch125. Original, incorrect note:
      independent confirmation that patch118's measurement reproduces the
      payload's own verdict:

      0x200000000  2048 MiB  classifier: cached    measured 4.7 MiB/s -> scan
      0x280200000  2048 MiB  classifier: UNCACHED  measured 2.8 MiB/s -> exclude

- [x] **ANSWERED, negatively: a value-freeze cannot hold a continuously
      rewritten value.** This closes the §3 item that had been resting on
      inference since session 2. Measured on `0x251042720` (ammo), with the
      user firing and also while idle:

          RDX freeze at its 200 ms tick   held  31/657 samples   4.7%
          tight write loop, no sleep      held 312/651 samples  47.9%
          game rewrites the address       every 8-20 ms (median 18)
          one ps5_write round trip        15.7 ms

      The game rewrites ammo every frame whether or not the player is
      shooting. A single write round trip costs about as long as the game's
      entire write period, so **the ceiling is set by the link, not by the
      tick rate** — no interval change wins this. The user confirmed the
      in-game counter never showed the frozen value.

      Freezing remains correct for values the game writes only on change
      (currency, item counts, unlock flags). It cannot work for anything
      updated per frame, which includes most of what people want to freeze.

      **The fix is code patching, not faster writing**: locate the
      instruction that writes the address and neutralise it, which is what
      Cheat Engine's AOB injection does (see UPSTREAM_AUDIT_PASS7). That
      requires knowing which instruction writes the address — i.e. the
      watchpoint / debug-register capability that has been unresolved since
      session 1.

      This reclassifies the DR read-back. It is not one more unknown among
      several; it gates whether RDX can affect frame-written values at all.

- [x] **§4 pointer resolve ran on ps5debug-NG** — 96 chains in 258 s for a
      live ammo address, and it exposed patch122: the plausibility guard
      tested `0 <= offsets[-1]`, so every *negative* field offset was called
      implausible. 0 of 8 top chains passed, the sort fell through to depth,
      and depth-1 coincidences at +0x42720 (the low bits of the target
      address) outranked the real chains ending in -0x60. Now magnitude.

- [x] **IL2CPP Type Scan: root cause found and fixed (patch123).** The scan
      required a type pointer's *target* to be a static/module region.
      `Il2CppClass` is heap-allocated: the holders of the "PlayerController"
      name string all sat at `0x2xxxxxxxx`, prot=3, `_is_static_region`
      False, class at `0x20362e560`. Every real type pointer was excluded and
      only vtable/callback pointers survived — the top five groups
      disassembled as x86-64 prologues, and 0 of 40 named.
      `_read_klass_name` was never at fault; given the real pointer it
      returned "PlayerController" from +0x18 immediately.
      After the fix, on the same console: 512 groups in 114 s (was 441 s),
      **10 of 40 named** — String, Boolean, Vector2[], Single.

- [x] **§2 pattern (AOB) scan validated on real memory.** A 16-byte pattern
      lifted from the live heap was found exactly once, at the right address,
      with and without wildcards, and a brute-force sweep of a 32 MiB window
      returned an identical hit list.
- [ ] **Open: AOB scanning has no console-side engine.** The same console
      scans an exact value with TurboScan in **0.90 s** and a 16-byte pattern
      in **129.79 s** — 144x — because pattern scanning always runs on the
      host path. Acceptable today, but UPSTREAM_AUDIT_PASS7 recommends
      signature-rooted chains, which would make pattern scanning routine
      rather than occasional. That recommendation should be costed against
      this number before it is adopted.

- [ ] **Open: `_KLASS_NAME_OFFSETS` order is load-bearing and undefended.**
      8 of 40 type pointers had more than one offset yielding a plausible
      name, because **+0x18 is the namespace**:

          0x10='String',    0x18='System'
          0x10='Vector2[]', 0x18='UnityEngine'

      `_read_klass_name` takes the first hit, and the shipped order
      `(0x10, 0x08, 0x18, ...)` is right — by luck. Put 0x18 first and every
      class is confidently labelled with its namespace. Needs a design pass:
      rank offsets, or recognise the name/namespace pair, rather than taking
      whichever matches first.

- [x] **ANSWERED and acted on (patch125): it is not a discriminator at
      all.** Re-measured with seven samples per region, the
      classifier-confirmed *uncached* mapping read **faster** than the
      cached one (median 5.0 vs 4.4 MiB/s), and throughput varies
      threefold within one mapping. Read-throughput cannot separate them;
      the probe is now documented as a readability floor at 0.5 MiB/s and
      errs toward inclusion. Original note:

- [ ] ~~**Open: `_OVERSIZE_MIN_RATE` has almost no margin.**~~
      Absolute read rates are payload-dependent, and the threshold is a fixed
      constant calibrated on MemDBG:

          MemDBG (9020, session 2)   cached 34-86 MiB/s   uncached n/a
          ps5debug-NG (744, this)    cached  4.7-5.3      uncached 1.7-4.2

      `_OVERSIZE_MIN_RATE = 4.0` sits inside the ps5debug-NG distribution, not
      outside it: cached measured 4.7 and uncached 2.8, a **0.7 MiB/s** gap
      that network jitter could cross. One uncached range (0x304e00000)
      measured 4.2, i.e. above the threshold — harmless only because it is
      under the 1 GiB cap so the probe never runs on it.

      The verdict should be **relative, not absolute**: probe a known-cached
      reference region in the same process over the same transport and compare
      against it, rather than against a constant. The danger case is a payload
      with slow absolute reads *and* no classifier; today's two payloads each
      avoid it for a different reason, which is luck, not design.


**Session 2 — 2026-08-29, 192.168.0.88, MemDBG 0.2.0-nightly.153.gc39ac30,
Enter the Gungeon (CUSA01659) as `eboot.bin` (pid 94).** First session with
MemDBG loaded; the native listener on 9020 and the compatibility listener on
744 were both up. Driven by read-only probe scripts against the module's own
functions rather than through the TUI.

- [x] §0 MemDBG on native 9020, RDX reports **native memory I/O**. HELLO:
      protocol 1, feature level 4, platform 5, udp_port 9023.
- [x] §0 Process list — 87 processes, carried over **9020 native**, not the
      744 compatibility listener.
- [x] Memory map: 307 regions, 4.26 GiB, **0.02 s**. Writable scope, which is
      what a scan actually walks: 142 regions, **4.18 GiB**.
- [x] Native read verified against known content — `/libexec/ld-elf.so.1` at
      `0x400000`, so the path returns real data and not zeroes.
- [x] Title confirmed IL2CPP Unity: `global-metadata` and `mscorlib.dll` both
      mapped by name, 4,187 `UnityEngine` and 107 `Il2Cpp` hits in the main
      module. **The type-pointer and class-name assumptions are testable on
      this title** — not yet exercised.
- [x] **Found and fixed: RDX opened a new TCP connection per memory read.**
      The native listener accepts 7 connect-read-close cycles and then
      refuses for ~60 s; every later read then paid the full native retry
      budget before falling back to 744. **311.9 ms/read against 4.8 ms once
      shared — 64x, and silent after one log line.** See patch117 in
      RELEASE_NOTES. Fix verified on this console by A/B.
- [x] Payload advertises `capabilities = 0xFFFFFFFF`, every bit set, so the
      bitmap carries no information. HELLO kept succeeding while
      `memory_read` failed, so it is not a usable health signal either.
      patch117 adds a failure latch rather than trusting either.
- [x] **Found and fixed: the scanner was looking at 4.4% of the game.**
      With no region classifier (MemDBG has none) the fallback dropped any
      mapping over 1 GiB — here, two 2 GiB `[device]` mappings holding
      4.000 GiB of 4.180 GiB writable. They read at **86 MiB/s, faster than
      the `[default]` regions that were kept (34 MiB/s)**, and held thousands
      of occurrences of the searched value. patch118 measures the region
      instead of guessing from its size. First scan for ammo `112`:
      **53 matches from 184.8 MiB → 21,743 from 4,280.8 MiB**, and every one
      of the first ten survivors is inside a region that had been dropped.
- [x] §2 First scan, exact `u32`, aligned, writable scope — 4,280.8 MiB in
      **177 s (24.2 MiB/s)** on the host engine. TurboScan and the console
      scan are ps5debug-NG commands and correctly reported unavailable.
- [x] §2 Next-scan narrowing — ammo 112 -> 109: **21,929 -> 2 survivors in
      8.0 s**. Both confirmed live at 105 on the next change, so they are
      mirrors of one value. Both addresses lie inside the `[device]` region
      the pre-patch118 size cap discarded — this workflow was not reachable
      before today.
- [x] §3 Write path on MemDBG, identity writes (no state change):
      `ps5_write` 2.4 ms; `ps5_write_verified` ack+verified 3.7 ms;
      `_target_checked_write` ok (78.5 ms first call, map fetch);
      `memdbg_write_multi` batch `[True, True]` 2.4 ms; `Target.write` ok.
      An unmapped address was correctly refused, logged
      `kernel space — write blocked`, with no exception.
- [x] §3 Freeze via `_toggle_cheat_freeze` — enables, reports
      `active @ 0x…`, writes, disables cleanly, releases.
- [x] **ANSWERED: it was contention, not lag.** Re-measured at 50 ms
      resolution: first effect at **1548 ms**, then the value tracked the
      game because it rewrites the address every 8-20 ms. The 5 s sampling
      below simply landed on the game's value twice. patch126/127 report
      this as **LOSE** rather than `active`. Original note kept for the
      record:

- [ ] ~~**Open: a freeze at 150 read back as 105 for the first ~10 s.**~~
      Samples (taken 5 s apart, the first *after* a 5 s sleep, so t≈5/10/15 s)
      read 105, 105, 150. The manager loop ticks every 0.2 s.

      **An earlier draft of this note blamed the `10.0` argument to
      `_validate_addr_in_maps`. That is wrong — it is a map-cache TTL
      override, not a timeout, and cannot block. Do not chase it.**

      Two readings remain, and this session cannot separate them:
      - the freeze genuinely took >10 s to begin writing; or
      - the freeze and the game were *both* writing, and two of the three
        samples happened to land after a game write.

      The second is the more likely and the more serious: it would mean a
      freeze does not reliably hold a contested value, which is the same gap
      as the "not proven" item below. Distinguishing them needs a tight
      sampling loop (~50 ms) around a freeze enable, which is cheap and
      should be the first thing run next session.
- [x] **ANSWERED, negatively: a freeze does not win.** The contested case
      was observed with the user firing: the value held in **31 of 657
      samples (4.7%)** at the 200 ms tick, and **312 of 651 (47.9%)** in a
      tight loop with no delay. The game rewrites every 8-20 ms; one write
      round trip costs 15.7 ms. The ceiling is the link, not the tick rate.
      No longer resting on inference.
- [x] **IL2CPP type scan: run, root-caused and fixed (patch123/124).** The
      target filter required type pointers to land in a *static* region;
      IL2CPP allocates `Il2CppClass` on the heap, so every real type pointer
      was excluded and only function pointers survived. After the fix: 512
      groups in 114 s, 10 of 40 named — String, Boolean, Vector2[], Single.
      The offset ambiguity is tracked separately below.
- [ ] Everything else below §0 — results, pointers, export — not reached
      this session.
- [x] **RESOLVED, by measurement, as "not a problem".** MemDBG's native
      listener serves exactly **6 concurrent** connections; the 7th is refused
      and the existing 6 keep working — a hard cap, not a collapse.
      ps5debug-NG did 40/40 connect-per-read cycles without refusal.
      `_MAX_CONSOLE_SOCKETS = 10` therefore does overrun MemDBG by four.

      A patch bounding the scan workers to 5 (6 measured, minus the one
      patch117's shared session holds) was written, tested and **reverted**
      after A/B on the console:

          budget 10   1 fallback to port 744   168.5 s   25.4 MiB/s
          budget  5   0 fallbacks              213.1 s   20.1 MiB/s

      The fix worked and cost 26% of scan throughput. The fallback to the
      compatibility listener is a working overflow valve, not degradation —
      port 744 is fast, and the warning it emits reads far worse than what is
      actually happening. Constraining the workers buys a tidier log and pays
      for it in time.

      **Correction to earlier notes in this file:** the line
      `MemDBG native scan read failed; trying port 744` was recorded three
      times this session as a cost. It is not. The only defensible change here
      is to stop logging normal overflow at `warn`.

- [ ] ~~**Open: the scan exceeds MemDBG's connection budget.**~~ (superseded)
      `_MAX_CONSOLE_SOCKETS = 10` is tuned for ps5debug-NG; MemDBG refused an
      8th connection in testing, and both full scans logged
      `MemDBG native scan read failed; trying port 744`. The scan completes
      over the compatibility listener, so this costs throughput rather than
      correctness — but the budget should be per-backend.

**Session 1 — 2026-08-27, 192.168.0.88, ps5debug-NG, Unity/IL2CPP title as
`eboot.bin` (pid 91).** MemDBG deliberately not loaded, so the whole MemDBG
column is still untested.

- [x] §0 ps5debug-NG loaded, RDX connects and lists processes — 87 processes,
      10 ms.
- [x] Memory map: 307 rows, 4.26 GiB, 31 ms. Main module `executable` at
      `0x400000`. Game identity `eboot.bin:f413b7d937a14416649e`.
- [x] TurboScan capability probe: version 1, engines `0x03FF` (all, including
      `TSE_SNAPSHOT`), 4 threads.
- [x] Region classifier: 265 ranges, 2,076 MiB correctly flagged uncached/GPU.
- [x] **Found and fixed: the value scanner was skipping 96% of the game.**
      See patch56 in RELEASE_NOTES. Writable scope 0.18 → 2.15 GiB.
- [x] §2 First Scan, exact value, `u32`, recommended scope — works on both
      engines. **TurboScan 965 matches in 0.8 s; host 965 in 138.8 s (165×).**
- [x] §2 Scan-engine parity, turbo vs host, same value/scope. Same count
      (965/965), 963 addresses shared, 2 turbo-only, 2 host-only.

      **This is agreement, not a discrepancy — verify the control before
      reading a difference here as a bug.** The two engines cannot be run
      simultaneously, and on a live title the heap churns continuously. Two
      *turbo* scans 0.8 s apart already differ by 2 addresses, and two turbo
      scans 141 s apart differ by 6 — more than the 4 that separated turbo
      from host over the same interval. An engine therefore agrees with the
      other engine at least as well as it agrees with itself. Judge parity by
      candidate *count* and by the size of the symmetric difference relative
      to a same-engine control at the same time gap, never by an exact
      address-set match.

**Title: Enter the Gungeon (CUSA01659), Unity/IL2CPP.**

- [x] §2 **Unknown-value snapshot scan (CC11 `TS_SNAPSHOT`)** — the feature
      the release notes call "explicitly the least-verified in this release".
      24 MiB probe scope returned 6,291,456 slots = exactly 24 MiB / 4,
      **100.0% coverage**. `TS_SNAPSHOT_INCLUDE_ZEROS` is honoured: 1,772,792
      of 2,000,000 fetched slots are zero, so the zero-parity assumption the
      release notes flag as a risk holds on this payload. Server kept the
      full 6.29 M list while the client fetched its 2 M cap, exactly as
      designed.
- [x] §2 **CC13 record layout VALIDATED.** 14 randomly sampled *non-zero*
      snapshot values matched independent `ps5_read` reads, 14/14. This is
      the "wire framing derived from two partially-conflicting protocol
      descriptions" the notes single out — it is correct.
      *Sample non-zero values only:* 88% of slots are zero, so a random
      sample is ~all zeros and a wrong field offset would also read zero.
      The first attempt at this check was vacuous for exactly that reason.
- [x] §2 **CC12 relational narrow.** unchanged 6,280,291 + changed 10,360 =
      6,290,651 of 6,291,456 — a **100.0% partition**, so the comparison is
      discriminating rather than passing everything through.
- [x] §3 Apply a value from Results — verified writes to two live heap
      addresses, ack + read-back both confirmed.
- [x] §3 **Confirmed visually in-game by the operator.** Writing 7777 to both
      ammo addresses changed the on-screen ammo counter. This is the half of
      the item that cannot be self-verified: RDX can prove a write landed in
      memory and survived read-back, but only a human can confirm the game
      actually consumed it. Both addresses also tracked together from 1000
      down to 988 while the operator was firing, independently corroborating
      that they are the live pair.
- [x] §4 **A death does NOT create a relocation epoch in this title.** After
      dying, 19 of 20 saved chains still resolved to the *same* address
      (`0x227a1e648`), which the operator confirmed still drives on-screen
      ammo. Enter the Gungeon recycles the allocation rather than freeing it,
      so `_validate_pointer_provisionals` correctly reports "reload not
      detected" (`same_pid AND same_target`) and survivals stay at 0.
      **Use a full game restart for reload validation on this title**; a
      death or floor change is too cheap. Zero chains failed to resolve
      across the death, which does confirm the module-rebasing path works —
      it simply had nothing to rebase.
- [x] §3 **Bulk freeze write (`0xBDAACC04`)** — two simultaneous flat
      freezes held 9999 for 84 s, both `ON` throughout, **386 bulk exchanges
      carrying 772 entries**. The one per-write call is the tick between the
      two toggles when only a single cheat was enabled, which is the
      documented single-target behaviour.
- [x] §5 Export skip logic — heap addresses are correctly kept in the RDX
      trainer as `session_bound` and correctly skipped from the GoldHEN JSON
      and `.mc4` ("address is not in the target module"), which cannot encode
      anything but module-relative patches.

**Operational note:** running two heavy scans concurrently (a 12-socket
batch read plus a 6-socket AOB scan) exhausted ps5debug-NG's connection
capacity and returned `ConnectionError: PS5 disconnected`. The UI cannot do
this on its own, but scripted sessions must not overlap heavy operations.

### Session 1, continued — full sweep

- [x] §2 **All ten value types** scan correctly on turbo:
      `u8 i8 u16 i16 u32 i32 u64 i64 f32 f64`. Counts are type-appropriate
      and the result cap reports `TRUNCATED` correctly (u8/i32/f32 hit 2 M).
- [x] §2 Aligned vs unaligned: 18 vs 19 hits for the same value — unaligned
      correctly finds a superset, at ~15× the cost (0.78 s vs 11.98 s).
- [x] §2 Region scope: readable 32 > writable 18 = recommended 18. Monotonic,
      as expected.
- [x] §2 **AOB scan with `??` wildcards** — validated by a four-way control,
      not by a bare hit count. Bytes were read from a known address, then:
      exact pattern → finds it; wildcarded pattern → finds it; a deliberately
      corrupted byte → correctly misses; that same byte wildcarded → correctly
      hits. `scan_next_pattern` revalidation also exercised.
      *A pattern that returns 0 proves nothing on its own — always run the
      positive control.* (`CUSA01659` is genuinely absent from eboot's 20 MiB
      image; it lives in `param.json`/SceShellCore, not process memory.)
- [x] §2 **Scan cancellation** — Esc during a ~140 s host scan stopped the
      worker in **0.0 s**, state usable afterwards. Confirms the F-08 fix;
      before it, cancellation waited on the in-flight 32 MiB chunk.
- [x] §2 **CC12 `increased by N`, the delta-operand path** — controlled test:
      wrote 4000, snapshotted 32,768 slots, wrote 4007, narrowed by
      "increased by 7" → **exactly 1 survivor, the address written**. This is
      the operand-carrying relational mode that had never run on hardware.
- [x] §3 **Drop a result stays dropped** (F-02 on hardware, `engine: auto`):
      18 candidates → drop one → Next Scan returns 17 and the dropped address
      does **not** reappear.
- [x] §3 Undo Scan restores the wider set (17 → 18) and discards the resident
      session.
- [x] §1 **Large batch-read correctness** — `ps5_read_batch` window
      coalescing: 12/12 random addresses matched individual reads, zero stray
      addresses outside the request, 99.8% coverage (the remainder is normal
      heap churn between scan and read).

#### New defects found on hardware

- **F-13 — `CMD_PROC_SCAN` (0xBDAA0009) is not implemented by this payload.**
  The "Console only" scan engine accepts both handshakes with
  `STATUS_SUCCESS`, then never emits a single result byte; RDX errors after
  `_recv_exact_cancel`'s 15 s inactivity budget. Reproduced on a **1.6 MiB**
  scope with a rare value, so it is not slowness — a scan that small should
  return in milliseconds.
  *Impact:* none on `auto` (turbo is tried first and wins), but "Console
  only" is dead, and any payload without TurboScan would pay a 15 s stall on
  every scan before falling through to host. Worth caching the failure per
  host the way `_memdbg_maps_v2_supported` already does for MemDBG.
  *Note:* an early attempt to test this scanned for value `0`, which matches
  nearly everything and streams millions of addresses 8 bytes at a time —
  that hang was the test's fault, not the payload's. Use a rare value.

- **F-14 — dropping a result under `engine: turbo` breaks the next scan.**
  The F-02 fix correctly discards the resident session on a drop, but
  `scan_next` re-raises in turbo-only mode rather than degrading, so the next
  Next Scan dies with `no matching resident TurboScan session` and keeps
  failing until a new First Scan. Raising is the documented turbo-only
  contract; the problem is that the message never mentions the drop, so the
  user cannot tell what they did or how to recover. Message-quality fix.

### Pointer workflow (§4)

- [x] **Find Permanent Pointer on a live moving address** — 21.3 min over
      4.24 GiB, four depth levels. **164 candidates, all 164 verified**, with
      genuine module-relative roots in `executable` and
      `Il2CppUserAssemblies.prx` at depth 2, confidence 94–95%.
- [x] Chains resolve correctly: 6/6 sampled chains rebased through
      `_pointer_module_base` and resolved to the live ammo address.
- [x] Pointer Project reads `0/2` with the provisionals persisted — RDX
      correctly refuses to call any of them permanent on same-session
      evidence.
- [x] **F-07 confirmed in the wild.** The run reported
      `method=locality-first, index_built=False` — the 8-aligned-only
      streaming scanner found chains and returned early, so the reverse
      index (the only path that scans the 4-byte residue) never ran. It
      found good chains here, but this is exactly why §1's alignment item
      must be read against the method the log names.
- [x] Disk-backed index selected correctly (4.24 GiB ≥ 1 GiB threshold),
      though locality-first won before it was needed.

### Import (§5)

- [x] `.mc4` import — 3 entries resolved against the **live** main module
      base (`0x400000`), not the file's claim. Multiple `<Cheatline>`s under
      one `<Cheat>` correctly de-duplicated to `Test Patch B` /
      `Test Patch B (2)`.
- [x] etaHEN/GoldHEN JSON import — identical result via the shared converter.
- [x] `.mc4` declaring a **different process** → rejected outright.
- [x] `.mc4` declaring a **different Title ID** → imported with a logged
      warning, which is the documented behaviour (there is no cryptographic
      way to verify a foreign file's claim).
- [x] Native `.rdx.json` export → re-import round trip: module-relative entry
      comes back `portable=True`, not import-locked.
- [x] Same file with a tampered `game_identity` → **rejected**: "trainer
      game-image fingerprint does not match the currently attached title".

### Still blocked — these need a human or a payload change

- [ ] **The MemDBG column below §0.** Session 2 loaded MemDBG and cleared
      §0 — connect, process list, maps and native reads all verified, and
      found the connection-per-read bug (patch117) doing it. Still unrun on
      MemDBG: §3's native batch write, `PROCESS_MAPS_V2`, and §6's
      pointer-seed alignment question.
- [ ] **Second half of the two-reload pointer validation.** 164 provisional
      chains are saved and verified; promoting them to permanent requires
      reloading the game, re-isolating the value at its new address, and
      re-running Resolve Permanent — twice.
- [ ] **`.mc4` against a live CheatRunner.** Port 9999 refused; CheatRunner
      is not running. The file format is validated (round trip + FIPS-197
      AES KAT + real third-party sample), but no CheatRunner has ever
      consumed a file this tool produced.
- [ ] **Numba relational scans** — N/A, numba is not installed. Deliberately
      not installed mid-session.

Report results by checking items off and noting the game/title used. A failure
on any item should include: console model/firmware, payload (ps5debug-NG or
MemDBG) and version, game/process, and the exact steps that reproduced it.

## 0. Environment

- [ ] ps5debug-NG loaded, RDX connects and lists processes.
- [x] MemDBG loaded on native TCP 9020, RDX auto-detects it and reports
      "native memory I/O" (not "compatibility fallback") in the connect log.
      *(Session 2: MemDBG 0.2.0-nightly.153.)*
- [ ] An early/older MemDBG build (if available) without native reads still
      works via its ps5debug-compatible TCP 744 listener.
- [ ] Reconnect (command palette) works after a payload restart without
      relaunching RDX.

## 1. Regression spot-checks (tied to specific fixes made this cycle)

These target the exact bugs found and fixed during code review. Each one
failed a specific way before its fix — reproduce the scenario below and
confirm the described correct behavior, not just "it doesn't crash."

- [ ] **4-byte-aligned pointer holders.** Pick a target value stored inside a
      packed/mixed-width struct (common in inventory/entity tables) where the
      pointer to it sits at a 4-byte-aligned (not 8-byte-aligned) offset.
      Run Find Permanent Pointer on a **large** process (enough readable
      memory to trigger the disk-backed reverse index — check the log for
      "Disk pointer index built") as well as a smaller one (RAM index).
      Both reverse indexes should find the holder; that is what was broken,
      and only in the disk-backed index for large processes.

      **Read a miss here carefully — it is not automatically a regression.**
      Only three code paths scan both the 0- and 4-byte alignment residues:
      the fast direct pass (`_fast_direct_pointer_hits`, but it looks
      ±0x100 around the target and no further) and the two reverse indexes.
      The streaming scanner `pointer_chain_scan` still reads the 8-aligned
      `uint64` grid only, and `_resolve_permanent_candidates` runs it as its
      "locality-first" phase *before* the reverse index, returning early if
      it verifies any static candidate. So a 4-byte-aligned holder outside
      that ±0x100 window is only reachable when the locality phase finds
      nothing at all. Check the log for which method won
      (`fast-direct` / `locality-first` / `reverse-index`) before filing it:
      a miss under `locality-first` is the known limitation, a miss under
      `reverse-index` is a real regression.
- [ ] **Results live values on the MemDBG backend.** With MemDBG active and
      native reads advertised (connect log says "native memory I/O"), run any
      First Scan and open Results. Every row's value column must fill in
      within ~2 s and keep refreshing; the age indicator in the status bar
      should cycle rather than sit at `⟳ fetching…`. Before patch55 the
      refresh thread died instantly on this backend and every row stayed `…`
      forever, with nothing logged — and only when MemDBG was working, since
      the port-744 fallback path was unaffected.

- [ ] **Drop, then Next Scan, with TurboScan active.** Drop a result (`D`)
      from Results *or* from the Address Inspector, then run a Next Scan.
      The dropped address must not come back. Same root cause as the Undo
      Scan item below — a server-resident TurboScan session outliving a
      client-side change to the candidate list — but via four call sites the
      original Undo fix did not cover. Also worth one pass of: run a Turbo
      first scan, switch **Scan Settings → engine: host**, Next Scan, switch
      back to `auto`, Next Scan. The final count must reflect the host-path
      narrowing, not the original scan's.

- [ ] **`libSce*` static-module recognition.** If a candidate holder resolves
      into a `libSce*`-named mapping without a `.sprx`/`.prx`/`.elf` suffix,
      confirm it's treated as a static root (visible in the pointer
      candidate's module name / static flag), not rejected as a transient
      anonymous allocation.
- [ ] **Undo Scan → Next Scan, via the Results "More actions" menu
      specifically** (not the direct `U` key). After a Next Scan, open
      **[M] More actions → Undo Scan**, then immediately run another Next
      Scan. Confirm the second scan's candidate count reflects the
      *restored* (larger, pre-undo) set, not the narrower pre-undo-attempt
      set. This only reproduces when TurboScan is the active engine (Scan
      Settings → engine `auto` or `turbo`, the default) — the bug was a
      stale server-resident TurboScan session being reused after an undo
      that went through the menu path.
- [ ] **Write/freeze into an address covered by an overlapping map row.**
      Hard to force directly, but watch for it: if any Apply/Freeze/Write
      is rejected with "mapped but not writable" on an address you're
      confident is writable game data, note the exact address and the
      output of a fresh `[?]`/log dump of that region's map rows — this was
      the class of bug (a smaller read-only overlay row shadowing a larger
      writable one).
- [ ] **Numba relational scans, only if `numba` is installed** (`pip install
      numba`; check the connect/startup log or `python3 -c "import numba"`).
      Run a "decreased by N" or "increased by N" Next Scan on a `u8`/`u16`/
      `u32` value where the arithmetic wraps (e.g. a small counter that's
      about to go from a low value to near its max, or vice versa). Confirm
      the wrapped match is still found. Without numba installed this path
      isn't exercised at all — the pure-NumPy fallback was never affected.
- [ ] **Large Next Scan batch correctness.** Run a First Scan with a large
      result set (tens of thousands+) on a big process, then Next Scan.
      Confirm results are plausible (spot-check a couple of addresses'
      values manually). This exercises `ps5_read_batch`'s window-coalescing
      path, which had an unreachable dead branch removed but whose live path
      had no direct test coverage before this cycle.

## 2. Scanning

- [ ] First Scan: each value type (`u8/i8/u16/i16/u32/i32/u64/i64/f32/f64`)
      against a known in-game value, both aligned and unaligned.
- [ ] First Scan: unknown-value (blank) snapshot, then narrow with
      changed/unchanged/increased/decreased.
- [ ] First Scan: AOB/raw-byte pattern with `??` wildcards.
- [ ] Next Scan: exact-value refinement to a small result set.
- [ ] Next Scan: relational modes, including "decreased by"/"increased by"
      with an explicit delta.
- [ ] Next Scan: AOB pattern revalidation.
- [ ] Scan cancellation (mid-scan) leaves state usable, doesn't hang.
- [ ] Scan engine override (Scan Settings): force `console`, `turbo`, `host`
      individually and confirm each still produces correct results.
- [ ] Recommended vs. all-writable vs. all-readable region scope produce the
      expected difference in result counts.
- [ ] **New: server-side unknown-value snapshot scan.** This is the single
      least-verified feature in this codebase — a large, new protocol
      surface (CC11 `TS_SNAPSHOT`, CC12 relational narrowing, CC13 typed
      value fetching) that has only ever been exercised against scripted
      mock sockets, never a real console, and it silently replaces the
      previously-working client-side unknown-value scan whenever it thinks
      it's available. Test deliberately and skeptically:
      - Unknown-value (blank) First Scan against a real known-changing value
        (e.g. HP), on a large region (Recommended scope on a big game) —
        confirm the found-candidate count is *plausible*, not just nonzero.
      - Narrow with **every** relational mode — changed, unchanged,
        increased, decreased, increased by N, decreased by N — each against
        a value you're independently changing in-game, and confirm the
        surviving addresses/values are correct at each step, not just that
        the scan "completes".
      - Compare candidate counts against forcing `Scan Settings → engine:
        host` for the *same* scan (blank value, same region) — the two
        counts should match. A mismatch means the `TS_SNAPSHOT_INCLUDE_ZEROS`
        parity assumption this was built on (see RELEASE_NOTES) is wrong for
        this payload/build, and every candidate the host path would have
        found via zero-valued slots is being silently dropped.
      - Force `engine: turbo` specifically for an unknown-value scan (not
        just Auto) and confirm it errors clearly rather than silently
        falling back, since Turbo-only mode is supposed to raise instead of
        degrade.
      - Undo Scan after a snapshot-mode narrow — confirm it restores the
        wider candidate set correctly, same as it already does for
        exact-value TurboScan sessions.

## 3. Results & cheats

- [ ] Apply a value once from Results; confirm it changes in-game.
- [ ] Create a freeze cheat and a write cheat from Results.
- [ ] Drop a result (`D`), confirm it's excluded from subsequent scans.
- [ ] Undo Scan via both the direct `U` key and the `[M]` menu path — both
      should behave identically (see the regression check above).
- [ ] Toggle two or more saved cheats' freezes independently (`F`/Space in
      Cheat List) — confirm they don't disable each other and both show
      ON/OFF/ERR status correctly.
- [ ] Manual timed freeze (`/` → Freeze): confirm it stops automatically on
      the configured duration and immediately on reconnect/process change.
- [ ] Edit a cheat's name/value/type — confirm memory is untouched until
      Apply/Freeze is explicitly used afterward.
- [ ] **New: bulk freeze write (`0xBDAACC04`).** Enable 2+ flat (non-pointer)
      cheats' freezes simultaneously and confirm both values stay locked
      in-game exactly as before — this is the first real console exercise
      of the new bulk-write protocol command; it was only verified against
      a scripted mock socket. If it silently regresses, freezes would still
      "work" via the per-write fallback for a single target but could
      misbehave with 2+ simultaneous flat freezes.
- [ ] **New: MemDBG native batch write.** With the MemDBG backend active
      (`memdbg-experimental`) and its daemon advertising `BATCH_WRITE`,
      repeat the 2+-simultaneous-flat-freeze check above — this exercises a
      completely separate wire command from the ps5debug-NG bulk write and
      has never touched a real MemDBG daemon.
- [ ] **New: MemDBG `PROCESS_MAPS_V2`.** With the MemDBG backend active,
      confirm scanning/pointer resolution/export still work normally (map
      listing is used everywhere). If the daemon doesn't support V2, RDX
      should silently fall back to the plain `PROCESS_MAPS` command after
      one failed attempt — check the log for a repeated V2 failure on every
      single map fetch, which would indicate the per-host "unsupported"
      cache isn't taking effect.

## 4. Pointer workflow

- [ ] Find Permanent Pointer on a moving address → provisional chains saved
      (check `[P] Pointer Project` shows 0/2 → after this step).
- [ ] Reload the game/scene, re-isolate the value at its new address, run
      Resolve Permanent again → survival 1/2.
- [ ] Reload once more, repeat → survival 2/2, chain offered for save.
- [ ] Save the resolved chain as a pointer cheat; confirm it resolves
      correctly to the live address afterward (inspect the cheat).
- [ ] Reconnect (new session, same game) and confirm the saved
      cross-reload-validated pointer cheat still resolves without needing
      to re-run the pointer workflow.
- [ ] Manual pointer verify (`V`): enter a known base + offsets directly,
      confirm it resolves and can be saved.

## AOB anchoring — VERIFIED ON HARDWARE (2026-08-30, pid 93)

All four checks pass against a live game over ps5debug-NG:

- [x] **Capture refused on writable memory.** A heap address returned None
      without a read being issued. The constraint that makes an anchor stable
      holds in practice.
- [x] **Capture on real code** — `0x420000`, 32 bytes, `lead=16`, 7 ms. The
      captured bytes are genuine x86-64 (`48 8B BD 30 FF FF FF` =
      `mov rdi,[rbp-0xD0]`, then `test r12,r12`, then a short jump), so it
      anchored on an instruction stream rather than padding.
- [x] **Matches where captured**, and a wrong-bytes control does not.
- [x] **Unique across the whole process** — `AOB scan: 1 matches`. Relocation
      returned `0x420000`, exactly the captured address, so the `lead`
      arithmetic is right in both directions.

**The measurement that mattered: 32 bytes is enough.**
`_AOB_SIGNATURE_BYTES = 32` does not need raising. `relocate_by_aob_signature`
refuses any signature matching more than one site, so had a 32-byte window over
real code been ambiguous the whole feature would have been unusable. No unit
test could have answered this — it needed real code with real repetition.

Cost: relocation is a full host AOB scan, 29.0 s. Acceptable for a once-per-run
operation; see the open note on AOB having no console-side engine.

## Superseded note (kept for the record)

`capture_aob_signature`, `aob_signature_matches` and
`relocate_by_aob_signature` are implemented and unit-tested, but the console
went off the network before any of them ran against real code. Two minutes of
console time settles it:

- [ ] **Capture on a real code address.** Pick an `r-x` module region, capture,
      and confirm a signature comes back rather than None.
- [ ] **It matches where it was captured**, and a wrong-bytes control does not.
- [ ] **It is unique across the process.** This is the one that actually
      matters. `relocate_by_aob_signature` refuses any signature matching more
      than one site, so if a 32-byte window over real game code is *not*
      unique, `_AOB_SIGNATURE_BYTES` is too small and the default needs
      raising. A single measurement decides it.
- [ ] **Round trip is exact** — relocate returns the address captured, not the
      window start. The `lead` arithmetic is the only place that can be off by
      a fixed amount, and a unit test cannot catch a wrong constant that both
      sides share.

Until that runs, treat AOB anchoring as implemented-not-proven. The refusal
paths are the tested ones; the success path is not.

## 5. Export / Import

- [x] **Round-trip verified with live console data (2026-08-30, pid 97).**
      Previously exercised only against fixtures; this is the first time the
      formats carried addresses RDX itself scanned off a running game.

      `.rdx.json`  3 cheats, `format=rdx-pointer-trainer-v1`, addresses
                   `0x169e230 / 0x16a163c / 0x16a4364` and values `0x64`
                   all preserved, `game_identity` carried through
      `.shn`       well-formed CheatRunner XML — `<Offset>0X169E230</Offset>`,
                   `<ValueOn>64-00-00-00</ValueOn>` (correct little-endian)
      `.mc4`       decrypts to XML byte-identical to the `.shn`, and
                   `mc4_xml_to_mods` returns all 3 mods with every offset and
                   value intact

      Note for future readers: `mc4_xml_to_mods` returns `(meta, mods)` — a
      metadata dict first, then the list. Unpacking it the other way round
      silently yields the metadata's key count as a "mod count", which is what
      first made this look like a 3-in / 5-out discrepancy.

- [x] **A live CheatRunner loaded a trainer RDX produced (2026-08-30).**
      Generated through RDX's real export path — `generate_etahen_json` ->
      `generate_mc4_bytes` / `generate_shn_text` — from addresses scanned off
      the running game, named `CUSA01659_01.00.{mc4,shn,json}` to GoldHEN's
      convention. CheatRunner loaded **all three** — the encrypted `.mc4`, the
      plaintext `.shn` and the etaHEN/GoldHEN `.json`. Schema, AES/base64
      container and JSON path are all acceptable to a real consumer, which no
      file this project produced had ever demonstrated.

      The `.shn`-beside-`.mc4` diagnostic went unused because nothing failed.
      Keep emitting it: it exists to separate a container fault from a schema
      fault, and the day either appears, the pair is what makes it legible.

      The cheats themselves did nothing when toggled, by construction — see
      the Address Mode item below. A user looking at them in CheatRunner will
      reasonably think the trainer is broken, which is worth remembering if
      this test is ever repeated.

      Offsets were module-relative, as GoldHEN expects:
      `0x169E230 - 0x400000 = 0x129E230`.

- [ ] **Still open: the Address Mode.** The test cheats were deliberately
      harmless — `ValueOn` and `ValueOff` both `64-00-00-00`, the value already
      at the address — so toggling wrote 100 over 100 and nothing observable
      happened. That safety is exactly what makes the result inconclusive on
      whether CheatRunner treats the offset as module-relative or absolute.
      Deciding it needs one cheat whose `ValueOn` differs from the live value,
      with RDX polling the address while the user toggles.

- [ ] ~~**No live CheatRunner has consumed a file RDX produced.**~~ (answered)
      Port 9999 has been refused in every session. The `.shn` is emitted beside
      the `.mc4` precisely to split that test: `.shn` accepted + `.mc4`
      rejected isolates the fault to the AES/base64 container; both rejected
      isolates it to the schema.


- [ ] Export with a `PPSA...` Title ID → confirm both the `.rdx.json` and
      the etaHEN `.json` are written, and the log reports
      `/data/etaHEN/cheats/json/` as the deploy path.
- [ ] Export with a `CUSA...` Title ID → confirm the GoldHEN JSON schema and
      `/user/data/GoldHEN/cheats/json/` deploy path.
- [ ] Export with multiple games'/sessions' cheats in the list → confirm
      only the currently-attached game's cheats are included, and the
      "Excluded stale/other-game cheats" count is accurate.
- [ ] Export a mix of pointer cheats and module-relative scalar cheats →
      confirm the etaHEN JSON only contains the scalar ones, and the log
      reports the pointer ones as skipped (not silently dropped without
      explanation).
- [ ] Import a previously exported `.rdx.json` on the same game/title →
      cheats load and resolve correctly.
- [ ] Import that same file against a **different** attached game → import
      is rejected with a game-identity-mismatch error, not silently
      accepted.
- [ ] **New: CheatRunner `.mc4` export.** Export a game with at least one
      static module-relative cheat, upload the generated `.mc4` to
      `/data/cheatrunner/cheats/mc4/` via FTP, and confirm CheatRunner's web
      UI (`http://<PS5-IP>:9999`) lists it and the game name/title match.
      Toggle the cheat on from CheatRunner and confirm the value actually
      changes in-game; toggle off and confirm it restores. This is the one
      export path that has only been verified against a real published
      third-party `.mc4` sample and a from-scratch AES-256 implementation —
      never against a live CheatRunner instance.
- [ ] **New: `.mc4`/etaHEN/GoldHEN JSON import.** While attached to a real
      game, import a real community `.mc4` trainer for that exact title (or
      an etaHEN/GoldHEN JSON exported earlier by this tool). Confirm each
      entry's resolved address is correct (it's computed live from the
      *currently attached* process's main module — never trusted from the
      file — so this only proves out against a live console) and that
      applying one actually changes the expected value in-game. Also import
      a `.mc4` for the **wrong** title while attached to a different game and
      confirm the addresses land somewhere sane rather than silently
      resolving to garbage (RDX warns on a Title ID mismatch but does not
      block the import, since it cannot verify the file's claimed identity
      the way it can its own `.rdx.json` format).
- [ ] **New: per-cheat export selection.** With 2+ eligible cheats, use the
      new checkbox picker before Export to deselect one, and confirm the
      written `.rdx.json`/etaHEN JSON/`.mc4` only contain the cheats left
      selected.
- [ ] **New: Cheat List delete-undo (`Z`).** Delete a cheat from the Cheat
      List screen, confirm it's gone, press `Z`, confirm it reappears at
      (approximately) its original position with its freeze state and values
      intact.

## 6. Open question to resolve during testing

- [ ] **MemDBG native pointer-seed alignment.** With MemDBG active (native
      backend), run Find Permanent Pointer on a target you've confirmed (via
      the ps5debug-compatible path, or by disabling MemDBG) has its *only*
      real holder at a 4-byte-aligned (not 8-byte) offset. See whether the
      MemDBG-backed run still finds a correct permanent pointer (it should,
      via fallback to the software scanner — see the code comment on
      `_MemDBGClient.pointer_holders`) or whether it misses it / finds a
      different, less direct chain than the ps5debug-compatible path would.
      This is the one remaining behavior that could not be verified without
      a live MemDBG daemon.


## Session — patch77/78 on Enter the Gungeon (CUSA01659), pid 101 then 138

Console 192.168.0.88, ps5debug-NG on 744 (+klog 3232). MemDBG not loaded, so
that column remains untested.

### Validated
- [x] Process list 87-88 entries, 13 ms; memory map 307 rows / 4.26 GiB, 23 ms.
      Both far under the patch74 caps (4096 / 65536) — no false rejection.
- [x] Exact scan i32: 195,469 hits in 1.32 s. Positive control 12/12 sampled
      hits genuinely held the value; negative control (rare value) 0 hits;
      unaligned >= aligned.
- [x] Narrowing across three real decrements: 1,485 -> 2 -> 2, no false drops.
- [x] `scan_next` batch path: 23,768 -> 2 in 6.28 s. (A naive per-address read
      loop over the same set timed out past 120 s — the coalesced path is the
      one to use, roughly 200x.)
- [x] `ps5_write_verified` correctly reported `verified=False` when the game
      overwrote a write to a mirror address, and `True` on the authoritative
      one. Source-vs-mirror separated by write persistence: 0/25 vs 25/25.
- [x] Writes visible on screen (777, 999 confirmed by the user).
- [x] **Freeze end to end**: held 999 in 24/25 samples over 10 s via the real
      `_toggle_cheat_freeze` worker; status `active @ 0x...`; clean stop.
- [x] Native `.rdx.json` export + round-trip through the importer's validation:
      identity matched, addresses validated, values parsed, `session_bound`
      correctly set.
- [x] GoldHEN/etaHEN export correctly **refused** both heap-address cheats
      ("address is not in the target module") — the skip report is the intended
      behaviour, and the UI only writes `.mc4` when mods exist.
- [x] **Debugger lifecycle, first hardware run**: attach -> arm DR7 -> wait ->
      clear watchpoint -> detach -> resume. Game remained fully responsive.
- [x] Fingerprint stability across a full game restart: identical
      (`eboot.bin:f332c6a77fb51d3a63c6` before and after). The earlier mismatch
      against a stored value came from an older session, not a design fault.

### Failed / found
- [x] **Two-reload pointer validation did its job.** Five chains that looked
      perfect in-session (`verified=True`, confidence 96) resolved to unrelated
      memory after a restart. Same-session evidence is not evidence; this is
      hardware proof of why the two-reload gate exists.
- [x] `_PTR_FAST_DIRECT_RANGE` was 256 bytes; the real IL2CPP displacement was
      0x90F8 (~37 KiB), so the cheap pass found nothing and the search fell
      through to a 30+ minute streaming scan. Fixed in patch77 (window is free
      to widen: 6.4-7.5 s at every width).
- [x] A failed debug detach permanently disabled tracing and said nothing.
      Fixed in patch86. Recovery requires reloading ps5debug-NG.

### Still open
- [ ] Watchpoint **capture** (the trace firing on a real write) — blocked on
      reloading the payload after the stuck session.
- [ ] Pointer promotion 0/2. Needs chains derived from a watchpoint trace
      rather than a range scan, then two reloads.
- [ ] MemDBG backend, entirely.
- [ ] `.mc4` against a live CheatRunner.


## Debugger attach: works, but crashed a live game (observed)

Over a LAN route (Tailscale off) the full lifecycle succeeded on the first try:

```
ATTACH pid 188 -> 0x80000000 CMD_SUCCESS
event channel: ('192.168.0.88', 59079)     <- console dialled back to the client
DETACH  -> 0x80000000 CMD_SUCCESS
```

That inbound connection on port 755 is the thing that can never happen over a
VPN/overlay route, and confirms patch86's diagnosis.

**However, seconds later the game was black with no sound, and ps5debug-NG and
klog (744, 3232) were gone.** The console itself stayed up: ICMP fine, the ELF
loader on 9295 still listening, the UI still able to close the game, so the
jailbreak survived and no reboot was needed.

Most likely mechanism: `CMD_DEBUG_ATTACH` calls `ptrace(PT_ATTACH)`, which
stops the process. A Unity/IL2CPP title stopped mid-GPU-submit can lose its
render context permanently, and because ps5debug-NG is hosted inside
`SceShellCore`, a fault propagating there takes the payload with it.

Both commands returned `CMD_SUCCESS` and the teardown ran correctly, so this is
not a client-side defect. It is the inherent risk of attaching a debugger to a
live title, and it is why `_DEBUG_TRACE_ENABLED` defaults to False.

Calibration note (third in this file): an earlier lifecycle test on pid 138
attached and detached with the game unaffected, and that single success was
treated as proof the operation was safe. It was not. "The lifecycle completes"
and "the game survives it" are different claims, and only the first had been
demonstrated.

- [x] Debugger attach/detach lifecycle over LAN — works, event channel confirmed
- [ ] Watchpoint capture — not attempted; the game died before a trace was armed
- [ ] Establish whether the crash is reproducible, and whether pausing at a
      safe moment (menu rather than active gameplay) avoids it


## Watchpoint: armed successfully, never trapped (unresolved)

Second and third attach attempts, both with the game paused first.

- Attach, arm and teardown all succeeded. The pause-menu approach appears to
  matter: the attempt that armed and resumed left the game running normally,
  where an earlier attach during active play black-screened it.
- The target has **40 threads**.
- With a write-only watchpoint armed on the verified ammo address, the value
  demonstrably changed (523 -> 518) and **no event fired** in 60 s.

Ruled out as causes:

- **Client packet shape.** RDX sends the documented 24-byte
  `cmd_debug_watchpt_packet` (`<IIIIQ`); encodings match reference 7.3
  (`breaktype=1` write-only, `length=3` for 4 bytes); the address is 4-byte
  aligned. `ps5dbg` sends 22 bytes for the same command, which is short of the
  documented struct — so RDX is the better-formed of the two.
- **Thread fan-out being the client's job.** Neither `ps5dbg` nor `ps4debug`
  iterates threads; both issue one process-wide call with no `lwpid`. An
  "arm every thread" change would diverge from the whole lineage with no
  evidence it is required.

Remaining hypotheses (all payload/firmware side, none testable without further
attaches): DR registers not honoured for this store on this firmware; the store
reaching the page through a different mapping; or the payload applying DRs to
only the debug-context thread.

- [x] Attach / arm / teardown lifecycle over LAN
- [x] Pause-before-attach appears to avoid the render-context crash (2 of 2)
- [ ] **Watchpoint capture — does not fire; cause unresolved, payload-side**
- [ ] Pointer promotion 0/2 (blocked on a non-coincidental chain)

Calibration note: an "arm all 40 threads" fix was drafted on the strength of
`_debug_free_watchpoint(lwpid)` taking a thread id. Checking the reference
clients first would have shown that the arming call carries no `lwpid` at all
and that no upstream client iterates threads. The repo comparison should have
come before the code change, not after.

## Session 4 — Phase 2: writer-instruction anchor, PROVEN on hardware

Console 192.168.0.88, fw 10.01, ps5debug-NG by OSR v1.3.0 (protocol 1.3),
game eboot.bin pid 89. One attach, one watchpoint, one resume, one event.

Chain executed end to end and verified:

    temporary address 0x00032a153f74 (u32, ammo)
      -> write watchpoint  slot 3, length_code=3 (4B), breaktype=1 (write-only)
         armed-state verified by decoding DR7=0xd00004c0:
           L3/G3 set, R/W3=0b01 (write), LEN3=0b11 (4 bytes)
      -> debug event  0x4A0 bytes, lwpid 101676, status 0x57F (SIGTRAP-stop)
      -> event RIP    0x018f5b5b
      -> writer       0x018f5b55  89 8B 24 01 00 00
                                  mov dword ptr [rbx+0x124], ecx
      -> AOB          32 bytes, mask all-FF, lead 16
      -> relocation   exactly 1 process-wide match -> 0x018f5b55

Writer proof (operand math, not event metadata):
  rbx = 0x32a153e50 ; rbx+0x124 = 0x00032a153f74 == watched address
  rcx = 0x25 = 37                                == live ammo value

FINDINGS TO CARRY FORWARD

1. The event RIP is NOT the writer. x86 hardware data breakpoints are
   trap-type: they report the instruction AFTER the store. The writer is
   at RIP - (length of the storing instruction) = RIP-6 here. Any code
   that equates event RIP with the writer will anchor one instruction too
   late. Decode backwards from RIP and confirm the operand resolves to
   the watched address before trusting it.

2. DR6 read back as 0x0 on a genuine watchpoint hit. The payload clears
   DR6 while handling the trap, so DR6 CANNOT be used to confirm that an
   event was a watchpoint hit, nor to identify which slot fired. Proof
   must come from the effective-address computation.

3. The ammo field is rewritten EVERY FRAME, not only on fire. The event
   arrived 0.0 s after resume, before any shot. A "trigger the action"
   step is not required for frequently-written fields, and an event
   arriving instantly is not evidence of a stale/queued stop event.

4. Cost: capture_aob_signature 0.01 s; relocate_by_aob_signature
   620.25 s (10.3 min, 4,651 ranges, writable_only=False). This is the
   already-recorded "AOB has no console-side engine" gap, now quantified
   against a real signature. Relocation is correct but far too slow for
   interactive use.

5. prot is an INTEGER bitmask, not a string. The writer's region
   0x400000-0x2308000 has prot=5 (READ|EXEC). Test it with
   `int(prot) & 0x4`; a string match on "x" silently fails. RDX's
   capture/patch paths already use the bitmask correctly — this bit me
   only in a scratch probe.

Attach budget for pid 89 is now spent. Do not reattach to it.

## Session 4 — patch138: writer resolution + executable-scoped AOB relocation

Audit of the event RIP -> writer path found the resolution logic already
correct and principled (backward decode preferring the instruction that *ends*
at the trap RIP, then an effective-address proof — never a hardcoded -6), but
unreachable: the DR6 slot-bit gate discarded every real event. Confirmed
offline against the captured event, which is now checked in as
`tests/golden_watchpoint_event.bin`.

CHANGES (only the two authorized areas)

  writer resolution
    - DR6 is a hint, not a gate. DR6 == 0 falls through to the operand check;
      a non-zero DR6 naming another slot is still rejected.
    - Every candidate event is validated inside the wait loop: the decoded
      effective address must equal the watched address, or the trace resumes
      past it and keeps waiting instead of aborting.
    - `_decoded_effective_address()` extracted so the loop and the post-loop
      invariant share one implementation.
    - The trace result now publishes `writer` alongside `rip`; the UI reports
      `writer` instead of falling back to the raw trap RIP.

  executable-scoped AOB scanning
    - `scan_first_pattern(region_scope="executable")` filters `prot & 0x4`.
    - `relocate_by_aob_signature` passes it.

GOLDEN REGRESSION (hardware, read-only, no attach)

    signature   0F94C5...C5FA1183   32 bytes, mask all-FF, lead 16
    expected    0x018f5b55
    relocated   0x018f5b55          1 match (unique)
    runtime     4.76 s              was 620.25 s  -> 130x

PROFILE THAT JUSTIFIED IT

    total mapped        13,151.2 MiB
    executable             47.0 MiB   (70 regions, all r-x)
    scanned before      9,765.2 MiB   99.52% could never hold a match
    read vs match       99.8% / 0.2%  (reads 3.0 MiB/s/thread,
                                       matching 1,627 MiB/s/thread)

  Rejected after measuring: coalescing adjacent executable ranges (70 -> 70,
  the modules are non-adjacent); larger reads (network-bound); faster matching
  (already 0.2% of time); RPC batching (already one socket per worker).

  The 4.76 s vs the 2.7 s projection is region-classifier startup — it still
  enumerates all 4,651 ranges before the scope filter applies. Not worth
  chasing at this size.

NOT CHANGED: the 32-byte signature, its mask, `_AOB_SIGNATURE_BYTES`, the
minimum-literal-byte guard, the zero/one/many match rules, the watchpoint
protocol.

STILL OPEN: `capture_aob_signature` and `relocate_by_aob_signature` have no
production callers. The anchor pipeline is validated end to end but is not yet
wired into the UI workflow.


## Session 4 — patch139: anchor pipeline wired

The primitives proven in patch138 now have production callers. Acquisition
(needs a debugger attach) and patching (does not) are deliberately separate, so
an anchor can be captured once and applied later without spending another
attach on the target process.

  _instruction_anchor_contract()   canonical result; `writer` mandatory,
                                   `trap_rip` diagnostic-only
  capture_instruction_anchor()     writer -> AOB -> unique relocation
  verify_instruction_anchor()      re-prove before any write
  patch_instruction_anchor()       refuses unless verification passes
  restore_instruction_anchor()     inverse, via the existing mechanism
  anchor_to_json / anchor_from_json    portable artifact, version-checked
  do_capture_instruction_anchor()  deliberate UI operation

STATIC AUDIT: capture_aob_signature has exactly one caller, which passes
anchor["writer"]; patch_instruction is reachable only through
patch_instruction_anchor (after verification) and restore_instruction. No path
can supply the raw trap RIP as an instruction address.

STILL NOT DONE ON HARDWARE: the patch/restore step. capture -> relocate is
hardware-proven; NOP-ing the writer and confirming the effect in-game is the
next live test, and it needs a fresh game process because pid 89's attach is
spent.

## Release status — patch140

Labels: VERIFIED (ran on hardware and passed) · NOT YET TESTED (never run) ·
KNOWN LIMITATION (ran, and this is the honest result) · OPTIONAL.

VERIFIED
  - Write watchpoint arms and fires (fw 10.01, ps5debug-NG v1.3.0, slot 3,
    DR7 = 0xd00004c0, watched 0x00032a153f74)
  - Writer resolution: trap RIP 0x018f5b5b -> writer 0x018f5b55, proved by
    rbx+0x124 == watched address and ecx == the live value
  - 32-byte AOB capture in executable, non-writable memory
  - Unique process-wide relocation back to 0x018f5b55, in 4.53 s
  - Detach clean; game survived every session
  - Scan/write/freeze/pointer/import/export coverage from earlier sessions

  - Patch and restore on hardware, through the production layer only
    (see the Session 4 patch/restore log below)

NOT YET TESTED
  - The in-game *effect* of a patch. The bytes were written and restored and
    verified by independent readback, but the player was not firing during the
    window, so ammo stayed at 33 throughout and no gameplay change was
    observed. The mechanism is verified; the gameplay consequence is not.
  - Exported-trainer Address Mode in CheatRunner (module-relative vs absolute)
  - Two-reload promotion of a pointer chain to permanent

KNOWN LIMITATION
  - One successful debugger attach per game-process lifetime. A second attach
    returns CMD_ERROR 0xF0000001. Relaunch the game to get another.
  - DR6 reads back as 0 on a genuine hit; it cannot identify which slot fired.
    Ownership is proved by the decoded operand instead.
  - AOB relocation has no console-side engine; it streams memory to the host.
    Restricting to executable mappings brought this from 620.25 s to 4.53 s.
  - _KLASS_NAME_OFFSETS ordering is correct by luck, not by construction.

OPTIONAL
  - Numba acceleration (requires `numba`)

CORRECTED PREVIOUS ENTRY: earlier notes recorded that watchpoints "do not work
on ps5debug-NG", based on two attaches that timed out on CMD_DEBUG_CONTINUE.
Those reached MemDBG's ps5debug-compat shim on port 744; the payload identity
was never checked with CMD_BRANDING at the time. Against real ps5debug-NG the
full chain works. Verify payload identity before drawing conclusions from it.


## Session 4 — patch/restore on hardware, VERIFIED

Ran entirely through the production layer (capture_instruction_anchor ->
patch_instruction_anchor -> restore_instruction_anchor). No debugger attach was
needed: patching writes with ps5_write, so pid 89's spent attach was not a
barrier. The verification layer was not bypassed at any point.

    pid                  89 (eboot.bin, still alive afterwards)
    writer address       0x018f5b55
    original bytes       898B24010000
    relocated address    0x018f5b55   (AOB scan: 1 match, 4.68 s)
    verification         verified, match_count = 1
    patched bytes        909090909090
    independent readback 909090909090  (matches)
    observed effect      none observable -- ammo held at 33, player not firing
    restored bytes       898B24010000  (matches original)
    restore verification ok
    detach               n/a, no attach was taken
    game after           alive, pid 89

NOTE ON RESTORE: the restore's own relocation reports "AOB scan: 0 matches",
because the patch has by then destroyed the very bytes the signature
describes. restore_instruction_anchor handles this deliberately -- it falls
back to the recorded, writer-derived address (anchor["relocated"], then
anchor["writer"]), never to the trap RIP. Confirmed correct on hardware.
