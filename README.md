<img width="1612" height="334" alt="image" src="https://github.com/user-attachments/assets/db4bede8-e81d-495e-b739-c35302c3a5d0" />

# RDX CheatMaker

A terminal memory scanner and cheat maker for jailbroken PS5/PS4, written in
Python. Built for memory research and homebrew development: locate values,
refine them to a single address, resolve moving addresses to permanent pointer
chains, and export trainers that GoldHEN, etaHEN and CheatRunner can load.

Runs entirely in a terminal on Linux, macOS and Windows. One file, one hard
dependency.

```bash
python3 -m pip install numpy
python3 RDX-CHEATMAKER-UI.py
```


---

## What it does

**Scanning** — exact-value and unknown-value scans across all ten numeric
types, or **Auto**, which tries the types your value could be and settles on
whichever finds matches (`u8 i8 u16 i16 u32 i32 u64 i64 f32 f64`), plus AOB/raw-byte patterns
with `??` wildcards. Aligned or unaligned. Relational next-scans (changed,
unchanged, increased, decreased, increased/decreased by N) for hunting values
you can't read off the screen.

**Three scan engines**, selected automatically or forced in Scan Settings:

| Engine | Where it runs | Speed on a 2 GiB title |
|---|---|---|
| TurboScan | on the console (ps5debug-NG) | **~0.8 s** |
| Console | on the console (legacy) | payload-dependent |
| Host | streams memory to your PC | ~140 s |

**Find what writes an address** — put a hardware watchpoint on a value, do
the thing that changes it, and RDX identifies the machine instruction that
wrote it. It does not trust the address the console reports: that names the
instruction *after* the store, so RDX decodes backwards and proves the
candidate by recomputing its target from the captured registers. If that does
not equal the address you watched, it keeps waiting rather than guessing.

**Instruction anchors** — the writing instruction sits at a different address
every launch, so RDX captures 32 bytes of surrounding code as a signature and
re-finds it later. A signature that matches in more than one place, or in none,
is refused rather than guessed at. Anchors are saved as portable artifacts, so
one capture can be re-verified and applied in a later session without spending
another debugger attach.

**Verification before writing** — a matching signature is evidence, not
permission. Before any patch RDX re-checks that the signature still relocates
uniquely, that the bytes there are the instruction it captured, and that the
memory is executable and not writable. Any failed check means nothing is
written at all. The original bytes are kept so the patch can be undone.

**Pointer chains** — a moving heap address is useless in a trainer. RDX walks
backwards to a module-relative root, then refuses to call the chain permanent
until it has survived **two real game reloads**. Same-session evidence is
explicitly rejected, because a chain that resolves right now is often just
this session's heap coincidence.

**Cheats** — independent freeze toggles with live ON/OFF/ERR/LOSE status,
one-shot writes, and per-cheat editing. Simultaneous freezes collapse into a
single bulk-write exchange per tick rather than one write per cheat.

> **A freeze cannot hold a value the game rewrites every frame.** This is a
> property of the link, not a tuning problem, and it is worth knowing before
> you reach for the feature. Measured on a real PS5: the game rewrote an ammo
> address every 8–20 ms, while one write round trip costs ~15.7 ms. The freeze
> held the value in 31 of 657 samples (4.7%) at its 200 ms tick, and 47.9% even
> in a tight loop with no delay at all.
>
> Freezing works exactly as advertised for values a game writes only when they
> *change* — currency, item counts, unlock flags, totals. For ammo, health,
> timers and anything else updated per frame, it will lose, and the cheat
> indicator reports **LOSE** rather than pretending otherwise.
>
> The established fix is to patch the instruction that performs the write
> rather than race it — which is what every cheat in the GoldHEN corpus does.
> RDX cannot yet author those; see `info/UPSTREAM_AUDIT_PASS8.md` and
> `PASS9.md`.

**Export / import** — native `.rdx.json` (the only format that carries pointer
chains), GoldHEN/etaHEN checkbox JSON, CheatRunner `.mc4`, and the plaintext
`.shn` twin of that `.mc4`. Import accepts all of them; addresses are always
re-resolved against the live process, never trusted from the file.

**Inspection** — a read-only hex viewer and a structure overlay that gives an
address named, typed fields, auto-dissected against the live memory map. Both
highlight bytes that changed since the last refresh, and pointer fields show
where they point. Both refuse to write: RDX has three audited write paths
already, and a fourth reachable by cursoring around a dump is how a running
game gets corrupted by accident.

**Finding objects, not just values** — Type Scan groups heap allocations by the
type pointer at their base. For an IL2CPP title that pointer is the
`Il2CppClass`, so each group is one class and its live instances. This finds
state that never appears on screen as a number, which the scan/narrow loop
cannot.

**Symbols** — Type Scan resolves class names from live memory by following the
object's own class pointer, so an IL2CPP title labels itself with no external
tooling. For field names, load an Il2CppDumper `dump.cs` and the structure view
uses real field names with their declared types instead of `field_0014`; RDX
does not produce the dump.

**Salvage** — a trainer written for a different build of the game is not a dead
end. Its pointer chains are re-verified against the running build, and the ones
that still resolve can be taken as bookmarks or cheats.

**Bookmarks** — keep an address you are still investigating without turning it
into a cheat. Attach a verified pointer chain and a bookmark rebases on every
attach, surviving reloads; without one it is a raw address and is marked stale
once the process or console session it was taken in is gone.

For the full walkthrough — first scan to exported trainer, screen by screen —
see **[`info/RDX-CHEATMAKER-PY_README.md`](info/RDX-CHEATMAKER-PY_README.md)**.

---



## Requirements

**Console** — a jailbroken PS5 or PS4 running
[ps5debug-NG](https://github.com/OpenSourcereR-dev/ps5debug-NG) (TCP 744), or
[MemDBG](https://github.com/seregonwar/MemDBG) on TCP 9020.

If you run MemDBG, consider keeping ps5debug-NG loaded too: TurboScan, the
console scanner and the region classifier are ps5debug commands with no MemDBG
equivalent, and without them every scan falls back to the slow host path. RDX
warns at connect time when port 744 is unreachable.

**Computer** — Python 3.8 or newer and a terminal at least **72×24**
(100×30 is more comfortable; the Results side panel needs 92 columns).

| Package | Required | Purpose |
|---|---|---|
| `numpy` | yes | vectorised scanning and filtering |
| `numba` | no | parallel JIT for relational scans |
| `psutil` | no | memory telemetry (falls back to `/proc`) |

Everything else is standard library — including the AES and LZ4 implementations,
so no crypto or compression package is needed.

---
## Settings

`T` from the main menu. The pointer bounds and region filter rules used to be
literals in the source; they are still the same defaults, now visible and
editable, and every one is clamped on load so a hand-edited preferences file
cannot request an unbounded pointer walk.

| Setting | Default | What it does |
|---|---|---|
| Scan engine | Auto | Turbo → Console → Host, or force one |
| Region exclude tokens | `.sprx,.prx,.so,…` | Substrings excluded from Recommended scope |
| Min region size | 0 (off) | Skip mappings below this size |
| Pointer max depth | 5 | Chain hops explored — matches PINCE's default |
| Pointer direct window | 0x800 | First-pass holder radius — matches PINCE's default |
| Pointer struct window | 0x4000 | Max \|offset\| per hop |
| Module bases only | off | Keep only chains rooted in a named module |

---

## Safety mechanisms

RDX writes to the memory of a running game. It validates addresses against the
process map before every write, refuses writes into non-writable mappings, and
fails closed when a cheat's process, session or game fingerprint does not match
what is currently attached. Reversible previews restore the original bytes even
when the operation fails partway.

Hardware-watchpoint tracing also requires the **console to be able to reach
this machine**: it is the console that opens the debug channel, outbound to the
client on port 755. A VPN or overlay route (Tailscale and similar) lets you
scan, read, write and freeze perfectly while making tracing impossible, so RDX
checks the return path and refuses before attaching rather than stopping the
game and hanging. Trace from a machine on the console's own network.

Hardware-watchpoint tracing exists but is **disabled by default** — it attaches
a debugger and stops the target process, and a failed teardown can leave the
game frozen.

Tracing also needs **elevated privileges**. The console dials out to the client
on TCP port 755 and ps5debug-NG hard-codes that port, so it cannot be moved;
being below 1024 it is privileged on Linux and macOS. Run RDX as root for
tracing, or grant the interpreter the capability once with
`sudo setcap 'cap_net_bind_service=+ep' $(readlink -f $(command -v python3))`
— noting that this applies to that interpreter for every program it runs.
Nothing else in RDX needs privileges; scanning, writing and freezing are all
outbound connections.

---

## Disclaimer

RDX CheatMaker is a memory inspection and modification tool for educational
and research use, and for homebrew development on hardware you own.

Writing to the memory of a running title can destabilise it and, in the worst
case, require a console restart. No responsibility is accepted for damage,
data loss, or any other consequence arising from use of this software.

---

## Credits

ps5debug and ps5debug-NG developers · MemDBG developers · PS4Cheater and
PointerFinder authors · PS5-MemoryPeeker, ps5dbg, PINCE and Cheat Engine
contributors · GoldHEN and etaHEN developers and cheat contributors · the PS5
homebrew community.

---
## License

No licence is specified, so all rights are reserved by default. You are welcome
to read the code and run it; if you want to reuse or redistribute it, ask.
