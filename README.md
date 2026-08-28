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
types (`u8 i8 u16 i16 u32 i32 u64 i64 f32 f64`), plus AOB/raw-byte patterns
with `??` wildcards. Aligned or unaligned. Relational next-scans (changed,
unchanged, increased, decreased, increased/decreased by N) for hunting values
you can't read off the screen.

**Three scan engines**, selected automatically or forced in Scan Settings:

| Engine | Where it runs | Speed on a 2 GiB title |
|---|---|---|
| TurboScan | on the console (ps5debug-NG) | **~0.8 s** |
| Console | on the console (legacy) | payload-dependent |
| Host | streams memory to your PC | ~140 s |

**Pointer chains** — a moving heap address is useless in a trainer. RDX walks
backwards to a module-relative root, then refuses to call the chain permanent
until it has survived **two real game reloads**. Same-session evidence is
explicitly rejected, because a chain that resolves right now is often just
this session's heap coincidence.

**Cheats** — independent freeze toggles with live ON/OFF/ERR status, one-shot
writes, and per-cheat editing. Simultaneous freezes collapse into a single
bulk-write exchange per tick rather than one write per cheat.

**Export / import** — native `.rdx.json` (the only format that carries pointer
chains), GoldHEN/etaHEN checkbox JSON, and CheatRunner `.mc4`. Import accepts
all three; addresses are always re-resolved against the live process, never
trusted from the file.

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

## Safety mechanisms

RDX writes to the memory of a running game. It validates addresses against the
process map before every write, refuses writes into non-writable mappings, and
fails closed when a cheat's process, session or game fingerprint does not match
what is currently attached. Reversible previews restore the original bytes even
when the operation fails partway.

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

## License

MIT.

