<img width="1612" height="334" alt="image" src="https://github.com/user-attachments/assets/db4bede8-e81d-495e-b739-c35302c3a5d0" />

# RDX CheatMaker

A memory scanner and cheat builder for a **jailbroken PlayStation 5**, run from
a terminal on your PC. Find a value in a running game, change it, freeze it,
and export a trainer other tools can load.

Version 1.0.0.

---

## What it does

- **Find values.** Search for a number you can see on screen (ammo, health,
  money) across ten numeric types, or search without knowing the value and
  narrow it down as it changes. Also searches raw byte patterns with wildcards.
- **Change and freeze.** Write new values, or hold a value fixed so the game
  cannot change it back.
- **Follow values that move.** A game usually stores things at a different
  address every launch. RDX can find a stable route to the value so a cheat
  keeps working after a restart.
- **Find what writes a value.** Put a hardware watchpoint on an address and RDX
  identifies the exact machine instruction that changed it.
- **Export trainers.** Saves in `.mc4`, `.shn`, GoldHEN/etaHEN JSON, and its own
  `.rdx.json`. Imports all of them too.

## Requirements

- A **jailbroken PS5** running a debug payload — **ps5debug-NG** (recommended)
  or **MemDBG**.
- Your PC and the PS5 on the **same network**.
- **Python 3.10 or newer** with **numpy**.
- A terminal at least 80×24.

## Install

```bash
pip install numpy
```

That is the only requirement. `numba` and `psutil` are optional and only add
speed or extra detail — RDX works fine without them. See `requirements.txt`.

## Run

```bash
python3 RDX-CHEATMAKER-UI-final.py
```

On first launch it asks for your PS5's IP address. Find it on the console under
**Settings → Network → View Connection Status**. RDX remembers it next time.

## Basic workflow

1. **Load your payload** on the PS5 and start the game.
2. **Connect** — enter the console's IP address.
3. **Pick the game** from the process list (usually `eboot.bin`).
4. **First scan** — enter the value you can see, e.g. ammo `112`.
5. **Change it in game** — fire a shot, so the value becomes something else.
6. **Next scan** — enter the new value. Repeat until few results remain.
7. **Edit or freeze** the value you found.
8. **Export** a trainer if you want to keep it.

The narrowing step is the whole trick: each scan keeps only the addresses that
changed the way your value changed.

## Experimental features

Two menu entries are marked **(experimental)**. Both attach a debugger to the
running game and ask for confirmation first:

- **Trace Write → Find Pointer** — uses a write event to find a stable route to
  the value.
- **Trace Write → Instruction Anchor** — identifies the instruction that writes
  the value, and can disable it.

**These are riskier than scanning.** They pause the game briefly and, on a bad
day, can crash it. Use a game you do not mind restarting.

**Important:** a game process allows **one debugger attach per launch**. If you
attach once and want to try again, **restart the game first**.

## Limitations

- The **in-game effect of disabling an instruction has not been confirmed** in
  testing — the mechanism works, but the gameplay result was never observed.
- **Exported trainer address mode is unverified** — trainers load in
  CheatRunner, but whether it reads offsets the way RDX writes them has not
  been confirmed with a live cheat.
- **Permanent pointer routes are not fully proven.** RDX requires a route to
  survive two game restarts before calling it permanent; that full cycle has not
  been completed end to end.
- Tested against **Enter the Gungeon** on firmware **10.01**. Other titles and
  firmwares should work but have not been checked.

## Troubleshooting

**Cannot connect.** Check the payload is loaded and the IP is right. RDX uses
port 744 (ps5debug-NG) or 9020 (MemDBG). Both devices must be on the same
network — a VPN on your PC will usually break it.

**No processes listed.** Start the game before connecting, then reconnect.

**Scans find nothing.** Your value may not be stored as you expect — try
**Auto** type, or unaligned scanning. Health is often a float, not an integer.

**Too many results.** Change the value in game and scan again. A few rounds
usually gets you to a handful.

**A trace says "no usable write event".** The value may not have been written
during the window. Trigger it more actively, or restart the game — the attach
may already be spent.

**The game froze or crashed.** Restart it. If RDX exits badly it tries to
release the debugger, but restarting the game is always the safe reset.

## Advanced documentation

- **`docs/DEVELOPMENT.md`** — architecture, critical invariants, hardware
  findings, and a handoff guide for developers and AI agents.
- **`docs/PS5DEBUG-NG-PROTOCOL.md`** — payload protocol reference.
- **`HARDWARE_TEST_CHECKLIST.md`** — exactly what has and has not been tested
  on real hardware.
- **`RELEASE_NOTES.md`** — what changed and when.

## Safety

RDX only touches the memory of the process you select. It refuses to attach to
system and jailbreak infrastructure. Instruction patches are verified before
being written and can be restored. Even so: **use a game you can afford to
restart.**

## Disclaimer

For your own console and your own games. Modifying game memory can crash the
game or corrupt saves. Do not use it to gain an advantage over other people in
online play. You are responsible for how you use it.

## License

No licence is specified, so all rights are reserved by default. You are welcome
to read the code and run it; if you want to reuse or redistribute it, ask. See
`LICENSE`.
