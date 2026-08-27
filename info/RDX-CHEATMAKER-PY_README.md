<img width="1612" height="334" alt="image" src="https://github.com/user-attachments/assets/db4bede8-e81d-495e-b739-c35302c3a5d0" />

# PS5/PS4 Python Cheat Maker (Terminal UI)
Memory inspection and modification tool for research and homebrew development.

A lightweight terminal-based PS5 memory scanner and cheat creator built for Linux, Windows, and macOS.
Connects to a PS5 running [`ps5debug-NG`](https://github.com/OpenSourcereR-dev/ps5debug-NG), or experimentally to [`MemDBG`](https://github.com/seregonwar/MemDBG), and allows you to:

- Scan signed/unsigned integers, floats, doubles, and wildcard byte patterns
- Refine scans to locate dynamic addresses
- Create freeze/write cheats
- Edit memory directly
- Resolve moving addresses to cross-reload-validated pointer chains
- Export native RDX trainers and compatible GoldHEN/etaHEN static patches
- Toggle multiple frozen cheats independently in real time

The project is designed to be fast and easy to use entirely from a terminal.
The source-by-source design review is recorded in
[`UPSTREAM_SCANNER_AUDIT.md`](UPSTREAM_SCANNER_AUDIT.md).

---

## Features

### Memory Scanning

- Exact value scanning
- First Scan / Next Scan workflow
- uint8 support
- int8 support
- uint16 support
- int16 support
- uint32 support
- int32 support
- uint64 support
- int64 support
- float32 and float64 support with configurable refinement tolerance
- AOB/raw-byte scanning with `??` wildcards
- Unknown/changed/unchanged/increased/decreased scans for every numeric type
- Recommended game-region filtering that excludes obvious payload/libraries
- Aligned scanning (fast)
- Unaligned scanning (thorough)
- Scan progress indicators
- Scan cancellation
- Undo scan refinement
- Memory-efficient result storage

### Memory Editing

- Read memory values
- Write memory values
- Persistent independent cheat toggles plus bounded manual timed freezes
- Address validation
- Remembered reconnect workflow

### Cheat Management

- Create cheats directly from scan results
- Freeze cheats
- Write cheats
- Edit cheats
- Delete cheats
- Multiple cheats per game

### Trainer Export

Generate pointer-capable `*.rdx.json` trainers. RDX also writes the shared
GoldHEN/etaHEN checkbox JSON schema when an entry is a main-module-relative
byte patch those managers can represent. Pointer chains and live freezes remain
in the native RDX trainer. Export includes a destination chooser and preflight.

### Terminal UI

- Pure curses interface
- No GUI required
- Keyboard navigation
- Live value refresh
- Scrollable logs
- Process filtering
- Remembered console IP, preferred process, and export destination
- Visible, resumable Pointer Project with 0/2 → 2/2 reload status

---

## Requirements

### PS5

- Jailbroken PS5
- Supported firmware
- ps5debug-NG loaded; or current MemDBG on native TCP 9020

### Computer

- Python 3.10 or newer
- NumPy is required
- Numba and psutil are optional accelerators/telemetry helpers

### Python Dependencies

Install the required dependency with `python3 -m pip install numpy`. Optional:
`python3 -m pip install numba psutil`.

---

## Quick Start

### 1. Load a compatible payload
Start ps5debug-NG, or MemDBG on native port 9020. Current MemDBG builds provide
native reads/writes; RDX uses port 744 only as a fallback for early builds.

### 2. Run the Tool

```bash
python3 RDX-CHEATMAKER-UI-final.py
```

### 3. Connect to Your PS5

Enter the IP address displayed on your console.

### 4. Select the Game Process

Choose the game process from the list. You can type to filter processes.

<img width="1632" height="744" alt="image" src="https://github.com/user-attachments/assets/8d7b35a5-0861-4bca-a601-f99be4e040f8" />

### 5. Perform a First Scan

Suppose your current health is:

```text
100
```

Select:

```text
[S] First Scan
```

Choose:

```text
Unsigned 32-bit (u32)
```

Enter:

```text
100
```

Keep `recommended game regions` unless the value is known to live in an
unusual read-only mapping.

### 6. Change the Value In-Game

Example:

```text
100 → 87
```

### 7. Perform a Next Scan

Select:

```text
[N] Next Scan
```

Enter:

```text
87
```

Repeat until only a few addresses remain.

### 8. Create a Cheat

Open:

```text
[R] Results
```

Select an address and press:

```text
C
```

Enter:

```text
Infinite Health
```

Choose:

```text
freeze
```

or

```text
write
```

### 9. Make a Moving Address Portable

If the result moves after a scene/save reload, select it in Results and press
`R` for **Find Permanent Pointer**. This stores provisional candidates. Reload
the game, isolate the value at its new address, and press `R` for survival one;
reload and repeat once more for survival two. Only then does RDX allow the
cross-reload pointer to be saved as permanent. The main-menu `P` Pointer
Project shows progress and resumes this workflow after a reconnect/reload.

### 10. Export the Cheat

Open the command palette with `/` and select:

```text
Export Trainers
```

Enter:

```text
Title ID
Version
Game Title
Cheat Author
Output Directory
```

Example:

```text
PPSA01234
01.00
Example Game
```

Generated output always includes the native pointer-capable trainer:

```text
PPSA01234_01.00.rdx.json
```

When eligible main-module scalar/raw-byte patches exist, RDX also writes
`PPSA01234_01.00.json` for etaHEN. A `CUSA` Title ID produces the same verified
schema for GoldHEN and reports `/user/data/GoldHEN/cheats/json/`; a `PPSA`
Title ID reports `/data/etaHEN/cheats/json/`. Pointer chains remain native.

---

## Main Menu

| Key | Function |
|-------|------------|
| S | First Scan |
| N | Next Scan |
| R | Results |
| P | Pointer Project |
| C | Cheat List |
| T | Scan Settings |
| / | Command Palette (export/import/write/freeze/pointer/log tools) |
| ? | Keyboard Help |
| Q | Quit |

---

## Results Screen

| Key | Function |
|-------|------------|
| ↑ ↓ | Navigate |
| Enter | Inspect selected address |
| A | Apply a value once |
| C | Add cheat |
| R | Find permanent pointer |
| N | Next Scan |
| D | Drop Address |
| U | Undo Scan |
| M | More actions |
| Q | Back |

---

## Cheat List Screen

| Key | Function |
|-------|------------|
| ↑ ↓ | Navigate |
| Enter | Inspect cheat |
| A | Apply once |
| F / Space | Toggle this cheat ON/OFF without disabling other cheats |
| E | Edit cheat metadata/value without writing memory |
| D | Delete Cheat |
| Q | Back |

---

## Native RDX Export Format

Example exported cheat:

```json
{
  "title": "Example Game",
  "titleid": "PPSA01234",
  "version": "01.004.000",
  "process": "eboot.bin",
  "format": "rdx-pointer-trainer-v1",
  "game_identity": "eboot.bin:example-fingerprint",
  "cheatList": [
    {
      "name": "Infinite Health",
      "type": "freeze",
      "address": "0x12345678",
      "module_name": "executable",
      "module_relative_offset": "0x345678",
      "value": "0x63",
      "value_type": "u32",
      "bytes": 4,
      "original_value": "0x64",
      "session_bound": false,
      "game_identity": "eboot.bin:example-fingerprint"
    }
  ]
}
```

## Supported Data Types

| Type | Size |
|--------|--------|
| uint8 | 1 byte |
| int8 | 1 byte |
| uint16 | 2 bytes |
| int16 | 2 bytes |
| uint32 | 4 bytes |
| int32 | 4 bytes |
| uint64 | 8 bytes |
| int64 | 8 bytes |
| float32 | 4 bytes |
| float64 | 8 bytes |
| raw bytes / AOB | 1–256 bytes; `??` wildcards while scanning |

---

## Safety Features

The application includes several safeguards:

- Blocks writes to address 0x0
- Blocks writes to kernel-space addresses
- Validates value sizes before writing
- Rejects NaN/infinity, signed overflow, malformed hex, and wildcard writes
- Binds writes/freezes to the owning process, session, and game image
- Keeps imported absolute/provisional entries locked until rebuilt live
- Makes cheat editing metadata-only; Apply and Freeze are explicit actions
- Handles connection failures gracefully
- Supports scan cancellation
- Supports scan undo

---

## Troubleshooting

### Cannot Connect

Verify:

- PS5 and PC are on the same network
- ps5debug-NG or MemDBG is running
- Correct IP address entered
- Firewall is not blocking TCP 744 (ps5debug-NG) or TCP 9020 (MemDBG)
- For an early MemDBG build without native reads, enable its optional
  ps5debug-compatible TCP 744 listener

### No Results Found

Try:

- Correct signed/unsigned/float value type
- Unaligned scanning
- All writable or all readable scope if Recommended omits the mapping
- A blank first value for an unknown-value snapshot, then changed/decreased
- Perform additional scan refinements

### Cheat Does Not Work

Verify:

- Correct game version
- Correct Title ID
- Correct address and typed data width
- For moving values, complete Find Permanent Pointer across two reloads
- Native RDX trainers are re-imported into RDX
- GoldHEN CUSA JSON goes to `/user/data/GoldHEN/cheats/json/`
- etaHEN PPSA JSON goes to `/data/etaHEN/cheats/json/`

---

## Disclaimer

This project is intended for educational, research, and homebrew development purposes on systems that you own and control.

The authors assume no responsibility for any damage, data loss, bans, or other consequences resulting from use of this software.

Use at your own risk.

---

## Credits

- ps5debug developers
- ps5debug-NG and MemDBG developers
- PS4Cheater / PointerFinder tool authors
- PS5-MemoryPeeker, ps5dbg, PINCE, and Cheat Engine contributors
- GoldHEN/etaHEN developers and cheat contributors
- PS5 homebrew community
- All contributors and testers

---

## License

MIT License

See the LICENSE file for details.
