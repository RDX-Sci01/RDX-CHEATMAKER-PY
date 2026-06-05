<img width="1612" height="334" alt="image" src="https://github.com/user-attachments/assets/db4bede8-e81d-495e-b739-c35302c3a5d0" />

# PS5/PS4 Python Cheat Maker (Terminal UI)
Memory inspection and modification tool for research and homebrew development.

A lightweight terminal-based PS5 memory scanner and cheat creator built for Linux, Windows, and macOS.
Connects to a PS5 running the [`ps5debug-NG`](https://github.com/OpenSourcereR-dev/ps5debug-NG) payload and allows you to:

- Scan game memory for values
- Refine scans to locate dynamic addresses
- Create freeze/write cheats
- Edit memory directly
- Export GoldHEN-compatible & Cheatrunner files
- Freeze values in real-time

The project is designed to be fast, dependency-free, and easy to use entirely from a terminal.

---

## Features

### Memory Scanning

- Exact value scanning
- First Scan / Next Scan workflow
- uint8 support
- uint16 support
- uint32 support
- uint64 support
- Aligned scanning (fast)
- Unaligned scanning (thorough)
- Scan progress indicators
- Scan cancellation
- Undo scan refinement
- Memory-efficient result storage

### Memory Editing

- Read memory values
- Write memory values
- Freeze addresses
- Address validation
- Automatic reconnect handling

### Cheat Management

- Create cheats directly from scan results
- Freeze cheats
- Write cheats
- Edit cheats
- Delete cheats
- Multiple cheats per game

### GoldHEN Export

Generate GoldHEN-compatible JSON cheat files.

### Terminal UI

- Pure curses interface
- No GUI required
- Keyboard navigation
- Live value refresh
- Scrollable logs
- Process filtering

---

## Requirements

### PS5

- Jailbroken PS5
- Supported firmware
- ps5debug payload loaded

### Computer

- Python 3.10 or newer

### Python Dependencies

None. Only Python standard library modules are used.

---

## Quick Start

### 1. Load ps5debug

Start the [`ps5debug-NG`](https://github.com/OpenSourcereR-dev/ps5debug-NG) payload on your PS5.

### 2. Run the Tool

```bash
python3 RDX-CHEATMAKER-UI.py
```

### 3. Connect to Your PS5

Enter the IP address displayed on your console.

Example:

```text
192.168.1.120
```

### 4. Select the Game Process

Choose the game process from the list.

You can type to filter processes.

### 5. Perform a First Scan

Suppose your current health is:

```text
100
```

Select:

```text
[S] First Scan
```

Enter:

```text
100
```

Choose:

```text
uint32
```

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
Enter
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

### 9. Export the Cheat

Select:

```text
[E] Export
```

Enter:

```text
Title ID
Version
Game Title
```

Example:

```text
PPSA01234
01.00
Example Game
```

Generated output:

```text
PPSA01234_01_00.json
```

---

## Main Menu

| Key | Function |
|-------|------------|
| S | First Scan |
| N | Next Scan |
| R | Results |
| W | Write Memory |
| F | Freeze Address |
| C | Cheat List |
| E | Export Cheats |
| L | Log Viewer |
| X | Clear Results |
| P | Change Process |
| Q | Quit |

---

## Results Screen

| Key | Function |
|-------|------------|
| ↑ ↓ | Navigate |
| Enter | Add Cheat |
| D | Drop Address |
| U | Undo Scan |
| Q | Back |

---

## Cheat List Screen

| Key | Function |
|-------|------------|
| ↑ ↓ | Navigate |
| Enter | Edit Cheat |
| D | Delete Cheat |
| Q | Back |

---

## Export Format

Example exported cheat:

```json
{
  "title": "Example Game",
  "titleid": "PPSA01234",
  "version": "01.00",
  "cheatList": [
    {
      "name": "Infinite Health",
      "type": "freeze",
      "address": "0x12345678",
      "value": "0x63",
      "bytes": 4
    }
  ]
}
```

---

## Memory Usage

The scanner uses compact 64-bit arrays for storing scan results.

Approximate memory consumption:

| Results | RAM Usage |
|----------|------------|
| 100,000 | ~0.8 MB |
| 250,000 | ~2 MB |
| 500,000 | ~4 MB |

A configurable scan cap prevents excessive memory usage.

---

## Supported Data Types

| Type | Size |
|--------|--------|
| uint8 | 1 byte |
| uint16 | 2 bytes |
| uint32 | 4 bytes |
| uint64 | 8 bytes |

---

## Supported Platforms

| Platform | Status |
|------------|----------|
| Linux | Supported |
| macOS | Supported |
| Windows | Not Supported |

---

## Safety Features

The application includes several safeguards:

- Blocks writes to address 0x0
- Blocks writes to kernel-space addresses
- Validates value sizes before writing
- Handles connection failures gracefully
- Supports scan cancellation
- Supports scan undo

---

## Troubleshooting

### Cannot Connect

Verify:

- PS5 and PC are on the same network
- ps5debug is running
- Correct IP address entered
- Firewall is not blocking connections

### No Results Found

Try:

- Different scan width
- Unaligned scanning
- Verify the value type
- Perform additional scan refinements

### Cheat Does Not Work

Verify:

- Correct game version
- Correct Title ID
- Correct address
- Address is not dynamic
- Cheat exported to the correct GoldHEN folder

---

## Disclaimer

This project is intended for educational, research, and homebrew development purposes on systems that you own and control.

The authors assume no responsibility for any damage, data loss, bans, or other consequences resulting from use of this software.

Use at your own risk.

---

## Credits

- ps5debug developers
- GoldHEN developers
- PS5 homebrew community
- All contributors and testers

---

## License

MIT License

See the LICENSE file for details.
