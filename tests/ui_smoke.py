#!/usr/bin/env python3
"""Drive the real curses UI in a real terminal, against a fake console.

Why this exists
---------------
Every test in `test_pointer_subsystem.py` stubs curses. That covers the logic
behind each screen but never the drawing, the key handling, or the terminal
itself -- and defects live there. Running the program for real found two that
the whole suite had missed:

  * an unreachable console froze the UI for 16.5 s with no cancel, because
    `memdbg_probe` (1.5 s) plus `ps5_connect`'s 15 s default both ran before
    anything was drawn (patch70);
  * the "terminal too small" branch repainted a static message ten times a
    second, because `getch()` returns -1 every 100 ms (patch70).

It also gave the first visual confirmation that the progress bar renders at
its full width after the `safe_addstr` fix in patch62 -- 58 cells at 110
columns, where the old codepoint-threshold rule drew about 37.

What it does
------------
Forks a pty, sets a known terminal size, launches the UI with every console
call replaced by a fake, and types a scripted key sequence into the terminal
exactly as a user would. Real curses, real terminal, real input -- only the
PS5 is fake. Fails on any traceback or `curses.error`.

Scenarios cover both layout branches of the results screen (the split detail
pane above 92 columns, the plain list below it), a SIGWINCH resize storm, the
progress bar's rendered width, and the sub-minimum terminal path.

Each scenario is checked against injected faults rather than trusted because
it is green: moving the detail pane off-screen makes the layout assertions
fire, and a raw addstr past the window bottom is caught as a traceback.

Usage
-----
    python3 ui_smoke.py            # run every scenario
    python3 ui_smoke.py --list     # show scenario names
    python3 ui_smoke.py screens    # run one scenario

Exit status is 0 when every scenario passes, 1 otherwise.
"""

import fcntl
import os
import pty
import re
import select
import signal
import struct
import sys
import termios
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE.parent / "RDX-CHEATMAKER-UI-final.py"

# The child process: patch out every console call, then hand control to main().
CHILD = r'''
import importlib.util, sys, curses, runpy, re
from pathlib import Path
import numpy as np

launcher = Path(__file__).with_name("RDX-CHEATMAKER-UI-final.py")
spec = importlib.util.spec_from_file_location("rdx", launcher)
m = importlib.util.module_from_spec(spec); sys.modules["rdx"] = m
spec.loader.exec_module(m)

MAPS = [{"start": 0x400000, "end": 0x500000, "prot": 5, "offset": 0,
         "name": "executable"},
        {"start": 0x500000, "end": 0x600000, "prot": 3, "offset": 0,
         "name": "executable"},
        {"start": 0x2000000, "end": 0x2400000, "prot": 3, "offset": 0,
         "name": ""}]

class FakeSock:
    def close(self): pass
    def settimeout(self, *_): pass
    def sendall(self, *_): pass

def slow_scan(ip, pid, value, width, aligned=True, progress_cb=None,
              cancel_event=None, **kw):
    """Slow enough that the progress bar is actually painted."""
    import time
    for done in range(0, 401, 20):
        if progress_cb:
            progress_cb(done, 400)
        time.sleep(0.09)
    return np.array([0x500100, 0x500200, 0x500300], dtype=np.uint64)

# A scenario may seed a damaged pointer-project file. Redirect the path to a
# temp copy so the user's real .rdx-pointer-candidates.json is never touched.
import os as _os, tempfile as _tf
_seed = _os.environ.get("RDX_SMOKE_PROJECT_FILE_CONTENT")
if _seed is not None:
    _p = Path(_tf.mkdtemp()) / ".rdx-pointer-candidates.json"
    _p.write_text(_seed)
    m._POINTER_PROVISIONAL_FILE = _p

m.memdbg_probe        = lambda ip, timeout=1.5: None
m._console_preflight  = lambda ip, timeout=3.0: True
m.ps5_connect         = lambda ip, timeout=15.0: FakeSock()
m.ps5_proc_list       = lambda ip: [{"pid": 91, "name": "eboot.bin"},
                                    {"pid": 57, "name": "SceShellCore"}]
m.ps5_maps            = lambda ip, pid: MAPS
m.ps5_read            = lambda ip, pid, a, n: b"\x2a" + b"\x00" * (n - 1)
m.ps5_write           = lambda *a, **kw: True
m.ps5_write_verified  = lambda ip, pid, a, d: (True, True, d)
m.ps5_classify_regions = lambda ip, pid, **kw: []
m.scan_first          = slow_scan
m.scan_next           = lambda *a, **kw: np.array([0x500100], dtype=np.uint64)
m.scan_first_unknown  = lambda *a, **kw: (np.array([0x500100], dtype=np.uint64),
                                          np.array([42], dtype=np.uint32))

curses.wrapper(m.main)
print("UIDRIVE_OK")
'''

UP, DOWN, ESC = b"\x1b[A", b"\x1b[B", b"\x1b"

# connect -> process -> First Scan -> value -> RESULTS, shared by several
# scenarios so each one only has to script the part it is actually testing.
SCAN_TO_RESULTS = [(b"\n", 1.5), (b"\n", 1.5), (b"s", 1.0), (b"\n", 0.7),
                   (b"42\n", 0.7), (b"\n", 0.7), (b"\n", 4.0)]

# (keys, seconds to wait after sending) plus the strings we expect to appear.
SCENARIOS = {
    "screens": {
        "size": (34, 110),
        "keys": [(b"\n", 1.5), (b"\n", 1.5), (b"?", 1.0), (b" ", 0.8),
                 (b"/", 1.0), (b"\x1b", 0.8), (b"t", 1.0), (b"\x1b", 0.8),
                 (b"c", 1.0), (b"q", 0.8), (b"p", 1.2), (b"\x1b", 0.8),
                 (b"q", 1.0)],
        # patch88 renamed SCAN SETTINGS -> SETTINGS: the screen now carries
        # the region and pointer tunables too, not just the scan engine.
        # "Pointer max depth" also proves the tunable rows actually render.
        "expect": ["CONNECT", "SELECT PROCESS", "RDX CHEAT MAKER",
                   "Keyboard Help", "COMMAND PALETTE", "SETTINGS",
                   "Pointer max depth", "CHEAT LIST", "POINTER PROJECT"],
    },
    "scan": {
        "size": (34, 110),
        "keys": [(b"\n", 1.5), (b"\n", 1.5), (b"s", 1.0), (b"\n", 0.7),
                 (b"42\n", 0.7), (b"\n", 0.7), (b"\n", 4.0), (b"\x1b", 1.0),
                 (b"q", 1.0)],
        "expect": ["FIRST SCAN", "SCANNING", "RESULTS"],
        # min(w - 8, 60) - 2, the inner width draw_progress_bar asks for
        "bar_width": 58,
    },
    # patch90/96: the hex pane and structure overlay only exist in a real
    # terminal -- the unit suite stubs curses, so nothing there proves these
    # actually render or that their key routing reaches them.
    "viewers": {
        "size": (34, 110),
        "keys": [(b"\n", 1.5), (b"\n", 1.5), (b"s", 1.0), (b"\n", 0.7),
                 (b"42\n", 0.7), (b"\n", 0.7), (b"\n", 4.0),
                 (b"b", 0.6),                      # bookmark a result
                 (b"\n", 1.0),                     # inspect it
                 (b"h", 1.2), (b"\x1b", 0.8),      # hex view, back
                 (b"s", 1.2), (b"\x1b", 0.8),      # structure view, back
                 (b"\x1b", 0.8), (b"\x1b", 0.8),
                 (b"q", 1.0)],
        "expect": ["RESULTS", "ADDRESS INSPECTOR", "HEX VIEW", "STRUCTURE"],
    },
    "tiny": {
        "size": (20, 60),
        "keys": [(b"q", 1.0)],
        "expect": ["Terminal too small"],
        # A static message must not be repainted on every -1 from getch().
        "max_repaints": ("Terminal too small", 2),
    },
    # do_show_results takes a different layout branch at w >= 92 (split view
    # with a detail pane) than below it. Both need driving in a real terminal:
    # the pane arithmetic is exactly where an off-by-one becomes curses.error.
    "results_wide": {
        "size": (34, 110),
        "keys": SCAN_TO_RESULTS + [
            (DOWN, 0.5), (DOWN, 0.5), (UP, 0.5),
            (b"\n", 1.0), (ESC, 0.7),      # inspect, back
            (b"m", 1.0), (ESC, 0.7),       # more menu, back
            (b"d", 1.0), (ESC, 0.7),       # drop, back out
            (b"u", 0.8),                   # undo
            (b"a", 1.0), (ESC, 0.8),       # add cheat, back out
            (b"q", 0.8), (b"q", 0.8), (b"\n", 0.8),
        ],
        # "SELECTED ADDRESS" is drawn only in the split branch, so it proves
        # the wide layout was actually reached rather than silently skipped.
        "expect": ["RESULTS", "SELECTED ADDRESS", "Create cheat"],
    },
    "results_narrow": {
        "size": (30, 80),                  # below the 92-column split threshold
        "keys": SCAN_TO_RESULTS + [
            (DOWN, 0.5), (b"\n", 1.0), (ESC, 0.7),
            (b"m", 1.0), (ESC, 0.7),
            (b"q", 0.8), (b"q", 0.8), (b"\n", 0.8),
        ],
        "expect": ["RESULTS"],
        "forbid": ["SELECTED ADDRESS"],   # must NOT take the split branch
    },
    # screen_main reads the persisted pointer project on every entry, so a
    # damaged file there took the UI down immediately after connecting
    # (patch76). The file is redirected to a temp copy inside the child.
    "corrupt_project_file": {
        "size": (34, 110),
        "env": {"RDX_SMOKE_PROJECT_FILE_CONTENT": "[]"},
        "keys": [(b"\n", 1.5), (b"\n", 1.5), (b"p", 1.2), (ESC, 0.8),
                 (b"q", 1.0)],
        "expect": ["RDX CHEAT MAKER", "POINTER PROJECT"],
    },
    # A resize mid-screen is the classic way to provoke curses.error: the
    # window shrinks between getmaxyx() and the addstr that trusted it.
    "resize": {
        "size": (34, 110),
        "keys": SCAN_TO_RESULTS + [(DOWN, 0.4)],
        "resize_to": [(24, 80), (40, 130), (24, 74), (34, 110)],
        # Reaching 130 columns must produce the split pane; if it never
        # appears the resize never took effect and the scenario is vacuous.
        "expect": ["RESULTS", "SELECTED ADDRESS"],
    },
}


def run(scenario: dict) -> tuple:
    rows, cols = scenario["size"]
    child = HERE.parent / ".ui_smoke_child.py"
    child.write_text(CHILD)
    try:
        pid, fd = pty.fork()
        if pid == 0:
            os.environ["TERM"] = "xterm-256color"
            for key, value in scenario.get("env", {}).items():
                os.environ[key] = value
            os.execvp("python3", ["python3", str(child)])
        fcntl.ioctl(fd, termios.TIOCSWINSZ,
                    struct.pack("HHHH", rows, cols, 0, 0))
        buf = bytearray()

        def pump(seconds: float) -> bool:
            end = time.time() + seconds
            while time.time() < end:
                ready, _, _ = select.select([fd], [], [], 0.12)
                if ready:
                    try:
                        chunk = os.read(fd, 1 << 16)
                    except OSError:
                        return False
                    if not chunk:
                        return False
                    buf.extend(chunk)
            return True

        try:
            pump(2.5)
            for keys, wait in scenario["keys"]:
                os.write(fd, keys)
                if not pump(wait):
                    break
            # Resize the real pty and signal it, the way a window manager
            # would. SIGWINCH is what turns this into a KEY_RESIZE.
            for new_rows, new_cols in scenario.get("resize_to", []):
                fcntl.ioctl(fd, termios.TIOCSWINSZ,
                            struct.pack("HHHH", new_rows, new_cols, 0, 0))
                try:
                    os.kill(pid, signal.SIGWINCH)
                except Exception:
                    pass
                if not pump(1.2):
                    break
            pump(1.5)
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except Exception:
                pass
    finally:
        child.unlink(missing_ok=True)

    text = buf.decode("utf-8", "replace")
    visible = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][B0]|\x1b[=>]|\r",
                     "", text)
    return visible, []


def check(name: str, scenario: dict) -> list:
    visible, problems = run(scenario)
    if "Traceback" in visible:
        tail = visible[visible.find("Traceback"):].splitlines()[:12]
        problems.append("traceback:\n      " + "\n      ".join(tail))
    if "curses.error" in visible:
        problems.append("curses.error raised")
    for marker in scenario.get("expect", []):
        if marker not in visible:
            problems.append(f"never rendered: {marker!r}")
    for marker in scenario.get("forbid", []):
        if marker in visible:
            problems.append(f"rendered but must not be: {marker!r} "
                            "(wrong layout branch taken)")
    want_bar = scenario.get("bar_width")
    if want_bar:
        bars = re.findall(r"\[([█░]+)\]", visible)
        if not bars:
            problems.append("progress bar never drawn")
        else:
            widest = max(len(b) for b in bars)
            if widest != want_bar:
                problems.append(
                    f"progress bar drew {widest} cells, expected {want_bar} "
                    "(safe_addstr width accounting)")
    repaint = scenario.get("max_repaints")
    if repaint:
        marker, limit = repaint
        seen = visible.count(marker)
        if seen > limit:
            problems.append(
                f"{marker!r} repainted {seen} times (limit {limit}) -- "
                "a static screen is being redrawn in a busy loop")
    return problems


def main(argv: list) -> int:
    if "--list" in argv:
        print("\n".join(SCENARIOS))
        return 0
    wanted = [a for a in argv[1:] if not a.startswith("-")] or list(SCENARIOS)
    failed = 0
    for name in wanted:
        scenario = SCENARIOS.get(name)
        if scenario is None:
            print(f"  {name}: unknown scenario")
            failed += 1
            continue
        problems = check(name, scenario)
        if problems:
            failed += 1
            print(f"  {name}: FAIL")
            for problem in problems:
                print(f"      {problem}")
        else:
            print(f"  {name}: ok")
    print(f"\n{len(wanted) - failed}/{len(wanted)} scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if not LAUNCHER.exists():
        print(f"launcher not found: {LAUNCHER}")
        sys.exit(1)
    sys.exit(main(sys.argv))
