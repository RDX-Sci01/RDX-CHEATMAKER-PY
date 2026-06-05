#!/usr/bin/env python3
"""
ps5cheats_tui.py — PS5 Cheat Maker with Terminal UI (Linux)
Requires: pip3 install windows-curses (Linux has curses built-in)

Usage:
    python3 ps5cheats_tui.py
"""

import curses
import socket
import struct
import json
import os
import time
import threading
from pathlib import Path

# ── ps5debug protocol ──────────────────────────────────────────────────────
CMD_MAGIC      = 0xFFAABBCC   # every packet starts with this uint32 (little-endian)
# Opcodes — top byte must be 0xBD; middle byte selects namespace
CMD_VERSION    = 0xBD000001   # info namespace (no status word in response)
CMD_PROC_LIST  = 0xBDAA0001   # proc namespace
CMD_PROC_READ  = 0xBDAA0002
CMD_PROC_WRITE = 0xBDAA0003
CMD_PROC_MAPS  = 0xBDAA0004
CMD_PROC_NOP   = 0xBDAACC06   # keepalive / ping
# Status words are bit-swapped on the wire by net_send_int32 (adjacent even/odd
# bit positions swapped). Clients compare the raw wire value directly.
# Server macro CMD_SUCCESS = 0x40000000 → wire value 0x80000000
# Server macro CMD_ERROR   = 0xF0000002 → wire value 0xF0000001
STATUS_SUCCESS = 0x80000000
STATUS_ERROR   = 0xF0000001
PS5_PORT       = 744   # TCP command server; 755 is debug-async (different channel)
# Scan type registry: label -> (struct_fmt, byte_width, is_float)
SCAN_TYPES = {
    "uint8":   ("B",  1, False),
    "uint16":  ("<H", 2, False),
    "uint32":  ("<I", 4, False),
    "uint64":  ("<Q", 8, False),
    "int8":    ("b",  1, False),
    "int16":   ("<h", 2, False),
    "int32":   ("<i", 4, False),
    "int64":   ("<q", 8, False),
    "float32": ("<f", 4, True),
    "float64": ("<d", 8, True),
}
SCAN_TYPE_NAMES = list(SCAN_TYPES.keys())
# Legacy maps kept for compatibility with existing cheat entries
WIDTH_FMT   = {1: "B", 2: "<H", 4: "<I", 8: "<Q"}
VALID_WIDTHS = [1, 2, 4, 8]
WIDTH_LABEL  = {1: "byte (u8)", 2: "uint16", 4: "uint32", 8: "uint64"}

def cmd_header(cmd, datalen=0):
    """
    Build the 12-byte packet header:
        magic   (uint32 LE)  0xFFAABBCC
        cmd     (uint32 LE)
        datalen (uint32 LE)  byte-length of the request body that follows
    """
    return struct.pack("<III", CMD_MAGIC, cmd, datalen)

# ── app state ──────────────────────────────────────────────────────────────
state = {
    "ip": "",
    "connected": False,
    "pid": None,
    "proc_name": "",
    "scan_results": [],
    "scan_width": 4,
    "cheats": [],
    "game_id": "",
    "game_ver": "01.00",
    "game_title": "",
    "log": [],
    "scan_history": [],   # stack of previous result sets for undo
    "scan_type":    "uint32",   # active scan type name
    "region_filter": "all",     # heap / exec / anon / all
    "prev_cache":   {},         # addr->bytes snapshot for change-mode scans
}

# ── ps5debug helpers ───────────────────────────────────────────────────────

def ps5_connect(ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(15)
    s.connect((ip, PS5_PORT))
    return s

def recv_exact(s, n):
    buf = b''
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("PS5 disconnected")
        buf += chunk
    return buf

def check_ok(s):
    return struct.unpack("<I", recv_exact(s, 4))[0] == STATUS_SUCCESS

def ps5_proc_list(ip):
    s = ps5_connect(ip)
    try:
        s.send(cmd_header(CMD_PROC_LIST, 0))
        if not check_ok(s):
            raise RuntimeError("proc list failed")
        count = struct.unpack("<I", recv_exact(s, 4))[0]
        procs = []
        # proc_list_entry (36 bytes): char name[32]; int32_t pid;
        # name comes first, then pid — order matters
        for _ in range(count):
            name = recv_exact(s, 32).rstrip(b'\x00').decode('utf-8', errors='replace')
            pid  = struct.unpack("<i", recv_exact(s, 4))[0]   # int32_t (signed)
            procs.append({"pid": pid, "name": name})
        return procs
    finally:
        s.close()

def ps5_maps(ip, pid):
    s = ps5_connect(ip)
    try:
        body = struct.pack("<I", pid)
        s.send(cmd_header(CMD_PROC_MAPS, len(body)) + body)
        if not check_ok(s):
            raise RuntimeError("maps failed")
        count = struct.unpack("<I", recv_exact(s, 4))[0]
        maps = []
        # proc_vm_map_entry (58 bytes packed):
        #   char name[32]; uint64_t start; uint64_t end; uint64_t offset; uint16_t prot;
        for _ in range(count):
            raw   = recv_exact(s, 58)
            name  = raw[0:32].rstrip(b'\x00').decode('utf-8', errors='replace')
            start = struct.unpack_from("<Q", raw, 32)[0]
            end   = struct.unpack_from("<Q", raw, 40)[0]
            # offset at 48 — not needed for scanning but consumed to keep stream in sync
            prot  = struct.unpack_from("<H", raw, 56)[0]   # uint16_t, not uint32_t
            maps.append({"start": start, "end": end, "prot": prot, "name": name})
        return maps
    finally:
        s.close()

def ps5_read(ip, pid, addr, length):
    s = ps5_connect(ip)
    try:
        # cmd_proc_read_packet (16 bytes): u32 pid; u64 address; u32 length
        body = struct.pack("<IQI", pid, addr, length)
        s.send(cmd_header(CMD_PROC_READ, len(body)) + body)
        if not check_ok(s):
            raise RuntimeError("read failed")
        return recv_exact(s, length)
    finally:
        s.close()

def ps5_write(ip, pid, addr, data: bytes):
    s = ps5_connect(ip)
    try:
        # cmd_proc_write_packet (16 bytes): u32 pid; u64 address; u32 length
        # Two-status-word protocol (spec §8 pattern 5):
        #   1. Send header + fixed request struct (no payload yet)
        #   2. Receive CMD_SUCCESS ack
        #   3. Send raw data bytes (the data phase)
        #   4. Receive final CMD_SUCCESS
        body = struct.pack("<IQI", pid, addr, len(data))
        s.send(cmd_header(CMD_PROC_WRITE, len(body)) + body)
        if not check_ok(s):
            return False        # ack rejected
        s.send(data)            # data phase — sent after the ack
        return check_ok(s)      # final status
    finally:
        s.close()

def ps5_read_multi(ip, pid, addr_list, width):
    """
    Read `width` bytes at each address in addr_list using a single TCP connection.
    Returns dict {addr: bytes} — missing/failed addresses are omitted.
    Each read still uses a separate CMD_PROC_READ (ps5debug has no bulk-read cmd),
    but shares the socket to avoid repeated TCP handshakes.
    """
    results = {}
    body_fmt = "<IQI"
    body_size = struct.calcsize(body_fmt)
    s = ps5_connect(ip)
    try:
        for addr in addr_list:
            try:
                body = struct.pack(body_fmt, pid, addr, width)
                s.send(cmd_header(CMD_PROC_READ, body_size) + body)
                if not check_ok(s):
                    continue
                results[addr] = recv_exact(s, width)
            except Exception:
                # Socket may be broken — stop trying further addresses
                break
    finally:
        s.close()
    return results


class FreezeSocket:
    """
    Persistent socket for the freeze loop.
    Opens once, reuses for every write, closes on exit.
    Falls back to a fresh socket if the connection drops mid-freeze.
    """
    def __init__(self, ip):
        self.ip  = ip
        self._s  = None

    def _ensure(self):
        if self._s is None:
            self._s = ps5_connect(self.ip)

    def write(self, pid, addr, data):
        for attempt in range(2):
            try:
                self._ensure()
                body = struct.pack("<IQI", pid, addr, len(data))
                self._s.send(cmd_header(CMD_PROC_WRITE, len(body)) + body)
                if not check_ok(self._s):
                    return False
                self._s.send(data)
                return check_ok(self._s)
            except Exception:
                # Connection dropped — close and retry once with a fresh socket
                try:
                    self._s.close()
                except Exception:
                    pass
                self._s = None
                if attempt == 1:
                    return False
        return False

    def close(self):
        if self._s:
            try:
                self._s.close()
            except Exception:
                pass
            self._s = None


def scan_first(ip, pid, value, width=4, progress_cb=None, region_filter=None):
    """
    Scan all readable memory regions for `value`.
    progress_cb(done_bytes, total_bytes) is called periodically if provided.
    region_filter: None = all regions, or one of "heap","exec","anon","all"
    Uses struct.iter_unpack for aligned scans (faster than byte-slice loop).
    """
    fmt    = WIDTH_FMT[width]
    target = struct.pack(fmt, value)
    maps   = ps5_maps(ip, pid)
    found  = []
    CHUNK  = 0x10000

    def region_ok(r):
        prot = r['prot']
        if not ((prot & 0x1) or (prot & 0x4)):
            return False
        if r['end'] - r['start'] > 0x10000000:
            return False
        if region_filter == "heap":
            return "heap" in r['name'].lower() or r['name'] == ""
        if region_filter == "exec":
            return bool(prot & 0x4)
        if region_filter == "anon":
            return r['name'] == ""
        return True  # "all" or None

    scannable   = [r for r in maps if region_ok(r)]
    total_bytes = sum(r['end'] - r['start'] for r in scannable)
    done_bytes  = 0

    for r in scannable:
        size = r['end'] - r['start']
        off  = 0
        while off < size:
            csz = min(CHUNK, size - off)
            try:
                chunk = ps5_read(ip, pid, r['start'] + off, csz)
                base  = r['start'] + off
                # Use iter_unpack for aligned scan (fast path)
                if csz % width == 0:
                    for idx, (v,) in enumerate(struct.iter_unpack(fmt, chunk)):
                        if struct.pack(fmt, v) == target:
                            found.append(base + idx * width)
                else:
                    # Unaligned tail — fall back to byte-slice
                    for i in range(0, len(chunk) - width + 1):
                        if chunk[i:i+width] == target:
                            found.append(base + i)
            except Exception:
                pass
            off        += csz
            done_bytes += csz
            if progress_cb:
                progress_cb(done_bytes, total_bytes)
    return found

def scan_unknown_initial(ip, pid, width, region_filter=None, progress_cb=None):
    """
    Snapshot all readable addresses (no value filter).
    Returns (addr_list, {addr: raw_bytes}) for use with scan_changed().
    """
    maps  = ps5_maps(ip, pid)
    CHUNK = 0x10000

    def region_ok(r):
        prot = r['prot']
        if not ((prot & 0x1) or (prot & 0x4)):
            return False
        if r['end'] - r['start'] > 0x10000000:
            return False
        if region_filter == "heap":
            return "heap" in r['name'].lower() or r['name'] == ""
        if region_filter == "exec":
            return bool(prot & 0x4)
        if region_filter == "anon":
            return r['name'] == ""
        return True

    scannable   = [r for r in maps if region_ok(r)]
    total_bytes = sum(r['end'] - r['start'] for r in scannable)
    done_bytes  = 0
    addr_list   = []
    cache       = {}
    fmt         = WIDTH_FMT.get(width, "<I")

    for r in scannable:
        size = r['end'] - r['start']
        off  = 0
        while off < size:
            csz = min(CHUNK, size - off)
            try:
                chunk = ps5_read(ip, pid, r['start'] + off, csz)
                base  = r['start'] + off
                if csz % width == 0:
                    for idx, _ in enumerate(struct.iter_unpack(fmt, chunk)):
                        a = base + idx * width
                        addr_list.append(a)
                        cache[a] = chunk[idx*width:(idx+1)*width]
            except Exception:
                pass
            off        += csz
            done_bytes += csz
            if progress_cb:
                progress_cb(done_bytes, total_bytes)
    return addr_list, cache


def scan_changed(ip, pid, mode, width, prev_cache, progress_cb=None):
    """
    Filter prev_cache (dict addr->bytes) by change mode:
      "changed"    — current value != previous
      "unchanged"  — current value == previous
      "increased"  — current > previous (unsigned)
      "decreased"  — current < previous (unsigned)
    Returns (surviving_addrs, new_cache_dict).
    """
    fmt      = WIDTH_FMT.get(width, "<I")
    addrs    = list(prev_cache.keys())
    survived = []
    new_cache = {}
    total = max(len(addrs), 1)

    s = ps5_connect(ip)
    try:
        for i, addr in enumerate(addrs):
            try:
                body = struct.pack("<IQI", pid, addr, width)
                s.send(cmd_header(CMD_PROC_READ, len(body)) + body)
                if not check_ok(s):
                    continue
                cur_bytes = recv_exact(s, width)
                prev_bytes = prev_cache[addr]
                cur_val  = struct.unpack(fmt, cur_bytes)[0]
                prev_val = struct.unpack(fmt, prev_bytes)[0]
                keep = (
                    (mode == "changed"   and cur_val != prev_val) or
                    (mode == "unchanged" and cur_val == prev_val) or
                    (mode == "increased" and cur_val >  prev_val) or
                    (mode == "decreased" and cur_val <  prev_val)
                )
                if keep:
                    survived.append(addr)
                    new_cache[addr] = cur_bytes
            except Exception:
                pass
            if progress_cb:
                progress_cb(i + 1, total)
    finally:
        s.close()
    return survived, new_cache


def scan_next(ip, pid, value, width, prev):
    fmt    = WIDTH_FMT[width]
    target = struct.pack(fmt, value)
    found  = []
    for addr in prev:
        try:
            if ps5_read(ip, pid, addr, width) == target:
                found.append(addr)
        except Exception:
            pass
    return found

def validate_write_addr(ip, pid, addr, length=1):
    """
    Check that addr..addr+length-1 falls within a writable (prot & 0x2) or
    readable (prot & 0x1) region. Returns (ok, reason_string).
    """
    try:
        maps = ps5_maps(ip, pid)
    except Exception as e:
        return False, f"Could not fetch maps: {e}"
    for r in maps:
        if r['start'] <= addr < r['end']:
            if addr + length > r['end']:
                return False, f"Address spans region boundary at {hex(r['end'])}"
            if not (r['prot'] & 0x3):   # neither read nor write
                return False, f"Region {hex(r['start'])}-{hex(r['end'])} not writable (prot={hex(r['prot'])})"
            return True, ""
    return False, f"{hex(addr)} not in any mapped region"


# ── pointer scanning ────────────────────────────────────────────────────────

PTR_FMT   = "<Q"   # PS5 is 64-bit LE; pointers are uint64
PTR_WIDTH = 8

def _is_static_region(r):
    """True if this region is a named module image (stable across restarts)."""
    return bool(r['name']) and bool(r['prot'] & 0x4)   # named + executable

def _is_heap_region(r):
    return not r['name'] or "heap" in r['name'].lower()

def scan_pointers(ip, pid, target_addrs, max_offset=2048, progress_cb=None,
                  depth=1):
    """
    Find pointer chains that resolve to any address in target_addrs.

    depth=1  : finds  base_ptr + offset == target   (one hop)
    depth=2  : finds  *(base_ptr) + offset == target  with base_ptr in
               a second set found by scanning for the intermediate addresses
               (two hops — slow but thorough)

    Returns list of dicts:
        {
          "chain":  [ptr_addr, offset]           # depth 1
                 or [ptr_addr, mid_offset, offset]  # depth 2
          "target": target_addr,
          "static": True/False   (ptr_addr is in a named module region)
        }
    """
    maps = ps5_maps(ip, pid)

    # Scannable regions: anything readable
    scannable = [r for r in maps
                 if (r['prot'] & 0x1) and (r['end'] - r['start']) <= 0x10000000]
    total_bytes = sum(r['end'] - r['start'] for r in scannable) * depth
    done_bytes  = 0
    CHUNK = 0x10000

    target_set = set(target_addrs)

    # Build a set of all regions for fast "which region?" lookups
    region_list = sorted(maps, key=lambda r: r['start'])

    def region_of(addr):
        for r in region_list:
            if r['start'] <= addr < r['end']:
                return r
        return None

    # ── Pass 1: find addresses whose u64 value lands within max_offset of a target ──
    level1 = []   # list of {ptr_addr, target, offset, static}

    for r in scannable:
        size = r['end'] - r['start']
        off  = 0
        while off < size:
            csz = (min(CHUNK, size - off) // PTR_WIDTH) * PTR_WIDTH
            if csz == 0:
                off += CHUNK
                done_bytes += CHUNK
                continue
            try:
                chunk = ps5_read(ip, pid, r['start'] + off, csz)
                base  = r['start'] + off
                for idx, (ptr_val,) in enumerate(struct.iter_unpack(PTR_FMT, chunk)):
                    for tgt in target_set:
                        delta = tgt - ptr_val
                        if 0 <= delta <= max_offset:
                            ptr_addr = base + idx * PTR_WIDTH
                            src_region = region_of(ptr_addr)
                            is_static  = _is_static_region(src_region) if src_region else False
                            level1.append({
                                "ptr_addr": ptr_addr,
                                "target":   tgt,
                                "offset":   delta,
                                "static":   is_static,
                            })
            except Exception:
                pass
            off        += csz
            done_bytes += csz
            if progress_cb:
                progress_cb(done_bytes, total_bytes)

    if depth == 1 or not level1:
        return [{"chain": [e["ptr_addr"], e["offset"]],
                 "target": e["target"], "static": e["static"]}
                for e in level1]

    # ── Pass 2: scan for pointers to each level1 ptr_addr ────────────────────
    mid_addrs = {e["ptr_addr"] for e in level1}
    level2    = []

    for r in scannable:
        size = r['end'] - r['start']
        off  = 0
        while off < size:
            csz = (min(CHUNK, size - off) // PTR_WIDTH) * PTR_WIDTH
            if csz == 0:
                off += CHUNK
                done_bytes += CHUNK
                continue
            try:
                chunk = ps5_read(ip, pid, r['start'] + off, csz)
                base  = r['start'] + off
                for idx, (ptr_val,) in enumerate(struct.iter_unpack(PTR_FMT, chunk)):
                    for mid in mid_addrs:
                        delta = mid - ptr_val
                        if 0 <= delta <= max_offset:
                            base_addr  = base + idx * PTR_WIDTH
                            src_region = region_of(base_addr)
                            is_static  = _is_static_region(src_region) if src_region else False
                            # Find the level1 entries that come from this mid
                            for e in level1:
                                if e["ptr_addr"] == mid:
                                    level2.append({
                                        "chain":  [base_addr, delta, e["offset"]],
                                        "target": e["target"],
                                        "static": is_static,
                                    })
            except Exception:
                pass
            off        += csz
            done_bytes += csz
            if progress_cb:
                progress_cb(done_bytes, total_bytes)

    return level2 if level2 else [{"chain": [e["ptr_addr"], e["offset"]],
                                    "target": e["target"], "static": e["static"]}
                                   for e in level1]


def resolve_pointer_chain(ip, pid, chain):
    """
    Walk a pointer chain and return the resolved final address.
    chain = [base_addr, offset]          → read u64 at base_addr, add offset
    chain = [base_addr, mid_off, offset] → read u64 at base_addr+0, add mid_off,
                                           read u64 at that address, add offset
    Returns resolved_addr or None on failure.
    """
    try:
        if len(chain) == 2:
            base_addr, final_off = chain
            ptr_bytes = ps5_read(ip, pid, base_addr, PTR_WIDTH)
            ptr_val   = struct.unpack(PTR_FMT, ptr_bytes)[0]
            return ptr_val + final_off
        else:
            base_addr, mid_off, final_off = chain
            ptr_bytes  = ps5_read(ip, pid, base_addr, PTR_WIDTH)
            ptr_val    = struct.unpack(PTR_FMT, ptr_bytes)[0]
            mid_addr   = ptr_val + mid_off
            mid_bytes  = ps5_read(ip, pid, mid_addr, PTR_WIDTH)
            mid_val    = struct.unpack(PTR_FMT, mid_bytes)[0]
            return mid_val + final_off
    except Exception:
        return None


def generate_cht(cheats, game_id, game_ver, game_title):
    """
    Generate a GoldHEN-compatible cheat file (JSON).
    Schema: { "title": ..., "titleid": ..., "version": ..., "cheatList": [ ... ] }
    Each entry: { "name": ..., "type": "write"|"freeze", "address": "0x...",
                  "value": "0x...", "bytes": N }
    """
    cheat_list = []
    for c in cheats:
        entry = {
            "name":  c["name"],
            "type":  c["type"],
            "bytes": c["width"],
        }
        if c["type"] == "pointer":
            # Pointer cheat: chain is [base_addr, offset] or [base, mid_off, offset]
            entry["chain"]  = [hex(a) for a in c["chain"]]
            entry["value"]  = hex(c["value"])
        else:
            entry["address"] = hex(c["address"])
            entry["value"]   = hex(c["value"])
        cheat_list.append(entry)
    payload = {
        "title":     game_title,
        "titleid":   game_id,
        "version":   game_ver,
        "cheatList": cheat_list,
    }
    return json.dumps(payload, indent=2)

# ── curses UI helpers ──────────────────────────────────────────────────────

def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,    -1)   # title / heading
    curses.init_pair(2, curses.COLOR_GREEN,   -1)   # success / highlight
    curses.init_pair(3, curses.COLOR_YELLOW,  -1)   # label / prompt
    curses.init_pair(4, curses.COLOR_RED,     -1)   # error
    curses.init_pair(5, curses.COLOR_WHITE,   -1)   # normal
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)   # accent
    curses.init_pair(7, curses.COLOR_BLACK,   curses.COLOR_CYAN)  # selected row
    curses.init_pair(8, curses.COLOR_BLACK,   curses.COLOR_RED)   # danger selected

# Color pair indices (used via color() after init_colors() is called)
C_TITLE = 1
C_OK    = 2
C_WARN  = 3
C_ERR   = 4
C_NORM  = 5
C_ACC   = 6
C_SEL   = 7
C_DSEL  = 8

def color(pair):
    """Return a curses color-pair attribute. Safe to call only after init_colors()."""
    return curses.color_pair(pair)

def safe_addstr(win, y, x, text, attr=0):
    """addstr that silently ignores out-of-bounds writes."""
    try:
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        win.addstr(y, x, text[:max(0, w - x)], attr)
    except curses.error:
        pass

def draw_border(win, title=""):
    win.box()
    if title:
        h, w = win.getmaxyx()
        label = f" {title} "
        safe_addstr(win, 0, max(2, (w - len(label)) // 2), label,
                    color(C_TITLE) | curses.A_BOLD)

def draw_statusbar(stdscr, segments):
    """
    Draw a status bar from [(text, color_pair), ...] segments joined by ' · '.
    """
    h, w = stdscr.getmaxyx()
    sep = "  ·  "
    x   = 0
    try:
        stdscr.addstr(h - 1, 0, " " * (w - 1), color(C_SEL))
    except curses.error:
        pass
    for i, (text, cp) in enumerate(segments):
        if x >= w - 1:
            break
        if i > 0:
            safe_addstr(stdscr, h - 1, x, sep, color(C_SEL))
            x += len(sep)
        chunk = text[:max(0, w - 1 - x)]
        safe_addstr(stdscr, h - 1, x, chunk, color(cp) | curses.A_BOLD)
        x += len(chunk)

def draw_progress_bar(win, y, x, width, fraction, label=""):
    """Draw a filled progress bar [████░░░░] followed by an optional label."""
    inner   = width - 2
    filled  = int(fraction * inner)
    bar     = "\u2588" * filled + "\u2591" * (inner - filled)
    safe_addstr(win, y, x, f"[{bar}]", color(C_OK))
    if label:
        safe_addstr(win, y, x + width + 1, label, color(C_WARN))

def add_log(msg, level="info"):
    """Append a timestamped log entry. level: 'info' | 'warn' | 'error'."""
    ts = time.strftime("%H:%M:%S")
    state["log"].append({"ts": ts, "msg": msg, "level": level})
    if len(state["log"]) > 200:
        state["log"] = state["log"][-200:]

def input_box(stdscr, prompt, y, x, width=30, default=""):
    """
    Inline input field. Returns the entered string (or default if empty).
    Gracefully skips rendering if y is out of terminal bounds.
    """
    h, _ = stdscr.getmaxyx()
    if y >= h - 1:
        return default
    safe_addstr(stdscr, y, x, prompt, color(C_WARN) | curses.A_BOLD)
    px = x + len(prompt)
    curses.echo()
    curses.curs_set(1)
    safe_addstr(stdscr, y, px, default)
    stdscr.refresh()
    try:
        val = stdscr.getstr(y, px, width).decode('utf-8').strip()
    except Exception:
        val = default
    curses.noecho()
    curses.curs_set(0)
    return val or default

def cycle_input(stdscr, prompt, y, x, options, default=None):
    """
    Inline selector: cycles through `options` with Tab / left-right arrow keys.
    Returns the chosen value.
    """
    h, _ = stdscr.getmaxyx()
    if y >= h - 1:
        return default if default is not None else options[0]
    idx = options.index(default) if default in options else 0
    curses.curs_set(0)
    while True:
        chosen = str(options[idx])
        safe_addstr(stdscr, y, x, prompt, color(C_WARN) | curses.A_BOLD)
        px = x + len(prompt)
        hint = f"< {chosen} >  (Tab/arrows to change, Enter to confirm)"
        safe_addstr(stdscr, y, px, hint, color(C_TITLE) | curses.A_BOLD)
        stdscr.refresh()
        k = stdscr.getch()
        if k in (ord('\t'), curses.KEY_RIGHT):
            idx = (idx + 1) % len(options)
        elif k == curses.KEY_LEFT:
            idx = (idx - 1) % len(options)
        elif k in (curses.KEY_ENTER, 10, 13):
            return options[idx]

def confirm_box(stdscr, question, title="Confirm"):
    """Yes/No dialog. Returns True if the user confirms."""
    h, w = stdscr.getmaxyx()
    lines = [question, "", "  [Y] Yes      [N / Esc] No"]
    bh = len(lines) + 4
    bw = max(len(l) for l in lines) + 6
    bw = min(bw, w - 4)
    win = curses.newwin(bh, bw, max(0, (h - bh) // 2), max(0, (w - bw) // 2))
    draw_border(win, title)
    for i, line in enumerate(lines):
        safe_addstr(win, i + 2, 3, line[:bw - 6], color(C_WARN))
    win.refresh()
    while True:
        k = win.getch()
        if k in (ord('y'), ord('Y'), curses.KEY_ENTER, 10, 13):
            return True
        if k in (ord('n'), ord('N'), 27):
            return False

def message_box(stdscr, lines, title="Info", color_pair=C_NORM):
    """Overlay a centered message box."""
    h, w = stdscr.getmaxyx()
    bh = len(lines) + 4
    bw = max((len(l) for l in lines), default=10) + 6
    bw = min(bw, w - 4)
    win = curses.newwin(bh, bw,
                        max(0, (h - bh) // 2),
                        max(0, (w - bw) // 2))
    draw_border(win, title)
    for i, line in enumerate(lines):
        safe_addstr(win, i + 2, 3, line[:bw - 6], color(color_pair))
    safe_addstr(win, bh - 2, max(1, (bw - 14) // 2), " Press any key ", color(C_WARN))
    win.refresh()
    win.getch()

# ── screens ────────────────────────────────────────────────────────────────

def draw_header_banner(stdscr):
    h, w = stdscr.getmaxyx()
    brand = "◈  PS5 CHEAT MAKER  ◈"
    safe_addstr(stdscr, 1, max(0, (w - len(brand)) // 2),
                brand, color(C_TITLE) | curses.A_BOLD)

def screen_connect(stdscr):
    stdscr.clear()
    draw_border(stdscr, "CONNECT")
    draw_header_banner(stdscr)
    hints = [
        "Ensure ps5debug payload is loaded on your PS5.",
        "Find PS5 IP:  Settings > Network > View Connection Status",
    ]
    for i, hint in enumerate(hints):
        safe_addstr(stdscr, 3 + i, 3, hint, color(C_NORM))
    stdscr.refresh()

    ip = input_box(stdscr, "PS5 IP address : ", 6, 3, 20, state["ip"] or "192.168.0.88")
    state["ip"] = ip

    safe_addstr(stdscr, 8, 3, "Connecting…", color(C_WARN))
    stdscr.refresh()
    try:
        procs = ps5_proc_list(ip)
        state["connected"] = True
        add_log(f"Connected to {ip}, {len(procs)} processes")
        return screen_proc_select(stdscr, procs)
    except Exception as e:
        safe_addstr(stdscr, 8, 3, f"X Failed: {e}".ljust(60), color(C_ERR))
        safe_addstr(stdscr, 10, 3, "Press any key to retry.", color(C_NORM))
        stdscr.refresh()
        stdscr.getch()
        return "connect"

def screen_proc_select(stdscr, procs):
    sel        = 0
    filter_str = ""
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "SELECT PROCESS")
        safe_addstr(stdscr, 2, 3,
            f"Connected: {state['ip']}   Processes: {len(procs)}",
            color(C_OK) | curses.A_BOLD)

        visible_procs = [p for p in procs
                         if filter_str.lower() in p['name'].lower()]
        filter_hint = filter_str if filter_str else "(none — type to filter)"
        safe_addstr(stdscr, 3, 3, f"Filter: {filter_hint}", color(C_WARN))

        visible = h - 9
        start   = max(0, sel - visible // 2)
        for i, p in enumerate(visible_procs[start:start + visible]):
            idx  = start + i
            dim  = p['pid'] < 10
            attr = (color(C_SEL) if idx == sel
                    else (color(C_NORM) | curses.A_DIM if dim
                          else color(C_NORM)))
            line = f"  PID {p['pid']:6d}   {p['name']}"
            safe_addstr(stdscr, 5 + i, 2, line[:w - 4].ljust(w - 4), attr)

        draw_statusbar(stdscr, [
            ("arrows navigate", C_NORM),
            ("Enter attach", C_OK),
            ("type to filter", C_WARN),
            ("Bksp clear", C_NORM),
            ("Q back", C_NORM),
        ])
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP and sel > 0:
            sel -= 1
        elif key == curses.KEY_DOWN and sel < len(visible_procs) - 1:
            sel += 1
        elif key in (curses.KEY_ENTER, 10, 13) and visible_procs:
            p = visible_procs[sel]
            state["pid"]       = p["pid"]
            state["proc_name"] = p["name"]
            add_log(f"Attached to PID {state['pid']} ({state['proc_name']})")
            return "main"
        elif key in (ord('q'), ord('Q')):
            return "connect"
        elif key in (curses.KEY_BACKSPACE, 127):
            filter_str = filter_str[:-1]
            sel = 0
        elif 32 <= key <= 126:
            filter_str += chr(key)
            sel = 0

def _draw_main_header(stdscr):
    h, w = stdscr.getmaxyx()
    conn = f" * {state['ip']}  PID {state['pid']} ({state['proc_name']}) "
    safe_addstr(stdscr, 2, 3, conn, color(C_OK) | curses.A_BOLD)
    width_label = {1: "byte", 2: "uint16", 4: "uint32", 8: "uint64"}.get(
        state["scan_width"], "?")
    stats = (f"  Scan results: {len(state['scan_results'])}   "
             f"Cheats: {len(state['cheats'])}   "
             f"Type: {width_label}")
    safe_addstr(stdscr, 3, 3, stats, color(C_WARN))

def screen_main(stdscr):
    menu = [
        ("S", "First Scan",       "scan_first",  C_NORM),
        ("N", "Next Scan",        "scan_next",   C_NORM),
        ("R", "Results",          "results",     C_NORM),
        ("W", "Write to Address", "write",       C_WARN),
        ("C", "Cheat List",       "cheat_list",  C_NORM),
        ("E", "Export .json",     "export",      C_OK),
        ("F", "Freeze Address",   "freeze",      C_WARN),
        ("L", "Log",              "log",         C_NORM),
        ("T", "Pointer Scan",     "pointer",     C_ACC),
        ("X", "Clear Results",    "clear",       C_WARN),
        ("P", "Change Process",   "proc",        C_NORM),
        ("Q", "Quit",             None,          C_ERR),
    ]
    sel = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "PS5 CHEAT MAKER")
        draw_header_banner(stdscr)
        _draw_main_header(stdscr)

        col2  = w > 60
        split = (len(menu) + 1) // 2 if col2 else len(menu)
        for i, (key, label, _, cp) in enumerate(menu):
            col = i // split if col2 else 0
            row = i % split
            cx  = 5 + col * 35
            cy  = 5 + row
            unavail = (
                (label == "Next Scan"    and not state["scan_results"]) or
                (label == "Results"      and not state["scan_results"]) or
                (label == "Pointer Scan" and not state["scan_results"]) or
                (label == "Export .json" and not state["cheats"])
            )
            if i == sel:
                attr = color(C_SEL) | curses.A_BOLD
            elif unavail:
                attr = color(C_NORM) | curses.A_DIM
            else:
                attr = color(cp)
            safe_addstr(stdscr, cy, cx, f"[{key}]  {label}".ljust(30), attr)

        if state["log"]:
            entry = state["log"][-1]
            lcp   = {"error": C_ERR, "warn": C_WARN, "info": C_OK}.get(
                entry["level"], C_NORM)
            safe_addstr(stdscr, h - 3, 3,
                        f"[{entry['ts']}] {entry['msg']}"[:w - 6],
                        color(lcp))

        draw_statusbar(stdscr, [
            ("arrows / letter", C_NORM),
            ("Enter select", C_OK),
            ("Q quit", C_ERR),
        ])
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP   and sel > 0:            sel -= 1
        elif key == curses.KEY_DOWN and sel < len(menu) - 1: sel += 1
        elif key in (curses.KEY_ENTER, 10, 13):
            action = menu[sel][2]
            if action is None:
                return None
            result = dispatch(stdscr, action)
            if result == "proc": return "proc"
        else:
            for i, (k, _, action, _) in enumerate(menu):
                if key in (ord(k.lower()), ord(k.upper())):
                    if action is None: return None
                    result = dispatch(stdscr, action)
                    if result == "proc": return "proc"
                    break

def dispatch(stdscr, action):
    if action == "scan_first":  return do_scan_first(stdscr)
    if action == "scan_next":   return do_scan_next(stdscr)
    if action == "results":     return do_show_results(stdscr)
    if action == "write":       return do_write(stdscr)
    if action == "cheat_list":  return do_cheat_list(stdscr)
    if action == "export":      return do_export(stdscr)
    if action == "freeze":      return do_freeze(stdscr)
    if action == "log":         return do_log(stdscr)
    if action == "pointer":     return do_pointer_scan(stdscr)
    if action == "clear":       return do_clear_results(stdscr)
    if action == "proc":        return "proc"

# ── scan screens ───────────────────────────────────────────────────────────

def do_scan_first(stdscr):
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    draw_border(stdscr, "FIRST SCAN")
    safe_addstr(stdscr, 2, 3, "Enter the current in-game value to search for.", color(C_WARN))
    stdscr.refresh()

    scan_type = cycle_input(stdscr, "Scan type       : ", 4, 3,
                            SCAN_TYPE_NAMES, state["scan_type"])
    state["scan_type"] = scan_type
    fmt, width, is_float = SCAN_TYPES[scan_type]
    state["scan_width"] = width

    # Unknown initial value — skip value entry, scan all addresses
    CHANGE_MODES = ["changed", "unchanged", "increased", "decreased"]
    val_s = input_box(stdscr, "Value (blank=unknown): ", 6, 3, 20)

    region_filter = cycle_input(stdscr, "Region filter   : ", 8, 3,
                                ["all", "heap", "exec", "anon"],
                                state["region_filter"])
    state["region_filter"] = region_filter

    # Parse value — empty = unknown initial value scan
    is_unknown = (val_s.strip() == "")
    val = None
    if not is_unknown:
        try:
            val = float(val_s) if is_float else int(val_s)
            struct.pack(fmt, val)   # validate fits
        except (ValueError, struct.error):
            message_box(stdscr,
                [f"Invalid value for {scan_type}."], "Error", C_ERR)
            return

    # Live-progress scan in a background thread
    progress = {"done": 0, "total": 1, "results": None, "error": None,
                "prev_cache": {}}

    def run_scan():
        try:
            def cb(done, total):
                progress["done"]  = done
                progress["total"] = max(total, 1)
            if is_unknown:
                # Snapshot all addresses — no value filtering, just collect
                addrs, cache = scan_unknown_initial(
                    state["ip"], state["pid"], width,
                    region_filter, cb)
                progress["results"]    = addrs
                progress["prev_cache"] = cache
            else:
                progress["results"] = scan_first(
                    state["ip"], state["pid"], val, width, cb,
                    region_filter)
        except Exception as e:
            progress["error"] = str(e)

    t = threading.Thread(target=run_scan, daemon=True)
    t.start()

    spinner = ["|", "/", "-", "\\"]
    spin_i  = 0
    while t.is_alive():
        frac = progress["done"] / progress["total"]
        safe_addstr(stdscr, 9, 3,
            f"{spinner[spin_i % len(spinner)]}  Scanning memory...  "
            f"{progress['done'] // 1024:,} KB / {progress['total'] // 1024:,} KB",
            color(C_WARN))
        draw_progress_bar(stdscr, 10, 3, min(w - 8, 60), frac,
                          f"  {int(frac * 100)}%")
        stdscr.refresh()
        time.sleep(0.1)
        spin_i += 1

    if progress["error"]:
        add_log(f"Scan error: {progress['error']}", "error")
        message_box(stdscr, [f"Error: {progress['error']}"], "Scan Failed", C_ERR)
        return

    results = progress["results"]
    state["scan_history"] = []   # fresh scan resets history
    state["scan_results"] = results
    state["prev_cache"]   = progress.get("prev_cache", {})
    lbl = "unknown initial" if is_unknown else str(val)
    add_log(f"First scan val={lbl} type={scan_type}: {len(results)} results")
    tip = ("Change the value in-game, then Next Scan (N)." if not is_unknown
           else "Value snapshot taken. Use Next Scan (N) with changed/increased/etc.")
    message_box(stdscr, [
        f"Found {len(results)} results.",
        "",
        tip,
        "Once narrowed down, use Results (R) to pick an address.",
    ], "Scan Complete", C_OK)

def do_scan_next(stdscr):
    if not state["scan_results"]:
        message_box(stdscr,
            ["No previous scan results.", "Run First Scan (S) first."], "Error", C_ERR)
        return
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    draw_border(stdscr, "NEXT SCAN")
    safe_addstr(stdscr, 2, 3,
        f"Candidates: {len(state['scan_results'])}  — enter the new in-game value.",
        color(C_WARN))
    stdscr.refresh()

    has_cache = bool(state.get("prev_cache"))
    fmt, width, is_float = SCAN_TYPES[state["scan_type"]]
    CHANGE_MODES = ["exact", "changed", "unchanged", "increased", "decreased"]
    if has_cache:
        mode = cycle_input(stdscr, "Filter mode     : ", 4, 3,
                           CHANGE_MODES, "exact")
    else:
        mode = "exact"
    val_s = ""
    if mode == "exact":
        val_s = input_box(stdscr, "New value       : ", 6, 3, 20)
    val = None
    if mode == "exact":
        try:
            val = float(val_s) if is_float else int(val_s)
        except ValueError:
            message_box(stdscr, ["Invalid value."], "Error", C_ERR)
            return

    # Run next scan in background thread with progress bar
    nxt_progress = {"done": 0, "total": max(len(state["scan_results"]), 1),
                    "results": None, "error": None}
    def _run_next():
        nonlocal mode, val
        if mode != "exact" and state.get("prev_cache"):
            def _cb(done, total):
                nxt_progress["done"]  = done
                nxt_progress["total"] = max(total, 1)
            survived, new_cache = scan_changed(
                state["ip"], state["pid"], mode,
                state["scan_width"], state["prev_cache"], _cb)
            nxt_progress["results"]    = survived
            nxt_progress["new_cache"]  = new_cache
            return
        # exact value scan
        found = []
        _fmt, _width, _is_float = SCAN_TYPES[state["scan_type"]]
        try:
            target = struct.pack(_fmt, val)
        except (struct.error, OverflowError) as e:
            nxt_progress["error"] = str(e)
            nxt_progress["results"] = []
            return
        addrs = list(state["scan_results"])
        for j, addr in enumerate(addrs):
            try:
                if ps5_read(state["ip"], state["pid"], addr, _width) == target:
                    found.append(addr)
            except Exception:
                pass
            nxt_progress["done"] = j + 1
        nxt_progress["results"]   = found
        nxt_progress["new_cache"] = {}
    nt = threading.Thread(target=_run_next, daemon=True)
    nt.start()
    spinner = ["|", "/", "-", "\\"]
    spin_i2 = 0
    while nt.is_alive():
        frac2 = nxt_progress["done"] / nxt_progress["total"]
        safe_addstr(stdscr, 7, 3,
            f"{spinner[spin_i2 % 4]}  Filtering {nxt_progress['done']:,}"
            f" / {nxt_progress['total']:,} addresses...",
            color(C_WARN))
        draw_progress_bar(stdscr, 8, 3, min(w - 8, 60), frac2,
                          f"  {int(frac2*100)}%")
        stdscr.refresh()
        time.sleep(0.1)
        spin_i2 += 1
    try:
        if nxt_progress.get("error"):
            raise ValueError(nxt_progress["error"])
        results = nxt_progress["results"]
        if results is None: results = []
        state["scan_history"].append(list(state["scan_results"]))  # save for undo
        if len(state["scan_history"]) > 10:
            state["scan_history"] = state["scan_history"][-10:]
        state["scan_results"] = results
        if nxt_progress.get("new_cache") is not None:
            state["prev_cache"] = nxt_progress["new_cache"]
        add_log(f"Next scan val={val}: {len(results)} remain")
        tip = ("Perfect! Use Results (R) to pick an address."
               if len(results) <= 10
               else "Still many — change value in-game and scan again (N).")
        undo_hint = f"  (U to undo — restores {len(state['scan_history'][-1])} candidates)" if state["scan_history"] else ""
        message_box(stdscr,
            [f"{len(results)} results remain.", "", tip, undo_hint],
            "Scan Complete", C_OK if len(results) <= 10 else C_WARN)
    except Exception as e:
        message_box(stdscr, [f"Error: {e}"], "Error", C_ERR)

_cache_lock = threading.Lock()

def _refresh_values(ip, pid, results, width, cache):
    """Background thread: batch-read current values using a single socket."""
    fmt      = WIDTH_FMT.get(width, "<I")
    batch    = ps5_read_multi(ip, pid, results, width)
    updates  = {}
    for addr in results:
        raw = batch.get(addr)
        try:
            val_str = str(struct.unpack(fmt, raw)[0]) if raw else "?"
        except Exception:
            val_str = "?"
        updates[addr] = val_str
    with _cache_lock:
        cache.update(updates)

def do_show_results(stdscr):
    results = state["scan_results"]
    if not results:
        message_box(stdscr,
            ["No scan results yet.", "Run First Scan (S) first."], "Results", C_WARN)
        return
    sel        = 0
    offset     = 0
    val_cache      = {}          # addr -> current value string
    last_refresh   = 0.0
    REFRESH_INTERVAL = 2.0  # re-read live values every 2s
    refresh_thread = None   # guard: only one refresh in flight at a time

    stdscr.nodelay(True)
    try:
        while True:
            now = time.time()
            # Trigger background refresh only when previous thread finished
            thread_idle = refresh_thread is None or not refresh_thread.is_alive()
            if thread_idle and (now - last_refresh >= REFRESH_INTERVAL or not val_cache):
                refresh_thread = threading.Thread(
                    target=_refresh_values,
                    args=(state["ip"], state["pid"], list(results),
                          state["scan_width"], val_cache),
                    daemon=True)
                refresh_thread.start()
                last_refresh = now

            stdscr.clear()
            h, w = stdscr.getmaxyx()
            draw_border(stdscr, f"RESULTS  ({len(results)} addresses)")
            wlabel = WIDTH_LABEL.get(state["scan_width"], str(state["scan_width"]))
            safe_addstr(stdscr, 2, 3,
                f"Type: {wlabel}   Process: {state['proc_name']} (PID {state['pid']})",
                color(C_WARN))
            safe_addstr(stdscr, 3, 3,
                "arrows navigate   Enter: add cheat   D: drop   U: undo scan   Q: back",
                color(C_NORM))

            visible = h - 7
            if sel < offset: offset = sel
            if sel >= offset + visible: offset = sel - visible + 1
            for i, addr in enumerate(results[offset:offset + visible]):
                idx   = offset + i
                with _cache_lock:
                    vstr = val_cache.get(addr, "...")
                marker = ">" if idx == sel else " "
                line   = f"{marker} {idx+1:4d}   {hex(addr):<20}  current = {vstr}"
                attr   = color(C_SEL) | curses.A_BOLD if idx == sel else color(C_NORM)
                safe_addstr(stdscr, 5 + i, 2, line[:w - 4].ljust(w - 4), attr)

            age = int(now - last_refresh)
            draw_statusbar(stdscr, [
                (f"{len(results)} results", C_WARN),
                ("arrows navigate", C_NORM),
                ("Enter add cheat", C_OK),
                ("D drop", C_ERR),
                ("U undo scan", C_WARN),
                (f"values {age}s ago", C_NORM),
                ("Q back", C_NORM),
            ])
            stdscr.refresh()
            time.sleep(0.05)

            key = stdscr.getch()
            if key == curses.KEY_UP and sel > 0:
                sel -= 1
            elif key == curses.KEY_DOWN and sel < len(results) - 1:
                sel += 1
            elif key in (curses.KEY_ENTER, 10, 13):
                stdscr.nodelay(False)
                _add_cheat_at(stdscr, results[sel])
                stdscr.nodelay(True)
                results = state["scan_results"]
                with _cache_lock:
                    val_cache.clear()
            elif key in (ord('d'), ord('D')):
                dropped = results[sel]          # capture before deletion
                del results[sel]
                state["scan_results"] = results
                with _cache_lock:
                    val_cache.pop(dropped, None)    # evict correct key
                sel = min(sel, len(results) - 1)
                if not results:
                    break
            elif key in (ord('u'), ord('U')):
                if state["scan_history"]:
                    state["scan_results"] = state["scan_history"].pop()
                    results = state["scan_results"]
                    with _cache_lock:
                        val_cache.clear()
                    sel = 0; offset = 0
                    add_log(f"Undo: restored {len(results)} candidates")
                else:
                    pass  # nothing to undo — ignore silently
            elif key in (ord('q'), ord('Q')):
                break
    finally:
        stdscr.nodelay(False)

def _add_cheat_at(stdscr, addr):
    """Prompt for cheat metadata and append to cheat list."""
    stdscr.clear()
    draw_border(stdscr, "ADD CHEAT")
    safe_addstr(stdscr, 2, 3, f"Address : {hex(addr)}", color(C_OK) | curses.A_BOLD)

    # Show current live value for reference
    try:
        raw = ps5_read(state["ip"], state["pid"], addr, state["scan_width"])
        cur = struct.unpack(WIDTH_FMT[state["scan_width"]], raw)[0]
        safe_addstr(stdscr, 3, 3, f"Current : {cur}", color(C_WARN))
    except Exception:
        pass

    stdscr.refresh()
    name  = input_box(stdscr, "Cheat name       : ", 5, 3, 40)
    val_s = input_box(stdscr, "Lock-in value    : ", 7, 3, 20)
    typ   = cycle_input(stdscr, "Cheat type       : ", 9, 3,
                        ["freeze", "write"], "freeze")
    try:
        val = int(val_s)
        entry = {
            "name":    name or f"Cheat@{hex(addr)}",
            "address": addr,
            "value":   val,
            "type":    typ,
            "width":   state["scan_width"],
        }
        state["cheats"].append(entry)
        add_log(f"Added '{entry['name']}' @ {hex(addr)} = {val}")
        message_box(stdscr,
            [f"  {entry['name']}", f"  {hex(addr)} = {val}  [{typ}]"],
            "Cheat Added", C_OK)
    except ValueError:
        message_box(stdscr,
            ["Invalid value — must be a whole number."], "Error", C_ERR)

def do_write(stdscr):
    stdscr.clear()
    draw_border(stdscr, "WRITE TO ADDRESS")
    safe_addstr(stdscr, 2, 3,
        "Write a single value directly to a memory address.", color(C_WARN))
    stdscr.refresh()
    addr_s = input_box(stdscr, "Address (hex)    : ", 4, 3, 20)
    val_s  = input_box(stdscr, "Value            : ", 6, 3, 20)
    _wd = [WIDTH_LABEL[ww] for ww in VALID_WIDTHS]
    _ws = cycle_input(stdscr, "Width            : ", 8, 3, _wd,
                      WIDTH_LABEL.get(state["scan_width"], "uint32"))
    width = VALID_WIDTHS[_wd.index(_ws)]
    try:
        addr = int(addr_s, 16)
        val  = int(val_s)
        fmt_key = {1:"B",2:"<H",4:"<I",8:"<Q"}[width]
        data = struct.pack(fmt_key, val)
        # Validate address before writing
        ok_addr, reason = validate_write_addr(state["ip"], state["pid"], addr, width)
        if not ok_addr:
            message_box(stdscr,
                [f"Address validation failed:", reason,
                 "", "Write aborted."], "Invalid Address", C_ERR)
            return
        ok = ps5_write(state["ip"], state["pid"], addr, data)
        lvl = "info" if ok else "error"
        add_log(f"Write {hex(addr)} = {val} {'OK' if ok else 'FAILED'}", lvl)
        if ok:
            message_box(stdscr,
                [f"Wrote {val} to {hex(addr)}"], "Write OK", C_OK)
        else:
            message_box(stdscr,
                ["Write command was rejected by ps5debug."], "Write Failed", C_ERR)
    except Exception as e:
        message_box(stdscr, [f"Error: {e}"], "Error", C_ERR)

def do_cheat_list(stdscr):
    cheats = state["cheats"]
    sel    = 0
    offset = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, f"CHEAT LIST  ({len(cheats)} cheats)")
        visible = max(1, h - 8)
        if not cheats:
            safe_addstr(stdscr, 4, 5,
                "No cheats yet — scan and add some!", color(C_WARN))
        else:
            safe_addstr(stdscr, 2, 3,
                "arrows select   Enter: edit   D: delete   Q: back",
                color(C_NORM))
            hdr = f"  {'Name':<28}  {'Address':<18}  {'Value':<10}  Type"
            safe_addstr(stdscr, 3, 2, hdr[:w - 4],
                        color(C_TITLE) | curses.A_UNDERLINE)
            # scroll offset tracking
            if sel < offset: offset = sel
            if sel >= offset + visible: offset = sel - visible + 1
            for i, c in enumerate(cheats[offset:offset + visible]):
                ri   = offset + i
                attr = color(C_SEL) | curses.A_BOLD if ri == sel else color(C_NORM)
                line = (f"  {c['name']:<28}  {hex(c['address']):<18}  "
                        f"{str(c['value']):<10}  [{c['type']}]")
                safe_addstr(stdscr, 5 + i, 2, line[:w - 4].ljust(w - 4), attr)
            if len(cheats) > visible:
                safe_addstr(stdscr, h - 3, w - 20,
                    f" {offset+1}-{min(offset+visible,len(cheats))}/{len(cheats)} ",
                    color(C_WARN))

        draw_statusbar(stdscr, [
            ("arrows navigate", C_NORM),
            ("Enter edit", C_OK),
            ("D delete", C_ERR),
            ("Q back", C_NORM),
        ])
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP   and sel > 0:              sel -= 1
        elif key == curses.KEY_DOWN and sel < len(cheats) - 1: sel += 1
        elif key in (curses.KEY_ENTER, 10, 13) and cheats:
            _edit_cheat(stdscr, sel)
            cheats = state["cheats"]
        elif key in (ord('d'), ord('D')) and cheats:
            name = cheats[sel]["name"]
            if confirm_box(stdscr, f"Delete '{name}'?", "Delete Cheat"):
                del cheats[sel]
                state["cheats"] = cheats
                add_log(f"Deleted cheat '{name}'", "warn")
                sel = max(0, min(sel, len(cheats) - 1))
                offset = min(offset, max(0, len(cheats) - visible))
        elif key in (ord('q'), ord('Q')):
            break

def _edit_cheat(stdscr, idx):
    """In-place editor for a single cheat entry."""
    c = state["cheats"][idx]
    stdscr.clear()
    draw_border(stdscr, "EDIT CHEAT")
    safe_addstr(stdscr, 2, 3, f"Editing: {c['name']}", color(C_TITLE) | curses.A_BOLD)
    safe_addstr(stdscr, 3, 3, "Leave blank to keep current value.", color(C_NORM))
    stdscr.refresh()

    new_name = input_box(stdscr, "Name    : ", 5, 3, 40, c["name"])
    val_s    = input_box(stdscr, "Value   : ", 7, 3, 20, str(c["value"]))
    new_type = cycle_input(stdscr, "Type    : ", 9, 3,
                           ["freeze", "write"], c["type"])

    try:
        new_val = int(val_s)
    except ValueError:
        new_val = c["value"]

    state["cheats"][idx].update({
        "name":  new_name,
        "value": new_val,
        "type":  new_type,
    })
    add_log(f"Edited cheat '{new_name}' -> val={new_val} type={new_type}")
    message_box(stdscr, [f"Updated '{new_name}'"], "Saved", C_OK)

def do_export(stdscr):
    stdscr.clear()
    draw_border(stdscr, "EXPORT GOLDHEN CHEAT JSON")
    safe_addstr(stdscr, 2, 3,
        f"Cheats to export: {len(state['cheats'])}", color(C_WARN))
    if not state["cheats"]:
        message_box(stdscr,
            ["No cheats to export.", "Build your cheat list first."], "Error", C_ERR)
        return
    stdscr.refresh()
    gid  = input_box(stdscr, "Title ID  (e.g. PPSA01234) : ", 4, 3, 20, state["game_id"])
    gver = input_box(stdscr, "Version   (e.g. 01.00)     : ", 6, 3, 10, state["game_ver"])
    gtit = input_box(stdscr, "Game Title                 : ", 8, 3, 40, state["game_title"])
    state.update(game_id=gid, game_ver=gver, game_title=gtit)

    fname = f"{gid}_{gver.replace('.','_')}.json"
    cht   = generate_cht(state["cheats"], gid, gver, gtit)
    try:
        with open(fname, 'w') as f:
            f.write(cht)
        add_log(f"Exported {fname}")
        message_box(stdscr, [
            f"Saved: {fname}",
            "",
            "Transfer to PS5 via FTP:",
            f"  /data/GoldHEN/cheats/{gid}/{fname}",
            "",
            "Activate: GoldHEN overlay > Options > Cheats",
        ], "Export OK", C_OK)
    except Exception as e:
        message_box(stdscr, [f"Could not write file: {e}"], "Export Failed", C_ERR)

def do_freeze(stdscr):
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    draw_border(stdscr, "FREEZE ADDRESS")
    safe_addstr(stdscr, 2, 3,
        "Continuously write a value to lock it in memory.", color(C_WARN))
    stdscr.refresh()
    addr_s = input_box(stdscr, "Address (hex)    : ", 4, 3, 20)
    val_s  = input_box(stdscr, "Freeze value     : ", 6, 3, 20)
    _wd = [WIDTH_LABEL[ww] for ww in VALID_WIDTHS]
    _ws = cycle_input(stdscr, "Width            : ", 8, 3, _wd,
                      WIDTH_LABEL.get(state["scan_width"], "uint32"))
    width = VALID_WIDTHS[_wd.index(_ws)]
    sec_s  = input_box(stdscr, "Duration (secs)  : ", 10, 3, 6, "30")
    try:
        addr = int(addr_s, 16)
        val  = int(val_s)
        secs = int(sec_s)
        data = struct.pack(WIDTH_FMT[width], val)
    except Exception as e:
        message_box(stdscr, [f"Bad input: {e}"], "Error", C_ERR)
        return

    safe_addstr(stdscr, 13, 3,
        f"Freezing {hex(addr)} = {val} for {secs}s",
        color(C_WARN) | curses.A_BOLD)
    safe_addstr(stdscr, 14, 3, "Press Q to stop early.", color(C_NORM))
    stdscr.refresh()

    write_errors = [0]
    deadline = time.time() + secs
    fsock = FreezeSocket(state["ip"])
    try:
        while time.time() < deadline:
            if not fsock.write(state["pid"], addr, data):
                write_errors[0] += 1
            elapsed   = time.time() - (deadline - secs)
            frac      = elapsed / secs
            remaining = int(deadline - time.time())
            safe_addstr(stdscr, 16, 3, f"Time left: {remaining:3d}s  ", color(C_OK))
            draw_progress_bar(stdscr, 17, 3, min(w - 8, 50), frac,
                              f"  {int(frac * 100)}%")
            if write_errors[0]:
                safe_addstr(stdscr, 18, 3,
                    f"Write errors: {write_errors[0]}  (connection issue?)",
                    color(C_ERR))
            stdscr.refresh()
            time.sleep(0.2)
            stdscr.nodelay(True)
            k = stdscr.getch()
            if k in (ord('q'), ord('Q')):
                break
    finally:
        stdscr.nodelay(False)
        fsock.close()
    add_log(f"Freeze done {hex(addr)} = {val}")
    message_box(stdscr, ["Freeze complete."], "Done", C_OK)

def do_pointer_scan(stdscr):
    """
    Pointer scan screen.
    Requires at least one address in scan_results as the target.
    """
    if not state["scan_results"]:
        message_box(stdscr,
            ["No scan results to use as target.",
             "First narrow down to the target address(es) with",
             "First Scan + Next Scan, then run Pointer Scan."],
            "No Target", C_WARN)
        return

    stdscr.clear()
    h, w = stdscr.getmaxyx()
    draw_border(stdscr, "POINTER SCAN")
    safe_addstr(stdscr, 2, 3,
        f"Target addresses: {len(state['scan_results'])}  "
        f"(using first 8 as targets)",
        color(C_WARN))
    safe_addstr(stdscr, 3, 3,
        "Finds pointers in static regions that resolve to your target.",
        color(C_NORM))
    stdscr.refresh()

    offset_s = input_box(stdscr, "Max struct offset (default 2048): ", 5, 3, 10, "2048")
    depth_s  = input_box(stdscr, "Depth 1=fast 2=thorough (default 1): ", 7, 3, 4, "1")

    try:
        max_off = int(offset_s)
        depth   = max(1, min(2, int(depth_s)))
    except ValueError:
        max_off, depth = 2048, 1

    targets = state["scan_results"][:8]   # cap to avoid absurdly long scans

    progress = {"done": 0, "total": 1, "results": None, "error": None}

    def _run_ptr():
        try:
            def cb(done, total):
                progress["done"]  = done
                progress["total"] = max(total, 1)
            progress["results"] = scan_pointers(
                state["ip"], state["pid"], targets,
                max_offset=max_off, progress_cb=cb, depth=depth)
        except Exception as e:
            progress["error"] = str(e)

    t = threading.Thread(target=_run_ptr, daemon=True)
    t.start()

    spinner = ["|", "/", "-", "\\"]
    spin_i  = 0
    safe_addstr(stdscr, 10, 3, "Scanning for pointers...", color(C_WARN))
    while t.is_alive():
        frac = progress["done"] / progress["total"]
        safe_addstr(stdscr, 11, 3,
            f"{spinner[spin_i % 4]}  "
            f"{progress['done'] // 1024:,} KB / {progress['total'] // 1024:,} KB",
            color(C_WARN))
        draw_progress_bar(stdscr, 12, 3, min(w - 8, 60), frac,
                          f"  {int(frac * 100)}%")
        stdscr.refresh()
        time.sleep(0.1)
        spin_i += 1

    if progress["error"]:
        add_log(f"Pointer scan error: {progress['error']}", "error")
        message_box(stdscr, [f"Error: {progress['error']}"], "Scan Failed", C_ERR)
        return

    results = progress["results"] or []
    # Sort: static (module-based) pointers first, then by chain base address
    results.sort(key=lambda r: (0 if r["static"] else 1, r["chain"][0]))
    add_log(f"Pointer scan: {len(results)} chains found (depth={depth})")

    if not results:
        message_box(stdscr,
            ["No pointer chains found.",
             "",
             "Try increasing Max Struct Offset,",
             "or use depth=2 for a two-hop search."],
            "No Results", C_WARN)
        return

    # ── Browse results ────────────────────────────────────────────────────────
    sel    = 0
    offset = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, f"POINTER RESULTS  ({len(results)} chains)")
        safe_addstr(stdscr, 2, 3,
            "* = static (stable across restarts)   arrows navigate   Enter: save   Q: back",
            color(C_NORM))

        hdr = f"  {'Base address':<20}  {'Chain':<30}  {'Target':<18}  St"
        safe_addstr(stdscr, 3, 2, hdr[:w - 4], color(C_TITLE) | curses.A_UNDERLINE)

        visible = h - 7
        if sel < offset:             offset = sel
        if sel >= offset + visible:  offset = sel - visible + 1

        for i, r in enumerate(results[offset:offset + visible]):
            ri      = offset + i
            chain   = r["chain"]
            static  = r["static"]
            tgt     = r["target"]

            # Format chain: [0x..., +offset] or [0x..., +mid, +final]
            if len(chain) == 2:
                chain_str = f"[{hex(chain[0])}] +{chain[1]}"
            else:
                chain_str = f"[{hex(chain[0])}] +{chain[1]} → +{chain[2]}"

            st_mark = "*" if static else " "
            line    = f"  {hex(chain[0]):<20}  {chain_str:<30}  {hex(tgt):<18}  {st_mark}"
            attr    = color(C_SEL) | curses.A_BOLD if ri == sel else (
                      color(C_OK) if static else color(C_NORM))
            safe_addstr(stdscr, 5 + i, 2, line[:w - 4].ljust(w - 4), attr)

        # Verify selected chain resolves correctly (live read)
        sel_r = results[sel]
        resolved = resolve_pointer_chain(state["ip"], state["pid"], sel_r["chain"])
        res_str  = hex(resolved) if resolved else "unresolvable"
        safe_addstr(stdscr, h - 3, 3,
            f"Selected resolves to: {res_str}  (target was {hex(sel_r['target'])})",
            color(C_OK if resolved == sel_r["target"] else C_WARN))

        draw_statusbar(stdscr, [
            (f"{len(results)} chains", C_WARN),
            ("* = stable", C_OK),
            ("Enter save as cheat", C_OK),
            ("Q back", C_NORM),
        ])
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP and sel > 0:
            sel -= 1
        elif key == curses.KEY_DOWN and sel < len(results) - 1:
            sel += 1
        elif key in (curses.KEY_ENTER, 10, 13):
            _save_pointer_cheat(stdscr, results[sel])
        elif key in (ord('q'), ord('Q')):
            break


def _save_pointer_cheat(stdscr, ptr_result):
    """Prompt and save a pointer chain as a cheat entry."""
    stdscr.clear()
    draw_border(stdscr, "SAVE POINTER CHEAT")
    chain = ptr_result["chain"]
    if len(chain) == 2:
        chain_disp = f"{hex(chain[0])} + {chain[1]}"
    else:
        chain_disp = f"{hex(chain[0])} +{chain[1]} -> +{chain[2]}"
    safe_addstr(stdscr, 2, 3, f"Chain  : {chain_disp}", color(C_OK) | curses.A_BOLD)
    safe_addstr(stdscr, 3, 3, f"Target : {hex(ptr_result['target'])}", color(C_WARN))
    safe_addstr(stdscr, 4, 3,
        "Stable: " + ("Yes — lives in module image" if ptr_result["static"]
                      else "No  — heap pointer, may shift on restart"),
        color(C_OK if ptr_result["static"] else C_WARN))
    stdscr.refresh()

    name  = input_box(stdscr, "Cheat name    : ", 6, 3, 40)
    val_s = input_box(stdscr, "Lock-in value : ", 8, 3, 20)
    typ   = cycle_input(stdscr, "Cheat type    : ", 10, 3,
                        ["freeze", "write"], "freeze")
    try:
        val = int(val_s)
    except ValueError:
        message_box(stdscr, ["Invalid value."], "Error", C_ERR)
        return

    entry = {
        "name":    name or f"Ptr@{hex(chain[0])}",
        "type":    "pointer",
        "chain":   chain,
        "address": ptr_result["target"],   # resolved target (may change; chain is canonical)
        "value":   val,
        "width":   state["scan_width"],
        "freeze_type": typ,   # how to apply value: freeze or write
    }
    state["cheats"].append(entry)
    add_log(f"Saved pointer cheat '{entry['name']}'")
    message_box(stdscr,
        [f"  {entry['name']}",
         f"  Chain: {chain_disp}",
         f"  Value: {val}  [{typ}]",
         "",
         "Pointer cheats resolve the address fresh each activation,",
         "so they survive game restarts."],
        "Pointer Cheat Saved", C_OK)


def do_clear_results(stdscr):
    if not state["scan_results"] and not state["scan_history"]:
        message_box(stdscr, ["No scan results to clear."], "Clear", C_WARN)
        return
    n = len(state["scan_results"])
    if confirm_box(stdscr, f"Clear {n} scan results and history?", "Clear Results"):
        state["scan_results"] = []
        state["scan_history"] = []
        add_log("Scan results cleared", "warn")
        message_box(stdscr, ["Results cleared.", "Ready for a fresh First Scan (S)."],
                    "Cleared", C_OK)


def do_log(stdscr):
    offset = max(0, len(state["log"]) - 20)
    level_colors = {"error": C_ERR, "warn": C_WARN, "info": C_OK}
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, f"LOG  ({len(state['log'])} entries)")
        visible = h - 6
        for i, entry in enumerate(state["log"][offset:offset + visible]):
            cp  = level_colors.get(entry["level"], C_NORM)
            tag = {"error": "ERR", "warn": "WRN", "info": "INF"}.get(
                entry["level"], "   ")
            line = f"[{entry['ts']}] [{tag}]  {entry['msg']}"
            safe_addstr(stdscr, 3 + i, 3, line[:w - 6], color(cp))
        draw_statusbar(stdscr, [
            (f"{offset+1}-{min(offset+visible, len(state['log']))}"
             f"/{len(state['log'])}", C_WARN),
            ("arrows scroll", C_NORM),
            ("Q back", C_NORM),
        ])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_UP   and offset > 0:                      offset -= 1
        elif key == curses.KEY_DOWN and offset < len(state["log"]) - 1: offset += 1
        elif key in (ord('q'), ord('Q')):
            break

# ── main loop ──────────────────────────────────────────────────────────────

def main(stdscr):
    curses.curs_set(0)
    curses.noecho()
    init_colors()
    stdscr.keypad(True)

    screen = "connect"
    while True:
        if screen == "connect":
            screen = screen_connect(stdscr)
        elif screen == "main":
            screen = screen_main(stdscr)
            if screen is None:
                break
        elif screen == "proc":
            try:
                procs  = ps5_proc_list(state["ip"])
                screen = screen_proc_select(stdscr, procs)
            except Exception as e:
                message_box(stdscr, [f"Error: {e}"], "Connection Error", C_ERR)
                screen = "connect"
        else:
            break

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    print("\nps5cheats_tui exited.")
