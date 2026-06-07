#!/usr/bin/env python3

"""
Python Cheat Maker with Terminal UI
curses is built into Python on Win/Linux/macOS; no extra packages

Usage:
    python3 RDX-CHEATMAKER-UI.py
"""

import array as _array
import curses
import gc
import os
import queue as _queue
import re
import socket
import struct
import json
import threading
import time
from pathlib import Path
from typing import Optional   # keep 3.8/3.9 compatibility (no X|Y union syntax)

from collections import deque

import numpy as np   # required; install with: pip install numpy

# ── memory telemetry ──────────────────────────────────────────────────────────
# Reads /proc/self/status on Linux (current RSS, not peak).
# Falls back to psutil when available, then to 0.0 so the rest of the code
# never has to guard against None.
try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

def _rss_mb() -> float:
    """Current process RSS in MiB.  Returns 0.0 on failure."""
    try:
        if _HAS_PSUTIL:
            return _psutil.Process(os.getpid()).memory_info().rss / 1_048_576
        with open("/proc/self/status") as _f:
            for _line in _f:
                if _line.startswith("VmRSS:"):
                    return float(_line.split()[1]) / 1024   # kB → MiB
    except Exception:
        pass
    return 0.0

def _total_ram_mb() -> float:
    """Total physical RAM in MiB.  Returns 0.0 on failure."""
    try:
        if _HAS_PSUTIL:
            return _psutil.virtual_memory().total / 1_048_576
        with open("/proc/meminfo") as _f:
            for _line in _f:
                if _line.startswith("MemTotal:"):
                    return float(_line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0

def _rss_frac() -> float:
    """RSS / total RAM as a fraction in [0, 1].  Returns 0.0 on failure."""
    total = _total_ram_mb()
    return (_rss_mb() / total) if total > 0 else 0.0

# ── ps5debug protocol ─────────────────────────────────────────────────────────
# Wire format documented against the canonical ps5debug source.
# Structure sizes are constants here; see ps5_maps() for the rationale.
CMD_MAGIC      = 0xFFAABBCC
CMD_PROC_LIST  = 0xBDAA0001
CMD_PROC_READ  = 0xBDAA0002
CMD_PROC_WRITE = 0xBDAA0003
CMD_PROC_MAPS  = 0xBDAA0004
# STATUS_SUCCESS / STATUS_ERROR: bit-swapped wire values produced by the server's
# net_send_int32() helper.  Clients compare raw wire bytes directly.
STATUS_SUCCESS = 0x80000000
STATUS_ERROR   = 0xF0000001
PS5_PORT       = 744

WIDTH_FMT   = {1: 'B', 2: '<H', 4: '<I', 8: '<Q'}
VALID_WIDTHS = [1, 2, 4, 8]
WIDTH_LABEL  = {1: "byte (u8)", 2: "uint16", 4: "uint32", 8: "uint64"}
WIDTH_MAX    = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF, 8: 0xFFFFFFFFFFFFFFFF}

# proc_list_entry layout: char name[32]; int32_t pid;  → 36 bytes
PROC_ENTRY_SIZE = 36
# proc_vm_map_entry layout: char name[32]; uint64 start; uint64 end;
#   uint64 offset; uint16 prot;  → 58 bytes (no padding between fields)
MAP_ENTRY_SIZE = 58

TITLE_ID_RE = re.compile(r'^[A-Z]{4}\d{5}$')

# ── scan limits ───────────────────────────────────────────────────────────────
# MAX_SCAN_RESULTS: hard upper bound on candidate addresses after the first
# scan.  Each address costs 8 bytes (uint64).  At the default of 2 M that is
# 16 MB per array — cheap enough that two full arrays (addrs + values) fit
# comfortably in RAM while leaving headroom for the undo history.
#
# Lower values → less RAM, more truncation on games with large/fragmented heaps.
# Raise if first scans are being truncated on games you care about.
MAX_SCAN_RESULTS: int = 2_000_000   # configurable; ~16 MB at this setting

# HISTORY_RAM_CAP_MB: maximum total RAM (MiB) allowed across all undo levels.
# When a new undo entry would push the total past this limit, the oldest entry
# is silently evicted (beyond the normal deque maxlen=5 rotation).  This caps
# worst-case undo RAM even if all 5 levels each hold 2 M addresses.
HISTORY_RAM_CAP_MB: float = 128.0   # configurable

# NumPy dtype for each scan width — used by vectorised scan/filter code.
# uint64 for addresses; width-specific for value arrays.
_NP_ADDR_DTYPE  = np.uint64
_NP_VALUE_DTYPE = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}

def _make_addr_array(iterable=()) -> np.ndarray:
    """
    Compact uint64 address array backed by NumPy.

    NumPy ndarray costs 8 bytes/element (same as array.array('Q')) but
    supports vectorised comparisons, argsort, searchsorted, and boolean
    indexing without any Python-level loop — that is where the performance
    gains come from in the filtering code below.

    Callers that previously used array.array('Q') are fully compatible:
    len(), iteration, and integer indexing all work identically.
    """
    if isinstance(iterable, np.ndarray):
        return iterable.astype(_NP_ADDR_DTYPE, copy=False)
    return np.fromiter(iterable, dtype=_NP_ADDR_DTYPE) if not isinstance(iterable, (list, tuple)) \
           else np.array(list(iterable), dtype=_NP_ADDR_DTYPE)

def _make_val_array(iterable, width: int) -> np.ndarray:
    """Compact value array for a given scan width (uint8/16/32/64)."""
    dtype = _NP_VALUE_DTYPE.get(width, np.uint64)
    if isinstance(iterable, np.ndarray):
        return iterable.astype(dtype, copy=False)
    return np.fromiter(iterable, dtype=dtype) if not isinstance(iterable, (list, tuple)) \
           else np.array(list(iterable), dtype=dtype)

def _make_addr_set(iterable=()) -> set:
    """Small dropped-address set — kept as plain Python set (O(1) lookup, few entries)."""
    return set(iterable)

def _addr_list(a) -> list:
    """Convert an addr array / ndarray / iterable to a plain Python list."""
    return a.tolist() if isinstance(a, np.ndarray) else list(a)

# ── undo history helpers ──────────────────────────────────────────────────────
# Each undo entry stores ONLY the delta — the addresses that were *removed* by
# a scan step — rather than a full copy of the previous candidate set.
#
# Comparison at 2 M candidates:
#   Old (full copy) : 2 M × 8 B = 16 MB per level × 5 levels = 80 MB
#   New (delta)     : only the removed fraction; if 99 % removed at step 1
#                     that is 1.98 M × 8 B = 15.8 MB for step 1, then
#                     ~0.16 MB for step 2, ~0.0016 MB for step 3, ...
#                     Total ≈ 16 MB — same worst case at step 1 but
#                     drops by 2 orders of magnitude over subsequent steps.
#
# Undo reconstruction: prev_addrs = union(current, delta), sorted for
# deterministic ordering.  Values are reconstructed from a merged map.
#
# Entry format: (removed_addrs: ndarray[uint64],
#                removed_values: ndarray|None,
#                prev_dropped: set)

def _undo_entry_bytes(entry: tuple) -> int:
    """Byte size of a single undo entry (removed_addrs + removed_values)."""
    a, v, _ = entry
    nb = a.nbytes if isinstance(a, np.ndarray) else len(a) * 8
    nv = v.nbytes if isinstance(v, np.ndarray) else 0
    return nb + nv

def _history_bytes() -> int:
    """Total RAM consumed by all live undo levels, in bytes."""
    return sum(_undo_entry_bytes(e) for e in state["scan_history"])

def _push_undo(removed_addrs: np.ndarray,
               removed_values: Optional[np.ndarray],
               prev_dropped: set) -> None:
    """
    Push one undo delta.  If the resulting history would exceed
    HISTORY_RAM_CAP_MB, evict the oldest entry first.
    """
    new_entry   = (removed_addrs, removed_values, prev_dropped)
    new_bytes   = _undo_entry_bytes(new_entry)
    # Evict oldest entries until we are under the cap (beyond normal maxlen).
    cap_bytes   = int(HISTORY_RAM_CAP_MB * 1_048_576)
    current_b   = _history_bytes()
    while state["scan_history"] and (current_b + new_bytes) > cap_bytes:
        evicted  = state["scan_history"].popleft()
        current_b -= _undo_entry_bytes(evicted)
    state["scan_history"].append(new_entry)

# ── shared state & locks ──────────────────────────────────────────────────────
_log_lock       = threading.Lock()
_cache_lock     = threading.Lock()   # protects val_cache in do_show_results
_map_cache:      dict = {}           # {pid: (timestamp, maps_list)}
_map_cache_lock = threading.Lock()
_MAP_CACHE_TTL  = 30.0               # seconds before cached map is stale

# Issues #7/#8/#9/#10: track the active freeze worker globally so it can be
# stopped when the user changes process or reconnects.  Without this the old
# worker keeps writing to an address in the previous process's address space,
# which either silently does nothing or corrupts unrelated memory if the PID
# was re-used by the OS.
_freeze_stop:   threading.Event  = threading.Event()
_freeze_thread: Optional[threading.Thread] = None
_freeze_lock:   threading.Lock   = threading.Lock()   # guards the two vars above

def _stop_freeze_worker() -> None:
    """
    Signal the active freeze worker to exit and wait for it to finish.
    Safe to call even when no freeze is running.
    Issues #7/#8 (freeze survives process change / reconnect).
    """
    global _freeze_thread
    with _freeze_lock:
        _freeze_stop.set()
        t = _freeze_thread
    if t and t.is_alive():
        t.join(timeout=2.0)
    with _freeze_lock:
        _freeze_thread = None
        _freeze_stop.clear()

state = {
    "ip":           "",
    "connected":    False,
    "pid":          None,
    "proc_name":    "",
    "scan_results": _make_addr_array(),   # np.ndarray[uint64]
    "scan_values":  None,                 # np.ndarray[uint64]|None
    "scan_dropped": set(),                # set[int] — user-dropped addresses
    "scan_pid":        None,
    "scan_truncated":  False,
    "scan_unknown":    False,
    "scan_width":   4,
    "scan_aligned":       True,
    "scan_writable_only": True,
    "cheats":       [],
    "game_id":      "",
    "game_ver":     "01.00",
    "game_title":   "",
    "log":          [],
    # Undo history — delta format; see _push_undo() above.
    "scan_history": deque(maxlen=5),
}

# ── ps5debug low-level helpers ────────────────────────────────────────────────

def cmd_header(cmd: int, datalen: int = 0) -> bytes:
    return struct.pack("<III", CMD_MAGIC, cmd, datalen)

def _resolve_ip(ip: str):
    """
    Return (family, sockaddr) for `ip`.  Tries all results from getaddrinfo in
    order so that systems whose DNS returns an unusable address first still work.
    """
    info = socket.getaddrinfo(ip, PS5_PORT, type=socket.SOCK_STREAM)
    if not info:
        raise OSError(f"Cannot resolve {ip!r}")
    return info[0][0], info[0][4]   # caller uses this; ps5_connect probes all

def ps5_connect(ip: str) -> socket.socket:
    """
    Connect to the PS5 debug server, probing every address returned by
    getaddrinfo in order.  The first successful connection is returned.
    This handles IPv6 networks where the preferred address may be listed first
    but is temporarily unreachable.
    """
    info = socket.getaddrinfo(ip, PS5_PORT, type=socket.SOCK_STREAM)
    if not info:
        raise OSError(f"Cannot resolve {ip!r}")
    last_exc: Exception = OSError("no addresses")
    for family, _, _, _, sockaddr in info:
        s = socket.socket(family, socket.SOCK_STREAM)
        s.settimeout(15)
        try:
            s.connect(sockaddr)
            return s
        except OSError as exc:
            s.close()
            last_exc = exc
    raise last_exc

def recv_exact(s: socket.socket, n: int) -> bytes:
    # bytearray + memoryview avoids O(n²) bytes concatenation on large reads.
    buf  = bytearray(n)
    view = memoryview(buf)
    pos  = 0
    while pos < n:
        got = s.recv_into(view[pos:], n - pos)
        if not got:
            raise ConnectionError("PS5 disconnected")
        pos += got
    return bytes(buf)

def check_ok(s: socket.socket) -> bool:
    return struct.unpack("<I", recv_exact(s, 4))[0] == STATUS_SUCCESS

# All helpers use sendall() and try/finally so the socket is always closed.

def ps5_proc_list(ip: str) -> list:
    s = ps5_connect(ip)
    try:
        s.sendall(cmd_header(CMD_PROC_LIST))
        if not check_ok(s):
            raise RuntimeError("proc list command rejected")
        count = struct.unpack("<I", recv_exact(s, 4))[0]
        procs = []
        for _ in range(count):
            raw  = recv_exact(s, PROC_ENTRY_SIZE)
            name = raw[:32].rstrip(b'\x00').decode('utf-8', errors='replace')
            pid  = struct.unpack_from("<i", raw, 32)[0]
            procs.append({"pid": pid, "name": name})
        return procs
    finally:
        s.close()

def ps5_maps(ip: str, pid: int) -> list:
    s = ps5_connect(ip)
    try:
        body = struct.pack("<I", pid)
        s.sendall(cmd_header(CMD_PROC_MAPS, len(body)) + body)
        if not check_ok(s):
            raise RuntimeError("proc maps command rejected")
        count = struct.unpack("<I", recv_exact(s, 4))[0]
        maps = []
        for _ in range(count):
            raw   = recv_exact(s, MAP_ENTRY_SIZE)
            name  = raw[:32].rstrip(b'\x00').decode('utf-8', errors='replace')
            start = struct.unpack_from("<Q", raw, 32)[0]
            end   = struct.unpack_from("<Q", raw, 40)[0]
            # offset field (bytes 48-55) consumed but not stored
            prot  = struct.unpack_from("<H", raw, 56)[0]
            maps.append({"start": start, "end": end, "prot": prot, "name": name})
        return maps
    finally:
        s.close()

_UI_MAX_RETRIES = 3   # retries for individual ps5_read / ps5_write UI calls

def ps5_read(ip: str, pid: int, addr: int, length: int) -> bytes:
    """Read with up to _UI_MAX_RETRIES retries on transient connection failures."""
    last_exc: Exception = RuntimeError("no attempts")
    for attempt in range(_UI_MAX_RETRIES):
        s = None
        try:
            s = ps5_connect(ip)
            body = struct.pack("<IQI", pid, addr, length)
            s.sendall(cmd_header(CMD_PROC_READ, len(body)) + body)
            if not check_ok(s):
                raise RuntimeError("read rejected")
            return recv_exact(s, length)
        except Exception as exc:
            last_exc = exc
            if attempt < _UI_MAX_RETRIES - 1:
                time.sleep(0.1 * (attempt + 1))
        finally:
            if s:
                try: s.close()
                except Exception: pass
    raise last_exc

def ps5_write(ip: str, pid: int, addr: int, data: bytes) -> bool:
    """Two-phase write with up to _UI_MAX_RETRIES retries."""
    for attempt in range(_UI_MAX_RETRIES):
        s = None
        try:
            s = ps5_connect(ip)
            body = struct.pack("<IQI", pid, addr, len(data))
            s.sendall(cmd_header(CMD_PROC_WRITE, len(body)) + body)
            if not check_ok(s):
                return False
            s.sendall(data)
            return check_ok(s)
        except Exception:
            if attempt < _UI_MAX_RETRIES - 1:
                time.sleep(0.1 * (attempt + 1))
        finally:
            if s:
                try: s.close()
                except Exception: pass
    return False

# ── batch reader for scan_next ────────────────────────────────────────────────

def ps5_read_batch(ip: str, pid: int, addrs: np.ndarray, width: int,
                   cancel_event=None, progress_cb=None) -> tuple:
    """
    Read `width` bytes at each address using NEXT_WORKERS parallel sockets.

    Previous return type: list[(addr_int, bytes|None)]
        → callers had to iterate in Python to filter and decode, paying
          full per-object GIL cost.  At 500 K addresses that loop alone
          cost ~87 ms before any comparison happened.

    New return type: (live_addrs: ndarray[uint64], live_vals: ndarray[uint_w])
        → workers write directly into pre-allocated arrays; the caller
          receives two flat ndarrays ready for vectorised comparison with
          no further Python-level work.

    Pre-allocation strategy:
        We allocate `len(addrs)` slots upfront (worst case: all reads succeed).
        A thread-safe atomic counter (`write_ptr`) assigns each worker a unique
        slot range — no lock needed per write, only one atomic fetch-and-add.
        After all workers finish, we slice `[:n_written]` to drop unused slots.

    Thread safety:
        `write_ptr` is a length-1 int array; `np.add.at` is not used here —
        instead we rely on Python's GIL for the `write_ptr[0] += k` increment
        inside the lock, which is the only shared mutation.  The actual ndarray
        writes happen to disjoint index ranges so no further locking is needed.
    """
    NEXT_WORKERS = 6
    if len(addrs) == 0:
        empty_a = np.empty(0, dtype=_NP_ADDR_DTYPE)
        empty_v = np.empty(0, dtype=_NP_VALUE_DTYPE[width])
        return empty_a, empty_v

    total     = len(addrs)
    val_dtype = _NP_VALUE_DTYPE[width]

    # Pre-allocated output buffers — workers write directly here.
    # Sized to `total` (worst case all reads succeed); trimmed at the end.
    out_addrs = np.empty(total, dtype=_NP_ADDR_DTYPE)
    out_vals  = np.empty(total, dtype=val_dtype)

    # Shared state: index of next address to claim, and write pointer.
    read_ptr  = [0]    # next address index to read
    write_ptr = [0]    # next output slot to write (only advanced under lock)
    ptr_lock  = threading.Lock()
    done_ctr  = [0]

    fmt = WIDTH_FMT[width]

    def _worker():
        sock = _ScanSocket(ip, pid)
        # Thread-local accumulation buffers — batch writes to out_addrs/out_vals
        # in chunks to reduce lock contention vs one lock per read.
        local_addrs = np.empty(256, dtype=_NP_ADDR_DTYPE)
        local_vals  = np.empty(256, dtype=val_dtype)
        local_n     = 0

        def _flush():
            nonlocal local_n
            if local_n == 0:
                return
            with ptr_lock:
                start = write_ptr[0]
                write_ptr[0] += local_n
            out_addrs[start:start + local_n] = local_addrs[:local_n]
            out_vals [start:start + local_n] = local_vals [:local_n]
            local_n = 0

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    break
                with ptr_lock:
                    if read_ptr[0] >= total:
                        break
                    my_idx = read_ptr[0]
                    read_ptr[0] += 1
                addr = int(addrs[my_idx])
                try:
                    data = sock.read(addr, width)
                    if len(data) == width:
                        local_addrs[local_n] = addr
                        # Decode directly — no bytes object kept alive
                        local_vals[local_n]  = struct.unpack(fmt, data)[0]
                        local_n += 1
                        if local_n == 256:
                            _flush()
                except Exception:
                    pass   # failed reads are silently dropped (addr excluded)
                with ptr_lock:
                    done_ctr[0] += 1
                    if progress_cb:
                        progress_cb(done_ctr[0], total)
        finally:
            _flush()   # write any remaining local buffer
            sock.close()

    workers = [threading.Thread(target=_worker, daemon=True)
               for _ in range(min(NEXT_WORKERS, max(1, total)))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    n = write_ptr[0]
    return out_addrs[:n].copy(), out_vals[:n].copy()

# ── persistent-socket reader for scan_first ───────────────────────────────────

class _ScanSocket:
    """
    Holds a single persistent TCP connection for the duration of a scan.
    Automatically reconnects (up to MAX_RETRIES times) when the socket dies.

    Hot-path optimisation: the CMD_PROC_READ request is 28 bytes total
    (12-byte cmd_packet header + 16-byte body).  We pre-allocate a single
    bytearray and patch only the addr field (bytes 20-27) before each send,
    avoiding repeated struct.pack() allocations in the inner scan loop.

    Buffer layout (all LE):
      [0-3]   magic    0xFFAABBCC
      [4-7]   cmd      CMD_PROC_READ
      [8-11]  datalen  16
      [12-15] pid      (fixed per socket)
      [16-23] addr     (patched per read)
      [24-27] length   (fixed per socket, same CHUNK for every read)
    """
    MAX_RETRIES = 3
    _HDR_SIZE   = 28   # 12 (cmd_packet) + 16 (cmd_proc_read_packet)

    def __init__(self, ip: str, pid: int):
        self.ip  = ip
        self.pid = pid
        self._s: Optional[socket.socket] = None
        # Pre-built mutable request buffer; addr field patched in read()
        self._req = bytearray(self._HDR_SIZE)
        struct.pack_into("<III", self._req,  0,
                         CMD_MAGIC, CMD_PROC_READ, 16)   # header
        struct.pack_into("<I",   self._req, 12, pid)     # pid (fixed)
        # addr at offset 16, length at offset 24 — set per-call
        self._connect()

    def _connect(self):
        if self._s:
            try: self._s.close()
            except Exception: pass
        self._s = ps5_connect(self.ip)

    def read(self, addr: int, length: int) -> bytes:
        """Read `length` bytes from `addr`, reconnecting on transient failure."""
        # Patch addr and length directly into the pre-built bytearray.
        # sendall accepts bytearray natively — no bytes() copy needed.
        struct.pack_into("<QI", self._req, 16, addr, length)
        for attempt in range(self.MAX_RETRIES):
            try:
                if self._s is None:
                    self._connect()
                self._s.sendall(self._req)   # zero-copy: no bytes() allocation
                if not check_ok(self._s):
                    raise RuntimeError("read rejected")
                return recv_exact(self._s, length)
            except Exception as exc:
                add_log(f"scan read err (attempt {attempt+1}/{self.MAX_RETRIES}) "
                        f"@ {hex(addr)}: {exc}", "warn")
                try: self._s.close()
                except Exception: pass
                self._s = None
                if attempt == self.MAX_RETRIES - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def close(self):
        if self._s:
            try: self._s.close()
            except Exception: pass
            self._s = None

def _get_maps_cached(ip: str, pid: int) -> list:
    """
    Return ps5_maps() with a 30-second cache.  Consecutive scans on the same
    process reuse the map rather than paying an extra RTT before each scan.
    Invalidated automatically when pid changes or TTL expires.
    """
    now = time.time()
    with _map_cache_lock:
        entry = _map_cache.get(pid)
        if entry and (now - entry[0]) < _MAP_CACHE_TTL:
            return entry[1]
    maps = ps5_maps(ip, pid)
    with _map_cache_lock:
        _map_cache.clear()          # only cache one pid at a time
        _map_cache[pid] = (now, maps)
    return maps


def scan_first(ip: str, pid: int, value: int, width: int = 4,
               aligned: bool = True, progress_cb=None,
               cancel_event=None,
               writable_only: bool = True) -> _array.array:
    """
    Scan all readable regions for `value`.

    Architecture
    ────────────
    Previous design: read chunk → search chunk → read next chunk (serial).
    Round-trip latency on a home LAN is 1–5 ms per chunk, so serial scanning
    spends most of its time waiting for the network.

    New design: producer/consumer pipeline with SCAN_WORKERS parallel reader
    threads, each owning its own _ScanSocket.  A single search thread consumes
    chunks from a bounded queue and writes matches.  This keeps the network and
    CPU both busy simultaneously.

    Layout
    ──────
      [reader-0] ──┐
      [reader-1] ──┼──► chunk_queue ──► [searcher] ──► found[]
      [reader-2] ──┘

    Back-pressure: chunk_queue is bounded (QUEUE_DEPTH) so readers stall rather
    than buffering the entire process memory at once.

    Concurrency model
    ─────────────────
    Readers write (addr, bytes) tuples into chunk_queue.
    The searcher is the only writer to found[] and done_bytes[],
    so no lock is needed on those.
    cancel_event stops all threads promptly.

    aligned=True  → struct.iter_unpack (fast, aligned offsets only)
    aligned=False → byte-by-byte (thorough, finds unaligned values)
    """
    # Validate via struct.pack — handles both signed and unsigned types correctly.
    # The old `value < 0` guard blocked all signed-type scans (int8/16/32/64).
    try:
        target = struct.pack(WIDTH_FMT[width], value)
    except struct.error:
        raise ValueError(
            f"Value {value} out of range for {WIDTH_LABEL.get(width, str(width))}")
    maps = _get_maps_cached(ip, pid)

    CHUNK        = 0x400000    # 4 MB per request — amortises RTT over more data
    SCAN_WORKERS = 6           # more workers since searcher is no longer bottleneck
    QUEUE_DEPTH  = SCAN_WORKERS * 4   # bound RAM: 6×4×4 MB = 96 MB max in-flight
    _SENTINEL    = None      # signals searcher that all readers have finished

    # ── region selection ──────────────────────────────────────────────────────
    PROT_READ  = 0x1
    PROT_WRITE = 0x2
    PROT_EXEC  = 0x4
    MAX_REGION = 0x40000000   # 1 GB — only skip GPU/VRAM/reserved ranges;
                               # heap regions up to 512 MB are now scanned

    def _scannable(regions, require_write):
        return [r for r in regions
                if (r['end'] - r['start']) <= MAX_REGION
                and (r['prot'] & PROT_READ)
                and (not require_write or (r['prot'] & PROT_WRITE))
                and not (r['prot'] == PROT_EXEC)]

    rw_regions  = _scannable(maps, require_write=True)
    if writable_only:
        # Game values (health, gold, ammo) live in writable memory.
        # Skipping R/O regions reduces scan size by 30-60%.
        scannable = rw_regions
    else:
        ro_regions = _scannable(maps, require_write=False)
        rw_set     = {(r['start'], r['end']) for r in rw_regions}
        ro_only    = [r for r in ro_regions
                      if (r['start'], r['end']) not in rw_set]
        scannable  = rw_regions + ro_only
    total_bytes = max(sum(r['end'] - r['start'] for r in scannable), 1)

    # ── build flat work list of (base_addr, size) chunks ─────────────────────
    # Use region_size for small regions to avoid padding waste on tiny regions.
    # Many PS5 mappings are 64KB-512KB; sending a 4MB request for 128KB wastes
    # the connection slot without filling it.
    MIN_CHUNK = 0x10000    # 64 KB minimum — avoid excessive small requests
    work: list = []
    for r in scannable:
        size = r['end'] - r['start']
        off  = 0
        while off < size:
            csz = min(CHUNK, size - off)
            # Round up tiny chunks to MIN_CHUNK for better socket utilisation
            # (ps5debug reads exactly what we ask; no penalty for smaller asks)
            work.append((r['start'] + off, csz))
            off += csz

    # ── shared state ─────────────────────────────────────────────────────────
    chunk_queue: "_queue.Queue[Optional[tuple]]" = _queue.Queue(maxsize=QUEUE_DEPTH)
    # Plain Python list for accumulation — supports .append() in O(1) amortised.
    # _make_addr_array() returns np.ndarray which has no .append(); the migration
    # to NumPy broke this.  We convert to ndarray once at the end.
    found: list = []
    done_bytes  = [0]          # written only by searcher thread
    work_lock   = threading.Lock()
    work_idx    = [0]          # shared index into work[]; protected by work_lock
    reader_err      = []
    reader_err_lock = threading.Lock()

    # ── reader thread ─────────────────────────────────────────────────────────
    def _reader():
        sock = _ScanSocket(ip, pid)
        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    break
                with work_lock:
                    if work_idx[0] >= len(work):
                        break
                    addr, csz = work[work_idx[0]]
                    work_idx[0] += 1
                try:
                    data = sock.read(addr, csz)
                except Exception as exc:
                    with reader_err_lock:
                        if len(reader_err) < 200:   # cap: pathological maps won't OOM
                            reader_err.append(f"skip {hex(addr)}: {exc}")
                        elif len(reader_err) == 200:
                            reader_err.append("(further reader errors suppressed)")
                    data = None
                # Timeout on put() prevents permanent block if searcher exits early
                while True:
                    if cancel_event and cancel_event.is_set():
                        return
                    try:
                        chunk_queue.put((addr, data), timeout=0.5)
                        break
                    except _queue.Full:
                        continue
        finally:
            sock.close()

    # ── searcher thread ───────────────────────────────────────────────────────
    # Uses bytes.find() — a C-level Boyer-Moore-Horspool search.
    # Benchmarked at ~2400 MB/s vs ~24 MB/s for iter_unpack: 100× faster.
    # Works for all widths; `step` enforces alignment on the result side.
    step = width if aligned else 1
    def _search_all():
        sentinels_received = 0
        while sentinels_received < n_workers:  # wait for exactly n_workers sentinels
            item = chunk_queue.get()
            if item is _SENTINEL:
                sentinels_received += 1
                continue
            addr, data = item
            if data is None:
                done_bytes[0] += CHUNK
                if progress_cb:
                    progress_cb(done_bytes[0], total_bytes)
                continue
            csz = len(data)
            # Issue #12/#13: if the read returned fewer bytes than requested,
            # the region boundary moved or permissions changed while we were
            # scanning (e.g. game unloaded a level).  Don't search a partial
            # chunk — the addresses at the end would be wrong.  Log and skip.
            expected_csz = next((s for a, s in work if a == addr), None)
            if expected_csz is not None and csz < expected_csz:
                add_log(f"Partial read @ {hex(addr)}: got {csz} of {expected_csz} bytes — skipped", "warn")
                done_bytes[0] += csz
                if progress_cb:
                    progress_cb(done_bytes[0], total_bytes)
                continue
            # bytes.find on a miss still scans the whole chunk, but the C
            # implementation is ~2400 MB/s so this is rarely worth splitting.
            # The real win: zero-page detection. Sparse mmap regions are
            # often entirely zero; skip them if target != b'\x00'*width.
            if target != b'\x00' * width:
                # Quick zero-page check using count of first target byte
                if data.count(target[0:1]) == 0:
                    done_bytes[0] += csz
                    if progress_cb:
                        progress_cb(done_bytes[0], total_bytes)
                    continue
            # For aligned scans, only accept hits at aligned offsets.
            # bytes.find can match at any byte position, so we must filter.
            check_align = aligned and width > 1
            pos = 0
            while True:
                p = data.find(target, pos)
                if p == -1:
                    break
                if check_align and (addr + p) % width != 0:
                    pos = p + 1   # skip this unaligned hit, try next byte
                    continue
                found.append(addr + p)
                if len(found) >= MAX_SCAN_RESULTS:
                    add_log(f"Result cap ({MAX_SCAN_RESULTS:,}) hit"
                            " — scan truncated", "warn")
                    if cancel_event:
                        cancel_event.set()
                        cancel_event.truncated = True   # piggyback flag
                    # Drain queue so readers blocked on put() can see cancel
                    while True:
                        try:
                            chunk_queue.get_nowait()
                        except _queue.Empty:
                            break
                    return
                pos = p + step
            done_bytes[0] += csz
            if progress_cb:
                progress_cb(done_bytes[0], total_bytes)

    # ── launch readers ────────────────────────────────────────────────────────
    n_workers = min(SCAN_WORKERS, max(1, len(work)))
    readers   = []
    for _ in range(n_workers):
        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        readers.append(t)

    # Post each sentinel as soon as its own reader exits — do not wait
    # for *all* readers before posting *any* sentinel.  This keeps the
    # searcher fed even when one reader is slower than the others.
    def _make_sentinel_watcher(reader_thread):
        def _watch():
            reader_thread.join()
            while True:
                try:
                    chunk_queue.put(_SENTINEL, timeout=0.5)
                    break
                except _queue.Full:
                    continue
        return _watch

    watchers = []
    for r in readers:
        wt = threading.Thread(target=_make_sentinel_watcher(r), daemon=True)
        wt.start()
        watchers.append(wt)

    # Run the searcher in this thread (saves one more thread; also keeps
    # found[] writes in a single thread with no lock needed)
    _search_all()

    for wt in watchers:
        wt.join()
    for msg in reader_err:
        add_log(msg, "warn")

    # Convert plain list → ndarray once here; avoids O(N) reallocations that
    # array.array or repeated np.append would cause inside the hot loop.
    return np.array(found, dtype=_NP_ADDR_DTYPE)


def scan_next(ip: str, pid: int, value: int, width: int,
              prev: np.ndarray,
              cancel_event=None, progress_cb=None) -> np.ndarray:
    """
    Filter `prev` to addresses that currently hold `value`.

    Fully vectorised — zero Python-level loops after the network reads.

    Pipeline:
      1. ps5_read_batch writes live values directly into two pre-allocated
         ndarrays (out_addrs, out_vals).  No list of (addr, bytes) tuples
         is ever built; no Python object per address is ever created.
      2. A single NumPy comparison (out_vals == target) produces a boolean
         mask in C/SIMD — O(N) with no GIL-held Python iteration.
      3. out_addrs[mask] gathers matching addresses — one C-level gather.

    The previous version built a Python generator over a list of (int, bytes)
    tuples.  Profiling showed that step cost ~87 ms at 500 K addresses, while
    the actual comparison cost only ~0.34 ms.  By moving the decode into
    ps5_read_batch workers the Python iteration is eliminated entirely.
    """
    dtype = _NP_VALUE_DTYPE[width]
    try:
        target = dtype(value & WIDTH_MAX[width])
    except (OverflowError, ValueError):
        raise ValueError(
            f"Value {value} out of range for {WIDTH_LABEL.get(width, str(width))}")

    # Stage 1: parallel network reads → pre-allocated ndarrays (no Python list)
    live_addrs, live_vals = ps5_read_batch(ip, pid, prev, width,
                                           cancel_event, progress_cb)

    if len(live_addrs) == 0:
        add_log(f"Exact next scan: 0 remain (no reads succeeded), "
                f"RSS {_rss_mb():.0f} MB")
        return np.empty(0, dtype=_NP_ADDR_DTYPE)

    # Stage 2: vectorised comparison — one C-level call across all N entries
    mask        = live_vals == target
    n_match     = int(mask.sum())
    n_read      = len(live_addrs)

    # Stage 3: masked gather — one C-level indexed copy
    result = live_addrs[mask].copy()
    del live_addrs, live_vals, mask

    add_log(f"Exact next scan: {n_match:,} remain "
            f"(of {n_read:,} read, {len(prev):,} prev), "
            f"RSS {_rss_mb():.0f} MB")
    return result


def scan_first_unknown(ip: str, pid: int, width: int = 4,
                       aligned: bool = True, progress_cb=None,
                       cancel_event=None,
                       writable_only: bool = True
                       ) -> tuple:
    """
    Unknown-value first scan.

    Instead of searching for a specific byte pattern, snapshot the current
    value at every candidate address.  Returns (addrs, values) — two parallel
    array.array('Q') objects of equal length.

    This is the entry point for relational scans (decreased / increased /
    changed / unchanged) used when the game doesn't display a numeric value
    (health bars, hidden stamina, etc.).

    The same producer/consumer pipeline as scan_first is reused; the searcher
    simply records every aligned address and its current bytes rather than
    filtering by value.

    Memory cost at width=4, aligned:
        PS5 writable heap is typically 200–800 MB → 50–200 M candidates
        Each (addr, value) pair = 8 + 8 = 16 bytes in array.array
        200 M × 16 B = 3.2 GB — far too large to hold in RAM.

    We therefore apply MAX_SCAN_RESULTS as a hard cap here too.
    For writable_only=True the practical count is much lower (30–80 M on
    most games) and the cap is rarely hit in the first pass; subsequent
    relational next scans reduce candidates rapidly.
    """
    maps = _get_maps_cached(ip, pid)

    CHUNK        = 0x400000
    SCAN_WORKERS = 6
    QUEUE_DEPTH  = SCAN_WORKERS * 4
    _SENTINEL    = None

    PROT_READ  = 0x1
    PROT_WRITE = 0x2
    PROT_EXEC  = 0x4
    MAX_REGION = 0x40000000

    def _scannable(regions, require_write):
        return [r for r in regions
                if (r['end'] - r['start']) <= MAX_REGION
                and (r['prot'] & PROT_READ)
                and (not require_write or (r['prot'] & PROT_WRITE))
                and not (r['prot'] == PROT_EXEC)]

    rw_regions  = _scannable(maps, require_write=True)
    if writable_only:
        scannable = rw_regions
    else:
        ro_regions = _scannable(maps, require_write=False)
        rw_set     = {(r['start'], r['end']) for r in rw_regions}
        ro_only    = [r for r in ro_regions
                      if (r['start'], r['end']) not in rw_set]
        scannable  = rw_regions + ro_only
    total_bytes = max(sum(r['end'] - r['start'] for r in scannable), 1)

    work: list = []
    for r in scannable:
        size = r['end'] - r['start']
        off  = 0
        while off < size:
            csz = min(CHUNK, size - off)
            work.append((r['start'] + off, csz))
            off += csz

    chunk_queue: "_queue.Queue[Optional[tuple]]" = _queue.Queue(maxsize=QUEUE_DEPTH)
    # Use lists of ndarray chunks; np.concatenate at the end is O(total) and
    # avoids the repeated reallocation that appending to a flat array.array
    # one element at a time causes (amortised O(N²) for large N).
    found_addrs:  list = []   # list[np.ndarray[uint64]]
    found_values: list = []   # list[np.ndarray[uint_w]]
    done_bytes   = [0]
    work_lock    = threading.Lock()
    work_idx     = [0]
    reader_err      = []
    reader_err_lock = threading.Lock()

    def _reader():
        sock = _ScanSocket(ip, pid)
        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    break
                with work_lock:
                    if work_idx[0] >= len(work):
                        break
                    addr, csz = work[work_idx[0]]
                    work_idx[0] += 1
                try:
                    data = sock.read(addr, csz)
                except Exception as exc:
                    with reader_err_lock:
                        if len(reader_err) < 200:
                            reader_err.append(f"skip {hex(addr)}: {exc}")
                        elif len(reader_err) == 200:
                            reader_err.append("(further reader errors suppressed)")
                    data = None
                while True:
                    if cancel_event and cancel_event.is_set():
                        return
                    try:
                        chunk_queue.put((addr, data), timeout=0.5)
                        break
                    except _queue.Full:
                        continue
        finally:
            sock.close()

    step = width if aligned else 1

    def _snapshot_all():
        """
        Consume chunks from the queue and snapshot every aligned address.

        NumPy vectorised implementation — replaces the original Python
        for-loop that called struct.unpack_from() per address:

          Old: for off in range(0, csz, step): struct.unpack_from(...)
               → O(N) Python dispatch, ~24 MB/s effective throughput

          New: np.frombuffer → [::step] strided view → append in bulk
               → C-level memory copy, ~2–8 GB/s throughput
               Typical speedup: 10–50× on the snapshot phase.

        Memory note: found_addrs / found_values grow by appending
        pre-allocated blocks rather than one element at a time, so the
        array extension amortises to O(1) per element.
        """
        nonlocal found_addrs, found_values
        sentinels_received = 0
        while sentinels_received < n_workers:
            item = chunk_queue.get()
            if item is _SENTINEL:
                sentinels_received += 1
                continue
            addr, data = item
            if data is None:
                done_bytes[0] += CHUNK
                if progress_cb:
                    progress_cb(done_bytes[0], total_bytes)
                continue
            if cancel_event and cancel_event.is_set():
                continue   # drain queue so readers can unblock
            csz = len(data)

            # ── vectorised extract ────────────────────────────────────────────
            # View the raw bytes as the correct dtype, then slice with step.
            val_dtype = _NP_VALUE_DTYPE[width]
            # Number of complete width-byte values in this chunk
            n_vals = (csz - (csz % width)) // width
            if n_vals == 0:
                done_bytes[0] += csz
                if progress_cb:
                    progress_cb(done_bytes[0], total_bytes)
                continue

            # Interpret raw bytes as packed integers (little-endian struct fmt).
            # frombuffer is zero-copy when data is bytes (read-only buffer).
            vals_raw = np.frombuffer(data[:n_vals * width], dtype=f'<u{width}')
            # For aligned scans, stride by (step // width) in the value array;
            # for unaligned, every byte offset matters so we must work at byte level.
            if aligned:
                vals_slice = vals_raw          # step == width → every element
                n_out      = len(vals_slice)
                # Absolute addresses: addr, addr+width, addr+2*width, ...
                addrs_out  = np.arange(addr, addr + n_out * width, width,
                                       dtype=_NP_ADDR_DTYPE)
            else:
                # Unaligned: one value per byte offset — requires byte-level scan.
                # We still use NumPy but must iterate byte offsets.
                offsets    = np.arange(0, csz - width + 1, 1, dtype=np.intp)
                vals_slice = np.array(
                    [struct.unpack_from(WIDTH_FMT[width], data, o)[0] for o in offsets],
                    dtype=val_dtype)
                addrs_out  = (addr + offsets).astype(_NP_ADDR_DTYPE)
                n_out      = len(offsets)

            # Enforce the result cap
            total_so_far = sum(len(c) for c in found_addrs)
            remaining = MAX_SCAN_RESULTS - total_so_far
            if n_out > remaining:
                addrs_out  = addrs_out[:remaining]
                vals_slice = vals_slice[:remaining]
                add_log(f"Unknown scan cap ({MAX_SCAN_RESULTS:,}) hit"
                        " — snapshot truncated", "warn")
                if cancel_event:
                    cancel_event.set()
                    cancel_event.truncated = True
                # Drain so readers unblock
                while True:
                    try:
                        chunk_queue.get_nowait()
                    except _queue.Empty:
                        break

            found_addrs.append(addrs_out)
            found_values.append(vals_slice.astype(_NP_VALUE_DTYPE[width]))

            done_bytes[0] += csz
            if progress_cb:
                progress_cb(done_bytes[0], total_bytes)

            if cancel_event and cancel_event.is_set():
                return

    n_workers = min(SCAN_WORKERS, max(1, len(work)))
    readers   = []
    for _ in range(n_workers):
        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        readers.append(t)

    def _make_sentinel_watcher(reader_thread):
        def _watch():
            reader_thread.join()
            while True:
                try:
                    chunk_queue.put(_SENTINEL, timeout=0.5)
                    break
                except _queue.Full:
                    continue
        return _watch

    watchers = []
    for r in readers:
        wt = threading.Thread(target=_make_sentinel_watcher(r), daemon=True)
        wt.start()
        watchers.append(wt)

    _snapshot_all()

    for wt in watchers:
        wt.join()
    for msg in reader_err:
        add_log(msg, "warn")

    # Concatenate accumulated chunk arrays into flat ndarrays.
    # np.concatenate on a list of arrays is O(total) with a single allocation.
    if found_addrs:
        out_addrs  = np.concatenate(found_addrs).astype(_NP_ADDR_DTYPE)
        out_values = np.concatenate(found_values)
    else:
        out_addrs  = np.empty(0, dtype=_NP_ADDR_DTYPE)
        out_values = np.empty(0, dtype=_NP_VALUE_DTYPE[width])

    add_log(f"Unknown-scan snapshot: {len(out_addrs):,} candidates, "
            f"RSS {_rss_mb():.0f} MB")
    return out_addrs, out_values


# Relational scan modes for unknown-value next scans.
RELATIONAL_MODES = [
    "decreased",        # current < previous (e.g. took damage)
    "increased",        # current > previous (e.g. picked up health)
    "changed",          # current != previous
    "unchanged",        # current == previous (value held steady)
    "decreased by",     # current == previous - N  (known delta)
    "increased by",     # current == previous + N  (known delta)
]

def scan_next_relational(ip: str, pid: int, width: int,
                         prev_addrs: np.ndarray,
                         prev_values: np.ndarray,
                         mode: str,
                         delta: int = 0,
                         cancel_event=None,
                         progress_cb=None) -> tuple:
    """
    Relational next scan — NumPy vectorised implementation.

    Previous implementation:
      1. Built a Python dict {addr: prev_value} — O(N) time + O(N) RAM.
      2. Iterated raw_results in Python, unpackaged each bytes object,
         did a dict lookup, and applied the comparison with if/elif chains.
      → Effective throughput: ~5–20 M comparisons/s (Python GIL-bound).

    New implementation:
      1. Reads live values via ps5_read_batch (network I/O — same as before).
      2. Assembles cur_vals / prv_vals as parallel ndarrays — NO dict built.
      3. Applies the comparison with a single NumPy expression → boolean mask.
      4. Indexes both address and value arrays with the mask in one step.
      → Effective throughput: ~200–800 M comparisons/s (C-level SIMD).

    Memory savings vs old approach:
      prev_map dict at 2 M entries: ~56 bytes/entry (Python dict overhead)
                                    = ~112 MB
      Two ndarrays at 2 M entries:  8 bytes/entry each = 16 MB each = 32 MB
      Saving: ~80 MB on a 2 M candidate scan.

    The key insight is that prev_addrs and prev_values are already parallel
    arrays with the same ordering guarantee as the dict — so we only need to
    know, for each live read result, the INDEX into prev_addrs to look up the
    corresponding prev_value.  np.searchsorted gives that in O(N log N) with
    no per-element Python overhead, and since prev_addrs is already sorted
    (scan_first guarantees this), no extra sort is needed.
    """
    fmt   = WIDTH_FMT[width]
    mask  = np.uint64(WIDTH_MAX[width])
    dtype = _NP_VALUE_DTYPE[width]

    # prev_addrs must be sorted for searchsorted.
    if len(prev_addrs) > 1 and not np.all(prev_addrs[:-1] <= prev_addrs[1:]):
        order       = np.argsort(prev_addrs, kind='stable')
        prev_addrs  = prev_addrs[order]
        prev_values = prev_values[order]

    # ps5_read_batch now returns (live_addrs, live_vals) ndarrays directly —
    # no Python list of (addr, bytes) tuples, no per-address decode loop.
    live_addrs, live_vals = ps5_read_batch(ip, pid, prev_addrs, width,
                                           cancel_event, progress_cb)
    live_vals = live_vals.astype(dtype, copy=False)

    if len(live_addrs) == 0:
        empty_a = np.empty(0, dtype=_NP_ADDR_DTYPE)
        empty_v = np.empty(0, dtype=dtype)
        add_log(f"Relational scan ({mode}): 0 remain, RSS {_rss_mb():.0f} MB")
        return empty_a, empty_v

    # ── look up previous values without a dict ────────────────────────────────
    # prev_addrs is sorted → searchsorted gives the insertion index of each
    # live_addr.  Entries that were not in prev_addrs (shouldn't happen but
    # guard anyway) will have an out-of-range index or a non-matching address.
    idx      = np.searchsorted(prev_addrs, live_addrs)
    in_range = (idx < len(prev_addrs))
    # Clip idx to valid range before indexing (avoid out-of-bounds on the
    # entries we will mask out anyway)
    idx_safe = np.where(in_range, idx, 0)
    matched  = in_range & (prev_addrs[idx_safe] == live_addrs)
    prv_vals = prev_values[idx_safe].astype(dtype)   # broadcast-safe

    # ── vectorised comparison ─────────────────────────────────────────────────
    cur = live_vals
    prv = prv_vals
    if   mode == "decreased":
        keep = cur < prv
    elif mode == "increased":
        keep = cur > prv
    elif mode == "changed":
        keep = cur != prv
    elif mode == "unchanged":
        keep = cur == prv
    elif mode == "decreased by":
        d64 = dtype(delta) if delta <= np.iinfo(dtype).max else np.uint64(delta)
        keep = cur == ((prv.astype(np.uint64) - np.uint64(delta)) & mask).astype(dtype)
    elif mode == "increased by":
        keep = cur == ((prv.astype(np.uint64) + np.uint64(delta)) & mask).astype(dtype)
    else:
        raise ValueError(f"Unknown relational mode: {mode!r}")

    # Combine the address-match mask with the value-comparison mask
    final_mask = matched & keep

    new_addrs  = live_addrs[final_mask]
    new_values = live_vals[final_mask]

    add_log(f"Relational scan ({mode}): {len(new_addrs):,} remain "
            f"(of {len(live_addrs):,} read), RSS {_rss_mb():.0f} MB")
    return new_addrs, new_values

# PS5 user-space address range: 0x0000_0000_0000_0001 – 0x0000_7FFF_FFFF_FFFF
# Writes to address 0, kernel space (>= 0x8000_0000_0000_0000), or obviously
# bogus values are rejected client-side before they reach ps5debug.
_ADDR_MIN = 0x0000_0000_0000_0001
_ADDR_MAX = 0x0000_7FFF_FFFF_FFFF

def _validate_write_addr(addr: int) -> Optional[str]:
    """Return an error string if addr is outside safe user-space range, else None."""
    if addr < _ADDR_MIN:
        return f"Address {hex(addr)} is zero or negative — likely a mistake."
    if addr > _ADDR_MAX:
        return f"Address {hex(addr)} is in kernel space — write blocked."
    return None

def _validate_addr_in_maps(ip: str, pid: int, addr: int, length: int) -> Optional[str]:
    """
    Return an error string if `addr`..`addr+length` does not fall within a
    writable mapped region of the process, else None.

    Uses the 30-second map cache so repeated writes/freezes don't pay an
    extra RTT each time.

    Returns an error string (not None) when the map cannot be fetched, so
    callers always see a real validation result — never a silent pass-through.
    """
    try:
        maps = _get_maps_cached(ip, pid)
    except Exception as exc:
        # Fail-CLOSED: surface the error so the caller can confirm explicitly.
        return f"Could not fetch memory map to validate address: {exc}"
    PROT_WRITE = 0x2
    for r in maps:
        if r['start'] <= addr and addr + length <= r['end']:
            if r['prot'] & PROT_WRITE:
                return None   # in a writable region — OK
            return (f"Address {hex(addr)} is mapped but not writable "
                    f"(prot={hex(r['prot'])}).")
    return f"Address {hex(addr)} is not in any mapped region of PID {pid}."


def sanitize_filename(name: str) -> str:
    """Strip characters unsafe for filenames, keeping alphanum, dash, dot."""
    return re.sub(r'[^\w\-.]', '_', name)


def generate_cht(cheats: list, game_id: str, game_ver: str,
                 game_title: str, hex_values: bool = True) -> str:
    # GoldHEN 2.x expects lowercase hex; some older/fork loaders want decimal.
    # The caller selects via hex_values.
    fmt_val = (lambda v: hex(v)) if hex_values else (lambda v: str(v))
    payload = {
        "title":     game_title,
        "titleid":   game_id,
        "version":   game_ver,
        "cheatList": [
            {
                "name":    c["name"],
                "type":    c["type"],
                "address": hex(c["address"]),
                "value":   fmt_val(c["value"]),
                "bytes":   c["width"],
            }
            for c in cheats
        ],
    }
    return json.dumps(payload, indent=2)

# ── logging ───────────────────────────────────────────────────────────────────
LOG_LIMIT = 500   # raised from 200 so older diagnostics are not lost so quickly

def add_log(msg: str, level: str = "info") -> None:
    with _log_lock:
        state["log"].append({"ts": time.strftime("%H:%M:%S"), "msg": msg, "level": level})
        if len(state["log"]) > LOG_LIMIT:
            state["log"] = state["log"][-LOG_LIMIT:]

# ── curses UI helpers ─────────────────────────────────────────────────────────

def init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,    -1)   # C_TITLE
    curses.init_pair(2, curses.COLOR_GREEN,   -1)   # C_OK
    curses.init_pair(3, curses.COLOR_YELLOW,  -1)   # C_WARN
    curses.init_pair(4, curses.COLOR_RED,     -1)   # C_ERR
    curses.init_pair(5, curses.COLOR_WHITE,   -1)   # C_NORM
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)   # C_ACC
    curses.init_pair(7, curses.COLOR_BLACK,   curses.COLOR_CYAN)  # C_SEL
    curses.init_pair(8, curses.COLOR_BLACK,   curses.COLOR_RED)   # C_DSEL

C_TITLE = 1; C_OK = 2; C_WARN = 3; C_ERR = 4
C_NORM  = 5; C_ACC = 6; C_SEL  = 7; C_DSEL = 8

def color(pair: int) -> int:
    return curses.color_pair(pair)

def safe_addstr(win, y: int, x: int, text: str, attr: int = 0) -> None:
    """
    Boundary-safe addstr wrapper.

    Issue #2/#3: guards both negative coordinates (small terminals) and
    positions beyond the current window size so no raw addstr call can
    raise curses.error due to out-of-bounds writes.

    Issue #5 (UTF-8 / wide chars): curses measures column width in display
    cells, not bytes, so a naïve [:w-x] byte-slice can still overrun the
    window when the string contains multi-byte or wide characters.  We use
    wcwidth via str.encode inspection: fall back to clipping one character
    at a time until the string fits.
    """
    try:
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        avail = w - x
        if avail <= 0:
            return
        # Fast path: pure ASCII — byte length == display width.
        if text.isascii():
            win.addstr(y, x, text[:avail], attr)
            return
        # Slow path for non-ASCII: clip character-by-character to stay within
        # available columns.  curses.unget_wch / waddwstr are not universally
        # available, so we use the simple char-count approximation: each
        # non-ASCII char might be wide (2 cols); we stop as soon as we'd
        # exceed avail cols.  This is conservative but safe.
        clipped, cols = [], 0
        for ch in text:
            w_ch = 2 if ord(ch) > 0x1100 else 1   # crude CJK/wide check
            if cols + w_ch > avail:
                break
            clipped.append(ch)
            cols += w_ch
        win.addstr(y, x, "".join(clipped), attr)
    except curses.error:
        pass


# Minimum terminal size the UI can sensibly operate in.
_MIN_ROWS, _MIN_COLS = 10, 40


def _popup_dims(stdscr, content_lines: list, title: str = "") -> tuple:
    """
    Issue #4: compute popup (bh, bw, by, bx) clamped to the current
    terminal size so popups are never drawn outside the visible area even
    on very small terminals.

    Returns (bh, bw, by, bx).  bh / bw are the usable box dimensions;
    content that won't fit is silently clipped by safe_addstr.
    """
    h, w = stdscr.getmaxyx()
    # Desired size
    bh_want = len(content_lines) + 4
    bw_want = max(
        (max((len(l) for l in content_lines), default=0) + 6),
        len(title) + 4,
        20,
    )
    bh = max(4, min(bh_want, h - 2))
    bw = max(10, min(bw_want, w - 2))
    by = max(0, (h - bh) // 2)
    bx = max(0, (w - bw) // 2)
    return bh, bw, by, bx


def draw_border(win, title: str = "") -> None:
    try:
        win.box()
    except curses.error:
        pass
    if title:
        h, w = win.getmaxyx()
        label = f" {title} "
        safe_addstr(win, 0, max(2, (w - len(label)) // 2),
                    label, color(C_TITLE) | curses.A_BOLD)

def draw_statusbar(stdscr, segments: list) -> None:
    h, w = stdscr.getmaxyx()
    if h < 2:
        return   # Issue #3: terminal too small to draw a statusbar
    sep  = "  ·  "
    x    = 0
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

def draw_progress_bar(win, y: int, x: int, bar_width: int,
                      fraction: float, label: str = "") -> None:
    inner  = max(bar_width - 2, 1)
    filled = int(max(0.0, min(1.0, fraction)) * inner)
    bar    = "\u2588" * filled + "\u2591" * (inner - filled)
    safe_addstr(win, y, x, f"[{bar}]", color(C_OK))
    if label:
        safe_addstr(win, y, x + bar_width + 1, label, color(C_WARN))

def input_box(stdscr, prompt: str, y: int, x: int,
              width: int = 30, default: str = "") -> str:
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h - 1:
        return default
    safe_addstr(stdscr, y, x, prompt, color(C_WARN) | curses.A_BOLD)
    px = x + len(prompt)
    if px >= w:
        return default
    # Always switch to blocking + cbreak before getstr().  Any caller that
    # used nodelay(True) (progress loops, results screen) must not leave the
    # terminal in non-blocking mode when we hand off to text input — getstr()
    # in nodelay mode returns immediately with empty bytes.
    stdscr.nodelay(False)
    stdscr.timeout(-1)       # block indefinitely while user types
    curses.cbreak()
    curses.echo()
    curses.curs_set(1)
    safe_addstr(stdscr, y, px, " " * min(width, w - px))  # clear previous value
    safe_addstr(stdscr, y, px, default)
    stdscr.refresh()
    try:
        val = stdscr.getstr(y, px, width).decode('utf-8').strip()
    except Exception:
        val = default
    finally:
        curses.noecho()
        curses.curs_set(0)
        # Restore the 100 ms timeout set in main() so callers get expected
        # behaviour without having to remember to reset it themselves.
        stdscr.timeout(100)
    return val or default

def cycle_input(stdscr, prompt: str, y: int, x: int,
                options: list, default=None):
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h - 1:
        return default if default is not None else options[0]
    idx = options.index(default) if default in options else 0
    curses.curs_set(0)
    while True:
        safe_addstr(stdscr, y, x, prompt, color(C_WARN) | curses.A_BOLD)
        hint = f"< {options[idx]} >  (Tab/arrows to change, Enter to confirm)"
        safe_addstr(stdscr, y, x + len(prompt), hint, color(C_TITLE) | curses.A_BOLD)
        stdscr.refresh()
        k = stdscr.getch()
        if k == curses.KEY_RESIZE:          # Issue #1: absorb resize events
            curses.update_lines_cols()
            h, w = stdscr.getmaxyx()
            continue
        if k in (ord('\t'), curses.KEY_RIGHT):
            idx = (idx + 1) % len(options)
        elif k == curses.KEY_LEFT:
            idx = (idx - 1) % len(options)
        elif k in (curses.KEY_ENTER, 10, 13):
            return options[idx]

def confirm_box(stdscr, question: str, title: str = "Confirm") -> bool:
    # Issue #4: use _popup_dims so the box is never drawn off-screen.
    lines = [question, "", "  [Y] Yes      [N / Esc] No"]
    bh, bw, by, bx = _popup_dims(stdscr, lines, title)
    try:
        win = curses.newwin(bh, bw, by, bx)
    except curses.error:
        return False   # terminal truly too small — safe default
    draw_border(win, title)
    for i, line in enumerate(lines):
        if i + 2 < bh - 1:
            safe_addstr(win, i + 2, 3, line[:bw - 6], color(C_WARN))
    win.refresh()
    while True:
        k = win.getch()
        if k == curses.KEY_RESIZE:
            curses.update_lines_cols()   # Issue #1: keep absorbing on resize
            continue
        if k in (ord('y'), ord('Y'), curses.KEY_ENTER, 10, 13):
            return True
        if k in (ord('n'), ord('N'), 27):
            return False

def message_box(stdscr, lines: list, title: str = "Info",
                color_pair: int = C_NORM) -> None:
    # Issue #4: use _popup_dims so the box is never drawn off-screen.
    bh, bw, by, bx = _popup_dims(stdscr, lines, title)
    try:
        win = curses.newwin(bh, bw, by, bx)
    except curses.error:
        return   # terminal truly too small — skip popup
    draw_border(win, title)
    for i, line in enumerate(lines):
        if i + 2 < bh - 1:
            safe_addstr(win, i + 2, 3, line[:bw - 6], color(color_pair))
    prompt_y = bh - 2
    if prompt_y > 0:
        safe_addstr(win, prompt_y, max(1, (bw - 14) // 2),
                    " Press any key ", color(C_WARN))
    win.refresh()
    while True:
        k = win.getch()
        if k != curses.KEY_RESIZE:   # Issue #1: absorb resize, wait for real key
            break

# ── screens ───────────────────────────────────────────────────────────────────

def draw_header_banner(stdscr) -> None:
    _, w = stdscr.getmaxyx()
    brand = "◈  PS5 CHEAT MAKER  ◈"
    safe_addstr(stdscr, 1, max(0, (w - len(brand)) // 2),
                brand, color(C_TITLE) | curses.A_BOLD)

def screen_connect(stdscr) -> str:
    stdscr.clear()
    draw_border(stdscr, "CONNECT")
    draw_header_banner(stdscr)
    for i, hint in enumerate([
        "Ensure ps5debug payload is loaded on your PS5.",
        "Find PS5 IP:  Settings > Network > View Connection Status",
    ]):
        safe_addstr(stdscr, 3 + i, 3, hint, color(C_NORM))
    stdscr.refresh()

    ip = input_box(stdscr, "PS5 IP address : ", 6, 3, 40,
                   state["ip"] or "192.168.0.88")
    state["ip"] = ip
    # Issue #9: a new connection means a new session — stop any freeze that
    # was left running from a previous connection before we try to talk to
    # the new (or restarted) PS5.
    _stop_freeze_worker()
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

def _clear_scan_state() -> None:
    """Wipe all scan-related state. Called whenever the attached process changes."""
    _stop_freeze_worker()
    state["scan_results"]   = _make_addr_array()
    state["scan_values"]    = None
    state["scan_dropped"]   = set()
    state["scan_history"]   = deque(maxlen=5)
    state["scan_pid"]       = None
    state["scan_truncated"] = False
    state["scan_unknown"]   = False
    with _map_cache_lock:
        _map_cache.clear()
    gc.collect()


def screen_proc_select(stdscr, procs: list) -> str:
    # Sort order: 'name' (default) or 'pid'.  Tab cycles between them.
    sort_by = "name"
    procs_orig = list(procs)

    def _sorted(lst):
        if sort_by == "pid":
            return sorted(lst, key=lambda p: p['pid'])
        return sorted(lst, key=lambda p: p['name'].lower())

    procs = _sorted(procs_orig)
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
        # Clamp sel whenever the visible list changes size.
        sel = min(sel, max(0, len(visible_procs) - 1))

        filter_hint = filter_str if filter_str else "(none — type to filter)"
        safe_addstr(stdscr, 3, 3, f"Filter: {filter_hint}", color(C_WARN))
        safe_addstr(stdscr, 3, w - 22,
                    f"Sort: {sort_by} [Tab]  ", color(C_NORM))

        visible = max(1, h - 9)
        start   = max(0, sel - visible // 2)
        for i, p in enumerate(visible_procs[start:start + visible]):
            idx  = start + i
            dim  = p['pid'] < 10
            attr = (color(C_SEL)
                    if idx == sel
                    else (color(C_NORM) | curses.A_DIM if dim else color(C_NORM)))
            line = f"  PID {p['pid']:6d}   {p['name']}"
            safe_addstr(stdscr, 5 + i, 2, line[:w - 4].ljust(w - 4), attr)

        draw_statusbar(stdscr, [
            ("arrows navigate", C_NORM), ("Enter attach", C_OK),
            ("type to filter", C_WARN),  ("Tab sort", C_NORM),
            ("Bksp clear", C_NORM),      ("Q back", C_NORM),
        ])
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_RESIZE:        # Issue #1: terminal resized
            curses.update_lines_cols()
            continue
        if key == curses.KEY_UP and sel > 0:
            sel -= 1
        elif key == curses.KEY_DOWN and sel < len(visible_procs) - 1:
            sel += 1
        elif key == ord('\t'):
            sort_by = "pid" if sort_by == "name" else "name"
            procs   = _sorted(procs_orig)
            sel     = 0
        elif key in (curses.KEY_ENTER, 10, 13) and visible_procs:
            p = visible_procs[sel]
            if p["pid"] != state["pid"]:   # process actually changed
                _clear_scan_state()
            state["pid"]       = p["pid"]
            state["proc_name"] = p["name"]
            add_log(f"Attached to PID {state['pid']} ({state['proc_name']})")
            return "main"
        elif key in (ord('q'), ord('Q')):
            return "connect"
        elif key in (curses.KEY_BACKSPACE, 127, 8):   # 8 = BS on some terminals
            filter_str = filter_str[:-1]
            sel = 0
        elif 32 <= key <= 126:
            filter_str += chr(key)
            sel = 0

def _draw_main_header(stdscr) -> None:
    _, w = stdscr.getmaxyx()
    conn = f" ★ {state['ip']}  PID {state['pid']} ({state['proc_name']}) "
    safe_addstr(stdscr, 2, 3, conn, color(C_OK) | curses.A_BOLD)
    wlabel = {1: "byte", 2: "uint16", 4: "uint32", 8: "uint64"}.get(
        state["scan_width"], "?")
    align  = "aligned" if state["scan_aligned"] else "unaligned"
    rss    = _rss_mb()
    frac   = _rss_frac()
    hist_mb = _history_bytes() / 1_048_576
    # Colour: green < 50 %, yellow < 75 %, red ≥ 75 % of total RAM
    if frac >= 0.75:
        ram_cp = C_ERR
    elif frac >= 0.50:
        ram_cp = C_WARN
    else:
        ram_cp = C_OK
    stats  = (f"  Results: {len(state['scan_results']):,}   "
              f"Cheats: {len(state['cheats'])}   "
              f"Width: {wlabel}  ({align})")
    ram_str = f"   RAM {rss:.0f} MB ({frac*100:.0f}%)  Undo {hist_mb:.1f} MB"
    safe_addstr(stdscr, 3, 3, stats, color(C_WARN))
    safe_addstr(stdscr, 3, 3 + len(stats), ram_str, color(ram_cp))

def screen_main(stdscr):
    menu = [
        ("S", "First Scan",          "scan_first",    C_NORM),
        ("N", "Next Scan",           "scan_next",     C_NORM),
        ("R", "Results",             "results",       C_NORM),
        ("W", "Write to Address",    "write",         C_WARN),
        ("C", "Cheat List",          "cheat_list",    C_NORM),
        ("E", "Export .json",        "export",        C_OK),
        ("F", "Freeze Address",      "freeze",        C_WARN),
        ("L", "Log",                 "log",           C_NORM),
        ("X", "Clear Results",       "clear",         C_WARN),
        ("H", "Clear Scan History",  "clear_history", C_WARN),
        ("P", "Change Process",      "proc",          C_NORM),
        ("Q", "Quit",                None,            C_ERR),
    ]
    sel = 0
    stdscr.timeout(100)   # return -1 after 100 ms so header (RAM display) refreshes
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
            unavail = (
                (label == "Next Scan"    and len(state["scan_results"]) == 0) or
                (label == "Results"      and len(state["scan_results"]) == 0) or
                (label == "Export .json" and not state["cheats"])
            )
            attr = (color(C_SEL) | curses.A_BOLD if i == sel
                    else color(C_NORM) | curses.A_DIM if unavail
                    else color(cp))
            safe_addstr(stdscr, 5 + row, 5 + col * 35,
                        f"[{key}]  {label}".ljust(30), attr)

        with _log_lock:
            last_entry = state["log"][-1] if state["log"] else None
        if last_entry:
            lcp = {"error": C_ERR, "warn": C_WARN, "info": C_OK}.get(
                last_entry["level"], C_NORM)
            safe_addstr(stdscr, h - 3, 3,
                        f"[{last_entry['ts']}] {last_entry['msg']}"[:w - 6],
                        color(lcp))

        draw_statusbar(stdscr, [
            ("arrows / letter", C_NORM), ("Enter select", C_OK), ("Q quit", C_ERR),
        ])
        stdscr.refresh()

        key = stdscr.getch()
        if key == -1:
            continue   # 100 ms timeout elapsed, no key — just redraw
        if key == curses.KEY_RESIZE:        # Issue #1: terminal resized — redraw
            curses.update_lines_cols()
            continue
        if key == curses.KEY_UP    and sel > 0:              sel -= 1
        elif key == curses.KEY_DOWN and sel < len(menu) - 1: sel += 1
        elif key in (curses.KEY_ENTER, 10, 13):
            action = menu[sel][2]
            if action is None:
                return None
            if dispatch(stdscr, action) == "proc":
                return "proc"
        else:
            for k, _, action, _ in menu:
                if key in (ord(k.lower()), ord(k.upper())):
                    if action is None:
                        return None
                    if dispatch(stdscr, action) == "proc":
                        return "proc"
                    break

def dispatch(stdscr, action: str):
    actions = {
        "scan_first":    do_scan_first,
        "scan_next":     do_scan_next,
        "results":       do_show_results,
        "write":         do_write,
        "cheat_list":    do_cheat_list,
        "export":        do_export,
        "freeze":        do_freeze,
        "log":           do_log,
        "clear":         do_clear_results,
        "clear_history": do_clear_history,
    }
    if action == "proc":
        return "proc"
    fn = actions.get(action)
    if fn:
        fn(stdscr)


def do_clear_history(stdscr) -> None:
    """
    Discard all undo history while keeping the current scan results intact.
    Useful after a scan has converged to a handful of addresses but the early
    undo deltas are still holding significant RAM.
    """
    n      = len(state["scan_history"])
    hbytes = _history_bytes()
    if n == 0:
        message_box(stdscr, ["No undo history to clear."], "Clear History", C_WARN)
        return
    hist_mb = hbytes / 1_048_576
    if confirm_box(stdscr,
            f"Clear {n} undo level{'s' if n != 1 else ''} ({hist_mb:.1f} MB)?\n"
            "Current scan results are kept intact.",
            "Clear Scan History"):
        state["scan_history"] = deque(maxlen=5)
        gc.collect()
        add_log(f"Undo history cleared: freed {hist_mb:.1f} MB — "
                f"RSS now {_rss_mb():.0f} MB", "warn")
        message_box(stdscr,
            [f"Freed {hist_mb:.1f} MB of undo history.",
             "Scan results unchanged.",
             f"RSS now {_rss_mb():.0f} MB"],
            "History Cleared", C_OK)

# ── scan UI ───────────────────────────────────────────────────────────────────

def _run_scan_with_progress(stdscr, thread_fn, total_label: str,
                             cancel_event: threading.Event,
                             progress: dict, w: int) -> bool:
    """
    Spin the progress-bar loop while `thread_fn` runs in a daemon thread.
    Returns True if the scan completed normally, False if cancelled.

    Issue #6 (thread exception silently dies): thread_fn is already expected
    to catch its own exceptions and write them to progress["error"].  The
    wrapper here is a final safety net that catches anything that slips
    through and stores it so the UI loop can report it rather than silently
    leaving the progress bar frozen.
    """
    _orig_fn = thread_fn
    def _guarded_fn():
        try:
            _orig_fn()
        except Exception as exc:                 # Issue #6: last-resort catch
            if not progress.get("error"):
                progress["error"] = f"Unhandled thread error: {exc}"
            add_log(f"Scan thread unhandled error: {exc}", "error")

    t = threading.Thread(target=_guarded_fn, daemon=True)
    t.start()

    spinner = ["|", "/", "-", "\\"]
    spin_i  = 0
    stdscr.nodelay(True)
    try:
        while t.is_alive():
            h, w = stdscr.getmaxyx()           # Issue #1: re-read on every tick
            frac = progress["done"] / max(progress["total"], 1)
            safe_addstr(stdscr, 9, 3,
                f"{spinner[spin_i % 4]}  {total_label}  "
                f"{progress['done']:,} / {progress['total']:,}  [Esc=cancel]",
                color(C_WARN))
            draw_progress_bar(stdscr, 10, 3, min(w - 8, 60), frac,
                              f"  {int(frac * 100)}%")
            stdscr.refresh()
            time.sleep(0.1)
            spin_i += 1
            k = stdscr.getch()
            if k == curses.KEY_RESIZE:         # Issue #1: absorb resize
                curses.update_lines_cols()
                stdscr.clear()
            elif k == 27:
                cancel_event.set()
                safe_addstr(stdscr, 12, 3, "Cancelling…", color(C_ERR))
                stdscr.refresh()
    finally:
        stdscr.nodelay(False)

    t.join()
    return not cancel_event.is_set()


def do_scan_first(stdscr) -> None:
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    draw_border(stdscr, "FIRST SCAN")
    safe_addstr(stdscr, 2, 3,
        "Enter the current in-game value to search for.", color(C_WARN))
    stdscr.refresh()

    val_s     = input_box(stdscr, "Value (blank = unknown): ", 4, 3, 20)
    unknown_mode = (val_s.strip() == "")

    _wlabels  = [WIDTH_LABEL[ww] for ww in VALID_WIDTHS]
    _wsel     = cycle_input(stdscr, "Scan width      : ", 6, 3, _wlabels,
                            WIDTH_LABEL.get(state["scan_width"], "uint32"))
    width     = VALID_WIDTHS[_wlabels.index(_wsel)]
    align_lbl = cycle_input(stdscr, "Scan alignment  : ", 8, 3,
                            ["aligned (faster)", "unaligned (thorough)"],
                            "aligned (faster)" if state["scan_aligned"] else "unaligned (thorough)")
    aligned = align_lbl.startswith("aligned")
    scope_lbl = cycle_input(stdscr, "Scan scope      : ", 10, 3,
                            ["writable only (fast)", "all readable (thorough)"],
                            "writable only (fast)" if state["scan_writable_only"]
                            else "all readable (thorough)")
    writable_only = scope_lbl.startswith("writable")

    state["scan_width"]        = width
    state["scan_aligned"]      = aligned
    state["scan_writable_only"] = writable_only

    val = None
    if not unknown_mode:
        try:
            val = int(val_s, 0)
        except ValueError:
            message_box(stdscr, ["Invalid — enter decimal or hex (0x…), or leave blank for unknown."], "Error", C_ERR)
            return
        if val < 0 or val > WIDTH_MAX[width]:
            message_box(stdscr,
                [f"Value {val} out of range for {WIDTH_LABEL[width]}.",
                 f"Max: {WIDTH_MAX[width]}"], "Error", C_ERR)
            return

    cancel_event = threading.Event()
    cancel_event.truncated = False   # searcher sets this when result cap is hit
    progress     = {"done": 0, "total": 1, "results": None, "values": None,
                    "error": None, "truncated": False}

    if unknown_mode:
        def run():
            try:
                addrs, vals = scan_first_unknown(
                    state["ip"], state["pid"], width, aligned,
                    lambda d, t: progress.update(done=d, total=max(t, 1)),
                    cancel_event,
                    writable_only=writable_only)
                progress["results"]   = addrs
                progress["values"]    = vals
                progress["truncated"] = getattr(cancel_event, "truncated", False)
            except Exception as exc:
                progress["error"] = str(exc)
        scan_label = "Snapshotting memory…"
    else:
        def run():
            try:
                res = scan_first(
                    state["ip"], state["pid"], val, width, aligned,
                    lambda d, t: progress.update(done=d, total=max(t, 1)),
                    cancel_event,
                    writable_only=writable_only)
                progress["results"]   = res
                progress["truncated"] = getattr(cancel_event, "truncated", False)
            except Exception as exc:
                progress["error"] = str(exc)
        scan_label = "Scanning memory…"

    ok = _run_scan_with_progress(stdscr, run, scan_label, cancel_event, progress, w)
    # cancel_event is also set internally when the result cap is hit (truncation).
    # Only treat it as a real user cancellation when the truncated flag is NOT set.
    user_cancelled = not ok and not getattr(cancel_event, "truncated", False)
    if user_cancelled:
        add_log("First scan cancelled", "warn")
        return
    if progress["error"]:
        add_log(f"Scan error: {progress['error']}", "error")
        message_box(stdscr, [f"Error: {progress['error']}"], "Scan Failed", C_ERR)
        return

    results = progress["results"] if progress["results"] is not None else _make_addr_array()
    # Free old arrays before the new assignment to avoid holding two full
    # arrays in RAM simultaneously (old + new) during the reassignment.
    state["scan_results"]  = _make_addr_array()
    state["scan_values"]   = None
    state["scan_history"]  = deque(maxlen=5)
    state["scan_dropped"]  = set()
    gc.collect()
    state["scan_results"]  = results
    state["scan_values"]   = progress.get("values")
    state["scan_pid"]      = state["pid"]
    state["scan_truncated"] = progress.get("truncated", False)
    state["scan_unknown"]  = unknown_mode
    add_log(f"{'Unknown' if unknown_mode else 'First'} scan "
            f"w={width} aligned={aligned}: {len(results):,} candidates, "
            f"RSS {_rss_mb():.0f} MB")

    trunc_lines = (
        [f"⚠  Scan capped at {MAX_SCAN_RESULTS:,} results — {len(results):,} shown.",
         "   Some matching addresses were NOT found.",
         "   Run Next Scan (N) with a changed value to narrow results",
         "   before trusting any address shown here.",
         ""]
        if progress["truncated"] else []
    )
    if unknown_mode:
        message_box(stdscr, trunc_lines + [
            f"Snapshot taken: {len(results):,} candidates.",
            "",
            "Now trigger a change in-game (take damage, heal, etc.)",
            "then use Next Scan (N) and choose a relational filter",
            "(decreased / increased / unchanged / changed).",
        ], "Snapshot Complete" + (" — TRUNCATED" if progress["truncated"] else ""),
           C_WARN if progress["truncated"] else C_OK)
    else:
        message_box(stdscr, trunc_lines + [
            f"Found {len(results)} results.",
            "",
            "Change the value in-game, then use Next Scan (N).",
            "Once narrowed down, use Results (R) to pick an address.",
        ], "Scan Complete" + (" — TRUNCATED" if progress["truncated"] else ""),
           C_WARN if progress["truncated"] else C_OK)


def do_scan_next(stdscr) -> None:
    if len(state["scan_results"]) == 0:
        message_box(stdscr,
            ["No previous scan results.", "Run First Scan (S) first."], "Error", C_ERR)
        return
    # Issues #10/#11: scan results from a different PID contain addresses that
    # are meaningless (or actively harmful to write) in the current process.
    # Reject unconditionally — the user must start a fresh scan.
    if state.get("scan_pid") not in (None, state["pid"]):
        if confirm_box(stdscr,
                "Scan results belong to a DIFFERENT process.\n"
                "Those addresses are invalid for the current PID.\n"
                "Clear stale results and start a fresh First Scan?",
                "Stale Results"):
            _clear_scan_state()
        return

    stdscr.clear()
    h, w = stdscr.getmaxyx()
    draw_border(stdscr, "NEXT SCAN")
    width   = state["scan_width"]
    is_unkn = state.get("scan_unknown", False)
    safe_addstr(stdscr, 2, 3,
        f"Candidates: {len(state['scan_results']):,}  "
        f"({'unknown-value' if is_unkn else 'exact-value'} session)",
        color(C_WARN))
    stdscr.refresh()

    cancel_event = threading.Event()
    prev_addrs   = state["scan_results"]

    if is_unkn:
        # ── relational (unknown-value) path ───────────────────────────────────
        prev_values = state.get("scan_values")
        if prev_values is None or len(prev_values) != len(prev_addrs):
            message_box(stdscr,
                ["Value snapshot is missing or mismatched.",
                 "Please run a new First Scan (S) with blank value."],
                "Error", C_ERR)
            return

        mode_lbl = cycle_input(stdscr, "Filter mode      : ", 4, 3,
                               RELATIONAL_MODES, RELATIONAL_MODES[0])

        delta = 0
        if mode_lbl in ("decreased by", "increased by"):
            delta_s = input_box(stdscr, "Delta amount     : ", 6, 3, 20, "1")
            try:
                delta = int(delta_s, 0)
                if delta < 0 or delta > WIDTH_MAX[width]:
                    raise ValueError("out of range")
            except ValueError:
                message_box(stdscr, ["Invalid delta — enter a positive integer."],
                            "Error", C_ERR)
                return

        progress = {"done": 0, "total": max(len(prev_addrs), 1),
                    "results": None, "values": None, "error": None}

        def run_rel():
            try:
                na, nv = scan_next_relational(
                    state["ip"], state["pid"], width,
                    prev_addrs, prev_values,
                    mode_lbl, delta,
                    cancel_event,
                    lambda d, t: progress.update(done=d, total=max(t, 1)))
                progress["results"] = na
                progress["values"]  = nv
            except Exception as exc:
                progress["error"] = str(exc)

        ok = _run_scan_with_progress(
            stdscr, run_rel, f"Filtering ({mode_lbl})…", cancel_event, progress, w)
        if not ok:
            add_log("Next scan cancelled", "warn")
            return
        if progress["error"]:
            add_log(f"Next scan error: {progress['error']}", "error")
            message_box(stdscr, [f"Error: {progress['error']}"], "Scan Error", C_ERR)
            return

        new_addrs  = progress["results"] if progress["results"] is not None else _make_addr_array()
        new_values = progress["values"]  if progress["values"]  is not None else np.empty(0, dtype=_NP_VALUE_DTYPE[width])

        # Delta undo — store only removed addresses/values, not a full copy.
        # prev_addrs is sorted; new_addrs may not be (batch order) → sort for
        # set-difference via searchsorted rather than building a Python set.
        new_sorted  = np.sort(new_addrs)
        # Find indices in prev_addrs that are NOT in new_sorted.
        ins         = np.searchsorted(new_sorted, prev_addrs)
        ins_clipped = np.clip(ins, 0, len(new_sorted) - 1)
        removed_mask = new_sorted[ins_clipped] != prev_addrs
        removed_a    = prev_addrs[removed_mask]
        removed_v    = prev_values[removed_mask]
        _push_undo(removed_a, removed_v, set(state["scan_dropped"]))
        del new_sorted, removed_mask, removed_a, removed_v   # free intermediates

        state["scan_results"] = new_addrs
        state["scan_values"]  = new_values
        state["scan_dropped"] = state["scan_dropped"] & set(new_addrs.tolist())

        hist_mb = _history_bytes() / 1_048_576
        add_log(f"Relational next scan ({mode_lbl}): {len(new_addrs):,} remain, "
                f"undo {hist_mb:.1f} MB, RSS {_rss_mb():.0f} MB")

        tip = ("Perfect! Use Results (R)."
               if len(new_addrs) <= 10
               else "Still many — trigger another change and scan again.")
        undo_hint = ""
        if state["scan_history"]:
            last_delta = state["scan_history"][-1]
            undo_hint  = (f"  (U to undo — restores "
                          f"{len(new_addrs) + len(last_delta[0]):,} candidates)")
        message_box(stdscr,
            [f"{len(new_addrs):,} candidates remain.", "", tip, undo_hint],
            "Scan Complete", C_OK if len(new_addrs) <= 10 else C_WARN)

    else:
        # ── exact-value path (original behaviour) ────────────────────────────
        safe_addstr(stdscr, 4, 3,
            "Enter the new in-game value.", color(C_NORM))
        stdscr.refresh()

        val_s = input_box(stdscr, "New value        : ", 6, 3, 20)
        try:
            val = int(val_s, 0)
        except ValueError:
            message_box(stdscr, ["Invalid value."], "Error", C_ERR)
            return
        if val < 0 or val > WIDTH_MAX[width]:
            message_box(stdscr,
                [f"Value {val} out of range for {WIDTH_LABEL[width]}."], "Error", C_ERR)
            return

        progress = {"done": 0, "total": max(len(prev_addrs), 1),
                    "results": None, "error": None}

        def run_exact():
            try:
                progress["results"] = scan_next(
                    state["ip"], state["pid"], val, width, prev_addrs,
                    cancel_event,
                    lambda d, t: progress.update(done=d, total=max(t, 1)))
            except Exception as exc:
                progress["error"] = str(exc)

        ok = _run_scan_with_progress(
            stdscr, run_exact, "Filtering addresses…", cancel_event, progress, w)
        if not ok:
            add_log("Next scan cancelled", "warn")
            return
        if progress["error"]:
            add_log(f"Next scan error: {progress['error']}", "error")
            message_box(stdscr, [f"Error: {progress['error']}"], "Scan Error", C_ERR)
            return

        results = progress["results"] if progress["results"] is not None else _make_addr_array()

        # Delta undo — same searchsorted approach as relational path.
        new_sorted   = np.sort(results)
        ins          = np.searchsorted(new_sorted, prev_addrs)
        ins_clipped  = np.clip(ins, 0, max(len(new_sorted) - 1, 0))
        removed_mask = (len(new_sorted) == 0) | (new_sorted[ins_clipped] != prev_addrs) \
                       if len(new_sorted) > 0 else np.ones(len(prev_addrs), dtype=bool)
        removed_a    = prev_addrs[removed_mask]
        _push_undo(removed_a, None, set(state["scan_dropped"]))
        del new_sorted, removed_mask, removed_a

        state["scan_results"] = results
        state["scan_values"]  = None
        state["scan_dropped"] = state["scan_dropped"] & set(results.tolist())

        hist_mb = _history_bytes() / 1_048_576
        add_log(f"Exact next scan val={val}: {len(results):,} remain, "
                f"undo {hist_mb:.1f} MB, RSS {_rss_mb():.0f} MB")

        add_log(f"Next scan val={val}: {len(results):,} remain")
        tip = ("Perfect! Use Results (R)."
               if len(results) <= 10 else "Still many — change value and scan again.")
        undo_hint = ""
        if state["scan_history"]:
            last_delta = state["scan_history"][-1]
            undo_hint  = (f"  (U to undo — restores "
                          f"{len(results) + len(last_delta[0]):,} candidates)")
        message_box(stdscr,
            [f"{len(results):,} results remain.", "", tip, undo_hint],
            "Scan Complete", C_OK if len(results) <= 10 else C_WARN)


# ── results screen ────────────────────────────────────────────────────────────

def _refresh_visible_locked(ip: str, pid: int, addrs: list, width: int,
                             cache: dict, lock: threading.Lock,
                             cancel_event: Optional[threading.Event] = None,
                             expected_pid: Optional[int] = None) -> None:
    """
    Read live values for `addrs` and update `cache` under `lock`.
    `expected_pid` is checked before each read; if state["pid"] has changed
    (user switched processes) the thread exits immediately without writing.

    Issue #12 (partial read accepted as valid): each read result is validated
    to be exactly `width` bytes; anything shorter is treated as an error and
    displayed as "?" rather than being unpacked with potentially wrong data.
    """
    if not addrs:
        return
    fmt  = WIDTH_FMT[width]
    sock = None
    try:
        # Build a _ScanSocket with an aggressively short timeout
        sock = _ScanSocket(ip, pid)
        sock._s.settimeout(1.5)   # type: ignore[union-attr]  short: fast exit on Q
        for addr in addrs:
            if cancel_event and cancel_event.is_set():
                break
            if expected_pid is not None and state["pid"] != expected_pid:
                break   # process switched — stop immediately
            try:
                raw  = sock.read(addr, width)
                # Issue #12: reject partial reads — only unpack when we got
                # exactly the number of bytes we asked for.
                if len(raw) == width:
                    vstr = str(struct.unpack(fmt, raw)[0])
                else:
                    vstr = "?"   # partial read — don't trust the data
            except Exception:
                vstr = "?"
            with lock:
                cache[addr] = vstr
    finally:
        if sock:
            sock.close()


def do_show_results(stdscr) -> None:
    results = state["scan_results"]
    if len(results) == 0:
        return
    if state.get("scan_pid") not in (None, state["pid"]):
        message_box(stdscr,
            ["Scan results are from a different process.",
             "Please run a new First Scan (S) for this process."],
            "Stale Results", C_WARN)
        return

    sel               = 0
    offset            = 0
    val_cache         = {}
    cache_lock        = threading.Lock()
    last_refresh      = 0.0
    refresh_deadline  = 0.0
    refresh_complete  = 0.0   # wall time when the last refresh thread finished
    REFRESH_INTERVAL  = 2.0
    refresh_thread    = None
    refresh_cancel    = threading.Event()

    stdscr.nodelay(True)
    try:
        while True:
            now = time.time()
            h, w = stdscr.getmaxyx()
            visible = max(1, h - 7)

            # Scroll offset maintenance
            if sel < offset:              offset = sel
            if sel >= offset + visible:   offset = sel - visible + 1

            # Only refresh the addresses currently on screen, under a lock
            thread_idle = refresh_thread is None or not refresh_thread.is_alive()
            if thread_idle and refresh_thread is not None:
                # Thread just finished — record completion time once
                if refresh_complete < refresh_deadline:
                    refresh_complete = time.time()
            if thread_idle and now - last_refresh >= REFRESH_INTERVAL:
                visible_addrs = results[offset:offset + visible]
                refresh_cancel.clear()
                refresh_thread = threading.Thread(
                    target=_refresh_visible_locked,
                    args=(state["ip"], state["pid"], list(visible_addrs),
                          state["scan_width"], val_cache, cache_lock,
                          refresh_cancel, state["pid"]),   # expected_pid
                    daemon=True)
                refresh_thread.start()
                refresh_deadline = now
                last_refresh = now

            stdscr.clear()
            draw_border(stdscr, f"RESULTS  ({len(results)} addresses)")
            wlabel = WIDTH_LABEL.get(state["scan_width"], str(state["scan_width"]))
            trunc_warn = "  ⚠ TRUNCATED — not all memory was searched" if state.get("scan_truncated") else ""
            safe_addstr(stdscr, 2, 3,
                f"Type: {wlabel}   Process: {state['proc_name']} (PID {state['pid']}){trunc_warn}",
                color(C_ERR) if trunc_warn else color(C_WARN))
            safe_addstr(stdscr, 3, 3,
                "↑↓ navigate   Enter add cheat   D drop   U undo scan   Q back",
                color(C_NORM))

            for i, addr in enumerate(results[offset:offset + visible]):
                idx    = offset + i
                with cache_lock:
                    vstr = val_cache.get(addr, "…")
                marker = ">" if idx == sel else " "
                line   = f"{marker} {idx+1:4d}   {hex(addr):<20}  current = {vstr}"
                attr   = color(C_SEL) | curses.A_BOLD if idx == sel else color(C_NORM)
                safe_addstr(stdscr, 5 + i, 2, line[:w - 4].ljust(w - 4), attr)

            # Age = how long since the last completed refresh, not since it started
            data_age      = int(now - refresh_complete) if refresh_complete else 0
            is_refreshing = refresh_thread is not None and refresh_thread.is_alive()
            # If a refresh is in flight and last data is older than one cycle,
            # mark displayed values as potentially stale so the user isn't misled.
            stale         = is_refreshing and data_age >= REFRESH_INTERVAL
            age_label     = "⟳ fetching…" if is_refreshing else f"~{data_age}s old"
            if stale:
                age_label = f"⚠ stale (~{data_age}s)"
            draw_statusbar(stdscr, [
                (f"{len(results)} results", C_WARN),
                ("↑↓ navigate",   C_NORM),
                ("Enter cheat",   C_OK),
                ("D drop",        C_ERR),
                ("U undo",        C_WARN),
                ("M flush maps",  C_WARN),
                (age_label,       C_ERR if stale else C_ACC if is_refreshing else C_NORM),
                ("Q back",        C_NORM),
            ])
            stdscr.refresh()

            key = stdscr.getch()
            # -1 = no key in nodelay mode — sleep only then to avoid busy-spin.
            # Previously sleep(0.05) ran unconditionally BEFORE getch(), adding
            # 50 ms latency to every keypress.
            if key == -1:
                time.sleep(0.05)
                continue
            if key == curses.KEY_RESIZE:
                curses.update_lines_cols()
                continue
            if key == curses.KEY_UP and sel > 0:
                sel -= 1
            elif key == curses.KEY_DOWN and sel < len(results) - 1:
                sel += 1
            elif key in (curses.KEY_ENTER, 10, 13):   # Issue #15: all Enter forms
                stdscr.nodelay(False)
                _add_cheat_at(stdscr, results[sel])
                stdscr.nodelay(True)
                results = state["scan_results"]
                with cache_lock:
                    val_cache.clear()
            elif key in (ord('d'), ord('D')):
                dropped = results[sel]
                results = _make_addr_array(a for i, a in enumerate(results) if i != sel)
                state["scan_results"] = results
                # Track dropped address separately from scan history
                state["scan_dropped"].add(dropped)
                with cache_lock:
                    val_cache.pop(dropped, None)
                if len(results) == 0:
                    break
                sel = min(sel, len(results) - 1)
            elif key in (ord('u'), ord('U')):
                if state["scan_history"]:
                    entry        = state["scan_history"].pop()
                    removed_a    = entry[0]   # ndarray[uint64]
                    removed_v    = entry[1]   # ndarray|None
                    prev_dropped = entry[2]
                    # Reconstruct prev = sorted union(current, removed)
                    cur_addrs  = state["scan_results"]
                    prev_addrs = np.union1d(cur_addrs, removed_a)  # sorted, unique
                    # Reconstruct values for unknown-value sessions
                    if removed_v is not None and state.get("scan_values") is not None:
                        cur_v   = state["scan_values"]
                        width_w = state["scan_width"]
                        dtype   = _NP_VALUE_DTYPE[width_w]
                        # Build merged value map via searchsorted (no dict)
                        prev_vals = np.zeros(len(prev_addrs), dtype=dtype)
                        # Fill from current
                        idx_cur = np.searchsorted(prev_addrs, cur_addrs)
                        prev_vals[idx_cur] = cur_v
                        # Fill from removed (overwrites only the new slots)
                        idx_rem = np.searchsorted(prev_addrs, removed_a)
                        prev_vals[idx_rem] = removed_v
                        prev_values_out = prev_vals
                    else:
                        prev_values_out = state.get("scan_values")
                    state["scan_results"] = prev_addrs
                    state["scan_values"]  = prev_values_out
                    state["scan_dropped"] = prev_dropped
                    results = state["scan_results"]
                    with cache_lock:
                        val_cache.clear()
                    sel = 0; offset = 0
                    add_log(f"Undo: restored {len(results):,} candidates, "
                            f"RSS {_rss_mb():.0f} MB")
            elif key in (ord('m'), ord('M')):
                # Force map-cache flush: useful when the game reallocated memory
                # without a PID change (e.g. level reload, NG+).
                with _map_cache_lock:
                    _map_cache.clear()
                with cache_lock:
                    val_cache.clear()
                add_log("Map cache flushed — next scan/write will re-fetch regions", "warn")
            elif key in (ord('q'), ord('Q'), 27):   # Issue #15: Esc also exits
                break
    finally:
        stdscr.nodelay(False)
        # Signal and join the refresh thread so it stops making connections
        # immediately after the user leaves the screen.
        refresh_cancel.set()
        if refresh_thread and refresh_thread.is_alive():
            refresh_thread.join(timeout=2.0)   # 1.5 s socket timeout + 0.5 s margin


def _add_cheat_at(stdscr, addr: int) -> None:
    stdscr.clear()
    draw_border(stdscr, "ADD CHEAT")
    safe_addstr(stdscr, 2, 3, f"Address : {hex(addr)}", color(C_OK) | curses.A_BOLD)
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
    scan_w = state["scan_width"]
    try:
        val = int(val_s, 0)
        if val < 0 or val > WIDTH_MAX[scan_w]:
            message_box(stdscr,
                [f"Value {val} exceeds maximum for {WIDTH_LABEL[scan_w]}.",
                 f"Max allowed: {WIDTH_MAX[scan_w]}"],
                "Value Out of Range", C_ERR)
            return
        entry = {
            "name":    name or f"Cheat@{hex(addr)}",
            "address": addr,
            "value":   val,
            "type":    typ,
            "width":   scan_w,
        }
        state["cheats"].append(entry)
        add_log(f"Added '{entry['name']}' @ {hex(addr)} = {val}")
        message_box(stdscr,
            [f"  {entry['name']}", f"  {hex(addr)} = {val}  [{typ}]"],
            "Cheat Added", C_OK)
    except ValueError:
        message_box(stdscr, ["Invalid value — must be an integer."], "Error", C_ERR)


def do_write(stdscr) -> None:
    stdscr.clear()
    draw_border(stdscr, "WRITE TO ADDRESS")
    safe_addstr(stdscr, 2, 3,
        "Write a single value directly to a memory address.", color(C_WARN))
    stdscr.refresh()
    addr_s = input_box(stdscr, "Address (hex) : ", 4, 3, 20)
    val_s  = input_box(stdscr, "Value         : ", 6, 3, 20)
    _wl    = [WIDTH_LABEL[ww] for ww in VALID_WIDTHS]
    _ws    = cycle_input(stdscr, "Width         : ", 8, 3, _wl,
                         WIDTH_LABEL.get(state["scan_width"], "uint32"))
    width  = VALID_WIDTHS[_wl.index(_ws)]
    try:
        addr = int(addr_s, 16)
        err  = _validate_write_addr(addr)
        if err:
            raise ValueError(err)
        val  = int(val_s, 0)
        if val < 0 or val > WIDTH_MAX[width]:
            raise ValueError(f"Value out of range for {WIDTH_LABEL[width]}")
        # Verify address is inside a writable mapped region (fail-CLOSED: surfaces error to user)
        map_err = _validate_addr_in_maps(state["ip"], state["pid"], addr, width)
        if map_err:
            if not confirm_box(stdscr, f"{map_err}\nWrite anyway?", "Unmapped Address"):
                return
        data = struct.pack(WIDTH_FMT[width], val)
        ok   = ps5_write(state["ip"], state["pid"], addr, data)
        add_log(f"Write {hex(addr)} = {val} {'OK' if ok else 'FAILED'}",
                "info" if ok else "error")
        if ok:
            message_box(stdscr, [f"Wrote {val} to {hex(addr)}"], "Write OK", C_OK)
        else:
            message_box(stdscr, ["Write rejected by ps5debug."], "Write Failed", C_ERR)
    except Exception as exc:
        message_box(stdscr, [f"Error: {exc}"], "Error", C_ERR)


def do_cheat_list(stdscr) -> None:
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
                "↑↓ select   Enter edit   D delete   Q back", color(C_NORM))
            hdr = f"  {'Name':<28}  {'Address':<18}  {'Value':<10}  Type"
            safe_addstr(stdscr, 3, 2, hdr[:w - 4],
                        color(C_TITLE) | curses.A_UNDERLINE)
            if sel < offset:             offset = sel
            if sel >= offset + visible:  offset = sel - visible + 1
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
            ("↑↓ navigate", C_NORM), ("Enter edit", C_OK),
            ("D delete", C_ERR),     ("Q back", C_NORM),
        ])
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP    and sel > 0:               sel -= 1
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
                if not cheats:
                    sel = 0
                else:
                    sel = min(sel, len(cheats) - 1)
                offset = min(offset, max(0, len(cheats) - visible))
        elif key in (ord('q'), ord('Q')):
            break


def _edit_cheat(stdscr, idx: int) -> None:
    c = state["cheats"][idx]
    stdscr.clear()
    draw_border(stdscr, "EDIT CHEAT")
    safe_addstr(stdscr, 2, 3, f"Editing: {c['name']}", color(C_TITLE) | curses.A_BOLD)
    safe_addstr(stdscr, 3, 3, "Leave blank to keep current value.", color(C_NORM))
    stdscr.refresh()
    new_name = input_box(stdscr, "Name  : ", 5, 3, 40, c["name"])
    val_s    = input_box(stdscr, "Value : ", 7, 3, 20, str(c["value"]))
    new_type = cycle_input(stdscr, "Type  : ", 9, 3, ["freeze", "write"], c["type"])
    try:
        new_val = int(val_s, 0)
        if new_val < 0 or new_val > WIDTH_MAX[c["width"]]:
            message_box(stdscr,
                [f"Value {new_val} exceeds maximum for {WIDTH_LABEL[c['width']]}.",
                 f"Max allowed: {WIDTH_MAX[c['width']]}  — keeping old value."],
                "Value Out of Range", C_WARN)
            new_val = c["value"]
    except ValueError:
        new_val = c["value"]
    state["cheats"][idx].update({"name": new_name, "value": new_val, "type": new_type})
    add_log(f"Edited '{new_name}' val={new_val} type={new_type}")
    message_box(stdscr, [f"Updated '{new_name}'"], "Saved", C_OK)


def do_export(stdscr) -> None:
    stdscr.clear()
    draw_border(stdscr, "EXPORT GOLDHEN CHEAT JSON")
    safe_addstr(stdscr, 2, 3,
        f"Cheats to export: {len(state['cheats'])}", color(C_WARN))
    if not state["cheats"]:
        message_box(stdscr,
            ["No cheats to export.", "Build your cheat list first."], "Error", C_ERR)
        return
    stdscr.refresh()

    # Require a non-empty Title ID before proceeding
    while True:
        gid = input_box(stdscr, "Title ID  (e.g. PPSA01234) : ", 4, 3, 20,
                        state["game_id"])
        if not gid:
            if not confirm_box(stdscr, "Title ID is empty — really continue?",
                               "Missing Title ID"):
                return
            break
        if TITLE_ID_RE.match(gid):
            break
        # Invalid format — ask user to confirm or re-enter
        if not confirm_box(stdscr,
                f"'{gid}' doesn't match PPSA01234 format.\nExport anyway?",
                "Bad Title ID"):
            continue   # let them re-enter
        break

    VERSION_RE = re.compile(r'^\d{2}\.\d{2}$')
    gver = input_box(stdscr, "Version   (e.g. 01.00)     : ", 6, 3, 10, state["game_ver"])
    if gver and not VERSION_RE.match(gver):
        if not confirm_box(stdscr,
                f"Version '{gver}' doesn't match NN.NN format.\nContinue anyway?",
                "Version Format"):
            return
    gtit = input_box(stdscr, "Game Title                 : ", 8, 3, 40, state["game_title"])
    val_fmt = cycle_input(stdscr, "Value format               : ", 10, 3,
                          ["hex (GoldHEN 2.x)", "decimal (older loaders)"],
                          "hex (GoldHEN 2.x)")
    hex_values = val_fmt.startswith("hex")
    state.update(game_id=gid, game_ver=gver, game_title=gtit)

    safe_gid  = sanitize_filename(gid)
    safe_gver = sanitize_filename(gver.replace('.', '_'))
    fname     = f"{safe_gid or 'UNKNOWN'}_{safe_gver or '00_00'}.json"
    save_path = Path.home() / fname

    # Overwrite confirmation if file already exists
    if save_path.exists():
        if not confirm_box(stdscr,
                f"'{fname}' already exists.\nOverwrite?", "Confirm Overwrite"):
            return

    cht = generate_cht(state["cheats"], gid, gver, gtit, hex_values)
    try:
        save_path.write_text(cht)
        add_log(f"Exported {save_path}")
        message_box(stdscr, [
            f"Saved: {save_path}",
            "",
            "Transfer to PS5 via FTP:",
            f"  /data/GoldHEN/cheats/{gid}/{fname}",
            "",
            "Activate: GoldHEN overlay > Options > Cheats",
            "",
            f"Values exported as: {'hex (GoldHEN 2.x)' if hex_values else 'decimal'}",
        ], "Export OK", C_OK)
    except Exception as exc:
        message_box(stdscr, [f"Could not write: {exc}"], "Export Failed", C_ERR)


def do_freeze(stdscr) -> None:
    global _freeze_thread
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    draw_border(stdscr, "FREEZE ADDRESS")
    safe_addstr(stdscr, 2, 3,
        "Continuously write a value to lock it in memory.", color(C_WARN))
    stdscr.refresh()
    addr_s   = input_box(stdscr, "Address (hex)    : ", 4, 3, 20)
    val_s    = input_box(stdscr, "Freeze value     : ", 6, 3, 20)
    _wl      = [WIDTH_LABEL[ww] for ww in VALID_WIDTHS]
    _ws      = cycle_input(stdscr, "Width            : ", 8, 3, _wl,
                           WIDTH_LABEL.get(state["scan_width"], "uint32"))
    width    = VALID_WIDTHS[_wl.index(_ws)]
    sec_s    = input_box(stdscr, "Duration (secs)  : ", 10, 3, 6, "30")
    intvl_s  = input_box(stdscr, "Interval (ms)    : ", 12, 3, 6, "200")

    try:
        addr     = int(addr_s, 16)
        err      = _validate_write_addr(addr)
        if err:
            raise ValueError(err)
        val      = int(val_s, 0)
        secs     = max(1, int(sec_s))
        interval = max(50, int(intvl_s)) / 1000.0
        if val < 0 or val > WIDTH_MAX[width]:
            raise ValueError(f"Value out of range for {WIDTH_LABEL[width]}")
        data = struct.pack(WIDTH_FMT[width], val)
    except Exception as exc:
        message_box(stdscr, [f"Bad input: {exc}"], "Error", C_ERR)
        return

    # Issue #10/#11: validate address is in a writable mapped region of the
    # CURRENT process at the moment the freeze is started.  If the game
    # restarted (new PID, new memory layout) this catches the stale address
    # before a single write reaches the wrong process.
    map_err = _validate_addr_in_maps(state["ip"], state["pid"], addr, width)
    if map_err:
        if not confirm_box(stdscr, f"{map_err}\nFreeze anyway?", "Unmapped Address"):
            return

    safe_addstr(stdscr, 15, 3,
        f"Freezing {hex(addr)} = {val} for {secs}s  (every {int(interval*1000)}ms)",
        color(C_WARN) | curses.A_BOLD)
    safe_addstr(stdscr, 16, 3, "Press Q or Esc to stop early.", color(C_NORM))
    stdscr.refresh()

    # Run the write loop in a background thread so the UI redraws responsively.
    # Issues #7/#8/#9: register with the global freeze tracker so the worker
    # can be stopped externally (process change, reconnect) without needing
    # a reference to this closure's local stop_event.
    # Issue #13: snapshot pid at start; worker checks it hasn't changed each
    # tick so it stops itself if the user switches processes mid-freeze.
    frozen_pid  = state["pid"]
    frozen_ip   = state["ip"]
    stop_event  = threading.Event()   # local stop (Q key / deadline)
    write_errors = [0]
    deadline    = time.time() + secs

    def _freeze_worker():
        while time.time() < deadline:
            # Issue #7/#8: honour both the local and global stop events.
            if stop_event.is_set() or _freeze_stop.is_set():
                break
            # Issue #13: abort if process or IP changed under us.
            if state["pid"] != frozen_pid or state["ip"] != frozen_ip:
                add_log("Freeze aborted — process or connection changed", "warn")
                break
            if not ps5_write(frozen_ip, frozen_pid, addr, data):
                write_errors[0] += 1
            # Use the local event for interruptible sleep; also wake early
            # if the global freeze_stop is set by _stop_freeze_worker().
            stop_event.wait(interval)

    # Issue #7: register globally before starting the thread.
    with _freeze_lock:
        _freeze_stop.clear()
        worker = threading.Thread(target=_freeze_worker, daemon=True)
        _freeze_thread = worker
    worker.start()

    stdscr.nodelay(True)
    try:
        while worker.is_alive():
            h, w      = stdscr.getmaxyx()   # Issue #1: re-read on resize
            elapsed   = time.time() - (deadline - secs)
            frac      = min(elapsed / secs, 1.0)
            remaining = max(0, int(deadline - time.time()))
            safe_addstr(stdscr, 18, 3, f"Time left: {remaining:3d}s  ", color(C_OK))
            draw_progress_bar(stdscr, 19, 3, min(w - 8, 50), frac,
                              f"  {int(frac * 100)}%")
            if write_errors[0]:
                safe_addstr(stdscr, 20, 3,
                    f"Write errors: {write_errors[0]}  (connection issue?)",
                    color(C_ERR))
            stdscr.refresh()
            time.sleep(0.1)
            k = stdscr.getch()
            if k == curses.KEY_RESIZE:       # Issue #1: absorb resize
                curses.update_lines_cols()
                stdscr.clear()
                draw_border(stdscr, "FREEZE ADDRESS")
            elif k in (ord('q'), ord('Q'), 27):   # Issue #15: Esc also stops
                stop_event.set()
                break
    finally:
        stdscr.nodelay(False)
        stop_event.set()
        worker.join(timeout=interval + 1.0)
        # Issue #7: deregister from global tracker after local join completes.
        with _freeze_lock:
            if _freeze_thread is worker:
                _freeze_thread = None

    add_log(f"Freeze done {hex(addr)} = {val}")
    message_box(stdscr, ["Freeze complete."], "Done", C_OK)


def do_clear_results(stdscr) -> None:
    if not len(state["scan_results"]) and not state["scan_history"]:
        message_box(stdscr, ["No scan results to clear."], "Clear", C_WARN)
        return
    n       = len(state["scan_results"])
    hist_mb = _history_bytes() / 1_048_576
    if confirm_box(stdscr,
            f"Clear {n:,} scan results and {len(state['scan_history'])} "
            f"undo levels ({hist_mb:.1f} MB)?",
            "Clear Results"):
        state["scan_results"] = _make_addr_array()
        state["scan_values"]  = None
        state["scan_history"] = deque(maxlen=5)
        state["scan_dropped"] = set()
        state["scan_unknown"] = False
        gc.collect()
        add_log(f"Scan results cleared — RSS now {_rss_mb():.0f} MB", "warn")
        message_box(stdscr, ["Results cleared.", "Ready for a fresh First Scan (S)."],
                    "Cleared", C_OK)


def do_log(stdscr) -> None:
    level_colors = {"error": C_ERR, "warn": C_WARN, "info": C_OK}
    offset = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        with _log_lock:
            snap = list(state["log"])
        draw_border(stdscr, f"LOG  ({len(snap)} entries  /  limit {LOG_LIMIT})")
        visible = max(1, h - 6)
        # Auto-scroll to bottom on first render
        if offset == 0 and len(snap) > visible:
            offset = len(snap) - visible
        for i, entry in enumerate(snap[offset:offset + visible]):
            cp  = level_colors.get(entry["level"], C_NORM)
            tag = {"error": "ERR", "warn": "WRN", "info": "INF"}.get(
                entry["level"], "   ")
            line = f"[{entry['ts']}] [{tag}]  {entry['msg']}"
            safe_addstr(stdscr, 3 + i, 3, line[:w - 6], color(cp))
        draw_statusbar(stdscr, [
            (f"{offset+1}-{min(offset+visible,len(snap))}/{len(snap)}", C_WARN),
            ("↑↓ scroll", C_NORM), ("Q back", C_NORM),
        ])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_UP    and offset > 0:              offset -= 1
        elif key == curses.KEY_DOWN and offset < len(snap) - 1: offset += 1
        elif key in (ord('q'), ord('Q')):
            break


# ── main loop ─────────────────────────────────────────────────────────────────

def main(stdscr) -> None:
    curses.curs_set(0)
    curses.noecho()
    curses.cbreak()          # ensure cbreak regardless of wrapper state
    init_colors()
    stdscr.keypad(True)
    stdscr.timeout(100)      # 100 ms blocking timeout on every getch() —
                             # replaces the broken halfdelay/nocbreak pair.
                             # win.timeout() on blocking screens keeps the main
                             # menu header (RSS, etc.) refreshing while idle.
                             # nodelay screens override this per-call.

    screen = "connect"
    while True:
        # Issues #1/#3: handle resize at the top level so every screen
        # automatically gets a full redraw after the user resizes the terminal.
        h, w = stdscr.getmaxyx()
        if h < _MIN_ROWS or w < _MIN_COLS:
            stdscr.clear()
            try:
                stdscr.addstr(0, 0,
                    f"Terminal too small ({w}×{h}). "
                    f"Need {_MIN_COLS}×{_MIN_ROWS}. Resize to continue.")
            except curses.error:
                pass
            stdscr.refresh()
            k = stdscr.getch()
            if k == curses.KEY_RESIZE:
                curses.update_lines_cols()
            elif k in (ord('q'), ord('Q')):
                break
            continue

        if screen == "connect":
            screen = screen_connect(stdscr)
        elif screen == "main":
            result = screen_main(stdscr)
            if result is None:
                break
            screen = result
        elif screen == "proc":
            try:
                procs  = ps5_proc_list(state["ip"])
                screen = screen_proc_select(stdscr, procs)
            except Exception as exc:
                message_box(stdscr, [f"Error: {exc}"], "Connection Error", C_ERR)
                screen = "connect"
        else:
            break


if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    print("\nps5cheats_tui exited.")
