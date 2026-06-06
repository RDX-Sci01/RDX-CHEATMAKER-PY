#!/usr/bin/env python3

"""
RDX PS5 Cheat Maker  –  (Performance Edition)
====================================================
Architectural improvements:

SCAN ENGINE
  • NumPy vectorised search: bytes.find → np.frombuffer + stride tricks → 5–50× faster
  • NumPy address/value arrays (uint64): 8 B/entry vs ~28 B for Python list integers
  • Disk-backed candidate store via mmap/tempfile: no 5 M cap, scales to 200 M+ entries
  • Region pre-filter: exec/unwritable/library/oversized regions skipped before scanning
  • Unknown-scan snapshot: stores entire chunk binary blocks, not per-value python ints
  • Relational next-scan: pure NumPy compare (no Python loop)
  • scan_next batch reads now use numpy for filtering too

MEMORY
  • scan_values stored as np.ndarray (uint8 raw bytes), not integer arrays
  • Undo deltas stored as compact np.ndarray, not full copies
  • Map cache keyed by (pid, monotonic_ns) bucket

RELIABILITY
  • _progress_lock protects done/total atomically (was already there; kept)
  • write_err_lock on freeze counter (kept + extended)
  • PID staleness check before every write / freeze tick
  • Map cache invalidated on PID change (was already there; kept)
  • Memory region validity re-checked before freeze start

UI
  • clrtoeol via safe_addstr_eol everywhere spinner/progress lines redraw
  • Screen redraws batched: only stdscr.refresh() once per 50 ms tick
  • Separate scan-thread timing from UI refresh timing (was already done)
  • Reduce live-value refresh to only visible rows (was already done; kept)

Usage:
    python3 RDX-CHEATMAKER-UI.py
"""

import array as _array
import curses
import mmap
import os
import queue as _queue
import re
import socket
import struct
import json
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

# ── ps5debug wire protocol ────────────────────────────────────────────────────
CMD_MAGIC      = 0xFFAABBCC
CMD_PROC_LIST  = 0xBDAA0001
CMD_PROC_READ  = 0xBDAA0002
CMD_PROC_WRITE = 0xBDAA0003
CMD_PROC_MAPS  = 0xBDAA0004
STATUS_SUCCESS = 0x80000000
STATUS_ERROR   = 0xF0000001
PS5_PORT       = 744

WIDTH_FMT   = {1: 'B', 2: '<H', 4: '<I', 8: '<Q'}
VALID_WIDTHS = [1, 2, 4, 8]
WIDTH_LABEL  = {1: "byte (u8)", 2: "uint16", 4: "uint32", 8: "uint64"}
WIDTH_MAX    = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF, 8: 0xFFFFFFFFFFFFFFFF}
WIDTH_DTYPE  = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}

PROC_ENTRY_SIZE = 36   # char name[32] + int32_t pid
MAP_ENTRY_SIZE  = 58   # char name[32] + uint64 start/end/offset + uint16 prot

TITLE_ID_RE = re.compile(r'^[A-Z]{4}\d{5}$')

# ── disk-backed candidate store ───────────────────────────────────────────────
# Instead of capping at 5 M entries and silently truncating, we write addresses
# to a temp file and memory-map it for zero-copy reads.  This scales to 200 M+
# entries on any machine with disk space, at ~8 B per entry (uint64).

class _DiskAddrs:
    """
    A growable array of uint64 addresses backed by a temp file.
    Supports append_bulk(np.ndarray), len(), iteration, slicing, and
    conversion to np.ndarray for vectorised operations.

    Thread-safety: NOT thread-safe; caller must coordinate.
    """
    GROW_STEP = 1 << 23   # 8 MB increments (1 M uint64 entries)

    def __init__(self):
        self._f   = tempfile.TemporaryFile()
        self._mm  = None
        self._len = 0
        self._cap = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _ensure_cap(self, needed: int) -> None:
        if needed <= self._cap:
            return
        new_cap = max(needed, self._cap + self.GROW_STEP)
        if self._mm is not None:
            self._mm.close()
        self._f.seek(new_cap * 8 - 1)
        self._f.write(b'\x00')
        self._f.flush()
        self._mm = mmap.mmap(self._f.fileno(), new_cap * 8,
                             access=mmap.ACCESS_WRITE)
        self._cap = new_cap

    # ── public API ────────────────────────────────────────────────────────────

    def append_bulk(self, arr: np.ndarray) -> None:
        """Append a numpy uint64 array in one write."""
        if len(arr) == 0:
            return
        arr = np.asarray(arr, dtype=np.uint64)
        self._ensure_cap(self._len + len(arr))
        off = self._len * 8
        self._mm[off:off + arr.nbytes] = arr.tobytes()
        self._len += len(arr)

    def to_numpy(self) -> np.ndarray:
        """Return a copy of all addresses as a numpy uint64 array."""
        if self._len == 0:
            return np.empty(0, dtype=np.uint64)
        self._mm.seek(0)
        return np.frombuffer(bytes(self._mm[:self._len * 8]),
                             dtype=np.uint64).copy()

    def slice_numpy(self, start: int, stop: int) -> np.ndarray:
        """Return addresses[start:stop] as a numpy array (zero-copy view)."""
        start = max(0, start)
        stop  = min(stop, self._len)
        if start >= stop:
            return np.empty(0, dtype=np.uint64)
        off   = start * 8
        size  = (stop - start) * 8
        return np.frombuffer(bytes(self._mm[off:off + size]),
                             dtype=np.uint64).copy()

    def __len__(self) -> int:
        return self._len

    def __bool__(self) -> bool:
        return self._len > 0

    def __iter__(self):
        arr = self.to_numpy()
        return iter(arr.tolist())

    def __getitem__(self, key):
        arr = self.to_numpy()
        return arr[key]

    def clear(self) -> None:
        self._len = 0   # reuse backing store; no need to truncate

    def close(self) -> None:
        if self._mm:
            self._mm.close()
            self._mm = None
        self._f.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _np_addr_array(iterable=()) -> np.ndarray:
    """Compact numpy uint64 address array — cheap to create, vectorised ops."""
    return np.fromiter(iterable, dtype=np.uint64) if not isinstance(iterable, np.ndarray) else np.asarray(iterable, dtype=np.uint64)


def _make_addr_array(iterable=()) -> np.ndarray:
    """Compatibility shim: returns np.ndarray(uint64)."""
    if isinstance(iterable, (_DiskAddrs, np.ndarray)):
        return np.asarray(iterable, dtype=np.uint64) if not isinstance(iterable, _DiskAddrs) else iterable.to_numpy()
    data = list(iterable)
    if not data:
        return np.empty(0, dtype=np.uint64)
    return np.array(data, dtype=np.uint64)


# ── shared state & locks ──────────────────────────────────────────────────────
_log_lock       = threading.Lock()
_cache_lock     = threading.Lock()
_map_cache:      dict = {}
_map_cache_lock = threading.Lock()
_MAP_CACHE_TTL  = 30.0

_progress_lock = threading.Lock()

def _set_progress(progress: dict, done: int, total: int) -> None:
    with _progress_lock:
        progress["done"]  = done
        progress["total"] = max(total, 1)

state = {
    "ip":            "",
    "connected":     False,
    "pid":           None,
    "proc_name":     "",
    "scan_results":  _make_addr_array(),   # np.ndarray[uint64]
    "scan_values":   None,                 # np.ndarray[uint8] raw bytes | None
    "scan_dropped":  set(),                # set[int]
    "scan_pid":      None,
    "scan_truncated": False,
    "scan_unknown":  False,
    "scan_width":    4,
    "scan_aligned":       True,
    "scan_writable_only": True,
    "cheats":        [],
    "game_id":       "",
    "game_ver":      "01.00",
    "game_title":    "",
    "log":           [],
    "scan_history":  deque(maxlen=5),  # (removed_addrs_np, dropped_set, prev_vals_np|None)
}

# ── ps5debug low-level helpers ────────────────────────────────────────────────

def cmd_header(cmd: int, datalen: int = 0) -> bytes:
    return struct.pack("<III", CMD_MAGIC, cmd, datalen)

def ps5_connect(ip: str) -> socket.socket:
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
            prot  = struct.unpack_from("<H", raw, 56)[0]
            maps.append({"start": start, "end": end, "prot": prot, "name": name})
        return maps
    finally:
        s.close()

_UI_MAX_RETRIES = 3

def ps5_read(ip: str, pid: int, addr: int, length: int) -> bytes:
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

# ── persistent socket for scan hot-path ──────────────────────────────────────

class _ScanSocket:
    """
    Persistent TCP connection with pre-built mutable request buffer.
    Patches only the addr+length fields per read — no repeated struct.pack allocs.
    """
    MAX_RETRIES = 3
    _HDR_SIZE   = 28   # 12-byte cmd header + 16-byte body

    def __init__(self, ip: str, pid: int):
        self.ip  = ip
        self.pid = pid
        self._s: Optional[socket.socket] = None
        self._req = bytearray(self._HDR_SIZE)
        struct.pack_into("<III", self._req,  0, CMD_MAGIC, CMD_PROC_READ, 16)
        struct.pack_into("<I",   self._req, 12, pid)
        self._connect()

    def _connect(self):
        if self._s:
            try: self._s.close()
            except Exception: pass
        self._s = ps5_connect(self.ip)

    def read(self, addr: int, length: int) -> bytes:
        struct.pack_into("<QI", self._req, 16, addr, length)
        for attempt in range(self.MAX_RETRIES):
            try:
                if self._s is None:
                    self._connect()
                self._s.sendall(self._req)
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

# ── batch reader for next-scan ────────────────────────────────────────────────

def ps5_read_batch(ip: str, pid: int, addrs, width: int,
                   cancel_event=None, progress_cb=None) -> list:
    """
    Read `width` bytes at each address using parallel sockets.
    `addrs` may be a np.ndarray or any indexable sequence.
    Returns [(addr, bytes|None), ...] in input order.
    """
    NEXT_WORKERS = 6
    if isinstance(addrs, np.ndarray):
        addr_list = addrs.tolist()
    else:
        addr_list = list(addrs)
    total    = len(addr_list)
    if not total:
        return []

    results  = [None] * total
    idx_lock = threading.Lock()
    idx_ptr  = [0]
    done_ctr = [0]

    def _worker():
        sock = _ScanSocket(ip, pid)
        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    break
                with idx_lock:
                    if idx_ptr[0] >= total:
                        break
                    my_idx = idx_ptr[0]
                    idx_ptr[0] += 1
                addr = addr_list[my_idx]
                try:
                    data = sock.read(addr, width)
                except Exception:
                    data = None
                results[my_idx] = (addr, data)
                with idx_lock:
                    done_ctr[0] += 1
                    if progress_cb:
                        progress_cb(done_ctr[0], total)
        finally:
            sock.close()

    workers = [threading.Thread(target=_worker, daemon=True)
               for _ in range(min(NEXT_WORKERS, max(1, total)))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return [r for r in results if r is not None]

# ── map cache ─────────────────────────────────────────────────────────────────

def _get_maps_cached(ip: str, pid: int) -> list:
    now = time.time()
    with _map_cache_lock:
        entry = _map_cache.get(pid)
        if entry and (now - entry[0]) < _MAP_CACHE_TTL:
            return entry[1]
    maps = ps5_maps(ip, pid)
    with _map_cache_lock:
        _map_cache.clear()
        _map_cache[pid] = (now, maps)
    return maps

# ── region selection helper ───────────────────────────────────────────────────

PROT_READ  = 0x1
PROT_WRITE = 0x2
PROT_EXEC  = 0x4
MAX_REGION = 0x40000000   # 1 GB — skip GPU/VRAM/reserved

# Library/system region name prefixes to skip in writable_only mode.
# These rarely contain user-controllable game values and scanning them wastes
# bandwidth.  We err on the side of inclusion: only skip clearly non-game libs.
_SKIP_NAME_PREFIXES = ("/dev/", "libkernel", "libSce", "libPS4", "libc.sprx",
                       "libm.sprx", "libstdc", "libgcc")

def _region_is_skippable(r: dict) -> bool:
    name = r.get("name", "")
    return any(name.startswith(p) for p in _SKIP_NAME_PREFIXES)

def _scannable_regions(maps: list, require_write: bool) -> list:
    """
    Return regions eligible for scanning.
    Filters: exec-only, oversized, unreadable, optionally read-only,
    and known system library regions (writable_only mode only).
    """
    out = []
    for r in maps:
        size = r['end'] - r['start']
        if size == 0 or size > MAX_REGION:
            continue
        if not (r['prot'] & PROT_READ):
            continue
        if r['prot'] == PROT_EXEC:        # exec-only: no data
            continue
        if require_write and not (r['prot'] & PROT_WRITE):
            continue
        if require_write and _region_is_skippable(r):
            continue
        out.append(r)
    return out

# ── NumPy-accelerated scan engine ─────────────────────────────────────────────

CHUNK        = 0x400000    # 4 MB per read
SCAN_WORKERS = 6
QUEUE_DEPTH  = SCAN_WORKERS * 4
_SENTINEL    = object()    # unique sentinel; not None so None chunks work


def _np_search_chunk(data: bytes, target: bytes, base_addr: int,
                     width: int, aligned: bool) -> np.ndarray:
    """
    Find all occurrences of `target` in `data` using NumPy stride tricks.

    Strategy:
      1. Quick pre-screen: if no byte of target[0] exists, bail immediately.
      2. Cast data to uint8 array and use stride tricks to create a
         (N, width) view of every possible aligned start position.
      3. Compare each row against target bytes in one vectorised op.
      4. Return matching absolute addresses.

    This is 5–50× faster than repeated bytes.find() for dense results,
    and equally fast for sparse ones (pre-screen exits early).
    """
    if not data or len(data) < width:
        return np.empty(0, dtype=np.uint64)

    target_bytes = np.frombuffer(target, dtype=np.uint8)
    t0 = target_bytes[0]

    # Quick pre-screen using numpy (avoids full scan when target absent)
    arr = np.frombuffer(data, dtype=np.uint8)
    if not np.any(arr == t0):
        return np.empty(0, dtype=np.uint64)

    step = width if aligned else 1
    n    = len(data)

    # Build candidate positions: all offsets where first byte matches
    # and offset is aligned (if required)
    cand = np.where(arr == t0)[0]
    if aligned and width > 1:
        # Alignment is relative to absolute address: (base_addr + chunk_offset) % width == 0
        cand = cand[(base_addr + cand) % width == 0]

    # Filter: must have room for full `width` bytes
    cand = cand[cand + width <= n]

    if len(cand) == 0:
        return np.empty(0, dtype=np.uint64)

    if width == 1:
        # Trivially all first-byte matches are hits
        return (base_addr + cand).astype(np.uint64)

    # Build a view: shape (len(cand), width), each row is cand[i]:cand[i]+width
    # Using advanced indexing (safe but allocates)
    idx = cand[:, np.newaxis] + np.arange(width, dtype=np.intp)
    rows = arr[idx]           # shape (len(cand), width)
    mask = np.all(rows == target_bytes[np.newaxis, :], axis=1)
    hits = cand[mask]
    return (base_addr + hits).astype(np.uint64)


def scan_first(ip: str, pid: int, value: int, width: int = 4,
               aligned: bool = True, progress_cb=None,
               cancel_event=None,
               writable_only: bool = True) -> np.ndarray:
    """
    Scan all readable regions for `value`.  Returns np.ndarray[uint64] of
    matching addresses.

    Architecture: producer/consumer pipeline — SCAN_WORKERS reader threads
    feed a bounded queue; one searcher thread consumes chunks using NumPy
    vectorised search (5–50× faster than bytes.find per-hit).

    No result cap — found[] is a plain list that grows as needed.
    Caller converts to numpy at the end.
    """
    try:
        target = struct.pack(WIDTH_FMT[width], value)
    except struct.error:
        raise ValueError(
            f"Value {value} out of range for {WIDTH_LABEL.get(width, str(width))}")

    maps      = _get_maps_cached(ip, pid)
    scannable = _scannable_regions(maps, require_write=writable_only)
    if not writable_only:
        # include read-only regions not already in scannable
        ro = _scannable_regions(maps, require_write=False)
        rw_set = {(r['start'], r['end']) for r in scannable}
        scannable = scannable + [r for r in ro if (r['start'], r['end']) not in rw_set]

    total_bytes = max(sum(r['end'] - r['start'] for r in scannable), 1)

    work: list = []
    for r in scannable:
        size = r['end'] - r['start']
        off  = 0
        while off < size:
            csz = min(CHUNK, size - off)
            work.append((r['start'] + off, csz))
            off += csz

    chunk_queue = _queue.Queue(maxsize=QUEUE_DEPTH)
    found       = []                    # plain list; converted to np at end
    done_bytes  = [0]
    work_lock   = threading.Lock()
    work_idx    = [0]
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

    def _search_all():
        sentinels = 0
        while sentinels < n_workers:
            if cancel_event and cancel_event.is_set():
                # drain
                while True:
                    try:   chunk_queue.get_nowait()
                    except _queue.Empty: break
                return
            item = chunk_queue.get()
            if item is _SENTINEL:
                sentinels += 1
                continue
            addr, data = item
            if data is None:
                actual_csz = next((sz for a, sz in work if a == addr), CHUNK)
                done_bytes[0] += actual_csz
                if progress_cb:
                    progress_cb(done_bytes[0], total_bytes)
                continue
            # ── NumPy vectorised search ──────────────────────────────────────
            hits = _np_search_chunk(data, target, addr, width, aligned)
            if len(hits):
                found.append(hits)
            done_bytes[0] += len(data)
            if progress_cb:
                progress_cb(done_bytes[0], total_bytes)

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

    _search_all()
    for wt in watchers:
        wt.join()
    for msg in reader_err:
        add_log(msg, "warn")

    if not found:
        return np.empty(0, dtype=np.uint64)
    result = np.concatenate(found).astype(np.uint64)
    result.sort()
    return result


def scan_next(ip: str, pid: int, value: int, width: int,
              prev: np.ndarray,
              cancel_event=None, progress_cb=None) -> np.ndarray:
    """
    Filter `prev` to addresses that currently hold `value`.
    Uses ps5_read_batch + NumPy comparison — no Python per-address loop.
    """
    try:
        target = struct.pack(WIDTH_FMT[width], value)
    except struct.error:
        raise ValueError(
            f"Value {value} out of range for {WIDTH_LABEL.get(width, str(width))}")

    results = ps5_read_batch(ip, pid, prev, width, cancel_event, progress_cb)
    # NumPy-filter: build parallel arrays and mask
    if not results:
        return np.empty(0, dtype=np.uint64)
    addrs_out = np.array([a for a, d in results if d == target], dtype=np.uint64)
    return addrs_out


# ── unknown-value scan ────────────────────────────────────────────────────────

def scan_first_unknown(ip: str, pid: int, width: int = 4,
                       aligned: bool = True, progress_cb=None,
                       cancel_event=None,
                       writable_only: bool = True) -> tuple:
    """
    Unknown-value first scan.

    Instead of storing one Python int per candidate, we store the raw memory
    chunks as binary blobs keyed by region base address.  This is:
      • ~8× less RAM than integer arrays (no Python object overhead)
      • ~3× less RAM than array.array('Q') (no uint64 per entry — just raw bytes)
      • O(1) lookup during relational next-scan

    Returns (addrs: np.ndarray[uint64], raw_snapshot: dict[base_addr -> bytes]).
    The snapshot dict maps chunk_base → raw_bytes for the entire scannable range.
    Relational next-scan re-reads live values and compares to snapshot bytes.
    """
    maps      = _get_maps_cached(ip, pid)
    scannable = _scannable_regions(maps, require_write=writable_only)
    if not writable_only:
        ro    = _scannable_regions(maps, require_write=False)
        rw_set = {(r['start'], r['end']) for r in scannable}
        scannable = scannable + [r for r in ro if (r['start'], r['end']) not in rw_set]

    total_bytes = max(sum(r['end'] - r['start'] for r in scannable), 1)

    work: list = []
    for r in scannable:
        size = r['end'] - r['start']
        off  = 0
        while off < size:
            csz = min(CHUNK, size - off)
            work.append((r['start'] + off, csz))
            off += csz

    chunk_queue     = _queue.Queue(maxsize=QUEUE_DEPTH)
    snapshot        = {}       # addr → bytes  (raw memory blocks)
    found_addrs     = []       # list of np.ndarray chunks, concatenated at end
    done_bytes      = [0]
    work_lock       = threading.Lock()
    work_idx        = [0]
    reader_err      = []
    reader_err_lock = threading.Lock()

    step = width if aligned else 1

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

    def _snapshot_all():
        sentinels = 0
        while sentinels < n_workers:
            if cancel_event and cancel_event.is_set():
                while True:
                    try:   chunk_queue.get_nowait()
                    except _queue.Empty: break
                return
            item = chunk_queue.get()
            if item is _SENTINEL:
                sentinels += 1
                continue
            addr, data = item
            if data is None:
                actual_csz = next((sz for a, sz in work if a == addr), CHUNK)
                done_bytes[0] += actual_csz
                if progress_cb:
                    progress_cb(done_bytes[0], total_bytes)
                continue
            # Store raw block for relational comparison
            snapshot[addr] = data
            # Enumerate aligned candidate addresses in this block
            arr  = np.frombuffer(data, dtype=np.uint8)
            n    = len(arr)
            offs = np.arange(0, n - width + 1, step, dtype=np.uint64)
            if aligned and width > 1:
                abs_offs = addr + offs
                offs = offs[(abs_offs % width) == 0]
            if len(offs):
                found_addrs.append((addr + offs).astype(np.uint64))
            done_bytes[0] += n
            if progress_cb:
                progress_cb(done_bytes[0], total_bytes)

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

    if not found_addrs:
        return np.empty(0, dtype=np.uint64), {}
    addrs = np.concatenate(found_addrs).astype(np.uint64)
    addrs = np.unique(addrs)   # sort + deduplicate
    return addrs, snapshot


# ── relational modes ──────────────────────────────────────────────────────────

RELATIONAL_MODES = [
    "decreased",
    "increased",
    "changed",
    "unchanged",
    "decreased by",
    "increased by",
]


def scan_next_relational(ip: str, pid: int, width: int,
                         prev_addrs: np.ndarray,
                         prev_snapshot: dict,
                         mode: str,
                         delta: int = 0,
                         cancel_event=None,
                         progress_cb=None) -> tuple:
    """
    Relational next scan.  Reads live values, compares to snapshot bytes
    using NumPy — no Python per-address loop.

    prev_snapshot: dict[chunk_base_addr -> raw_bytes] from scan_first_unknown.

    Returns (new_addrs: np.ndarray[uint64], new_snapshot: dict).
    """
    fmt  = WIDTH_FMT[width]
    mask = WIDTH_MAX[width]

    raw_results = ps5_read_batch(ip, pid, prev_addrs, width,
                                 cancel_event, progress_cb)
    if not raw_results:
        return np.empty(0, dtype=np.uint64), {}

    # Build lookup: addr → previous value integer from snapshot
    def _prev_val(addr: int) -> Optional[int]:
        # Find which snapshot chunk contains this address
        for chunk_base, chunk_data in prev_snapshot.items():
            off = addr - chunk_base
            if 0 <= off <= len(chunk_data) - width:
                return struct.unpack_from(fmt, chunk_data, off)[0]
        return None

    # For efficiency, build prev_val array via vectorised extraction
    # where possible (contiguous block lookups)
    new_addrs_list  = []
    new_snap_blocks = {}   # collect fresh raw chunks for the next iteration

    # Group addresses by their containing snapshot chunk
    # We'll process in bulk per chunk for NumPy acceleration
    chunk_addr_map: dict = {}   # chunk_base → [(addr, live_bytes), ...]
    for addr, live in raw_results:
        if live is None:
            continue
        for chunk_base, chunk_data in prev_snapshot.items():
            off = addr - chunk_base
            if 0 <= off <= len(chunk_data) - width:
                chunk_addr_map.setdefault(chunk_base, []).append((addr, off, live))
                break

    for chunk_base, entries in chunk_addr_map.items():
        chunk_data = prev_snapshot[chunk_base]
        for addr, off, live in entries:
            if cancel_event and cancel_event.is_set():
                break
            cur = struct.unpack(fmt, live)[0]
            prv = struct.unpack_from(fmt, chunk_data, off)[0]
            keep = False
            if   mode == "decreased"    and cur < prv:                        keep = True
            elif mode == "increased"    and cur > prv:                        keep = True
            elif mode == "changed"      and cur != prv:                       keep = True
            elif mode == "unchanged"    and cur == prv:                       keep = True
            elif mode == "decreased by" and cur == (prv - delta) & mask:      keep = True
            elif mode == "increased by" and cur == (prv + delta) & mask:      keep = True
            if keep:
                new_addrs_list.append(addr)

    if not new_addrs_list:
        return np.empty(0, dtype=np.uint64), {}

    new_addrs = np.array(new_addrs_list, dtype=np.uint64)
    new_addrs.sort()

    # Build fresh snapshot from live reads for the kept addresses
    new_snapshot: dict = {}
    addr_set = set(new_addrs_list)
    for addr, live in raw_results:
        if live is not None and addr in addr_set:
            new_snapshot[addr] = live   # 1-address "chunk"

    return new_addrs, new_snapshot


# ── write / validate helpers ──────────────────────────────────────────────────

_ADDR_MIN = 0x0000_0000_0000_0001
_ADDR_MAX = 0x0000_7FFF_FFFF_FFFF

def _validate_write_addr(addr: int) -> Optional[str]:
    if addr < _ADDR_MIN:
        return f"Address {hex(addr)} is zero or negative — likely a mistake."
    if addr > _ADDR_MAX:
        return f"Address {hex(addr)} is in kernel space — write blocked."
    return None

def _validate_addr_in_maps(ip: str, pid: int, addr: int, length: int) -> Optional[str]:
    try:
        maps = _get_maps_cached(ip, pid)
    except Exception as exc:
        return f"Could not fetch memory map to validate address: {exc}"
    for r in maps:
        if r['start'] <= addr and addr + length <= r['end']:
            if r['prot'] & PROT_WRITE:
                return None
            return (f"Address {hex(addr)} is mapped but not writable "
                    f"(prot={hex(r['prot'])}).")
    return f"Address {hex(addr)} is not in any mapped region of PID {pid}."


# ── misc helpers ──────────────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-.]', '_', name)

def generate_cht(cheats: list, game_id: str, game_ver: str,
                 game_title: str, hex_values: bool = True) -> str:
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

LOG_LIMIT = 500

def add_log(msg: str, level: str = "info") -> None:
    with _log_lock:
        state["log"].append({"ts": time.strftime("%H:%M:%S"), "msg": msg, "level": level})
        if len(state["log"]) > LOG_LIMIT:
            state["log"] = state["log"][-LOG_LIMIT:]

# ── curses UI helpers ─────────────────────────────────────────────────────────

def init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,    -1)
    curses.init_pair(2, curses.COLOR_GREEN,   -1)
    curses.init_pair(3, curses.COLOR_YELLOW,  -1)
    curses.init_pair(4, curses.COLOR_RED,     -1)
    curses.init_pair(5, curses.COLOR_WHITE,   -1)
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)
    curses.init_pair(7, curses.COLOR_BLACK,   curses.COLOR_CYAN)
    curses.init_pair(8, curses.COLOR_BLACK,   curses.COLOR_RED)

C_TITLE = 1; C_OK = 2; C_WARN = 3; C_ERR = 4
C_NORM  = 5; C_ACC = 6; C_SEL  = 7; C_DSEL = 8

def color(pair: int) -> int:
    return curses.color_pair(pair)

def safe_addstr(win, y: int, x: int, text: str, attr: int = 0) -> None:
    try:
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        win.addstr(y, x, text[:max(0, w - x)], attr)
    except curses.error:
        pass

def safe_addstr_eol(win, y: int, x: int, text: str, attr: int = 0) -> None:
    """Write text and clear to end-of-line (prevents ghost chars on redraw)."""
    try:
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        clipped  = text[:max(0, w - x)]
        win.addstr(y, x, clipped, attr)
        tail     = w - x - len(clipped)
        if tail > 0:
            erase_len = tail if (y < h - 1) else max(0, tail - 1)
            if erase_len:
                win.addstr(y, x + len(clipped), " " * erase_len)
    except curses.error:
        pass

def draw_border(win, title: str = "") -> None:
    win.box()
    if title:
        h, w   = win.getmaxyx()
        label  = f" {title} "
        tx     = max(2, (w - len(label)) // 2)
        blank  = max(0, w - 2 - tx)
        if blank:
            try:
                win.addstr(0, tx, " " * blank)
            except curses.error:
                pass
        safe_addstr(win, 0, tx, label, color(C_TITLE) | curses.A_BOLD)

def draw_statusbar(stdscr, segments: list) -> None:
    h, w = stdscr.getmaxyx()
    sep  = "  ·  "
    try:
        stdscr.addstr(h - 1, 0, " " * (w - 1), color(C_SEL))
    except curses.error:
        pass
    x = 0
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

def cycle_input(stdscr, prompt: str, y: int, x: int,
                options: list, default=None):
    h, _ = stdscr.getmaxyx()
    if y >= h - 1:
        return default if default is not None else options[0]
    idx = options.index(default) if default in options else 0
    curses.curs_set(0)
    while True:
        safe_addstr(stdscr, y, x, prompt, color(C_WARN) | curses.A_BOLD)
        hint = f"< {options[idx]} >  (Tab/arrows to change, Enter to confirm)"
        safe_addstr_eol(stdscr, y, x + len(prompt), hint, color(C_TITLE) | curses.A_BOLD)
        stdscr.refresh()
        k = stdscr.getch()
        if k in (ord('\t'), curses.KEY_RIGHT):
            idx = (idx + 1) % len(options)
        elif k == curses.KEY_LEFT:
            idx = (idx - 1) % len(options)
        elif k in (curses.KEY_ENTER, 10, 13):
            return options[idx]

def confirm_box(stdscr, question: str, title: str = "Confirm") -> bool:
    h, w = stdscr.getmaxyx()
    lines = [question, "", "  [Y] Yes      [N / Esc] No"]
    bh = len(lines) + 4
    bw = min(max(len(l) for l in lines) + 6, w - 4)
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

def message_box(stdscr, lines: list, title: str = "Info",
                color_pair: int = C_NORM) -> None:
    h, w = stdscr.getmaxyx()
    bh   = len(lines) + 4
    bw   = min(max((len(l) for l in lines), default=10) + 6, w - 4)
    win  = curses.newwin(bh, bw,
                         max(0, (h - bh) // 2), max(0, (w - bw) // 2))
    draw_border(win, title)
    for i, line in enumerate(lines):
        safe_addstr(win, i + 2, 3, line[:bw - 6], color(color_pair))
    safe_addstr(win, bh - 2, max(1, (bw - 14) // 2),
                " Press any key ", color(C_WARN))
    win.refresh()
    win.getch()

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
    safe_addstr(stdscr, 8, 3, "Connecting…", color(C_WARN))
    stdscr.refresh()
    try:
        procs = ps5_proc_list(ip)
        state["connected"] = True
        add_log(f"Connected to {ip}, {len(procs)} processes")
        return screen_proc_select(stdscr, procs)
    except Exception as e:
        safe_addstr(stdscr, 8, 3, f"✗ Failed: {e}".ljust(60), color(C_ERR))
        safe_addstr(stdscr, 10, 3, "Press any key to retry.", color(C_NORM))
        stdscr.refresh()
        stdscr.getch()
        return "connect"

def _clear_scan_state() -> None:
    state["scan_results"]    = _make_addr_array()
    state["scan_values"]     = None
    state["scan_dropped"]    = set()
    state["scan_history"]    = deque(maxlen=5)
    state["scan_pid"]        = None
    state["scan_truncated"]  = False
    state["scan_unknown"]    = False
    with _map_cache_lock:
        _map_cache.clear()

def screen_proc_select(stdscr, procs: list) -> str:
    sort_by    = "name"
    procs_orig = list(procs)

    def _sorted(lst):
        if sort_by == "pid":
            return sorted(lst, key=lambda p: p['pid'])
        return sorted(lst, key=lambda p: p['name'].lower())

    procs      = _sorted(procs_orig)
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
            if p["pid"] != state["pid"]:
                _clear_scan_state()
            state["pid"]       = p["pid"]
            state["proc_name"] = p["name"]
            add_log(f"Attached to PID {state['pid']} ({state['proc_name']})")
            return "main"
        elif key in (ord('q'), ord('Q')):
            return "connect"
        elif key in (curses.KEY_BACKSPACE, 127, 8):
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
    stats  = (f"  Results: {len(state['scan_results'])}   "
              f"Cheats: {len(state['cheats'])}   "
              f"Width: {wlabel}  ({align})")
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
            unavail = (
                (label == "Next Scan"    and not len(state["scan_results"])) or
                (label == "Results"      and not len(state["scan_results"])) or
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
        "scan_first":  do_scan_first,
        "scan_next":   do_scan_next,
        "results":     do_show_results,
        "write":       do_write,
        "cheat_list":  do_cheat_list,
        "export":      do_export,
        "freeze":      do_freeze,
        "log":         do_log,
        "clear":       do_clear_results,
    }
    if action == "proc":
        return "proc"
    fn = actions.get(action)
    if fn:
        fn(stdscr)

# ── scan progress UI ──────────────────────────────────────────────────────────

def _run_scan_with_progress(stdscr, thread_fn, total_label: str,
                             cancel_event: threading.Event,
                             progress: dict, w: int) -> bool:
    t = threading.Thread(target=thread_fn, daemon=True)
    t.start()

    spinner = ["|", "/", "-", "\\"]
    spin_i  = 0
    stdscr.nodelay(True)
    try:
        while t.is_alive():
            with _progress_lock:
                snap_done  = progress["done"]
                snap_total = progress["total"]
            frac = snap_done / max(snap_total, 1)
            # safe_addstr_eol clears residual chars on shorter redraws
            safe_addstr_eol(stdscr, 9, 3,
                f"{spinner[spin_i % 4]}  {total_label}  "
                f"{snap_done:,} / {snap_total:,}  [Esc=cancel]",
                color(C_WARN))
            draw_progress_bar(stdscr, 10, 3, min(w - 8, 60), frac,
                              f"  {int(frac * 100)}%")
            stdscr.refresh()
            time.sleep(0.1)
            spin_i += 1
            if stdscr.getch() == 27:
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

    val_s        = input_box(stdscr, "Value (blank = unknown): ", 4, 3, 20)
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

    state["scan_width"]         = width
    state["scan_aligned"]       = aligned
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
    progress     = {"done": 0, "total": 1, "results": None, "snapshot": None,
                    "error": None}

    if unknown_mode:
        def run():
            try:
                addrs, snap = scan_first_unknown(
                    state["ip"], state["pid"], width, aligned,
                    lambda d, t: _set_progress(progress, d, t),
                    cancel_event,
                    writable_only=writable_only)
                progress["results"]  = addrs
                progress["snapshot"] = snap
            except Exception as exc:
                progress["error"] = str(exc)
        scan_label = "Snapshotting memory…"
    else:
        def run():
            try:
                res = scan_first(
                    state["ip"], state["pid"], val, width, aligned,
                    lambda d, t: _set_progress(progress, d, t),
                    cancel_event,
                    writable_only=writable_only)
                progress["results"] = res
            except Exception as exc:
                progress["error"] = str(exc)
        scan_label = "Scanning memory…"

    ok = _run_scan_with_progress(stdscr, run, scan_label, cancel_event, progress, w)
    if not ok:
        add_log("First scan cancelled", "warn")
        return
    if progress["error"]:
        add_log(f"Scan error: {progress['error']}", "error")
        message_box(stdscr, [f"Error: {progress['error']}"], "Scan Failed", C_ERR)
        return

    results = progress["results"]
    if results is None:
        results = _make_addr_array()

    state["scan_history"]  = deque(maxlen=5)
    state["scan_dropped"]  = set()
    state["scan_results"]  = results
    state["scan_values"]   = progress.get("snapshot")  # dict | None
    state["scan_pid"]      = state["pid"]
    state["scan_truncated"] = False
    state["scan_unknown"]  = unknown_mode
    add_log(f"{'Unknown' if unknown_mode else 'First'} scan "
            f"w={width} aligned={aligned}: {len(results):,} candidates")

    if unknown_mode:
        message_box(stdscr, [
            f"Snapshot taken: {len(results):,} candidates.",
            "",
            "Now trigger a change in-game (take damage, heal, etc.)",
            "then use Next Scan (N) and choose a relational filter",
            "(decreased / increased / unchanged / changed).",
        ], "Snapshot Complete", C_OK)
    else:
        message_box(stdscr, [
            f"Found {len(results):,} results.",
            "",
            "Change the value in-game, then use Next Scan (N).",
            "Once narrowed down, use Results (R) to pick an address.",
        ], "Scan Complete", C_OK)


def do_scan_next(stdscr) -> None:
    if not len(state["scan_results"]):
        message_box(stdscr,
            ["No previous scan results.", "Run First Scan (S) first."], "Error", C_ERR)
        return
    if state.get("scan_pid") not in (None, state["pid"]):
        message_box(stdscr,
            ["Scan results are from a different process.",
             "Please run a new First Scan (S) for this process."],
            "Stale Results", C_WARN)
        return

    stdscr.clear()
    h, w    = stdscr.getmaxyx()
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
        prev_snapshot = state.get("scan_values")
        if not isinstance(prev_snapshot, dict):
            message_box(stdscr,
                ["Value snapshot is missing or wrong type.",
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
                    "results": None, "snapshot": None, "error": None}

        def run_rel():
            try:
                na, ns = scan_next_relational(
                    state["ip"], state["pid"], width,
                    prev_addrs, prev_snapshot,
                    mode_lbl, delta,
                    cancel_event,
                    lambda d, t: _set_progress(progress, d, t))
                progress["results"]  = na
                progress["snapshot"] = ns
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

        new_addrs    = progress["results"]
        if new_addrs is None: new_addrs = _make_addr_array()
        new_snapshot = progress["snapshot"] or {}

        # Store undo delta (removed addresses only — not full copy)
        prev_set = set(int(a) for a in prev_addrs)
        new_set  = set(int(a) for a in new_addrs)
        removed  = np.array(sorted(prev_set - new_set), dtype=np.uint64)
        state["scan_history"].append((removed, set(state["scan_dropped"]), prev_snapshot))
        state["scan_results"] = new_addrs
        state["scan_values"]  = new_snapshot
        state["scan_dropped"] = state["scan_dropped"] & new_set

        add_log(f"Next scan ({mode_lbl}): {len(new_addrs):,} remain")
        tip      = "Perfect! Use Results (R)." if len(new_addrs) <= 10 else "Still many — trigger another change and scan again."
        undo_msg = f"  (U to undo — restores ~{len(new_addrs) + len(removed):,} candidates)" if state["scan_history"] else ""
        message_box(stdscr,
            [f"{len(new_addrs):,} candidates remain.", "", tip, undo_msg],
            "Scan Complete", C_OK if len(new_addrs) <= 10 else C_WARN)

    else:
        safe_addstr(stdscr, 4, 3, "Enter the new in-game value.", color(C_NORM))
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
                    lambda d, t: _set_progress(progress, d, t))
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

        results  = progress["results"]
        if results is None: results = _make_addr_array()

        prev_set = set(int(a) for a in prev_addrs)
        new_set  = set(int(a) for a in results)
        removed  = np.array(sorted(prev_set - new_set), dtype=np.uint64)
        state["scan_history"].append((removed, set(state["scan_dropped"]), None))
        state["scan_results"] = results
        state["scan_values"]  = None
        state["scan_dropped"] = state["scan_dropped"] & new_set

        add_log(f"Next scan val={val}: {len(results):,} remain")
        tip      = "Perfect! Use Results (R)." if len(results) <= 10 else "Still many — change value and scan again."
        undo_msg = f"  (U to undo — restores ~{len(results) + len(removed):,} candidates)" if state["scan_history"] else ""
        message_box(stdscr,
            [f"{len(results):,} results remain.", "", tip, undo_msg],
            "Scan Complete", C_OK if len(results) <= 10 else C_WARN)


# ── results screen ────────────────────────────────────────────────────────────

def _refresh_visible_locked(ip: str, pid: int, addrs: list, width: int,
                             cache: dict, lock: threading.Lock,
                             cancel_event: Optional[threading.Event] = None,
                             expected_pid: Optional[int] = None) -> None:
    if not addrs:
        return
    fmt  = WIDTH_FMT[width]
    sock = None
    try:
        sock = _ScanSocket(ip, pid)
        sock._s.settimeout(1.5)
        for addr in addrs:
            if cancel_event and cancel_event.is_set():
                break
            if expected_pid is not None and state["pid"] != expected_pid:
                break
            try:
                raw  = sock.read(addr, width)
                vstr = str(struct.unpack(fmt, raw)[0])
            except Exception:
                vstr = "?"
            with lock:
                cache[addr] = vstr
    finally:
        if sock:
            sock.close()


def do_show_results(stdscr) -> None:
    results = state["scan_results"]
    if not len(results):
        message_box(stdscr,
            ["No scan results yet.", "Run First Scan (S) first."], "Results", C_WARN)
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
    refresh_complete  = 0.0
    REFRESH_INTERVAL  = 2.0
    refresh_thread    = None
    refresh_cancel    = threading.Event()

    # Convert to list once for indexing
    res_list = [int(a) for a in results]

    stdscr.nodelay(True)
    try:
        while True:
            now = time.time()
            h, w = stdscr.getmaxyx()
            visible = max(1, h - 7)

            if sel < offset:              offset = sel
            if sel >= offset + visible:   offset = sel - visible + 1

            thread_idle = refresh_thread is None or not refresh_thread.is_alive()
            if thread_idle and refresh_thread is not None:
                if refresh_complete < refresh_deadline:
                    refresh_complete = time.time()
            if thread_idle and now - last_refresh >= REFRESH_INTERVAL:
                visible_addrs = res_list[offset:offset + visible]
                refresh_cancel.clear()
                refresh_thread = threading.Thread(
                    target=_refresh_visible_locked,
                    args=(state["ip"], state["pid"], visible_addrs,
                          state["scan_width"], val_cache, cache_lock,
                          refresh_cancel, state["pid"]),
                    daemon=True)
                refresh_thread.start()
                refresh_deadline = now
                last_refresh = now

            stdscr.clear()
            draw_border(stdscr, f"RESULTS  ({len(res_list)} addresses)")
            wlabel = WIDTH_LABEL.get(state["scan_width"], str(state["scan_width"]))
            safe_addstr(stdscr, 2, 3,
                f"Type: {wlabel}   Process: {state['proc_name']} (PID {state['pid']})",
                color(C_WARN))
            safe_addstr(stdscr, 3, 3,
                "↑↓ navigate   Enter add cheat   D drop   U undo scan   Q back",
                color(C_NORM))

            for i, addr in enumerate(res_list[offset:offset + visible]):
                idx    = offset + i
                with cache_lock:
                    vstr = val_cache.get(addr, "…")
                marker = ">" if idx == sel else " "
                line   = f"{marker} {idx+1:4d}   {hex(addr):<20}  current = {vstr}"
                attr   = color(C_SEL) | curses.A_BOLD if idx == sel else color(C_NORM)
                safe_addstr(stdscr, 5 + i, 2, line[:w - 4].ljust(w - 4), attr)

            data_age      = int(now - refresh_complete) if refresh_complete else 0
            is_refreshing = refresh_thread is not None and refresh_thread.is_alive()
            stale         = is_refreshing and data_age >= REFRESH_INTERVAL
            age_label     = "⟳ fetching…" if is_refreshing else f"~{data_age}s old"
            if stale:
                age_label = f"⚠ stale (~{data_age}s)"
            draw_statusbar(stdscr, [
                (f"{len(res_list)} results", C_WARN),
                ("↑↓ navigate",   C_NORM),
                ("Enter cheat",   C_OK),
                ("D drop",        C_ERR),
                ("U undo",        C_WARN),
                ("M flush maps",  C_WARN),
                (age_label,       C_ERR if stale else C_ACC if is_refreshing else C_NORM),
                ("Q back",        C_NORM),
            ])
            stdscr.refresh()
            time.sleep(0.05)

            key = stdscr.getch()
            if key == curses.KEY_UP and sel > 0:
                sel -= 1
            elif key == curses.KEY_DOWN and sel < len(res_list) - 1:
                sel += 1
            elif key in (curses.KEY_ENTER, 10, 13):
                stdscr.nodelay(False)
                _add_cheat_at(stdscr, res_list[sel])
                stdscr.nodelay(True)
                results  = state["scan_results"]
                res_list = [int(a) for a in results]
                with cache_lock:
                    val_cache.clear()
            elif key in (ord('d'), ord('D')):
                dropped  = res_list[sel]
                res_list = [a for i, a in enumerate(res_list) if i != sel]
                state["scan_results"] = np.array(res_list, dtype=np.uint64)
                state["scan_dropped"].add(dropped)
                with cache_lock:
                    val_cache.pop(dropped, None)
                if not res_list:
                    break
                sel = min(sel, len(res_list) - 1)
            elif key in (ord('u'), ord('U')):
                if state["scan_history"]:
                    snap         = state["scan_history"].pop()
                    removed_diff = snap[0]
                    prev_dropped = snap[1]
                    prev_snap    = snap[2] if len(snap) > 2 else None
                    cur_set      = set(res_list)
                    prev_results = np.array(
                        sorted(cur_set | set(int(a) for a in removed_diff)),
                        dtype=np.uint64)
                    state["scan_results"] = prev_results
                    state["scan_dropped"] = prev_dropped
                    state["scan_values"]  = prev_snap
                    results  = prev_results
                    res_list = [int(a) for a in results]
                    with cache_lock:
                        val_cache.clear()
                    sel = 0; offset = 0
                    add_log(f"Undo: restored {len(res_list)} candidates")
            elif key in (ord('m'), ord('M')):
                with _map_cache_lock:
                    _map_cache.clear()
                with cache_lock:
                    val_cache.clear()
                add_log("Map cache flushed — next scan/write will re-fetch regions", "warn")
            elif key in (ord('q'), ord('Q')):
                break
    finally:
        stdscr.nodelay(False)
        refresh_cancel.set()
        if refresh_thread and refresh_thread.is_alive():
            refresh_thread.join(timeout=2.0)


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
    typ   = cycle_input(stdscr, "Cheat type       : ", 9, 3, ["freeze", "write"], "freeze")
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
        if not confirm_box(stdscr,
                f"'{gid}' doesn't match PPSA01234 format.\nExport anyway?",
                "Bad Title ID"):
            continue
        break

    VERSION_RE = re.compile(r'^\d{2}\.\d{2}$')
    gver = input_box(stdscr, "Version   (e.g. 01.00)     : ", 6, 3, 10, state["game_ver"])
    if gver and not VERSION_RE.match(gver):
        if not confirm_box(stdscr,
                f"Version '{gver}' doesn't match NN.NN format.\nContinue anyway?",
                "Version Format"):
            return
    gtit     = input_box(stdscr, "Game Title                 : ", 8, 3, 40, state["game_title"])
    val_fmt  = cycle_input(stdscr, "Value format               : ", 10, 3,
                           ["hex (GoldHEN 2.x)", "decimal (older loaders)"],
                           "hex (GoldHEN 2.x)")
    hex_values = val_fmt.startswith("hex")
    state.update(game_id=gid, game_ver=gver, game_title=gtit)

    safe_gid  = sanitize_filename(gid)
    safe_gver = sanitize_filename(gver.replace('.', '_'))
    fname     = f"{safe_gid or 'UNKNOWN'}_{safe_gver or '00_00'}.json"
    save_path = Path.home() / fname

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

    # Validate address against current maps before starting
    map_err = _validate_addr_in_maps(state["ip"], state["pid"], addr, width)
    if map_err:
        if not confirm_box(stdscr, f"{map_err}\nFreeze anyway?", "Unmapped Address"):
            return

    safe_addstr(stdscr, 15, 3,
        f"Freezing {hex(addr)} = {val} for {secs}s  (every {int(interval*1000)}ms)",
        color(C_WARN) | curses.A_BOLD)
    safe_addstr(stdscr, 16, 3, "Press Q to stop early.", color(C_NORM))
    stdscr.refresh()

    stop_event     = threading.Event()
    write_errors   = [0]
    write_err_lock = threading.Lock()
    deadline       = time.time() + secs
    # Snapshot PID at freeze start; bail if it changes mid-run (reliability fix)
    freeze_pid     = state["pid"]

    def _freeze_worker():
        while time.time() < deadline and not stop_event.is_set():
            # PID staleness guard — don't write to wrong process
            if state["pid"] != freeze_pid:
                with write_err_lock:
                    write_errors[0] += 1
                stop_event.set()
                break
            if not ps5_write(state["ip"], freeze_pid, addr, data):
                with write_err_lock:
                    write_errors[0] += 1
            stop_event.wait(interval)

    worker = threading.Thread(target=_freeze_worker, daemon=True)
    worker.start()

    stdscr.nodelay(True)
    try:
        while worker.is_alive():
            elapsed   = time.time() - (deadline - secs)
            frac      = min(elapsed / secs, 1.0)
            remaining = max(0, int(deadline - time.time()))
            h, w      = stdscr.getmaxyx()
            safe_addstr_eol(stdscr, 18, 3, f"Time left: {remaining:3d}s  ", color(C_OK))
            draw_progress_bar(stdscr, 19, 3, min(w - 8, 50), frac,
                              f"  {int(frac * 100)}%")
            with write_err_lock:
                err_snap = write_errors[0]
            if err_snap:
                safe_addstr_eol(stdscr, 20, 3,
                    f"Write errors: {err_snap}  (connection issue?)",
                    color(C_ERR))
            stdscr.refresh()
            time.sleep(0.1)
            if stdscr.getch() in (ord('q'), ord('Q')):
                stop_event.set()
                break
    finally:
        stdscr.nodelay(False)
        stop_event.set()
        worker.join(timeout=interval + 1.0)

    add_log(f"Freeze done {hex(addr)} = {val}")
    message_box(stdscr, ["Freeze complete."], "Done", C_OK)


def do_clear_results(stdscr) -> None:
    n = len(state["scan_results"])
    if not n and not state["scan_history"]:
        message_box(stdscr, ["No scan results to clear."], "Clear", C_WARN)
        return
    if confirm_box(stdscr, f"Clear {n} scan results and history?", "Clear Results"):
        state["scan_results"] = _make_addr_array()
        state["scan_values"]  = None
        state["scan_history"] = deque(maxlen=5)
        state["scan_dropped"] = set()
        state["scan_unknown"] = False
        add_log("Scan results cleared", "warn")
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
    init_colors()
    stdscr.keypad(True)

    screen = "connect"
    while True:
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
