#!/usr/bin/env python3

"""
Python Cheat Maker with Terminal UI
curses is built into Python on Win/Linux/macOS; no extra packages

Usage:
    python3 RDX-CHEATMAKER-UI.py
"""

import array as _array
import curses
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

# Maximum addresses kept in memory after a first scan.
# Each PS5 address is a uint64 (8 bytes).  Stored in array.array('Q') rather
# than a plain Python list, each entry costs ~8 bytes vs ~28 bytes in CPython.
#   500 000 × 8 B  ≈  4 MB   (array)
#   500 000 × 28 B ≈ 14 MB   (list)
# Exceeding the cap is handled gracefully: truncation + warning.
MAX_SCAN_RESULTS = 500_000

def _make_addr_array(iterable=()) -> _array.array:
    """Return a compact array.array of uint64 addresses."""
    return _array.array('Q', iterable)

def _make_addr_set(iterable=()) -> set:
    """
    Compact dropped-address set. We keep it as a plain Python set since we need
    O(1) membership tests and the number of manually-dropped addresses is small
    (users drop a handful, not millions).  Comment kept as a decision record.
    """
    return set(iterable)

def _addr_list(a) -> list:
    """Convert an addr array (or plain list/iterable) to a plain Python list."""
    return list(a)

# ── shared state & locks ──────────────────────────────────────────────────────
_log_lock       = threading.Lock()
_cache_lock     = threading.Lock()   # protects val_cache in do_show_results
_map_cache:      dict = {}           # {pid: (timestamp, maps_list)}
_map_cache_lock = threading.Lock()
_MAP_CACHE_TTL  = 30.0               # seconds before cached map is stale

state = {
    "ip":           "",
    "connected":    False,
    "pid":          None,
    "proc_name":    "",
    "scan_results": _make_addr_array(),  # array.array('Q') — always this type
    "scan_values":  None,        # array.array('Q')|None — parallel last-known values
                                 # (only populated for unknown-value / relational scans)
    "scan_dropped": set(),       # set[int]  — addresses the user manually dropped
    "scan_pid":        None,     # pid that scan_results belong to
    "scan_truncated":  False,    # True when result cap was hit
    "scan_unknown":    False,    # True when scan was started as unknown-value
    "scan_width":   4,
    "scan_aligned":       True,   # True = step by width; False = every byte
    "scan_writable_only": True,   # True = skip R/O regions (faster, recommended)
    "cheats":       [],
    "game_id":      "",
    "game_ver":     "01.00",
    "game_title":   "",
    "log":          [],
    "scan_history": deque(maxlen=5),  # (results_array, dropped_set) — deque caps RAM
                                      # and O(1) pop vs list.pop(0) which is O(n)
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

def ps5_read_batch(ip: str, pid: int, addrs: list, width: int,
                   cancel_event=None, progress_cb=None) -> list:
    """
    Read `width` bytes at each address using NEXT_WORKERS parallel sockets.
    Returns a list of (addr, bytes|None) pairs in input order.

    Previously used a single serial socket; parallelising here gives the
    same speedup as scan_first's producer/consumer pipeline.
    """
    NEXT_WORKERS = 6   # match SCAN_WORKERS; bytes.find removed searcher bottleneck
    if not addrs:
        return []

    total    = len(addrs)
    results  = [None] * total          # pre-allocated, indexed by position
    idx_lock = threading.Lock()
    idx_ptr  = [0]                     # next address index to claim
    done_ctr = [0]                     # completed reads (for progress)

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
                addr = addrs[my_idx]
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
    # Filter out any slots not filled (cancelled mid-run)
    return [r for r in results if r is not None]

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
    found       = _make_addr_array()
    done_bytes  = [0]          # written only by searcher thread
    work_lock   = threading.Lock()
    work_idx    = [0]          # shared index into work[]; protected by work_lock
    reader_err      = []           # collects non-fatal reader warnings
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
            # Fast pre-screen: if target not in chunk at all, skip immediately.
            # bytes.find on a miss still scans the whole chunk, but the C
            # implementation is ~2400 MB/s so this is rarely worth splitting.
            # The real win: zero-page detection. Sparse mmap regions are
            # often entirely zero; skip them if target != b'\x00'*width.
            csz = len(data)
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

    return found


def scan_next(ip: str, pid: int, value: int, width: int,
              prev: "_array.array",
              cancel_event=None, progress_cb=None) -> "_array.array":
    """
    Filter `prev` to addresses that currently hold `value`.
    Accepts an array.array directly; converts to list once for ps5_read_batch.
    """
    try:
        target = struct.pack(WIDTH_FMT[width], value)
    except struct.error:
        raise ValueError(
            f"Value {value} out of range for {WIDTH_LABEL.get(width, str(width))}")
    # Pass array.array directly — ps5_read_batch accepts any indexable
    # sequence so the _addr_list() copy is unnecessary.
    results = ps5_read_batch(ip, pid, prev, width,
                             cancel_event, progress_cb)
    return _make_addr_array(addr for addr, data in results if data == target)


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
    found_addrs  = _make_addr_array()
    found_values = _make_addr_array()   # parallel; same index = same address
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
            csz = len(data)
            # Record every aligned address and its current value.
            for off in range(0, csz - width + 1, step):
                if cancel_event and cancel_event.is_set():
                    return
                raw_val = data[off:off + width]
                if len(raw_val) < width:
                    break
                int_val = struct.unpack_from(WIDTH_FMT[width], raw_val)[0]
                found_addrs.append(addr + off)
                found_values.append(int_val)
                if len(found_addrs) >= MAX_SCAN_RESULTS:
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
                    return
            done_bytes[0] += csz
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

    return found_addrs, found_values


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
                         prev_addrs: "_array.array",
                         prev_values: "_array.array",
                         mode: str,
                         delta: int = 0,
                         cancel_event=None,
                         progress_cb=None) -> tuple:
    """
    Relational next scan for unknown-value sessions.

    Reads the current value at every address in prev_addrs, then keeps only
    those that satisfy `mode` relative to their entry in prev_values.

    Returns (new_addrs, new_values) — parallel array.array('Q') pairs
    reflecting the *current* values so the next relational pass has a fresh
    baseline.

    mode must be one of RELATIONAL_MODES.
    delta is only used for 'decreased by' and 'increased by'.
    """
    fmt  = WIDTH_FMT[width]
    mask = WIDTH_MAX[width]

    raw_results = ps5_read_batch(ip, pid, prev_addrs, width,
                                 cancel_event, progress_cb)

    new_addrs  = _make_addr_array()
    new_values = _make_addr_array()

    # Build a fast addr→prev_value lookup dict from the parallel arrays.
    # Build addr→prev_value lookup.  Duplicate addresses (which should not
    # occur after the deduplication in scan_first_unknown, but can appear if
    # the snapshot was truncated mid-chunk) are resolved by keeping the first
    # occurrence — consistent with array order — rather than silently
    # overwriting with the last, which would compare against a stale baseline.
    prev_map: dict = {}
    for i in range(len(prev_addrs)):
        addr_i = prev_addrs[i]
        if addr_i not in prev_map:   # first-wins: preserves snapshot order
            prev_map[addr_i] = prev_values[i]

    for addr, data in raw_results:
        if data is None:
            continue
        cur = struct.unpack(fmt, data)[0]
        prv = prev_map.get(addr)
        if prv is None:
            continue
        if   mode == "decreased"   and cur < prv:
            pass
        elif mode == "increased"   and cur > prv:
            pass
        elif mode == "changed"     and cur != prv:
            pass
        elif mode == "unchanged"   and cur == prv:
            pass
        elif mode == "decreased by" and cur == (prv - delta) & mask:
            pass
        elif mode == "increased by" and cur == (prv + delta) & mask:
            pass
        else:
            continue   # did not match — drop this address
        new_addrs.append(addr)
        new_values.append(cur)

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
    try:
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        win.addstr(y, x, text[:max(0, w - x)], attr)
    except curses.error:
        pass

def draw_border(win, title: str = "") -> None:
    win.box()
    if title:
        h, w = win.getmaxyx()
        label = f" {title} "
        safe_addstr(win, 0, max(2, (w - len(label)) // 2),
                    label, color(C_TITLE) | curses.A_BOLD)

def draw_statusbar(stdscr, segments: list) -> None:
    h, w = stdscr.getmaxyx()
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
        safe_addstr(stdscr, y, x + len(prompt), hint, color(C_TITLE) | curses.A_BOLD)
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
    bh = len(lines) + 4
    bw = min(max((len(l) for l in lines), default=10) + 6, w - 4)
    win = curses.newwin(bh, bw,
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
        safe_addstr(stdscr, 8, 3, f"X Failed: {e}".ljust(60), color(C_ERR))
        safe_addstr(stdscr, 10, 3, "Press any key to retry.", color(C_NORM))
        stdscr.refresh()
        stdscr.getch()
        return "connect"

def _clear_scan_state() -> None:
    """Wipe all scan-related state. Called whenever the attached process changes."""
    state["scan_results"]   = _make_addr_array()   # must stay array.array('Q')
    state["scan_values"]    = None
    state["scan_dropped"]   = set()
    state["scan_history"]   = deque(maxlen=5)
    state["scan_pid"]       = None
    state["scan_truncated"] = False
    state["scan_unknown"]   = False
    with _map_cache_lock:
        _map_cache.clear()


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
                (label == "Next Scan"    and not state["scan_results"]) or
                (label == "Results"      and not state["scan_results"]) or
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

# ── scan UI ───────────────────────────────────────────────────────────────────

def _run_scan_with_progress(stdscr, thread_fn, total_label: str,
                             cancel_event: threading.Event,
                             progress: dict, w: int) -> bool:
    """
    Spin the progress-bar loop while `thread_fn` runs in a daemon thread.
    Returns True if the scan completed normally, False if cancelled.
    """
    t = threading.Thread(target=thread_fn, daemon=True)
    t.start()

    spinner = ["|", "/", "-", "\\"]
    spin_i  = 0
    stdscr.nodelay(True)
    try:
        while t.is_alive():
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

    results = progress["results"] or _make_addr_array()
    state["scan_history"]  = deque(maxlen=5)
    state["scan_dropped"]  = set()
    state["scan_results"]  = results
    state["scan_values"]   = progress.get("values")   # None for exact scans
    state["scan_pid"]      = state["pid"]
    state["scan_truncated"] = progress.get("truncated", False)
    state["scan_unknown"]  = unknown_mode
    add_log(f"{'Unknown' if unknown_mode else 'First'} scan "
            f"w={width} aligned={aligned}: {len(results):,} candidates")

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
    if not state["scan_results"]:
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

        new_addrs  = progress["results"] or _make_addr_array()
        new_values = progress["values"]  or _make_addr_array()

        # Save undo snapshot: include values so undo restores the full baseline.
        state["scan_history"].append(
            (_make_addr_array(prev_addrs), set(state["scan_dropped"]),
             _make_addr_array(prev_values)))
        state["scan_results"] = new_addrs
        state["scan_values"]  = new_values
        state["scan_dropped"] = state["scan_dropped"] & set(new_addrs)

        add_log(f"Next scan ({mode_lbl}): {len(new_addrs):,} remain")
        tip = ("Perfect! Use Results (R)."
               if len(new_addrs) <= 10
               else "Still many — trigger another change and scan again.")
        undo_hint = ""
        if state["scan_history"]:
            prev_snap = state["scan_history"][-1][0]
            undo_hint = f"  (U to undo — restores {len(prev_snap):,} candidates)"
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

        results = progress["results"] or _make_addr_array()
        state["scan_history"].append(
            (_make_addr_array(prev_addrs), set(state["scan_dropped"]), None))
        state["scan_results"] = results
        state["scan_values"]  = None
        state["scan_dropped"] = state["scan_dropped"] & set(results)

        add_log(f"Next scan val={val}: {len(results)} remain")
        tip = ("Perfect! Use Results (R)."
               if len(results) <= 10 else "Still many — change value and scan again.")
        undo_hint = (f"  (U to undo — restores {len(state['scan_history'][-1][0])} candidates)"
                     if state["scan_history"] else "")
        message_box(stdscr,
            [f"{len(results)} results remain.", "", tip, undo_hint],
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
    if not results:
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
                if not results:
                    break
                sel = min(sel, len(results) - 1)
            elif key in (ord('u'), ord('U')):
                if state["scan_history"]:
                    snap = state["scan_history"].pop()
                    prev_results, prev_dropped = snap[0], snap[1]
                    prev_vals = snap[2] if len(snap) > 2 else None
                    state["scan_results"] = prev_results
                    state["scan_dropped"] = prev_dropped
                    state["scan_values"]  = prev_vals
                    results = state["scan_results"]
                    with cache_lock:
                        val_cache.clear()
                    sel = 0; offset = 0
                    add_log(f"Undo: restored {len(results)} candidates")
            elif key in (ord('m'), ord('M')):
                # Force map-cache flush: useful when the game reallocated memory
                # without a PID change (e.g. level reload, NG+).
                with _map_cache_lock:
                    _map_cache.clear()
                with cache_lock:
                    val_cache.clear()
                add_log("Map cache flushed — next scan/write will re-fetch regions", "warn")
            elif key in (ord('q'), ord('Q')):
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

    # Verify address is inside a writable mapped region (fail-CLOSED: surfaces error to user)
    map_err = _validate_addr_in_maps(state["ip"], state["pid"], addr, width)
    if map_err:
        if not confirm_box(stdscr, f"{map_err}\nFreeze anyway?", "Unmapped Address"):
            return

    safe_addstr(stdscr, 15, 3,
        f"Freezing {hex(addr)} = {val} for {secs}s  (every {int(interval*1000)}ms)",
        color(C_WARN) | curses.A_BOLD)
    safe_addstr(stdscr, 16, 3, "Press Q to stop early.", color(C_NORM))
    stdscr.refresh()

    # Run the write loop in a background thread so the UI redraws responsively.
    # Freeze uses one fresh ps5_write() per tick — the WRITE command is a
    # stateless request/response pair; persistent multi-write sessions are not
    # guaranteed compatible across ps5debug forks.
    stop_event   = threading.Event()
    write_errors = [0]
    deadline     = time.time() + secs

    def _freeze_worker():
        while time.time() < deadline and not stop_event.is_set():
            if not ps5_write(state["ip"], state["pid"], addr, data):
                write_errors[0] += 1
            stop_event.wait(interval)   # interruptible sleep

    worker = threading.Thread(target=_freeze_worker, daemon=True)
    worker.start()

    stdscr.nodelay(True)
    try:
        while worker.is_alive():
            elapsed   = time.time() - (deadline - secs)
            frac      = min(elapsed / secs, 1.0)
            remaining = max(0, int(deadline - time.time()))
            h, w      = stdscr.getmaxyx()
            safe_addstr(stdscr, 18, 3, f"Time left: {remaining:3d}s  ", color(C_OK))
            draw_progress_bar(stdscr, 19, 3, min(w - 8, 50), frac,
                              f"  {int(frac * 100)}%")
            if write_errors[0]:
                safe_addstr(stdscr, 20, 3,
                    f"Write errors: {write_errors[0]}  (connection issue?)",
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
    if not state["scan_results"] and not state["scan_history"]:
        message_box(stdscr, ["No scan results to clear."], "Clear", C_WARN)
        return
    n = len(state["scan_results"])
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
