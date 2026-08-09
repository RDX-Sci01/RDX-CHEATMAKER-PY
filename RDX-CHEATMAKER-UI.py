#!/usr/bin/env python3

"""
Python Cheat Maker with Terminal UI

Usage:
    python3 RDX-CHEATMAKER-UI.py
"""

import array as _array
import bisect
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

# ── Phase 3: Numba JIT — optional, graceful fallback ─────────────────────────
# numba accelerates the relational comparison filter by compiling it to
# native LLVM code with OpenMP parallelism, bypassing the Python GIL
# entirely for the pure-integer inner loop.
#
# Install:  pip install numba
# Expected speedup vs NumPy boolean expression: 4–16× on multi-core machines.
# Falls back to the existing pure-NumPy path if numba is not installed —
# zero loss of correctness, only the parallel speedup is unavailable.
try:
    import numba as nb
    from numba import prange as _prange
    _NUMBA_OK = True

    @nb.njit(parallel=True, cache=True, fastmath=True)
    def _nb_relational_mask(cur_vals, prv_vals, mode_id: int, delta: int):
        """
        Compute a boolean mask for the relational filter in parallel.

        mode_id values (must match RELATIONAL_MODE_IDS below):
            0 = decreased       cur < prv
            1 = increased       cur > prv
            2 = changed         cur != prv
            3 = unchanged       cur == prv
            4 = decreased by    cur == prv - delta
            5 = increased by    cur == prv + delta

        prange compiles to a parallel for-loop (OpenMP on Linux/macOS,
        TBB on Windows); all iterations are independent and GIL-free.
        """
        n    = len(cur_vals)
        mask = np.empty(n, dtype=nb.boolean)
        for i in _prange(n):
            c = cur_vals[i]
            p = prv_vals[i]
            if   mode_id == 0: mask[i] = c < p
            elif mode_id == 1: mask[i] = c > p
            elif mode_id == 2: mask[i] = c != p
            elif mode_id == 3: mask[i] = c == p
            elif mode_id == 4: mask[i] = c == p - delta
            else:              mask[i] = c == p + delta
        return mask

    @nb.njit(parallel=True, cache=True, fastmath=True)
    def _nb_search_aligned(data: np.ndarray, target: np.ndarray,
                           base_addr: np.uint64, width: np.int32) -> np.ndarray:
        """
        Parallel aligned search — stride = width.

        prange divides the aligned-offset space across all CPU cores.
        Each thread checks only offsets that are multiples of `width`
        relative to `base_addr`, so no alignment filtering is needed
        inside the loop — every candidate is already aligned by construction.

        Two-pass design (count then collect) avoids any shared-append lock.
        """
        n      = len(data)
        w      = int(width)
        counts = np.zeros(n, dtype=np.int32)

        # Numba's prange supports a unit step; map each parallel iteration to
        # an aligned byte offset explicitly.
        first = int((-base_addr) % np.uint64(w))
        candidate_count = max(0, (n - first - w) // w + 1)
        for k in _prange(candidate_count):
            i = first + k * w
            match = True
            for j in range(w):
                if data[i + j] != target[j]:
                    match = False
                    break
            if match:
                counts[i] = 1

        hit_offsets = np.where(counts)[0]
        return (base_addr + hit_offsets.astype(np.uint64)).astype(np.uint64)

    @nb.njit(parallel=True, cache=True, fastmath=True)
    def _nb_search_unaligned(data: np.ndarray, target: np.ndarray,
                             base_addr: np.uint64, width: np.int32) -> np.ndarray:
        """
        Parallel unaligned search — stride = 1.

        Previous behaviour (bug): _nb_search used step=width unconditionally,
        so unaligned scans skipped every non-aligned offset — silently missing
        values stored at misaligned addresses.  This kernel fixes that by
        iterating every byte offset from 0 to len(data)-width.

        Performance note: stride=1 means N threads each check N/cores offsets.
        At 12 cores on a 32 MB chunk with width=4, each thread handles
        ~700 K offsets.  Total work is 4× more than aligned (32 M vs 8 M
        comparisons per chunk) but still GIL-free and fully parallelised.
        Expected throughput: ~600–800 MB/s effective (network-bound anyway).
        """
        n      = len(data)
        w      = int(width)
        counts = np.zeros(n, dtype=np.int32)

        # stride=1: check every byte offset — this is the key difference
        # from _nb_search_aligned which strides by w.
        candidate_count = max(0, n - w + 1)
        for i in _prange(candidate_count):
            match = True
            for j in range(w):
                if data[i + j] != target[j]:
                    match = False
                    break
            if match:
                counts[i] = 1

        hit_offsets = np.where(counts)[0]
        return (base_addr + hit_offsets.astype(np.uint64)).astype(np.uint64)

    # Keep the old name as an alias for any external callers.
    def _nb_search(data, target, base_addr, width, check_align):
        if check_align:
            return _nb_search_aligned(data, target, base_addr, width)
        else:
            return _nb_search_unaligned(data, target, base_addr, width)

except ImportError:
    _NUMBA_OK = False
    _nb_relational_mask = None
    _nb_search          = None

# Map relational mode strings to integer IDs for the Numba kernel.
RELATIONAL_MODE_IDS: dict = {
    "decreased":    0,
    "increased":    1,
    "changed":      2,
    "unchanged":    3,
    "decreased by": 4,
    "increased by": 5,
}

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
CMD_PROC_SCAN  = 0xBDAA0009
CMD_PROC_AUTH  = 0xBDAACCFF
CMD_TURBO_CAPS = 0xBDAACC10
CMD_TURBO_START = 0xBDAACC11
CMD_TURBO_COUNT = 0xBDAACC12
CMD_TURBO_GET   = 0xBDAACC13
CMD_TURBO_END   = 0xBDAACC14
PROC_AUTH_MAGIC = 0xBB40E64D
# STATUS_SUCCESS / STATUS_ERROR: bit-swapped wire values produced by the server's
# net_send_int32() helper.  Clients compare raw wire bytes directly.
STATUS_SUCCESS = 0x80000000
STATUS_ERROR   = 0xF0000001
PS5_PORT       = 744

# A TurboScan list session lives on its TCP connection.  Retaining that
# connection lets subsequent exact scans use the payload's resident COUNT
# command instead of reading millions of candidate addresses back over LAN.
_turbo_session_lock = threading.RLock()
_turbo_session = None

WIDTH_FMT   = {1: 'B', 2: '<H', 4: '<I', 8: '<Q'}
VALID_WIDTHS = [1, 2, 4, 8]
WIDTH_LABEL  = {1: "byte (u8)", 2: "uint16", 4: "uint32", 8: "uint64"}
WIDTH_MAX    = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF, 8: 0xFFFFFFFFFFFFFFFF}

# PS5 user-space is 0x0001 – 0x00007FFF_FFFF_FFFF.
# Static/module segments on PS5 (orbis-ld output) are loaded in the low
# portion of that range.  Heap and mmap() regions occupy the upper portion.
# These thresholds are heuristics — confirmed against multiple retail titles.
_STATIC_ADDR_MAX  = 0x0000_0100_0000_0000   # below ≈ 1 TB → likely module/static
_HEAP_NAME_HINTS  = frozenset({"", "anon", "heap", "stack", "scePthread",
                                "SceKernelPrimary", "SceLibcInternal"})

# Maximum pointer-chain depth.  Chains longer than 6 are almost never seen
# in retail titles and exponential scan cost grows quickly beyond this.
MAX_CHAIN_DEPTH   = 6

# How many pointer-scan results to keep per depth level before filtering.
# Raising this finds more candidates but uses more RAM and takes longer.
MAX_PTR_RESULTS   = 500_000


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
#                prev_dropped: set,
#                prev_truncated: bool)

def _undo_entry_bytes(entry: tuple) -> int:
    """Byte size of a single undo entry (removed_addrs + removed_values)."""
    a, v, _, _ = entry
    nb = a.nbytes if isinstance(a, np.ndarray) else len(a) * 8
    nv = v.nbytes if isinstance(v, np.ndarray) else 0
    return nb + nv

def _history_bytes() -> int:
    """Total RAM consumed by all live undo levels, in bytes."""
    return sum(_undo_entry_bytes(e) for e in state["scan_history"])

def _push_undo(removed_addrs: np.ndarray,
               removed_values: Optional[np.ndarray],
               prev_dropped: set,
               prev_truncated: bool = False) -> None:
    """
    Push one undo delta.  If the resulting history would exceed
    HISTORY_RAM_CAP_MB, evict the oldest entry first.
    """
    new_entry   = (removed_addrs, removed_values, prev_dropped,
                   bool(prev_truncated))
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
_map_cache:      dict = {}           # {(ip, pid): (timestamp, maps_list)}
_map_cache_lock = threading.Lock()
_MAP_CACHE_TTL  = 30.0               # general scan cache TTL
_WRITE_MAP_CACHE_TTL = 10.0          # shorter TTL for writes/freezes

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
        if t and t.is_alive():
            # Keep the signal asserted and retain the reference.  Clearing it
            # here would allow a still-blocked worker to resume later.
            _freeze_thread = t
        else:
            _freeze_thread = None
            _freeze_stop.clear()

state = {
    "ip":           "",
    "connected":    False,
    "session":      0,                    # increments after every reconnect
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
    "scan_engine": "auto",          # auto / turbo / console / host
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

def ps5_connect(ip: str, timeout: float = 15.0) -> socket.socket:
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
        s.settimeout(timeout)
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

def _auth_keystream(length: int) -> bytes:
    """ps5debug-NG auth.c LFSR keystream, seeded with 200/300/400/500."""
    s1, s2, s3, s4 = 200, 300, 400, 500
    out = bytearray(length)
    mask = 0xFFFFFFFF
    for i in range(length):
        s1 = ((s1 << 18) & 0xFFF80000) ^ ((s1 ^ ((s1 << 6) & mask)) >> 13)
        s2 = ((s2 << 2)  & 0xFFFFFFE0) ^ ((s2 ^ ((s2 << 2) & mask)) >> 27)
        s3 = ((s3 << 7)  & 0xFFFFF800) ^ ((s3 ^ ((s3 << 13) & mask)) >> 21)
        s4 = ((s4 << 13) & 0xFFF00000) ^ ((s4 ^ ((s4 << 3) & mask)) >> 12)
        s1 &= mask; s2 &= mask; s3 &= mask; s4 &= mask
        out[i] = (s1 ^ s2 ^ s3 ^ s4) & 0xFF
    return bytes(out)

def ps5_auth_scanner(ip: str) -> None:
    """Enable ps5debug-NG's authenticated iterative/TurboScan commands."""
    s = ps5_connect(ip)
    try:
        body = struct.pack("<II", PROC_AUTH_MAGIC, 2)
        s.sendall(cmd_header(CMD_PROC_AUTH, len(body)) + body)
        if not check_ok(s):
            raise RuntimeError("scanner authentication rejected")
        length = struct.unpack("<H", recv_exact(s, 2))[0]
        if length <= 0 or length > 256:
            raise RuntimeError(f"invalid auth challenge length: {length}")
        challenge = recv_exact(s, length)
        key = _auth_keystream(length)
        s.sendall(bytes(a ^ b for a, b in zip(challenge, key)))
        if not check_ok(s):
            raise RuntimeError("scanner authentication response rejected")
    finally:
        s.close()

def ps5_turboscan_caps(ip: str) -> tuple:
    """Return (version, engines, max_threads) from the read-only capability probe."""
    s = ps5_connect(ip)
    try:
        s.sendall(cmd_header(CMD_TURBO_CAPS))
        if not check_ok(s):
            raise RuntimeError("TurboScan unavailable")
        version, engines, max_threads, _ = struct.unpack("<IIII", recv_exact(s, 16))
        return version, engines, max_threads
    finally:
        s.close()

def _recv_exact_cancel(s: socket.socket, n: int,
                       cancel_event: Optional[threading.Event]) -> bytes:
    """recv_exact variant that remains cancellable while the server is busy."""
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    old_timeout = s.gettimeout()
    s.settimeout(0.5)
    try:
        while pos < n:
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("scan cancelled")
            try:
                got = s.recv_into(view[pos:], n - pos)
            except socket.timeout:
                continue
            if not got:
                raise ConnectionError("PS5 disconnected")
            pos += got
    finally:
        s.settimeout(old_timeout)
    return bytes(buf)

def ps5_scan_exact_server(ip: str, pid: int, value: int, width: int,
                          regions: list, aligned: bool = True,
                          cancel_event=None,
                          progress_cb=None) -> np.ndarray:
    """Run ps5debug's legacy exact-value scanner on-console."""
    # VM maps can contain overlapping entries after page-table augmentation.
    # Merge them so bisecting by start cannot select a short overlapping entry
    # and incorrectly reject an address covered by a longer one.
    merged = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    regions = merged
    if not regions:
        return np.empty(0, dtype=_NP_ADDR_DTYPE)

    value_type = {1: 0, 2: 2, 4: 4, 8: 6}[width]
    target = struct.pack(WIDTH_FMT[width], value)
    body = struct.pack("<IBBI", pid, value_type, 0, len(target))
    starts = [r[0] for r in regions]
    total_bytes = max(sum(end - start for start, end in regions), 1)
    if progress_cb:
        progress_cb(0, total_bytes)

    s = ps5_connect(ip)
    found = []
    truncated = False
    try:
        s.sendall(cmd_header(CMD_PROC_SCAN, len(body)) + body)
        status = struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0]
        if status != STATUS_SUCCESS:
            raise RuntimeError("server-side scan command rejected")
        s.sendall(target)
        status = struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0]
        if status != STATUS_SUCCESS:
            raise RuntimeError("server-side scan failed")

        _last_progress_n = 0
        while True:
            addr = struct.unpack("<Q", _recv_exact_cancel(s, 8, cancel_event))[0]
            if addr == 0xFFFFFFFFFFFFFFFF:
                break
            i = bisect.bisect_right(starts, addr) - 1
            if i < 0 or addr + width > regions[i][1]:
                continue
            if aligned and addr % width != 0:
                continue
            found.append(addr)
            # Emit progress every 10 000 addresses so the UI bar advances
            # smoothly instead of sitting at 0% until the sentinel arrives.
            # Use MAX_SCAN_RESULTS as denominator so the bar tracks fill-level.
            if progress_cb and len(found) - _last_progress_n >= 10_000:
                progress_cb(min(len(found), MAX_SCAN_RESULTS), MAX_SCAN_RESULTS)
                _last_progress_n = len(found)
            if len(found) >= MAX_SCAN_RESULTS:
                truncated = True
                if cancel_event:
                    cancel_event.truncated = True
                break
    finally:
        s.close()

    if progress_cb:
        progress_cb(total_bytes, total_bytes)
    result = np.asarray(found, dtype=_NP_ADDR_DTYPE)
    add_log(f"Console exact scan: {len(result):,} matches"
            f"{' (truncated)' if truncated else ''}")
    return result

def _turbo_fetch_addresses(s: socket.socket, width: int, count: int,
                           cancel_event=None) -> np.ndarray:
    """Fetch up to the UI cap from the current resident TurboScan session."""
    wanted = min(int(count), MAX_SCAN_RESULTS)
    out = np.empty(wanted, dtype=_NP_ADDR_DTYPE)
    pos = 0
    page = 10_000
    while pos < wanted:
        take = min(page, wanted - pos)
        get_body = struct.pack("<III", pos, take, 0)
        s.sendall(cmd_header(CMD_TURBO_GET, len(get_body)) + get_body)
        if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
            raise RuntimeError("TurboScan result fetch rejected")
        header = struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0]
        actual = header & 0x7FFFFFFF
        has_first = bool(header & 0x80000000)
        rec_size = 8 + width * (3 if has_first else 2)
        raw = _recv_exact_cancel(s, actual * rec_size, cancel_event)
        if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
            raise RuntimeError("TurboScan result fetch incomplete")
        if actual == 0:
            break
        records = np.ndarray(shape=(actual,), dtype={
            "names": ["addr", "rest"],
            "formats": ["<u8", f"V{rec_size - 8}"],
            "offsets": [0, 8], "itemsize": rec_size}, buffer=raw)
        out[pos:pos + actual] = records["addr"]
        pos += actual
    return out[:pos]


def _close_turbo_session() -> None:
    """Best-effort release of the console-resident result list and socket."""
    global _turbo_session
    with _turbo_session_lock:
        session, _turbo_session = _turbo_session, None
        if not session:
            return
        s = session["socket"]
        try:
            s.sendall(cmd_header(CMD_TURBO_END))
            recv_exact(s, 4)
        except Exception:
            pass
        finally:
            s.close()


def ps5_scan_exact_turbo(ip: str, pid: int, value: int, width: int,
                         regions: list, aligned: bool = True,
                         cancel_event=None, progress_cb=None) -> np.ndarray:
    """Segmented SIMD exact scan, retaining its server-resident result list."""
    global _turbo_session
    _close_turbo_session()
    ps5_auth_scanner(ip)
    version, engines, _ = ps5_turboscan_caps(ip)
    required = 0x01 | 0x04 | 0x10  # SIMD, resident results, segmented scans
    if version < 1 or (engines & required) != required:
        raise RuntimeError("required TurboScan engines are unavailable")

    combined = []
    for start, end in sorted(regions):
        if combined and start <= combined[-1][1]:
            combined[-1] = (combined[-1][0], max(combined[-1][1], end))
        else:
            combined.append((start, end))
    merged = []
    for start, end in combined:
        if aligned:
            start += (-start) % width
            if start + width > end:
                continue
        while start < end:
            max_piece = 0xFFFFFFFF
            if aligned and width > 1:
                # Keep the next segment's base on the same absolute alignment;
                # otherwise the server's per-segment stepping can skip values.
                max_piece -= max_piece % width
            piece_end = min(end, start + max_piece)
            merged.append((start, piece_end))
            start = piece_end
    if not merged:
        return np.empty(0, dtype=_NP_ADDR_DTYPE)
    if len(merged) > 1_048_576:
        raise RuntimeError("too many TurboScan segments")

    target = struct.pack(WIDTH_FMT[width], value)
    value_type = {1: 0, 2: 2, 4: 4, 8: 6}[width]
    flags = 0x02 | 0x10  # server resident + segmented
    if (engines & 0x02) and os.environ.get("RDX_TURBO_ALIAS", "1") != "0":
        flags |= 0x01
        if engines & 0x100:
            flags |= 0x80
    # scan_turbo.c uses `alignment ? alignment : value_length`; therefore 0
    # means width-aligned.  A byte step of 1 is the true unaligned mode.
    alignment = width if aligned else 1
    body = struct.pack("<IQIBBBII", pid, 0, 0, value_type, 0,
                       alignment, len(target), flags)
    total_bytes = sum(end - start for start, end in merged)
    if progress_cb:
        progress_cb(0, max(total_bytes, 1))

    s = ps5_connect(ip)
    session_created = False
    retain_session = False
    try:
        s.sendall(cmd_header(CMD_TURBO_START, len(body)) + body)
        if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
            raise RuntimeError("TurboScan start rejected")
        s.sendall(target)
        if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
            raise RuntimeError("TurboScan value rejected")

        segment_data = bytearray(struct.pack("<I", len(merged)))
        for start, end in merged:
            segment_data.extend(struct.pack("<QI", start, end - start))
        s.sendall(segment_data)

        # Heartbeat: advance progress 0→90% while blocking on the server scan.
        # The server gives no per-byte feedback for segmented scans, so we use
        # a time-based estimate.  The heartbeat stops when the recv completes.
        _hb_stop = threading.Event()
        if progress_cb and total_bytes > 0:
            def _heartbeat():
                # Assume ~500 MB/s effective throughput as a baseline estimate.
                # Advances done from 0 to 99% of total_bytes at that rate,
                # then holds at 99% until the real completion fires.
                # The final 1% is covered by the post-scan ramp in
                # _run_scan_with_progress so the bar reaches 100% visibly.
                _estimated_secs = max(total_bytes / (500 * 1024 * 1024), 0.5)
                _step = max(int(total_bytes * 0.01), 1)   # 1% per tick
                _interval = _estimated_secs / 99           # tick every 1% of est. time
                _done = 0
                _cap  = int(total_bytes * 0.99)
                while not _hb_stop.wait(max(_interval, 0.05)):
                    _done = min(_done + _step, _cap)
                    try:
                        progress_cb(_done, total_bytes)
                    except Exception:
                        pass
            _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            _hb_thread.start()
        else:
            _hb_thread = None

        try:
            stored, count = struct.unpack("<IQ", _recv_exact_cancel(s, 12, cancel_event))
        finally:
            # Stop heartbeat on every exit path — normal completion, cancel,
            # or network error — so the daemon thread never outlives this scope.
            _hb_stop.set()
            if _hb_thread is not None:
                _hb_thread.join(timeout=0.5)

        if not stored:
            # Segmented decline includes an empty stream sentinel before status.
            _recv_exact_cancel(s, 8, cancel_event)
            _recv_exact_cancel(s, 4, cancel_event)
            raise RuntimeError("TurboScan resident result capacity exceeded")
        session_created = True
        if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
            raise RuntimeError("TurboScan did not complete")

        out = _turbo_fetch_addresses(s, width, count, cancel_event)
        pos = len(out)

        if count > MAX_SCAN_RESULTS and cancel_event:
            cancel_event.truncated = True
        if progress_cb:
            progress_cb(max(total_bytes, 1), max(total_bytes, 1))
        add_log(f"TurboScan exact: {pos:,}/{count:,} matches, "
                f"{total_bytes / 1_073_741_824:.2f} GiB scanned")
        with _turbo_session_lock:
            _turbo_session = {"socket": s, "ip": ip, "pid": pid,
                              "width": width, "count": int(count),
                              "engines": engines}
        retain_session = True
        return out
    finally:
        if not retain_session:
            if session_created:
                try:
                    s.sendall(cmd_header(CMD_TURBO_END))
                    recv_exact(s, 4)
                except Exception:
                    pass
            s.close()


def ps5_scan_next_turbo(ip: str, pid: int, value: int, width: int,
                         cancel_event=None, progress_cb=None) -> np.ndarray:
    """Refine the complete resident result set on-console using COUNT."""
    global _turbo_session
    with _turbo_session_lock:
        session = _turbo_session
        if not session or any((session["ip"] != ip, session["pid"] != pid,
                               session["width"] != width)):
            raise RuntimeError("no matching resident TurboScan session")
        s = session["socket"]
        old_count = int(session["count"])
        target = struct.pack(WIDTH_FMT[width], value)
        value_type = {1: 0, 2: 2, 4: 4, 8: 6}[width]
        flags = 0x02  # TS_SERVER_RESIDENT
        if session["engines"] & 0x200:
            flags |= 0x100  # TS_RESCAN_ALIASING
        body = struct.pack("<IQBBII", pid, 0, value_type, 0,
                           len(target), flags)
        try:
            if progress_cb:
                progress_cb(0, max(old_count, 1))
            s.sendall(cmd_header(CMD_TURBO_COUNT, len(body)) + body)
            if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
                raise RuntimeError("TurboScan rescan rejected")
            s.sendall(target)

            # COUNT may report progress; list-backed sessions commonly send
            # only the sentinel, so the UI still remains responsive/spinning.
            while True:
                scanned = struct.unpack("<Q", _recv_exact_cancel(s, 8, cancel_event))[0]
                if scanned == 0xFFFFFFFFFFFFFFFF:
                    break
                if progress_cb:
                    progress_cb(min(int(scanned), old_count), max(old_count, 1))
            new_count = struct.unpack("<Q", _recv_exact_cancel(s, 8, cancel_event))[0]
            if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
                raise RuntimeError("TurboScan rescan failed")
            session["count"] = int(new_count)
            out = _turbo_fetch_addresses(s, width, new_count, cancel_event)
            if cancel_event is not None:
                cancel_event.truncated = new_count > MAX_SCAN_RESULTS
            if progress_cb:
                progress_cb(max(old_count, 1), max(old_count, 1))
            add_log(f"Turbo next scan: {len(out):,}/{new_count:,} remain")
            return out
        except Exception:
            # A cancelled or failed command leaves stream framing uncertain.
            # Drop the session so subsequent scans safely use host fallback.
            _turbo_session = None
            s.close()
            raise

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

def ps5_write(ip: str, pid: int, addr: int, data: bytes,
              cancel_event: Optional[threading.Event] = None,
              timeout: float = 15.0) -> bool:
    """Two-phase write with up to _UI_MAX_RETRIES retries."""
    for attempt in range(_UI_MAX_RETRIES):
        if cancel_event and cancel_event.is_set():
            return False
        s = None
        try:
            s = ps5_connect(ip, timeout=timeout)
            body = struct.pack("<IQI", pid, addr, len(data))
            s.sendall(cmd_header(CMD_PROC_WRITE, len(body)) + body)
            if not check_ok(s):
                return False
            s.sendall(data)
            return check_ok(s)
        except Exception:
            if attempt < _UI_MAX_RETRIES - 1:
                delay = 0.1 * (attempt + 1)
                if cancel_event:
                    if cancel_event.wait(delay):
                        return False
                else:
                    time.sleep(delay)
        finally:
            if s:
                try: s.close()
                except Exception: pass
    return False

def ps5_write_verified(ip: str, pid: int, addr: int, data: bytes) -> tuple:
    """Write and immediately read back; returns (ack, verified, actual_bytes)."""
    if not ps5_write(ip, pid, addr, data):
        return False, False, None
    try:
        actual = ps5_read(ip, pid, addr, len(data))
    except Exception:
        return True, None, None
    return True, actual == data, actual

# ── batch reader for scan_next ────────────────────────────────────────────────

def ps5_read_batch(ip: str, pid: int, addrs: np.ndarray, width: int,
                   cancel_event=None, progress_cb=None) -> tuple:
    """
    Read `width` bytes at each address using NEXT_WORKERS parallel sockets.

    Phase 2 — Coalesced batch (window) reads
    ─────────────────────────────────────────
    Previous architecture: one ps5debug READ command per address.
    At 500 K addresses × 1–2 ms RTT = 8–16 minutes — completely impractical.

    New architecture: sort the candidate addresses, then detect contiguous
    aligned runs.  Each run whose span ≤ COALESCE_MAX bytes is read with a
    single ps5debug READ command instead of one per element.  For a typical
    post-first-scan candidate set (addresses clustered in a few heap regions)
    this reduces the number of network round-trips by 5–20×.

    After the window read, a NumPy view+slice decodes all values in the
    window without any Python loop, giving the same zero-Python-loop
    guarantee as before while also cutting RTT overhead.

    For sparse / isolated addresses (gap > COALESCE_MAX) the code falls
    back to individual reads — identical to the previous version.

    COALESCE_MAX = 4096 bytes (1 KB per candidate at most).
        Rationale: a contiguous run of N candidates at width=4 occupies
        N×4 bytes.  Typical next-scan sets are densely clustered within a
        few KB of each other (e.g. an array of health values).  Reading a
        4 KB window that contains 100 candidates costs the same RTT as one
        individual read but recovers 100 values in one shot.

    Return type: (live_addrs: ndarray[uint64], live_vals: ndarray[uint_w])
        Same as before — callers unchanged.
    """
    NEXT_WORKERS  = 12          # Phase 1: raised from 6 → 12
    COALESCE_MAX  = 8 * 1024 * 1024
    MAX_BYTES_PER_CANDIDATE = 4096
    LOCAL_BUF     = 256         # thread-local accumulation size before flush

    if len(addrs) == 0:
        return (np.empty(0, dtype=_NP_ADDR_DTYPE),
                np.empty(0, dtype=_NP_VALUE_DTYPE[width]))

    started = time.monotonic()

    # ── Phase 2: build coalesced work items ────────────────────────────────────
    # Sort addresses so contiguous runs are adjacent.
    sorted_idx  = np.argsort(addrs, kind='stable')
    sorted_addr = addrs[sorted_idx]

    # Each work item: (window_base_addr, window_size_bytes, local_addr_array)
    # The local_addr_array holds the individual candidate addresses within the
    # window so we can slice the right offsets after the single window read.
    #
    # Bound windows by their total span, not merely by adjacent gaps.  The old
    # grouping joined an entire dense region into one huge run and then fell
    # back to one request per address when that run exceeded 4 KiB.
    work_items: list = []
    rs = 0
    while rs < len(sorted_addr):
        if cancel_event and cancel_event.is_set():
            return (np.empty(0, dtype=_NP_ADDR_DTYPE),
                    np.empty(0, dtype=_NP_VALUE_DTYPE[width]))
        first_addr = int(sorted_addr[rs])
        max_last   = first_addr + COALESCE_MAX - width
        span_re = int(np.searchsorted(sorted_addr, max_last, side='right'))
        # Take the largest prefix whose window does not read more than a
        # bounded amount of unrelated memory.  Density is not monotonic when
        # gaps exist, so evaluate all possible endpoints with NumPy instead
        # of using a binary search that can reject a later dense prefix.
        endpoints = sorted_addr[rs:span_re]
        spans = endpoints.astype(np.uint64) - np.uint64(first_addr + width) \
                + np.uint64(2 * width)
        budgets = np.arange(1, len(endpoints) + 1, dtype=np.uint64) \
                  * np.uint64(MAX_BYTES_PER_CANDIDATE)
        valid_ends = np.flatnonzero(spans <= budgets)
        re = rs + (int(valid_ends[-1]) + 1 if len(valid_ends) else 1)
        last_addr = int(sorted_addr[re - 1])
        span      = last_addr - first_addr + width
        work_items.append(('window', first_addr, span,
                           sorted_addr[rs:re].copy()))
        rs = re

    total     = len(addrs)
    val_dtype = _NP_VALUE_DTYPE[width]
    fmt       = WIDTH_FMT[width]

    # Pre-allocated output buffers — worst case all reads succeed.
    out_addrs = np.empty(total, dtype=_NP_ADDR_DTYPE)
    out_vals  = np.empty(total, dtype=val_dtype)
    write_ptr = [0]
    work_ptr  = [0]
    ptr_lock  = threading.Lock()
    done_ctr  = [0]
    connected_workers = [0]
    worker_errors: list = []

    def _worker():
        sock = None
        local_addrs = np.empty(LOCAL_BUF, dtype=_NP_ADDR_DTYPE)
        local_vals  = np.empty(LOCAL_BUF, dtype=val_dtype)
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
            try:
                sock = _ScanSocket(ip, pid)
                with ptr_lock:
                    connected_workers[0] += 1
            except Exception as exc:
                with ptr_lock:
                    worker_errors.append(str(exc))
                return
            while True:
                if cancel_event and cancel_event.is_set():
                    break
                with ptr_lock:
                    if work_ptr[0] >= len(work_items):
                        break
                    item = work_items[work_ptr[0]]
                    work_ptr[0] += 1

                kind = item[0]
                if kind == 'window':
                    _, base_addr, span, cand_addrs = item
                    try:
                        window = sock.read(base_addr, span, cancel_event)
                        if len(window) == span:
                            # Vectorised decode: interpret entire window as packed
                            # integers, then gather only the candidate offsets.
                            # Replaces the Python for-loop (one struct.unpack per
                            # candidate) with a single NumPy advanced-index gather.
                            #
                            # Old (Python loop over cand_addrs):
                            #   for ca in cand_addrs:  # O(N) GIL-held iterations
                            #       off = int(ca) - base_addr
                            #       val = struct.unpack(fmt, window[off:off+w])[0]
                            #
                            # New: build byte indices for every candidate and
                            # combine their little-endian bytes in NumPy.
                            byte_offsets = cand_addrs.astype(np.int64) - base_addr
                            # Decode at each candidate's exact byte offset.  In
                            # unaligned scan sessions these offsets need not be
                            # multiples of `width`, and the window length need
                            # not be divisible by `width`.
                            valid    = ((byte_offsets >= 0) &
                                        (byte_offsets + width <= len(window)))
                            v_off    = byte_offsets[valid].astype(np.intp)
                            v_addrs  = cand_addrs[valid]
                            # An overlapping typed view decodes values at every
                            # possible byte offset without allocating a 2-D
                            # byte-index matrix.
                            all_vals = np.ndarray(
                                shape=(len(window) - width + 1,),
                                dtype=f'<u{width}', buffer=window, strides=(1,))
                            v_vals = all_vals[v_off].astype(val_dtype, copy=False)
                            # Write into local buffer in one slice assignment;
                            # flush in chunks of LOCAL_BUF if needed.
                            n_v = len(v_addrs)
                            i   = 0
                            while i < n_v:
                                space = LOCAL_BUF - local_n
                                take  = min(space, n_v - i)
                                local_addrs[local_n:local_n + take] = v_addrs[i:i + take]
                                local_vals [local_n:local_n + take] = v_vals [i:i + take]
                                local_n += take
                                i       += take
                                if local_n == LOCAL_BUF:
                                    _flush()
                        n_done = len(cand_addrs)
                    except Exception:
                        n_done = len(cand_addrs)
                else:
                    # 'single' fallback — identical to old per-address read
                    _, addr, _w, _ = item
                    n_done = 1
                    try:
                        data = sock.read(addr, _w, cancel_event)
                        if len(data) == _w:
                            local_addrs[local_n] = addr
                            local_vals [local_n] = struct.unpack(fmt, data)[0]
                            local_n += 1
                            if local_n == LOCAL_BUF:
                                _flush()
                    except Exception:
                        pass

                with ptr_lock:
                    done_ctr[0] += n_done
                    if progress_cb:
                        progress_cb(done_ctr[0], total)
        finally:
            _flush()
            if sock:
                sock.close()

    workers = [threading.Thread(target=_worker, daemon=True)
               for _ in range(min(NEXT_WORKERS, max(1, len(work_items))))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    if connected_workers[0] == 0:
        detail = worker_errors[0] if worker_errors else "unknown connection error"
        raise ConnectionError(f"No batch-read worker could connect: {detail}")
    for error in worker_errors:
        add_log(f"Batch-read worker unavailable: {error}", "warn")

    n = write_ptr[0]
    elapsed = max(time.monotonic() - started, 1e-9)
    add_log(f"Batch read: {n:,}/{total:,} candidates via "
            f"{len(work_items):,} windows in {elapsed:.2f}s "
            f"({n / elapsed:,.0f} candidates/s)")
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
    _POOL_MAX   = 6
    _pool_lock  = threading.Lock()
    _pool       = {}  # {(ip, pid): [socket, ...]}

    def __init__(self, ip: str, pid: int):
        self.ip  = ip
        self.pid = pid
        self._s: Optional[socket.socket] = None
        self._from_pool = False
        # Pre-built mutable request buffer; addr field patched in read()
        self._req = bytearray(self._HDR_SIZE)
        struct.pack_into("<III", self._req,  0,
                         CMD_MAGIC, CMD_PROC_READ, 16)   # header
        struct.pack_into("<I",   self._req, 12, pid)     # pid (fixed)
        # addr at offset 16, length at offset 24 — set per-call
        self._connect()

    @classmethod
    def clear_pool(cls, ip=None, pid=None):
        with cls._pool_lock:
            for key in list(cls._pool):
                if ip is not None and key[0] != ip: continue
                if pid is not None and key[1] != pid: continue
                for sock in cls._pool.pop(key, []):
                    try: sock.close()
                    except Exception: pass

    def _connect(self):
        if self._s:
            try: self._s.close()
            except Exception: pass
            self._s = None
        key = (self.ip, self.pid)
        with self._pool_lock:
            bucket = self._pool.get(key)
            if bucket:
                self._s = bucket.pop()
                self._from_pool = True
                if not bucket: self._pool.pop(key, None)
                return
        self._s = ps5_connect(self.ip)
        self._from_pool = False

    def read(self, addr: int, length: int,
             cancel_event: Optional[threading.Event] = None) -> bytes:
        """Read `length` bytes from `addr`, reconnecting on transient failure."""
        # Patch addr and length directly into the pre-built bytearray.
        # sendall accepts bytearray natively — no bytes() copy needed.
        struct.pack_into("<QI", self._req, 16, addr, length)
        for attempt in range(self.MAX_RETRIES):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("scan cancelled")
            try:
                if self._s is None:
                    self._connect()
                self._s.sendall(self._req)   # zero-copy: no bytes() allocation
                if not check_ok(self._s):
                    raise RuntimeError("read rejected")
                return _recv_exact_cancel(self._s, length, cancel_event)
            except Exception as exc:
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("scan cancelled") from exc
                add_log(f"scan read err (attempt {attempt+1}/{self.MAX_RETRIES}) "
                        f"@ {hex(addr)}: {exc}", "warn")
                try: self._s.close()
                except Exception: pass
                self._s = None
                self._from_pool = False
                if attempt == self.MAX_RETRIES - 1:
                    raise
                delay = 0.1 * (attempt + 1)
                if cancel_event:
                    if cancel_event.wait(delay):
                        raise InterruptedError("scan cancelled")
                else:
                    time.sleep(delay)

    def close(self):
        if not self._s:
            return
        sock, self._s = self._s, None
        key = (self.ip, self.pid)
        with self._pool_lock:
            bucket = self._pool.setdefault(key, [])
            if len(bucket) < self._POOL_MAX:
                bucket.append(sock)
                self._from_pool = True
                return
        try: sock.close()
        except Exception: pass
        self._from_pool = False

def _get_maps_cached(ip: str, pid: int) -> list:
    """
    Return ps5_maps() with a 30-second cache.  Consecutive scans on the same
    process reuse the map rather than paying an extra RTT before each scan.
    Invalidated automatically when the endpoint or pid changes, or TTL expires.
    """
    now = time.time()
    with _map_cache_lock:
        cache_key = (ip, pid)
        entry = _map_cache.get(cache_key)
        if entry and (now - entry[0]) < _MAP_CACHE_TTL:
            return entry[1]
    maps = ps5_maps(ip, pid)
    with _map_cache_lock:
        _map_cache.clear()          # only cache one pid at a time
        _map_cache[cache_key] = (now, maps)
    return maps


def scan_first(ip: str, pid: int, value: int, width: int = 4,
               aligned: bool = True, progress_cb=None,
               cancel_event=None,
               writable_only: bool = True) -> np.ndarray:
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
    started = time.monotonic()
    if cancel_event is None:
        cancel_event = threading.Event()
        cancel_event.truncated = False
    # Validate via struct.pack — handles both signed and unsigned types correctly.
    # The old `value < 0` guard blocked all signed-type scans (int8/16/32/64).
    try:
        target = struct.pack(WIDTH_FMT[width], value)
    except struct.error:
        raise ValueError(
            f"Value {value} out of range for {WIDTH_LABEL.get(width, str(width))}")
    maps = _get_maps_cached(ip, pid)

    CHUNK        = 0x2000000   # 32 MB per request — amortises RTT 8× more than 4 MB;
                               # each request now carries ~8M uint32 values, so a
                               # single RTT covers far more heap than before.
                               # RAM budget: 12 workers × 4 slots × 32 MB = 1.5 GB
                               # max in-flight; bounded by QUEUE_DEPTH.
    SCAN_WORKERS = 12          # 12 parallel readers to saturate the GbE link
                               # (ps5debug is server-side; 12 concurrent TCP streams
                               # keeps the scanner from stalling on any one RTT).
    QUEUE_DEPTH  = SCAN_WORKERS * 4   # 48 slots × 32 MB = 1.5 GB max in-flight
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
    # Phase 4a: sort regions largest-first.  Benefits:
    #   • Workers pick up the biggest chunks immediately → CPU/network both
    #     saturated from the first second rather than warming up on tiny regions.
    #   • If the user cancels early, the most data-rich regions were scanned
    #     first, improving the chance of having found the target already.
    #   • UX: progress bar advances fastest in the first few seconds.
    scannable.sort(key=lambda r: r['end'] - r['start'], reverse=True)
    total_bytes = max(sum(r['end'] - r['start'] for r in scannable), 1)

    if not scannable:
        if progress_cb:
            progress_cb(1, 1)
        add_log("First scan: no eligible memory regions", "warn")
        return np.empty(0, dtype=_NP_ADDR_DTYPE)

    # Select the scanner engine from the UI setting.
    engine = state.get("scan_engine", "auto")
    selected_ranges = sorted((r['start'], r['end']) for r in scannable)
    if engine in ("auto", "turbo") and os.environ.get("RDX_TURBO_SCAN", "1") != "0":
        try:
            result = ps5_scan_exact_turbo(ip, pid, value, width, selected_ranges, aligned,
                                          cancel_event, progress_cb)
            add_log(f"Turbo first scan completed in {max(time.monotonic()-started,1e-9):.2f}s")
            return result
        except InterruptedError:
            raise
        except Exception as exc:
            add_log(f"TurboScan unavailable ({exc})", "warn")
            if engine == "turbo":
                raise
    if engine in ("auto", "console"):
        try:
            result = ps5_scan_exact_server(ip, pid, value, width, selected_ranges, aligned,
                                           cancel_event, progress_cb)
            add_log(f"Console first scan completed in {max(time.monotonic()-started,1e-9):.2f}s")
            return result
        except InterruptedError:
            raise
        except Exception as exc:
            add_log(f"Console scan unavailable ({exc})", "warn")
            if engine == "console":
                raise

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
            # Include enough look-ahead bytes to find a value that begins at
            # the end of this chunk and continues into the next one.
            csz = min(CHUNK + width - 1, size - off)
            work.append((r['start'] + off, csz))
            off += CHUNK

    # Pre-build O(1) lookup dict for partial-read detection in the searcher.
    # The old code used next((s for a,s in work if a==addr), None) which is
    # O(N) per chunk — for a 400 MB scan with 32 MB chunks that is 12 linear
    # scans × 12 chunks = 144 comparisons per chunk, totally avoidable.
    work_sizes: dict = {addr: csz for addr, csz in work}

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
    connected_readers = [0]

    # ── reader thread ─────────────────────────────────────────────────────────
    def _reader():
        sock = None
        try:
            try:
                sock = _ScanSocket(ip, pid)
                with reader_err_lock:
                    connected_readers[0] += 1
            except Exception as exc:
                with reader_err_lock:
                    reader_err.append(f"scan connection failed: {exc}")
                return
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
            if sock:
                sock.close()

    # ── searcher ──────────────────────────────────────────────────────────────
    # Fast path: _nb_search (Numba, parallel across all CPU cores, GIL-free).
    # Fallback: bytes.find() (C-level Boyer-Moore-Horspool, ~2400 MB/s, 1 core).
    # Both produce identical results; Numba wins on multi-core machines because
    # the search across a 32 MB chunk is split across N threads simultaneously.
    step        = width if aligned else 1
    check_align = aligned and width > 1
    target_np   = np.frombuffer(target, dtype=np.uint8)
    target_zero = (target == b'\x00' * width)
    target_b0   = target[0:1]

    if _NUMBA_OK:
        if aligned:
            def _search_chunk(data: bytes, addr: int) -> list:
                arr  = np.frombuffer(data, dtype=np.uint8)
                hits = _nb_search_aligned(arr, target_np,
                                          np.uint64(addr),
                                          np.int32(width))
                return hits.tolist()
        else:
            def _search_chunk(data: bytes, addr: int) -> list:
                arr  = np.frombuffer(data, dtype=np.uint8)
                hits = _nb_search_unaligned(arr, target_np,
                                            np.uint64(addr),
                                            np.int32(width))
                return hits.tolist()
    else:
        def _search_chunk(data: bytes, addr: int) -> list:
            hits = []
            pos  = 0
            while True:
                p = data.find(target, pos)
                if p == -1:
                    break
                if check_align and (addr + p) % width != 0:
                    pos = p + 1
                    continue
                hits.append(addr + p)
                pos = p + step
            return hits

    def _search_all():
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
            expected_csz = work_sizes.get(addr)
            if expected_csz is not None and csz < expected_csz:
                add_log(f"Partial read @ {hex(addr)}: got {csz} of {expected_csz} B — skipped", "warn")
                done_bytes[0] += csz
                if progress_cb:
                    progress_cb(done_bytes[0], total_bytes)
                continue
            # Zero-page fast-path: if the target's first byte is absent in
            # the whole chunk, skip without searching (O(N) C scan, ~2400 MB/s).
            if not target_zero and target_b0 not in data:
                done_bytes[0] += csz
                if progress_cb:
                    progress_cb(done_bytes[0], total_bytes)
                continue
            for h in _search_chunk(data, addr):
                found.append(h)
                if len(found) >= MAX_SCAN_RESULTS:
                    add_log(f"Result cap ({MAX_SCAN_RESULTS:,}) hit — scan truncated", "warn")
                    if cancel_event:
                        cancel_event.set()
                        cancel_event.truncated = True
                    while True:
                        try:
                            chunk_queue.get_nowait()
                        except _queue.Empty:
                            break
                    return
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
    if connected_readers[0] == 0:
        detail = reader_err[0] if reader_err else "unknown connection error"
        raise ConnectionError(f"No first-scan reader could connect: {detail}")
    for msg in reader_err:
        add_log(msg, "warn")

    # Convert plain list → ndarray once here; avoids O(N) reallocations that
    # array.array or repeated np.append would cause inside the hot loop.
    result = np.array(found, dtype=_NP_ADDR_DTYPE)
    elapsed = max(time.monotonic() - started, 1e-9)
    processed_mb = min(done_bytes[0], total_bytes) / 1_048_576
    add_log(f"First-scan transfer/search: {processed_mb:.1f} MiB "
            f"in {elapsed:.2f}s ({processed_mb / elapsed:.1f} MiB/s)")
    return result


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
    if state.get("scan_engine", "auto") in ("auto", "turbo"):
        try:
            return ps5_scan_next_turbo(ip, pid, value, width, cancel_event, progress_cb)
        except InterruptedError:
            raise
        except Exception as exc:
            add_log(f"Resident Turbo rescan unavailable ({exc}); using host filter", "warn")
            if state.get("scan_engine") == "turbo":
                raise

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
    started = time.monotonic()
    if cancel_event is None:
        cancel_event = threading.Event()
        cancel_event.truncated = False
    maps = _get_maps_cached(ip, pid)

    CHUNK        = 0x2000000   # 32 MB — matches scan_first for consistent RTT amortisation
    SCAN_WORKERS = 12          # 12 parallel readers
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
    # Phase 4a: sort largest-first — same rationale as scan_first.
    scannable.sort(key=lambda r: r['end'] - r['start'], reverse=True)
    total_bytes = max(sum(r['end'] - r['start'] for r in scannable), 1)

    if not scannable:
        if progress_cb:
            progress_cb(1, 1)
        add_log("Unknown scan: no eligible memory regions", "warn")
        return (np.empty(0, dtype=_NP_ADDR_DTYPE),
                np.empty(0, dtype=_NP_VALUE_DTYPE[width]))

    work: list = []
    for r in scannable:
        size = r['end'] - r['start']
        off  = 0
        while off < size:
            # Overlap adjacent reads by width-1 bytes so unaligned values that
            # cross a chunk boundary are included in the snapshot.
            csz = min(CHUNK + width - 1, size - off)
            work.append((r['start'] + off, csz))
            off += CHUNK

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
    connected_readers = [0]

    def _reader():
        sock = None
        try:
            try:
                sock = _ScanSocket(ip, pid)
                with reader_err_lock:
                    connected_readers[0] += 1
            except Exception as exc:
                with reader_err_lock:
                    reader_err.append(f"snapshot connection failed: {exc}")
                return
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
            if sock:
                sock.close()

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
        total_so_far = 0
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

            # Phase 4b — zero-page fast-path.
            # PS5 heap regions often contain large runs of zero pages (unused
            # allocator arenas, zeroed BSS, stack guard pages).  A single
            # bytes.count(b'\x00') call is C-level ~2400 MB/s and lets us skip
            # the entire frombuffer + arange + append pipeline for any chunk
            # that is entirely zero — which is both the most common case for
            # sparse heaps and the least interesting for value hunting.
            # We only skip when ALL bytes are zero AND 0 is not the target
            # width-boundary value (which would make zero pages valuable).
            if data.count(b'\x00') == csz:
                # Entire chunk is zero.  Record only if the user specifically
                # wants zero values (unusual but valid — skip only when ALL
                # widths of zero are uninteresting, i.e. always in a snapshot
                # because we record all values and prune later via relational).
                # For the snapshot path we keep zero pages: the user may have
                # a zero health value.  Instead we only skip TRULY empty pages:
                # if the chunk is shorter than `width` nothing can be extracted.
                if csz < width:
                    done_bytes[0] += csz
                    if progress_cb:
                        progress_cb(done_bytes[0], total_bytes)
                    continue
                # Non-trivial zero page — still extract (values may be 0).

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

            # For aligned scans, stride by width in the value array;
            # for unaligned, every byte offset matters so we must work at byte level.
            if aligned:
                # Memory mappings are usually page-aligned, but the protocol
                # does not guarantee it.  Start at the first absolute address
                # divisible by the selected width.
                lead       = (-addr) % width
                aligned_n  = (csz - lead) // width
                vals_slice = np.frombuffer(
                    data[lead:lead + aligned_n * width], dtype=f'<u{width}')
                n_out      = len(vals_slice)
                first_addr = addr + lead
                addrs_out  = np.arange(first_addr, first_addr + n_out * width, width,
                                       dtype=_NP_ADDR_DTYPE)
            else:
                # Overlapping typed view: one value at every byte offset, with
                # all decoding performed in NumPy rather than Python.
                n_out      = csz - width + 1
                offsets    = np.arange(n_out, dtype=np.intp)
                vals_slice = np.ndarray(
                    shape=(n_out,), dtype=f'<u{width}',
                    buffer=data, strides=(1,))
                addrs_out  = (addr + offsets).astype(_NP_ADDR_DTYPE)

            # Enforce the result cap
            remaining = MAX_SCAN_RESULTS - total_so_far
            if n_out > remaining:
                addrs_out  = addrs_out[:remaining]
                vals_slice = vals_slice[:remaining]
                found_addrs.append(addrs_out)
                found_values.append(vals_slice.astype(_NP_VALUE_DTYPE[width]))
                total_so_far += len(addrs_out)
                add_log(f"Unknown scan cap ({MAX_SCAN_RESULTS:,}) hit"
                        " — snapshot truncated", "warn")
                if cancel_event:
                    cancel_event.set()
                    cancel_event.truncated = True
                # Drain the queue so readers can unblock and exit, then stop.
                # Do NOT use get_nowait() alone — it discards sentinels and
                # leaves sentinels_received < n_workers, causing an infinite loop.
                # Instead just return; readers will drain naturally via cancel.
                return

            found_addrs.append(addrs_out)
            found_values.append(vals_slice.astype(_NP_VALUE_DTYPE[width]))
            total_so_far += len(addrs_out)

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
    if connected_readers[0] == 0:
        detail = reader_err[0] if reader_err else "unknown connection error"
        raise ConnectionError(f"No snapshot reader could connect: {detail}")
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
    elapsed = max(time.monotonic() - started, 1e-9)
    processed_mb = min(done_bytes[0], total_bytes) / 1_048_576
    add_log(f"Unknown-scan transfer/decode: {processed_mb:.1f} MiB "
            f"in {elapsed:.2f}s ({processed_mb / elapsed:.1f} MiB/s)")
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

    # ── vectorised comparison — Phase 3: Numba parallel kernel or NumPy ────────
    cur = live_vals
    prv = prv_vals
    if _NUMBA_OK and mode in RELATIONAL_MODE_IDS:
        # Phase 3 fast path: Numba compiles _nb_relational_mask to native
        # LLVM code on first call (cached to disk after that).  Subsequent
        # calls skip compilation entirely.  The parallel=True flag enables
        # OpenMP on Linux/macOS and TBB on Windows — all CPU cores, GIL-free.
        #
        # "decreased by" / "increased by" use uint64 arithmetic; cast both
        # arrays so the Numba kernel works with a uniform dtype.
        keep = _nb_relational_mask(
            cur.astype(np.uint64),
            prv.astype(np.uint64),
            RELATIONAL_MODE_IDS[mode],
            int(delta) & 0xFFFF_FFFF_FFFF_FFFF,
        ).astype(bool)
    else:
        # Phase 3 fallback: pure NumPy (always correct, single-threaded SIMD).
        if   mode == "decreased":
            keep = cur < prv
        elif mode == "increased":
            keep = cur > prv
        elif mode == "changed":
            keep = cur != prv
        elif mode == "unchanged":
            keep = cur == prv
        elif mode == "decreased by":
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

def _is_static_region(region: dict) -> bool:
    """
    Heuristic: return True when a map entry looks like a static/module segment
    rather than a heap or anonymous allocation.

    Criteria (any one is sufficient):
      • address below _STATIC_ADDR_MAX  (module segments load low)
      • region name is a known module or game binary  (non-empty, not a hint)
      • region is NOT writable but IS readable  (typical for .data/.rodata)

    The combination catches the vast majority of static segments while
    excluding stack, heap, and large anonymous mmaps.
    """
    name  = region.get("name", "")
    start = region.get("start", 0)
    prot  = region.get("prot", 0)
    PROT_READ  = 0x1
    PROT_WRITE = 0x2
    if start < _STATIC_ADDR_MAX:
        return True
    if name and name not in _HEAP_NAME_HINTS:
        return True
    if (prot & PROT_READ) and not (prot & PROT_WRITE):
        return True
    return False


def ps5_read_pointer(ip: str, pid: int, addr: int) -> int:
    """
    Read a single uint64 pointer value from `addr`.
    Returns 0 on failure (invalid pointers are treated as null).
    """
    try:
        raw = ps5_read(ip, pid, addr, 8)
        if len(raw) == 8:
            return struct.unpack("<Q", raw)[0]
    except Exception:
        pass
    return 0


def _resolve_pointer_chain(ip: str, pid: int,
                            base_addr: int,
                            offsets: list) -> tuple:
    """
    Follow a pointer chain starting at `base_addr` and applying each offset
    in `offsets` in order.  Every level reads a uint64 pointer then adds the
    corresponding offset before proceeding to the next level.

    Algorithm (corrected — a pointer read occurs at every level including the
    last, so that depth=1 with offsets=[0x0] resolves to *base_addr + 0):
        current = read_u64(base_addr) + offsets[0]
        current = read_u64(current)   + offsets[1]
        ...
        final   = read_u64(current)   + offsets[-1]

    Returns (success: bool, final_addr: int, steps: list[int])
        steps contains the intermediate resolved addresses for debugging.
    """
    if not offsets:
        return True, base_addr, []

    steps   = []
    current = base_addr
    # Every offset requires a pointer read at the current address, then add.
    for offset in offsets:
        ptr_val = ps5_read_pointer(ip, pid, current)
        if ptr_val == 0 or ptr_val > _ADDR_MAX:
            return False, 0, steps
        current = ptr_val + offset
        steps.append(current)

    if current < _ADDR_MIN or current > _ADDR_MAX:
        return False, 0, steps
    return True, current, steps


def scan_for_pointer(ip: str, pid: int,
                     target_addr: int,
                     cancel_event=None,
                     progress_cb=None) -> np.ndarray:
    """
    Scan all readable memory for uint64 values equal to `target_addr`.

    This is functionally identical to scan_first() with width=8 and the target
    being the bytes of the address itself.  We call the existing scan_first()
    machinery directly rather than duplicating it.

    Returns ndarray[uint64] of addresses that contain a pointer to target_addr.
    """
    # Validate target is a plausible PS5 user-space address.
    if target_addr < _ADDR_MIN or target_addr > _ADDR_MAX:
        add_log(f"Pointer scan: target {hex(target_addr)} is not a valid "
                "user-space address", "warn")
        return np.empty(0, dtype=_NP_ADDR_DTYPE)

    add_log(f"Pointer scan: searching for pointers to {hex(target_addr)}")
    results = scan_first(
        ip, pid, target_addr,
        width=8,
        aligned=True,          # pointers are always 8-byte aligned on PS5
        progress_cb=progress_cb,
        cancel_event=cancel_event,
        writable_only=False,   # static regions may not be writable
    )
    add_log(f"Pointer scan: found {len(results):,} locations containing "
            f"{hex(target_addr)}")
    return results


def classify_pointer_results(ip: str, pid: int,
                              ptr_addrs: np.ndarray) -> tuple:
    """
    Split pointer scan results into (static_addrs, heap_addrs) using the
    current memory map.

    Returns:
        static_addrs : ndarray[uint64]  — addresses in static/module regions
        heap_addrs   : ndarray[uint64]  — addresses in heap/anon regions
        region_map   : dict {addr: region_name}  — for display
    """
    try:
        maps = _get_maps_cached(ip, pid)
    except Exception:
        # If we can't get the map, return everything as heap (conservative).
        return np.empty(0, dtype=_NP_ADDR_DTYPE), ptr_addrs, {}

    # Build sorted arrays of region boundaries for O(log N) lookup.
    starts = np.array([r["start"] for r in maps], dtype=np.uint64)
    ends   = np.array([r["end"]   for r in maps], dtype=np.uint64)
    order  = np.argsort(starts)
    starts = starts[order]
    ends   = ends[order]
    maps_s = [maps[i] for i in order]

    static_list  = []
    heap_list    = []
    region_map   = {}

    for addr in ptr_addrs.tolist():
        idx = int(np.searchsorted(starts, addr, side="right")) - 1
        if idx < 0 or addr >= ends[idx]:
            heap_list.append(addr)
            region_map[addr] = "unmapped"
            continue
        region = maps_s[idx]
        name   = region.get("name", "")
        region_map[addr] = name or "anon"
        if _is_static_region(region):
            static_list.append(addr)
        else:
            heap_list.append(addr)

    return (np.array(static_list, dtype=_NP_ADDR_DTYPE),
            np.array(heap_list,   dtype=_NP_ADDR_DTYPE),
            region_map)


# A pointer scan normally finds an exact pointer value.  That is not enough
# to find common struct-member pointers, where the memory cell contains
# (target_address - struct_offset).  The old _auto_pointer_offset() tried to
# infer the offset by reading the exact-match cell, which can only ever produce
# zero.  Keep the zero case cheap and use an explicit bounded sweep for the
# actual struct-offset case.
POINTER_STRUCT_SWEEP_MAX = 0x200
POINTER_STRUCT_SWEEP_STEP = 8

def _pointer_target_sweep(ip: str, pid: int, target_addr: int,
                          cancel_event=None) -> list:
    """Return (holder_address, struct_offset) hits for a target.

    For each aligned offset O, a holder containing ``target_addr - O`` is a
    valid pointer to ``target_addr`` at ``holder + O``.  The sweep is bounded
    to 0x200 bytes to avoid turning pointer scanning into an unbounded search.
    Results are deduplicated by holder/offset and capped like normal pointer
    scan results.
    """
    hits = []
    seen = set()
    max_off = min(POINTER_STRUCT_SWEEP_MAX, target_addr - _ADDR_MIN)
    for off in range(0, max_off + 1, POINTER_STRUCT_SWEEP_STEP):
        if cancel_event and cancel_event.is_set():
            break
        wanted = target_addr - off
        if wanted < _ADDR_MIN or wanted > _ADDR_MAX:
            continue
        ptr_addrs = scan_for_pointer(ip, pid, wanted,
                                     cancel_event=cancel_event,
                                     progress_cb=None)
        for addr in ptr_addrs.tolist():
            key = (int(addr), int(off))
            if key in seen:
                continue
            seen.add(key)
            hits.append(key)
            if len(hits) >= MAX_PTR_RESULTS:
                add_log("Pointer struct-offset sweep reached result cap", "warn")
                return hits
    return hits


def pointer_chain_scan(ip: str, pid: int,
                       target_addr: int,
                       max_depth: int = 3,
                       cancel_event=None,
                       progress_cb=None) -> list:
    """
    Multi-level pointer chain scanner.

    Iteratively scans for pointers TO the target, then pointers TO those
    pointers, up to `max_depth` levels.  At each level the scan is restricted
    to the writable+readable region so we don't spend time scanning ROM.

    Returns a list of PointerCandidate dicts:
        {
            "base":    int,          # static address that anchors the chain
            "offsets": [int, ...],   # offsets applied at each level
            "depth":   int,          # chain length
            "region":  str,          # name of the region containing base
            "static":  bool,         # True if base is in a static region
        }

    The list is sorted: static candidates first, then by ascending depth,
    then by ascending base address (most predictable / lowest first).
    """
    candidates = []
    # Level 0: addresses that directly hold target_addr.
    current_targets = {target_addr: []}   # {addr_being_pointed_to: offsets_so_far}

    for depth in range(1, max_depth + 1):
        if cancel_event and cancel_event.is_set():
            break
        next_targets = {}

        for pointed_to, chain_so_far in current_targets.items():
            if cancel_event and cancel_event.is_set():
                break
            if progress_cb:
                progress_cb(depth - 1, max_depth)

            # The first level gets the bounded struct-member sweep because
            # this is where a static pointer commonly anchors a heap object.
            # Deeper levels use exact pointer matches to avoid multiplying a
            # 65-scan sweep by every intermediate candidate.
            if depth == 1:
                pointer_hits = _pointer_target_sweep(
                    ip, pid, pointed_to, cancel_event=cancel_event)
            else:
                ptr_addrs = scan_for_pointer(
                    ip, pid, pointed_to, cancel_event=cancel_event,
                    progress_cb=None)
                pointer_hits = [(int(a), 0) for a in ptr_addrs.tolist()]
            if not pointer_hits:
                continue

            # Group/deduplicate by holder while retaining distinct offsets.
            # A holder may legitimately produce more than one candidate if the
            # same address can be interpreted through different pointer values.
            static_hits = []
            heap_hits = []
            try:
                maps = _get_maps_cached(ip, pid)
            except Exception:
                maps = []
            # Classification is independent of the offset, so build the map
            # index once instead of fetching/rebuilding it for every hit.
            starts = np.array([r["start"] for r in maps], dtype=np.uint64)
            ends = np.array([r["end"] for r in maps], dtype=np.uint64)
            if len(starts):
                order = np.argsort(starts)
                starts = starts[order]; ends = ends[order]
                maps_s = [maps[i] for i in order]
            else:
                maps_s = []
            for addr, step_offset in pointer_hits:
                mi = int(np.searchsorted(starts, addr, side="right")) - 1 if len(starts) else -1
                if mi >= 0 and addr < int(ends[mi]):
                    region = maps_s[mi]
                    rname = region.get("name", "") or "anon"
                    is_static = _is_static_region(region)
                else:
                    rname = "unmapped"; is_static = False
                (static_hits if is_static else heap_hits).append((addr, step_offset, rname))

            for addr, step_offset, rname in static_hits:
                candidates.append({
                    "base": addr,
                    "offsets": chain_so_far + [step_offset],
                    "depth": depth,
                    "region": rname,
                    "static": True,
                })

            # Track heap results as seeds for deeper levels.
            heap_list = heap_hits
            if len(next_targets) + len(heap_list) > MAX_PTR_RESULTS:
                heap_list = heap_list[:max(0, MAX_PTR_RESULTS - len(next_targets))]
                add_log(f"Depth {depth}: next_targets capped at {MAX_PTR_RESULTS:,}", "warn")
            for addr, step_offset, rname in heap_list:
                new_chain = chain_so_far + [step_offset]
                # Keep one chain per heap holder for the next depth.  If the
                # same holder was discovered through multiple offsets, prefer
                # the shorter chain; emitting every duplicate would explode the
                # search space without improving the next pointer level.
                old_chain = next_targets.get(addr)
                if old_chain is None or len(new_chain) < len(old_chain):
                    next_targets[addr] = new_chain
                if depth == max_depth:
                    candidates.append({
                        "base": addr,
                        "offsets": new_chain,
                        "depth": depth,
                        "region": rname,
                        "static": False,
                    })


        current_targets = next_targets
        if not current_targets:
            break

    if progress_cb:
        progress_cb(max_depth, max_depth)

    # Sort: static first, then by depth, then by base address.
    candidates.sort(key=lambda c: (not c["static"], c["depth"], c["base"]))
    add_log(f"Pointer chain scan: {len(candidates)} candidates "
            f"({sum(1 for c in candidates if c['static'])} static)")
    return candidates


def _validate_write_addr(addr: int) -> Optional[str]:
    """Return an error string if addr is outside safe user-space range, else None."""
    if addr < _ADDR_MIN:
        return f"Address {hex(addr)} is zero or negative — likely a mistake."
    if addr > _ADDR_MAX:
        return f"Address {hex(addr)} is in kernel space — write blocked."
    return None

def _validate_addr_in_maps(ip: str, pid: int, addr: int, length: int,
                           ttl_override: Optional[float] = None) -> Optional[str]:
    """Validate a writable address using a shorter cache for write paths."""
    try:
        ttl = _WRITE_MAP_CACHE_TTL if ttl_override is None else max(0.0, float(ttl_override))
        now = time.time()
        key = (ip, pid)
        with _map_cache_lock:
            entry = _map_cache.get(key)
            maps = entry[1] if entry and (now - entry[0]) < ttl else None
        if maps is None:
            maps = ps5_maps(ip, pid)
            with _map_cache_lock:
                _map_cache.clear()
                _map_cache[key] = (now, maps)
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
    fmt_val = (lambda v: hex(v)) if hex_values else (lambda v: str(v))
    cheat_list = []
    for c in cheats:
        is_pointer = "offsets" in c and c.get("offsets") is not None
        if is_pointer:
            entry = {
                "name":    c["name"],
                "type":    c["type"],                    # pointer_freeze / pointer_write
                "base":    hex(c["base"]) if isinstance(c["base"], int) else c["base"],
                "offsets": [hex(o) for o in c["offsets"]],
                "value":   fmt_val(c["value"]),
                "bytes":   c["width"],
            }
        else:
            entry = {
                "name":    c["name"],
                "type":    c["type"],
                "address": hex(c["address"]),
                "value":   fmt_val(c["value"]),
                "bytes":   c["width"],
            }
        cheat_list.append(entry)
    payload = {
        "title":     game_title,
        "titleid":   game_id,
        "version":   game_ver,
        "cheatList": cheat_list,
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
        # A successful connection starts a new protocol session.  PIDs can be
        # reused after rest mode/restart and can coincide across consoles, so
        # no address or map state from the prior session is safe to retain.
        # Increment session BEFORE clearing state so cheats stamped during
        # the clear cannot pass the subsequent session-match check.
        state["session"] += 1
        _clear_scan_state()
        state["pid"] = None
        state["proc_name"] = ""
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
    _close_turbo_session()
    state["scan_results"]   = _make_addr_array()
    state["scan_values"]    = None
    state["scan_dropped"]   = set()
    state["scan_history"]   = deque(maxlen=5)
    state["scan_pid"]       = None
    state["scan_truncated"] = False
    state["scan_unknown"]   = False
    with _map_cache_lock:
        _map_cache.clear()
    _ScanSocket.clear_pool()
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
    """Compact persistent application header: connection/process first, telemetry second."""
    _, w = stdscr.getmaxyx()
    conn = f"● {state['ip']}   {state['proc_name']}   PID {state['pid']}"
    safe_addstr(stdscr, 2, 3, conn[:max(w - 6, 0)], color(C_OK) | curses.A_BOLD)
    results = len(state["scan_results"])
    cheats = len(state["cheats"])
    width = WIDTH_LABEL.get(state["scan_width"], "?")
    scan_note = f"Results {results:,}   Cheats {cheats}   {width}"
    safe_addstr(stdscr, 3, 3, scan_note[:max(w - 6, 0)], color(C_WARN))


def _draw_toast(stdscr, cp=C_OK) -> None:
    """Draw the latest non-modal status message above the action bar."""
    h, w = stdscr.getmaxyx()
    if h < 4 or w < 8:
        return
    with _log_lock:
        last_entry = state["log"][-1] if state["log"] else None
    if last_entry:
        msg = f"✓ {last_entry['msg']}"
        if last_entry.get("level") == "warn":
            cp = C_WARN
            msg = f"⚠ {last_entry['msg']}"
        elif last_entry.get("level") == "error":
            cp = C_ERR
            msg = f"✗ {last_entry['msg']}"
        safe_addstr(stdscr, h - 2, 3, msg[:max(w - 6, 0)], color(cp))


def do_command_palette(stdscr) -> None:
    commands = [
        ("First Scan", "scan_first"), ("Next Scan", "scan_next"),
        ("Results", "results"), ("Cheat List", "cheat_list"),
        ("Pointer Scan", "pointer_scan"), ("Write Address", "write"),
        ("Freeze Address", "freeze"), ("Import Cheats", "import"),
        ("Export Cheats", "export"), ("Scan Settings", "scan_settings"),
        ("Logs", "log"), ("Clear Results", "clear"),
        ("Clear Scan History", "clear_history"), ("Change Process", "proc"),
        ("Verify Pointer", "ptr_verify"),
    ]
    query = ""
    sel = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "COMMAND PALETTE")
        safe_addstr(stdscr, 2, 3, f"> {query}_", color(C_ACC) | curses.A_BOLD)
        visible = max(1, min(12, h - 7))
        q = query.lower()
        matches = [c for c in commands if q in c[0].lower()]
        if matches:
            sel = min(sel, len(matches) - 1)
            for i, (label, _) in enumerate(matches[:visible]):
                attr = color(C_SEL) | curses.A_BOLD if i == sel else color(C_NORM)
                safe_addstr(stdscr, 4 + i, 4, ("▶ " if i == sel else "  ") + label, attr)
        else:
            safe_addstr(stdscr, 4, 4, "No matching commands.", color(C_WARN))
        draw_statusbar(stdscr, [("↑↓", C_NORM), ("Enter run", C_OK), ("Backspace", C_NORM), ("Esc", C_NORM)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols(); continue
        if key in (27, ord('q'), ord('Q')) and not query:
            return
        if key in (curses.KEY_UP,):
            sel = max(0, sel - 1)
        elif key in (curses.KEY_DOWN,):
            sel = min(max(len(matches) - 1, 0), sel + 1)
        elif key in (curses.KEY_ENTER, 10, 13) and matches:
            result = dispatch(stdscr, matches[sel][1])
            if result == "proc":
                return "proc"
            return
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            query = query[:-1]; sel = 0
        elif 32 <= key <= 126:
            query += chr(key); sel = 0


def _main_menu_entries():
    """Return the deliberately small primary workflow menu.

    Advanced/destructive utilities remain discoverable through the command
    palette instead of competing with the scan/results/cheat workflow.
    """
    return [
        ("S", "First Scan",  "scan_first",  C_NORM),
        ("N", "Next Scan",   "scan_next",   C_NORM),
        ("R", "Results",     "results",     C_ACC),
        ("C", "Cheat List",  "cheat_list",  C_NORM),
        ("P", "Pointer Scan", "pointer_scan", C_ACC),
        ("I", "Import Cheats", "import",    C_OK),
        ("E", "Export Cheats", "export",    C_OK),
        ("T", "Settings",    "scan_settings", C_ACC),
        ("L", "Logs",        "log",         C_NORM),
        ("Q", "Quit",        None,           C_ERR),
    ]


def screen_main(stdscr):
    """Compact primary navigation: workflow first, advanced tools elsewhere."""
    menu = _main_menu_entries()
    sel = 0
    stdscr.timeout(100)
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "RDX CHEAT MAKER")
        _draw_main_header(stdscr)

        # The primary workflow is intentionally linear and small.  Do not
        # expose low-frequency destructive/debug utilities here.
        sections = [
            ("SCAN", 0, 3),
            ("CHEATS", 3, 2),
            ("TOOLS", 5, 4),
        ]
        if w >= 78:
            col_x = [3, max(25, w // 3), max(50, (2 * w) // 3)]
            for title, start_i, count in sections:
                x = col_x[sections.index((title, start_i, count))]
                safe_addstr(stdscr, 5, x, title, color(C_TITLE) | curses.A_BOLD)
                for j in range(count):
                    i = start_i + j
                    if i >= len(menu):
                        continue
                    key, label, _, cp = menu[i]
                    unavailable = (
                        (label == "Next Scan" and not state["scan_results"]) or
                        (label == "Results" and not state["scan_results"]) or
                        (label == "Export Cheats" and not state["cheats"])
                    )
                    attr = (color(C_SEL) | curses.A_BOLD if i == sel else
                            color(C_NORM) | curses.A_DIM if unavailable else color(cp))
                    safe_addstr(stdscr, 7 + j, x,
                                f"[{key}] {label}"[:max(w - x - 2, 0)], attr)
        else:
            safe_addstr(stdscr, 5, 3, "WORKFLOW", color(C_TITLE) | curses.A_BOLD)
            for i, (key, label, _, cp) in enumerate(menu):
                unavailable = (
                    (label == "Next Scan" and not state["scan_results"]) or
                    (label == "Results" and not state["scan_results"]) or
                    (label == "Export Cheats" and not state["cheats"])
                )
                attr = (color(C_SEL) | curses.A_BOLD if i == sel else
                        color(C_NORM) | curses.A_DIM if unavailable else color(cp))
                safe_addstr(stdscr, 7 + i, 3,
                            f"[{key}] {label}"[:max(w - 6, 0)], attr)

        _draw_toast(stdscr)
        draw_statusbar(stdscr, [
            ("↑↓", C_NORM), ("Enter", C_OK), ("/ Commands", C_ACC),
            ("? Help", C_ACC), ("Q Quit", C_ERR)
        ])
        stdscr.refresh()
        key = stdscr.getch()
        if key == -1:
            continue
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols()
            continue
        if key == curses.KEY_UP:
            sel = max(0, sel - 1)
        elif key == curses.KEY_DOWN:
            sel = min(len(menu) - 1, sel + 1)
        elif key in (curses.KEY_ENTER, 10, 13):
            action = menu[sel][2]
            if action is None:
                return None
            result = dispatch(stdscr, action)
            if result == "proc":
                return "proc"
        elif key == ord('/'):
            result = do_command_palette(stdscr)
            if result == "proc":
                return "proc"
        elif key == ord('?'):
            do_help(stdscr)
        else:
            for k, _, action, _ in menu:
                if key in (ord(k.lower()), ord(k.upper())):
                    if action is None:
                        return None
                    result = dispatch(stdscr, action)
                    if result == "proc":
                        return "proc"
                    break

def do_help(stdscr) -> None:
    lines = [
        "Navigation   ↑↓ Select   Enter Run   Esc Back   Q Quit",
        "Global       / Command Palette   ? Help",
        "Scanning     S First Scan   N Next Scan   R Results",
        "Results      A Apply   C Cheat   P Pointer   N Next   M More",
        "Cheats       Enter Inspect   A Apply   D Delete",
        "Pointer      E Edit   T Test   S Save Cheat",
        "Tools        I Import   E Export   T Settings   L Logs",
        "",
        "Routine success messages stay in the status line;",
        "errors and destructive operations remain modal.",
    ]
    message_box(stdscr, lines, "Keyboard Help", C_ACC)

def dispatch(stdscr, action: str):
    actions = {
        "pointer_scan": do_pointer_scan,
        "ptr_verify":   do_ptr_verify_manual,
        "scan_first":    do_scan_first,
        "scan_next":     do_scan_next,
        "scan_settings": do_scan_settings,
        "results":       do_show_results,
        "write":         do_write,
        "cheat_list":    do_cheat_list,
        "export":        do_export,
        "import":        do_import,
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
                             progress: dict) -> bool:
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
    started = time.monotonic()
    t.start()

    # Clear the screen before the progress loop so that any preceding input
    # prompts (value, width, depth selectors) don't linger behind the bar.
    stdscr.clear()
    draw_border(stdscr, "SCANNING…")
    safe_addstr(stdscr, 2, 3, total_label, color(C_WARN))
    stdscr.refresh()

    spinner = ["|", "/", "-", "\\"]
    spin_i  = 0
    stdscr.nodelay(True)
    try:
        while t.is_alive():
            h, w = stdscr.getmaxyx()           # Issue #1: re-read on every tick
            frac = progress["done"] / max(progress["total"], 1)
            elapsed = max(time.monotonic() - started, 0.0)
            eta = ((elapsed / frac) - elapsed) if frac > 0 else None
            timing = f"elapsed {elapsed:.1f}s"
            if eta is not None and frac < 1.0:
                timing += f", ETA {max(eta, 0.0):.1f}s"
            # Erase the full previous status line first.  Without this, a
            # shorter count/ETA leaves stale trailing characters on screen.
            safe_addstr(stdscr, 9, 3, " " * max(w - 6, 0), color(C_NORM))
            safe_addstr(stdscr, 9, 3,
                f"{spinner[spin_i % 4]}  {total_label}  "
                f"{progress['done']:,} / {progress['total']:,}  "
                f"[{timing}]  [Esc=cancel]",
                color(C_WARN))
            draw_progress_bar(stdscr, 10, 3, min(w - 8, 60), frac,
                              f"  {int(frac * 100)}%")
            stdscr.refresh()
            time.sleep(0.1)
            spin_i += 1
            k = stdscr.getch()
            if k == curses.KEY_RESIZE:         # absorb resize, redraw frame
                curses.update_lines_cols()
                stdscr.clear()
                draw_border(stdscr, "SCANNING…")
                safe_addstr(stdscr, 2, 3, total_label, color(C_WARN))
            elif k == 27:
                cancel_event.set()
                safe_addstr(stdscr, 12, 3, "Cancelling…", color(C_ERR))
                stdscr.refresh()
    finally:
        stdscr.nodelay(False)

    t.join()

    # Post-scan: smoothly ramp the bar to 100% so it always visibly
    # reaches the end before the next screen appears.  Without this,
    # TurboScan's heartbeat caps at 95% and the final progress_cb(total,total)
    # may fire between two UI ticks leaving the bar stuck below 100%.
    if not cancel_event.is_set():
        total = max(progress["total"], 1)
        h, w = stdscr.getmaxyx()
        start_done = min(max(int(progress.get("done", 0)), 0), total)
        ramp_start = min(start_done, max(total - 1, 0))
        for step in range(1, 6):
            done = ramp_start + ((total - ramp_start) * step // 5)
            progress["done"] = done
            frac = done / total
            safe_addstr(stdscr, 9, 3, " " * max(w - 6, 0), color(C_NORM))
            safe_addstr(stdscr, 9, 3, f"✓  {total_label}  Completing…", color(C_OK) | curses.A_BOLD)
            draw_progress_bar(stdscr, 10, 3, min(w - 8, 60), frac, f"  {int(frac * 100)}%")
            stdscr.refresh()
            time.sleep(0.05)
        progress["done"] = total
        safe_addstr(stdscr, 9, 3, " " * max(w - 6, 0), color(C_NORM))
        safe_addstr(stdscr, 9, 3, f"✓  {total_label}  Complete!", color(C_OK) | curses.A_BOLD)
        draw_progress_bar(stdscr, 10, 3, min(w - 8, 60), 1.0, "  100%")
        stdscr.refresh()
        time.sleep(0.12)

    return not cancel_event.is_set()


def do_scan_settings(stdscr) -> None:
    options = ["Auto (Turbo → Console → Host)", "Turbo only", "Console only", "Host only"]
    keys = ["auto", "turbo", "console", "host"]
    current = state.get("scan_engine", "auto")
    selected = cycle_input(stdscr, "Scan engine: ", 5, 3, options, options[keys.index(current)])
    state["scan_engine"] = keys[options.index(selected)]
    add_log(f"Scan engine set to {state['scan_engine']}")
    message_box(stdscr, [f"Scan engine: {selected}"], "Scan Settings", C_OK)

def do_scan_first(stdscr) -> None:
    stdscr.clear()
    _, w = stdscr.getmaxyx()
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

    # A new first scan supersedes any retained resident refinement session,
    # including when this scan is unknown-value or uses a fallback engine.
    _close_turbo_session()
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

    ok = _run_scan_with_progress(stdscr, run, scan_label, cancel_event, progress)
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
         "   Additional matches are retained in the console scan session.",
         "   Run Next Scan (N) with a changed value to narrow results",
         "   across the complete result set.",
         ""]
        if progress["truncated"] else []
    )
    if unknown_mode:
        add_log(f"Snapshot complete — {len(results):,} candidates", "warn" if progress["truncated"] else "info")
        do_show_results(stdscr)
    else:
        add_log(f"First scan complete — {len(results):,} candidates", "warn" if progress["truncated"] else "info")
        # Results is the primary workflow: go there immediately after a scan.
        do_show_results(stdscr)


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
    _, w = stdscr.getmaxyx()
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
            stdscr, run_rel, f"Filtering ({mode_lbl})…", cancel_event, progress)
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
        if len(new_sorted) == 0:
            removed_mask = np.ones(len(prev_addrs), dtype=bool)
        else:
            ins          = np.searchsorted(new_sorted, prev_addrs)
            ins_clipped  = np.clip(ins, 0, len(new_sorted) - 1)
            removed_mask = new_sorted[ins_clipped] != prev_addrs
        removed_a    = prev_addrs[removed_mask]
        removed_v    = prev_values[removed_mask]
        _push_undo(removed_a, removed_v, set(state["scan_dropped"]),
                   state.get("scan_truncated", False))
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
        add_log(f"Next scan complete — {len(new_addrs):,} candidates remain",
                "info" if len(new_addrs) <= 10 else "warn")
        do_show_results(stdscr)

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

        cancel_event.truncated = bool(state.get("scan_truncated", False))
        progress = {"done": 0, "total": max(len(prev_addrs), 1),
                    "results": None, "error": None,
                    "truncated": bool(state.get("scan_truncated", False))}

        def run_exact():
            try:
                progress["results"] = scan_next(
                    state["ip"], state["pid"], val, width, prev_addrs,
                    cancel_event,
                    lambda d, t: progress.update(done=d, total=max(t, 1)))
                progress["truncated"] = getattr(cancel_event, "truncated", False)
            except Exception as exc:
                progress["error"] = str(exc)

        ok = _run_scan_with_progress(
            stdscr, run_exact, "Filtering addresses…", cancel_event, progress)
        if not ok:
            add_log("Next scan cancelled", "warn")
            return
        if progress["error"]:
            add_log(f"Next scan error: {progress['error']}", "error")
            message_box(stdscr, [f"Error: {progress['error']}"], "Scan Error", C_ERR)
            return

        results = progress["results"] if progress["results"] is not None else _make_addr_array()

        # Delta undo — same searchsorted approach as relational path.
        new_sorted  = np.sort(results)
        if len(new_sorted) == 0:
            removed_mask = np.ones(len(prev_addrs), dtype=bool)
        else:
            ins          = np.searchsorted(new_sorted, prev_addrs)
            ins_clipped  = np.clip(ins, 0, len(new_sorted) - 1)
            removed_mask = new_sorted[ins_clipped] != prev_addrs
        removed_a    = prev_addrs[removed_mask]
        _push_undo(removed_a, None, set(state["scan_dropped"]),
                   state.get("scan_truncated", False))
        del new_sorted, removed_mask, removed_a

        state["scan_results"] = results
        state["scan_values"]  = None
        state["scan_dropped"] = state["scan_dropped"] & set(results.tolist())
        state["scan_truncated"] = progress.get("truncated", False)

        hist_mb = _history_bytes() / 1_048_576
        add_log(f"Exact next scan val={val}: {len(results):,} remain, "
                f"undo {hist_mb:.1f} MB, RSS {_rss_mb():.0f} MB")

        tip = ("Perfect! Use Results (R)."
               if len(results) <= 10 else "Still many — change value and scan again.")
        undo_hint = ""
        if state["scan_history"]:
            last_delta = state["scan_history"][-1]
            undo_hint  = (f"  (U to undo — restores "
                          f"{len(results) + len(last_delta[0]):,} candidates)")
        add_log(f"Next scan complete — {len(results):,} candidates remain",
                "info" if len(results) <= 10 and not state["scan_truncated"] else "warn")
        do_show_results(stdscr)


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



def _results_more_menu(stdscr):
    """Low-frequency Results actions, kept out of the primary action bar."""
    options = [
        ("Write Address", "write"),
        ("Verify Pointer", "ptr_verify"),
        ("Undo Scan", "undo_results"),
        ("Flush Memory Maps", "flush_maps"),
        ("Clear Results", "clear"),
        ("Clear Scan History", "clear_history"),
        ("Logs", "log"),
    ]
    sel = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "RESULTS · MORE")
        safe_addstr(stdscr, 2, 3, "Advanced actions", color(C_WARN))
        visible = max(1, min(len(options), h - 7))
        for i, (label, _) in enumerate(options[:visible]):
            attr = color(C_SEL) | curses.A_BOLD if i == sel else color(C_NORM)
            safe_addstr(stdscr, 4 + i, 4,
                        ("▶ " if i == sel else "  ") + label,
                        attr)
        draw_statusbar(stdscr, [("↑↓", C_NORM), ("Enter run", C_OK),
                                ("Esc", C_NORM)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols()
            continue
        if key in (27, ord('q'), ord('Q')):
            return
        if key == curses.KEY_UP:
            sel = max(0, sel - 1)
        elif key == curses.KEY_DOWN:
            sel = min(len(options) - 1, sel + 1)
        elif key in (curses.KEY_ENTER, 10, 13):
            action = options[sel][1]
            if action == "undo_results":
                # Results already owns the undo stack; dispatching a separate
                # screen would lose the selected context.  Perform one undo
                # through the same path as U below by returning its action.
                return "undo"
            if action == "flush_maps":
                return "flush_maps"
            result = dispatch(stdscr, action)
            if result == "proc":
                return "proc"
            return


def do_show_results(stdscr) -> None:
    results = state["scan_results"]
    if len(results) == 0:
        return
    if state.get("scan_pid") not in (None, state["pid"]):
        add_log("Stale results blocked — PID changed; start a new First Scan", "warn")
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
    value_filter      = None

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
            trunc_warn = "  ⚠ CAPPED — additional matches not displayed" if state.get("scan_truncated") else ""
            safe_addstr(stdscr, 2, 3,
                f"Type: {wlabel}   Process: {state['proc_name']} (PID {state['pid']}){trunc_warn}",
                color(C_ERR) if trunc_warn else color(C_WARN))
            safe_addstr(stdscr, 3, 3,
                "↑↓/PgUp/PgDn navigate   G jump   F filter   Enter inspect   D drop   U undo   M more   Q back",
                color(C_NORM))

            split_view = w >= 92
            list_right = (w // 2 - 2) if split_view else (w - 3)
            shown = 0
            for idx in range(offset, min(offset + visible, len(results))):
                addr = results[idx]
                with cache_lock: vstr = val_cache.get(addr, "…")
                if value_filter is not None and vstr not in ("…", "?"):
                    try:
                        if int(vstr, 0) != value_filter: continue
                    except ValueError: continue
                marker = ">" if idx == sel else " "
                line = f"{marker} {idx+1:4d}  {hex(addr):<18} {vstr:>12}"
                attr = color(C_SEL) | curses.A_BOLD if idx == sel else color(C_NORM)
                safe_addstr(stdscr, 5 + shown, 2, line[:max(list_right-2, 1)].ljust(max(list_right-2, 1)), attr)
                shown += 1
                if shown >= visible: break

            if split_view:
                pane_x = w // 2 + 1
                safe_addstr(stdscr, 4, pane_x, "SELECTED ADDRESS", color(C_TITLE) | curses.A_BOLD)
                selected_addr = int(results[sel])
                with cache_lock: selected_value = val_cache.get(selected_addr, "…")
                safe_addstr(stdscr, 6, pane_x, f"Address   {hex(selected_addr)}", color(C_OK) | curses.A_BOLD)
                safe_addstr(stdscr, 7, pane_x, f"Current   {selected_value}", color(C_WARN))
                safe_addstr(stdscr, 8, pane_x, f"Type      {wlabel}", color(C_NORM))
                safe_addstr(stdscr, 10, pane_x, "A  Apply value", color(C_OK))
                safe_addstr(stdscr, 11, pane_x, "C  Create cheat", color(C_OK))
                safe_addstr(stdscr, 12, pane_x, "P  Pointer scan", color(C_ACC))
                safe_addstr(stdscr, 13, pane_x, "D  Drop result", color(C_ERR))
                safe_addstr(stdscr, 15, pane_x, "N  Next scan", color(C_ACC))
                safe_addstr(stdscr, 16, pane_x, "M  More actions", color(C_WARN))

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
                (f"{len(results):,} results", C_WARN),
                ("↑↓/PgUp/PgDn", C_NORM),
                ("Enter inspect", C_OK), ("A apply", C_OK), ("C cheat", C_OK),
                ("P pointer", C_ACC), ("N next", C_ACC),
                ("D drop", C_ERR), ("M more", C_WARN),
                (age_label, C_ERR if stale else C_ACC if is_refreshing else C_NORM),
                ("Q back", C_NORM),
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
            elif key == curses.KEY_PPAGE:
                sel = max(0, sel - visible)
            elif key == curses.KEY_NPAGE:
                sel = min(len(results)-1, sel + visible)
            elif key in (ord('g'), ord('G')):
                stdscr.nodelay(False)
                idx_s = input_box(stdscr, "Jump to result index: ", h-2, 3, 12, str(sel+1))
                stdscr.nodelay(True)
                try: sel = max(0, min(int(idx_s)-1, len(results)-1))
                except ValueError: pass
            elif key in (ord('f'), ord('F')):
                stdscr.nodelay(False)
                fs = input_box(stdscr, "Live value filter (blank clears): ", h-2, 3, 24,
                               "" if value_filter is None else str(value_filter))
                stdscr.nodelay(True)
                if not fs.strip(): value_filter = None
                else:
                    try: value_filter = int(fs, 0)
                    except ValueError:
                        message_box(stdscr, ["Enter decimal or hex (0x...)."], "Invalid Filter", C_ERR)
                        continue
                sel = 0; offset = 0
            elif key in (ord('a'), ord('A')) and len(results) > 0:
                stdscr.nodelay(False)
                addr = int(results[sel])
                try:
                    value_s = input_box(stdscr, "Apply value: ", h-2, 3, 20)
                    value = int(value_s, 0)
                    width = state["scan_width"]
                    if value < 0 or value > WIDTH_MAX[width]:
                        raise ValueError(f"Value out of range for {WIDTH_LABEL[width]}")
                    ack, verified, actual = _write_value_verified(state["ip"], state["pid"], addr, value, width)
                    if ack and verified:
                        add_log(f"Applied {value} → {hex(addr)} verified")
                    elif ack and verified is None:
                        add_log(f"Applied {value} → {hex(addr)} but read-back failed", "warn")
                    elif ack:
                        actual_val = struct.unpack(WIDTH_FMT[width], actual)[0]
                        add_log(f"Write mismatch {hex(addr)}: wanted {value}, read {actual_val}", "error")
                    else:
                        add_log(f"Write rejected at {hex(addr)}", "error")
                except Exception as exc:
                    add_log(f"Apply failed at {hex(addr)}: {exc}", "error")
                stdscr.nodelay(True)
            elif key in (curses.KEY_ENTER, 10, 13):
                stdscr.nodelay(False)
                selected_addr = int(results[sel])
                with cache_lock:
                    selected_live = val_cache.get(selected_addr, "…")
                _inspect_result(stdscr, selected_addr, selected_live)
                stdscr.nodelay(True)
                results = state["scan_results"]
                with cache_lock:
                    val_cache.clear()
            elif key in (ord('c'), ord('C')) and len(results) > 0:
                stdscr.nodelay(False)
                _add_cheat_at(stdscr, results[sel])
                stdscr.nodelay(True)
                results = state["scan_results"]
                with cache_lock:
                    val_cache.clear()
            elif key in (ord('p'), ord('P')) and len(results) > 0:
                stdscr.nodelay(False)
                do_pointer_scan(stdscr, int(results[sel]))
                stdscr.nodelay(True)
                results = state["scan_results"]
            elif key in (ord('n'), ord('N')):
                stdscr.nodelay(False)
                do_scan_next(stdscr)
                return

            elif key in (ord('m'), ord('M')):
                stdscr.nodelay(False)
                more_result = _results_more_menu(stdscr)
                stdscr.nodelay(True)
                if more_result == "proc":
                    return "proc"
                if more_result == "undo":
                    if state["scan_history"]:
                        entry = state["scan_history"].pop()
                        removed_a, removed_v, prev_dropped, prev_truncated = entry
                        cur_addrs = state["scan_results"]
                        prev_addrs = np.union1d(cur_addrs, removed_a)
                        if removed_v is not None and state.get("scan_values") is not None:
                            cur_v = state["scan_values"]
                            dtype = _NP_VALUE_DTYPE[state["scan_width"]]
                            prev_vals = np.zeros(len(prev_addrs), dtype=dtype)
                            prev_vals[np.searchsorted(prev_addrs, cur_addrs)] = cur_v
                            prev_vals[np.searchsorted(prev_addrs, removed_a)] = removed_v
                        else:
                            prev_vals = state.get("scan_values")
                        state["scan_results"] = prev_addrs
                        state["scan_values"] = prev_vals
                        state["scan_dropped"] = prev_dropped
                        state["scan_truncated"] = prev_truncated
                        results = state["scan_results"]
                        sel = min(sel, max(0, len(results) - 1))
                        offset = min(offset, max(0, len(results) - 1))
                        add_log(f"Undo scan → {len(results):,} candidates remain")
                    else:
                        add_log("No scan history to undo", "warn")
                elif more_result == "flush_maps":
                    with _map_cache_lock:
                        _map_cache.clear()
                    add_log("Memory-map cache flushed")
                results = state["scan_results"]
                if len(results) == 0:
                    break
                continue
            elif key in (ord('d'), ord('D')):
                dropped_idx = sel
                dropped = results[sel]
                results = _make_addr_array(a for i, a in enumerate(results) if i != sel)
                state["scan_results"] = results
                # Unknown-value scans keep a parallel value snapshot.  Remove
                # the matching element or the next relational scan sees a
                # mismatched/corrupted address-value pair.
                if state.get("scan_values") is not None:
                    state["scan_values"] = np.delete(
                        state["scan_values"], dropped_idx)
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
                    prev_truncated = entry[3]
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
                    state["scan_truncated"] = prev_truncated
                    # COUNT narrows the server list in place and has no rewind
                    # operation.  Discard it after UI undo; the next scan will
                    # safely refine the reconstructed client-side candidates.
                    _close_turbo_session()
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


def _inspect_result(stdscr, addr: int, live_value: str = "…") -> None:
    """Compact address inspector; keeps common actions one screen away."""
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "ADDRESS INSPECTOR")
        width = state["scan_width"]
        wlabel = WIDTH_LABEL.get(width, str(width))
        safe_addstr(stdscr, 2, 3, f"Address   {hex(addr)}", color(C_OK) | curses.A_BOLD)
        safe_addstr(stdscr, 3, 3, f"Current   {live_value}", color(C_WARN))
        safe_addstr(stdscr, 4, 3, f"Type      {wlabel}", color(C_NORM))
        safe_addstr(stdscr, 5, 3, f"Process   {state['proc_name']} (PID {state['pid']})", color(C_NORM))
        safe_addstr(stdscr, 7, 3, "Actions", color(C_TITLE) | curses.A_BOLD)
        safe_addstr(stdscr, 8, 5, "A  Apply value", color(C_OK))
        safe_addstr(stdscr, 9, 5, "C  Create cheat", color(C_OK))
        safe_addstr(stdscr, 10, 5, "P  Pointer scan", color(C_ACC))
        safe_addstr(stdscr, 11, 5, "D  Drop result", color(C_ERR))
        draw_statusbar(stdscr, [("A apply", C_OK), ("C cheat", C_OK), ("P pointer", C_ACC), ("D drop", C_ERR), ("Esc back", C_NORM)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols(); continue
        if key in (27, ord('q'), ord('Q')):
            return
        if key in (ord('a'), ord('A')):
            value_s = input_box(stdscr, "Apply value: ", h - 2, 3, 20)
            try:
                value = int(value_s, 0)
                if value < 0 or value > WIDTH_MAX[width]:
                    raise ValueError(f"Value out of range for {WIDTH_LABEL[width]}")
                ack, verified, actual = _write_value_verified(state["ip"], state["pid"], addr, value, width)
                if ack and verified:
                    add_log(f"Applied {value} → {hex(addr)} verified")
                elif ack and verified is None:
                    add_log(f"Applied {value} → {hex(addr)} but read-back failed", "warn")
                elif ack:
                    actual_val = struct.unpack(WIDTH_FMT[width], actual)[0]
                    add_log(f"Write mismatch {hex(addr)}: wanted {value}, read {actual_val}", "error")
                else:
                    add_log(f"Write rejected at {hex(addr)}", "error")
            except Exception as exc:
                add_log(f"Apply failed at {hex(addr)}: {exc}", "error")
            try:
                raw = ps5_read(state["ip"], state["pid"], addr, width)
                live_value = str(struct.unpack(WIDTH_FMT[width], raw)[0]) if len(raw) == width else "?"
            except Exception:
                live_value = "?"
        elif key in (ord('c'), ord('C')):
            _add_cheat_at(stdscr, addr)
            return
        elif key in (ord('p'), ord('P')):
            do_pointer_scan(stdscr, addr)
            return
        elif key in (ord('d'), ord('D')):
            old_results = state["scan_results"]
            old_values = state.get("scan_values")
            try:
                drop_idx = int(np.searchsorted(old_results, addr))
            except Exception:
                drop_idx = -1
            state["scan_results"] = _make_addr_array(a for a in old_results if int(a) != addr)
            if old_values is not None and 0 <= drop_idx < len(old_values):
                state["scan_values"] = np.delete(old_values, drop_idx)
            add_log(f"Dropped result {hex(addr)}", "warn")
            return


def _write_value_verified(ip: str, pid: int, addr: int, value: int, width: int,
                          cancel_event: Optional[threading.Event] = None) -> tuple:
    """Validate, write, and verify one memory value using the standard write path."""
    err = _validate_write_addr(addr)
    if err:
        raise ValueError(err)
    map_err = _validate_addr_in_maps(ip, pid, addr, width)
    if map_err:
        raise ValueError(map_err)
    data = struct.pack(WIDTH_FMT[width], value)
    return ps5_write_verified(ip, pid, addr, data)


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
            # Local safety metadata; generate_cht intentionally does not export it.
            "pid":     state["pid"],
            "process": state["proc_name"],
            "session": state["session"],
        }
        state["cheats"].append(entry)
        add_log(f"Added '{entry['name']}' @ {hex(addr)} = {val}")

        add_log(f"Created cheat '{entry['name']}' @ {hex(addr)} — not applied")
        # Creation and application are deliberately separate actions.
        # Apply from Cheat List (A) or Results (A) when the user explicitly asks.
        return
    except ValueError as exc:
        message_box(stdscr, [f"Could not add/apply cheat: {exc}"], "Error", C_ERR)


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
        addr = int(addr_s, 0)
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
        ack, verified, actual = ps5_write_verified(
            state["ip"], state["pid"], addr, data)
        if ack and verified:
            add_log(f"Write {hex(addr)} = {val} verified")
        elif ack and verified is None:
            add_log(f"Write {hex(addr)} = {val} acknowledged; read-back failed", "warn")
            message_box(stdscr,
                ["ps5debug acknowledged the write,",
                 "but the address could not be read back.",
                 "Check the Log and connection."],
                "Write Unverified", C_WARN)
        elif ack:
            actual_val = struct.unpack(WIDTH_FMT[width], actual)[0]
            add_log(f"Write mismatch {hex(addr)}: wanted {val}, read {actual_val}", "error")
            message_box(stdscr,
                ["ps5debug acknowledged the command, but memory did not change.",
                 f"Requested: {val}", f"Read back: {actual_val}",
                 "The game may restore the value, or this payload/firmware",
                 "may not support writes to that mapping."],
                "Write Mismatch", C_ERR)
        else:
            add_log(f"Write {hex(addr)} = {val} rejected", "error")
            message_box(stdscr, ["Write rejected by ps5debug."], "Write Failed", C_ERR)
    except Exception as exc:
        message_box(stdscr, [f"Error: {exc}"], "Error", C_ERR)


def _read_cheat_live_value(cheat: dict) -> str:
    """Read the current live value at a cheat's address. Returns str or '?'."""
    try:
        width = int(cheat["width"])
        is_pointer = "offsets" in cheat and cheat.get("offsets") is not None
        if is_pointer:
            base    = int(cheat["base"], 0) if isinstance(cheat["base"], str) else int(cheat["base"])
            offsets = [int(o, 0) if isinstance(o, str) else int(o) for o in cheat["offsets"]]
            ok, addr, _ = _resolve_pointer_chain(state["ip"], state["pid"], base, offsets)
            if not ok:
                return "?(chain)"
        else:
            addr = int(cheat["address"])
        raw = ps5_read(state["ip"], state["pid"], addr, width)
        if len(raw) == width:
            return str(struct.unpack(WIDTH_FMT[width], raw)[0])
    except Exception:
        pass
    return "?"


def _inspect_cheat(stdscr, idx: int) -> None:
    """Read-only cheat inspector with explicit edit/apply/delete actions."""
    while 0 <= idx < len(state["cheats"]):
        c = state["cheats"][idx]
        live = _read_cheat_live_value(c)
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "CHEAT INSPECTOR")
        safe_addstr(stdscr, 2, 3, c["name"], color(C_TITLE) | curses.A_BOLD)
        is_pointer = "offsets" in c and c.get("offsets") is not None
        addr_text = (hex(c["base"]) if isinstance(c.get("base"), int) else str(c.get("base"))) if is_pointer else hex(int(c["address"]))
        safe_addstr(stdscr, 4, 3, f"Address   {addr_text}{' (pointer)' if is_pointer else ''}", color(C_OK))
        safe_addstr(stdscr, 5, 3, f"Set       {c.get('value')}", color(C_NORM))
        safe_addstr(stdscr, 6, 3, f"Live      {live}", color(C_OK) if str(live) == str(c.get('value')) else color(C_WARN))
        safe_addstr(stdscr, 7, 3, f"Mode      {c.get('type')}", color(C_NORM))
        safe_addstr(stdscr, 8, 3, f"Width     {WIDTH_LABEL.get(int(c.get('width', state['scan_width'])), c.get('width'))}", color(C_NORM))
        if is_pointer:
            offs = "+".join(hex(int(o, 0) if isinstance(o, str) else int(o)) for o in c.get("offsets", []))
            safe_addstr(stdscr, 9, 3, f"Chain     {addr_text} + {offs}", color(C_ACC))
        draw_statusbar(stdscr, [("A apply", C_OK), ("E edit", C_WARN), ("F freeze", C_ACC), ("D delete", C_ERR), ("Esc back", C_NORM)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols(); continue
        if key in (27, ord('q'), ord('Q')):
            return
        if key in (ord('a'), ord('A')):
            _apply_cheat_once(stdscr, c)
        elif key in (ord('e'), ord('E')):
            _edit_cheat(stdscr, idx)
        elif key in (ord('f'), ord('F')):
            # Reuse the existing freeze manager; it can target this saved cheat.
            do_freeze(stdscr)
        elif key in (ord('d'), ord('D')):
            if confirm_box(stdscr, f"Delete '{c['name']}'?", "Delete Cheat"):
                name = c["name"]
                del state["cheats"][idx]
                add_log(f"Deleted cheat '{name}'", "warn")
                return


def do_cheat_list(stdscr) -> None:
    cheats      = state["cheats"]
    sel         = 0
    offset      = 0
    live_cache  = {}          # {cheat_index: live_value_str}
    cache_lock  = threading.Lock()
    last_refresh = 0.0
    refresh_thread = None
    refresh_cancel = threading.Event()
    REFRESH_INTERVAL = 2.0

    def _refresh_live_values(indices: list):
        """Background: read live value for each visible cheat."""
        for idx in indices:
            if refresh_cancel.is_set():
                break
            if idx >= len(state["cheats"]):
                continue
            val_str = _read_cheat_live_value(state["cheats"][idx])
            with cache_lock:
                live_cache[idx] = val_str

    stdscr.nodelay(True)
    try:
        while True:
            now = time.time()
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            cheats  = state["cheats"]
            visible = max(1, h - 8)
            draw_border(stdscr, f"CHEAT LIST  ({len(cheats)} cheats)")

            # Kick off a background refresh for visible cheats
            thread_idle = refresh_thread is None or not refresh_thread.is_alive()
            if thread_idle and now - last_refresh >= REFRESH_INTERVAL and cheats:
                visible_indices = list(range(offset, min(offset + visible, len(cheats))))
                refresh_cancel.clear()
                refresh_thread = threading.Thread(
                    target=_refresh_live_values, args=(visible_indices,), daemon=True)
                refresh_thread.start()
                last_refresh = now

            if not cheats:
                safe_addstr(stdscr, 4, 5,
                    "No cheats yet — scan and add some!", color(C_WARN))
            else:
                safe_addstr(stdscr, 2, 3,
                    "↑↓ select   Enter inspect   A apply once   D delete   Q back", color(C_NORM))
                hdr = f"  {'Name':<26}  {'Address':<22}  {'Set':<8}  {'Live':<10}  Type"
                safe_addstr(stdscr, 3, 2, hdr[:w - 4],
                            color(C_TITLE) | curses.A_UNDERLINE)
                if sel < offset:             offset = sel
                if sel >= offset + visible:  offset = sel - visible + 1
                for i, c in enumerate(cheats[offset:offset + visible]):
                    ri   = offset + i
                    attr = color(C_SEL) | curses.A_BOLD if ri == sel else color(C_NORM)
                    _disp_addr = ((hex(c["base"]) if isinstance(c["base"], int) else c["base"]) + " (ptr)"
                                  if "offsets" in c and c.get("offsets") is not None
                                  else hex(c["address"]))
                    with cache_lock:
                        live_val = live_cache.get(ri, "…")
                    # Colour live value: green if matches set value, yellow if differs
                    set_val_str = str(c["value"])
                    if live_val not in ("…", "?", "?(chain)"):
                        live_attr = color(C_OK) if live_val == set_val_str else color(C_WARN)
                    else:
                        live_attr = color(C_NORM)
                    line_base = (f"  {c['name']:<26}  {_disp_addr:<22}  "
                                 f"{set_val_str:<8}  ")
                    live_part = f"{live_val:<10}  [{c['type']}]"
                    if ri == sel:
                        safe_addstr(stdscr, 5 + i, 2,
                                    (line_base + live_part)[:w - 4].ljust(w - 4), attr)
                    else:
                        safe_addstr(stdscr, 5 + i, 2, line_base[:w - 4], color(C_NORM))
                        live_col = 2 + len(line_base)
                        if live_col < w - 4:
                            safe_addstr(stdscr, 5 + i, live_col,
                                        live_part[:w - 4 - live_col], live_attr)
                if len(cheats) > visible:
                    safe_addstr(stdscr, h - 3, w - 20,
                        f" {offset+1}-{min(offset+visible,len(cheats))}/{len(cheats)} ",
                        color(C_WARN))

            is_refreshing = refresh_thread is not None and refresh_thread.is_alive()
            draw_statusbar(stdscr, [
                ("↑↓ navigate", C_NORM), ("Enter inspect", C_OK),
                ("A apply once", C_WARN), ("D delete", C_ERR),
                ("⟳ live" if is_refreshing else "live values", C_ACC),
                ("Q back", C_NORM),
            ])
            stdscr.refresh()

            key = stdscr.getch()
            if key == -1:
                time.sleep(0.05)
                continue
            if key == curses.KEY_RESIZE:
                curses.update_lines_cols()
                continue
            if key == curses.KEY_UP    and sel > 0:               sel -= 1
            elif key == curses.KEY_DOWN and sel < len(cheats) - 1: sel += 1
            elif key in (curses.KEY_ENTER, 10, 13) and cheats:
                stdscr.nodelay(False)
                _inspect_cheat(stdscr, sel)
                stdscr.nodelay(True)
                cheats = state["cheats"]
                sel = min(sel, max(0, len(cheats) - 1))
                with cache_lock:
                    live_cache.clear()
            elif key in (ord('a'), ord('A')) and cheats:
                stdscr.nodelay(False)
                _apply_cheat_once(stdscr, cheats[sel])
                stdscr.nodelay(True)
                with cache_lock:
                    live_cache.pop(sel, None)   # force re-read after apply
            elif key in (ord('d'), ord('D')) and cheats:
                stdscr.nodelay(False)
                name = cheats[sel]["name"]
                if confirm_box(stdscr, f"Delete '{name}'?", "Delete Cheat"):
                    del cheats[sel]
                    state["cheats"] = cheats
                    add_log(f"Deleted cheat '{name}'", "warn")
                    with cache_lock:
                        live_cache.clear()
                    if not cheats:
                        sel = 0
                    else:
                        sel = min(sel, len(cheats) - 1)
                    offset = min(offset, max(0, len(cheats) - visible))
                stdscr.nodelay(True)
            elif key in (ord('q'), ord('Q')):
                break
    finally:
        stdscr.nodelay(False)
        refresh_cancel.set()
        if refresh_thread and refresh_thread.is_alive():
            refresh_thread.join(timeout=2.0)


def _apply_cheat_once(stdscr, cheat: dict) -> None:
    """Apply one saved cheat value immediately and verify the result.
    Supports both flat (address) cheats and pointer-chain cheats."""
    owner_pid = cheat.get("pid")
    if owner_pid is None:
        message_box(stdscr,
            ["This cheat predates process ownership tracking.",
             "Re-add it from current scan results before applying it."],
            "Unowned Cheat", C_WARN)
        return
    if cheat.get("session") != state["session"]:
        message_box(stdscr,
            ["This cheat belongs to an earlier PS5 connection session.",
             "The game or payload may have restarted and reused its PID.",
             "Re-add the address from fresh scan results."],
            "Stale Cheat", C_ERR)
        return
    if owner_pid != state["pid"]:
        owner_name = cheat.get("process") or "unknown process"
        message_box(stdscr,
            [f"This address belongs to PID {owner_pid} ({owner_name}).",
             f"Current process is PID {state['pid']} ({state['proc_name']}).",
             "Application blocked to avoid writing to the wrong process."],
            "Stale Cheat", C_ERR)
        return

    width = int(cheat["width"])
    value = int(cheat["value"])
    is_pointer = "offsets" in cheat and cheat.get("offsets") is not None

    if is_pointer:
        # ── pointer cheat: resolve chain first ───────────────────────────
        base    = int(cheat["base"], 0) if isinstance(cheat["base"], str) else int(cheat["base"])
        offsets = [int(o, 0) if isinstance(o, str) else int(o) for o in cheat["offsets"]]
        ok, addr, steps = _resolve_pointer_chain(
            state["ip"], state["pid"], base, offsets)
        if not ok:
            add_log(f"Pointer chain broken for '{cheat['name']}' "
                    f"base={hex(base)}", "error")
            message_box(stdscr,
                [f"Pointer chain could not be resolved.",
                 f"Base: {hex(base)}",
                 "The game may have restarted or reloaded.",
                 "Try re-scanning and rebuilding the chain."],
                "Chain Broken", C_ERR)
            return
        add_log(f"Pointer chain resolved: {hex(base)} → {hex(addr)} "
                f"(steps: {[hex(s) for s in steps]})")
    else:
        addr = int(cheat["address"])

    map_err = _validate_addr_in_maps(state["ip"], state["pid"], addr, width)
    if map_err and not confirm_box(stdscr, f"{map_err}\nWrite anyway?", "Unmapped Address"):
        return

    data = struct.pack(WIDTH_FMT[width], value)
    ack, verified, actual = ps5_write_verified(
        state["ip"], state["pid"], addr, data)

    chain_note = f" (resolved {hex(addr)})" if is_pointer else ""
    if ack and verified:
        add_log(f"Applied '{cheat['name']}' @ {hex(addr)} = {value} (verified){chain_note}")
    elif ack and verified is None:
        add_log(f"Applied '{cheat['name']}' but read-back failed{chain_note}", "warn")
        message_box(stdscr, ["Write acknowledged, but read-back failed."],
                    "Apply Unverified", C_WARN)
    elif ack:
        with _map_cache_lock:
            _map_cache.clear()
        actual_value = struct.unpack(WIDTH_FMT[width], actual)[0]
        add_log(f"Apply mismatch '{cheat['name']}': read {actual_value}", "error")
        message_box(stdscr,
                    ["Write acknowledged but did not stick.",
                     f"Requested: {value}", f"Read back: {actual_value}"],
                    "Apply Mismatch", C_ERR)
    else:
        add_log(f"Apply rejected for '{cheat['name']}'", "error")
        message_box(stdscr, ["Write rejected by ps5debug."], "Apply Failed", C_ERR)


def _edit_cheat(stdscr, idx: int) -> None:
    c = state["cheats"][idx]
    is_pointer = "offsets" in c and c.get("offsets") is not None
    stdscr.clear()
    draw_border(stdscr, "EDIT CHEAT")
    safe_addstr(stdscr, 2, 3, f"Editing: {c['name']}", color(C_TITLE) | curses.A_BOLD)
    safe_addstr(stdscr, 3, 3, "Leave blank to keep current value.", color(C_NORM))
    if is_pointer:
        base_hex = hex(c["base"]) if isinstance(c["base"], int) else c["base"]
        safe_addstr(stdscr, 4, 3, f"Pointer chain — base: {base_hex}", color(C_WARN))
    stdscr.refresh()
    new_name = input_box(stdscr, "Name  : ", 5, 3, 40, c["name"])
    val_s    = input_box(stdscr, "Value : ", 7, 3, 20, str(c["value"]))
    # Pointer cheats use pointer_freeze/pointer_write; flat cheats use freeze/write.
    # Offering the wrong set would crash cycle_input (options.index raises ValueError).
    if is_pointer:
        type_opts = ["pointer_freeze", "pointer_write"]
    else:
        type_opts = ["freeze", "write"]
    new_type = cycle_input(stdscr, "Type  : ", 9, 3, type_opts, c["type"])
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
    old_val = int(c["value"])
    state["cheats"][idx].update({"name": new_name, "value": new_val, "type": new_type})
    apply_status = "Value unchanged."
    apply_color = C_OK
    if new_val != old_val:
        try:
            if is_pointer:
                base = int(c["base"], 0) if isinstance(c["base"], str) else int(c["base"])
                offsets = [int(o, 0) if isinstance(o, str) else int(o) for o in c["offsets"]]
                ok_chain, target_addr, _ = _resolve_pointer_chain(state["ip"], state["pid"], base, offsets)
                if not ok_chain:
                    raise ValueError("pointer chain could not be resolved")
                ack, verified, actual = _write_value_verified(state["ip"], state["pid"], target_addr, new_val, int(c["width"]))
            else:
                ack, verified, actual = _write_value_verified(state["ip"], state["pid"], int(c["address"]), new_val, int(c["width"]))
            if ack and verified:
                apply_status = f"Applied new value: {new_val} (verified)."
            elif ack and verified is None:
                apply_status = "Value saved; write acknowledged but read-back failed."
                apply_color = C_WARN
            elif ack and actual is not None:
                actual_val = struct.unpack(WIDTH_FMT[int(c["width"])], actual)[0]
                apply_status = f"Value saved, but memory read back {actual_val}."
                apply_color = C_ERR
            else:
                apply_status = "Value saved, but the memory write was rejected."
                apply_color = C_ERR
        except Exception as exc:
            apply_status = f"Value saved, but could not apply: {exc}"
            apply_color = C_ERR
    add_log(f"Edited '{new_name}' val={new_val} type={new_type}")
    message_box(stdscr, [f"Updated '{new_name}'", apply_status], "Saved", apply_color)


def _parse_int_field(value, field_name):
    try:
        return int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {field_name}: {value!r}")


def do_import(stdscr) -> None:
    stdscr.clear()
    draw_border(stdscr, "IMPORT CHEATS")
    path = Path(input_box(stdscr, "JSON path: ", 4, 3, 70, str(Path.home()))).expanduser()
    if not path.exists() or not path.is_file():
        message_box(stdscr, [f"File not found: {path}"], "Import Failed", C_ERR)
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("cheatList", [])
        if not isinstance(items, list): raise ValueError("cheatList is not an array")
        imported = []
        for c in items:
            if not isinstance(c, dict) or not c.get("name"): continue
            width = int(c.get("bytes", 4))
            value = _parse_int_field(c.get("value", 0), "value")
            if width not in WIDTH_MAX or not 0 <= value <= WIDTH_MAX[width]:
                raise ValueError(f"Invalid width/value in '{c['name']}'")
            e = {"name": str(c["name"]), "type": str(c.get("type", "write")),
                 "value": value, "width": width, "pid": state["pid"],
                 "process": state["proc_name"], "session": state["session"],
                 "imported_from": str(path)}
            if "base" in c:
                e["base"] = _parse_int_field(c["base"], "base")
                e["offsets"] = [_parse_int_field(x, "offset") for x in c.get("offsets", [])]
                e["address"] = 0
            elif "address" in c:
                e["address"] = _parse_int_field(c["address"], "address")
            else:
                raise ValueError(f"'{e['name']}' has no address/base")
            imported.append(e)
        if not imported: raise ValueError("No usable cheats found")
        if state["cheats"] and not confirm_box(stdscr,
                f"Import {len(imported)} cheats and keep existing {len(state['cheats'])}?",
                "Import Cheats"):
            return
        state["cheats"].extend(imported)
        add_log(f"Imported {len(imported)} cheats from {path}")
        message_box(stdscr, [f"Imported {len(imported)} cheats.",
                             "They are bound to the current process/session."],
                    "Import Complete", C_OK)
    except Exception as exc:
        add_log(f"Import failed: {exc}", "error")
        message_box(stdscr, [f"Import failed: {exc}"], "Import Failed", C_ERR)


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
    """Freeze a manual address or resolve a saved pointer chain every tick."""
    global _freeze_thread
    _stop_freeze_worker()
    with _freeze_lock:
        if _freeze_thread and _freeze_thread.is_alive():
            message_box(stdscr, ["Previous freeze worker is still shutting down."],
                        "Freeze Still Active", C_ERR)
            return
    stdscr.clear(); draw_border(stdscr, "FREEZE ADDRESS / CHEAT")
    choices = ["Manual address"] + (["Saved cheat"] if state["cheats"] else [])
    mode = cycle_input(stdscr, "Target            : ", 4, 3, choices, choices[0])
    is_pointer = False; cheat = None; base = None; offsets = []
    if mode == "Saved cheat":
        names = [c.get("name", "Unnamed") for c in state["cheats"]]
        selected = cycle_input(stdscr, "Cheat             : ", 6, 3, names, names[0])
        cheat = state["cheats"][names.index(selected)]
        if cheat.get("pid") != state["pid"] or cheat.get("session") != state["session"]:
            message_box(stdscr, ["Selected cheat belongs to another process/session."], "Stale Cheat", C_ERR); return
        width = int(cheat["width"]); val = int(cheat["value"])
        is_pointer = "offsets" in cheat and cheat.get("offsets") is not None
        if is_pointer:
            base = int(cheat["base"],0) if isinstance(cheat["base"],str) else int(cheat["base"])
            offsets = [int(x,0) if isinstance(x,str) else int(x) for x in cheat["offsets"]]
            addr = 0
        else: addr = int(cheat["address"])
    else:
        addr_s = input_box(stdscr, "Address (hex)    : ", 6, 3, 20)
        val_s = input_box(stdscr, "Freeze value     : ", 8, 3, 20)
        labels = [WIDTH_LABEL[x] for x in VALID_WIDTHS]
        ws = cycle_input(stdscr, "Width            : ", 10, 3, labels, WIDTH_LABEL.get(state["scan_width"],"uint32"))
        width = VALID_WIDTHS[labels.index(ws)]
        try:
            addr = int(addr_s,0); val = int(val_s,0)
            err = _validate_write_addr(addr)
            if err: raise ValueError(err)
            if not 0 <= val <= WIDTH_MAX[width]: raise ValueError("Value out of range")
        except Exception as exc:
            message_box(stdscr,[f"Bad input: {exc}"],"Error",C_ERR); return
    try:
        sec = max(1,int(input_box(stdscr,"Duration (secs)  : ",12,3,6,"30")))
        interval = max(50,int(input_box(stdscr,"Interval (ms)    : ",14,3,6,"200")))/1000.0
        data = struct.pack(WIDTH_FMT[width], val)
    except Exception as exc:
        message_box(stdscr,[f"Bad input: {exc}"],"Error",C_ERR); return
    frozen_ip, frozen_pid = state["ip"], state["pid"]
    stop_event = threading.Event(); errors=[0]; deadline=time.time()+sec
    def _freeze_worker():
        while time.time() < deadline:
            if stop_event.is_set() or _freeze_stop.is_set(): break
            if state["ip"] != frozen_ip or state["pid"] != frozen_pid:
                add_log("Freeze aborted — process or connection changed","warn"); break
            target = addr
            if is_pointer:
                ok,target,_ = _resolve_pointer_chain(frozen_ip,frozen_pid,base,offsets)
                if not ok:
                    add_log("Pointer freeze: chain broken — retrying next tick","warn")
                    if stop_event.wait(interval) or _freeze_stop.is_set(): break
                    continue
            if _validate_addr_in_maps(frozen_ip,frozen_pid,target,width,10.0):
                with _map_cache_lock: _map_cache.clear()
                if stop_event.wait(interval) or _freeze_stop.is_set(): break
                continue
            if not ps5_write(frozen_ip,frozen_pid,target,data,cancel_event=_freeze_stop,timeout=1.0):
                errors[0]+=1
                with _map_cache_lock: _map_cache.clear()
            if stop_event.wait(interval) or _freeze_stop.is_set(): break
    with _freeze_lock:
        _freeze_stop.clear(); worker=threading.Thread(target=_freeze_worker,daemon=True); _freeze_thread=worker
    worker.start(); stdscr.nodelay(True)
    try:
        while worker.is_alive():
            h,w=stdscr.getmaxyx(); frac=min((time.time()-(deadline-sec))/sec,1.0)
            safe_addstr(stdscr,20,3,f"Time left: {max(0,int(deadline-time.time())):3d}s",color(C_OK))
            draw_progress_bar(stdscr,21,3,min(max(w-8,10),50),frac,f"  {int(frac*100)}%")
            if errors[0]: safe_addstr(stdscr,22,3,f"Write errors: {errors[0]}",color(C_ERR))
            stdscr.refresh(); k=stdscr.getch()
            if k==curses.KEY_RESIZE: curses.update_lines_cols(); stdscr.clear(); draw_border(stdscr,"FREEZE ADDRESS / CHEAT")
            elif k in (ord('q'),ord('Q'),27): stop_event.set(); break
            time.sleep(.05)
    finally:
        stdscr.nodelay(False); stop_event.set(); worker.join(timeout=interval+1.0)
        with _freeze_lock:
            if _freeze_thread is worker: _freeze_thread=None
    message_box(stdscr,["Freeze complete."],"Done",C_OK)


def do_ptr_verify_manual(stdscr) -> None:
    """
    Manual pointer-chain verify entry point (V menu key).

    Prompts the user for a base address and up to MAX_CHAIN_DEPTH offsets,
    then opens the chain-verify screen so they can test/refine/save the chain
    without running a full pointer scan first.

    This is useful when the user already knows a static base address (e.g.
    from a prior session or external tool) and just wants to verify or adjust
    the offset chain.
    """
    stdscr.clear()
    draw_border(stdscr, "MANUAL POINTER VERIFY")
    safe_addstr(stdscr, 2, 3,
        "Enter a known static base address to verify a pointer chain manually.",
        color(C_WARN))
    stdscr.refresh()

    base_s = input_box(stdscr, "Static base address (hex) : ", 4, 3, 20)
    try:
        base = int(base_s, 0)
        if base < _ADDR_MIN or base > _ADDR_MAX:
            raise ValueError("address out of PS5 user-space range")
    except ValueError as exc:
        message_box(stdscr, [f"Invalid address: {exc}"], "Error", C_ERR)
        return

    target_s = input_box(stdscr, "Target address (hex, 0=unknown): ", 6, 3, 20, "0")
    try:
        target = int(target_s, 0)
    except ValueError:
        target = 0

    offsets = []
    stdscr.clear()
    draw_border(stdscr, "ENTER OFFSETS")
    safe_addstr(stdscr, 2, 3,
        "Enter offsets one by one (hex or decimal).  Leave blank to finish.",
        color(C_NORM))
    safe_addstr(stdscr, 3, 3,
        f"Enter at least 1 offset (max {MAX_CHAIN_DEPTH}).",
        color(C_WARN))
    stdscr.refresh()
    for i in range(MAX_CHAIN_DEPTH):
        default_val = "0x0" if i == 0 else ""
        val_s = input_box(stdscr, f"  Offset [{i+1}] : ", 5 + i, 3, 20, default_val)
        # Empty input on any slot after the first means the user is done.
        if not val_s:
            if i == 0:
                # First offset is mandatory; treat blank as 0x0.
                offsets.append(0)
            break
        try:
            offsets.append(int(val_s, 0))
        except ValueError:
            message_box(stdscr, [f"Invalid offset: {val_s!r}"], "Error", C_ERR)
            return

    if not offsets:
        offsets = [0]

    candidate = {
        "base":   base,
        "offsets": offsets,
        "depth":  len(offsets),
        "region": "manual",
        "static": True,
    }
    do_pointer_chain_verify(stdscr, candidate, target)


def do_pointer_scan(stdscr, target_addr: Optional[int] = None) -> None:
    """
    Guided pointer-scan wizard.

    Step 1: pick the temporary address (from scan results list or manual entry).
    Step 2: scan runs at MAX_CHAIN_DEPTH automatically.
    Step 3: auto-test all static candidates and highlight matches.
    Step 3: user picks a verified chain and saves it as a permanent cheat.
    """
    # ── STEP 1: pick the target address ──────────────────────────────────────
    stdscr.clear()
    draw_border(stdscr, "POINTER SCAN — Step 1 of 3: Pick Address")

    guide = [
        "Pointer Scan finds a PERMANENT address for a value that moves",
        "every time the game restarts (a 'temporary' address).",
        "",
        "How it works:",
        "  1. Pick the temporary address you found via scanning.",
        "  2. The tool searches memory for anything that POINTS to it.",
        "  3. Static (permanent) pointer candidates are highlighted.",
        "  4. Verified chains are saved as cheats that survive restarts.",
    ]
    for i, line in enumerate(guide):
        safe_addstr(stdscr, 2 + i, 3, line,
                    color(C_WARN) if i == 0 else color(C_NORM))
    stdscr.refresh()

    scan_results = state["scan_results"]
    if target_addr is not None:
        target_addr = int(target_addr)
    if target_addr is None and len(scan_results) > 0:
        # Offer the results list as selectable options
        sel    = 0
        offset = 0
        stdscr.nodelay(True)
        try:
            while True:
                stdscr.clear()
                h, w = stdscr.getmaxyx()
                visible = max(1, h - 9)
                draw_border(stdscr, "POINTER SCAN — Step 1 of 3: Pick Temporary Address")
                safe_addstr(stdscr, 2, 3,
                    "Select the address you want to find a permanent pointer for.",
                    color(C_WARN) | curses.A_BOLD)
                safe_addstr(stdscr, 3, 3,
                    "These are your current scan results.  ↑↓ to choose, Enter to confirm.",
                    color(C_NORM))
                safe_addstr(stdscr, 4, 3,
                    "M = enter manually instead   Q = cancel",
                    color(C_NORM))

                if sel < offset:             offset = sel
                if sel >= offset + visible:  offset = sel - visible + 1

                for i, addr in enumerate(scan_results[offset:offset + visible]):
                    idx  = offset + i
                    attr = color(C_SEL) | curses.A_BOLD if idx == sel else color(C_NORM)
                    line = f"  {idx+1:4d}.   {hex(addr)}"
                    safe_addstr(stdscr, 6 + i, 2, line[:w - 4].ljust(w - 4), attr)

                draw_statusbar(stdscr, [
                    (f"{len(scan_results)} results", C_WARN),
                    ("↑↓ navigate", C_NORM),
                    ("Enter select", C_OK),
                    ("M manual", C_NORM),
                    ("Q cancel", C_NORM),
                ])
                stdscr.refresh()

                key = stdscr.getch()
                if key == -1:
                    time.sleep(0.05)
                    continue
                if key == curses.KEY_RESIZE:
                    curses.update_lines_cols()
                    continue
                if key == curses.KEY_UP and sel > 0:
                    sel -= 1
                elif key == curses.KEY_DOWN and sel < len(scan_results) - 1:
                    sel += 1
                elif key in (curses.KEY_ENTER, 10, 13):
                    target_addr = int(scan_results[sel])
                    break
                elif key in (ord('m'), ord('M')):
                    break   # fall through to manual entry
                elif key in (ord('q'), ord('Q'), 27):
                    return
        finally:
            stdscr.nodelay(False)

    if target_addr is None:
        # Manual entry fallback (no results, or user pressed M)
        stdscr.clear()
        draw_border(stdscr, "POINTER SCAN — Step 1 of 3: Enter Address Manually")
        safe_addstr(stdscr, 2, 3,
            "Enter the temporary address you found from scanning.",
            color(C_WARN))
        safe_addstr(stdscr, 3, 3,
            "You can find it in Results (R) — look at the hex address column.",
            color(C_NORM))
        stdscr.refresh()
        addr_s = input_box(stdscr, "Temporary address (hex) : ", 5, 3, 24)
        try:
            target_addr = int(addr_s, 0)
            if target_addr < _ADDR_MIN or target_addr > _ADDR_MAX:
                raise ValueError("address out of PS5 user-space range")
        except ValueError as exc:
            message_box(stdscr, [f"Invalid address: {exc}"], "Error", C_ERR)
            return

    # ── STEP 2: scan (always at maximum depth) ───────────────────────────────
    # Depth is fixed at MAX_CHAIN_DEPTH — no reason to ask the user.
    # Deeper scans find everything shallower ones find plus more; the extra
    # cost is acceptable and removes a technical question ordinary users
    # should never have to answer.
    max_depth    = MAX_CHAIN_DEPTH
    cancel_event = threading.Event()
    progress     = {"done": 0, "total": max_depth, "results": None, "error": None}

    def run():
        try:
            progress["results"] = pointer_chain_scan(
                state["ip"], state["pid"],
                target_addr,
                max_depth=max_depth,
                cancel_event=cancel_event,
                progress_cb=lambda d, t: progress.update(done=d, total=max(t, 1)),
            )
        except Exception as exc:
            progress["error"] = str(exc)

    ok = _run_scan_with_progress(
        stdscr, run, f"Scanning for pointer chains (depth {max_depth})…", cancel_event, progress)
    if not ok:
        add_log("Pointer scan cancelled", "warn")
        return
    if progress["error"]:
        message_box(stdscr, [f"Error: {progress['error']}"], "Pointer Scan Failed", C_ERR)
        return

    candidates = progress["results"] or []
    if not candidates:
        message_box(stdscr,
            ["No pointer candidates found.",
             "",
             "Tips:",
             "  • Try a deeper scan depth (3 or 4).",
             "  • Make sure the game is running and the value",
             "    is still at the address you selected.",
             "  • Run a fresh First Scan → Next Scan to confirm",
             "    the address before retrying."],
            "No Results", C_WARN)
        return

    # ── STEP 4: auto-test static candidates and show verified matches ─────────
    static_candidates = [c for c in candidates if c["static"]]
    verified_matches  = []   # chains that resolve exactly to target_addr

    if static_candidates:
        stdscr.clear()
        draw_border(stdscr, "POINTER SCAN — Step 2 of 3: Auto-Testing Chains")
        safe_addstr(stdscr, 2, 3,
            f"Testing {len(static_candidates)} static candidates against {hex(target_addr)}…",
            color(C_WARN))
        safe_addstr(stdscr, 3, 3,
            "This verifies which chains currently resolve to your target address.",
            color(C_NORM))
        stdscr.refresh()

        for i, cand in enumerate(static_candidates):
            safe_addstr(stdscr, 5, 3,
                f"Testing {i+1}/{len(static_candidates)}: {hex(cand['base'])}…",
                color(C_NORM))
            stdscr.refresh()
            ok_c, resolved, _ = _resolve_pointer_chain(
                state["ip"], state["pid"], cand["base"], cand["offsets"])
            if ok_c and resolved == target_addr:
                verified_matches.append(cand)

        safe_addstr(stdscr, 6, 3,
            f"✓ {len(verified_matches)} chains verified  "
            f"({len(static_candidates) - len(verified_matches)} did not resolve).",
            color(C_OK) if verified_matches else color(C_ERR))
        stdscr.refresh()
        time.sleep(0.8)

    # ── STEP 5: result browser ────────────────────────────────────────────────
    # Show verified matches first; fall back to all candidates if none verified.
    display_list   = verified_matches if verified_matches else candidates
    show_all       = not bool(verified_matches)
    sel    = 0
    offset = 0
    stdscr.nodelay(True)
    try:
        while True:
            stdscr.clear()
            h, w    = stdscr.getmaxyx()
            visible = max(1, h - 9)
            n_static = sum(1 for c in candidates if c["static"])

            if show_all:
                title = f"POINTER SCAN — All candidates ({len(candidates)} total, {n_static} static)"
            else:
                title = f"POINTER SCAN — Step 3 of 3: {len(verified_matches)} Verified Chains"
            draw_border(stdscr, title)

            if verified_matches and not show_all:
                safe_addstr(stdscr, 2, 3,
                    "✓ These chains currently point to your target address.",
                    color(C_OK) | curses.A_BOLD)
                safe_addstr(stdscr, 3, 3,
                    "Select one and press Enter to save it as a permanent cheat.",
                    color(C_NORM))
            else:
                safe_addstr(stdscr, 2, 3,
                    f"Target: {hex(target_addr)}   Static: {n_static}   Heap: {len(candidates)-n_static}",
                    color(C_WARN))
                if not verified_matches:
                    safe_addstr(stdscr, 3, 3,
                        "No chains verified yet — select one to test/adjust manually.",
                        color(C_WARN))
                else:
                    safe_addstr(stdscr, 3, 3,
                        "Showing all results.  Enter=verify  V=show verified only  Q=back",
                        color(C_NORM))

            hint = "↑↓ navigate   Enter save/verify   "
            hint += ("A show all" if not show_all else "V verified only") + "   Q back"
            safe_addstr(stdscr, 4, 3, hint, color(C_NORM))

            if sel < offset:             offset = sel
            if sel >= offset + visible:  offset = sel - visible + 1

            for i, cand in enumerate(display_list[offset:offset + visible]):
                idx    = offset + i
                attr   = color(C_SEL) | curses.A_BOLD if idx == sel else color(C_NORM)
                s_flag = "✓VERIFIED" if cand in verified_matches else (
                         "★STATIC  " if cand["static"] else "  heap   ")
                flag_color = color(C_OK) if cand in verified_matches else (
                             color(C_ACC) if cand["static"] else color(C_NORM))
                offs   = "+".join(hex(o) for o in cand["offsets"]) or "0x0"
                line_a = f"  {s_flag}  {hex(cand['base']):<18}"
                line_b = f"  d={cand['depth']} [{offs}]  {cand['region'][:14]}"
                if idx == sel:
                    safe_addstr(stdscr, 6 + i, 2,
                                (line_a + line_b)[:w - 4].ljust(w - 4), attr)
                else:
                    safe_addstr(stdscr, 6 + i, 2, line_a[:w//2], flag_color)
                    safe_addstr(stdscr, 6 + i, 2 + len(line_a),
                                line_b[:w - 4 - len(line_a)], color(C_NORM))

            draw_statusbar(stdscr, [
                (f"{len(display_list)} shown", C_WARN),
                (f"{len(verified_matches)} verified", C_OK),
                ("Enter save", C_OK),
                ("A all  V verified", C_NORM),
                ("Q back", C_NORM),
            ])
            stdscr.refresh()

            key = stdscr.getch()
            if key == -1:
                time.sleep(0.05)
                continue
            if key == curses.KEY_RESIZE:
                curses.update_lines_cols()
                continue
            if key == curses.KEY_UP and sel > 0:
                sel -= 1
            elif key == curses.KEY_DOWN and sel < len(display_list) - 1:
                sel += 1
            elif key in (curses.KEY_ENTER, 10, 13):
                stdscr.nodelay(False)
                do_pointer_chain_verify(stdscr, display_list[sel], target_addr)
                stdscr.nodelay(True)
            elif key in (ord('a'), ord('A')):
                display_list = candidates
                show_all     = True
                sel = 0; offset = 0
            elif key in (ord('v'), ord('V')) and verified_matches:
                display_list = verified_matches
                show_all     = False
                sel = 0; offset = 0
            elif key in (ord('q'), ord('Q'), 27):
                break
    finally:
        stdscr.nodelay(False)


def do_pointer_chain_verify(stdscr, candidate: dict, original_target: int) -> None:
    """
    Chain verification and refinement screen.

    Shows the chain, lets the user edit offsets, tests resolution live,
    and optionally saves the chain as a pointer cheat.

    A pointer cheat is stored as:
        {
            "name":    str,
            "type":    "pointer_freeze" | "pointer_write",
            "base":    int,          # static anchor address
            "offsets": [int, ...],   # e.g. [0x10, 0x58, 0x0]
            "value":   int,          # value to lock in
            "width":   int,          # byte width of the final value
            "pid":     int,
            "process": str,
            "session": int,
        }
    """
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    draw_border(stdscr, "VERIFY POINTER CHAIN")

    base    = candidate["base"]
    # Use a one-element list so closures always see the current offsets list
    # even after reassignment in the E branch (closures capture the cell, not
    # the value; a mutable container avoids the stale-reference problem).
    _offsets = [list(candidate["offsets"])]
    region   = candidate["region"]

    safe_addstr(stdscr, 2, 3,
        f"Base address : {hex(base)}  [{region}]", color(C_OK) | curses.A_BOLD)

    def _test_chain():
        """Resolve chain and return (ok, final_addr, steps)."""
        return _resolve_pointer_chain(state["ip"], state["pid"], base, _offsets[0])

    def _draw_chain(start_row: int) -> int:
        """Draw the current chain and return the row after the last line."""
        safe_addstr(stdscr, start_row, 3, "Chain:", color(C_TITLE) | curses.A_BOLD)
        row = start_row + 1
        safe_addstr(stdscr, row, 5,
            f"[base]  {hex(base)}", color(C_OK))
        row += 1
        for i, off in enumerate(_offsets[0]):
            label = f"[+{i+1}]  +{hex(off)}"
            safe_addstr(stdscr, row, 5, label, color(C_WARN))
            row += 1
        return row

    while True:
        offsets = _offsets[0]   # local alias for readability in this loop body
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "VERIFY POINTER CHAIN")
        safe_addstr(stdscr, 2, 3,
            f"Base: {hex(base)}  [{region}]  "
            f"({'★ static' if candidate['static'] else 'heap'})",
            color(C_OK) | curses.A_BOLD)

        chain_end_row = _draw_chain(4)

        # Live resolve
        ok, final_addr, steps = _test_chain()
        status_row = chain_end_row + 1
        if ok:
            match = (final_addr == original_target)
            status_color = C_OK if match else C_WARN
            safe_addstr(stdscr, status_row, 3,
                f"→ Resolves to: {hex(final_addr)}"
                f"  {'✓ MATCHES target' if match else '✗ differs from original target'}",
                color(status_color) | curses.A_BOLD)
            if steps:
                safe_addstr(stdscr, status_row + 1, 5,
                    "Steps: " + " → ".join(hex(s) for s in steps),
                    color(C_NORM))
        else:
            safe_addstr(stdscr, status_row, 3,
                "→ Chain BROKEN (null/invalid pointer)", color(C_ERR) | curses.A_BOLD)

        menu_row = min(status_row + 3, h - 5)
        safe_addstr(stdscr, menu_row, 3,
            "[E] Edit offsets   [T] Test again   [S] Save as cheat   [Q] Back",
            color(C_NORM))
        draw_statusbar(stdscr, [
            ("E edit offsets", C_WARN),
            ("T re-test",      C_OK),
            ("S save cheat",   C_OK),
            ("Q back",         C_NORM),
        ])
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols()
            continue

        elif key in (ord('e'), ord('E')):
            # Edit offsets one by one
            stdscr.clear()
            draw_border(stdscr, "EDIT OFFSETS")
            safe_addstr(stdscr, 2, 3,
                "Enter offsets as hex (e.g. 0x10) or decimal.  "
                "Empty = keep current.",
                color(C_NORM))
            safe_addstr(stdscr, 3, 3,
                "Every offset dereferences a pointer then adds the offset value.",
                color(C_WARN))
            safe_addstr(stdscr, 4, 3,
                f"Current depth: {len(offsets)}  "
                "Add more offsets? (enter values below, blank to stop)",
                color(C_NORM))
            stdscr.refresh()
            new_offsets = []
            row = 6
            for i in range(MAX_CHAIN_DEPTH):
                cur = hex(offsets[i]) if i < len(offsets) else ""
                val_s = input_box(stdscr, f"  Offset [{i+1}] : ", row + i, 3, 20, cur)
                if not val_s:
                    break
                try:
                    new_offsets.append(int(val_s, 0))
                except ValueError:
                    message_box(stdscr, [f"Invalid offset: {val_s!r}"], "Error", C_ERR)
                    break
            if new_offsets:
                _offsets[0] = new_offsets   # update shared container; closures see it

        elif key in (ord('t'), ord('T')):
            # Re-test (loop redraws automatically)
            continue

        elif key in (ord('s'), ord('S')):
            # Save as pointer cheat
            offsets = _offsets[0]   # re-alias after any edits
            stdscr.clear()
            draw_border(stdscr, "SAVE POINTER CHEAT")
            safe_addstr(stdscr, 2, 3,
                f"Chain: {hex(base)} + [{'+'.join(hex(o) for o in offsets)}]",
                color(C_OK))
            stdscr.refresh()

            name  = input_box(stdscr, "Cheat name   : ", 4, 3, 40)
            val_s = input_box(stdscr, "Lock-in value: ", 6, 3, 20)
            _wl   = [WIDTH_LABEL[ww] for ww in VALID_WIDTHS]
            _ws   = cycle_input(stdscr, "Width        : ", 8, 3, _wl,
                                WIDTH_LABEL.get(state["scan_width"], "uint32"))
            width = VALID_WIDTHS[_wl.index(_ws)]
            typ   = cycle_input(stdscr, "Type         : ", 10, 3,
                                ["pointer_freeze", "pointer_write"],
                                "pointer_freeze")
            try:
                val = int(val_s, 0)
                if val < 0 or val > WIDTH_MAX[width]:
                    raise ValueError(f"out of range for {WIDTH_LABEL[width]}")
            except ValueError as exc:
                message_box(stdscr, [f"Invalid value: {exc}"], "Error", C_ERR)
                continue

            entry = {
                "name":    name or f"PTR@{hex(base)}",
                "type":    typ,
                "base":    base,
                "offsets": list(offsets),
                "value":   val,
                "width":   width,
                "pid":     state["pid"],
                "process": state["proc_name"],
                "session": state["session"],
                # For display compatibility with non-pointer cheats:
                "address": 0,    # resolved at apply time
            }
            state["cheats"].append(entry)
            add_log(f"Added pointer cheat '{entry['name']}' "
                    f"base={hex(base)} offsets={[hex(o) for o in offsets]} val={val}")
            message_box(stdscr,
                [f"  {entry['name']}",
                 f"  base={hex(base)}",
                 f"  offsets=[{'+'.join(hex(o) for o in offsets)}]",
                 f"  value={val}  [{typ}]",
                 "",
                 "This cheat resolves the chain at apply-time,",
                 "so it works even after the game restarts."],
                "Pointer Cheat Saved", C_OK)
            break

        elif key in (ord('q'), ord('Q'), 27):
            break


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
        _clear_scan_state()
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
            ("↑↓/PgUp/PgDn", C_NORM), ("S save", C_OK), ("Q back", C_NORM),
        ])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_UP and offset > 0: offset -= 1
        elif key == curses.KEY_DOWN and offset < max(0, len(snap)-1): offset += 1
        elif key == curses.KEY_PPAGE: offset = max(0, offset - visible)
        elif key == curses.KEY_NPAGE: offset = min(max(0, len(snap)-visible), offset + visible)
        elif key in (ord('s'), ord('S')):
            fname = Path.home() / f"rdx-debug-{time.strftime('%Y%m%d-%H%M%S')}.txt"
            try:
                with fname.open('w', encoding='utf-8') as f:
                    for e in snap: f.write(f"[{e['ts']}] [{e['level'].upper()[:4]}]  {e['msg']}\n")
                message_box(stdscr, [f"Log saved to {fname}"], "Saved", C_OK)
            except OSError as exc:
                message_box(stdscr, [f"Could not save log: {exc}"], "Save Failed", C_ERR)
        elif key in (ord('q'), ord('Q'), 27): break


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
    finally:
        _stop_freeze_worker()
        _close_turbo_session()
    print("\nps5cheats_tui exited.")
