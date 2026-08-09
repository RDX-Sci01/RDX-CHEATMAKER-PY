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
import heapq
import os
import queue as _queue
import re
import socket
import struct
import json
import threading
import tempfile
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
CMD_REGION_CLASSIFY = 0xBDAACC16
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
MAX_CHAIN_DEPTH   = 8

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

# Smart temporary → permanent resolver cache.  The index is built once per
# process/map layout and reused for subsequent temporary addresses.
_pointer_index_cache = {}
_pointer_index_lock = threading.RLock()
_PTR_INDEX_CHUNK = 0x2000000          # 32 MiB
_PTR_RESOLVE_OFFSET_MAX = 0x2000   # ±8 KiB maximum field-offset search
_PTR_RESOLVE_OFFSET_TIERS = (0x100, 0x400, 0x1000, 0x2000)
_PTR_RESOLVE_OFFSET_STEP = 4      # common 32-bit structure-field alignment
_PTR_RESOLVE_MAX_HITS = 192       # deterministic candidate window per target
_PTR_RESOLVE_MAX_NODES = 2500
_PTR_RESOLVE_MAX_FOUND = 96
_PTR_FAST_DIRECT_RANGE = 0x100
_PTR_FAST_DIRECT_HITS = 24

# Bounded streaming pointer scanner.  These limits are separate from the
# reverse-index resolver above and must remain defined for pointer_chain_scan.
_PTR_STRUCT_MAX = 0x1000             # interval matching keeps wide offsets cheap
_PTR_STRUCT_STEP = 4
_PTR_CHUNK = 0x2000000              # 32 MiB network reads
_PTR_BEAM_MAX = 12_000              # heap targets carried to the next depth
_PTR_SEARCH_TARGET_MAX = 4_096      # bounded deep-pass value set
_PTR_DEPTH_DEFAULT = 3
_PTR_DISK_INDEX_THRESHOLD = 0x40000000  # 1 GiB readable memory
_PTR_DISK_SHARD_BYTES = 0x2000000       # 32 MiB per sorted disk shard
_PTR_DISK_WORKERS = 6                    # persistent parallel reader/indexers
_pointer_region_class_cache = {}         # map fingerprint -> classified rows

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

def ps5_classify_regions(ip: str, pid: int, max_regions: int = 8192,
                         probe_bytes: int = 0x10000) -> list:
    """Return ps5debug-NG's read-throughput classification for process maps.

    This read-only authenticated command identifies uncached/GPU-backed ranges
    which are unsuitable for pointer indexing.  An unsupported payload simply
    raises; callers deliberately retain the normal map-based fallback.
    """
    s = ps5_connect(ip)
    try:
        body = struct.pack("<II", PROC_AUTH_MAGIC, 2)
        s.sendall(cmd_header(CMD_PROC_AUTH, len(body)) + body)
        if not check_ok(s):
            raise RuntimeError("region classifier authentication rejected")
        length = struct.unpack("<H", recv_exact(s, 2))[0]
        if length <= 0 or length > 256:
            raise RuntimeError(f"invalid auth challenge length: {length}")
        challenge = recv_exact(s, length)
        key = _auth_keystream(length)
        s.sendall(bytes(a ^ b for a, b in zip(challenge, key)))
        if not check_ok(s):
            raise RuntimeError("region classifier auth response rejected")

        request = struct.pack("<IIII", int(pid), int(max_regions),
                              int(probe_bytes), 0)
        s.sendall(cmd_header(CMD_REGION_CLASSIFY, len(request)) + request)
        if not check_ok(s):
            raise RuntimeError("region classifier unavailable")
        count = struct.unpack("<I", recv_exact(s, 4))[0]
        if count > max_regions:
            raise RuntimeError(f"invalid region classifier count: {count}")
        records = []
        for _ in range(count):
            start, end, prot, flags, mbps, _reserved = struct.unpack(
                "<QQIIII", recv_exact(s, 32))
            records.append({"start": start, "end": end, "prot": prot,
                            "flags": flags, "mbps": mbps})
        if not check_ok(s):
            raise RuntimeError("region classifier final status failed")
        return records
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
    transfer_started = time.monotonic()
    last_progress = time.monotonic()
    inactivity_limit = 15.0
    # Even a connection that trickles a few bytes must not hold the scanner
    # forever.  Allow roughly 1 MiB/s plus setup slack; normal LAN reads are
    # substantially faster and therefore unaffected.
    total_limit = max(15.0, (n / 1048576.0) + 10.0)
    try:
        while pos < n:
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("scan cancelled")
            if time.monotonic() - transfer_started >= total_limit:
                raise TimeoutError(
                    f"PS5 read exceeded {total_limit:.0f}s after "
                    f"{pos:,}/{n:,} bytes")
            try:
                got = s.recv_into(view[pos:], n - pos)
            except socket.timeout:
                if time.monotonic() - last_progress >= inactivity_limit:
                    raise TimeoutError(
                        f"PS5 read stalled for {inactivity_limit:.0f}s "
                        f"after {pos:,}/{n:,} bytes")
                continue
            if not got:
                raise ConnectionError("PS5 disconnected")
            pos += got
            last_progress = time.monotonic()
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
            # Protocol names are fixed-width C strings.  Bytes after the first
            # NUL are padding and may contain stale data from the server struct.
            name = raw[:32].split(b'\x00', 1)[0].decode('utf-8', errors='replace')
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
            name  = raw[:32].split(b'\x00', 1)[0].decode('utf-8', errors='replace')
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


# ── Change-triggered debugger resolver ────────────────────────────────────────
#
# Prefer a hardware-watchpoint trace over a broad pointer-index build.
# ps5debug-NG exposes a real debug session, four hardware watchpoints, and an
# async interrupt channel on TCP/755.  The trace changes the target briefly,
# waits for a real target-process access, captures RIP/registers, disassembles
# the accessor, and then feeds the observed base pointer into the existing
# bounded pointer scanner.  The original reverse-index resolver remains the
# fallback when tracing is unavailable or produces no useful accessor.

CMD_DEBUG_ATTACH        = 0xBDBB0001
CMD_DEBUG_DETACH        = 0xBDBB0002
CMD_DEBUG_SET_WATCHPOINT = 0xBDBB0004
CMD_DEBUG_GET_THREAD_LIST = 0xBDBB0005
CMD_DEBUG_GETDBREGS     = 0xBDBB000C
CMD_PROC_DISASM_REGION  = 0xBDAA0020

_DEBUG_EVENT_SIZE = 0x4A0
_DEBUG_REG_OFFSET = 0x30
_DEBUG_DBREG_OFFSET = 0x420
_DEBUG_TRACE_TIMEOUT = 10.0
_DEBUG_TRACE_MAX_HITS = 8
_DEBUG_TRACE_WP_INDEX = None
# Disabled after live testing exposed a console-crashing attach/watchpoint
# lifecycle.  Keep the implementation for protocol work, but never enter it
# from a production resolver until it has an explicit capability/safety gate.
_DEBUG_TRACE_ENABLED = False

# FreeBSD/amd64 struct reg: 15 GP registers followed by trap/segment fields.
_REG_OFFSETS = {
    "r15": 0, "r14": 8, "r13": 16, "r12": 24, "r11": 32, "r10": 40,
    "r9": 48, "r8": 56, "rdi": 64, "rsi": 72, "rbp": 80, "rbx": 88,
    "rdx": 96, "rcx": 104, "rax": 112, "rip": 136, "rflags": 152,
    "rsp": 160,
}

# Zydis 4.x register IDs used by ps5debug-NG's disassembler.
_ZYDIS_GPR64 = {
    53:"rax", 54:"rcx", 55:"rdx", 56:"rbx", 57:"rsp", 58:"rbp",
    59:"rsi", 60:"rdi", 61:"r8", 62:"r9", 63:"r10", 64:"r11",
    65:"r12", 66:"r13", 67:"r14", 68:"r15",
}
_ZYDIS_RIP = 197

def _debug_status_ok(s: socket.socket) -> bool:
    return struct.unpack("<I", recv_exact(s, 4))[0] == STATUS_SUCCESS

def _debug_send(s: socket.socket, cmd: int, body: bytes = b"") -> None:
    s.sendall(cmd_header(cmd, len(body)) + body)
    if not _debug_status_ok(s):
        raise RuntimeError(f"debug command 0x{cmd:X} rejected")

def _debug_thread_list(s: socket.socket) -> list:
    s.sendall(cmd_header(CMD_DEBUG_GET_THREAD_LIST))
    if not _debug_status_ok(s):
        raise RuntimeError("debug thread-list rejected")
    n = struct.unpack("<I", recv_exact(s, 4))[0]
    if n > 4096:
        raise RuntimeError("invalid debug thread count")
    return [struct.unpack("<I", recv_exact(s, 4))[0] for _ in range(n)]

def _debug_get_dbregs(s: socket.socket, lwpid: int) -> bytes:
    body = struct.pack("<I", int(lwpid))
    s.sendall(cmd_header(CMD_DEBUG_GETDBREGS, len(body)) + body)
    if not _debug_status_ok(s):
        raise RuntimeError("debug DBREG read rejected")
    return recv_exact(s, 128)

def _debug_free_watchpoint(s: socket.socket, lwpid: int) -> Optional[int]:
    try:
        db = _debug_get_dbregs(s, lwpid)
        regs = struct.unpack("<16Q", db)
        dr7 = int(regs[7])
        for i in range(4):
            if not (dr7 & (1 << (2 * i))) and not (dr7 & (1 << (2 * i + 1))):
                return i
    except Exception:
        return None
    return None

def _debug_set_watchpoint(s: socket.socket, index: int, address: int,
                          length_code: int = 0, breaktype: int = 3) -> None:
    # index, enabled, DR7 length encoding, DR7 access encoding, address
    body = struct.pack("<IIIIQ", int(index), 1, int(length_code),
                       int(breaktype), int(address))
    _debug_send(s, CMD_DEBUG_SET_WATCHPOINT, body)

def _debug_clear_watchpoint(s: socket.socket, index: int) -> None:
    body = struct.pack("<IIIIQ", int(index), 0, 0, 3, 0)
    _debug_send(s, CMD_DEBUG_SET_WATCHPOINT, body)

def _debug_continue(s: socket.socket, action: int) -> None:
    """Resume (0) or stop (1) the attached target process."""
    _debug_send(s, 0xBDBB0010, struct.pack("<I", int(action)))

def _debug_parse_event(packet: bytes) -> dict:
    if len(packet) != _DEBUG_EVENT_SIZE:
        raise RuntimeError("invalid debug event size")
    regs_blob = packet[_DEBUG_REG_OFFSET:_DEBUG_REG_OFFSET + 176]
    db_blob = packet[_DEBUG_DBREG_OFFSET:_DEBUG_DBREG_OFFSET + 128]
    regs = {name: struct.unpack_from("<Q", regs_blob, off)[0]
            for name, off in _REG_OFFSETS.items()}
    db = struct.unpack("<16Q", db_blob)
    return {
        "lwpid": struct.unpack_from("<I", packet, 0)[0],
        "status": struct.unpack_from("<I", packet, 4)[0],
        "regs": regs,
        "dbregs": db,
    }

def _debug_disasm(s: socket.socket, pid: int, address: int,
                  length: int = 32, max_entries: int = 16) -> list:
    body = struct.pack("<IQII", int(pid), int(address), int(length), int(max_entries))
    s.sendall(cmd_header(CMD_PROC_DISASM_REGION, len(body)) + body)
    if not _debug_status_ok(s):
        raise RuntimeError("disassembly request rejected")
    out = []
    while True:
        raw = recv_exact(s, 32)
        if raw == b"\xFF" * 32:
            break
        addr, rip_rel, mem_disp = struct.unpack_from("<QQq", raw, 0)
        insn_len, kind, mem_base, mem_index, mem_scale = struct.unpack_from("<BBBBB", raw, 24)
        mnemonic = struct.unpack_from("<B", raw, 29)[0]
        out.append({
            "addr": addr, "rip_rel_target": rip_rel, "mem_disp": mem_disp,
            "length": insn_len, "kind": kind, "mem_base_reg": mem_base,
            "mem_index_reg": mem_index, "mem_scale": mem_scale,
            "mnemonic_lo": mnemonic,
        })
    return out

def _trace_temporary_access(ip: str, pid: int, target_addr: int,
                            width: int, timeout: float = _DEBUG_TRACE_TIMEOUT) -> dict:
    """
    Change-triggered resolver:
      1) attach debugger;
      2) install a read/write hardware watchpoint on target;
      3) wait for a real target-process access;
      5) capture RIP/registers and decode the memory operand;
      6) restore the original bytes and detach.

    Returns a trace dictionary.  It never leaves the probe value installed.
    """
    if not _DEBUG_TRACE_ENABLED:
        raise RuntimeError(
            "hardware-watchpoint tracing is disabled: unsafe debugger lifecycle")
    target_addr = int(target_addr)
    width = int(width)
    original = ps5_read(ip, pid, target_addr, width)
    if len(original) != width:
        raise RuntimeError("could not read original target value")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", 755))
    listener.listen(1)
    listener.settimeout(max(timeout + 1.0, 2.0))

    cmd = None
    event_sock = None
    attached = False
    wp_index = None
    try:
        cmd = ps5_connect(ip, timeout=10.0)
        body = struct.pack("<I", int(pid))
        cmd.sendall(cmd_header(CMD_DEBUG_ATTACH, len(body)) + body)
        if not _debug_status_ok(cmd):
            raise RuntimeError("debug attach rejected")
        attached = True

        event_sock, _ = listener.accept()
        event_sock.settimeout(timeout)

        threads = _debug_thread_list(cmd)
        if not threads:
            raise RuntimeError("debugger attached but no target threads were reported")
        lwpid = threads[0]
        wp_index = _debug_free_watchpoint(cmd, lwpid)
        if wp_index is None:
            raise RuntimeError("no free hardware watchpoint slot")

        # Stop briefly while installing DR7.  Do not write a probe value here:
        # restoring one later can overwrite a legitimate in-game inventory
        # change made while the trace is active.
        _debug_continue(cmd, 1)
        # DR7 length encoding: 0=1 byte, 1=2 bytes, 2=8 bytes, 3=4 bytes.
        wp_length = {1: 0, 2: 1, 4: 3, 8: 2}.get(width)
        if wp_length is None:
            raise RuntimeError(f"unsupported watchpoint width: {width}")
        _debug_set_watchpoint(cmd, wp_index, target_addr, wp_length, 3)
        _debug_continue(cmd, 0)

        # Collect a few genuine target accesses.  The first hit is not always
        # the useful accessor (for example a housekeeping read), so keep trying
        # until we decode a usable memory operand or the trace window expires.
        deadline = time.monotonic() + max(timeout, 0.1)
        event = None
        insn = None
        last_reason = "no matching hardware watchpoint event"
        hits = 0
        while time.monotonic() < deadline and hits < _DEBUG_TRACE_MAX_HITS:
            remaining = max(deadline - time.monotonic(), 0.05)
            event_sock.settimeout(remaining)
            try:
                packet = recv_exact(event_sock, _DEBUG_EVENT_SIZE)
            except socket.timeout:
                break
            candidate = _debug_parse_event(packet)
            dr6 = int(candidate["dbregs"][6])
            if not (dr6 & (1 << int(wp_index))):
                # Every debug event stops the target.  Ignoring an unrelated
                # event without resuming leaves the game visibly frozen.
                _debug_continue(cmd, 0)
                continue
            hits += 1
            regs = candidate["regs"]
            rip = int(regs["rip"])
            try:
                insns = _debug_disasm(cmd, pid, max(rip - 8, _ADDR_MIN), 32, 16)
                decoded = next((x for x in insns if int(x["addr"]) == rip), None)
            except Exception as exc:
                decoded = None
                last_reason = f"disassembly failed: {exc}"
            if not decoded or not (int(decoded["kind"]) & 0x10):
                last_reason = "watchpoint hit but accessor instruction was not decoded"
                _debug_continue(cmd, 0)
                continue
            event = candidate
            insn = decoded
            break
        if event is None or insn is None:
            raise TimeoutError(last_reason)

        regs = event["regs"]
        rip = int(regs["rip"])

        base_reg_id = int(insn["mem_base_reg"])
        index_reg_id = int(insn["mem_index_reg"])
        base_name = _ZYDIS_GPR64.get(base_reg_id)
        index_name = _ZYDIS_GPR64.get(index_reg_id)

        # Resolve the actual effective address represented by the decoded
        # operand.  RIP-relative accesses are stable code references, not object
        # pointers, so they are reported but not used as permanent pointer roots.
        index_val = int(regs.get(index_name, 0)) if index_name else 0
        if base_reg_id == _ZYDIS_RIP:
            base_val = rip
            base_name = "rip"
        elif base_name:
            base_val = int(regs[base_name])
        else:
            base_val = 0

        effective = base_val + (index_val * int(insn["mem_scale"] or 1)) + int(insn["mem_disp"])
        if effective != target_addr:
            raise RuntimeError(
                f"decoded accessor mismatch: {hex(effective)} != {hex(target_addr)}"
            )

        kind = int(insn.get("kind", 0))
        access_mode = "readwrite"
        if kind & 0x40 and not (kind & 0x80):
            access_mode = "read"
        elif kind & 0x80 and not (kind & 0x40):
            access_mode = "write"

        return {
            "success": True,
            "target": target_addr,
            "rip": rip,
            "base_reg": base_name or f"reg#{base_reg_id}",
            "base_value": base_val,
            "index_reg": index_name,
            "index_value": index_val,
            "scale": int(insn["mem_scale"] or 1),
            "final_offset": int(insn["mem_disp"]),
            "access_mode": access_mode,
            "instruction": insn,
            "lwpid": int(event["lwpid"]),
        }
    finally:
        # Clear the watchpoint before detaching.
        if cmd is not None and wp_index is not None:
            try:
                _debug_clear_watchpoint(cmd, wp_index)
            except Exception:
                pass
        if cmd is not None:
            if attached:
                try:
                    cmd.sendall(cmd_header(CMD_DEBUG_DETACH))
                    _debug_status_ok(cmd)
                except Exception:
                    pass
        if event_sock is not None:
            try: event_sock.close()
            except Exception: pass
        if cmd is not None:
            try: cmd.close()
            except Exception: pass
        try: listener.close()
        except Exception: pass

def _resolve_trace_first(ip: str, pid: int, target_addr: int,
                         width: int, cancel_event=None,
                         progress_cb=None) -> dict:
    """
    Trace first, then resolve the observed object pointer with the existing
    bounded pointer scanner.  Falls back to the cached reverse index if the
    trace backend is unavailable or the accessor is not pointer-like.
    """
    trace = _trace_temporary_access(ip, pid, target_addr, width)
    if cancel_event and cancel_event.is_set():
        return {"candidates": [], "trace": trace, "method": "trace-cancelled"}

    # A stable pointer root needs a general-purpose base register.  Indexed
    # addressing is reported but deliberately not promoted to a permanent chain
    # because its index can change at runtime.
    if (not trace.get("base_value") or trace.get("base_reg") in ("rip", "rsp", "rbp")
            or trace.get("index_reg")):
        return {"candidates": [], "trace": trace, "method": "trace-no-stable-base"}

    if progress_cb:
        progress_cb(0, max(_PTR_RESOLVE_MAX_NODES, 1))

    base_target = int(trace["base_value"])
    candidates = pointer_chain_scan(
        ip, pid, base_target,
        max_depth=min(5, MAX_CHAIN_DEPTH),
        cancel_event=cancel_event,
        progress_cb=progress_cb,
    )

    verified = []
    final_off = int(trace["final_offset"])
    maps = _get_maps_cached(ip, pid)
    for c in candidates:
        if not c.get("static"):
            continue
        c2 = dict(c)
        # The traced instruction is the terminal field access. The reverse
        # chain resolves to the object/base pointer; the traced displacement is
        # then added directly. Do not treat the displacement as another pointer
        # dereference.
        c2["offsets"] = list(c["offsets"])
        c2["terminal_offset"] = final_off
        c2["depth"] = len(c2["offsets"])
        c2["trace_rip"] = int(trace["rip"])
        c2["trace_base_reg"] = trace.get("base_reg")
        c2["trace_base_value"] = base_target
        c2["trace_final_offset"] = final_off
        c2["trace_access_mode"] = trace.get("access_mode", "readwrite")
        c2["trace_instruction"] = trace.get("instruction")
        mod, mb, mr = _module_info_for_addr(int(c2["base"]), maps)
        c2["module_name"] = mod or "main"
        c2["module_base"] = mb
        c2["module_relative_offset"] = mr
        ok, resolved, steps = _resolve_pointer_chain(
            ip, pid, int(c2["base"]), c2["offsets"], final_off)
        c2["verified"] = bool(ok and resolved == int(target_addr))
        c2["resolved"] = int(resolved) if ok else 0
        c2["resolved_base"] = int(resolved - final_off) if ok else 0
        c2["steps"] = steps
        if c2["verified"]:
            c2["score"] = float(c2.get("score", 0.0)) + 150.0
            verified.append(c2)

    verified.sort(key=lambda c: (-c["score"], c["depth"]))
    return {
        "candidates": verified,
        "trace": trace,
        "method": "change-triggered",
        "index_built": False,
        "maps": maps,
    }


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
                # The caller can immediately retry this range with a smaller
                # chunk. Repeating the same oversized, slow transfer here only
                # multiplies the stall duration.
                if isinstance(exc, TimeoutError):
                    raise
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
    """Return True only for plausible persistent/module-backed regions.

    Avoid treating arbitrary named anonymous mappings as permanent roots.
    Writable anonymous/heap/stack regions remain transient candidates unless
    the map name clearly identifies a module-backed image.
    """
    name = str(region.get("name", "") or "").strip()
    start = int(region.get("start", 0))
    prot = int(region.get("prot", 0))
    PROT_READ, PROT_WRITE, PROT_EXEC = 0x1, 0x2, 0x4

    low_name = name.lower()
    transient_prefixes = ("[heap", "[stack", "[anon", "anon", "heap", "stack",
                           "scepthread", "scelibcinternal")
    if low_name.startswith(transient_prefixes) or low_name in {"", "anon"}:
        return bool((prot & PROT_READ) and not (prot & PROT_WRITE)
                    and start < _STATIC_ADDR_MAX)

    # Executable mappings and named non-writable image mappings are strong
    # module/static signals. Named writable module .data/.bss mappings are also
    # accepted because they are commonly where global pointers live.
    if name and (prot & PROT_READ):
        if prot & PROT_EXEC:
            return True
        if not (prot & PROT_WRITE):
            return True
        # Writable named image mapping: require a module-like name rather than
        # an arbitrary allocation label.
        module_like = (low_name == "executable" or
                       ".elf" in low_name or ".prx" in low_name or
                       ".sprx" in low_name or "/" in name or
                       low_name.endswith((".bin", ".self", ".so")))
        if module_like:
            return True

    # Low, readable, non-writable anonymous mappings can still be image-backed
    # segments when the debugger omits the name.
    return bool(start < _STATIC_ADDR_MAX and (prot & PROT_READ)
                and not (prot & PROT_WRITE))


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
                            offsets: list,
                            terminal_offset: int = 0) -> tuple:
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
        current = int(base_addr) + int(terminal_offset)
        if current < _ADDR_MIN or current > _ADDR_MAX:
            return False, 0, []
        return True, current, []

    steps   = []
    current = base_addr
    # Every offset requires a pointer read at the current address, then add.
    for offset in offsets:
        ptr_val = ps5_read_pointer(ip, pid, current)
        if ptr_val == 0 or ptr_val > _ADDR_MAX:
            return False, 0, steps
        current = ptr_val + offset
        steps.append(current)

    current += int(terminal_offset)
    if current < _ADDR_MIN or current > _ADDR_MAX:
        return False, 0, steps
    if terminal_offset:
        steps.append(current)
    return True, current, steps


# ── Smart temporary → permanent resolver ────────────────────────────────────
#
# This is deliberately separate from pointer_chain_scan().  The older scanner
# remains available as an advanced/fallback search.  The resolver builds one
# reverse pointer index for the process and then answers many temporary-address
# queries from that index instead of rescanning memory for every address.


def _pointer_map_fingerprint(maps: list) -> tuple:
    """Compact fingerprint used to invalidate the reverse index on remaps."""
    return tuple(sorted((int(r.get("start", 0)), int(r.get("end", 0)),
                         int(r.get("prot", 0)), str(r.get("name", "")))
                        for r in maps))


def _module_info_for_addr(addr: int, maps: list) -> tuple:
    """Return (module_name, module_base, relative_offset) for a static address."""
    containing = None
    for r in maps:
        if int(r["start"]) <= addr < int(r["end"]):
            containing = r
            break
    if containing is None:
        return ("", 0, 0)
    name = str(containing.get("name", "") or "")
    static_maps = [r for r in maps if _is_static_region(r)]
    if name:
        same = [r for r in static_maps if str(r.get("name", "") or "") == name]
    else:
        same = []
    if same:
        base = min(int(r["start"]) for r in same)
        return (name, base, int(addr - base))
    # For unnamed low/static mappings, use the lowest static mapping as the
    # main module anchor.  This is still ASLR-relative rather than absolute.
    if _is_static_region(containing) and static_maps:
        base = min(int(r["start"]) for r in static_maps)
        return (name or "main", base, int(addr - base))
    return (name, int(containing["start"]), int(addr - containing["start"]))


def _build_region_lookup(maps: list):
    """Build a binary-searchable region table once per resolver run."""
    rows = sorted(
        (int(r.get("start", 0)), int(r.get("end", 0)), r)
        for r in maps if int(r.get("end", 0)) > int(r.get("start", 0))
    )
    starts = [x[0] for x in rows]
    return starts, rows


def _region_for_addr(addr: int, region_starts: list, region_rows: list):
    """O(log n) memory-map ownership lookup."""
    i = bisect.bisect_right(region_starts, int(addr)) - 1
    if i >= 0:
        start, end, region = region_rows[i]
        if start <= int(addr) < end:
            return region
    return None


def _region_priority(region: dict) -> int:
    """Higher score means more likely to contain a useful persistent pointer."""
    prot = int(region.get("prot", 0))
    name = str(region.get("name", "") or "")
    if _is_static_region(region):
        return 100 + (20 if prot & 0x4 else 0) + (10 if name else 0)
    if name and name not in _HEAP_NAME_HINTS and (prot & 0x2):
        return 70
    if prot & 0x2:
        return 45
    if name in _HEAP_NAME_HINTS:
        return 20
    return 10


def _fast_direct_pointer_hits(ip: str, pid: int, target: int, maps: list,
                              cancel_event=None, max_hits: int = _PTR_FAST_DIRECT_HITS) -> list:
    """Cheap first pass: find direct target-near pointers without building the full index."""
    readable = [r for r in _pointer_readable_regions(maps) if _is_static_region(r)]
    readable.sort(key=lambda r: (-_region_priority(r), int(r["start"])))
    low = max(_ADDR_MIN, int(target) - _PTR_FAST_DIRECT_RANGE)
    high = min(_ADDR_MAX, int(target) + _PTR_FAST_DIRECT_RANGE)
    starts = np.asarray([int(r["start"]) for r in maps], dtype=np.uint64)
    ends = np.asarray([int(r["end"]) for r in maps], dtype=np.uint64)
    hits = []
    sock = _ScanSocket(ip, pid)
    try:
        for region in readable:
            if cancel_event and cancel_event.is_set():
                break
            rs, re_ = int(region["start"]), int(region["end"])
            pos = rs + ((-rs) % 8)
            while pos < re_ and len(hits) < max_hits:
                size = min(_PTR_INDEX_CHUNK, re_ - pos)
                size -= size % 8
                if size < 8:
                    break
                try:
                    raw = sock.read(pos, size, cancel_event)
                except Exception:
                    pos += size
                    continue
                trim = len(raw) - (len(raw) % 8)
                if trim >= 8:
                    a = np.frombuffer(raw[:trim], dtype=np.uint64)
                    mask = (a >= np.uint64(low)) & (a <= np.uint64(high))
                    if mask.any():
                        idxs = np.flatnonzero(mask)
                        for j in idxs.tolist():
                            value = int(a[j])
                            delta = int(target) - value
                            if delta % _PTR_RESOLVE_OFFSET_STEP:
                                continue
                            holder = pos + j * 8
                            hits.append((holder, delta, region))
                            if len(hits) >= max_hits:
                                break
                pos += size
    finally:
        sock.close()
    hits.sort(key=lambda x: (-_region_priority(x[2]), abs(x[1]), x[0]))
    return hits[:max_hits]


class _ReversePointerIndex:
    """Compact reverse pointer index: pointer value -> holder addresses.

    Two uint64 arrays are kept sorted by pointer value.  This is considerably
    smaller than a Python dict containing millions of integer/list objects and
    allows repeated binary-search queries for different temporary addresses.
    """
    def __init__(self, ip: str, pid: int, maps: list, cancel_event=None,
                 progress_cb=None):
        self.ip = ip
        self.pid = pid
        self.maps = maps
        self.fingerprint = _pointer_map_fingerprint(maps)
        self.values = np.empty(0, dtype=np.uint64)
        self.holders = np.empty(0, dtype=np.uint64)
        self.holder_priority = np.empty(0, dtype=np.uint8)
        self.total_bytes = 0
        self.done_bytes = 0
        self._build(cancel_event, progress_cb)

    def _build(self, cancel_event=None, progress_cb=None):
        readable = _pointer_readable_regions(self.maps)
        if not readable:
            return
        # ps5debug map order is not guaranteed.  searchsorted requires sorted
        # starts; using the raw map order can silently discard valid pointers.
        lookup_maps = sorted(
            (r for r in self.maps if int(r.get("end", 0)) > int(r.get("start", 0))),
            key=lambda r: int(r["start"])
        )
        starts = np.asarray([int(r["start"]) for r in lookup_maps], dtype=np.uint64)
        ends = np.asarray([int(r["end"]) for r in lookup_maps], dtype=np.uint64)
        total = max(sum(int(r["end"]) - int(r["start"]) for r in readable), 1)
        vals_parts, holder_parts = [], []
        sock = _ScanSocket(self.ip, self.pid)
        try:
            for region in readable:
                if cancel_event and cancel_event.is_set():
                    break
                rs, re_ = int(region["start"]), int(region["end"])
                pos = rs + ((-rs) % 8)
                while pos < re_:
                    if cancel_event and cancel_event.is_set():
                        break
                    size = min(_PTR_INDEX_CHUNK, re_ - pos)
                    size -= size % 8
                    if size < 8:
                        break
                    try:
                        raw = sock.read(pos, size, cancel_event)
                    except Exception as exc:
                        add_log(f"Pointer index read error @ {hex(pos)}: {exc}", "warn")
                        self.done_bytes += size
                        pos += size
                        if progress_cb:
                            progress_cb(self.done_bytes, total)
                        continue
                    trim = len(raw) - (len(raw) % 8)
                    if trim >= 8:
                        a = np.frombuffer(raw[:trim], dtype=np.uint64)
                        # A pointer candidate must be a user-space address that
                        # currently falls inside a mapped region.  This removes
                        # ordinary integers while retaining heap/module pointers.
                        plausible = (a >= _ADDR_MIN) & (a <= _ADDR_MAX)
                        idx = np.searchsorted(starts, a, side="right") - 1
                        idx_i = idx.astype(np.int64, copy=False)
                        valid_idx = (idx_i >= 0)
                        if valid_idx.any():
                            ii = np.where(valid_idx, idx_i, 0)
                            plausible &= valid_idx & (a < ends[ii])
                        hit_idx = np.flatnonzero(plausible)
                        if hit_idx.size:
                            vals_parts.append(a[hit_idx].copy())
                            holder_parts.append(
                                pos + hit_idx.astype(np.uint64, copy=False) * np.uint64(8))
                    self.done_bytes += size
                    pos += size
                    if progress_cb:
                        progress_cb(self.done_bytes, total)
        finally:
            sock.close()
        if vals_parts:
            # Concatenation allocates the final arrays; release each family of
            # chunk arrays immediately instead of retaining both chunk lists
            # throughout priority construction and sorting.
            self.values = np.concatenate(vals_parts)
            vals_parts.clear()
            self.holders = np.concatenate(holder_parts)
            holder_parts.clear()
            gc.collect()
            # Cache holder-region priority once.  Querying a dense pointer
            # range must not discard a good module/static pointer merely
            # because many heap pointers occur at smaller offsets.
            map_priority = np.fromiter(
                (_region_priority(r) for r in lookup_maps),
                dtype=np.uint8, count=len(lookup_maps))
            holder_map = np.searchsorted(starts, self.holders, side="right") - 1
            holder_map_i = holder_map.astype(np.int64, copy=False)
            valid = holder_map_i >= 0
            hp = np.zeros(self.holders.shape, dtype=np.uint8)
            if valid.any():
                vi = np.where(valid, holder_map_i, 0)
                inside = valid & (self.holders < ends[vi])
                hp[inside] = map_priority[vi[inside]]
            self.holder_priority = hp
            order = np.argsort(self.values, kind="mergesort")
            self.values = self.values[order]
            self.holders = self.holders[order]
            self.holder_priority = self.holder_priority[order]
        add_log(f"Reverse pointer index built: {len(self.values):,} pointers, "
                f"{self.done_bytes / 1048576:.1f} MiB scanned")

    def query(self, target: int, max_offset: int = _PTR_RESOLVE_OFFSET_MAX,
              step: int = _PTR_RESOLVE_OFFSET_STEP,
              max_hits: int = _PTR_RESOLVE_MAX_HITS) -> list:
        """Return (holder, offset) for values in [target-max_offset, target].

        The old implementation performed one binary search per 8-byte offset.
        That becomes expensive with an 8 KiB window.  This version performs one
        range lookup and derives the offset directly from the matched pointer
        value, which keeps the larger search window cheap.
        """
        if self.values.size == 0:
            return []
        target = int(target)
        max_offset = max(0, int(max_offset))
        step = max(1, int(step))
        max_hits = max(1, int(max_hits))
        # Search on both sides of the target.  A valid field offset may be
        # positive OR negative: target = pointer_value + signed_offset.
        low = max(_ADDR_MIN, target - max_offset)
        high = min(_ADDR_MAX, target + max_offset)
        if low > high:
            return []
        lo = int(np.searchsorted(self.values, np.uint64(low), side="left"))
        hi = int(np.searchsorted(self.values, np.uint64(high), side="right"))
        if hi <= lo:
            return []

        values = self.values[lo:hi]
        holders = self.holders[lo:hi]
        priorities = self.holder_priority[lo:hi]
        # Filter alignment first, then rank candidates by region quality before
        # distance.  This prevents a dense heap from hiding a useful module root.
        deltas = np.asarray(target, dtype=np.int64) - values.astype(np.int64, copy=False)
        mask = (np.abs(deltas) <= max_offset) & ((deltas % step) == 0)
        if not mask.any():
            return []
        idxs = np.flatnonzero(mask)
        # Stable deterministic ordering: better region, shorter offset, holder.
        idxs = sorted(idxs.tolist(),
                      key=lambda i: (-int(priorities[i]), abs(int(deltas[i])),
                                     int(deltas[i]) < 0, int(holders[i])))[:max_hits]
        out = []
        seen = set()
        for i in idxs:
            holder = int(holders[i])
            delta = int(deltas[i])
            key = (holder, delta)
            if key not in seen:
                seen.add(key)
                out.append((holder, delta))
        return out


class _DiskReversePointerIndex:
    """Shard-backed reverse index for processes too large to retain in RAM.

    Each shard is independently sorted and stored as two ``.npy`` arrays.
    Queries binary-search every memory-mapped value shard and globally rank the
    small union of hits.  Peak construction memory is bounded by one shard.
    """
    def __init__(self, ip: str, pid: int, maps: list, cancel_event=None,
                 progress_cb=None):
        self.ip = ip
        self.pid = pid
        self.maps = maps
        self.fingerprint = _pointer_map_fingerprint(maps)
        self.total_bytes = 0
        self.done_bytes = 0
        self.shards = []
        self._tmpdir = Path(tempfile.mkdtemp(prefix="rdx_ptr_"))
        try:
            self._build(cancel_event, progress_cb)
        except BaseException:
            self.close()
            raise

    def close(self):
        """Close mapped shards and remove this index's private temporary files."""
        self.shards.clear()
        try:
            for path in self._tmpdir.iterdir():
                path.unlink(missing_ok=True)
            self._tmpdir.rmdir()
        except Exception:
            pass

    def _build(self, cancel_event=None, progress_cb=None):
        readable = _coalesce_pointer_regions(_pointer_readable_regions(self.maps))
        lookup_maps = sorted(
            (r for r in self.maps if int(r.get("end", 0)) > int(r.get("start", 0))),
            key=lambda r: int(r["start"]))
        starts = np.asarray([int(r["start"]) for r in lookup_maps], dtype=np.uint64)
        ends = np.asarray([int(r["end"]) for r in lookup_maps], dtype=np.uint64)
        total = max(sum(int(r["end"]) - int(r["start"]) for r in readable), 1)
        self.total_bytes = total
        tasks = _queue.Queue()
        shard_no = 0
        for region in readable:
            pos = int(region["start"]) + ((-int(region["start"])) % 8)
            region_end = int(region["end"])
            while pos < region_end:
                size = min(_PTR_DISK_SHARD_BYTES, region_end - pos)
                size -= size % 8
                if size < 8:
                    break
                tasks.put((shard_no, pos, size))
                shard_no += 1
                pos += size

        requested_workers = min(_PTR_DISK_WORKERS, max(tasks.qsize(), 1))
        result_lock = threading.Lock()
        fatal_errors = []

        # Establish readers before queueing sentinels.  Previously, if every
        # worker failed during connection, Queue.join() waited forever because
        # no thread remained to acknowledge the queued scan tasks.
        reader_sockets = []
        for _ in range(requested_workers):
            try:
                reader_sockets.append(_ScanSocket(self.ip, self.pid))
            except Exception as exc:
                fatal_errors.append((None, exc))
        if not reader_sockets:
            raise ConnectionError(f"disk pointer index could not connect: "
                                  f"{fatal_errors[0][1]}")
        for _ in reader_sockets:
            tasks.put(None)

        def _worker(sock):
            try:
                while True:
                    task = tasks.get()
                    try:
                        if task is None:
                            return
                        number, pos, size = task
                        if cancel_event and cancel_event.is_set():
                            continue
                        raw = sock.read(pos, size, cancel_event)
                        trim = len(raw) - (len(raw) % 8)
                        paths = []
                        if trim >= 8:
                            values = np.frombuffer(raw[:trim], dtype=np.uint64)
                            plausible = (values >= _ADDR_MIN) & (values <= _ADDR_MAX)
                            map_idx = np.searchsorted(starts, values, side="right") - 1
                            map_idx_i = map_idx.astype(np.int64, copy=False)
                            valid = map_idx_i >= 0
                            if valid.any():
                                safe_idx = np.where(valid, map_idx_i, 0)
                                plausible &= valid & (values < ends[safe_idx])
                            hit_idx = np.flatnonzero(plausible)
                            if hit_idx.size:
                                kept_values = values[hit_idx].copy()
                                kept_holders = (np.uint64(pos) +
                                    hit_idx.astype(np.uint64, copy=False) * np.uint64(8))
                                # Partition by the pointer's upper 32 bits.  A
                                # graph query then opens only the address family
                                # containing its target instead of every shard.
                                prefixes = kept_values >> np.uint64(32)
                                for prefix in np.unique(prefixes).tolist():
                                    group = prefixes == np.uint64(prefix)
                                    group_values = kept_values[group]
                                    group_holders = kept_holders[group]
                                    order = np.argsort(group_values, kind="mergesort")
                                    value_path = self._tmpdir / (
                                        f"v{number:05d}_{int(prefix):08x}.npy")
                                    holder_path = self._tmpdir / (
                                        f"h{number:05d}_{int(prefix):08x}.npy")
                                    np.save(value_path, group_values[order],
                                            allow_pickle=False)
                                    np.save(holder_path, group_holders[order],
                                            allow_pickle=False)
                                    paths.append((int(prefix), value_path,
                                                  holder_path))
                        with result_lock:
                            if paths:
                                self.shards.extend(paths)
                            self.done_bytes += size
                            if progress_cb:
                                progress_cb(self.done_bytes, total)
                    except InterruptedError:
                        if cancel_event:
                            cancel_event.set()
                    except Exception as exc:
                        with result_lock:
                            fatal_errors.append((task, exc))
                            if task is not None:
                                self.done_bytes += task[2]
                    finally:
                        tasks.task_done()
            finally:
                sock.close()

        workers = [threading.Thread(target=_worker, args=(sock,), daemon=True)
                   for sock in reader_sockets]
        for worker in workers:
            worker.start()
        tasks.join()
        for worker in workers:
            worker.join()
        self.shards.sort(key=lambda shard: (shard[0], shard[1].name))
        if fatal_errors and not self.shards and not (cancel_event and cancel_event.is_set()):
            raise ConnectionError(f"disk pointer index failed: {fatal_errors[0][1]}")
        for task, exc in fatal_errors[:8]:
            where = hex(task[1]) if task is not None else "connect"
            add_log(f"Disk pointer index error @ {where}: {exc}", "warn")
        add_log(f"Disk pointer index built: {len(self.shards):,} shards, "
                f"{self.done_bytes / 1048576:.1f} MiB scanned")

    def query(self, target: int, max_offset: int = _PTR_RESOLVE_OFFSET_MAX,
              step: int = _PTR_RESOLVE_OFFSET_STEP,
              max_hits: int = _PTR_RESOLVE_MAX_HITS) -> list:
        target = int(target)
        low = max(_ADDR_MIN, target - max(0, int(max_offset)))
        high = min(_ADDR_MAX, target + max(0, int(max_offset)))
        candidates = []
        region_starts, region_rows = _build_region_lookup(self.maps)
        wanted_prefixes = set(range(low >> 32, (high >> 32) + 1))
        for prefix, value_path, holder_path in self.shards:
            if prefix not in wanted_prefixes:
                continue
            values = np.load(value_path, mmap_mode="r", allow_pickle=False)
            lo = int(np.searchsorted(values, np.uint64(low), side="left"))
            hi = int(np.searchsorted(values, np.uint64(high), side="right"))
            if hi <= lo:
                continue
            holders = np.load(holder_path, mmap_mode="r", allow_pickle=False)
            local = []
            for value, holder in zip(values[lo:hi].tolist(), holders[lo:hi].tolist()):
                delta = target - int(value)
                if abs(delta) <= max_offset and delta % max(1, int(step)) == 0:
                    local.append((int(holder), delta))
            if len(local) > max_hits:
                local.sort(key=lambda item: (
                    -_region_priority(_region_for_addr(
                        item[0], region_starts, region_rows) or {}),
                    abs(item[1]), item[1] < 0, item[0]))
                del local[max_hits:]
            candidates.extend(local)
        candidates.sort(key=lambda item: (
            -_region_priority(_region_for_addr(item[0], region_starts, region_rows) or {}),
            abs(item[1]), item[1] < 0, item[0]))
        out, seen = [], set()
        for item in candidates:
            if item not in seen:
                seen.add(item)
                out.append(item)
                if len(out) >= max(1, int(max_hits)):
                    break
        return out


def _get_reverse_pointer_index(ip: str, pid: int, cancel_event=None,
                               progress_cb=None):
    """Get or build the cached reverse pointer index for the current map layout."""
    maps = _get_maps_cached(ip, pid)
    fp = _pointer_map_fingerprint(maps)
    if fp not in _pointer_region_class_cache:
        try:
            classified = ps5_classify_regions(ip, pid)
            if len(_pointer_region_class_cache) >= 4:
                _pointer_region_class_cache.clear()
            _pointer_region_class_cache[fp] = classified
            uncached_mib = sum(
                max(0, int(r["end"]) - int(r["start"]))
                for r in classified if int(r.get("flags", 0)) & 1
            ) / 1048576
            add_log(f"Region classifier: {len(classified):,} ranges; "
                    f"{uncached_mib:,.1f} MiB uncached/GPU memory excluded")
            if len(classified) == 8192:
                add_log("Region classifier reached the payload's 8,192-row cap; "
                        "unreported maps retain normal safe fallback handling",
                        "warn")
        except Exception as exc:
            _pointer_region_class_cache[fp] = []
            add_log(f"Region classifier unavailable; using map safeguards: {exc}",
                    "warn")
    key = (ip, int(pid), int(state.get("session", 0)))
    with _pointer_index_lock:
        cached = _pointer_index_cache.get(key)
        if cached and cached[0] == fp:
            return cached[1], maps, False
    readable_bytes = sum(
        int(r["end"]) - int(r["start"])
        for r in _pointer_readable_regions(maps))
    index_type = (_DiskReversePointerIndex
                  if readable_bytes >= _PTR_DISK_INDEX_THRESHOLD
                  else _ReversePointerIndex)
    idx = index_type(ip, pid, maps, cancel_event, progress_cb)
    if cancel_event and cancel_event.is_set():
        if hasattr(idx, "close"):
            idx.close()
        return idx, maps, True
    with _pointer_index_lock:
        # A later builder may have won the race; keep the first completed index.
        cached = _pointer_index_cache.get(key)
        if cached and cached[0] == fp:
            if hasattr(idx, "close"):
                idx.close()
            return cached[1], maps, False
        _pointer_index_cache[key] = (fp, idx)
    return idx, maps, True


def _invalidate_pointer_index(ip=None, pid=None):
    with _pointer_index_lock:
        if ip is None and pid is None:
            for _, idx in _pointer_index_cache.values():
                if hasattr(idx, "close"):
                    idx.close()
            _pointer_index_cache.clear()
            return
        for key in list(_pointer_index_cache):
            if isinstance(key, tuple) and len(key) >= 2 and key[0] == ip and int(key[1]) == int(pid):
                entry = _pointer_index_cache.pop(key, None)
                if entry and hasattr(entry[1], "close"):
                    entry[1].close()


def _candidate_confidence(c: dict) -> int:
    """Convert resolver evidence into a stable 0..100 user-facing confidence."""
    if not c.get("verified"):
        return 0
    depth = max(1, int(c.get("depth", 1)))
    conf = 62
    if c.get("module_name"):
        conf += 18
    if c.get("static"):
        conf += 8
    conf += max(0, 8 - depth * 2)
    offsets = [int(x) for x in c.get("offsets", [])]
    if offsets:
        aligned = sum(1 for x in offsets if x % _PTR_RESOLVE_OFFSET_STEP == 0)
        reasonable = sum(1 for x in offsets if abs(x) <= 0x400)
        conf += round(2 * aligned / len(offsets))
        conf += round(2 * reasonable / len(offsets))
    return max(0, min(100, int(conf)))


def _verify_candidate_twice(ip: str, pid: int, candidate: dict, target_addr: int,
                            cancel_event=None) -> bool:
    """Require two fresh pointer-chain resolutions before accepting a candidate."""
    if cancel_event and cancel_event.is_set():
        return False
    base = int(candidate.get("base", 0))
    offsets = [int(x) for x in candidate.get("offsets", [])]
    ok1, resolved1, steps1 = _resolve_pointer_chain(
        ip, pid, base, offsets, int(candidate.get("terminal_offset", 0)))
    if not (ok1 and int(resolved1) == int(target_addr)):
        candidate["verified"] = False
        candidate["resolved"] = int(resolved1 or 0)
        candidate["steps"] = steps1
        return False
    # Fresh map/read pass: do not rely on the first read or cached result.
    time.sleep(0.01)
    if cancel_event and cancel_event.is_set():
        return False
    ok2, resolved2, steps2 = _resolve_pointer_chain(
        ip, pid, base, offsets, int(candidate.get("terminal_offset", 0)))
    verified = bool(ok2 and int(resolved2) == int(target_addr))
    candidate["verified"] = verified
    candidate["resolved"] = int(resolved2 or 0)
    candidate["steps"] = steps2
    candidate["verification_passes"] = 2 if verified else 1
    return verified


def _resolve_permanent_candidates(ip: str, pid: int, target_addr: int,
                                   max_depth: int = 5, cancel_event=None,
                                   progress_cb=None) -> dict:
    """Resolve a dynamic address using a fast direct pass, then a priority-guided graph."""
    max_depth = max(1, min(int(max_depth), MAX_CHAIN_DEPTH))
    maps = _get_maps_cached(ip, pid)
    region_starts, region_rows = _build_region_lookup(maps)

    # Fast path: a direct reference is common and avoids constructing millions
    # of index entries when the target is already held by a stable object.
    direct = _fast_direct_pointer_hits(ip, pid, int(target_addr), maps, cancel_event)
    fast_candidates = []
    for holder, off, region in direct:
        if not _is_static_region(region):
            continue
        module_name, module_base, module_rel = _module_info_for_addr(holder, maps)
        c = {
            "base": holder, "offsets": [off], "depth": 1,
            "region": region.get("name", "") or "static", "static": True,
            "module_name": module_name or "main", "module_base": module_base,
            "module_relative_offset": module_rel, "score": 0.0, "verified": False,
        }
        if _verify_candidate_twice(ip, pid, c, int(target_addr), cancel_event):
            c["score"] = 220 + (35 if int(region.get("prot", 0)) & 0x4 else 0)
            c["confidence"] = _candidate_confidence(c)
            fast_candidates.append(c)
    if fast_candidates:
        fast_candidates.sort(key=lambda c: -c["score"])
        return {"candidates": fast_candidates, "index_built": False,
                "maps": maps, "method": "fast-direct"}

    # Before constructing a multi-gigabyte exhaustive index, walk the natural
    # object locality: modules first, then the address family containing each
    # discovered parent.  This is the common-case algorithm used by practical
    # pointer scanners and reuses the exhaustive index only when locality fails.
    local_hits = pointer_chain_scan(
        ip, pid, int(target_addr), max_depth=max_depth,
        cancel_event=cancel_event, progress_cb=progress_cb)
    local_candidates = []
    for hit in local_hits:
        if not hit.get("static"):
            continue
        holder = int(hit["base"])
        module_name, module_base, module_rel = _module_info_for_addr(holder, maps)
        candidate = dict(hit)
        candidate.update({
            "module_name": module_name or "main",
            "module_base": module_base,
            "module_relative_offset": module_rel,
            "score": float(210 - 7 * int(hit.get("depth", 1))),
            "verified": False,
        })
        if _verify_candidate_twice(
                ip, pid, candidate, int(target_addr), cancel_event):
            candidate["confidence"] = _candidate_confidence(candidate)
            local_candidates.append(candidate)
    if local_candidates:
        local_candidates.sort(key=lambda c: (-c["score"], c["base"]))
        return {"candidates": local_candidates, "index_built": False,
                "maps": maps, "method": "locality-first"}

    index, maps, built = _get_reverse_pointer_index(
        ip, pid, cancel_event, progress_cb)
    if cancel_event and cancel_event.is_set():
        return {"candidates": [], "index_built": built, "maps": maps}

    # Priority queue: higher-quality regions, shorter offsets and shorter chains
    # are expanded first. This replaces list.pop(0), which is O(n).
    queue = []
    serial = 0
    heapq.heappush(queue, (0, 0, serial, int(target_addr), [], {int(target_addr)}))
    found = []
    best_depth_for_node = {}
    processed = 0

    while queue and processed < _PTR_RESOLVE_MAX_NODES:
        if cancel_event and cancel_event.is_set():
            break
        _, _, _, current, rev_edges, visited = heapq.heappop(queue)
        depth = len(rev_edges) + 1
        if depth > max_depth:
            continue

        # Deterministic search: every node uses the same full offset window.
        # Do not stop early because a nearby heap reference happened to exist;
        # that can hide a valid module/static root farther from the target.
        hits = index.query(current, _PTR_RESOLVE_OFFSET_MAX,
                           _PTR_RESOLVE_OFFSET_STEP, _PTR_RESOLVE_MAX_HITS)

        for holder, off in hits:
            if holder in visited:
                continue
            region = _region_for_addr(holder, region_starts, region_rows)
            if region is None:
                continue
            is_static = _is_static_region(region)
            edge = (holder, off, region)
            new_rev = rev_edges + [edge]
            if is_static:
                chain = [e[1] for e in reversed(new_rev)]
                module_name, module_base, module_rel = _module_info_for_addr(holder, maps)
                aligned = sum(1 for x in chain if int(x) % 8 == 0)
                reasonable = sum(1 for x in chain if abs(int(x)) <= 0x400)
                prot = int(region.get("prot", 0))
                name = str(region.get("name", "") or "")
                executable_root = bool(prot & 0x4) or name.startswith("executable")
                named_module = bool(name) and name not in _HEAP_NAME_HINTS
                score = 150 + max(0, 50 - len(chain) * 7)
                score += 35 if executable_root else 0
                score += 20 if named_module else 0
                score += int(14 * aligned / max(len(chain), 1))
                score += int(12 * reasonable / max(len(chain), 1))
                score -= min(abs(int(off)) // 0x400, 12)
                found.append({
                    "base": holder, "offsets": chain, "depth": len(chain),
                    "region": name or "static", "static": True,
                    "module_name": module_name or "main",
                    "module_base": module_base,
                    "module_relative_offset": module_rel,
                    "score": float(score), "verified": False,
                })
                if len(found) >= _PTR_RESOLVE_MAX_FOUND:
                    break
            elif depth < max_depth:
                old = best_depth_for_node.get(holder)
                if old is None or len(new_rev) < old:
                    best_depth_for_node[holder] = len(new_rev)
                    serial += 1
                    priority = (
                        -_region_priority(region),
                        abs(int(off)) + len(new_rev) * 64,
                        serial, holder, new_rev, visited | {holder}
                    )
                    if len(queue) < _PTR_RESOLVE_MAX_NODES:
                        heapq.heappush(queue, priority)
            processed += 1
        if len(found) >= _PTR_RESOLVE_MAX_FOUND:
            break
        if progress_cb:
            progress_cb(processed, max(_PTR_RESOLVE_MAX_NODES, 1))

    unique = {}
    for c in found:
        key = (c["module_name"], c["module_relative_offset"], tuple(c["offsets"]))
        old = unique.get(key)
        if old is None or c["score"] > old["score"]:
            unique[key] = c
    candidates = sorted(unique.values(), key=lambda c: -c["score"])

    def verify_batch(items):
        verified_count = 0
        for c in items:
            if cancel_event and cancel_event.is_set():
                break
            if _verify_candidate_twice(ip, pid, c, int(target_addr), cancel_event):
                verified_count += 1
        return verified_count

    # Verify candidates in deterministic batches.  The old fixed [:32] slice
    # silently ignored valid lower-ranked chains. Stop once the UI has enough
    # verified alternatives, otherwise exhaust the bounded candidate list.
    verified_total = 0
    for batch_start in range(0, len(candidates), 32):
        verified_total += verify_batch(candidates[batch_start:batch_start + 32])
        if verified_total >= 8 or (cancel_event and cancel_event.is_set()):
            break

    # Heap/object pointers can move while the map layout remains unchanged. If
    # the cached index yielded no verified result, rebuild it once and retry.
    if (not any(c.get("verified") for c in candidates) and
            not built and not (cancel_event and cancel_event.is_set())):
        add_log("No verified chain from cached pointer index; rebuilding once", "warn")
        _invalidate_pointer_index(ip, pid)
        fresh_index, fresh_maps, fresh_built = _get_reverse_pointer_index(
            ip, pid, cancel_event, progress_cb)
        if fresh_index is not None:
            # Re-run the graph search against the fresh index by temporarily
            # using the same deterministic traversal logic. Keep this bounded
            # and avoid recursively calling the whole resolver.
            maps = fresh_maps
            region_starts, region_rows = _build_region_lookup(maps)
            queue = [(0, 0, 0, int(target_addr), [], {int(target_addr)})]
            fresh_found = []
            best_depth_for_node = {}
            serial2 = 0
            processed2 = 0
            while queue and processed2 < _PTR_RESOLVE_MAX_NODES:
                if cancel_event and cancel_event.is_set():
                    break
                _, _, _, current, rev_edges, visited = heapq.heappop(queue)
                depth = len(rev_edges) + 1
                if depth > max_depth:
                    continue
                hits = fresh_index.query(current, _PTR_RESOLVE_OFFSET_MAX,
                                         _PTR_RESOLVE_OFFSET_STEP, _PTR_RESOLVE_MAX_HITS)
                for holder, off in hits:
                    if holder in visited:
                        continue
                    region = _region_for_addr(holder, region_starts, region_rows)
                    if region is None:
                        continue
                    new_rev = rev_edges + [(holder, off, region)]
                    if _is_static_region(region):
                        chain = [e[1] for e in reversed(new_rev)]
                        mn, mb, mr = _module_info_for_addr(holder, maps)
                        fresh_found.append({
                            "base": holder, "offsets": chain, "depth": len(chain),
                            "region": region.get("name", "") or "static", "static": True,
                            "module_name": mn, "module_base": mb,
                            "module_relative_offset": mr, "score": 0.0,
                            "verified": False,
                        })
                    elif depth < max_depth:
                        old = best_depth_for_node.get(holder)
                        if old is None or len(new_rev) < old:
                            best_depth_for_node[holder] = len(new_rev)
                            serial2 += 1
                            heapq.heappush(queue, (
                                -_region_priority(region),
                                abs(int(off)) + len(new_rev) * 64, serial2,
                                holder, new_rev, visited | {holder}))
                    processed2 += 1
                    if len(fresh_found) >= _PTR_RESOLVE_MAX_FOUND:
                        break
                if len(fresh_found) >= _PTR_RESOLVE_MAX_FOUND:
                    break
            candidates = fresh_found
            built = fresh_built
            verified_total = 0
            for batch_start in range(0, len(candidates), 32):
                verified_total += verify_batch(
                    candidates[batch_start:batch_start + 32])
                if verified_total >= 8 or (cancel_event and cancel_event.is_set()):
                    break

    for c in candidates:
        c["confidence"] = _candidate_confidence(c)

    # Single canonical ranking used for both the displayed winner and selection.
    candidates.sort(key=lambda c: (
        not bool(c.get("verified")),
        not bool(c.get("module_name")),
        int(c.get("depth", 99)),
        -_region_priority(_region_for_addr(int(c.get("base", 0)), region_starts, region_rows) or {}),
        sum(0 if int(x) % 8 == 0 else 1 for x in c.get("offsets", [])),
        sum(abs(int(x)) for x in c.get("offsets", [])),
        str(c.get("module_name", "")),
        int(c.get("module_relative_offset", 0)),
        tuple(int(x) for x in c.get("offsets", [])),
        int(c.get("base", 0)),
    ))
    return {"candidates": candidates, "index_built": built,
            "maps": maps, "method": "reverse-index"}


def _pointer_readable_regions(maps: list) -> list:
    """Return useful readable regions, excluding obvious giant reserved maps."""
    PROT_READ = 0x1
    PROT_EXEC = 0x4
    MAX_REGION = 0x40000000  # 1 GiB; avoid obvious GPU/VM reservations
    out = []
    classified = _pointer_region_class_cache.get(_pointer_map_fingerprint(maps), [])
    uncached = sorted(
        (int(row["start"]), int(row["end"])) for row in classified
        if int(row.get("flags", 0)) & 1 and int(row["end"]) > int(row["start"])
    )
    uncached_starts = [row[0] for row in uncached]
    for r in maps:
        start, end = int(r.get("start", 0)), int(r.get("end", 0))
        prot = int(r.get("prot", 0))
        if end <= start or not (prot & PROT_READ):
            continue
        if prot == PROT_EXEC:
            continue
        if end - start > MAX_REGION:
            continue
        name = str(r.get("name", "") or "")
        if name.startswith("libSce"):
            continue
        # Never discard a module/static root.  For anonymous writable memory,
        # however, an overlap with an explicitly uncached classifier row means
        # GPU/Garlic backing: reads are exceptionally slow and cannot contain a
        # stable CPU pointer chain.
        if not _is_static_region(r) and uncached:
            ui = bisect.bisect_left(uncached_starts, end)
            if ui > 0 and start < uncached[ui - 1][1]:
                continue
        out.append(r)
    # Static/module regions first.  This lets a useful short chain be surfaced
    # quickly, while heap regions are still searched when they are needed to
    # extend a chain.
    out.sort(key=lambda r: (not _is_static_region(r), r["start"]))
    return out


def _coalesce_pointer_regions(regions: list) -> list:
    """Merge adjacent scan ranges to avoid one network request per tiny map.

    PS5 games commonly expose thousands of contiguous 64 KiB mappings.  Their
    individual names/protections are still retained in the original map table
    used to classify hits; these rows are only transport ranges.
    """
    # Preserve the caller's priority order (static roots first).  Sorting here
    # used to silently undo _pointer_readable_regions()'s smart ordering and
    # made the scanner trawl low-value heaps before module roots.
    ordered = list(regions)
    merged = []
    for region in ordered:
        start, end = int(region["start"]), int(region["end"])
        is_static = bool(_is_static_region(region))
        if end <= start:
            continue
        if (merged and start == merged[-1]["end"] and
                is_static == merged[-1]["static"]):
            merged[-1]["end"] = end
        else:
            merged.append({"start": start, "end": end,
                           "static": is_static})
    return merged


def _pointer_scan_chunk(sock: _ScanSocket, start: int, size: int,
                        target_arr: np.ndarray, target_chains: dict,
                        maps_sorted: list, map_starts: np.ndarray,
                        map_ends: np.ndarray, cancel_event=None):
    """Read one chunk and return matching (holder, target, offset, chain) hits."""
    raw = sock.read(start, size, cancel_event)
    trim = len(raw) - (len(raw) % 8)
    if trim < 8:
        return []
    vals = np.frombuffer(raw[:trim], dtype=np.uint64)
    # Match pointer values to target *ranges* instead of enumerating every
    # target-offset value. The nearest sorted target is sufficient to decide
    # whether at least one target lies within the signed offset window; exact
    # target/chain combinations are expanded only for the tiny set of hits.
    positions = np.searchsorted(target_arr, vals, side="left")
    mask = np.zeros(vals.shape, dtype=bool)
    right = positions < target_arr.size
    if right.any():
        ri = positions[right]
        rv = vals[right]
        rd = target_arr[ri].astype(np.int64) - rv.astype(np.int64)
        mask[right] |= (np.abs(rd) <= _PTR_STRUCT_MAX) & (
            (rd % _PTR_STRUCT_STEP) == 0)
    left = positions > 0
    if left.any():
        li = positions[left] - 1
        lv = vals[left]
        ld = target_arr[li].astype(np.int64) - lv.astype(np.int64)
        mask[left] |= (np.abs(ld) <= _PTR_STRUCT_MAX) & (
            (ld % _PTR_STRUCT_STEP) == 0)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []

    holders = start + indices.astype(np.uint64) * 8
    hits = []
    # Usually only a tiny number of slots match a pointer target.  Keep the
    # Python work limited to those hits rather than iterating over the chunk.
    for idx, holder_u in zip(indices.tolist(), holders.tolist()):
        value = int(vals[idx])
        lo = int(np.searchsorted(target_arr,
                                 np.uint64(max(_ADDR_MIN, value - _PTR_STRUCT_MAX)),
                                 side="left"))
        hi = int(np.searchsorted(target_arr,
                                 np.uint64(min(_ADDR_MAX, value + _PTR_STRUCT_MAX)),
                                 side="right"))
        for target_u in target_arr[lo:hi].tolist():
            target = int(target_u)
            soff = target - value
            if soff % _PTR_STRUCT_STEP:
                continue
            chain = target_chains[target]
            # A pointer holder itself must be inside a mapped region.
            mi = int(np.searchsorted(map_starts, holder_u, side="right") - 1)
            if mi < 0 or holder_u >= int(map_ends[mi]):
                continue
            region = maps_sorted[mi]
            hits.append((int(holder_u), int(target), int(soff), chain,
                         bool(_is_static_region(region)),
                         region.get("name", "") or "anon"))
    return hits


def pointer_chain_scan(ip: str, pid: int,
                       target_addr: int,
                       max_depth: int = _PTR_DEPTH_DEFAULT,
                       cancel_event=None,
                       progress_cb=None) -> list:
    """Find static pointer chains to ``target_addr`` without full-memory caching.

    The search works backwards from the temporary address.  At each level it
    looks for pointers to ``target - offset`` for aligned offsets 0..0x200.  A
    heap holder becomes the target for the next level; a static holder becomes
    a candidate immediately.

    Important differences from the old scanner:
      * memory is streamed in 32 MiB chunks instead of retained for every level;
      * offsets are tested at every pointer level, not only level 1;
      * heap expansion is bounded (beam search) so one noisy heap cannot create
        millions of chains;
      * the default depth is 4, with the UI able to request deeper scans later;
      * the first static hit can be returned without waiting for six full passes.
    """
    started = time.monotonic()
    candidates = []

    max_depth = max(1, min(int(max_depth), MAX_CHAIN_DEPTH))
    try:
        maps = _get_maps_cached(ip, pid)
    except Exception as exc:
        add_log(f"Pointer scan: could not fetch memory map: {exc}", "error")
        return []
    if not maps:
        add_log("Pointer scan: memory map is empty", "warn")
        return []

    readable_maps = _pointer_readable_regions(maps)
    readable = _coalesce_pointer_regions(readable_maps)
    if not readable:
        add_log("Pointer scan: no readable data regions", "warn")
        return []

    maps_sorted = sorted(maps, key=lambda r: r["start"])
    map_starts = np.asarray([r["start"] for r in maps_sorted], dtype=np.uint64)
    map_ends   = np.asarray([r["end"] for r in maps_sorted], dtype=np.uint64)

    # target -> offsets already discovered, stored as tuples so we can keep the
    # best (shortest) chain when multiple paths reach the same holder.
    current_targets = {int(target_addr): []}
    total_bytes = max(sum(int(r["end"]) - int(r["start"]) for r in readable), 1)

    for depth in range(1, max_depth + 1):
        if cancel_event and cancel_event.is_set():
            break
        if not current_targets:
            add_log(f"Pointer scan depth {depth}: no heap targets remain")
            break

        # Keep the beam bounded before multiplying by struct offsets.
        if len(current_targets) > _PTR_BEAM_MAX:
            current_targets = dict(list(current_targets.items())[:_PTR_BEAM_MAX])
            add_log(f"Depth {depth}: beam capped at {_PTR_BEAM_MAX:,} targets", "warn")

        target_arr = np.sort(np.fromiter(current_targets.keys(), dtype=np.uint64))
        next_targets = {}
        static_hits = 0
        heap_hits = 0
        depth_done = 0

        add_log(f"Pointer scan depth {depth}: {len(current_targets):,} targets, "
                f"interval-matched ±{hex(_PTR_STRUCT_MAX)}")

        sock = None
        try:
            sock = _ScanSocket(ip, pid)
            target_prefixes = {int(tgt) >> 32 for tgt in current_targets}
            depth_regions = sorted(
                readable,
                key=lambda r: (
                    not bool(r.get("static")),
                    (int(r["start"]) >> 32) not in target_prefixes,
                    int(r["start"])))
            for region in depth_regions:
                if cancel_event and cancel_event.is_set():
                    break
                # On the final level only static holders can produce a
                # permanent chain. Heap holders cannot be expanded again, so
                # reading them is both useless and potentially hundreds of MiB.
                if depth == max_depth and not region.get("static"):
                    add_log(f"Depth {depth}: static-root pass complete; "
                            "skipping terminal heap scan")
                    break
                is_local = (int(region["start"]) >> 32) in target_prefixes
                if (not region.get("static") and not is_local and next_targets):
                    add_log(f"Depth {depth}: local heap produced "
                            f"{len(next_targets):,} parents; deferring unrelated "
                            "address families")
                    break
                rstart, rend = int(region["start"]), int(region["end"])
                start = rstart + ((-rstart) % 8)
                chunk_limit = _PTR_CHUNK
                while start < rend:
                    if cancel_event and cancel_event.is_set():
                        break
                    size = min(chunk_limit, rend - start)
                    size -= size % 8
                    if size < 8:
                        break
                    try:
                        hits = _pointer_scan_chunk(
                            sock, start, size, target_arr, current_targets,
                            maps_sorted, map_starts, map_ends, cancel_event)
                    except Exception as exc:
                        add_log(f"ptr scan read err @ {hex(start)}: {exc}", "warn")
                        if size > 0x100000:
                            chunk_limit = max(0x100000, size // 2)
                            add_log(f"Retrying {hex(start)} with "
                                    f"{chunk_limit / 1048576:.0f} MiB chunks",
                                    "warn")
                            continue
                        start += size
                        depth_done += size
                        continue

                    for holder, target, soff, old_chain, is_static, rname in hits:
                        # The search walks backward (target -> holder), while
                        # resolution walks forward (static root -> target).
                        # Each newly discovered outer offset therefore belongs
                        # at the front. Appending produced invalid chains for
                        # every depth greater than one.
                        new_chain = [soff] + list(old_chain)
                        if is_static:
                            candidates.append({
                                "base": holder,
                                "offsets": new_chain,
                                "depth": depth,
                                "region": rname,
                                "static": True,
                            })
                            static_hits += 1
                        elif depth < max_depth:
                            old = next_targets.get(holder)
                            if old is None or len(new_chain) < len(old):
                                if len(next_targets) < _PTR_BEAM_MAX or old is not None:
                                    next_targets[holder] = new_chain
                            heap_hits += 1
                        else:
                            candidates.append({
                                "base": holder,
                                "offsets": new_chain,
                                "depth": depth,
                                "region": rname,
                                "static": False,
                            })

                    start += size
                    depth_done += size
                    if chunk_limit < _PTR_CHUNK:
                        chunk_limit = min(_PTR_CHUNK, chunk_limit * 2)
                    if progress_cb:
                        # Map the whole pass into this depth's slice.  This is
                        # real progress, unlike the old spinner/heartbeat.
                        overall = ((depth - 1) * total_bytes + depth_done)
                        progress_cb(overall, total_bytes * max_depth)
        finally:
            if sock:
                sock.close()

        add_log(f"Depth {depth}: {static_hits} static, {heap_hits} heap expansions")
        current_targets = next_targets

        # A broad signed-offset sweep can produce a static coincidence whose
        # heap object changes before the pass ends.  Never stop on an unverified
        # hit: resolve each fresh chain twice and discard stale candidates.
        depth_static = [c for c in candidates
                        if c["static"] and c["depth"] == depth]
        verified_static = []
        for candidate in depth_static:
            if _verify_candidate_twice(
                    ip, pid, candidate, int(target_addr), cancel_event):
                verified_static.append(candidate)
        if depth_static:
            stale_ids = {id(c) for c in depth_static if c not in verified_static}
            candidates = [c for c in candidates if id(c) not in stale_ids]
            if stale_ids:
                add_log(f"Depth {depth}: rejected {len(stale_ids)} stale static "
                        "candidate(s)", "warn")

        # Once a verified static root exists at a short depth, avoid unrelated
        # deeper passes.
        if verified_static:
            add_log(f"Pointer scan found {sum(1 for c in candidates if c['static'])} "
                    f"verified static candidate(s) at depth {depth}")
            break

    candidates.sort(key=lambda c: (not c["static"], c["depth"], c["base"]))
    elapsed = max(time.monotonic() - started, 1e-9)
    add_log(f"Pointer chain scan: {len(candidates)} candidates "
            f"({sum(1 for c in candidates if c['static'])} static) "
            f"in {elapsed:.1f}s")
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


def _runtime_pointer_base(cheat: dict) -> int:
    """Resolve a stored module-relative pointer root for the current ASLR layout.

    New resolver cheats store module_name + module_relative_offset.  Legacy
    cheats without those fields continue using their absolute ``base`` field.
    """
    if cheat.get("module_name") and "module_relative_offset" in cheat:
        maps = _get_maps_cached(state["ip"], state["pid"])
        wanted = str(cheat["module_name"])
        same = [r for r in maps if str(r.get("name", "") or "") == wanted]
        if same:
            base = min(int(r["start"]) for r in same)
            return base + int(cheat["module_relative_offset"])
        if wanted == "main":
            static = [r for r in maps if _is_static_region(r)]
            if static:
                return min(int(r["start"]) for r in static) + int(cheat["module_relative_offset"])
        raise RuntimeError(f"module '{wanted}' is not currently mapped")
    base = cheat.get("base", 0)
    return int(base, 0) if isinstance(base, str) else int(base)


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
                "base":    hex(c["base"]) if isinstance(c.get("base"), int) else c.get("base"),
                "offsets": [hex(o) for o in c["offsets"]],
                "module":  c.get("module_name"),
                "module_offset": (hex(c["module_relative_offset"])
                                   if c.get("module_relative_offset") is not None else None),
                "terminal_offset": (hex(int(c["terminal_offset"]))
                                     if c.get("terminal_offset") else None),
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
        safe_addstr(stdscr, 10, 3,
                    "Reminder: start the ps5debug payload on your console,",
                    color(C_WARN) | curses.A_BOLD)
        safe_addstr(stdscr, 11, 3,
                    "then verify the PS5 IP address and try again.", color(C_WARN))
        safe_addstr(stdscr, 13, 3, "Press any key to retry.", color(C_NORM))
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
    _invalidate_pointer_index()
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
        ("S", "First Scan", "scan_first", C_NORM),
        ("N", "Next Scan",  "scan_next",  C_NORM),
        ("R", "Results",    "results",    C_ACC),
        ("C", "Cheats",     "cheat_list", C_NORM),
        ("T", "Settings",   "scan_settings", C_ACC),
        ("Q", "Quit",       None,          C_ERR),
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
            ("CHEATS", 3, 1),
            ("SETUP", 4, 2),
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
                        (label == "Next Scan" and len(state["scan_results"]) == 0) or
                        (label == "Results" and len(state["scan_results"]) == 0) or
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
                    (label == "Next Scan" and len(state["scan_results"]) == 0) or
                    (label == "Results" and len(state["scan_results"]) == 0) or
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
            for k, label, action, _ in menu:
                if key in (ord(k.lower()), ord(k.upper())):
                    unavailable = (
                        (label == "Next Scan" and len(state["scan_results"]) == 0) or
                        (label == "Results" and len(state["scan_results"]) == 0)
                    )
                    if unavailable:
                        add_log(f"{label} is unavailable until a scan is complete", "warn")
                        break
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
        "Results      A Apply   C Cheat   P Pointer   R Permanent",
        "Cheats       Enter Inspect   A Apply   D Delete",
        "Advanced     / Command Palette → import/export/logs/pointer tools",
        "Setup        T Settings",
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
        ("Explore Values Near This Item", "nearby_browse"),
        ("Find a Nearby Item by Changing It", "nearby"),
        ("Preview Selected Value", "preview"),
        ("Find Matching Nearby Item (Group Test)", "batch_preview"),
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
            if action == "nearby":
                return "nearby"
            if action == "nearby_browse":
                return "nearby_browse"
            if action == "preview":
                return "preview"
            if action == "batch_preview":
                return "batch_preview"
            result = dispatch(stdscr, action)
            if result == "proc":
                return "proc"
            return


def _snapshot_anchor_window(ip: str, pid: int, anchor: int, width: int,
                            radius: int) -> tuple:
    """Snapshot aligned values near an anchor, clipped to its readable map."""
    maps = _get_maps_cached(ip, pid)
    owner = next((r for r in maps if int(r["start"]) <= int(anchor) < int(r["end"])),
                 None)
    if owner is None or not (int(owner.get("prot", 0)) & 1):
        raise ValueError("anchor is not inside a readable memory map")
    radius = max(width, min(int(radius), 0x10000))
    start = max(int(owner["start"]), int(anchor) - radius)
    end = min(int(owner["end"]), int(anchor) + radius + width)
    start += (-start) % width
    length = end - start
    length -= length % width
    if length < width:
        raise ValueError("anchor window is smaller than one value")
    raw = ps5_read(ip, pid, start, length)
    values = np.frombuffer(raw, dtype=_NP_VALUE_DTYPE[width]).copy()
    addresses = (np.uint64(start) +
                 np.arange(values.size, dtype=np.uint64) * np.uint64(width))
    return addresses, values


def do_browse_nearby(stdscr, anchor: int) -> None:
    """Populate Results with plausible nearby values without requiring a change."""
    width = int(state.get("scan_width", 4))
    try:
        addresses, values = _snapshot_anchor_window(
            state["ip"], int(state["pid"]), int(anchor), width, 0x400)
    except Exception as exc:
        message_box(stdscr, [f"Could not inspect nearby values: {exc}"],
                    "Nearby Candidates", C_ERR)
        return
    # Inventory counts/counters are normally nonzero and modest. Keep the
    # ceiling generous, exclude the known anchor, and cap the visual trial list
    # so ordinary users are not handed hundreds of meaningless fields.
    plausible = (values > 0) & (values <= min(WIDTH_MAX[width], 999999))
    plausible &= addresses != np.uint64(anchor)
    candidate_addr = addresses[plausible]
    candidate_values = values[plausible]
    if not len(candidate_addr):
        message_box(stdscr, ["No plausible nearby item values were found."],
                    "Nearby Candidates", C_WARN)
        return
    order = np.argsort(np.abs(candidate_addr.astype(np.int64) - int(anchor)))
    order = order[:256]
    candidate_addr = candidate_addr[order].copy()
    candidate_values = candidate_values[order].copy()
    if not confirm_box(
            stdscr,
            f"Found {len(candidate_addr):,} nearby candidates. Open them for safe visual preview?",
            "Nearby Candidates"):
        return
    state["scan_results"] = candidate_addr
    state["scan_values"] = candidate_values
    state["scan_pid"] = state["pid"]
    state["scan_unknown"] = False
    state["scan_truncated"] = False
    state["scan_dropped"] = set()
    state["scan_history"].clear()
    add_log(f"Nearby browse @ {hex(int(anchor))}: "
            f"{len(candidate_addr):,} plausible candidates")


def do_discover_nearby(stdscr, anchor: int) -> None:
    """Find neighboring fields that change after an anchored item change."""
    width = int(state.get("scan_width", 4))
    stdscr.clear()
    draw_border(stdscr, "ANCHORED GROUP DISCOVERY")
    radius_text = input_box(stdscr, "Radius (hex): ", 3, 3,
                            width=12, default="0x400")
    try:
        radius = int(radius_text, 0)
    except ValueError:
        message_box(stdscr, ["Invalid radius."], "Error", C_ERR)
        return
    try:
        before_addr, before_val = _snapshot_anchor_window(
            state["ip"], int(state["pid"]), int(anchor), width, radius)
    except Exception as exc:
        message_box(stdscr, [f"Could not snapshot anchor: {exc}"],
                    "Nearby Discovery", C_ERR)
        return

    message_box(stdscr, [
        f"Captured {len(before_addr):,} nearby {WIDTH_LABEL[width]} fields.",
        f"Anchor: {hex(int(anchor))}   Radius: ±{hex(radius)}",
        "Change Red-Item, Green-Item, or another nearby item now.",
        "Do not change Blue-Item during this comparison.",
        "Press any key here only after the in-game change.",
    ], "Nearby Baseline", C_WARN)
    try:
        after_addr, after_val = _snapshot_anchor_window(
            state["ip"], int(state["pid"]), int(anchor), width, radius)
    except Exception as exc:
        message_box(stdscr, [f"Could not read comparison snapshot: {exc}"],
                    "Nearby Discovery", C_ERR)
        return
    if not np.array_equal(before_addr, after_addr):
        message_box(stdscr, ["The containing memory map changed; retry after the game settles."],
                    "Nearby Discovery", C_ERR)
        return
    changed = before_val != after_val
    changed_addr = before_addr[changed].copy()
    old_values = before_val[changed].copy()
    new_values = after_val[changed].copy()
    if not len(changed_addr):
        message_box(stdscr, ["No nearby aligned values changed.",
                             "Try a larger radius or verify the item changed."],
                    "Nearby Discovery", C_WARN)
        return
    order = np.argsort(np.abs(changed_addr.astype(np.int64) - int(anchor)))
    changed_addr, old_values, new_values = (
        changed_addr[order], old_values[order], new_values[order])
    lines = [f"{len(changed_addr):,} nearby fields changed:"]
    for addr, old, new in zip(changed_addr[:18], old_values[:18], new_values[:18]):
        lines.append(f"{hex(int(addr))}  {int(old)} → {int(new)}  "
                     f"({int(addr) - int(anchor):+#x})")
    if len(changed_addr) > 18:
        lines.append(f"…and {len(changed_addr) - 18:,} more")
    message_box(stdscr, lines, "Nearby Candidates", C_OK)
    if confirm_box(stdscr, "Replace Results with these nearby candidates?",
                   "Nearby Discovery"):
        state["scan_results"] = changed_addr
        state["scan_values"] = new_values
        state["scan_pid"] = state["pid"]
        state["scan_unknown"] = False
        state["scan_truncated"] = False
        state["scan_dropped"] = set()
        state["scan_history"].clear()
        add_log(f"Anchored discovery @ {hex(int(anchor))}: "
                f"{len(changed_addr):,} nearby candidates")


def do_preview_candidate(stdscr, address: int) -> None:
    """Temporarily change one candidate, then always restore its original value."""
    width = int(state.get("scan_width", 4))
    validation = _validate_addr_in_maps(
        state["ip"], int(state["pid"]), int(address), width,
        ttl_override=0.0)
    if validation:
        message_box(stdscr, [validation], "Preview Blocked", C_ERR)
        return
    try:
        original_raw = ps5_read(state["ip"], int(state["pid"]),
                                int(address), width)
        original = struct.unpack(WIDTH_FMT[width], original_raw)[0]
    except Exception as exc:
        message_box(stdscr, [f"Could not read the original value: {exc}"],
                    "Preview Blocked", C_ERR)
        return

    stdscr.clear()
    draw_border(stdscr, "SAFE CANDIDATE PREVIEW")
    safe_addstr(stdscr, 2, 3, f"Current value: {original}", color(C_NORM))
    preview_text = input_box(stdscr, "Temporary value: ", 4, 3,
                             width=24, default=str(original + 1))
    try:
        preview = int(preview_text, 0)
        if not 0 <= preview <= WIDTH_MAX[width]:
            raise ValueError
    except ValueError:
        message_box(stdscr, [f"Enter a value from 0 to {WIDTH_MAX[width]}."],
                    "Invalid Preview", C_ERR)
        return
    if preview == original:
        message_box(stdscr, ["The preview value must differ from the original."],
                    "Invalid Preview", C_WARN)
        return

    try:
        acknowledged, verified, actual = _write_value_verified(
            state["ip"], int(state["pid"]), int(address), preview, width)
        if not (acknowledged and verified):
            raise RuntimeError(f"write verification returned {actual}")
    except Exception as exc:
        message_box(stdscr, [f"Preview write failed: {exc}",
                             "The original value was not intentionally changed."],
                    "Preview Failed", C_ERR)
        return

    identified = False
    restore_error = None
    try:
        identified = confirm_box(
            stdscr,
            f"Temporary value {preview} is active. Did the intended in-game item change?",
            "Inspect the Game")
    finally:
        try:
            acknowledged, verified, actual = _write_value_verified(
                state["ip"], int(state["pid"]), int(address), original, width)
            if not (acknowledged and verified):
                restore_error = f"restore verification returned {actual}"
        except Exception as exc:
            restore_error = str(exc)

    if restore_error:
        add_log(f"URGENT: preview restore failed @ {hex(int(address))}: "
                f"{restore_error}", "error")
        message_box(stdscr, [
            "Automatic restore FAILED.",
            f"Address: {hex(int(address))}",
            f"Expected original value: {original}",
            f"Error: {restore_error}",
            "Do not preview another candidate until this is corrected.",
        ], "Restore Failed", C_ERR)
        return

    add_log(f"Preview restored {hex(int(address))} to {original}")
    if identified:
        message_box(stdscr, ["Candidate confirmed and original value restored.",
                             "Name the discovered item on the next screen."],
                    "Item Identified", C_OK)
        _add_cheat_at(stdscr, int(address))
    else:
        message_box(stdscr, ["Original value restored.",
                             "Select another nearby candidate to continue."],
                    "Candidate Rejected", C_NORM)


def do_batch_preview_matching(stdscr) -> None:
    """Temporarily rewrite matching Results as one verified transaction."""
    width = int(state.get("scan_width", 4))
    results = state.get("scan_results")
    if results is None or len(results) == 0:
        message_box(stdscr, ["There are no nearby Results to preview."],
                    "Batch Preview", C_WARN)
        return

    stdscr.clear()
    draw_border(stdscr, "TRANSACTIONAL BATCH PREVIEW")
    match_text = input_box(stdscr, "Current value to match: ", 3, 3,
                           width=20, default="1")
    replacement_text = input_box(stdscr, "Temporary replacement: ", 5, 3,
                                 width=20, default="3")
    try:
        match_value = int(match_text, 0)
        replacement = int(replacement_text, 0)
        if not (0 <= match_value <= WIDTH_MAX[width] and
                0 <= replacement <= WIDTH_MAX[width]):
            raise ValueError
        if match_value == replacement:
            raise ValueError
    except ValueError:
        message_box(stdscr, [f"Enter two different values from 0 to {WIDTH_MAX[width]}."],
                    "Invalid Batch Preview", C_ERR)
        return

    originals = []
    rejected = []
    result_addrs = sorted({int(raw_addr) for raw_addr in results})
    try:
        maps = _get_maps_cached(state["ip"], int(state["pid"]), ttl_override=0.0)
        first, last = result_addrs[0], result_addrs[-1]
        owner = next((m for m in maps
                      if int(m["start"]) <= first and last + width <= int(m["end"])),
                     None)
        if owner is None or not (int(owner.get("prot", 0)) & 2):
            raise RuntimeError("nearby Results are not inside one writable memory map")
        span = last - first + width
        if span > 0x20000:
            raise RuntimeError("nearby Results span is unexpectedly large")
        live_raw = ps5_read(state["ip"], int(state["pid"]), first, span)
        if len(live_raw) != span:
            raise RuntimeError("partial nearby snapshot")
        for addr in result_addrs:
            offset = addr - first
            live = struct.unpack_from(WIDTH_FMT[width], live_raw, offset)[0]
            if live == match_value:
                originals.append((addr, live))
    except Exception as exc:
        message_box(stdscr, [f"Could not capture a safe live snapshot: {exc}"],
                    "Batch Preview Blocked", C_ERR)
        return

    # A broad nearby window can contain flags and object metadata.  Keep the
    # transaction bounded so rollback remains quick even on a weak connection.
    if len(originals) > 64:
        message_box(stdscr, [
            f"{len(originals):,} matching fields exceed the safe batch limit of 64.",
            "Narrow Results or preview smaller groups first.",
        ], "Batch Preview Blocked", C_ERR)
        return
    if not originals:
        message_box(stdscr, [f"No live Results currently equal {match_value}."],
                    "Batch Preview", C_WARN)
        return
    if not confirm_box(stdscr,
            f"Temporarily change {len(originals)} nearby fields from "
            f"{match_value} to {replacement}, then restore all of them?",
            "Confirm Batch Preview"):
        return

    written = []
    write_error = None
    identified = False
    try:
        for addr, original in originals:
            ack, verified, actual = _write_value_verified(
                state["ip"], int(state["pid"]), addr, replacement, width)
            if not (ack and verified):
                raise RuntimeError(
                    f"write verification failed at {hex(addr)}: {actual}")
            written.append((addr, original))
        identified = confirm_box(stdscr,
            f"{len(written)} matching fields are temporarily {replacement}. "
            "Did any intended in-game item change?",
            "Inspect the Game")
    except Exception as exc:
        write_error = str(exc)
    finally:
        unresolved = []
        for addr, original in reversed(written):
            restored = False
            last_error = "unknown restore error"
            for _attempt in range(3):
                try:
                    ack, verified, actual = _write_value_verified(
                        state["ip"], int(state["pid"]), addr, original, width)
                    if ack and verified:
                        restored = True
                        break
                    last_error = f"verification returned {actual}"
                except Exception as exc:
                    last_error = str(exc)
            if not restored:
                unresolved.append((addr, original, last_error))

    if unresolved:
        for addr, original, error in unresolved:
            add_log(f"URGENT: batch restore failed @ {hex(addr)} to "
                    f"{original}: {error}", "error")
        lines = ["Automatic batch restore FAILED for:"]
        lines.extend(f"{hex(addr)} → {original}: {error}"
                     for addr, original, error in unresolved[:8])
        lines.append("Do not run another preview until these are corrected.")
        message_box(stdscr, lines, "Batch Restore Failed", C_ERR)
        return
    if write_error:
        message_box(stdscr, [f"Batch stopped: {write_error}",
                             f"Restored all {len(written)} changed fields."],
                    "Batch Preview Rolled Back", C_ERR)
        return
    add_log(f"Batch preview restored {len(written)} fields "
            f"({match_value} → {replacement} → {match_value})")
    note = (f"Skipped {len(rejected)} unreadable/non-writable Results."
            if rejected else "All selected fields were writable and verified.")
    message_box(stdscr, [f"Restored all {len(written)} fields to {match_value}.", note],
                "Batch Preview Complete", C_OK)
    if identified and len(originals) > 1 and confirm_box(
            stdscr,
            "The item is in this group. Isolate its exact address now?",
            "Guided Isolation"):
        _isolate_matching_group(stdscr, originals, replacement, width)


def _preview_group_once(stdscr, candidates: list, replacement: int,
                        width: int) -> tuple:
    """Preview one candidate group and restore it; return (changed, error)."""
    written = []
    changed = False
    failure = None
    try:
        for addr, original in candidates:
            ack, verified, actual = _write_value_verified(
                state["ip"], int(state["pid"]), int(addr), replacement, width)
            if not (ack and verified):
                raise RuntimeError(f"write verification failed at {hex(int(addr))}: {actual}")
            written.append((int(addr), original))
        changed = confirm_box(
            stdscr,
            f"Testing {len(candidates)} possible fields at value {replacement}. "
            "Did the same item change?",
            "Guided Isolation")
    except Exception as exc:
        failure = str(exc)
    finally:
        unresolved = []
        for addr, original in reversed(written):
            restored = False
            last_error = "unknown restore error"
            for _attempt in range(3):
                try:
                    ack, verified, actual = _write_value_verified(
                        state["ip"], int(state["pid"]), addr, original, width)
                    if ack and verified:
                        restored = True
                        break
                    last_error = f"verification returned {actual}"
                except Exception as exc:
                    last_error = str(exc)
            if not restored:
                unresolved.append(f"{hex(addr)}: {last_error}")
        if unresolved:
            failure = "restore failed — " + "; ".join(unresolved[:4])
    return changed, failure


def _isolate_matching_group(stdscr, candidates: list, replacement: int,
                            width: int) -> None:
    """Binary-search a confirmed group with reversible visual previews."""
    remaining = list(candidates)
    while len(remaining) > 1:
        test_group = remaining[:(len(remaining) + 1) // 2]
        changed, failure = _preview_group_once(
            stdscr, test_group, replacement, width)
        if failure:
            add_log(f"Guided isolation stopped: {failure}", "error")
            message_box(stdscr, [failure,
                                 "Isolation stopped; no further group was tested."],
                        "Isolation Stopped", C_ERR)
            return
        tested = {int(addr) for addr, _ in test_group}
        remaining = (test_group if changed else
                     [item for item in remaining if int(item[0]) not in tested])
        if not remaining:
            message_box(stdscr, [
                "The visual answers were inconsistent; no candidate remains.",
                "Retry when the item count is visible and the game is settled.",
            ], "Isolation Inconclusive", C_WARN)
            return
    address, original = remaining[0]
    message_box(stdscr, [
        "Exact nearby item field identified.",
        f"Address: {hex(int(address))}",
        f"Current value: {original}",
        "The original value is restored.",
    ], "Item Identified", C_OK)
    if confirm_box(stdscr, "Save this address as a cheat now?", "Save Item"):
        _add_cheat_at(stdscr, int(address))


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
                safe_addstr(stdscr, 13, pane_x, "R  Resolve permanent", color(C_ACC))
                safe_addstr(stdscr, 14, pane_x, "D  Drop result", color(C_ERR))
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
                ("P pointer", C_ACC), ("R permanent", C_ACC), ("N next", C_ACC),
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
            elif key in (ord('r'), ord('R')) and len(results) > 0:
                stdscr.nodelay(False)
                do_resolve_permanent(stdscr, int(results[sel]))
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
                if more_result == "nearby":
                    stdscr.nodelay(False)
                    do_discover_nearby(stdscr, int(results[sel]))
                    stdscr.nodelay(True)
                    results = state["scan_results"]
                if more_result == "nearby_browse":
                    stdscr.nodelay(False)
                    do_browse_nearby(stdscr, int(results[sel]))
                    stdscr.nodelay(True)
                    results = state["scan_results"]
                if more_result == "preview":
                    stdscr.nodelay(False)
                    do_preview_candidate(stdscr, int(results[sel]))
                    stdscr.nodelay(True)
                    results = state["scan_results"]
                if more_result == "batch_preview":
                    stdscr.nodelay(False)
                    do_batch_preview_matching(stdscr)
                    stdscr.nodelay(True)
                    results = state["scan_results"]
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
            base    = _runtime_pointer_base(cheat)
            offsets = [int(o, 0) if isinstance(o, str) else int(o) for o in cheat["offsets"]]
            ok, addr, _ = _resolve_pointer_chain(
                state["ip"], state["pid"], base, offsets, int(cheat.get("terminal_offset", 0)))
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
        base    = _runtime_pointer_base(cheat)
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
                base = _runtime_pointer_base(c)
                offsets = [int(o, 0) if isinstance(o, str) else int(o) for o in c["offsets"]]
                ok_chain, target_addr, _ = _resolve_pointer_chain(
                    state["ip"], state["pid"], base, offsets, int(c.get("terminal_offset", 0)))
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
                base = _parse_int_field(c["base"], "base")
                if not (_ADDR_MIN <= base <= _ADDR_MAX):
                    raise ValueError(f"'{e['name']}' has an invalid base address")
                raw_offsets = c.get("offsets", [])
                if not isinstance(raw_offsets, list) or not (1 <= len(raw_offsets) <= MAX_CHAIN_DEPTH):
                    raise ValueError(f"'{e['name']}' has an invalid pointer depth")
                offsets = [_parse_int_field(x, "offset") for x in raw_offsets]
                if any(abs(x) > _PTR_RESOLVE_OFFSET_MAX for x in offsets):
                    raise ValueError(f"'{e['name']}' has an unreasonable pointer offset")
                e["base"] = base
                e["offsets"] = offsets
                e["address"] = 0
                if c.get("module_name") is not None:
                    module_name = str(c.get("module_name")).strip()
                    if not module_name or len(module_name) > 512:
                        raise ValueError(f"'{e['name']}' has an invalid module name")
                    e["module_name"] = module_name
                if c.get("module_relative_offset") is not None:
                    mrel = _parse_int_field(c["module_relative_offset"], "module relative offset")
                    if mrel < 0 or mrel > _ADDR_MAX:
                        raise ValueError(f"'{e['name']}' has an invalid module-relative offset")
                    e["module_relative_offset"] = mrel
            elif "address" in c:
                address = _parse_int_field(c["address"], "address")
                if not (_ADDR_MIN <= address <= _ADDR_MAX):
                    raise ValueError(f"'{e['name']}' has an invalid address")
                e["address"] = address
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
            base = _runtime_pointer_base(cheat)
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
    frozen_session = state["session"]
    stop_event = threading.Event(); errors=[0]; deadline=time.time()+sec
    def _freeze_worker():
        while time.time() < deadline:
            if stop_event.is_set() or _freeze_stop.is_set(): break
            if state["ip"] != frozen_ip or state["pid"] != frozen_pid:
                add_log("Freeze aborted — process or connection changed","warn"); break
            target = addr
            if state.get("session") != frozen_session:
                add_log("Freeze aborted — session changed", "warn"); break
            if is_pointer:
                try:
                    base_now = _runtime_pointer_base(cheat)
                except Exception as exc:
                    add_log(f"Pointer freeze: module unavailable — {exc}", "warn")
                    break
                ok,target,_ = _resolve_pointer_chain(
                    frozen_ip, frozen_pid, base_now, offsets, int(cheat.get("terminal_offset", 0)))
                if not ok:
                    add_log("Pointer freeze: chain broken — retrying next tick","warn")
                    if stop_event.wait(interval) or _freeze_stop.is_set(): break
                    continue
            validation_error = _validate_addr_in_maps(frozen_ip,frozen_pid,target,width,10.0)
            if validation_error is not None:
                errors[0] += 1
                add_log(f"Freeze write blocked: {validation_error}", "warn")
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


def do_resolve_permanent(stdscr, target_addr: int) -> None:
    """Resolve a known temporary address to the best verified permanent chain."""
    target_addr = int(target_addr)
    if not (_ADDR_MIN <= target_addr <= _ADDR_MAX):
        message_box(stdscr, [f"Invalid target address: {hex(target_addr)}"],
                    "Resolve Permanent Address", C_ERR)
        return

    cancel_event = threading.Event()
    progress = {"done": 0, "total": _PTR_RESOLVE_MAX_NODES,
                "results": None, "error": None, "built": False}

    def run():
        try:
            progress["results"] = _resolve_permanent_candidates(
                state["ip"], state["pid"], target_addr,
                max_depth=min(6, MAX_CHAIN_DEPTH),
                cancel_event=cancel_event,
                progress_cb=lambda d, t: progress.update(
                    done=d, total=max(int(t), 1)))
            progress["built"] = bool(progress["results"].get("index_built"))
        except Exception as exc:
            progress["error"] = str(exc)

    ok = _run_scan_with_progress(
        stdscr, run,
        "Finding a stable location…",
        cancel_event, progress)
    if not ok:
        add_log("Permanent resolver cancelled", "warn")
        return
    if progress["error"]:
        message_box(stdscr, ["Could not find a permanent location.",
                             str(progress["error"])],
                    "Resolve Failed", C_ERR)
        return

    data = progress["results"] or {}
    candidates = [c for c in data.get("candidates", []) if c.get("verified")]
    if not candidates:
        message_box(
            stdscr,
            ["No stable location was found.",
             "Try again after the value has been used in-game."],
            "Permanent Location Not Found", C_WARN)
        return

    # Keep only the strongest verified choices; the user should choose between
    # a few meaningful alternatives, not inspect raw pointer-search output.
    candidates = candidates[:8]
    sel = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "PERMANENT LOCATION")
        safe_addstr(stdscr, 2, 3, "Your temporary address was found.",
                    color(C_NORM))
        safe_addstr(stdscr, 3, 3, f"Temporary: {hex(target_addr)}",
                    color(C_WARN))
        safe_addstr(stdscr, 5, 3, "BEST MATCH", color(C_TITLE) | curses.A_BOLD)

        c = candidates[sel]
        module = c.get("module_name") or "Game module"
        rel = hex(int(c.get("module_relative_offset", 0)))
        path = " → ".join(hex(int(x)) for x in c.get("offsets", []))
        safe_addstr(stdscr, 6, 3, f"Permanent: {module} + {rel}",
                    color(C_OK) | curses.A_BOLD)
        safe_addstr(stdscr, 7, 3, f"Pointer path: {path or 'direct'}", color(C_WARN))
        safe_addstr(stdscr, 8, 3, "Verification: ✓ Confirmed", color(C_OK))
        safe_addstr(stdscr, 9, 3, f"Confidence: {int(c.get('confidence', 0))}%",
                    color(C_OK))

        if len(candidates) > 1:
            safe_addstr(stdscr, 11, 3, "Other verified matches",
                        color(C_TITLE) | curses.A_BOLD)
            for i, alt in enumerate(candidates[1:5], 1):
                a_mod = alt.get("module_name") or "Game module"
                a_rel = hex(int(alt.get("module_relative_offset", 0)))
                marker = ">" if i == sel else " "
                line = f"{marker} {i+1}. {a_mod} + {a_rel}  ({int(alt.get('confidence', 0))}%)"
                safe_addstr(stdscr, 12 + i - 1, 3, line[:max(w - 6, 1)],
                            color(C_NORM))

        draw_statusbar(stdscr, [
            ("↑↓ choose", C_NORM), ("Enter save", C_OK),
            ("R search again", C_WARN), ("Q back", C_NORM)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols()
            continue
        if key == curses.KEY_UP:
            sel = max(0, sel - 1)
        elif key == curses.KEY_DOWN:
            sel = min(len(candidates) - 1, sel + 1)
        elif key in (10, 13, curses.KEY_ENTER):
            c2 = dict(c)
            c2["module_name"] = module
            c2["module_relative_offset"] = int(c.get("module_relative_offset", 0))
            do_pointer_chain_verify(stdscr, c2, target_addr)
        elif key in (ord('r'), ord('R')):
            _invalidate_pointer_index(state["ip"], state["pid"])
            return do_resolve_permanent(stdscr, target_addr)
        elif key in (ord('q'), ord('Q'), 27):
            return


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

    # ── STEP 2: scan ─────────────────────────────────────────────────────────
    # Four levels cover the common static -> heap -> object -> value layout
    # without forcing every user through six full memory passes.  The pointer
    # scanner itself still accepts deeper values for advanced callers.
    max_depth    = _PTR_DEPTH_DEFAULT
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
    terminal_offset = int(candidate.get("terminal_offset", 0))
    # Use a one-element list so closures always see the current offsets list
    # even after reassignment in the E branch (closures capture the cell, not
    # the value; a mutable container avoids the stale-reference problem).
    _offsets = [list(candidate["offsets"])]
    region   = candidate["region"]

    safe_addstr(stdscr, 2, 3,
        f"Base address : {hex(base)}  [{region}]", color(C_OK) | curses.A_BOLD)
    if terminal_offset:
        safe_addstr(stdscr, 3, 3,
                    f"Terminal field offset: +{hex(terminal_offset)}", color(C_ACC))

    def _test_chain():
        """Resolve chain and return (ok, final_addr, steps)."""
        return _resolve_pointer_chain(
            state["ip"], state["pid"], base, _offsets[0], terminal_offset)

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
                **({"terminal_offset": terminal_offset} if terminal_offset else {}),
                "value":   val,
                "width":   width,
                "pid":     state["pid"],
                "process": state["proc_name"],
                "session": state["session"],
                # Smart resolver metadata: survives ASLR/module relocation.
                **({"module_name": candidate["module_name"],
                    "module_relative_offset": int(candidate["module_relative_offset"])}
                   if candidate.get("module_name") is not None else {}),
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
                message_box(stdscr, [
                    f"Error: {exc}",
                    "",
                    "Reminder: make sure the ps5debug payload is running",
                    "on your console, then verify its IP address.",
                ], "Connection Error", C_ERR)
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
