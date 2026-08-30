#!/usr/bin/env python3

"""
RDX CheatMaker Final Release with Terminal UI

Usage:
    python3 RDX-CHEATMAKER-UI.py
"""

RDX_VERSION = "1.0.0"

import array as _array
import base64
import bisect
import contextlib
import curses
import gc
import hashlib
import heapq
import math
import os
import queue as _queue
import re
import socket
import struct
import ipaddress
import json
import threading
import tempfile
import time
import unicodedata
import warnings
import uuid
import xml.etree.ElementTree as ET
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
    def _nb_relational_mask(cur_vals, prv_vals, mode_id: int, delta: int,
                            width_mask: int):
        """
        Compute a boolean mask for the relational filter in parallel.

        mode_id values (must match RELATIONAL_MODE_IDS below):
            0 = decreased       cur < prv
            1 = increased       cur > prv
            2 = changed         cur != prv
            3 = unchanged       cur == prv
            4 = decreased by    cur == (prv - delta) & width_mask
            5 = increased by    cur == (prv + delta) & width_mask

        cur_vals/prv_vals arrive zero-extended to uint64 from their native
        scan width, so modes 0-3 compare correctly as-is. Modes 4-5 perform
        real subtraction/addition, which must wrap at the value's own width
        (e.g. a u8 counter) rather than at 64 bits, or a legitimate wrapped
        match (say u8 3 "decreased by" 10 == 249) is silently missed —
        width_mask (WIDTH_MAX[width]) reproduces the same wraparound the
        pure-NumPy fallback applies.

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
            elif mode_id == 4: mask[i] = c == ((p - delta) & width_mask)
            else:              mask[i] = c == ((p + delta) & width_mask)
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
CMD_PROC_WRITE_MULTI = 0xBDAACC04
PROC_WRITE_MULTI_F_STATUS = 0x1
PROC_WRITE_MULTI_MAX_COUNT = 0xFFFF
PROC_WRITE_MULTI_MAX_LEN   = 0x100000
PROC_AUTH_MAGIC = 0xBB40E64D
# STATUS_SUCCESS / STATUS_ERROR: bit-swapped wire values produced by the server's
# net_send_int32() helper.  Clients compare raw wire bytes directly.
STATUS_SUCCESS = 0x80000000
STATUS_ERROR   = 0xF0000001
PS5_PORT       = 744
MEMDBG_PORT    = 9020
MEMDBG_MAGIC   = 0x4742444D             # "MDBG", little-endian
MEMDBG_VERSION = 1
MEMDBG_CMD_HELLO = 0x0001
MEMDBG_CMD_PROCESS_LIST = 0x0100
MEMDBG_CMD_PROCESS_MAPS = 0x0101
MEMDBG_CMD_PROCESS_MAPS_V2 = 0x0110
MEMDBG_CMD_MEMORY_READ = 0x0200
MEMDBG_CMD_MEMORY_WRITE = 0x0201
MEMDBG_CMD_BATCH_WRITE = 0x0203
MEMDBG_CMD_SCAN_POINTER = 0x0303
MEMDBG_CAP_PROCESS_LIST = 1 << 0
MEMDBG_CAP_PROCESS_MAPS = 1 << 1
MEMDBG_CAP_MEMORY_READ = 1 << 2
MEMDBG_CAP_MEMORY_WRITE = 1 << 3
MEMDBG_CAP_BATCH_WRITE = 1 << 16
MEMDBG_CAP_SCAN_POINTER = 1 << 10
# Bits RDX had no name for. The console reported capabilities = 0xFFFFFFFF on
# 2026-08-30 and that was recorded as uninformative -- correctly, since every
# bit including undefined ones was set. What went unremarked is that bit 20 was
# among them: the payload was announcing a debugger in the same handshake, and
# RDX could not read most of what the bitmap said.
MEMDBG_CAP_DEBUGGER = 1 << 20
MEMDBG_CAP_TRACER = 1 << 21

# ── MemDBG debugger: the write-watchpoint path ────────────────────────────────
# Wire format taken from MemDBG's published protocol header (GPL-3.0-or-later,
# seregonwar/MemDBG). Implementing a documented protocol for interoperability
# is not a derived work; no source is copied here.
#
# Only the commands the AOB anchor method needs are implemented. RDX is not
# growing a general debugger: the goal is "which instruction writes this
# address", and nothing more.
MEMDBG_CMD_DEBUG_GET_THREADS = 0x0605
MEMDBG_CMD_DEBUG_GET_REGS = 0x0606
MEMDBG_CMD_DEBUG_SET_WATCHPOINT = 0x060C
MEMDBG_CMD_DEBUG_CLEAR_WATCHPOINT = 0x060D
MEMDBG_CMD_DEBUG_POLL_EVENTS = 0x0610

_MEMDBG_WP_WRITE = 1                 # 0=exec 1=write 2=read 3=read-write
_MEMDBG_WP_LENGTHS = (1, 2, 4, 8)
_MEMDBG_REGS_SIZE = 176
# Offset computed from the header's field order, not assumed from the textbook
# FreeBSD layout -- the two disagree. r_rip follows r_ds at 134 + 2.
_MEMDBG_REGS_RIP_OFFSET = 136
_MEMDBG_POLL_SIZE = 8
MEMDBG_MAX_MEMORY_READ = 1024 * 1024
MEMDBG_MAX_WRITE_DATA = 1024 * 1024 - 16  # request body includes 16-byte header
MEMDBG_BATCH_WRITE_MAX_ITEMS = 64

# A TurboScan list session lives on its TCP connection.  Retaining that
# connection lets subsequent exact scans use the payload's resident COUNT
# command instead of reading millions of candidate addresses back over LAN.
_turbo_session_lock = threading.RLock()
_turbo_session = None

# ``WIDTH_*`` remain as the backwards-compatible unsigned view used by the
# pointer subsystem and older external callers.  User-facing scans and cheats
# use VALUE_TYPES below, which carries the signed/float/raw-byte semantics that
# a width by itself cannot express.
WIDTH_FMT   = {1: 'B', 2: '<H', 4: '<I', 8: '<Q'}
VALID_WIDTHS = [1, 2, 4, 8]
WIDTH_LABEL  = {1: "byte (u8)", 2: "uint16", 4: "uint32", 8: "uint64"}
WIDTH_MAX    = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF, 8: 0xFFFFFFFFFFFFFFFF}

VALUE_TYPES = {
    "u8":  {"label": "Unsigned 8-bit (u8)",  "width": 1, "fmt": "<B", "dtype": "<u1", "kind": "uint", "min": 0, "max": 0xFF},
    "i8":  {"label": "Signed 8-bit (i8)",    "width": 1, "fmt": "<b", "dtype": "<i1", "kind": "sint", "min": -0x80, "max": 0x7F},
    "u16": {"label": "Unsigned 16-bit (u16)", "width": 2, "fmt": "<H", "dtype": "<u2", "kind": "uint", "min": 0, "max": 0xFFFF},
    "i16": {"label": "Signed 16-bit (i16)",   "width": 2, "fmt": "<h", "dtype": "<i2", "kind": "sint", "min": -0x8000, "max": 0x7FFF},
    "u32": {"label": "Unsigned 32-bit (u32)", "width": 4, "fmt": "<I", "dtype": "<u4", "kind": "uint", "min": 0, "max": 0xFFFFFFFF},
    "i32": {"label": "Signed 32-bit (i32)",   "width": 4, "fmt": "<i", "dtype": "<i4", "kind": "sint", "min": -0x80000000, "max": 0x7FFFFFFF},
    "f32": {"label": "Float 32-bit (f32)",    "width": 4, "fmt": "<f", "dtype": "<f4", "kind": "float"},
    "u64": {"label": "Unsigned 64-bit (u64)", "width": 8, "fmt": "<Q", "dtype": "<u8", "kind": "uint", "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
    "i64": {"label": "Signed 64-bit (i64)",   "width": 8, "fmt": "<q", "dtype": "<i8", "kind": "sint", "min": -0x8000000000000000, "max": 0x7FFFFFFFFFFFFFFF},
    "f64": {"label": "Float 64-bit (f64)",    "width": 8, "fmt": "<d", "dtype": "<f8", "kind": "float"},
    # Raw byte cheats use a per-entry width and canonical uppercase hex value.
    "bytes": {"label": "Byte pattern / raw bytes", "width": None,
              "fmt": None, "dtype": None, "kind": "bytes"},
}
VALUE_TYPE_ORDER = ["u8", "i8", "u16", "i16", "u32", "i32", "f32",
                    "u64", "i64", "f64", "bytes"]
LEGACY_VALUE_TYPE = {1: "u8", 2: "u16", 4: "u32", 8: "u64"}
SCAN_VALUE_TYPE_ID = {
    "u8": 0, "i8": 1, "u16": 2, "i16": 3,
    "u32": 4, "i32": 5, "u64": 6, "i64": 7,
    "f32": 8, "f64": 9, "bytes": 10,
}


def _normalise_value_type(value_type=None, width: Optional[int] = None) -> str:
    """Return a supported type key, preserving unsigned legacy behaviour."""
    key = str(value_type or "").strip().lower()
    aliases = {
        "byte": "u8", "uint8": "u8", "int8": "i8",
        "uint16": "u16", "int16": "i16", "short": "i16",
        "uint32": "u32", "int32": "i32", "int": "i32",
        "float": "f32", "single": "f32",
        "uint64": "u64", "int64": "i64", "double": "f64",
        "aob": "bytes", "hex": "bytes", "raw": "bytes",
    }
    key = aliases.get(key, key)
    if key in VALUE_TYPES:
        return key
    if width is not None and int(width) in LEGACY_VALUE_TYPE:
        return LEGACY_VALUE_TYPE[int(width)]
    return "u32"


def _value_spec(value_type=None, width: Optional[int] = None) -> dict:
    return VALUE_TYPES[_normalise_value_type(value_type, width)]


def _value_width(value_type=None, width: Optional[int] = None) -> int:
    spec = _value_spec(value_type, width)
    resolved = spec.get("width") if spec.get("width") is not None else width
    if resolved is None or int(resolved) <= 0 or int(resolved) > 256:
        raise ValueError("raw-byte values require a width from 1 to 256")
    return int(resolved)


def _parse_byte_pattern(text: str, allow_wildcards: bool = True) -> tuple:
    """Parse ``AA BB ?? CC`` or compact hex into (bytes, mask, canonical)."""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("byte pattern is empty")
    if re.fullmatch(r"[0-9a-fA-F]+", raw) and len(raw) % 2 == 0:
        tokens = [raw[i:i + 2] for i in range(0, len(raw), 2)]
    else:
        tokens = [t for t in re.split(r"[\s,;:-]+", raw) if t]
    if not 1 <= len(tokens) <= 256:
        raise ValueError("byte pattern must contain 1 to 256 bytes")
    values = bytearray()
    mask = bytearray()
    canonical = []
    for token in tokens:
        if token in {"?", "??", "**"}:
            if not allow_wildcards:
                raise ValueError("wildcards are not allowed for a write value")
            values.append(0)
            mask.append(0)
            canonical.append("??")
            continue
        if not re.fullmatch(r"[0-9a-fA-F]{2}", token):
            raise ValueError(f"invalid byte token: {token!r}")
        values.append(int(token, 16))
        mask.append(0xFF)
        canonical.append(token.upper())
    if not any(mask):
        raise ValueError("a pattern cannot consist entirely of wildcards")
    return bytes(values), bytes(mask), " ".join(canonical)


def _parse_value_text(text: str, value_type=None,
                      width: Optional[int] = None):
    key = _normalise_value_type(value_type, width)
    spec = VALUE_TYPES[key]
    if spec["kind"] == "bytes":
        raw, _mask, _canonical = _parse_byte_pattern(text, False)
        if width is not None and len(raw) != int(width):
            raise ValueError(f"expected {int(width)} bytes, got {len(raw)}")
        return raw.hex().upper()
    if spec["kind"] == "float":
        try:
            value = float(str(text).strip())
        except ValueError:
            raise ValueError(f"invalid {key} value: {text!r}")
        if not math.isfinite(value):
            raise ValueError("NaN and infinity are not supported")
        # Packing catches f32 overflow and also rounds to the value that is
        # actually representable in game memory.
        try:
            return struct.unpack(spec["fmt"], struct.pack(spec["fmt"], value))[0]
        except (OverflowError, struct.error):
            raise ValueError(f"value is out of range for {key}")
    try:
        value = int(str(text).strip(), 0)
    except ValueError:
        raise ValueError(f"invalid {key} value: {text!r}")
    if not int(spec["min"]) <= value <= int(spec["max"]):
        raise ValueError(
            f"value must be between {spec['min']} and {spec['max']} for {key}")
    return value


def _pack_typed_value(value, value_type=None,
                      width: Optional[int] = None) -> bytes:
    key = _normalise_value_type(value_type, width)
    spec = VALUE_TYPES[key]
    if spec["kind"] == "bytes":
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
        else:
            raw, _mask, _canonical = _parse_byte_pattern(str(value), False)
        if width is not None and len(raw) != int(width):
            raise ValueError(f"expected {int(width)} bytes, got {len(raw)}")
        return raw
    parsed = value
    if isinstance(value, str):
        parsed = _parse_value_text(value, key, width)
    try:
        return struct.pack(spec["fmt"], parsed)
    except (OverflowError, TypeError, ValueError, struct.error):
        raise ValueError(f"value {value!r} is not valid for {key}")


def _unpack_typed_value(raw: bytes, value_type=None,
                        width: Optional[int] = None):
    key = _normalise_value_type(value_type, width)
    if key == "bytes":
        expected = _value_width(key, width)
        if len(raw) != expected:
            raise ValueError(f"expected {expected} bytes, got {len(raw)}")
        return bytes(raw).hex().upper()
    expected = _value_width(key, width)
    if len(raw) != expected:
        raise ValueError(f"expected {expected} bytes, got {len(raw)}")
    return struct.unpack(VALUE_TYPES[key]["fmt"], raw)[0]


def _format_typed_value(value, value_type=None,
                        width: Optional[int] = None) -> str:
    key = _normalise_value_type(value_type, width)
    if key == "bytes":
        try:
            raw = _pack_typed_value(value, key, width)
            return " ".join(f"{b:02X}" for b in raw)
        except ValueError:
            return str(value)
    if VALUE_TYPES[key]["kind"] == "float":
        return format(float(value), ".9g" if key == "f32" else ".17g")
    return str(int(value))


def _cheat_value_type(cheat: dict) -> str:
    return _normalise_value_type(cheat.get("value_type"), cheat.get("width", 4))


def _cheat_value_bytes(cheat: dict, field: str = "value") -> bytes:
    return _pack_typed_value(
        cheat[field], _cheat_value_type(cheat), int(cheat.get("width", 4)))

# PS5 user-space is 0x0001 – 0x00007FFF_FFFF_FFFF.
# Static/module segments on PS5 (orbis-ld output) are loaded in the low
# portion of that range.  Heap and mmap() regions occupy the upper portion.
# These thresholds are heuristics — confirmed against multiple retail titles.
_STATIC_ADDR_MAX  = 0x0000_0100_0000_0000   # below ≈ 1 TB → likely module/static
_HEAP_NAME_HINTS  = frozenset({"", "anon", "heap", "stack", "scePthread",
                                "SceKernelPrimary", "SceLibcInternal"})

# Maximum pointer-chain depth. The normal guided scan uses five levels and the
# exhaustive resolver uses six; manual/advanced callers can opt into eight.
MAX_CHAIN_DEPTH   = 8

# How many pointer-scan results to keep per depth level before filtering.
# Raising this finds more candidates but uses more RAM and takes longer.
MAX_PTR_RESULTS   = 500_000


# proc_list_entry layout: char name[32]; int32_t pid;  → 36 bytes
PROC_ENTRY_SIZE = 36
# proc_vm_map_entry layout: char name[32]; uint64 start; uint64 end;
#   uint64 offset; uint16 prot;  → 58 bytes (no padding between fields)
MAP_ENTRY_SIZE = 58

# Sanity bounds on the uint32 entry counts that CMD_PROC_LIST and CMD_PROC_MAPS
# put on the wire.  The payload declares no server-side cap for either (see
# protocol reference 254-299), and the protocol itself notes that a stream can
# become "untrustworthy" after a cap violation elsewhere.  An unbounded count
# is therefore attacker-free but not garbage-free: on a desynced stream those
# four bytes are whatever happened to be in the buffer, and feeding
# range(0xFFFFFFFF) a 58-byte-per-entry loop turns ~400 MB of stream into
# >3 GB of Python dicts in about seven seconds.  That matters beyond a wasted
# allocation -- an OOM kill is SIGKILL, which _install_signal_teardown()
# explicitly cannot catch, so it would leave a SIGSTOPped game on the console.
# Real hardware measured 307 map rows and 87 processes; these caps sit far
# above anything real while still bounding the failure.
MAX_PROC_ENTRIES = 4096
MAX_MAP_ENTRIES = 65536


def _checked_entry_count(raw: bytes, limit: int, what: str) -> int:
    """Decode a uint32 wire count, refusing implausible values."""
    count = struct.unpack("<I", raw)[0]
    if count > limit:
        raise RuntimeError(
            f"{what} count of {count:,} exceeds the sane maximum of {limit:,}; "
            f"the connection stream is out of sync — reconnect to the console")
    return count

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

# TURBO_MIN_SURVIVORS: below this many candidates, "auto" filters on the host
# rather than negotiating a resident TurboScan session.
#
# Auto previously chose purely by availability -- turbo, then console, then
# host -- which is the right first filter but not the only one. Squalr's rules
# engine additionally weighs "region size, SIMD compatibility, and data type
# properties" when picking a strategy. The console round trips that set up and
# tear down a resident session cost more than reading a few hundred addresses
# outright, so on a nearly-converged candidate list turbo is the slower path.
# Only "auto" consults this; an explicit "turbo" is still honoured exactly.
TURBO_MIN_SURVIVORS: int = 512

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

# ── undo delta compression ────────────────────────────────────────────────────
# Undo levels were the project's real RAM problem: HISTORY_RAM_CAP_MB exists to
# bound them, and "Clear Scan History" is a whole user-facing screen whose only
# job is freeing them. A delta is the set of addresses a Next Scan removed, and
# on an aligned scan those are overwhelmingly long constant-stride runs -- the
# scan walks memory at a fixed step, so the addresses it discards are mostly
# consecutive.
#
# Squalr makes the same observation about scan results and answers it with
# run-length encoding: "if scanning for value 0x00 and the region was entirely
# zeros at size 0x2000 at address 0x10000, it would yield a single result."
# Applied to a delta, one 4-byte-strided run of a million addresses collapses
# from 8 MB to three numbers.
#
# RLE is not unconditionally better -- a shattered delta with no runs costs 3x
# raw -- so both encodings are produced and the smaller one is kept. Decoding
# is exact either way; this is a storage format, not an approximation.
_RLE_MIN_RUN = 4          # shorter runs cost more to describe than to store


def _rle_encode_addrs(addrs: np.ndarray) -> Optional[tuple]:
    """Encode a sorted address array as (starts, steps, counts).

    Returns None when the encoding would not be smaller than the raw array.
    """
    arr = np.asarray(addrs, dtype=np.uint64)
    if len(arr) < _RLE_MIN_RUN:
        return None
    # Signed diffs: addresses are sorted so steps are positive, but uint64
    # subtraction would wrap on any out-of-order input rather than showing it.
    diffs = np.diff(arr.astype(np.int64))
    if len(diffs) == 0:
        return None
    # A run boundary is where the stride changes.
    boundaries = np.flatnonzero(diffs[1:] != diffs[:-1]) + 1
    starts_idx = np.concatenate(([0], boundaries + 1))
    run_lengths = np.diff(np.concatenate((starts_idx, [len(arr)])))

    starts, steps, counts = [], [], []
    cursor = 0
    while cursor < len(arr):
        if cursor == len(arr) - 1:
            starts.append(arr[cursor]); steps.append(0); counts.append(1)
            cursor += 1
            continue
        step = int(arr[cursor + 1]) - int(arr[cursor])
        length = 2
        while (cursor + length < len(arr)
               and int(arr[cursor + length]) - int(arr[cursor + length - 1]) == step):
            length += 1
        starts.append(arr[cursor]); steps.append(step); counts.append(length)
        cursor += length

    encoded = (np.asarray(starts, dtype=np.uint64),
               np.asarray(steps, dtype=np.int64),
               np.asarray(counts, dtype=np.int64))
    if sum(part.nbytes for part in encoded) >= arr.nbytes:
        return None
    return encoded


def _rle_decode_addrs(encoded: tuple) -> np.ndarray:
    """Rebuild the exact array a _rle_encode_addrs() result describes."""
    starts, steps, counts = encoded
    total = int(counts.sum())
    out = np.empty(total, dtype=np.uint64)
    cursor = 0
    for start, step, count in zip(starts.tolist(), steps.tolist(),
                                  counts.tolist()):
        out[cursor:cursor + count] = (
            np.uint64(start)
            + (np.arange(count, dtype=np.int64) * step).astype(np.uint64))
        cursor += count
    return out


class _UndoAddrs:
    """One undo level's removed-address set, stored in whichever form is
    smaller. Callers only ever see the decoded array."""

    __slots__ = ("_raw", "_encoded", "_nbytes", "_length")

    def __init__(self, addrs: np.ndarray):
        arr = np.asarray(addrs, dtype=np.uint64)
        self._length = len(arr)
        encoded = _rle_encode_addrs(arr)
        if encoded is None:
            self._raw, self._encoded = arr, None
            self._nbytes = arr.nbytes
        else:
            self._raw, self._encoded = None, encoded
            self._nbytes = sum(part.nbytes for part in encoded)

    def __len__(self) -> int:
        return self._length

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def compressed(self) -> bool:
        return self._encoded is not None

    def array(self) -> np.ndarray:
        return self._raw if self._encoded is None else _rle_decode_addrs(
            self._encoded)


def _undo_entry_bytes(entry: tuple) -> int:
    """Byte size of a single undo entry (removed_addrs + removed_values)."""
    a, v, _, _ = entry
    if isinstance(a, _UndoAddrs):
        nb = a.nbytes
    elif isinstance(a, np.ndarray):
        nb = a.nbytes
    else:
        nb = len(a) * 8
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
    new_entry   = (_UndoAddrs(removed_addrs), removed_values, prev_dropped,
                   bool(prev_truncated))
    new_bytes   = _undo_entry_bytes(new_entry)
    # Evict oldest entries until we are under the cap (beyond normal maxlen).
    cap_bytes   = int(HISTORY_RAM_CAP_MB * 1_048_576)
    current_b   = _history_bytes()
    while state["scan_history"] and (current_b + new_bytes) > cap_bytes:
        evicted  = state["scan_history"].popleft()
        current_b -= _undo_entry_bytes(evicted)
    state["scan_history"].append(new_entry)


def _apply_scan_undo() -> Optional[np.ndarray]:
    """Pop and apply one undo delta from state["scan_history"].

    Reconstructs the previous, sorted/deduplicated candidate set (and, for an
    unknown-value session, the previous value array) via union1d + searchsorted
    scatter. Also discards any resident TurboScan session: COUNT narrows the
    server's candidate list in place with no rewind operation, so a stale
    session would let the next Next Scan silently rescan the pre-undo
    server-side list instead of the client-reconstructed one. Returns the new
    scan_results, or None if there was no history to undo.
    """
    if not state["scan_history"]:
        return None
    stored_a, removed_v, prev_dropped, prev_truncated = (
        state["scan_history"].pop())
    removed_a = (stored_a.array() if isinstance(stored_a, _UndoAddrs)
                 else stored_a)
    cur_addrs = state["scan_results"]
    prev_addrs = np.union1d(cur_addrs, removed_a)
    if removed_v is not None and state.get("scan_values") is not None:
        cur_v = state["scan_values"]
        dtype = np.dtype(VALUE_TYPES[_current_scan_type()]["dtype"])
        prev_vals = np.zeros(len(prev_addrs), dtype=dtype)
        prev_vals[np.searchsorted(prev_addrs, cur_addrs)] = cur_v
        prev_vals[np.searchsorted(prev_addrs, removed_a)] = removed_v
    else:
        prev_vals = state.get("scan_values")
    scan.restore((prev_addrs, prev_vals, state.get("scan_pid"), prev_dropped,
                  prev_truncated, state.get("scan_unknown", False)))
    _close_turbo_session()
    return prev_addrs

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
# PS4CheaterNeo permits an unlimited pointer offset.  RDX keeps a finite safety
# bound, but the exhaustive sorted index can search a much wider interval with
# two binary searches; it need not enumerate every possible offset.  Small
# tiers preserve common-case ranking while the 1 MiB fallback prevents the old
# ±8 KiB ceiling from guaranteeing false negatives on manager/pool layouts.
_PTR_RESOLVE_OFFSET_MAX = 0x100000
_PTR_RESOLVE_OFFSET_TIERS = (0x100, 0x1000, 0x4000, 0x10000, 0x100000)
_PTR_RESOLVE_OFFSET_STEP = 4      # common 32-bit structure-field alignment
_PTR_RESOLVE_MAX_HITS = 192       # deterministic candidate window per target
_PTR_RESOLVE_MAX_NODES = 2500
_PTR_RESOLVE_MAX_FOUND = 96
# Displacement window the cheap first pass will accept between a holder's
# pointer value and the target.
#
# History, because this constant has been wrong twice in opposite directions.
# It began at 0x100 (256 B), which assumed the target is a field in a small
# struct and found nothing on a real title. patch77 then widened it to 0x10000
# (64 KiB) after an IL2CPP field appeared to sit 0x90F8 from its holder, and
# that looked like a success: five module-rooted chains in 7.3 s where the
# previous search had taken over 30 minutes.
#
# It was not a success. All five failed the two-reload test -- they were this
# session's heap coincidences. A later session produced 24 more of the same
# shape, offsets -0x18C8 to -0x2408 in an exact 48-byte arithmetic series,
# which is an IL2CPP static-field pointer table rather than parents of the
# target. Speed had been measured and mistaken for correctness.
#
# Cheat Engine, the canonical implementation, defaults its maximum offset to
# **2048** and documents that it "should be changed only if you suspect that
# any of the offsets will be larger than 2048 bytes ... which is not so common
# for most of the values that cheaters are looking for". Every coincidence
# above lies beyond that bound, so the canonical setting would have rejected
# them and let the search fall through to the deeper tiers -- which is where
# Cheat Engine, PS4CheaterNeo and PINCE all actually find pointers.
#
# Depth, not proximity, is what finds a real chain. Widening this window only
# buys coincidences that cannot be told apart from real chains until the heap
# moves.
_PTR_FAST_DIRECT_RANGE = 0x800
_PTR_FAST_DIRECT_HITS = 24
# Collect several times the returned cap before ranking. The scan stops as soon
# as it has max_hits, and a wide window makes coincidental holders far more
# likely, so stopping at the first 24 could lock in junk from whichever region
# was scanned first and never reach the real holder. Gather more, rank by
# region priority and smallest displacement, then truncate.
_PTR_FAST_DIRECT_POOL = 8
# A real parent pointer points at an object's base, with the field of interest
# a short way inside it. A holder whose pointer lands *above* the target, or
# thousands of bytes below it, is a coincidence: some unrelated object that
# happens to sit nearby in this session's heap.
#
# This distinction is what patch77's wider window lost. With the old 256-byte
# window tier 1 usually found nothing and the search fell through to the deeper
# tiers; at 64 KiB it matches coincidences and returns early, so the deeper
# search that could find the real chain never runs. Measured on Enter the
# Gungeon: 24 "verified" depth-1 candidates, offsets -9224 to -6344 in an exact
# 48-byte arithmetic series -- an IL2CPP static-field pointer table, not
# parents. Five of the same shape from an earlier session survived 0/5 reloads.
_PTR_PLAUSIBLE_FIELD_MAX = 0x200


def _candidate_field_offset_is_plausible(candidate: dict) -> bool:
    """Whether a chain's final hop looks like a field inside an object.

    The rule above was only ever enforced on the fast-direct short-circuit.
    The locality and reverse-index passes returned candidates in a ranking
    that did not consider it at all, so a holder pointing thousands of bytes
    past the target -- the exact shape recorded above as surviving 0/5
    reloads -- could and did outrank a real multi-hop chain. Following that
    recommendation costs two game reloads to disprove, with the real chain
    sitting one row below the whole time.

    The last offset is the displacement inside the final object, which is
    what the rule is about; for a depth-1 candidate it is also offsets[0],
    so this agrees with the fast-direct filter rather than competing with it.
    """
    offsets = candidate.get("offsets") or []
    if not offsets:
        return True
    # Magnitude, not sign. The rule is about *distance* from the object base:
    # a displacement of hundreds of KiB is the coincidence shape. A negative
    # displacement is ordinary -- a chain that lands inside a sub-object and
    # reads a field earlier in its parent has a small negative final offset,
    # and Cheat Engine has always allowed them.
    #
    # The `0 <=` form rejected every negative offset as implausible. Measured
    # on CUSA01659 (ps5debug-NG, 2026-08-30) against a real ammo address: of
    # the eight top-ranked chains, four ended in -0x60 -- 96 bytes, textbook
    # field shape -- and all eight were judged implausible. With nothing
    # separating them, the sort fell through to `depth`, which promoted the
    # depth-1 holders at +0x42720 (271,648 bytes, the low bits of the target
    # address) above the real chains. The guard was not merely absent, it was
    # inverted in effect: it demoted the plausible candidates it exists to
    # promote.
    return abs(int(offsets[-1])) <= _PTR_PLAUSIBLE_FIELD_MAX


def _rank_pointer_candidates(ip: str, pid: int, candidates: list,
                             region_starts=None, region_rows=None) -> list:
    """Drop self-revisiting chains, then rank plausible candidates first.

    _resolve_permanent_candidates has three return paths -- fast-direct,
    locality-first and reverse-index -- and each had grown its own sort.
    Only the first applied the structural-plausibility rule, so which
    ranking a user saw depended on which tier happened to answer, and the
    locality tier would put a coincidence-shaped holder above a real chain.
    One function for all three is the fix; the divergence was the bug.
    """
    kept, dropped = [], 0
    for candidate in candidates:
        try:
            _ok, _final, steps = _resolve_pointer_chain(
                ip, pid, int(candidate.get("base", 0)),
                [int(x) for x in candidate.get("offsets", ())],
                int(candidate.get("terminal_offset", 0)))
        except Exception:
            steps = ()
        if _chain_revisits_an_address(steps):
            dropped += 1
            continue
        kept.append(candidate)
    if dropped:
        add_log(f"Pointer search: dropped {dropped} chain(s) that revisit an "
                f"address already on their own path", "warn")

    def region_rank(candidate):
        if region_starts is None:
            return 0
        return -_region_priority(
            _region_for_addr(int(candidate.get("base", 0)),
                             region_starts, region_rows) or {})

    kept.sort(key=lambda c: (
        # Structural plausibility first, ahead of score and depth: a holder
        # pointing thousands of bytes from the target is the coincidence
        # shape recorded at _PTR_PLAUSIBLE_FIELD_MAX, and 24 of them once
        # verified clean and then survived 0/5 reloads. Still kept -- when
        # nothing better exists they are the only lead -- but never first.
        not _candidate_field_offset_is_plausible(c),
        not bool(c.get("verified")),
        not bool(c.get("module_name")),
        -float(c.get("score", 0.0)),
        int(c.get("depth", 99)),
        region_rank(c),
        sum(0 if int(x) % 8 == 0 else 1 for x in c.get("offsets", [])),
        sum(abs(int(x)) for x in c.get("offsets", [])),
        str(c.get("module_name", "")),
        int(c.get("module_relative_offset", 0)),
        tuple(int(x) for x in c.get("offsets", [])),
        int(c.get("base", 0)),
    ))
    return kept


def _chain_revisits_an_address(steps) -> bool:
    """True when a resolved chain passes through the same address twice.

    A walker with no cycle guard emits [16, -16368, 48] and
    [16, -16368, -16368, 48] beside the real [16, 48]: all three resolve to
    the target, because the extra hops step off an address and back onto it.
    They are longer restatements of the same chain, and they push the real
    one down the list."""
    seen = set()
    for address in steps or ():
        if int(address) in seen:
            return True
        seen.add(int(address))
    return False
# Seconds the locality (tier 2) pass may run before the search gives up on it
# and falls through to the reverse index, whose cost is bandwidth-bound and
# therefore predictable. Two minutes is far longer than tier 2 needs when it is
# going to succeed at all, and far shorter than the 30+ minutes it can spend
# failing.
_PTR_LOCALITY_TIME_BUDGET = 120.0

# Bounded streaming pointer scanner.  These limits are separate from the
# reverse-index resolver above and must remain defined for pointer_chain_scan.
_PTR_STRUCT_MAX = 0x4000             # ±16 KiB; interval matching keeps this cheap


def _ptr_struct_max() -> int:
    """User-overridable ±window for a single pointer hop (default above)."""
    return int(setting("ptr_offset_max"))

_PTR_STRUCT_STEP = 4
_PTR_CHUNK = 0x2000000              # 32 MiB network reads
_PTR_BEAM_MAX = 12_000              # heap targets carried to the next depth
_PTR_VERIFY_MAX = 64                # bounded same-session network validations
_PTR_VERIFY_PER_FAMILY = 2          # duplicate roots per converged heap path
_PTR_SEARCH_TARGET_MAX = 4_096      # bounded deep-pass value set
_PTR_DEPTH_DEFAULT = 5
_PTR_DISK_INDEX_THRESHOLD = 0x40000000  # 1 GiB readable memory
_PTR_DISK_SHARD_BYTES = 0x2000000       # 32 MiB per sorted disk shard
_PTR_DISK_WORKERS = 6                    # persistent parallel reader/indexers
                                         # (kept under _MAX_CONSOLE_SOCKETS)
_pointer_region_class_cache = {}         # map fingerprint -> classified rows
_region_class_supported: dict = {}       # map fingerprint -> probe actually ran
_region_class_lock = threading.Lock()
_POINTER_PROVISIONAL_FILE = Path(__file__).with_name(
    ".rdx-pointer-candidates.json")
_PREFERENCES_FILE = Path(__file__).with_name(".rdx-preferences.json")


# ── scan state aggregate ──────────────────────────────────────────────────────
# scan_results, scan_values, scan_pid, scan_dropped, scan_truncated and
# scan_unknown are one fact, not six. They were kept consistent by convention,
# re-established by hand at each of thirteen mutation sites -- which is why
# dropping a single result needed a manual np.delete on the parallel value
# array *and* a _close_turbo_session() call, and why replacing the candidate
# list took six assignments in a fixed order.
#
# ScanState owns those invariants at the boundary instead:
#
#   * values stay the same length as addresses, or are dropped entirely;
#   * any replacement of the candidate set closes a resident TurboScan
#     session, because the server holds its own copy of the survivor list and
#     would otherwise hand back the pre-replacement one on the next scan;
#   * a snapshot is a plain tuple, so undo is a swap rather than a
#     field-by-field restore.
#
# The dict remains the single source of truth so the ~200 read sites are
# untouched; this is a controller over it, not a parallel copy.
class ScanState:
    """Invariant-preserving controller for the correlated scan fields."""

    __slots__ = ("_state",)

    FIELDS = ("scan_results", "scan_values", "scan_pid", "scan_dropped",
              "scan_truncated", "scan_unknown")

    # `pid=None` on replace() means "these results came from the process we
    # are attached to now". Clearing has to say something different -- "these
    # results belong to no process" -- because do_show_results distinguishes
    # a null scan_pid from a mismatched one when it blocks stale results.
    # One sentinel keeps both meanings expressible through one parameter.
    UNSET = object()

    def __init__(self, backing: dict):
        self._state = backing

    # ── reads ──
    @property
    def addrs(self):
        return self._state["scan_results"]

    @property
    def values(self):
        return self._state.get("scan_values")

    def __len__(self) -> int:
        return len(self._state["scan_results"])

    # ── invariants ──
    def _check(self) -> None:
        values = self._state.get("scan_values")
        if values is not None and len(values) != len(self._state["scan_results"]):
            # Never seen in production, but a mismatch here silently corrupts
            # the next relational scan by pairing an address with another
            # address's previous value. Fail loudly instead of scanning wrong.
            raise AssertionError(
                f"scan value/address length mismatch: "
                f"{len(values)} values, {len(self._state['scan_results'])} addresses")

    # ── writes ──
    def replace(self, addrs, values=None, *, pid=None, unknown=False,
                truncated=False, keep_dropped=False,
                close_turbo=True) -> None:
        """Install a new candidate set, consistently."""
        if close_turbo:
            _close_turbo_session()
        self._state["scan_results"] = addrs
        self._state["scan_values"] = values
        if pid is None:
            self._state["scan_pid"] = self._state["pid"]
        elif pid is ScanState.UNSET:
            self._state["scan_pid"] = None
        else:
            self._state["scan_pid"] = pid
        self._state["scan_unknown"] = bool(unknown)
        self._state["scan_truncated"] = bool(truncated)
        if not keep_dropped:
            self._state["scan_dropped"] = set()
        self._check()

    def narrow(self, addrs, values=None, *, truncated=False) -> None:
        """Record the result of a next-scan over the existing candidates."""
        self._state["scan_results"] = addrs
        self._state["scan_values"] = values
        self._state["scan_truncated"] = bool(truncated)
        self._check()

    def drop_index(self, index: int):
        """Remove one candidate by position, keeping values aligned."""
        addrs = self._state["scan_results"]
        if not 0 <= index < len(addrs):
            return None
        dropped = addrs[index]
        # A resident session is matched by connection/PID/width/value-type
        # alone, never by candidate count, so it would happily narrow the
        # server's pre-drop list and hand this address straight back.
        _close_turbo_session()
        self._state["scan_results"] = _make_addr_array(
            a for i, a in enumerate(addrs) if i != index)
        values = self._state.get("scan_values")
        if values is not None:
            self._state["scan_values"] = np.delete(values, index)
        self._state["scan_dropped"].add(dropped)
        self._check()
        return dropped

    def drop_address(self, address: int):
        """Remove one candidate by address."""
        addrs = self._state["scan_results"]
        matches = [i for i, a in enumerate(addrs) if int(a) == int(address)]
        return self.drop_index(matches[0]) if matches else None

    def clear(self) -> None:
        self.replace(_make_addr_array(), None, pid=ScanState.UNSET,
                     close_turbo=False)

    # ── undo ──
    def snapshot(self) -> tuple:
        return tuple(self._state.get(field) for field in self.FIELDS)

    def restore(self, snap: tuple) -> None:
        for field, value in zip(self.FIELDS, snap):
            self._state[field] = value
        self._check()


# Defined before the preference loader that reads it.
_GUIDE_PREF_KEY = "guide_seen"


def _load_preferences(path: Optional[Path] = None) -> dict:
    """Load small, non-sensitive UI preferences; corrupt files fail closed."""
    src = Path(path or _PREFERENCES_FILE)
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or int(data.get("version", 0)) != 1:
            return {}
        out = {}
        for key in ("last_ip", "last_process", "export_dir"):
            value = data.get(key)
            if isinstance(value, str) and len(value) <= 1024:
                out[key] = value
        if data.get(_GUIDE_PREF_KEY):
            out[_GUIDE_PREF_KEY] = True
        # Tunables are bounded on the way in: a hand-edited file must not be
        # able to set an unbounded pointer depth or a zero-size scan window.
        for key in _SETTING_SPECS:
            if key in data:
                out[key] = _coerce_setting(key, data[key])
        return out
    except (OSError, ValueError, TypeError):
        return {}


def _save_preferences(updates: Optional[dict] = None,
                      path: Optional[Path] = None) -> None:
    """Atomically persist connection/export convenience settings."""
    dst = Path(path or _PREFERENCES_FILE)
    payload = {"version": 1}
    payload.update(_load_preferences(dst))
    for key, value in (updates or {}).items():
        if key in {"last_ip", "last_process", "export_dir"}:
            payload[key] = str(value or "")[:1024]
        elif key == _GUIDE_PREF_KEY:
            payload[key] = bool(value)
        elif key in _SETTING_SPECS:
            payload[key] = _coerce_setting(key, value)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, dst)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a UTF-8 text export without leaving a half-written trainer."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, dst)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write a binary/base64 export (e.g. .mc4) without a half-written file."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, dst)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


# ── tunable settings ──────────────────────────────────────────────────────────
# The constants above are defaults, not policy. Every one of them was reasoned
# about and several carry their own history (see _PTR_FAST_DIRECT_RANGE), so
# they stay exactly as they are — what changes here is that the user can see
# and override them without editing the source.
#
# PINCE exposes the same five pointer knobs (max depth, max positive/negative
# offset, module-bases-only, scan scope) in its Pointer Scanner window, and
# PS4CheaterNeo exposes its section-filter rules and a minimum section size in
# Options. RDX had all of them as literals. Note that RDX's depth default of 5
# and its 0x800 direct window match PINCE's independently chosen defaults
# exactly, which is why this is a visibility change and not a tuning one.
#
# Every setting is bounded on load: a hand-edited preferences file must not be
# able to talk RDX into an unbounded pointer walk or a zero-width scan window.
_SETTING_SPECS = {
    "ptr_max_depth": {
        "label": "Pointer max depth",
        "kind": "int", "default": _PTR_DEPTH_DEFAULT,
        "min": 1, "max": MAX_CHAIN_DEPTH,
        "help": "Chain hops to explore. Deeper finds more, costs more.",
    },
    "ptr_direct_range": {
        "label": "Pointer direct window",
        "kind": "hex", "default": _PTR_FAST_DIRECT_RANGE,
        "min": 0x40, "max": 0x10000,
        "help": "First-pass holder search radius. 0x800 matches PINCE; "
                "widening it buys coincidences, not chains.",
    },
    "ptr_offset_max": {
        "label": "Pointer struct window",
        "kind": "hex", "default": _PTR_STRUCT_MAX,
        "min": 0x100, "max": 0x100000,
        "help": "Max |offset| accepted at each hop of the streaming scan.",
    },
    "ptr_module_bases_only": {
        "label": "Module bases only",
        "kind": "bool", "default": False,
        "help": "Keep only chains rooted in a named module. Fewer results, "
                "all of them expressible as module+offset.",
    },
    "region_min_size": {
        "label": "Min region size",
        "kind": "hex", "default": 0,
        "min": 0, "max": 0x10000000,
        "help": "Skip mappings smaller than this when scanning. 0 = off; "
                "PS4CheaterNeo defaults to 0x32000 (200K).",
    },
    "region_exclude": {
        "label": "Region exclude tokens",
        "kind": "csv", "default": (".sprx,.prx,.so,/lib/,libkernel,libsce,"
                                   "ps5debug,ps4debug,memdbg,etahen,goldhen"),
        "help": "Comma-separated substrings excluded from Recommended scope.",
    },
}


def _coerce_setting(key: str, value):
    """Clamp/normalise one setting; returns the default when unusable."""
    spec = _SETTING_SPECS[key]
    kind = spec["kind"]
    try:
        if kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if kind == "csv":
            if isinstance(value, (list, tuple)):
                value = ",".join(str(v) for v in value)
            parts = [t.strip().lower() for t in str(value).split(",")]
            parts = [t for t in parts if t][:64]
            return ",".join(parts)
        number = int(str(value), 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        return spec["default"]
    return max(spec["min"], min(spec["max"], number))


_settings = {key: spec["default"] for key, spec in _SETTING_SPECS.items()}


def setting(key: str):
    """Read a tunable. Always returns a bounded, usable value."""
    return _settings.get(key, _SETTING_SPECS[key]["default"])


def _region_exclude_tokens() -> tuple:
    """Exclusion substrings for the Recommended scan scope."""
    raw = str(setting("region_exclude"))
    return tuple(t for t in (p.strip() for p in raw.split(",")) if t)


_preferences = _load_preferences()
for _key in _SETTING_SPECS:
    if _key in _preferences:
        _settings[_key] = _preferences[_key]

# Issues #7/#8/#9/#10: track the active freeze worker globally so it can be
# stopped when the user changes process or reconnects.  Without this the old
# worker keeps writing to an address in the previous process's address space,
# which either silently does nothing or corrupts unrelated memory if the PID
# was re-used by the OS.
_freeze_stop:   threading.Event  = threading.Event()
_freeze_thread: Optional[threading.Thread] = None
_freeze_lock:   threading.RLock  = threading.RLock()
_freeze_targets: dict = {}       # runtime id -> saved cheat object
_freeze_status: dict = {}        # runtime id -> last success/error text


# Trainer files are third-party input. HEN-Cheats-Collection alone carries
# 2,364 games' worth in JSON/MC4/SHN, so the common case is importing a file
# somebody else wrote, and a corrupt or hostile one is a normal thing to meet.
#
# Every other collection in this program is bounded -- bookmarks at 256,
# remembered structures at 64, symbol classes at 20,000, scan results at 2 M,
# undo history at 128 MB, wire-decoded process and map entries at 4,096 and
# 65,536. Imported cheats were the sole exception: 200,000 entries from a
# 20 MB file were accepted at 283 MB peak, and each enabled freeze costs a
# per-tick address resolution that explicitly cannot be batched.
#
# 1,024 is far above any real trainer (published ones run to tens of entries)
# and far below the point where the cheat list or the freeze tick stops being
# usable.
MAX_IMPORT_CHEATS = 1024
# Read guard applied before parsing rather than after: the parse peaks at
# roughly ten times the file size, so rejecting a 20 MB trainer costs 200 MB
# if the check comes afterwards.
MAX_TRAINER_FILE_BYTES = 8 * 1024 * 1024

# Appended to an exported cheat's name when its freeze semantics do not
# survive the container. Short on purpose: it shares a line with the cheat's
# own name in someone else's menu.
_ONE_SHOT_MARKER = "[1-shot]"
# A cheat whose on and off values are identical writes what is already there.
# It cannot do anything, and in a manager it is indistinguishable from a
# broken one. Observed for real on 2026-08-30: a CheatRunner trainer built
# with deliberately inert values loaded correctly, did nothing when toggled,
# and the reasonable question back was "what is this supposed to do?".
#
# The person who discovers an inert cheat is the person least able to diagnose
# it -- they see a toggle that does not work, not an equal pair of byte
# strings in a file they cannot read. Same reasoning as _ONE_SHOT_MARKER, and
# the same mechanism: the name is the one field every manager displays.
_INERT_MARKER = "[no-op]"

_BOOKMARK_MAX = 256


def _bookmark_key(address: int, value_type: str) -> tuple:
    return (int(address), str(value_type))


def _add_bookmark(address: int, value_type: str, note: str = "",
                  chain: Optional[dict] = None) -> str:
    """Record an address for later. Returns a status line for the caller.

    `chain` optionally carries {module_name, module_relative_offset, offsets,
    terminal_offset} from a verified pointer search. A bookmark holding one
    rebases on the next attach instead of expiring -- see
    _bookmark_is_current.
    """
    bookmarks = state.setdefault("bookmarks", [])
    key = _bookmark_key(address, value_type)
    for existing in bookmarks:
        if _bookmark_key(existing["address"], existing["value_type"]) == key:
            if note:
                existing["note"] = str(note)[:64]
            if chain and not existing.get("chain"):
                existing["chain"] = dict(chain)
                return (f"Bookmark at {hex(int(address))} now carries a "
                        f"pointer chain")
            return f"Bookmark already exists at {hex(int(address))}"
    if len(bookmarks) >= _BOOKMARK_MAX:
        return f"Bookmark limit ({_BOOKMARK_MAX}) reached"
    bookmarks.append({
        "address": int(address),
        "value_type": str(value_type),
        "note": str(note or "")[:64],
        "pid": state.get("pid"),
        "process": state.get("proc_name", ""),
        "session": int(state.get("session", 0)),
        "chain": dict(chain) if chain else None,
    })
    return (f"Bookmarked {hex(int(address))}"
            + (" with its pointer chain" if chain else ""))


def _salvageable_chains(items: list) -> list:
    """Pointer chains inside a trainer written for a different game build.

    A version-mismatched trainer is normally a dead end: the addresses in it
    are wrong for the build that is running. But an entry carrying a
    module-rooted chain is not just an address -- it is a record of a
    structure someone already worked out. Game layouts change far less
    between patches than absolute addresses do, so those offsets are usually
    still correct and are worth re-verifying rather than discarding.

    This is the common case, not an edge case: HEN-Cheats-Collection carries
    2,364 games and updates, organised by title ID *and* version, so holding
    a trainer for a version you are not running is ordinary.

    EdiZon SE does the same thing from the other direction, extracting a
    chain from a cheat "made for a previous version of the game".

    Returns [{name, module_name, module_relative_offset, offsets,
    terminal_offset}] for the entries worth offering.
    """
    salvage = []
    for c in items:
        if not isinstance(c, dict):
            continue
        module_name = str(c.get("module_name", c.get("module", "")) or "")
        raw_offsets = c.get("offsets")
        if not module_name or not isinstance(raw_offsets, list):
            continue
        if not (1 <= len(raw_offsets) <= MAX_CHAIN_DEPTH):
            continue
        try:
            offsets = [int(x, 0) if isinstance(x, str) else int(x)
                       for x in raw_offsets]
            module_rel = c.get("module_relative_offset", c.get("module_offset"))
            module_rel = (int(module_rel, 0) if isinstance(module_rel, str)
                          else int(module_rel))
        except (TypeError, ValueError):
            continue
        if module_rel < 0 or module_rel > _ADDR_MAX:
            continue
        if any(abs(x) > _PTR_RESOLVE_OFFSET_MAX for x in offsets):
            continue
        salvage.append({
            "name": str(c.get("name", "unnamed")),
            "module_name": module_name,
            "module_relative_offset": module_rel,
            "offsets": offsets,
            "terminal_offset": int(c.get("terminal_offset", 0) or 0),
        })
        if len(salvage) >= MAX_IMPORT_CHEATS:
            break
    return salvage


def _verify_salvaged_chain(chain: dict) -> Optional[int]:
    """Resolve a salvaged chain against the running build, or None.

    Nothing is trusted from the file: the module base comes from the live
    map and the chain is walked from there, exactly as a saved cheat's is.
    """
    try:
        maps = _get_maps_cached(state["ip"], int(state["pid"]))
        module_base = _pointer_module_base(maps, chain["module_name"])
        if module_base is None:
            return None
        base = module_base + int(chain["module_relative_offset"])
        ok, final, _steps = _resolve_pointer_chain(
            state["ip"], int(state["pid"]), base,
            [int(x) for x in chain["offsets"]],
            int(chain.get("terminal_offset", 0)))
        if not ok:
            return None
        if _validate_addr_in_maps(state["ip"], int(state["pid"]),
                                  int(final), 1):
            return None
        return int(final)
    except Exception:
        return None


def _attach_chain_to_bookmark(address: int, candidate: dict) -> Optional[str]:
    """Give any bookmark at `address` the verified chain just found for it.

    Returns a status line when one was attached, else None. Silent when no
    bookmark is on that address -- the pointer search is used far more often
    from Results than from the bookmark list, and it should not start
    creating bookmarks nobody asked for.
    """
    chain = {
        "module_name": str(candidate.get("module_name", "") or ""),
        "module_relative_offset": int(
            candidate.get("module_relative_offset", 0)),
        "offsets": [int(x) for x in candidate.get("offsets", ())],
        "terminal_offset": int(candidate.get("terminal_offset", 0)),
    }
    if not chain["module_name"]:
        return None
    for bookmark in state.get("bookmarks", []):
        if int(bookmark.get("address", 0)) != int(address):
            continue
        if bookmark.get("chain"):
            return None
        bookmark["chain"] = chain
        return (f"Bookmark {hex(int(address))} now carries "
                f"{chain['module_name']}+{chain['module_relative_offset']:#x} "
                f"— it will survive a reload")
    return None


def _remove_bookmark(index: int) -> Optional[dict]:
    bookmarks = state.get("bookmarks", [])
    if 0 <= index < len(bookmarks):
        return bookmarks.pop(index)
    return None


def _bookmark_is_current(bookmark: dict) -> bool:
    """Whether a bookmark still refers to the thing it was taken on.

    A bookmark with no chain is a raw address, so after a reload it names
    whatever now occupies that memory. Marking those stale is the honest
    presentation; silently reading one would show a plausible number
    belonging to something else entirely. That reasoning is unchanged.

    What changed is the premise: a bookmark does not *have* to be a raw
    address. One carrying a verified module-rooted chain rebases on the next
    attach exactly as a saved cheat does, so it survives the reload that
    would have expired it. EdiZon SE reaches the same conclusion from the
    other direction -- its bookmarks "adjust to changing main and heap start
    address on subsequent launch of the game".
    """
    if bookmark.get("chain"):
        return _bookmark_chain_resolves(bookmark) is not None
    return (int(bookmark.get("session", -1)) == int(state.get("session", 0))
            and bookmark.get("pid") == state.get("pid"))


def _bookmark_chain_resolves(bookmark: dict) -> Optional[int]:
    """Live address for a chained bookmark, or None when it cannot rebase.

    Reuses the cheat path's rebasing primitives rather than repeating them:
    the module base is looked up in the current maps and the chain walked
    from there, so a bookmark and a cheat built on the same chain resolve
    identically.
    """
    chain = bookmark.get("chain") or {}
    module_name = str(chain.get("module_name", "") or "")
    if not module_name:
        return None
    try:
        maps = _get_maps_cached(state["ip"], int(state["pid"]))
        module_base = _pointer_module_base(maps, module_name)
        if module_base is None:
            return None
        base = module_base + int(chain.get("module_relative_offset", 0))
        offsets = [int(x) for x in chain.get("offsets", ())]
        if not offsets:
            return base
        ok, final, _steps = _resolve_pointer_chain(
            state["ip"], int(state["pid"]), base, offsets,
            int(chain.get("terminal_offset", 0)))
        return int(final) if ok else None
    except Exception:
        return None


def _bookmark_live_address(bookmark: dict) -> int:
    """The address to read now: rebased when chained, stored otherwise."""
    if bookmark.get("chain"):
        resolved = _bookmark_chain_resolves(bookmark)
        if resolved is not None:
            return resolved
    return int(bookmark.get("address", 0))


def _cheat_runtime_id(cheat: dict) -> str:
    runtime_id = str(cheat.get("_runtime_id", "") or "")
    if not runtime_id:
        runtime_id = uuid.uuid4().hex
        cheat["_runtime_id"] = runtime_id
    return runtime_id


def _is_cheat_frozen(cheat: dict) -> bool:
    runtime_id = str(cheat.get("_runtime_id", "") or "")
    with _freeze_lock:
        return bool(runtime_id and runtime_id in _freeze_targets)


# ── freeze contention ─────────────────────────────────────────────────────────
# A freeze writes on a timer; the game writes whenever it likes. Measured on
# CUSA01659 against a live ammo address:
#
#     game rewrites the address    every 8-20 ms (median 18)
#     one ps5_write round trip     15.7 ms
#     freeze at its 200 ms tick    held the value 31/657 samples -- 4.7%
#     tight loop, no sleep         held 312/651 -- 47.9%
#
# A network round trip costs about as long as the game's whole write period,
# so this is a race the tick rate cannot win. For values a game writes only on
# change -- currency, item counts, unlock flags -- freezing works exactly as
# advertised. For anything written per frame it does not, and RDX reported
# "active @ 0x..." throughout while the value it claimed to be holding tracked
# the game the entire time. A status that cannot fail is not a status.
#
# So every few ticks the freeze reads back what it wrote. Losing the race is
# reported as contention, not as success. The write is still attempted -- a
# partially-held value is occasionally what the user wants -- but the UI stops
# claiming it worked.
_FREEZE_VERIFY_EVERY = 5        # ticks between read-backs (~1 s at 0.2 s)
_FREEZE_CONTESTED_AT = 0.5      # lost more than half the window -> contested
_FREEZE_VERIFY_WINDOW = 10      # checks kept; bounds recovery as well as detection
_FREEZE_VERIFY_MIN = 3          # below this, one unlucky read decides nothing

# Contention is kept apart from _freeze_status on purpose. The write phase
# rewrites that entry every tick, so a verdict stored there survives ~200 ms
# and is invisible to anything sampling more slowly than the worker loops --
# which is every caller. The first cut of this patch did exactly that and
# reported "active" for a freeze holding 0% of its writes, on hardware, which
# is the bug it was written to fix.
_freeze_contested: dict = {}    # runtime_id -> human-readable verdict


def _freeze_note_verification(state_map: dict, runtime_id: str,
                              held: bool) -> Optional[str]:
    """Record one read-back outcome; return a status when it is conclusive.

    A sliding window, not a lifetime tally. The first version counted every
    check ever taken, which bounded how fast contention was *detected* but
    left recovery unbounded: a freeze contested for five minutes and then
    holding perfectly needed five minutes of wins before the indicator caught
    up, because the losses never aged out. Observed on hardware -- the
    indicator sat on LOSE well after the game had stopped fighting the write.

    With a window both directions cost at most _FREEZE_VERIFY_WINDOW checks,
    so the badge tracks what is happening now rather than what happened when
    the freeze was switched on.
    """
    window = state_map.get(runtime_id)
    if window is None:
        window = state_map[runtime_id] = deque(maxlen=_FREEZE_VERIFY_WINDOW)
    window.append(bool(held))
    if len(window) < _FREEZE_VERIFY_MIN:
        return None
    kept = sum(1 for x in window if x)
    seen = len(window)
    if (1.0 - (kept / float(seen))) > _FREEZE_CONTESTED_AT:
        return (f"contested — the game is overwriting this "
                f"({kept}/{seen} recent checks held)")
    return None


# How long a cheat's address stays meaningful. RDX has computed this for a
# while -- _is_cross_reload_pointer, _is_module_relative_scalar -- and showed
# it only on the detail screen, only for chain cheats, and never at export.
# So a raw heap address and a twice-promoted module-rooted chain look
# identical in the list, and a trainer can be exported without any indication
# that half of it dies on the next launch. Both of the addresses found on
# hardware this session were raw pointers into the managed heap; they would
# have exported cleanly and been dead on the next boot.
_DURABILITY_SESSION = ("SESSION", "current session only — a raw address")
_DURABILITY_RELOAD = ("RELOAD", "survives a reload — module-rooted chain")
_DURABILITY_STATIC = ("STATIC", "module-relative static patch")


def cheat_durability(cheat: dict) -> tuple:
    """(short, long) description of how long this cheat's address survives.

    Deliberately three states rather than a guess at a fourth: nothing here
    survives a *game update*, because RDX has no signature-rooted root yet
    (UPSTREAM_AUDIT_PASS7). Claiming otherwise would be the kind of confident
    label this session has spent its time removing.
    """
    if _is_module_relative_scalar(cheat):
        return _DURABILITY_STATIC
    if _is_cross_reload_pointer(cheat):
        return _DURABILITY_RELOAD
    return _DURABILITY_SESSION


def summarise_durability(cheats: list) -> str:
    """One line naming how many cheats of each durability are in a set."""
    counts: dict = {}
    for cheat in cheats:
        short = cheat_durability(cheat)[0]
        counts[short] = counts.get(short, 0) + 1
    if not counts:
        return "no cheats"
    order = [_DURABILITY_STATIC[0], _DURABILITY_RELOAD[0],
             _DURABILITY_SESSION[0]]
    return ", ".join(f"{counts[k]} {k.lower()}"
                     for k in order if k in counts)


def _cheat_freeze_indicator(cheat: dict) -> str:
    runtime_id = str(cheat.get("_runtime_id", "") or "")
    with _freeze_lock:
        if not runtime_id or runtime_id not in _freeze_targets:
            return "OFF"
        status = str(_freeze_status.get(runtime_id, "") or "")
        contested = runtime_id in _freeze_contested
    if status.startswith("error:"):
        return "ERR"
    # Distinct from ON: the write is landing and being undone. Reporting this
    # as ON is what let a freeze that held 4.7% of the time look like it was
    # working.
    return "LOSE" if contested else "ON"


def freeze_contention_note(cheat: dict) -> Optional[str]:
    """The contention verdict for a cheat, or None when it is holding."""
    runtime_id = str(cheat.get("_runtime_id", "") or "")
    with _freeze_lock:
        return _freeze_contested.get(runtime_id)


def _resolve_cheat_runtime_address(cheat: dict) -> int:
    """Resolve a saved cheat without prompting; raise on stale ownership."""
    portable = _is_portable_cheat(cheat)
    same_process = str(cheat.get("process", "") or "") in (
        "", str(state.get("proc_name", "") or ""))
    portable_here = (portable and same_process and
                     _portable_cheat_matches_current_game(cheat))
    stale = (cheat.get("pid") != state.get("pid") or
             cheat.get("session") != state.get("session"))
    if stale and not portable_here:
        raise ValueError("cheat belongs to another process/session")
    if cheat.get("offsets") is not None:
        base = _runtime_pointer_base(cheat)
        offsets = [int(item, 0) if isinstance(item, str) else int(item)
                   for item in cheat.get("offsets", [])]
        ok, address, _steps = _resolve_pointer_chain(
            state["ip"], state["pid"], base, offsets,
            int(cheat.get("terminal_offset", 0)))
        if not ok:
            raise ValueError("pointer chain is currently unresolved")
        return int(address)
    return int(_runtime_scalar_address(cheat))


def _freeze_manager_worker(stop_event: threading.Event) -> None:
    """Keep every enabled saved cheat active until individually disabled.

    Takes its own stop event rather than reading the module-level one.
    _stop_freeze_worker deliberately leaves that signal asserted when its
    join times out, so that a worker still blocked in a write cannot
    resume later -- which means the next worker cannot share it.
    """
    global _freeze_thread
    ticks = 0
    verified: dict = {}          # runtime_id -> (checks, times it held)
    try:
        while not stop_event.is_set():
            with _freeze_lock:
                targets = list(_freeze_targets.items())
            if not targets:
                # Keep one lightweight manager alive between toggle changes.
                # Exiting here races with a new enable that sees the old
                # thread as alive and therefore does not start a replacement.
                stop_event.wait(0.5)
                continue
            # Phase 1: resolve each target's live address + value. This is
            # inherently per-cheat (a pointer chain dereferences one hop at
            # a time), so it cannot be batched — only the final writes can.
            resolved = []   # [(runtime_id, address, data)]
            for runtime_id, cheat in targets:
                if stop_event.is_set():
                    break
                try:
                    address = _resolve_cheat_runtime_address(cheat)
                    width = int(cheat["width"])
                    error = _validate_addr_in_maps(
                        state["ip"], state["pid"], address, width, 10.0)
                    if error:
                        raise ValueError(error)
                    resolved.append(
                        (runtime_id, address, _cheat_value_bytes(cheat)))
                except Exception as exc:
                    with _freeze_lock:
                        _freeze_status[runtime_id] = f"error: {exc}"

            # Phase 2: collapse every resolved write into one round trip
            # instead of one write per cheat. Three paths, most-specific
            # first: MemDBG's own native batch write when available, else
            # ps5debug-NG's 0xBDAACC04 bulk write when not on MemDBG's
            # native write path, else the plain per-write loop (single
            # entry has no batching benefit either way).
            if len(resolved) >= 2 and _memdbg_has(MEMDBG_CAP_BATCH_WRITE):
                try:
                    results = memdbg_write_multi(
                        state["ip"], state["pid"],
                        [(addr, data) for _rid, addr, data in resolved],
                        cancel_event=stop_event, timeout=2.0)
                except Exception as exc:
                    results = [False] * len(resolved)
                    add_log(f"MemDBG bulk freeze write failed: {exc}", "warn")
                with _freeze_lock:
                    for (runtime_id, address, _data), ok in zip(resolved, results):
                        _freeze_status[runtime_id] = (
                            f"active @ {hex(address)}" if ok
                            else "error: payload rejected the write")
            elif len(resolved) >= 2 and not _memdbg_has(MEMDBG_CAP_MEMORY_WRITE):
                try:
                    results = ps5_write_multi(
                        state["ip"], state["pid"],
                        [(addr, data) for _rid, addr, data in resolved],
                        cancel_event=stop_event, timeout=2.0)
                except Exception as exc:
                    results = [False] * len(resolved)
                    add_log(f"Bulk freeze write failed: {exc}", "warn")
                with _freeze_lock:
                    for (runtime_id, address, _data), ok in zip(resolved, results):
                        _freeze_status[runtime_id] = (
                            f"active @ {hex(address)}" if ok
                            else "error: payload rejected the write")
            else:
                for runtime_id, address, data in resolved:
                    if stop_event.is_set():
                        break
                    try:
                        if not ps5_write(
                                state["ip"], state["pid"], address, data,
                                cancel_event=stop_event, timeout=1.0):
                            raise IOError("payload rejected the write")
                        with _freeze_lock:
                            _freeze_status[runtime_id] = f"active @ {hex(address)}"
                    except Exception as exc:
                        with _freeze_lock:
                            _freeze_status[runtime_id] = f"error: {exc}"

            # Every few ticks, check the writes are actually sticking.
            ticks += 1
            if resolved and ticks % _FREEZE_VERIFY_EVERY == 0:
                for runtime_id, address, data in resolved:
                    if stop_event.is_set():
                        break
                    try:
                        live = ps5_read(state["ip"], state["pid"],
                                        address, len(data))
                    except Exception:
                        continue
                    verdict = _freeze_note_verification(
                        verified, runtime_id, live == data)
                    with _freeze_lock:
                        if verdict is not None:
                            _freeze_contested[runtime_id] = verdict
                        else:
                            _freeze_contested.pop(runtime_id, None)

            stop_event.wait(0.2)
    finally:
        with _freeze_lock:
            if threading.current_thread() is _freeze_thread:
                _freeze_thread = None


def _ensure_freeze_worker() -> None:
    """Guarantee a running worker for the currently enabled freezes.

    The alive check alone was not enough. _stop_freeze_worker keeps the stop
    signal asserted when its join times out -- a worker blocked in a slow
    write -- and retains the thread reference. Enabling a freeze during that
    window found a live thread and returned, leaving the signal asserted; the
    old worker then exited and cleared the reference, so the cheat sat at
    "starting @ ..." with no worker and no way back short of toggling it
    again. Reachable whenever a write outlives the 2 s join, which a degraded
    console makes easy, and silent when it happens.

    Clearing the shared event instead would resume the outgoing worker, which
    is exactly what _stop_freeze_worker is preventing. So a new worker gets a
    new event and the two cannot interfere.
    """
    global _freeze_thread, _freeze_stop
    with _freeze_lock:
        if (_freeze_thread and _freeze_thread.is_alive()
                and not _freeze_stop.is_set()):
            return
        _freeze_stop = threading.Event()
        _freeze_thread = threading.Thread(
            target=_freeze_manager_worker, args=(_freeze_stop,),
            name="rdx-freeze-manager", daemon=True)
        _freeze_thread.start()


def _toggle_cheat_freeze(cheat: dict) -> bool:
    """Toggle one saved cheat while leaving every other toggle untouched."""
    runtime_id = _cheat_runtime_id(cheat)
    with _freeze_lock:
        if runtime_id in _freeze_targets:
            _freeze_targets.pop(runtime_id, None)
            _freeze_status.pop(runtime_id, None)
            _freeze_contested.pop(runtime_id, None)
            cheat["enabled"] = False
            add_log(f"Disabled freeze '{cheat.get('name', 'Unnamed')}'")
            return False
    # Resolve and validate before presenting the toggle as enabled.
    address = _resolve_cheat_runtime_address(cheat)
    error = _validate_addr_in_maps(
        state["ip"], state["pid"], address, int(cheat["width"]), 0.0)
    if error:
        raise ValueError(error)
    with _freeze_lock:
        _freeze_targets[runtime_id] = cheat
        _freeze_status[runtime_id] = f"starting @ {hex(address)}"
        cheat["enabled"] = True
    _ensure_freeze_worker()
    add_log(f"Enabled freeze '{cheat.get('name', 'Unnamed')}'")
    return True

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
        for cheat in _freeze_targets.values():
            cheat["enabled"] = False
        _freeze_targets.clear()
        _freeze_status.clear()
        _freeze_contested.clear()
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
    "ip":           _preferences.get("last_ip", ""),
    "connected":    False,
    # ps5debug remains the default transport. Current MemDBG payloads advertise
    # native read/write capabilities; older builds transparently fall back to
    # their optional ps5debug-compatible listener.
    "backend":      "ps5debug",           # ps5debug / memdbg-experimental
    "memdbg":       None,                 # HELLO metadata/capabilities
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
    "scan_type":    "u32",
    "scan_tolerance": 0.0001,
    "scan_pattern": "",
    "scan_aligned":       True,
    "scan_writable_only": True,
    "scan_scope":         "recommended",
    "scan_engine": "auto",          # auto / turbo / console / host
    "cheats":       [],
    # Bookmarks are "addresses I am still investigating". Before this, the
    # only way to keep an address across screens was to create a cheat --
    # which sets cheats_dirty, joins export selection and triggers the
    # unsaved-cheats quit guard. There was no way to say "keep this, I am
    # not finished". PINCE keeps an unlimited bookmark list for the same
    # reason. Session-scoped on purpose: a bookmark is a scratch note, and
    # persisting it would resurrect stale addresses after a reload.
    "bookmarks":    [],
    # {base_address: [field, ...]} — session-scoped like bookmarks, and for
    # the same reason: a structure describes one process's layout.
    "structures":   {},
    # {class_name: [field, ...]} from an Il2CppDumper dump.cs. Unlike
    # structures and bookmarks this survives a process change: it describes
    # the title's layout, not this session's memory, so clearing it on every
    # reattach would mean reloading the file constantly.
    "symbols":      {},
    "cheats_dirty": False,   # True after add/delete since the last export
    "last_deleted_cheat": None,   # (cheat_dict, original_index) — single-slot undo
    "game_id":      "",
    "game_ver":     "01.00",
    "game_title":   "",
    "export_dir":   _preferences.get("export_dir", str(Path.home())),
    "last_process": _preferences.get("last_process", "eboot.bin"),
    "log":          [],
    # Undo history — delta format; see _push_undo() above.
    "scan_history": deque(maxlen=5),
}

# Controller over the correlated scan fields above; see ScanState.
scan = ScanState(state)

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


def _lz4_decompress_block(data: bytes, expected_length: int) -> bytes:
    """Decode one raw LZ4 block with strict bounds and length validation.

    MemDBG wraps compressible memory reads in a tiny command-local frame rather
    than an LZ4 frame/container.  Keeping the block decoder here avoids making
    a third-party Python package mandatory for the experimental native backend.
    """
    expected_length = int(expected_length)
    if expected_length < 0 or expected_length > MEMDBG_MAX_MEMORY_READ:
        raise RuntimeError("invalid MemDBG LZ4 output length")
    source = memoryview(data)
    out = bytearray()
    pos = 0

    def extended_length(initial: int) -> int:
        nonlocal pos
        length = initial
        if initial == 15:
            while True:
                if pos >= len(source):
                    raise RuntimeError("truncated MemDBG LZ4 length")
                value = int(source[pos])
                pos += 1
                length += value
                if value != 255:
                    break
        return length

    while pos < len(source):
        token = int(source[pos])
        pos += 1
        literal_length = extended_length(token >> 4)
        if (literal_length > expected_length - len(out) or
                pos + literal_length > len(source)):
            raise RuntimeError("invalid MemDBG LZ4 literals")
        out.extend(source[pos:pos + literal_length])
        pos += literal_length
        if pos == len(source):
            break
        if pos + 2 > len(source):
            raise RuntimeError("truncated MemDBG LZ4 match offset")
        offset = int(source[pos]) | (int(source[pos + 1]) << 8)
        pos += 2
        if offset == 0 or offset > len(out):
            raise RuntimeError("invalid MemDBG LZ4 match offset")
        match_length = extended_length(token & 0x0F) + 4
        if match_length > expected_length - len(out):
            raise RuntimeError("MemDBG LZ4 output exceeds declared length")
        # LZ4 matches may overlap their own output.  Copy from a fixed start in
        # increasingly large slices; this preserves overlap semantics without
        # a byte-at-a-time Python loop for highly compressible game memory.
        match_start = len(out) - offset
        remaining = match_length
        while remaining:
            take = min(remaining, len(out) - match_start)
            if take <= 0:
                raise RuntimeError("invalid MemDBG LZ4 match")
            out.extend(out[match_start:match_start + take])
            remaining -= take
    if len(out) != expected_length:
        raise RuntimeError(
            f"MemDBG LZ4 length mismatch: {len(out)} != {expected_length}")
    return bytes(out)


_memdbg_maps_v2_lock = threading.Lock()
_memdbg_maps_v2_supported: dict = {}   # ip -> bool, learned once per host
_console_scan_lock = threading.Lock()
_console_scan_supported: dict = {}     # ip -> bool, learned once per host

# Same idea for TurboScan, and for the same reason. ps5_scan_exact_turbo runs
# ps5_auth_scanner then ps5_turboscan_caps, both over port 744 with
# ps5_connect's 15 s default. MemDBG *accepts* connections on 744 but never
# answers those commands, so the probe does not fail fast -- it times out.
# Nothing remembered that, so every scan of every kind re-probed and paid the
# timeout again: measured on a live PS5 (MemDBG 0.2.0-nightly.153), every
# first scan logged "TurboScan unavailable (timed out)" before starting real
# work. The console-scan path two branches below already learns this per host
# ("not retrying it on this console"); TurboScan simply never did.
_turbo_supported: dict = {}            # ip -> bool, learned once per host
_turbo_lock = threading.Lock()


def _turbo_worth_probing(ip: str) -> bool:
    """False once this host has shown it has no TurboScan at all."""
    with _turbo_lock:
        return _turbo_supported.get(ip) is not False


def _note_turbo_outcome(ip: str, ok: bool) -> None:
    """Record whether the TurboScan capability probe succeeded on this host.

    Only call this where the failure really is evidence about the payload.
    A resident-rescan failure is not: "no matching resident session" happens
    routinely on consoles whose TurboScan works perfectly.
    """
    with _turbo_lock:
        first = ip not in _turbo_supported
        _turbo_supported[ip] = bool(ok)
    if not ok and first:
        add_log("TurboScan is unavailable on this console; not probing for "
                "it again this session", "warn")


def _memdbg_unframe_memory(raw: bytes) -> bytes:
    """Decode MemDBG's command-local raw/LZ4 response frame (used for both
    native memory reads and, since it's the same generic framing, the
    PROCESS_MAPS_V2 map list)."""
    if not raw:
        raise RuntimeError("empty MemDBG memory frame")
    if raw[0] == 0:
        return raw[1:]
    if raw[0] == 1:
        if len(raw) < 5:
            raise RuntimeError("short MemDBG LZ4 frame")
        expected = struct.unpack_from("<I", raw, 1)[0]
        return _lz4_decompress_block(raw[5:], expected)
    raise RuntimeError(f"invalid MemDBG memory frame marker: {raw[0]}")


class _MemDBGClient:
    """Small dependency-free client for MemDBG's native control protocol.

    RDX keeps the adapter capability-gated: current payloads supply process
    metadata, framed native memory I/O and one-hop pointer seeds, while older
    builds fall back to their ps5debug-compatible listener.
    """
    _next_id = 1

    def __init__(self, ip: str, timeout: float = 5.0):
        self.ip = ip
        self.sock = None
        self.timeout = timeout
        self.hello = None

    def connect(self):
        info = socket.getaddrinfo(self.ip, MEMDBG_PORT, type=socket.SOCK_STREAM)
        last_exc = OSError("no MemDBG addresses")
        for family, _, _, _, sockaddr in info:
            s = socket.socket(family, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            try:
                s.connect(sockaddr)
                self.sock = s
                # Empty HELLO is part of the stable v1 contract and works with
                # both old and current MemDBG payloads.
                raw = self.request(MEMDBG_CMD_HELLO)
                if len(raw) < 44:
                    raise RuntimeError("short MemDBG HELLO")
                protocol, platform, caps, debug_port, udp_port = struct.unpack_from(
                    "<HHIHH", raw, 0)
                version = raw[12:28].split(b"\0", 1)[0].decode("utf-8", "replace")
                name = raw[28:44].split(b"\0", 1)[0].decode("utf-8", "replace")
                feature = struct.unpack_from("<H", raw, 44)[0] if len(raw) >= 46 else 1
                if len(raw) >= 112:
                    full_version = raw[64:112].split(b"\0", 1)[0].decode(
                        "utf-8", "replace")
                    if full_version:
                        version = full_version
                self.hello = {"protocol": protocol, "platform": platform,
                              "capabilities": caps, "debug_port": debug_port,
                              "udp_port": udp_port, "version": version,
                              "name": name, "feature_level": feature}
                return self
            except Exception as exc:
                last_exc = exc
                try: s.close()
                except Exception: pass
                self.sock = None
        raise last_exc

    def close(self):
        if self.sock is not None:
            try: self.sock.close()
            except Exception: pass
            self.sock = None

    def request(self, command: int, body: bytes = b"") -> bytes:
        if self.sock is None:
            raise ConnectionError("MemDBG client is not connected")
        request_id = _MemDBGClient._next_id & 0xFFFFFFFF
        _MemDBGClient._next_id = (_MemDBGClient._next_id + 1) & 0xFFFFFFFF
        header = struct.pack("<IHHII", MEMDBG_MAGIC, MEMDBG_VERSION,
                             int(command), request_id, len(body))
        self.sock.sendall(header + body)
        response = recv_exact(self.sock, 20)
        magic, version, echoed_cmd, echoed_id, status, length = struct.unpack(
            "<IHHIiI", response)
        if (magic != MEMDBG_MAGIC or version != MEMDBG_VERSION or
                echoed_cmd != int(command) or echoed_id != request_id):
            raise RuntimeError("invalid MemDBG response header")
        if length > 8 * 1024 * 1024:
            raise RuntimeError(f"oversized MemDBG response: {length}")
        payload = recv_exact(self.sock, length) if length else b""
        if status != 0:
            raise RuntimeError(f"MemDBG command 0x{command:04X} failed: {status}")
        return payload

    def process_list(self) -> list:
        raw = self.request(MEMDBG_CMD_PROCESS_LIST)
        if len(raw) < 4:
            raise RuntimeError("short MemDBG process list")
        count = struct.unpack_from("<I", raw, 0)[0]
        payload_len = len(raw) - 4
        stride = 56 if payload_len == count * 56 else (
            52 if payload_len == count * 52 else 0)
        if count > 65536 or not stride:
            raise RuntimeError("invalid MemDBG process list length")
        out = []
        for i in range(count):
            off = 4 + i * stride
            pid = struct.unpack_from("<i", raw, off)[0]
            name_off = off + (8 if stride == 56 else 4)
            name = raw[name_off:off + stride].split(b"\0", 1)[0].decode(
                "utf-8", "replace")
            out.append({"pid": pid, "name": name})
        return out

    @staticmethod
    def _parse_map_list_body(raw: bytes) -> list:
        """Parse the `uint32 count + memdbg_map_entry_t[]` body shared by
        PROCESS_MAPS and the unframed PROCESS_MAPS_V2 payload."""
        if len(raw) < 4:
            raise RuntimeError("short MemDBG map list")
        count = struct.unpack_from("<I", raw, 0)[0]
        expected = 4 + count * 88
        if count > 131072 or len(raw) != expected:
            raise RuntimeError("invalid MemDBG map list length")
        out = []
        for i in range(count):
            off = 4 + i * 88
            start, end, prot, flags = struct.unpack_from("<QQII", raw, off)
            name = raw[off + 24:off + 88].split(b"\0", 1)[0].decode(
                "utf-8", "replace")
            out.append({"start": start, "end": end, "prot": prot,
                        "flags": flags, "name": name})
        return out

    def process_maps(self, pid: int) -> list:
        """Map list, preferring the raw/LZ4-framed PROCESS_MAPS_V2 (smaller
        transfer on large map tables). Probed once per host: on the first
        failure for a given IP, V2 is remembered as unsupported so later
        calls (map fetches happen constantly — every scan, every pointer
        resolution, every export) don't keep retrying a doomed command.
        """
        with _memdbg_maps_v2_lock:
            try_v2 = _memdbg_maps_v2_supported.get(self.ip) is not False
        if try_v2:
            try:
                raw = _memdbg_unframe_memory(self.request(
                    MEMDBG_CMD_PROCESS_MAPS_V2, struct.pack("<i", int(pid))))
                maps = self._parse_map_list_body(raw)
            except Exception:
                with _memdbg_maps_v2_lock:
                    _memdbg_maps_v2_supported[self.ip] = False
            else:
                with _memdbg_maps_v2_lock:
                    _memdbg_maps_v2_supported[self.ip] = True
                return maps
        raw = self.request(MEMDBG_CMD_PROCESS_MAPS, struct.pack("<i", int(pid)))
        return self._parse_map_list_body(raw)

    def memory_write_multi(self, pid: int, entries: list) -> list:
        """Bulk write via MEMDBG_CMD_BATCH_WRITE.

        `entries` is [(address, data), ...], at most
        MEMDBG_BATCH_WRITE_MAX_ITEMS long. Returns a list of bool the same
        length as `entries` (True = that entry's write succeeded — both the
        server status and the reported byte count must confirm it).
        """
        caps = int((self.hello or {}).get("capabilities", 0))
        if not (caps & MEMDBG_CAP_BATCH_WRITE):
            raise RuntimeError("MemDBG does not advertise native batch writes")
        if not entries:
            return []
        if len(entries) > MEMDBG_BATCH_WRITE_MAX_ITEMS:
            raise ValueError(
                f"too many entries for one MemDBG batch write "
                f"({len(entries)} > {MEMDBG_BATCH_WRITE_MAX_ITEMS})")
        body = bytearray(struct.pack("<iII", int(pid), len(entries), 0))
        for addr, data in entries:
            if len(data) > MEMDBG_MAX_WRITE_DATA:
                raise ValueError(
                    f"entry at {hex(addr)} is {len(data)} bytes, over the "
                    f"{MEMDBG_MAX_WRITE_DATA}-byte per-entry cap")
            body += struct.pack("<QII", int(addr), len(data), 0)
            body += data
        raw = self.request(MEMDBG_CMD_BATCH_WRITE, bytes(body))
        if len(raw) != len(entries) * 16:
            raise RuntimeError("short MemDBG batch write response")
        results = []
        for i, (_addr, data) in enumerate(entries):
            _r_addr, written, status = struct.unpack_from("<QII", raw, i * 16)
            results.append(status == 0 and written == len(data))
        return results

    def memory_read(self, pid: int, address: int, length: int) -> bytes:
        """Read through native MemDBG, splitting at its public 1 MiB limit."""
        caps = int((self.hello or {}).get("capabilities", 0))
        if not (caps & MEMDBG_CAP_MEMORY_READ):
            raise RuntimeError("MemDBG does not advertise native memory reads")
        length = int(length)
        if length < 0:
            raise ValueError("negative memory read length")
        result = bytearray(length)
        pos = 0
        while pos < length:
            take = min(MEMDBG_MAX_MEMORY_READ, length - pos)
            body = struct.pack("<iQI", int(pid), int(address) + pos, take)
            chunk = _memdbg_unframe_memory(
                self.request(MEMDBG_CMD_MEMORY_READ, body))
            if len(chunk) != take:
                raise RuntimeError(
                    f"short MemDBG memory read: {len(chunk)} != {take}")
            result[pos:pos + take] = chunk
            pos += take
        return bytes(result)

    def memory_write(self, pid: int, address: int, data: bytes) -> bool:
        """Write through native MemDBG and require its exact byte count."""
        caps = int((self.hello or {}).get("capabilities", 0))
        if not (caps & MEMDBG_CAP_MEMORY_WRITE):
            raise RuntimeError("MemDBG does not advertise native memory writes")
        view = memoryview(data)
        pos = 0
        while pos < len(view):
            take = min(MEMDBG_MAX_WRITE_DATA, len(view) - pos)
            body = (struct.pack("<iQI", int(pid), int(address) + pos, take) +
                    bytes(view[pos:pos + take]))
            raw = self.request(MEMDBG_CMD_MEMORY_WRITE, body)
            if len(raw) != 4 or struct.unpack("<I", raw)[0] != take:
                raise RuntimeError("short MemDBG memory write")
            pos += take
        return True

    def pointer_holders(self, pid: int, target: int, regions: list,
                        max_results: int = 50000) -> list:
        """Return native one-hop exact holders, explicitly not full chains.

        NEEDS HARDWARE VERIFICATION: the trailing `8` below is passed to
        MemDBG's native SCAN_POINTER as an alignment/stride parameter, so
        this native seed only reports 8-byte-aligned holders even though
        RDX's own software scanner (_scan_pointer_hits) also checks 4-byte
        alignment. This has no confirmed console behavior either way -- no
        real MemDBG daemon has been available to test it -- so leave the
        `8` alone until it can be checked live. It is very unlikely to cause
        a missed pointer regardless: _resolve_permanent_candidates only
        trusts this fast path when it returns a verified candidate, and
        falls through to the alignment-complete pointer_chain_scan whenever
        it returns nothing (see _fast_direct_pointer_hits). The only
        theoretical gap is the rarer case where this narrower native scan
        finds and verifies a *different* holder before the fuller scan ever
        runs -- confirm on real hardware whether that ever produces a
        different (not wrong, just less optimal) permanent-pointer choice
        than the software path would have found on its own.
        """
        caps = int((self.hello or {}).get("capabilities", 0))
        if not (caps & MEMDBG_CAP_SCAN_POINTER):
            return []
        found = []
        for region in regions:
            start, end = int(region["start"]), int(region["end"])
            if end <= start:
                continue
            body = struct.pack("<iQQQIIII", int(pid), start, end - start,
                               int(target), 1, max(1, max_results - len(found)),
                               8, 0)
            raw = self.request(MEMDBG_CMD_SCAN_POINTER, body)
            if len(raw) < 40:
                raise RuntimeError("short MemDBG pointer response")
            count = struct.unpack_from("<I", raw, 0)[0]
            remaining = len(raw) - 40
            # Current MemDBG main declares 16-byte pointer-chain entries but
            # its daemon's generic result sender emits 8-byte address entries.
            # Accept both so RDX works before and after that upstream mismatch
            # is corrected.
            stride = 16 if remaining == count * 16 else (
                8 if remaining == count * 8 else 0)
            if not stride:
                raise RuntimeError("invalid MemDBG pointer response length")
            found.extend(struct.unpack_from("<Q", raw, 40 + i * stride)[0]
                         for i in range(count))
            if len(found) >= max_results:
                break
        return found[:max_results]


# ── one shared native connection ──────────────────────────────────────────────
# MemDBG's native listener accepts a small fixed number of connections and
# leaves closed ones lingering in FIN-WAIT-2, so the connection-per-call
# pattern these helpers used to follow exhausted it almost immediately.
# Measured against MemDBG 0.2.0-nightly.153 on a live PS5 running CUSA01659:
#
#     connect-read-close, repeated  ->  7 cycles, then "PS5 disconnected"
#                                       for ~60 s, at 0/200/500 ms pacing
#                                       alike, and with explicit shutdown()
#     one connection, 200 reads     ->  200/200 OK at 4.1 ms/read
#
# Pacing made no difference, so this is a count of live connections, not a
# rate limit, and the compatibility listener on 744 does not share it
# (40/40 connect-per-read cycles fine).  Once exhausted, every ps5_read spent
# its whole native retry budget -- three connect attempts plus 0.3 s of
# backoff -- before falling back to 744: 279 ms per read against 5 ms, a 53x
# penalty.  _note_memdbg_fallback reports that once per session, so after one
# log line the slowdown was silent; over a 4.18 GiB scan at 64 KiB per read it
# is about five hours of pure backoff.
#
# Sharing one connection keeps RDX inside the console's budget and makes the
# native path the fast path it was meant to be.  The scan reader already kept
# its own long-lived client, which is why scanning never hit this.
_memdbg_shared = None
_memdbg_shared_lock = threading.RLock()
_memdbg_native_failures = 0

# Consecutive whole-operation failures (every retry exhausted) after which the
# native path is abandoned for the session.  A payload that answers HELLO but
# cannot serve reads -- exactly what an exhausted listener looks like -- would
# otherwise be retried, at full retry cost, for every operation forever.
_MEMDBG_NATIVE_FAILURE_LIMIT = 3


def memdbg_native_ready() -> bool:
    """False once the native path has failed enough times to abandon it."""
    return _memdbg_native_failures < _MEMDBG_NATIVE_FAILURE_LIMIT


def _memdbg_note_native_outcome(ok: bool) -> None:
    """Record one whole-operation outcome and latch the path off on repeats."""
    global _memdbg_native_failures
    if ok:
        _memdbg_native_failures = 0
        return
    _memdbg_native_failures += 1
    if _memdbg_native_failures == _MEMDBG_NATIVE_FAILURE_LIMIT:
        add_log(f"MemDBG native I/O failed {_MEMDBG_NATIVE_FAILURE_LIMIT} times "
                "in a row; using the compatibility port for the rest of this "
                "session", "warn")


@contextlib.contextmanager
def memdbg_session(ip: str, timeout: float = 5.0):
    """Yield the process-wide native client, connecting or reconnecting.

    Serialised on purpose: one connection cannot carry two overlapping
    exchanges, and the console's connection budget is far too small to give
    every caller its own.  Reads cost about 4 ms, so the queue is cheap.
    """
    global _memdbg_shared
    with _memdbg_shared_lock:
        client = _memdbg_shared
        if client is not None and (client.sock is None or client.ip != ip):
            client.close()
            client = _memdbg_shared = None
        if client is None:
            client = _MemDBGClient(ip, timeout)
            client.connect()
            _memdbg_shared = client
        client.timeout = float(timeout)
        if client.sock is not None:
            try:
                client.sock.settimeout(float(timeout))
            except OSError:
                pass
        try:
            yield client
        except BaseException:
            # A failed exchange may have left unread bytes in the stream; the
            # next caller must not inherit a desynchronised connection.
            #
            # BaseException, not Exception: a cancelled scan, a Ctrl-C or a
            # closed generator abandons the exchange exactly as a socket error
            # does, and patch117 caught only Exception -- so an interrupt kept
            # the client cached with a half-read stream and handed it straight
            # to the next caller. Verified: after KeyboardInterrupt the shared
            # client survived with its socket open and was reused.
            client.close()
            _memdbg_shared = None
            raise


def memdbg_debugger_attached(ip: str) -> Optional[list]:
    """Thread list if a debug session is live, else None. Read-only, no attach.

    This is the cheap question that must precede the expensive one. Two
    attaches through ps5debug-NG on 2026-08-30 froze the game before anything
    established that its debugger answered at all; asking for threads costs one
    round trip and cannot stop a process.
    """
    try:
        with memdbg_session(ip) as client:
            raw = client.request(MEMDBG_CMD_DEBUG_GET_THREADS)
    except Exception:
        return None
    if len(raw) < 4:
        return None
    count = struct.unpack_from("<I", raw, 0)[0]
    if count > MAX_PROC_ENTRIES:
        return None
    return [count, len(raw)]


def memdbg_debug_set_watchpoint(ip: str, address: int, length: int,
                                wp_type: int = _MEMDBG_WP_WRITE) -> None:
    """Arm a hardware watchpoint. Raises on refusal."""
    if int(length) not in _MEMDBG_WP_LENGTHS:
        raise ValueError(f"watchpoint length must be 1, 2, 4 or 8, not {length}")
    body = struct.pack("<QII", int(address), int(length), int(wp_type))
    with memdbg_session(ip) as client:
        client.request(MEMDBG_CMD_DEBUG_SET_WATCHPOINT, body)


def memdbg_debug_clear_watchpoint(ip: str, address: int, length: int,
                                  wp_type: int = _MEMDBG_WP_WRITE) -> bool:
    """Disarm a watchpoint. Never raises: this runs on the cleanup path."""
    try:
        body = struct.pack("<QII", int(address), int(length), int(wp_type))
        with memdbg_session(ip) as client:
            client.request(MEMDBG_CMD_DEBUG_CLEAR_WATCHPOINT, body)
        return True
    except Exception as exc:
        add_log(f"Watchpoint at {int(address):#x} could not be cleared: {exc}",
                "warn")
        return False


def memdbg_debug_poll(ip: str) -> tuple:
    """(stopped, stop_lwp) for the active debug session."""
    with memdbg_session(ip) as client:
        raw = client.request(MEMDBG_CMD_DEBUG_POLL_EVENTS)
    if len(raw) < _MEMDBG_POLL_SIZE:
        raise RuntimeError(f"short debug poll response: {len(raw)} bytes")
    stopped, lwp = struct.unpack_from("<ii", raw, 0)
    return bool(stopped), int(lwp)


def memdbg_debug_rip(ip: str, pid: int, lwp: int) -> int:
    """Instruction pointer of a stopped thread.

    POLL_EVENTS reports only that the process stopped and on which thread; it
    carries no event record naming the trapping instruction. The instruction
    address therefore has to come from the register block, which makes
    GET_REGS required rather than optional for this workflow.
    """
    body = struct.pack("<ii", int(pid), int(lwp))
    with memdbg_session(ip) as client:
        raw = client.request(MEMDBG_CMD_DEBUG_GET_REGS, body)
    if len(raw) < _MEMDBG_REGS_SIZE:
        raise RuntimeError(f"short register block: {len(raw)} bytes")
    return int(struct.unpack_from("<q", raw, _MEMDBG_REGS_RIP_OFFSET)[0])


def trace_writer_instruction(ip: str, pid: int, address: int, width: int = 4,
                             timeout: float = 20.0,
                             poll_interval: float = 0.2,
                             cancel_event=None) -> dict:
    """Which instruction writes `address`, via a MemDBG write watchpoint.

    Requires a debug session to already exist. It will not attach: attaching is
    what froze the game twice on 2026-08-30, and keeping it out of this path
    isolates the new capability from that instability. If nothing is attached,
    this says so and does nothing.

    The watchpoint is always removed, including on timeout, cancellation and
    error. A watchpoint left armed on a live game is worse than no answer.

    Returns a dict with `stage`; `instruction` is present only when found.
    """
    result = {"stage": "no-debug-session", "instruction": None,
              "lwp": None, "target": int(address), "width": int(width)}
    if int(width) not in _MEMDBG_WP_LENGTHS:
        result["stage"] = "bad-width"
        return result
    if memdbg_debugger_attached(ip) is None:
        result["note"] = ("no active MemDBG debug session; attach before "
                          "tracing (RDX will not attach for you)")
        return result

    armed = False
    try:
        memdbg_debug_set_watchpoint(ip, address, width, _MEMDBG_WP_WRITE)
        armed = True
    except Exception as exc:
        result["stage"] = "watchpoint-refused"
        result["note"] = str(exc)
        return result

    try:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                result["stage"] = "cancelled"
                return result
            try:
                stopped, lwp = memdbg_debug_poll(ip)
            except Exception as exc:
                result["stage"] = "poll-failed"
                result["note"] = str(exc)
                return result
            if stopped:
                result["lwp"] = lwp
                try:
                    result["instruction"] = memdbg_debug_rip(ip, pid, lwp)
                    result["stage"] = "found"
                except Exception as exc:
                    result["stage"] = "regs-unreadable"
                    result["note"] = str(exc)
                return result
            time.sleep(max(float(poll_interval), 0.01))
        result["stage"] = "no-write-observed"
        result["note"] = (f"nothing wrote {int(address):#x} within "
                          f"{timeout:.0f}s")
        return result
    finally:
        if armed:
            memdbg_debug_clear_watchpoint(ip, address, width, _MEMDBG_WP_WRITE)


def memdbg_reset_session() -> None:
    """Drop the shared connection and re-arm the native path.

    Called by Reconnect: after a payload restart the old connection is dead
    and the failure latch, if it tripped, describes a console that no longer
    exists.
    """
    global _memdbg_shared, _memdbg_native_failures
    with _memdbg_shared_lock:
        if _memdbg_shared is not None:
            _memdbg_shared.close()
            _memdbg_shared = None
        _memdbg_native_failures = 0


def memdbg_probe(ip: str, timeout: float = 1.5):
    """Return native MemDBG HELLO information, or None when unavailable.

    Borrows the shared session, so the connection opened to identify the
    payload is the same one the first read uses rather than a spent slot.
    """
    try:
        with memdbg_session(ip, timeout) as client:
            return dict(client.hello or {})
    except Exception:
        return None

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

def ps5_scan_exact_server(ip: str, pid: int, value, width: int,
                          regions: list, aligned: bool = True,
                          cancel_event=None,
                          progress_cb=None,
                          value_type: Optional[str] = None) -> np.ndarray:
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

    type_key = _normalise_value_type(value_type, width)
    wire_type = SCAN_VALUE_TYPE_ID[type_key]
    target = _pack_typed_value(value, type_key, width)
    body = struct.pack("<IBBI", pid, wire_type, 0, len(target))
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


def _turbo_fetch_addresses_and_values(s: socket.socket, width: int, count: int,
                                      value_type: str,
                                      cancel_event=None) -> tuple:
    """Like _turbo_fetch_addresses, but also decodes each record's `current`
    value field — used by the snapshot (unknown-value) path, which needs
    values for display and as the next narrow's client-side baseline,
    unlike the exact-match path where the value is already known."""
    wanted = min(int(count), MAX_SCAN_RESULTS)
    value_dtype = np.dtype(VALUE_TYPES[value_type]["dtype"])
    out_addr = np.empty(wanted, dtype=_NP_ADDR_DTYPE)
    out_val = np.empty(wanted, dtype=value_dtype)
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
            "names": ["addr", "current", "rest"],
            "formats": ["<u8", value_dtype, f"V{rec_size - 8 - width}"],
            "offsets": [0, 8, 8 + width], "itemsize": rec_size}, buffer=raw)
        out_addr[pos:pos + actual] = records["addr"]
        out_val[pos:pos + actual] = records["current"]
        pos += actual
    return out_addr[:pos], out_val[:pos]


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


def ps5_scan_exact_turbo(ip: str, pid: int, value, width: int,
                         regions: list, aligned: bool = True,
                         cancel_event=None, progress_cb=None,
                         value_type: Optional[str] = None) -> np.ndarray:
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

    type_key = _normalise_value_type(value_type, width)
    target = _pack_typed_value(value, type_key, width)
    wire_type = SCAN_VALUE_TYPE_ID[type_key]
    flags = 0x02 | 0x10  # server resident + segmented
    if (engines & 0x02) and os.environ.get("RDX_TURBO_ALIAS", "1") != "0":
        flags |= 0x01
        if engines & 0x100:
            flags |= 0x80
    # scan_turbo.c uses `alignment ? alignment : value_length`; therefore 0
    # means width-aligned.  A byte step of 1 is the true unaligned mode.
    alignment = width if aligned else 1
    body = struct.pack("<IQIBBBII", pid, 0, 0, wire_type, 0,
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
                              "engines": engines, "value_type": type_key,
                              "mode": "exact"}
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


def ps5_scan_next_turbo(ip: str, pid: int, value, width: int,
                         cancel_event=None, progress_cb=None,
                         value_type: Optional[str] = None) -> np.ndarray:
    """Refine the complete resident result set on-console using COUNT."""
    global _turbo_session
    with _turbo_session_lock:
        session = _turbo_session
        type_key = _normalise_value_type(value_type, width)
        if not session or any((session["ip"] != ip, session["pid"] != pid,
                               session["width"] != width,
                               session.get("value_type", type_key) != type_key,
                               session.get("mode", "exact") != "exact")):
            raise RuntimeError("no matching resident TurboScan session")
        s = session["socket"]
        old_count = int(session["count"])
        target = _pack_typed_value(value, type_key, width)
        wire_type = SCAN_VALUE_TYPE_ID[type_key]
        flags = 0x02  # TS_SERVER_RESIDENT
        if session["engines"] & 0x200:
            flags |= 0x100  # TS_RESCAN_ALIASING
        body = struct.pack("<IQBBII", pid, 0, wire_type, 0,
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


# Protocol 2.2: the snapshot value store is RAM-backed under this threshold
# (server default, tunable by the client via CC15 which RDX does not use); the
# overflow spills to a file under the console's /data.
_TS_SNAPSHOT_RAM_THRESHOLD = 512 * 1024 * 1024
# Above this much spill, say so as a warning rather than an informational line.
_TS_SNAPSHOT_SPILL_WARN = 1024 * 1024 * 1024


# Wire compareType codes for CC12's relational narrow of a resident snapshot
# session (protocol.md 7.2, cmpType 0-12). Modes without "by" compare against
# the session's own tracked prior value server-side and need no operand.
_SNAPSHOT_COMPARE_TYPE = {
    "increased":    5,   # IncreasedValue    — no operand
    "increased by": 6,   # IncreasedValueBy  — operand = delta
    "decreased":    7,   # DecreasedValue    — no operand
    "decreased by": 8,   # DecreasedValueBy  — operand = delta
    "changed":      9,   # ChangedValue      — no operand
    "unchanged":    10,  # UnchangedValue    — no operand
}


def ps5_scan_unknown_turbo(ip: str, pid: int, width: int,
                           regions: list, aligned: bool = True,
                           cancel_event=None, progress_cb=None,
                           value_type: Optional[str] = None) -> tuple:
    """Unknown-initial-value First Scan via CC11 + TS_SNAPSHOT: the console
    holds the baseline itself (RAM/disk-hybrid, per-connection) instead of
    RDX transferring and holding every candidate's raw bytes locally.
    Retains its server-resident snapshot session for narrowing via
    ps5_scan_relational_turbo, exactly as ps5_scan_exact_turbo retains its
    own session for ps5_scan_next_turbo. Returns (addrs, values) matching
    scan_first_unknown()'s contract exactly, so callers can try this first
    and fall back to that unchanged.

    TS_SNAPSHOT_INCLUDE_ZEROS is always set: scan_first_unknown's client-side
    path records every aligned address regardless of value, but the server's
    snapshot default *drops* all-zero slots — this flag is required for
    behavioral parity with the existing path, not an optional tuning knob.
    """
    global _turbo_session
    _close_turbo_session()
    ps5_auth_scanner(ip)
    version, engines, _ = ps5_turboscan_caps(ip)
    required = 0x08 | 0x10  # TSE_SNAPSHOT, TSE_SNAPSHOT_SEGMENTS
    if version < 1 or (engines & required) != required:
        raise RuntimeError("required TurboScan snapshot engines are unavailable")

    type_key = _normalise_value_type(value_type, width)
    value_dtype = np.dtype(VALUE_TYPES[type_key]["dtype"])
    empty = (np.empty(0, dtype=_NP_ADDR_DTYPE), np.empty(0, dtype=value_dtype))

    # Segment merge + alignment: identical to ps5_scan_exact_turbo's, so a
    # snapshot scan covers exactly the same slots an exact-value turbo scan
    # over the same regions/alignment would.
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
                max_piece -= max_piece % width
            piece_end = min(end, start + max_piece)
            merged.append((start, piece_end))
            start = piece_end
    if not merged:
        return empty
    if len(merged) > 1_048_576:
        raise RuntimeError("too many TurboScan segments")

    wire_type = SCAN_VALUE_TYPE_ID[type_key]
    flags = 0x04 | 0x08 | 0x10  # TS_SNAPSHOT | TS_SNAPSHOT_INCLUDE_ZEROS | TS_SNAPSHOT_SEGMENTS

    # The snapshot's value store lives on the CONSOLE. Protocol 2.2: it is
    # RAM-backed under a 512 MiB threshold and the overflow spills to
    # ps5dbg_snap_NN.bin under /data. With INCLUDE_ZEROS every aligned slot is
    # seeded, so a whole-heap unknown scan is far larger than the bytes read:
    # 2.15 GiB at width 4 is ~577 M slots and roughly 4.3 GiB of store, i.e.
    # ~3.8 GiB written to the console's /data. That is self-limiting (unlinked
    # on END/disconnect, /data swept at startup) and the server declines
    # cleanly with snapshot_ok = 0 if it cannot allocate — but it should never
    # be invisible. Say the number before committing to it.
    alignment = width if aligned else 1
    _slots = sum((end - start) // max(alignment, 1) for start, end in merged)
    _store_bytes = _slots * width * 2          # current + previous stores
    _spill_bytes = max(0, _store_bytes - _TS_SNAPSHOT_RAM_THRESHOLD)
    if _spill_bytes:
        add_log(
            f"Snapshot scan will build a ~{_store_bytes / 1073741824:.2f} GiB "
            f"value store on the console ({_slots:,} slots); about "
            f"{_spill_bytes / 1073741824:.2f} GiB spills to its /data "
            f"partition until the session ends.",
            "warn" if _spill_bytes >= _TS_SNAPSHOT_SPILL_WARN else "info")
    # compareType 11 = UnknownInitialValue; lenData 0 — TS_SNAPSHOT needs no seed.
    body = struct.pack("<IQIBBBII", pid, 0, 0, wire_type, 11,
                       alignment, 0, flags)
    total_bytes = sum(end - start for start, end in merged)
    if progress_cb:
        progress_cb(0, max(total_bytes, 1))

    s = ps5_connect(ip)
    session_created = False
    retain_session = False
    try:
        s.sendall(cmd_header(CMD_TURBO_START, len(body)) + body)
        if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
            raise RuntimeError("TurboScan snapshot start rejected")
        # TS_SNAPSHOT needs no seed value, but the wire still has two leading
        # acks (protocol.md: "after the two leading CMD_SUCCESS acks... reads
        # the trailing segment list") — sending zero bytes here mirrors
        # ps5_scan_exact_turbo's ack1-then-value-then-ack2 shape with an
        # empty value phase, matching the exact-segmented case's documented
        # "after the two acks, reads the segment list" ordering exactly.
        s.sendall(b"")
        if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
            raise RuntimeError("TurboScan snapshot value phase rejected")

        segment_data = bytearray(struct.pack("<I", len(merged)))
        for start, end in merged:
            segment_data.extend(struct.pack("<QI", start, end - start))
        s.sendall(segment_data)
        # No ack after the segment list — straight into the plan/progress/
        # summary stream, exactly like the exact-segmented case goes
        # straight into its 12-byte summary with no ack in between.

        # Heartbeat: advance progress 0→90% while blocking on the server scan,
        # same time-based estimate ps5_scan_exact_turbo uses for the same
        # reason (segmented scans give no fine-grained per-byte feedback
        # here either — the real progress records arrive after this phase).
        _hb_stop = threading.Event()
        if progress_cb and total_bytes > 0:
            def _heartbeat():
                _estimated_secs = max(total_bytes / (500 * 1024 * 1024), 0.5)
                _step = max(int(total_bytes * 0.01), 1)
                _interval = _estimated_secs / 99
                _done = 0
                _cap = int(total_bytes * 0.99)
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
            _slot_count, _plan_total_bytes = struct.unpack(
                "<QQ", _recv_exact_cancel(s, 16, cancel_event))
            # The server's own plan, which RDX previously read and discarded.
            # Log it: it is the authoritative slot count and the only way to
            # see that the estimate above matched what the console actually
            # committed to.
            add_log(f"Snapshot plan: {_slot_count:,} slots over "
                    f"{_plan_total_bytes / 1073741824:.2f} GiB "
                    f"(estimated {_slots:,} slots)")
            while True:
                bytes_done = struct.unpack(
                    "<Q", _recv_exact_cancel(s, 8, cancel_event))[0]
                if bytes_done == 0xFFFFFFFFFFFFFFFF:
                    break
                if progress_cb:
                    progress_cb(min(bytes_done, total_bytes), max(total_bytes, 1))
            snapshot_ok, survivor_count = struct.unpack(
                "<IQ", _recv_exact_cancel(s, 12, cancel_event))
        finally:
            _hb_stop.set()
            if _hb_thread is not None:
                _hb_thread.join(timeout=0.5)

        if not snapshot_ok:
            raise RuntimeError("TurboScan snapshot storage exceeded capacity")
        session_created = True
        if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
            raise RuntimeError("TurboScan snapshot did not complete")

        addrs, vals = _turbo_fetch_addresses_and_values(
            s, width, survivor_count, type_key, cancel_event)

        if survivor_count > MAX_SCAN_RESULTS and cancel_event:
            cancel_event.truncated = True
        if progress_cb:
            progress_cb(max(total_bytes, 1), max(total_bytes, 1))
        add_log(f"TurboScan snapshot: {len(addrs):,}/{survivor_count:,} "
                f"candidates, {total_bytes / 1_073_741_824:.2f} GiB scanned")
        with _turbo_session_lock:
            _turbo_session = {"socket": s, "ip": ip, "pid": pid,
                              "width": width, "count": int(survivor_count),
                              "engines": engines, "value_type": type_key,
                              "mode": "snapshot"}
        retain_session = True
        return addrs, vals
    finally:
        if not retain_session:
            if session_created:
                try:
                    s.sendall(cmd_header(CMD_TURBO_END))
                    recv_exact(s, 4)
                except Exception:
                    pass
            s.close()


def ps5_scan_relational_turbo(ip: str, pid: int, width: int,
                              mode_lbl: str, delta,
                              cancel_event=None, progress_cb=None,
                              value_type: Optional[str] = None) -> tuple:
    """Narrow a resident snapshot session (see ps5_scan_unknown_turbo) via
    CC12 + TS_SERVER_RESIDENT with a relational compareType, instead of
    scan_next_relational's client-driven re-read-and-compare over every
    candidate. Requires a matching snapshot-mode session; raises otherwise
    so the caller can fall back to that unchanged. Returns (addrs, values)
    matching scan_next_relational()'s contract.
    """
    global _turbo_session
    compare_type = _SNAPSHOT_COMPARE_TYPE.get(str(mode_lbl))
    if compare_type is None:
        raise ValueError(f"unsupported relational mode for TurboScan: {mode_lbl!r}")
    needs_operand = mode_lbl in ("increased by", "decreased by")

    with _turbo_session_lock:
        session = _turbo_session
        type_key = _normalise_value_type(value_type, width)
        if not session or any((session["ip"] != ip, session["pid"] != pid,
                               session["width"] != width,
                               session.get("value_type", type_key) != type_key,
                               session.get("mode") != "snapshot")):
            raise RuntimeError("no matching resident TurboScan snapshot session")
        s = session["socket"]
        old_count = int(session["count"])
        wire_type = SCAN_VALUE_TYPE_ID[type_key]
        operand = _pack_typed_value(delta, type_key, width) if needs_operand else b""
        flags = 0x02  # TS_SERVER_RESIDENT
        if session["engines"] & 0x200:
            flags |= 0x100  # TS_RESCAN_ALIASING
        body = struct.pack("<IQBBII", pid, 0, wire_type, compare_type,
                           len(operand), flags)
        try:
            if progress_cb:
                progress_cb(0, max(old_count, 1))
            s.sendall(cmd_header(CMD_TURBO_COUNT, len(body)) + body)
            if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
                raise RuntimeError("TurboScan snapshot rescan rejected")
            s.sendall(operand)

            # COUNT has a single ack (above), unlike START's two — matches
            # ps5_scan_next_turbo's proven wire sequence exactly: send
            # header+body, one ack, send the operand (here: none for
            # operand-free relational modes), then straight into the
            # progress stream.
            while True:
                scanned = struct.unpack(
                    "<Q", _recv_exact_cancel(s, 8, cancel_event))[0]
                if scanned == 0xFFFFFFFFFFFFFFFF:
                    break
                if progress_cb:
                    progress_cb(min(int(scanned), old_count), max(old_count, 1))
            new_count = struct.unpack("<Q", _recv_exact_cancel(s, 8, cancel_event))[0]
            if struct.unpack("<I", _recv_exact_cancel(s, 4, cancel_event))[0] != STATUS_SUCCESS:
                raise RuntimeError("TurboScan snapshot rescan failed")
            session["count"] = int(new_count)
            addrs, vals = _turbo_fetch_addresses_and_values(
                s, width, new_count, type_key, cancel_event)
            if cancel_event is not None:
                cancel_event.truncated = new_count > MAX_SCAN_RESULTS
            if progress_cb:
                progress_cb(max(old_count, 1), max(old_count, 1))
            add_log(f"Turbo snapshot next scan ({mode_lbl}): "
                    f"{len(addrs):,}/{new_count:,} remain")
            return addrs, vals
        except Exception:
            _turbo_session = None
            s.close()
            raise

# All helpers use sendall() and try/finally so the socket is always closed.

def ps5_proc_list(ip: str) -> list:
    if state.get("backend") == "memdbg-experimental" and memdbg_native_ready():
        try:
            with memdbg_session(ip) as client:
                procs = client.process_list()
            _memdbg_note_native_outcome(True)
            return procs
        except Exception as exc:
            _memdbg_note_native_outcome(False)
            add_log(f"MemDBG process list failed; using compatibility port: {exc}",
                    "warn")
    s = ps5_connect(ip)
    try:
        s.sendall(cmd_header(CMD_PROC_LIST))
        if not check_ok(s):
            raise RuntimeError("proc list command rejected")
        count = _checked_entry_count(recv_exact(s, 4), MAX_PROC_ENTRIES,
                                     "process list")
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
    if state.get("backend") == "memdbg-experimental" and memdbg_native_ready():
        try:
            with memdbg_session(ip) as client:
                entries = client.process_maps(pid)
            _memdbg_note_native_outcome(True)
            return entries
        except Exception as exc:
            _memdbg_note_native_outcome(False)
            add_log(f"MemDBG maps failed; using compatibility port: {exc}", "warn")
    s = ps5_connect(ip)
    try:
        body = struct.pack("<I", pid)
        s.sendall(cmd_header(CMD_PROC_MAPS, len(body)) + body)
        if not check_ok(s):
            raise RuntimeError("proc maps command rejected")
        count = _checked_entry_count(recv_exact(s, 4), MAX_MAP_ENTRIES,
                                     "memory map")
        maps = []
        for _ in range(count):
            raw   = recv_exact(s, MAP_ENTRY_SIZE)
            name  = raw[:32].split(b'\x00', 1)[0].decode('utf-8', errors='replace')
            start = struct.unpack_from("<Q", raw, 32)[0]
            end   = struct.unpack_from("<Q", raw, 40)[0]
            offset = struct.unpack_from("<Q", raw, 48)[0]
            prot  = struct.unpack_from("<H", raw, 56)[0]
            maps.append({"start": start, "end": end, "prot": prot,
                         "offset": offset, "name": name})
        return maps
    finally:
        s.close()

_UI_MAX_RETRIES = 3   # retries for individual ps5_read / ps5_write UI calls
_memdbg_fallback_notes = set()
_memdbg_fallback_lock = threading.Lock()


# ── target abstraction ────────────────────────────────────────────────────────
# RDX speaks to two payloads: ps5debug-NG on TCP 744 and MemDBG on 9020. That
# choice was made ad hoc at each call site -- _memdbg_has(CAP) guarding a
# native attempt, then a fall-through to port 744 -- which works, but leaves no
# single seam a backend implements, and therefore nothing a test can stand in
# for. That matters here specifically: HARDWARE_TEST_CHECKLIST lists "the
# MemDBG backend, in its entirety" as never having run against a real daemon,
# and the only way to exercise it today is to fake sockets underneath it.
#
# Squalr isolates this as squalr-engine-targets with swappable native
# implementations; MemoryEngine360 keeps Playstation and Xbox platform folders
# under one core with the connection chosen at File > Connect.
#
# The wire functions below stay exactly as they are and remain the code path
# the app uses -- this is a seam over them, not a rewrite of them. What it buys
# is a single place that answers "which backend, and what can it do", and a
# MockTarget that lets the MemDBG capability matrix be tested without a daemon.
def _target_checked_write(ip: str, pid: int, addr: int, data: bytes) -> bool:
    """Map-validated write, used by every Target implementation.

    The wire helper `ps5_write` performs no validation of its own -- every
    caller in the app validates first, which is how the README's "validates
    addresses against the process map before every write" holds. `Target` is
    a seam that *looks* like the backend write API, so a future migration of
    call sites onto it would have silently dropped that property. Validating
    here makes the seam safe by default instead of safe by convention.
    """
    error = _validate_write_addr(int(addr))
    if error:
        add_log(f"Target write refused at {hex(int(addr))}: {error}", "error")
        return False
    error = _validate_addr_in_maps(ip, int(pid), int(addr), len(data))
    if error:
        add_log(f"Target write refused at {hex(int(addr))}: {error}", "error")
        return False
    return ps5_write(ip, int(pid), int(addr), data)


class Target:
    """One console transport: capability reporting plus process/memory I/O."""

    name = "target"
    port = 0

    # Capabilities every target is expected to answer about. These are the
    # operations RDX degrades gracefully without, so a target that lacks one
    # must say so rather than raise at the point of use.
    CAP_PROCESSES = "processes"
    CAP_MAPS = "maps"
    CAP_READ = "read"
    CAP_WRITE = "write"
    CAP_WRITE_MULTI = "write_multi"
    CAP_TURBO = "turbo"
    CAP_REGION_CLASSIFY = "region_classify"

    ALL_CAPS = (CAP_PROCESSES, CAP_MAPS, CAP_READ, CAP_WRITE,
                CAP_WRITE_MULTI, CAP_TURBO, CAP_REGION_CLASSIFY)

    def __init__(self, ip: str):
        self.ip = str(ip)

    def capabilities(self) -> frozenset:
        raise NotImplementedError

    def has(self, capability: str) -> bool:
        return capability in self.capabilities()

    def processes(self) -> list:
        raise NotImplementedError

    def maps(self, pid: int) -> list:
        raise NotImplementedError

    def read(self, pid: int, addr: int, length: int) -> bytes:
        raise NotImplementedError

    def write(self, pid: int, addr: int, data: bytes) -> bool:
        raise NotImplementedError

    def describe(self) -> str:
        missing = [c for c in self.ALL_CAPS if not self.has(c)]
        if not missing:
            return f"{self.name}: all capabilities"
        return f"{self.name}: missing {', '.join(missing)}"


class Ps5DebugTarget(Target):
    """ps5debug-NG on TCP 744. The reference transport."""

    name = "ps5debug"
    port = PS5_PORT

    def capabilities(self) -> frozenset:
        # ps5debug-NG implements the whole surface; TurboScan and the region
        # classifier are probed per-console elsewhere and degrade on their own.
        return frozenset(self.ALL_CAPS)

    def processes(self) -> list:
        return ps5_proc_list(self.ip)

    def maps(self, pid: int) -> list:
        return ps5_maps(self.ip, pid)

    def read(self, pid: int, addr: int, length: int) -> bytes:
        return ps5_read(self.ip, pid, addr, length)

    def write(self, pid: int, addr: int, data: bytes) -> bool:
        return _target_checked_write(self.ip, pid, addr, data)


class MemDbgTarget(Target):
    """MemDBG on TCP 9020, gated by the capability bitmap it advertises.

    TurboScan and the region classifier are ps5debug commands with no MemDBG
    equivalent, so this target reports them missing and RDX falls back to the
    slow host scan path -- which is why the connect screen warns when port 744
    is unreachable even though MemDBG alone is enough to read and write.
    """

    name = "memdbg"
    port = MEMDBG_PORT

    _CAP_BITS = {
        Target.CAP_PROCESSES: MEMDBG_CAP_PROCESS_LIST,
        Target.CAP_MAPS: MEMDBG_CAP_PROCESS_MAPS,
        Target.CAP_READ: MEMDBG_CAP_MEMORY_READ,
        Target.CAP_WRITE: MEMDBG_CAP_MEMORY_WRITE,
        Target.CAP_WRITE_MULTI: MEMDBG_CAP_BATCH_WRITE,
    }

    def __init__(self, ip: str, hello: Optional[dict] = None):
        super().__init__(ip)
        self.hello = hello if hello is not None else (state.get("memdbg") or {})

    def capabilities(self) -> frozenset:
        bits = int(self.hello.get("capabilities", 0) or 0)
        return frozenset(cap for cap, bit in self._CAP_BITS.items()
                         if bits & int(bit))

    def processes(self) -> list:
        return ps5_proc_list(self.ip)

    def maps(self, pid: int) -> list:
        return ps5_maps(self.ip, pid)

    def read(self, pid: int, addr: int, length: int) -> bytes:
        return ps5_read(self.ip, pid, addr, length)

    def write(self, pid: int, addr: int, data: bytes) -> bool:
        return _target_checked_write(self.ip, pid, addr, data)


def current_target(ip: Optional[str] = None) -> Target:
    """The target for the active backend."""
    endpoint = state.get("ip", "") if ip is None else ip
    if state.get("backend") == "memdbg-experimental":
        return MemDbgTarget(endpoint)
    return Ps5DebugTarget(endpoint)


def _memdbg_has(capability: int) -> bool:
    """Whether the native path both advertises `capability` and still works.

    The advertised bitmap is cached from the HELLO taken at connect time, so
    on its own it keeps claiming a capability long after the listener stopped
    serving it -- and one real payload advertises 0xFFFFFFFF, every bit set,
    which makes the bitmap alone worthless as a health signal.
    """
    return (state.get("backend") == "memdbg-experimental" and
            memdbg_native_ready() and
            bool(int((state.get("memdbg") or {}).get("capabilities", 0)) &
                 int(capability)))


# A scan opens more connections than MemDBG's native listener serves, so it
# overflows to the compatibility listener by design. Measured: MemDBG accepts
# exactly 6 concurrent native connections and refuses the 7th, with the
# existing 6 unaffected; RDX's budget is 10.
#
# That overflow is not degradation. A/B on the console, same 4,280.8 MiB scan:
#
#     budget 10   1 overflow to port 744   168.5 s   25.4 MiB/s
#     budget  5   0 overflows              213.1 s   20.1 MiB/s
#
# Constraining the workers to avoid it was written, tested and reverted: it
# worked and cost 26% of throughput. Port 744 is fast, and the valve is doing
# its job.
#
# The message, though, said "failed" at `warn` level, and this session's own
# notes recorded it three times as a cost before the measurement showed it was
# not one. A scan overflowing is routine and is logged as such; a one-off read
# or write falling back is not routine and still warrants a warning.
_MEMDBG_ROUTINE_FALLBACKS = frozenset({"scan read"})


def _note_memdbg_fallback(operation: str, exc: Exception) -> None:
    """Log one native-to-compatibility fallback per operation and session."""
    key = (int(state.get("session", 0)), str(operation))
    with _memdbg_fallback_lock:
        if key in _memdbg_fallback_notes:
            return
        _memdbg_fallback_notes.add(key)
    if str(operation) in _MEMDBG_ROUTINE_FALLBACKS:
        add_log(f"MemDBG native {operation}: connection budget reached, "
                f"using port 744 for the overflow (expected; the compatibility "
                f"listener is not slower)")
    else:
        add_log(f"MemDBG native {operation} failed; trying port 744: {exc}",
                "warn")

def ps5_read(ip: str, pid: int, addr: int, length: int) -> bytes:
    """Read with up to _UI_MAX_RETRIES retries on transient connection failures."""
    last_exc: Exception = RuntimeError("no attempts")
    if _memdbg_has(MEMDBG_CAP_MEMORY_READ):
        for attempt in range(_UI_MAX_RETRIES):
            try:
                with memdbg_session(ip) as client:
                    data = client.memory_read(pid, addr, length)
                _memdbg_note_native_outcome(True)
                return data
            except Exception as exc:
                last_exc = exc
                if attempt < _UI_MAX_RETRIES - 1:
                    time.sleep(0.1 * (attempt + 1))
        _memdbg_note_native_outcome(False)
        _note_memdbg_fallback("read", last_exc)
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
    native_exc: Exception = RuntimeError("no attempts")
    if _memdbg_has(MEMDBG_CAP_MEMORY_WRITE):
        for attempt in range(_UI_MAX_RETRIES):
            if cancel_event and cancel_event.is_set():
                return False
            try:
                with memdbg_session(ip, timeout=timeout) as client:
                    written = client.memory_write(pid, addr, data)
                _memdbg_note_native_outcome(True)
                return written
            except Exception as exc:
                native_exc = exc
                if attempt < _UI_MAX_RETRIES - 1:
                    delay = 0.1 * (attempt + 1)
                    if cancel_event:
                        if cancel_event.wait(delay):
                            return False
                    else:
                        time.sleep(delay)
        _memdbg_note_native_outcome(False)
        _note_memdbg_fallback("write", native_exc)
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


def ps5_write_multi(ip: str, pid: int, entries: list,
                    cancel_event: Optional[threading.Event] = None,
                    timeout: float = 15.0) -> list:
    """Bulk write via 0xBDAACC04 — collapses N single CMD_PROC_WRITEs (e.g.
    one freeze tick over several cheats) into one exchange.

    `entries` is [(address: int, data: bytes), ...]. Returns a list of bool
    the same length as `entries` (True = that entry's write succeeded).
    Server-side application is best-effort and non-atomic — one entry
    failing does not stop the rest — so a per-entry status array is always
    requested (PROC_WRITE_MULTI_F_STATUS) rather than only getting a single
    all-or-nothing result.
    """
    if not entries:
        return []
    if len(entries) > PROC_WRITE_MULTI_MAX_COUNT:
        raise ValueError(
            f"too many entries for one bulk write "
            f"({len(entries)} > {PROC_WRITE_MULTI_MAX_COUNT})")
    for addr, data in entries:
        if len(data) > PROC_WRITE_MULTI_MAX_LEN:
            raise ValueError(
                f"entry at {hex(addr)} is {len(data)} bytes, over the "
                f"{PROC_WRITE_MULTI_MAX_LEN}-byte per-entry cap")

    payload = bytearray()
    for addr, data in entries:
        payload += struct.pack("<QI", addr, len(data))
        payload += data
    body = struct.pack("<III", pid, len(entries), PROC_WRITE_MULTI_F_STATUS)

    for attempt in range(_UI_MAX_RETRIES):
        if cancel_event and cancel_event.is_set():
            return [False] * len(entries)
        s = None
        try:
            s = ps5_connect(ip, timeout=timeout)
            s.sendall(cmd_header(CMD_PROC_WRITE_MULTI, len(body)) + body)
            if not check_ok(s):
                return [False] * len(entries)
            s.sendall(bytes(payload))
            status = recv_exact(s, len(entries))
            if not check_ok(s):
                return [False] * len(entries)
            return [b == 0 for b in status]
        except Exception:
            if attempt < _UI_MAX_RETRIES - 1:
                delay = 0.1 * (attempt + 1)
                if cancel_event:
                    if cancel_event.wait(delay):
                        return [False] * len(entries)
                else:
                    time.sleep(delay)
        finally:
            if s:
                try: s.close()
                except Exception: pass
    return [False] * len(entries)


def memdbg_write_multi(ip: str, pid: int, entries: list,
                       cancel_event: Optional[threading.Event] = None,
                       timeout: float = 5.0) -> list:
    """Retry-wrapping free function around _MemDBGClient.memory_write_multi,
    mirroring ps5_write's MemDBG-native retry loop. Raises on total failure
    (every attempt exhausted) so the caller can fall back to per-write."""
    if not entries:
        return []
    native_exc: Exception = RuntimeError("no attempts")
    for attempt in range(_UI_MAX_RETRIES):
        if cancel_event and cancel_event.is_set():
            return [False] * len(entries)
        try:
            with memdbg_session(ip, timeout=timeout) as client:
                # One BATCH_WRITE exchange is capped at
                # MEMDBG_BATCH_WRITE_MAX_ITEMS entries.  Split a larger freeze
                # tick across several exchanges on the same connection: passing
                # the whole list straight through raised ValueError, which this
                # loop then treated as a transient network fault, so a user with
                # 65+ simultaneous freezes got nothing written at all, every
                # tick, with no fall-through to the per-write path.
                results = []
                for start in range(0, len(entries), MEMDBG_BATCH_WRITE_MAX_ITEMS):
                    results.extend(client.memory_write_multi(
                        pid, entries[start:start + MEMDBG_BATCH_WRITE_MAX_ITEMS]))
            _memdbg_note_native_outcome(True)
            return results
        except Exception as exc:
            native_exc = exc
            if attempt < _UI_MAX_RETRIES - 1:
                delay = 0.1 * (attempt + 1)
                if cancel_event:
                    if cancel_event.wait(delay):
                        return [False] * len(entries)
                else:
                    time.sleep(delay)
    _memdbg_note_native_outcome(False)
    raise native_exc


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
_DEBUG_TRACE_MAX_HITS = 1
_DEBUG_TRACE_WP_INDEX = None
# Never enter hardware tracing implicitly.  It is available only through the
# explicitly confirmed, one-shot experimental UI action.
_DEBUG_TRACE_ENABLED = False

# ── debug-session safety net ──────────────────────────────────────────────────
#
# An attached debug session that is never torn down is the one thing in this
# tool that can take the console down with it. ps5debug-NG allows exactly one
# session (main.c handle_client only records the debugger slot when
# g_debug_attached == 0), the target can be left SIGSTOPped, and hardware
# watchpoints stay armed in DR0-DR3. PS4CheaterNeo's own documentation warns
# that closing the game while its debugger is attached crashes the console.
#
# So: the moment we attach, the session is recorded here, and every exit path
# — normal return, exception, Ctrl-C, SIGTERM, interpreter shutdown — runs the
# teardown. This is the same pattern the freeze worker uses for its writes.
# Arming a watchpoint means SIGSTOPping the target. Doing that to a system
# process stops the console itself: ps5dbg gates its own debug-attach test
# behind --risky because "attaching to SceShellCore can freeze the system UI".
# None of these are cheat targets, and RDX previously attached to whatever
# process happened to be selected.
_DEBUG_ATTACH_BLOCKLIST = frozenset({
    "kernel", "mini-syscore.elf", "SceSysCore.elf", "SceShellCore",
    "SceShellUI", "SceSysAvControl.elf", "SceRemotePlay",
    "SceGameLiveStreaming", "SceVideoCore2K", "SceAvCapture",
    "orbis_audiod.elf", "AgcCompositor.elf",
})


def _debug_attach_refusal(process: str) -> Optional[str]:
    """Why the debugger must not attach to `process`, or None if it may.

    Hard-refuses the processes that run the console UI and audio/video
    pipeline. Anything else that is plainly a system service still attaches,
    but the caller is expected to confirm it explicitly — see
    _debug_attach_is_unusual().
    """
    name = str(process or "").strip()
    if not name:
        return "no process is attached"
    if name in _DEBUG_ATTACH_BLOCKLIST:
        return (f"'{name}' runs the console itself. Stopping it to arm a "
                "hardware watchpoint can freeze the system UI, and it is "
                "never a cheat target.")
    return None


def _debug_attach_is_unusual(process: str) -> bool:
    """True for a process that is probably a system service, not a game.

    Games run as `eboot.bin`; homebrew as a named `.elf`. A `Sce*` daemon is
    almost certainly not what the user meant to trace, so it earns a second,
    explicit confirmation rather than a silent attach.
    """
    name = str(process or "").strip()
    return bool(name) and name.startswith("Sce")


CMD_DEBUG_PROCESS_STOP = 0xBDBB0500
_debug_session_lock = threading.Lock()
_debug_session: Optional[dict] = None   # {"ip","pid","sock","wp_index","stopped"}


def _register_debug_session(ip: str, pid: int, sock) -> None:
    global _debug_session
    with _debug_session_lock:
        _debug_session = {"ip": ip, "pid": int(pid), "sock": sock,
                          "wp_index": None, "stopped": False}


def _update_debug_session(**fields) -> None:
    with _debug_session_lock:
        if _debug_session is not None:
            _debug_session.update(fields)


def _clear_debug_session() -> None:
    global _debug_session
    with _debug_session_lock:
        _debug_session = None


_debug_session_stuck = False      # payload still holds a session we cannot clear
_DETACH_TIMEOUT = 5.0


def _debug_session_is_stuck() -> bool:
    """True when a previous teardown failed to release the payload's session."""
    return _debug_session_stuck


def _debug_detach_or_report(cmd: socket.socket, ip: str, pid: int) -> bool:
    """Detach, and make failure loud instead of invisible.

    The previous code sent `CMD_DEBUG_DETACH` on `cmd` and swallowed every
    exception. That is precisely backwards: a trace fails most often by the
    command socket going bad, so the one moment detach matters is the one
    moment this socket cannot deliver it. Observed on hardware -- a trace timed
    out, the swallowed detach never landed, and `g_debug_attached` stayed 1.
    Every later attach was refused with "already debugging" and nothing
    connected that to the earlier run.

    Recovery is limited by the protocol: the server binds the session to the
    connection slot recorded at attach (reference 1.5), so a detach from a new
    connection acks success without clearing anything -- verified on hardware.
    The realistic options are therefore to try harder on the owning socket, and
    to tell the user the truth when that fails.
    """
    global _debug_session_stuck
    try:
        cmd.settimeout(_DETACH_TIMEOUT)
    except Exception:
        pass
    last = None
    for attempt in range(3):
        try:
            cmd.sendall(cmd_header(CMD_DEBUG_DETACH))
            if _debug_status_ok(cmd):
                _debug_session_stuck = False
                return True
            last = "payload refused the detach"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(0.2 * (attempt + 1))

    # Last resort: at least make sure the game is not left stopped. This does
    # not clear the session -- CMD_DEBUG_PROCESS_STOP is a bare kill(pid,
    # SIGCONT) -- so its success must not be mistaken for a detach.
    resumed = _debug_force_resume(ip, pid)
    _debug_session_stuck = True
    add_log(
        f"Debug detach failed ({last}). The target is most likely still traced "
        f"by this session, so the next attach will fail. "
        + ("The game was resumed, so play is unaffected."
           if resumed else
           "The game may still be stopped; use Force Resume."),
        "error")
    add_log("Relaunch the game before tracing again — a fresh process clears a "
            "leaked trace. Reload ps5debug-NG only if that does not help.",
            "error")
    return False


def _local_address_towards(ip: str, port: int = 744) -> Optional[str]:
    """The source address this host would use to reach `ip`.

    Uses a connectionless UDP socket, so nothing is sent and no server has to
    be listening; the kernel just resolves the route.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((ip, int(port)))
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


def _trace_network_refusal(ip: str) -> Optional[str]:
    """Why the console could not deliver debug events here, or None if it can.

    The debug interrupt channel is the console dialling *out* to the client on
    port 755 (protocol 1.9), using the address it sees on the command
    connection. Every other RDX operation is client-to-console, so a route that
    is fine for scanning can still be useless for tracing -- and the failure is
    ugly: the attach stops the game, then blocks waiting for a callback that
    can never arrive, and the target is left traced.

    Seen in testing over a Tailscale subnet route: the console saw the client
    as 100.122.106.94 and had no route back to it. Scans, reads and writes all
    worked perfectly, which makes this genuinely confusing without a check.
    """
    local = _local_address_towards(ip)
    if local is None:
        return f"no route to {ip}"
    try:
        local_addr = ipaddress.ip_address(local)
        console_addr = ipaddress.ip_address(str(ip))
    except ValueError:
        return None

    # 100.64.0.0/10 is carrier-grade NAT, which is also what Tailscale and
    # similar overlays hand out. A console on a normal LAN cannot route to it.
    if local_addr in ipaddress.ip_network("100.64.0.0/10"):
        return (f"this host reaches the console from {local}, a carrier-grade "
                f"NAT/VPN address (Tailscale and similar overlays use "
                f"100.64.0.0/10). The console must open a connection back to "
                f"the client on port 755 and cannot route to that address")

    # Otherwise require the client to look local to the console. A /24 is the
    # common case; this is a heuristic, so it only fires when both addresses
    # are private and clearly on different networks.
    if local_addr.is_private and console_addr.is_private:
        if local_addr.packed[:3] != console_addr.packed[:3]:
            return (f"this host reaches the console from {local}, which is not "
                    f"on the console's network ({ip}). The console opens the "
                    f"debug channel back to the client on port 755 and is "
                    f"unlikely to have a route to that address")
    return None


def _debug_force_resume(ip: str, pid: int) -> bool:
    """Resume a target over a fresh connection, with no debug session needed.

    `CMD_DEBUG_PROCESS_STOP` is handled even when no session is active: the
    server falls through to a direct `kill(pid, sig)` with `0 -> SIGCONT`
    (protocol 2.3). That makes it the one way to un-stick a game left stopped
    by a session that died without detaching — including one leaked by a
    previous run of this program.
    """
    try:
        s = ps5_connect(ip, timeout=5.0)
    except OSError:
        return False
    try:
        body = struct.pack("<IB", int(pid), 0)   # 5 raw bytes: pid, state=0
        s.sendall(cmd_header(CMD_DEBUG_PROCESS_STOP, len(body)) + body)
        return _debug_status_ok(s)
    except Exception:
        return False
    finally:
        try: s.close()
        except Exception: pass


def _emergency_debug_teardown() -> None:
    """Clear the watchpoint, resume the target and detach. Never raises.

    Registered with atexit and called from every trace exit path, so a crash,
    Ctrl-C or a closed terminal cannot leave the console attached with a live
    hardware watchpoint.
    """
    with _debug_session_lock:
        session, globals()["_debug_session"] = _debug_session, None
    if not session:
        return
    sock = session.get("sock")
    if sock is not None:
        for step in ("watchpoint", "resume", "detach"):
            try:
                if step == "watchpoint" and session.get("wp_index") is not None:
                    _debug_clear_watchpoint(sock, int(session["wp_index"]))
                elif step == "resume" and session.get("stopped"):
                    _debug_continue(sock, 0)
                elif step == "detach":
                    sock.sendall(cmd_header(CMD_DEBUG_DETACH))
                    # Record a failed release here too. This path runs from
                    # atexit and the signal handlers, so it is the last chance
                    # to notice; leaving it silent is what made a stuck
                    # session look like an unexplained refusal later.
                    if not _debug_status_ok(sock):
                        globals()["_debug_session_stuck"] = True
            except Exception:
                globals()["_debug_session_stuck"] = (
                    step == "detach" or _debug_session_stuck)
                pass          # keep going; a later step may still land
        try: sock.close()
        except Exception: pass
    # Belt and braces: DETACH already runs debug_full_teardown (which resumes),
    # but if the command socket was the thing that broke, the target may still
    # be stopped and only a fresh connection can reach it.
    if session.get("stopped"):
        _debug_force_resume(str(session.get("ip", "")), int(session.get("pid", 0)))


import atexit as _atexit
_atexit.register(_emergency_debug_teardown)

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

# On-the-wire status words (protocol reference 1.6 — the server bit-swaps, and
# clients compare the raw wire word, so these are the post-swap values).
_DEBUG_STATUS_NAMES = {
    0x80000000: "CMD_SUCCESS",
    0xF0000001: "CMD_ERROR",
    0xF0000003: "CMD_DATA_NULL",
    0xF0000004: "CMD_ALREADY_DEBUG",
    0xF0000005: "CMD_INVALID_INDEX",
}


def _debug_status_word(s: socket.socket) -> int:
    """Read one raw status word."""
    return struct.unpack("<I", recv_exact(s, 4))[0]


def _debug_status_name(word: int) -> str:
    return _DEBUG_STATUS_NAMES.get(int(word), f"unknown status {int(word):#010x}")


def _debug_status_ok(s: socket.socket) -> bool:
    return _debug_status_word(s) == STATUS_SUCCESS

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

# ── hardware debug-register inspection ────────────────────────────────────────
# The watchpoint arms cleanly and never fires (HARDWARE_TEST_CHECKLIST). Client
# packet shape and thread fan-out were both ruled out there, leaving three
# payload-side hypotheses -- and the session that made the observation could not
# tell them apart, because RDX read the debug registers only *before* arming, on
# one thread, to pick a free slot. Nothing read them back afterwards, so "no
# event in 60 s" could equally mean the DRs were never set or that they were set
# and the store did not trap. Those have different causes and different fixes.
#
# These helpers close that gap with the command RDX already implements
# (CMD_DEBUG_GETDBREGS, which takes an lwpid). They are read-only: nothing here
# writes a debug register. Deciding whether the *write* half
# (CMD_DEBUG_SETDBREGS, 0xBDBB000D -- documented, not implemented) is needed is
# exactly what the read-back is for, and it must not be implemented on
# speculation. See the checklist's calibration note.
_DR_PROBE_MAX_THREADS = 128       # bounded round trips; the observed target had 40
# The debug command socket carries a 10 s timeout that persists per recv, so a
# console that stops answering mid-sweep could stall 40 threads x 10 s ~= 7
# minutes with the diagnostic holding the trace open. A wall-clock budget keeps
# a diagnostic from outlasting the thing it is diagnosing.
_DR_PROBE_TIME_BUDGET = 8.0


def _debug_decode_dbregs(blob: bytes) -> dict:
    """Decode a 128-byte dbreg blob into DR7 plus the four slot addresses.

    Layout matches _debug_free_watchpoint's existing assumption: 16 x uint64
    with DR0-DR3 at indices 0-3 and DR7 at index 7.
    """
    regs = struct.unpack("<16Q", blob)
    dr7 = int(regs[7])
    slots = []
    for i in range(4):
        # DR7 bits 2i (local) and 2i+1 (global) enable slot i.
        enabled = bool(dr7 & (1 << (2 * i))) or bool(dr7 & (1 << (2 * i + 1)))
        slots.append({"index": i, "enabled": enabled, "address": int(regs[i])})
    return {"dr7": dr7, "slots": slots}


def _debug_thread_dr_state(s: socket.socket, lwpid: int) -> Optional[dict]:
    """DR state for one thread, or None when it cannot be read."""
    try:
        state_ = _debug_decode_dbregs(_debug_get_dbregs(s, lwpid))
    except Exception:
        return None
    state_["lwpid"] = int(lwpid)
    return state_


def _debug_free_watchpoint_all(s: socket.socket, threads: list) -> Optional[int]:
    """Pick a DR slot that is free on *every* readable thread.

    The previous version read DR7 from threads[0] alone. If the payload applies
    debug registers per-thread -- which is one of the open hypotheses -- a slot
    free on thread 0 can be occupied on thread 7, so the index chosen could be
    wrong for the thread that matters. Taking the union costs one round trip per
    thread and cannot pick a busy slot.
    """
    occupied = set()
    seen_any = False
    deadline = time.monotonic() + _DR_PROBE_TIME_BUDGET
    for lwpid in list(threads)[:_DR_PROBE_MAX_THREADS]:
        if time.monotonic() >= deadline:
            # Partial knowledge is still safe here: every slot seen busy on
            # any thread stays excluded, so a short sweep can only make the
            # choice more conservative, never pick an occupied slot.
            break
        state_ = _debug_thread_dr_state(s, lwpid)
        if state_ is None:
            continue
        seen_any = True
        for slot in state_["slots"]:
            if slot["enabled"]:
                occupied.add(slot["index"])
    if not seen_any:
        return None
    for i in range(4):
        if i not in occupied:
            return i
    return None


def _debug_verify_watchpoint(s: socket.socket, threads: list, address: int,
                             wp_index: int, cancel_event=None,
                             time_budget: float = _DR_PROBE_TIME_BUDGET) -> dict:
    """Read the debug registers back on every thread after arming.

    Returns coverage, never raises: this is a diagnostic and must not be able
    to fail a trace that would otherwise work -- which includes not being able
    to hang it, hence the budget.
    """
    armed, absent, unreadable = [], [], []
    wanted = list(threads)[:_DR_PROBE_MAX_THREADS]
    deadline = time.monotonic() + max(float(time_budget), 0.1)
    truncated = False
    for index, lwpid in enumerate(wanted):
        if (cancel_event is not None and cancel_event.is_set()) or \
                time.monotonic() >= deadline:
            truncated = True
            break
        state_ = _debug_thread_dr_state(s, lwpid)
        if state_ is None:
            unreadable.append(int(lwpid))
            continue
        slot = next((x for x in state_["slots"]
                     if x["index"] == int(wp_index)), None)
        # Match on the address too: a slot enabled for some *other* address is
        # not this watchpoint, and counting it would manufacture coverage.
        if slot and slot["enabled"] and int(slot["address"]) == int(address):
            armed.append(int(lwpid))
        else:
            absent.append(int(lwpid))
    return {"armed": armed, "absent": absent, "unreadable": unreadable,
            "checked": len(armed) + len(absent),
            "total": len(list(threads)),
            "truncated": truncated,
            "unchecked": max(0, len(wanted) - len(armed) - len(absent)
                             - len(unreadable))}


def _debug_watchpoint_verdict(coverage: dict) -> tuple:
    """(verdict_key, human_sentence) for a _debug_verify_watchpoint result.

    The three outcomes discriminate the checklist's remaining hypotheses.
    """
    armed, checked = len(coverage.get("armed", [])), coverage.get("checked", 0)
    if checked == 0:
        return ("unknown",
                "debug registers could not be read back on any thread")
    # A sweep cut short by the budget saw a sample, not the whole target.
    # Saying "ruled out" or "the payload applies DRs per-thread" from a
    # sample would be the same overreach the checklist's calibration note
    # already records once.
    partial_sample = bool(coverage.get("truncated"))
    caveat = (f" (sample only: {checked} of {coverage.get('total', checked)} "
              f"thread(s) read before the time budget expired)"
              if partial_sample else "")
    if armed == 0:
        return ("none",
                "the watchpoint was acknowledged but is set on no thread "
                "that was read — payload-side; worth reporting upstream"
                + caveat)
    if armed == checked:
        return ("all",
                f"the watchpoint is set on all {armed} thread(s) read"
                + (caveat if partial_sample else
                   " — per-thread application is ruled out; a store that does "
                   "not trap points at DR honouring or another mapping"))
    return ("partial",
            f"the watchpoint is set on {armed} of {checked} thread(s) read — "
            "the payload applies debug registers per-thread, so arming needs "
            "CMD_DEBUG_SETDBREGS per thread" + caveat)


def _debug_watchpoint_preliminary(coverage: Optional[dict]) -> Optional[str]:
    """What the pre-resume, single-thread read-back says on its own.

    Deliberately NOT a coverage verdict. One thread cannot rule per-thread
    application in or out, and saying it could is the overreach the
    calibration note already records once.

    This exists because the ordering lost a real measurement. patch102 takes
    this read while the target is stopped, then resumes, then sweeps every
    thread -- and only the sweep produced a verdict. On 2026-08-30 the resume
    itself timed out (`CMD_DEBUG_CONTINUE`, 61 s), the exception unwound, and
    the single-thread data already in hand was discarded. The attach cost a
    stopped game, a wedged payload and a console restart, and answered
    nothing.

    A broken debugger lifecycle must not be able to destroy the diagnostic
    whose job is to explain it, so whatever is known is reported before the
    next thing that can hang.
    """
    if not coverage:
        return None
    checked = coverage.get("checked", 0)
    if checked == 0:
        return "debug registers could not be read back before the resume"
    if len(coverage.get("armed", [])) == 0:
        return ("the watchpoint is NOT set on the thread read while stopped — "
                "the payload acknowledged the arm and did not keep it")
    return ("the watchpoint IS set on the thread read while stopped; whether "
            "other threads carry it is what the post-resume sweep answers")


def _debug_free_watchpoint(s: socket.socket, lwpid: int) -> Optional[int]:
    """Single-thread free-slot probe. Retained for callers with one thread;
    _debug_free_watchpoint_all is what the trace path uses."""
    try:
        state_ = _debug_decode_dbregs(_debug_get_dbregs(s, lwpid))
    except Exception:
        return None
    for slot in state_["slots"]:
        if not slot["enabled"]:
            return slot["index"]
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
    # The payload validates max_entries as 1..1000000 and answers CMD_ERROR
    # outside that, so a bad argument must not reach the wire at all.
    max_entries = int(max_entries)
    if not 1 <= max_entries <= 1_000_000:
        raise ValueError(
            f"max_entries must be between 1 and 1,000,000, got {max_entries}")
    body = struct.pack("<IQII", int(pid), int(address), int(length), max_entries)
    s.sendall(cmd_header(CMD_PROC_DISASM_REGION, len(body)) + body)
    if not _debug_status_ok(s):
        raise RuntimeError("disassembly request rejected")
    out = []
    while True:
        raw = recv_exact(s, 32)
        if raw == b"\xFF" * 32:
            break
        # The server sends at most max_entries records and then the sentinel
        # (protocol reference 380-385), so a further record means the stream
        # has desynced -- typically unread bytes left on this socket by an
        # earlier command. Without this bound the loop is infinite and
        # accumulating: a desynced stream produced 6.5 million entries and
        # 2.7 GB of RSS in 5.4 s for a request that asked for 16. That is far
        # worse here than elsewhere, because this runs with the target process
        # stopped inside _trace_temporary_access -- an OOM kill is SIGKILL,
        # the teardown cannot run, and the game is left SIGSTOPped.
        if len(out) >= max_entries:
            raise RuntimeError(
                f"disassembly stream did not terminate after {max_entries} "
                f"entries; the connection stream is out of sync — reconnect "
                f"to the console")
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

def _decoded_effective_address(insn: dict, regs: dict):
    """Effective address of a decoded instruction's memory operand.

    Returns ``(effective, base_name, base_value, index_name, index_value)``.
    A RIP-relative operand resolves against the *end* of the instruction, which
    is what the architecture defines and what makes such an operand a stable
    code reference rather than an object pointer.
    """
    base_reg_id = int(insn["mem_base_reg"])
    index_reg_id = int(insn["mem_index_reg"])
    base_name = _ZYDIS_GPR64.get(base_reg_id)
    index_name = _ZYDIS_GPR64.get(index_reg_id)
    index_val = int(regs.get(index_name, 0)) if index_name else 0
    if base_reg_id == _ZYDIS_RIP:
        base_val = int(insn["addr"]) + int(insn["length"])
        base_name = "rip"
    elif base_name:
        base_val = int(regs[base_name])
    else:
        base_val = 0
    effective = (base_val + index_val * int(insn["mem_scale"] or 1)
                 + int(insn["mem_disp"]))
    return effective, base_name, base_val, index_name, index_val


def _trace_temporary_access(ip: str, pid: int, target_addr: int,
                            width: int, timeout: float = _DEBUG_TRACE_TIMEOUT,
                            experimental: bool = False,
                            _attach_retried: bool = False) -> dict:
    """
    Change-triggered resolver:
      1) attach debugger;
      2) install a read/write hardware watchpoint on target;
      3) wait for a real target-process access;
      5) capture RIP/registers and decode the memory operand;
      6) restore the original bytes and detach.

    Returns a trace dictionary.  It never leaves the probe value installed.
    """
    if not (_DEBUG_TRACE_ENABLED or experimental):
        raise RuntimeError(
            "hardware-watchpoint tracing is disabled: unsafe debugger lifecycle")
    refusal = _debug_attach_refusal(state.get("proc_name", ""))
    if refusal:
        raise RuntimeError(refusal)
    # Check the return path BEFORE attaching. Attaching stops the game and then
    # waits for a callback; if that callback can never arrive the game is left
    # traced and the next attach fails. Cheap to check, expensive to discover.
    network = _trace_network_refusal(ip)
    if network:
        raise RuntimeError(
            f"hardware watchpoint tracing is not possible over this network "
            f"path: {network}. Scanning, reading, writing and freezing are "
            f"unaffected -- only tracing needs the console to reach this host. "
            f"Run RDX on a machine on the same network as the console to trace.")
    target_addr = int(target_addr)
    width = int(width)
    original = ps5_read(ip, pid, target_addr, width)
    if len(original) != width:
        raise RuntimeError("could not read original target value")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Port 755 is not ours to choose: the console connects *out* to the client
    # and ps5debug-NG hard-codes htons(755) in debug.c (protocol ref 1.9), so
    # the client must already be listening there before CMD_DEBUG_ATTACH. It is
    # below 1024, which is privileged on Linux and macOS -- so an ordinary user
    # run fails here with a bare "[Errno 13] Permission denied" that names
    # neither the port nor the remedy. Bind before attaching, and translate the
    # two failures that actually happen into something actionable.
    try:
        listener.bind(("0.0.0.0", 755))
        listener.listen(1)
    except PermissionError as exc:
        listener.close()
        raise RuntimeError(
            f"cannot listen on TCP port 755 for the debug interrupt channel: "
            f"{exc}. The console dials this fixed port itself, so it cannot be "
            f"changed. Either run RDX as root, or grant the interpreter the "
            f"bind capability once:\n"
            f"    sudo setcap 'cap_net_bind_service=+ep' "
            f"$(readlink -f $(command -v python3))\n"
            f"(the capability applies to that interpreter for every program it "
            f"runs, so prefer sudo if that is not wanted)") from exc
    except OSError as exc:
        listener.close()
        raise RuntimeError(
            f"cannot listen on TCP port 755 for the debug interrupt channel: "
            f"{exc}. Another process is holding it -- typically an earlier RDX "
            f"run whose listener has not been released yet, or a second "
            f"debugger. Close it and retry.") from exc
    listener.settimeout(max(timeout + 1.0, 2.0))

    cmd = None
    event_sock = None
    attached = False
    wp_index = None
    target_stopped = False
    try:
        cmd = ps5_connect(ip, timeout=10.0)
        body = struct.pack("<I", int(pid))
        cmd.sendall(cmd_header(CMD_DEBUG_ATTACH, len(body)) + body)
        attach_status = _debug_status_word(cmd)
        if attach_status != STATUS_SUCCESS:
            # ps5debug-NG permits one session at a time, so this is usually a
            # session leaked by an earlier run rather than a real failure.
            # Try to un-stick the game before giving up: a target left
            # SIGSTOPped is indistinguishable from a hung console to the user.
            # Report what the payload actually said. Treating every non-success
            # word as "already debugging" sends the user to reload the payload
            # when the real cause is usually per-process and cleared by
            # relaunching the game -- a much cheaper remedy.
            status_name = _debug_status_name(attach_status)
            if attach_status == 0xF0000004 and not _attach_retried:
                # A session left behind by an earlier attempt is clearable:
                # CMD_DEBUG_DETACH runs debug_full_teardown even from a new
                # connection. Verified on hardware -- a held session answered
                # CMD_SUCCESS and the flag was released. Recover once
                # automatically rather than making the user do it by hand.
                add_log("A debug session was still held; releasing it and "
                        "retrying the attach.", "warn")
                try:
                    cmd.sendall(cmd_header(CMD_DEBUG_DETACH))
                    _debug_status_word(cmd)
                except Exception:
                    pass
                try:
                    cmd.close()
                except Exception:
                    pass
                return _trace_temporary_access(
                    ip, pid, target_addr, width, timeout=timeout,
                    experimental=experimental, _attach_retried=True)
            if attach_status == 0xF0000004:      # CMD_ALREADY_DEBUG
                # Only this word actually means a session is held. Resuming is
                # still worth attempting so a stopped game is not left frozen.
                if _debug_force_resume(ip, pid):
                    add_log("A previous debug session was still attached; the "
                            "target was resumed.", "warn")
                raise RuntimeError(
                    f"debug attach refused with {status_name}: another debug "
                    f"session is already attached. Close the other debugger, "
                    f"or reload ps5debug-NG if nothing else is using it.")
            raise RuntimeError(
                f"debug attach failed with {status_name}. The payload elevates, "
                f"calls ptrace(PT_ATTACH) and then dials back to port 755; this "
                f"status means it did not get that far. The usual cause is that "
                f"'{state.get('proc_name', 'the target')}' is still traced by an "
                f"earlier session, which a fresh launch of the game clears. "
                f"Relaunch the title and try again; reload ps5debug-NG only if "
                f"that does not help.")
        attached = True
        _register_debug_session(ip, pid, cmd)

        event_sock, _ = listener.accept()
        event_sock.settimeout(timeout)

        threads = _debug_thread_list(cmd)
        if not threads:
            raise RuntimeError("debugger attached but no target threads were reported")
        # Filled by the post-arm read-back below; merged into the result so a
        # caller can report *why* a trace found nothing, not just that it did.
        wp_diagnostic: dict = {}
        lwpid = threads[0]
        # Union across every thread, not threads[0] alone: if the payload
        # applies debug registers per-thread, a slot free on thread 0 can be
        # occupied on another, and the index picked would be wrong for the
        # thread that matters.
        wp_index = _debug_free_watchpoint_all(cmd, threads)
        if wp_index is None:
            raise RuntimeError("no free hardware watchpoint slot")
        _update_debug_session(wp_index=wp_index)

        # Stop briefly while installing DR7.  Do not write a probe value here:
        # restoring one later can overwrite a legitimate in-game inventory
        # change made while the trace is active.
        _debug_continue(cmd, 1)
        target_stopped = True
        _update_debug_session(stopped=True)
        # DR7 length encoding: 0=1 byte, 1=2 bytes, 2=8 bytes, 3=4 bytes.
        wp_length = {1: 0, 2: 1, 4: 3, 8: 2}.get(width)
        if wp_length is None:
            raise RuntimeError(f"unsupported watchpoint width: {width}")
        # Write-only (DR7 RW=01) avoids stopping on harmless UI/inventory reads.
        _debug_set_watchpoint(cmd, wp_index, target_addr, wp_length, 1)
        # One thread only, while stopped. patch97 swept every thread here,
        # which put up to 128 sequential round trips between the stop and the
        # resume -- in the one path this project has already watched
        # black-screen a live game. The full sweep now runs after the resume
        # (below); this single read costs one round trip and exists to catch
        # the case where the two disagree, which would itself say something
        # about whether the payload stages debug registers in the pcb.
        try:
            stopped_check = _debug_verify_watchpoint(
                cmd, threads[:1], target_addr, wp_index)
        except Exception as exc:
            stopped_check = None
            add_log(f"Watchpoint DR pre-resume check unavailable: {exc}", "warn")
        # Report it NOW. Everything after this point can hang -- the resume
        # did, for 61 s, on 2026-08-30 -- and an exception there used to take
        # this measurement with it.
        preliminary = _debug_watchpoint_preliminary(stopped_check)
        if preliminary:
            add_log(f"Watchpoint DR pre-resume (1 thread, slot {wp_index} @ "
                    f"{hex(target_addr)}): {preliminary}",
                    "warn" if "NOT set" in preliminary else "info")
        _debug_continue(cmd, 0)
        target_stopped = False
        _update_debug_session(stopped=False)
        # Full sweep, with the game running again. This is also the state that
        # actually matters: the debug registers as they stand while the store
        # under investigation executes.
        try:
            wp_coverage = _debug_verify_watchpoint(
                cmd, threads, target_addr, wp_index)
            if wp_coverage.get("truncated"):
                add_log("Watchpoint DR sweep stopped at its time budget; "
                        "the verdict below is from a sample", "warn")
            verdict_key, verdict_text = _debug_watchpoint_verdict(wp_coverage)
            add_log(
                f"Watchpoint DR verify: slot {wp_index} @ {hex(target_addr)} — "
                f"set on {len(wp_coverage['armed'])}/{wp_coverage['checked']} "
                f"readable thread(s) of {wp_coverage['total']}"
                + (f", {len(wp_coverage['unreadable'])} unreadable"
                   if wp_coverage["unreadable"] else ""),
                "warn" if verdict_key != "all" else "info")
            add_log(f"Watchpoint DR verdict: {verdict_text}",
                    "error" if verdict_key in ("none", "unknown") else
                    "warn" if verdict_key == "partial" else "info")
            _update_debug_session(wp_coverage=wp_coverage,
                                  wp_verdict=verdict_key)
            wp_diagnostic = {"wp_coverage": wp_coverage,
                             "wp_verdict": verdict_key,
                             "wp_verdict_text": verdict_text,
                             "wp_stopped_check": stopped_check}
            # A disagreement between the two reads is itself evidence: it
            # would mean the arm is visible to a stopped thread and not to a
            # running one, i.e. the payload stages debug registers rather
            # than applying them. Worth saying out loud, because it changes
            # what the verdict above means.
            if stopped_check is not None and threads:
                first = int(threads[0])
                was_armed = first in stopped_check.get("armed", [])
                now_armed = first in wp_coverage.get("armed", [])
                if was_armed != now_armed:
                    add_log(
                        f"Watchpoint DR mismatch on lwpid {first}: "
                        f"{'set' if was_armed else 'clear'} while stopped, "
                        f"{'set' if now_armed else 'clear'} while running — "
                        f"the payload appears to stage debug registers",
                        "warn")
                    wp_diagnostic["wp_staged"] = True
        except Exception as exc:
            add_log(f"Watchpoint DR verify unavailable: {exc}", "warn")

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
            target_stopped = True
            _update_debug_session(stopped=True)
            dr6 = int(candidate["dbregs"][6])
            # DR6 is a hint here, not a gate.  ps5debug-NG clears DR6 while it
            # handles the trap, so a genuine hit arrives with DR6 == 0 —
            # confirmed on firmware 10.01 with payload v1.3.0, where DR7 still
            # showed the watchpoint armed (L3/G3 set, R/W3=write, LEN3=4).
            # Requiring our slot's bit discarded every real event and made the
            # trace time out every time.  A *non-zero* DR6 naming some other
            # slot genuinely is another watchpoint's event; DR6 == 0 means the
            # payload consumed it, and the decoded operand below decides.
            if dr6 and not (dr6 & (1 << int(wp_index))):
                # Every debug event stops the target.  Ignoring an unrelated
                # event without resuming leaves the game visibly frozen.
                _debug_continue(cmd, 0)
                target_stopped = False
                _update_debug_session(stopped=False)
                continue
            regs = candidate["regs"]
            rip = int(regs["rip"])
            try:
                insns = _debug_disasm(cmd, pid, max(rip - 8, _ADDR_MIN), 32, 16)
                # x86 data-breakpoint #DB is reported after the memory
                # instruction.  Prefer an instruction ending at event RIP;
                # retain addr==RIP as a compatibility fallback.
                decoded = next((x for x in reversed(insns)
                                if int(x["addr"]) + int(x["length"]) == rip
                                and (int(x["kind"]) & 0x10)), None)
                if decoded is None:
                    decoded = next((x for x in insns
                                    if int(x["addr"]) == rip), None)
            except Exception as exc:
                decoded = None
                last_reason = f"disassembly failed: {exc}"
            if not decoded or not (int(decoded["kind"]) & 0x10):
                last_reason = "watchpoint hit but accessor instruction was not decoded"
                _debug_continue(cmd, 0)
                target_stopped = False
                _update_debug_session(stopped=False)
                continue
            # With DR6 unusable as a discriminator, the decoded operand is what
            # proves this event belongs to *this* watchpoint.  An accessor that
            # resolves elsewhere is another trap, so it must not consume the
            # hit budget -- keep waiting rather than aborting the trace.
            try:
                probe = _decoded_effective_address(decoded, regs)[0]
            except Exception as exc:
                probe = None
                last_reason = f"effective address unavailable: {exc}"
            if probe != int(target_addr):
                if probe is not None:
                    last_reason = (f"accessor resolved to {hex(probe)}, not the "
                                   f"watched {hex(int(target_addr))}")
                _debug_continue(cmd, 0)
                target_stopped = False
                _update_debug_session(stopped=False)
                continue
            hits += 1
            event = candidate
            insn = decoded
            break
        if event is None or insn is None:
            # A timeout is precisely the case the read-back exists for: the old
            # message said only that nothing fired, which is the observation
            # that could not be acted on. Attach the verdict to it.
            verdict_text = wp_diagnostic.get("wp_verdict_text")
            raise TimeoutError(
                f"{last_reason} — {verdict_text}" if verdict_text
                else last_reason)

        regs = event["regs"]
        rip = int(regs["rip"])

        # Resolve the actual effective address represented by the decoded
        # operand.  RIP-relative accesses are stable code references, not object
        # pointers, so they are reported but not used as permanent pointer roots.
        (effective, base_name, base_val,
         index_name, index_val) = _decoded_effective_address(insn, regs)
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
            # `rip` is the raw trap RIP the payload reported.  x86 data
            # breakpoints are trap-type, so it names the instruction *after*
            # the access; `writer` is the instruction that actually performed
            # it and is the address an AOB anchor must be captured at.
            "rip": rip,
            "writer": int(insn["addr"]),
            "base_reg": base_name or f"reg#{base_reg_id}",
            "base_value": base_val,
            "index_reg": index_name,
            "index_value": index_val,
            "scale": int(insn["mem_scale"] or 1),
            "final_offset": int(insn["mem_disp"]),
            "access_mode": access_mode,
            "instruction": insn,
            "lwpid": int(event["lwpid"]),
            **wp_diagnostic,
        }
    finally:
        # Clear the watchpoint before detaching.
        if cmd is not None and wp_index is not None:
            try:
                _debug_clear_watchpoint(cmd, wp_index)
            except Exception:
                pass
        # A captured/exception event leaves the process stopped.  Resume
        # explicitly even though DETACH also performs a full resume teardown.
        if cmd is not None and attached and target_stopped:
            try:
                _debug_continue(cmd, 0)
                target_stopped = False
                _update_debug_session(stopped=False)
            except Exception:
                pass
        if cmd is not None and attached:
            _debug_detach_or_report(cmd, ip, pid)
        # Whatever happened above, the session is finished from here on; drop
        # the registration so the atexit net does not act on a dead socket.
        _clear_debug_session()
        if event_sock is not None:
            try: event_sock.close()
            except Exception: pass
        if cmd is not None:
            try: cmd.close()
            except Exception: pass
        try: listener.close()
        except Exception: pass

def _trace_base_is_resolvable(trace: dict) -> Optional[str]:
    """Why this trace cannot seed a permanent chain, or None if it can.

    A chain root has to be reachable from a module every run.  `rip` is a code
    reference rather than an object pointer; `rsp`/`rbp` are stack frames that
    exist only for the duration of the call; and an indexed access
    (`[base + index*scale]`) has a runtime-varying element that no fixed offset
    chain can reproduce.  Returning the reason rather than a bool lets the UI
    tell the user which of these it hit, since the remedy differs.
    """
    if not trace.get("base_value"):
        return "the accessor had no base register (absolute or computed address)"
    if trace.get("base_reg") in ("rip", "rsp", "rbp"):
        return (f"the base register is {trace.get('base_reg')}, which is a code "
                "or stack reference rather than a heap object pointer")
    if trace.get("index_reg"):
        return (f"the access is indexed via {trace.get('index_reg')}, whose value "
                "varies at runtime and cannot be baked into a fixed chain")
    return None


def _exact_pointer_holders(ip: str, pid: int, value: int,
                           cancel_event=None,
                           max_hits: int = 4096) -> list:
    """Addresses whose 64-bit contents are exactly ``value``.

    Step 4 of the documented watcher workflow: once the traced instruction has
    named the object's base pointer, you scan for that pointer *as a value* --
    hex, exact, pointer width -- and the static results are the answer. One
    bounded scan, not a graph search.
    """
    value = int(value)
    # A degenerate value matches an enormous share of memory and would return
    # a meaningless flood. The trace guards base_value, but this helper is
    # reachable on its own.
    if not (_ADDR_MIN <= value <= _ADDR_MAX):
        return []
    hits = scan_first(ip, pid, value, 8, aligned=True,
                      value_type="u64", writable_only=False,
                      cancel_event=cancel_event)
    return [int(a) for a in hits[:max_hits]]


def _walk_from_traced_base(ip: str, pid: int, base_value: int, maps: list,
                           max_depth: Optional[int] = None, cancel_event=None,
                           progress_cb=None, fanout: int = 4) -> list:
    """Walk outward from a traced object pointer to module-rooted holders.

    Level 1 is an exact-value scan, because the traced instruction named the
    object base precisely -- that is the whole benefit of having traced.

    Deeper levels cannot be exact. A parent object points at the *base* of the
    object that holds the pointer, with the pointer sitting at some field
    offset inside it; searching for the holder's own address would only find
    parents that happen to point exactly at that word, which is rare. The
    manual method solves this by re-running "what accesses" at every level.
    Without a fresh trace the honest substitute is a bounded window, and the
    real displacement is recorded rather than assumed to be zero.

    Returns [{"base", "offsets", "depth", "static"}] with offsets ordered
    outermost-first, matching every other chain in RDX.
    """
    region_starts, region_rows = _build_region_lookup(maps)
    results = []
    frontier = [(int(base_value), [])]   # (address to find a holder for, trail)
    seen = {int(base_value)}
    if max_depth is None:
        max_depth = int(setting("ptr_max_depth"))
    depth_cap = max(1, min(int(max_depth), MAX_CHAIN_DEPTH))
    for depth in range(1, depth_cap + 1):
        if cancel_event is not None and cancel_event.is_set():
            break
        next_frontier = []
        for wanted, trail in frontier[:fanout]:
            if depth == 1:
                found = [(h, 0) for h in
                         _exact_pointer_holders(ip, pid, wanted, cancel_event)]
            else:
                found = [(h, int(off)) for h, off, _rg in
                         _fast_direct_pointer_hits(ip, pid, wanted, maps,
                                                   cancel_event,
                                                   static_only=False)]
            for holder, off in found:
                region = _region_for_addr(holder, region_starts, region_rows)
                if region is None:
                    continue
                if _is_static_region(region):
                    # The displacement belongs to the chain on every branch:
                    # resolving is deref(base) + offsets[0] -> deref -> ... so
                    # the offset applied after dereferencing THIS holder must
                    # be carried, not just the trail behind it. Level 1 is an
                    # exact match, so its entry is 0.
                    results.append({"base": holder,
                                    "offsets": [off] + list(trail),
                                    "depth": depth, "static": True,
                                    "region": region.get("name", "") or "static"})
                elif holder not in seen:
                    seen.add(holder)
                    next_frontier.append((holder, [off] + list(trail)))
        if progress_cb:
            progress_cb(depth, depth_cap)
        if results or not next_frontier:
            break
        frontier = next_frontier
    return results


def _pointer_candidates_from_trace(ip: str, pid: int, trace: dict,
                                   target_addr: int, cancel_event=None,
                                   progress_cb=None,
                                   max_depth: Optional[int] = None) -> dict:
    """Turn one captured write-trace into verified module-rooted chains.

    This is the half of the change-triggered workflow that runs *after* the
    watchpoint fires, split out so the UI can capture a trace (which needs the
    user to interact with the game) and then run the search under a progress
    bar without tracing a second time.

    The win over searching backwards from the value itself: the traced
    instruction hands us the object's base pointer and the field displacement
    exactly, read off the opcode.  Searching for one known object pointer is
    far more constrained than accepting any pointer that happens to land within
    ``_PTR_STRUCT_MAX`` of the target, which is where a backwards scan's
    coincidental chains come from.
    """
    base_target = int(trace["base_value"])
    maps_for_walk = _get_maps_cached(ip, pid)
    # The documented method: scan for the traced base pointer *as a value* and
    # take the static holders. Previously this handed base_target to
    # pointer_chain_scan -- the graph search whose cost tracks heap complexity,
    # measured at over 30 minutes at depth 4 on a 4.26 GiB title. That threw
    # away the whole point of tracing: once the instruction has named the exact
    # object pointer, finding its holders is one bounded scan per level.
    candidates = _walk_from_traced_base(
        ip, pid, base_target, maps_for_walk,
        max_depth=min(int(max_depth if max_depth is not None
                          else setting("ptr_max_depth")), MAX_CHAIN_DEPTH),
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
        # The raw trap RIP, kept for diagnostics only.  x86 data breakpoints
        # are trap-type, so this is the instruction *after* the store; the
        # instruction that performed it is `trace_writer`.  Never anchor or
        # patch using this field.
        c2["trace_trap_rip"] = int(trace["rip"])
        # Diagnostic only, and optional: this path resolves pointer chains and
        # never anchors on an instruction, so a trace without a resolved writer
        # is still perfectly usable here.  The places that *do* anchor require
        # the writer outright rather than falling back to the trap RIP.
        if trace.get("writer") is not None:
            c2["trace_writer"] = int(trace["writer"])
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
            c2["confidence"] = _candidate_confidence(c2)
            verified.append(c2)

    verified.sort(key=lambda c: (-c["score"], c["depth"]))
    return {
        "candidates": verified,
        "trace": trace,
        "method": "change-triggered",
        "index_built": False,
        "maps": maps,
    }


def _resolve_trace_first(ip: str, pid: int, target_addr: int,
                         width: int, cancel_event=None,
                         progress_cb=None, experimental: bool = False) -> dict:
    """
    Trace first, then resolve the observed object pointer with the existing
    bounded pointer scanner.  Falls back to the cached reverse index if the
    trace backend is unavailable or the accessor is not pointer-like.
    """
    trace = _trace_temporary_access(ip, pid, target_addr, width,
                                    experimental=experimental)
    if cancel_event and cancel_event.is_set():
        return {"candidates": [], "trace": trace, "method": "trace-cancelled"}

    reason = _trace_base_is_resolvable(trace)
    if reason:
        return {"candidates": [], "trace": trace,
                "method": "trace-no-stable-base", "reason": reason}

    if progress_cb:
        progress_cb(0, max(_PTR_RESOLVE_MAX_NODES, 1))
    return _pointer_candidates_from_trace(
        ip, pid, trace, target_addr, cancel_event, progress_cb)


# ── batch reader for scan_next ────────────────────────────────────────────────

def ps5_read_batch(ip: str, pid: int, addrs: np.ndarray, width: int,
                   cancel_event=None, progress_cb=None,
                   value_type: Optional[str] = None) -> tuple:
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
    # Bounded by the console connection budget, not just by CPU: see
    # _MAX_CONSOLE_SOCKETS.
    NEXT_WORKERS  = min(12, _MAX_CONSOLE_SOCKETS)
    COALESCE_MAX  = 8 * 1024 * 1024
    MAX_BYTES_PER_CANDIDATE = 4096
    LOCAL_BUF     = 256         # thread-local accumulation size before flush

    type_key = _normalise_value_type(value_type, width)
    if type_key == "bytes":
        raise ValueError("typed batch reads do not decode variable raw bytes")
    width = _value_width(type_key, width)
    val_dtype = np.dtype(VALUE_TYPES[type_key]["dtype"])

    if len(addrs) == 0:
        return (np.empty(0, dtype=_NP_ADDR_DTYPE),
                np.empty(0, dtype=val_dtype))

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
                    np.empty(0, dtype=val_dtype))
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

                # Every work item is built as a coalesced ('window', ...)
                # group above (a lone isolated address is simply a window of
                # one candidate), so there is only ever this one kind here.
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
                            dtype=val_dtype, buffer=window, strides=(1,))
                        v_vals = all_vals[v_off]
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

# Concurrent console connections are a finite, shared resource and nothing
# bounded them. MemDBG documents a hard cap of 16 ("16 accepted, 4 rejected out
# of 20") and closes idle connections after 30 s. ps5debug-NG documents no cap,
# but this project observed it drop *every* connection with
# `ConnectionError: PS5 disconnected` when a 12-socket batch read and a 6-socket
# AOB scan overlapped.
#
# The budget counts sockets that are OPEN, not sockets that are busy: a pooled
# _ScanSocket is idle but still occupies a connection on the console, so it
# keeps holding its slot until it is genuinely closed. Headroom is left for the
# connections that never go through _ScanSocket at all — the resident TurboScan
# session, map fetches, the region classifier, writes, and the Results
# live-value refresh.
_MAX_CONSOLE_SOCKETS = 10
_console_socket_slots = threading.BoundedSemaphore(_MAX_CONSOLE_SOCKETS)


class _ScanSocket:
    """
    Holds one persistent native-MemDBG or ps5debug connection for a scan.
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
        self._native = _memdbg_has(MEMDBG_CAP_MEMORY_READ)
        self._native_client: Optional[_MemDBGClient] = None
        self._from_pool = False
        self._holds_slot = False
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
                    try: _console_socket_slots.release()
                    except ValueError: pass

    @classmethod
    def _acquire_slot(cls):
        """Take a connection slot, evicting idle pooled sockets if needed.

        Pooled sockets are idle for us but still open on the console, so they
        hold slots. Without this, a pool left full by a previous operation
        would starve the workers of the next one: they would block until the
        pool happened to be cleared. Active work always wins over a cached
        connection.
        """
        if _console_socket_slots.acquire(blocking=False):
            return
        cls.clear_pool()          # frees every slot the pool was holding
        _console_socket_slots.acquire()

    def __del__(self):
        """Last-resort budget recovery for a socket nobody closed.

        Every current caller closes in a `finally`, but a slot that is never
        returned is worse than the problem the budget solves: the ceiling
        drops permanently, and once it reaches zero every scan blocks for
        ever. One missed `close()` in future code would be enough. Closing
        here also pools or shuts the socket properly rather than just
        reclaiming the number.
        """
        try:
            if self._holds_slot:
                self.close()
        except Exception:
            pass
        finally:
            try:
                self._release_slot()
            except Exception:
                pass          # interpreter shutdown can gut the globals

    def _release_slot(self):
        """Give the connection budget back exactly once per open socket."""
        if self._holds_slot:
            self._holds_slot = False
            try:
                _console_socket_slots.release()
            except ValueError:
                pass

    def _connect(self):
        if self._native_client is not None:
            self._native_client.close()
            self._native_client = None
            self._release_slot()
        if self._s:
            try: self._s.close()
            except Exception: pass
            self._s = None
            self._release_slot()
        if self._native:
            self._acquire_slot()
            client = _MemDBGClient(self.ip, timeout=15.0)
            try:
                client.connect()
                if not (int((client.hello or {}).get("capabilities", 0)) &
                        MEMDBG_CAP_MEMORY_READ):
                    raise RuntimeError("native reads are not advertised")
                self._native_client = client
                self._holds_slot = True
                self._from_pool = False
                return
            except Exception as exc:
                client.close()
                _console_socket_slots.release()
                # A development/older payload may still expose the compatibility
                # service.  Try it without making every scan worker repeat the
                # native failure on each reconnect.
                self._native = False
                _note_memdbg_fallback("scan read", exc)
        key = (self.ip, self.pid)
        with self._pool_lock:
            bucket = self._pool.get(key)
            if bucket:
                self._s = bucket.pop()
                self._from_pool = True
                self._holds_slot = True     # inherited from the pooled socket
                if not bucket: self._pool.pop(key, None)
                return
        self._acquire_slot()
        try:
            self._s = ps5_connect(self.ip)
        except BaseException:
            _console_socket_slots.release()
            raise
        self._holds_slot = True
        self._from_pool = False

    def read(self, addr: int, length: int,
             cancel_event: Optional[threading.Event] = None) -> bytes:
        """Read `length` bytes from `addr`, reconnecting on transient failure."""
        # Both native MemDBG and its compatibility bridge cap public reads at
        # 1 MiB. RDX's scan engine uses larger logical chunks, so split here
        # while retaining one persistent connection and return a contiguous
        # buffer to callers. Without this, every experimental pointer scan was
        # rejected before examining a single byte.
        if (state.get("backend") == "memdbg-experimental" and length > 0x100000):
            parts = bytearray(length)
            pos = 0
            while pos < length:
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("scan cancelled")
                take = min(0x100000, length - pos)
                parts[pos:pos + take] = self._read_single(
                    addr + pos, take, cancel_event)
                pos += take
            return bytes(parts)
        return self._read_single(addr, length, cancel_event)

    def _read_single(self, addr: int, length: int,
                     cancel_event: Optional[threading.Event] = None) -> bytes:
        """Perform one payload-sized read with reconnect handling."""
        # Patch addr and length directly into the pre-built bytearray.
        # sendall accepts bytearray natively — no bytes() copy needed.
        struct.pack_into("<QI", self._req, 16, addr, length)
        for attempt in range(self.MAX_RETRIES):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("scan cancelled")
            try:
                if self._native and self._native_client is None:
                    self._connect()
                elif not self._native and self._s is None:
                    self._connect()
                if self._native_client is not None:
                    return self._native_client.memory_read(
                        self.pid, addr, length)
                self._s.sendall(self._req)   # zero-copy: no bytes() allocation
                if not check_ok(self._s):
                    raise RuntimeError("read rejected")
                return _recv_exact_cancel(self._s, length, cancel_event)
            except Exception as exc:
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("scan cancelled") from exc
                add_log(f"scan read err (attempt {attempt+1}/{self.MAX_RETRIES}) "
                        f"@ {hex(addr)}: {exc}", "warn")
                if self._native_client is not None:
                    self._native_client.close()
                    self._native_client = None
                    self._release_slot()
                if self._s is not None:
                    try: self._s.close()
                    except Exception: pass
                    self._release_slot()
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

    def set_timeout(self, seconds: float) -> None:
        """Apply a socket timeout to whichever transport is actually live.

        Callers that want a short, responsive read budget (the Results
        screen's live-value refresh) must not reach for ``self._s``
        directly: after a successful native MemDBG connect, ``_s`` is None
        and only ``_native_client`` holds a socket.
        """
        if self._native_client is not None:
            self._native_client.timeout = float(seconds)
            if self._native_client.sock is not None:
                try:
                    self._native_client.sock.settimeout(float(seconds))
                except OSError:
                    pass
        elif self._s is not None:
            try:
                self._s.settimeout(float(seconds))
            except OSError:
                pass

    def close(self):
        if self._native_client is not None:
            self._native_client.close()
            self._native_client = None
            self._release_slot()
            return
        if not self._s:
            self._release_slot()
            return
        sock, self._s = self._s, None
        key = (self.ip, self.pid)
        with self._pool_lock:
            bucket = self._pool.setdefault(key, [])
            if len(bucket) < self._POOL_MAX:
                # Still open, so it still costs a console connection: hand the
                # slot to the pool rather than releasing it here.
                bucket.append(sock)
                self._holds_slot = False
                self._from_pool = True
                return
        try: sock.close()
        except Exception: pass
        self._release_slot()
        self._from_pool = False

def _get_maps_cached(ip: str, pid: int,
                     ttl_override: Optional[float] = None) -> list:
    """
    Return ps5_maps() with a 30-second cache.  Consecutive scans on the same
    process reuse the map rather than paying an extra RTT before each scan.
    Invalidated automatically when the endpoint or pid changes, or TTL expires.
    """
    now = time.time()
    ttl = (_MAP_CACHE_TTL if ttl_override is None else
           max(0.0, float(ttl_override)))
    with _map_cache_lock:
        cache_key = (ip, pid)
        entry = _map_cache.get(cache_key)
        if entry and (now - entry[0]) < ttl:
            return entry[1]
    maps = ps5_maps(ip, pid)
    with _map_cache_lock:
        _map_cache.clear()          # only cache one pid at a time
        _map_cache[cache_key] = (now, maps)
    return maps


def _coalesce_scan_regions(regions: list) -> list:
    """Merge adjacent/overlapping eligible mappings into single spans.

    Two things fall out of this, and the second is a correctness fix.

    Fewer, larger reads. MemoryEngine360 reports the same optimisation as a
    Next Scan speedup -- "using a union of fragments to join smaller reads
    into single large reads" -- and RDX is on a slower link than it is.

    Array-of-bytes matches that straddle a mapping boundary. The AOB scanner
    extends each chunk read by len(pattern)-1 so a match spanning two chunks
    is still found, but it clamps that overlap to the region end, so a match
    spanning two *adjacent* regions was invisible. Squalr merges adjacent
    pages for exactly this reason and notes the consequence: "scanning for an
    array of bytes that crosses a page boundary is trivially supported".

    RDX already had the merge -- _coalesce_ranges and
    _coalesce_pointer_regions -- but wired only into the pointer index.

    Callers must apply their protection and scope filters *first*: this
    merges whatever it is given, so feeding it unfiltered maps would splice
    an excluded library onto the game's heap.
    """
    spans = []
    for region in sorted(regions, key=lambda r: int(r.get("start", 0))):
        start, end = int(region.get("start", 0)), int(region.get("end", 0))
        if end <= start:
            continue
        if spans and start <= spans[-1]["end"]:
            spans[-1]["end"] = max(spans[-1]["end"], end)
            spans[-1]["merged"] += 1
            # Intersect the protection bits rather than keeping the first
            # region's. No scan path reads prot after coalescing today, but a
            # span that merged a writable region with a read-only one is not
            # writable throughout, and a future reader assuming otherwise
            # would be wrong in the unsafe direction.
            spans[-1]["prot"] &= int(region.get("prot", 0))
        else:
            spans.append({"start": start, "end": end,
                          "name": region.get("name", ""),
                          "prot": int(region.get("prot", 0)),
                          "merged": 1})
    return spans


def _region_settings_are_default() -> bool:
    """True while the user has not changed the region filter settings."""
    return all(setting(key) == _SETTING_SPECS[key]["default"]
               for key in ("region_min_size", "region_exclude"))


def _note_recommended_filter(kept: int, total: int, where: str) -> None:
    """Say so when the user's own region settings are what narrowed a scan.

    Exposing these settings (patch88) created a failure mode that did not
    exist while they were literals: a scan can now come back thin, or empty,
    because of a value the user set on a different screen some time ago. The
    existing "no eligible memory regions" message does not connect the two,
    so the obvious next move is to suspect the console -- which during a
    hardware session costs an attach and a reload to rule out.

    Silent while the settings are at their defaults, so an ordinary scan gains
    no noise.
    """
    if _region_settings_are_default():
        return
    dropped = int(total) - int(kept)
    if dropped <= 0:
        return
    add_log(
        f"{where}: your region settings excluded {dropped} of {total} "
        f"mapping(s) — min size {hex(int(setting('region_min_size')))}, "
        f"exclude '{setting('region_exclude')}'. Change them in Settings > "
        f"Regions, or scan with the Writable/Readable scope.",
        "error" if kept <= 0 else "warn")


def _recommended_game_scan_region(region: dict, process: str = "") -> bool:
    """Exclude obvious payload/library mappings from the default game scan.

    The exclusion tokens and the minimum mapping size are user-editable
    settings (Settings -> Regions); PS4CheaterNeo exposes the equivalent
    preset list and its SectionFilterSize the same way. The defaults
    reproduce the behaviour this function had when the list was a literal.
    """
    name = str(region.get("name", "") or "").replace("\\", "/").lower()
    process_name = str(process or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    basename = name.rsplit("/", 1)[-1]
    main_image = (name == "executable" or
                  (process_name and basename == process_name) or
                  "/app0/" in name or "eboot" in basename)
    library_or_payload = (
        any(token in name for token in _region_exclude_tokens())
        and not main_image)
    if library_or_payload:
        return False
    # Small mappings are overwhelmingly loader/TLS/guard pages. Skipping them
    # cuts first-scan cost on fragmented heaps, which is exactly where the
    # result cap bites. Never applied to the main image, whose size is not a
    # signal about whether the user's value lives in it.
    min_size = int(setting("region_min_size"))
    if min_size and not main_image:
        try:
            span = int(region.get("end", 0)) - int(region.get("start", 0))
        except (TypeError, ValueError):
            span = 0
        if 0 < span < min_size:
            return False
    prot = int(region.get("prot", 0))
    heap_named = any(token in name for token in
                     ("anon", "heap", "dlmalloc", "game"))
    return bool(main_image or heap_named or (prot & 0x2))


# ── measured fallback for payloads with no region classifier ──────────────────
# ps5debug-NG answers "is this mapping worth scanning?" with its read-throughput
# classifier.  MemDBG has no equivalent, and the fallback that stood in for it --
# skip any region larger than MAX_REGION -- is exactly the blunt size heuristic
# the classifier was introduced to replace.  scan_first's own comment says so:
#
#     A blanket size cap is the wrong instrument for excluding GPU/VRAM, and on
#     a real title it is actively harmful: retail games commonly place the whole
#     managed heap in one multi-GiB cached mapping, so the cap silently skipped
#     the only memory worth scanning.
#
# Measured against MemDBG 0.2.0-nightly.153 on a PS5 running CUSA01659, the cap
# excluded 4.000 GiB of the 4.180 GiB of writable memory -- 95.7% of the game --
# in two 2 GiB mappings named "[device]".  Both read at 86 MiB/s, *faster* than
# the "[default]" regions that were kept (34 MiB/s), and both held thousands of
# occurrences of the value being searched for.  A first scan therefore looked at
# 4.4% of the game and reported its 53 matches as though that were the answer.
#
# So do what the classifier does: read a little, and time it.
_OVERSIZE_PROBE_BYTES = 0x40000        # 256 KiB -- one round trip, not a scan
# A readability floor, NOT a cached/uncached discriminator.
#
# patch118 introduced this as the latter, on a single sample per region under
# MemDBG where everything read at 30-90 MiB/s. Re-measured under ps5debug-NG
# with seven samples per region, against the payload classifier as ground
# truth, that reading does not survive:
#
#     0x200000000  classifier: cached     min 1.9  med 4.4  max 5.6
#     0x280200000  classifier: UNCACHED   min 3.3  med 5.0  max 5.3
#
# The uncached mapping measures *faster* than the cached one on most samples.
# Single-sample throughput varies by 3x within one region, so the earlier
# "the probe agrees with the classifier" was two noisy samples landing either
# side of a constant, not a signal.
#
# The payload classifier remains the only thing that actually distinguishes
# these; where it exists it is used and this probe never runs. Where it does
# not, throughput cannot substitute, so the fallback stops pretending and
# errs the way the costs point: wrongly excluding a mapping means the user
# cannot find their value at all (the patch118 bug -- 95.7% of the game), while
# wrongly including one only makes the scan slower. So the floor is set low
# enough to reject only mappings that are genuinely unusable.
_OVERSIZE_MIN_RATE    = 0.5            # MiB/s
_oversize_probe_cache: dict = {}
_oversize_probe_lock = threading.Lock()


def _oversize_region_is_scannable(ip: str, pid: int, region: dict) -> bool:
    """Whether an oversized region is readable enough to be worth scanning.

    Cached per (host, pid, region), so a scan pays at most one extra round
    trip per oversized mapping and repeat scans pay none.  An unreadable
    region is excluded: it cannot be scanned regardless of its size.

    This is a readability floor, not a GPU detector -- see
    _OVERSIZE_MIN_RATE for the measurements that ruled the latter out.
    Two samples are taken from different points because single-sample
    throughput was measured varying threefold within one mapping, and the
    faster is used: on an inconclusive read the cost asymmetry says include.
    """
    start, end = int(region.get("start", 0)), int(region.get("end", 0))
    key = (ip, int(pid), start, end)
    with _oversize_probe_lock:
        if key in _oversize_probe_cache:
            return _oversize_probe_cache[key]
    span = min(_OVERSIZE_PROBE_BYTES, max(end - start, 0))
    verdict = False
    rate = 0.0
    if span > 0:
        # Read from inside the mapping rather than its first page, which is a
        # guard page often enough to make the probe answer the wrong question.
        # Two points, not one: throughput within a single mapping was
        # measured varying from 1.9 to 5.6 MiB/s, so one sample decides
        # nothing. Keep the faster -- an inconclusive probe should include.
        errors = []
        for fraction in (3, 5):
            offset = min(span, (end - start) * fraction // 8)
            try:
                began = time.monotonic()
                data = ps5_read(ip, pid, start + offset, span)
                elapsed = max(time.monotonic() - began, 1e-9)
                rate = max(rate, (len(data) / (1024.0 * 1024.0)) / elapsed)
            except Exception as exc:
                errors.append(exc)
        if len(errors) == 2:
            add_log(f"Region {start:#x} could not be read, excluding it: "
                    f"{errors[0]}", "warn")
            verdict = False
        else:
            verdict = rate >= _OVERSIZE_MIN_RATE
    with _oversize_probe_lock:
        if len(_oversize_probe_cache) >= 256:
            _oversize_probe_cache.clear()
        _oversize_probe_cache[key] = verdict
    add_log(f"Region {start:#x}-{end:#x} "
            f"({(end - start) / (1024 ** 3):.2f} GiB) probed at {rate:.1f} MiB/s "
            f"— {'scanning it' if verdict else 'excluded as too slow'}")
    return verdict


def _invalidate_oversize_probes() -> None:
    """Region layout is process-scoped; drop verdicts when it changes."""
    with _oversize_probe_lock:
        _oversize_probe_cache.clear()


def scan_first(ip: str, pid: int, value, width: int = 4,
               aligned: bool = True, progress_cb=None,
               cancel_event=None,
               writable_only: bool = True,
               value_type: Optional[str] = None,
               region_scope: Optional[str] = None) -> np.ndarray:
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
    type_key = _normalise_value_type(value_type, width)
    if type_key == "bytes":
        raise ValueError("use scan_first_pattern for raw byte patterns")
    width = _value_width(type_key, width)
    target = _pack_typed_value(value, type_key, width)
    # Console scanners accept the unsigned bit-pattern for integer types.  This
    # gives signed scans their correct two's-complement byte representation
    # without requiring a new payload ABI.
    maps = _get_maps_cached(ip, pid)

    CHUNK        = 0x2000000   # 32 MB per request — amortises RTT 8× more than 4 MB;
                               # each request now carries ~8M uint32 values, so a
                               # single RTT covers far more heap than before.
                               # RAM budget: 12 workers × 4 slots × 32 MB = 1.5 GB
                               # max in-flight; bounded by QUEUE_DEPTH.
    SCAN_WORKERS = min(12, _MAX_CONSOLE_SOCKETS)   # bounded by the budget
                               # (ps5debug is server-side; 12 concurrent TCP streams
                               # keeps the scanner from stalling on any one RTT).
    QUEUE_DEPTH  = SCAN_WORKERS * 4   # 48 slots × 32 MB = 1.5 GB max in-flight
    _SENTINEL    = None      # signals searcher that all readers have finished

    # ── region selection ──────────────────────────────────────────────────────
    PROT_READ  = 0x1
    PROT_WRITE = 0x2
    PROT_EXEC  = 0x4
    # Fallback only: used when the payload cannot classify regions for us.
    MAX_REGION = 0x40000000

    # A blanket size cap is the wrong instrument for excluding GPU/VRAM, and
    # on a real title it is actively harmful: retail games commonly place the
    # whole managed heap in one multi-GiB cached mapping, so the cap silently
    # skipped the only memory worth scanning. Ask the payload instead, exactly
    # as the pointer scanner already does — it distinguishes a 2 GiB cached
    # game heap from a 2 GiB uncached GPU mapping, which a size test cannot.
    _uncached, _classifier_ok = _classify_regions_cached(ip, pid, maps)
    _uncached_starts = [row[0] for row in _uncached]

    def _scannable(regions, require_write):
        out = []
        for r in regions:
            if not (r['prot'] & PROT_READ):
                continue
            if require_write and not (r['prot'] & PROT_WRITE):
                continue
            if r['prot'] == PROT_EXEC:
                continue
            if _classifier_ok:
                if _region_is_uncached(r, _uncached, _uncached_starts):
                    continue
            elif ((r['end'] - r['start']) > MAX_REGION and
                  not _oversize_region_is_scannable(ip, pid, r)):
                # No classifier on this payload: measure the region rather
                # than assuming a big mapping is GPU memory.
                continue
            out.append(r)
        return out

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
    if str(region_scope or "") == "recommended":
        _before = len(scannable)
        scannable = [r for r in scannable
                     if _recommended_game_scan_region(
                         r, state.get("proc_name", ""))]
        _note_recommended_filter(len(scannable), _before, "First scan")
    # Merge adjacent mappings into single spans before chunking: fewer and
    # larger reads on RDX's slowest link. See _coalesce_scan_regions.
    scannable = _coalesce_scan_regions(scannable)
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
        add_log("First scan: no eligible memory regions"
                + ("" if _region_settings_are_default()
                   else " — your region settings excluded them all"), "warn")
        return np.empty(0, dtype=_NP_ADDR_DTYPE)

    # Select the scanner engine from the UI setting.
    engine = state.get("scan_engine", "auto")
    selected_ranges = sorted((r['start'], r['end']) for r in scannable)
    payload_exact_ok = VALUE_TYPES[type_key]["kind"] in {"uint", "sint", "float"}
    if (payload_exact_ok and engine in ("auto", "turbo") and
            os.environ.get("RDX_TURBO_SCAN", "1") != "0" and
            (_turbo_worth_probing(ip) or engine == "turbo")):
        try:
            result = ps5_scan_exact_turbo(ip, pid, value, width,
                                          selected_ranges, aligned,
                                          cancel_event, progress_cb,
                                          value_type=type_key)
            _note_turbo_outcome(ip, True)
            add_log(f"Turbo first scan completed in {max(time.monotonic()-started,1e-9):.2f}s")
            return result
        except InterruptedError:
            raise
        except Exception as exc:
            _note_turbo_outcome(ip, False)
            add_log(f"TurboScan unavailable ({exc})", "warn")
            if engine == "turbo":
                raise
    elif engine == "turbo" and not payload_exact_ok:
        raise ValueError(f"Turbo-only scanning does not support {type_key}; use Auto or Host")
    if payload_exact_ok and engine in ("auto", "console"):
        # ps5debug-NG builds exist that acknowledge CMD_PROC_SCAN with
        # STATUS_SUCCESS and then never send a single result byte, so the
        # only way to discover it is to wait out _recv_exact_cancel's 15 s
        # inactivity budget. Learn that once per host rather than paying it
        # on every scan, the same way MemDBG's PROCESS_MAPS_V2 probe does.
        # _clear_scan_state() drops the cache, so loading a different payload
        # gets a fresh probe.
        with _console_scan_lock:
            known_bad = _console_scan_supported.get(ip) is False
        if known_bad:
            if engine == "console":
                raise RuntimeError(
                    "this payload accepts the console scan command but never "
                    "returns results; use Auto or Host")
        else:
            try:
                result = ps5_scan_exact_server(ip, pid, value, width,
                                               selected_ranges, aligned,
                                               cancel_event, progress_cb,
                                               value_type=type_key)
                with _console_scan_lock:
                    _console_scan_supported[ip] = True
                add_log(f"Console first scan completed in {max(time.monotonic()-started,1e-9):.2f}s")
                return result
            except InterruptedError:
                raise
            except Exception as exc:
                with _console_scan_lock:
                    _console_scan_supported[ip] = False
                add_log(f"Console scan unavailable ({exc}); "
                        "not retrying it on this console", "warn")
                if engine == "console":
                    raise
    elif engine == "console" and not payload_exact_ok:
        raise ValueError(f"Console-only scanning does not support {type_key}; use Auto or Host")

    # ── build flat work list of (base_addr, size) chunks ─────────────────────
    # csz below already caps to the remaining region size, so small regions
    # (many PS5 mappings are 64KB-512KB) naturally get a request sized to
    # their own bytes instead of a padded, wasteful full CHUNK request.
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
                    # Pass cancel_event: without it a 32 MiB read runs to
                    # completion (or to _recv_exact_cancel's ~42 s budget)
                    # before Esc is noticed, with twelve readers typically
                    # mid-chunk. scan_first_pattern already does this.
                    data = sock.read(addr, csz, cancel_event)
                except InterruptedError:
                    break
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


def _find_pattern_offsets(data: bytes, pattern: bytes, mask: bytes,
                          absolute_base: int = 0,
                          alignment: int = 1) -> list:
    """Return pattern starts, supporting byte wildcards without regex copies."""
    if len(pattern) != len(mask) or not pattern:
        raise ValueError("invalid byte pattern")
    alignment = max(1, int(alignment))
    limit = len(data) - len(pattern)
    if limit < 0:
        return []
    if all(m == 0xFF for m in mask):
        hits = []
        pos = 0
        while pos <= limit:
            found = data.find(pattern, pos)
            if found < 0:
                break
            if (absolute_base + found) % alignment == 0:
                hits.append(found)
            pos = found + 1
        return hits

    # Anchor wildcard searches on the first concrete byte so candidate
    # generation stays in CPython's fast bytes.find implementation.
    anchor = next(i for i, m in enumerate(mask) if m)
    anchor_byte = pattern[anchor:anchor + 1]
    concrete = [(i, pattern[i]) for i, m in enumerate(mask) if m]
    hits = []
    pos = anchor
    while pos < len(data):
        found = data.find(anchor_byte, pos)
        if found < 0:
            break
        start = found - anchor
        if (0 <= start <= limit and
                (absolute_base + start) % alignment == 0 and
                all(data[start + i] == value for i, value in concrete)):
            hits.append(start)
        pos = found + 1
    return hits


# ── AOB anchoring ─────────────────────────────────────────────────────────────
# A cheat that names a raw address dies when the address moves. RDX's answer so
# far has been the pointer chain, which reaches a *data* address and is the
# more capable technique -- but the trainer formats cannot express one
# (UPSTREAM_AUDIT_PASS8) and a chain does not survive a game update
# (PASS7).
#
# The ecosystem's answer, and the default in mainstream tooling, is to anchor on
# a unique array of bytes instead. From a published PS5 walkthrough
# (info/reference/mw-guide.txt, ~12:20): after patching an instruction the tool
# asks whether to find it in future by a unique array of bytes or by a static
# address, and recommends AOB because the PS5 uses ASLR and because an AOB
# anchor is likely to keep working on other versions of the game.
#
# The constraint that walkthrough leaves implicit is the important one: it is
# anchoring an *instruction*. Code bytes are stable. Data bytes are not, and an
# AOB captured over a writable mapping re-finds nothing the moment the value it
# contains changes -- which for a cheat target is immediately, by definition.
#
# So capture is refused on writable memory rather than producing an anchor that
# silently rots. A signature is only offered where it can actually hold.
_AOB_SIGNATURE_BYTES = 32          # window captured around the anchor point
_AOB_MIN_LITERAL_BYTES = 8         # below this a "unique" match is luck


# ── instruction patching ──────────────────────────────────────────────────────
# The one link the AOB anchor chain had no implementation for, and it is needed
# twice: once after the watchpoint identifies the writing instruction, and again
# after relocation in a later run.
#
# It cannot use _target_checked_write. That guard validates the address against
# a *writable* mapping, which is exactly right for a value cheat and exactly
# wrong here: instructions live in r-x memory, and patching them is a
# deliberate write to non-writable pages. So this path validates the opposite
# property -- the target must be executable -- rather than bypassing the check.
#
# The dangerous failure is patching the wrong instruction: a plausible-looking
# anchor, a confident match, and a NOP written over something else. Every step
# therefore refuses rather than guesses.
_X86_NOP = 0x90


def nop_bytes(length: int) -> bytes:
    """`length` single-byte NOPs."""
    if int(length) <= 0:
        raise ValueError("patch length must be positive")
    return bytes([_X86_NOP]) * int(length)


def _instruction_region(ip: str, pid: int, address: int,
                        length: int, maps: Optional[list] = None):
    """The executable region wholly containing [address, address+length)."""
    maps = maps if maps is not None else _get_maps_cached(ip, pid)
    starts, rows = _build_region_lookup(maps)
    region = _region_for_addr(int(address), starts, rows)
    if not region:
        return None, "address is not in any mapped region"
    if not int(region.get("prot", 0)) & 0x4:
        return None, "address is not in an executable region"
    if int(address) + int(length) > int(region.get("end", 0)):
        # Never let a patch run off the end of the code it belongs to.
        return None, "patch would extend past the end of the region"
    return region, None


def patch_instruction(ip: str, pid: int, address: int, new_bytes: bytes,
                      expected_original: bytes,
                      maps: Optional[list] = None) -> dict:
    """Overwrite an instruction, but only if it is still the expected one.

    `expected_original` must be the exact bytes currently at `address`, and
    `new_bytes` must be the same length: a patch that is shorter would leave
    part of the old instruction behind, and one that is longer would run into
    the next. Callers get the length from the disassembler; this function will
    not infer an instruction boundary, because guessing one is how the wrong
    thing gets overwritten.

    An AOB match is not on its own proof that the site is correct, so the
    original bytes are re-read and compared immediately before writing.
    """
    out = {"ok": False, "address": int(address), "stage": "", "note": ""}
    expected_original = bytes(expected_original)
    new_bytes = bytes(new_bytes)
    if not expected_original or len(new_bytes) != len(expected_original):
        out["stage"] = "length-mismatch"
        out["note"] = (f"patch is {len(new_bytes)} bytes but the instruction "
                       f"is {len(expected_original)}; refusing to partially "
                       f"overwrite an instruction")
        return out
    region, why = _instruction_region(ip, pid, address, len(new_bytes), maps)
    if region is None:
        out["stage"] = "not-patchable"
        out["note"] = why
        return out
    try:
        live = ps5_read(ip, pid, int(address), len(expected_original))
    except Exception as exc:
        out["stage"] = "unreadable"
        out["note"] = str(exc)
        return out
    if live != expected_original:
        out["stage"] = "bytes-changed"
        out["note"] = (f"expected {expected_original.hex().upper()} but found "
                       f"{live.hex().upper()}; the instruction is not the one "
                       f"this patch was made for")
        return out
    try:
        # Deliberately not _target_checked_write: see the note above.
        if not ps5_write(ip, pid, int(address), new_bytes):
            out["stage"] = "write-rejected"
            out["note"] = "the payload refused the write"
            return out
        readback = ps5_read(ip, pid, int(address), len(new_bytes))
    except Exception as exc:
        out["stage"] = "write-failed"
        out["note"] = str(exc)
        return out
    if readback != new_bytes:
        out["stage"] = "verify-failed"
        out["note"] = (f"wrote {new_bytes.hex().upper()} but read back "
                       f"{readback.hex().upper()}")
        return out
    out.update(ok=True, stage="patched", original=expected_original.hex().upper(),
               applied=new_bytes.hex().upper(),
               module=str(region.get("name", "") or ""))
    add_log(f"Patched {int(address):#x}: {expected_original.hex().upper()} -> "
            f"{new_bytes.hex().upper()}")
    return out


def restore_instruction(ip: str, pid: int, address: int,
                        original_bytes: bytes, applied_bytes: bytes,
                        maps: Optional[list] = None) -> dict:
    """Put an instruction back. The inverse of patch_instruction.

    `applied_bytes` is what the patch wrote and must still be present, so a
    restore cannot clobber a change made by something else after the patch.
    It is a required argument rather than an assumed run of NOPs: a caller
    that applied a non-NOP patch would otherwise be unable to restore, and
    defaulting to NOPs would make the guard pass for the wrong reason.
    """
    return patch_instruction(ip, pid, address, bytes(original_bytes),
                             bytes(applied_bytes), maps)


def capture_aob_signature(ip: str, pid: int, address: int,
                          span: int = _AOB_SIGNATURE_BYTES,
                          maps: Optional[list] = None) -> Optional[dict]:
    """Capture a relocatable byte signature covering `address`, or None.

    Returns {pattern, mask, lead} where `lead` is how far into the pattern the
    anchor sits, so a later match can be converted back to an address.

    None -- deliberately, not an exception -- when the site cannot support an
    anchor: a writable mapping, an unmapped address, or a window too uniform
    to identify anything.
    """
    maps = maps if maps is not None else _get_maps_cached(ip, pid)
    starts, rows = _build_region_lookup(maps)
    region = _region_for_addr(int(address), starts, rows)
    if not region:
        return None
    if int(region.get("prot", 0)) & 0x2:
        # Writable: the bytes around a live value are the live value. An
        # anchor captured here stops matching as soon as the game writes.
        return None
    # The window must lie entirely inside the region. patch132 clamped only
    # its lower edge, so an anchor near the end of a mapping read past it --
    # measured: an anchor at 0x400038 in a region ending at 0x400040 issued a
    # read of 0x400028..0x400048, eight bytes over, and returned a signature.
    #
    # That is not merely a bad read. The writability check above applies to the
    # anchor's region; bytes pulled from the *neighbouring* mapping were never
    # checked, and may be writable and volatile. A signature containing them
    # stops matching for exactly the reason this function refuses writable
    # memory in the first place.
    region_start = int(region.get("start", 0))
    region_end = int(region.get("end", 0))
    address = int(address)
    if not (region_start <= address < region_end):
        return None
    start = max(region_start, address - span // 2)
    end = min(region_end, start + span)
    start = max(region_start, end - span)   # pull back off the far edge
    actual = end - start
    if actual < _AOB_MIN_LITERAL_BYTES:
        return None                          # region too small to anchor in
    lead = address - start
    if not 0 <= lead < actual:
        return None
    try:
        window = ps5_read(ip, pid, start, actual)
    except Exception:
        return None
    if len(window) != actual or len(set(window)) < 4:
        # A run of identical bytes matches everywhere and anchors nothing.
        return None
    return {"pattern": window.hex().upper(),
            "mask": "FF" * actual,
            "lead": int(lead)}


def aob_signature_matches(ip: str, pid: int, address: int,
                          signature: dict) -> bool:
    """Whether `signature` still describes the bytes at `address`."""
    try:
        pattern = bytes.fromhex(str(signature["pattern"]))
        mask = bytes.fromhex(str(signature["mask"]))
        lead = int(signature["lead"])
    except (KeyError, TypeError, ValueError):
        return False
    if len(pattern) != len(mask) or not pattern:
        return False
    start = int(address) - lead
    if start < 0:
        return False
    try:
        live = ps5_read(ip, pid, start, len(pattern))
    except Exception:
        return False
    return all(m == 0 or a == b
               for a, b, m in zip(live, pattern, mask))


def relocate_by_aob_signature(ip: str, pid: int, signature: dict,
                              cancel_event=None) -> Optional[int]:
    """Re-find an anchor's address after it moved, or None.

    A signature that matches in more than one place is refused. The
    walkthrough this follows is explicit that the array of bytes must be
    *unique*; two matches means the anchor cannot say which site it meant, and
    guessing would relocate a cheat onto the wrong instruction.
    """
    try:
        pattern = bytes.fromhex(str(signature["pattern"]))
        mask = bytes.fromhex(str(signature["mask"]))
        lead = int(signature["lead"])
    except (KeyError, TypeError, ValueError):
        return None
    if sum(1 for m in mask if m) < _AOB_MIN_LITERAL_BYTES:
        add_log("AOB anchor has too few fixed bytes to be unique; not "
                "relocating", "warn")
        return None
    hits = np.asarray(scan_first_pattern(ip, pid, pattern, mask,
                                         cancel_event=cancel_event,
                                         writable_only=False,
                                         region_scope="executable"))
    if hits.size == 0:
        return None
    if hits.size > 1:
        add_log(f"AOB anchor matched {hits.size} sites; refusing to guess "
                f"which one the cheat meant", "warn")
        return None
    return int(hits[0]) + lead


# ── instruction anchors: trace -> AOB -> relocation -> patch ─────────────────
#
# The acquisition half of this pipeline needs a live debugger session; the
# patching half does not.  Keeping them separable means an anchor captured once
# can be re-verified and applied later without attaching again -- which matters
# because a target process allows exactly one successful attach per lifetime.

_ANCHOR_VERSION = 1


def _instruction_anchor_contract(trace: dict, width: int = 4) -> dict:
    """Canonical description of one traced write.

    `writer` is *the* instruction address.  The raw trap RIP is carried as
    `trap_rip` for diagnostics only: x86 data breakpoints are trap-type, so it
    names the instruction after the store and is never the anchor.  A trace
    without a resolved writer raises rather than degrading to the trap RIP.
    """
    if not isinstance(trace, dict):
        raise TypeError("trace must be a dict")
    writer = trace.get("writer")
    if writer is None:
        raise KeyError("trace has no resolved writer address")
    insn = trace.get("instruction") or {}
    base_value = int(trace.get("base_value", 0))
    index_value = int(trace.get("index_value", 0))
    scale = int(trace.get("scale", 1) or 1)
    disp = int(trace.get("final_offset", 0))
    return {
        "version": _ANCHOR_VERSION,
        "temporary_address": int(trace["target"]),
        "writer": int(writer),
        "trap_rip": int(trace.get("rip", 0)),
        "instruction_length": int(insn.get("length", 0)),
        "instruction_bytes": "",
        "base_reg": trace.get("base_reg"),
        "base_value": base_value,
        "index_reg": trace.get("index_reg"),
        "index_value": index_value,
        "scale": scale,
        "displacement": disp,
        "effective_address": base_value + index_value * scale + disp,
        "access_width": int(width),
        "access_mode": str(trace.get("access_mode", "write")),
        "lwpid": int(trace.get("lwpid", 0)),
        "signature": None,
        "relocated": None,
        "verified": False,
    }


def capture_instruction_anchor(ip: str, pid: int, trace: dict, width: int = 4,
                               cancel_event=None,
                               maps: Optional[list] = None) -> dict:
    """Turn a verified write trace into a relocatable instruction anchor.

    Runs the two proven steps in order -- signature capture at the writer, then
    a uniqueness-checked relocation that must land back on it.  Every failure
    aborts; none of them falls back to the trap RIP or invents an anchor.
    """
    out = {"ok": False, "stage": "", "note": "", "anchor": None}
    try:
        anchor = _instruction_anchor_contract(trace, width)
    except (KeyError, TypeError, ValueError) as exc:
        out["stage"] = "no-writer"
        out["note"] = (f"the trace carries no resolved writer ({exc}); the raw "
                       f"trap RIP is not a substitute")
        return out
    if anchor["effective_address"] != anchor["temporary_address"]:
        out["stage"] = "operand-mismatch"
        out["note"] = (f"the decoded operand resolves to "
                       f"{hex(anchor['effective_address'])}, not the watched "
                       f"{hex(anchor['temporary_address'])}")
        return out
    length = int(anchor["instruction_length"])
    if length <= 0:
        out["stage"] = "unknown-length"
        out["note"] = "the trace did not report an instruction length"
        return out
    maps = maps if maps is not None else _get_maps_cached(ip, pid)
    try:
        original = ps5_read(ip, pid, anchor["writer"], length)
    except Exception as exc:
        original = None
        out["note"] = str(exc)
    if not original or len(original) != length:
        out["stage"] = "unreadable"
        out["note"] = out["note"] or "could not read the writer instruction"
        return out
    anchor["instruction_bytes"] = original.hex().upper()

    signature = capture_aob_signature(ip, pid, anchor["writer"], maps=maps)
    if not signature:
        out["stage"] = "capture-refused"
        out["note"] = ("no signature could be captured there -- the window is "
                       "writable, unmapped, or too uniform to be unique")
        return out
    anchor["signature"] = dict(signature)

    relocated = relocate_by_aob_signature(ip, pid, signature,
                                          cancel_event=cancel_event)
    if relocated is None:
        out["stage"] = "not-unique"
        out["note"] = ("the signature did not match exactly once; an anchor "
                       "that cannot say which site it meant is not usable")
        return out
    if int(relocated) != int(anchor["writer"]):
        out["stage"] = "relocation-mismatch"
        out["note"] = (f"the signature relocated to {hex(int(relocated))} but "
                       f"was captured at {hex(anchor['writer'])}")
        return out
    anchor["relocated"] = int(relocated)
    anchor["verified"] = True
    out.update(ok=True, stage="anchored", anchor=anchor,
               note=f"anchored at {hex(anchor['writer'])}")
    return out


def verify_instruction_anchor(ip: str, pid: int, anchor: dict,
                              cancel_event=None,
                              maps: Optional[list] = None) -> dict:
    """Re-prove an anchor against the live process. No write happens without it.

    Checked, in order: the signature still relocates to exactly one site; the
    bytes there are the instruction that was captured; the mapping is
    executable and not writable.  Returns the address only when all of them
    hold, so a caller cannot accidentally patch on a partial result.
    """
    out = {"ok": False, "stage": "", "note": "", "address": None,
           "match_count": None}
    if not isinstance(anchor, dict) or not anchor.get("signature"):
        out["stage"] = "no-signature"
        out["note"] = "this anchor carries no captured signature"
        return out
    expected = str(anchor.get("instruction_bytes") or "")
    if not expected:
        out["stage"] = "no-instruction-bytes"
        out["note"] = "this anchor recorded no original instruction bytes"
        return out
    try:
        original = bytes.fromhex(expected)
    except ValueError:
        out["stage"] = "no-instruction-bytes"
        out["note"] = "the recorded instruction bytes are not valid hex"
        return out

    address = relocate_by_aob_signature(ip, pid, anchor["signature"],
                                        cancel_event=cancel_event)
    if address is None:
        out["stage"] = "not-unique"
        out["note"] = ("the signature no longer matches exactly one site; "
                       "refusing to guess which instruction was meant")
        return out
    out["match_count"] = 1

    maps = maps if maps is not None else _get_maps_cached(ip, pid)
    region = next((r for r in maps
                   if int(r.get("start", 0)) <= address < int(r.get("end", 0))),
                  None)
    if region is None:
        out["stage"] = "unmapped"
        out["note"] = f"{hex(address)} is not in any mapping"
        return out
    prot = int(region.get("prot", 0))
    if not prot & 0x4:
        out["stage"] = "not-executable"
        out["note"] = f"{hex(address)} is not in executable memory"
        return out
    if prot & 0x2:
        out["stage"] = "writable-region"
        out["note"] = (f"{hex(address)} is in writable memory, so the bytes "
                       f"there are not a stable code anchor")
        return out
    try:
        live = ps5_read(ip, pid, address, len(original))
    except Exception as exc:
        live = None
        out["note"] = str(exc)
    if not live or len(live) != len(original):
        out["stage"] = "unreadable"
        out["note"] = out["note"] or "could not read the relocated instruction"
        return out
    if live != original:
        out["stage"] = "bytes-changed"
        out["note"] = (f"expected {original.hex().upper()} at {hex(address)}, "
                       f"found {live.hex().upper()}")
        return out
    out.update(ok=True, stage="verified", address=int(address),
               note=f"verified at {hex(int(address))}")
    return out


def patch_instruction_anchor(ip: str, pid: int, anchor: dict,
                             new_bytes: Optional[bytes] = None,
                             cancel_event=None,
                             maps: Optional[list] = None) -> dict:
    """Patch the instruction an anchor names, and only after re-proving it.

    A matching AOB on its own is not authority to write: the anchor is
    re-verified against live memory first, and a failed check means no write at
    all.  `new_bytes` defaults to NOPs of exactly the instruction's length.
    """
    verdict = verify_instruction_anchor(ip, pid, anchor,
                                        cancel_event=cancel_event, maps=maps)
    if not verdict["ok"]:
        return {"ok": False, "stage": verdict["stage"], "address": None,
                "note": f"not patched: {verdict['note']}",
                "verification": verdict}
    original = bytes.fromhex(str(anchor["instruction_bytes"]))
    payload = nop_bytes(len(original)) if new_bytes is None else bytes(new_bytes)
    result = patch_instruction(ip, pid, verdict["address"], payload, original,
                               maps=maps)
    result["verification"] = verdict
    result["applied"] = payload.hex().upper()
    return result


def restore_instruction_anchor(ip: str, pid: int, anchor: dict,
                               applied_bytes: bytes,
                               maps: Optional[list] = None) -> dict:
    """Undo a patch applied through `patch_instruction_anchor`."""
    verdict = verify_instruction_anchor(ip, pid, anchor, maps=maps)
    address = verdict.get("address")
    if address is None:
        # The signature cannot be re-found because the patch changed the very
        # bytes it describes.  Fall back to the recorded site, which is where
        # the patch was written.
        address = anchor.get("relocated") or anchor.get("writer")
    if address is None:
        return {"ok": False, "stage": "no-address", "note": "nowhere to restore"}
    original = bytes.fromhex(str(anchor["instruction_bytes"]))
    return restore_instruction(ip, pid, int(address), original,
                               bytes(applied_bytes), maps)


def anchor_to_json(anchor: dict) -> str:
    """Serialise an anchor so acquisition and patching can be separate runs."""
    return json.dumps(anchor, indent=2, sort_keys=True)


def anchor_from_json(blob: str) -> dict:
    """Load an anchor artifact, refusing one this build cannot interpret."""
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("anchor artifact is not an object")
    if int(data.get("version", 0)) != _ANCHOR_VERSION:
        raise ValueError(f"unsupported anchor version {data.get('version')!r}")
    for key in ("writer", "signature", "instruction_bytes"):
        if not data.get(key):
            raise ValueError(f"anchor artifact is missing {key!r}")
    return data


def scan_first_pattern(ip: str, pid: int, pattern: bytes, mask: bytes,
                       alignment: int = 1, progress_cb=None,
                       cancel_event=None,
                       writable_only: bool = False,
                       region_scope: Optional[str] = None) -> np.ndarray:
    """Host-side AOB scan with ``??`` wildcard support and bounded memory."""
    pattern = bytes(pattern)
    mask = bytes(mask)
    if not pattern or len(pattern) != len(mask) or len(pattern) > 256:
        raise ValueError("invalid byte pattern")
    if not any(mask):
        raise ValueError("a pattern cannot consist entirely of wildcards")
    if cancel_event is None:
        cancel_event = threading.Event()
        cancel_event.truncated = False

    maps = _get_maps_cached(ip, pid)
    # Same classifier-over-size-cap correction as scan_first's _scannable.
    uncached, classifier_ok = _classify_regions_cached(ip, pid, maps)
    uncached_starts = [row[0] for row in uncached]
    regions = [r for r in maps
               if int(r.get("end", 0)) > int(r.get("start", 0))
               and (not _region_is_uncached(r, uncached, uncached_starts)
                    if classifier_ok else
                    int(r.get("end", 0)) - int(r.get("start", 0)) <= 0x40000000)
               and (int(r.get("prot", 0)) & 0x1)
               and (not writable_only or (int(r.get("prot", 0)) & 0x2))]
    if str(region_scope or "") == "executable":
        # An instruction anchor can only live in executable memory.  Measured
        # on hardware: 47.0 MiB of r-x against 9,765 MiB readable, so 99.52%
        # of the bytes read could never hold a match.  Narrowing here cannot
        # lose the true site -- capture_aob_signature refuses writable memory,
        # so every signature it produces was captured in an executable
        # mapping -- and it drops incidental copies sitting in data buffers,
        # which are not candidate writers.
        regions = [r for r in regions if int(r.get("prot", 0)) & 0x4]
    if str(region_scope or "") == "recommended":
        _before = len(regions)
        regions = [r for r in regions if _recommended_game_scan_region(
            r, state.get("proc_name", ""))]
        _note_recommended_filter(len(regions), _before, "AOB scan")
    # Merge before chunking: a pattern straddling two adjacent mappings was
    # unreachable while each region was chunked in isolation.
    regions = _coalesce_scan_regions(regions)
    regions.sort(key=lambda r: int(r["end"]) - int(r["start"]), reverse=True)
    total_bytes = max(sum(int(r["end"]) - int(r["start"])
                          for r in regions), 1)
    if not regions:
        if progress_cb:
            progress_cb(1, 1)
        return np.empty(0, dtype=_NP_ADDR_DTYPE)

    chunk_size = 0x400000  # 4 MiB: responsive cancellation, modest RAM.
    work = []
    for region in regions:
        start, end = int(region["start"]), int(region["end"])
        cursor = start
        while cursor < end:
            primary_end = min(cursor + chunk_size, end)
            read_end = min(primary_end + len(pattern) - 1, end)
            work.append((cursor, read_end - cursor, primary_end))
            cursor = primary_end

    work_index = [0]
    done_bytes = [0]
    found = []
    lock = threading.Lock()
    errors = []

    def worker():
        sock = None
        try:
            sock = _ScanSocket(ip, pid)
            while not cancel_event.is_set():
                with lock:
                    if work_index[0] >= len(work):
                        break
                    address, size, primary_end = work[work_index[0]]
                    work_index[0] += 1
                try:
                    data = sock.read(address, size, cancel_event)
                    if len(data) != size:
                        raise IOError(f"partial read: {len(data)}/{size}")
                    local = [address + off for off in _find_pattern_offsets(
                        data, pattern, mask, address, alignment)
                             if address + off < primary_end]
                    with lock:
                        remaining = MAX_SCAN_RESULTS - len(found)
                        if remaining > 0:
                            found.extend(local[:remaining])
                        if len(local) > remaining:
                            cancel_event.truncated = True
                            cancel_event.set()
                except InterruptedError:
                    break
                except Exception as exc:
                    with lock:
                        if len(errors) < 20:
                            errors.append(f"{hex(address)}: {exc}")
                finally:
                    with lock:
                        done_bytes[0] += min(size, chunk_size)
                        if progress_cb:
                            progress_cb(min(done_bytes[0], total_bytes),
                                        total_bytes)
        finally:
            if sock:
                sock.close()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(min(6, max(1, len(work))))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for error in errors:
        add_log(f"AOB read skipped at {error}", "warn")
    if progress_cb and not cancel_event.is_set():
        progress_cb(total_bytes, total_bytes)
    result = np.asarray(sorted(set(found)), dtype=_NP_ADDR_DTYPE)
    add_log(f"AOB scan: {len(result):,} matches"
            f"{' (truncated)' if getattr(cancel_event, 'truncated', False) else ''}")
    return result


def scan_next_pattern(ip: str, pid: int, pattern: bytes, mask: bytes,
                      prev: np.ndarray, cancel_event=None,
                      progress_cb=None) -> np.ndarray:
    """Revalidate AOB candidates without rescanning unrelated memory."""
    pattern, mask = bytes(pattern), bytes(mask)
    if not pattern or len(pattern) != len(mask):
        raise ValueError("invalid byte pattern")
    if cancel_event is None:
        cancel_event = threading.Event()
    addresses = np.asarray(prev, dtype=_NP_ADDR_DTYPE)
    index = [0]
    found = []
    lock = threading.Lock()

    def worker():
        sock = None
        try:
            sock = _ScanSocket(ip, pid)
            while not cancel_event.is_set():
                with lock:
                    if index[0] >= len(addresses):
                        break
                    item_index = index[0]
                    index[0] += 1
                address = int(addresses[item_index])
                try:
                    raw = sock.read(address, len(pattern), cancel_event)
                    if (len(raw) == len(pattern) and
                            all(not m or raw[i] == pattern[i]
                                for i, m in enumerate(mask))):
                        with lock:
                            found.append(address)
                except Exception:
                    pass
                if progress_cb:
                    with lock:
                        progress_cb(index[0], max(len(addresses), 1))
        finally:
            if sock:
                sock.close()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(min(8, max(1, len(addresses))))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return np.asarray(sorted(found), dtype=_NP_ADDR_DTYPE)


def scan_next(ip: str, pid: int, value, width: int,
              prev: np.ndarray,
              cancel_event=None, progress_cb=None,
              value_type: Optional[str] = None,
              tolerance: float = 0.0) -> np.ndarray:
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
    type_key = _normalise_value_type(value_type, width)
    if type_key == "bytes":
        raise ValueError("use scan_next_pattern for raw byte patterns")
    width = _value_width(type_key, width)
    packed_target = _pack_typed_value(value, type_key, width)
    kind = VALUE_TYPES[type_key]["kind"]
    engine = state.get("scan_engine", "auto")
    # Shape, not just availability: a nearly-converged list is cheaper to
    # filter on the host than to negotiate a resident session for.
    #
    # A session that already exists forces the turbo path regardless of size,
    # and that is not an optimisation -- it is required. The server holds the
    # survivor list, so narrowing on the host while leaving that session
    # resident would let the next turbo scan re-adopt the *pre-narrowing*
    # list and silently undo this filter. (Same hazard the engine switch,
    # the result drop and the nearby browse each call _close_turbo_session
    # for.) Its setup cost is already paid, so there is nothing to save.
    with _turbo_session_lock:
        turbo_resident = _turbo_session is not None
    turbo_worth_it = (engine == "turbo" or turbo_resident
                      or len(prev) >= TURBO_MIN_SURVIVORS)
    if (kind in {"uint", "sint", "float"} and
            not (kind == "float" and float(tolerance) > 0) and
            engine in ("auto", "turbo") and turbo_worth_it):
        try:
            return ps5_scan_next_turbo(
                ip, pid, value, width, cancel_event, progress_cb,
                value_type=type_key)
        except InterruptedError:
            raise
        except Exception as exc:
            add_log(f"Resident Turbo rescan unavailable ({exc}); using host filter", "warn")
            if state.get("scan_engine") == "turbo":
                raise RuntimeError(_TURBO_ONLY_NO_SESSION.format(reason=exc)) from exc

    dtype = np.dtype(VALUE_TYPES[type_key]["dtype"])
    target = np.asarray([_unpack_typed_value(
        packed_target, type_key, width)], dtype=dtype)[0]

    # Stage 1: parallel network reads → pre-allocated ndarrays (no Python list)
    live_addrs, live_vals = ps5_read_batch(ip, pid, prev, width,
                                           cancel_event, progress_cb,
                                           value_type=type_key)

    if len(live_addrs) == 0:
        add_log(f"Exact next scan: 0 remain (no reads succeeded), "
                f"RSS {_rss_mb():.0f} MB")
        return np.empty(0, dtype=_NP_ADDR_DTYPE)

    # Stage 2: vectorised comparison — one C-level call across all N entries
    if kind == "float" and float(tolerance) > 0:
        mask = np.isfinite(live_vals) & np.isclose(
            live_vals, target, rtol=0.0, atol=float(tolerance))
    else:
        mask = live_vals == target
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
                       writable_only: bool = True,
                       value_type: Optional[str] = None,
                       region_scope: Optional[str] = None
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
    type_key = _normalise_value_type(value_type, width)
    if type_key == "bytes":
        raise ValueError("unknown-value scans require a numeric type")
    width = _value_width(type_key, width)
    value_dtype = np.dtype(VALUE_TYPES[type_key]["dtype"])
    if cancel_event is None:
        cancel_event = threading.Event()
        cancel_event.truncated = False
    maps = _get_maps_cached(ip, pid)

    CHUNK        = 0x2000000   # 32 MB — matches scan_first for consistent RTT amortisation
    SCAN_WORKERS = min(12, _MAX_CONSOLE_SOCKETS)   # bounded by the budget
    QUEUE_DEPTH  = SCAN_WORKERS * 4
    _SENTINEL    = None

    PROT_READ  = 0x1
    PROT_WRITE = 0x2
    PROT_EXEC  = 0x4
    MAX_REGION = 0x40000000   # fallback only; see scan_first's _scannable

    _uncached, _classifier_ok = _classify_regions_cached(ip, pid, maps)
    _uncached_starts = [row[0] for row in _uncached]

    def _scannable(regions, require_write):
        out = []
        for r in regions:
            if not (r['prot'] & PROT_READ):
                continue
            if require_write and not (r['prot'] & PROT_WRITE):
                continue
            if r['prot'] == PROT_EXEC:
                continue
            if _classifier_ok:
                if _region_is_uncached(r, _uncached, _uncached_starts):
                    continue
            elif ((r['end'] - r['start']) > MAX_REGION and
                  not _oversize_region_is_scannable(ip, pid, r)):
                continue
            out.append(r)
        return out

    rw_regions  = _scannable(maps, require_write=True)
    if writable_only:
        scannable = rw_regions
    else:
        ro_regions = _scannable(maps, require_write=False)
        rw_set     = {(r['start'], r['end']) for r in rw_regions}
        ro_only    = [r for r in ro_regions
                      if (r['start'], r['end']) not in rw_set]
        scannable  = rw_regions + ro_only
    if str(region_scope or "") == "recommended":
        _before = len(scannable)
        scannable = [r for r in scannable
                     if _recommended_game_scan_region(
                         r, state.get("proc_name", ""))]
        _note_recommended_filter(len(scannable), _before, "Unknown scan")
    # Phase 4a: sort largest-first — same rationale as scan_first.
    # Same merge the exact and AOB paths do; see _coalesce_scan_regions.
    scannable = _coalesce_scan_regions(scannable)
    scannable.sort(key=lambda r: r['end'] - r['start'], reverse=True)
    total_bytes = max(sum(r['end'] - r['start'] for r in scannable), 1)

    if not scannable:
        if progress_cb:
            progress_cb(1, 1)
        add_log("Unknown scan: no eligible memory regions"
                + ("" if _region_settings_are_default()
                   else " — your region settings excluded them all"), "warn")
        return (np.empty(0, dtype=_NP_ADDR_DTYPE),
                np.empty(0, dtype=value_dtype))

    # Prefer TurboScan's server-side snapshot — the console holds the
    # baseline itself instead of RDX transferring the whole region and
    # holding it in its own process — matching scan_first's exact-value
    # engine-selection pattern exactly. Falls back to the client-side
    # pipeline below on any failure (older payload, TS_SNAPSHOT engine
    # unavailable, etc.), same as scan_first's own turbo fallback.
    engine = state.get("scan_engine", "auto")
    if (engine in ("auto", "turbo") and
            os.environ.get("RDX_TURBO_SCAN", "1") != "0" and
            (_turbo_worth_probing(ip) or engine == "turbo")):
        try:
            selected_ranges = sorted((r['start'], r['end']) for r in scannable)
            addrs, vals = ps5_scan_unknown_turbo(
                ip, pid, width, selected_ranges, aligned,
                cancel_event, progress_cb, value_type=type_key)
            _note_turbo_outcome(ip, True)
            add_log("Turbo unknown-value scan completed in "
                    f"{max(time.monotonic() - started, 1e-9):.2f}s")
            return addrs, vals
        except InterruptedError:
            raise
        except Exception as exc:
            _note_turbo_outcome(ip, False)
            add_log(f"TurboScan snapshot unavailable ({exc})", "warn")
            if engine == "turbo":
                raise

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
                    data = sock.read(addr, csz, cancel_event)   # see scan_first
                except InterruptedError:
                    break
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
            val_dtype = value_dtype
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
                    data[lead:lead + aligned_n * width], dtype=value_dtype)
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
                    shape=(n_out,), dtype=value_dtype,
                    buffer=data, strides=(1,))
                addrs_out  = (addr + offsets).astype(_NP_ADDR_DTYPE)

            # Enforce the result cap
            remaining = MAX_SCAN_RESULTS - total_so_far
            if n_out > remaining:
                addrs_out  = addrs_out[:remaining]
                vals_slice = vals_slice[:remaining]
                found_addrs.append(addrs_out)
                found_values.append(vals_slice.astype(value_dtype, copy=False))
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
            found_values.append(vals_slice.astype(value_dtype, copy=False))
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
        out_values = np.empty(0, dtype=value_dtype)

    add_log(f"Unknown-scan snapshot: {len(out_addrs):,} candidates, "
            f"RSS {_rss_mb():.0f} MB")
    elapsed = max(time.monotonic() - started, 1e-9)
    processed_mb = min(done_bytes[0], total_bytes) / 1_048_576
    add_log(f"Unknown-scan transfer/decode: {processed_mb:.1f} MiB "
            f"in {elapsed:.2f}s ({processed_mb / elapsed:.1f} MiB/s)")
    return out_addrs, out_values


# Turbo-only mode deliberately refuses to degrade to the host filter. That is
# the right contract, but the underlying "no matching resident TurboScan
# session" is opaque: the usual cause is the user's own previous action
# (dropping a result, exploring nearby values, switching engine) discarding
# the console-resident list, and nothing in the bare message says so or says
# how to get moving again.
_TURBO_ONLY_NO_SESSION = (
    "{reason}. Turbo-only mode narrows the list the console is holding, and "
    "that list was discarded — dropping a result, exploring nearby values, "
    "undoing a scan or changing the scan engine all discard it. Run a new "
    "First Scan, or set Scan Settings -> engine: auto to fall back to the "
    "host filter."
)

# Relational scan modes for unknown-value next scans.
RELATIONAL_MODES = [
    "decreased",        # current < previous (e.g. took damage)
    "increased",        # current > previous (e.g. picked up health)
    "changed",          # current != previous
    "unchanged",        # current == previous (value held steady)
    "decreased by",     # current == previous - N  (known delta)
    "increased by",     # current == previous + N  (known delta)
]

def _wrapped_delta(prev_values: np.ndarray, delta, width: int,
                   dtype: np.dtype, sign: int) -> np.ndarray:
    """Return ``prev_values ± delta`` wrapped at the scanned value's own width.

    Integer game values wrap at their storage width, so the comparison has to
    as well: a u8 counter really does go from 3 to 249 when it loses 10, and
    an i8 one really does go from 100 to 44 when it gains 200.  Doing the
    arithmetic in the array's native dtype does not reproduce that — NumPy
    promotes to a wider type whenever the Python ``delta`` does not fit the
    dtype, so the sum never wraps and a legitimate match is silently dropped
    (and on NumPy 2 / NEP 50 the same expression raises OverflowError
    instead).  Compute in uint64, mask to ``width``, then reinterpret through
    the unsigned view of the scan dtype so signed types land back on their
    correct two's-complement value.
    """
    step = np.uint64(int(delta) & 0xFFFF_FFFF_FFFF_FFFF)
    wide = prev_values.astype(np.uint64)
    # Wrapping here is the entire point of the function, so it must not be
    # reported as an error once install_warning_router() makes overflow warn.
    with np.errstate(over="ignore"):
        wide = (wide + step) if sign > 0 else (wide - step)
    masked = wide & np.uint64(WIDTH_MAX[width])
    return masked.astype(np.dtype(f"<u{width}")).view(dtype)


def scan_next_relational(ip: str, pid: int, width: int,
                         prev_addrs: np.ndarray,
                         prev_values: np.ndarray,
                         mode: str,
                         delta=0,
                         cancel_event=None,
                         progress_cb=None,
                         value_type: Optional[str] = None,
                         tolerance: float = 0.0) -> tuple:
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
    type_key = _normalise_value_type(value_type, width)
    if type_key == "bytes":
        raise ValueError("relational scans require a numeric type")
    width = _value_width(type_key, width)
    spec = VALUE_TYPES[type_key]
    kind = spec["kind"]
    dtype = np.dtype(spec["dtype"])

    # Prefer narrowing a resident TurboScan snapshot session server-side —
    # matches scan_next's own exact-value engine-selection pattern,
    # including its same float-tolerance gate: the server's relational
    # compareTypes do an exact comparison with no tolerance concept, so a
    # nonzero float tolerance must keep using the host path, which honours
    # it. Falls back to the client-driven read-and-compare below whenever
    # there's no matching resident session (e.g. the First Scan that built
    # `prev_addrs`/`prev_values` ran on the host path) or any other failure.
    if (mode in _SNAPSHOT_COMPARE_TYPE and
            not (kind == "float" and float(tolerance) > 0) and
            state.get("scan_engine", "auto") in ("auto", "turbo") and
            (_turbo_worth_probing(ip) or
             state.get("scan_engine") == "turbo")):
        try:
            return ps5_scan_relational_turbo(
                ip, pid, width, mode, delta, cancel_event, progress_cb,
                value_type=type_key)
        except InterruptedError:
            raise
        except Exception as exc:
            add_log(f"Resident Turbo snapshot rescan unavailable ({exc}); "
                    "using host filter", "warn")
            if state.get("scan_engine") == "turbo":
                raise RuntimeError(_TURBO_ONLY_NO_SESSION.format(reason=exc)) from exc

    # prev_addrs must be sorted for searchsorted.
    if len(prev_addrs) > 1 and not np.all(prev_addrs[:-1] <= prev_addrs[1:]):
        order       = np.argsort(prev_addrs, kind='stable')
        prev_addrs  = prev_addrs[order]
        prev_values = prev_values[order]

    # ps5_read_batch now returns (live_addrs, live_vals) ndarrays directly —
    # no Python list of (addr, bytes) tuples, no per-address decode loop.
    live_addrs, live_vals = ps5_read_batch(ip, pid, prev_addrs, width,
                                           cancel_event, progress_cb,
                                           value_type=type_key)
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
    if _NUMBA_OK and kind == "uint" and mode in RELATIONAL_MODE_IDS:
        # Phase 3 fast path: Numba compiles _nb_relational_mask to native
        # LLVM code on first call (cached to disk after that).  Subsequent
        # calls skip compilation entirely.  The parallel=True flag enables
        # OpenMP on Linux/macOS and TBB on Windows — all CPU cores, GIL-free.
        #
        # "decreased by" / "increased by" use uint64 arithmetic; cast both
        # arrays so the Numba kernel works with a uniform dtype. Pass the
        # value's own width mask so the kernel wraps at the scanned width
        # (e.g. 8 bits for u8) instead of at the full 64 bits it computes in.
        # delta/width_mask must be np.uint64, not plain Python int: Numba
        # infers a plain int as signed int64, and mixing that with a uint64
        # array element in `p - delta` silently produces a signed result
        # that only happens to survive a narrower mask by coincidence of
        # two's-complement bit patterns — at width=8 (the full 64 bits, no
        # narrowing) it does not, and the comparison against `cur` fails.
        keep = _nb_relational_mask(
            cur.astype(np.uint64),
            prv.astype(np.uint64),
            RELATIONAL_MODE_IDS[mode],
            np.uint64(int(delta) & 0xFFFF_FFFF_FFFF_FFFF),
            np.uint64(WIDTH_MAX[width]),
        ).astype(bool)
    else:
        # Phase 3 fallback: pure NumPy (always correct, single-threaded SIMD).
        if mode == "decreased":
            keep = cur < prv
        elif mode == "increased":
            keep = cur > prv
        elif mode == "changed":
            keep = cur != prv
        elif mode == "unchanged":
            if kind == "float" and float(tolerance) > 0:
                keep = np.isfinite(cur) & np.isfinite(prv) & np.isclose(
                    cur, prv, rtol=0.0, atol=float(tolerance))
            else:
                keep = cur == prv
        elif mode == "decreased by":
            if kind in ("uint", "sint"):
                expected = _wrapped_delta(prv, delta, width, dtype, -1)
            else:
                expected = prv - delta
            keep = (np.isclose(cur, expected, rtol=0.0,
                               atol=float(tolerance))
                    if kind == "float" and float(tolerance) > 0
                    else cur == expected)
        elif mode == "increased by":
            if kind in ("uint", "sint"):
                expected = _wrapped_delta(prv, delta, width, dtype, +1)
            else:
                expected = prv + delta
            keep = (np.isclose(cur, expected, rtol=0.0,
                               atol=float(tolerance))
                    if kind == "float" and float(tolerance) > 0
                    else cur == expected)
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
    flags = int(region.get("flags", 0))
    PROT_READ, PROT_WRITE, PROT_EXEC = 0x1, 0x2, 0x4

    low_name = name.lower()
    # MemDBG preserves the native VM object type in the high flag byte.  A
    # readable vnode/file mapping is image-backed even when the PS5 omitted its
    # filename; this is stronger evidence than the writable/name heuristics.
    if ((flags >> 24) & 0xFF) == 2 and (prot & PROT_READ):
        return True
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
                       low_name.startswith("libsce") or
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
                         int(r.get("prot", 0)), int(r.get("flags", 0)),
                         int(r.get("offset", 0)), str(r.get("name", "")))
                        for r in maps))


def _module_info_for_addr(addr: int, maps: list) -> tuple:
    """Return (module_name, module_base, relative_offset) for a static address."""
    region_starts, region_rows = _build_region_lookup(maps)
    containing = _region_for_addr(int(addr), region_starts, region_rows)
    if containing is None:
        return ("", 0, 0)
    name = str(containing.get("name", "") or "")
    static_maps = [r for r in maps if _is_static_region(r)]
    # MemDBG must synthesize names such as ``[file]`` when the kernel does not
    # expose a vnode path.  Treating every such row as one giant module made a
    # root in a library rebase from the first file mapping in the process.
    # PS4Cheater's proven representation is section-relative (section ID plus
    # offset), so use a stable metadata/ordinal section identity for generic
    # rows while retaining friendlier module-relative roots for named images.
    if _is_static_region(containing) and _is_generic_map_name(name):
        section_id = _section_module_identity(containing, static_maps)
        return (section_id, int(containing["start"]),
                int(addr - int(containing["start"])))
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
    """Build an overlap-safe binary-searchable region table.

    PS5 map enumeration may contain a large reservation plus smaller augmented
    page-table rows.  A plain ``bisect(start)`` can land on a short overlapping
    row that does not contain the address and incorrectly report it unmapped.
    Prefix maximum ends let lookup walk back only while an earlier row can
    still cover the address.
    """
    raw_rows = sorted(
        (int(r.get("start", 0)), int(r.get("end", 0)), r)
        for r in maps if int(r.get("end", 0)) > int(r.get("start", 0))
    )
    rows = []
    max_end = 0
    for start, end, region in raw_rows:
        max_end = max(max_end, end)
        rows.append((start, end, region, max_end))
    starts = [x[0] for x in rows]
    return starts, rows


def _region_for_addr(addr: int, region_starts: list, region_rows: list):
    """Overlap-safe memory-map ownership lookup, preferring static identity."""
    i = bisect.bisect_right(region_starts, int(addr)) - 1
    matches = []
    while i >= 0:
        start, end, region, prefix_max_end = region_rows[i]
        if start <= int(addr) < end:
            matches.append(region)
        i -= 1
        if i < 0 or region_rows[i][3] <= int(addr):
            break
    if not matches:
        return None
    return max(matches, key=lambda r: (_is_static_region(r),
                                       _region_priority(r),
                                       -(int(r["end"]) - int(r["start"]))))


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


_SECTION_MODULE_PREFIX = "@section:"


def _is_generic_map_name(name: str) -> bool:
    """Whether a map label identifies a backing type rather than a module."""
    low = str(name or "").strip().lower()
    return (not low or low in {"anon", "heap", "[file]", "[vnode]",
                               "[untyped]", "[default]", "[unknown]"})


def _section_signature(region: dict) -> tuple:
    """Stable fields available in both ps5debug and MemDBG map records."""
    return (int(region.get("prot", 0)),
            (int(region.get("flags", 0)) >> 24) & 0xFF,
            int(region.get("end", 0)) - int(region.get("start", 0)))


def _section_module_identity(region: dict, static_maps: list) -> str:
    """Encode PS4Cheater-style section identity without persisting an address."""
    prot, map_type, size = _section_signature(region)
    peers = sorted((r for r in static_maps
                    if _section_signature(r) == (prot, map_type, size)
                    and _is_generic_map_name(str(r.get("name", "") or ""))),
                   key=lambda r: (int(r["start"]), int(r["end"])))
    ordinal = 0
    wanted_bounds = (int(region["start"]), int(region["end"]))
    for i, peer in enumerate(peers):
        if (int(peer["start"]), int(peer["end"])) == wanted_bounds:
            ordinal = i
            break
    return (f"{_SECTION_MODULE_PREFIX}{prot:x}:{map_type:x}:"
            f"{size:x}:{ordinal}")


def _build_region_priority_intervals(maps: list):
    """Return disjoint mapped intervals carrying the best overlap priority.

    A pointer value only needs to land in *some* mapped row.  Using a simple
    ``searchsorted(starts)`` against overlapping VM records can select a short
    overlay and reject an address that is still covered by a larger mapping.
    This sweep converts the map into non-overlapping intervals once, allowing
    the RAM and disk reverse indexes to keep their vectorized hot paths.
    """
    events = {}
    priorities = {}
    row_id = 0
    for region in maps:
        start = int(region.get("start", 0))
        end = int(region.get("end", 0))
        if end <= start:
            continue
        priorities[row_id] = int(_region_priority(region))
        events.setdefault(start, [[], []])[0].append(row_id)  # additions
        events.setdefault(end, [[], []])[1].append(row_id)    # removals
        row_id += 1

    active = set()
    priority_heap = []
    intervals = []
    previous = None
    for coordinate in sorted(events):
        while priority_heap and priority_heap[0][1] not in active:
            heapq.heappop(priority_heap)
        if previous is not None and coordinate > previous and priority_heap:
            priority = -int(priority_heap[0][0])
            if (intervals and intervals[-1][1] == previous and
                    intervals[-1][2] == priority):
                intervals[-1] = (intervals[-1][0], coordinate, priority)
            else:
                intervals.append((previous, coordinate, priority))

        additions, removals = events[coordinate]
        for rid in removals:
            active.discard(rid)
        for rid in additions:
            active.add(rid)
            heapq.heappush(priority_heap, (-priorities[rid], rid))
        while priority_heap and priority_heap[0][1] not in active:
            heapq.heappop(priority_heap)
        previous = coordinate

    return (
        np.asarray([row[0] for row in intervals], dtype=np.uint64),
        np.asarray([row[1] for row in intervals], dtype=np.uint64),
        np.asarray([row[2] for row in intervals], dtype=np.uint8),
    )


def _pointer_word_views(raw: bytes, start: int,
                        holder_limit: Optional[int] = None):
    """Yield uint64 pointer slots at the two common 64-bit field alignments.

    A native pointer has an eight-byte width, but it can live at an offset of
    four inside a packed or mixed-width game structure.  Scanning only an
    eight-byte grid therefore makes an entire class of valid roots invisible.
    The caller may append a four-byte transport overlap and pass the logical
    chunk end as ``holder_limit``; this includes a cross-boundary 4-aligned
    word once without returning the next chunk's ordinary 8-aligned word.
    """
    view = memoryview(raw)
    start = int(start)
    limit = int(holder_limit) if holder_limit is not None else start + len(view)
    residues = (0,) if _PTR_STRUCT_STEP <= 0 or _PTR_STRUCT_STEP >= 8 else (
        0, _PTR_STRUCT_STEP)
    seen = set()
    for residue in residues:
        residue = int(residue)
        if residue in seen:
            continue
        seen.add(residue)
        available = len(view) - residue
        usable = available - (available % 8)
        if usable < 8:
            continue
        holder_base = start + residue
        count = max(0, (limit - holder_base + 7) // 8)
        if count <= 0:
            continue
        values = np.frombuffer(view[residue:residue + usable], dtype="<u8")
        if count < len(values):
            values = values[:count]
        if len(values):
            yield holder_base, values


def _scan_pointer_hits(raw: bytes, pos: int, holder_limit: int,
                       starts: np.ndarray, ends: np.ndarray):
    """Return (values, holders) for plausible pointer candidates in ``raw``.

    Scans both alignment residues via ``_pointer_word_views`` and keeps only
    values that are a plausible user-space address AND currently fall inside
    a mapped region (``starts``/``ends``, from
    ``_build_region_priority_intervals``). Shared by the in-memory and
    disk-backed reverse pointer indexes so a scanning correction only needs
    to land in one place.
    """
    values_parts, holders_parts = [], []
    for holder_base, values in _pointer_word_views(
            raw, pos, holder_limit=holder_limit):
        plausible = (values >= _ADDR_MIN) & (values <= _ADDR_MAX)
        map_idx = np.searchsorted(starts, values, side="right") - 1
        map_idx_i = map_idx.astype(np.int64, copy=False)
        valid = map_idx_i >= 0
        if valid.any():
            safe_idx = np.where(valid, map_idx_i, 0)
            plausible &= valid & (values < ends[safe_idx])
        hit_idx = np.flatnonzero(plausible)
        if hit_idx.size:
            values_parts.append(values[hit_idx].copy())
            holders_parts.append(
                np.uint64(holder_base) +
                hit_idx.astype(np.uint64, copy=False) * np.uint64(8))
    if not values_parts:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    return np.concatenate(values_parts), np.concatenate(holders_parts)


def _fast_direct_pointer_hits(ip: str, pid: int, target: int, maps: list,
                              cancel_event=None, max_hits: int = _PTR_FAST_DIRECT_HITS,
                              static_only: bool = True) -> list:
    """Cheap first pass: find direct target-near pointers without building the full index."""
    readable = _pointer_readable_regions(maps)
    if static_only:
        readable = [r for r in readable if _is_static_region(r)]
    readable.sort(key=lambda r: (-_region_priority(r), int(r["start"])))
    direct_range = int(setting("ptr_direct_range"))
    low = max(_ADDR_MIN, int(target) - direct_range)
    high = min(_ADDR_MAX, int(target) + direct_range)
    hits = []

    # MemDBG currently implements a fast exact one-hop holder scan.  Use it as
    # a seed only; RDX remains responsible for offsets, recursion, module roots
    # and verification.  This avoids trusting MemDBG's presently-unused depth
    # field as if it represented a complete chain.
    if state.get("backend") == "memdbg-experimental" and memdbg_native_ready():
        try:
            with memdbg_session(ip, timeout=10.0) as client:
                region_starts, region_rows = _build_region_lookup(maps)
                for holder in client.pointer_holders(pid, target, readable,
                                                     max_hits):
                    region = _region_for_addr(holder, region_starts, region_rows)
                    if region is not None and _is_static_region(region):
                        hits.append((int(holder), 0, region))
            _memdbg_note_native_outcome(True)
            if hits:
                add_log(f"MemDBG native pointer seed: {len(hits)} exact static holder(s)")
                return hits[:max_hits]
        except Exception as exc:
            _memdbg_note_native_outcome(False)
            add_log(f"MemDBG pointer seed unavailable; using RDX scan: {exc}", "warn")
    # Gather beyond the returned cap so ranking, not scan order, decides which
    # holders survive; see _PTR_FAST_DIRECT_POOL.
    pool_limit = max(int(max_hits), 1) * _PTR_FAST_DIRECT_POOL
    sock = _ScanSocket(ip, pid)
    try:
        for region in readable:
            if cancel_event and cancel_event.is_set():
                break
            rs, re_ = int(region["start"]), int(region["end"])
            pos = rs + ((-rs) % 8)
            while pos < re_ and len(hits) < pool_limit:
                size = min(_PTR_INDEX_CHUNK, re_ - pos)
                size -= size % 8
                if size < 8:
                    break
                # Include four bytes from the following logical chunk so a
                # 4-byte-aligned uint64 holder at its end is not lost.
                read_size = min(size + _PTR_STRUCT_STEP, re_ - pos)
                try:
                    raw = sock.read(pos, read_size, cancel_event)
                except Exception:
                    pos += size
                    continue
                for holder_base, a in _pointer_word_views(
                        raw, pos, holder_limit=pos + size):
                    mask = (a >= np.uint64(low)) & (a <= np.uint64(high))
                    if mask.any():
                        idxs = np.flatnonzero(mask)
                        for j in idxs.tolist():
                            value = int(a[j])
                            delta = int(target) - value
                            if delta % _PTR_RESOLVE_OFFSET_STEP:
                                continue
                            holder = holder_base + j * 8
                            hits.append((holder, delta, region))
                            if len(hits) >= pool_limit:
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
        starts, ends, interval_priority = _build_region_priority_intervals(
            self.maps)
        if starts.size == 0:
            return
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
                    # Preserve a 4-byte-aligned pointer which straddles this
                    # logical chunk boundary.  ``holder_limit`` below keeps
                    # the overlap from producing duplicate holders.
                    read_size = min(size + _PTR_STRUCT_STEP, re_ - pos)
                    try:
                        raw = sock.read(pos, read_size, cancel_event)
                    except Exception as exc:
                        add_log(f"Pointer index read error @ {hex(pos)}: {exc}", "warn")
                        self.done_bytes += size
                        pos += size
                        if progress_cb:
                            progress_cb(self.done_bytes, total)
                        continue
                    # A pointer candidate must be a user-space address that
                    # currently falls inside a mapped region.  This removes
                    # ordinary integers while retaining heap/module pointers.
                    values_hit, holders_hit = _scan_pointer_hits(
                        raw, pos, pos + size, starts, ends)
                    if values_hit.size:
                        vals_parts.append(values_hit)
                        holder_parts.append(holders_hit)
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
            holder_map = np.searchsorted(starts, self.holders, side="right") - 1
            holder_map_i = holder_map.astype(np.int64, copy=False)
            valid = holder_map_i >= 0
            hp = np.zeros(self.holders.shape, dtype=np.uint8)
            if valid.any():
                vi = np.where(valid, holder_map_i, 0)
                inside = valid & (self.holders < ends[vi])
                hp[inside] = interval_priority[vi[inside]]
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
        """Return (holder, offset) for values within max_offset of target.

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


# Live disk indexes, so a crash or Ctrl-C cannot strand their shard files.
# One index for a multi-GiB process can hold hundreds of MiB of .npy shards,
# and close() was previously the only thing that removed them.
_disk_index_registry = set()
_disk_index_lock = threading.Lock()
_DISK_INDEX_PREFIX = "rdx_ptr_"
# Orphans from a previous run that died before its atexit hook could fire.
# ps5debug-NG does the same for its own snapshot spill files ("/data is swept
# at startup"); an hour is well past any plausible in-progress build.
_DISK_INDEX_ORPHAN_AGE = 3600.0


def _close_all_disk_indexes() -> None:
    with _disk_index_lock:
        live = list(_disk_index_registry)
    for index in live:
        try:
            index.close()
        except Exception:
            pass


def _sweep_orphaned_disk_indexes() -> int:
    """Remove shard directories left behind by a previous run. Never raises."""
    removed = 0
    try:
        root = Path(tempfile.gettempdir())
        now = time.time()
        for entry in root.glob(_DISK_INDEX_PREFIX + "*"):
            try:
                if not entry.is_dir():
                    continue
                if now - entry.stat().st_mtime < _DISK_INDEX_ORPHAN_AGE:
                    continue          # may belong to a running instance
                for child in entry.iterdir():
                    child.unlink(missing_ok=True)
                entry.rmdir()
                removed += 1
            except OSError:
                continue
    except Exception:
        pass
    return removed


_atexit.register(_close_all_disk_indexes)


class _DiskReversePointerIndex:
    """Shard-backed reverse index for processes too large to retain in RAM.

    Each shard is independently sorted and stored as three ``.npy`` arrays
    (values, holders, and each holder's precomputed region priority).
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
        # Shard arrays are written once by _build() and never change, but
        # query() used to np.load() every one of them again on every call.
        # The graph search runs up to _PTR_RESOLVE_MAX_NODES nodes x
        # len(_PTR_RESOLVE_OFFSET_TIERS) tiers, so on a 4.24 GiB process
        # (135 shards) that is ~1.7 M reopen+remap operations -- about 1.3
        # minutes of pure overhead on top of an already slow resolve. Map
        # each file once and keep it; memmaps are virtual, so holding a few
        # hundred costs address space rather than RSS.
        self._mapped = {}
        self._tmpdir = Path(tempfile.mkdtemp(prefix=_DISK_INDEX_PREFIX))
        with _disk_index_lock:
            _disk_index_registry.add(self)
        try:
            self._build(cancel_event, progress_cb)
        except BaseException:
            self.close()
            raise

    def _mapped_array(self, path):
        """Memory-map a shard array once and reuse it for later queries."""
        arr = self._mapped.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            self._mapped[path] = arr
        return arr

    def close(self):
        """Close mapped shards and remove this index's private temporary files."""
        # Drop the mappings before unlinking so the files are not still mapped
        # when the directory goes.
        with _disk_index_lock:
            _disk_index_registry.discard(self)
        self._mapped.clear()
        self.shards.clear()
        try:
            for path in self._tmpdir.iterdir():
                path.unlink(missing_ok=True)
            self._tmpdir.rmdir()
        except Exception:
            pass

    def _build(self, cancel_event=None, progress_cb=None):
        readable = _coalesce_pointer_regions(_pointer_readable_regions(self.maps))
        starts, ends, interval_priority = _build_region_priority_intervals(
            self.maps)
        if starts.size == 0:
            return
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
                # Request a small transport overlap so a 4-byte-aligned
                # pointer holder straddling this shard's logical boundary is
                # still read once, matching the in-memory reverse index.
                read_size = min(size + _PTR_STRUCT_STEP, region_end - pos)
                tasks.put((shard_no, pos, size, read_size))
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
                        number, pos, size, read_size = task
                        if cancel_event and cancel_event.is_set():
                            continue
                        raw = sock.read(pos, read_size, cancel_event)
                        paths = []
                        kept_values, kept_holders = _scan_pointer_hits(
                            raw, pos, pos + size, starts, ends)
                        if kept_values.size:
                            # Rank holders by their own region quality once,
                            # at build time, so a query never needs to look
                            # this up again per candidate.
                            holder_map = np.searchsorted(
                                starts, kept_holders, side="right") - 1
                            holder_map_i = holder_map.astype(np.int64, copy=False)
                            holder_valid = holder_map_i >= 0
                            kept_priority = np.zeros(kept_holders.shape, dtype=np.uint8)
                            if holder_valid.any():
                                vi = np.where(holder_valid, holder_map_i, 0)
                                inside = holder_valid & (kept_holders < ends[vi])
                                kept_priority[inside] = interval_priority[vi[inside]]
                            # Partition by the pointer's upper 32 bits.  A
                            # graph query then opens only the address family
                            # containing its target instead of every shard.
                            prefixes = kept_values >> np.uint64(32)
                            for prefix in np.unique(prefixes).tolist():
                                group = prefixes == np.uint64(prefix)
                                group_values = kept_values[group]
                                group_holders = kept_holders[group]
                                group_priority = kept_priority[group]
                                order = np.argsort(group_values, kind="mergesort")
                                value_path = self._tmpdir / (
                                    f"v{number:05d}_{int(prefix):08x}.npy")
                                holder_path = self._tmpdir / (
                                    f"h{number:05d}_{int(prefix):08x}.npy")
                                priority_path = self._tmpdir / (
                                    f"p{number:05d}_{int(prefix):08x}.npy")
                                np.save(value_path, group_values[order],
                                        allow_pickle=False)
                                np.save(holder_path, group_holders[order],
                                        allow_pickle=False)
                                np.save(priority_path, group_priority[order],
                                        allow_pickle=False)
                                # Each shard is sorted, so its first and
                                # last value bound everything it holds. Keep
                                # them: query() can then skip a shard whole
                                # instead of binary-searching it, which is
                                # what makes a 135-shard index tractable.
                                sorted_values = group_values[order]
                                paths.append((int(prefix), value_path,
                                              holder_path, priority_path,
                                              int(sorted_values[0]),
                                              int(sorted_values[-1])))
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
        # Holder region priority is precomputed once per shard at build time
        # (see _build), so a query never has to rebuild the region lookup or
        # re-resolve each candidate's owning region.
        target = int(target)
        step = max(1, int(step))
        low = max(_ADDR_MIN, target - max(0, int(max_offset)))
        high = min(_ADDR_MAX, target + max(0, int(max_offset)))
        candidates = []
        wanted_prefixes = set(range(low >> 32, (high >> 32) + 1))
        for (prefix, value_path, holder_path, priority_path,
             shard_lo, shard_hi) in self.shards:
            if prefix not in wanted_prefixes:
                continue
            # Cheap interval reject before touching the file at all: the
            # shard is sorted, so if its range does not intersect the query
            # window it cannot contribute a single hit.
            if shard_hi < low or shard_lo > high:
                continue
            values = self._mapped_array(value_path)
            lo = int(np.searchsorted(values, np.uint64(low), side="left"))
            hi = int(np.searchsorted(values, np.uint64(high), side="right"))
            if hi <= lo:
                continue
            holders = self._mapped_array(holder_path)
            priorities = self._mapped_array(priority_path)
            seg_values = values[lo:hi].astype(np.int64, copy=False)
            seg_holders = holders[lo:hi]
            seg_priorities = priorities[lo:hi]
            deltas = np.int64(target) - seg_values
            mask = (np.abs(deltas) <= max_offset) & (deltas % step == 0)
            idxs = np.flatnonzero(mask)
            local = [(int(seg_priorities[i]), int(seg_holders[i]), int(deltas[i]))
                     for i in idxs.tolist()]
            if len(local) > max_hits:
                local.sort(key=lambda item: (-item[0], abs(item[2]), item[2] < 0, item[1]))
                del local[max_hits:]
            candidates.extend(local)
        candidates.sort(key=lambda item: (-item[0], abs(item[2]), item[2] < 0, item[1]))
        out, seen = [], set()
        for _priority, holder, delta in candidates:
            key = (holder, delta)
            if key not in seen:
                seen.add(key)
                out.append(key)
                if len(out) >= max(1, int(max_hits)):
                    break
        return out


def _classify_regions_cached(ip: str, pid: int, maps: list) -> tuple:
    """Fetch ps5debug-NG's read-throughput classification once per map layout.

    Returns ``(uncached_ranges, supported)``.  ``uncached_ranges`` is the
    sorted ``(start, end)`` list the payload reported as uncached/GPU-backed.
    ``supported`` reports whether the probe actually ran, which callers need
    in order to tell "the payload says nothing here is GPU-backed" apart from
    "this payload cannot tell us" — only the latter should fall back to a
    blunt size heuristic.
    """
    fingerprint = _pointer_map_fingerprint(maps)
    # _pointer_region_class_cache is also read from index-builder threads via
    # _pointer_readable_regions, so guard the mutation rather than relying on
    # the UI happening to serialise scans.
    with _region_class_lock:
        needs_probe = fingerprint not in _pointer_region_class_cache
    if needs_probe:
        try:
            classified = ps5_classify_regions(ip, pid)
            with _region_class_lock:
                if len(_pointer_region_class_cache) >= 4:
                    _pointer_region_class_cache.clear()
                    _region_class_supported.clear()
                _pointer_region_class_cache[fingerprint] = classified
                _region_class_supported[fingerprint] = True
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
            with _region_class_lock:
                _pointer_region_class_cache[fingerprint] = []
                _region_class_supported[fingerprint] = False
            add_log(f"Region classifier unavailable; using map safeguards: {exc}",
                    "warn")
    with _region_class_lock:
        rows = _pointer_region_class_cache.get(fingerprint) or []
        supported = bool(_region_class_supported.get(fingerprint))
    raw = sorted(
        (int(r["start"]), int(r["end"])) for r in rows
        if int(r.get("flags", 0)) & 1 and int(r["end"]) > int(r["start"]))
    # Coalesce before returning. Callers test membership with a single
    # bisect against the range starts, which is only correct on disjoint
    # ranges: with overlaps, a short later-starting range shadows a longer
    # earlier one and an address inside the long range is reported as
    # cached. The classifier derives its rows from the VM map, which this
    # codebase documents as containing overlapping records, so that is a
    # real shape rather than a theoretical one.
    return _coalesce_ranges(raw), supported


def _coalesce_ranges(raw: list) -> list:
    """Merge a sorted (start, end) list into disjoint ranges."""
    out = []
    for start, end in sorted(raw):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _coalesced_uncached_ranges(maps: list) -> list:
    """Already-classified uncached ranges for `maps`, merged and disjoint.

    Reads the cache only — never probes — so callers that merely want to
    honour a previous classification (the pointer scanner) do not trigger an
    authenticated round trip of their own.
    """
    with _region_class_lock:
        rows = _pointer_region_class_cache.get(_pointer_map_fingerprint(maps), [])
    return _coalesce_ranges(
        [(int(r["start"]), int(r["end"])) for r in rows
         if int(r.get("flags", 0)) & 1 and int(r["end"]) > int(r["start"])])


def _region_is_uncached(region: dict, uncached: list, starts: list) -> bool:
    """Whether ``region`` overlaps any classifier-reported uncached range."""
    if not uncached:
        return False
    start, end = int(region.get("start", 0)), int(region.get("end", 0))
    index = bisect.bisect_left(starts, end)
    return index > 0 and start < uncached[index - 1][1]


def _get_reverse_pointer_index(ip: str, pid: int, cancel_event=None,
                               progress_cb=None):
    """Get or build the cached reverse pointer index for the current map layout."""
    maps = _get_maps_cached(ip, pid)
    fp = _pointer_map_fingerprint(maps)
    _classify_regions_cached(ip, pid, maps)
    key = (ip, int(pid), int(state.get("session", 0)))
    stale_index = None
    with _pointer_index_lock:
        cached = _pointer_index_cache.get(key)
        if cached and cached[0] == fp:
            return cached[1], maps, False
        if cached:
            _pointer_index_cache.pop(key, None)
            stale_index = cached[1]
    if stale_index is not None and hasattr(stale_index, "close"):
        stale_index.close()
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


def _query_pointer_index_tiered(index, target: int) -> list:
    """Query narrow-to-wide without letting wide-range noise hide near hits."""
    out = []
    seen = set()
    per_tier = max(32, _PTR_RESOLVE_MAX_HITS // 2)
    for window in _PTR_RESOLVE_OFFSET_TIERS:
        for holder, offset in index.query(
                int(target), int(window), _PTR_RESOLVE_OFFSET_STEP, per_tier):
            key = (int(holder), int(offset))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        # Preserve all close candidates; allow progressively wider tiers to
        # add alternatives up to a bounded multiple of the former single pass.
        if len(out) >= _PTR_RESOLVE_MAX_HITS * 3:
            break
    return out


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


def resolve_via_object_identity(ip: str, pid: int, target_addr: int,
                                groups: list, width: int = 4,
                                value_type: str = "u32",
                                max_depth: Optional[int] = None,
                                cancel_event=None,
                                progress_cb=None) -> dict:
    """Resolve a temporary address by first identifying the object that owns it.

    The usual route asks the resolver for pointers to `target_addr` itself --
    an address in the *middle* of an object. Very little code holds such a
    pointer: what code holds is the object base, and reaches the field by a
    constant displacement. So the search is aimed at the wrong thing, and the
    chains that come back are dominated by whatever happens to land nearby.
    Measured this session: 96 candidates for one live ammo address, with the
    top-ranked ones pointing 271,648 bytes away -- the low bits of the target,
    matched by coincidence.

    Type Scan already knows which object owns an address and at what offset.
    That is the same object-base-plus-field-offset the evidence-driven
    literature obtains by trapping the writing instruction and reading a
    register -- reached here without a debugger, which matters because this
    payload's does not work.

    So: identify the object, corroborate the field against sibling instances,
    then resolve a chain to the **object base** and carry the field offset as
    the chain's terminal. Fewer candidates, better ones, and each carries a
    structural claim that can be checked rather than only a numeric score.

    Returns a dict describing what was established, including the failure
    stage when it could not be. Never raises for an ordinary miss: "no object
    owns this address" is a normal answer for a value that is not a managed
    object field, and the caller should fall back to the direct resolver.
    """
    finding = locate_field_in_type(int(target_addr), groups)
    if finding is None:
        return {"stage": "no-object", "finding": None, "corroboration": None,
                "candidates": [],
                "note": "no known object instance owns this address; "
                        "use the direct resolver"}

    corroboration = corroborate_field_across_instances(
        ip, pid, finding, width=width, value_type=value_type,
        cancel_event=cancel_event)

    # A field no sibling shares is weak evidence that this is a field at all.
    # Say so, and keep going -- the object may simply be the only live
    # instance -- but do not present the result as corroborated.
    read = int(corroboration.get("read", 0))
    plausible = int(corroboration.get("plausible", 0))
    corroborated = read > 0 and plausible * 2 >= read

    add_log(
        f"Object identity: {finding.get('class_name') or hex(finding['type_ptr'])}"
        f" + {finding['field_offset']:#x}"
        + (f"; {plausible}/{read} sibling instance(s) agree"
           if read else "; no sibling instances readable"))

    result = _resolve_permanent_candidates(
        ip, pid, int(finding["instance_base"]), max_depth=max_depth,
        cancel_event=cancel_event, progress_cb=progress_cb)

    candidates = list(result.get("candidates", []) or [])
    for candidate in candidates:
        # The chain reaches the object; the field is a constant hop from it.
        candidate["terminal_offset"] = int(finding["field_offset"])
        candidate["object_type_ptr"] = int(finding["type_ptr"])
        if finding.get("class_name"):
            candidate["object_class_name"] = str(finding["class_name"])
        candidate["field_corroborated"] = bool(corroborated)
        candidate["sibling_agreement"] = f"{plausible}/{read}" if read else "0/0"

    return {"stage": "resolved" if candidates else "no-chain-to-object",
            "finding": finding, "corroboration": corroboration,
            "corroborated": corroborated,
            "candidates": candidates,
            "method": result.get("method"),
            "maps": result.get("maps")}


def _resolve_permanent_candidates(ip: str, pid: int, target_addr: int,
                                   max_depth: Optional[int] = None,
                                   cancel_event=None,
                                   progress_cb=None) -> dict:
    """Resolve a dynamic address using a fast direct pass, then a priority-guided graph."""
    if max_depth is None:
        max_depth = int(setting("ptr_max_depth"))
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
    # Only a structurally plausible hit is worth short-circuiting on.
    plausible = [c for c in fast_candidates
                 if abs(int(c["offsets"][0])) <= _PTR_PLAUSIBLE_FIELD_MAX]
    if plausible:
        plausible = _rank_pointer_candidates(ip, pid, plausible)
        return {"candidates": plausible, "index_built": False,
                "maps": maps, "method": "fast-direct"}
    if fast_candidates:
        # Keep them, but do not stop here: they are near-misses, and the deeper
        # search may still find a real chain. Returned only as a fallback.
        add_log(
            f"Fast pointer pass found {len(fast_candidates)} holder(s), but "
            f"none point at a plausible object base (nearest displacement "
            f"{min(abs(int(c['offsets'][0])) for c in fast_candidates):#x}). "
            f"Continuing with the deeper search.", "warn")
        for c in fast_candidates:
            c["score"] = float(c["score"]) - 120.0
            c["coincidence_risk"] = "displacement is not a plausible field offset"
            c["confidence"] = _candidate_confidence(c)

    # Before constructing a multi-gigabyte exhaustive index, walk the natural
    # object locality: modules first, then the address family containing each
    # discovered parent.  This is the common-case algorithm used by practical
    # pointer scanners and reuses the exhaustive index only when locality fails.
    # Tier 2 is a graph exploration: it accepts any pointer landing within
    # _PTR_STRUCT_MAX of the target and expands heap holders level by level, so
    # its cost tracks heap complexity rather than memory bandwidth. Measured on
    # a 4.24 GiB title it reached only 30% of a depth-4 search in 10 minutes.
    # Left unbounded it can therefore run for tens of minutes and *then* still
    # fall through to the indexed tier, which is the one whose cost is
    # predictable. Bound it so the fallthrough happens while the user is still
    # willing to wait.
    budget_event = threading.Event()

    def _budget_watch():
        deadline = time.monotonic() + _PTR_LOCALITY_TIME_BUDGET
        while not budget_event.wait(0.25):
            if cancel_event is not None and cancel_event.is_set():
                budget_event.set()
                return
            if time.monotonic() >= deadline:
                add_log(
                    f"Locality pointer pass exceeded "
                    f"{_PTR_LOCALITY_TIME_BUDGET:.0f}s; falling through to the "
                    f"reverse index", "warn")
                budget_event.set()
                return

    watcher = threading.Thread(target=_budget_watch, daemon=True)
    watcher.start()
    try:
        local_hits = pointer_chain_scan(
            ip, pid, int(target_addr), max_depth=max_depth,
            cancel_event=budget_event, progress_cb=progress_cb)
    finally:
        budget_event.set()
        watcher.join(timeout=1.0)
    # A caller-requested cancel must stay a cancel, not look like a timeout.
    if cancel_event is not None and cancel_event.is_set():
        return {"candidates": [], "index_built": False, "maps": maps,
                "method": "cancelled"}
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
        local_candidates = _rank_pointer_candidates(
            ip, pid, local_candidates, region_starts, region_rows)
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
        hits = _query_pointer_index_tiered(index, current)

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
                hits = _query_pointer_index_tiered(fresh_index, current)
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

    # PINCE's "Module Bases Only". A chain rooted in an anonymous mapping
    # cannot be written as module+offset, so it can never survive a reboot
    # even if it survives two reloads. Filtering here rather than during the
    # walk keeps the search itself unchanged and the setting reversible.
    if setting("ptr_module_bases_only"):
        rooted = [c for c in candidates if c.get("module_name")]
        dropped = len(candidates) - len(rooted)
        if dropped:
            add_log(f"Module-bases-only: dropped {dropped} chain(s) with no "
                    f"named module root", "warn")
        candidates = rooted

    candidates = _rank_pointer_candidates(
        ip, pid, candidates, region_starts, region_rows)

    if not candidates and fast_candidates:
        # Every tier has now run and found nothing better. Hand back the
        # near-misses from the fast pass rather than nothing at all, clearly
        # marked: they are holders that point near the target but not at a
        # plausible object base, so they are likely this session's heap
        # coincidences and will not survive a reload.
        fast_candidates.sort(key=lambda c: (abs(int(c["offsets"][0])), c["base"]))
        return {"candidates": fast_candidates, "index_built": built,
                "maps": maps, "method": "fast-direct-unverified"}
    return {"candidates": candidates, "index_built": built,
            "maps": maps, "method": "reverse-index"}


def _pointer_readable_regions(maps: list) -> list:
    """Return useful readable regions, excluding confirmed uncached maps.

    Region size alone is not evidence of a GPU reservation.  Large PS5 heaps
    are legitimate pointer sources, and excluding every map above 1 GiB made
    some games mathematically impossible to resolve.  The authenticated region
    classifier remains the authoritative way to omit uncached/GPU memory.
    """
    PROT_READ = 0x1
    PROT_EXEC = 0x4
    out = []
    # Share one coalescing implementation with the value scanners. This used
    # to build its own uncached list inline, which is how the un-merged
    # overlap bug got copied into the scanners in the first place.
    uncached = _coalesced_uncached_ranges(maps)
    uncached_starts = [row[0] for row in uncached]
    for r in maps:
        start, end = int(r.get("start", 0)), int(r.get("end", 0))
        prot = int(r.get("prot", 0))
        if end <= start or not (prot & PROT_READ):
            continue
        if prot == PROT_EXEC:
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
        if (merged and start <= merged[-1]["end"] and
                is_static == merged[-1]["static"]):
            merged[-1]["end"] = max(merged[-1]["end"], end)
        else:
            merged.append({"start": start, "end": end,
                           "static": is_static})
    return merged


def _pointer_scan_chunk(sock: _ScanSocket, start: int, size: int,
                        target_arr: np.ndarray, target_chains: dict,
                        region_starts: list, region_rows: list, cancel_event=None,
                        diagnostic: Optional[dict] = None):
    """Read one chunk and return matching (holder, target, offset, chain) hits."""
    raw = sock.read(start, size, cancel_event)
    trim = len(raw) - (len(raw) % 8)
    if trim < 8:
        return []
    vals = np.frombuffer(raw[:trim], dtype=np.uint64)
    if diagnostic is not None:
        diagnostic["slots_scanned"] = diagnostic.get("slots_scanned", 0) + len(vals)
    # Match pointer values to target *ranges* instead of enumerating every
    # target-offset value. Neighbouring targets cheaply identify whether at
    # least one target lies within the signed offset window; exact aligned
    # target/chain combinations are expanded only for the tiny set of hits.
    positions = np.searchsorted(target_arr, vals, side="left")
    range_mask = np.zeros(vals.shape, dtype=bool)
    right = positions < target_arr.size
    if right.any():
        ri = positions[right]
        rv = vals[right]
        rd = target_arr[ri].astype(np.int64) - rv.astype(np.int64)
        in_range = np.abs(rd) <= _ptr_struct_max()
        range_mask[right] |= in_range
    left = positions > 0
    if left.any():
        li = positions[left] - 1
        lv = vals[left]
        ld = target_arr[li].astype(np.int64) - lv.astype(np.int64)
        in_range = np.abs(ld) <= _ptr_struct_max()
        range_mask[left] |= in_range
    if diagnostic is not None:
        raw_count = int(np.count_nonzero(range_mask))
        diagnostic["raw_range_matches"] = diagnostic.get("raw_range_matches", 0) + raw_count
    # The range prefilter only checks the nearest target on either side. With multiple
    # targets, that nearest target can have the wrong 4-byte residue while a
    # slightly farther in-range target is a valid aligned edge.  PS4Cheater's
    # exhaustive address walk does not make this mistake.  Expand every cheap
    # interval match here and apply exact alignment in the small Python loop
    # below, preventing a systematic deep-chain false negative.
    indices = np.flatnonzero(range_mask)
    if not len(indices):
        return []

    holders = start + indices.astype(np.uint64) * 8
    hits = []
    # Usually only a tiny number of slots match a pointer target.  Keep the
    # Python work limited to those hits rather than iterating over the chunk.
    for idx, holder_u in zip(indices.tolist(), holders.tolist()):
        value = int(vals[idx])
        lo = int(np.searchsorted(target_arr,
                                 np.uint64(max(_ADDR_MIN, value - _ptr_struct_max())),
                                 side="left"))
        hi = int(np.searchsorted(target_arr,
                                 np.uint64(min(_ADDR_MAX, value + _ptr_struct_max())),
                                 side="right"))
        for target_u in target_arr[lo:hi].tolist():
            target = int(target_u)
            soff = target - value
            if soff % _PTR_STRUCT_STEP:
                continue
            chain = target_chains[target]
            # A pointer holder itself must be inside a mapped region.
            region = _region_for_addr(int(holder_u), region_starts, region_rows)
            if region is None:
                continue
            hits.append((int(holder_u), int(target), int(soff), chain,
                         bool(_is_static_region(region)),
                         region.get("name", "") or "anon"))
    if diagnostic is not None:
        aligned_slots = len({hit[0] for hit in hits})
        diagnostic["aligned_matches"] = (
            diagnostic.get("aligned_matches", 0) + aligned_slots)
        diagnostic["alignment_rejected"] = (
            diagnostic.get("alignment_rejected", 0) +
            max(0, int(np.count_nonzero(range_mask)) - aligned_slots))
        diagnostic["expanded_hits"] = diagnostic.get("expanded_hits", 0) + len(hits)
    return hits


def pointer_chain_scan(ip: str, pid: int,
                       target_addr: int,
                       max_depth: Optional[int] = None,
                       cancel_event=None,
                       progress_cb=None,
                       diagnostic_report: Optional[dict] = None) -> list:
    """Find static pointer chains to ``target_addr`` without full-memory caching.

    The search works backwards from the temporary address.  At each level it
    looks for pointers within a signed ±16 KiB field-offset window, accepting
    4-byte-aligned offsets.  A heap holder becomes the target for the next
    level; a static holder becomes a candidate immediately.

    Important differences from the old scanner:
      * memory is streamed in 32 MiB chunks instead of retained for every level;
      * offsets are tested at every pointer level, not only level 1;
      * heap expansion is bounded (beam search) so one noisy heap cannot create
        millions of chains;
      * the default depth is five, with up to eight supported;
      * all heap address families remain eligible until a static root is found.

    ``max_depth`` resolves to the user's Settings value when not given, so a
    caller that passes nothing follows the setting rather than a literal.
    """
    if max_depth is None:
        max_depth = int(setting("ptr_max_depth"))
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
    if diagnostic_report is not None:
        readable_ids = {id(r) for r in readable_maps}
        excluded = {"not_readable": 0, "execute_only": 0,
                    "libsce": 0, "classified_uncached": 0, "other": 0}
        for r in maps:
            if id(r) in readable_ids:
                continue
            start, end = int(r.get("start", 0)), int(r.get("end", 0))
            prot = int(r.get("prot", 0)); name = str(r.get("name", "") or "")
            if end <= start or not (prot & 1): excluded["not_readable"] += 1
            elif prot == 4: excluded["execute_only"] += 1
            elif name.startswith("libSce"): excluded["libsce"] += 1
            else: excluded["other"] += 1
        diagnostic_report.update({
            "target": int(target_addr), "maps_total": len(maps),
            "maps_scanned": len(readable_maps), "excluded_regions": excluded,
            "depths": [], "limits": {
                "offset_max": _ptr_struct_max(), "offset_step": _PTR_STRUCT_STEP,
                "beam_max": _PTR_BEAM_MAX, "max_depth": max_depth,
            }})
    readable = _coalesce_pointer_regions(readable_maps)
    if not readable:
        add_log("Pointer scan: no readable data regions", "warn")
        return []

    region_starts, region_rows = _build_region_lookup(maps)

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
        pre_depth_beam_dropped = 0
        if len(current_targets) > _PTR_BEAM_MAX:
            pre_depth_beam_dropped = len(current_targets) - _PTR_BEAM_MAX
            current_targets = dict(list(current_targets.items())[:_PTR_BEAM_MAX])
            add_log(f"Depth {depth}: beam capped at {_PTR_BEAM_MAX:,} targets", "warn")

        target_arr = np.sort(np.fromiter(current_targets.keys(), dtype=np.uint64))
        next_targets = {}
        static_hits = 0
        heap_hits = 0
        depth_done = 0
        depth_diag = {"depth": depth, "targets_entering": len(current_targets),
                      "pre_depth_beam_dropped": pre_depth_beam_dropped,
                      "slots_scanned": 0, "raw_range_matches": 0,
                      "alignment_rejected": 0, "aligned_matches": 0,
                      "expanded_hits": 0, "heap_hits": 0, "static_hits": 0,
                      "parents_retained": 0, "beam_dropped": 0,
                      "verification_failures": 0, "partial_chains": [],
                      "read_failures": 0, "terminal_heap_regions_skipped": 0,
                      "nonlocal_heap_regions_deferred": 0}
        if diagnostic_report is not None:
            diagnostic_report["depths"].append(depth_diag)

        add_log(f"Pointer scan depth {depth}: {len(current_targets):,} targets, "
                f"interval-matched ±{hex(_ptr_struct_max())}")

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
                    if diagnostic_report is not None:
                        depth_diag["terminal_heap_regions_skipped"] += 1
                    add_log(f"Depth {depth}: static-root pass complete; "
                            "skipping terminal heap scan")
                    break
                # Do not stop after the first address family yields a parent.
                # PS5 allocators commonly keep manager objects, pools and game
                # objects in different high-address families.  The old locality
                # shortcut made every later family invisible and was a major
                # systematic source of zero static-root results.
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
                            region_starts, region_rows, cancel_event,
                            depth_diag if diagnostic_report is not None else None)
                    except Exception as exc:
                        if diagnostic_report is not None:
                            depth_diag["read_failures"] += 1
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
                                "status": "provisional",
                                # Static slots often converge on the exact same
                                # heap parent.  Retain this key so verification
                                # can collapse those duplicates first.
                                "family_anchor": target,
                            })
                            static_hits += 1
                            if diagnostic_report is not None and len(depth_diag["partial_chains"]) < 20:
                                depth_diag["partial_chains"].append({
                                    "kind": "static", "base": holder,
                                    "target": target, "offsets": new_chain,
                                    "region": rname})
                        elif depth < max_depth:
                            old = next_targets.get(holder)
                            if old is None or len(new_chain) < len(old):
                                if len(next_targets) < _PTR_BEAM_MAX or old is not None:
                                    next_targets[holder] = new_chain
                                elif diagnostic_report is not None:
                                    depth_diag["beam_dropped"] += 1
                            heap_hits += 1
                            if diagnostic_report is not None and len(depth_diag["partial_chains"]) < 20:
                                depth_diag["partial_chains"].append({
                                    "kind": "heap", "base": holder,
                                    "target": target, "offsets": new_chain,
                                    "region": rname})
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
        if diagnostic_report is not None:
            depth_diag["static_hits"] = static_hits
            depth_diag["heap_hits"] = heap_hits
            depth_diag["parents_retained"] = len(next_targets)
        current_targets = next_targets

        # A broad signed-offset sweep can produce a static coincidence whose
        # heap object changes before the pass ends.  Never stop on an unverified
        # hit: resolve each fresh chain twice and discard stale candidates.
        depth_static = [c for c in candidates
                        if c["static"] and c["depth"] == depth]
        if len(depth_static) > _PTR_VERIFY_MAX:
            families = {}
            ranked = sorted(depth_static, key=lambda c: (
                sum(1 for x in c.get("offsets", []) if int(x) % 8),
                sum(abs(int(x)) for x in c.get("offsets", [])),
                int(c.get("base", 0))))
            selected = []
            for candidate in ranked:
                family = (int(candidate.get("family_anchor", 0)),
                          tuple(int(x) for x in candidate.get("offsets", [])[1:]))
                kept = families.get(family, 0)
                if kept >= _PTR_VERIFY_PER_FAMILY:
                    continue
                families[family] = kept + 1
                selected.append(candidate)
                if len(selected) >= _PTR_VERIFY_MAX:
                    break
            selected_ids = {id(c) for c in selected}
            pruned_ids = {id(c) for c in depth_static
                          if id(c) not in selected_ids}
            candidates = [c for c in candidates if id(c) not in pruned_ids]
            if diagnostic_report is not None:
                depth_diag["static_candidates_pruned"] = len(pruned_ids)
                depth_diag["static_families_retained"] = len(families)
            add_log(f"Depth {depth}: grouped {len(depth_static):,} static hits "
                    f"into {len(families):,} families; verifying "
                    f"{len(selected):,}")
            depth_static = selected
        verified_static = []
        for candidate in depth_static:
            if _verify_candidate_twice(
                    ip, pid, candidate, int(target_addr), cancel_event):
                verified_static.append(candidate)
        if depth_static:
            # Compare by identity: `c not in verified_static` is dict
            # value-equality, which is both O(n²) here and the wrong question
            # to ask about two candidate records that happen to match.
            verified_ids = {id(c) for c in verified_static}
            stale_ids = {id(c) for c in depth_static if id(c) not in verified_ids}
            candidates = [c for c in candidates if id(c) not in stale_ids]
            if stale_ids:
                if diagnostic_report is not None:
                    depth_diag["verification_failures"] = len(stale_ids)
                add_log(f"Depth {depth}: rejected {len(stale_ids)} stale static "
                        "candidate(s)", "warn")

        # Same-session verification only proves that a chain is internally
        # consistent right now.  It does not prove reload stability: ordinary
        # executable data slots can coincidentally lead into the current heap.
        # Keep walking the retained heap frontier through the requested depth
        # so shallow coincidences cannot hide a longer, stable chain.
        if verified_static:
            add_log(f"Pointer scan found {sum(1 for c in candidates if c['static'])} "
                    f"same-session candidate(s) at depth {depth}; "
                    "continuing for deeper roots")

    candidates.sort(key=lambda c: (not c["static"], c["depth"], c["base"]))
    elapsed = max(time.monotonic() - started, 1e-9)
    if diagnostic_report is not None:
        diagnostic_report["elapsed_seconds"] = elapsed
        diagnostic_report["candidates_returned"] = len(candidates)
        diagnostic_report["static_candidates_returned"] = sum(
            1 for c in candidates if c["static"])
    add_log(f"Pointer chain scan: {len(candidates)} candidates "
            f"({sum(1 for c in candidates if c['static'])} static) "
            f"in {elapsed:.1f}s")
    return candidates


# ── instance discovery by type pointer ────────────────────────────────────────
# XenoScan's headline primitive, which RDX had no equivalent of: enumerate live
# class instances and group them by their underlying type. Its stated mechanism
# is "any class with a virtual-function table".
#
# That transfers to RDX's actual target better than to a generic PC scanner.
# Every IL2CPP object begins with a pointer to its Il2CppClass, which is the
# same signature a vtable pointer is: a qword at offset 0 that points into a
# module/static region and is SHARED BY EVERY INSTANCE of that type. So the
# discriminator is not the value itself but how often it repeats.
#
# The whole workflow up to now has been "know a number, scan for it, narrow
# it", which cannot find state that never appears on screen. This adds "show me
# the objects", and hands the pointer search a typed base rather than a bare
# heap address.
#
# Everything here is read-only.
_TYPE_SCAN_MIN_INSTANCES = 8        # below this a repeat is not evidence of a type
# Naming is the discriminator now that the target filter admits heap
# memory, so it gets a real budget rather than a cosmetic one.
_TYPE_SCAN_LABEL_LIMIT = 256
_TYPE_SCAN_LABEL_BUDGET = 25.0
_TYPE_SCAN_MAX_CANDIDATES = 8_000_000   # bounded collected slots (~128 MB peak)
_TYPE_SCAN_MAX_TYPES = 512          # distinct types returned
_TYPE_SCAN_MAX_INSTANCES = 4096     # instance addresses retained per type
_TYPE_SCAN_MAX_DISTINCT = 250_000   # distinct type pointers tallied
# Consecutive failed chunk reads that mean "the console is gone" rather than
# "this span is unmapped". Unmapped spans are scattered; a dead link fails
# every read from the point it dies.
_TYPE_SCAN_MAX_CONSECUTIVE_FAILS = 8
_TYPE_SCAN_CHUNK = 0x1000000        # 16 MiB reads


def _type_target_interval_arrays(maps: list) -> tuple:
    """Coalesced (starts, ends) for addresses a type pointer may legitimately
    target: any mapped, readable, non-executable range.

    Not `_static_interval_arrays`. IL2CPP allocates class metadata on the
    heap, so restricting targets to module/static memory excluded every real
    type pointer and admitted the code pointers that dominate a heap sweep.
    Executable ranges stay out because a class never lives in code, and that
    is what the old filter was accidentally selecting for.
    """
    ranges = _coalesce_ranges([
        (int(r.get("start", 0)), int(r.get("end", 0)))
        for r in maps
        if int(r.get("end", 0)) > int(r.get("start", 0))
        and (int(r.get("prot", 0)) & 0x1)
        and not (int(r.get("prot", 0)) & 0x4)])
    if not ranges:
        return (np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64))
    return (np.asarray([a for a, _ in ranges], dtype=np.uint64),
            np.asarray([b for _, b in ranges], dtype=np.uint64))


def _static_interval_arrays(maps: list) -> tuple:
    """Coalesced (starts, ends) arrays for regions a type pointer may target."""
    ranges = _coalesce_ranges([
        (int(r.get("start", 0)), int(r.get("end", 0)))
        for r in maps
        if int(r.get("end", 0)) > int(r.get("start", 0)) and _is_static_region(r)])
    if not ranges:
        return (np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64))
    return (np.asarray([a for a, _ in ranges], dtype=np.uint64),
            np.asarray([b for _, b in ranges], dtype=np.uint64))


def _values_in_intervals(values: np.ndarray, starts: np.ndarray,
                         ends: np.ndarray) -> np.ndarray:
    """Boolean mask: which values fall inside any [start, end) interval."""
    if not len(starts) or not len(values):
        return np.zeros(len(values), dtype=bool)
    idx = np.searchsorted(starts, values, side="right") - 1
    inside = idx >= 0
    clipped = np.clip(idx, 0, len(starts) - 1)
    return inside & (values < ends[clipped])


def _group_type_pointers(values: np.ndarray, holders: np.ndarray,
                         min_instances: int = _TYPE_SCAN_MIN_INSTANCES) -> list:
    """Group holder addresses by the repeated pointer value they contain.

    A value seen once is an ordinary pointer. A value seen at the base of many
    similarly-shaped allocations is a type identity, which is the whole basis
    of the technique.
    """
    if not len(values):
        return []
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_holders = holders[order]
    unique, first_index, counts = np.unique(
        sorted_values, return_index=True, return_counts=True)
    keep = counts >= int(min_instances)
    groups = []
    for value, start, count in zip(unique[keep], first_index[keep],
                                   counts[keep]):
        instances = sorted_holders[int(start):int(start) + int(count)]
        groups.append({
            "type_ptr": int(value),
            "count": int(count),
            "instances": np.sort(instances[:_TYPE_SCAN_MAX_INSTANCES]).copy(),
        })
    groups.sort(key=lambda g: (-g["count"], g["type_ptr"]))
    return groups[:_TYPE_SCAN_MAX_TYPES]


def _group_counted_types(counts: dict, instances: dict,
                         min_instances: int = _TYPE_SCAN_MIN_INSTANCES) -> list:
    """Shape already-tallied counts into the same rows _group_type_pointers
    returns, so both paths hand the UI one format."""
    groups = []
    for value, count in counts.items():
        if count < int(min_instances):
            continue
        addrs = sorted(instances.get(value, ()))[:_TYPE_SCAN_MAX_INSTANCES]
        groups.append({
            "type_ptr": int(value),
            "count": int(count),
            "instances": np.asarray(addrs, dtype=np.uint64),
        })
    groups.sort(key=lambda g: (-g["count"], g["type_ptr"]))
    return groups[:_TYPE_SCAN_MAX_TYPES]


# ── field corroboration across sibling instances ──────────────────────────────
# A candidate address that follows the value through several scans is still
# only correlated with it. RDX's existing answers to that are the causal test
# (write it and watch the game) and the two-reload promotion rule -- both good,
# both requiring either a risky write or a game restart.
#
# There is a third kind of evidence available for free once Type Scan works,
# and it needs neither: if the address is a *field* of an object, every other
# live instance of the same type has that field too. Reading the same offset
# across siblings either corroborates the structural claim or refutes it.
#
# This is the "multiple observations" and "instance comparison" argument -- a
# cluster of related fields, and a consistent offset across instances of one
# type, make an accidental identification much less likely. It is a
# falsification test in the proper sense: it can fail, and failing means the
# address is probably not the field it looks like.
#
# Deliberately reports a distribution rather than a verdict. "17 of 20 siblings
# hold a small integer here" is evidence a person can weigh; a boolean would
# hide how thin the evidence sometimes is.
_FIELD_SIBLING_SAMPLE = 24         # siblings read; enough to be evidence, cheap
_FIELD_MAX_OBJECT_SPAN = 0x2000    # a candidate further than this from a base
                                   # is not plausibly a field of that object


def locate_field_in_type(address: int, groups: list,
                         max_span: int = _FIELD_MAX_OBJECT_SPAN) -> Optional[dict]:
    """Which type's instance contains `address`, and at what field offset.

    Instances are object bases. The owning instance is the greatest base at or
    below the address, provided the distance is small enough to be a field
    rather than a coincidence of ordering.
    """
    address = int(address)
    best = None
    for group in groups or ():
        instances = group.get("instances")
        if instances is None or len(instances) == 0:
            continue
        arr = np.asarray(instances, dtype=np.uint64)
        idx = int(np.searchsorted(arr, np.uint64(address), side="right")) - 1
        if idx < 0:
            continue
        base = int(arr[idx])
        offset = address - base
        if not 0 <= offset < int(max_span):
            continue
        # Nearest owning base wins: a smaller offset is a more specific claim.
        if best is None or offset < best["field_offset"]:
            best = {"type_ptr": int(group.get("type_ptr", 0)),
                    "class_name": group.get("class_name"),
                    "instance_base": base,
                    "field_offset": int(offset),
                    "instances": arr}
    return best


def corroborate_field_across_instances(ip: str, pid: int, finding: dict,
                                       width: int = 4,
                                       value_type: str = "u32",
                                       sample: int = _FIELD_SIBLING_SAMPLE,
                                       cancel_event=None) -> dict:
    """Read the same field offset in sibling instances of the same type.

    Returns counts, never a verdict. `plausible` counts siblings whose value
    at that offset shares the shape of the candidate's -- for an integer
    field, the same order of magnitude; that is weak evidence individually and
    meaningful in aggregate.
    """
    arr = finding.get("instances")
    offset = int(finding.get("field_offset", 0))
    base = int(finding.get("instance_base", 0))
    out = {"read": 0, "plausible": 0, "sampled": 0, "values": []}
    if arr is None or not len(arr):
        return out
    siblings = [int(a) for a in np.asarray(arr) if int(a) != base]
    if not siblings:
        return out
    step = max(1, len(siblings) // max(1, int(sample)))
    siblings = siblings[::step][:int(sample)]
    try:
        anchor_value = _unpack_typed_value(
            ps5_read(ip, pid, base + offset, width), value_type, width)
    except Exception:
        return out
    for addr in siblings:
        if cancel_event is not None and cancel_event.is_set():
            break
        out["sampled"] += 1
        try:
            raw = ps5_read(ip, pid, addr + offset, width)
            value = _unpack_typed_value(raw, value_type, width)
        except Exception:
            continue
        out["read"] += 1
        out["values"].append(value)
        if _same_magnitude(anchor_value, value):
            out["plausible"] += 1
    return out


def _same_magnitude(a, b) -> bool:
    """Whether two field values look like the same kind of quantity.

    Intentionally crude. The claim being tested is structural -- "this offset
    holds the same sort of thing in every instance" -- not that the values
    match, which for per-object state they should not.
    """
    try:
        a, b = abs(float(a)), abs(float(b))
    except (TypeError, ValueError):
        return False
    if a == 0.0 or b == 0.0:
        return a == b
    ratio = a / b if a > b else b / a
    return ratio <= 1000.0


def scan_type_instances(ip: str, pid: int,

                        min_instances: int = _TYPE_SCAN_MIN_INSTANCES,
                        cancel_event=None, progress_cb=None) -> list:
    """Find live object instances grouped by the type pointer at their base.

    Read-only. Streams heap regions, keeps 8-aligned qwords that point at a
    mapped, non-executable address, groups the holders by that value, and
    keeps only the groups whose pointer resolves to a class name.

    The target filter used to require a *static/module* region, on the
    assumption that a type identity lives in the image. Measured against
    CUSA01659 on ps5debug-NG, that assumption is wrong for IL2CPP: every
    `Il2CppClass` for this title is heap-allocated. The nine holders of the
    "PlayerController" name string all sat at 0x2xxxxxxxx with prot=3 and
    `_is_static_region` False, and the class itself resolved at
    0x20362e560 -- heap, not module.

    So every real type pointer was excluded, and what survived were the
    pointers that *do* target the image: vtable and callback entries. The
    top five groups of a 512-group scan disassembled as x86-64 prologues
    (`55 48 89 e5 41 56 53` and friends), and `label_type_groups` named 0 of
    40 because none of them were classes. `_read_klass_name` was never at
    fault -- handed the real pointer it returns "PlayerController" from
    offset +0x18 on the first try.

    Frequency alone cannot replace the removed filter: without it every
    repeated pointer is a candidate. The discriminator is that a type
    pointer resolves to a plausible class name, which is a check this module
    already had. Executable targets are excluded up front because a class
    never lives in code, which removes the old dominant noise cheaply.
    """
    if cancel_event is None:
        cancel_event = threading.Event()
    maps = _get_maps_cached(ip, pid)
    if not maps:
        return []
    starts, ends = _type_target_interval_arrays(maps)
    if not len(starts):
        add_log("Type scan: no regions a type pointer could target", "warn")
        return []

    # Objects live in writable, non-static mappings. Excluding static regions
    # keeps the module's own pointer tables out of the instance set -- those
    # repeat too, and would otherwise dominate the result.
    heap = [r for r in _pointer_readable_regions(maps)
            if (int(r.get("prot", 0)) & 0x2) and not _is_static_region(r)]
    heap = _coalesce_scan_regions(heap)
    if not heap:
        add_log("Type scan: no writable heap regions", "warn")
        return []

    total_bytes = max(sum(r["end"] - r["start"] for r in heap), 1)
    done_bytes = 0
    # Aggregate per chunk rather than collecting every candidate and grouping
    # at the end. The collect-then-group form peaked at 268 MB on a 256 MiB
    # heap and would reach roughly 500 MB at the candidate cap -- badly out of
    # proportion for a tool that caps its whole undo history at 128 MB.
    # Counting as we go makes peak memory a function of how many distinct
    # type pointers exist, which is thousands, not of how much heap was read.
    counts: dict = {}
    instances: dict = {}
    collected = 0
    truncated = False
    read_failures = 0
    consecutive_failures = 0
    successful_reads = 0
    last_failure: Optional[Exception] = None
    sock = None
    try:
        sock = _ScanSocket(ip, pid)
        for region in heap:
            cursor = int(region["start"])
            end = int(region["end"])
            while cursor < end:
                if cancel_event.is_set():
                    raise InterruptedError("type scan cancelled")
                size = min(_TYPE_SCAN_CHUNK, end - cursor)
                try:
                    raw = sock.read(cursor, size, cancel_event)
                    consecutive_failures = 0
                    successful_reads += 1
                except Exception as exc:
                    # An unreadable span mid-heap is normal and must not
                    # abandon a scan that is otherwise working. A *run* of
                    # failures is not normal -- it means the console went
                    # away, and swallowing that produced "0 types found",
                    # which reads exactly like "this title has no type
                    # pointers". Two very different things, same output.
                    read_failures += 1
                    consecutive_failures += 1
                    last_failure = exc
                    if consecutive_failures >= _TYPE_SCAN_MAX_CONSECUTIVE_FAILS:
                        raise ConnectionError(
                            f"{consecutive_failures} consecutive read failures "
                            f"from {hex(cursor)}; the console stopped "
                            f"responding ({exc})") from exc
                    cursor += size
                    done_bytes += size
                    continue
                usable = len(raw) - (len(raw) % 8)
                if usable >= 8:
                    values = np.frombuffer(raw[:usable], dtype="<u8")
                    mask = _values_in_intervals(values, starts, ends)
                    if mask.any():
                        hits = np.flatnonzero(mask)
                        room = _TYPE_SCAN_MAX_CANDIDATES - collected
                        if len(hits) > room:
                            hits = hits[:room]
                            truncated = True
                        chunk_values = values[hits]
                        chunk_holders = (np.uint64(cursor)
                                         + hits.astype(np.uint64) * 8)
                        uniq, first, cnt = np.unique(
                            chunk_values, return_index=True,
                            return_counts=True)
                        for value, idx, count in zip(uniq.tolist(),
                                                     first.tolist(),
                                                     cnt.tolist()):
                            if (value not in counts
                                    and len(counts) >= _TYPE_SCAN_MAX_DISTINCT):
                                # Refusing new keys keeps the dict bounded
                                # without discarding tallies already built.
                                truncated = True
                                continue
                            counts[value] = counts.get(value, 0) + count
                            kept = instances.setdefault(value, [])
                            if len(kept) < _TYPE_SCAN_MAX_INSTANCES:
                                same = chunk_holders[chunk_values == value]
                                room_i = _TYPE_SCAN_MAX_INSTANCES - len(kept)
                                kept.extend(int(a) for a in same[:room_i])
                        collected += len(hits)
                cursor += size
                done_bytes += size
                if progress_cb:
                    progress_cb(min(done_bytes, total_bytes), total_bytes)
                if collected >= _TYPE_SCAN_MAX_CANDIDATES:
                    truncated = True
                    break
            if collected >= _TYPE_SCAN_MAX_CANDIDATES:
                break
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    # A heap small enough to fit in one or two chunks can never reach the
    # consecutive-failure threshold, so that rule alone would still let a dead
    # link look like an empty result. Nothing read at all is the unambiguous
    # case and is checked regardless of how many chunks there were.
    if read_failures and successful_reads == 0:
        raise ConnectionError(
            f"every chunk read failed ({read_failures}); the console stopped "
            f"responding ({last_failure})")
    if read_failures:
        # Say so even on a successful scan: a partial read set changes what a
        # thin result means.
        add_log(f"Type scan: {read_failures} chunk read(s) failed and were "
                f"skipped (last: {last_failure})", "warn")
    if not counts:
        if read_failures:
            add_log("Type scan found nothing, but reads were failing — this "
                    "is not evidence that the title has no type pointers",
                    "warn")
        return []
    groups = _group_counted_types(counts, instances, min_instances)
    if truncated:
        add_log(f"Type scan: candidate cap ({_TYPE_SCAN_MAX_CANDIDATES:,}) hit; "
                f"results are partial", "warn")
    # Name the type pointer's home module -- that is what makes a row readable.
    for group in groups:
        module_name, _base, rel = _module_info_for_addr(group["type_ptr"], maps)
        group["module_name"] = module_name or ""
        group["module_relative_offset"] = rel
    add_log(f"Type scan: {len(groups)} type(s) with >= {min_instances} "
            f"instances from {collected:,} candidate slot(s)")
    # Follow each type pointer one more hop for its class name. Bounded and
    # entirely optional -- a title that does not use this layout simply shows
    # pointers, exactly as before.
    #
    # With the target filter widened to all readable non-executable memory,
    # frequency alone no longer separates a type from any other repeated
    # pointer. Resolving the class name is what does, so this pass is now
    # load-bearing rather than cosmetic: named groups are known types and are
    # ranked first. Unnamed ones are kept -- a title whose layout is unknown
    # must still show its pointers, which is the behaviour this feature has
    # always promised -- but they no longer bury the real answers.
    groups.sort(key=lambda g: -int(g.get("count", 0)))
    try:
        named = label_type_groups(ip, pid, groups, maps,
                                  limit=_TYPE_SCAN_LABEL_LIMIT,
                                  time_budget=_TYPE_SCAN_LABEL_BUDGET,
                                  cancel_event=cancel_event)
        if named:
            add_log(f"Type scan: resolved {named} class name(s) from live "
                    f"memory (no dump.cs needed)")
            groups.sort(key=lambda g: (not g.get("class_name"),
                                       -int(g.get("count", 0))))
        elif groups:
            add_log("Type scan: no class names resolved — this title may not "
                    "use an IL2CPP layout, or its name offset is unknown",
                    "warn")
    except Exception as exc:
        add_log(f"Type scan: class-name lookup unavailable: {exc}", "warn")
    return groups


# ── IL2CPP class names from live memory ───────────────────────────────────────
# Type Scan already finds the thing this needs. Every IL2CPP object begins with
# a pointer to its Il2CppClass, and Type Scan groups objects by exactly that
# qword, returning it as `type_ptr`. The pointer was being used only to name
# the module it lives in; following it one more hop yields the class name.
#
# Breeze does this on Switch and states the shape plainly:
#
#     class_info = [candidate_base]
#     class_name = [[candidate_base] + 0x10]     # c-string
#
# The offset is the part not to copy. Il2CppClass has been reordered between
# IL2CPP releases, so a hardcoded 0x10 would name classes correctly on some
# titles and confidently produce garbage on others -- which, for a feature
# whose entire job is labelling, is worse than having no feature. So: probe a
# small set of plausible offsets, follow each to a string, and accept one only
# if it reads like a real type name. When nothing does, say nothing and leave
# the raw pointer on screen.
#
# All read-only.
_KLASS_NAME_OFFSETS = (0x10, 0x08, 0x18, 0x20, 0x28, 0x00)
_KLASS_NAME_MAX_LEN = 128
_KLASS_NAME_CACHE_MAX = 4096
_klass_name_cache: dict = {}
# klass pointer -> [(offset, name), ...] when more than one offset resolved.
# Consulted by label_type_groups so a contested name is shown as contested.
_klass_name_ambiguous: dict = {}
_klass_name_lock = threading.Lock()
# Enough C# identifier characters to accept generics, nested types and
# namespaced names without accepting arbitrary binary that happens to be
# printable.
_KLASS_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.<>`|+\[\], ]{0,127}$")


def _plausible_class_name(raw: bytes) -> Optional[str]:
    """Decode a candidate C string, or None when it is not a type name."""
    end = raw.find(b"\x00")
    if end <= 0:                     # empty string, or no terminator in range
        return None
    try:
        text = raw[:end].decode("ascii")
    except UnicodeDecodeError:
        return None
    if not _KLASS_NAME_RE.match(text):
        return None
    # A name that is all punctuation-ish, or a single character, is far more
    # likely to be coincidence than a type.
    if len(text) < 2 or not any(c.isalpha() for c in text):
        return None
    return text


def _read_klass_name(ip: str, pid: int, klass_ptr: int,
                     maps: Optional[list] = None) -> Optional[str]:
    """Resolve an Il2CppClass pointer to its type name, or None.

    Never raises: this is a labelling convenience layered on Type Scan, and a
    title that does not use this layout must simply go unlabelled rather than
    fail the scan that found it.
    """
    key = int(klass_ptr)
    with _klass_name_lock:
        if key in _klass_name_cache:
            return _klass_name_cache[key]
    name = None
    try:
        maps = maps if maps is not None else _get_maps_cached(ip, pid)
        starts, rows = _build_region_lookup(maps)
        # One read covers every candidate offset instead of one read each.
        span = max(_KLASS_NAME_OFFSETS) + 8
        header = ps5_read(ip, pid, key, span)
        matches = []
        for offset in _KLASS_NAME_OFFSETS:
            if offset + 8 > len(header):
                continue
            name_ptr = int.from_bytes(header[offset:offset + 8], "little")
            if not (_ADDR_MIN <= name_ptr <= _ADDR_MAX):
                continue
            # Check the target is mapped before dereferencing it; an
            # unmapped read is a wasted round trip and a logged failure.
            if not _region_for_addr(name_ptr, starts, rows):
                continue
            try:
                raw = ps5_read(ip, pid, name_ptr, _KLASS_NAME_MAX_LEN)
            except Exception:
                continue
            candidate = _plausible_class_name(raw)
            if candidate:
                matches.append((offset, candidate))
        if matches:
            name = matches[0][1]
        # Do not let a coin-flip look like a fact. Measured on CUSA01659:
        # 8 of 40 type pointers had more than one offset yielding a plausible
        # name, and the offset is not consistent even between real classes --
        # PlayerController's name is at +0x18 with +0x10 empty, while String
        # and Boolean carry theirs at +0x10 with the *namespace* at +0x18.
        # One structure returned 'TargetPlayer' from +0x10 and
        # 'PlayerController' from +0x18: a field name winning over a class
        # name purely because of probe order.
        #
        # There is no rule here worth trusting yet, so the first hit is still
        # what is reported -- unchanged behaviour -- but a contested read is
        # recorded rather than presented as settled.
        if len(matches) > 1:
            with _klass_name_lock:
                _klass_name_ambiguous[key] = list(matches)
    except Exception:
        name = None
    with _klass_name_lock:
        if len(_klass_name_cache) >= _KLASS_NAME_CACHE_MAX:
            _klass_name_cache.clear()
        _klass_name_cache[key] = name
    return name


def _invalidate_klass_names() -> None:
    """Class pointers are process-scoped; drop them when the process changes."""
    with _klass_name_lock:
        _klass_name_cache.clear()
        _klass_name_ambiguous.clear()


def label_type_groups(ip: str, pid: int, groups: list,
                      maps: Optional[list] = None,
                      limit: int = 64,
                      time_budget: float = 8.0,
                      cancel_event=None) -> int:
    """Attach a live class name to as many groups as the budget allows.

    Bounded on purpose. Each name costs up to two round trips and a scan can
    return hundreds of types, so this labels the rows a user will actually
    look at first and leaves the rest showing their pointer. Returns how many
    were named.
    """
    named = 0
    deadline = time.monotonic() + max(float(time_budget), 0.1)
    for group in list(groups)[:max(int(limit), 0)]:
        if time.monotonic() >= deadline:
            break
        if cancel_event is not None and cancel_event.is_set():
            break
        name = _read_klass_name(ip, pid, int(group.get("type_ptr", 0)), maps)
        if name:
            group["class_name"] = name
            with _klass_name_lock:
                contested = _klass_name_ambiguous.get(
                    int(group.get("type_ptr", 0)))
            if contested:
                group["class_name_ambiguous"] = [
                    (int(o), str(v)) for o, v in contested]
                add_log(
                    f"Type {int(group.get('type_ptr', 0)):#x}: name is "
                    + " / ".join(f"{v!r} at +{o:#x}" for o, v in contested)
                    + f" — showing {name!r}", "warn")
            named += 1
    return named


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
    # PS5 map enumeration can report a large reservation alongside smaller
    # overlapping augmented rows (see _build_region_lookup's docstring for the
    # same hazard). Trusting whichever covering row happens to appear first
    # lets a non-writable overlay stub falsely block a write into a mapping
    # that is legitimately writable underneath it.
    covering = [r for r in maps if r['start'] <= addr and addr + length <= r['end']]
    if not covering:
        return f"Address {hex(addr)} is not in any mapped region of PID {pid}."
    for r in covering:
        if r['prot'] & PROT_WRITE:
            return None   # some covering region grants write access — OK
    return (f"Address {hex(addr)} is mapped but not writable "
            f"(prot={hex(covering[0]['prot'])}).")


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
        base = _pointer_module_base(maps, wanted)
        if base is not None:
            return base + int(cheat["module_relative_offset"])
        raise RuntimeError(f"module '{wanted}' is not currently mapped")
    base = cheat.get("base", 0)
    return int(base, 0) if isinstance(base, str) else int(base)


def _is_cross_reload_pointer(cheat: dict) -> bool:
    """Whether a cheat has enough persisted evidence to survive reconnects."""
    return bool(
        cheat.get("offsets") is not None
        and cheat.get("module_name")
        and cheat.get("module_relative_offset") is not None
        and cheat.get("cross_reload_validated") is True
        and cheat.get("game_identity")
    )


def _is_module_relative_scalar(cheat: dict) -> bool:
    """Whether a flat write has a relocatable static-module address."""
    return bool(
        cheat.get("offsets") is None
        and cheat.get("module_name")
        and cheat.get("module_relative_offset") is not None
        and cheat.get("game_identity")
    )


def _is_portable_cheat(cheat: dict) -> bool:
    return bool(cheat.get("game_identity")) and (
        _is_cross_reload_pointer(cheat) or _is_module_relative_scalar(cheat))


def _portable_cheat_matches_current_game(cheat: dict) -> bool:
    """Fail closed before rebasing a trainer into another eboot.bin title."""
    if not _is_portable_cheat(cheat):
        return False
    try:
        maps = _get_maps_cached(state["ip"], state["pid"])
        current = _pointer_game_identity(state.get("proc_name", ""), maps)
        return str(cheat.get("game_identity", "")) == current
    except Exception:
        return False


def _runtime_scalar_address(cheat: dict) -> int:
    """Rebase a static scalar patch, or return its legacy absolute address."""
    if _is_module_relative_scalar(cheat):
        maps = _get_maps_cached(state["ip"], state["pid"])
        base = _pointer_module_base(maps, str(cheat["module_name"]))
        if base is None:
            raise RuntimeError(
                f"module '{cheat['module_name']}' is not currently mapped")
        return int(base) + int(cheat["module_relative_offset"])
    return int(cheat["address"])


def _pointer_module_base(maps: list, module_name: str) -> Optional[int]:
    """Return the current base for a persisted pointer-root module."""
    wanted = str(module_name or "main")
    if wanted.startswith(_SECTION_MODULE_PREFIX):
        match = re.fullmatch(
            re.escape(_SECTION_MODULE_PREFIX) +
            r"([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)",
            wanted)
        if not match:
            return None
        prot, map_type, size = (int(match.group(i), 16) for i in range(1, 4))
        ordinal = int(match.group(4))
        peers = sorted((r for r in maps
                        if _is_static_region(r)
                        and _is_generic_map_name(str(r.get("name", "") or ""))
                        and _section_signature(r) == (prot, map_type, size)),
                       key=lambda r: (int(r["start"]), int(r["end"])))
        if 0 <= ordinal < len(peers):
            return int(peers[ordinal]["start"])
        return None
    same = [r for r in maps if str(r.get("name", "") or "") == wanted]
    if not same:
        # ps5debug reports a bare module name ("Il2CppUserAssemblies.prx")
        # where MemDBG reports the full vnode path
        # ("/app0/Il2CppUserAssemblies.prx"), so a pointer chain rooted in a
        # library and validated under one backend could not rebase under the
        # other -- the chain silently stopped resolving and the trainer was
        # dead. _is_main_module_name already does basename matching for the
        # main image; library roots need the same. Exact match still wins,
        # so this only ever adds a fallback.
        wanted_base = wanted.replace("\\", "/").rsplit("/", 1)[-1]
        if wanted_base and not _is_generic_map_name(wanted_base):
            same = [r for r in maps
                    if str(r.get("name", "") or "").replace(
                        "\\", "/").rsplit("/", 1)[-1] == wanted_base]
            # Two modules can share a basename in different directories. The
            # min(start) below would then silently pick one and rebase the
            # chain into the wrong image -- which resolves to a plausible
            # address and fails silently. Say so rather than guessing quietly.
            distinct = {str(r.get("name", "") or "") for r in same}
            if len(distinct) > 1:
                add_log(f"Pointer root '{wanted}' matches {len(distinct)} "
                        f"differently-pathed modules with the same name "
                        f"({', '.join(sorted(distinct)[:3])}); rebasing to the "
                        f"lowest-addressed one, which may be wrong", "warn")
    if same:
        return min(int(r["start"]) for r in same)
    if wanted == "main":
        static = [r for r in maps if _is_static_region(r)]
        return min((int(r["start"]) for r in static), default=None)
    return None


def _save_pointer_provisionals(records: list,
                               path: Optional[Path] = None,
                               epochs: Optional[list] = None) -> None:
    """Atomically persist provisional chains for validation after a reload."""
    dst = Path(path or _POINTER_PROVISIONAL_FILE)
    payload = {"version": 1, "candidates": records}
    if epochs is not None:
        payload["epochs"] = list(epochs)[-_PTR_EPOCH_LOG_MAX:]
    else:
        # Preserve a log written by an earlier call: callers that only
        # rewrite the candidate list must not silently drop the history.
        existing = _load_pointer_epochs(dst)
        if existing:
            payload["epochs"] = existing
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, dst)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def _load_pointer_provisionals(path: Optional[Path] = None) -> list:
    """Load the two-reload pointer project; a damaged file fails closed.

    This runs from screen_main on every entry to the main menu, so anything
    that escapes here takes down the UI immediately after connecting, with no
    way back in short of knowing which hidden file to delete. `data` must be
    shape-checked before use: valid JSON of the wrong type (`[]`, `null`, a
    bare number or string) makes `data.get` raise AttributeError, which is not
    a subclass of any exception listed below. _load_preferences already guards
    this way; this loader did not.
    """
    src = Path(path or _POINTER_PROVISIONAL_FILE)
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or int(data.get("version", 0)) != 1:
            return []
        candidates = data.get("candidates", [])
        if not isinstance(candidates, list):
            return []
        return [x for x in candidates if isinstance(x, dict)]
    except (OSError, ValueError, TypeError, AttributeError):
        return []


_PTR_EPOCH_LOG_MAX = 12          # reload epochs retained for the funnel view


def _load_pointer_epochs(path: Optional[Path] = None) -> list:
    """Load the per-reload survivor log; a damaged file fails closed.

    Kept separate from _load_pointer_provisionals so a corrupt or absent log
    can never stop the candidates themselves from loading — the log is a
    narration of how the project got here, not part of the promotion rule.
    """
    src = Path(path or _POINTER_PROVISIONAL_FILE)
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or int(data.get("version", 0)) != 1:
            return []
        epochs = data.get("epochs", [])
        if not isinstance(epochs, list):
            return []
        return [e for e in epochs if isinstance(e, dict)][-_PTR_EPOCH_LOG_MAX:]
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def _record_pointer_epoch(result: dict, process: str = "",
                          path: Optional[Path] = None) -> list:
    """Append one reload epoch to the log and return the whole log.

    PINCE's user watches a pointer-map file shrink at each filter pass, so
    the narrowing is visible and it is obvious when it has stopped paying.
    RDX's two-reload gate is stricter and stays exactly as it is; this only
    makes the same narrowing legible, by recording what each epoch started
    with, what survived, and why the rest died.
    """
    survivors = result.get("survivors", []) or []
    rejected = result.get("rejected", []) or []
    reasons: dict = {}
    for record in rejected:
        reason = str(record.get("rejection_reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    epochs = _load_pointer_epochs(path)
    epochs.append({
        "epoch": len(epochs) + 1,
        "considered": len(survivors) + len(rejected),
        "survived": len(survivors),
        "rejected": len(rejected),
        "reasons": reasons,
        "process": str(process or ""),
        "at": time.strftime("%Y-%m-%d %H:%M"),
    })
    return epochs[-_PTR_EPOCH_LOG_MAX:]


def _format_epoch_rows(epochs: list) -> list:
    """Render the epoch log as aligned funnel rows for the project screen."""
    rows = []
    for entry in epochs:
        considered = int(entry.get("considered", 0) or 0)
        survived = int(entry.get("survived", 0) or 0)
        pct = (100.0 * survived / considered) if considered else 0.0
        reasons = entry.get("reasons", {}) or {}
        top = ""
        if reasons:
            worst = max(reasons.items(), key=lambda kv: kv[1])
            top = f"   mostly: {worst[0]} ({worst[1]})"
        rows.append(
            f"Reload {int(entry.get('epoch', 0)):<2} "
            f"{considered:>5} -> {survived:<5} kept ({pct:4.0f}%)"
            f"{top}")
    return rows


def _coerce_int_field(record: dict, key: str, default: int = 0) -> int:
    """Read an int from an untrusted persisted record without raising.

    Records survive across releases and hand edits, so a field can hold a
    string, None, or a float. int() raises ValueError/TypeError on those, and
    the summary below runs inside screen_main where that is fatal.
    """
    try:
        return int(record.get(key, default))
    except (TypeError, ValueError):
        return default


def _pointer_project_summary(process: str = "", maps: Optional[list] = None,
                             path: Optional[Path] = None) -> dict:
    """Summarise the persisted two-reload pointer validation project."""
    records = _load_pointer_provisionals(path)
    wanted_process = str(process or "")
    if wanted_process:
        records = [r for r in records
                   if str(r.get("observed_process", "") or "") in
                   ("", wanted_process)]
    game_identity = (_pointer_game_identity(wanted_process, maps)
                     if maps else "")
    if game_identity:
        records = [r for r in records
                   if str(r.get("observed_game", "") or "") in
                   ("", game_identity)]
    survivals = max((_coerce_int_field(r, "reload_survivals") for r in records),
                    default=0)
    epochs = _load_pointer_epochs(path)
    return {
        "count": len(records),
        "survivals": min(max(survivals, 0), 2),
        "complete": survivals >= 2,
        "epochs": epochs,
        "target": (_coerce_int_field(records[0], "observed_target")
                   if records else None),
        "process": wanted_process,
        "game_identity": game_identity,
    }


def _clear_pointer_project(process: str = "", maps: Optional[list] = None,
                           path: Optional[Path] = None) -> int:
    """Remove only this game/process project; preserve unrelated candidates."""
    dst = Path(path or _POINTER_PROVISIONAL_FILE)
    all_records = _load_pointer_provisionals(dst)
    wanted_process = str(process or "")
    game_identity = (_pointer_game_identity(wanted_process, maps)
                     if maps else "")

    def belongs(record):
        process_matches = (not wanted_process or
                           str(record.get("observed_process", "") or "") in
                           ("", wanted_process))
        game_matches = (not game_identity or
                        str(record.get("observed_game", "") or "") in
                        ("", game_identity))
        return process_matches and game_matches

    kept = [record for record in all_records if not belongs(record)]
    removed = len(all_records) - len(kept)
    if removed:
        _save_pointer_provisionals(kept, dst)
    return removed


def _is_main_module_name(map_name, process: str) -> bool:
    """Match ps5debug labels and MemDBG full paths to the target image."""
    name = str(map_name or "").strip()
    process = str(process or "").strip()
    if not name:
        return False
    if name == "executable" or (process and name == process):
        return True
    name_base = name.replace("\\", "/").rsplit("/", 1)[-1]
    process_base = process.replace("\\", "/").rsplit("/", 1)[-1]
    return bool(process_base and name_base == process_base)


def _pointer_game_identity(process: str, maps: list) -> str:
    """Build an ASLR-independent identity for an attached game image.

    Process names alone are not sufficient because unrelated titles normally
    run as ``eboot.bin``. Prefer the named main image; when MemDBG only exposes
    generic vnode rows, use their ordered section metadata. Addresses are
    deliberately excluded so the identity survives relocation.
    """
    process = str(process or "")
    basename = process.replace("\\", "/").rsplit("/", 1)[-1]
    static = [r for r in maps if _is_static_region(r)]
    main_rows = [r for r in static
                 if _is_main_module_name(r.get("name", ""), process)]
    chosen = main_rows or sorted(static, key=lambda r: int(r["start"]))[:32]
    # Only fields BOTH backends report. ps5debug's map rows carry `offset`
    # and no `flags`; MemDBG's carry `flags` and no `offset`, so including
    # either made the same game on the same console fingerprint differently
    # depending on which payload was loaded — and this identity is the gate
    # for reusing a trainer, a portable cheat or a pointer project. A
    # ps5debug-built trainer was therefore rejected the moment the user
    # switched to MemDBG, with "game-image fingerprint does not match".
    # Name, protection and section size are ASLR-independent and reported
    # identically by both, which is what this fingerprint actually needs.
    signature = sorted((
        str(r.get("name", "") or ""),
        int(r.get("prot", 0)),
        int(r.get("end", 0)) - int(r.get("start", 0)),
    ) for r in chosen)
    encoded = json.dumps(signature, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return f"{basename or 'process'}:{digest}"


def _merge_pointer_provisionals(records: list, process: str,
                                path: Optional[Path] = None,
                                game_identity: Optional[str] = None,
                                epochs: Optional[list] = None) -> list:
    """Replace one game's candidates without erasing other games."""
    wanted = str(process or "")
    def same_scope(record):
        if str(record.get("observed_process", "")) not in ("", wanted):
            return False
        if game_identity is None:
            return True
        # Empty identities are prerelease records. Replace them in this process
        # scope so they cannot continue colliding with every eboot.bin title.
        return str(record.get("observed_game", "")) in ("", game_identity)

    preserved = [x for x in _load_pointer_provisionals(path)
                 if not same_scope(x)]
    merged = preserved + list(records)
    _save_pointer_provisionals(merged, path, epochs=epochs)
    return merged


def _make_pointer_provisionals(candidates: list, maps: list, pid: int,
                               process: str, target_addr: int) -> list:
    """Convert same-session hits into bounded, module-relative records."""
    unique = {}
    game_identity = _pointer_game_identity(process, maps)
    for candidate in candidates:
        if not candidate.get("verified") or not candidate.get("static"):
            continue
        holder = int(candidate.get("base", 0))
        module_name = candidate.get("module_name")
        module_base = candidate.get("module_base")
        module_rel = candidate.get("module_relative_offset")
        if module_rel is None:
            module_name, module_base, module_rel = _module_info_for_addr(holder, maps)
        if module_rel is None:
            continue
        offsets = [int(x) for x in candidate.get("offsets", [])]
        key = (str(module_name or "main"), int(module_rel), tuple(offsets),
               int(candidate.get("terminal_offset", 0)))
        record = {
            "status": "provisional",
            "module_name": str(module_name or "main"),
            "module_relative_offset": int(module_rel),
            "offsets": offsets,
            "terminal_offset": int(candidate.get("terminal_offset", 0)),
            "depth": len(offsets),
            "observed_pid": int(pid),
            "observed_process": str(process or ""),
            "observed_game": game_identity,
            "observed_target": int(target_addr),
            "reload_survivals": 0,
        }
        unique[key] = record
    # Prefer shorter paths and smaller offsets, and bound persistence so a
    # noisy executable data section cannot create an unwieldy reload pass.
    records = sorted(unique.values(), key=lambda c: (
        c["depth"], sum(abs(int(x)) for x in c["offsets"]),
        c["module_name"], c["module_relative_offset"]))
    return records[:256]


def _validate_pointer_provisionals(ip: str, pid: int, process: str,
                                   target_addr: int, records: list,
                                   maps: Optional[list] = None) -> dict:
    """Rebase saved chains and validate them in a new relocation epoch."""
    maps = maps if maps is not None else _get_maps_cached(ip, pid)
    game_identity = _pointer_game_identity(process, maps)
    survivors, rejected = [], []
    for saved in records:
        candidate = dict(saved)
        if str(saved.get("observed_process", "")) not in ("", str(process or "")):
            candidate["rejection_reason"] = "different process"
            rejected.append(candidate)
            continue
        if str(saved.get("observed_game", "")) not in ("", game_identity):
            candidate["rejection_reason"] = "different game image"
            rejected.append(candidate)
            continue
        same_pid = int(saved.get("observed_pid", pid)) == int(pid)
        same_target = int(saved.get("observed_target", target_addr)) == int(target_addr)
        # A relocated address is a new validation epoch even when the process
        # survives a scene/save reload. Requiring a new PID rejected the normal
        # PointerFinder-style "Next Scan" workflow.
        if same_pid and same_target:
            candidate["rejection_reason"] = "reload not detected"
            rejected.append(candidate)
            continue
        module_base = _pointer_module_base(maps, saved.get("module_name", "main"))
        if module_base is None:
            candidate["rejection_reason"] = "module not mapped"
            rejected.append(candidate)
            continue
        candidate["base"] = module_base + int(saved["module_relative_offset"])
        candidate["static"] = True
        candidate["region"] = saved.get("module_name", "main")
        if _verify_candidate_twice(ip, pid, candidate, int(target_addr)):
            survivals = int(saved.get("reload_survivals", 0)) + 1
            candidate["reload_survivals"] = survivals
            candidate["status"] = ("permanent" if survivals >= 2
                                     else "provisional")
            # A second survival must come from another actual game reload.
            candidate["observed_pid"] = int(pid)
            candidate["observed_process"] = str(process or "")
            candidate["observed_game"] = game_identity
            candidate["observed_target"] = int(target_addr)
            candidate["confidence"] = _candidate_confidence(candidate)
            survivors.append(candidate)
        else:
            candidate["status"] = "rejected"
            candidate["rejection_reason"] = "chain changed after reload"
            rejected.append(candidate)
    return {"survivors": survivors, "rejected": rejected}


def _hex_or_none(value):
    """hex(value), or None if value is missing — the common None-guard
    used repeatedly when serialising optional numeric trainer fields."""
    return hex(value) if value is not None else None


def generate_cht(cheats: list, game_id: str, game_ver: str,
                 game_title: str, hex_values: bool = True,
                 process: str = "eboot.bin") -> str:
    """Generate RDX's pointer-capable, round-trippable trainer format."""
    def fmt_val(c, value):
        type_key = _cheat_value_type(c)
        if type_key == "bytes":
            return _pack_typed_value(value, type_key, int(c["width"])).hex().upper()
        if VALUE_TYPES[type_key]["kind"] == "float":
            return float(value)
        return hex(int(value)) if hex_values else str(int(value))

    cheat_list = []
    for c in cheats:
        type_key = _cheat_value_type(c)
        is_pointer = "offsets" in c and c.get("offsets") is not None
        module_rel_hex = _hex_or_none(c.get("module_relative_offset"))
        original_value_fmt = (fmt_val(c, c["original_value"])
                              if c.get("original_value") is not None else None)
        if is_pointer:
            entry = {
                "name":    c["name"],
                "type":    c["type"],                    # pointer_freeze / pointer_write
                "base":    hex(c["base"]) if isinstance(c.get("base"), int) else c.get("base"),
                "offsets": [hex(o) for o in c["offsets"]],
                "module_name": c.get("module_name"),
                "module_relative_offset": module_rel_hex,
                "terminal_offset": (hex(int(c["terminal_offset"]))
                                     if c.get("terminal_offset") else None),
                "value":   fmt_val(c, c["value"]),
                "value_type": type_key,
                "bytes":   c["width"],
                "original_value": original_value_fmt,
                "cross_reload_validated": bool(
                    c.get("cross_reload_validated", False)
                    and c.get("game_identity")),
                "game_identity": str(c.get("game_identity", "") or ""),
            }
        else:
            entry = {
                "name":    c["name"],
                "type":    c["type"],
                "address": hex(c["address"]),
                "module_name": c.get("module_name"),
                "module_relative_offset": module_rel_hex,
                "value":   fmt_val(c, c["value"]),
                "value_type": type_key,
                "bytes":   c["width"],
                "original_value": original_value_fmt,
                "session_bound": not _is_module_relative_scalar(c),
                "game_identity": str(c.get("game_identity", "") or ""),
            }
        cheat_list.append(entry)
    identities = sorted({str(c.get("game_identity", "") or "")
                         for c in cheats if c.get("game_identity")})
    payload = {
        "title":     game_title,
        "titleid":   game_id,
        "version":   game_ver,
        "process":   process,
        "format":    "rdx-pointer-trainer-v1",
        "game_identity": identities[0] if len(identities) == 1 else None,
        "cheatList": cheat_list,
    }
    return json.dumps(payload, indent=2)


def _etahen_main_module(maps: list, process: str) -> tuple:
    """Return (base, accepted map names) for etaHEN's target module.

    etaHEN resolves every JSON patch as ``module.sections[0].vaddr + offset``.
    ps5debug normally labels the corresponding image ``executable`` whereas
    MemDBG may expose a real process/module name.  Generic ``[file]`` rows are
    intentionally not guessed: etaHEN cannot select one of several anonymous
    vnode images from a per-patch field.
    """
    process = str(process or "eboot.bin")
    same = [r for r in maps
            if _is_static_region(r)
            and _is_main_module_name(r.get("name", ""), process)]
    if same:
        return (min(int(r["start"]) for r in same),
                {str(r.get("name", "") or "") for r in same})
    return (None, set())


def _mods_to_import_entries(mods: list, source_path, module_base: int) -> list:
    """Convert an eligible-mods list (an etaHEN/GoldHEN JSON 'mods' array, or
    a decrypted .mc4's Cheat/Cheatline list via mc4_xml_to_mods()) into RDX's
    internal flat-cheat dicts.

    Neither format carries a value's semantic type (signed/float/etc.) — only
    raw on/off byte patches — so entries import as raw-byte ("bytes") writes:
    a faithful, lossless representation of exactly what will be written,
    rather than guessing a numeric interpretation that could be wrong.
    Addresses are resolved against `module_base`, the *currently attached*
    process's live main-module base — never trusted from the file — so a
    freshly imported entry is session-bound exactly like one built from a
    fresh scan, and only becomes portable (module_relative + game_identity)
    the same way any flat cheat does: via Export's own promotion logic.
    """
    entries = []
    for mod in mods:
        if not isinstance(mod, dict):
            continue
        name = str(mod.get("name") or "Unnamed cheat")
        for mem in mod.get("memory", []):
            if not isinstance(mem, dict):
                continue
            offset_hex = str(mem.get("offset", "")).strip()
            on_hex = str(mem.get("on", "")).strip().replace("-", "")
            off_hex = str(mem.get("off", "")).strip().replace("-", "")
            if not (offset_hex and on_hex):
                continue
            # A non-zero <Section> means the offset belongs to some other
            # module. RDX only resolves the main image here, so placing it
            # against module_base would be a confident write to the wrong
            # address — skip loudly instead. (.mc4 only; the etaHEN/GoldHEN
            # JSON schema has no section concept and omits the key.)
            section = mem.get("section", 0)
            if section != 0:
                add_log(f"Import: skipped '{name}' — it patches section "
                        f"{section}, not the main image, and RDX cannot "
                        "resolve a non-main section offset.", "warn")
                continue
            try:
                offset = int(offset_hex, 16)
                on_bytes = bytes.fromhex(on_hex)
            except ValueError:
                continue
            width = len(on_bytes)
            if width == 0:
                continue
            # A negative offset resolves below the module base, i.e. outside
            # the image entirely. Export already refuses these; import must
            # too, or a malformed file writes into whatever precedes it.
            if offset < 0:
                add_log(f"Import: skipped '{name}' — negative offset "
                        f"{offset_hex} resolves below the module base.", "warn")
                continue
            # _value_width() caps raw-byte values at 256, so a longer patch
            # produces a cheat that raises on every apply/export. The native
            # .rdx.json path already rejects these; this one silently kept
            # them.
            if width > 256:
                add_log(f"Import: skipped '{name}' — {width}-byte patch "
                        "exceeds the 256-byte raw-value limit.", "warn")
                continue
            if not (_ADDR_MIN <= module_base + offset <= _ADDR_MAX):
                continue
            entry = {
                "name": name, "type": "write",
                "address": module_base + offset,
                "value": on_hex.upper(), "value_type": "bytes", "width": width,
                "pid": state["pid"], "process": state["proc_name"],
                "session": state["session"], "imported_from": str(source_path),
            }
            if off_hex:
                try:
                    off_bytes = bytes.fromhex(off_hex)
                except ValueError:
                    add_log(f"Import: '{name}' has an invalid off-value "
                            "hex string — original value omitted.", "warn")
                else:
                    if len(off_bytes) == width:
                        entry["original_value"] = off_hex.upper()
                    else:
                        add_log(
                            f"Import: '{name}' off-value is "
                            f"{len(off_bytes)} byte(s), on-value is "
                            f"{width} — inconsistent widths, original "
                            "value omitted.", "warn")
            entries.append(entry)

    # A single <Cheat> can carry several <Cheatline>s under one name (RDX's
    # flat cheat model imports each as a separate entry), and a missing name
    # always falls back to the same string — either way, make every name
    # unique so cheat-list lookups that key by name (e.g. Freeze's cheat
    # picker) can't silently resolve to the wrong entry.
    seen = {}
    for entry in entries:
        base_name = entry["name"]
        seen[base_name] = seen.get(base_name, 0) + 1
        if seen[base_name] > 1:
            entry["name"] = f"{base_name} ({seen[base_name]})"
    return entries


def _scalar_hex_bytes(value, width: int,
                      value_type: Optional[str] = None) -> str:
    """etaHEN JSON stores raw little-endian bytes as uppercase hex."""
    width = int(width)
    return _pack_typed_value(value, value_type, width).hex().upper()


def generate_etahen_json(cheats: list, game_id: str, game_ver: str,
                         game_title: str, process: str, maps: list,
                         author: str = "RDX CheatMaker") -> tuple:
    """Generate an etaHEN-compatible static-patch JSON and a skip report.

    etaHEN's loader accepts module-relative byte patches only.  It has no field
    for RDX/PointerFinder-style dereference chains and it does not implement a
    repeated freeze.  Pointer/dynamic entries therefore remain in the native
    RDX trainer and are never misrepresented as etaHEN-compatible patches.
    """
    module_base, accepted_names = _etahen_main_module(maps, process)
    region_starts, region_rows = _build_region_lookup(maps)
    mods = []
    skipped = []
    for cheat in cheats:
        name = str(cheat.get("name", "Unnamed cheat"))
        if cheat.get("offsets") is not None:
            skipped.append((name, "pointer chain requires the RDX runtime"))
            continue
        if module_base is None:
            skipped.append((name, "main executable module was not identifiable"))
            continue
        try:
            address = int(cheat["address"])
            width = int(cheat["width"])
            saved_module = str(cheat.get("module_name", "") or "")
            saved_relative = cheat.get("module_relative_offset")
            if saved_module in accepted_names and saved_relative is not None:
                relative = int(saved_relative)
            else:
                region = _region_for_addr(address, region_starts, region_rows)
                region_name = str((region or {}).get("name", "") or "")
                if (region is None or not _is_static_region(region) or
                        region_name not in accepted_names):
                    skipped.append((name, "address is not in the target module"))
                    continue
                relative = address - int(module_base)
            if relative < 0:
                skipped.append((name, "address precedes the target module base"))
                continue
            original = cheat.get("original_value")
            if original is None:
                skipped.append((name, "original/off value is unknown"))
                continue
            value_type = _cheat_value_type(cheat)
            on_bytes = _scalar_hex_bytes(cheat["value"], width, value_type)
            off_bytes = _scalar_hex_bytes(original, width, value_type)
        except (KeyError, TypeError, ValueError, struct.error) as exc:
            skipped.append((name, f"invalid scalar patch: {exc}"))
            continue
        description = "RDX module-relative scalar write."
        inert = on_bytes == off_bytes
        if inert:
            description += (" On and off values are identical, so toggling "
                            "this writes back what is already there.")
        downgraded = str(cheat.get("type", "")) == "freeze"
        if downgraded:
            description += " etaHEN applies it once per toggle; it is not a live freeze."
        # The hint above never reaches the person running the cheat. Measured:
        # `.shn`/`.mc4` carry name, offset, on and off, and the Trainer XML
        # schema has no field for `hint` or `type` at all, so both are dropped
        # in transit. Provenance survives (the `Moder`/`credits` attributes),
        # but this particular note does not -- and it is the one that changes
        # what the cheat *does*.
        #
        # A cheat set up here as a continuous freeze becomes a one-shot write
        # in every manager that consumes these formats. That is a real
        # semantic downgrade, it varies per entry, and nothing else tells the
        # user about it. EdiZon SE solves the same class of problem by putting
        # what matters in the label, which is the one field every manager
        # shows.
        #
        # Marked only on the entries that are actually downgraded: a marker on
        # every row would be noise, and the name is what a player reads in a
        # menu.
        exported_name = name
        if downgraded:
            exported_name += f" {_ONE_SHOT_MARKER}"
        if inert:
            exported_name += f" {_INERT_MARKER}"
        mods.append({
            "name": exported_name,
            # "hint", not "description": a real etaHEN file
            # (PS5_Cheats json/CUSA00004_01.07.json) uses
            # {name, hint, type, memory} and GoldHEN's equivalent uses
            # {name, type, memory} with no such field at all. RDX emitted
            # "description", which neither manager reads -- so the note
            # explaining that a toggle is a one-shot write, not a live
            # freeze, was never shown to anyone. GoldHEN ignores the extra
            # key exactly as it ignored the old one.
            "hint": description,
            "type": "checkbox",
            "memory": [{
                "offset": f"{relative:X}",
                "on": on_bytes,
                "off": off_bytes,
            }],
        })
    payload = {
        "name": game_title,
        "id": game_id,
        "version": game_ver,
        "process": str(process or "eboot.bin"),
        "mods": mods,
        "credits": [str(author or "RDX CheatMaker")],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False), mods, skipped


# ── .mc4 (CheatRunner) trainer export ────────────────────────────────────────
#
# .mc4 is base64(AES-256-CBC(PKCS7(xml))) of the classic PS3/PS4 "Cheat
# Device" Trainer/Cheat/Cheatline XML schema.  Key/IV are constants fixed by
# etaHEN/CheatRunner (vendored in CheatRunner's third_party/mc4, GPLv3) —
# every consumer on a jailbroken console uses the same pair, so this is a
# public container format, not a secret.  No third-party crypto package is
# added for this (numpy is the project's only hard dependency, matching the
# existing hand-rolled LZ4 decoder for MemDBG); this is a compact from-scratch
# AES-256 block cipher, verified against the FIPS-197 AES-256 known-answer
# test and against a real published .mc4 sample decrypting to valid XML.

_MC4_AES256CBC_KEY = b"304c6528f659c766110239a51cl5dd9c"[:32]
_MC4_AES256CBC_IV = b"u@}kzW2u[u(8DWar"

_AES_SBOX = bytes((
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16))
_AES_INV_SBOX = bytes(_AES_SBOX.index(i) for i in range(256))
_AES_RCON = (0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36,0x6c,0xd8,0xab,0x4d)


def _aes_mul(a: int, b: int) -> int:
    """GF(2^8) multiply (AES's field, reduction poly 0x11b)."""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xff
        if hi:
            a ^= 0x1b
        b >>= 1
    return p


def _aes_key_expansion(key: bytes):
    nk = len(key) // 4
    nr = nk + 6
    w = [list(key[4 * i:4 * i + 4]) for i in range(nk)]
    for i in range(nk, 4 * (nr + 1)):
        temp = list(w[i - 1])
        if i % nk == 0:
            temp = temp[1:] + temp[:1]
            temp = [_AES_SBOX[b] for b in temp]
            temp[0] ^= _AES_RCON[i // nk - 1]
        elif nk > 6 and i % nk == 4:
            temp = [_AES_SBOX[b] for b in temp]
        w.append([w[i - nk][j] ^ temp[j] for j in range(4)])
    return w, nr


def _aes_encrypt_block(block: bytes, w, nr: int) -> bytes:
    state = [[block[r + 4 * c] for c in range(4)] for r in range(4)]

    def add_round_key(rnd):
        for c in range(4):
            for r in range(4):
                state[r][c] ^= w[rnd * 4 + c][r]

    add_round_key(0)
    for rnd in range(1, nr + 1):
        for r in range(4):
            for c in range(4):
                state[r][c] = _AES_SBOX[state[r][c]]
        for r in range(1, 4):
            state[r] = state[r][r:] + state[r][:r]
        if rnd != nr:
            for c in range(4):
                col = [state[r][c] for r in range(4)]
                state[0][c] = _aes_mul(col[0], 2) ^ _aes_mul(col[1], 3) ^ col[2] ^ col[3]
                state[1][c] = col[0] ^ _aes_mul(col[1], 2) ^ _aes_mul(col[2], 3) ^ col[3]
                state[2][c] = col[0] ^ col[1] ^ _aes_mul(col[2], 2) ^ _aes_mul(col[3], 3)
                state[3][c] = _aes_mul(col[0], 3) ^ col[1] ^ col[2] ^ _aes_mul(col[3], 2)
        add_round_key(rnd)
    return bytes(state[r][c] for c in range(4) for r in range(4))


def _aes_decrypt_block(block: bytes, w, nr: int) -> bytes:
    state = [[block[r + 4 * c] for c in range(4)] for r in range(4)]

    def add_round_key(rnd):
        for c in range(4):
            for r in range(4):
                state[r][c] ^= w[rnd * 4 + c][r]

    add_round_key(nr)
    for rnd in range(nr - 1, -1, -1):
        for r in range(1, 4):
            state[r] = state[r][-r:] + state[r][:-r]
        for r in range(4):
            for c in range(4):
                state[r][c] = _AES_INV_SBOX[state[r][c]]
        add_round_key(rnd)
        if rnd != 0:
            for c in range(4):
                col = [state[r][c] for r in range(4)]
                state[0][c] = (_aes_mul(col[0], 14) ^ _aes_mul(col[1], 11) ^
                               _aes_mul(col[2], 13) ^ _aes_mul(col[3], 9))
                state[1][c] = (_aes_mul(col[0], 9) ^ _aes_mul(col[1], 14) ^
                               _aes_mul(col[2], 11) ^ _aes_mul(col[3], 13))
                state[2][c] = (_aes_mul(col[0], 13) ^ _aes_mul(col[1], 9) ^
                               _aes_mul(col[2], 14) ^ _aes_mul(col[3], 11))
                state[3][c] = (_aes_mul(col[0], 11) ^ _aes_mul(col[1], 13) ^
                               _aes_mul(col[2], 9) ^ _aes_mul(col[3], 14))
    return bytes(state[r][c] for c in range(4) for r in range(4))


def _aes256_cbc_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    w, nr = _aes_key_expansion(key)
    pad = 16 - (len(plaintext) % 16)
    data = plaintext + bytes([pad]) * pad
    out = bytearray()
    prev = iv
    for i in range(0, len(data), 16):
        block = bytes(a ^ b for a, b in zip(data[i:i + 16], prev))
        enc = _aes_encrypt_block(block, w, nr)
        out += enc
        prev = enc
    return bytes(out)


def _aes256_cbc_decrypt(key: bytes, iv: bytes, ciphertext: bytes) -> bytes:
    if len(ciphertext) % 16:
        # Say what is actually wrong.  Without this the short final block
        # raises IndexError from inside the cipher, which the import path
        # reports as the vaguer "it may be corrupt".
        raise ValueError(
            f"ciphertext is {len(ciphertext)} bytes, not a multiple of the "
            "16-byte AES block size")
    w, nr = _aes_key_expansion(key)
    out = bytearray()
    prev = iv
    for i in range(0, len(ciphertext), 16):
        block = ciphertext[i:i + 16]
        dec = _aes_decrypt_block(block, w, nr)
        out += bytes(a ^ b for a, b in zip(dec, prev))
        prev = block
    if out:
        pad = out[-1]
        if 1 <= pad <= 16 and out[-pad:] == bytes([pad]) * pad:
            out = out[:-pad]
    return bytes(out)


def _mc4_encrypt(xml: bytes) -> bytes:
    cipher = _aes256_cbc_encrypt(_MC4_AES256CBC_KEY, _MC4_AES256CBC_IV, xml)
    return base64.b64encode(cipher)


def _mc4_decrypt(mc4_bytes: bytes) -> bytes:
    """Inverse of _mc4_encrypt. Used by _do_import_mc4() to decode a real
    .mc4 for import, and by the regression suite to verify a generated
    .mc4 round-trips to the exact XML that produced it."""
    cipher = base64.b64decode(mc4_bytes)
    return _aes256_cbc_decrypt(_MC4_AES256CBC_KEY, _MC4_AES256CBC_IV, cipher)


def _xml_escape(text) -> str:
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text


def _dash_hex(hexstr: str) -> str:
    """'90909090' -> '90-90-90-90', matching the community mc4 convention."""
    hexstr = str(hexstr).upper()
    return "-".join(hexstr[i:i + 2] for i in range(0, len(hexstr), 2))


def generate_trainer_xml(mods: list, game_id: str, game_ver: str, game_title: str,
                         process: str, author: str = "RDX CheatMaker") -> str:
    """Build the plaintext Trainer/Cheat/Cheatline XML document.

    This is the shared body of both console trainer containers: `.shn` is
    this string written as-is, and `.mc4` is the same string passed through
    _mc4_encrypt().  They are the same schema and the same consumers —
    CheatRunner and PS4CheaterNeo both read either — so splitting the
    document out of generate_mc4_bytes() is what lets RDX emit the pair
    without a second XML builder that could drift from this one.

    Takes the same already-resolved module-relative scalar patch list
    generate_etahen_json() produces (see its docstring): <Cheatline> has no
    field for pointer/dereference chains and these managers apply a toggle
    as a one-shot write like etaHEN, not a live freeze, so the eligible-cheat
    set is identical and is computed once by the caller.
    """
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             '<Trainer xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
             'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             f'Game="{_xml_escape(game_title)}" '
             f'Moder="{_xml_escape(author or "RDX CheatMaker")}" '
             f'Cusa="{_xml_escape(game_id)}" Version="{_xml_escape(game_ver)}" '
             f'Process="{_xml_escape(process or "eboot.bin")}">']
    for mod in mods:
        name = _xml_escape(mod.get("name", "Unnamed cheat"))
        lines.append(f'  <Cheat Control="Toggel" Text="{name}">')
        for entry in mod.get("memory", []):
            offset = str(entry.get("offset", "")).upper()
            on_hex = _dash_hex(entry.get("on", ""))
            off_hex = _dash_hex(entry.get("off", ""))
            lines.append('    <Cheatline>')
            lines.append(f'      <Offset>{offset}</Offset>')
            lines.append('      <Section>0</Section>')
            lines.append(f'      <ValueOn>{on_hex}</ValueOn>')
            lines.append(f'      <ValueOff>{off_hex}</ValueOff>')
            lines.append('    </Cheatline>')
        lines.append('  </Cheat>')
    lines.append('</Trainer>')
    return "\n".join(lines) + "\n"


def generate_shn_text(mods: list, game_id: str, game_ver: str, game_title: str,
                      process: str, author: str = "RDX CheatMaker") -> str:
    """Build a CheatRunner/PS4CheaterNeo-compatible .shn trainer.

    `.shn` is the plaintext form of the very document `.mc4` encrypts, so
    this is generate_trainer_xml() under the name its consumers use.

    Emitting it beside the .mc4 is also a diagnostic. The .mc4 path has never
    been consumed by a live CheatRunner (see HARDWARE_TEST_CHECKLIST.md), and
    a rejection there currently cannot distinguish a wrong schema from a
    wrong container — both fail identically. With the pair on disk, .shn
    accepted + .mc4 rejected isolates the fault to the AES/base64 container;
    both rejected isolates it to the schema.
    """
    return generate_trainer_xml(mods, game_id, game_ver, game_title,
                                process, author)


def generate_mc4_bytes(mods: list, game_id: str, game_ver: str, game_title: str,
                       process: str, author: str = "RDX CheatMaker") -> bytes:
    """Build a CheatRunner-compatible .mc4 trainer: the encrypted .shn."""
    xml_text = generate_trainer_xml(mods, game_id, game_ver, game_title,
                                    process, author)
    return _mc4_encrypt(xml_text.encode("utf-8"))


def mc4_xml_to_mods(xml_text: str) -> tuple:
    """Parse a decrypted .mc4 Trainer/Cheat/Cheatline XML document into the
    same {"name", "memory": [{"offset", "on", "off"}]} mod shape
    generate_etahen_json()/generate_mc4_bytes() consume, so import can share
    one converter with the etaHEN/GoldHEN JSON path. Returns
    (trainer_attrs, mods); a malformed document raises ET.ParseError, and a
    well-formed but empty/unexpected one simply yields an empty mods list.
    """
    root = ET.fromstring(xml_text)
    trainer_attrs = dict(root.attrib)
    mods = []
    for cheat_el in root.findall("Cheat"):
        name = (cheat_el.get("Text") or cheat_el.get("CheatName") or
                cheat_el.get("Name") or "Unnamed cheat")
        memory = []
        for line_el in cheat_el.findall("Cheatline"):
            offset = (line_el.findtext("Offset") or "").strip()
            on = (line_el.findtext("ValueOn") or "").strip().replace("-", "")
            off = (line_el.findtext("ValueOff") or "").strip().replace("-", "")
            # <Section> selects WHICH module the offset is relative to.
            # Dropping it silently made a section-1 library patch resolve
            # against the main module — the same address a section-0 patch
            # would get, so a community trainer wrote into the wrong image
            # with no warning. Carry it so the importer can refuse what it
            # cannot place. Absent/blank means section 0 (main image).
            section_text = (line_el.findtext("Section") or "").strip()
            try:
                section = int(section_text) if section_text else 0
            except ValueError:
                section = -1          # unparseable: treat as un-placeable
            if offset and on:
                memory.append({"offset": offset, "on": on, "off": off,
                               "section": section})
        if memory:
            mods.append({"name": name, "memory": memory})
    return trainer_attrs, mods


# ── logging ───────────────────────────────────────────────────────────────────
LOG_LIMIT = 500   # raised from 200 so older diagnostics are not lost so quickly

def add_log(msg: str, level: str = "info") -> None:
    with _log_lock:
        state["log"].append({"ts": time.strftime("%H:%M:%S"), "msg": msg, "level": level})
        if len(state["log"]) > LOG_LIMIT:
            state["log"] = state["log"][-LOG_LIMIT:]

# ── curses UI helpers ─────────────────────────────────────────────────────────

_COLORS_OK = False

# ── warning routing ───────────────────────────────────────────────────────────
# Python writes warnings to sys.stderr, and curses does not redirect stderr --
# it takes over the terminal display. So during a session a RuntimeWarning
# lands on the terminal *underneath* the TUI: it either corrupts the drawn
# screen until the next full refresh, or is lost entirely when the program
# restores the terminal on exit. Neither puts it in front of the user, and
# neither puts it in the log they can save and send.
#
# That matters for this program specifically. RDX does heavy NumPy arithmetic
# on scan data, and _wrapped_delta's docstring records what that class of
# failure looks like here: "the sum never wraps and a legitimate match is
# silently dropped". That case is handled and tested. The point is that if a
# future change reintroduces one, the warning announcing it is currently
# invisible.
#
# Routing warnings into add_log puts them where the user already looks, and
# into the saved log, without anyone having to reproduce the problem under a
# terminal they can read.
_warning_seen: set = set()
_warning_lock = threading.Lock()
_WARNING_DEDUP_MAX = 256
_original_showwarning = None


def _log_warning(message, category, filename, lineno, file=None, line=None):
    """Route a Python/NumPy warning into the in-app log.

    Deduplicated by origin: NumPy can raise the same warning once per element
    in a hot loop, and 500 identical lines would push every other diagnostic
    out of a log that holds 500 entries.
    """
    key = (getattr(category, "__name__", str(category)),
           str(filename), int(lineno))
    with _warning_lock:
        if key in _warning_seen:
            return
        if len(_warning_seen) >= _WARNING_DEDUP_MAX:
            _warning_seen.clear()
        _warning_seen.add(key)
    try:
        where = str(filename).rsplit("/", 1)[-1]
        add_log(f"{key[0]}: {message} ({where}:{lineno})", "warn")
    except Exception:
        # A logging failure must never turn into a second warning, which
        # would recurse straight back into here.
        pass


def install_warning_router() -> None:
    """Send warnings to the log instead of to a terminal curses is using.

    NumPy's error policy is set alongside it: without this, an overflow or an
    invalid float operation is silent by default, so there is nothing for the
    router to carry. Underflow stays ignored -- it is normal and harmless in
    float scanning, and warning about it would be noise.

    Sites that wrap on purpose (see _wrapped_delta) opt out locally with
    np.errstate, so deliberate behaviour does not report itself as a fault.
    """
    global _original_showwarning
    if _original_showwarning is None:
        _original_showwarning = warnings.showwarning
    warnings.showwarning = _log_warning
    np.seterr(over="warn", divide="warn", invalid="warn", under="ignore")


def init_colors() -> None:
    global _COLORS_OK
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN,    -1)   # C_TITLE
        curses.init_pair(2, curses.COLOR_GREEN,   -1)   # C_OK
        curses.init_pair(3, curses.COLOR_YELLOW,  -1)   # C_WARN
        curses.init_pair(4, curses.COLOR_RED,     -1)   # C_ERR
        curses.init_pair(5, curses.COLOR_WHITE,   -1)   # C_NORM
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)   # C_ACC
        curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_CYAN)  # C_SEL
        curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_RED)   # C_DSEL
        _COLORS_OK = True
    except curses.error:
        _COLORS_OK = False


def _safe_curs_set(visibility: int) -> None:
    """Best-effort cursor mode for terminals that reject curs_set()."""
    try:
        curses.curs_set(int(visibility))
    except curses.error:
        pass

C_TITLE = 1; C_OK = 2; C_WARN = 3; C_ERR = 4
C_NORM  = 5; C_ACC = 6; C_SEL  = 7; C_DSEL = 8

def color(pair: int) -> int:
    if not _COLORS_OK:
        return 0
    try:
        return curses.color_pair(pair)
    except curses.error:
        return 0

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
        # available columns.  Width comes from unicodedata rather than a
        # codepoint threshold: the old "> 0x1100 means wide" rule called
        # every decorative glyph this UI draws double-width -- the progress
        # bar's block characters, the check/warn/cross marks, the arrows,
        # the spinner -- so a 60-column bar rendered in 39 columns on an
        # 80-column terminal, and any status line containing them was
        # clipped early.  East Asian Width W/F is the real double-width set
        # (CJK); Ambiguous and Neutral render as one column in terminals.
        clipped, cols = [], 0
        for ch in text:
            if unicodedata.combining(ch):
                w_ch = 0          # accents attach to the previous cell
            else:
                w_ch = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if cols + w_ch > avail:
                break
            clipped.append(ch)
            cols += w_ch
        win.addstr(y, x, "".join(clipped), attr)
    except curses.error:
        pass


# The pointer verifier, freeze monitor, and guided scan forms need these bounds
# to keep their actions and cancel/status rows visible. Smaller terminals now
# get one clear resize prompt instead of silently accepting hidden defaults.
_MIN_ROWS, _MIN_COLS = 24, 72


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
              width: int = 30, default: str = "",
              allow_cancel: bool = False,
              empty_uses_default: bool = True,
              cancel_with_q: bool = True) -> Optional[str]:
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h - 1:
        return None if allow_cancel else default
    safe_addstr(stdscr, y, x, prompt, color(C_WARN) | curses.A_BOLD)
    px = x + len(prompt)
    if px >= w:
        return None if allow_cancel else default
    # Always switch to blocking + cbreak before getstr().  Any caller that
    # used nodelay(True) (progress loops, results screen) must not leave the
    # terminal in non-blocking mode when we hand off to text input — getstr()
    # in nodelay mode returns immediately with empty bytes.
    stdscr.nodelay(False)
    stdscr.timeout(-1)       # block indefinitely while user types
    curses.cbreak()
    # getstr() cannot implement an immediate cancel: it line-buffers Esc/Q
    # until Enter is pressed.  Cancellable numeric prompts therefore read one
    # key at a time below.  Regular text prompts retain curses' normal editor.
    curses.noecho() if allow_cancel else curses.echo()
    _safe_curs_set(1)
    safe_addstr(stdscr, y, px, " " * min(width, w - px))  # clear previous value
    safe_addstr(stdscr, y, px, default)
    stdscr.refresh()
    try:
        if allow_cancel:
            chars = []
            while True:
                key = stdscr.getch()
                if key == 27 or (cancel_with_q and key in (ord('q'), ord('Q'))):
                    return None
                if key in (curses.KEY_ENTER, 10, 13):
                    break
                if key in (curses.KEY_BACKSPACE, 8, 127):
                    if chars:
                        chars.pop()
                        safe_addstr(stdscr, y, px, " " * min(width, w - px))
                        safe_addstr(stdscr, y, px, "".join(chars))
                        stdscr.refresh()
                    continue
                if 32 <= key <= 126 and len(chars) < width:
                    if not chars:
                        safe_addstr(stdscr, y, px,
                                    " " * min(width, w - px))
                    chars.append(chr(key))
                    safe_addstr(stdscr, y, px, "".join(chars))
                    stdscr.refresh()
            raw = "".join(chars).encode("utf-8")
        else:
            raw = stdscr.getstr(y, px, width)
        val = raw.decode('utf-8').strip()
    except Exception:
        # A terminal/input failure must not silently submit displayed defaults
        # in a cancellable, state-changing form.
        val = None if allow_cancel else default
    finally:
        curses.noecho()
        _safe_curs_set(0)
        # Restore the 100 ms timeout set in main() so callers get expected
        # behaviour without having to remember to reset it themselves.
        stdscr.timeout(100)
    if val is None:
        return None
    return (val or default) if empty_uses_default else val

def cycle_input(stdscr, prompt: str, y: int, x: int,
                options: list, default=None, allow_cancel: bool = False):
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h - 1:
        return (None if allow_cancel else
                (default if default is not None else options[0]))
    idx = options.index(default) if default in options else 0
    _safe_curs_set(0)
    while True:
        safe_addstr(stdscr, y, x, prompt, color(C_WARN) | curses.A_BOLD)
        hint = f"< {options[idx]} >  (Tab/arrows to change, Enter to confirm)"
        safe_addstr(stdscr, y, x + len(prompt), hint, color(C_TITLE) | curses.A_BOLD)
        stdscr.refresh()
        try:
            k = stdscr.getch()
        except curses.error:
            return None if allow_cancel else (
                default if default is not None else options[0])
        if k == -1:
            time.sleep(0.02)
            continue
        if k == curses.KEY_RESIZE:          # Issue #1: absorb resize events
            curses.update_lines_cols()
            h, w = stdscr.getmaxyx()
            continue
        if allow_cancel and k in (27, ord('q'), ord('Q')):
            return None
        if k in (ord('\t'), curses.KEY_RIGHT):
            idx = (idx + 1) % len(options)
        elif k == curses.KEY_LEFT:
            idx = (idx - 1) % len(options)
        elif k in (curses.KEY_ENTER, 10, 13):
            return options[idx]

def confirm_box(stdscr, question: str, title: str = "Confirm") -> bool:
    # Issue #4: use _popup_dims so the box is never drawn off-screen.
    # Preserve deliberate newlines in safety/preflight questions.  Older
    # versions passed the whole string to one curses row and silently clipped
    # every line after the first.
    lines = str(question).splitlines() + ["", "  [Y] Yes      [N / Esc] No"]
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
    backend = "MemDBG EXP" if state.get("backend") == "memdbg-experimental" else "ps5debug"
    brand = f"◈  RDX CHEATMAKER {RDX_VERSION}  [{backend}]  ◈"
    safe_addstr(stdscr, 1, max(0, (w - len(brand)) // 2),
                brand, color(C_TITLE) | curses.A_BOLD)

def _console_preflight(ip: str, timeout: float = 3.0) -> bool:
    """Is anything listening on either payload port? Fails fast when not.

    A console on the LAN answers in milliseconds. When it is powered off,
    asleep, or the address is a typo, the normal path costs `memdbg_probe`
    (1.5 s) plus `ps5_connect`'s 15 s default -- about 16.5 s during which
    the UI shows only "Connecting..." and offers no way to cancel. Both
    ports are tried because a MemDBG-only console has 744 closed and a
    ps5debug-only console has 9020 closed. Uses ps5_connect/memdbg_probe so
    the same getaddrinfo handling applies as in the real connection.
    """
    try:
        probe = ps5_connect(ip, timeout=timeout)
        probe.close()
        return True
    except Exception:
        pass
    return memdbg_probe(ip, timeout=max(1.0, timeout / 2)) is not None


def screen_connect(stdscr) -> str:
    stdscr.clear()
    draw_border(stdscr, "CONNECT")
    draw_header_banner(stdscr)
    for i, hint in enumerate([
        "Load ps5debug-NG, or MemDBG (native 9020; legacy 744 is optional).",
        "Find PS5 IP:  Settings > Network > View Connection Status",
        "Press Esc to exit.",
    ]):
        safe_addstr(stdscr, 3 + i, 3, hint, color(C_NORM))
    stdscr.refresh()

    # Prefill from this user's own last console (loaded into state["ip"] from
    # the `last_ip` preference), and from nothing at all on a first run.  A
    # literal address here would be one particular development console, which
    # is not a sensible default for anyone else and reads as a real suggestion.
    ip = input_box(stdscr, "PS5 IP address : ", 6, 3, 40,
                   state["ip"] or "", allow_cancel=True,
                   cancel_with_q=False)
    if ip is None:
        return "quit"
    state["ip"] = ip
    # Issue #9: a new connection means a new session — stop any freeze that
    # was left running from a previous connection before we try to talk to
    # the new (or restarted) PS5.
    _stop_freeze_worker()
    safe_addstr(stdscr, 8, 3, "Connecting…", color(C_WARN))
    stdscr.refresh()
    if not _console_preflight(ip):
        safe_addstr(stdscr, 8, 3,
                    f"X Nothing is listening on {ip} (ports 744 / 9020)".ljust(60),
                    color(C_ERR))
        safe_addstr(stdscr, 10, 3,
                    "The console may be powered off, asleep, or on another",
                    color(C_WARN))
        safe_addstr(stdscr, 11, 3,
                    "address; the payload may also not be loaded yet.",
                    color(C_WARN))
        safe_addstr(stdscr, 13, 3, "Press any key to retry.", color(C_NORM))
        stdscr.refresh()
        stdscr.getch()
        return "connect"
    try:
        native = memdbg_probe(ip)
        if native is not None:
            state["backend"] = "memdbg-experimental"
            state["memdbg"] = native
            # Early MemDBG revisions did not advertise native memory reads.
            # Those builds still need the optional compatibility listener;
            # current payloads can run the entire RDX scan workflow on 9020.
            if not _memdbg_has(MEMDBG_CAP_MEMORY_READ):
                bridge = ps5_connect(ip, timeout=3.0)
                bridge.close()
        else:
            state["backend"] = "ps5debug"
            state["memdbg"] = None
        # One line naming the transport and anything it cannot do, so a
        # session that silently fell back to the slow host scan path says so
        # in the log rather than only in its timings.
        add_log(f"Transport — {current_target(ip).describe()}")
        procs = ps5_proc_list(ip)
        # A successful connection starts a new protocol session.  PIDs can be
        # reused after rest mode/restart and can coincide across consoles, so
        # no address or map state from the prior session is safe to retain.
        # Increment session BEFORE clearing state so cheats stamped during
        # the clear cannot pass the subsequent session-match check.
        state["session"] += 1
        _reset_learned_payload_support()   # may be a different payload build
        _clear_scan_state()
        state["pid"] = None
        state["proc_name"] = ""
        state["connected"] = True
        _save_preferences({"last_ip": ip})
        if native is not None:
            native_io = (_memdbg_has(MEMDBG_CAP_MEMORY_READ) and
                         _memdbg_has(MEMDBG_CAP_MEMORY_WRITE))
            add_log(f"Connected to {ip} via experimental MemDBG "
                    f"{native.get('version') or ''}; "
                    f"{'native memory I/O' if native_io else 'compatibility fallback'}")
            # TurboScan, the console-side scanner, and the region classifier
            # are all ps5debug-NG commands on port 744 and have no MemDBG
            # equivalent. With MemDBG alone every scan silently falls back to
            # transferring the whole region over the network -- on a 2 GiB
            # game that is ~140 s instead of ~1 s. Say so at connect time
            # rather than letting the user rediscover it per scan.
            # Diagnostic only: never let a failed probe fail the connect,
            # since native MemDBG is explicitly supposed to work with no
            # legacy listener at all.
            try:
                probe = ps5_connect(ip, timeout=2.0)
                probe.close()
            except Exception:
                add_log("ps5debug-NG (port 744) is not listening: TurboScan, "
                        "the console scanner and the region classifier are "
                        "unavailable. Scans will use the slower host path; "
                        "load ps5debug-NG alongside MemDBG to keep them.",
                        "warn")
        else:
            add_log(f"Connected to {ip} via ps5debug, {len(procs)} processes")
        return screen_proc_select(stdscr, procs)
    except Exception as e:
        safe_addstr(stdscr, 8, 3, f"X Failed: {e}".ljust(60), color(C_ERR))
        safe_addstr(stdscr, 10, 3,
                    "Start ps5debug-NG or MemDBG on your console,",
                    color(C_WARN) | curses.A_BOLD)
        safe_addstr(stdscr, 11, 3,
                    "then verify the PS5 IP address and try again.", color(C_WARN))
        safe_addstr(stdscr, 13, 3, "Press any key to retry.", color(C_NORM))
        stdscr.refresh()
        stdscr.getch()
        return "connect"

def _reset_learned_payload_support() -> None:
    """Forget which optional commands this console's payload supports.

    Only a reconnect can land on a different payload build, so this is
    deliberately separate from _clear_scan_state(): clearing results or
    switching process must not make RDX re-pay the discovery cost.
    """
    with _console_scan_lock:
        _console_scan_supported.clear()
    with _turbo_lock:
        _turbo_supported.clear()
    with _memdbg_maps_v2_lock:
        _memdbg_maps_v2_supported.clear()
    # The shared native connection points at the old payload instance, and a
    # tripped failure latch describes a console that no longer exists.
    memdbg_reset_session()


def _clear_scan_state(stop_freezes: bool = True) -> None:
    """Wipe scan state; process/session changes also stop active toggles."""
    if stop_freezes:
        _stop_freeze_worker()
    _close_turbo_session()
    # A bookmark with no chain is a raw address and means nothing once the
    # scan state it was taken alongside is gone. One that carries a verified
    # module-rooted chain rebases against the new maps, so it is kept.
    state["bookmarks"]      = [b for b in state.get("bookmarks", [])
                               if b.get("chain")]
    state["structures"]     = {}
    _invalidate_klass_names()
    scan.clear()
    state["scan_history"]   = deque(maxlen=5)
    with _map_cache_lock:
        _map_cache.clear()
    _invalidate_oversize_probes()
    # NOTE: learned command-support caches are deliberately NOT cleared here.
    # _clear_scan_state also runs on a plain "Clear Results" and on a process
    # change, and forgetting there means the next scan pays the 15 s stall to
    # rediscover a command this payload still does not implement. Only a
    # reconnect can reach a different payload build, so screen_connect owns
    # that reset (see _reset_learned_payload_support).
    _invalidate_pointer_index()
    _ScanSocket.clear_pool()
    gc.collect()


# ── game-process identification ───────────────────────────────────────────────
# The process list carries only pid and name, so on a first run every row looks
# alike and the user has to know that the game is the one called "eboot.bin".
# Two signals separate it, in increasing cost and confidence:
#
#   1. name — the game/app is "eboot.bin" on both consoles. Free, but system
#      applications use the same name, so it is a candidate test, not proof.
#   2. an /app0/ mapping — /app0 is the mounted title image, so a process
#      owning one *is* the running title. Costs a maps call per candidate,
#      which is why it runs only against processes that pass test 1.
#
# Test 2 runs on a daemon thread so the picker stays interactive on a slow
# link, and every failure degrades silently to test 1 rather than blocking
# attach behind an identification that is only ever a convenience.
_GAME_NAME_HINTS = ("eboot.bin", "eboot")
_GAME_IDENT_MAX_PROBES = 4


def _is_game_candidate(name) -> bool:
    """True for processes worth spending an /app0/ probe on."""
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base in _GAME_NAME_HINTS


def _process_owns_app0(ip: str, pid: int) -> bool:
    """True when the process has an /app0/ mapping, i.e. it is the title."""
    maps = _get_maps_cached(ip, pid)
    return any("/app0/" in str(r.get("name", "") or "").replace("\\", "/").lower()
               for r in maps)


def _identify_game_processes(ip: str, candidates: list, confirmed: dict,
                             lock: threading.Lock, cancel) -> None:
    """Fill `confirmed` with {pid: bool} for each candidate that answers.

    Never raises: identification is a convenience, and a console that will
    not serve maps for one process must not stop the user attaching to it.
    """
    for proc in candidates[:_GAME_IDENT_MAX_PROBES]:
        if cancel.is_set():
            return
        pid = int(proc.get("pid", 0))
        try:
            owns = _process_owns_app0(ip, pid)
        except Exception:
            continue
        with lock:
            confirmed[pid] = owns


def screen_proc_select(stdscr, procs: list) -> str:
    # Sort order: 'name' (default) or 'pid'.  Tab cycles between them.
    sort_by = "name"
    procs_orig = list(procs)

    def _sorted(lst):
        if sort_by == "pid":
            return sorted(lst, key=lambda p: p['pid'])
        return sorted(lst, key=lambda p: p['name'].lower())

    procs = _sorted(procs_orig)
    preferred = str(state.get("last_process", "eboot.bin") or "eboot.bin")
    game_candidates = [p for p in procs_orig if _is_game_candidate(p.get("name"))]
    confirmed_games: dict = {}
    ident_lock = threading.Lock()
    ident_cancel = threading.Event()
    ident_thread = None
    if game_candidates:
        ident_thread = threading.Thread(
            target=_identify_game_processes,
            args=(state["ip"], game_candidates, confirmed_games,
                  ident_lock, ident_cancel),
            daemon=True)
        ident_thread.start()

    def _game_rank(proc) -> int:
        """0 = confirmed title, 1 = name candidate, 2 = everything else."""
        with ident_lock:
            if confirmed_games.get(int(proc.get("pid", 0))) is True:
                return 0
        return 1 if _is_game_candidate(proc.get("name")) else 2

    # Preference order for the initial cursor: the process the user attached
    # to last, then the likeliest game, then the top of the list. Without the
    # middle term a first run always landed on row 0, which is a system
    # process on every console this was tested against.
    sel = next((i for i, p in enumerate(procs)
                if str(p.get("name", "")) == preferred), None)
    if sel is None:
        sel = next((i for i, p in enumerate(procs)
                    if _is_game_candidate(p.get("name"))), 0)
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
        # Stable sort by game-rank keeps the chosen sort order (name/pid)
        # inside each group while floating the title to the top.
        visible_procs.sort(key=_game_rank)
        # Clamp sel whenever the visible list changes size.
        sel = min(sel, max(0, len(visible_procs) - 1))

        filter_hint = filter_str if filter_str else "(none — type to filter)"
        safe_addstr(stdscr, 3, 3, f"Filter: {filter_hint}", color(C_WARN))
        safe_addstr(stdscr, 3, w - 22,
                    f"Sort: {sort_by} [Tab]  ", color(C_NORM))
        with ident_lock:
            n_confirmed = sum(1 for v in confirmed_games.values() if v)
            still_probing = (ident_thread is not None
                             and ident_thread.is_alive())
        if n_confirmed:
            safe_addstr(stdscr, 4, 3,
                        "▶ = running title (owns an /app0/ mapping)",
                        color(C_OK))
        elif still_probing:
            safe_addstr(stdscr, 4, 3, "· identifying the running title…",
                        color(C_ACC))

        visible = max(1, h - 9)
        start   = max(0, sel - visible // 2)
        for i, p in enumerate(visible_procs[start:start + visible]):
            idx  = start + i
            dim  = p['pid'] < 10
            rank = _game_rank(p)
            attr = (color(C_SEL)
                    if idx == sel
                    else (color(C_OK) | curses.A_BOLD if rank == 0
                          else color(C_NORM) | curses.A_DIM if dim
                          else color(C_NORM)))
            marker = "▶ " if rank == 0 else "· " if rank == 1 else "  "
            suffix = "   ← game" if rank == 0 else ""
            line = f"{marker}PID {p['pid']:6d}   {p['name']}{suffix}"
            safe_addstr(stdscr, 5 + i, 2, line[:w - 4].ljust(w - 4), attr)

        draw_statusbar(stdscr, [
            ("arrows navigate", C_NORM), ("Enter attach", C_OK),
            ("type to filter", C_WARN),  ("Tab sort", C_NORM),
            ("Bksp clear", C_NORM),      ("Esc back", C_NORM),
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
            ident_cancel.set()
            if p["pid"] != state["pid"]:   # process actually changed
                _clear_scan_state()
            state["pid"]       = p["pid"]
            state["proc_name"] = p["name"]
            state["last_process"] = p["name"]
            _save_preferences({"last_ip": state["ip"],
                               "last_process": p["name"]})
            add_log(f"Attached to PID {state['pid']} ({state['proc_name']})")
            return "main"
        elif key == 27:
            # 'q'/'Q' are deliberately NOT bound to quit here: this screen
            # has a live typeahead filter, and 'q' is a normal, common
            # character in process names — treating it as a quit shortcut
            # would make it impossible to ever type. Esc is unambiguous.
            ident_cancel.set()
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
    project = state.get("pointer_project_summary") or {
        "count": 0, "survivals": 0}
    project_note = (f"   Pointer project {project['survivals']}/2"
                    if project["count"] else "")
    scan_note = (f"Results {results:,}   Cheats {cheats}   "
                 f"{_current_scan_label()}{project_note}")
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


def _fuzzy_subsequence_rank(term: str, text: str):
    """None if `term`'s characters don't all appear in `text`, in order
    (case-insensitive callers pass already-lowered strings); otherwise a
    (span, start) rank where a tighter, earlier match sorts first."""
    ti = 0
    first = None
    last = -1
    for ci, ch in enumerate(text):
        if ti < len(term) and ch == term[ti]:
            if first is None:
                first = ci
            last = ci
            ti += 1
    if ti < len(term):
        return None
    return (last - first, first)


def _command_palette_rank(query: str, label: str):
    """None if `label` doesn't match every whitespace-separated term in
    `query` as a fuzzy subsequence; otherwise a sort key, best match first.
    Lets "exp trn" find "Export Trainers" without requiring a contiguous
    substring."""
    terms = query.lower().split()
    if not terms:
        return (0, 0)
    text = label.lower()
    total_span = total_start = 0
    for term in terms:
        ranked = _fuzzy_subsequence_rank(term, text)
        if ranked is None:
            return None
        span, start = ranked
        total_span += span
        total_start += start
    return (total_span, total_start)


def do_command_palette(stdscr) -> None:
    # One registry, so a command added there appears here automatically and
    # carries its own availability rule with it.
    commands = [(command.label, command.name)
                for command in _commands().values() if command.in_palette]
    query = ""
    sel = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "COMMAND PALETTE")
        safe_addstr(stdscr, 2, 3, f"> {query}_", color(C_ACC) | curses.A_BOLD)
        visible = max(1, min(12, h - 7))
        ranked = [(rank, c) for c in commands
                 if (rank := _command_palette_rank(query, c[0])) is not None]
        ranked.sort(key=lambda item: item[0])
        matches = [c for _, c in ranked]
        if matches:
            sel = min(sel, len(matches) - 1)
            for i, (label, name) in enumerate(matches[:visible]):
                reason = _commands()[name].unavailable_reason()
                if i == sel:
                    attr = color(C_SEL) | curses.A_BOLD
                elif reason:
                    attr = color(C_NORM) | curses.A_DIM
                else:
                    attr = color(C_NORM)
                # The palette used to offer every command unconditionally,
                # including ones the main menu already knew could not run.
                suffix = f"   — {reason}" if reason else ""
                safe_addstr(stdscr, 4 + i, 4,
                            ("▶ " if i == sel else "  ") + label + suffix, attr)
        else:
            safe_addstr(stdscr, 4, 4, "No matching commands.", color(C_WARN))
        draw_statusbar(stdscr, [("↑↓", C_NORM), ("Enter run", C_OK), ("Backspace", C_NORM), ("Esc", C_NORM)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols(); continue
        if key == 27:
            # 'q'/'Q' are deliberately NOT bound to quit here: this is a
            # live fuzzy-search query field, and 'q' is a normal character
            # in command names — treating it as a quit shortcut would make
            # it impossible to ever start a query with "q". Esc is
            # unambiguous. (Same class of bug as screen_proc_select's
            # process filter, fixed the same way.)
            return
        if key in (curses.KEY_UP,):
            sel = max(0, sel - 1)
        elif key in (curses.KEY_DOWN,):
            sel = min(max(len(matches) - 1, 0), sel + 1)
        elif key in (curses.KEY_ENTER, 10, 13) and matches:
            result = dispatch(stdscr, matches[sel][1])
            if result in {"proc", "connect"}:
                return result
            return
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            query = query[:-1]; sel = 0
        elif 32 <= key <= 126:
            query += chr(key); sel = 0


# ── command registry ──────────────────────────────────────────────────────────
# One table, three consumers. dispatch() held a 15-entry action dict,
# do_command_palette() held the same actions again as display labels, and
# _main_menu_entries() held a third overlapping copy -- so a new command had to
# be added in three places and an availability rule (like "Next Scan needs
# results") existed only in the main menu, which is why the palette happily
# offered commands that could not run.
#
# This is the informal command enum those three tables already were, made
# explicit. Squalr does the same thing deliberately -- "shared command models
# live in squalr-engine-api; command-line parsing is an adapter that lowers
# CLI/REPL-style input into those shared commands" -- which is what lets one
# engine serve a GUI, a CLI and a TUI. RDX has one front end today, but the
# duplication was already costing it correctness.
class Command:
    """One user-invokable action."""

    __slots__ = ("name", "label", "handler", "menu_key", "color",
                 "requires_results", "requires_process", "in_palette")

    def __init__(self, name, label, handler, *, menu_key=None, color=None,
                 requires_results=False, requires_process=False,
                 in_palette=True):
        self.name = name
        self.label = label
        self.handler = handler
        self.menu_key = menu_key
        self.color = color
        self.requires_results = requires_results
        self.requires_process = requires_process
        self.in_palette = in_palette

    def unavailable_reason(self) -> Optional[str]:
        """Why this cannot run right now, or None when it can."""
        if self.requires_process and state.get("pid") is None:
            return "attach to a process first"
        if self.requires_results and len(state.get("scan_results", ())) == 0:
            return "no scan results yet"
        return None

    def is_available(self) -> bool:
        return self.unavailable_reason() is None


def _build_command_registry() -> dict:
    """The single source of truth for dispatch, palette and main menu."""
    commands = [
        Command("scan_first", "First Scan", do_scan_first,
                menu_key="S", color=C_NORM, requires_process=True),
        Command("scan_next", "Next Scan", do_scan_next,
                menu_key="N", color=C_NORM, requires_results=True),
        Command("results", "Results", do_show_results,
                menu_key="R", color=C_ACC, requires_results=True),
        Command("pointer_project", "Pointer Project", do_pointer_project,
                menu_key="P", color=C_ACC, requires_process=True),
        Command("cheat_list", "Cheats", do_cheat_list,
                menu_key="C", color=C_NORM),
        Command("scan_settings", "Settings", do_scan_settings,
                menu_key="T", color=C_ACC),
        # The workflow menu is deliberately small, but "small" stopped
        # meaning "most of the program" some time ago: 16 of 22 commands
        # were reachable only by typing their name into the palette, and a
        # palette is a recall tool -- it cannot help someone who does not
        # know the thing exists. This entry is the doorway; the ALSO
        # AVAILABLE column beside it is the sign on the door.
        Command("more_tools", "More Tools", do_command_palette,
                menu_key="M", color=C_ACC, in_palette=False),
        Command("guide", "Getting Started", do_guide),
        Command("bookmarks", "Bookmarks", do_bookmarks),
        Command("hex_view", "Hex View", _dispatch_hex_view,
                requires_process=True),
        Command("structure_view", "Structure View", _dispatch_structure_view,
                requires_process=True),
        Command("type_scan", "Type Scan (find objects)", do_type_scan,
                requires_process=True),
        Command("load_symbols", "Load Symbols (Il2CppDumper)", do_load_symbols),
        Command("pointer_scan", "Find Permanent Pointer", do_pointer_scan,
                requires_process=True),
        Command("ptr_verify", "Verify Pointer", do_ptr_verify_manual,
                requires_process=True),
        Command("write", "Write Address", do_write, requires_process=True),
        Command("freeze", "Freeze Address", do_freeze,
                requires_process=True),
        Command("import", "Import Trainer", do_import),
        Command("export", "Export Trainers", do_export),
        Command("log", "Logs", do_log),
        Command("clear", "Clear Results", do_clear_results),
        Command("clear_history", "Clear Scan History", do_clear_history),
        Command("proc", "Change Process", None),
        Command("reconnect", "Reconnect Console", None),
    ]
    return {command.name: command for command in commands}


_COMMANDS: dict = {}


def _commands() -> dict:
    """Registry, built lazily so it can reference handlers defined later."""
    global _COMMANDS
    if not _COMMANDS:
        _COMMANDS = _build_command_registry()
    return _COMMANDS


def _command_unavailable_reason(name) -> Optional[str]:
    """Why a menu/palette entry cannot run, or None. Quit has no command."""
    command = _commands().get(name) if name else None
    return command.unavailable_reason() if command else None


def _command_available(name) -> bool:
    return _command_unavailable_reason(name) is None


# ── first-run guidance ────────────────────────────────────────────────────────
# There was no onboarding of any kind: zero occurrences of tutorial, first_run
# or walkthrough in the whole program, and the teaching material was a README
# that lives outside it.
#
# Cheat Engine ships a nine-step interactive tutorial under Help, and it is the
# single most-recommended starting point in every third-party guide to that
# tool. It exists because the obstacles are structural, and the same three
# apply here: too many results, refinement being non-obvious, and pointer
# scanning looking daunting.
#
# A full interactive tutorial is a large build and probably the wrong first
# step. This is the smaller thing that covers the loop every other feature in
# RDX sits on top of -- shown once, dismissable forever, and never in the way
# of somebody who already knows what they are doing.



def _guide_seen() -> bool:
    return bool(_preferences.get(_GUIDE_PREF_KEY))


def first_run_guide_lines() -> list:
    """The one screen a first-time user gets, in the order they need it."""
    return [
        "RDX finds a value in a running game, then lets you change it.",
        "",
        "THE LOOP — this is the whole tool:",
        "",
        "  1. Note a number you can see in the game (ammo, gold, health).",
        "  2. First Scan for it. You will get thousands of matches.",
        "  3. Change it in the game — fire a shot, spend a coin.",
        "  4. Next Scan for the new number. Most matches disappear.",
        "  5. Repeat 3 and 4 until a handful are left.",
        "",
        "That narrowing is the point. One scan can never find an address;",
        "two or three almost always can.",
        "",
        "IF THE FIRST SCAN FINDS NOTHING",
        "  The value type is usually wrong. Unity games keep health, ammo",
        "  and timers in floats — try f32 rather than the u32 default.",
        "",
        "WHEN YOU HAVE A FEW ADDRESSES",
        "  Enter inspects one. C makes it a cheat. R searches for a",
        "  permanent pointer so it survives a reload — that part needs two",
        "  real game reloads and RDX will not skip them.",
        "",
        "There are more tools than the menu shows: press M, or / to search",
        "them, or ? for every key.",
    ]


def maybe_show_first_run_guide(stdscr) -> None:
    """Show the guide once, then never again unless asked.

    Silent for anyone whose preferences file already records having seen it,
    so it cannot become something a returning user dismisses every session.
    Failing to persist that is not worth an error: the worst case is seeing
    a help screen twice.
    """
    if _guide_seen():
        return
    message_box(stdscr, first_run_guide_lines(), "Getting Started", C_ACC)
    try:
        _preferences[_GUIDE_PREF_KEY] = True
        _save_preferences({_GUIDE_PREF_KEY: True})
    except Exception:
        pass


def do_guide(stdscr) -> None:
    """The same screen, on demand, from the palette."""
    message_box(stdscr, first_run_guide_lines(), "Getting Started", C_ACC)


def _menu_only_labels() -> list:
    """Labels of commands that exist but are not on the main menu.

    Shown so the front door advertises the building. Ordered by how likely a
    user is to want them rather than by registry order.
    """
    preferred = ["Type Scan (find objects)", "Hex View", "Structure View",
                 "Bookmarks", "Find Permanent Pointer", "Export Trainers",
                 "Import Trainer", "Load Symbols (Il2CppDumper)",
                 "Freeze Address", "Write Address", "Logs"]
    available = {c.label for c in _commands().values()
                 if c.in_palette and not c.menu_key}
    ordered = [label for label in preferred if label in available]
    ordered += sorted(available - set(ordered))
    return ordered


def _main_menu_entries():
    """The deliberately small primary workflow menu, from the registry.

    Advanced/destructive utilities remain discoverable through the command
    palette instead of competing with the scan/results/cheat workflow.
    """
    entries = [(command.menu_key, command.label, command.name, command.color)
               for command in _commands().values()
               if command.menu_key]
    entries.append(("Q", "Quit", None, C_ERR))
    return entries


def _confirm_quit(stdscr) -> bool:
    """True if it's OK to quit — asks first when there are cheats that
    haven't survived an Export since they were last added/deleted."""
    if not (state["cheats"] and state.get("cheats_dirty")):
        return True
    n = len(state["cheats"])
    return confirm_box(
        stdscr,
        f"{n} cheat{'s' if n != 1 else ''} not exported since the last "
        "change.\nQuit anyway? This will discard them.",
        "Unsaved Cheats")


def screen_main(stdscr):
    """Compact primary navigation: workflow first, advanced tools elsewhere."""
    menu = _main_menu_entries()
    sel = 0
    # Once, on the first main menu a new install ever draws.
    maybe_show_first_run_guide(stdscr)
    state["pointer_project_summary"] = _pointer_project_summary(
        state.get("proc_name", ""))
    stdscr.timeout(100)
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "RDX CHEAT MAKER")
        _draw_main_header(stdscr)

        # The primary workflow is intentionally linear and small.  Do not
        # expose low-frequency destructive/debug utilities here.
        # Derived from the menu's own length rather than hardcoded counts.
        # Adding "More Tools" pushed Quit to index 7, past the end of a
        # hardcoded SETUP(5, 2) -- so it silently stopped being drawn in
        # wide mode. Letting the last section absorb whatever remains means
        # the next menu addition cannot repeat that.
        sections = [
            ("SCAN", 0, 4),
            ("CHEATS", 4, 1),
            ("SETUP", 5, max(1, len(menu) - 5)),
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
                    key, label, action, cp = menu[i]
                    # Availability travels with the command now, so the menu
                    # and the palette cannot drift apart.
                    unavailable = not _command_available(action)
                    attr = (color(C_SEL) | curses.A_BOLD if i == sel else
                            color(C_NORM) | curses.A_DIM if unavailable else color(cp))
                    safe_addstr(stdscr, 7 + j, x,
                                f"[{key}] {label}"[:max(w - x - 2, 0)], attr)
        else:
            safe_addstr(stdscr, 5, 3, "WORKFLOW", color(C_TITLE) | curses.A_BOLD)
            for i, (key, label, action, cp) in enumerate(menu):
                unavailable = not _command_available(action)
                attr = (color(C_SEL) | curses.A_BOLD if i == sel else
                        color(C_NORM) | curses.A_DIM if unavailable else color(cp))
                safe_addstr(stdscr, 7 + i, 3,
                            f"[{key}] {label}"[:max(w - 6, 0)], attr)

        # Named below the menu rather than in a fourth column. As a column it
        # needed a terminal 160 columns wide before it drew at all -- the
        # three existing columns already reach 2/3 of the width -- so on the
        # 80-to-120-column terminals people actually use, the thing built to
        # fix discoverability was itself invisible. Under the menu it fits at
        # every width, wrapping to however many rows are free.
        extras = _menu_only_labels()
        row = 7 + max(4, len(menu) - 4) + 1
        if extras and row + 2 < h - 3:
            safe_addstr(stdscr, row, 3, "ALSO AVAILABLE",
                        color(C_TITLE) | curses.A_BOLD)
            for j, line in enumerate(_wrap_help(" · ".join(extras),
                                                max(w - 8, 20))):
                if row + 1 + j >= h - 3:
                    break
                safe_addstr(stdscr, row + 1 + j, 3, line,
                            color(C_NORM) | curses.A_DIM)
        hidden = len(extras)
        if hidden:
            safe_addstr(stdscr, h - 4, 3,
                        f"{hidden} more tools — press M, or / to search them"
                        f"   ? for the full key list",
                        color(C_ACC))
        _draw_toast(stdscr)
        draw_statusbar(stdscr, [
            ("↑↓", C_NORM), ("Enter", C_OK), ("M more", C_ACC),
            ("/ Commands", C_ACC), ("? Help", C_ACC), ("Q Quit", C_ERR)
        ])
        stdscr.refresh()
        key = stdscr.getch()
        if key == -1:
            continue
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols()
            continue
        # j/k/g/G are aliases only on screens with no live typeahead filter.
        # screen_proc_select and do_command_palette deliberately have none:
        # there every printable character is query text, which is the same
        # reason 'q' is not bound to quit on those two screens.
        if key in (curses.KEY_UP, ord('k')):
            sel = max(0, sel - 1)
        elif key in (curses.KEY_DOWN, ord('j')):
            sel = min(len(menu) - 1, sel + 1)
        elif key == ord('g'):
            sel = 0
        elif key == ord('G'):
            sel = len(menu) - 1
        elif key in (curses.KEY_ENTER, 10, 13):
            action = menu[sel][2]
            if action is None:
                if _confirm_quit(stdscr):
                    return None
                continue
            result = dispatch(stdscr, action)
            if result in {"proc", "connect"}:
                return result
        elif key == ord('/'):
            result = do_command_palette(stdscr)
            if result in {"proc", "connect"}:
                return result
        elif key == ord('?'):
            do_help(stdscr)
        else:
            for k, label, action, _ in menu:
                if key in (ord(k.lower()), ord(k.upper())):
                    reason = _command_unavailable_reason(action)
                    if reason:
                        add_log(f"{label} unavailable — {reason}", "warn")
                        break
                    if action is None:
                        if _confirm_quit(stdscr):
                            return None
                        break
                    result = dispatch(stdscr, action)
                    if result in {"proc", "connect"}:
                        return result
                    break

def do_help(stdscr) -> None:
    lines = [
        "Navigation   ↑↓ or j/k   g/G top/bottom   Enter Run   Esc Back",
        "Global       / Command Palette   ? Help   M More Tools",
        "New here     / then \"Getting Started\" for the scan loop",
        "Scanning     S First Scan   N Next Scan   R Results",
        "Pointers     P Pointer Project (persisted 2-reload workflow)",
        "Results      A Apply   C Cheat   R Find permanent   N Refine",
        "Results      B Bookmark   Enter inspect → H for a hex view",
        "Type Scan    / then \"Type Scan\" — groups heap objects by the",
        "             type pointer at their base. Read-only.",
        "Hex view     Read-only. ↑↓/jk row, PgUp/PgDn page, a address,",
        "             n back to anchor, s structure. It never writes.",
        "Structure    Named typed fields over an address. Enter renames,",
        "             T retypes, R re-dissects, +/- resize, C changes,",
        "             Y overlays a class from a loaded dump.cs. Read-only.",
        "Symbols      / then \"Load Symbols\" — reads an Il2CppDumper",
        "             dump.cs so fields get real names, not field_0014.",
        "Cheats       F/Space Toggle   A Apply   E Edit   D Delete",
        "Bookmarks    / then \"Bookmarks\"   Enter inspect   C promote",
        "             (j/k work everywhere except the process picker and",
        "             the palette, where letters are filter text.)",
        "Advanced     Export/Import/Freeze/Logs have no direct key —",
        "             press / then type the command name to run them.",
        "Setup        T Settings",
        "",
        "Routine success messages stay in the status line;",
        "errors and destructive operations remain modal.",
    ]
    message_box(stdscr, lines, "Keyboard Help", C_ACC)

def dispatch(stdscr, action: str):
    """Run one registered command by name."""
    if action == "proc":
        return "proc"
    if action == "reconnect":
        _stop_freeze_worker()
        state["connected"] = False
        return "connect"
    command = _commands().get(action)
    if command is None:
        add_log(f"Unknown command: {action}", "warn")
        return None
    reason = command.unavailable_reason()
    if reason is not None:
        add_log(f"{command.label} unavailable — {reason}", "warn")
        return None
    if command.handler is None:
        return None
    return command.handler(stdscr)


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


def _format_setting(key: str) -> str:
    """Render a tunable the way it is entered."""
    spec, value = _SETTING_SPECS[key], setting(key)
    if spec["kind"] == "bool":
        return "on" if value else "off"
    if spec["kind"] == "hex":
        return hex(int(value))
    if spec["kind"] == "csv":
        text = str(value)
        return text if len(text) <= 46 else text[:43] + "..."
    return str(value)


def _edit_setting(stdscr, key: str, y: int) -> bool:
    """Prompt for one tunable. True when the stored value changed."""
    spec = _SETTING_SPECS[key]
    before = setting(key)
    if spec["kind"] == "bool":
        chosen = cycle_input(stdscr, f"{spec['label']}: ", y, 3,
                             ["off", "on"], "on" if before else "off",
                             allow_cancel=True)
        if chosen is None:
            return False
        after = _coerce_setting(key, chosen)
    else:
        raw = input_box(stdscr, f"{spec['label']}: ", y, 3, 52,
                        _format_setting(key), allow_cancel=True,
                        cancel_with_q=False)
        if raw is None:
            return False
        after = _coerce_setting(key, raw)
        # _coerce_setting clamps rather than rejects, so tell the user when
        # what they typed is not what got stored.
        if spec["kind"] in ("int", "hex"):
            try:
                asked = int(str(raw), 0)
            except (TypeError, ValueError):
                message_box(stdscr,
                            [f"{raw!r} is not a number — {spec['label']} "
                             f"left at {_format_setting(key)}."],
                            "Unchanged", C_WARN)
                return False
            if asked != after:
                message_box(stdscr,
                            [f"Clamped to the supported range "
                             f"{hex(spec['min'])}..{hex(spec['max'])}.",
                             f"{spec['label']} is now {hex(after)}."],
                            "Clamped", C_WARN)
    if after == before:
        return False
    _settings[key] = after
    _save_preferences({key: after})
    add_log(f"{spec['label']} = {_format_setting(key)}")
    return True


def do_scan_settings(stdscr) -> None:
    """Scan, region and pointer tunables.

    Previously this screen held one dropdown. The pointer bounds and the
    region filter rules lived as literals in the source, which is where PINCE
    and PS4CheaterNeo both put them on screen instead. The defaults are
    unchanged — RDX's depth of 5 and 0x800 window match PINCE's own defaults
    — so this exposes them rather than retuning anything.
    """
    engine_options = ["Auto (Turbo → Console → Host)", "Turbo only",
                      "Console only", "Host only"]
    engine_keys = ["auto", "turbo", "console", "host"]
    rows = [
        ("section", "SCAN"),
        ("engine", None),
        ("section", "REGIONS"),
        ("setting", "region_exclude"),
        ("setting", "region_min_size"),
        ("section", "POINTERS"),
        ("setting", "ptr_max_depth"),
        ("setting", "ptr_direct_range"),
        ("setting", "ptr_offset_max"),
        ("setting", "ptr_module_bases_only"),
        ("action", "reset"),
    ]
    selectable = [i for i, (kind, _) in enumerate(rows) if kind != "section"]
    sel = selectable[0]

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "SETTINGS")
        engine = state.get("scan_engine", "auto")
        if engine not in engine_keys:
            engine = "auto"
        for i, (kind, key) in enumerate(rows):
            y = 3 + i
            if y >= h - 3:
                break
            if kind == "section":
                safe_addstr(stdscr, y, 3, key,
                            color(C_TITLE) | curses.A_BOLD)
                continue
            chosen = i == sel
            attr = color(C_SEL) | curses.A_BOLD if chosen else color(C_NORM)
            if kind == "engine":
                label, value = "Scan engine", engine_options[engine_keys.index(engine)]
            elif kind == "action":
                label, value = "Restore defaults", ""
            else:
                label, value = _SETTING_SPECS[key]["label"], _format_setting(key)
            marker = "▸ " if chosen else "  "
            safe_addstr(stdscr, y, 3,
                        f"{marker}{label:<24} {value}"[:max(w - 6, 0)], attr)

        kind, key = rows[sel]
        helptext = ("Reset every setting on this screen to its built-in default."
                    if kind == "action" else
                    "Auto tries the fastest available engine and falls back safely."
                    if kind == "engine" else _SETTING_SPECS[key]["help"])
        for n, line in enumerate(_wrap_help(helptext, max(w - 8, 20))[:2]):
            safe_addstr(stdscr, h - 4 + n, 3, line, color(C_WARN))
        draw_statusbar(stdscr, [("↑↓ / jk", C_NORM), ("Enter edit", C_OK),
                                ("Esc/Q back", C_NORM)])
        stdscr.refresh()

        ch = stdscr.getch()
        if ch == curses.KEY_RESIZE:
            curses.update_lines_cols(); continue
        if ch in (curses.KEY_UP, ord('k')):
            prior = [i for i in selectable if i < sel]
            sel = prior[-1] if prior else selectable[-1]
        elif ch in (curses.KEY_DOWN, ord('j')):
            later = [i for i in selectable if i > sel]
            sel = later[0] if later else selectable[0]
        elif ch in (curses.KEY_ENTER, 10, 13):
            kind, key = rows[sel]
            if kind == "engine":
                chosen = cycle_input(stdscr, "Scan engine: ", h - 6, 3,
                                     engine_options,
                                     engine_options[engine_keys.index(engine)],
                                     allow_cancel=True)
                if chosen is not None:
                    picked = engine_keys[engine_options.index(chosen)]
                    if picked != engine:
                        # Switching away and back would otherwise re-adopt the
                        # session the last turbo scan left resident, silently
                        # discarding every narrowing the host path did between.
                        _close_turbo_session()
                        state["scan_engine"] = picked
                        add_log(f"Scan engine set to {picked}")
            elif kind == "action":
                if confirm_box(stdscr,
                               "Restore every setting on this screen to its "
                               "built-in default?", "Restore Defaults"):
                    for skey, spec in _SETTING_SPECS.items():
                        _settings[skey] = spec["default"]
                        _save_preferences({skey: spec["default"]})
                    add_log("Settings restored to defaults", "warn")
            else:
                changed = _edit_setting(stdscr, key, h - 6)
                # A changed pointer/region bound invalidates work computed
                # under the old one; a stale reverse index would otherwise be
                # reused and silently ignore the new setting.
                if changed and key in ("ptr_direct_range", "ptr_offset_max",
                                       "region_exclude", "region_min_size"):
                    _invalidate_pointer_index()
                    with _map_cache_lock:
                        _map_cache.clear()
        elif ch in (ord('q'), ord('Q'), 27):
            return


def _wrap_help(text: str, width: int) -> list:
    """Wrap one help string to the screen without importing textwrap."""
    words, lines, current = str(text).split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current); current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


# ── automatic value type ──────────────────────────────────────────────────────
# GameGuardian offers a data type called Auto alongside its explicit ones,
# and it exists because value-type confusion is the documented first obstacle
# for newcomers -- Cheat Engine's own material names it. RDX has more
# configuration than Cheat Engine, not less, and defaults to u32 while its one
# hardware-validated title is Unity/IL2CPP, where health and ammo are floats.
#
# patch110 made a zero-result scan explain that. This removes the decision
# instead of explaining it, for the user who does not yet know enough to make
# it. The cost is real -- one pass per candidate type -- so it is an explicit
# choice rather than the default, and it stops at the first type that finds
# anything rather than scanning all of them.
_AUTO_TYPE_ORDER = ("u32", "f32", "i32", "u64", "f64", "u16", "u8")
# Each attempt is a *full* scan. Seven of them on a 2 GiB title over the host
# path is roughly sixteen minutes for a value that was never there, which is
# not an automatic mode, it is a hang. Three covers the overwhelming majority
# -- an unsigned integer, a float, and a signed integer -- and the types that
# get dropped are named in the log so the choice is visible rather than
# silent.
_AUTO_MAX_ATTEMPTS = 3


def _auto_candidate_types(value_text: str) -> list:
    """Types worth trying for this text, most likely first.

    Only types the value actually parses as: scanning u8 for 70000 would be
    a guaranteed-empty pass, and three of those in a row is how an automatic
    mode earns a reputation for being slow and useless.
    """
    text = str(value_text or "").strip()
    if not text:
        return []
    candidates = []
    for key in _AUTO_TYPE_ORDER:
        spec = VALUE_TYPES.get(key)
        if not spec:
            continue
        try:
            _parse_value_text(text, key, spec["width"])
        except Exception:
            continue
        candidates.append(key)
    return candidates


def _zero_result_advice(value_text: str, type_key: str, scope: str,
                        aligned: bool) -> list:
    """Why a scan probably found nothing, likeliest cause first.

    A scan that matches nothing used to log one line and return to the main
    menu, which is what a beginner's first attempt most often produces -- and
    a UI that does nothing at all reads as a bug rather than a result.

    Every fact below is already known at the call site; none of it was being
    said. The float case leads because it is both the most common mistake and
    the one most specific to what RDX is used on: the default type is u32,
    and Unity/IL2CPP titles -- the only kind this project has validated
    against hardware -- keep health, ammo, position, speed and timers in
    floats. Cheat Engine's own documentation names value-type confusion as
    beginner obstacle number one.
    """
    lines = ["The scan completed and matched nothing.", ""]
    spec = VALUE_TYPES.get(type_key, {})
    kind = spec.get("kind", "")
    text = str(value_text or "").strip()

    # float() alone is too generous -- it accepts "1_000", "nan" and "inf",
    # none of which mean "the user typed a decimal". Require a decimal point
    # or an exponent before claiming the value looks like a float, so the
    # headline is only shown when it is actually true.
    float_parseable = False
    if text and kind in ("uint", "sint"):
        try:
            parsed = float(text)
            float_parseable = (parsed == parsed              # not NaN
                               and parsed not in (float("inf"), float("-inf")))
        except ValueError:
            float_parseable = False

    if float_parseable:
        lines += [
            f"Most likely: the value is a float, not {type_key}.",
            "",
            "Unity games keep health, ammo, position, speed and timers",
            "in floats. Run First Scan again and pick f32.",
            "",
        ]
    lines.append("Other things to check:")
    # Only worth raising when a float actually IS the alternative. Telling
    # someone already scanning f32 that the value is "usually f32" is noise,
    # and advice that is obviously wrong once teaches the reader to skip the
    # rest of it.
    if not float_parseable and kind in ("uint", "sint"):
        lines.append(f"  • Wrong value type — you scanned {type_key}. Health and")
        lines.append("    similar values in Unity titles are usually f32.")
    elif kind == "float":
        lines.append(f"  • Float precision — an exact {type_key} match is strict.")
        lines.append("    Set a tolerance on the First Scan screen, or scan")
        lines.append("    for an unknown value and narrow by 'changed'.")
    lines.append("  • The value on screen is not the value in memory")
    lines.append("    (a bar may be a percentage; ammo may be capped).")
    if str(scope or "") == "recommended":
        lines.append("  • Scope is Recommended, which skips libraries and")
        lines.append("    payloads. Try Writable in Settings if the value")
        lines.append("    might live outside the game's own mappings.")
    if not _region_settings_are_default():
        lines.append("  • Your region settings are not at their defaults and")
        lines.append("    may be excluding the mapping that holds it.")
    if aligned:
        lines.append("  • Aligned scanning is on. A packed structure can hold")
        lines.append("    a value at an unaligned offset.")
    return lines


def _current_scan_type() -> str:
    """Return the active scan type, including legacy width-only sessions."""
    return _normalise_value_type(state.get("scan_type"),
                                 state.get("scan_width", 4))


def _current_scan_label() -> str:
    key = _current_scan_type()
    if key == "bytes" and state.get("scan_pattern"):
        return f"AOB ({state['scan_pattern']})"
    return VALUE_TYPES[key]["label"]

def do_scan_first(stdscr) -> None:
    stdscr.clear()
    draw_border(stdscr, "FIRST SCAN")
    safe_addstr(stdscr, 2, 3,
        "Choose the value's real in-memory type before entering it.",
        color(C_WARN))
    stdscr.refresh()

    auto_label = "Auto (try likely types)"
    labels = [auto_label] + [VALUE_TYPES[key]["label"] for key in VALUE_TYPE_ORDER]
    current_type = _current_scan_type()
    type_label = cycle_input(
        stdscr, "Value type      : ", 4, 3, labels,
        VALUE_TYPES[current_type]["label"], allow_cancel=True)
    if type_label is None:
        add_log("First scan setup cancelled")
        return
    auto_mode = type_label == auto_label
    if auto_mode:
        # Resolved once the value is known; u32 is only a placeholder so the
        # prompts below behave normally.
        type_key = "u32"
    else:
        type_key = (VALUE_TYPE_ORDER[labels.index(type_label) - 1]
                    if type_label in labels else
                    _normalise_value_type(type_label))

    if type_key == "bytes":
        value_prompt = "Pattern (AA BB ?? CC): "
        value_default = state.get("scan_pattern", "")
    else:
        value_prompt = "Value (blank = unknown): "
        value_default = ""
    val_s = input_box(stdscr, value_prompt, 6, 3, 38, value_default,
                      allow_cancel=True)
    if val_s is None:
        add_log("First scan setup cancelled")
        return
    unknown_mode = type_key != "bytes" and not val_s.strip()

    auto_candidates = []
    if auto_mode:
        if unknown_mode:
            # An unknown-value scan snapshots memory; there is no value to
            # infer a type from, so Auto has nothing to decide.
            message_box(stdscr,
                        ["Auto needs a value to work from.",
                         "",
                         "For an unknown-value scan, pick the type you expect",
                         "— u32 for counters, f32 for health and timers."],
                        "Auto Needs a Value", C_WARN)
            return
        all_candidates = _auto_candidate_types(val_s)
        if not all_candidates:
            message_box(stdscr, [f"{val_s!r} does not parse as any scannable type."],
                        "Invalid Value", C_ERR)
            return
        auto_candidates = all_candidates[:_AUTO_MAX_ATTEMPTS]
        dropped = all_candidates[_AUTO_MAX_ATTEMPTS:]
        type_key = auto_candidates[0]
        add_log(f"Auto: trying {', '.join(auto_candidates)} in order until "
                f"one finds matches"
                + (f" (not trying {', '.join(dropped)} — pick one explicitly "
                   f"if you need it)" if dropped else ""))

    if type_key == "bytes":
        try:
            pattern, pattern_mask, canonical = _parse_byte_pattern(val_s, True)
        except ValueError as exc:
            message_box(stdscr, [str(exc)], "Invalid Pattern", C_ERR)
            return
        width = len(pattern)
        align_options = ["every byte (recommended)", "pattern-width aligned"]
        align_default = align_options[0]
    else:
        width = _value_width(type_key)
        pattern = pattern_mask = None
        canonical = ""
        align_options = ["aligned (faster)", "unaligned (thorough)"]
        align_default = (align_options[0] if state["scan_aligned"]
                         else align_options[1])

    align_lbl = cycle_input(stdscr, "Scan alignment  : ", 8, 3,
                            align_options, align_default, allow_cancel=True)
    if align_lbl is None:
        add_log("First scan setup cancelled")
        return
    aligned = (align_lbl.startswith("aligned") or
               align_lbl.startswith("pattern-width"))
    scope_options = [
        "recommended game regions",
        "all writable regions",
        "all readable regions (thorough)",
    ]
    scope_keys = ["recommended", "writable", "readable"]
    current_scope = state.get("scan_scope", "recommended")
    if current_scope not in scope_keys:
        current_scope = "writable" if state["scan_writable_only"] else "readable"
    scope_lbl = cycle_input(
        stdscr, "Scan scope      : ", 10, 3, scope_options,
        scope_options[scope_keys.index(current_scope)], allow_cancel=True)
    if scope_lbl is None:
        add_log("First scan setup cancelled")
        return
    region_scope = scope_keys[scope_options.index(scope_lbl)]
    writable_only = region_scope != "readable"

    state["scan_width"]        = width
    state["scan_type"]         = type_key
    state["scan_pattern"]      = canonical
    state["scan_aligned"]      = aligned
    state["scan_writable_only"] = writable_only
    state["scan_scope"]        = region_scope

    val = None
    if not unknown_mode:
        try:
            val = (canonical if type_key == "bytes" else
                   _parse_value_text(val_s, type_key, width))
        except ValueError as exc:
            message_box(stdscr, [str(exc)], "Invalid Value", C_ERR)
            return

    if VALUE_TYPES[type_key]["kind"] == "float":
        tol_s = input_box(stdscr, "Float tolerance : ", 12, 3, 20,
                          str(state.get("scan_tolerance", 0.0001)),
                          allow_cancel=True)
        if tol_s is None:
            add_log("First scan setup cancelled")
            return
        try:
            tolerance = float(tol_s)
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError
        except ValueError:
            message_box(stdscr, ["Tolerance must be a finite number >= 0."],
                        "Invalid Tolerance", C_ERR)
            return
        state["scan_tolerance"] = tolerance

    # A new first scan supersedes any retained resident refinement session,
    # including when this scan is unknown-value or uses a fallback engine.
    _close_turbo_session()
    cancel_event = threading.Event()
    cancel_event.truncated = False   # searcher sets this when result cap is hit
    progress     = {"done": 0, "total": 1, "results": None, "values": None,
                    "error": None, "truncated": False}

    if type_key == "bytes":
        def run():
            try:
                progress["results"] = scan_first_pattern(
                    state["ip"], state["pid"], pattern, pattern_mask,
                    width if aligned else 1,
                    lambda d, t: progress.update(done=d, total=max(t, 1)),
                    cancel_event, writable_only=writable_only,
                    region_scope=region_scope)
                progress["truncated"] = getattr(cancel_event, "truncated", False)
            except Exception as exc:
                progress["error"] = str(exc)
        scan_label = "Scanning byte pattern…"
    elif unknown_mode:
        def run():
            try:
                addrs, vals = scan_first_unknown(
                    state["ip"], state["pid"], width, aligned,
                    lambda d, t: progress.update(done=d, total=max(t, 1)),
                    cancel_event,
                    writable_only=writable_only, value_type=type_key,
                    region_scope=region_scope)
                progress["results"]   = addrs
                progress["values"]    = vals
                progress["truncated"] = getattr(cancel_event, "truncated", False)
            except Exception as exc:
                progress["error"] = str(exc)
        scan_label = "Snapshotting memory…"
    else:
        def run():
            try:
                # In Auto mode this walks the candidate types in likelihood
                # order and stops at the first that finds anything, so the
                # user makes no type decision at all. A single pass for an
                # explicit type is the same loop with one entry.
                attempts = auto_candidates or [type_key]
                res = _make_addr_array()
                chosen = type_key
                for candidate in attempts:
                    if cancel_event.is_set():
                        break
                    spec = VALUE_TYPES[candidate]
                    try:
                        candidate_val = _parse_value_text(
                            val_s, candidate, spec["width"])
                    except Exception:
                        continue
                    if len(attempts) > 1:
                        # _run_scan_with_progress takes its caption as an
                        # argument, not from this dict, so writing a label
                        # here did nothing. The log is the channel that
                        # actually reaches the user.
                        add_log(f"Auto: scanning as {candidate}…")
                    res = scan_first(
                        state["ip"], state["pid"], candidate_val,
                        spec["width"], aligned,
                        lambda d, t: progress.update(done=d, total=max(t, 1)),
                        cancel_event,
                        writable_only=writable_only, value_type=candidate,
                        region_scope=region_scope)
                    chosen = candidate
                    if len(res):
                        break
                progress["results"]   = res
                progress["chosen_type"] = chosen
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
    scan.clear()
    state["scan_history"]  = deque(maxlen=5)
    gc.collect()
    # Auto may have settled on a different type than the one it started
    # with; everything downstream (Results, Next Scan, cheats) must agree.
    chosen_type = progress.get("chosen_type") or type_key
    if chosen_type != type_key:
        add_log(f"Auto: {chosen_type} is the type that matched", "info")
    type_key = chosen_type
    width = _value_width(type_key) or width
    state["scan_type"] = type_key
    state["scan_width"] = width
    scan.replace(results, progress.get("values"),
                 unknown=unknown_mode,
                 truncated=progress.get("truncated", False),
                 close_turbo=False)
    add_log(f"{'Unknown' if unknown_mode else 'First'} scan "
            f"type={type_key} w={width} aligned={aligned}: "
            f"{len(results):,} candidates, "
            f"RSS {_rss_mb():.0f} MB")

    if unknown_mode:
        add_log(f"Snapshot complete — {len(results):,} candidates", "warn" if progress["truncated"] else "info")
        do_show_results(stdscr)
    elif len(results) == 0:
        # do_show_results returns immediately on an empty set, so without
        # this the user is dropped back on the main menu with one status
        # line and no idea what happened.
        add_log("First scan complete — 0 candidates", "warn")
        advice = _zero_result_advice(val_s, type_key,
                                     state.get("scan_scope", ""), aligned)
        if auto_candidates:
            # Auto already ruled these out, so repeating "try f32" would be
            # advice the user has provably already followed.
            advice = ([f"Auto tried {', '.join(auto_candidates)} and none",
                       "matched.", ""]
                      + [l for l in advice[1:] if "Most likely" not in l
                         and "f32" not in l and "Unity games keep" not in l
                         and "in floats. Run First Scan" not in l])
        message_box(stdscr, advice, "No Matches", C_WARN)
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
    type_key = _current_scan_type()
    is_unkn = state.get("scan_unknown", False)
    safe_addstr(stdscr, 2, 3,
        f"Candidates: {len(state['scan_results']):,}  "
        f"{_current_scan_label()}  "
        f"({'unknown-value' if is_unkn else 'exact-value'} session)",
        color(C_WARN))
    stdscr.refresh()

    cancel_event = threading.Event()
    prev_addrs   = state["scan_results"]

    if type_key == "bytes":
        pattern_s = input_box(
            stdscr, "Pattern          : ", 5, 3, 38,
            state.get("scan_pattern", ""), allow_cancel=True)
        if pattern_s is None:
            add_log("Next scan setup cancelled")
            return
        try:
            pattern, pattern_mask, canonical = _parse_byte_pattern(
                pattern_s, True)
        except ValueError as exc:
            message_box(stdscr, [str(exc)], "Invalid Pattern", C_ERR)
            return
        if len(pattern) != width:
            message_box(
                stdscr,
                [f"This scan expects {width} pattern bytes.",
                 "Start a new First Scan to change pattern length."],
                "Pattern Length", C_ERR)
            return

        progress = {"done": 0, "total": max(len(prev_addrs), 1),
                    "results": None, "error": None}

        def run_pattern():
            try:
                progress["results"] = scan_next_pattern(
                    state["ip"], state["pid"], pattern, pattern_mask,
                    prev_addrs, cancel_event,
                    lambda d, t: progress.update(done=d, total=max(t, 1)))
            except Exception as exc:
                progress["error"] = str(exc)

        ok = _run_scan_with_progress(
            stdscr, run_pattern, "Revalidating byte pattern…",
            cancel_event, progress)
        if not ok:
            add_log("Next pattern scan cancelled", "warn")
            return
        if progress["error"]:
            message_box(stdscr, [f"Error: {progress['error']}"],
                        "Scan Error", C_ERR)
            return
        results = (progress["results"] if progress["results"] is not None
                   else _make_addr_array())
        removed_a = np.setdiff1d(prev_addrs, results, assume_unique=True)
        _push_undo(removed_a, None, set(state["scan_dropped"]),
                   state.get("scan_truncated", False))
        scan.narrow(results, None)
        state["scan_pattern"] = canonical
        # A dropped address is removed from scan_results at drop time, so it
        # can never reappear in a later Next Scan's (necessarily narrower)
        # output -- there is nothing left worth carrying forward.
        state["scan_dropped"] = set()
        add_log(f"AOB next scan: {len(results):,} candidates remain")
        do_show_results(stdscr)

    elif is_unkn:
        # ── relational (unknown-value) path ───────────────────────────────────
        prev_values = state.get("scan_values")
        if prev_values is None or len(prev_values) != len(prev_addrs):
            message_box(stdscr,
                ["Value snapshot is missing or mismatched.",
                 "Please run a new First Scan (S) with blank value."],
                "Error", C_ERR)
            return

        mode_lbl = cycle_input(stdscr, "Filter mode      : ", 4, 3,
                               RELATIONAL_MODES, RELATIONAL_MODES[0],
                               allow_cancel=True)
        if mode_lbl is None:
            add_log("Next scan setup cancelled")
            return

        delta = 0.0 if VALUE_TYPES[type_key]["kind"] == "float" else 0
        if mode_lbl in ("decreased by", "increased by"):
            delta_s = input_box(
                stdscr, "Delta amount     : ", 6, 3, 20, "1",
                allow_cancel=True)
            if delta_s is None:
                add_log("Next scan setup cancelled")
                return
            try:
                delta = (float(delta_s)
                         if VALUE_TYPES[type_key]["kind"] == "float"
                         else int(delta_s, 0))
                if not math.isfinite(float(delta)) or delta < 0:
                    raise ValueError("out of range")
                # A delta is a magnitude, not a value of the scanned type, so
                # it is bounded by the width rather than by the signed range:
                # 200 is a legitimate delta for an i8 counter that wraps.
                # Anything past the width would wrap to a different, very
                # surprising number.
                if (VALUE_TYPES[type_key]["kind"] != "float" and
                        delta > WIDTH_MAX[width]):
                    raise ValueError("larger than the scanned width holds")
            except ValueError as exc:
                hint = ("Enter a positive number."
                        if VALUE_TYPES[type_key]["kind"] == "float" else
                        f"Enter a whole number from 0 to "
                        f"{WIDTH_MAX[width]:,}.")
                message_box(stdscr, [f"Invalid delta: {exc}", hint],
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
                    lambda d, t: progress.update(done=d, total=max(t, 1)),
                    value_type=type_key,
                    tolerance=float(state.get("scan_tolerance", 0.0)))
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
        new_values = (progress["values"] if progress["values"] is not None
                      else np.empty(
                          0, dtype=np.dtype(VALUE_TYPES[type_key]["dtype"])))

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

        scan.narrow(new_addrs, new_values)
        # A dropped address is removed from scan_results at drop time, so it
        # can never reappear in a later Next Scan's (necessarily narrower)
        # output -- there is nothing left worth carrying forward.
        state["scan_dropped"] = set()

        hist_mb = _history_bytes() / 1_048_576
        add_log(f"Relational next scan ({mode_lbl}): {len(new_addrs):,} remain, "
                f"undo {hist_mb:.1f} MB, RSS {_rss_mb():.0f} MB")

        add_log(f"Next scan complete — {len(new_addrs):,} candidates remain",
                "info" if len(new_addrs) <= 10 else "warn")
        do_show_results(stdscr)

    else:
        # ── exact-value path (original behaviour) ────────────────────────────
        safe_addstr(stdscr, 4, 3,
            "Enter the new in-game value.", color(C_NORM))
        stdscr.refresh()

        val_s = input_box(stdscr, "New value        : ", 6, 3, 20,
                          allow_cancel=True)
        if val_s is None:
            add_log("Next scan setup cancelled")
            return
        try:
            val = _parse_value_text(val_s, type_key, width)
        except ValueError as exc:
            message_box(stdscr, [str(exc)], "Invalid Value", C_ERR)
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
                    lambda d, t: progress.update(done=d, total=max(t, 1)),
                    value_type=type_key,
                    tolerance=float(state.get("scan_tolerance", 0.0)))
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

        scan.narrow(results, None,
                    truncated=progress.get("truncated", False))
        # A dropped address is removed from scan_results at drop time, so it
        # can never reappear in a later Next Scan's (necessarily narrower)
        # output -- there is nothing left worth carrying forward.
        state["scan_dropped"] = set()

        hist_mb = _history_bytes() / 1_048_576
        add_log(f"Exact next scan type={type_key} val={val}: "
                f"{len(results):,} remain, "
                f"undo {hist_mb:.1f} MB, RSS {_rss_mb():.0f} MB")

        add_log(f"Next scan complete — {len(results):,} candidates remain",
                "info" if len(results) <= 10 and not state["scan_truncated"] else "warn")
        do_show_results(stdscr)


# ── results screen ────────────────────────────────────────────────────────────

def _refresh_visible_locked(ip: str, pid: int, addrs: list, width: int,
                             cache: dict, lock: threading.Lock,
                             cancel_event: Optional[threading.Event] = None,
                             expected_pid: Optional[int] = None,
                             value_type: Optional[str] = None) -> None:
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
    type_key = _normalise_value_type(value_type, width)
    sock = None
    try:
        # Build a _ScanSocket with an aggressively short timeout.  Go through
        # set_timeout(): on the native MemDBG backend there is no ``_s`` to
        # poke, and touching it directly killed this thread on every refresh
        # tick, leaving every Results row stuck showing "…".
        sock = _ScanSocket(ip, pid)
        sock.set_timeout(1.5)   # short: fast exit on Q
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
                    value = _unpack_typed_value(raw, type_key, width)
                    vstr = _format_typed_value(value, type_key, width)
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
        ("Trace Write → Find Pointer (experimental)", "trace_write"),
        ("Trace Write → Instruction Anchor (experimental)", "trace_anchor"),
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
            if action == "trace_write":
                return "trace_write"
            if action == "trace_anchor":
                return "trace_anchor"
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


def _unsigned_inventory_action_ok(stdscr, action: str) -> bool:
    """The visual inventory helpers intentionally operate on counters only."""
    type_key = _current_scan_type()
    if VALUE_TYPES[type_key]["kind"] == "uint":
        return True
    message_box(
        stdscr,
        [f"{action} is designed for unsigned inventory counters.",
         f"The active scan type is {VALUE_TYPES[type_key]['label']}.",
         "Use Apply/Create Cheat for this type, or start an unsigned scan."],
        "Action Not Applicable", C_WARN)
    return False


def do_browse_nearby(stdscr, anchor: int) -> None:
    """Populate Results with plausible nearby values without requiring a change."""
    if not _unsigned_inventory_action_ok(stdscr, "Nearby browsing"):
        return
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
    # This replaces the candidate list wholesale; a surviving resident
    # TurboScan session would make the next Next Scan silently discard the
    # whole nearby set and narrow the old server-side list instead.
    _close_turbo_session()
    scan.replace(candidate_addr, candidate_values, close_turbo=False)
    state["scan_history"].clear()
    add_log(f"Nearby browse @ {hex(int(anchor))}: "
            f"{len(candidate_addr):,} plausible candidates")


def do_discover_nearby(stdscr, anchor: int) -> None:
    """Find neighboring fields that change after an anchored item change."""
    if not _unsigned_inventory_action_ok(stdscr, "Nearby change discovery"):
        return
    width = int(state.get("scan_width", 4))
    stdscr.clear()
    draw_border(stdscr, "ANCHORED GROUP DISCOVERY")
    radius_text = input_box(stdscr, "Radius (hex): ", 3, 3,
                            width=12, default="0x400", allow_cancel=True)
    if radius_text is None:
        add_log("Nearby discovery cancelled")
        return
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
        # Wholesale replacement — discard any resident session first, same
        # reason as do_browse_nearby.
        _close_turbo_session()
        scan.replace(changed_addr, new_values, close_turbo=False)
        state["scan_history"].clear()
        add_log(f"Anchored discovery @ {hex(int(anchor))}: "
                f"{len(changed_addr):,} nearby candidates")


def do_preview_candidate(stdscr, address: int) -> None:
    """Temporarily change one candidate, then always restore its original value."""
    if not _unsigned_inventory_action_ok(stdscr, "Safe candidate preview"):
        return
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
                             width=24, default=str(original + 1),
                             allow_cancel=True)
    if preview_text is None:
        add_log("Candidate preview cancelled")
        return
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


def do_capture_instruction_anchor(stdscr, address: int) -> None:
    """Capture a stable instruction anchor from one traced write.

    Deliberately separate from "Trace Write -> Find Pointer": this path does
    not scan for pointers at all.  It resolves the writing instruction, turns
    it into an executable-memory AOB, and proves the AOB relocates uniquely
    back to it.  Patching is a further explicit confirmation, because a
    matching signature is evidence, not permission.
    """
    width = int(state.get("scan_width", 4))
    if not state.get("connected") or not state.get("pid"):
        message_box(stdscr, ["Connect to a console and select a process first."],
                    "Not Connected", C_WARN)
        return
    if not confirm_box(stdscr,
                       f"Trace writes to {hex(int(address))} with a hardware "
                       f"watchpoint? This attaches the debugger to the game.",
                       "Capture Instruction Anchor"):
        return

    message_box(stdscr, [
        "Attaching and arming a write watchpoint.",
        "Trigger the value now (fire, use the item, take damage...).",
        "",
        "This uses the one debugger attach the game allows.",
    ], "Waiting For A Write", C_NORM)
    try:
        trace = _trace_temporary_access(
            state["ip"], int(state["pid"]), int(address), width,
            timeout=15.0, experimental=True)
    except Exception as exc:
        add_log(f"Instruction-anchor trace failed: {exc}", "warn")
        message_box(stdscr, [
            "No usable write event was captured.",
            str(exc),
            "The watchpoint was cleared and detach was requested.",
        ], "Trace Finished", C_WARN)
        return

    state["last_access_trace"] = trace
    result = capture_instruction_anchor(state["ip"], int(state["pid"]),
                                        trace, width)
    if not result["ok"]:
        add_log(f"Instruction anchor refused ({result['stage']}): "
                f"{result['note']}", "warn")
        message_box(stdscr, [
            "The write was traced, but no stable anchor could be made.",
            f"Stage: {result['stage']}",
            f"  {result['note']}",
            "",
            f"Trap RIP was {hex(int(trace.get('rip', 0)))}, which is never",
            "used as the anchor: it names the instruction after the write.",
        ], "No Stable Anchor", C_WARN)
        return

    anchor = result["anchor"]
    state["last_instruction_anchor"] = anchor
    add_log(f"Instruction anchor captured: writer={hex(anchor['writer'])} "
            f"trap_rip={hex(anchor['trap_rip'])} "
            f"aob={len(anchor['signature']['mask']) // 2}B unique")
    message_box(stdscr, [
        f"Temporary address : {hex(anchor['temporary_address'])}",
        f"TRAP RIP          : {hex(anchor['trap_rip'])}   (not the writer)",
        f"WRITER            : {hex(anchor['writer'])}",
        f"  {anchor['base_reg']} = {hex(anchor['base_value'])} "
        f"{anchor['displacement']:+#x}  ->  "
        f"{hex(anchor['effective_address'])}",
        f"Access            : {anchor['access_mode']}, "
        f"{anchor['access_width']} bytes",
        f"Instruction       : {anchor['instruction_bytes']} "
        f"({anchor['instruction_length']} bytes)",
        "",
        f"AOB length        : {len(anchor['signature']['mask']) // 2} bytes",
        f"Match count       : 1 (unique, executable memory only)",
        f"STABLE ANCHOR     : {hex(anchor['relocated'])}",
        f"Verification      : verified",
    ], "Instruction Anchor Captured", C_OK)

    if not confirm_box(stdscr,
                       f"NOP the instruction at {hex(anchor['relocated'])}? "
                       f"It will be re-verified against live memory first.",
                       "Patch Instruction"):
        message_box(stdscr, [
            "Anchor kept, nothing patched.",
            "It stays available for this session and can be applied later.",
        ], "Anchor Saved", C_NORM)
        return

    patched = patch_instruction_anchor(state["ip"], int(state["pid"]), anchor)
    if not patched.get("ok"):
        add_log(f"Anchor patch refused ({patched.get('stage')}): "
                f"{patched.get('note')}", "warn")
        message_box(stdscr, [
            "Nothing was written.",
            f"Stage: {patched.get('stage')}",
            f"  {patched.get('note')}",
        ], "Patch Refused", C_WARN)
        return
    state["last_anchor_patch"] = patched
    add_log(f"Anchor patched at {hex(int(patched['address']))}: "
            f"{patched.get('original')} -> {patched.get('applied')}")
    message_box(stdscr, [
        f"Patched {hex(int(patched['address']))}.",
        f"  was {patched.get('original')}",
        f"  now {patched.get('applied')}",
        "",
        "The original bytes are recorded for this session so the patch",
        "can be restored.",
    ], "Instruction Patched", C_OK)


def do_trace_item_write(stdscr, address: int) -> None:
    """Run one explicitly confirmed write-only hardware watchpoint trace."""
    width = int(state.get("scan_width", 4))
    validation = _validate_addr_in_maps(
        state["ip"], int(state["pid"]), int(address), width,
        ttl_override=0.0)
    if validation:
        message_box(stdscr, [validation], "Trace Blocked", C_ERR)
        return
    # Refuse outright before anything is attached: arming a watchpoint stops
    # the target, and stopping the console's own UI process freezes the
    # console.
    refusal = _debug_attach_refusal(state.get("proc_name", ""))
    if refusal:
        message_box(stdscr, [
            "Tracing is not available for this process.",
            "",
            refusal,
            "",
            "Attach to the game process and try again.",
        ], "Trace Refused", C_ERR)
        return
    if _debug_attach_is_unusual(state.get("proc_name", "")):
        if not confirm_box(
                stdscr,
                f"'{state.get('proc_name')}' looks like a system service, not "
                "a game.\nAttaching a debugger to it stops it while the "
                "watchpoint is armed,\nwhich may disturb the console.\n\n"
                "Trace it anyway?",
                "Unusual Trace Target"):
            return
    if not confirm_box(stdscr,
            "Experimental one-shot trace: the game may pause briefly. Continue?",
            "Trace Item Write"):
        return

    stdscr.clear()
    draw_border(stdscr, "EXPERIMENTAL ITEM-WRITE TRACE")
    safe_addstr(stdscr, 2, 3, f"Watching: {hex(int(address))}", color(C_WARN))
    safe_addstr(stdscr, 4, 3, "The write-only watchpoint is being armed.",
                color(C_NORM))
    safe_addstr(stdscr, 5, 3, "Change this item exactly once in the game now.",
                color(C_OK) | curses.A_BOLD)
    safe_addstr(stdscr, 7, 3, "Timeout: 15 seconds. Cleanup always runs.",
                color(C_WARN))
    stdscr.refresh()
    try:
        trace = _trace_temporary_access(
            state["ip"], int(state["pid"]), int(address), width,
            timeout=15.0, experimental=True)
    except Exception as exc:
        add_log(f"Experimental item-write trace failed: {exc}", "warn")
        message_box(stdscr, [
            "No usable write event was captured.",
            str(exc),
            "The watchpoint was cleared and debugger teardown was requested.",
        ], "Trace Finished", C_WARN)
        return

    state["last_access_trace"] = trace
    insn = trace.get("instruction") or {}
    # The resolved writer, with no fallback.  The raw trap RIP names the
    # instruction *after* the access, so substituting it would silently report
    # -- and later anchor -- one instruction past the real writer.  A trace that
    # reached here without a writer is a bug, and should fail loudly.
    instruction_addr = int(trace["writer"])
    base_name = str(trace.get("base_reg") or "unknown")
    base_value = int(trace.get("base_value", 0))
    final_offset = int(trace.get("final_offset", 0))
    add_log(
        f"Write trace captured target={hex(int(address))} "
        f"instruction={hex(instruction_addr)} base={base_name}:"
        f"{hex(base_value)} offset={final_offset:+#x}")
    lines = [
        "Write instruction captured successfully.",
        f"Instruction: {hex(instruction_addr)}",
        f"Object base: {base_name} = {hex(base_value)}",
        f"Item-field offset: {final_offset:+#x}",
        f"Access: {trace.get('access_mode', 'write')}",
        "Watchpoint cleared; target resume and detach were requested.",
    ]
    message_box(stdscr, lines, "Item Write Captured", C_OK)

    # The trace is only half the value. Searching backwards from the value
    # itself accepts any pointer landing within _PTR_STRUCT_MAX of it, which
    # is where a backwards scan's coincidental chains come from. The traced
    # instruction hands us the object's base pointer and the field
    # displacement exactly, so the search becomes "find this one known
    # object pointer" with the terminal offset already known.
    reason = _trace_base_is_resolvable(trace)
    if reason:
        message_box(stdscr, [
            "This accessor cannot seed a permanent pointer chain:",
            f"  {reason}.",
            "",
            "The capture is still recorded in the log. Try tracing a",
            "different write to the same value, or use Find Permanent",
            "Pointer, which searches backwards instead.",
        ], "No Stable Base", C_WARN)
        return

    if not confirm_box(
            stdscr,
            f"Search for pointer chains to the traced object at "
            f"{hex(base_value)}?\n"
            "This is the same bounded scan Find Permanent Pointer uses, but\n"
            "aimed at the exact object the game itself dereferenced.",
            "Resolve Traced Object"):
        return

    cancel_event = threading.Event()
    progress = {"done": 0, "total": _PTR_RESOLVE_MAX_NODES,
                "results": None, "error": None}

    def run():
        try:
            progress["results"] = _pointer_candidates_from_trace(
                state["ip"], int(state["pid"]), trace, int(address),
                cancel_event=cancel_event,
                progress_cb=lambda d, t: progress.update(
                    done=d, total=max(int(t), 1)))
        except Exception as exc:
            progress["error"] = str(exc)

    if not _run_scan_with_progress(
            stdscr, run, "Resolving the traced object…", cancel_event, progress):
        add_log("Traced-object resolution cancelled", "warn")
        return
    if progress["error"]:
        message_box(stdscr, [f"Error: {progress['error']}"],
                    "Resolve Failed", C_ERR)
        return

    data = progress["results"] or {}
    candidates = [c for c in data.get("candidates", []) if c.get("verified")]
    if not candidates:
        message_box(stdscr, [
            "No module-rooted chain reached the traced object.",
            "",
            "The object is reachable from the accessor but not from any",
            "static root within the search depth. A deeper Find Permanent",
            "Pointer run may still find one.",
        ], "No Chain Found", C_WARN)
        return

    # Same persistence path as do_resolve_permanent, so these chains enter
    # the identical two-reload validation workflow rather than a parallel one.
    try:
        maps = data.get("maps") or _get_maps_cached(state["ip"], state["pid"])
        game_identity = _pointer_game_identity(state.get("proc_name", ""), maps)
        provisional = _make_pointer_provisionals(
            candidates, maps, state["pid"], state["proc_name"], int(address))
        _merge_pointer_provisionals(
            provisional, state.get("proc_name", ""),
            game_identity=game_identity)
        state["pointer_project_summary"] = _pointer_project_summary(
            state.get("proc_name", ""), maps)
    except Exception as exc:
        add_log(f"Could not persist traced pointer chains: {exc}", "error")
        message_box(stdscr, [f"Chains found but not saved: {exc}"],
                    "Save Failed", C_ERR)
        return

    best = candidates[0]
    add_log(f"Change-triggered resolve: {len(candidates)} verified chain(s) "
            f"from base {base_name}:{hex(base_value)}")
    message_box(stdscr, [
        f"{len(candidates)} verified chain(s) reach the traced object.",
        f"Best: {best.get('module_name')} + "
        f"{int(best.get('module_relative_offset', 0)):#x}",
        f"  offsets {[hex(int(x)) for x in best.get('offsets', [])]}"
        f" then field {final_offset:+#x}",
        f"  confidence {int(best.get('confidence', 0))}%",
        "",
        f"Saved {len(provisional)} provisional chain(s). They are NOT",
        "permanent yet — reload the game, isolate the value again, and run",
        "Resolve permanent twice to promote them.",
    ], "Traced Chains Saved", C_OK)


def do_batch_preview_matching(stdscr) -> None:
    """Temporarily rewrite matching Results as one verified transaction."""
    if not _unsigned_inventory_action_ok(stdscr, "Transactional batch preview"):
        return
    width = int(state.get("scan_width", 4))
    results = state.get("scan_results")
    if results is None or len(results) == 0:
        message_box(stdscr, ["There are no nearby Results to preview."],
                    "Batch Preview", C_WARN)
        return

    stdscr.clear()
    draw_border(stdscr, "TRANSACTIONAL BATCH PREVIEW")
    match_text = input_box(stdscr, "Current value to match: ", 3, 3,
                           width=20, default="1", allow_cancel=True)
    if match_text is None:
        add_log("Batch preview cancelled")
        return
    replacement_text = input_box(stdscr, "Temporary replacement: ", 5, 3,
                                 width=20, default="3", allow_cancel=True)
    if replacement_text is None:
        add_log("Batch preview cancelled")
        return
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
    # Every candidate is snapshot-read and writability-checked before the
    # transaction starts, so anything that reaches here was verified; the old
    # "skipped" branch read a list nothing ever appended to.
    note = "All selected fields were writable and verified."
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
                          refresh_cancel, state["pid"],
                          _current_scan_type()),
                    daemon=True)
                refresh_thread.start()
                refresh_deadline = now
                last_refresh = now

            stdscr.clear()
            draw_border(stdscr, f"RESULTS  ({len(results)} addresses)")
            wlabel = _current_scan_label()
            trunc_warn = "  ⚠ CAPPED — additional matches not displayed" if state.get("scan_truncated") else ""
            safe_addstr(stdscr, 2, 3,
                f"Type: {wlabel}   Process: {state['proc_name']} (PID {state['pid']}){trunc_warn}",
                color(C_ERR) if trunc_warn else color(C_WARN))
            safe_addstr(stdscr, 3, 3,
                "↑↓/jk navigate   G jump   Enter inspect   B bookmark   D drop   U undo   M more   Q back",
                color(C_NORM))

            split_view = w >= 92
            list_right = (w // 2 - 2) if split_view else (w - 3)
            shown = 0
            for idx in range(offset, min(offset + visible, len(results))):
                addr = results[idx]
                with cache_lock: vstr = val_cache.get(addr, "…")
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
                safe_addstr(stdscr, 16, pane_x, "B  Bookmark", color(C_NORM))
                safe_addstr(stdscr, 12, pane_x,
                            "R  Find permanent pointer", color(C_ACC))
                safe_addstr(stdscr, 13, pane_x, "D  Drop result", color(C_ERR))
                safe_addstr(stdscr, 14, pane_x, "N  Next scan", color(C_ACC))
                safe_addstr(stdscr, 15, pane_x, "M  More actions", color(C_WARN))

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
                ("↑↓ Enter inspect", C_NORM), ("Esc/Q back", C_NORM),
                ("N next", C_ACC), ("R permanent", C_ACC),
                ("C cheat", C_OK), ("M more", C_WARN),
                (age_label, C_ERR if stale else C_ACC if is_refreshing else C_NORM),
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
            # Only j/k here: 'g'/'G' already open "jump to result index" on
            # this screen, and that binding predates the vim aliases. Taking
            # it for go-to-bottom would silently change what an existing key
            # does, which is worse than the alias being incomplete.
            if key in (curses.KEY_UP, ord('k')) and sel > 0:
                sel -= 1
            elif key in (curses.KEY_DOWN, ord('j')) and sel < len(results) - 1:
                sel += 1
            elif key == curses.KEY_PPAGE:
                sel = max(0, sel - visible)
            elif key == curses.KEY_NPAGE:
                sel = min(len(results)-1, sel + visible)
            elif key in (ord('g'), ord('G')):
                stdscr.nodelay(False)
                idx_s = input_box(
                    stdscr, "Jump to result index: ", h-2, 3, 12,
                    str(sel+1), allow_cancel=True)
                stdscr.nodelay(True)
                if idx_s is None:
                    continue
                try: sel = max(0, min(int(idx_s)-1, len(results)-1))
                except ValueError: pass
            elif key in (ord('a'), ord('A')) and len(results) > 0:
                stdscr.nodelay(False)
                addr = int(results[sel])
                try:
                    value_s = input_box(
                        stdscr, "Apply value: ", h-2, 3, 20,
                        allow_cancel=True)
                    if value_s is None:
                        continue
                    width = state["scan_width"]
                    type_key = _current_scan_type()
                    value = _parse_value_text(value_s, type_key, width)
                    ack, verified, actual = _write_value_verified(
                        state["ip"], state["pid"], addr, value, width,
                        value_type=type_key)
                    if ack and verified:
                        add_log(f"Applied {value} → {hex(addr)} verified")
                    elif ack and verified is None:
                        add_log(f"Applied {value} → {hex(addr)} but read-back failed", "warn")
                    elif ack:
                        actual_val = _format_typed_value(
                            _unpack_typed_value(actual, type_key, width),
                            type_key, width)
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
            elif key in (ord('b'), ord('B')) and len(results) > 0:
                add_log(_add_bookmark(int(results[sel]), _current_scan_type()))
            elif key in (ord('c'), ord('C')) and len(results) > 0:
                stdscr.nodelay(False)
                _add_cheat_at(stdscr, results[sel])
                stdscr.nodelay(True)
                results = state["scan_results"]
                with cache_lock:
                    val_cache.clear()
            elif key in (ord('p'), ord('P')) and len(results) > 0:
                stdscr.nodelay(False)
                do_resolve_permanent(stdscr, int(results[sel]))
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
                if more_result == "trace_write":
                    stdscr.nodelay(False)
                    do_trace_item_write(stdscr, int(results[sel]))
                    stdscr.nodelay(True)
                    results = state["scan_results"]
                if more_result == "trace_anchor":
                    stdscr.nodelay(False)
                    do_capture_instruction_anchor(stdscr, int(results[sel]))
                    stdscr.nodelay(True)
                    results = state["scan_results"]
                if more_result == "batch_preview":
                    stdscr.nodelay(False)
                    do_batch_preview_matching(stdscr)
                    stdscr.nodelay(True)
                    results = state["scan_results"]
                if more_result == "undo":
                    if _apply_scan_undo() is not None:
                        with cache_lock:
                            val_cache.clear()
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
                # drop_index owns all three consequences that used to be
                # spelled out here: closing the resident TurboScan session
                # (which is matched by connection/PID/width/value-type, never
                # by candidate count, so it would hand this address straight
                # back), removing the parallel previous-value element so the
                # next relational scan cannot pair an address with another
                # address's value, and recording the drop.
                dropped = scan.drop_index(sel)
                results = state["scan_results"]
                with cache_lock:
                    val_cache.pop(dropped, None)
                if len(results) == 0:
                    break
                sel = min(sel, len(results) - 1)
            elif key in (ord('u'), ord('U')):
                if _apply_scan_undo() is not None:
                    results = state["scan_results"]
                    with cache_lock:
                        val_cache.clear()
                    sel = 0; offset = 0
                    add_log(f"Undo: restored {len(results):,} candidates, "
                            f"RSS {_rss_mb():.0f} MB")
            elif key in (ord('q'), ord('Q'), 27):   # Issue #15: Esc also exits
                break
    finally:
        stdscr.nodelay(False)
        # Signal and join the refresh thread so it stops making connections
        # immediately after the user leaves the screen.
        refresh_cancel.set()
        if refresh_thread and refresh_thread.is_alive():
            refresh_thread.join(timeout=2.0)   # 1.5 s socket timeout + 0.5 s margin


# ── hex viewer ────────────────────────────────────────────────────────────────
# Read-only by design. Every comparable tool ships a memory viewer -- Cheat
# Engine's Memory Viewer, PINCE's MemoryView, PS4CheaterNeo's hex editor,
# MemoryEngine360's -- and RDX had none, which matters here more than it would
# on a PC tool: RDX expresses chains as [base+0x18]-0x10, so checking that a
# candidate lands where it should is inherently "look at the bytes at this
# base", and there was no way to do that.
#
# It does not write. RDX already has three audited write paths (Apply, cheats,
# freeze) that validate against the process map first; a fourth one reachable
# by cursoring around a hex dump would be the easiest way in the program to
# corrupt a running game by accident.
_HEX_BYTES_PER_ROW = 16
_HEX_WINDOW_ROWS = 64            # rows fetched per read, not rows displayed
_HEX_REFRESH_INTERVAL = 2.0


def _hex_render_rows(base: int, data: bytes, unreadable: bool = False) -> list:
    """Render a byte window as (address, hex column, ascii column) rows."""
    rows = []
    for offset in range(0, len(data), _HEX_BYTES_PER_ROW):
        chunk = data[offset:offset + _HEX_BYTES_PER_ROW]
        if unreadable:
            hex_col = " ".join("??" for _ in range(_HEX_BYTES_PER_ROW))
            ascii_col = "?" * _HEX_BYTES_PER_ROW
        else:
            hex_col = " ".join(f"{b:02X}" for b in chunk)
            hex_col += "   " * (_HEX_BYTES_PER_ROW - len(chunk))
            ascii_col = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        rows.append((base + offset, hex_col, ascii_col))
    return rows


def _hex_changed_offsets(previous: Optional[bytes], current: bytes) -> set:
    """Byte offsets that differ between two reads of the same window.

    ReClass.NET ships this as "highlight changed memory", and it earns its
    place here for a reason specific to RDX: "find the thing that changes when
    I do X" is exactly what the unknown-value and relational scans exist to
    serve. Shown on a window that is already being re-read on a timer, it
    answers the same question at a glance, on one object, with no scan at all.

    Returns an empty set when there is nothing to compare against, so a first
    frame highlights nothing rather than everything.
    """
    if not previous or len(previous) != len(current):
        return set()
    return {i for i, (a, b) in enumerate(zip(previous, current)) if a != b}


def _hex_fetch(ip: str, pid: int, base: int, length: int) -> tuple:
    """Read a window. Returns (data, unreadable) and never raises.

    Unmapped memory is normal while scrolling -- the viewer walks straight
    off the end of a mapping -- so a failed read renders as '??' rather than
    an error box the user has to dismiss on every keypress.
    """
    try:
        data = ps5_read(ip, pid, base, length)
        if len(data) < length:
            data = data + b"\x00" * (length - len(data))
            return data, False
        return data, False
    except Exception:
        return b"\x00" * length, True


def do_hex_view(stdscr, address: int) -> None:
    """Read-only hex dump anchored at `address`."""
    if state.get("pid") is None:
        message_box(stdscr, ["Attach to a process first."], "Hex View", C_WARN)
        return
    anchor = int(address)
    base = anchor - (anchor % _HEX_BYTES_PER_ROW)
    data, unreadable = b"", True
    window_base = None
    last_read = 0.0
    previous_data = None
    changed = set()
    highlight = True

    stdscr.nodelay(True)
    try:
        while True:
            now = time.time()
            h, w = stdscr.getmaxyx()
            visible = max(1, h - 7)
            span = visible * _HEX_BYTES_PER_ROW
            need_refetch = (window_base != base
                            or now - last_read >= _HEX_REFRESH_INTERVAL)
            if need_refetch:
                fresh, unreadable = _hex_fetch(
                    state["ip"], int(state["pid"]), base, span)
                # Only diff against the same window: scrolling is not a change.
                changed = (_hex_changed_offsets(previous_data, fresh)
                           if (highlight and window_base == base
                               and not unreadable) else set())
                previous_data, data = fresh, fresh
                window_base, last_read = base, now

            stdscr.clear()
            draw_border(stdscr, f"HEX VIEW  {hex(base)}  (read-only)")
            safe_addstr(stdscr, 2, 3,
                        f"Process {state['proc_name']} (PID {state['pid']})"
                        f"   anchor {hex(anchor)}",
                        color(C_NORM))
            if unreadable:
                safe_addstr(stdscr, 2, max(3, w - 26), "UNREADABLE REGION",
                            color(C_ERR) | curses.A_BOLD)
            safe_addstr(stdscr, 3, 3,
                        # Kept under 72 columns: safe_addstr clips, and the
                        # longer form lost "Q back" at the documented minimum
                        # terminal size.
                        "↑↓/jk row  a addr  n anchor  s struct  "
                        "b mark  c changes  Q back",
                        color(C_NORM))

            for i, (row_addr, hex_col, ascii_col) in enumerate(
                    _hex_render_rows(base, data, unreadable)[:visible]):
                on_anchor = row_addr <= anchor < row_addr + _HEX_BYTES_PER_ROW
                attr = (color(C_ACC) | curses.A_BOLD if on_anchor
                        else color(C_ERR) if unreadable else color(C_NORM))
                y = 5 + i
                row_off = row_addr - base
                safe_addstr(stdscr, y, 2, f"{row_addr:016X} ", attr)
                # Per-byte so a changed byte can be picked out of its row.
                # Drawn as segments rather than one string; 16 writes a row is
                # nothing next to the network read that produced the row.
                for b in range(_HEX_BYTES_PER_ROW):
                    x = 2 + 17 + b * 3
                    if x + 2 >= w - 2:
                        break
                    cell = hex_col[b * 3:b * 3 + 2]
                    hot = (row_off + b) in changed
                    safe_addstr(stdscr, y, x, cell,
                                color(C_WARN) | curses.A_BOLD if hot else attr)
                ax = 2 + 17 + _HEX_BYTES_PER_ROW * 3
                if ax + _HEX_BYTES_PER_ROW + 2 < w - 2:
                    safe_addstr(stdscr, y, ax, "|", attr)
                    for b, ch in enumerate(ascii_col):
                        hot = (row_off + b) in changed
                        safe_addstr(stdscr, y, ax + 1 + b, ch,
                                    color(C_WARN) | curses.A_BOLD if hot
                                    else attr)
                    safe_addstr(stdscr, y, ax + 1 + len(ascii_col), "|", attr)

            age = int(now - last_read)
            draw_statusbar(stdscr, [
                (f"{hex(base)}", C_WARN), ("↑↓/jk", C_NORM),
                ("a address", C_ACC), ("n anchor", C_ACC),
                ("b bookmark", C_OK), ("Esc/Q back", C_NORM),
                (f"{len(changed)} changed" if changed else
                 "changes on" if highlight else "changes off",
                 C_WARN if changed else C_NORM),
                ("?? unreadable" if unreadable else f"~{age}s old",
                 C_ERR if unreadable else C_NORM),
            ])
            stdscr.refresh()

            key = stdscr.getch()
            if key == -1:
                time.sleep(0.05)
                continue
            if key == curses.KEY_RESIZE:
                curses.update_lines_cols(); window_base = None; continue
            if key in (curses.KEY_UP, ord('k')):
                base = max(0, base - _HEX_BYTES_PER_ROW)
            elif key in (curses.KEY_DOWN, ord('j')):
                base += _HEX_BYTES_PER_ROW
            elif key == curses.KEY_PPAGE:
                base = max(0, base - span)
            elif key == curses.KEY_NPAGE:
                base += span
            elif key in (ord('n'), ord('N')):
                base = anchor - (anchor % _HEX_BYTES_PER_ROW)
            elif key in (ord('c'), ord('C')):
                highlight = not highlight
                if not highlight:
                    changed = set()
            elif key in (ord('s'), ord('S')):
                stdscr.nodelay(False)
                do_structure_view(stdscr, base)
                stdscr.nodelay(True)
            elif key in (ord('b'), ord('B')):
                add_log(_add_bookmark(base, _current_scan_type()))
            elif key in (ord('a'), ord('A')):
                stdscr.nodelay(False)
                raw = input_box(stdscr, "Go to address: ", h - 2, 3, 20,
                                hex(base), allow_cancel=True,
                                cancel_with_q=False)
                stdscr.nodelay(True)
                if raw:
                    try:
                        target = int(str(raw).strip(), 0)
                        base = max(0, target - (target % _HEX_BYTES_PER_ROW))
                    except ValueError:
                        add_log(f"Not an address: {raw}", "warn")
            elif key in (ord('q'), ord('Q'), 27):
                return
    finally:
        stdscr.nodelay(False)


# ── structure view ────────────────────────────────────────────────────────────
# The second half of the memory-viewer gap. PINCE describes it as "define and
# view memory structures with named, typed members, and overlay them on any
# address"; Cheat Engine reaches the same thing as Ctrl+D from a memory record.
#
# It earns its place in RDX for the same reason the hex pane did: chains are
# expressed as [base+0x18]-0x10, so the question the user actually has is
# "what lives at this base, and which field is mine". A hex dump answers that
# in bytes; a structure answers it in fields.
#
# Auto-dissect classifies each aligned slot by what its bytes could plausibly
# be, checking candidate pointers against the live memory map. It is a
# starting point the user then names and corrects, not an authority -- an
# integer that happens to look like a mapped address is indistinguishable from
# a pointer at this level, and is labelled the way it reads.
_STRUCT_DEFAULT_SPAN = 0x80         # bytes dissected by default
_STRUCT_MAX_SPAN = 0x400
# Structure layouts are remembered per base address so field names survive
# leaving and re-entering the screen. Nothing bounded that dict: Type Scan can
# hand back 4096 instances, and walking them with S left one entry each, every
# entry holding up to _STRUCT_MAX_SPAN/4 field dicts. Oldest-out keeps the
# convenience without the unbounded session growth.
_STRUCT_MAX_REMEMBERED = 64


def _remember_structure(base: int, fields: list) -> None:
    """Store a layout for `base`, evicting the oldest beyond the cap."""
    layouts = state.setdefault("structures", {})
    layouts[int(base)] = fields
    while len(layouts) > _STRUCT_MAX_REMEMBERED:
        # dicts preserve insertion order, so the first key is the oldest.
        layouts.pop(next(iter(layouts)))
_STRUCT_TYPES = ("u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8",
                 "f32", "f64", "ptr", "bytes")


def _struct_slot_type(raw: bytes, offset: int, maps: list,
                      region_starts=None, region_rows=None) -> str:
    """Classify one 8-byte-aligned slot by what its bytes plausibly are."""
    chunk = raw[offset:offset + 8]
    if len(chunk) < 8:
        return "u32" if len(chunk) >= 4 else "u8"
    qword = int.from_bytes(chunk, "little")
    # A value that resolves inside a mapped region is far more likely to be a
    # pointer than a coincidental integer of that magnitude.
    if qword and _ADDR_MIN <= qword <= _ADDR_MAX and maps:
        if region_starts is not None:
            if _region_for_addr(qword, region_starts, region_rows):
                return "ptr"
        elif any(int(r.get("start", 0)) <= qword < int(r.get("end", 0))
                 for r in maps):
            return "ptr"
    dword = int.from_bytes(chunk[:4], "little")
    if dword:
        as_float = struct.unpack("<f", chunk[:4])[0]
        # Game floats are overwhelmingly modest magnitudes; the exponent test
        # rejects the denormal/huge patterns that arbitrary integers produce.
        if (as_float == as_float                      # not NaN
                and abs(as_float) not in (float("inf"),)
                and 1e-4 < abs(as_float) < 1e9):
            return "f32"
    return "u32"


def _struct_auto_fields(raw: bytes, maps: Optional[list] = None) -> list:
    """Propose a field list for a freshly-read window."""
    maps = maps or []
    region_starts, region_rows = (_build_region_lookup(maps) if maps
                                  else (None, None))
    fields = []
    offset = 0
    while offset + 8 <= len(raw):
        kind = _struct_slot_type(raw, offset, maps, region_starts, region_rows)
        fields.append({"offset": offset, "name": f"field_{offset:04X}",
                       "type": kind})
        # A pointer occupies the whole qword; everything else is read as a
        # 4-byte slot so adjacent 32-bit fields stay separately addressable.
        offset += 8 if kind == "ptr" else 4
    return fields


_STRUCT_TYPE_WIDTH = {
    "u64": 8, "i64": 8, "f64": 8, "ptr": 8,
    "u32": 4, "i32": 4, "f32": 4,
    "u16": 2, "i16": 2,
    "u8": 1, "i8": 1,
    # A one-byte "bytes" field was useless: the type is offered in the picker
    # and rendered a single pair of hex digits. A short run is what anyone
    # selecting it actually wants to see.
    "bytes": 8,
}


def _struct_field_width(field: dict) -> int:
    """Bytes one field occupies. Shared so change-detection and rendering
    cannot disagree about how much memory a field covers."""
    return _STRUCT_TYPE_WIDTH.get(str(field.get("type", "u32")), 4)


def _struct_field_value(raw: bytes, field: dict) -> str:
    """Render one field's current value from an already-read window."""
    offset = int(field.get("offset", 0))
    kind = str(field.get("type", "u32"))
    width = _struct_field_width(field)
    chunk = raw[offset:offset + width]
    if len(chunk) < width:
        return "??"
    try:
        if kind == "ptr":
            return hex(int.from_bytes(chunk, "little"))
        if kind == "bytes":
            return " ".join(f"{b:02X}" for b in chunk)
        return _format_typed_value(
            _unpack_typed_value(chunk, kind, width), kind, width)
    except Exception:
        return "??"


# ── IL2CPP symbol import ──────────────────────────────────────────────────────
# The structure view names fields field_0014 because auto-dissect can only
# infer from bytes. Il2CppDumper already knows the answer: for a Unity IL2CPP
# title it emits dump.cs (classes and fields with reconstructed names and
# offsets) and il2cpp.h (C struct definitions). Loading one replaces the
# generated names with real ones and, more usefully, gives each slot a
# *declared* type -- strictly better than the heuristic, because an integer
# that happens to look like a mapped address is indistinguishable from a
# pointer by inspection and is not by declaration.
#
# Scope is deliberately narrow: RDX imports a dump the user already has. It
# does not produce one -- that needs the title's global-metadata.dat and IL2CPP
# binary extracted from the game, which is a user-side step and not something a
# memory scanner should be doing. With no symbols loaded the structure view
# behaves exactly as it did before, so this is purely additive.
_SYMBOL_MAX_CLASSES = 20_000
_SYMBOL_MAX_FIELDS = 512          # per class

# dump.cs field line, e.g.
#   public int currentHealth; // 0x18
#   private static readonly System.String Name; // 0x0
_DUMPCS_CLASS_RE = re.compile(
    r'^\s*(?:\[[^\]]*\]\s*)*'
    r'(?:public|private|protected|internal)?\s*'
    r'(?:static\s+|sealed\s+|abstract\s+|partial\s+|readonly\s+)*'
    r'(?:class|struct)\s+([A-Za-z_][\w.<>`]*)')
_DUMPCS_FIELD_RE = re.compile(
    r'^\s*(?:\[[^\]]*\]\s*)*'
    r'(?:public|private|protected|internal)\s+'
    r'(?:static\s+|readonly\s+|const\s+|volatile\s+)*'
    r'([\w.<>\[\]`,]+)\s+([A-Za-z_]\w*)\s*;\s*//\s*(0x[0-9A-Fa-f]+)')

# Map IL2CPP/C# declared types onto RDX's structure field kinds.
_IL2CPP_TYPE_MAP = {
    "byte": "u8", "sbyte": "i8", "bool": "u8", "char": "u16",
    "short": "i16", "ushort": "u16",
    "int": "i32", "uint": "u32", "Int32": "i32", "UInt32": "u32",
    "long": "i64", "ulong": "u64", "Int64": "i64", "UInt64": "u64",
    "float": "f32", "Single": "f32",
    "double": "f64", "Double": "f64",
    "System.Int32": "i32", "System.UInt32": "u32",
    "System.Single": "f32", "System.Double": "f64",
    "System.Boolean": "u8", "System.Byte": "u8",
    "System.Int64": "i64", "System.UInt64": "u64",
}


def _il2cpp_field_kind(declared: str) -> str:
    """RDX structure type for one declared C# type.

    Anything not a known value type is a managed reference, which in memory is
    a pointer -- that is the case worth getting right, because it is exactly
    what auto-dissect guesses at.
    """
    name = str(declared or "").strip()
    if name in _IL2CPP_TYPE_MAP:
        return _IL2CPP_TYPE_MAP[name]
    short = name.rsplit(".", 1)[-1]
    if short in _IL2CPP_TYPE_MAP:
        return _IL2CPP_TYPE_MAP[short]
    return "ptr"


def parse_il2cpp_dump(text: str, cancel_event=None, progress_cb=None) -> dict:
    """Parse Il2CppDumper `dump.cs` into {class_name: [field, ...]}.

    Fields carry the same shape the structure view already uses
    ({offset, name, type}) so they can be dropped straight in.

    Takes a cancel event and progress callback because a real dump.cs for a
    large title runs to tens of megabytes and this parses at roughly 10 MB/s
    -- long enough that running it inline froze the terminal with no feedback,
    which is not how any other long operation in RDX behaves.
    """
    classes: dict = {}
    current = None
    lines = str(text).splitlines()
    total = max(len(lines), 1)
    for index, line in enumerate(lines):
        if cancel_event is not None and (index & 0x3FF) == 0:
            if cancel_event.is_set():
                raise InterruptedError("symbol parse cancelled")
            if progress_cb:
                progress_cb(index, total)
        class_match = _DUMPCS_CLASS_RE.match(line)
        if class_match:
            if len(classes) >= _SYMBOL_MAX_CLASSES:
                break
            current = class_match.group(1)
            classes.setdefault(current, [])
            continue
        if current is None:
            continue
        field_match = _DUMPCS_FIELD_RE.match(line)
        if not field_match:
            continue
        declared, name, offset_hex = field_match.groups()
        fields = classes[current]
        if len(fields) >= _SYMBOL_MAX_FIELDS:
            continue
        try:
            offset = int(offset_hex, 16)
        except ValueError:
            continue
        fields.append({"offset": offset, "name": name,
                       "type": _il2cpp_field_kind(declared),
                       "declared": declared})
    # Drop classes that yielded no located fields; they cannot overlay anything.
    return {k: sorted(v, key=lambda f: f["offset"])
            for k, v in classes.items() if v}


def _symbol_class_names() -> list:
    return sorted(state.get("symbols", {}).keys())


def do_load_symbols(stdscr) -> None:
    """Load an Il2CppDumper dump.cs for use by the structure view."""
    stdscr.clear()
    draw_border(stdscr, "LOAD SYMBOLS")
    safe_addstr(stdscr, 2, 3,
                "Loads an Il2CppDumper dump.cs so the structure view can use",
                color(C_NORM))
    safe_addstr(stdscr, 3, 3,
                "real class and field names instead of field_0014.",
                color(C_NORM))
    safe_addstr(stdscr, 5, 3,
                "RDX does not produce the dump — run Il2CppDumper against the",
                color(C_WARN))
    safe_addstr(stdscr, 6, 3,
                "title's IL2CPP binary and global-metadata.dat yourself.",
                color(C_WARN))
    raw_path = input_box(stdscr, "dump.cs path: ", 8, 3, 70,
                         state.get("export_dir", str(Path.home())),
                         allow_cancel=True, cancel_with_q=False)
    if raw_path is None:
        return
    path = Path(raw_path).expanduser()
    if not path.exists() or not path.is_file():
        message_box(stdscr, [f"File not found: {path}"], "Load Symbols", C_ERR)
        return
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        message_box(stdscr, [f"Could not read: {exc}"], "Load Symbols", C_ERR)
        return
    cancel_event = threading.Event()
    progress = {"done": 0, "total": 1, "results": None, "error": None}

    def worker():
        try:
            progress["results"] = parse_il2cpp_dump(
                text, cancel_event,
                lambda d, t: progress.update(done=d, total=max(t, 1)))
        except InterruptedError:
            progress["error"] = "cancelled"
        except Exception as exc:
            progress["error"] = str(exc)

    if not _run_scan_with_progress(stdscr, worker, "Parsing dump.cs",
                                   cancel_event, progress):
        return
    if progress["error"]:
        if progress["error"] != "cancelled":
            message_box(stdscr, [f"Could not parse dump.cs: {progress['error']}"],
                        "Load Symbols", C_ERR)
        return
    classes = progress["results"] or {}
    if not classes:
        message_box(stdscr,
                    ["No classes with located fields were found.",
                     "",
                     "This does not look like an Il2CppDumper dump.cs, or it",
                     "was produced without DumpFieldOffset enabled."],
                    "Load Symbols", C_WARN)
        return
    state["symbols"] = classes
    total_fields = sum(len(v) for v in classes.values())
    add_log(f"Symbols loaded: {len(classes)} class(es), "
            f"{total_fields} located field(s) from {path.name}")
    message_box(stdscr,
                [f"Loaded {len(classes)} class(es), {total_fields} field(s).",
                 "",
                 "In the structure view press Y to overlay a class."],
                "Symbols Loaded", C_OK)


def _pick_symbol_class(stdscr) -> Optional[str]:
    """Filterable class picker for the structure view's Y action."""
    names = _symbol_class_names()
    if not names:
        message_box(stdscr,
                    ["No symbols loaded.",
                     "",
                     "Use the command palette -> Load Symbols to load an",
                     "Il2CppDumper dump.cs first."],
                    "Structure", C_WARN)
        return None
    query, sel = "", 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, f"OVERLAY CLASS  ({len(names)} loaded)")
        safe_addstr(stdscr, 2, 3, f"> {query}_", color(C_ACC) | curses.A_BOLD)
        matches = [n for n in names if query.lower() in n.lower()]
        visible = max(1, min(14, h - 7))
        if matches:
            sel = min(sel, len(matches) - 1)
            for i, name in enumerate(matches[:visible]):
                attr = (color(C_SEL) | curses.A_BOLD if i == sel
                        else color(C_NORM))
                count = len(state.get("symbols", {}).get(name, ()))
                safe_addstr(stdscr, 4 + i, 4,
                            f"{'▶ ' if i == sel else '  '}{name}  "
                            f"({count} fields)"[:w - 8], attr)
        else:
            safe_addstr(stdscr, 4, 4, "No matching class.", color(C_WARN))
        draw_statusbar(stdscr, [("type to filter", C_WARN), ("↑↓", C_NORM),
                                ("Enter overlay", C_OK), ("Esc cancel", C_NORM)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols(); continue
        if key == 27:
            # Same rule as the other filtered screens: 'q' is query text here.
            return None
        if key == curses.KEY_UP:
            sel = max(0, sel - 1)
        elif key == curses.KEY_DOWN:
            sel = min(max(len(matches) - 1, 0), sel + 1)
        elif key in (curses.KEY_ENTER, 10, 13) and matches:
            return matches[sel]
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            query = query[:-1]; sel = 0
        elif 32 <= key <= 126:
            query += chr(key); sel = 0


def _struct_pointer_target(field: dict, raw: bytes,
                           maps: Optional[list] = None,
                           region_starts=None, region_rows=None) -> str:
    """Describe where a pointer field points, for the structure view.

    ReClass.NET calls this Pointer Preview. It matters more here than the
    feature name suggests: telling a real object pointer from an integer that
    happens to look like a mapped address is the hardest judgement in RDX's
    whole pointer workflow, and the one _PTR_FAST_DIRECT_RANGE's comment block
    documents getting wrong twice. The region a pointer lands in is the cheap
    half of that judgement and needs no network read.
    """
    if str(field.get("type")) != "ptr":
        return ""
    offset = int(field.get("offset", 0))
    chunk = raw[offset:offset + 8]
    if len(chunk) < 8:
        return ""
    value = int.from_bytes(chunk, "little")
    if not value:
        return "NULL"
    region = None
    if region_starts is not None:
        region = _region_for_addr(value, region_starts, region_rows)
    elif maps:
        region = next((r for r in maps
                       if int(r.get("start", 0)) <= value < int(r.get("end", 0))),
                      None)
    if region is None:
        # Points nowhere mapped: almost certainly not a pointer at all.
        return "unmapped"
    name = str(region.get("name", "") or "anon").rsplit("/", 1)[-1]
    return f"-> {name}+{value - int(region.get('start', 0)):#x}"


def do_structure_view(stdscr, address: int) -> None:
    """Overlay a named, typed field list on `address`."""
    if state.get("pid") is None:
        message_box(stdscr, ["Attach to a process first."], "Structure", C_WARN)
        return
    base = int(address)
    span = _STRUCT_DEFAULT_SPAN
    fields = state.setdefault("structures", {}).get(base)
    raw, unreadable = _hex_fetch(state["ip"], int(state["pid"]), base, span)
    if fields is None:
        try:
            maps = _get_maps_cached(state["ip"], int(state["pid"]))
        except Exception:
            maps = []
        fields = _struct_auto_fields(raw, maps)
        _remember_structure(base, fields)
    sel = 0
    last_read = time.time()
    previous_raw = None
    changed_fields: set = set()
    highlight = True
    overlay_class = None
    try:
        struct_maps = _get_maps_cached(state["ip"], int(state["pid"]))
    except Exception:
        struct_maps = []
    region_starts, region_rows = (_build_region_lookup(struct_maps)
                                  if struct_maps else (None, None))

    stdscr.nodelay(True)
    try:
        while True:
            now = time.time()
            if now - last_read >= 1.0:
                fresh, unreadable = _hex_fetch(
                    state["ip"], int(state["pid"]), base, span)
                if highlight and not unreadable:
                    byte_changes = _hex_changed_offsets(previous_raw, fresh)
                    # A field is changed if any byte it covers moved.
                    # The field's own width, not a flat 8 bytes. With a flat
                    # span two adjacent u8 fields both lit up when one byte
                    # moved, and a symbol-overlaid class of 32-bit fields lit
                    # up two rows per change.
                    changed_fields = {
                        int(f["offset"]) for f in fields
                        if any(o in byte_changes
                               for o in range(int(f["offset"]),
                                              int(f["offset"])
                                              + _struct_field_width(f)))}
                else:
                    changed_fields = set()
                previous_raw, raw = fresh, fresh
                last_read = now
            h, w = stdscr.getmaxyx()
            stdscr.clear()
            draw_border(stdscr,
                    f"STRUCTURE  {hex(base)}"
                    + (f"  [{overlay_class}]" if overlay_class else "")
                    + "  (read-only)")
            safe_addstr(stdscr, 2, 3,
                        f"{len(fields)} field(s) over {span} bytes"
                        f"   {state['proc_name']} (PID {state['pid']})",
                        color(C_ERR) if unreadable else color(C_NORM))
            safe_addstr(stdscr, 3, 3,
                        "Enter rename  T type  +/- span  R re-dissect  "
                        "C changes  Y symbols  H hex  Q back", color(C_NORM))
            visible = max(1, h - 7)
            sel = max(0, min(sel, max(0, len(fields) - 1)))
            start = max(0, sel - visible // 2)
            for i, field in enumerate(fields[start:start + visible]):
                idx = start + i
                hot = int(field["offset"]) in changed_fields
                attr = (color(C_SEL) | curses.A_BOLD if idx == sel
                        else color(C_WARN) | curses.A_BOLD if hot
                        else color(C_ACC) if field["type"] == "ptr"
                        else color(C_NORM))
                target = _struct_pointer_target(
                    field, raw, struct_maps, region_starts, region_rows)
                line = (f"{'>' if idx == sel else ' '}"
                        f"{'*' if hot else ' '}"
                        f"+{int(field['offset']):04X}  "
                        f"{str(field['type']):<6} "
                        f"{str(field['name'])[:22]:<22} "
                        f"{_struct_field_value(raw, field)}"
                        f"{'  ' + target if target else ''}")
                safe_addstr(stdscr, 5 + i, 2, line[:w - 4].ljust(w - 4), attr)
            draw_statusbar(stdscr, [("↑↓ / jk", C_NORM), ("Enter rename", C_OK),
                                    ("T type", C_ACC), ("R re-dissect", C_WARN),
                                    (f"{len(changed_fields)} changed"
                                     if changed_fields else
                                     "changes on" if highlight
                                     else "changes off",
                                     C_WARN if changed_fields else C_NORM),
                                    ("H hex", C_NORM), ("Esc/Q back", C_NORM)])
            stdscr.refresh()

            key = stdscr.getch()
            if key == -1:
                time.sleep(0.05); continue
            if key == curses.KEY_RESIZE:
                curses.update_lines_cols(); continue
            if key in (curses.KEY_UP, ord('k')):
                sel = max(0, sel - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                sel = min(max(0, len(fields) - 1), sel + 1)
            elif key == ord('g'):
                sel = 0
            elif key == ord('G'):
                sel = max(0, len(fields) - 1)
            elif key in (ord('y'), ord('Y')):
                stdscr.nodelay(False)
                chosen = _pick_symbol_class(stdscr)
                stdscr.nodelay(True)
                if chosen:
                    # Replace the guessed layout wholesale. A declared field
                    # list is better information than auto-dissect can infer,
                    # so merging the two would only reintroduce guesses.
                    symbol_fields = [dict(f) for f in
                                     state.get("symbols", {}).get(chosen, ())]
                    if symbol_fields:
                        fields = symbol_fields
                        _remember_structure(base, fields)
                        overlay_class = chosen
                        sel = 0
                        changed_fields = set()
                        add_log(f"Overlaid {chosen} ({len(fields)} fields) "
                                f"at {hex(base)}")
            elif key in (ord('c'), ord('C')):
                highlight = not highlight
                if not highlight:
                    changed_fields = set()
            elif key in (ord('h'), ord('H')):
                stdscr.nodelay(False)
                do_hex_view(stdscr, base + int(fields[sel]["offset"])
                            if fields else base)
                stdscr.nodelay(True)
            elif key in (ord('+'), ord('=')):
                span = min(_STRUCT_MAX_SPAN, span * 2); last_read = 0.0
            elif key == ord('-'):
                span = max(0x20, span // 2); last_read = 0.0
            elif key in (ord('r'), ord('R')):
                stdscr.nodelay(False)
                if confirm_box(stdscr,
                               "Re-dissect this address? Field names you have "
                               "set will be lost.", "Re-dissect"):
                    try:
                        maps = _get_maps_cached(state["ip"], int(state["pid"]))
                    except Exception:
                        maps = []
                    raw, unreadable = _hex_fetch(
                        state["ip"], int(state["pid"]), base, span)
                    fields = _struct_auto_fields(raw, maps)
                    _remember_structure(base, fields)
                stdscr.nodelay(True)
            elif key in (curses.KEY_ENTER, 10, 13) and fields:
                stdscr.nodelay(False)
                name = input_box(stdscr, "Field name: ", h - 2, 3, 24,
                                 str(fields[sel]["name"]), allow_cancel=True,
                                 cancel_with_q=False)
                stdscr.nodelay(True)
                if name:
                    fields[sel]["name"] = str(name)[:32]
            elif key in (ord('t'), ord('T')) and fields:
                stdscr.nodelay(False)
                chosen = cycle_input(stdscr, "Field type: ", h - 2, 3,
                                     list(_STRUCT_TYPES),
                                     str(fields[sel]["type"]),
                                     allow_cancel=True)
                stdscr.nodelay(True)
                if chosen:
                    fields[sel]["type"] = chosen
            elif key in (ord('q'), ord('Q'), 27):
                return
    finally:
        stdscr.nodelay(False)


def _inspect_result(stdscr, addr: int, live_value: str = "…") -> None:
    """Compact address inspector; keeps common actions one screen away."""
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, "ADDRESS INSPECTOR")
        width = state["scan_width"]
        type_key = _current_scan_type()
        wlabel = _current_scan_label()
        safe_addstr(stdscr, 2, 3, f"Address   {hex(addr)}", color(C_OK) | curses.A_BOLD)
        safe_addstr(stdscr, 3, 3, f"Current   {live_value}", color(C_WARN))
        safe_addstr(stdscr, 4, 3, f"Type      {wlabel}", color(C_NORM))
        safe_addstr(stdscr, 5, 3, f"Process   {state['proc_name']} (PID {state['pid']})", color(C_NORM))
        safe_addstr(stdscr, 7, 3, "Actions", color(C_TITLE) | curses.A_BOLD)
        safe_addstr(stdscr, 8, 5, "A  Apply value", color(C_OK))
        safe_addstr(stdscr, 9, 5, "C  Create cheat", color(C_OK))
        safe_addstr(stdscr, 10, 5, "P  Find permanent pointer", color(C_ACC))
        safe_addstr(stdscr, 11, 5, "D  Drop result", color(C_ERR))
        safe_addstr(stdscr, 12, 5, "H  Hex view (read-only)", color(C_NORM))
        safe_addstr(stdscr, 14, 5, "S  Structure view", color(C_NORM))
        safe_addstr(stdscr, 13, 5, "B  Bookmark", color(C_NORM))
        draw_statusbar(stdscr, [("A apply", C_OK), ("C cheat", C_OK),
                                ("P permanent", C_ACC), ("H hex", C_NORM),
                                ("B bookmark", C_OK), ("D drop", C_ERR),
                                ("Esc/Q back", C_NORM)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols(); continue
        if key in (27, ord('q'), ord('Q')):
            return
        if key in (ord('a'), ord('A')):
            value_s = input_box(
                stdscr, "Apply value: ", h - 2, 3, 20,
                allow_cancel=True)
            if value_s is None:
                continue
            try:
                value = _parse_value_text(value_s, type_key, width)
                ack, verified, actual = _write_value_verified(
                    state["ip"], state["pid"], addr, value, width,
                    value_type=type_key)
                if ack and verified:
                    add_log(f"Applied {value} → {hex(addr)} verified")
                elif ack and verified is None:
                    add_log(f"Applied {value} → {hex(addr)} but read-back failed", "warn")
                elif ack:
                    actual_val = _format_typed_value(
                        _unpack_typed_value(actual, type_key, width),
                        type_key, width)
                    add_log(f"Write mismatch {hex(addr)}: wanted {value}, read {actual_val}", "error")
                else:
                    add_log(f"Write rejected at {hex(addr)}", "error")
            except Exception as exc:
                add_log(f"Apply failed at {hex(addr)}: {exc}", "error")
            try:
                raw = ps5_read(state["ip"], state["pid"], addr, width)
                live_value = (_format_typed_value(
                    _unpack_typed_value(raw, type_key, width), type_key, width)
                    if len(raw) == width else "?")
            except Exception:
                live_value = "?"
        elif key in (ord('h'), ord('H')):
            do_hex_view(stdscr, addr)
        elif key in (ord('s'), ord('S')):
            do_structure_view(stdscr, addr)
        elif key in (ord('b'), ord('B')):
            add_log(_add_bookmark(addr, type_key))
        elif key in (ord('c'), ord('C')):
            _add_cheat_at(stdscr, addr)
            return
        elif key in (ord('p'), ord('P')):
            do_resolve_permanent(stdscr, addr)
            return
        elif key in (ord('d'), ord('D')):
            old_results = state["scan_results"]
            old_values = state.get("scan_values")
            # searchsorted would be wrong here: after any host-path Next Scan
            # scan_results comes back in ps5_read_batch's worker-flush order,
            # not sorted, so it returned an index belonging to a different
            # address and np.delete silently desynchronised scan_values from
            # scan_results for every later relational scan.
            # drop_address keeps the parallel previous-value array aligned
            # and closes the resident session; see ScanState.drop_index.
            scan.drop_address(addr)
            add_log(f"Dropped result {hex(addr)}", "warn")
            return


def _write_value_verified(ip: str, pid: int, addr: int, value, width: int,
                          cancel_event: Optional[threading.Event] = None,
                          value_type: Optional[str] = None) -> tuple:
    """Validate, write, and verify one memory value using the standard write path."""
    err = _validate_write_addr(addr)
    if err:
        raise ValueError(err)
    map_err = _validate_addr_in_maps(ip, pid, addr, width)
    if map_err:
        raise ValueError(map_err)
    data = _pack_typed_value(value, value_type, width)
    return ps5_write_verified(ip, pid, addr, data)


def _add_cheat_at(stdscr, addr: int) -> None:
    stdscr.clear()
    draw_border(stdscr, "ADD CHEAT")
    safe_addstr(stdscr, 2, 3, f"Address : {hex(addr)}", color(C_OK) | curses.A_BOLD)
    type_key = _current_scan_type()
    scan_w = state["scan_width"]
    cur = None
    try:
        raw = ps5_read(state["ip"], state["pid"], addr, scan_w)
        cur = _unpack_typed_value(raw, type_key, scan_w)
        safe_addstr(stdscr, 3, 3,
                    f"Current : {_format_typed_value(cur, type_key, scan_w)}",
                    color(C_WARN))
    except Exception:
        pass
    stdscr.refresh()
    name = input_box(stdscr, "Cheat name       : ", 5, 3, 40,
                     allow_cancel=True, cancel_with_q=False)
    if name is None:
        add_log("Add cheat cancelled")
        return
    val_s = input_box(
        stdscr, "Lock-in value    : ", 7, 3, 38,
        _format_typed_value(cur, type_key, scan_w) if cur is not None else "",
        allow_cancel=True)
    if val_s is None:
        add_log("Add cheat cancelled")
        return
    typ   = cycle_input(stdscr, "Cheat type       : ", 9, 3,
                        ["freeze", "write"], "freeze", allow_cancel=True)
    if typ is None:
        add_log("Add cheat cancelled")
        return
    try:
        val = _parse_value_text(val_s, type_key, scan_w)
        module_metadata = {}
        try:
            maps = _get_maps_cached(state["ip"], state["pid"])
            starts, rows = _build_region_lookup(maps)
            region = _region_for_addr(int(addr), starts, rows)
            if region is not None and _is_static_region(region):
                module_name, _module_base, module_rel = (
                    _module_info_for_addr(int(addr), maps))
                if module_name and module_rel is not None:
                    module_metadata = {
                        "module_name": str(module_name),
                        "module_relative_offset": int(module_rel),
                        "game_identity": _pointer_game_identity(
                            state.get("proc_name", ""), maps),
                    }
        except Exception as exc:
            add_log(f"Could not classify cheat address {hex(int(addr))}: {exc}",
                    "warn")
        entry = {
            "name":    name or f"Cheat@{hex(addr)}",
            "address": addr,
            "value":   val,
            "type":    typ,
            "width":   scan_w,
            "value_type": type_key,
            **({"original_value": cur} if cur is not None else {}),
            # Local safety metadata; generate_cht intentionally does not export it.
            "pid":     state["pid"],
            "process": state["proc_name"],
            "session": state["session"],
            **module_metadata,
        }
        state["cheats"].append(entry)
        state["cheats_dirty"] = True
        add_log(f"Added '{entry['name']}' @ {hex(addr)} = "
                f"{_format_typed_value(val, type_key, scan_w)} ({type_key})")

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
    safe_addstr(stdscr, 10, 3, "Esc/Q cancels without writing", color(C_NORM))
    stdscr.refresh()
    addr_s = input_box(stdscr, "Address (hex) : ", 4, 3, 20,
                       allow_cancel=True)
    if addr_s is None:
        add_log("Write setup cancelled")
        return
    labels = [VALUE_TYPES[key]["label"] for key in VALUE_TYPE_ORDER]
    type_label = cycle_input(
        stdscr, "Value type    : ", 6, 3, labels,
        VALUE_TYPES[_current_scan_type()]["label"], allow_cancel=True)
    if type_label is None:
        add_log("Write setup cancelled")
        return
    type_key = (VALUE_TYPE_ORDER[labels.index(type_label)]
                if type_label in labels else _normalise_value_type(type_label))
    val_s = input_box(stdscr,
                      "Raw bytes     : " if type_key == "bytes" else "Value         : ",
                      8, 3, 38, allow_cancel=True)
    if val_s is None:
        add_log("Write setup cancelled")
        return
    try:
        addr = int(addr_s, 0)
        err  = _validate_write_addr(addr)
        if err:
            raise ValueError(err)
        val = _parse_value_text(val_s, type_key)
        width = (len(bytes.fromhex(val)) if type_key == "bytes" else
                 _value_width(type_key))
        # Verify address is inside a writable mapped region (fail-CLOSED: surfaces error to user)
        map_err = _validate_addr_in_maps(state["ip"], state["pid"], addr, width)
        if map_err:
            if not confirm_box(stdscr, f"{map_err}\nWrite anyway?", "Unmapped Address"):
                return
        data = _pack_typed_value(val, type_key, width)
        ack, verified, actual = ps5_write_verified(
            state["ip"], state["pid"], addr, data)
        if ack and verified:
            add_log(f"Write {hex(addr)} = "
                    f"{_format_typed_value(val, type_key, width)} verified")
        elif ack and verified is None:
            add_log(f"Write {hex(addr)} = {val} acknowledged; read-back failed", "warn")
            message_box(stdscr,
                ["The debug payload acknowledged the write,",
                 "but the address could not be read back.",
                 "Check the Log and connection."],
                "Write Unverified", C_WARN)
        elif ack:
            actual_val = _format_typed_value(
                _unpack_typed_value(actual, type_key, width), type_key, width)
            add_log(f"Write mismatch {hex(addr)}: wanted {val}, read {actual_val}", "error")
            message_box(stdscr,
                ["The debug payload acknowledged the command, but memory did not change.",
                 f"Requested: {val}", f"Read back: {actual_val}",
                 "The game may restore the value, or this payload/firmware",
                 "may not support writes to that mapping."],
                "Write Mismatch", C_ERR)
        else:
            add_log(f"Write {hex(addr)} = {val} rejected", "error")
            message_box(stdscr, ["Write rejected by the debug payload."],
                        "Write Failed", C_ERR)
    except Exception as exc:
        message_box(stdscr, [f"Error: {exc}"], "Error", C_ERR)


def _read_cheat_live_value(cheat: dict) -> str:
    """Read the current live value at a cheat's address. Returns str or '?'."""
    try:
        portable = _is_portable_cheat(cheat)
        same_process = str(cheat.get("process", "") or "") in (
            "", str(state.get("proc_name", "") or ""))
        portable_here = (portable and same_process and
                         _portable_cheat_matches_current_game(cheat))
        stale = (cheat.get("pid") != state.get("pid") or
                 cheat.get("session") != state.get("session"))
        if stale and not portable_here:
            return "stale"
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
            addr = _runtime_scalar_address(cheat)
        raw = ps5_read(state["ip"], state["pid"], addr, width)
        if len(raw) == width:
            type_key = _cheat_value_type(cheat)
            return _format_typed_value(
                _unpack_typed_value(raw, type_key, width), type_key, width)
    except Exception:
        pass
    return "?"


def _delete_cheat_with_undo(index: int) -> str:
    """Delete state['cheats'][index], stopping any active freeze first, and
    stash it (with its original position) as a single-slot undo buffer.
    Returns the deleted cheat's name."""
    cheat = state["cheats"][index]
    name = cheat["name"]
    if _is_cheat_frozen(cheat):
        _toggle_cheat_freeze(cheat)
    state["last_deleted_cheat"] = (dict(cheat), index)
    del state["cheats"][index]
    state["cheats_dirty"] = True
    add_log(f"Deleted cheat '{name}' — press Z in Cheat List to restore", "warn")
    return name


def _restore_last_deleted_cheat() -> Optional[str]:
    """Undo the most recent single delete, if any and not already consumed."""
    saved = state.get("last_deleted_cheat")
    state["last_deleted_cheat"] = None
    if saved is None:
        return None
    cheat, index = saved
    index = min(max(index, 0), len(state["cheats"]))
    state["cheats"].insert(index, cheat)
    state["cheats_dirty"] = True
    add_log(f"Restored deleted cheat '{cheat['name']}'")
    return cheat["name"]


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
        if is_pointer and c.get("module_name") and c.get("module_relative_offset") is not None:
            addr_text = (f"{c['module_name']} + "
                         f"{hex(int(c['module_relative_offset']))}")
        elif is_pointer:
            addr_text = (hex(c["base"]) if isinstance(c.get("base"), int)
                         else str(c.get("base")))
        elif _is_module_relative_scalar(c):
            addr_text = (f"{c['module_name']} + "
                         f"{hex(int(c['module_relative_offset']))}")
        else:
            addr_text = hex(int(c["address"]))
        safe_addstr(stdscr, 4, 3, f"Address   {addr_text}{' (pointer)' if is_pointer else ''}", color(C_OK))
        type_key = _cheat_value_type(c)
        set_value = _format_typed_value(c.get("value"), type_key,
                                        int(c.get("width", 4)))
        safe_addstr(stdscr, 5, 3, f"Set       {set_value}", color(C_NORM))
        safe_addstr(stdscr, 6, 3, f"Live      {live}", color(C_OK) if str(live) == set_value else color(C_WARN))
        safe_addstr(stdscr, 7, 3, f"Mode      {c.get('type')}", color(C_NORM))
        safe_addstr(stdscr, 8, 3,
                    f"Type      {VALUE_TYPES[type_key]['label']} "
                    f"({int(c.get('width', state['scan_width']))} byte(s))",
                    color(C_NORM))
        if is_pointer:
            offsets = [int(o, 0) if isinstance(o, str) else int(o)
                       for o in c.get("offsets", [])]
            offs = " → ".join(f"{off:+#x}" for off in offsets)
            terminal = int(c.get("terminal_offset", 0))
            if terminal:
                offs += f"; field {terminal:+#x}"
            safe_addstr(stdscr, 9, 3,
                        f"Chain     {addr_text} → {offs}", color(C_ACC))
            lifecycle = ("cross-reload validated" if _is_cross_reload_pointer(c)
                         else "current session only")
            safe_addstr(stdscr, 10, 3, f"Lifetime  {lifecycle}",
                        color(C_OK) if _is_cross_reload_pointer(c) else color(C_WARN))
        else:
            short, long_text = cheat_durability(c)
            safe_addstr(stdscr, 9, 3, f"Lifetime  {long_text}",
                        color(C_OK) if short != _DURABILITY_SESSION[0]
                        else color(C_WARN))
        frozen = _is_cheat_frozen(c)
        freeze_indicator = _cheat_freeze_indicator(c)
        safe_addstr(stdscr, 11, 3,
                    f"Toggle    {freeze_indicator}",
                    (color(C_ERR) if freeze_indicator == "ERR" else
                     color(C_OK) if frozen else color(C_NORM)))
        draw_statusbar(stdscr, [("A apply", C_OK), ("E edit", C_WARN),
                                ("F/Space toggle", C_ACC),
                                ("D delete", C_ERR), ("Esc/Q back", C_NORM)])
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
        elif key in (ord('f'), ord('F'), ord(' ')):
            do_freeze(stdscr, c)
        elif key in (ord('d'), ord('D')):
            if confirm_box(stdscr, f"Delete '{c['name']}'?", "Delete Cheat"):
                _delete_cheat_with_undo(idx)
                return


def _filter_type_groups(groups: list, query: str) -> list:
    """Groups whose class name or type pointer matches `query`.

    Matches the class name case-insensitively, and also a hex pointer, so a
    pointer copied out of an earlier session or a dump still finds its row.
    """
    text = str(query or "").strip().lower()
    if not text:
        return list(groups)
    out = []
    for group in groups:
        name = str(group.get("class_name", "") or "").lower()
        pointer = hex(int(group.get("type_ptr", 0))).lower()
        module = str(group.get("module_name", "") or "").lower()
        if text in name or text in pointer or text in module:
            out.append(group)
    return out


def do_type_scan(stdscr) -> None:
    """Find live objects grouped by the type pointer at their base."""
    if state.get("pid") is None:
        message_box(stdscr, ["Attach to a process first."], "Type Scan", C_WARN)
        return
    stdscr.clear()
    draw_border(stdscr, "TYPE SCAN")
    safe_addstr(stdscr, 2, 3,
                "Groups heap objects by the type pointer at their base.",
                color(C_NORM))
    safe_addstr(stdscr, 3, 3,
                "For IL2CPP titles that is the Il2CppClass pointer, so each "
                "group is one class.", color(C_NORM))
    safe_addstr(stdscr, 4, 3, "Read-only: nothing is written.", color(C_OK))
    raw_min = input_box(stdscr, "Minimum instances: ", 6, 3, 8,
                        str(_TYPE_SCAN_MIN_INSTANCES), allow_cancel=True,
                        cancel_with_q=False)
    if raw_min is None:
        return
    try:
        min_instances = max(2, int(str(raw_min).strip(), 0))
    except ValueError:
        message_box(stdscr, [f"Not a number: {raw_min}"], "Type Scan", C_ERR)
        return

    cancel_event = threading.Event()
    progress = {"done": 0, "total": 1, "results": None, "error": None}

    def worker():
        try:
            progress["results"] = scan_type_instances(
                state["ip"], int(state["pid"]), min_instances,
                cancel_event,
                lambda d, t: progress.update(done=d, total=max(t, 1)))
        except InterruptedError:
            progress["error"] = "cancelled"
        except Exception as exc:
            progress["error"] = str(exc)

    if not _run_scan_with_progress(stdscr, worker, "Scanning heap for types",
                                   cancel_event, progress):
        return
    if progress["error"]:
        message_box(stdscr, [f"Type scan failed: {progress['error']}"],
                    "Type Scan", C_ERR)
        return
    groups = progress["results"] or []
    if not groups:
        message_box(stdscr,
                    ["No type pointers found.",
                     "",
                     "Either the title does not use a type-pointer layout,",
                     "or the minimum instance count is too high."],
                    "Type Scan", C_WARN)
        return

    sel = 0
    # Type Scan answers "what objects are in this heap?". Once its rows carry
    # class names (patch107) the next question is the inverse -- "take me to
    # PlayerController" -- which is what Il2CppGG's class search does. Both
    # halves already existed here; this is the filter that joins them.
    query = ""
    while True:
        visible_groups = _filter_type_groups(groups, query)
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, f"TYPES  ({len(visible_groups)}/{len(groups)})"
                            if query else f"TYPES  ({len(groups)} found)")
        safe_addstr(stdscr, 2, 3,
                    "Enter instances  Tab structure  type to filter  Esc back",
                    color(C_NORM))
        if query:
            safe_addstr(stdscr, 2, max(3, w - 34),
                        f"filter: {query}_"[:30], color(C_ACC) | curses.A_BOLD)
        named_count = sum(1 for g in visible_groups if g.get("class_name"))
        if named_count:
            safe_addstr(stdscr, 3, 3,
                        f"{named_count} of {len(groups)} named from live "
                        f"class data", color(C_OK))
        visible = max(1, h - 8)
        sel = max(0, min(sel, max(0, len(visible_groups) - 1)))
        start = max(0, sel - visible // 2)
        for i, group in enumerate(visible_groups[start:start + visible]):
            idx = start + i
            attr = (color(C_SEL) | curses.A_BOLD if idx == sel
                    else color(C_ACC) if group.get("class_name")
                    else color(C_NORM))
            module = group.get("module_name") or "?"
            rel = group.get("module_relative_offset")
            where = (f"{module}+{rel:#x}" if rel is not None else module)
            # The class name is the useful column when it resolved; the raw
            # pointer stays visible when it did not, rather than a blank.
            label = group.get("class_name") or hex(group["type_ptr"])
            line = (f"{'>' if idx == sel else ' '} "
                    f"{group['count']:>7,} x  "
                    f"{label:<30.30} {where}")
            safe_addstr(stdscr, 5 + i, 2, line[:w - 4].ljust(w - 4), attr)
        draw_statusbar(stdscr, [("↑↓", C_NORM),
                                ("Enter instances", C_OK),
                                ("Tab structure", C_ACC),
                                ("type to filter", C_WARN),
                                ("Esc back", C_NORM)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols(); continue
        # This screen now has a live typeahead filter, so printable
        # characters are query text. That is the same rule
        # screen_proc_select and the command palette document, and it is why
        # 'q' cannot mean quit here: a class name may well start with one.
        # Esc is unambiguous. Navigation moves to the arrow keys only.
        if key == 27:
            if query:
                query = ""; sel = 0
                continue
            return
        if key == curses.KEY_UP:
            sel = max(0, sel - 1)
        elif key == curses.KEY_DOWN:
            sel = min(max(0, len(visible_groups) - 1), sel + 1)
        elif key == curses.KEY_PPAGE:
            sel = max(0, sel - visible)
        elif key == curses.KEY_NPAGE:
            sel = min(max(0, len(visible_groups) - 1), sel + visible)
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            query = query[:-1]; sel = 0
        elif key == ord('\t') and visible_groups:
            # Tab, not 'S'. Every printable character is filter text on this
            # screen, and the letter branch ran first -- so a capital S was
            # swallowed and "System", "Slot" and "Sprite" could not be typed
            # at all, in a filter whose whole purpose is finding a class by
            # name. Tab is unambiguous and matches the process picker.
            instances = visible_groups[sel]["instances"]
            if len(instances):
                do_structure_view(stdscr, int(instances[0]))
        elif 32 <= key <= 126:
            query += chr(key); sel = 0
        elif key in (curses.KEY_ENTER, 10, 13) and visible_groups:
            group = visible_groups[sel]
            instances = _make_addr_array(int(a) for a in group["instances"])
            if not len(instances):
                continue
            if not confirm_box(
                    stdscr,
                    f"Load {len(instances):,} instance address(es) of "
                    f"{hex(group['type_ptr'])} into Results?\n"
                    "This replaces the current scan results.",
                    "Open Instances"):
                continue
            # These are object bases, not values of the current scan type, so
            # there is no meaningful previous-value array to carry. Switch the
            # display type to u64 as well: left on u32 the Results screen
            # renders the low half of each object's type pointer, which is a
            # number that means nothing. As u64 every row shows the whole type
            # pointer, identical down the list, which is a useful confirmation
            # that these really are instances of one type.
            state["scan_type"] = "u64"
            state["scan_width"] = 8
            scan.replace(instances, None)
            add_log(f"Type scan: loaded {len(instances):,} instance(s) of "
                    f"{group.get('class_name') or hex(group['type_ptr'])} "
                    f"into Results; display type set to u64 so each row "
                    f"shows its type pointer")
            do_show_results(stdscr)
            return


def _dispatch_structure_view(stdscr) -> None:
    """Palette entry: ask for a base address, then overlay a structure."""
    if state.get("pid") is None:
        message_box(stdscr, ["Attach to a process first."], "Structure", C_WARN)
        return
    seed = (hex(int(state["scan_results"][0]))
            if len(state.get("scan_results", ())) else "0x0")
    raw = input_box(stdscr, "Structure base address: ", 4, 3, 20, seed,
                    allow_cancel=True, cancel_with_q=False)
    if not raw:
        return
    try:
        do_structure_view(stdscr, int(str(raw).strip(), 0))
    except ValueError:
        message_box(stdscr, [f"Not an address: {raw}"], "Structure", C_ERR)


def _dispatch_hex_view(stdscr) -> None:
    """Palette entry: ask for an address, then open the viewer there."""
    if state.get("pid") is None:
        message_box(stdscr, ["Attach to a process first."], "Hex View", C_WARN)
        return
    seed = (hex(int(state["scan_results"][0]))
            if len(state.get("scan_results", ())) else "0x0")
    raw = input_box(stdscr, "Hex view address: ", 4, 3, 20, seed,
                    allow_cancel=True, cancel_with_q=False)
    if not raw:
        return
    try:
        do_hex_view(stdscr, int(str(raw).strip(), 0))
    except ValueError:
        message_box(stdscr, [f"Not an address: {raw}"], "Hex View", C_ERR)


def do_bookmarks(stdscr) -> None:
    """Addresses kept for investigation, separate from saved cheats."""
    sel = 0
    while True:
        bookmarks = state.get("bookmarks", [])
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, f"BOOKMARKS  ({len(bookmarks)})")
        if not bookmarks:
            safe_addstr(stdscr, 3, 3, "No bookmarks yet.", color(C_WARN))
            safe_addstr(stdscr, 5, 3,
                        "Press B on a result or in the address inspector to",
                        color(C_NORM))
            safe_addstr(stdscr, 6, 3,
                        "keep an address here without creating a cheat.",
                        color(C_NORM))
            safe_addstr(stdscr, 8, 3,
                        "Attach a pointer chain with P and the bookmark",
                        color(C_ACC))
            safe_addstr(stdscr, 9, 3,
                        "survives a reload instead of going stale.",
                        color(C_ACC))
            draw_statusbar(stdscr, [("Esc/Q back", C_NORM)])
            stdscr.refresh()
            if stdscr.getch() in (ord('q'), ord('Q'), 27):
                return
            continue

        sel = max(0, min(sel, len(bookmarks) - 1))
        safe_addstr(stdscr, 2, 3,
                    "Enter inspect   C cheat   P attach chain   D delete   Q back",
                    color(C_NORM))
        visible = max(1, h - 7)
        start = max(0, sel - visible // 2)
        for i, bookmark in enumerate(bookmarks[start:start + visible]):
            idx = start + i
            chained = bool(bookmark.get("chain"))
            stale = not _bookmark_is_current(bookmark)
            attr = (color(C_SEL) | curses.A_BOLD if idx == sel
                    else color(C_ERR) if stale
                    else color(C_ACC) if chained else color(C_NORM))
            note = bookmark.get("note", "")
            # A chained bookmark shows where it resolves *now*, which is the
            # whole point of it having a chain.
            shown = (_bookmark_live_address(bookmark) if chained and not stale
                     else int(bookmark["address"]))
            flag = (" STALE" if stale else " CHAIN" if chained else "")
            line = (f"{'>' if idx == sel else ' '} "
                    f"{hex(int(shown)):<18} "
                    f"{bookmark['value_type']:<6}{flag:<7} {note}")
            safe_addstr(stdscr, 4 + i, 2, line[:w - 4].ljust(w - 4), attr)

        chained_n = sum(1 for b in bookmarks if b.get("chain"))
        draw_statusbar(stdscr, [("↑↓ / jk", C_NORM), ("Enter inspect", C_OK),
                                ("C cheat", C_OK), ("P chain", C_ACC),
                                (f"{chained_n} chained" if chained_n else
                                 "none chained", C_ACC if chained_n else C_NORM),
                                ("D delete", C_ERR), ("Esc/Q back", C_NORM)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols(); continue
        if key in (curses.KEY_UP, ord('k')):
            sel = max(0, sel - 1)
        elif key in (curses.KEY_DOWN, ord('j')):
            sel = min(len(bookmarks) - 1, sel + 1)
        elif key == ord('g'):
            sel = 0
        elif key == ord('G'):
            sel = len(bookmarks) - 1
        elif key in (curses.KEY_ENTER, 10, 13):
            bookmark = bookmarks[sel]
            if not _bookmark_is_current(bookmark):
                message_box(stdscr,
                            ["This bookmark was taken in a different process",
                             "or console session, so its address no longer",
                             "refers to the same thing. Delete it and scan again."],
                            "Stale Bookmark", C_ERR)
                continue
            _inspect_result(stdscr, _bookmark_live_address(bookmark))
        elif key in (ord('c'), ord('C')):
            bookmark = bookmarks[sel]
            if not _bookmark_is_current(bookmark):
                message_box(stdscr, ["Stale bookmark — cannot become a cheat."],
                            "Stale Bookmark", C_ERR)
                continue
            _add_cheat_at(stdscr, _bookmark_live_address(bookmark))
        elif key in (ord('p'), ord('P')):
            bookmark = bookmarks[sel]
            if bookmark.get("chain"):
                message_box(stdscr,
                            ["This bookmark already carries a chain.",
                             "It rebases on every attach."],
                            "Pointer Chain", C_OK)
                continue
            if not _bookmark_is_current(bookmark):
                message_box(stdscr,
                            ["Stale bookmark — its address no longer refers",
                             "to the thing it was taken on, so a chain found",
                             "for it now would be meaningless."],
                            "Stale Bookmark", C_ERR)
                continue
            # Reuse the existing resolver wholesale; a chain good enough for
            # a cheat is good enough for a bookmark.
            do_resolve_permanent(stdscr, int(bookmark["address"]))
        elif key in (ord('d'), ord('D')):
            removed = _remove_bookmark(sel)
            if removed:
                add_log(f"Removed bookmark {hex(int(removed['address']))}")
                sel = max(0, sel - 1)
        elif key in (ord('q'), ord('Q'), 27):
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
                    "↑↓ select   Enter inspect   F/Space toggle   A apply once   D delete",
                    color(C_NORM))
                hdr = f"  {'Name':<26}  {'Address':<22}  {'Set':<8}  {'Live':<10}  Type"
                safe_addstr(stdscr, 3, 2, hdr[:w - 4],
                            color(C_TITLE) | curses.A_UNDERLINE)
                if sel < offset:             offset = sel
                if sel >= offset + visible:  offset = sel - visible + 1
                for i, c in enumerate(cheats[offset:offset + visible]):
                    ri   = offset + i
                    attr = color(C_SEL) | curses.A_BOLD if ri == sel else color(C_NORM)
                    is_pointer = ("offsets" in c and
                                  c.get("offsets") is not None)
                    if ((is_pointer or _is_module_relative_scalar(c)) and
                            c.get("module_name") and
                            c.get("module_relative_offset") is not None):
                        _disp_addr = (f"{c['module_name']}+"
                                      f"{hex(int(c['module_relative_offset']))}")
                    elif is_pointer:
                        raw_base = c.get("base")
                        _disp_addr = ((hex(raw_base) if isinstance(raw_base, int)
                                       else str(raw_base)) + " (ptr)")
                    else:
                        _disp_addr = hex(int(c["address"]))
                    with cache_lock:
                        live_val = live_cache.get(ri, "…")
                    # Colour live value: green if matches set value, yellow if differs
                    set_val_str = _format_typed_value(
                        c["value"], _cheat_value_type(c), int(c["width"]))
                    if live_val not in ("…", "?", "?(chain)"):
                        live_attr = color(C_OK) if live_val == set_val_str else color(C_WARN)
                    else:
                        live_attr = color(C_NORM)
                    line_base = (f"  {c['name']:<26}  {_disp_addr:<22}  "
                                 f"{set_val_str:<8}  ")
                    toggle = _cheat_freeze_indicator(c)
                    durability = cheat_durability(c)[0]
                    live_part = (f"{live_val:<10}  [{toggle}] "
                                 f"{c['type']:<8} {durability}")
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
            status_items = [
                ("↑↓ navigate", C_NORM), ("Enter inspect", C_OK),
                ("F/Space toggle", C_ACC),
                ("A apply once", C_OK), ("D delete", C_ERR),
            ]
            if state.get("last_deleted_cheat"):
                status_items.append(("Z restore", C_OK))
            status_items.append(
                ("⟳ live" if is_refreshing else "live values", C_ACC))
            status_items.append(("Esc/Q back", C_NORM))
            draw_statusbar(stdscr, status_items)
            stdscr.refresh()

            key = stdscr.getch()
            if key == -1:
                time.sleep(0.05)
                continue
            if key == curses.KEY_RESIZE:
                curses.update_lines_cols()
                continue
            if key in (curses.KEY_UP, ord('k')) and sel > 0:               sel -= 1
            elif key in (curses.KEY_DOWN, ord('j')) and sel < len(cheats) - 1: sel += 1
            elif key == ord('g') and cheats:                       sel = 0
            elif key == ord('G') and cheats:                       sel = len(cheats) - 1
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
            elif key in (ord('f'), ord('F'), ord(' ')) and cheats:
                try:
                    _toggle_cheat_freeze(cheats[sel])
                except Exception as exc:
                    add_log(f"Freeze toggle failed: {exc}", "error")
            elif key in (ord('d'), ord('D')) and cheats:
                stdscr.nodelay(False)
                name = cheats[sel]["name"]
                if confirm_box(stdscr, f"Delete '{name}'?", "Delete Cheat"):
                    _delete_cheat_with_undo(sel)
                    cheats = state["cheats"]
                    with cache_lock:
                        live_cache.clear()
                    if not cheats:
                        sel = 0
                    else:
                        sel = min(sel, len(cheats) - 1)
                    offset = min(offset, max(0, len(cheats) - visible))
                stdscr.nodelay(True)
            elif key in (ord('z'), ord('Z')) and state.get("last_deleted_cheat"):
                restored = _restore_last_deleted_cheat()
                if restored:
                    cheats = state["cheats"]
                    with cache_lock:
                        live_cache.clear()
            elif key in (ord('q'), ord('Q'), 27):
                break
    finally:
        stdscr.nodelay(False)
        refresh_cancel.set()
        if refresh_thread and refresh_thread.is_alive():
            refresh_thread.join(timeout=2.0)


def _apply_cheat_once(stdscr, cheat: dict) -> None:
    """Apply one saved cheat value immediately and verify the result.
    Supports both flat (address) cheats and pointer-chain cheats."""
    is_pointer = "offsets" in cheat and cheat.get("offsets") is not None
    portable_cheat = _is_portable_cheat(cheat)
    same_process = str(cheat.get("process", "") or "") in (
        "", str(state.get("proc_name", "") or ""))
    portable_here = (portable_cheat and same_process and
                     _portable_cheat_matches_current_game(cheat))
    owner_pid = cheat.get("pid")
    if owner_pid is None:
        message_box(stdscr,
            ["This cheat predates process ownership tracking.",
             "Re-add it from current scan results before applying it."],
            "Unowned Cheat", C_WARN)
        return
    if (cheat.get("session") != state["session"] and
            not portable_here):
        message_box(stdscr,
            ["This cheat belongs to an earlier PS5 connection session.",
             "The game or payload may have restarted and reused its PID.",
             "Only module-relative or validated pointer cheats can be rebased."],
            "Stale Cheat", C_ERR)
        return
    if owner_pid != state["pid"] and not portable_here:
        owner_name = cheat.get("process") or "unknown process"
        message_box(stdscr,
            [f"This address belongs to PID {owner_pid} ({owner_name}).",
             f"Current process is PID {state['pid']} ({state['proc_name']}).",
             "Application blocked to avoid writing to the wrong process."],
            "Stale Cheat", C_ERR)
        return

    width = int(cheat["width"])
    type_key = _cheat_value_type(cheat)
    value = cheat["value"]

    if is_pointer:
        # ── pointer cheat: resolve chain first ───────────────────────────
        try:
            base = _runtime_pointer_base(cheat)
            offsets = [int(o, 0) if isinstance(o, str) else int(o)
                       for o in cheat["offsets"]]
            ok, addr, steps = _resolve_pointer_chain(
                state["ip"], state["pid"], base, offsets,
                int(cheat.get("terminal_offset", 0)))
        except Exception as exc:
            add_log(f"Pointer base unavailable for '{cheat['name']}': {exc}",
                    "error")
            message_box(stdscr, [str(exc)], "Pointer Module Unavailable", C_ERR)
            return
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
        try:
            addr = _runtime_scalar_address(cheat)
        except Exception as exc:
            message_box(stdscr, [str(exc)],
                        "Cheat Module Unavailable", C_ERR)
            return

    map_err = _validate_addr_in_maps(state["ip"], state["pid"], addr, width)
    if map_err and not confirm_box(stdscr, f"{map_err}\nWrite anyway?", "Unmapped Address"):
        return

    data = _cheat_value_bytes(cheat)
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
        actual_value = _format_typed_value(
            _unpack_typed_value(actual, type_key, width), type_key, width)
        add_log(f"Apply mismatch '{cheat['name']}': read {actual_value}", "error")
        message_box(stdscr,
                    ["Write acknowledged but did not stick.",
                     f"Requested: {value}", f"Read back: {actual_value}"],
                    "Apply Mismatch", C_ERR)
    else:
        add_log(f"Apply rejected for '{cheat['name']}'", "error")
        message_box(stdscr, ["Write rejected by the debug payload."],
                    "Apply Failed", C_ERR)


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
    new_name = input_box(
        stdscr, "Name  : ", 5, 3, 40, c["name"],
        allow_cancel=True, cancel_with_q=False)
    if new_name is None:
        add_log("Edit cheat cancelled")
        return
    type_key = _cheat_value_type(c)
    val_s = input_box(
        stdscr, "Value : ", 7, 3, 38,
        _format_typed_value(c["value"], type_key, int(c["width"])),
        allow_cancel=True)
    if val_s is None:
        add_log("Edit cheat cancelled")
        return
    # Pointer cheats use pointer_freeze/pointer_write; flat cheats use freeze/write.
    # Offering the wrong set would crash cycle_input (options.index raises ValueError).
    if is_pointer:
        type_opts = ["pointer_freeze", "pointer_write"]
    else:
        type_opts = ["freeze", "write"]
    new_type = cycle_input(
        stdscr, "Type  : ", 9, 3, type_opts, c["type"],
        allow_cancel=True)
    if new_type is None:
        add_log("Edit cheat cancelled")
        return
    try:
        new_val = _parse_value_text(val_s, type_key, int(c["width"]))
    except ValueError as exc:
        message_box(stdscr, [str(exc), "Keeping the previous value."],
                    "Invalid Value", C_WARN)
        new_val = c["value"]
    state["cheats"][idx].update({"name": new_name, "value": new_val, "type": new_type})
    add_log(f"Edited '{new_name}' val={new_val} type={new_type}")
    message_box(
        stdscr,
        [f"Updated '{new_name}'.", "Memory was not changed; use Apply explicitly."],
        "Cheat Updated", C_OK)


def _parse_int_field(value, field_name):
    try:
        return int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {field_name}: {value!r}")


def _offer_salvaged_chains(stdscr, path, salvage: list) -> bool:
    """Re-verify a mismatched trainer's chains against the running build.

    Returns True when the user took something from it, so the caller can stop
    rather than falling through to the mismatch error.

    Nothing here is imported on the file's word. Every chain is walked
    against the live memory map, and only the ones that resolve to a
    currently-writable address are offered -- which is exactly the test a
    saved cheat's chain has to pass before RDX will apply it.
    """
    if not confirm_box(
            stdscr,
            f"{Path(path).name} was made for a different build of this game,\n"
            f"so its addresses are wrong.\n\n"
            f"It carries {len(salvage)} pointer chain(s), and those usually\n"
            f"survive a patch even when addresses do not.\n\n"
            f"Re-verify them against the running build?",
            "Different Game Build"):
        return False

    cancel_event = threading.Event()
    progress = {"done": 0, "total": max(len(salvage), 1),
                "results": None, "error": None}

    def worker():
        found = []
        try:
            for i, chain in enumerate(salvage):
                if cancel_event.is_set():
                    raise InterruptedError("cancelled")
                resolved = _verify_salvaged_chain(chain)
                if resolved is not None:
                    found.append((chain, resolved))
                progress["done"] = i + 1
            progress["results"] = found
        except InterruptedError:
            progress["error"] = "cancelled"
        except Exception as exc:
            progress["error"] = str(exc)

    if not _run_scan_with_progress(stdscr, worker, "Re-verifying chains",
                                   cancel_event, progress):
        return False
    if progress["error"]:
        if progress["error"] != "cancelled":
            message_box(stdscr, [f"Chain re-verification failed: "
                                 f"{progress['error']}"],
                        "Salvage Failed", C_ERR)
        return False

    survivors = progress["results"] or []
    if not survivors:
        message_box(stdscr,
                    [f"None of the {len(salvage)} chain(s) resolve against",
                     "the running build.",
                     "",
                     "The layout changed too, not just the addresses."],
                    "Nothing Salvaged", C_WARN)
        return False

    sel = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_border(stdscr, f"SALVAGED CHAINS  ({len(survivors)} resolve)")
        safe_addstr(stdscr, 2, 3,
                    f"From {Path(path).name}, re-verified against this build:",
                    color(C_NORM))
        safe_addstr(stdscr, 3, 3,
                    "B bookmark   C create cheat   A take all as bookmarks   Q back",
                    color(C_NORM))
        visible = max(1, h - 8)
        sel = max(0, min(sel, len(survivors) - 1))
        start = max(0, sel - visible // 2)
        for i, (chain, resolved) in enumerate(survivors[start:start + visible]):
            idx = start + i
            attr = (color(C_SEL) | curses.A_BOLD if idx == sel
                    else color(C_OK))
            line = (f"{'>' if idx == sel else ' '} "
                    f"{hex(resolved):<18} "
                    f"{chain['module_name']}+{chain['module_relative_offset']:#x} "
                    f"{chain['name'][:24]}")
            safe_addstr(stdscr, 5 + i, 2, line[:w - 4].ljust(w - 4), attr)
        draw_statusbar(stdscr, [("↑↓ / jk", C_NORM), ("B bookmark", C_OK),
                                ("C cheat", C_OK), ("A all", C_ACC),
                                ("Esc/Q back", C_NORM)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols(); continue
        if key in (curses.KEY_UP, ord('k')):
            sel = max(0, sel - 1)
        elif key in (curses.KEY_DOWN, ord('j')):
            sel = min(len(survivors) - 1, sel + 1)
        elif key in (ord('b'), ord('B')):
            chain, resolved = survivors[sel]
            add_log(_add_bookmark(resolved, _current_scan_type(),
                                  f"salvaged: {chain['name']}"[:64],
                                  chain=chain))
        elif key in (ord('a'), ord('A')):
            taken = 0
            for chain, resolved in survivors:
                _add_bookmark(resolved, _current_scan_type(),
                              f"salvaged: {chain['name']}"[:64], chain=chain)
                taken += 1
            add_log(f"Salvaged {taken} chain(s) from {Path(path).name} "
                    f"into bookmarks")
            message_box(stdscr,
                        [f"Added {taken} bookmark(s), each carrying its chain.",
                         "",
                         "They rebase on every attach, so they survive a",
                         "reload. Verify the values look right before",
                         "promoting any of them to a cheat."],
                        "Salvaged", C_OK)
            return True
        elif key in (ord('c'), ord('C')):
            _chain, resolved = survivors[sel]
            _add_cheat_at(stdscr, resolved)
        elif key in (ord('q'), ord('Q'), 27):
            return bool(state.get("bookmarks") or state.get("cheats"))


def _do_import_static_patch_mods(stdscr, path: Path, mods: list,
                                 kind_label: str, file_title_id: str,
                                 file_process: str = "") -> None:
    """Shared tail for importing etaHEN/GoldHEN JSON or a decrypted .mc4's
    mods into RDX's cheat list, resolved against the currently attached
    process's live main module (never trusted from the file itself — see
    _mods_to_import_entries)."""
    if len(mods) > MAX_IMPORT_CHEATS:
        message_box(stdscr,
            [f"{kind_label} holds {len(mods):,} entries, over the "
             f"{MAX_IMPORT_CHEATS:,} limit.",
             "",
             "It is corrupt, or is not a hand-made trainer."],
            "Import Failed", C_ERR)
        add_log(f"Import refused: {kind_label} holds {len(mods):,} entries",
                "error")
        return
    attached_process = str(state.get("proc_name", "") or "")
    if file_process and attached_process and file_process != attached_process:
        message_box(stdscr,
            [f"{kind_label} targets process '{file_process}', but RDX is",
             f"attached to '{attached_process}'.",
             "Attach to the matching process before importing."],
            "Import Failed", C_ERR)
        return
    if (file_title_id and state.get("game_id") and
            file_title_id.upper() != str(state["game_id"]).upper()):
        add_log(
            f"{kind_label} Title ID '{file_title_id}' does not match the "
            f"attached game_id '{state['game_id']}' — importing anyway; "
            "addresses are resolved against the live process, not the file.",
            "warn")
    process = str(state.get("proc_name", "") or "eboot.bin")
    try:
        maps = _get_maps_cached(state["ip"], state["pid"])
    except Exception as exc:
        message_box(stdscr,
            [f"Could not read the target process's memory map: {exc}",
             "A live connection is required to resolve static-patch",
             "offsets — connect and attach to the game first."],
            "Import Failed", C_ERR)
        return
    module_base, _accepted = _etahen_main_module(maps, process)
    if module_base is None:
        message_box(stdscr,
            [f"Could not identify {process}'s main module in the",
             "current process — cannot resolve static-patch offsets."],
            "Import Failed", C_ERR)
        return
    imported = _mods_to_import_entries(mods, path, module_base)
    if not imported:
        message_box(stdscr,
            [f"No usable static patches found in this {kind_label} file."],
            "Import Failed", C_ERR)
        return
    if state["cheats"] and not confirm_box(
            stdscr,
            f"Import {len(imported)} cheats from {kind_label} and keep "
            f"existing {len(state['cheats'])}?",
            "Import Cheats"):
        return
    state["cheats"].extend(imported)
    state["cheats_dirty"] = True
    add_log(f"Imported {len(imported)} cheats from {kind_label} {path}")
    message_box(stdscr,
        [f"Imported {len(imported)} cheats as raw-byte static-module",
         "patches, bound to this session at the current live address.",
         "Export once attached to make them portable across reloads."],
        "Import Complete", C_OK)


def _do_import_mc4(stdscr, path: Path, encrypted: bool = True) -> None:
    """Import a .mc4 (encrypted) or .shn (plaintext) trainer.

    Both are the same Trainer/Cheat/Cheatline document; only the container
    differs, so they share mc4_xml_to_mods() and differ by one decode step.
    """
    label = "CheatRunner .mc4" if encrypted else ".shn trainer"
    suffix = ".mc4" if encrypted else ".shn"
    try:
        if encrypted:
            xml_text = _mc4_decrypt(path.read_bytes()).decode("utf-8")
        else:
            # utf-8-sig for the same reason the JSON path uses it: a
            # Windows-authored trainer carries a BOM, and ET.fromstring
            # rejects one with a bare "syntax error" that says nothing.
            xml_text = path.read_text(encoding="utf-8-sig")
        trainer_attrs, mods = mc4_xml_to_mods(xml_text)
    except Exception as exc:
        message_box(stdscr,
            [f"Could not decode {suffix}: {exc}",
             f"It may be corrupt, or not a real {suffix} trainer."],
            "Import Failed", C_ERR)
        return
    if not mods:
        message_box(stdscr,
            [f"No usable <Cheat>/<Cheatline> entries found in this {suffix}."],
            "Import Failed", C_ERR)
        return
    _do_import_static_patch_mods(
        stdscr, path, mods, label, trainer_attrs.get("Cusa", ""),
        trainer_attrs.get("Process", ""))


def do_import(stdscr) -> None:
    stdscr.clear()
    draw_border(stdscr, "IMPORT TRAINER")
    safe_addstr(stdscr, 2, 3,
                "Imports an RDX .rdx.json, an etaHEN/GoldHEN JSON, a .mc4 or a .shn.",
                color(C_NORM))
    raw_path = input_box(
        stdscr, "Trainer path: ", 4, 3, 70,
        state.get("export_dir", str(Path.home())),
        allow_cancel=True, cancel_with_q=False)
    if raw_path is None:
        add_log("Trainer import cancelled")
        return
    path = Path(raw_path).expanduser()
    if not path.exists() or not path.is_file():
        message_box(stdscr, [f"File not found: {path}"], "Import Failed", C_ERR)
        return
    try:
        file_bytes = path.stat().st_size
    except OSError as exc:
        message_box(stdscr, [f"Could not read {path}: {exc}"],
                    "Import Failed", C_ERR)
        return
    if file_bytes > MAX_TRAINER_FILE_BYTES:
        message_box(stdscr,
            [f"{path.name} is {file_bytes / 1048576:.1f} MB.",
             f"The limit is {MAX_TRAINER_FILE_BYTES / 1048576:.0f} MB.",
             "",
             "Real trainers are a few kilobytes. A file this large is",
             "corrupt, or is not a trainer at all."],
            "Import Failed", C_ERR)
        add_log(f"Import refused: {path.name} is {file_bytes:,} bytes", "error")
        return

    if path.suffix.lower() == ".mc4":
        _do_import_mc4(stdscr, path, encrypted=True)
        return
    if path.suffix.lower() == ".shn":
        _do_import_mc4(stdscr, path, encrypted=False)
        return

    # Sniff for the etaHEN/GoldHEN static-patch JSON schema (a top-level
    # "mods" array, no "cheatList") before falling into the native RDX
    # format's own parsing below, which owns .rdx.json's error handling.
    try:
        # utf-8-sig, not utf-8: trainer files are user-supplied and a
        # Windows editor leaves a BOM, which made json.loads fail with a
        # raw "Unexpected UTF-8 BOM" and the whole import die. The codec
        # is identical to utf-8 when no BOM is present.
        sniffed = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        sniffed = None
    if (isinstance(sniffed, dict) and "cheatList" not in sniffed and
            isinstance(sniffed.get("mods"), list)):
        _do_import_static_patch_mods(
            stdscr, path, sniffed["mods"], "etaHEN/GoldHEN JSON",
            str(sniffed.get("id", "")), str(sniffed.get("process", "")))
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        # Shape check before any .get(): a file holding valid JSON that is not
        # an object otherwise surfaces as "'list' object has no attribute
        # 'get'", which tells the user nothing about their file.
        if not isinstance(data, dict):
            raise ValueError(
                "trainer file must contain a JSON object, but this one holds "
                f"a {type(data).__name__}")
        trainer_process = str(data.get("process", "") or "")
        if trainer_process and trainer_process != str(state.get("proc_name", "") or ""):
            raise ValueError(
                f"trainer targets process '{trainer_process}', but RDX is "
                f"attached to '{state.get('proc_name', '')}'")
        items = data.get("cheatList", [])
        if not isinstance(items, list): raise ValueError("cheatList is not an array")
        if len(items) > MAX_IMPORT_CHEATS:
            raise ValueError(
                f"trainer holds {len(items):,} entries, over the "
                f"{MAX_IMPORT_CHEATS:,} limit — it is corrupt, or is not a "
                f"hand-made trainer")
        trainer_identity = str(data.get("game_identity", "") or "")
        identities = {str(c.get("game_identity", trainer_identity) or "")
                      for c in items if isinstance(c, dict)
                      and (c.get("game_identity") or trainer_identity)}
        if identities:
            current_maps = _get_maps_cached(state["ip"], state["pid"])
            current_identity = _pointer_game_identity(
                state.get("proc_name", ""), current_maps)
            if identities != {current_identity}:
                # A mismatch means the addresses are wrong, not that the
                # structure is. Offer whatever chains the file carries before
                # giving up on it.
                salvage = _salvageable_chains(items)
                if salvage and _offer_salvaged_chains(stdscr, path, salvage):
                    return
                raise ValueError(
                    "trainer game-image fingerprint does not match the "
                    "currently attached title")
        imported = []
        for c in items:
            if not isinstance(c, dict) or not c.get("name"): continue
            width = int(c.get("bytes", 4))
            value_type = _normalise_value_type(c.get("value_type"), width)
            if value_type == "bytes":
                if not 1 <= width <= 256:
                    raise ValueError(f"Invalid raw-byte width in '{c['name']}'")
            elif _value_width(value_type) != width:
                raise ValueError(f"Type/width mismatch in '{c['name']}'")
            value = _parse_value_text(str(c.get("value", 0)),
                                      value_type, width)
            e = {"name": str(c["name"]), "type": str(c.get("type", "write")),
                 "value": value, "value_type": value_type, "width": width,
                 "pid": state["pid"],
                 "process": state["proc_name"], "session": state["session"],
                 "imported_from": str(path)}
            item_identity = str(
                c.get("game_identity", trainer_identity) or "")
            if item_identity:
                e["game_identity"] = item_identity
            if c.get("original_value") is not None:
                original = _parse_value_text(str(c["original_value"]),
                                             value_type, width)
                e["original_value"] = original
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
                # v1 release keys are round-trippable. Accept the shorter key
                # names written by late prerelease builds for compatibility.
                raw_module_name = c.get("module_name", c.get("module"))
                raw_module_offset = c.get(
                    "module_relative_offset", c.get("module_offset"))
                if raw_module_name is not None:
                    module_name = str(raw_module_name).strip()
                    if not module_name or len(module_name) > 512:
                        raise ValueError(f"'{e['name']}' has an invalid module name")
                    e["module_name"] = module_name
                if raw_module_offset is not None:
                    mrel = _parse_int_field(raw_module_offset,
                                            "module relative offset")
                    if mrel < 0 or mrel > _ADDR_MAX:
                        raise ValueError(f"'{e['name']}' has an invalid module-relative offset")
                    e["module_relative_offset"] = mrel
                if c.get("terminal_offset") is not None:
                    terminal = _parse_int_field(
                        c["terminal_offset"], "terminal offset")
                    if abs(terminal) > _PTR_RESOLVE_OFFSET_MAX:
                        raise ValueError(
                            f"'{e['name']}' has an unreasonable terminal offset")
                    if terminal:
                        e["terminal_offset"] = terminal
                if e["type"] in {"freeze", "write"}:
                    e["type"] = "pointer_" + e["type"]
                if e["type"] not in {"pointer_freeze", "pointer_write"}:
                    raise ValueError(f"'{e['name']}' has an invalid pointer type")
                if (c.get("cross_reload_validated") is True and
                        e.get("module_name") and
                        e.get("module_relative_offset") is not None and
                        e.get("game_identity")):
                    e["cross_reload_validated"] = True
            elif "address" in c:
                address = _parse_int_field(c["address"], "address")
                if not (_ADDR_MIN <= address <= _ADDR_MAX):
                    raise ValueError(f"'{e['name']}' has an invalid address")
                e["address"] = address
                raw_module_name = c.get("module_name", c.get("module"))
                raw_module_offset = c.get(
                    "module_relative_offset", c.get("module_offset"))
                if ((raw_module_name is None) !=
                        (raw_module_offset is None)):
                    raise ValueError(
                        f"'{e['name']}' has incomplete module-relative metadata")
                if raw_module_name is not None:
                    module_name = str(raw_module_name).strip()
                    mrel = _parse_int_field(
                        raw_module_offset, "module relative offset")
                    if (not module_name or len(module_name) > 512 or
                            mrel < 0 or mrel > _ADDR_MAX):
                        raise ValueError(
                            f"'{e['name']}' has invalid module-relative metadata")
                    e["module_name"] = module_name
                    e["module_relative_offset"] = mrel
                if e["type"] not in {"freeze", "write"}:
                    raise ValueError(f"'{e['name']}' has an invalid scalar type")
            else:
                raise ValueError(f"'{e['name']}' has no address/base")
            # A file cannot prove that an absolute address or provisional
            # pointer belongs to this runtime session. Keep it visible for
            # inspection, but do not stamp it as current and thereby turn an
            # old address into an authorized write target. Portable entries
            # have already passed the process/game-image checks above.
            if not _is_portable_cheat(e):
                e["pid"] = None
                e["session"] = None
                e["import_locked"] = True
            imported.append(e)
        if not imported: raise ValueError("No usable cheats found")
        if state["cheats"] and not confirm_box(stdscr,
                f"Import {len(imported)} cheats and keep existing {len(state['cheats'])}?",
                "Import Cheats"):
            return
        state["cheats"].extend(imported)
        # Imported cheats are unsaved work like any other, so Quit must warn
        # about them; the .mc4/etaHEN import path already sets this.
        state["cheats_dirty"] = True
        add_log(f"Imported {len(imported)} cheats from {path}")
        portable = sum(1 for cheat in imported if _is_portable_cheat(cheat))
        locked = sum(1 for cheat in imported if cheat.get("import_locked"))
        message_box(stdscr, [f"Imported {len(imported)} cheats.",
                             f"Portable entries ready: {portable}.",
                             f"Absolute/provisional entries locked: {locked}."],
                    "Import Complete", C_OK)
    except Exception as exc:
        add_log(f"Import failed: {exc}", "error")
        message_box(stdscr, [f"Import failed: {exc}"], "Import Failed", C_ERR)


def _select_export_cheats(stdscr, cheats: list) -> Optional[list]:
    """Checkbox picker for which of the eligible cheats to include in this
    export. Everything starts selected — most exports want everything, so
    this only costs a keystroke (Enter) in the common case. Returns the
    chosen subset (possibly empty), or None if the user cancels."""
    marked = [True] * len(cheats)
    sel = 0
    offset = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        n_selected = sum(marked)
        draw_border(stdscr,
                    f"SELECT CHEATS TO EXPORT  ({n_selected}/{len(cheats)})")
        visible = max(1, h - 7)
        if sel < offset:
            offset = sel
        if sel >= offset + visible:
            offset = sel - visible + 1
        for i, c in enumerate(cheats[offset:offset + visible]):
            ri = offset + i
            box = "[x]" if marked[ri] else "[ ]"
            attr = color(C_SEL) | curses.A_BOLD if ri == sel else color(C_NORM)
            line = f"{box} {c.get('name', 'Unnamed cheat')}"
            safe_addstr(stdscr, 3 + i, 3, line[:w - 6], attr)
        draw_statusbar(stdscr, [
            ("↑↓ navigate", C_NORM), ("Space toggle", C_ACC),
            ("A all", C_OK), ("N none", C_WARN),
            ("Enter continue", C_OK), ("Esc/Q cancel", C_NORM),
        ])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols()
            continue
        if key == curses.KEY_UP:
            sel = max(0, sel - 1)
        elif key == curses.KEY_DOWN:
            sel = min(len(cheats) - 1, sel + 1)
        elif key == ord(' '):
            marked[sel] = not marked[sel]
        elif key in (ord('a'), ord('A')):
            marked = [True] * len(cheats)
        elif key in (ord('n'), ord('N')):
            marked = [False] * len(cheats)
        elif key in (curses.KEY_ENTER, 10, 13):
            return [c for c, m in zip(cheats, marked) if m]
        elif key in (27, ord('q'), ord('Q')):
            return None


def do_export(stdscr) -> None:
    stdscr.clear()
    draw_border(stdscr, "EXPORT TRAINERS")
    if not state["cheats"]:
        message_box(stdscr,
            ["No cheats to export.", "Build your cheat list first."], "Error", C_ERR)
        return

    # Resolve ownership before collecting metadata.  Asking for four fields and
    # only then discovering that every entry belongs to a stale session or a
    # different title makes Export feel broken and risks packaging the wrong
    # game's entries when several sessions remain in the cheat list.
    process = str(state.get("proc_name", "") or "eboot.bin")
    try:
        maps = _get_maps_cached(state["ip"], state["pid"])
    except Exception as exc:
        maps = []
        add_log(f"Trainer export map lookup failed: {exc}", "warn")
    current_identity = (_pointer_game_identity(process, maps) if maps else "")

    def belongs_to_current_game(cheat):
        current_session = (cheat.get("pid") == state.get("pid") and
                           cheat.get("session") == state.get("session"))
        portable_current = (
            _is_portable_cheat(cheat)
            and str(cheat.get("process", "") or "") in ("", process)
            and str(cheat.get("game_identity", "") or "") == current_identity)
        return current_session or portable_current

    export_cheats = [c for c in state["cheats"] if belongs_to_current_game(c)]
    excluded_count = len(state["cheats"]) - len(export_cheats)
    # Say what is actually being shipped. A raw heap address exports exactly
    # as cleanly as a validated chain and is dead on the next launch; that
    # difference belongs in front of the user at the moment they export, not
    # only on a detail screen they may never open.
    add_log(f"Export durability: {summarise_durability(export_cheats)}")
    # belongs_to_current_game admits a cheat on either arm of an OR, and the
    # two arms mean different things once the file is written: a portable entry
    # survives a reload, a same-session one is a raw address that will point at
    # whatever occupies that memory next time. The predicate has already run;
    # only the answer was being discarded. Counting it is what lets the export
    # screen say which kind of trainer it just produced.
    session_bound = [c for c in export_cheats if not _is_portable_cheat(c)]
    if not export_cheats:
        message_box(
            stdscr,
            ["No cheats belong to the currently attached game/session.",
             "Switch to the owning process or build a new cheat first."],
            "Nothing to Export", C_WARN)
        return
    safe_addstr(stdscr, 2, 3,
        f"Eligible cheats: {len(export_cheats)}", color(C_WARN))
    if excluded_count:
        safe_addstr(stdscr, 3, 3,
            f"Excluded stale/other-game cheats: {excluded_count}", color(C_NORM))
    stdscr.refresh()

    deselected_count = 0
    if len(export_cheats) > 1:
        picked = _select_export_cheats(stdscr, export_cheats)
        if picked is None:
            add_log("Trainer export cancelled")
            return
        if not picked:
            message_box(stdscr,
                ["No cheats selected — nothing to export."],
                "Nothing to Export", C_WARN)
            return
        deselected_count = len(export_cheats) - len(picked)
        export_cheats = picked

    # Require a non-empty Title ID before proceeding
    while True:
        gid = input_box(stdscr, "Title ID  (e.g. PPSA01234) : ", 4, 3, 20,
                        state["game_id"], allow_cancel=True,
                        cancel_with_q=False)
        if gid is None:
            add_log("Trainer export cancelled")
            return
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

    # PS4 package versions normally use 01.00; native PS5 entries in etaHEN's
    # database use the longer 01.004.000 form.
    VERSION_RE = re.compile(r'^(?:\d{2}\.\d{2}|\d{2}\.\d{3}\.\d{3})$')
    gver = input_box(stdscr, "Version (01.00 / 01.004.000): ",
                     6, 3, 16, state["game_ver"], allow_cancel=True,
                     cancel_with_q=False)
    if gver is None:
        add_log("Trainer export cancelled")
        return
    if gver and not VERSION_RE.match(gver):
        if not confirm_box(stdscr,
                f"Version '{gver}' is not a standard PS4/PS5 form.\nContinue anyway?",
                "Version Format"):
            return
    gtit = input_box(
        stdscr, "Game Title                 : ", 8, 3, 40,
        state["game_title"], allow_cancel=True, cancel_with_q=False)
    if gtit is None:
        add_log("Trainer export cancelled")
        return
    author = input_box(stdscr, "Cheat author               : ", 10, 3, 30,
                       "RDX CheatMaker", allow_cancel=True,
                       cancel_with_q=False)
    if author is None:
        add_log("Trainer export cancelled")
        return
    out_raw = input_box(
        stdscr, "Output directory           : ", 12, 3, 60,
        state.get("export_dir", str(Path.home())), allow_cancel=True,
        cancel_with_q=False)
    if out_raw is None:
        add_log("Trainer export cancelled")
        return
    output_dir = Path(out_raw).expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        message_box(stdscr, [f"Not a directory: {output_dir}"],
                    "Export Failed", C_ERR)
        return
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        message_box(stdscr, [f"Cannot create output directory: {exc}"],
                    "Export Failed", C_ERR)
        return
    state.update(game_id=gid, game_ver=gver, game_title=gtit)
    state["export_dir"] = str(output_dir)
    _save_preferences({"export_dir": str(output_dir)})

    safe_gid  = sanitize_filename(gid)
    safe_gver = sanitize_filename(gver)
    base_name = f"{safe_gid or 'UNKNOWN'}_{safe_gver or '00.00'}"
    rdx_name = base_name + ".rdx.json"
    process_suffix = ("" if process.lower() == "eboot.bin" else
                      "_" + sanitize_filename(process))
    etahen_name = base_name + process_suffix + ".json"
    mc4_name = base_name + process_suffix + ".mc4"
    shn_name = base_name + process_suffix + ".shn"
    rdx_path = output_dir / rdx_name
    etahen_path = output_dir / etahen_name
    mc4_path = output_dir / mc4_name
    shn_path = output_dir / shn_name

    # Capture the disable value at creation time whenever possible.  Older
    # in-memory entries predate that field, so make one best-effort live read
    # before export; never invent an etaHEN "off" byte sequence.
    for cheat in export_cheats:
        if cheat.get("offsets") is not None or cheat.get("original_value") is not None:
            continue
        try:
            width = int(cheat["width"])
            raw = ps5_read(state["ip"], state["pid"],
                           _runtime_scalar_address(cheat), width)
            if len(raw) == width:
                cheat["original_value"] = _unpack_typed_value(
                    raw, _cheat_value_type(cheat), width)
        except Exception as exc:
            add_log(f"Could not capture off-value for '{cheat.get('name', '?')}': "
                    f"{exc}", "warn")

    # Upgrade older in-memory flat cheats when their current address is in a
    # static module. This turns a session address into a relocatable native RDX
    # trainer entry and an etaHEN-compatible module patch where possible.
    if maps:
        region_starts, region_rows = _build_region_lookup(maps)
        for cheat in export_cheats:
            if (cheat.get("offsets") is not None or
                    cheat.get("module_name") or
                    cheat.get("pid") != state.get("pid") or
                    cheat.get("session") != state.get("session")):
                continue
            address = int(cheat.get("address", 0))
            region = _region_for_addr(address, region_starts, region_rows)
            if region is None or not _is_static_region(region):
                continue
            module_name, _module_base, module_rel = _module_info_for_addr(
                address, maps)
            if module_name and module_rel is not None:
                cheat["module_name"] = str(module_name)
                cheat["module_relative_offset"] = int(module_rel)
                cheat["game_identity"] = _pointer_game_identity(process, maps)

    rdx_text = generate_cht(
        export_cheats, gid, gver, gtit, True, process)
    etahen_text, etahen_mods, skipped = generate_etahen_json(
        export_cheats, gid, gver, gtit, process, maps, author)
    mc4_bytes = (generate_mc4_bytes(etahen_mods, gid, gver, gtit, process, author)
                if etahen_mods else None)
    # Plaintext twin of the .mc4, same schema and same consumers. Written
    # unconditionally beside it so a CheatRunner rejection can be attributed
    # to the container or the schema — see generate_shn_text().
    shn_text = (generate_shn_text(etahen_mods, gid, gver, gtit, process, author)
                if etahen_mods else None)
    # .mc4 has more than one consumer. CheatRunner reads it from its own
    # directory, but GoldHEN and etaHEN each keep a cheats/mc4/ folder
    # alongside their cheats/json/ one, so naming only CheatRunner's path
    # sent the file somewhere the manager RDX had just selected would never
    # look for it.
    mc4_dirs = ["/data/cheatrunner/cheats/mc4/"]
    shn_dirs = ["/data/cheatrunner/cheats/shn/"]
    if str(gid).upper().startswith("CUSA"):
        platform_name = "GoldHEN"
        deploy_dir = "/user/data/GoldHEN/cheats/json/"
        mc4_dirs.insert(0, "/user/data/GoldHEN/cheats/mc4/")
        shn_dirs.insert(0, "/user/data/GoldHEN/cheats/shn/")
    elif str(gid).upper().startswith("PPSA"):
        platform_name = "etaHEN"
        deploy_dir = "/data/etaHEN/cheats/json/"
        mc4_dirs.insert(0, "/data/etaHEN/cheats/mc4/")
        shn_dirs.insert(0, "/data/etaHEN/cheats/shn/")
    else:
        platform_name = "GoldHEN/etaHEN-compatible"
        deploy_dir = "the console manager's cheats/json directory"

    preflight = [
        f"Native RDX entries: {len(export_cheats)}",
        (f"  ⚠ {len(session_bound)} of these are session-bound and will NOT "
         f"resolve after a reload" if session_bound else
         "  all carry a module root or verified chain — reload-safe"),
        f"{platform_name} static patches: {len(etahen_mods)}",
        f"Static export skipped: {len(skipped)}",
        f"Other-game/stale excluded: {excluded_count}",
    ]
    if deselected_count:
        preflight.append(f"Deselected by you: {deselected_count}")
    preflight.append(f"Destination: {output_dir}")
    if etahen_mods:
        preflight.append(f"Console deploy: {deploy_dir}{etahen_name}")
        preflight.append(
            f".mc4: {len(etahen_mods)} patches -> {mc4_dirs[0]}")
        preflight.append(
            f".shn (plaintext twin): {len(etahen_mods)} patches -> {shn_dirs[0]}")
    if not confirm_box(stdscr, "\n".join(preflight) + "\n\nWrite these files?",
                       "Export Preflight"):
        add_log("Trainer export cancelled at preflight")
        return
    existing = [p.name for p in (rdx_path, etahen_path, mc4_path, shn_path)
                if p.exists() and (p == rdx_path or etahen_mods)]
    if existing and not confirm_box(
            stdscr, "Overwrite existing file(s)?\n" + "\n".join(existing),
            "Confirm Overwrite"):
        return

    try:
        _atomic_write_text(rdx_path, rdx_text)
        # Only clear the unsaved-work flag if every cheat in the list was
        # actually written — excluded stale/other-game entries and cheats
        # the user deselected in the picker were not.
        state["cheats_dirty"] = excluded_count > 0 or deselected_count > 0
        add_log(f"Exported RDX trainer {rdx_path}"
                + (f" — {len(session_bound)}/{len(export_cheats)} entries "
                   f"session-bound (not reload-safe)" if session_bound
                   else " — all entries reload-safe"),
                "warn" if session_bound else "info")
        lines = [f"RDX trainer: {rdx_path}",
                 f"  {len(export_cheats)} entry/entries; pointer chains supported."]
        if session_bound:
            # "pointer chains supported" describes the format, not these
            # entries. Without this the user finds out their trainer was
            # session-bound when they try to use it, not when they wrote it.
            lines.extend([
                f"  ⚠ {len(session_bound)} of {len(export_cheats)} entry/entries "
                f"are session-bound",
                "    raw heap addresses with no module root or verified chain.",
                "    They work now and will not resolve after a game reload.",
                "    Use Pointer Project to promote them to permanent chains.",
            ])
        if excluded_count:
            lines.append(
                f"  Skipped {excluded_count} stale/other-game entry/entries.")
        if etahen_mods:
            _atomic_write_text(etahen_path, etahen_text)
            add_log(f"Exported {platform_name} patch {etahen_path} "
                    f"({len(etahen_mods)} mods, {len(skipped)} skipped)")
            lines.extend([
                "",
                f"{platform_name} patch: {etahen_path}",
                f"  {len(etahen_mods)} static module patch(es).",
                "Upload via FTP to:",
                f"  {deploy_dir}{etahen_name}",
            ])
            if mc4_bytes is not None:
                _atomic_write_bytes(mc4_path, mc4_bytes)
                add_log(f"Exported CheatRunner .mc4 {mc4_path} "
                        f"({len(etahen_mods)} patches)")
                lines.extend([
                    "",
                    f".mc4 trainer: {mc4_path}",
                    f"  {len(etahen_mods)} static module patch(es).",
                    "Upload via FTP to whichever manager you use:",
                ] + [f"  {d}{mc4_name}" for d in mc4_dirs])
            if shn_text is not None:
                _atomic_write_text(shn_path, shn_text)
                add_log(f"Exported .shn {shn_path} "
                        f"({len(etahen_mods)} patches, plaintext twin of the .mc4)")
                lines.extend([
                    "",
                    f".shn trainer: {shn_path}",
                    "  Same patches as the .mc4, unencrypted. Load this one if",
                    "  the .mc4 is rejected — it tells you which layer failed.",
                ] + [f"  {d}{shn_name}" for d in shn_dirs])
        else:
            lines.extend([
                "",
                f"No {platform_name} JSON or .mc4 was written: neither can execute",
                "pointer chains or live freezes; use the RDX trainer for those entries.",
            ])
        if skipped:
            lines.append(
                f"Static-manager-incompatible entries kept in RDX: {len(skipped)}")
        message_box(stdscr, lines, "Export OK", C_OK)
    except Exception as exc:
        message_box(stdscr, [f"Could not write: {exc}"], "Export Failed", C_ERR)


def do_freeze(stdscr, selected_cheat: Optional[dict] = None) -> None:
    """Toggle saved cheats persistently or run an isolated timed manual freeze."""
    stdscr.clear()
    draw_border(stdscr, "FREEZE ADDRESS / CHEAT")
    choices = ["Manual timed freeze"] + (
        ["Saved cheat toggle"] if state["cheats"] else [])
    mode = ("Saved cheat toggle" if selected_cheat is not None else
            cycle_input(stdscr, "Target            : ", 4, 3,
                        choices, choices[0], allow_cancel=True))
    if mode is None:
        add_log("Freeze setup cancelled")
        return

    if mode == "Saved cheat toggle":
        cheat = selected_cheat
        if cheat is None:
            # Number the labels: two cheats can legitimately share a name
            # (only the import path de-duplicates), and a bare name lookup
            # silently resolved every duplicate to the first one.
            names = [f"{i + 1}. {c.get('name', 'Unnamed')}"
                     for i, c in enumerate(state["cheats"])]
            selected = cycle_input(
                stdscr, "Cheat             : ", 6, 3, names, names[0],
                allow_cancel=True)
            if selected is None:
                add_log("Freeze setup cancelled")
                return
            cheat = state["cheats"][names.index(selected)]
        try:
            enabled = _toggle_cheat_freeze(cheat)
        except Exception as exc:
            message_box(stdscr, [f"Could not toggle freeze: {exc}"],
                        "Freeze Failed", C_ERR)
            return
        message_box(
            stdscr,
            [f"'{cheat.get('name', 'Unnamed')}' is now "
             f"{'ON' if enabled else 'OFF'}.",
             "Other enabled cheats remain active.",
             "Use F or Space in Cheat List to toggle it again."],
            "Cheat Toggle", C_OK if enabled else C_WARN)
        return

    addr_s = input_box(stdscr, "Address (hex)    : ", 6, 3, 20,
                       allow_cancel=True)
    if addr_s is None:
        add_log("Freeze setup cancelled")
        return
    labels = [VALUE_TYPES[key]["label"] for key in VALUE_TYPE_ORDER]
    type_label = cycle_input(
        stdscr, "Value type       : ", 8, 3, labels,
        VALUE_TYPES[_current_scan_type()]["label"], allow_cancel=True)
    if type_label is None:
        add_log("Freeze setup cancelled")
        return
    type_key = (VALUE_TYPE_ORDER[labels.index(type_label)]
                if type_label in labels else _normalise_value_type(type_label))
    val_s = input_box(stdscr, "Freeze value     : ", 10, 3, 38,
                      allow_cancel=True)
    sec_s = input_box(stdscr, "Duration (secs)  : ", 12, 3, 6, "30",
                      allow_cancel=True) if val_s is not None else None
    interval_s = input_box(stdscr, "Interval (ms)    : ", 14, 3, 6, "200",
                           allow_cancel=True) if sec_s is not None else None
    if val_s is None or sec_s is None or interval_s is None:
        add_log("Freeze setup cancelled")
        return
    try:
        address = int(addr_s, 0)
        err = _validate_write_addr(address)
        if err:
            raise ValueError(err)
        value = _parse_value_text(val_s, type_key)
        width = (len(bytes.fromhex(value)) if type_key == "bytes" else
                 _value_width(type_key))
        map_error = _validate_addr_in_maps(
            state["ip"], state["pid"], address, width, 0.0)
        if map_error:
            raise ValueError(map_error)
        data = _pack_typed_value(value, type_key, width)
        seconds = max(1, int(sec_s))
        interval = max(50, int(interval_s)) / 1000.0
    except Exception as exc:
        message_box(stdscr, [f"Bad input: {exc}"], "Freeze Failed", C_ERR)
        return

    frozen_endpoint = (state["ip"], state["pid"], state["session"])
    deadline = time.time() + seconds
    stop_event = threading.Event()
    errors = [0]

    def worker_fn():
        while time.time() < deadline and not stop_event.is_set():
            if frozen_endpoint != (state["ip"], state["pid"], state["session"]):
                add_log("Manual freeze stopped because the session changed", "warn")
                break
            # Re-validate the mapping every tick, as the saved-cheat freeze
            # worker already does. Validating once and then writing for the
            # rest of the window meant that if the title unmapped or moved
            # that region mid-freeze, this kept writing into whatever now
            # occupies the address -- the one thing in this tool that can
            # corrupt a running game without the user doing anything wrong.
            # _WRITE_MAP_CACHE_TTL makes the repeat check cheap.
            map_error = _validate_addr_in_maps(
                state["ip"], state["pid"], address, len(data),
                _WRITE_MAP_CACHE_TTL)
            if map_error:
                add_log(f"Manual freeze stopped: {hex(address)} is no longer "
                        f"a valid write target ({map_error})", "error")
                break
            if not ps5_write(state["ip"], state["pid"], address, data,
                             cancel_event=stop_event, timeout=1.0):
                errors[0] += 1
            if stop_event.wait(interval):
                break

    worker = threading.Thread(target=worker_fn, name="rdx-manual-freeze",
                              daemon=True)
    worker.start()
    stdscr.nodelay(True)
    try:
        while worker.is_alive():
            h, w = stdscr.getmaxyx()
            elapsed = seconds - max(0.0, deadline - time.time())
            fraction = min(max(elapsed / seconds, 0.0), 1.0)
            safe_addstr(stdscr, 18, 3,
                        f"Time left: {max(0, int(deadline-time.time())):3d}s",
                        color(C_OK))
            draw_progress_bar(stdscr, 19, 3, min(max(w - 8, 10), 50),
                              fraction, f"  {int(fraction * 100)}%")
            if errors[0]:
                safe_addstr(stdscr, 20, 3, f"Write errors: {errors[0]}",
                            color(C_ERR))
            draw_statusbar(stdscr, [("Esc/Q stop", C_WARN)])
            stdscr.refresh()
            key = stdscr.getch()
            if key == curses.KEY_RESIZE:
                curses.update_lines_cols()
                stdscr.clear()
                draw_border(stdscr, "MANUAL TIMED FREEZE")
            elif key in (27, ord('q'), ord('Q')):
                stop_event.set()
                break
            time.sleep(0.05)
    finally:
        stdscr.nodelay(False)
        stop_event.set()
        worker.join(timeout=interval + 1.0)
    add_log(f"Manual freeze finished with {errors[0]} write error(s)")


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

    base_s = input_box(stdscr, "Static base address (hex) : ", 4, 3, 20,
                       allow_cancel=True)
    if base_s is None:
        add_log("Manual pointer verification cancelled")
        return
    try:
        base = int(base_s, 0)
        if base < _ADDR_MIN or base > _ADDR_MAX:
            raise ValueError("address out of PS5 user-space range")
    except ValueError as exc:
        message_box(stdscr, [f"Invalid address: {exc}"], "Error", C_ERR)
        return

    target_s = input_box(
        stdscr, "Target address (hex, 0=unknown): ", 6, 3, 20, "0",
        allow_cancel=True)
    if target_s is None:
        add_log("Manual pointer verification cancelled")
        return
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
        val_s = input_box(
            stdscr, f"  Offset [{i+1}] : ", 5 + i, 3, 20,
            default_val, allow_cancel=True)
        if val_s is None:
            add_log("Manual pointer verification cancelled")
            return
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


def do_pointer_project(stdscr) -> None:
    """Visible home for the persisted scan → reload → rescan workflow."""
    try:
        maps = _get_maps_cached(state["ip"], state["pid"])
    except Exception:
        maps = []
    summary = _pointer_project_summary(state.get("proc_name", ""), maps)
    state["pointer_project_summary"] = summary
    options = []
    if len(state["scan_results"]):
        options.append("Resume using a scan result")
        options.append("Open current Results")
    options.append("Start a value scan")
    if summary["count"]:
        options.append("Clear this pointer project")
    options.append("Back")

    while True:
        stdscr.clear()
        draw_border(stdscr, "POINTER PROJECT")
        safe_addstr(stdscr, 2, 3,
                    f"Reload validation: {summary['survivals']}/2",
                    color(C_OK) if summary["complete"] else color(C_WARN) |
                    curses.A_BOLD)
        safe_addstr(stdscr, 3, 3,
                    f"Persisted candidates: {summary['count']}", color(C_NORM))
        if summary["target"]:
            safe_addstr(stdscr, 4, 3,
                        f"Previous temporary address: {hex(summary['target'])}",
                        color(C_NORM))
        # Per-reload funnel. Without it the two-reload wait is opaque: the
        # user sees "not yet permanent" and cannot tell a project that is
        # converging from one that is killing every chain it has.
        row = 6
        epoch_rows = _format_epoch_rows(summary.get("epochs", []))
        if epoch_rows:
            safe_addstr(stdscr, row, 3, "SURVIVORS PER RELOAD",
                        color(C_TITLE) | curses.A_BOLD)
            row += 1
            h, w = stdscr.getmaxyx()
            for line in epoch_rows[-4:]:
                if row >= h - 8:
                    break
                safe_addstr(stdscr, row, 3, line[:max(w - 6, 0)],
                            color(C_NORM))
                row += 1
            row += 1
        safe_addstr(stdscr, row, 3,
                    "After each real game reload, find the value's new address,",
                    color(C_ACC))
        safe_addstr(stdscr, row + 1, 3,
                    "then resume with that result. Two survivals unlock saving.",
                    color(C_ACC))
        selected = cycle_input(stdscr, "Action: ", row + 3, 3, options,
                               options[0], allow_cancel=True)
        if selected is None or selected == "Back":
            return
        if selected == "Open current Results":
            do_show_results(stdscr)
            return
        if selected == "Start a value scan":
            do_scan_first(stdscr)
            return
        if selected == "Clear this pointer project":
            if confirm_box(stdscr,
                           "Remove the persisted candidates for this game?",
                           "Clear Pointer Project"):
                removed = _clear_pointer_project(
                    state.get("proc_name", ""), maps)
                # The funnel describes candidates that no longer exist.
                _save_pointer_provisionals(
                    _load_pointer_provisionals(), epochs=[])
                state["pointer_project_summary"] = _pointer_project_summary(
                    state.get("proc_name", ""), maps)
                add_log(f"Cleared {removed} pointer-project candidate(s)",
                        "warn")
            return
        if selected == "Resume using a scan result":
            result_count = len(state["scan_results"])
            idx_s = input_box(
                stdscr, f"Result index (1-{result_count}): ", 12, 3, 12,
                "1", allow_cancel=True)
            if idx_s is None:
                continue
            try:
                idx = int(idx_s) - 1
                if not 0 <= idx < result_count:
                    raise ValueError
            except ValueError:
                message_box(stdscr, ["Invalid result index."],
                            "Pointer Project", C_ERR)
                continue
            do_resolve_permanent(stdscr, int(state["scan_results"][idx]))
            return


def do_resolve_permanent(stdscr, target_addr: int) -> None:
    """Resolve a known temporary address to the best verified permanent chain."""
    target_addr = int(target_addr)
    if not (_ADDR_MIN <= target_addr <= _ADDR_MAX):
        message_box(stdscr, [f"Invalid target address: {hex(target_addr)}"],
                    "Resolve Permanent Address", C_ERR)
        return

    try:
        # A scene reload can remap sections without changing the PID. Bypass
        # the normal 30-second UI cache before deciding section identity/base.
        with _map_cache_lock:
            _map_cache.pop((state["ip"], state["pid"]), None)
        current_maps = _get_maps_cached(state["ip"], state["pid"])
    except Exception as exc:
        message_box(stdscr, [f"Could not read the current memory map: {exc}"],
                    "Resolve Permanent Address", C_ERR)
        return
    game_identity = _pointer_game_identity(
        state.get("proc_name", ""), current_maps)

    # A chain cannot be called permanent from same-session reads.  If a prior
    # search exists and the game PID changed, validate that saved set first.
    all_saved = _load_pointer_provisionals()
    saved = [x for x in all_saved
             if str(x.get("observed_process", "")) in
             ("", str(state.get("proc_name", "") or ""))
             and str(x.get("observed_game", "")) == game_identity]
    reload_result = None
    relocation_detected = any(
            int(x.get("observed_pid", state["pid"])) != int(state["pid"]) or
            int(x.get("observed_target", target_addr)) != target_addr
            for x in saved)
    if saved and not relocation_detected:
        message_box(
            stdscr,
            ["A provisional search already exists for this exact address.",
             "RDX cannot count another same-session scan as validation.",
             "",
             "Reload the game or scene, isolate the moved value again,",
             "then choose Find Permanent Pointer on its new address."],
            "Reload Not Detected", C_WARN)
        return
    if saved and relocation_detected:
        try:
            reload_result = _validate_pointer_provisionals(
                state["ip"], state["pid"], state["proc_name"], target_addr,
                saved, current_maps)
        except Exception as exc:
            add_log(f"Reload pointer validation failed: {exc}", "warn")
            reload_result = {"survivors": [], "rejected": saved}
        add_log(f"Reload validation: {len(reload_result['survivors'])} survived, "
                f"{len(reload_result['rejected'])} rejected")
        epochs = _record_pointer_epoch(reload_result, state.get("proc_name", ""))
        for row in _format_epoch_rows(epochs[-1:]):
            add_log(row)
        # Do not repeatedly reconsider chains that already failed this reload.
        _merge_pointer_provisionals(
            reload_result["survivors"], state.get("proc_name", ""),
            game_identity=game_identity, epochs=epochs)
        state["pointer_project_summary"] = _pointer_project_summary(
            state.get("proc_name", ""), current_maps)

    reload_survivors = (reload_result or {}).get("survivors", [])
    promoted = [c for c in reload_survivors
                if c.get("status") == "permanent"
                and int(c.get("reload_survivals", 0)) >= 2]
    if promoted:
        data = {"candidates": promoted,
                "method": "cross-reload-validation", "index_built": False}
    else:
        data = None

    if reload_survivors and not promoted:
        message_box(
            stdscr,
            [f"{len(reload_survivors)} chain(s) survived the first reload.",
             "",
             "They remain provisional until one more game reload.",
             "Reload, isolate the address again, then choose",
             "Resolve permanent for the final validation."],
            "One More Reload Required", C_WARN)
        return

    cancel_event = threading.Event()
    progress = {"done": 0, "total": _PTR_RESOLVE_MAX_NODES,
                "results": data, "error": None, "built": False}

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

    ok = True
    if data is None:
        ok = _run_scan_with_progress(
            stdscr, run,
            "Finding provisional pointer chains…",
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
             "Confirm the selected address still holds the intended value.",
             "Change it in-game and run Next Scan again if results are broad.",
             "For difficult titles, use all-readable scope in First Scan,",
             "then retry after the object has been created/used in-game."],
            "Permanent Location Not Found", C_WARN)
        return


    if data.get("method") != "cross-reload-validation":
        maps = data.get("maps") or current_maps
        provisional = _make_pointer_provisionals(
            candidates, maps, state["pid"], state["proc_name"], target_addr)
        _merge_pointer_provisionals(
            provisional, state.get("proc_name", ""),
            game_identity=game_identity)
        state["pointer_project_summary"] = _pointer_project_summary(
            state.get("proc_name", ""), current_maps)
        message_box(
            stdscr,
            [f"Saved {len(provisional)} provisional pointer chain(s).",
             "",
             "They are not permanent yet.",
             "Reload the game, isolate the value's new address, then",
             "choose Resolve permanent again to validate automatically."],
            "Reload Validation Required", C_WARN)
        return

    # Keep only the strongest verified choices; the user should choose between
    # a few meaningful alternatives, not inspect raw pointer-search output.
    # The compact chooser renders one primary plus four alternatives. Never
    # leave additional selectable rows off-screen.
    candidates = candidates[:5]
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
        safe_addstr(stdscr, 8, 3,
                    "Verification: ✓ Survived two relocation epochs",
                    color(C_OK))
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
            ("Esc/Q back", C_NORM)])
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
            # These candidates have already survived two relocation epochs,
            # which is the same bar a saved cheat's chain has to clear. If a
            # bookmark is sitting on this address, give it the chain: that is
            # what stops it expiring on the next attach.
            attached = _attach_chain_to_bookmark(target_addr, c2)
            if attached:
                add_log(attached)
            do_pointer_chain_verify(stdscr, c2, target_addr)
        elif key in (ord('q'), ord('Q'), 27):
            return


def do_pointer_scan(stdscr, target_addr: Optional[int] = None,
                    diagnostic: bool = False) -> None:
    """
    Pick a temporary address and start the cross-reload permanent resolver.

    ``diagnostic=True`` retains the same-session candidate browser for
    troubleshooting, but those candidates are explicitly provisional and
    cannot be represented as permanent trainers.
    """
    # ── STEP 1: pick the target address ──────────────────────────────────────
    stdscr.clear()
    draw_border(stdscr, "POINTER SCAN — Step 1 of 3: Pick Address")

    guide = [
        "Find Permanent Pointer follows a value that moves after reloads.",
        "It never treats same-session evidence as a reusable trainer chain.",
        "",
        "How it works:",
        "  1. Pick the temporary address you found via scanning.",
        "  2. RDX finds and records module-rooted pointer candidates.",
        "  3. Reload, isolate the moved value, then run this again.",
        "  4. Two relocation survivals promote a reusable trainer chain.",
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
                    ("Esc/Q cancel", C_NORM),
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
        addr_s = input_box(stdscr, "Temporary address (hex) : ", 5, 3, 24,
                           allow_cancel=True)
        if addr_s is None:
            add_log("Permanent pointer setup cancelled")
            return
        try:
            target_addr = int(addr_s, 0)
            if target_addr < _ADDR_MIN or target_addr > _ADDR_MAX:
                raise ValueError("address out of PS5 user-space range")
        except ValueError as exc:
            message_box(stdscr, [f"Invalid address: {exc}"], "Error", C_ERR)
            return

    # The normal UI has one pointer workflow. The legacy candidate browser is
    # retained only as an explicit diagnostic mode so same-session evidence is
    # never presented to users as a permanent trainer.
    if not diagnostic:
        return do_resolve_permanent(stdscr, target_addr)

    # ── STEP 2: scan ─────────────────────────────────────────────────────────
    # Five levels cover common module -> manager -> object -> component ->
    # value layouts. The permanent resolver can exhaustively fall back through
    # six levels and the scanner API supports up to MAX_CHAIN_DEPTH.
    max_depth    = int(setting("ptr_max_depth"))
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
    for candidate in candidates:
        candidate["status"] = "provisional"
    if not candidates:
        message_box(stdscr,
            ["No pointer candidates found.",
             "",
             "Tips:",
             "  • Use Resolve permanent for the exhaustive fallback.",
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
                    "Same-session diagnostic only; Enter inspects the chain.",
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
                ("Esc/Q back", C_NORM),
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
    chain_edited = False
    # Use a one-element list so closures always see the current offsets list
    # even after reassignment in the E branch (closures capture the cell, not
    # the value; a mutable container avoids the stale-reference problem).
    _offsets = [list(candidate["offsets"])]
    region   = candidate["region"]

    safe_addstr(stdscr, 2, 3,
        f"Base address : {hex(base)}  [{region}]", color(C_OK) | curses.A_BOLD)
    if terminal_offset:
        safe_addstr(stdscr, 3, 3,
                    f"Terminal field offset: {terminal_offset:+#x}", color(C_ACC))

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
            label = f"[{i+1}]  {int(off):+#x}"
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
            target_known = _ADDR_MIN <= int(original_target) <= _ADDR_MAX
            match = (not target_known or final_addr == original_target)
            status_color = C_OK if match else C_WARN
            match_text = ("✓ MATCHES target" if target_known and match else
                          "target not supplied" if not target_known else
                          "✗ differs from original target")
            safe_addstr(stdscr, status_row, 3,
                f"→ Resolves to: {hex(final_addr)}"
                f"  {match_text}",
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
            "[E] Edit offsets   [T] Test again   [S] Save as cheat   [Esc/Q] Back",
            color(C_NORM))
        draw_statusbar(stdscr, [
            ("E edit offsets", C_WARN),
            ("T re-test",      C_OK),
            ("S save cheat",   C_OK),
            ("Esc/Q back",         C_NORM),
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
                val_s = input_box(
                    stdscr, f"  Offset [{i+1}] : ", row + i, 3, 20,
                    cur, allow_cancel=True)
                if val_s is None:
                    new_offsets = []
                    break
                if not val_s:
                    break
                try:
                    new_offsets.append(int(val_s, 0))
                except ValueError:
                    message_box(stdscr, [f"Invalid offset: {val_s!r}"], "Error", C_ERR)
                    break
            if new_offsets:
                _offsets[0] = new_offsets   # update shared container; closures see it
                chain_edited = True

        elif key in (ord('t'), ord('T')):
            # Re-test (loop redraws automatically)
            continue

        elif key in (ord('s'), ord('S')):
            # Save as pointer cheat
            if candidate.get("status") == "provisional":
                message_box(
                    stdscr,
                    ["This chain is provisional and cannot be saved yet.",
                     "Reload the game, isolate the address again, and use",
                     "Resolve permanent to perform cross-reload validation."],
                    "Reload Validation Required", C_WARN)
                continue
            save_ok, save_addr, _ = _test_chain()
            target_known = _ADDR_MIN <= int(original_target) <= _ADDR_MAX
            if not save_ok or (target_known and save_addr != int(original_target)):
                message_box(
                    stdscr,
                    ["This chain is broken or no longer resolves to the target.",
                     "Re-test it or return to Find Permanent Pointer."],
                    "Cannot Save Chain", C_ERR)
                continue
            offsets = _offsets[0]   # re-alias after any edits
            stdscr.clear()
            draw_border(stdscr, "SAVE POINTER CHEAT")
            safe_addstr(stdscr, 2, 3,
                f"Chain: {hex(base)} + [{'+'.join(hex(o) for o in offsets)}]",
                color(C_OK))
            stdscr.refresh()

            name = input_box(
                stdscr, "Cheat name   : ", 4, 3, 40,
                allow_cancel=True, cancel_with_q=False)
            if name is None:
                add_log("Save pointer cheat cancelled")
                continue
            type_labels = [VALUE_TYPES[key]["label"]
                           for key in VALUE_TYPE_ORDER]
            type_label = cycle_input(
                stdscr, "Value type   : ", 6, 3, type_labels,
                VALUE_TYPES[_current_scan_type()]["label"],
                allow_cancel=True)
            if type_label is None:
                add_log("Save pointer cheat cancelled")
                continue
            value_type = VALUE_TYPE_ORDER[type_labels.index(type_label)]
            val_s = input_box(stdscr, "Lock-in value: ", 8, 3, 38,
                              allow_cancel=True)
            if val_s is None:
                add_log("Save pointer cheat cancelled")
                continue
            typ   = cycle_input(stdscr, "Type         : ", 10, 3,
                                ["pointer_freeze", "pointer_write"],
                                "pointer_freeze", allow_cancel=True)
            if typ is None:
                add_log("Save pointer cheat cancelled")
                continue
            try:
                val = _parse_value_text(val_s, value_type)
                width = (len(bytes.fromhex(val)) if value_type == "bytes"
                         else _value_width(value_type))
            except ValueError as exc:
                message_box(stdscr, [f"Invalid value: {exc}"], "Error", C_ERR)
                continue

            original_value = None
            try:
                original_raw = ps5_read(
                    state["ip"], state["pid"], save_addr, width)
                if len(original_raw) == width:
                    original_value = _unpack_typed_value(
                        original_raw, value_type, width)
            except Exception as exc:
                add_log(f"Could not capture pointer cheat off-value: {exc}",
                        "warn")

            entry = {
                "name":    name or f"PTR@{hex(base)}",
                "type":    typ,
                "base":    base,
                "offsets": list(offsets),
                **({"terminal_offset": terminal_offset} if terminal_offset else {}),
                "value":   val,
                "value_type": value_type,
                "width":   width,
                **({"original_value": original_value}
                   if original_value is not None else {}),
                "pid":     state["pid"],
                "process": state["proc_name"],
                "session": state["session"],
                "cross_reload_validated": bool(
                    candidate.get("status") == "permanent"
                    and not chain_edited
                    and candidate.get("module_name")
                    and candidate.get("module_relative_offset") is not None),
                "game_identity": str(candidate.get("observed_game", "") or ""),
                # Smart resolver metadata: survives ASLR/module relocation.
                **({"module_name": candidate["module_name"],
                    "module_relative_offset": int(candidate["module_relative_offset"])}
                   if candidate.get("module_name") is not None else {}),
                # For display compatibility with non-pointer cheats:
                "address": 0,    # resolved at apply time
            }
            state["cheats"].append(entry)
            state["cheats_dirty"] = True
            add_log(f"Added pointer cheat '{entry['name']}' "
                    f"base={hex(base)} offsets={[hex(o) for o in offsets]} val={val}")
            if entry["cross_reload_validated"]:
                try:
                    current_maps = _get_maps_cached(state["ip"], state["pid"])
                    _clear_pointer_project(state.get("proc_name", ""),
                                           current_maps)
                    state["pointer_project_summary"] = (
                        _pointer_project_summary(
                            state.get("proc_name", ""), current_maps))
                except Exception as exc:
                    add_log(f"Could not clear completed pointer project: {exc}",
                            "warn")
                lifecycle = [
                    "Cross-reload validation retained.",
                    "RDX will rebase the module and resolve it after reconnects.",
                ]
            else:
                lifecycle = [
                    "Saved for this connection session only.",
                    "Use Find Permanent Pointer to validate it across reloads.",
                ]
            message_box(stdscr,
                [f"  {entry['name']}",
                 f"  base={hex(base)}",
                 f"  offsets=[{'+'.join(hex(o) for o in offsets)}]",
                 f"  value={val}  [{typ}]", ""] + lifecycle,
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
        _clear_scan_state(stop_freezes=False)
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
            ("↑↓/PgUp/PgDn", C_NORM), ("S save", C_OK), ("Esc/Q back", C_NORM),
        ])
        stdscr.refresh()
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord('k')) and offset > 0: offset -= 1
        elif key in (curses.KEY_DOWN, ord('j')) and offset < max(0, len(snap)-1): offset += 1
        elif key == ord('g'): offset = 0
        elif key == ord('G'): offset = max(0, len(snap) - visible)
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
    _safe_curs_set(0)
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
    too_small_drawn = None      # last size the warning was drawn at
    while True:
        # Issues #1/#3: handle resize at the top level so every screen
        # automatically gets a full redraw after the user resizes the terminal.
        h, w = stdscr.getmaxyx()
        if h < _MIN_ROWS or w < _MIN_COLS:
            # getch() returns -1 every 100 ms (stdscr.timeout), so redrawing
            # unconditionally clears and repaints a static message ten times
            # a second: visible flicker and constant CPU for as long as the
            # terminal stays small. Only repaint when the size actually
            # changes.
            if too_small_drawn != (h, w):
                too_small_drawn = (h, w)
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
        too_small_drawn = None

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
                    "Make sure ps5debug-NG or MemDBG is running on your console,",
                    "then verify the console IP address.",
                ], "Connection Error", C_ERR)
                screen = "connect"
        else:
            break


def _install_signal_teardown() -> None:
    """Tear the debugger down on SIGTERM/SIGHUP as well as on normal exit.

    atexit does not run for a signal-terminated process, and a closed terminal
    (SIGHUP) is a realistic way to lose this program mid-trace. SIGKILL cannot
    be caught, which is why _debug_force_resume() exists as a manual escape.
    """
    import signal

    def _handler(signum, _frame):
        _emergency_debug_teardown()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass          # not the main thread, or unsupported on this OS


if __name__ == '__main__':
    # Before curses takes the terminal, so nothing can be written to a
    # display the user cannot read.
    install_warning_router()
    _install_signal_teardown()
    _swept = _sweep_orphaned_disk_indexes()
    if _swept:
        add_log(f"Removed {_swept} orphaned pointer-index director"
                f"{'y' if _swept == 1 else 'ies'} from a previous run")
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    finally:
        # Order matters: drop the debugger first. A leaked session can leave
        # the game SIGSTOPped with a live hardware watchpoint, which is the
        # one failure in this tool that takes the console down with it.
        _emergency_debug_teardown()
        _stop_freeze_worker()
        _close_turbo_session()
    print("\nRDX CheatMaker exited.")
