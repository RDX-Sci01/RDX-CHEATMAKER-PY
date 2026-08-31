#!/usr/bin/env python3
"""
A ps5debug-NG protocol server, so a hardware session can be rehearsed offline.

Why this exists
---------------
`test_pointer_subsystem.py` mocks at the *function* level and `ui_smoke.py`
replaces `scan_first` outright with a stub, using a socket object whose
`sendall` does nothing. Both are the right tools for what they cover, and
neither has ever executed the wire layer: `_ScanSocket` and its pool, the
`recv_exact` framing, the resident TurboScan session lifecycle,
`ps5_read_batch`, Type Scan's streaming reads, or the watchpoint DR
diagnostic. Every one of those runs for the first time when the tool meets a
real console -- which is the one setting where a defect costs a crashed game
and a reload rather than a red test.

This serves the real protocol over a real TCP socket against a simulated
process, so those paths run end to end with nothing stubbed.

It is a **test double, not an emulator**. It implements the request/response
shapes RDX actually sends, taken from the client in
`RDX-CHEATMAKER-UI-patch103.py` and the reference in
`info/ps5debug-NG_PROTOCOL.md`. Where the real payload's behaviour is unknown
-- most importantly whether debug registers are applied per-thread -- this
makes it a *setting* rather than guessing, so the diagnostic that has to
decide that question can be exercised against every answer before the single
expensive attach that will decide it for real.

Usage
-----
    from fake_console import FakeConsole
    with FakeConsole() as con:            # binds 127.0.0.1 on a free port
        ...                               # point RDX at con.host / con.port

    # Model a payload that applies DRs to only the thread that armed them:
    with FakeConsole(dr_mode="first-thread-only") as con:
        ...
"""

import socket
import struct
import threading

# ── protocol constants (mirrored from the client) ────────────────────────────
CMD_MAGIC = 0xFFAABBCC
STATUS_SUCCESS = 0x80000000
STATUS_ERROR = 0xF0000001

CMD_PROC_LIST = 0xBDAA0001
CMD_PROC_READ = 0xBDAA0002
CMD_PROC_WRITE = 0xBDAA0003
CMD_PROC_MAPS = 0xBDAA0004
CMD_PROC_AUTH = 0xBDAACCFF
CMD_TURBO_CAPS = 0xBDAACC10
CMD_REGION_CLASSIFY = 0xBDAACC16
CMD_PROC_WRITE_MULTI = 0xBDAACC04

CMD_DEBUG_ATTACH = 0xBDBB0001
CMD_DEBUG_DETACH = 0xBDBB0002
CMD_DEBUG_SET_WATCHPOINT = 0xBDBB0004
CMD_DEBUG_GET_THREAD_LIST = 0xBDBB0005
CMD_DEBUG_GETDBREGS = 0xBDBB000C
CMD_DEBUG_CONTINUE = 0xBDBB0010

PROC_ENTRY_SIZE = 36
MAP_ENTRY_SIZE = 58

# The client XORs the challenge with this LFSR keystream; reproduced from
# ps5debug-NG's auth.c (seeded 200/300/400/500) so the handshake is genuinely
# exercised rather than waved through -- a client-side keystream regression
# must fail here, at the handshake, not silently later.
def auth_keystream(length: int) -> bytes:
    s1, s2, s3, s4 = 200, 300, 400, 500
    out = bytearray(length)
    mask = 0xFFFFFFFF
    for i in range(length):
        s1 = ((s1 << 18) & 0xFFF80000) ^ ((s1 ^ ((s1 << 6) & mask)) >> 13)
        s2 = ((s2 << 2) & 0xFFFFFFE0) ^ ((s2 ^ ((s2 << 2) & mask)) >> 27)
        s3 = ((s3 << 7) & 0xFFFFF800) ^ ((s3 ^ ((s3 << 13) & mask)) >> 21)
        s4 = ((s4 << 13) & 0xFFF00000) ^ ((s4 ^ ((s4 << 3) & mask)) >> 12)
        s1 &= mask; s2 &= mask; s3 &= mask; s4 &= mask
        out[i] = (s1 ^ s2 ^ s3 ^ s4) & 0xFF
    return bytes(out)


def _default_maps():
    """A layout shaped like the validated title: a module, then heap."""
    return [
        {"name": "executable", "start": 0x400000, "end": 0x480000, "prot": 5},
        {"name": "executable", "start": 0x480000, "end": 0x500000, "prot": 3},
        {"name": "libkernel.sprx", "start": 0x900000, "end": 0x910000, "prot": 5},
        # Two ADJACENT heap mappings: the coalescing path and the AOB
        # cross-boundary case both depend on this shape existing.
        {"name": "", "start": 0x2000000, "end": 0x2200000, "prot": 3},
        {"name": "", "start": 0x2200000, "end": 0x2400000, "prot": 3},
    ]


class FakeConsole:
    """Serves the ps5debug-NG protocol subset RDX uses, over real sockets."""

    def __init__(self, host="127.0.0.1", port=0, maps=None, procs=None,
                 dr_mode="all-threads", thread_count=40, turbo=False,
                 memory=None):
        """
        dr_mode selects how the simulated payload applies debug registers.
        It is the whole reason this file can settle a question the checklist
        currently cannot:

          "all-threads"       DR7 set on every thread (arming works globally)
          "first-thread-only" DR7 only on the lowest lwpid  (hypothesis 3)
          "none"              the arm is acknowledged and discarded
        """
        self.host = host
        self._requested_port = port
        self.maps = list(maps if maps is not None else _default_maps())
        self.procs = list(procs if procs is not None else
                          [{"pid": 91, "name": "eboot.bin"},
                           {"pid": 3, "name": "SceShellCore"}])
        self.dr_mode = dr_mode
        self.thread_count = int(thread_count)
        self.turbo = bool(turbo)
        # Sparse backing store: {address: byte}. Unwritten memory reads zero,
        # which matches an untouched page well enough for these paths.
        self.memory = dict(memory or {})
        self.writes = []          # every accepted write, for assertions
        self.commands = []        # every command served, for assertions
        self._watchpoints = {}    # index -> (address, enabled)
        self._lock = threading.Lock()
        self._sock = None
        self._thread = None
        self._stop = threading.Event()

    # ── lifecycle ──
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.stop()
        return False

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self._requested_port))
        self.port = self._sock.getsockname()[1]
        self._sock.listen(16)
        self._sock.settimeout(0.25)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True,
                                        name="fake-console")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True,
                             name="fake-console-conn").start()

    # ── helpers ──
    @staticmethod
    def _recv_exact(conn, n):
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("client closed")
            buf += chunk
        return buf

    @staticmethod
    def _ok(conn):
        conn.sendall(struct.pack("<I", STATUS_SUCCESS))

    @staticmethod
    def _err(conn):
        conn.sendall(struct.pack("<I", STATUS_ERROR))

    def _read_memory(self, addr, length):
        mem = self.memory
        return bytes(mem.get(addr + i, 0) for i in range(length))

    def _addr_is_mapped(self, addr, length):
        """True when every byte of the range is backed by some mapping.

        Deliberately tests the *union* of mappings, not one region: RDX
        coalesces adjacent mappings and issues reads that span them, which a
        real console serves without complaint because the pages are
        contiguous. Requiring a single containing region made this double
        stricter than the thing it stands in for, and turned every
        cross-boundary read into a rejection -- which is precisely the case
        the coalescing work exists to exercise.
        """
        if length <= 0:
            return False
        merged = []
        for start, end in sorted((int(r["start"]), int(r["end"]))
                                 for r in self.maps):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return any(lo <= addr and addr + length <= hi for lo, hi in merged)

    # ── connection handler ──
    def _serve(self, conn):
        conn.settimeout(10.0)
        try:
            while not self._stop.is_set():
                header = self._recv_exact(conn, 12)
                magic, cmd, datalen = struct.unpack("<III", header)
                if magic != CMD_MAGIC:
                    self._err(conn)
                    return
                body = self._recv_exact(conn, datalen) if datalen else b""
                with self._lock:
                    self.commands.append(cmd)
                if not self._dispatch(conn, cmd, body):
                    return
        except (ConnectionError, socket.timeout, OSError, struct.error):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _dispatch(self, conn, cmd, body):
        """Return False to close the connection."""
        if cmd == CMD_PROC_LIST:
            self._ok(conn)
            conn.sendall(struct.pack("<I", len(self.procs)))
            for p in self.procs:
                entry = p["name"].encode()[:31].ljust(32, b"\x00")
                entry += struct.pack("<i", int(p["pid"]))
                conn.sendall(entry.ljust(PROC_ENTRY_SIZE, b"\x00"))
            return True

        if cmd == CMD_PROC_MAPS:
            self._ok(conn)
            conn.sendall(struct.pack("<I", len(self.maps)))
            for r in self.maps:
                entry = str(r.get("name", "")).encode()[:31].ljust(32, b"\x00")
                entry += struct.pack("<QQQH", int(r["start"]), int(r["end"]),
                                     int(r.get("offset", 0)),
                                     int(r.get("prot", 3)))
                conn.sendall(entry[:MAP_ENTRY_SIZE])
            return True

        if cmd == CMD_PROC_READ:
            _pid, addr, length = struct.unpack("<IQI", body)
            if not self._addr_is_mapped(addr, length):
                self._err(conn)
                return True
            self._ok(conn)
            conn.sendall(self._read_memory(addr, length))
            return True

        if cmd == CMD_PROC_WRITE:
            _pid, addr, length = struct.unpack("<IQI", body)
            if not self._addr_is_mapped(addr, length):
                self._err(conn)
                return True
            self._ok(conn)
            data = self._recv_exact(conn, length)
            with self._lock:
                for i, b in enumerate(data):
                    self.memory[addr + i] = b
                self.writes.append((addr, bytes(data)))
            self._ok(conn)
            return True

        if cmd == CMD_PROC_WRITE_MULTI:
            _pid, count, flags = struct.unpack("<III", body)
            self._ok(conn)
            statuses = bytearray()
            for _ in range(count):
                # Entries are {uint64 address; uint32 length; length bytes}
                # CONCATENATED (protocol reference 0xBDAACC04) -- header and
                # data interleave per entry. Reading the headers as one block
                # up front desynchronises the stream from the second entry on.
                header = self._recv_exact(conn, 12)
                addr, length = struct.unpack("<QI", header)
                data = self._recv_exact(conn, length)
                if self._addr_is_mapped(addr, length):
                    with self._lock:
                        for j, b in enumerate(data):
                            self.memory[addr + j] = b
                        self.writes.append((addr, bytes(data)))
                    statuses.append(0)
                else:
                    statuses.append(1)
            # The status array is sent only when the client asked for it.
            if flags & 0x1:
                conn.sendall(bytes(statuses))
            self._ok(conn)
            return True

        if cmd == CMD_PROC_AUTH:
            self._ok(conn)
            challenge = bytes(range(1, 33))
            conn.sendall(struct.pack("<H", len(challenge)) + challenge)
            response = self._recv_exact(conn, len(challenge))
            expected = bytes(a ^ b for a, b in
                             zip(challenge, auth_keystream(len(challenge))))
            # Genuinely checked: a broken keystream must fail here, not later.
            (self._ok if response == expected else self._err)(conn)
            return True

        if cmd == CMD_TURBO_CAPS:
            if not self.turbo:
                self._err(conn)          # forces the host scan path
                return True
            self._ok(conn)
            conn.sendall(struct.pack("<IIII", 1, 0x03FF, 4, 0))
            return True

        if cmd == CMD_REGION_CLASSIFY:
            self._err(conn)              # no classifier; client falls back
            return True

        return self._dispatch_debug(conn, cmd, body)

    # ── debugger ──
    def _dispatch_debug(self, conn, cmd, body):
        if cmd == CMD_DEBUG_ATTACH:
            self._ok(conn)
            return True

        if cmd == CMD_DEBUG_DETACH:
            with self._lock:
                self._watchpoints.clear()
            self._ok(conn)
            return True

        if cmd == CMD_DEBUG_CONTINUE:
            self._ok(conn)
            return True

        if cmd == CMD_DEBUG_GET_THREAD_LIST:
            self._ok(conn)
            conn.sendall(struct.pack("<I", self.thread_count))
            for i in range(self.thread_count):
                conn.sendall(struct.pack("<I", 100 + i))
            return True

        if cmd == CMD_DEBUG_SET_WATCHPOINT:
            index, enabled, _length, _breaktype, address = \
                struct.unpack("<IIIIQ", body)
            if index > 3:
                self._err(conn)          # CMD_INVALID_INDEX
                return True
            with self._lock:
                if enabled:
                    self._watchpoints[index] = address
                else:
                    self._watchpoints.pop(index, None)
            self._ok(conn)
            return True

        if cmd == CMD_DEBUG_GETDBREGS:
            lwpid = struct.unpack("<I", body)[0]
            self._ok(conn)
            conn.sendall(self._dbregs_for(lwpid))
            return True

        self._err(conn)
        return True

    def _dbregs_for(self, lwpid):
        """The 128-byte dbreg blob this thread would report.

        dr[0..3] hold the watchpoint addresses and dr[7] the enable bits, which
        is the layout the client decodes. `dr_mode` decides which threads see
        them -- the question the hardware diagnostic exists to answer.
        """
        regs = [0] * 16
        with self._lock:
            watchpoints = dict(self._watchpoints)
        show = (self.dr_mode == "all-threads"
                or (self.dr_mode == "first-thread-only" and lwpid == 100))
        if show:
            dr7 = 0
            for index, address in watchpoints.items():
                regs[index] = int(address)
                dr7 |= 1 << (2 * index)      # local-enable bit for this slot
            regs[7] = dr7
        return struct.pack("<16Q", *regs)


def seed_type_pointers(console, base=0x2000000, count=64, stride=0x40,
                       type_ptr=0x480280):
    """Write a type pointer at the base of `count` synthetic objects.

    Gives Type Scan something real to find over the wire: every object begins
    with the same pointer to a class, which is the signature the scan is
    built around.

    The default points into the module's *data* mapping (0x480000, prot 3),
    not its executable one. It used to be 0x400280, inside prot=5 — a class
    living in code, which no real title does. That fixture is part of why the
    old static-only target filter looked correct offline while excluding
    every real Il2CppClass on hardware.
    """
    for i in range(count):
        addr = base + i * stride
        for j, b in enumerate(int(type_ptr).to_bytes(8, "little")):
            console.memory[addr + j] = b
    return type_ptr


def seed_value(console, addr, value, width=4, signed=False):
    """Place one integer in the simulated address space."""
    raw = int(value).to_bytes(width, "little", signed=signed)
    for i, b in enumerate(raw):
        console.memory[addr + i] = b
    return addr
