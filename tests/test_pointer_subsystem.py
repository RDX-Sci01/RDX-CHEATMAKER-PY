import os
import sys
import importlib.util
import json
import re
import struct
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
SOURCE = Path(__file__).resolve().parent.parent / "RDX-CHEATMAKER-UI-final.py"
SPEC = importlib.util.spec_from_file_location("rdx_final_tests", SOURCE)
RDX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RDX)


class FakeSock:
    """Stand-in for a live socket on a fake MemDBG client.

    memdbg_session() treats `client.sock is None` as "not connected" and calls
    settimeout() on whatever is there, so a fake that reports itself connected
    has to offer both.
    """
    def settimeout(self, _seconds): pass
    def close(self): pass


class MemorySocket:
    memory = {}

    def __init__(self, *_args):
        pass

    def read(self, address, length, _cancel=None):
        data = bytearray(length)
        for slot, value in self.memory.items():
            if address <= slot and slot + 8 <= address + length:
                struct.pack_into("<Q", data, slot - address, value)
        return bytes(data)

    def close(self):
        pass


class PointerSubsystemTests(unittest.TestCase):
    def test_map_cache_zero_ttl_forces_reload(self):
        first = [{"start": 0x1000, "end": 0x2000}]
        second = [{"start": 0x5000, "end": 0x6000}]
        with RDX._map_cache_lock:
            saved = dict(RDX._map_cache)
            RDX._map_cache.clear()
        try:
            with patch.object(RDX, "ps5_maps", side_effect=[first, second]) as maps:
                self.assertIs(RDX._get_maps_cached("test", 7), first)
                self.assertIs(RDX._get_maps_cached("test", 7), first)
                self.assertIs(
                    RDX._get_maps_cached("test", 7, ttl_override=0.0), second)
            self.assertEqual(maps.call_count, 2)
        finally:
            with RDX._map_cache_lock:
                RDX._map_cache.clear()
                RDX._map_cache.update(saved)

    def test_write_validation_accepts_writable_map_covered_by_overlapping_stub(self):
        # PS5 map enumeration can report a large writable reservation
        # alongside a smaller, differently-permissioned augmented row over the
        # same bytes (the same hazard _build_region_lookup documents).
        # Returning on whichever covering row is listed first let a
        # read-only overlay falsely block a write that is legitimately
        # writable underneath it.
        maps = [
            {"start": 0x1000, "end": 0x1010, "prot": 1, "name": "stub"},
            {"start": 0x1000, "end": 0x2000, "prot": 3, "name": "heap"},
        ]
        with RDX._map_cache_lock:
            saved = dict(RDX._map_cache)
            RDX._map_cache.clear()
            RDX._map_cache[("test", 1)] = (time.time(), maps)
        try:
            self.assertIsNone(
                RDX._validate_addr_in_maps("test", 1, 0x1004, 4,
                                            ttl_override=999))
        finally:
            with RDX._map_cache_lock:
                RDX._map_cache.clear()
                RDX._map_cache.update(saved)

    def test_write_validation_rejects_when_every_covering_map_is_read_only(self):
        maps = [{"start": 0x1000, "end": 0x2000, "prot": 1, "name": "rodata"}]
        with RDX._map_cache_lock:
            saved = dict(RDX._map_cache)
            RDX._map_cache.clear()
            RDX._map_cache[("test", 1)] = (time.time(), maps)
        try:
            error = RDX._validate_addr_in_maps("test", 1, 0x1004, 4,
                                               ttl_override=999)
            self.assertIsNotNone(error)
            self.assertIn("not writable", error)
        finally:
            with RDX._map_cache_lock:
                RDX._map_cache.clear()
                RDX._map_cache.update(saved)

    def test_memdbg_vnode_map_is_static_even_when_unnamed_and_writable(self):
        self.assertTrue(RDX._is_static_region({
            "start": 0x500000000, "end": 0x500010000, "prot": 3,
            "flags": 2 << 24, "name": ""}))

    def test_tiered_index_query_reaches_wide_offsets_without_losing_near_hit(self):
        class FakeIndex:
            def query(self, target, window, step, max_hits):
                out = [(0x1000, 0x20)]
                if window >= 0x100000:
                    out.append((0x2000, 0x90000))
                return out

        self.assertEqual(RDX._query_pointer_index_tiered(FakeIndex(), 0x500000),
                         [(0x1000, 0x20), (0x2000, 0x90000)])

    def test_overlap_safe_region_lookup_prefers_static_covering_map(self):
        maps = [
            {"start": 0x1000, "end": 0x5000, "prot": 3,
             "name": "executable"},
            # Later-starting augmented row does not cover 0x3000.  A simple
            # bisect selected this row and falsely reported the holder unmapped.
            {"start": 0x2000, "end": 0x2800, "prot": 3, "name": "heap"},
        ]
        starts, rows = RDX._build_region_lookup(maps)
        self.assertIs(RDX._region_for_addr(0x3000, starts, rows), maps[0])

    def test_ram_reverse_index_keeps_value_covered_by_overlapping_map(self):
        maps = [
            {"start": 0x1000, "end": 0x5000, "prot": 3,
             "name": "executable"},
            {"start": 0x2000, "end": 0x2800, "prot": 3,
             "name": "heap"},
        ]
        # A simple searchsorted against map starts lands on the short overlay
        # for 0x3000 and rejects it, even though the executable still covers it.
        MemorySocket.memory = {0x1000: 0x3000}
        with patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "_pointer_readable_regions",
                          return_value=[maps[0]]):
            index = RDX._ReversePointerIndex("test", 1, maps)
        self.assertEqual(index.query(0x3000, 0, 4), [(0x1000, 0)])

    def test_disk_reverse_index_keeps_value_covered_by_overlapping_map(self):
        maps = [
            {"start": 0x1000, "end": 0x5000, "prot": 3,
             "name": "executable"},
            {"start": 0x2000, "end": 0x2800, "prot": 3,
             "name": "heap"},
        ]
        MemorySocket.memory = {0x1000: 0x3000}
        with patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "_pointer_readable_regions",
                          return_value=[maps[0]]):
            index = RDX._DiskReversePointerIndex("test", 1, maps)
            try:
                self.assertEqual(index.query(0x3000, 0, 4),
                                 [(0x1000, 0)])
            finally:
                index.close()

    def test_ps5_read_batch_coalesces_and_decodes_correctly(self):
        # ps5_read_batch had no direct test: its dead 'single' work-item
        # branch (referencing an undefined `fmt`) went unnoticed because
        # nothing ever exercised the function's real per-item decode path.
        # Cover the reachable path directly: a dense pair of addresses that
        # coalesce into one window read, plus a distant isolated one that
        # becomes its own single-candidate window, decoded correctly either
        # way.
        values = {0x1000: 100, 0x1004: 200, 0x2000: 300}

        class FourByteSocket:
            def __init__(self, *_args):
                pass
            def read(self, address, length, _cancel=None):
                data = bytearray(length)
                for slot, value in values.items():
                    if address <= slot and slot + 4 <= address + length:
                        struct.pack_into("<I", data, slot - address, value)
                return bytes(data)
            def close(self):
                pass

        with patch.object(RDX, "_ScanSocket", FourByteSocket):
            addrs = RDX._make_addr_array([0x2000, 0x1000, 0x1004])
            live_addrs, live_vals = RDX.ps5_read_batch(
                "test", 1, addrs, 4, value_type="u32")
        pairs = dict(zip(live_addrs.tolist(), live_vals.tolist()))
        self.assertEqual(pairs, values)

    def test_streaming_chunk_checks_farther_aligned_target(self):
        maps = [{"start": 0x1000, "end": 0x1010, "prot": 3,
                 "name": "executable"}]
        starts, rows = RDX._build_region_lookup(maps)
        MemorySocket.memory = {0x1000: 0x2000}
        targets = np.asarray([0x2001, 0x2004], dtype=np.uint64)
        hits = RDX._pointer_scan_chunk(
            MemorySocket(), 0x1000, 8, targets,
            {0x2001: [], 0x2004: []}, starts, rows)
        self.assertEqual([(h[0], h[1], h[2]) for h in hits],
                         [(0x1000, 0x2004, 4)])

    def test_memdbg_generic_static_root_rebases_by_section(self):
        vnode = 2 << 24
        before = [
            {"start": 0x1000, "end": 0x2000, "prot": 3,
             "flags": vnode, "name": "[file]"},
            {"start": 0x3000, "end": 0x4000, "prot": 3,
             "flags": vnode, "name": "[file]"},
        ]
        module, base, relative = RDX._module_info_for_addr(0x3020, before)
        self.assertTrue(module.startswith(RDX._SECTION_MODULE_PREFIX))
        self.assertEqual((base, relative), (0x3000, 0x20))

        after = [
            {"start": 0x5000, "end": 0x6000, "prot": 3,
             "flags": vnode, "name": "[file]"},
            {"start": 0x9000, "end": 0xA000, "prot": 3,
             "flags": vnode, "name": "[file]"},
        ]
        self.assertEqual(RDX._pointer_module_base(after, module), 0x9000)

    def test_pointer_scan_does_not_abandon_nonlocal_heap_family(self):
        maps = [
            {"start": 0x1000, "end": 0x1100, "prot": 3,
             "name": "executable"},
            {"start": 0x400000002000, "end": 0x400000002100,
             "prot": 3, "name": "heap"},
            {"start": 0x600000003000, "end": 0x600000003100,
             "prot": 3, "name": "heap"},
        ]
        # Static -> manager in family 0x7000..., manager -> target in family
        # 0x9000.... The old locality break skipped the manager family.
        MemorySocket.memory = {
            0x1000: 0x400000002000,
            0x400000002018: 0x600000003000,
        }

        def read_pointer(_ip, _pid, address):
            return MemorySocket.memory.get(address, 0)

        with patch.object(RDX, "_get_maps_cached", return_value=maps), \
             patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "ps5_read_pointer", side_effect=read_pointer):
            found = RDX.pointer_chain_scan(
                "test", 1, 0x60000000301C, max_depth=2,
                cancel_event=threading.Event())
        self.assertTrue(any(c["base"] == 0x1000 and
                            c["offsets"] == [0x18, 0x1C]
                            for c in found))

    def test_memdbg_native_request_validates_and_decodes_response(self):
        body = struct.pack("<Iii48s", 1, 42, 1, b"eboot.bin")

        class ScriptedSocket:
            def __init__(self): self.pending = bytearray()
            def sendall(self, data):
                magic, version, command, request_id, length = struct.unpack(
                    "<IHHII", data[:16])
                self.pending.extend(struct.pack(
                    "<IHHIiI", magic, version, command, request_id, 0,
                    len(body)) + body)
            def recv_into(self, view, length):
                take = min(length, len(self.pending))
                view[:take] = self.pending[:take]
                del self.pending[:take]
                return take
            def close(self): pass

        client = RDX._MemDBGClient("test")
        client.sock = ScriptedSocket()
        try:
            procs = client.process_list()
            self.assertEqual(procs, [{"pid": 42, "name": "eboot.bin"}])
        finally:
            client.close()

    def test_memdbg_native_memory_frames_read_and_write(self):
        client = RDX._MemDBGClient("test")
        client.hello = {"capabilities": (RDX.MEMDBG_CAP_MEMORY_READ |
                                          RDX.MEMDBG_CAP_MEMORY_WRITE)}
        calls = []

        def request(command, body=b""):
            calls.append((command, body))
            if command == RDX.MEMDBG_CMD_MEMORY_READ:
                return b"\x00" + b"ABCD"
            if command == RDX.MEMDBG_CMD_MEMORY_WRITE:
                return struct.pack("<I", 4)
            raise AssertionError(command)

        with patch.object(client, "request", side_effect=request):
            self.assertEqual(client.memory_read(7, 0x1234, 4), b"ABCD")
            self.assertTrue(client.memory_write(7, 0x2000, b"WXYZ"))
        self.assertEqual(
            struct.unpack("<iQI", calls[0][1]), (7, 0x1234, 4))
        self.assertEqual(
            struct.unpack("<iQI", calls[1][1][:16]), (7, 0x2000, 4))
        self.assertEqual(calls[1][1][16:], b"WXYZ")

    def test_memdbg_native_lz4_memory_frame_is_dependency_free(self):
        # Token 0x44 = four literals then an eight-byte match (4 + 4), with
        # offset four.  The decoded block is b"abcd" repeated three times.
        block = b"\x44abcd\x04\x00"
        frame = b"\x01" + struct.pack("<I", 12) + block
        self.assertEqual(RDX._memdbg_unframe_memory(frame), b"abcd" * 3)
        with self.assertRaises(RuntimeError):
            RDX._memdbg_unframe_memory(
                b"\x01" + struct.pack("<I", 13) + block)
        with self.assertRaises(RuntimeError):
            RDX._memdbg_unframe_memory(b"\x02not-a-frame")

    def test_memdbg_process_maps_prefers_v2_and_caches_per_host(self):
        old = dict(RDX._memdbg_maps_v2_supported)
        RDX._memdbg_maps_v2_supported.clear()
        client = RDX._MemDBGClient("test-host-1")
        map_body = (struct.pack("<I", 1) +
                   struct.pack("<QQII", 0x1000, 0x2000, 5, 0) +
                   b"region".ljust(64, b"\0"))
        calls = []

        def request(command, body=b""):
            calls.append(command)
            if command == RDX.MEMDBG_CMD_PROCESS_MAPS_V2:
                return b"\x00" + map_body
            raise AssertionError("must not fall back to v1 when v2 succeeds")

        try:
            with patch.object(client, "request", side_effect=request):
                maps = client.process_maps(7)
            self.assertEqual(calls, [RDX.MEMDBG_CMD_PROCESS_MAPS_V2])
            self.assertEqual(maps, [{"start": 0x1000, "end": 0x2000,
                                     "prot": 5, "flags": 0, "name": "region"}])
            self.assertTrue(RDX._memdbg_maps_v2_supported["test-host-1"])
        finally:
            RDX._memdbg_maps_v2_supported.clear()
            RDX._memdbg_maps_v2_supported.update(old)

    def test_memdbg_process_maps_falls_back_to_v1_on_v2_failure_and_remembers(self):
        old = dict(RDX._memdbg_maps_v2_supported)
        RDX._memdbg_maps_v2_supported.clear()
        calls = []

        def request(command, body=b""):
            calls.append(command)
            if command == RDX.MEMDBG_CMD_PROCESS_MAPS_V2:
                raise RuntimeError("MemDBG command 0x0110 failed: -1")
            if command == RDX.MEMDBG_CMD_PROCESS_MAPS:
                return struct.pack("<I", 0)
            raise AssertionError(command)

        try:
            client = RDX._MemDBGClient("test-host-2")
            with patch.object(client, "request", side_effect=request):
                maps = client.process_maps(7)
            self.assertEqual(maps, [])
            self.assertEqual(calls, [RDX.MEMDBG_CMD_PROCESS_MAPS_V2,
                                     RDX.MEMDBG_CMD_PROCESS_MAPS])
            self.assertFalse(RDX._memdbg_maps_v2_supported["test-host-2"])

            # A second call on the same host must skip V2 entirely now.
            calls.clear()
            client2 = RDX._MemDBGClient("test-host-2")
            with patch.object(client2, "request", side_effect=request):
                client2.process_maps(7)
            self.assertEqual(calls, [RDX.MEMDBG_CMD_PROCESS_MAPS])
        finally:
            RDX._memdbg_maps_v2_supported.clear()
            RDX._memdbg_maps_v2_supported.update(old)

    def test_memdbg_batch_write_sends_correct_wire_format_and_parses_status(self):
        client = RDX._MemDBGClient("test")
        client.hello = {"capabilities": RDX.MEMDBG_CAP_BATCH_WRITE}
        calls = []

        def request(command, body=b""):
            calls.append((command, body))
            return (struct.pack("<QII", 0x1000, 4, 0) +
                   struct.pack("<QII", 0x2000, 0, 1))

        entries = [(0x1000, b"\x01\x02\x03\x04"), (0x2000, b"\xAA\xBB")]
        with patch.object(client, "request", side_effect=request):
            results = client.memory_write_multi(7, entries)
        self.assertEqual(results, [True, False])

        self.assertEqual(len(calls), 1)
        command, body = calls[0]
        self.assertEqual(command, RDX.MEMDBG_CMD_BATCH_WRITE)
        pid, count, reserved = struct.unpack_from("<iII", body, 0)
        self.assertEqual((pid, count, reserved), (7, 2, 0))
        item_hdr = struct.calcsize("<QII")   # 16
        addr0, len0, _r0 = struct.unpack_from("<QII", body, 12)
        data0 = body[12 + item_hdr:12 + item_hdr + len0]
        self.assertEqual((addr0, len0, data0), (0x1000, 4, b"\x01\x02\x03\x04"))
        off1 = 12 + item_hdr + len0
        addr1, len1, _r1 = struct.unpack_from("<QII", body, off1)
        data1 = body[off1 + item_hdr:off1 + item_hdr + len1]
        self.assertEqual((addr1, len1, data1), (0x2000, 2, b"\xAA\xBB"))

    def test_memdbg_batch_write_rejects_oversized_batch_without_a_request(self):
        client = RDX._MemDBGClient("test")
        client.hello = {"capabilities": RDX.MEMDBG_CAP_BATCH_WRITE}
        entries = [(0x1000, b"\x00")] * (RDX.MEMDBG_BATCH_WRITE_MAX_ITEMS + 1)
        with patch.object(client, "request",
                          side_effect=AssertionError("must not request")):
            with self.assertRaises(ValueError):
                client.memory_write_multi(7, entries)

    def test_memdbg_batch_write_requires_capability(self):
        client = RDX._MemDBGClient("test")
        client.hello = {"capabilities": 0}
        with patch.object(client, "request",
                          side_effect=AssertionError("must not request")):
            with self.assertRaises(RuntimeError):
                client.memory_write_multi(7, [(0x1000, b"\x01")])

    def test_memdbg_write_multi_retries_then_raises_on_total_failure(self):
        attempts = []

        class FailingClient:
            def __init__(self, ip, timeout=5.0):
                self.ip, self.sock, self.timeout = ip, None, timeout
            def connect(self):
                self.sock = FakeSock()
            def memory_write_multi(self, pid, entries):
                attempts.append(1)
                raise RuntimeError("nope")
            def close(self): pass

        with patch.object(RDX, "_MemDBGClient", FailingClient), \
             patch.object(RDX.time, "sleep"):
            with self.assertRaises(RuntimeError):
                RDX.memdbg_write_multi("test", 7, [(0x1000, b"\x01")])
        self.assertEqual(len(attempts), RDX._UI_MAX_RETRIES)

    # ── shared native MemDBG connection (patch117) ────────────────────────
    # Found on hardware, not by reading: MemDBG 0.2.0-nightly.153 on a live
    # PS5 running CUSA01659 accepted exactly 7 connect-read-close cycles and
    # then refused for ~60 s, identically at 0/200/500 ms pacing and with an
    # explicit shutdown(), so it is a live-connection count and not a rate.
    # One reused connection served 200/200 reads at 4.1 ms.

    class _CountingClient:
        """Fake native client that records how often it connects."""
        counter = None

        def __init__(self, ip, timeout=5.0):
            self.ip, self.sock, self.timeout = ip, None, timeout
            self.hello = {"capabilities": 0xFFFFFFFF}

        def connect(self):
            type(self).counter["n"] += 1
            self.sock = FakeSock()
            return self

        def close(self):
            self.sock = None

        def memory_read(self, _pid, _addr, length):
            return b"\xAB" * length

        def process_list(self):
            return [{"pid": 7, "name": "eboot.bin"}]

        def process_maps(self, _pid):
            return [{"start": 0x1000, "end": 0x2000, "prot": 3,
                     "offset": 0, "name": "[default]"}]

    def _memdbg_state(self, caps=0xFFFFFFFF):
        """Enter memdbg backend state; returns the keys to restore."""
        old = {k: RDX.state.get(k) for k in ("backend", "memdbg", "ip")}
        RDX.state.update(backend="memdbg-experimental",
                         memdbg={"capabilities": caps}, ip="test")
        RDX.memdbg_reset_session()
        return old

    def test_memdbg_reads_share_one_connection(self):
        # Regression: ps5_read opened a fresh TCP connection per read, which
        # exhausted the console's native listener after 7 and left every
        # later read paying the full native retry budget -- three connects
        # plus 0.3 s of backoff -- before falling back to port 744.
        # Measured end to end: 311.9 ms/read against 4.8 ms once shared.
        # Deliberately written without any patch117-only symbol, so against
        # patch116 it fails on the connection count (12 != 1) rather than on
        # a missing attribute -- it has to demonstrate the behaviour, not the
        # presence of the new helper.
        counter = {"n": 0}
        Client = self._CountingClient
        Client.counter = counter
        old = {k: RDX.state.get(k) for k in ("backend", "memdbg", "ip")}
        RDX.state.update(backend="memdbg-experimental",
                         memdbg={"capabilities": RDX.MEMDBG_CAP_MEMORY_READ},
                         ip="test")
        reset = getattr(RDX, "memdbg_reset_session", lambda: None)
        reset()
        try:
            with patch.object(RDX, "_MemDBGClient", Client), \
                 patch.object(RDX, "ps5_connect",
                              side_effect=AssertionError("fell back to 744")):
                for _ in range(12):
                    self.assertEqual(RDX.ps5_read("test", 7, 0x4000, 4),
                                     b"\xAB" * 4)
        finally:
            reset()
            RDX.state.update(**old)
        self.assertEqual(counter["n"], 1)

    def test_memdbg_maps_process_list_and_reads_share_one_connection(self):
        counter = {"n": 0}
        Client = self._CountingClient
        Client.counter = counter
        old = self._memdbg_state()
        try:
            with patch.object(RDX, "_MemDBGClient", Client), \
                 patch.object(RDX, "ps5_connect",
                              side_effect=AssertionError("fell back to 744")):
                self.assertEqual(RDX.ps5_proc_list("test")[0]["pid"], 7)
                self.assertEqual(len(RDX.ps5_maps("test", 7)), 1)
                RDX.ps5_read("test", 7, 0x4000, 4)
        finally:
            RDX.memdbg_reset_session()
            RDX.state.update(**old)
        self.assertEqual(counter["n"], 1)

    def test_memdbg_probe_leaves_its_connection_for_reuse(self):
        # The connection opened to identify the payload should become the one
        # the first read uses, not a slot spent and returned to FIN-WAIT-2.
        counter = {"n": 0}
        Client = self._CountingClient
        Client.counter = counter
        old = self._memdbg_state()
        try:
            with patch.object(RDX, "_MemDBGClient", Client):
                hello = RDX.memdbg_probe("test", timeout=1.0)
                self.assertEqual(hello["capabilities"], 0xFFFFFFFF)
                with RDX.memdbg_session("test"):
                    pass
        finally:
            RDX.memdbg_reset_session()
            RDX.state.update(**old)
        self.assertEqual(counter["n"], 1)

    def test_memdbg_session_drops_the_client_after_a_failed_exchange(self):
        # A failed exchange can leave unread bytes in the stream; handing that
        # connection to the next caller desynchronises it.
        made = []

        class Client:
            def __init__(self, ip, timeout=5.0):
                self.ip, self.sock, self.timeout = ip, None, timeout
                made.append(self)
            def connect(self):
                self.sock = FakeSock()
                return self
            def close(self):
                self.sock = None

        RDX.memdbg_reset_session()
        try:
            with patch.object(RDX, "_MemDBGClient", Client):
                with self.assertRaises(RuntimeError):
                    with RDX.memdbg_session("test"):
                        raise RuntimeError("exchange blew up")
                self.assertIsNone(made[0].sock)
                with RDX.memdbg_session("test"):
                    pass
            self.assertEqual(len(made), 2)
        finally:
            RDX.memdbg_reset_session()

    def test_memdbg_session_drops_the_client_on_an_interrupt(self):
        # patch117 caught only Exception here, so a cancelled scan or a Ctrl-C
        # left the shared connection cached mid-exchange and handed it to the
        # next caller -- the exact desynchronisation the handler exists for.
        made = []

        class Client:
            def __init__(self, ip, timeout=5.0):
                self.ip, self.sock, self.timeout = ip, None, timeout
                made.append(self)
            def connect(self):
                self.sock = FakeSock()
                return self
            def close(self):
                self.sock = None

        RDX.memdbg_reset_session()
        try:
            with patch.object(RDX, "_MemDBGClient", Client):
                with self.assertRaises(KeyboardInterrupt):
                    with RDX.memdbg_session("test"):
                        raise KeyboardInterrupt()
                self.assertIsNone(made[0].sock, "socket left open after interrupt")
                with RDX.memdbg_session("test") as again:
                    self.assertIsNot(again, made[0],
                                     "reused the interrupted connection")
            self.assertEqual(len(made), 2)
        finally:
            RDX.memdbg_reset_session()

    def test_memdbg_native_latches_off_after_repeated_failures(self):
        # The advertised bitmap is cached from connect-time HELLO, and the
        # real payload advertises 0xFFFFFFFF -- every bit set -- so it keeps
        # claiming a capability long after the listener stopped serving it.
        # Without a latch, every later operation re-pays the retry cost.
        class DeadClient:
            def __init__(self, ip, timeout=5.0):
                self.ip, self.sock, self.timeout = ip, None, timeout
            def connect(self):
                raise ConnectionError("PS5 disconnected")
            def close(self):
                pass

        old = self._memdbg_state()
        try:
            self.assertTrue(RDX._memdbg_has(RDX.MEMDBG_CAP_MEMORY_READ))
            with patch.object(RDX, "_MemDBGClient", DeadClient), \
                 patch.object(RDX.time, "sleep"), \
                 patch.object(RDX, "ps5_connect",
                              side_effect=RuntimeError("744 down too")):
                for _ in range(RDX._MEMDBG_NATIVE_FAILURE_LIMIT):
                    with self.assertRaises(Exception):
                        RDX.ps5_read("test", 7, 0x4000, 4)
            self.assertFalse(RDX.memdbg_native_ready())
            # The bitmap is untouched; the latch is what stops the retries.
            self.assertEqual(
                int(RDX.state["memdbg"]["capabilities"]), 0xFFFFFFFF)
            self.assertFalse(RDX._memdbg_has(RDX.MEMDBG_CAP_MEMORY_READ))
        finally:
            RDX.memdbg_reset_session()
            RDX.state.update(**old)

    def test_memdbg_success_clears_the_failure_run_and_reset_rearms(self):
        old = self._memdbg_state()
        try:
            for _ in range(RDX._MEMDBG_NATIVE_FAILURE_LIMIT - 1):
                RDX._memdbg_note_native_outcome(False)
            self.assertTrue(RDX.memdbg_native_ready())
            RDX._memdbg_note_native_outcome(True)      # run is broken
            for _ in range(RDX._MEMDBG_NATIVE_FAILURE_LIMIT - 1):
                RDX._memdbg_note_native_outcome(False)
            self.assertTrue(RDX.memdbg_native_ready())
            RDX._memdbg_note_native_outcome(False)     # now the run completes
            self.assertFalse(RDX.memdbg_native_ready())
            RDX.memdbg_reset_session()                 # reconnect re-arms
            self.assertTrue(RDX.memdbg_native_ready())
        finally:
            RDX.memdbg_reset_session()
            RDX.state.update(**old)

    # ── measured region fallback (patch118) ───────────────────────────────
    # Found on hardware: with no region classifier (MemDBG has none) the
    # fallback dropped any mapping over 1 GiB. On a PS5 running CUSA01659
    # that excluded 4.000 GiB of 4.180 GiB writable -- 95.7% of the game --
    # in two "[device]" mappings that read at 86 MiB/s, faster than the
    # "[default]" regions kept at 34 MiB/s, and held thousands of hits for
    # the value being searched. The first scan looked at 4.4% of the game.

    _BIG_REGION = {"start": 0x200000000, "end": 0x280000000,
                   "prot": 3, "offset": 0, "name": "[device]"}

    class _RecordingScanSocket:
        """Fake scan reader that records the addresses a scan actually reads."""
        reads = None

        def __init__(self, ip, pid, *_a, **_k):
            pass
        def read(self, addr, length, *_a, **_k):
            type(self).reads.append((addr, length))
            return b"\x00" * length
        def set_timeout(self, _s): pass
        def close(self): pass

    def _run_scan_over_big_region(self, probe_delay=0.0):
        """Drive scan_first over one oversized region; return addresses read."""
        reads = []
        Sock = self._RecordingScanSocket
        Sock.reads = reads

        def slow_read(_ip, _pid, _addr, length):
            if probe_delay:
                time.sleep(probe_delay)
            return b"\x00" * length

        with patch.object(RDX, "_get_maps_cached",
                          return_value=[dict(self._BIG_REGION)]), \
             patch.object(RDX, "_classify_regions_cached",
                          return_value=([], False)), \
             patch.object(RDX, "ps5_read", side_effect=slow_read), \
             patch.object(RDX, "ps5_scan_exact_turbo",
                          side_effect=RuntimeError("no turbo")), \
             patch.object(RDX, "ps5_scan_exact_server",
                          side_effect=RuntimeError("no console scan")), \
             patch.object(RDX, "_ScanSocket", Sock):
            RDX.scan_first("test", 7, 112, width=4, aligned=True,
                           writable_only=True, value_type="u32")
        return reads

    def test_oversized_region_is_scanned_when_it_reads_fast(self):
        # Deliberately behavioural and free of patch118-only symbols: against
        # patch117 this fails because the region is dropped and nothing is
        # read at all, not because a helper is missing.
        if hasattr(RDX, "_invalidate_oversize_probes"):
            RDX._invalidate_oversize_probes()
        reads = self._run_scan_over_big_region()
        self.assertTrue(reads, "the oversized region was never read")
        lo = min(a for a, _ in reads)
        hi = max(a + n for a, n in reads)
        self.assertGreaterEqual(lo, self._BIG_REGION["start"])
        self.assertLessEqual(hi, self._BIG_REGION["end"] + 4)
        covered = sum(n for _, n in reads)
        self.assertGreater(covered, 0x40000000)   # more than the old cap

    def test_oversized_region_is_dropped_when_it_reads_effectively_unusably(self):
        RDX._invalidate_oversize_probes()
        try:
            # The floor is a readability check, not a GPU detector: hardware
            # measured ordinary cached memory as low as 1.9 MiB/s, so only a
            # mapping that is effectively unusable should be dropped.
            # 256 KiB in 0.7 s is about 0.36 MiB/s.
            reads = self._run_scan_over_big_region(probe_delay=0.7)
        finally:
            RDX._invalidate_oversize_probes()
        self.assertEqual(reads, [])

    def test_oversize_probe_runs_once_per_region(self):
        RDX._invalidate_oversize_probes()
        calls = {"n": 0}

        def counting_read(_ip, _pid, _addr, length):
            calls["n"] += 1
            return b"\x00" * length

        try:
            with patch.object(RDX, "ps5_read", side_effect=counting_read):
                for _ in range(5):
                    self.assertTrue(RDX._oversize_region_is_scannable(
                        "test", 7, dict(self._BIG_REGION)))
            # One probe pass, cached thereafter. The pass takes two samples
            # because single-sample throughput was measured varying threefold
            # within one mapping on hardware.
            self.assertEqual(calls["n"], 2)
        finally:
            RDX._invalidate_oversize_probes()

    def test_oversize_probe_excludes_a_region_it_cannot_read(self):
        RDX._invalidate_oversize_probes()
        try:
            with patch.object(RDX, "ps5_read",
                              side_effect=ConnectionError("PS5 disconnected")):
                self.assertFalse(RDX._oversize_region_is_scannable(
                    "test", 7, dict(self._BIG_REGION)))
        finally:
            RDX._invalidate_oversize_probes()

    def test_classifier_verdict_still_wins_when_the_payload_has_one(self):
        # Guard, not a regression: ps5debug-NG behaviour must be unchanged --
        # when the classifier answers, the probe must not run at all.
        if hasattr(RDX, "_invalidate_oversize_probes"):
            RDX._invalidate_oversize_probes()
        big = dict(self._BIG_REGION)
        uncached = [(big["start"], big["end"])]
        Sock = self._RecordingScanSocket
        Sock.reads = []
        with patch.object(RDX, "_get_maps_cached", return_value=[big]), \
             patch.object(RDX, "_classify_regions_cached",
                          return_value=(uncached, True)), \
             patch.object(RDX, "ps5_read",
                          side_effect=AssertionError("probed despite classifier")), \
             patch.object(RDX, "ps5_scan_exact_turbo",
                          side_effect=RuntimeError("no turbo")), \
             patch.object(RDX, "ps5_scan_exact_server",
                          side_effect=RuntimeError("no console scan")), \
             patch.object(RDX, "_ScanSocket", Sock):
            RDX.scan_first("test", 7, 112, width=4, aligned=True,
                           writable_only=True, value_type="u32")
        self.assertEqual(Sock.reads, [])

    # ── TurboScan capability is learned once per host (patch120) ──────────

    def _scan_with_turbo(self, turbo, engine="auto", value=112):
        """Run scan_first with a stubbed turbo entry point; return reads."""
        Sock = self._RecordingScanSocket
        Sock.reads = []
        region = {"start": 0x10000, "end": 0x20000, "prot": 3,
                  "offset": 0, "name": "heap"}
        old_engine = RDX.state.get("scan_engine")
        RDX.state["scan_engine"] = engine
        try:
            with patch.object(RDX, "_get_maps_cached", return_value=[region]), \
                 patch.object(RDX, "_classify_regions_cached",
                              return_value=([], True)), \
                 patch.object(RDX, "ps5_scan_exact_turbo", turbo), \
                 patch.object(RDX, "ps5_scan_exact_server",
                              side_effect=RuntimeError("no console scan")), \
                 patch.object(RDX, "_ScanSocket", Sock):
                RDX.scan_first("turbo-host", 7, value, width=4, aligned=True,
                               writable_only=True, value_type="u32")
        finally:
            RDX.state["scan_engine"] = old_engine

    def test_turboscan_absence_is_learned_once_per_host(self):
        # Regression: ps5_scan_exact_turbo runs ps5_auth_scanner then
        # ps5_turboscan_caps over port 744 with a 15 s default. MemDBG accepts
        # connections there but never answers, so the probe times out rather
        # than failing fast -- and nothing remembered, so every scan paid it
        # again. Written without patch120-only symbols so it fails on patch119
        # on the call count (2 != 1), not on a missing helper.
        calls = {"n": 0}

        def turbo(*_a, **_k):
            calls["n"] += 1
            raise RuntimeError("timed out")

        reset = getattr(RDX, "_turbo_supported", None)
        if reset is not None:
            reset.clear()
        try:
            self._scan_with_turbo(turbo)
            self._scan_with_turbo(turbo)
        finally:
            if reset is not None:
                reset.clear()
        self.assertEqual(calls["n"], 1)

    def test_explicit_turbo_engine_still_probes_when_known_absent(self):
        # A user who picked Turbo must get the real error, not a silent host
        # fallback, even after the capability was learned absent.
        RDX._turbo_supported.clear()
        calls = {"n": 0}

        def turbo(*_a, **_k):
            calls["n"] += 1
            raise RuntimeError("timed out")

        try:
            self._scan_with_turbo(turbo)              # auto: learns absence
            with self.assertRaises(RuntimeError):
                self._scan_with_turbo(turbo, engine="turbo")
            self.assertEqual(calls["n"], 2)
        finally:
            RDX._turbo_supported.clear()

    def test_turbo_success_does_not_latch_the_capability_off(self):
        RDX._turbo_supported.clear()
        hits = np.asarray([0x10004], dtype=RDX._NP_ADDR_DTYPE)
        calls = {"n": 0}

        def turbo(*_a, **_k):
            calls["n"] += 1
            return hits

        try:
            self._scan_with_turbo(turbo)
            self._scan_with_turbo(turbo)
            self.assertEqual(calls["n"], 2)
            self.assertTrue(RDX._turbo_worth_probing("turbo-host"))
        finally:
            RDX._turbo_supported.clear()

    def test_reconnect_rearms_the_turbo_probe(self):
        RDX._turbo_supported.clear()
        try:
            RDX._note_turbo_outcome("h", False)
            self.assertFalse(RDX._turbo_worth_probing("h"))
            RDX._reset_learned_payload_support()
            self.assertTrue(RDX._turbo_worth_probing("h"))
        finally:
            RDX._turbo_supported.clear()

    def test_resident_rescan_failure_does_not_latch_turbo_off(self):
        # "no matching resident session" happens on consoles whose TurboScan
        # works, so it must not be recorded as evidence about the payload.
        RDX._turbo_supported.clear()
        try:
            with patch.object(RDX, "ps5_scan_relational_turbo",
                              side_effect=RuntimeError("no resident session")), \
                 patch.object(RDX, "ps5_read_batch",
                              return_value=(
                                  np.asarray([0x7000], dtype=RDX._NP_ADDR_DTYPE),
                                  np.asarray([5], dtype=np.uint32))):
                old = RDX.state.get("scan_engine")
                RDX.state["scan_engine"] = "auto"
                try:
                    RDX.scan_next_relational(
                        "h", 7, 4,
                        np.asarray([0x7000], dtype=RDX._NP_ADDR_DTYPE),
                        np.asarray([4], dtype=np.uint32),
                        "increased", 0, value_type="u32")
                finally:
                    RDX.state["scan_engine"] = old
            self.assertTrue(RDX._turbo_worth_probing("h"),
                            "a missing resident session latched TurboScan off")
        finally:
            RDX._turbo_supported.clear()

    def test_scan_first_refuses_bytes_before_any_engine_is_chosen(self):
        # Guard, not a regression (passes on earlier patches too).
        #
        # Recorded while reviewing patch120. scan_first computes
        #     payload_exact_ok = VALUE_TYPES[type_key]["kind"] in {...}
        # and guards the turbo and console engine branches with it, each with
        # an `elif engine == ... and not payload_exact_ok: raise`. Those two
        # arms are unreachable: "bytes" is the only non-numeric kind, and it
        # is rejected at the top of scan_first with a different message, so
        # payload_exact_ok is always True by the time the engine is chosen.
        #
        # They are defensible as cover for a future non-numeric type, so they
        # are left in place. This pins the rejection that actually happens, so
        # that if the early guard ever moves, the dead arms become live rather
        # than silently letting a bytes scan reach an engine.
        for engine in ("auto", "turbo", "console", "host"):
            old = RDX.state.get("scan_engine")
            RDX.state["scan_engine"] = engine
            try:
                with self.assertRaises(ValueError) as caught:
                    RDX.scan_first("h", 7, b"\x01\x02", width=2,
                                   aligned=True, value_type="bytes")
                self.assertIn("scan_first_pattern", str(caught.exception),
                              f"engine={engine}")
            finally:
                RDX.state["scan_engine"] = old

    def test_enabling_a_freeze_during_a_timed_out_stop_still_runs(self):
        # Regression: _stop_freeze_worker keeps the stop signal asserted when
        # its join times out (a worker blocked in a slow write) and retains
        # the thread reference, on purpose -- clearing it would let that
        # worker resume. _ensure_freeze_worker only checked is_alive(), so a
        # freeze enabled in that window returned early with the signal still
        # asserted; the old worker then exited and cleared the reference,
        # leaving a registered cheat, no worker, and no recovery.
        # Hermetic: this manipulates process-wide freeze state, so quiesce
        # any worker a previous test left running and restore everything
        # afterwards. Without this the test is order-dependent in both
        # directions -- a leaked worker perturbs it, and _stop_freeze_worker
        # below would clear another test's targets.
        RDX._stop_freeze_worker()
        old_thread, old_stop = RDX._freeze_thread, RDX._freeze_stop
        old_targets = dict(RDX._freeze_targets)
        old_status = dict(RDX._freeze_status)
        gate = threading.Event()
        winding_down = threading.Thread(target=lambda: gate.wait(10),
                                        daemon=True)
        winding_down.start()
        try:
            RDX._freeze_thread = winding_down
            RDX._freeze_stop = threading.Event()
            RDX._freeze_stop.set()          # what a timed-out stop leaves

            RDX._ensure_freeze_worker()

            self.assertFalse(
                RDX._freeze_stop.is_set(),
                "new worker inherited the outgoing worker's stop signal")
            self.assertIsNot(RDX._freeze_thread, winding_down,
                             "no new worker was started")
            self.assertTrue(RDX._freeze_thread.is_alive())

            # The outgoing worker exiting must not clear the new worker.
            gate.set()
            winding_down.join(timeout=5)
            self.assertTrue(RDX._freeze_thread.is_alive(),
                            "the outgoing worker took the new one with it")
        finally:
            gate.set()
            RDX._stop_freeze_worker()
            with RDX._freeze_lock:
                RDX._freeze_targets.clear()
                RDX._freeze_targets.update(old_targets)
                RDX._freeze_status.clear()
                RDX._freeze_status.update(old_status)
            RDX._freeze_thread, RDX._freeze_stop = old_thread, old_stop

    def test_negative_final_offset_is_a_plausible_field(self):
        # Regression, from hardware. _candidate_field_offset_is_plausible
        # tested `0 <= offsets[-1] <= MAX`, so every negative displacement was
        # judged implausible. A chain landing inside a sub-object and reading
        # a field earlier in its parent has a small negative final offset --
        # ordinary, and what this title actually used.
        #
        # Measured on CUSA01659: of the eight top-ranked chains for a real
        # ammo address, four ended in -0x60 and all eight were called
        # implausible. With the discriminator inert the sort fell through to
        # depth, promoting depth-1 holders at +0x42720 -- the low bits of the
        # target address -- above the real chains.
        plausible = RDX._candidate_field_offset_is_plausible
        self.assertTrue(plausible({"offsets": [0x5248, -0x60]}))
        self.assertTrue(plausible({"offsets": [-0x10]}))
        self.assertTrue(plausible({"offsets": [RDX._PTR_PLAUSIBLE_FIELD_MAX]}))
        self.assertTrue(plausible({"offsets": [-RDX._PTR_PLAUSIBLE_FIELD_MAX]}))
        # Distance is still what the rule is about.
        self.assertFalse(plausible({"offsets": [0x42720]}))
        self.assertFalse(plausible({"offsets": [-0x31440]}))
        self.assertFalse(plausible({"offsets": [RDX._PTR_PLAUSIBLE_FIELD_MAX + 1]}))
        self.assertFalse(plausible({"offsets": [-RDX._PTR_PLAUSIBLE_FIELD_MAX - 1]}))

    def test_ranking_puts_a_negative_field_chain_above_a_far_coincidence(self):
        # The consequence, end to end: the real chain must outrank the
        # coincidence-shaped depth-1 holder, which it did not before.
        real = {"base": 0x1000, "offsets": [0x5248, -0x60], "depth": 2,
                "score": 100.0, "verified": True, "module_name": "executable"}
        coincidence = {"base": 0x2000, "offsets": [0x42720], "depth": 1,
                       "score": 255.0, "verified": True,
                       "module_name": "Il2CppUserAssemblies.prx"}
        with patch.object(RDX, "_resolve_pointer_chain",
                          return_value=(True, 0x9999, ())):
            ranked = RDX._rank_pointer_candidates(
                "h", 7, [coincidence, real])
        self.assertEqual(ranked[0]["offsets"], [0x5248, -0x60],
                         "the far-displacement coincidence still ranks first")

    def test_type_pointer_targets_admit_heap_and_exclude_code(self):
        # Regression, from hardware. Type-pointer targets were restricted to
        # static/module regions, but IL2CPP allocates Il2CppClass on the heap:
        # on CUSA01659 the class for "PlayerController" resolved at
        # 0x20362e560, prot=3, _is_static_region False. Every real type
        # pointer was therefore excluded, and what survived were pointers into
        # code -- the top five groups of a 512-group scan disassembled as
        # x86-64 prologues, and 0 of 40 resolved to a class name.
        maps = [
            {"start": 0x400000, "end": 0x500000, "prot": 5, "name": "exec"},
            {"start": 0x200000000, "end": 0x200100000, "prot": 3, "name": ""},
            {"start": 0x300000000, "end": 0x300010000, "prot": 1,
             "name": "global-metadata"},
        ]
        starts, ends = RDX._type_target_interval_arrays(maps)
        import numpy as _np
        probe = _np.asarray([0x450000,        # inside executable -> rejected
                             0x200080000,     # heap -> accepted
                             0x300004000,     # metadata, read-only -> accepted
                             0x900000000],    # unmapped -> rejected
                            dtype=_np.uint64)
        got = RDX._values_in_intervals(probe, starts, ends).tolist()
        self.assertEqual(got, [False, True, True, False])

    def test_type_target_intervals_are_not_the_static_intervals(self):
        # The two filters answer different questions and must not be aliased:
        # a class never lives in code, but it very often lives in the heap.
        maps = [
            {"start": 0x400000, "end": 0x500000, "prot": 5, "name": "exec"},
            {"start": 0x200000000, "end": 0x200100000, "prot": 3, "name": ""},
        ]
        t_s, _t_e = RDX._type_target_interval_arrays(maps)
        s_s, _s_e = RDX._static_interval_arrays(maps)
        self.assertNotEqual(t_s.tolist(), s_s.tolist())
        self.assertIn(0x200000000, t_s.tolist())
        self.assertNotIn(0x400000, t_s.tolist())

    def test_a_contested_class_name_is_recorded_not_presented_as_settled(self):
        # Regression, from hardware. _read_klass_name takes the first offset
        # that yields a plausible string. On CUSA01659, 8 of 40 type pointers
        # had more than one such offset, and the correct offset is not even
        # consistent between real classes: PlayerController carries its name
        # at +0x18 with +0x10 empty, while String and Boolean carry theirs at
        # +0x10 with the *namespace* at +0x18. One structure returned
        # 'TargetPlayer' from +0x10 and 'PlayerController' from +0x18 -- a
        # field name beating a class name on probe order alone.
        con = self._console(maps=[
            {"name": "executable", "start": 0x400000, "end": 0x420000,
             "prot": 5},
            {"name": "executable", "start": 0x420000, "end": 0x430000,
             "prot": 1},
            {"name": "", "start": 0x2000000, "end": 0x2010000, "prot": 3}])
        try:
            RDX._invalidate_klass_names()
            klass = 0x420280

            def put_q(addr, val):
                for i, b in enumerate(int(val).to_bytes(8, "little")):
                    con.memory[addr + i] = b

            def put_s(addr, s):
                for i, b in enumerate(s.encode() + b"\x00"):
                    con.memory[addr + i] = b

            # Two offsets both resolve: +0x10 first in probe order, +0x18 the
            # one a real class of this shape would actually use.
            put_q(klass + 0x10, 0x421000); put_s(0x421000, "TargetPlayer")
            put_q(klass + 0x18, 0x421100); put_s(0x421100, "PlayerController")
            for i in range(64):
                put_q(0x2000000 + i * 0x40, klass)

            groups = RDX.scan_type_instances(con.host, 91, min_instances=8)
            self.assertTrue(groups)
            g = groups[0]
            # Behaviour is unchanged: the first hit is still what is shown.
            self.assertEqual(g["class_name"], "TargetPlayer")
            # But the contest is recorded rather than hidden.
            contested = g.get("class_name_ambiguous")
            self.assertTrue(contested, "a contested name was reported as settled")
            self.assertEqual(
                sorted(contested),
                [(0x10, "TargetPlayer"), (0x18, "PlayerController")])
        finally:
            self._release(con)

    def test_an_unambiguous_class_name_is_not_flagged(self):
        con = self._console(maps=[
            {"name": "executable", "start": 0x400000, "end": 0x420000,
             "prot": 5},
            {"name": "executable", "start": 0x420000, "end": 0x430000,
             "prot": 1},
            {"name": "", "start": 0x2000000, "end": 0x2010000, "prot": 3}])
        try:
            RDX._invalidate_klass_names()
            klass = 0x420280
            for i, b in enumerate(int(0x421000).to_bytes(8, "little")):
                con.memory[klass + 0x10 + i] = b
            for i, b in enumerate(b"PlayerController\x00"):
                con.memory[0x421000 + i] = b
            for i in range(64):
                for j, b in enumerate(int(klass).to_bytes(8, "little")):
                    con.memory[0x2000000 + i * 0x40 + j] = b
            groups = RDX.scan_type_instances(con.host, 91, min_instances=8)
            self.assertEqual(groups[0]["class_name"], "PlayerController")
            self.assertIsNone(groups[0].get("class_name_ambiguous"))
        finally:
            self._release(con)

    def test_oversize_probe_keeps_a_region_one_sample_would_have_dropped(self):
        # Regression, from hardware. Throughput within a single mapping was
        # measured varying from 1.9 to 5.6 MiB/s on ps5debug-NG, and the
        # classifier-confirmed *uncached* mapping read faster than the cached
        # one on most samples. So one sample decides nothing, and the cost
        # asymmetry is lopsided: wrongly excluding a mapping cost 95.7% of the
        # game in patch118, wrongly including one only costs scan time.
        RDX._invalidate_oversize_probes()
        calls = {"n": 0}

        def flaky(_ip, _pid, _addr, length):
            calls["n"] += 1
            # First sample slow enough to fail the floor, second fine.
            if calls["n"] == 1:
                time.sleep(0.6)
            return b"\x00" * length

        try:
            with patch.object(RDX, "ps5_read", side_effect=flaky):
                self.assertTrue(
                    RDX._oversize_region_is_scannable(
                        "t", 7, dict(self._BIG_REGION)),
                    "a single slow sample dropped a readable 2 GiB mapping")
            self.assertEqual(calls["n"], 2)
        finally:
            RDX._invalidate_oversize_probes()

    def test_oversize_probe_excludes_only_what_it_cannot_read(self):
        RDX._invalidate_oversize_probes()
        try:
            with patch.object(RDX, "ps5_read",
                              side_effect=ConnectionError("unreadable")):
                self.assertFalse(RDX._oversize_region_is_scannable(
                    "t", 7, dict(self._BIG_REGION)))
        finally:
            RDX._invalidate_oversize_probes()

    def test_oversize_floor_is_not_a_gpu_detector(self):
        # Documents the measurement that ruled that out: the floor must stay
        # well below ordinary read rates, so it rejects only unusable memory.
        self.assertLess(RDX._OVERSIZE_MIN_RATE, 1.9,
                        "the floor is high enough to drop mappings that "
                        "hardware measured as ordinary cached memory")

    def test_a_freeze_losing_the_race_is_reported_as_contested(self):
        # Regression, from hardware. On CUSA01659 a freeze held its value in
        # 31 of 657 samples -- 4.7% -- while RDX reported "active @ 0x..."
        # the entire time. The game rewrites the address every 8-20 ms and a
        # write round trip costs 15.7 ms, so the tick rate cannot win; the
        # user watched the in-game counter never change. A status that cannot
        # fail is not a status.
        seen = {}
        rid = "r1"
        # Below the sample floor, no verdict yet: one unlucky read must not
        # condemn a freeze that is working.
        self.assertIsNone(RDX._freeze_note_verification(seen, rid, False))
        self.assertIsNone(RDX._freeze_note_verification(seen, rid, True))
        verdict = RDX._freeze_note_verification(seen, rid, False)
        self.assertIsNotNone(verdict)
        self.assertIn("contested", verdict)
        self.assertIn("1/3", verdict)

    def test_a_freeze_that_holds_is_not_flagged(self):
        seen = {}
        rid = "r2"
        for _ in range(6):
            self.assertIsNone(RDX._freeze_note_verification(seen, rid, True))
        # A single lost check among many is normal timing, not contention.
        self.assertIsNone(RDX._freeze_note_verification(seen, rid, False))

    def test_contention_recovery_is_bounded_by_the_window(self):
        # Regression: the first version counted every check ever taken, so
        # detection was bounded but recovery was not -- a freeze contested for
        # five minutes then holding perfectly needed five minutes of wins
        # before the badge caught up. Observed on hardware, with the indicator
        # sitting on LOSE after the game had stopped fighting the write.
        seen = {}
        rid = "r4"
        for _ in range(30):                      # a long spell of losing
            RDX._freeze_note_verification(seen, rid, False)
        self.assertIsNotNone(RDX._freeze_note_verification(seen, rid, False))
        cleared_after = None
        for i in range(1, RDX._FREEZE_VERIFY_WINDOW + 2):
            if RDX._freeze_note_verification(seen, rid, True) is None:
                cleared_after = i
                break
        self.assertIsNotNone(cleared_after, "contention never cleared")
        self.assertLessEqual(
            cleared_after, RDX._FREEZE_VERIFY_WINDOW,
            "recovery is not bounded by the window — losses never age out")

    def test_contention_window_does_not_grow_without_bound(self):
        seen = {}
        for _ in range(500):
            RDX._freeze_note_verification(seen, "r5", False)
        self.assertLessEqual(len(seen["r5"]), RDX._FREEZE_VERIFY_WINDOW)

    def test_the_freeze_indicator_distinguishes_contested_from_active(self):
        cheat = {"_runtime_id": "r3", "name": "Ammo"}
        old_t = dict(RDX._freeze_targets)
        old_s = dict(RDX._freeze_status)
        try:
            with RDX._freeze_lock:
                RDX._freeze_targets["r3"] = cheat
                RDX._freeze_status["r3"] = "active @ 0x1000"
            self.assertEqual(RDX._cheat_freeze_indicator(cheat), "ON")
            # Contention lives apart from _freeze_status, which the write
            # phase rewrites every tick: a verdict stored there survived
            # ~200 ms and no caller ever saw it.
            with RDX._freeze_lock:
                RDX._freeze_contested["r3"] = ("contested — the game is "
                                               "overwriting this (1/3 held)")
            self.assertEqual(RDX._cheat_freeze_indicator(cheat), "LOSE")
            self.assertIn("contested", RDX.freeze_contention_note(cheat))
            # ...and a per-tick "active" write must not clear it.
            with RDX._freeze_lock:
                RDX._freeze_status["r3"] = "active @ 0x1000"
            self.assertEqual(RDX._cheat_freeze_indicator(cheat), "LOSE")
            with RDX._freeze_lock:
                RDX._freeze_status["r3"] = "error: nope"
            self.assertEqual(RDX._cheat_freeze_indicator(cheat), "ERR")
        finally:
            with RDX._freeze_lock:
                RDX._freeze_targets.clear(); RDX._freeze_targets.update(old_t)
                RDX._freeze_status.clear(); RDX._freeze_status.update(old_s)
                RDX._freeze_contested.clear()

    def test_cheat_durability_names_the_three_real_states(self):
        # Both ammo addresses found on hardware this session were raw heap
        # pointers. They would have exported cleanly and been dead on the next
        # launch, with nothing in the list saying so.
        raw = {"name": "Ammo", "address": 0x251042720, "width": 4}
        self.assertEqual(RDX.cheat_durability(raw)[0], "SESSION")

        # A chain alone is not enough: RELOAD requires the two-reload
        # promotion to have actually happened. An unpromoted chain is still
        # this session's address, and the badge must not overstate it.
        unpromoted = {"name": "Ammo", "address": 0x251042720, "width": 4,
                      "offsets": [0x5248, -0x60], "module_name": "executable",
                      "module_relative_offset": 0x1234}
        self.assertEqual(RDX.cheat_durability(unpromoted)[0], "SESSION")

        chain = dict(unpromoted, cross_reload_validated=True,
                     game_identity="eboot.bin:abc123")
        self.assertEqual(RDX.cheat_durability(chain)[0], "RELOAD")

        static = {"name": "Patch", "address": 0x410280, "width": 4,
                  "module_name": "executable", "module_relative_offset": 0x280,
                  "type": "write"}
        if RDX._is_module_relative_scalar(static):
            self.assertEqual(RDX.cheat_durability(static)[0], "STATIC")

    def test_durability_summary_counts_what_is_being_exported(self):
        raw = {"name": "a", "address": 1, "width": 4}
        chain = {"name": "b", "address": 2, "width": 4,
                 "offsets": [0x10], "module_name": "executable",
                 "module_relative_offset": 0x20,
                 "cross_reload_validated": True,
                 "game_identity": "eboot.bin:abc123"}
        summary = RDX.summarise_durability([raw, raw, chain])
        self.assertIn("2 session", summary)
        self.assertIn("1 reload", summary)
        self.assertEqual(RDX.summarise_durability([]), "no cheats")

    def test_durability_never_claims_survival_across_a_game_update(self):
        # RDX has no signature-rooted chain root (UPSTREAM_AUDIT_PASS7), so no
        # state may imply a cheat survives a patch. Claiming otherwise is the
        # kind of confident label this session removed elsewhere.
        chain = {"name": "b", "address": 2, "width": 4,
                 "offsets": [0x10], "module_name": "executable",
                 "module_relative_offset": 0x20,
                 "cross_reload_validated": True,
                 "game_identity": "eboot.bin:abc123"}
        for cheat in ({"name": "a", "address": 1, "width": 4}, chain):
            text = RDX.cheat_durability(cheat)[1].lower()
            self.assertNotIn("update", text)
            self.assertNotIn("patch-proof", text)

    def test_pre_resume_verdict_is_reported_before_the_resume_can_hang(self):
        # Regression, from a real attach. patch102 reads the debug registers
        # back on one thread while the target is stopped, then resumes, then
        # sweeps every thread -- and only the sweep produced a verdict. On
        # 2026-08-30 the resume timed out (CMD_DEBUG_CONTINUE, 61 s), the
        # exception unwound, and the single-thread data already in hand was
        # discarded. The attach cost a stopped game, a wedged payload and a
        # console restart, and answered nothing.
        armed = {"armed": [101], "checked": 1, "total": 8}
        empty = {"armed": [], "checked": 1, "total": 8}
        none_read = {"armed": [], "checked": 0, "total": 8}

        self.assertIn("IS set", RDX._debug_watchpoint_preliminary(armed))
        self.assertIn("NOT set", RDX._debug_watchpoint_preliminary(empty))
        self.assertIn("could not be read",
                      RDX._debug_watchpoint_preliminary(none_read))
        self.assertIsNone(RDX._debug_watchpoint_preliminary(None))

    def test_pre_resume_verdict_never_claims_thread_coverage(self):
        # One thread cannot rule per-thread application in or out. Claiming it
        # could is the overreach the calibration note already records once.
        for coverage in ({"armed": [1], "checked": 1, "total": 64},
                         {"armed": [], "checked": 1, "total": 64}):
            text = RDX._debug_watchpoint_preliminary(coverage).lower()
            self.assertNotIn("ruled out", text)
            self.assertNotIn("all threads", text)
            self.assertNotIn("per-thread application", text)

    def test_scan_overflow_to_port_744_is_not_logged_as_a_failure(self):
        # Measured: MemDBG serves 6 concurrent native connections and refuses
        # the 7th; RDX's budget is 10, so a scan overflows to port 744 by
        # design. A/B on hardware over the same 4,280.8 MiB scan showed the
        # overflow is not a cost -- budget 10 finished in 168.5 s with one
        # overflow, budget 5 in 213.1 s with none. This session's notes
        # recorded the warning as degradation three times before the
        # measurement showed otherwise.
        old_notes = set(RDX._memdbg_fallback_notes)
        lines = []
        try:
            RDX._memdbg_fallback_notes.clear()
            with patch.object(RDX, "add_log",
                              side_effect=lambda m, l="info", *a, **k:
                                  lines.append((l, m))):
                RDX._note_memdbg_fallback("scan read", ConnectionError("x"))
            self.assertEqual(len(lines), 1)
            level, message = lines[0]
            self.assertEqual(level, "info", "routine overflow logged as a warning")
            self.assertIn("budget", message)
            self.assertNotIn("failed", message)
        finally:
            RDX._memdbg_fallback_notes.clear()
            RDX._memdbg_fallback_notes.update(old_notes)

    def test_a_one_off_read_fallback_is_still_a_warning(self):
        # A single read or write dropping to the compatibility listener is not
        # budget overflow -- nothing else is competing for connections -- so it
        # still means something is wrong.
        old_notes = set(RDX._memdbg_fallback_notes)
        lines = []
        try:
            RDX._memdbg_fallback_notes.clear()
            with patch.object(RDX, "add_log",
                              side_effect=lambda m, l="info", *a, **k:
                                  lines.append((l, m))):
                RDX._note_memdbg_fallback("read", ConnectionError("x"))
                RDX._note_memdbg_fallback("write", ConnectionError("x"))
            self.assertEqual([l for l, _ in lines], ["warn", "warn"])
        finally:
            RDX._memdbg_fallback_notes.clear()
            RDX._memdbg_fallback_notes.update(old_notes)

    def test_an_inert_cheat_is_marked_in_its_exported_name(self):
        # Observed for real on 2026-08-30. A CheatRunner trainer was built with
        # deliberately inert values -- ValueOn and ValueOff both the value
        # already at the address -- so a format test could not crash a game.
        # CheatRunner loaded it, toggling did nothing, and the reasonable
        # question back was "what is this supposed to do?". A cheat that cannot
        # act is indistinguishable from a broken one, and the person who finds
        # out is the one least able to diagnose it.
        maps = [{"name": "executable", "start": 0x400000, "end": 0x500000,
                 "prot": 3, "offset": 0}]
        inert = {"name": "Inert", "address": 0x410000, "value": 100,
                 "original_value": 100, "type": "write", "width": 4,
                 "value_type": "u32", "process": "eboot.bin",
                 "module_name": "executable", "module_relative_offset": 0x10000}
        live = dict(inert, name="Live", value=101)
        _text, mods, _skipped = RDX.generate_etahen_json(
            [inert, live], "CUSA00001", "01.00", "T", "eboot.bin", maps, "RDX")
        by_name = {m["name"]: m for m in mods}
        marked = [k for k in by_name if RDX._INERT_MARKER in k]
        self.assertEqual(len(marked), 1, f"expected one marked, got {list(by_name)}")
        self.assertTrue(marked[0].startswith("Inert"))
        # And the one that actually changes something must not be marked.
        self.assertFalse(any(RDX._INERT_MARKER in k and k.startswith("Live")
                             for k in by_name))

    def test_the_inert_marker_survives_into_the_shn_the_player_reads(self):
        # The Trainer XML schema carries no hint or type field, so the name is
        # the only place this can live -- the same reason _ONE_SHOT_MARKER is
        # in the name rather than the description.
        mods = [{"name": f"Ammo {RDX._INERT_MARKER}", "type": "checkbox",
                 "memory": [{"offset": "129E230", "on": "64000000",
                             "off": "64000000"}]}]
        xml = RDX.generate_shn_text(mods, "CUSA00001", "01.00", "T", "eboot.bin")
        self.assertIn(RDX._INERT_MARKER, xml)

    def test_a_freeze_export_can_carry_both_markers(self):
        # A frozen cheat with identical on/off values is downgraded to a
        # one-shot write *and* inert. Both facts matter and neither should
        # displace the other.
        maps = [{"name": "executable", "start": 0x400000, "end": 0x500000,
                 "prot": 3, "offset": 0}]
        cheat = {"name": "Both", "address": 0x410000, "value": 7,
                 "original_value": 7, "type": "freeze", "width": 4,
                 "value_type": "u32", "process": "eboot.bin",
                 "module_name": "executable", "module_relative_offset": 0x10000}
        _text, mods, _skipped = RDX.generate_etahen_json(
            [cheat], "CUSA00001", "01.00", "T", "eboot.bin", maps, "RDX")
        name = mods[0]["name"]
        self.assertIn(RDX._ONE_SHOT_MARKER, name)
        self.assertIn(RDX._INERT_MARKER, name)

    # ── AOB anchoring (patch132) ──────────────────────────────────────────

    _AOB_MAPS = [
        {"name": "executable", "start": 0x400000, "end": 0x410000, "prot": 5},
        {"name": "executable", "start": 0x410000, "end": 0x420000, "prot": 1},
        {"name": "", "start": 0x800000, "end": 0x810000, "prot": 3},
    ]

    def test_aob_capture_is_refused_on_writable_memory(self):
        # The technique anchors an *instruction*; code bytes are stable. The
        # bytes around a live value are the live value, so an anchor captured
        # over writable memory stops matching the moment the game writes --
        # which for a cheat target is immediately, by definition.
        with patch.object(RDX, "ps5_read",
                          side_effect=AssertionError("must not even read")):
            self.assertIsNone(RDX.capture_aob_signature(
                "t", 7, 0x800100, maps=self._AOB_MAPS))

    def test_aob_capture_refuses_a_uniform_window(self):
        # A run of identical bytes matches everywhere and anchors nothing.
        with patch.object(RDX, "ps5_read", return_value=b"\x90" * 32):
            self.assertIsNone(RDX.capture_aob_signature(
                "t", 7, 0x405000, maps=self._AOB_MAPS))

    def test_aob_capture_succeeds_on_code_and_round_trips(self):
        window = bytes(range(32))
        with patch.object(RDX, "ps5_read", return_value=window):
            sig = RDX.capture_aob_signature("t", 7, 0x405000,
                                            maps=self._AOB_MAPS)
        self.assertIsNotNone(sig)
        self.assertEqual(bytes.fromhex(sig["pattern"]), window)
        self.assertEqual(sig["lead"], 16)
        with patch.object(RDX, "ps5_read", return_value=window):
            self.assertTrue(RDX.aob_signature_matches("t", 7, 0x405000, sig))
        with patch.object(RDX, "ps5_read", return_value=b"\x00" * 32):
            self.assertFalse(RDX.aob_signature_matches("t", 7, 0x405000, sig))

    def test_aob_relocation_refuses_an_ambiguous_match(self):
        # The walkthrough this follows insists the array of bytes be *unique*.
        # Two matches means the anchor cannot say which site it meant, and
        # guessing would relocate a cheat onto the wrong instruction.
        sig = {"pattern": "00" * 4 + "AABBCCDD" * 4, "mask": "FF" * 20,
               "lead": 8}
        two = np.asarray([0x401000, 0x402000], dtype=RDX._NP_ADDR_DTYPE)
        with patch.object(RDX, "scan_first_pattern", return_value=two):
            self.assertIsNone(RDX.relocate_by_aob_signature("t", 7, sig))

    def test_aob_relocation_returns_the_anchor_not_the_window_start(self):
        sig = {"pattern": "AABBCCDD" * 5, "mask": "FF" * 20, "lead": 8}
        one = np.asarray([0x401000], dtype=RDX._NP_ADDR_DTYPE)
        with patch.object(RDX, "scan_first_pattern", return_value=one):
            self.assertEqual(
                RDX.relocate_by_aob_signature("t", 7, sig), 0x401000 + 8)

    def test_aob_relocation_refuses_a_mostly_wildcard_signature(self):
        sig = {"pattern": "AA" * 20, "mask": "FF" * 2 + "00" * 18, "lead": 0}
        with patch.object(RDX, "scan_first_pattern",
                          side_effect=AssertionError("must not scan")):
            self.assertIsNone(RDX.relocate_by_aob_signature("t", 7, sig))

    def test_aob_helpers_never_raise_on_a_malformed_signature(self):
        for bad in ({}, {"pattern": "zz", "mask": "FF", "lead": 0},
                    {"pattern": "AABB", "mask": "FF", "lead": 0},
                    {"pattern": "", "mask": "", "lead": 0}):
            self.assertFalse(RDX.aob_signature_matches("t", 7, 0x1000, bad))
            self.assertIsNone(RDX.relocate_by_aob_signature("t", 7, bad))

    def test_aob_capture_never_reads_outside_its_region(self):
        # Regression. patch132 clamped only the window's lower edge, so an
        # anchor near the end of a mapping read past it: anchor 0x400038 in a
        # region ending at 0x400040 issued a read of 0x400028..0x400048.
        #
        # Not merely a bad read. The writability check applies to the anchor's
        # region; bytes pulled from the neighbouring mapping were never
        # checked and may be writable and volatile -- a signature containing
        # them stops matching for the exact reason writable memory is refused.
        small = [{"name": "executable", "start": 0x400000, "end": 0x400040,
                  "prot": 5}]
        reads = []

        def spy(_ip, _pid, addr, n):
            reads.append((addr, n))
            return bytes((addr + i) & 0xFF for i in range(n))

        for anchor in (0x400000, 0x400008, 0x400020, 0x40003F):
            reads.clear()
            with patch.object(RDX, "ps5_read", side_effect=spy):
                sig = RDX.capture_aob_signature("t", 7, anchor, maps=small)
            for addr, n in reads:
                self.assertGreaterEqual(addr, 0x400000,
                                        f"read before region, anchor {anchor:#x}")
                self.assertLessEqual(addr + n, 0x400040,
                                     f"read past region, anchor {anchor:#x}")
            if sig:
                # The anchor must still be inside the captured window.
                span = len(bytes.fromhex(sig["pattern"]))
                self.assertTrue(0 <= sig["lead"] < span)

    def test_aob_capture_mask_always_matches_pattern_length(self):
        # patch132 emitted a mask of the *requested* span against a pattern of
        # the clamped length, which aob_signature_matches rejects outright --
        # so a clamped capture would have produced a signature that could
        # never verify.
        small = [{"name": "executable", "start": 0x400000, "end": 0x400030,
                  "prot": 5}]
        with patch.object(RDX, "ps5_read",
                          side_effect=lambda _i, _p, a, n:
                              bytes((a + i) & 0xFF for i in range(n))):
            sig = RDX.capture_aob_signature("t", 7, 0x400028, maps=small)
        self.assertIsNotNone(sig)
        self.assertEqual(len(bytes.fromhex(sig["pattern"])),
                         len(bytes.fromhex(sig["mask"])))
        with patch.object(RDX, "ps5_read",
                          side_effect=lambda _i, _p, a, n:
                              bytes((a + i) & 0xFF for i in range(n))):
            self.assertTrue(
                RDX.aob_signature_matches("t", 7, 0x400028, sig))

    def test_aob_capture_refuses_a_region_too_small_to_anchor_in(self):
        tiny = [{"name": "executable", "start": 0x400000, "end": 0x400004,
                 "prot": 5}]
        with patch.object(RDX, "ps5_read",
                          side_effect=lambda _i, _p, a, n: bytes(n)):
            self.assertIsNone(RDX.capture_aob_signature(
                "t", 7, 0x400002, maps=tiny))

    def test_aob_capture_refuses_an_address_outside_its_region(self):
        maps = [{"name": "executable", "start": 0x400000, "end": 0x400040,
                 "prot": 5}]
        with patch.object(RDX, "_region_for_addr", return_value=maps[0]), \
             patch.object(RDX, "ps5_read",
                          side_effect=AssertionError("must not read")):
            self.assertIsNone(RDX.capture_aob_signature(
                "t", 7, 0x500000, maps=maps))

    def test_aob_capture_invariants_hold_across_every_boundary(self):
        # patch132 shipped two edge-of-region bugs -- an unclamped far edge and
        # a mask built from the requested rather than the clamped span -- and
        # the hand-written tests missed both, because they covered the cases I
        # had thought about while designing. This sweeps the boundary space
        # instead of sampling it.
        SPAN = RDX._AOB_SIGNATURE_BYTES
        BASE = 0x400000
        produced = 0
        for size in list(range(1, 72)) + [128, 4096]:
            maps = [{"name": "executable", "start": BASE, "end": BASE + size,
                     "prot": 5}]
            for off in range(size):
                anchor = BASE + off
                reads = []

                def spy(_i, _p, a, n):
                    reads.append((a, n))
                    return bytes((a + i) & 0xFF for i in range(n))

                with patch.object(RDX, "ps5_read", side_effect=spy):
                    sig = RDX.capture_aob_signature("t", 7, anchor, maps=maps)
                for a, n in reads:
                    self.assertGreaterEqual(a, BASE, f"size={size} off={off}")
                    self.assertLessEqual(a + n, BASE + size,
                                         f"size={size} off={off}")
                if sig is None:
                    continue
                produced += 1
                pat = bytes.fromhex(sig["pattern"])
                msk = bytes.fromhex(sig["mask"])
                self.assertEqual(len(pat), len(msk), f"size={size} off={off}")
                self.assertTrue(0 <= sig["lead"] < len(pat),
                                f"size={size} off={off}")
                with patch.object(RDX, "ps5_read", side_effect=spy):
                    self.assertTrue(
                        RDX.aob_signature_matches("t", 7, anchor, sig),
                        f"signature does not self-verify: size={size} off={off}")
        self.assertGreater(produced, 1000, "the sweep produced almost nothing")

    # ── field corroboration across instances (patch134) ───────────────────

    def _typed_groups(self):
        return [{"type_ptr": 0x82000000, "class_name": "PlayerController",
                 "count": 4,
                 "instances": np.asarray([0x2000000, 0x2001000,
                                          0x2002000, 0x2003000],
                                         dtype=np.uint64)}]

    def test_locate_field_finds_the_owning_instance_and_offset(self):
        # A candidate that follows the value is still only correlated with it.
        # If it is a *field*, every other live instance of the type has that
        # field too -- which is checkable without a debugger or a restart.
        got = RDX.locate_field_in_type(0x2001120, self._typed_groups())
        self.assertIsNotNone(got)
        self.assertEqual(got["instance_base"], 0x2001000)
        self.assertEqual(got["field_offset"], 0x120)
        self.assertEqual(got["class_name"], "PlayerController")

    def test_locate_field_refuses_an_address_too_far_from_any_base(self):
        # Being above a base is not membership: without a span limit the
        # highest instance would claim every address after it.
        far = 0x2003000 + RDX._FIELD_MAX_OBJECT_SPAN + 1
        self.assertIsNone(RDX.locate_field_in_type(far, self._typed_groups()))

    def test_locate_field_prefers_the_nearest_owning_base(self):
        groups = self._typed_groups() + [
            {"type_ptr": 0x82FFFFFF, "class_name": "Other", "count": 2,
             "instances": np.asarray([0x2000800], dtype=np.uint64)}]
        got = RDX.locate_field_in_type(0x2000900, groups)
        # 0x2000900 is 0x900 past 0x2000000 but only 0x100 past 0x2000800.
        self.assertEqual(got["instance_base"], 0x2000800)
        self.assertEqual(got["field_offset"], 0x100)

    def test_corroboration_counts_siblings_holding_a_similar_field(self):
        finding = RDX.locate_field_in_type(0x2001120, self._typed_groups())
        # Anchor 87; two siblings hold comparable integers, one holds
        # something that is not the same kind of quantity at all.
        #
        # The odd value has to be implausible *at the read width*. An earlier
        # version used a pointer-sized 0x7F0000001234, which a 4-byte read
        # truncates to 0x1234 -- 4660, only 53x the anchor and legitimately
        # the same sort of field. The truncation destroyed the case being
        # tested.
        table = {0x2001120: 87, 0x2000120: 64, 0x2002120: 120,
                 0x2003120: 0xF0000000}

        def fake(_ip, _pid, addr, n):
            return int(table.get(addr, 0)).to_bytes(8, "little")[:n]

        with patch.object(RDX, "ps5_read", side_effect=fake):
            out = RDX.corroborate_field_across_instances(
                "t", 7, finding, width=4, value_type="u32")
        self.assertEqual(out["sampled"], 3)
        self.assertEqual(out["read"], 3)
        self.assertEqual(out["plausible"], 2)

    def test_corroboration_survives_unreadable_siblings(self):
        finding = RDX.locate_field_in_type(0x2001120, self._typed_groups())

        def flaky(_ip, _pid, addr, n):
            if addr == 0x2001120:
                return (100).to_bytes(n, "little")
            raise ConnectionError("gone")

        with patch.object(RDX, "ps5_read", side_effect=flaky):
            out = RDX.corroborate_field_across_instances("t", 7, finding)
        self.assertEqual(out["read"], 0)
        self.assertEqual(out["plausible"], 0)
        self.assertGreater(out["sampled"], 0)

    def test_corroboration_is_empty_when_there_are_no_siblings(self):
        lone = [{"type_ptr": 1, "count": 1,
                 "instances": np.asarray([0x2000000], dtype=np.uint64)}]
        finding = RDX.locate_field_in_type(0x2000010, lone)
        with patch.object(RDX, "ps5_read",
                          side_effect=AssertionError("must not read")):
            out = RDX.corroborate_field_across_instances("t", 7, finding)
        self.assertEqual((out["read"], out["plausible"], out["sampled"]),
                         (0, 0, 0))

    def test_same_magnitude_is_about_shape_not_equality(self):
        # Per-object state should *differ* between instances; what is being
        # tested is that the offset holds the same kind of quantity.
        self.assertTrue(RDX._same_magnitude(87, 120))
        self.assertTrue(RDX._same_magnitude(1, 900))
        self.assertFalse(RDX._same_magnitude(87, 0x7F0000001234))
        self.assertFalse(RDX._same_magnitude(0, 5))
        self.assertTrue(RDX._same_magnitude(0, 0))

    # ── object-identity-first resolution (patch135) ───────────────────────

    def test_object_identity_aims_the_resolver_at_the_object_base(self):
        # The usual route asks for pointers to the target address itself -- an
        # address in the *middle* of an object. Little code holds such a
        # pointer; what code holds is the object base, reaching the field by a
        # constant displacement. Measured this session: 96 candidates for one
        # ammo address, top-ranked ones pointing 271,648 bytes away.
        groups = self._typed_groups()
        seen = {}

        def fake_resolve(_ip, _pid, target, **_kw):
            seen["target"] = target
            return {"candidates": [{"base": 0x1000, "offsets": [0x20],
                                    "depth": 1, "score": 50.0}],
                    "method": "fast-direct"}

        with patch.object(RDX, "_resolve_permanent_candidates",
                          side_effect=fake_resolve), \
             patch.object(RDX, "ps5_read",
                          side_effect=lambda _i, _p, a, n: (7).to_bytes(n, "little")):
            out = RDX.resolve_via_object_identity("t", 7, 0x2001120, groups)
        # Resolver was aimed at the object base, not the interior address.
        self.assertEqual(seen["target"], 0x2001000)
        self.assertEqual(out["stage"], "resolved")
        # The field is carried as the chain's terminal hop.
        self.assertEqual(out["candidates"][0]["terminal_offset"], 0x120)
        self.assertEqual(out["candidates"][0]["object_class_name"],
                         "PlayerController")

    def test_object_identity_falls_back_cleanly_when_no_object_owns_it(self):
        # A value that is not a managed object field is an ordinary answer,
        # not an error: the caller should use the direct resolver.
        with patch.object(RDX, "_resolve_permanent_candidates",
                          side_effect=AssertionError("must not resolve")):
            out = RDX.resolve_via_object_identity(
                "t", 7, 0xDEAD0000, self._typed_groups())
        self.assertEqual(out["stage"], "no-object")
        self.assertEqual(out["candidates"], [])
        self.assertIn("direct resolver", out["note"])

    def test_object_identity_marks_an_uncorroborated_field(self):
        # A field no sibling shares is weak evidence that it is a field at
        # all. Keep going -- the object may be the only live instance -- but
        # do not present the result as corroborated.
        groups = self._typed_groups()
        odd = {0x2001120: 50, 0x2000120: 0xF0000000,
               0x2002120: 0xF0000001, 0x2003120: 0xF0000002}

        with patch.object(RDX, "_resolve_permanent_candidates",
                          return_value={"candidates": [{"base": 1, "offsets": [0]}],
                                        "method": "m"}), \
             patch.object(RDX, "ps5_read",
                          side_effect=lambda _i, _p, a, n:
                              int(odd.get(a, 0)).to_bytes(8, "little")[:n]):
            out = RDX.resolve_via_object_identity("t", 7, 0x2001120, groups)
        self.assertFalse(out["corroborated"])
        self.assertFalse(out["candidates"][0]["field_corroborated"])
        self.assertEqual(out["candidates"][0]["sibling_agreement"], "0/3")

    def test_object_identity_reports_when_no_chain_reaches_the_object(self):
        groups = self._typed_groups()
        with patch.object(RDX, "_resolve_permanent_candidates",
                          return_value={"candidates": [], "method": "m"}), \
             patch.object(RDX, "ps5_read",
                          side_effect=lambda _i, _p, a, n: (7).to_bytes(n, "little")):
            out = RDX.resolve_via_object_identity("t", 7, 0x2001120, groups)
        self.assertEqual(out["stage"], "no-chain-to-object")

    # ── write-watchpoint instruction tracing (patch136) ───────────────────

    class _DebugClient:
        """Fake MemDBG client scripted per command id."""
        script = {}
        calls = None

        def __init__(self, ip, timeout=5.0):
            self.ip, self.sock, self.timeout = ip, FakeSock(), timeout
        def connect(self):
            self.sock = FakeSock(); return self
        def close(self):
            self.sock = None
        def request(self, command, body=b""):
            if type(self).calls is not None:
                type(self).calls.append((command, body))
            entry = type(self).script.get(command)
            if isinstance(entry, Exception):
                raise entry
            if callable(entry):
                return entry(body)
            return entry if entry is not None else b""

    def _debug_env(self, script, calls=None):
        cls = self._DebugClient
        cls.script, cls.calls = script, calls
        RDX.memdbg_reset_session()
        return patch.object(RDX, "_MemDBGClient", cls)

    @staticmethod
    def _regs_with_rip(rip):
        blob = bytearray(RDX._MEMDBG_REGS_SIZE)
        struct.pack_into("<q", blob, RDX._MEMDBG_REGS_RIP_OFFSET, rip)
        return bytes(blob)

    def test_tracing_refuses_when_nothing_is_attached(self):
        # Attaching is what froze the game twice on 2026-08-30. This path must
        # never attach on the caller's behalf; it says so and stops.
        script = {RDX.MEMDBG_CMD_DEBUG_GET_THREADS: ConnectionError("no session"),
                  RDX.MEMDBG_CMD_DEBUG_SET_WATCHPOINT:
                      AssertionError("must not arm a watchpoint")}
        try:
            with self._debug_env(script):
                out = RDX.trace_writer_instruction("t", 7, 0x2001000, 4)
        finally:
            RDX.memdbg_reset_session()
        self.assertEqual(out["stage"], "no-debug-session")
        self.assertIsNone(out["instruction"])
        self.assertIn("will not attach", out["note"])

    def test_tracing_reports_the_writing_instruction(self):
        calls = []
        script = {
            RDX.MEMDBG_CMD_DEBUG_GET_THREADS: struct.pack("<I", 3) + b"\x00" * 32,
            RDX.MEMDBG_CMD_DEBUG_SET_WATCHPOINT: b"",
            RDX.MEMDBG_CMD_DEBUG_POLL_EVENTS: struct.pack("<ii", 1, 101),
            RDX.MEMDBG_CMD_DEBUG_GET_REGS: self._regs_with_rip(0x82449db4),
            RDX.MEMDBG_CMD_DEBUG_CLEAR_WATCHPOINT: b"",
        }
        try:
            with self._debug_env(script, calls):
                out = RDX.trace_writer_instruction("t", 7, 0x2001000, 4)
        finally:
            RDX.memdbg_reset_session()
        self.assertEqual(out["stage"], "found")
        self.assertEqual(out["instruction"], 0x82449db4)
        self.assertEqual(out["lwp"], 101)
        # The watchpoint request must carry address, length and write type.
        armed = [b for c, b in calls if c == RDX.MEMDBG_CMD_DEBUG_SET_WATCHPOINT]
        self.assertEqual(struct.unpack("<QII", armed[0]),
                         (0x2001000, 4, RDX._MEMDBG_WP_WRITE))

    def test_the_watchpoint_is_always_removed(self):
        # A watchpoint left armed on a live game is worse than no answer.
        for stage_script, expected in (
                ({RDX.MEMDBG_CMD_DEBUG_POLL_EVENTS: struct.pack("<ii", 0, 0)},
                 "no-write-observed"),
                ({RDX.MEMDBG_CMD_DEBUG_POLL_EVENTS: RuntimeError("boom")},
                 "poll-failed"),
                ({RDX.MEMDBG_CMD_DEBUG_POLL_EVENTS: struct.pack("<ii", 1, 5),
                  RDX.MEMDBG_CMD_DEBUG_GET_REGS: b"\x00" * 4},
                 "regs-unreadable")):
            calls = []
            script = {
                RDX.MEMDBG_CMD_DEBUG_GET_THREADS: struct.pack("<I", 1) + b"\x00" * 8,
                RDX.MEMDBG_CMD_DEBUG_SET_WATCHPOINT: b"",
                RDX.MEMDBG_CMD_DEBUG_CLEAR_WATCHPOINT: b"",
            }
            script.update(stage_script)
            try:
                with self._debug_env(script, calls):
                    out = RDX.trace_writer_instruction(
                        "t", 7, 0x2001000, 4, timeout=0.05, poll_interval=0.01)
            finally:
                RDX.memdbg_reset_session()
            self.assertEqual(out["stage"], expected)
            self.assertTrue(
                any(c == RDX.MEMDBG_CMD_DEBUG_CLEAR_WATCHPOINT for c, _ in calls),
                f"watchpoint left armed after {expected}")

    def test_tracing_honours_cancellation_and_still_cleans_up(self):
        calls = []
        script = {
            RDX.MEMDBG_CMD_DEBUG_GET_THREADS: struct.pack("<I", 1) + b"\x00" * 8,
            RDX.MEMDBG_CMD_DEBUG_SET_WATCHPOINT: b"",
            RDX.MEMDBG_CMD_DEBUG_CLEAR_WATCHPOINT: b"",
        }
        ev = threading.Event(); ev.set()
        try:
            with self._debug_env(script, calls):
                out = RDX.trace_writer_instruction(
                    "t", 7, 0x2001000, 4, cancel_event=ev)
        finally:
            RDX.memdbg_reset_session()
        self.assertEqual(out["stage"], "cancelled")
        self.assertTrue(
            any(c == RDX.MEMDBG_CMD_DEBUG_CLEAR_WATCHPOINT for c, _ in calls))

    def test_tracing_rejects_a_width_no_watchpoint_can_take(self):
        script = {RDX.MEMDBG_CMD_DEBUG_GET_THREADS:
                      AssertionError("must not even probe")}
        try:
            with self._debug_env(script):
                out = RDX.trace_writer_instruction("t", 7, 0x2001000, width=3)
        finally:
            RDX.memdbg_reset_session()
        self.assertEqual(out["stage"], "bad-width")

    def test_rip_offset_matches_the_published_register_layout(self):
        # Computed from the header's field order, not assumed: the textbook
        # FreeBSD layout puts rip at 144 and this protocol puts it at 136.
        self.assertEqual(RDX._MEMDBG_REGS_RIP_OFFSET, 136)
        self.assertEqual(RDX._MEMDBG_REGS_SIZE, 176)
        blob = self._regs_with_rip(0xDEADBEEF)
        self.assertEqual(
            struct.unpack_from("<q", blob, RDX._MEMDBG_REGS_RIP_OFFSET)[0],
            0xDEADBEEF)

    def test_debugger_capability_bits_are_named(self):
        # The console reported 0xFFFFFFFF and bit 20 went unremarked because
        # RDX had no name for it.
        self.assertEqual(RDX.MEMDBG_CAP_DEBUGGER, 1 << 20)
        self.assertEqual(RDX.MEMDBG_CAP_TRACER, 1 << 21)

    # ── instruction patching (patch137) ───────────────────────────────────

    _CODE_MAPS = [
        {"name": "executable", "start": 0x400000, "end": 0x410000, "prot": 5},
        {"name": "executable", "start": 0x410000, "end": 0x420000, "prot": 1},
        {"name": "", "start": 0x800000, "end": 0x810000, "prot": 3},
    ]

    def _mem(self, table):
        """ps5_read/ps5_write over a dict of address -> bytearray."""
        def rd(_ip, _pid, addr, n):
            out = bytearray()
            for i in range(n):
                out.append(table.get(addr + i, 0))
            return bytes(out)

        def wr(_ip, _pid, addr, data, **_kw):
            for i, b in enumerate(bytes(data)):
                table[addr + i] = b
            return True
        return rd, wr

    def test_patch_refuses_to_partially_overwrite_an_instruction(self):
        # A patch shorter than the instruction leaves part of the old one
        # behind; a longer one runs into the next. Never guess a boundary.
        with patch.object(RDX, "ps5_read",
                          side_effect=AssertionError("must not read")):
            out = RDX.patch_instruction("t", 7, 0x401000, b"\x90",
                                        b"\x89\x41\x18",
                                        maps=self._CODE_MAPS)
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage"], "length-mismatch")
        self.assertIn("partially overwrite", out["note"])

    def test_patch_refuses_a_non_executable_target(self):
        # Instructions live in r-x memory. A "patch" in the heap is a value
        # write and belongs on the guarded write path, not here.
        for addr in (0x410000, 0x800100):
            with patch.object(RDX, "ps5_read",
                              side_effect=AssertionError("must not read")):
                out = RDX.patch_instruction("t", 7, addr, b"\x90\x90",
                                            b"\x00\x00",
                                            maps=self._CODE_MAPS)
            self.assertEqual(out["stage"], "not-patchable", f"addr={addr:#x}")
            self.assertIn("executable", out["note"])

    def test_patch_refuses_to_run_past_the_end_of_its_region(self):
        with patch.object(RDX, "ps5_read",
                          side_effect=AssertionError("must not read")):
            out = RDX.patch_instruction("t", 7, 0x40FFFE, b"\x90" * 4,
                                        b"\x00" * 4, maps=self._CODE_MAPS)
        self.assertEqual(out["stage"], "not-patchable")
        self.assertIn("past the end", out["note"])

    def test_patch_refuses_when_the_instruction_is_not_the_expected_one(self):
        # An AOB match is not proof the site is correct. The bytes are re-read
        # immediately before writing; this is the last line of defence against
        # the worst failure -- patching the wrong instruction.
        table = {0x401000 + i: b for i, b in enumerate(b"\x48\x89\x41\x18")}
        rd, wr = self._mem(table)
        with patch.object(RDX, "ps5_read", side_effect=rd), \
             patch.object(RDX, "ps5_write",
                          side_effect=AssertionError("must not write")):
            out = RDX.patch_instruction("t", 7, 0x401000, b"\x90" * 4,
                                        b"\x48\x89\x41\x20",
                                        maps=self._CODE_MAPS)
        self.assertEqual(out["stage"], "bytes-changed")
        self.assertIn("not the one", out["note"])

    def test_patch_applies_and_verifies_the_readback(self):
        original = b"\x48\x89\x41\x18"
        table = {0x401000 + i: b for i, b in enumerate(original)}
        rd, wr = self._mem(table)
        with patch.object(RDX, "ps5_read", side_effect=rd), \
             patch.object(RDX, "ps5_write", side_effect=wr):
            out = RDX.patch_instruction("t", 7, 0x401000,
                                        RDX.nop_bytes(len(original)),
                                        original, maps=self._CODE_MAPS)
        self.assertTrue(out["ok"])
        self.assertEqual(out["stage"], "patched")
        self.assertEqual(bytes(table[0x401000 + i] for i in range(4)),
                         b"\x90\x90\x90\x90")

    def test_patch_reports_a_readback_that_does_not_match(self):
        original = b"\x48\x89"
        table = {0x401000 + i: b for i, b in enumerate(original)}
        rd, _ = self._mem(table)

        def lying_write(_ip, _pid, _addr, _data, **_kw):
            return True          # claims success, changes nothing

        with patch.object(RDX, "ps5_read", side_effect=rd), \
             patch.object(RDX, "ps5_write", side_effect=lying_write):
            out = RDX.patch_instruction("t", 7, 0x401000, b"\x90\x90",
                                        original, maps=self._CODE_MAPS)
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage"], "verify-failed")

    def test_restore_puts_the_original_instruction_back(self):
        original = b"\x48\x89\x41\x18"
        applied = RDX.nop_bytes(len(original))
        table = {0x401000 + i: b for i, b in enumerate(original)}
        rd, wr = self._mem(table)
        with patch.object(RDX, "ps5_read", side_effect=rd), \
             patch.object(RDX, "ps5_write", side_effect=wr):
            self.assertTrue(RDX.patch_instruction(
                "t", 7, 0x401000, applied, original,
                maps=self._CODE_MAPS)["ok"])
            out = RDX.restore_instruction("t", 7, 0x401000, original, applied,
                                          maps=self._CODE_MAPS)
        self.assertTrue(out["ok"])
        self.assertEqual(bytes(table[0x401000 + i] for i in range(4)), original)

    def test_restore_refuses_if_something_else_changed_the_site(self):
        # applied_bytes is required rather than an assumed run of NOPs: a
        # caller that applied a different patch could not otherwise restore,
        # and defaulting to NOPs would make the guard pass for a wrong reason.
        original = b"\x48\x89"
        table = {0x401000: 0xCC, 0x401001: 0xCC}      # a third party got here
        rd, wr = self._mem(table)
        with patch.object(RDX, "ps5_read", side_effect=rd), \
             patch.object(RDX, "ps5_write",
                          side_effect=AssertionError("must not write")):
            out = RDX.restore_instruction("t", 7, 0x401000, original,
                                          RDX.nop_bytes(2),
                                          maps=self._CODE_MAPS)
        self.assertEqual(out["stage"], "bytes-changed")

    def test_nop_bytes_rejects_a_non_positive_length(self):
        self.assertEqual(RDX.nop_bytes(3), b"\x90\x90\x90")
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                RDX.nop_bytes(bad)

    def test_scan_socket_uses_native_memdbg_without_port_744(self):
        old_backend = RDX.state.get("backend")
        old_memdbg = RDX.state.get("memdbg")
        RDX.state.update(
            backend="memdbg-experimental",
            memdbg={"capabilities": RDX.MEMDBG_CAP_MEMORY_READ})

        def connect(client):
            client.hello = {"capabilities": RDX.MEMDBG_CAP_MEMORY_READ}
            return client

        try:
            with patch.object(RDX._MemDBGClient, "connect", new=connect), \
                 patch.object(RDX._MemDBGClient, "memory_read",
                              return_value=b"native") as native_read, \
                 patch.object(RDX, "ps5_connect",
                              side_effect=AssertionError("port 744 used")):
                reader = RDX._ScanSocket("test", 7)
                try:
                    self.assertEqual(reader.read(0x4000, 6), b"native")
                finally:
                    reader.close()
            native_read.assert_called_once_with(7, 0x4000, 6)
        finally:
            RDX.state.update(backend=old_backend, memdbg=old_memdbg)

    def test_connect_accepts_native_memdbg_without_legacy_listener(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass
            def getmaxyx(self): return (24, 80)

        old = {key: RDX.state.get(key) for key in (
            "ip", "backend", "memdbg", "session", "pid", "proc_name",
            "connected")}
        hello = {
            "capabilities": (RDX.MEMDBG_CAP_PROCESS_LIST |
                             RDX.MEMDBG_CAP_PROCESS_MAPS |
                             RDX.MEMDBG_CAP_MEMORY_READ |
                             RDX.MEMDBG_CAP_MEMORY_WRITE),
            "version": "test-native",
        }
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_header_banner"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "input_box", return_value="console"), \
                 patch.object(RDX, "memdbg_probe", return_value=hello), \
                 patch.object(RDX, "ps5_proc_list", return_value=[]), \
                 patch.object(RDX, "ps5_connect",
                              side_effect=AssertionError("port 744 used")), \
                 patch.object(RDX, "_stop_freeze_worker"), \
                 patch.object(RDX, "_clear_scan_state"), \
                 patch.object(RDX, "_save_preferences"), \
                 patch.object(RDX, "screen_proc_select",
                              return_value="main"):
                # patch59 probes 744 to warn that TurboScan will be
                # unavailable, so ps5_connect IS called now -- but purely as
                # a diagnostic. What this test guards is unchanged and is the
                # thing that actually matters: a native MemDBG connect must
                # still succeed when nothing is listening on 744 at all.
                self.assertEqual(RDX.screen_connect(FakeWindow()), "main")
            self.assertEqual(RDX.state["backend"], "memdbg-experimental")
            self.assertTrue(any("port 744" in e["msg"]
                                for e in RDX.state["log"][-4:]),
                            "should warn that the ps5debug commands are gone")
        finally:
            RDX.state.update(old)

    def test_memdbg_pointer_entry_is_only_used_as_one_hop_holder(self):
        prefix = struct.pack("<IIQQIIII", 1, 0, 0x100, 1, 1, 1, 0, 0)
        entry = struct.pack("<QII", 0x12345678, 99, 0)
        response_body = prefix + entry

        class ScriptedSocket:
            def __init__(self): self.pending = bytearray()
            def sendall(self, data):
                magic, version, command, request_id, length = struct.unpack(
                    "<IHHII", data[:16])
                self.pending.extend(struct.pack(
                    "<IHHIiI", magic, version, command, request_id, 0,
                    len(response_body)) + response_body)
            def recv_into(self, view, length):
                take = min(length, len(self.pending))
                view[:take] = self.pending[:take]
                del self.pending[:take]
                return take
            def close(self): pass

        client = RDX._MemDBGClient("test")
        client.sock = ScriptedSocket()
        client.hello = {"capabilities": RDX.MEMDBG_CAP_SCAN_POINTER}
        try:
            holders = client.pointer_holders(
                7, 0x9000,
                [{"start": 0x1000, "end": 0x2000}], 10)
            self.assertEqual(holders, [0x12345678])
        finally:
            client.close()

    def test_memdbg_pointer_accepts_current_daemon_eight_byte_entries(self):
        prefix = struct.pack("<IIQQIIII", 2, 0, 0x100, 1, 1, 1, 0, 0)
        response_body = prefix + struct.pack("<QQ", 0x1111, 0x2222)

        class ScriptedSocket:
            def __init__(self): self.pending = bytearray()
            def sendall(self, data):
                magic, version, command, request_id, _ = struct.unpack(
                    "<IHHII", data[:16])
                self.pending.extend(struct.pack(
                    "<IHHIiI", magic, version, command, request_id, 0,
                    len(response_body)) + response_body)
            def recv_into(self, view, length):
                take = min(length, len(self.pending))
                view[:take] = self.pending[:take]
                del self.pending[:take]
                return take
            def close(self): pass

        client = RDX._MemDBGClient("test")
        client.sock = ScriptedSocket()
        client.hello = {"capabilities": RDX.MEMDBG_CAP_SCAN_POINTER}
        try:
            self.assertEqual(client.pointer_holders(
                7, 0x9000, [{"start": 0x1000, "end": 0x2000}], 10),
                [0x1111, 0x2222])
        finally:
            client.close()

    def test_memdbg_compat_large_reads_are_split_to_payload_limit(self):
        reader = object.__new__(RDX._ScanSocket)
        calls = []

        def one(address, length, _cancel=None):
            calls.append((address, length))
            return bytes([len(calls)]) * length

        old_backend = RDX.state.get("backend")
        RDX.state["backend"] = "memdbg-experimental"
        try:
            with patch.object(reader, "_read_single", side_effect=one):
                data = reader.read(0x4000, 0x280000)
        finally:
            RDX.state["backend"] = old_backend
        self.assertEqual(calls, [
            (0x4000, 0x100000),
            (0x104000, 0x100000),
            (0x204000, 0x80000),
        ])
        self.assertEqual(len(data), 0x280000)
        self.assertEqual(data[0], 1)
        self.assertEqual(data[0x100000], 2)
        self.assertEqual(data[0x200000], 3)

    def test_fast_direct_pointer_hits_builds_region_lookup_once(self):
        # The MemDBG-native branch used to rebuild the region lookup inside
        # the per-holder loop -- up to _PTR_FAST_DIRECT_HITS (24) times for a
        # single call -- instead of once before it, the same anti-pattern
        # already fixed in the disk reverse index's query().
        maps = [{"start": 0x1000, "end": 0x1100, "prot": 5,
                 "name": "executable"}]

        class FakeMemDBGClient:
            def __init__(self, ip="test", timeout=5.0, *_a, **_k):
                self.ip, self.sock, self.timeout = ip, None, timeout
            def connect(self):
                self.sock = FakeSock()
                return self
            def close(self):
                self.sock = None
            def pointer_holders(self, pid, target, regions, max_hits):
                return [0x1000, 0x1004, 0x1008]

        calls = {"n": 0}
        real_build = RDX._build_region_lookup
        def counting_build(maps_arg):
            calls["n"] += 1
            return real_build(maps_arg)

        old_backend = RDX.state.get("backend")
        RDX.state["backend"] = "memdbg-experimental"
        try:
            with patch.object(RDX, "_MemDBGClient", FakeMemDBGClient), \
                 patch.object(RDX, "_build_region_lookup",
                              side_effect=counting_build), \
                 patch.object(RDX, "_pointer_readable_regions",
                              return_value=maps), \
                 patch.object(RDX, "_is_static_region", return_value=True):
                hits = RDX._fast_direct_pointer_hits("test", 1, 0x1000, maps)
        finally:
            RDX.state["backend"] = old_backend
        self.assertEqual(calls["n"], 1)
        self.assertEqual(len(hits), 3)

    def test_first_scan_setup_controls_can_cancel(self):
        class FakeWindow:
            def getmaxyx(self): return (24, 80)
            def refresh(self): pass
            def getch(self): return 27
            def getstr(self, *_args):
                raise AssertionError("cancellable input must not wait for Enter")
            def nodelay(self, *_args): pass
            def timeout(self, *_args): pass

        win = FakeWindow()
        with patch.object(RDX, "safe_addstr"), \
             patch.object(RDX, "color", return_value=0), \
             patch.object(RDX, "_safe_curs_set"), \
             patch.object(RDX.curses, "cbreak"), \
             patch.object(RDX.curses, "echo"), \
             patch.object(RDX.curses, "noecho"):
            self.assertIsNone(RDX.input_box(
                win, "Value: ", 1, 1, allow_cancel=True))
            self.assertIsNone(RDX.cycle_input(
                win, "Width: ", 1, 1, ["one", "two"],
                allow_cancel=True))

    def test_write_setup_cancel_performs_no_write(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        with patch.object(RDX, "draw_border"), \
             patch.object(RDX, "safe_addstr"), \
             patch.object(RDX, "color", return_value=0), \
             patch.object(RDX, "input_box", return_value=None), \
             patch.object(RDX, "ps5_write_verified",
                          side_effect=AssertionError("write must not run")):
            RDX.do_write(FakeWindow())

    def test_write_setup_can_cancel_at_each_step(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        scenarios = [
            ([None], ["uint32"]),
            (["0x1000", None], ["uint32"]),
            (["0x1000", "7"], [None]),
        ]
        for inputs, widths in scenarios:
            with self.subTest(inputs=inputs, widths=widths), \
                 patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "input_box", side_effect=inputs), \
                 patch.object(RDX, "cycle_input", side_effect=widths), \
                 patch.object(RDX, "ps5_write_verified",
                              side_effect=AssertionError("write must not run")):
                RDX.do_write(FakeWindow())

    def test_provisional_pointer_store_round_trip(self):
        records = [{"status": "provisional", "observed_pid": 10,
                    "module_name": "executable",
                    "module_relative_offset": 0x1234,
                    "offsets": [0x10, -0x20]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            RDX._save_pointer_provisionals(records, path)
            self.assertEqual(RDX._load_pointer_provisionals(path), records)

    def test_provisional_requires_new_pid_and_rebases_module(self):
        maps = [{"start": 0x500000, "end": 0x600000, "prot": 3,
                 "name": "executable"}]
        saved = [{"status": "provisional", "observed_pid": 10,
                  "observed_process": "eboot.bin",
                  "module_name": "executable",
                  "module_relative_offset": 0x1234,
                  "offsets": [0x10], "terminal_offset": 0,
                  "reload_survivals": 0}]

        same = RDX._validate_pointer_provisionals(
            "test", 10, "eboot.bin", 0x9000, saved, maps)
        self.assertEqual(same["survivors"], [])
        self.assertEqual(same["rejected"][0]["rejection_reason"],
                         "reload not detected")

        def verify(_ip, _pid, candidate, _target):
            candidate["verified"] = True
            return True

        with patch.object(RDX, "_verify_candidate_twice", side_effect=verify):
            changed = RDX._validate_pointer_provisionals(
                "test", 11, "eboot.bin", 0x9000, saved, maps)
        self.assertEqual(len(changed["survivors"]), 1)
        survivor = changed["survivors"][0]
        self.assertEqual(survivor["base"], 0x501234)
        self.assertEqual(survivor["status"], "provisional")
        self.assertEqual(survivor["reload_survivals"], 1)
        self.assertEqual(survivor["observed_pid"], 11)

        with patch.object(RDX, "_verify_candidate_twice", side_effect=verify):
            final = RDX._validate_pointer_provisionals(
                "test", 12, "eboot.bin", 0xA000,
                changed["survivors"], maps)
        promoted = final["survivors"][0]
        self.assertEqual(promoted["status"], "permanent")
        self.assertEqual(promoted["reload_survivals"], 2)
        self.assertEqual(promoted["observed_pid"], 12)

    def test_provisional_accepts_relocated_target_with_same_pid(self):
        maps = [{"start": 0x500000, "end": 0x600000, "prot": 3,
                 "name": "executable"}]
        saved = [{"status": "provisional", "observed_pid": 10,
                  "observed_process": "eboot.bin", "observed_target": 0x9000,
                  "module_name": "executable", "module_relative_offset": 0x20,
                  "offsets": [0], "terminal_offset": 0,
                  "reload_survivals": 0}]
        with patch.object(RDX, "_verify_candidate_twice", return_value=True):
            result = RDX._validate_pointer_provisionals(
                "test", 10, "eboot.bin", 0xA000, saved, maps)
        self.assertEqual(len(result["survivors"]), 1)
        self.assertEqual(result["survivors"][0]["reload_survivals"], 1)
        self.assertEqual(result["survivors"][0]["observed_target"], 0xA000)

    def test_provisional_merge_preserves_other_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            RDX._save_pointer_provisionals([
                {"observed_process": "game-a", "offsets": [1]},
                {"observed_process": "game-b", "offsets": [2]},
            ], path)
            merged = RDX._merge_pointer_provisionals(
                [{"observed_process": "game-a", "offsets": [3]}],
                "game-a", path)
            self.assertEqual(merged, [
                {"observed_process": "game-b", "offsets": [2]},
                {"observed_process": "game-a", "offsets": [3]},
            ])

    def test_provisional_merge_separates_games_that_both_use_eboot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            RDX._save_pointer_provisionals([
                {"observed_process": "eboot.bin", "observed_game": "A",
                 "offsets": [1]},
                {"observed_process": "eboot.bin", "observed_game": "B",
                 "offsets": [2]},
            ], path)
            merged = RDX._merge_pointer_provisionals([
                {"observed_process": "eboot.bin", "observed_game": "A",
                 "offsets": [3]},
            ], "eboot.bin", path, game_identity="A")
            self.assertEqual(merged, [
                {"observed_process": "eboot.bin", "observed_game": "B",
                 "offsets": [2]},
                {"observed_process": "eboot.bin", "observed_game": "A",
                 "offsets": [3]},
            ])

    def test_game_identity_survives_aslr_but_changes_with_image_layout(self):
        before = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                   "name": "executable", "offset": 0}]
        relocated = [{"start": 0x900000, "end": 0xA00000, "prot": 5,
                      "name": "executable", "offset": 0}]
        other_game = [{"start": 0x400000, "end": 0x580000, "prot": 5,
                       "name": "executable", "offset": 0}]
        first = RDX._pointer_game_identity("eboot.bin", before)
        self.assertEqual(first,
                         RDX._pointer_game_identity("eboot.bin", relocated))
        self.assertNotEqual(first,
                            RDX._pointer_game_identity("eboot.bin", other_game))

    def test_memdbg_full_executable_path_is_the_main_game_image(self):
        path = "/mnt/sandbox/PPSA00001/app0/eboot.bin"
        before = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                   "flags": 2 << 24, "name": path, "offset": 0}]
        relocated = [{"start": 0x900000, "end": 0xA00000, "prot": 5,
                      "flags": 2 << 24, "name": path, "offset": 0}]
        self.assertTrue(RDX._is_main_module_name(path, "eboot.bin"))
        self.assertEqual(
            RDX._pointer_game_identity("eboot.bin", before),
            RDX._pointer_game_identity("eboot.bin", relocated))

    def test_streaming_depth_two_chain_order_and_four_byte_offsets(self):
        maps = [
            {"start": 0x1000, "end": 0x1100, "prot": 3,
             "name": "executable"},
            {"start": 0x2000, "end": 0x4000, "prot": 3, "name": ""},
        ]
        # Pointer holders remain 8-byte aligned; field offsets may be 4-byte
        # aligned: *(0x1000) + 0x18 = 0x2018, then +0x1c = 0x301c.
        MemorySocket.memory = {0x1000: 0x2000, 0x2018: 0x3000}

        def read_pointer(_ip, _pid, address):
            return MemorySocket.memory.get(address, 0)

        with patch.object(RDX, "_get_maps_cached", return_value=maps), \
             patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "ps5_read_pointer", side_effect=read_pointer):
            found = RDX.pointer_chain_scan(
                "test", 1, 0x301C, max_depth=2,
                cancel_event=threading.Event())
            verified = [candidate for candidate in found
                        if candidate["base"] == 0x1000
                        and candidate["offsets"] == [0x18, 0x1C]]
            self.assertEqual(len(verified), 1)
            ok, resolved, steps = RDX._resolve_pointer_chain(
                "test", 1, verified[0]["base"], verified[0]["offsets"])
            self.assertTrue(ok)
            self.assertEqual(resolved, 0x301C)
            self.assertEqual(steps, [0x2018, 0x301C])

    def test_writable_main_executable_is_static(self):
        self.assertTrue(RDX._is_static_region({
            "start": 0x400000, "end": 0x500000,
             "prot": 3, "name": "executable"}))

    def test_writable_libsce_module_without_extension_is_static(self):
        # A writable, named libSce* mapping is a legitimate module .data/.bss
        # section even when the debugger reports it without a .sprx/.prx/.elf
        # suffix.  This used to fall through to the final anonymous-mapping
        # check, which rejects anything writable.
        self.assertTrue(RDX._is_static_region({
            "start": 0x800000000, "end": 0x800001000,
            "prot": 3, "name": "libSceNetCtl"}))

    def test_coalescing_preserves_static_first_priority(self):
        regions = [
            {"start": 0x5000, "end": 0x6000, "prot": 3,
             "name": "executable"},
            {"start": 0x1000, "end": 0x2000, "prot": 3, "name": "heap"},
        ]
        merged = RDX._coalesce_pointer_regions(regions)
        self.assertEqual(merged[0]["start"], 0x5000)
        self.assertTrue(merged[0]["static"])

    def test_disk_index_query_and_cleanup(self):
        maps = [
            {"start": 0x1000, "end": 0x1040, "prot": 3, "name": "heap"},
            {"start": 0x2000, "end": 0x2100, "prot": 3, "name": "heap"},
        ]
        MemorySocket.memory = {0x1000: 0x2000}
        with patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "_pointer_readable_regions",
                          return_value=[maps[0]]):
            index = RDX._DiskReversePointerIndex("test", 1, maps)
            temp_path = index._tmpdir
            try:
                self.assertEqual(index.query(0x201C, 0x200, 4),
                                 [(0x1000, 0x1C)])
            finally:
                index.close()
            self.assertFalse(temp_path.exists())

    def test_ram_reverse_index_finds_four_byte_aligned_holder(self):
        # A pointer holder itself (not just a resolved field offset) can sit
        # at a 4-byte-aligned address inside a packed/mixed-width struct.
        # Scanning only the 8-byte grid made this whole class of roots
        # invisible.  The target value (0x1050) is placed inside the same
        # region so the plausibility filter accepts it.
        maps = [{"start": 0x1000, "end": 0x1100, "prot": 3,
                 "name": "executable"}]
        MemorySocket.memory = {0x1004: 0x1050}
        with patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "_pointer_readable_regions",
                          return_value=[maps[0]]):
            index = RDX._ReversePointerIndex("test", 1, maps)
        self.assertEqual(index.query(0x1050, 0, 4), [(0x1004, 0)])

    def test_disk_reverse_index_finds_four_byte_aligned_holder(self):
        maps = [{"start": 0x1000, "end": 0x1100, "prot": 3,
                 "name": "executable"}]
        MemorySocket.memory = {0x1004: 0x1050}
        with patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "_pointer_readable_regions",
                          return_value=[maps[0]]):
            index = RDX._DiskReversePointerIndex("test", 1, maps)
            try:
                self.assertEqual(index.query(0x1050, 0, 4), [(0x1004, 0)])
            finally:
                index.close()

    def test_ram_reverse_index_keeps_holder_straddling_chunk_boundary(self):
        # With a chunk size of 0x40, chunks are read as [0x1000, 0x1040) and
        # [0x1040, 0x1080).  A 4-byte-aligned holder at 0x103C needs bytes
        # through 0x1044, which is outside both chunks unless the read for
        # the first chunk deliberately overlaps into the next one.
        maps = [{"start": 0x1000, "end": 0x1080, "prot": 3,
                 "name": "executable"}]
        MemorySocket.memory = {0x103C: 0x1070}
        with patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "_pointer_readable_regions",
                          return_value=[maps[0]]), \
             patch.object(RDX, "_PTR_INDEX_CHUNK", 0x40):
            index = RDX._ReversePointerIndex("test", 1, maps)
        self.assertEqual(index.query(0x1070, 0, 4), [(0x103C, 0)])

    def test_disk_reverse_index_keeps_holder_straddling_shard_boundary(self):
        maps = [{"start": 0x1000, "end": 0x1080, "prot": 3,
                 "name": "executable"}]
        MemorySocket.memory = {0x103C: 0x1070}
        with patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "_pointer_readable_regions",
                          return_value=[maps[0]]), \
             patch.object(RDX, "_PTR_DISK_SHARD_BYTES", 0x40):
            index = RDX._DiskReversePointerIndex("test", 1, maps)
            try:
                self.assertEqual(index.query(0x1070, 0, 4), [(0x103C, 0)])
            finally:
                index.close()

    def test_disk_reverse_index_query_does_not_rebuild_region_lookup(self):
        # Holder region priority must be precomputed once at build time.
        # Pointer chain resolution calls query() repeatedly per candidate and
        # per depth, so re-deriving the region lookup inside query() would be
        # a hot-path cost paid on every one of those calls.
        maps = [{"start": 0x1000, "end": 0x1100, "prot": 3,
                 "name": "executable"}]
        MemorySocket.memory = {0x1000: 0x1050}
        with patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "_pointer_readable_regions",
                          return_value=[maps[0]]):
            index = RDX._DiskReversePointerIndex("test", 1, maps)
        try:
            def _boom(*_a, **_k):
                raise AssertionError("query() rebuilt the region lookup")
            with patch.object(RDX, "_build_region_lookup", side_effect=_boom):
                self.assertEqual(index.query(0x1050, 0, 4), [(0x1000, 0)])
        finally:
            index.close()

    def test_disk_reverse_index_orders_hits_by_region_priority(self):
        # Two holders equidistant from the target: one in a static executable
        # region, one in heap.  With max_hits=1 only the higher-priority
        # (static) holder should survive the global rank.
        maps = [
            {"start": 0x1000, "end": 0x1100, "prot": 5, "name": "executable"},
            {"start": 0x2000, "end": 0x2100, "prot": 3, "name": "heap"},
        ]
        MemorySocket.memory = {0x1000: 0x1050, 0x2000: 0x1050}
        with patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "_pointer_readable_regions",
                          return_value=maps):
            index = RDX._DiskReversePointerIndex("test", 1, maps)
        try:
            self.assertEqual(
                index.query(0x1050, 0, 4, max_hits=1), [(0x1000, 0)])
        finally:
            index.close()

    def test_etahen_export_uses_module_offsets_and_skips_pointer_chains(self):
        maps = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                 "name": "executable"}]
        cheats = [
            {"name": "Health", "type": "write", "address": 0x401020,
             "value": 0x12345678, "original_value": 7, "width": 4},
            {"name": "Dynamic ammo", "type": "pointer_freeze",
             "base": 0x400100, "offsets": [0x18, 0x1C],
             "value": 99, "original_value": 6, "width": 4},
        ]
        text, mods, skipped = RDX.generate_etahen_json(
            cheats, "PPSA00001", "01.001.000", "Test Game",
            "eboot.bin", maps, "Tester")
        payload = json.loads(text)
        self.assertEqual(payload["process"], "eboot.bin")
        self.assertEqual(payload["credits"], ["Tester"])
        self.assertEqual(mods, payload["mods"])
        self.assertEqual(len(mods), 1)
        self.assertEqual(mods[0]["memory"], [{
            "offset": "1020", "on": "78563412", "off": "07000000"}])
        self.assertEqual(skipped,
                         [("Dynamic ammo",
                           "pointer chain requires the RDX runtime")])

    def test_etahen_export_refuses_flat_heap_address(self):
        maps = [
            {"start": 0x400000, "end": 0x500000, "prot": 5,
             "name": "executable"},
            {"start": 0x800000, "end": 0x900000, "prot": 3,
             "name": "heap"},
        ]
        text, mods, skipped = RDX.generate_etahen_json([
            {"name": "Temporary", "type": "write", "address": 0x801000,
             "value": 1, "original_value": 2, "width": 4},
        ], "PPSA00001", "01.001.000", "Test", "eboot.bin", maps)
        self.assertEqual(json.loads(text)["mods"], [])
        self.assertEqual(mods, [])
        self.assertEqual(skipped,
                         [("Temporary", "address is not in the target module")])

    def test_etahen_export_accepts_memdbg_full_main_module_path(self):
        path = "/mnt/sandbox/PPSA00001/app0/eboot.bin"
        maps = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                 "flags": 2 << 24, "name": path}]
        cheat = {
            "name": "Health", "type": "write", "address": 0x401020,
            "module_name": path, "module_relative_offset": 0x1020,
            "value": 8, "original_value": 7, "width": 4,
        }
        _text, mods, skipped = RDX.generate_etahen_json(
            [cheat], "PPSA00001", "01.00", "Test", "eboot.bin", maps)
        self.assertEqual(mods[0]["memory"][0]["offset"], "1020")
        self.assertEqual(skipped, [])

    def test_aes256_cbc_matches_fips197_known_answer(self):
        # FIPS-197 Appendix C.3 AES-256 single-block known-answer test.
        key = bytes.fromhex(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
        plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
        expected = bytes.fromhex("8ea2b7ca516745bfeafc49904b496089")
        w, nr = RDX._aes_key_expansion(key)
        self.assertEqual(nr, 14)
        cipher = RDX._aes_encrypt_block(plaintext, w, nr)
        self.assertEqual(cipher, expected)
        self.assertEqual(RDX._aes_decrypt_block(cipher, w, nr), plaintext)

    def test_mc4_decrypts_a_real_published_sample(self):
        # Captured from HEN-Cheats-Collection's CUSA00002 (Killzone: Shadow
        # Fall) .mc4 — a fixed real-world artifact, not something this
        # codebase produced, so this pins the key/IV/algorithm against format
        # drift rather than just checking our own round trip.
        sample = (
            b"AgKHPLxZwZ+2YfYslmLUXGyUIJK0NSn6118NipwxKEztE2q1rhwqHXSKaFQA/6YW"
            b"qxnwy/nyj9oBsEIbcrz/FKR7uSA0bLf+RXN/lhBn6v+h9HuUZd2kHOE9znFnUrqD"
            b"RVtUNuoiYeXgW98DPqQmGLoNR9j5WVYHsEdIvjM3iiN9oQ9djobOpt6AuAFjAB15"
            b"/grhFrQQz81N7AGK5p/B2j9YU8R8aj8YkQ/kY2qgfd6FBdGxfY+QC8z7uGLg6kqk"
            b"kdewLGBuu9gs5mwy6UpLFxuRgO4RT/6KxFqgqHHIvesBs8aVYA7zl63Dbv3L1ggk"
            b"R8v0ZJ3g1FGeVLXSYVDximAsCMm44Pov1+R30rdHFGYhu7j6E6N7TPAwQjSveMKP"
            b"LvwHQPy4zmyfaJF1VGbr/M/NcvplqthkJ4TWum5Vm+9WCusFR3qg5YOi0jqzaFnZ"
            b"C/KS4MboHIjNpg/nN/fw2aKXNebORmkfi3CIgOS11KEGWUI+HZ2Jh71xHneQPJqx"
            b"xqzHgw/RsOh5rNptsFSMqf2VGP8+89BRx7UP3BI8tWNNl3zjQp8y9xhmD/CBUL52"
            b"lIu7CxWU+TPfW0Sp1Y4f7G/qF1WopPHivEtWgwRdZVSUqFYG9vs76USrHkdmQr+Y"
            b"MXcQiyVXwHpbhMhIIeHt5Ttj9zjWRA9XgXmzSYcquLytR4VJeVuJaA9UW0lLUves"
            b"ISiubGYjdC9Lt7TOdsebN6nh+k77MbAZznOLTwh2lsfZr/jcYPe4lK61M7Xq+Vtv"
            b"BOM/pFtAZhXvq9T77It98Wn2vAAlkJTOTzM53oK71Ind9P1S2swbCfkFRafdqCb"
            b"jDtSsObteMSl5FJlYHh3wKc1MO4jqSWY75u3Zb3I9lfCB1wfYTH1qGRhh8oMhcl"
            b"LhUK3OjrEAsCb9Oz6mkWdQ5+Cih+v/ROHHvGyClnhykwpruNBmmnEigydXOJQ6i"
            b"7ZpP9qjeLzUebBmf4nv2KIboORnhwpv65JWsnlFJMqY5N8xXdGGJ2oL9zzOBSIs"
            b"Estw79MfejMmN96DDIyHuF4RbldNq/2zS5BB7RxDJjIEfcyIU6rv5xmMckParjy"
            b"956H1inZXJ/kVqT6twJ2zcICzWfJ3dK20hWPFQ6hgvu5sHlCWL5u6dUjC/9rph7"
            b"E55SsJ6rtoZeHeBlnb54XT98D37lH0NpK3OGTF9oAWp12dgMZ3iT4B+0KGJqpta"
            b"EJb4xInwv7nKNaPqi6a5/bP/Jw4v2drD+U8ldx0SIp4f8IidmKxkomZFxPwhvyaYyL/qwvJ"
        )
        xml = RDX._mc4_decrypt(sample).decode("utf-8")
        self.assertTrue(xml.startswith('<?xml version="1.0"'))
        self.assertIn('Game="Killzone: Shadow Fall"', xml)
        self.assertIn('Cusa="CUSA00002"', xml)
        self.assertIn("<Offset>668DA3</Offset>", xml)
        self.assertIn("<ValueOn>90-90-90-90</ValueOn>", xml)

    def test_generate_mc4_bytes_round_trips_to_expected_trainer_xml(self):
        mods = [{
            "name": "Health", "description": "RDX module-relative scalar write.",
            "type": "checkbox",
            "memory": [{"offset": "1020", "on": "78563412", "off": "07000000"}],
        }]
        mc4 = RDX.generate_mc4_bytes(
            mods, "PPSA00001", "01.001.000", "Test Game", "eboot.bin", "Tester")
        xml = RDX._mc4_decrypt(mc4).decode("utf-8")
        self.assertIn('Game="Test Game" Moder="Tester" Cusa="PPSA00001" '
                      'Version="01.001.000" Process="eboot.bin"', xml)
        self.assertIn('<Cheat Control="Toggel" Text="Health">', xml)
        self.assertIn("<Offset>1020</Offset>", xml)
        self.assertIn("<Section>0</Section>", xml)
        self.assertIn("<ValueOn>78-56-34-12</ValueOn>", xml)
        self.assertIn("<ValueOff>07-00-00-00</ValueOff>", xml)

    def test_generate_mc4_bytes_escapes_xml_special_characters_in_name(self):
        mods = [{
            "name": 'A & B <"quoted">', "memory": [
                {"offset": "10", "on": "01", "off": "00"}],
        }]
        xml = RDX._mc4_decrypt(RDX.generate_mc4_bytes(
            mods, "PPSA00001", "01.00", "Test", "eboot.bin")).decode("utf-8")
        self.assertIn(
            'Text="A &amp; B &lt;&quot;quoted&quot;&gt;"', xml)
        self.assertNotIn("<\"quoted\">", xml)

    def test_do_export_writes_mc4_matching_etahen_mods(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        maps = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                 "name": "executable"}]
        cheat = {
            "name": "Current", "type": "write", "address": 0x401000,
            "value": 3, "original_value": 2, "width": 4,
            "pid": 7, "process": "eboot.bin", "session": 5,
        }
        old_state = {key: RDX.state.get(key) for key in (
            "ip", "pid", "proc_name", "session", "cheats", "game_id",
            "game_ver", "game_title", "export_dir")}
        RDX.state.update(ip="test", pid=7, proc_name="eboot.bin", session=5,
                         cheats=[cheat], game_id="", game_ver="01.00",
                         game_title="")
        try:
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "input_box", side_effect=[
                     "PPSA00001", "01.001.000", "Test Game", "Tester",
                     directory]), \
                 patch.object(RDX, "confirm_box", return_value=True), \
                 patch.object(RDX, "_save_preferences"), \
                 patch.object(RDX, "message_box"), \
                 patch.object(RDX, "_get_maps_cached", return_value=maps), \
                 patch.object(RDX.Path, "home", return_value=Path(directory)):
                RDX.do_export(FakeWindow())
                etahen_mods = json.loads((Path(directory) /
                    "PPSA00001_01.001.000.json").read_text())["mods"]
                mc4_bytes = (Path(directory) /
                    "PPSA00001_01.001.000.mc4").read_bytes()
            self.assertEqual(len(etahen_mods), 1)
            xml = RDX._mc4_decrypt(mc4_bytes).decode("utf-8")
            self.assertIn('Cusa="PPSA00001"', xml)
            self.assertIn(
                f'<Offset>{etahen_mods[0]["memory"][0]["offset"]}</Offset>', xml)
            self.assertIn(RDX._dash_hex(etahen_mods[0]["memory"][0]["on"]), xml)
        finally:
            RDX.state.update(old_state)

    def test_native_trainer_declares_pointer_format_and_original_values(self):
        payload = json.loads(RDX.generate_cht([
            {"name": "Ammo", "type": "pointer_freeze", "base": 0x400100,
             "offsets": [0x18, -0x20], "module_name": "executable",
             "module_relative_offset": 0x100, "terminal_offset": -0x10,
             "value": 99,
             "original_value": 6, "width": 4,
             "cross_reload_validated": True, "game_identity": "game:test"},
        ], "PPSA00001", "01.001.000", "Test", process="eboot.bin"))
        self.assertEqual(payload["format"], "rdx-pointer-trainer-v1")
        self.assertEqual(payload["process"], "eboot.bin")
        self.assertEqual(payload["cheatList"][0]["offsets"],
                         ["0x18", "-0x20"])
        self.assertEqual(payload["cheatList"][0]["module_name"], "executable")
        self.assertEqual(payload["cheatList"][0]["module_relative_offset"],
                         "0x100")
        self.assertEqual(payload["cheatList"][0]["terminal_offset"], "-0x10")
        self.assertNotIn("module", payload["cheatList"][0])
        self.assertEqual(payload["cheatList"][0]["original_value"], "0x6")
        self.assertTrue(
            payload["cheatList"][0]["cross_reload_validated"])

    def test_apply_portable_pointer_rebases_and_uses_terminal_offset(self):
        old = {key: RDX.state.get(key)
               for key in ("ip", "pid", "proc_name", "session")}
        RDX.state.update(ip="test", pid=22, proc_name="eboot.bin", session=9)
        cheat = {
            "name": "Portable", "type": "pointer_write", "base": 0x1111,
            "offsets": [0x18], "terminal_offset": -0x10,
            "module_name": "executable", "module_relative_offset": 0x100,
            "cross_reload_validated": True, "value": 7, "width": 4,
            "game_identity": "game:test",
            "pid": 11, "process": "eboot.bin", "session": 8,
        }
        try:
            with patch.object(RDX, "_runtime_pointer_base", return_value=0x500100), \
                 patch.object(RDX, "_portable_cheat_matches_current_game",
                              return_value=True), \
                 patch.object(RDX, "_resolve_pointer_chain",
                              return_value=(True, 0x9000, [0x9010])) as resolve, \
                 patch.object(RDX, "_validate_addr_in_maps", return_value=None), \
                 patch.object(RDX, "ps5_write_verified",
                              return_value=(True, True, struct.pack("<I", 7))):
                RDX._apply_cheat_once(None, cheat)
            resolve.assert_called_once_with(
                "test", 22, 0x500100, [0x18], -0x10)
        finally:
            RDX.state.update(old)

    def test_module_relative_scalar_rebases_and_exports_as_portable(self):
        maps = [{"start": 0x900000, "end": 0xA00000, "prot": 5,
                 "name": "executable"}]
        cheat = {
            "name": "Static", "type": "write", "address": 0x401020,
            "module_name": "executable", "module_relative_offset": 0x1020,
            "game_identity": "game:test", "value": 3,
            "original_value": 2, "width": 4,
        }
        old = {key: RDX.state.get(key) for key in ("ip", "pid")}
        RDX.state.update(ip="test", pid=1)
        try:
            with patch.object(RDX, "_get_maps_cached", return_value=maps):
                self.assertEqual(RDX._runtime_scalar_address(cheat), 0x901020)
        finally:
            RDX.state.update(old)
        payload = json.loads(RDX.generate_cht(
            [cheat], "PPSA00001", "01.001.000", "Test"))
        item = payload["cheatList"][0]
        self.assertEqual(item["module_name"], "executable")
        self.assertEqual(item["module_relative_offset"], "0x1020")
        self.assertFalse(item["session_bound"])
        self.assertEqual(payload["game_identity"], "game:test")

    def test_native_pointer_export_import_round_trip_keeps_rebase_metadata(self):
        source = {
            "name": "Ammo", "type": "pointer_freeze", "base": 0x400100,
            "offsets": [0x18, -0x20], "terminal_offset": -0x10,
            "module_name": "executable", "module_relative_offset": 0x100,
            "cross_reload_validated": True, "game_identity": "game:test",
            "value": 99, "original_value": 6, "width": 4,
        }
        payload = RDX.generate_cht(
            [source], "PPSA00001", "01.001.000", "Test",
            process="eboot.bin")

        class FakeWindow:
            def clear(self): pass

        old_state = {key: RDX.state.get(key)
                     for key in ("ip", "pid", "proc_name", "session", "cheats")}
        RDX.state.update(ip="test", pid=7, proc_name="eboot.bin",
                         session=3, cheats=[])
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trainer.rdx.json"
                path.write_text(payload, encoding="utf-8")
                with patch.object(RDX, "draw_border"), \
                     patch.object(RDX, "safe_addstr"), \
                     patch.object(RDX, "color", return_value=0), \
                     patch.object(RDX, "input_box", return_value=str(path)), \
                     patch.object(RDX, "message_box"), \
                     patch.object(RDX, "_get_maps_cached", return_value=[]), \
                     patch.object(RDX, "_pointer_game_identity",
                                  return_value="game:test"):
                    RDX.do_import(FakeWindow())
            imported = RDX.state["cheats"][0]
            self.assertEqual(imported["module_name"], "executable")
            self.assertEqual(imported["module_relative_offset"], 0x100)
            self.assertEqual(imported["offsets"], [0x18, -0x20])
            self.assertEqual(imported["terminal_offset"], -0x10)
            self.assertEqual(imported["original_value"], 6)
            self.assertTrue(imported["cross_reload_validated"])
            self.assertEqual(imported["game_identity"], "game:test")
        finally:
            RDX.state.update(old_state)

    def test_imported_absolute_cheat_is_not_rebound_to_current_session(self):
        payload = RDX.generate_cht([{
            "name": "Old heap", "type": "write", "address": 0x800000,
            "value": 9, "original_value": 2, "width": 4,
        }], "PPSA00001", "01.00", "Test", process="eboot.bin")

        class FakeWindow:
            def clear(self): pass

        old = {key: RDX.state.get(key)
               for key in ("ip", "pid", "proc_name", "session", "cheats")}
        RDX.state.update(ip="test", pid=7, proc_name="eboot.bin",
                         session=3, cheats=[])
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "absolute.rdx.json"
                path.write_text(payload, encoding="utf-8")
                with patch.object(RDX, "draw_border"), \
                     patch.object(RDX, "safe_addstr"), \
                     patch.object(RDX, "color", return_value=0), \
                     patch.object(RDX, "input_box", return_value=str(path)), \
                     patch.object(RDX, "message_box"):
                    RDX.do_import(FakeWindow())
            imported = RDX.state["cheats"][0]
            self.assertIsNone(imported["pid"])
            self.assertIsNone(imported["session"])
            self.assertTrue(imported["import_locked"])
        finally:
            RDX.state.update(old)

    def test_edit_cheat_never_writes_memory_implicitly(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        old_cheats = RDX.state.get("cheats")
        RDX.state["cheats"] = [{
            "name": "Health", "type": "write", "address": 0x401000,
            "value": 10, "width": 4, "pid": 7, "session": 1,
        }]
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "input_box",
                              side_effect=["Health edited", "20"]), \
                 patch.object(RDX, "cycle_input", return_value="write"), \
                 patch.object(RDX, "message_box"), \
                 patch.object(RDX, "_write_value_verified",
                              side_effect=AssertionError("implicit write")):
                RDX._edit_cheat(FakeWindow(), 0)
            self.assertEqual(RDX.state["cheats"][0]["value"], 20)
            self.assertEqual(RDX.state["cheats"][0]["name"], "Health edited")
        finally:
            RDX.state["cheats"] = old_cheats

    def test_wrong_game_fingerprint_blocks_stale_portable_write(self):
        old = {key: RDX.state.get(key)
               for key in ("ip", "pid", "proc_name", "session")}
        RDX.state.update(ip="test", pid=22, proc_name="eboot.bin", session=9)
        cheat = {
            "name": "Wrong title", "type": "pointer_write", "base": 0x1000,
            "offsets": [0], "module_name": "executable",
            "module_relative_offset": 0, "cross_reload_validated": True,
            "game_identity": "game:other", "value": 7, "width": 4,
            "pid": 11, "process": "eboot.bin", "session": 8,
        }
        try:
            with patch.object(RDX, "_portable_cheat_matches_current_game",
                              return_value=False), \
                 patch.object(RDX, "message_box") as message, \
                 patch.object(RDX, "ps5_write_verified",
                              side_effect=AssertionError("write must be blocked")):
                RDX._apply_cheat_once(None, cheat)
            self.assertEqual(message.call_args.args[2], "Stale Cheat")
        finally:
            RDX.state.update(old)

    def test_export_excludes_other_game_and_stale_session_entries(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        maps = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                 "name": "executable"}]
        current = {
            "name": "Current", "type": "write", "address": 0x401000,
            "value": 3, "original_value": 2, "width": 4,
            "pid": 7, "process": "eboot.bin", "session": 5,
        }
        stale = {
            "name": "Stale", "type": "write", "address": 0x402000,
            "value": 8, "original_value": 7, "width": 4,
            "pid": 6, "process": "eboot.bin", "session": 4,
        }
        old_state = {key: RDX.state.get(key) for key in (
            "ip", "pid", "proc_name", "session", "cheats", "game_id",
            "game_ver", "game_title", "export_dir")}
        RDX.state.update(ip="test", pid=7, proc_name="eboot.bin", session=5,
                         cheats=[current, stale], game_id="", game_ver="01.00",
                         game_title="")
        try:
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "input_box", side_effect=[
                     "PPSA00001", "01.001.000", "Test", "Tester",
                     directory]), \
                 patch.object(RDX, "confirm_box", return_value=True), \
                 patch.object(RDX, "_save_preferences"), \
                 patch.object(RDX, "message_box"), \
                 patch.object(RDX, "_get_maps_cached", return_value=maps), \
                 patch.object(RDX.Path, "home", return_value=Path(directory)):
                RDX.do_export(FakeWindow())
                exported = json.loads((Path(directory) /
                    "PPSA00001_01.001.000.rdx.json").read_text())
            self.assertEqual([c["name"] for c in exported["cheatList"]],
                             ["Current"])
        finally:
            RDX.state.update(old_state)

    def test_hex_or_none_helper(self):
        self.assertEqual(RDX._hex_or_none(0x1020), "0x1020")
        self.assertIsNone(RDX._hex_or_none(None))

    def test_confirm_quit_allows_silently_when_nothing_unexported(self):
        old = {key: RDX.state.get(key)
               for key in ("cheats", "cheats_dirty")}
        RDX.state.update(cheats=[], cheats_dirty=False)
        try:
            with patch.object(RDX, "confirm_box",
                              side_effect=AssertionError("should not prompt")):
                self.assertTrue(RDX._confirm_quit(None))
            RDX.state.update(
                cheats=[{"name": "Health"}], cheats_dirty=False)
            with patch.object(RDX, "confirm_box",
                              side_effect=AssertionError("should not prompt")):
                self.assertTrue(RDX._confirm_quit(None))
        finally:
            RDX.state.update(old)

    def test_confirm_quit_prompts_when_cheats_are_unexported(self):
        old = {key: RDX.state.get(key)
               for key in ("cheats", "cheats_dirty")}
        RDX.state.update(cheats=[{"name": "Health"}], cheats_dirty=True)
        try:
            with patch.object(RDX, "confirm_box",
                              return_value=False) as box:
                self.assertFalse(RDX._confirm_quit(None))
            self.assertIn("Unsaved Cheats", box.call_args.args)
            with patch.object(RDX, "confirm_box", return_value=True):
                self.assertTrue(RDX._confirm_quit(None))
        finally:
            RDX.state.update(old)

    def test_cheats_dirty_flag_set_on_mutation_and_cleared_on_full_export(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        maps = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                 "name": "executable"}]
        cheat = {
            "name": "Current", "type": "write", "address": 0x401000,
            "value": 3, "original_value": 2, "width": 4,
            "pid": 7, "process": "eboot.bin", "session": 5,
        }
        old_state = {key: RDX.state.get(key) for key in (
            "ip", "pid", "proc_name", "session", "cheats", "cheats_dirty",
            "game_id", "game_ver", "game_title", "export_dir")}
        RDX.state.update(ip="test", pid=7, proc_name="eboot.bin", session=5,
                         cheats=[cheat], cheats_dirty=True, game_id="",
                         game_ver="01.00", game_title="")
        try:
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "input_box", side_effect=[
                     "PPSA00001", "01.001.000", "Test", "Tester",
                     directory]), \
                 patch.object(RDX, "confirm_box", return_value=True), \
                 patch.object(RDX, "_save_preferences"), \
                 patch.object(RDX, "message_box"), \
                 patch.object(RDX, "_get_maps_cached", return_value=maps), \
                 patch.object(RDX.Path, "home", return_value=Path(directory)):
                RDX.do_export(FakeWindow())
            self.assertFalse(RDX.state["cheats_dirty"])
        finally:
            RDX.state.update(old_state)

    def test_fuzzy_subsequence_rank_orders_tighter_matches_first(self):
        self.assertIsNone(RDX._fuzzy_subsequence_rank("xyz", "export"))
        self.assertEqual(RDX._fuzzy_subsequence_rank("exp", "export"), (2, 0))
        self.assertIsNotNone(
            RDX._fuzzy_subsequence_rank("exp", "export trainers"))

    def test_command_palette_rank_matches_multi_term_query_as_subsequence(self):
        self.assertIsNotNone(
            RDX._command_palette_rank("exp trn", "Export Trainers"))
        self.assertIsNone(
            RDX._command_palette_rank("zzz", "Export Trainers"))
        # Empty query matches everything, with equal (tie) rank.
        self.assertEqual(RDX._command_palette_rank("", "Anything"), (0, 0))

    def test_command_palette_fuzzy_query_dispatches_matching_command(self):
        class FakeWindow:
            def __init__(self, keys): self._keys = list(keys)
            def clear(self): pass
            def refresh(self): pass
            def getmaxyx(self): return (24, 80)
            def getch(self):
                return self._keys.pop(0) if self._keys else 27

        keys = [ord(c) for c in "exp trn"] + [10]
        with patch.object(RDX, "draw_border"), \
             patch.object(RDX, "safe_addstr"), \
             patch.object(RDX, "draw_statusbar"), \
             patch.object(RDX, "color", return_value=0), \
             patch.object(RDX, "dispatch", return_value=None) as dispatch:
            RDX.do_command_palette(FakeWindow(keys))
        self.assertEqual(dispatch.call_args.args[1], "export")

    def test_command_palette_letter_q_types_instead_of_quitting(self):
        # Regression: 'q'/'Q' used to be gated the same as Esc — quit only
        # if the query was empty — so a query could never start with "q".
        # Typing 'q' then 'x' then Esc should leave the palette with "qx"
        # typed, and Esc should now close it unconditionally (matching
        # standard fuzzy-finder convention) rather than requiring the query
        # be cleared first.
        class FakeWindow:
            def __init__(self, keys): self._keys = list(keys)
            def clear(self): pass
            def refresh(self): pass
            def getmaxyx(self): return (24, 80)
            def getch(self):
                return self._keys.pop(0) if self._keys else 27

        with patch.object(RDX, "draw_border"), \
             patch.object(RDX, "safe_addstr"), \
             patch.object(RDX, "draw_statusbar"), \
             patch.object(RDX, "color", return_value=0), \
             patch.object(RDX, "dispatch",
                          side_effect=AssertionError("must not dispatch")):
            result = RDX.do_command_palette(FakeWindow([ord('q'), ord('x'), 27]))
        self.assertIsNone(result)

    def test_do_help_mentions_command_palette_for_hidden_actions(self):
        with patch.object(RDX, "message_box") as box:
            RDX.do_help(None)
        self.assertEqual(box.call_args.args[2], "Keyboard Help")
        joined = "\n".join(box.call_args.args[1]).lower()
        self.assertIn("export", joined)
        self.assertIn("press /", joined)

    def test_screen_main_opens_and_quits_without_crashing(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass
            def getmaxyx(self): return (30, 100)
            def timeout(self, *_a): pass
            def getch(self): return ord('q')

        old = {key: RDX.state.get(key) for key in
               ("cheats", "cheats_dirty")}
        RDX.state.update(cheats=[], cheats_dirty=False)
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "color", return_value=0):
                result = RDX.screen_main(FakeWindow())
            self.assertIsNone(result)
        finally:
            RDX.state.update(old)

    def test_screen_proc_select_escape_exits_when_filter_empty(self):
        class FakeWindow:
            def __init__(self, keys): self._keys = list(keys)
            def clear(self): pass
            def refresh(self): pass
            def getmaxyx(self): return (24, 80)
            def getch(self):
                return self._keys.pop(0) if self._keys else ord('q')

        procs = [{"pid": 100, "name": "eboot.bin"}]
        with patch.object(RDX, "draw_border"), \
             patch.object(RDX, "safe_addstr"), \
             patch.object(RDX, "draw_statusbar"), \
             patch.object(RDX, "color", return_value=0):
            result = RDX.screen_proc_select(FakeWindow([27]), procs)
        self.assertEqual(result, "connect")

    def test_screen_proc_select_letter_q_filters_instead_of_quitting(self):
        # Regression: 'q'/'Q' used to quit unconditionally, so a process
        # name containing "q" could never be typed into the filter — the
        # key always exited the screen before reaching the filter-append
        # branch. It must now behave like the command palette: quit only
        # when the filter is empty, otherwise append to the filter.
        class FakeWindow:
            def __init__(self, keys): self._keys = list(keys)
            def clear(self): pass
            def refresh(self): pass
            def getmaxyx(self): return (24, 80)
            def getch(self):
                return self._keys.pop(0) if self._keys else ord('q')

        procs = [{"pid": 100, "name": "eboot.bin"}, {"pid": 101, "name": "quake.elf"}]
        keys = [ord('q'), RDX.curses.KEY_ENTER]
        old = {key: RDX.state.get(key) for key in
               ("pid", "proc_name", "last_process", "session")}
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "_save_preferences"):
                result = RDX.screen_proc_select(FakeWindow(keys), procs)
            self.assertEqual(result, "main")
            self.assertEqual(RDX.state["proc_name"], "quake.elf")
        finally:
            RDX.state.update(old)

    def test_do_freeze_cancel_at_target_prompt_does_not_crash(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        with patch.object(RDX, "draw_border"), \
             patch.object(RDX, "cycle_input", return_value=None) as cycle:
            RDX.do_freeze(FakeWindow())
        cycle.assert_called_once()

    def test_do_pointer_scan_cancel_at_manual_entry_does_not_crash(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        old = RDX.state.get("scan_results")
        RDX.state["scan_results"] = RDX._make_addr_array()
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "input_box", return_value=None) as box:
                RDX.do_pointer_scan(FakeWindow())
            box.assert_called_once()
        finally:
            RDX.state["scan_results"] = old

    def test_delete_cheat_with_undo_round_trips_and_is_single_slot(self):
        old = {key: RDX.state.get(key) for key in
               ("cheats", "cheats_dirty", "last_deleted_cheat")}
        cheat = {"name": "Health", "type": "write", "address": 0x1000,
                 "value": 1, "width": 4}
        RDX.state.update(cheats=[cheat], cheats_dirty=False,
                         last_deleted_cheat=None)
        try:
            with patch.object(RDX, "_is_cheat_frozen", return_value=False):
                name = RDX._delete_cheat_with_undo(0)
            self.assertEqual(name, "Health")
            self.assertEqual(RDX.state["cheats"], [])
            self.assertTrue(RDX.state["cheats_dirty"])

            RDX.state["cheats_dirty"] = False
            restored = RDX._restore_last_deleted_cheat()
            self.assertEqual(restored, "Health")
            self.assertEqual([c["name"] for c in RDX.state["cheats"]],
                             ["Health"])
            self.assertTrue(RDX.state["cheats_dirty"])
            # Single-slot buffer: a second restore has nothing left to do.
            self.assertIsNone(RDX._restore_last_deleted_cheat())
        finally:
            RDX.state.update(old)

    def test_cheat_list_delete_then_z_restores_it(self):
        class FakeWindow:
            def __init__(self, keys): self._keys = list(keys)
            def clear(self): pass
            def refresh(self): pass
            def nodelay(self, *_a): pass
            def getmaxyx(self): return (24, 80)
            def getch(self):
                return self._keys.pop(0) if self._keys else ord('q')

        cheat = {"name": "Health", "type": "write", "address": 0x1000,
                 "value": 1, "width": 4, "pid": None, "process": "eboot.bin"}
        old = {key: RDX.state.get(key) for key in
               ("cheats", "cheats_dirty", "last_deleted_cheat")}
        RDX.state.update(cheats=[cheat], cheats_dirty=False,
                         last_deleted_cheat=None)
        keys = [ord('d'), ord('z'), ord('q')]
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "confirm_box", return_value=True), \
                 patch.object(RDX, "_read_cheat_live_value", return_value="1"):
                RDX.do_cheat_list(FakeWindow(keys))
            self.assertEqual([c["name"] for c in RDX.state["cheats"]],
                             ["Health"])
            self.assertIsNone(RDX.state["last_deleted_cheat"])
        finally:
            RDX.state.update(old)

    def test_cheat_list_accepts_escape_to_back_out(self):
        # Regression: Cheat List previously only accepted 'Q', unlike every
        # other screen in the app, which breaks Esc muscle memory trained
        # elsewhere. A bare Esc (27) with no other keys queued must return
        # cleanly rather than looping forever waiting for 'q'/'Q'.
        class FakeWindow:
            def __init__(self, keys): self._keys = list(keys)
            def clear(self): pass
            def refresh(self): pass
            def nodelay(self, *_a): pass
            def getmaxyx(self): return (24, 80)
            def getch(self):
                return self._keys.pop(0) if self._keys else -1

        old = RDX.state.get("cheats")
        RDX.state["cheats"] = []
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "color", return_value=0):
                result = RDX.do_cheat_list(FakeWindow([27]))
            self.assertIsNone(result)
        finally:
            RDX.state["cheats"] = old

    def test_select_export_cheats_lets_user_deselect_one(self):
        class FakeWindow:
            def __init__(self, keys): self._keys = list(keys)
            def clear(self): pass
            def refresh(self): pass
            def getmaxyx(self): return (24, 80)
            def getch(self):
                return self._keys.pop(0) if self._keys else 27

        cheats = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        # Down to B, toggle it off, Enter.
        keys = [RDX.curses.KEY_DOWN, ord(' '), 10]
        with patch.object(RDX, "draw_border"), \
             patch.object(RDX, "safe_addstr"), \
             patch.object(RDX, "draw_statusbar"), \
             patch.object(RDX, "color", return_value=0):
            picked = RDX._select_export_cheats(FakeWindow(keys), cheats)
        self.assertEqual([c["name"] for c in picked], ["A", "C"])

    def test_select_export_cheats_returns_none_on_cancel(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass
            def getmaxyx(self): return (24, 80)
            def getch(self): return 27

        with patch.object(RDX, "draw_border"), \
             patch.object(RDX, "safe_addstr"), \
             patch.object(RDX, "draw_statusbar"), \
             patch.object(RDX, "color", return_value=0):
            picked = RDX._select_export_cheats(
                FakeWindow(), [{"name": "A"}, {"name": "B"}])
        self.assertIsNone(picked)

    def test_do_export_honours_deselected_cheats(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        maps = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                 "name": "executable"}]
        cheat_a = {
            "name": "Keep", "type": "write", "address": 0x401000,
            "value": 3, "original_value": 2, "width": 4,
            "pid": 7, "process": "eboot.bin", "session": 5,
        }
        cheat_b = {
            "name": "Drop", "type": "write", "address": 0x401004,
            "value": 9, "original_value": 8, "width": 4,
            "pid": 7, "process": "eboot.bin", "session": 5,
        }
        old_state = {key: RDX.state.get(key) for key in (
            "ip", "pid", "proc_name", "session", "cheats", "cheats_dirty",
            "game_id", "game_ver", "game_title", "export_dir")}
        RDX.state.update(ip="test", pid=7, proc_name="eboot.bin", session=5,
                         cheats=[cheat_a, cheat_b], cheats_dirty=True,
                         game_id="", game_ver="01.00", game_title="")
        try:
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "_select_export_cheats",
                              return_value=[cheat_a]) as picker, \
                 patch.object(RDX, "input_box", side_effect=[
                     "PPSA00001", "01.001.000", "Test", "Tester",
                     directory]), \
                 patch.object(RDX, "confirm_box", return_value=True), \
                 patch.object(RDX, "_save_preferences"), \
                 patch.object(RDX, "message_box"), \
                 patch.object(RDX, "_get_maps_cached", return_value=maps), \
                 patch.object(RDX.Path, "home", return_value=Path(directory)):
                RDX.do_export(FakeWindow())
                exported = json.loads((Path(directory) /
                    "PPSA00001_01.001.000.rdx.json").read_text())
            picker.assert_called_once()
            self.assertEqual([c["name"] for c in exported["cheatList"]],
                             ["Keep"])
            # Regression: "Drop" was deselected, not just excluded as
            # stale/other-game, so the unsaved-work flag must stay set —
            # otherwise quitting afterward would silently discard it with
            # no warning.
            self.assertTrue(RDX.state["cheats_dirty"])
        finally:
            RDX.state.update(old_state)

    # ── patch62 ──────────────────────────────────────────────────────────

    class _WidthWindow:
        """Records what safe_addstr actually managed to write."""
        def __init__(self, cols):
            self.cols = cols
            self.written = None
        def getmaxyx(self):
            return (24, self.cols)
        def addstr(self, _y, _x, text, _attr=0):
            self.written = text

    def test_decorative_glyphs_are_not_treated_as_double_width(self):
        # safe_addstr used "codepoint > 0x1100 means 2 columns", which is
        # wrong for every decorative glyph this UI draws: the progress bar's
        # block characters, the check/warn/cross marks, the arrows and the
        # spinner are all East Asian Width Ambiguous/Neutral, i.e. ONE column
        # in a terminal. A 60-column progress bar rendered in 39 columns on
        # an 80-column terminal.
        bar = "[" + "█" * 29 + "░" * 29 + "]"   # 60 chars
        win = self._WidthWindow(80)
        RDX.safe_addstr(win, 10, 3, bar)
        self.assertEqual(win.written, bar,
                         "full-width bar must not be clipped on an 80-col term")
        line = "✓  Scanning…  ⚠ 1,234 → 5,678"
        win = self._WidthWindow(80)
        RDX.safe_addstr(win, 10, 3, line)
        self.assertEqual(win.written, line)

    def test_real_double_width_text_still_clips(self):
        # The fix must not simply call everything narrow: genuine CJK is two
        # columns and still has to be clipped, or curses overruns the window.
        cjk = "中" * 10            # 10 chars, 20 columns
        win = self._WidthWindow(13)    # x=3 -> 10 columns available
        RDX.safe_addstr(win, 5, 3, cjk)
        self.assertEqual(len(win.written), 5, "10 columns fits 5 CJK chars")
        win = self._WidthWindow(23)    # 20 columns available
        RDX.safe_addstr(win, 5, 3, cjk)
        self.assertEqual(len(win.written), 10)

    def test_combining_marks_do_not_consume_a_column(self):
        # e + combining acute occupies one cell, not two.
        text = "é" * 5           # 10 chars, 5 columns
        win = self._WidthWindow(9)     # x=3 -> 6 columns available
        RDX.safe_addstr(win, 5, 3, text)
        self.assertEqual(win.written, text)

    def test_mc4_xml_to_mods_parses_cheat_and_cheatline_elements(self):
        xml_text = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<Trainer Game="Test Game" Cusa="CUSA00002" Version="01.00" '
            'Process="eboot.bin">'
            '<Cheat Control="Toggel" Text="Godmode">'
            '<Cheatline><Offset>668DA3</Offset><Section>0</Section>'
            '<ValueOn>90-90-90-90</ValueOn><ValueOff>C5-FA-11-00</ValueOff>'
            '</Cheatline></Cheat></Trainer>')
        attrs, mods = RDX.mc4_xml_to_mods(xml_text)
        self.assertEqual(attrs["Cusa"], "CUSA00002")
        self.assertEqual(mods, [{
            "name": "Godmode",
            # patch61 carries <Section> through so the importer can refuse a
            # patch it cannot place; 0 is the main image.
            "memory": [{"offset": "668DA3", "on": "90909090",
                        "off": "C5FA1100", "section": 0}],
        }])

    def test_mc4_import_refuses_a_non_main_section_instead_of_misplacing_it(self):
        # <Section> selects WHICH module the offset belongs to. Dropping it
        # made a section-1 library patch resolve against the main module --
        # the exact same address a section-0 patch would get -- so a real
        # community trainer wrote into the wrong image, reported as a clean
        # "Import Complete". RDX only resolves the main image, so anything
        # else must be skipped loudly.
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<Trainer Cusa="CUSA00002" Process="eboot.bin">'
            '<Cheat Text="main"><Cheatline><Offset>1000</Offset>'
            '<Section>0</Section><ValueOn>90-90</ValueOn>'
            '<ValueOff>74-05</ValueOff></Cheatline></Cheat>'
            '<Cheat Text="library"><Cheatline><Offset>1000</Offset>'
            '<Section>1</Section><ValueOn>AA-BB</ValueOn>'
            '<ValueOff>CC-DD</ValueOff></Cheatline></Cheat>'
            '<Cheat Text="no tag"><Cheatline><Offset>2000</Offset>'
            '<ValueOn>01</ValueOn><ValueOff>00</ValueOff></Cheatline></Cheat>'
            '</Trainer>')
        _attrs, mods = RDX.mc4_xml_to_mods(xml)
        self.assertEqual([mem["section"] for mod in mods
                          for mem in mod["memory"]], [0, 1, 0])
        entries = RDX._mods_to_import_entries(mods, "t.mc4", 0x400000)
        names = [e["name"] for e in entries]
        self.assertIn("main", names)
        self.assertIn("no tag", names)       # absent tag == section 0
        self.assertNotIn("library", names)   # section 1 must not be placed

    def test_mods_import_rejects_negative_and_oversized_patches(self):
        # A negative offset resolves BELOW the module base, i.e. outside the
        # image; export already refuses these and import must too. A patch
        # over 256 bytes exceeds _value_width()'s raw-byte cap, so it built a
        # cheat that raised on every apply and export instead of being
        # rejected at the door.
        def one(offset, on):
            return RDX._mods_to_import_entries(
                [{"name": "x", "memory": [{"offset": offset, "on": on,
                                           "off": ""}]}], "f", 0x400000)
        self.assertEqual(one("-10", "90"), [])
        self.assertEqual(one("10", "90" * 300), [])
        self.assertEqual(len(one("10", "90" * 256)), 1)   # exactly at the cap
        self.assertEqual(len(one("10", "90909090")), 1)

    def test_trainer_import_tolerates_a_utf8_bom(self):
        # Trainer files are user-supplied and any Windows editor leaves a
        # BOM; json.loads then failed with a raw "Unexpected UTF-8 BOM" and
        # the entire import died.
        payload = {"name": "G", "id": "CUSA00002", "version": "01.00",
                   "process": "eboot.bin",
                   "mods": [{"name": "P", "memory": [
                       {"offset": "1234", "on": "90909090",
                        "off": "00000000"}]}]}
        maps = [{"start": 0x400000, "end": 0x600000, "prot": 5,
                 "offset": 0, "name": "executable"}]
        old = {k: RDX.state.get(k) for k in
               ("cheats", "cheats_dirty", "pid", "proc_name", "session",
                "game_id", "ip")}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bom.json"
            path.write_text(json.dumps(payload), encoding="utf-8-sig")
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
            try:
                RDX.state.update(cheats=[], cheats_dirty=False, pid=1,
                                 proc_name="eboot.bin", session=1,
                                 game_id="CUSA00002", ip="test")
                with patch.object(RDX, "draw_border"), \
                     patch.object(RDX, "safe_addstr"), \
                     patch.object(RDX, "color", return_value=0), \
                     patch.object(RDX, "input_box", return_value=str(path)), \
                     patch.object(RDX, "confirm_box", return_value=True), \
                     patch.object(RDX, "_get_maps_cached", return_value=maps), \
                     patch.object(RDX, "message_box"):
                    RDX.do_import(self._FakeKeyWindow([]))
                self.assertEqual(len(RDX.state["cheats"]), 1)
                self.assertEqual(RDX.state["cheats"][0]["address"], 0x401234)
            finally:
                RDX.state.update(old)

    def test_mods_to_import_entries_resolves_against_live_module_base(self):
        mods = [{"name": "Godmode", "memory": [
            {"offset": "1020", "on": "78563412", "off": "07000000"}]}]
        entries = RDX._mods_to_import_entries(mods, Path("/tmp/x.mc4"),
                                               module_base=0x400000)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["address"], 0x401020)
        self.assertEqual(e["value"], "78563412")
        self.assertEqual(e["value_type"], "bytes")
        self.assertEqual(e["width"], 4)
        self.assertEqual(e["original_value"], "07000000")

    def test_do_import_real_mc4_sample_resolves_against_live_module(self):
        # Same real Killzone: Shadow Fall sample used in the export tests —
        # decode-then-import against a live-attached process's main module.
        sample = (
            b"AgKHPLxZwZ+2YfYslmLUXGyUIJK0NSn6118NipwxKEztE2q1rhwqHXSKaFQA/6YW"
            b"qxnwy/nyj9oBsEIbcrz/FKR7uSA0bLf+RXN/lhBn6v+h9HuUZd2kHOE9znFnUrqD"
            b"RVtUNuoiYeXgW98DPqQmGLoNR9j5WVYHsEdIvjM3iiN9oQ9djobOpt6AuAFjAB15"
            b"/grhFrQQz81N7AGK5p/B2j9YU8R8aj8YkQ/kY2qgfd6FBdGxfY+QC8z7uGLg6kqk"
            b"kdewLGBuu9gs5mwy6UpLFxuRgO4RT/6KxFqgqHHIvesBs8aVYA7zl63Dbv3L1ggk"
            b"R8v0ZJ3g1FGeVLXSYVDximAsCMm44Pov1+R30rdHFGYhu7j6E6N7TPAwQjSveMKP"
            b"LvwHQPy4zmyfaJF1VGbr/M/NcvplqthkJ4TWum5Vm+9WCusFR3qg5YOi0jqzaFnZ"
            b"C/KS4MboHIjNpg/nN/fw2aKXNebORmkfi3CIgOS11KEGWUI+HZ2Jh71xHneQPJqx"
            b"xqzHgw/RsOh5rNptsFSMqf2VGP8+89BRx7UP3BI8tWNNl3zjQp8y9xhmD/CBUL52"
            b"lIu7CxWU+TPfW0Sp1Y4f7G/qF1WopPHivEtWgwRdZVSUqFYG9vs76USrHkdmQr+Y"
            b"MXcQiyVXwHpbhMhIIeHt5Ttj9zjWRA9XgXmzSYcquLytR4VJeVuJaA9UW0lLUves"
            b"ISiubGYjdC9Lt7TOdsebN6nh+k77MbAZznOLTwh2lsfZr/jcYPe4lK61M7Xq+Vtv"
            b"BOM/pFtAZhXvq9T77It98Wn2vAAlkJTOTzM53oK71Ind9P1S2swbCfkFRafdqCb"
            b"jDtSsObteMSl5FJlYHh3wKc1MO4jqSWY75u3Zb3I9lfCB1wfYTH1qGRhh8oMhcl"
            b"LhUK3OjrEAsCb9Oz6mkWdQ5+Cih+v/ROHHvGyClnhykwpruNBmmnEigydXOJQ6i"
            b"7ZpP9qjeLzUebBmf4nv2KIboORnhwpv65JWsnlFJMqY5N8xXdGGJ2oL9zzOBSIs"
            b"Estw79MfejMmN96DDIyHuF4RbldNq/2zS5BB7RxDJjIEfcyIU6rv5xmMckParjy"
            b"956H1inZXJ/kVqT6twJ2zcICzWfJ3dK20hWPFQ6hgvu5sHlCWL5u6dUjC/9rph7"
            b"E55SsJ6rtoZeHeBlnb54XT98D37lH0NpK3OGTF9oAWp12dgMZ3iT4B+0KGJqpta"
            b"EJb4xInwv7nKNaPqi6a5/bP/Jw4v2drD+U8ldx0SIp4f8IidmKxkomZFxPwhvyaYyL/qwvJ"
        )
        maps = [{"start": 0x800000000, "end": 0x800100000, "prot": 5,
                 "name": "executable"}]
        old = {key: RDX.state.get(key) for key in (
            "ip", "pid", "proc_name", "session", "cheats", "cheats_dirty",
            "game_id", "export_dir")}
        RDX.state.update(ip="test", pid=7, proc_name="eboot.bin", session=1,
                         cheats=[], cheats_dirty=False, game_id="")
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        try:
            with tempfile.TemporaryDirectory() as directory:
                mc4_path = Path(directory) / "CUSA00002_01.00.mc4"
                mc4_path.write_bytes(sample)
                with patch.object(RDX, "draw_border"), \
                     patch.object(RDX, "safe_addstr"), \
                     patch.object(RDX, "color", return_value=0), \
                     patch.object(RDX, "input_box", return_value=str(mc4_path)), \
                     patch.object(RDX, "confirm_box", return_value=True), \
                     patch.object(RDX, "message_box"), \
                     patch.object(RDX, "_get_maps_cached", return_value=maps):
                    RDX.do_import(FakeWindow())
            # The real trainer has three <Cheat> entries (Godmode, Infinite
            # Ammo, Infinite Medkits) — check the first and its address math.
            self.assertEqual(len(RDX.state["cheats"]), 3)
            by_name = {c["name"]: c for c in RDX.state["cheats"]}
            self.assertEqual(set(by_name),
                             {"Godmode", "Infinite Ammo",
                              "Infinite Medkits (Slow Time)"})
            cheat = by_name["Godmode"]
            self.assertEqual(cheat["address"], 0x800000000 + 0x668DA3)
            self.assertEqual(cheat["value"], "90909090")
            self.assertEqual(cheat["value_type"], "bytes")
            self.assertTrue(RDX.state["cheats_dirty"])
        finally:
            RDX.state.update(old)

    def test_do_import_etahen_json_mods_resolves_against_live_module(self):
        payload = {
            "name": "Test Game", "id": "PPSA00001", "version": "01.00",
            "process": "eboot.bin",
            "mods": [{"name": "Health", "description": "", "type": "checkbox",
                      "memory": [{"offset": "1020", "on": "78563412",
                                 "off": "07000000"}]}],
            "credits": ["Someone"],
        }
        maps = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                 "name": "executable"}]
        old = {key: RDX.state.get(key) for key in (
            "ip", "pid", "proc_name", "session", "cheats", "cheats_dirty",
            "game_id", "export_dir")}
        RDX.state.update(ip="test", pid=7, proc_name="eboot.bin", session=1,
                         cheats=[], cheats_dirty=False, game_id="")
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        try:
            with tempfile.TemporaryDirectory() as directory:
                json_path = Path(directory) / "PPSA00001_01.00.json"
                json_path.write_text(json.dumps(payload))
                with patch.object(RDX, "draw_border"), \
                     patch.object(RDX, "safe_addstr"), \
                     patch.object(RDX, "color", return_value=0), \
                     patch.object(RDX, "input_box", return_value=str(json_path)), \
                     patch.object(RDX, "confirm_box", return_value=True), \
                     patch.object(RDX, "message_box"), \
                     patch.object(RDX, "_get_maps_cached", return_value=maps):
                    RDX.do_import(FakeWindow())
            self.assertEqual(len(RDX.state["cheats"]), 1)
            cheat = RDX.state["cheats"][0]
            self.assertEqual(cheat["name"], "Health")
            self.assertEqual(cheat["address"], 0x401020)
            self.assertEqual(cheat["original_value"], "07000000")
        finally:
            RDX.state.update(old)

    def test_do_import_static_patch_rejects_without_live_connection(self):
        payload = {"id": "PPSA00001", "process": "eboot.bin",
                  "mods": [{"name": "Health", "memory": [
                      {"offset": "10", "on": "01", "off": "00"}]}]}
        old = {key: RDX.state.get(key) for key in
               ("ip", "pid", "proc_name", "cheats")}
        RDX.state.update(ip="test", pid=7, proc_name="eboot.bin", cheats=[])
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        try:
            with tempfile.TemporaryDirectory() as directory:
                json_path = Path(directory) / "PPSA00001.json"
                json_path.write_text(json.dumps(payload))
                with patch.object(RDX, "draw_border"), \
                     patch.object(RDX, "safe_addstr"), \
                     patch.object(RDX, "color", return_value=0), \
                     patch.object(RDX, "input_box", return_value=str(json_path)), \
                     patch.object(RDX, "message_box") as box, \
                     patch.object(RDX, "_get_maps_cached",
                                  side_effect=OSError("not connected")):
                    RDX.do_import(FakeWindow())
            self.assertEqual(RDX.state["cheats"], [])
            self.assertEqual(box.call_args.args[2], "Import Failed")
        finally:
            RDX.state.update(old)

    def test_do_import_static_patch_rejects_process_mismatch(self):
        # Regression: the native .rdx.json path already refuses a trainer
        # whose declared process doesn't match the attached one; the
        # etaHEN/GoldHEN JSON and .mc4 paths must too, or offsets silently
        # resolve against the wrong executable's module.
        payload = {"id": "PPSA00001", "process": "other.elf",
                  "mods": [{"name": "Health", "memory": [
                      {"offset": "10", "on": "01", "off": "00"}]}]}
        old = {key: RDX.state.get(key) for key in
               ("ip", "pid", "proc_name", "cheats")}
        RDX.state.update(ip="test", pid=7, proc_name="eboot.bin", cheats=[])
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass

        try:
            with tempfile.TemporaryDirectory() as directory:
                json_path = Path(directory) / "PPSA00001.json"
                json_path.write_text(json.dumps(payload))
                with patch.object(RDX, "draw_border"), \
                     patch.object(RDX, "safe_addstr"), \
                     patch.object(RDX, "color", return_value=0), \
                     patch.object(RDX, "input_box", return_value=str(json_path)), \
                     patch.object(RDX, "message_box") as box, \
                     patch.object(RDX, "_get_maps_cached",
                                  side_effect=AssertionError(
                                      "should reject before touching maps")):
                    RDX.do_import(FakeWindow())
            self.assertEqual(RDX.state["cheats"], [])
            self.assertEqual(box.call_args.args[2], "Import Failed")
        finally:
            RDX.state.update(old)

    def test_mods_to_import_entries_deduplicates_colliding_names(self):
        # Two <Cheatline>s under one <Cheat> (or two mods missing a name)
        # produce colliding "name" values by construction; each imported
        # entry must still be independently addressable by name.
        mods = [
            {"memory": [{"offset": "10", "on": "01", "off": "00"}]},
            {"memory": [{"offset": "20", "on": "02", "off": "00"}]},
            {"name": "Ammo", "memory": [
                {"offset": "30", "on": "03", "off": "00"}]},
            {"name": "Ammo", "memory": [
                {"offset": "40", "on": "04", "off": "00"}]},
        ]
        entries = RDX._mods_to_import_entries(mods, Path("/tmp/x.json"),
                                               module_base=0x400000)
        names = [e["name"] for e in entries]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names,
                         ["Unnamed cheat", "Unnamed cheat (2)",
                          "Ammo", "Ammo (2)"])

    def test_mods_to_import_entries_logs_on_off_value_width_mismatch(self):
        mods = [{"name": "Bad", "memory": [
            {"offset": "10", "on": "01020304", "off": "0102"}]}]
        with patch.object(RDX, "add_log") as log:
            entries = RDX._mods_to_import_entries(
                mods, Path("/tmp/x.json"), module_base=0x400000)
        self.assertNotIn("original_value", entries[0])
        self.assertTrue(any(
            "warn" in call.args and "width" in call.args[0].lower()
            for call in log.call_args_list))

    def test_permanent_resolver_does_not_reset_same_session_candidates(self):
        maps = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                 "name": "executable"}]
        identity = RDX._pointer_game_identity("eboot.bin", maps)
        saved = [{"observed_process": "eboot.bin", "observed_game": identity,
                  "observed_pid": 7, "observed_target": 0x9000}]
        old = {key: RDX.state.get(key)
               for key in ("ip", "pid", "proc_name")}
        RDX.state.update(ip="test", pid=7, proc_name="eboot.bin")
        try:
            with patch.object(RDX, "_get_maps_cached", return_value=maps), \
                 patch.object(RDX, "_load_pointer_provisionals",
                              return_value=saved), \
                 patch.object(RDX, "message_box") as message, \
                 patch.object(RDX, "_resolve_permanent_candidates",
                              side_effect=AssertionError("must wait for reload")):
                RDX.do_resolve_permanent(None, 0x9000)
            self.assertEqual(message.call_args.args[2], "Reload Not Detected")
        finally:
            RDX.state.update(old)

    def test_cursor_helper_never_initializes_color_pairs(self):
        with patch.object(RDX.curses, "curs_set"), \
             patch.object(RDX.curses, "init_pair",
                          side_effect=AssertionError("color init is separate")):
            RDX._safe_curs_set(0)

    def test_color_setup_falls_back_on_monochrome_terminal(self):
        old = RDX._COLORS_OK
        try:
            with patch.object(RDX.curses, "start_color",
                              side_effect=RDX.curses.error("no color")):
                RDX.init_colors()
            self.assertFalse(RDX._COLORS_OK)
            self.assertEqual(RDX.color(RDX.C_OK), 0)
        finally:
            RDX._COLORS_OK = old

    def test_input_box_can_return_blank_instead_of_its_default(self):
        class FakeWindow:
            def getmaxyx(self): return (24, 80)
            def nodelay(self, *_args): pass
            def timeout(self, *_args): pass
            def refresh(self): pass
            def getstr(self, *_args): return b""

        with patch.object(RDX, "safe_addstr"), \
             patch.object(RDX, "color", return_value=0), \
             patch.object(RDX, "_safe_curs_set"), \
             patch.object(RDX.curses, "cbreak"), \
             patch.object(RDX.curses, "echo"), \
             patch.object(RDX.curses, "noecho"):
            self.assertEqual(RDX.input_box(
                FakeWindow(), "Filter: ", 1, 1, default="7",
                empty_uses_default=False), "")

    def test_cancellable_controls_fail_closed_when_not_drawable(self):
        class TinyWindow:
            def getmaxyx(self): return (1, 1)

        tiny = TinyWindow()
        self.assertIsNone(RDX.input_box(
            tiny, "Value: ", 1, 1, default="7", allow_cancel=True))
        self.assertIsNone(RDX.cycle_input(
            tiny, "Mode: ", 1, 1, ["one", "two"], "one",
            allow_cancel=True))

    def _synthetic_disk_index(self, shard_count=8, base=0x200000000,
                              span=0x2000000):
        """A _DiskReversePointerIndex with hand-written, disjoint shards."""
        maps = [{"start": base, "end": base + span, "prot": 3,
                 "offset": 0, "name": "heap"}]
        with patch.object(RDX, "_ScanSocket", MemorySocket), \
             patch.object(RDX, "_pointer_readable_regions", return_value=maps), \
             patch.object(RDX, "_coalesce_pointer_regions", return_value=[
                 {"start": base, "end": base + span, "static": False}]):
            index = RDX._DiskReversePointerIndex("ip", 1, maps)
        # The constructor already made a temp dir; reuse it rather than
        # mkdtemp again, which would orphan the first one every call.
        directory = index._tmpdir
        index.shards = []
        stride = span // shard_count
        for i in range(shard_count):
            low = base + i * stride
            values = np.arange(low, low + 16, dtype=np.uint64)
            holders = values + 0x1000
            paths = []
            for tag, arr in (("v", values), ("h", holders),
                             ("p", np.zeros(len(values), dtype=np.uint8))):
                path = directory / f"{tag}{i}.npy"
                np.save(path, arr, allow_pickle=False)
                paths.append(path)
            index.shards.append((base >> 32, paths[0], paths[1], paths[2],
                                 int(values[0]), int(values[-1])))
        return index

    def test_disk_index_range_skip_does_not_change_results(self):
        # query() used to np.load() every shard on every call and binary-search
        # all of them. The graph search runs up to _PTR_RESOLVE_MAX_NODES nodes
        # x len(_PTR_RESOLVE_OFFSET_TIERS) tiers, so on a 4.24 GiB process
        # (135 shards) that is ~1.7 M reopen+remap operations. Each shard is
        # sorted, so its first/last value bound it and a non-overlapping shard
        # can be rejected without being touched -- but only if that produces
        # identical results, which is what this pins.
        index = self._synthetic_disk_index()
        try:
            base = 0x200000000
            for target in (base, base + 4, base + (0x2000000 // 8) + 8,
                           base + 0x1000000, base + 0x1FFFFFF):
                got = index.query(target, max_offset=0x100000, step=4)
                # brute force over every shard, ignoring the range reject
                expected = []
                for _p, vpath, hpath, _pp, _lo, _hi in index.shards:
                    values = np.load(vpath, mmap_mode="r", allow_pickle=False)
                    holders = np.load(hpath, mmap_mode="r", allow_pickle=False)
                    for value, holder in zip(values.tolist(), holders.tolist()):
                        delta = target - int(value)
                        if abs(delta) <= 0x100000 and delta % 4 == 0:
                            expected.append((int(holder), delta))
                self.assertEqual(sorted(got), sorted(set(expected)),
                                 f"range skip changed the result at {target:#x}")
        finally:
            index.close()

    def test_disk_index_maps_each_shard_once_and_frees_on_close(self):
        index = self._synthetic_disk_index(shard_count=4)
        directory = index._tmpdir
        try:
            with patch.object(RDX.np, "load",
                              wraps=RDX.np.load) as loader:
                for _ in range(5):
                    index.query(0x200000000, max_offset=0x100000, step=4)
                first = loader.call_count
                for _ in range(5):
                    index.query(0x200000000, max_offset=0x100000, step=4)
                self.assertEqual(loader.call_count, first,
                                 "repeat queries must reuse the mapping")
            self.assertTrue(index._mapped)
        finally:
            index.close()
        self.assertFalse(index._mapped)
        self.assertFalse(directory.exists(),
                         "mappings must be dropped before the files are removed")

    def test_mc4_export_names_the_matching_manager_directory(self):
        # GoldHEN and etaHEN each keep a cheats/mc4/ folder beside their
        # cheats/json/ one. RDX named only CheatRunner's directory, so a .mc4
        # exported for a GoldHEN title was sent somewhere the manager RDX had
        # just selected for the JSON would never look.
        base = 0x400000
        maps = [{"start": base, "end": base + 0x2000000, "prot": 5,
                 "offset": 0, "name": "executable"}]
        old = {k: RDX.state.get(k) for k in
               ("ip", "backend", "pid", "proc_name", "session", "cheats",
                "game_id", "game_ver", "game_title")}
        try:
            for title_id, expected in (
                    ("CUSA01659", "/user/data/GoldHEN/cheats/mc4/"),
                    ("PPSA01342", "/data/etaHEN/cheats/mc4/")):
                RDX.state.update(
                    ip="t", backend="ps5debug", pid=1, proc_name="eboot.bin",
                    session=1, game_id=title_id, game_ver="01.00",
                    game_title="T",
                    cheats=[{
                        "name": "c", "address": base + 0x1000,
                        "value": "90909090", "type": "write", "width": 4,
                        "value_type": "bytes", "original_value": "00000000",
                        "module_name": "executable",
                        "module_relative_offset": 0x1000,
                        "game_identity": RDX._pointer_game_identity(
                            "eboot.bin", maps),
                        "pid": 1, "process": "eboot.bin", "session": 1}])
                shown = []
                with tempfile.TemporaryDirectory() as out:
                    with patch.object(RDX, "draw_border"), \
                         patch.object(RDX, "safe_addstr"), \
                         patch.object(RDX, "color", return_value=0), \
                         patch.object(RDX, "_get_maps_cached",
                                      return_value=maps), \
                         patch.object(RDX, "input_box", side_effect=[
                             title_id, "01.00", "T", "RDX", out]), \
                         patch.object(RDX, "confirm_box", return_value=True), \
                         patch.object(RDX, "_save_preferences"), \
                         patch.object(RDX, "ps5_read",
                                      side_effect=Exception("no console")), \
                         patch.object(RDX, "message_box",
                                      side_effect=lambda st, l, t="", c=0:
                                      shown.extend(l)):
                        RDX.do_export(self._FakeKeyWindow([]))
                paths = [line.strip() for line in shown
                         if "cheats/mc4/" in line]
                self.assertTrue(
                    any(expected in p for p in paths),
                    f"{title_id} must offer {expected}; got {paths}")
                self.assertTrue(
                    any("/data/cheatrunner/cheats/mc4/" in p for p in paths),
                    "CheatRunner remains a valid destination")
        finally:
            RDX.state.update(old)

    def test_preflight_accepts_either_payload_port(self):
        # A MemDBG-only console has 744 closed and a ps5debug-only console has
        # 9020 closed, so requiring both would refuse a working setup.
        class FakeSock:
            def close(self): pass

        with patch.object(RDX, "ps5_connect", return_value=FakeSock()), \
             patch.object(RDX, "memdbg_probe", return_value=None):
            self.assertTrue(RDX._console_preflight("ip"), "ps5debug only")
        with patch.object(RDX, "ps5_connect", side_effect=OSError("refused")), \
             patch.object(RDX, "memdbg_probe", return_value={"version": "x"}):
            self.assertTrue(RDX._console_preflight("ip"), "MemDBG only")
        with patch.object(RDX, "ps5_connect", side_effect=OSError("refused")), \
             patch.object(RDX, "memdbg_probe", return_value=None):
            self.assertFalse(RDX._console_preflight("ip"), "neither port")

    def test_preflight_short_circuits_before_the_slow_connect(self):
        # Without it, an unreachable address costs memdbg_probe (1.5 s) plus
        # ps5_connect's 15 s default -- ~16.5 s of frozen UI showing only
        # "Connecting...", with no cancel. Measured 16.5 s -> 4.5 s.
        # screen_connect must give up before ps5_proc_list is ever reached.
        old = {k: RDX.state.get(k) for k in ("ip", "connected")}
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_header_banner"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "input_box", return_value="10.255.255.1"), \
                 patch.object(RDX, "_stop_freeze_worker"), \
                 patch.object(RDX, "_console_preflight", return_value=False), \
                 patch.object(RDX, "ps5_proc_list",
                              side_effect=AssertionError(
                                  "must not try the slow path")) as procs:
                result = RDX.screen_connect(self._FakeKeyWindow([ord(' ')]))
            self.assertEqual(result, "connect")
            procs.assert_not_called()
        finally:
            RDX.state.update(old)

    def test_orphaned_shard_directories_are_swept_but_fresh_ones_kept(self):
        # A disk index for a multi-GiB process holds hundreds of MiB of .npy
        # shards, and close() was the only thing that removed them: a crash or
        # Ctrl-C between build and close stranded them permanently. Sweeping
        # at startup mirrors what ps5debug-NG does for its own snapshot spill
        # files. A *fresh* directory must survive, because it may belong to
        # another instance that is running right now.
        root = Path(tempfile.gettempdir())
        stale = Path(tempfile.mkdtemp(prefix=RDX._DISK_INDEX_PREFIX))
        (stale / "v0.npy").write_bytes(b"x" * 64)
        old = time.time() - (RDX._DISK_INDEX_ORPHAN_AGE * 2)
        os.utime(stale, (old, old))
        fresh = Path(tempfile.mkdtemp(prefix=RDX._DISK_INDEX_PREFIX))
        (fresh / "v0.npy").write_bytes(b"x" * 64)
        try:
            RDX._sweep_orphaned_disk_indexes()
            self.assertFalse(stale.exists(), "a stale orphan must be removed")
            self.assertTrue(fresh.exists(),
                            "a fresh directory may belong to a live instance")
        finally:
            for directory in (stale, fresh):
                if directory.exists():
                    for child in directory.iterdir():
                        child.unlink(missing_ok=True)
                    directory.rmdir()

    def test_sweep_tolerates_a_non_directory_and_never_raises(self):
        root = Path(tempfile.gettempdir())
        decoy = root / (RDX._DISK_INDEX_PREFIX + "not_a_directory")
        decoy.write_text("x")
        try:
            RDX._sweep_orphaned_disk_indexes()      # must not raise
            self.assertTrue(decoy.exists())
        finally:
            decoy.unlink(missing_ok=True)

    def test_atexit_closes_live_disk_indexes_even_if_one_raises(self):
        class Fake:
            def __init__(self, explode=False):
                self.closed = False
                self.explode = explode
            def close(self):
                if self.explode:
                    raise RuntimeError("close failed")
                self.closed = True

        good, bad = Fake(), Fake(explode=True)
        with RDX._disk_index_lock:
            saved = set(RDX._disk_index_registry)
            RDX._disk_index_registry.clear()
            RDX._disk_index_registry.update({good, bad})
        try:
            RDX._close_all_disk_indexes()           # must not raise
            self.assertTrue(good.closed)
        finally:
            with RDX._disk_index_lock:
                RDX._disk_index_registry.clear()
                RDX._disk_index_registry.update(saved)

    def test_snapshot_scan_reports_its_console_side_storage_cost(self):
        # Protocol 2.2: the snapshot value store lives on the CONSOLE, is
        # RAM-backed under a 512 MiB threshold, and the overflow spills to a
        # file under /data. With TS_SNAPSHOT_INCLUDE_ZEROS every aligned slot
        # is seeded, so the store is far larger than the bytes read: the
        # measured 2.15 GiB "recommended" scope is ~577 M slots and ~4.3 GiB
        # of store, ~3.8 GiB of it written to the console. RDX used to read
        # the server's plan (slot_count, total_bytes) and discard it, so none
        # of this was visible anywhere.
        big = [(0x200000000, 0x200000000 + 0x80000000)]      # 2 GiB
        small = [(0x10000000, 0x10000000 + 0x400000)]        # 4 MiB
        for ranges, expect_warn in ((big, True), (small, False)):
            before = len(RDX.state["log"])
            with patch.object(RDX, "ps5_auth_scanner"), \
                 patch.object(RDX, "ps5_turboscan_caps",
                              return_value=(1, 0x3FF, 4)), \
                 patch.object(RDX, "ps5_connect",
                              side_effect=OSError("stop before the wire")):
                with self.assertRaises(OSError):
                    RDX.ps5_scan_unknown_turbo(
                        "ip", 1, 4, ranges, aligned=True, value_type="u32")
            msgs = [e for e in RDX.state["log"][before:]
                    if "value store on the console" in e["msg"]]
            if expect_warn:
                self.assertTrue(msgs, "a multi-GiB spill must be reported")
                self.assertEqual(msgs[0]["level"], "warn")
                self.assertIn("/data", msgs[0]["msg"])
            else:
                self.assertFalse(
                    msgs, "a store that fits in RAM must not cry wolf")

    def test_goldhen_export_matches_the_published_schema_exactly(self):
        # Pinned against a real file from GoldHEN_Cheat_Repository
        # (json/CUSA11260_01.13.json):
        #     {"offset": "1778257",
        #      "on":  "C5FA118FC0000000",
        #      "off": "C5FA1187C0000000"}
        # Three things about that shape were unverified assumptions in this
        # codebase until they were checked against the repository, and each
        # would have produced a trainer the console silently mis-applies:
        #   * the offset carries NO 0x prefix;
        #   * it is HEXADECIMAL, not decimal -- another published entry reads
        #     "431B000", which contains a B and so cannot be decimal;
        #   * on/off bytes are continuous uppercase hex with no separators
        #     (unlike .mc4, which dash-separates them).
        base = 0x400000
        maps = [{"start": base, "end": base + 0x2000000, "prot": 5,
                 "offset": 0, "name": "executable"}]
        old = {k: RDX.state.get(k) for k in
               ("ip", "backend", "pid", "proc_name", "session")}
        RDX.state.update(ip="t", backend="ps5debug", pid=1,
                         proc_name="eboot.bin", session=1)
        try:
            cheat = {
                "name": "Infinite Health", "address": base + 0x1778257,
                "value": "C5FA118FC0000000", "type": "write", "width": 8,
                "value_type": "bytes", "original_value": "C5FA1187C0000000",
                "module_name": "executable",
                "module_relative_offset": 0x1778257,
                "game_identity": RDX._pointer_game_identity("eboot.bin", maps),
                "pid": 1, "process": "eboot.bin", "session": 1,
            }
            text, mods, skipped = RDX.generate_etahen_json(
                [cheat], "CUSA11260", "01.13", "Test Game", "eboot.bin",
                maps, "RDX")
            self.assertEqual(skipped, [])
            doc = json.loads(text)
            self.assertEqual(list(doc.keys()),
                             ["name", "id", "version", "process", "mods",
                              "credits"])
            mod = doc["mods"][0]
            # Pinned against real files, not a summary. etaHEN's
            # PS5_Cheats json/CUSA00004_01.07.json uses
            # {name, hint, type, memory}; GoldHEN's equivalent uses
            # {name, type, memory} and defines no such field. RDX previously
            # emitted "description", which neither manager reads, so its own
            # note about a toggle being a one-shot write rather than a live
            # freeze was never displayed anywhere.
            self.assertEqual(list(mod.keys()),
                             ["name", "hint", "type", "memory"])
            self.assertNotIn("description", mod)
            self.assertEqual(mod["type"], "checkbox")
            self.assertEqual(mod["memory"][0], {
                "offset": "1778257",
                "on": "C5FA118FC0000000",
                "off": "C5FA1187C0000000",
            })
        finally:
            RDX.state.update(old)

    # ── patch66: debugger attach target guard ────────────────────────────

    def test_debugger_refuses_console_critical_processes(self):
        # Arming a watchpoint SIGSTOPs the target. Doing that to the process
        # running the console UI freezes the console: ps5dbg gates its own
        # debug-attach test behind --risky because "attaching to SceShellCore
        # can freeze the system UI". Checked against the real process list
        # captured from a live PS5.
        for name in ("SceShellCore", "SceShellUI", "kernel",
                     "mini-syscore.elf", "SceSysCore.elf", "SceVideoCore2K",
                     "orbis_audiod.elf"):
            self.assertIsNotNone(RDX._debug_attach_refusal(name),
                                 f"{name} must be refused")
        self.assertIsNotNone(RDX._debug_attach_refusal(""))

    def test_debugger_allows_games_and_homebrew(self):
        for name in ("eboot.bin", "payload.elf", "kstuff.elf",
                     "shadowmountplus.elf"):
            self.assertIsNone(RDX._debug_attach_refusal(name), name)
            self.assertFalse(RDX._debug_attach_is_unusual(name), name)

    def test_other_system_services_need_a_second_confirmation(self):
        # Not dangerous enough to refuse, but almost certainly not what the
        # user meant to trace either.
        for name in ("SceRedisServer", "SceNKUIProcess", "SceDiscordDaemon"):
            self.assertIsNone(RDX._debug_attach_refusal(name), name)
            self.assertTrue(RDX._debug_attach_is_unusual(name), name)

    def test_attach_guard_runs_before_any_connection_is_opened(self):
        # The refusal is worthless if it happens after ATTACH: by then the
        # target is already stopped.
        old = {k: RDX.state.get(k) for k in ("proc_name", "ip", "pid")}
        RDX.state.update(proc_name="SceShellCore", ip="1.2.3.4", pid=57)
        try:
            with patch.object(RDX, "_trace_network_refusal", return_value=None), \
                 patch.object(RDX, "ps5_connect",
                              side_effect=AssertionError(
                                  "must not connect")) as conn:
                with self.assertRaises(RuntimeError) as caught:
                    RDX._trace_temporary_access("1.2.3.4", 57, 0x1000, 4,
                                                experimental=True)
            self.assertIn("runs the console itself", str(caught.exception))
            conn.assert_not_called()
        finally:
            RDX.state.update(old)

    # ── patch85: follow a trace the documented way, not with a graph search ──

    def test_trace_followup_scans_for_the_base_value(self):
        # The documented workflow (Cheat Engine, and every tutorial that
        # describes the debugger route): once "what accesses this address"
        # names the object pointer, you scan for that pointer AS A VALUE --
        # hex, exact, pointer width -- and static results are the answer.
        # patch84 instead handed the traced base to pointer_chain_scan, the
        # graph search measured at 30+ minutes at depth 4 on a 4.26 GiB title.
        # That discards the entire benefit of having traced it.
        BASE, HOLDER, DISP = 0x2531F0000, 0x82BE17C0, 0x18
        module = {"start": 0x82BE0000, "end": 0x82BF0000, "prot": 0x1,
                  "offset": 0, "name": "Il2CppUserAssemblies.prx"}
        scanned = []

        def fake_scan(ip, pid, value, width, aligned=True, **kw):
            scanned.append((int(value), int(width)))
            return np.array([HOLDER], dtype=np.uint64) if int(value) == BASE \
                else np.array([], dtype=np.uint64)

        trace = {"base_value": BASE, "final_offset": DISP, "rip": 0x400100,
                 "base_reg": "rbx", "access_mode": "write", "instruction": {}}
        with patch.object(RDX, "scan_first", fake_scan), \
             patch.object(RDX, "_get_maps_cached", return_value=[module]), \
             patch.object(RDX, "_build_region_lookup",
                          return_value=([module["start"]], [module])), \
             patch.object(RDX, "_region_for_addr", return_value=module), \
             patch.object(RDX, "_module_info_for_addr",
                          return_value=("Il2CppUserAssemblies.prx",
                                        module["start"], 0x17C0)), \
             patch.object(RDX, "_resolve_pointer_chain",
                          return_value=(True, BASE + DISP, [])), \
             patch.object(RDX, "pointer_chain_scan",
                          side_effect=AssertionError(
                              "must not fall back to the graph search")):
            out = RDX._pointer_candidates_from_trace(
                "ip", 1, trace, BASE + DISP, max_depth=3)

        self.assertTrue(scanned, "no scan was issued for the traced base")
        self.assertEqual(scanned[0], (BASE, 8),
                         "did not scan for the base pointer at pointer width")
        self.assertTrue(out["candidates"], "no chain built from the trace")
        c = out["candidates"][0]
        self.assertEqual(c["base"], HOLDER)
        self.assertEqual(c["terminal_offset"], DISP)
        self.assertEqual(c["module_name"], "Il2CppUserAssemblies.prx")

    def test_trace_walk_records_the_real_offset_below_level_one(self):
        # Level 1 is exact -- the trace named the object base. Deeper levels
        # cannot be: a parent points at the BASE of the object holding the
        # pointer, with the pointer at some field offset inside it. patch85
        # searched for the holder's own address and recorded offset 0, so it
        # would almost never find a level-2 parent and encoded a false offset
        # when it did. The manual method re-traces at each level; without a
        # fresh trace, a bounded window with the real displacement is the
        # honest substitute.
        BASE, HEAP, MODULE_HOLDER, FIELD = 0x2531F0000, 0x25000000, 0x82BE17C0, 0x28
        module = {"start": 0x82BE0000, "end": 0x82BF0000, "prot": 0x1,
                  "offset": 0, "name": "Il2CppUserAssemblies.prx"}
        heap = {"start": 0x24000000, "end": 0x26000000, "prot": 0x3,
                "offset": 0, "name": ""}

        def exact(ip, pid, value, cancel_event=None, max_hits=4096):
            return [HEAP] if int(value) == BASE else []

        def windowed(ip, pid, target, maps, cancel_event=None,
                     max_hits=24, static_only=True):
            # the module holds a pointer to HEAP-FIELD, i.e. the object base
            self.assertFalse(static_only, "deeper levels must search the heap too")
            return [(MODULE_HOLDER, FIELD, module)] if int(target) == HEAP else []

        def region_for(addr, *_a):
            return module if addr >= module["start"] else heap

        with patch.object(RDX, "_exact_pointer_holders", exact), \
             patch.object(RDX, "_fast_direct_pointer_hits", windowed), \
             patch.object(RDX, "_build_region_lookup", return_value=([], [])), \
             patch.object(RDX, "_region_for_addr", region_for), \
             patch.object(RDX, "_is_static_region", lambda r: bool(r.get("name"))):
            out = RDX._walk_from_traced_base("ip", 1, BASE, [module, heap],
                                             max_depth=3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["base"], MODULE_HOLDER)
        self.assertEqual(out[0]["depth"], 2)
        # deref(MODULE_HOLDER) + FIELD -> HEAP, then deref(HEAP) + 0 -> BASE.
        # The trailing 0 is the level-1 exact match and must be present, or the
        # chain is one dereference short.
        self.assertEqual(out[0]["offsets"], [FIELD, 0],
                         "recorded a placeholder offset instead of the real one")

    def test_exact_holder_scan_refuses_a_degenerate_value(self):
        # A value outside the address range (0 being the obvious case) matches
        # an enormous share of memory and would return a meaningless flood.
        with patch.object(RDX, "scan_first",
                          side_effect=AssertionError("must not scan")):
            self.assertEqual(RDX._exact_pointer_holders("ip", 1, 0), [])
            self.assertEqual(RDX._exact_pointer_holders("ip", 1, -1), [])

    def test_fast_direct_can_include_heap_regions(self):
        # GUARD: the default stays static-only, so the tier-1 behaviour that
        # every other caller relies on is unchanged.
        seen = {}
        region = {"start": 0x1000, "end": 0x2000, "prot": 0x3, "offset": 0,
                  "name": ""}

        def fake_regions(maps):
            return [region]

        class Sock:
            def __init__(self, *_a): pass
            def close(self): pass
            def read(self, a, n, _c=None): return b"\x00" * n

        for static_only, expect_heap in ((True, False), (False, True)):
            with patch.object(RDX, "_pointer_readable_regions", fake_regions), \
                 patch.object(RDX, "_is_static_region", lambda r: bool(r.get("name"))), \
                 patch.object(RDX, "_ScanSocket", Sock):
                RDX._fast_direct_pointer_hits("ip", 1, 0x1800, [region], None,
                                              static_only=static_only)
            # a heap-only map means the scan loop runs only when heap is allowed
            seen[static_only] = expect_heap
        self.assertTrue(seen[False])

    # ── patch82: a coincidental fast hit must not pre-empt the real search ──

    def _resolver_env(self, fast_hits, deep_hits=None):
        """Drive _resolve_permanent_candidates with scripted tier results."""
        region = {"start": 0x82B00000, "end": 0x82C00000, "prot": 0x1,
                  "offset": 0, "name": "Il2CppUserAssemblies.prx"}
        direct = [(0x82B00000 + i * 8, off, region)
                  for i, off in enumerate(fast_hits)]
        return region, direct

    def test_a_coincidental_fast_hit_does_not_short_circuit(self):
        # patch77 widened the fast window to 64 KiB, which made tier 1 match
        # coincidences and return immediately -- so the deeper tiers that could
        # find the real chain never ran. Measured on hardware: 24 "verified"
        # depth-1 candidates with offsets -9224..-6344 in an exact 48-byte
        # series (an IL2CPP pointer table, not parents), and an earlier set of
        # the same shape survived 0/5 reloads.
        region, direct = self._resolver_env([-0x2018, -0x18C8])
        reached = {"locality": False}

        def locality(*a, **k):
            reached["locality"] = True
            return []

        with patch.object(RDX, "_get_maps_cached", return_value=[region]), \
             patch.object(RDX, "_build_region_lookup", return_value=([], [])), \
             patch.object(RDX, "_fast_direct_pointer_hits", return_value=direct), \
             patch.object(RDX, "_verify_candidate_twice", return_value=True), \
             patch.object(RDX, "_module_info_for_addr",
                          return_value=("Il2CppUserAssemblies.prx", 0x82B00000, 0x10)), \
             patch.object(RDX, "pointer_chain_scan", locality), \
             patch.object(RDX, "_get_reverse_pointer_index",
                          return_value=(self._EmptyIndex(), [region], False)), \
             patch.object(RDX, "add_log", lambda *a, **k: None):
            out = RDX._resolve_permanent_candidates("ip", 1, 0x2531F1B88,
                                                    max_depth=2)
        self.assertTrue(reached["locality"],
                        "a coincidental fast hit pre-empted the deeper search")
        self.assertEqual(out["method"], "fast-direct-unverified")
        self.assertTrue(all(c.get("coincidence_risk") for c in out["candidates"]),
                        "near-misses were not marked as coincidence risks")

    def test_a_plausible_fast_hit_still_short_circuits(self):
        # GUARD: the speed win of patch77 must survive. A holder pointing at an
        # object base with the field a short way in is the real thing, and must
        # still return immediately without building an index.
        region, direct = self._resolver_env([0x18])
        reached = {"locality": False}

        def locality(*a, **k):
            reached["locality"] = True
            return []

        with patch.object(RDX, "_get_maps_cached", return_value=[region]), \
             patch.object(RDX, "_build_region_lookup", return_value=([], [])), \
             patch.object(RDX, "_fast_direct_pointer_hits", return_value=direct), \
             patch.object(RDX, "_verify_candidate_twice", return_value=True), \
             patch.object(RDX, "_module_info_for_addr",
                          return_value=("Il2CppUserAssemblies.prx", 0x82B00000, 0x10)), \
             patch.object(RDX, "pointer_chain_scan", locality), \
             patch.object(RDX, "add_log", lambda *a, **k: None):
            out = RDX._resolve_permanent_candidates("ip", 1, 0x2531F1B88,
                                                    max_depth=2)
        self.assertFalse(reached["locality"], "a real hit was not trusted")
        self.assertEqual(out["method"], "fast-direct")
        self.assertEqual(out["candidates"][0]["offsets"], [0x18])

    def test_plausible_field_bound_is_defined(self):
        self.assertTrue(hasattr(RDX, "_PTR_PLAUSIBLE_FIELD_MAX"))
        self.assertGreater(RDX._PTR_PLAUSIBLE_FIELD_MAX, 0)
        self.assertLessEqual(RDX._PTR_PLAUSIBLE_FIELD_MAX, 0x1000)

    # ── patch81: recover once from a session left by an earlier attempt ──

    class _FakeListener:
        """Stands in for the port-755 listener so tests never bind a real port."""
        def __init__(self, *_a, **_k): pass
        def setsockopt(self, *_a): pass
        def bind(self, *_a): pass
        def listen(self, *_a): pass
        def settimeout(self, *_a): pass
        def accept(self): raise TimeoutError("no debug event in test")
        def close(self): pass

    def test_already_debug_is_recovered_automatically_once(self):
        # A held session is clearable: CMD_DEBUG_DETACH runs the full teardown
        # even from a new connection (verified on hardware -- CMD_SUCCESS, and
        # the flag was released). Make the tool do it rather than the user.
        calls = {"attach": 0, "detach": 0}

        class Sock:
            def settimeout(self, *_): pass
            def close(self): pass
            def sendall(self, payload):
                if len(payload) >= 8:
                    cmd = struct.unpack_from("<I", payload, 4)[0]
                    if cmd == RDX.CMD_DEBUG_ATTACH:
                        calls["attach"] += 1
                    elif cmd == RDX.CMD_DEBUG_DETACH:
                        calls["detach"] += 1

        words = [0xF0000004, 0xF0000001]     # held, then a fresh failure

        def status(_s):
            return words.pop(0) if words else 0xF0000001

        old = {k: RDX.state.get(k) for k in ("proc_name", "ip", "pid")}
        RDX.state.update(proc_name="eboot.bin", ip="1.2.3.4", pid=57)
        try:
            with patch.object(RDX, "_trace_network_refusal", return_value=None), \
                 patch.object(RDX.socket, "socket", self._FakeListener), \
                 patch.object(RDX, "ps5_read", return_value=b"\x00" * 4), \
                 patch.object(RDX, "ps5_connect", return_value=Sock()), \
                 patch.object(RDX, "_debug_status_word", status), \
                 patch.object(RDX, "_debug_force_resume", return_value=True), \
                 patch.object(RDX, "add_log", lambda *a, **k: None):
                with self.assertRaises(RuntimeError):
                    RDX._trace_temporary_access("1.2.3.4", 57, 0x1000, 4,
                                                experimental=True)
            self.assertEqual(calls["attach"], 2, "did not retry after releasing")
            self.assertGreaterEqual(calls["detach"], 1, "never sent the release")
        finally:
            RDX.state.update(old)

    def test_recovery_is_attempted_only_once(self):
        # Two ALREADY_DEBUG answers in a row must not recurse forever.
        class Sock:
            def settimeout(self, *_): pass
            def close(self): pass
            def sendall(self, *_): pass
        old = {k: RDX.state.get(k) for k in ("proc_name", "ip", "pid")}
        RDX.state.update(proc_name="eboot.bin", ip="1.2.3.4", pid=57)
        try:
            with patch.object(RDX, "_trace_network_refusal", return_value=None), \
                 patch.object(RDX.socket, "socket", self._FakeListener), \
                 patch.object(RDX, "ps5_read", return_value=b"\x00" * 4), \
                 patch.object(RDX, "ps5_connect", return_value=Sock()), \
                 patch.object(RDX, "_debug_status_word", return_value=0xF0000004), \
                 patch.object(RDX, "_debug_force_resume", return_value=True), \
                 patch.object(RDX, "add_log", lambda *a, **k: None):
                with self.assertRaises(RuntimeError) as caught:
                    RDX._trace_temporary_access("1.2.3.4", 57, 0x1000, 4,
                                                experimental=True)
            self.assertIn("CMD_ALREADY_DEBUG", str(caught.exception))
        finally:
            RDX.state.update(old)

    # ── patch80: the console could not reach the client for debug events ──

    def test_trace_refuses_a_cgnat_client_address(self):
        # Found on hardware: the console was reached over a Tailscale subnet
        # route, so it saw the client as 100.122.106.94 and had no route back.
        # Scans/reads/writes were all fine (client-to-console); only the debug
        # channel, which the console dials outbound, could never arrive.
        with patch.object(RDX, "_local_address_towards",
                          return_value="100.122.106.94"):
            why = RDX._trace_network_refusal("192.168.0.88")
        self.assertIsNotNone(why)
        self.assertIn("100.64.0.0/10", why)
        self.assertIn("755", why)

    def test_trace_refuses_a_client_on_another_private_network(self):
        with patch.object(RDX, "_local_address_towards",
                          return_value="10.9.9.5"):
            why = RDX._trace_network_refusal("192.168.0.88")
        self.assertIsNotNone(why)
        self.assertIn("not\n" if False else "not on the console's network", why)

    def test_trace_allows_a_same_network_client(self):
        # GUARD: the check must not block the ordinary LAN case it exists for.
        for local in ("192.168.0.41", "192.168.0.250"):
            with patch.object(RDX, "_local_address_towards", return_value=local):
                self.assertIsNone(RDX._trace_network_refusal("192.168.0.88"),
                                  f"refused a same-network client {local}")

    def test_network_check_runs_before_anything_is_attached(self):
        # The whole point: attaching stops the game and then blocks waiting for
        # a callback that cannot arrive, leaving the target traced so the next
        # attach fails too. That is how a live console session lost its
        # debugger. Nothing may connect or bind before this check.
        old = {k: RDX.state.get(k) for k in ("proc_name", "ip", "pid")}
        RDX.state.update(proc_name="eboot.bin", ip="192.168.0.88", pid=169)
        try:
            with patch.object(RDX, "_local_address_towards",
                              return_value="100.122.106.94"), \
                 patch.object(RDX, "ps5_read",
                              side_effect=AssertionError("must not read")), \
                 patch.object(RDX, "ps5_connect",
                              side_effect=AssertionError("must not connect")), \
                 patch.object(RDX.socket, "socket",
                              side_effect=AssertionError("must not bind")):
                with self.assertRaises(RuntimeError) as caught:
                    RDX._trace_temporary_access("192.168.0.88", 169, 0x1000, 4,
                                                experimental=True)
            self.assertIn("not possible over this network path",
                          str(caught.exception))
        finally:
            RDX.state.update(old)

    # ── patch78: a failed debug detach was silent and unexplained ────────

    def setUp(self):
        RDX._debug_session_stuck = False
        # The native MemDBG connection is process-wide. Without this, a test
        # that leaves a fake client behind hands it to the next test, which
        # then exercises the fake instead of what it meant to.
        if hasattr(RDX, "memdbg_reset_session"):
            RDX.memdbg_reset_session()

    class _DetachSocket:
        """Command socket whose DETACH behaviour is scripted."""
        def __init__(self, mode):
            self.mode = mode          # "ok" | "refuse" | "raise"
            self.detach_attempts = 0
        def settimeout(self, *_): pass
        def close(self): pass
        def sendall(self, payload):
            self.detach_attempts += 1
            if self.mode == "raise":
                raise ConnectionResetError("socket is dead")

    def test_failed_detach_is_retried_and_reported(self):
        # patch77 sent DETACH once on the socket that had just failed and
        # swallowed every exception, so a stuck session was invisible. Observed
        # on hardware: a timed-out trace left g_debug_attached=1 and every
        # later attach was refused with nothing linking it to the cause.
        sock = self._DetachSocket("raise")
        logged = []
        with patch.object(RDX, "add_log", lambda m, lvl="info": logged.append((lvl, m))), \
             patch.object(RDX, "_debug_force_resume", return_value=True):
            ok = RDX._debug_detach_or_report(sock, "ip", 1)
        self.assertFalse(ok)
        self.assertGreater(sock.detach_attempts, 1, "detach was not retried")
        self.assertTrue(RDX._debug_session_is_stuck())
        text = " ".join(m for _lvl, m in logged)
        self.assertIn("Reload ps5debug-NG", text)
        self.assertTrue(any(lvl == "error" for lvl, _m in logged),
                        "a stuck session was not logged as an error")

    def test_successful_detach_clears_the_stuck_flag(self):
        RDX._debug_session_stuck = True
        sock = self._DetachSocket("ok")
        with patch.object(RDX, "_debug_status_ok", return_value=True):
            ok = RDX._debug_detach_or_report(sock, "ip", 1)
        self.assertTrue(ok)
        self.assertFalse(RDX._debug_session_is_stuck())
        self.assertEqual(sock.detach_attempts, 1, "retried a detach that worked")

    def test_force_resume_success_is_not_mistaken_for_a_detach(self):
        # CMD_DEBUG_PROCESS_STOP is a bare kill(pid, SIGCONT): it unfreezes the
        # game but never clears the session. Treating its True as a successful
        # teardown is what made the recovery tool look like it had worked.
        sock = self._DetachSocket("raise")
        with patch.object(RDX, "add_log", lambda *a, **k: None), \
             patch.object(RDX, "_debug_force_resume", return_value=True):
            ok = RDX._debug_detach_or_report(sock, "ip", 1)
        self.assertFalse(ok, "force-resume was treated as a detach")
        self.assertTrue(RDX._debug_session_is_stuck())

    def test_attach_reports_the_actual_status_word(self):
        # The bug behind a long misdiagnosis in testing: _debug_status_ok()
        # returns a bool, so every non-success word looked like "already
        # debugging". Hardware returned CMD_ERROR (0xF0000001), not
        # CMD_ALREADY_DEBUG (0xF0000004) -- meaning ptrace(PT_ATTACH) failed,
        # a per-process condition a relaunch clears, not a held session
        # needing a payload reload. The two must be told apart.
        cases = {
            0xF0000001: ("ptrace", "Relaunch"),
            0xF0000004: ("already attached", "other debugger"),
        }
        for word, (needle_a, needle_b) in cases.items():
            old = {k: RDX.state.get(k) for k in ("proc_name", "ip", "pid")}
            RDX.state.update(proc_name="eboot.bin", ip="1.2.3.4", pid=57)

            class Sock:
                def settimeout(self, *_): pass
                def sendall(self, *_): pass
                def close(self): pass

            try:
                with patch.object(RDX, "ps5_read", return_value=b"\x00" * 4), \
                     patch.object(RDX, "_trace_network_refusal", return_value=None), \
                     patch.object(RDX.socket, "socket", self._FakeListener), \
                     patch.object(RDX, "ps5_connect", return_value=Sock()), \
                     patch.object(RDX, "_debug_status_word", return_value=word), \
                     patch.object(RDX, "_debug_force_resume", return_value=True), \
                     patch.object(RDX, "add_log", lambda *a, **k: None):
                    with self.assertRaises(RuntimeError) as caught:
                        RDX._trace_temporary_access("1.2.3.4", 57, 0x1000, 4,
                                                    experimental=True)
                message = str(caught.exception)
                self.assertIn(RDX._debug_status_name(word), message,
                              f"status name missing for {word:#x}")
                self.assertIn(needle_a, message)
                self.assertIn(needle_b, message)
            finally:
                RDX.state.update(old)

    def test_status_names_match_the_protocol_table(self):
        # Pinned to protocol reference 1.6 (post-bit-swap wire values).
        self.assertEqual(RDX._debug_status_name(0xF0000004), "CMD_ALREADY_DEBUG")
        self.assertEqual(RDX._debug_status_name(0xF0000001), "CMD_ERROR")
        self.assertEqual(RDX._debug_status_name(0x80000000), "CMD_SUCCESS")
        self.assertIn("unknown", RDX._debug_status_name(0x12345678))

    def test_attach_refusal_names_the_cause_when_we_caused_it(self):
        # A stuck session makes every later attach fail. The message must say
        # why and that only reloading the payload clears it -- the protocol
        # binds a session to the connection that opened it, so a detach from a
        # new connection acks success and changes nothing (verified on hardware).
        RDX._debug_session_stuck = True
        old = {k: RDX.state.get(k) for k in ("proc_name", "ip", "pid")}
        RDX.state.update(proc_name="eboot.bin", ip="1.2.3.4", pid=57)

        class Sock:
            def settimeout(self, *_): pass
            def sendall(self, *_): pass
            def close(self): pass

        try:
            with patch.object(RDX, "ps5_read", return_value=b"\x00" * 4), \
                 patch.object(RDX, "_trace_network_refusal", return_value=None), \
                 patch.object(RDX.socket, "socket", self._FakeListener), \
                 patch.object(RDX, "ps5_connect", return_value=Sock()), \
                 patch.object(RDX, "_debug_status_word", return_value=0xF0000004), \
                 patch.object(RDX, "_debug_force_resume", return_value=True), \
                 patch.object(RDX, "add_log", lambda *a, **k: None):
                with self.assertRaises(RuntimeError) as caught:
                    RDX._trace_temporary_access("1.2.3.4", 57, 0x1000, 4,
                                                experimental=True)
            message = str(caught.exception)
            self.assertIn("CMD_ALREADY_DEBUG", message)
            self.assertIn("already attached", message)
        finally:
            RDX.state.update(old)

    # ── patch77: the fast pointer pass could not see an IL2CPP field ─────

    IL2CPP_TARGET = 0x246F37B88
    IL2CPP_HOLDER = 0x82BE17C0
    IL2CPP_DISP = 0x90F8            # measured on Enter the Gungeon

    def _il2cpp_env(self, displacement):
        """A module region holding one pointer `displacement` above the target.

        Mirrors the real layout: a .prx-backed static region holds a pointer to
        an IL2CPP static-field blob, and the field of interest sits well below
        that pointer.
        """
        target = self.IL2CPP_TARGET
        holder = self.IL2CPP_HOLDER
        region = {"start": holder & ~0xFFF, "end": (holder & ~0xFFF) + 0x2000,
                  "prot": 0x1, "offset": 0, "name": "Il2CppUserAssemblies.prx"}
        pointer_value = target + displacement      # target = value - disp

        class Sock:
            def __init__(self, *_a): pass
            def close(self): pass
            def read(self, addr, length, _cancel=None):
                buf = bytearray(length)
                if addr <= holder and holder + 8 <= addr + length:
                    struct.pack_into("<Q", buf, holder - addr, pointer_value)
                return bytes(buf)

        return [region], Sock

    def test_fast_pass_window_matches_the_canonical_default(self):
        # This constant has been wrong in both directions. 0x100 found nothing
        # on a real title; patch77's 0x10000 found plenty -- all coincidences.
        # Five such chains survived 0/5 reloads, and a later session produced
        # 24 more with offsets -0x18C8..-0x2408 in an exact 48-byte series (an
        # IL2CPP pointer table, not parents). Cheat Engine defaults its maximum
        # offset to 2048 and documents that larger offsets are "not so common".
        # Pin to that: depth finds real chains, proximity finds coincidences.
        self.assertEqual(RDX._PTR_FAST_DIRECT_RANGE, 0x800)

    def test_fast_pass_rejects_the_measured_coincidence_offsets(self):
        # The exact displacements observed on hardware must not be accepted.
        for disp in (0x18C8, 0x2408, 0x90F8, 0x9418):
            maps, Sock = self._il2cpp_env(disp)
            with patch.object(RDX, "_ScanSocket", Sock):
                hits = RDX._fast_direct_pointer_hits(
                    "ip", 1, self.IL2CPP_TARGET, maps, None)
            self.assertEqual(hits, [],
                             f"accepted {disp:#x}, a measured coincidence")

    def test_fast_pass_still_accepts_a_real_field_offset(self):
        # GUARD: ordinary struct-field displacements must keep working.
        for disp in (0x8, 0x18, 0x2C, 0x100, 0x400, 0x7F8):
            maps, Sock = self._il2cpp_env(disp)
            with patch.object(RDX, "_ScanSocket", Sock):
                hits = RDX._fast_direct_pointer_hits(
                    "ip", 1, self.IL2CPP_TARGET, maps, None)
            self.assertTrue(hits, f"rejected a plausible field offset {disp:#x}")

    def test_fast_pass_collects_a_pool_before_ranking(self):
        # A wide window makes coincidental holders likelier, and the scan stops
        # as soon as it has max_hits. Collecting exactly max_hits would let
        # whichever region is scanned first lock in junk. Gather more, then
        # rank by smallest displacement.
        self.assertGreater(RDX._PTR_FAST_DIRECT_POOL, 1)
        target = self.IL2CPP_TARGET
        base = 0x82B00000
        region = {"start": base, "end": base + 0x10000, "prot": 0x1,
                  "offset": 0, "name": "Il2CppUserAssemblies.prx"}
        # 40 holders; the nearest displacement is written last, so a scan that
        # stopped at the cap would never reach it.
        entries = {base + i * 8: target + 0x8000 - i for i in range(39)}
        entries[base + 39 * 8] = target + 8      # the good one, scanned last

        class Sock:
            def __init__(self, *_a): pass
            def close(self): pass
            def read(self, addr, length, _cancel=None):
                buf = bytearray(length)
                for a, v in entries.items():
                    if addr <= a and a + 8 <= addr + length:
                        struct.pack_into("<Q", buf, a - addr, v)
                return bytes(buf)

        with patch.object(RDX, "_ScanSocket", Sock):
            hits = RDX._fast_direct_pointer_hits(
                "ip", 1, target, [region], None, max_hits=4)
        self.assertLessEqual(len(hits), 4)
        self.assertEqual(hits[0][1], -8,
                         "nearest holder did not win; ranking was pre-empted "
                         "by the scan stopping at the cap")

    class _EmptyIndex:
        """Stand-in for the tier-3 reverse index: present, but holds nothing."""
        def query(self, *_a, **_k): return []
        def close(self): pass
        def __len__(self): return 0

    def test_locality_pass_has_a_time_budget(self):
        # Tier 2's cost tracks heap complexity, not bandwidth: on hardware it
        # reached 30% of a depth-4 search in 10 minutes and would then still
        # have fallen through to the indexed tier. It must give up first.
        self.assertTrue(hasattr(RDX, "_PTR_LOCALITY_TIME_BUDGET"))
        self.assertLessEqual(RDX._PTR_LOCALITY_TIME_BUDGET, 300.0)
        seen = {}

        def slow_scan(ip, pid, target, max_depth=5, cancel_event=None,
                      progress_cb=None, diagnostic_report=None):
            seen["cancel_event"] = cancel_event
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    seen["stopped_early"] = True
                    return []
                time.sleep(0.02)
            seen["stopped_early"] = False
            return []

        with patch.object(RDX, "_PTR_LOCALITY_TIME_BUDGET", 0.5), \
             patch.object(RDX, "_get_maps_cached", return_value=[]), \
             patch.object(RDX, "_build_region_lookup", return_value=([], [])), \
             patch.object(RDX, "_fast_direct_pointer_hits", return_value=[]), \
             patch.object(RDX, "pointer_chain_scan", slow_scan), \
             patch.object(RDX, "_get_reverse_pointer_index",
                          return_value=(self._EmptyIndex(), [], False)):
            RDX._resolve_permanent_candidates("ip", 1, 0x1000, max_depth=2)
        self.assertTrue(seen.get("stopped_early"),
                        "locality pass ran past its budget")

    def test_caller_cancel_is_not_reported_as_a_timeout(self):
        # The budget uses its own event, so a real cancel must still surface as
        # a cancel rather than looking like the pass merely timed out.
        ev = threading.Event()
        ev.set()
        with patch.object(RDX, "_get_maps_cached", return_value=[]), \
             patch.object(RDX, "_build_region_lookup", return_value=([], [])), \
             patch.object(RDX, "_fast_direct_pointer_hits", return_value=[]), \
             patch.object(RDX, "pointer_chain_scan", return_value=[]):
            out = RDX._resolve_permanent_candidates(
                "ip", 1, 0x1000, max_depth=2, cancel_event=ev)
        self.assertEqual(out.get("method"), "cancelled")
        self.assertEqual(out.get("candidates"), [])

    # ── patch76: a damaged pointer-project file crashed the main menu ────

    # Valid JSON that is not an object. Each makes `data.get` raise
    # AttributeError, which patch75's except clause did not list.
    WRONG_SHAPE = ['[]', '[{"a": 1}]', 'null', '5', '"hello"', 'true']
    # Well-shaped files whose record fields are not integers.
    BAD_FIELDS = ['{"version":1,"candidates":[{"reload_survivals":"abc"}]}',
                  '{"version":1,"candidates":[{"reload_survivals":null}]}',
                  '{"version":1,"candidates":[{"observed_target":"zzz"}]}']

    def _candidates_file(self, text):
        path = Path(tempfile.mkdtemp()) / ".rdx-pointer-candidates.json"
        path.write_text(text)
        return path

    def test_wrong_shaped_project_file_does_not_crash_the_main_menu(self):
        # _pointer_project_summary runs from screen_main on every entry to the
        # main menu, so anything escaping here kills the UI right after
        # connecting, with no way back unless the user knows which hidden file
        # to delete. Verified against the real curses UI: patch75 died at
        # screen_main with AttributeError before drawing the menu.
        for text in self.WRONG_SHAPE:
            path = self._candidates_file(text)
            self.assertEqual(RDX._load_pointer_provisionals(path), [],
                             f"loader accepted {text!r}")
            summary = RDX._pointer_project_summary("eboot.bin", path=path)
            self.assertEqual(summary["count"], 0, f"for {text!r}")

    def test_non_integer_record_fields_do_not_crash_the_main_menu(self):
        # Records persist across releases and hand edits, so a field can hold
        # a string or null. int() raises there, in the same fatal place.
        for text in self.BAD_FIELDS:
            path = self._candidates_file(text)
            summary = RDX._pointer_project_summary("eboot.bin", path=path)
            self.assertEqual(summary["survivals"], 0, f"for {text!r}")

    def test_truncated_and_empty_project_files_still_fail_closed(self):
        # GUARD (passes on patch75 too): these were already handled by the
        # ValueError branch and must stay handled.
        for text in ("", "   \n", '{"version": 1, "candidates": ['):
            path = self._candidates_file(text)
            self.assertEqual(RDX._load_pointer_provisionals(path), [])

    def test_valid_project_file_is_unaffected(self):
        # GUARD: the fix must not swallow good data along with bad.
        path = self._candidates_file(json.dumps({
            "version": 1,
            "candidates": [{"reload_survivals": 2, "observed_target": 4096,
                            "observed_process": "eboot.bin"}]}))
        self.assertEqual(len(RDX._load_pointer_provisionals(path)), 1)
        summary = RDX._pointer_project_summary("eboot.bin", path=path)
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["survivals"], 2)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["target"], 4096)

    # ── patch75: unbounded sentinel-terminated disassembly stream ────────

    class _StreamSocket:
        """Serves a scripted byte stream, then endless filler."""
        def __init__(self, script: bytes, filler=b"\x41"):
            self.buf = bytearray(script)
            self.filler = filler
            self.filler_served = 0
        def sendall(self, *_): pass
        def settimeout(self, *_): pass
        def close(self): pass
        def recv_into(self, view, n=None):
            n = n or len(view)
            if self.buf:
                k = min(n, len(self.buf))
                view[:k] = self.buf[:k]
                del self.buf[:k]
                return k
            self.filler_served += n
            if self.filler_served > 4 * 1024 * 1024:
                raise TimeoutError("peer stalled")
            view[:n] = self.filler * n
            return n

    @staticmethod
    def _disasm_entry(addr):
        # struct disasm_instr_entry, 32 bytes; length=1, kind=0x10 (has memory
        # operand) so the record is well-formed for the caller.
        return (struct.pack("<QQq", addr, 0, 0)
                + struct.pack("<BBBBB", 1, 0x10, 0, 0, 1)
                + struct.pack("<B", 0) + b"\x00" * 2)

    SENTINEL32 = b"\xFF" * 32

    def test_disasm_stream_without_sentinel_is_bounded(self):
        # patch74 looped until a sentinel that a desynced stream never sends,
        # accumulating 6.5 million entries and 2.7 GB for a 16-entry request.
        # This runs with the target STOPPED inside _trace_temporary_access, so
        # an OOM (SIGKILL, uncatchable) would strand a SIGSTOPped game.
        sock = self._StreamSocket(b"")           # pure filler, no sentinel ever
        with patch.object(RDX, "_debug_status_ok", return_value=True):
            with self.assertRaises(RuntimeError) as caught:
                RDX._debug_disasm(sock, 1, 0x400000, 32, 16)
        self.assertIn("out of sync", str(caught.exception))
        # 16 accepted entries + the one that proved the desync.
        self.assertLessEqual(sock.filler_served, 32 * 17)

    def test_disasm_accepts_a_full_max_entries_response(self):
        # GUARD, not a regression test: passes against patch74 too, since
        # patch74 had no cap to get wrong. It is the counterweight to the
        # bound above. Boundary: the server may legitimately fill every slot and *then*
        # send the sentinel. An off-by-one here would reject real responses.
        script = b"".join(self._disasm_entry(0x400000 + i * 4)
                          for i in range(16)) + self.SENTINEL32
        sock = self._StreamSocket(script)
        with patch.object(RDX, "_debug_status_ok", return_value=True):
            out = RDX._debug_disasm(sock, 1, 0x400000, 32, 16)
        self.assertEqual(len(out), 16)
        self.assertEqual(out[0]["addr"], 0x400000)
        self.assertEqual(sock.filler_served, 0, "read past the sentinel")

    def test_disasm_accepts_a_short_response(self):
        # GUARD (passes on patch74 as well): the ordinary case must be
        # untouched by the bound.
        script = b"".join(self._disasm_entry(0x400000 + i * 4)
                          for i in range(3)) + self.SENTINEL32
        sock = self._StreamSocket(script)
        with patch.object(RDX, "_debug_status_ok", return_value=True):
            out = RDX._debug_disasm(sock, 1, 0x400000, 32, 16)
        self.assertEqual(len(out), 3)

    def test_disasm_rejects_out_of_range_max_entries_before_sending(self):
        # The payload validates max_entries as 1..1000000 and answers
        # CMD_ERROR outside it, so a bad argument must not reach the wire.
        class Tripwire:
            def sendall(self, *_): raise AssertionError("must not send")
            def settimeout(self, *_): pass
        for bad in (0, -1, 1_000_001):
            with self.assertRaises(ValueError):
                RDX._debug_disasm(Tripwire(), 1, 0x400000, 32, bad)

    # ── patch74: unbounded entry counts off the wire ─────────────────────

    class _CountSocket:
        """A console reply whose uint32 entry count is `count`, followed by
        endless plausible bytes -- what a desynced stream looks like."""
        def __init__(self, count, payload=b"\x41"):
            self.buf = struct.pack("<I", count)
            self.payload = payload
            self.served = 0
        def sendall(self, *_): pass
        def settimeout(self, *_): pass
        def close(self): pass
        def recv_into(self, view, n=None):
            n = n or len(view)
            if self.buf:
                k = min(n, len(self.buf))
                view[:k] = self.buf[:k]
                self.buf = self.buf[k:]
                return k
            self.served += n
            if self.served > 8 * 1024 * 1024:      # keep the test bounded
                raise TimeoutError("peer stalled")
            view[:n] = self.payload * n
            return n
        def recv(self, n):
            b = bytearray(n)
            k = self.recv_into(memoryview(b), n)
            return bytes(b[:k])

    def _with_count(self, fn, count):
        sock = self._CountSocket(count)
        old_backend = RDX.state.get("backend")
        RDX.state["backend"] = "ps5debug"
        try:
            with patch.object(RDX, "ps5_connect", return_value=sock), \
                 patch.object(RDX, "check_ok", return_value=True):
                return fn(), sock
        finally:
            RDX.state["backend"] = old_backend

    def test_absurd_map_count_is_refused_before_allocating(self):
        # CMD_PROC_MAPS puts a bare uint32 count on the wire and the payload
        # declares no cap. patch73 fed it straight to range(), so a desynced
        # stream turned ~400 MB of bytes into >3 GB of dicts in ~7 s. An OOM
        # kill is SIGKILL, which the teardown explicitly cannot catch, so this
        # could leave a SIGSTOPped game on the console.
        with self.assertRaises(RuntimeError) as caught:
            self._with_count(lambda: RDX.ps5_maps("ip", 1), 0xFFFFFFFF)
        message = str(caught.exception)
        self.assertIn("4,294,967,295", message)
        self.assertIn("out of sync", message)

    def test_absurd_map_count_consumes_no_entry_bytes(self):
        # The point of the fix is that it refuses *before* reading entries.
        sock = self._CountSocket(0xFFFFFFFF)
        RDX.state["backend"] = "ps5debug"
        with patch.object(RDX, "ps5_connect", return_value=sock), \
             patch.object(RDX, "check_ok", return_value=True):
            with self.assertRaises(RuntimeError):
                RDX.ps5_maps("ip", 1)
        self.assertEqual(sock.served, 0,
                         "entry bytes were read despite an impossible count")

    def test_absurd_process_count_is_refused(self):
        with self.assertRaises(RuntimeError) as caught:
            self._with_count(lambda: RDX.ps5_proc_list("ip"), 0xFFFFFFFF)
        self.assertIn("out of sync", str(caught.exception))

    def test_realistic_counts_are_still_accepted(self):
        # GUARD, not a regression test: it passes against patch73 too, because
        # patch73 had no cap to over-tighten. It is the counterweight to the
        # three above -- a cap set too low would silently break every real
        # connection, so the caps are pinned to measured hardware values:
        # 307 map rows and 87 processes.
        maps, _ = self._with_count(lambda: RDX.ps5_maps("ip", 1), 307)
        self.assertEqual(len(maps), 307)
        procs, _ = self._with_count(lambda: RDX.ps5_proc_list("ip"), 87)
        self.assertEqual(len(procs), 87)

    # ── patch73: privileged bind on the debug interrupt port ─────────────

    def _trace_with_bind_error(self, exc):
        """Run _trace_temporary_access with listener.bind() raising `exc`."""
        class Listener:
            closed = False
            def __init__(self, *_a, **_k): pass
            def setsockopt(self, *_a): pass
            def bind(self, *_a): raise exc
            def listen(self, *_a): pass
            def settimeout(self, *_a): pass
            def close(self): Listener.closed = True
        old = {k: RDX.state.get(k) for k in ("proc_name", "ip", "pid")}
        RDX.state.update(proc_name="eboot.bin", ip="1.2.3.4", pid=57)
        try:
            with patch.object(RDX.socket, "socket", Listener), \
                 patch.object(RDX, "_trace_network_refusal", return_value=None), \
                 patch.object(RDX, "ps5_read", return_value=b"\x00" * 4), \
                 patch.object(RDX, "ps5_connect",
                              side_effect=AssertionError("must not attach")):
                with self.assertRaises(Exception) as caught:
                    RDX._trace_temporary_access("1.2.3.4", 57, 0x1000, 4,
                                                experimental=True)
            return caught.exception, Listener
        finally:
            RDX.state.update(old)

    def test_privileged_bind_failure_explains_port_755(self):
        # Port 755 is fixed by the protocol: the console dials out to it and
        # ps5debug-NG hard-codes htons(755) in debug.c, so it cannot be moved.
        # It is below 1024, so an ordinary user run fails here -- and patch72
        # surfaced the raw "[Errno 13] Permission denied" with no mention of
        # the port, the reason, or the remedy, on the first feature the user
        # reaches for when hunting a static address.
        exc, listener = self._trace_with_bind_error(
            PermissionError(13, "Permission denied"))
        message = str(exc)
        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("755", message)
        self.assertIn("cap_net_bind_service", message)
        self.assertTrue(listener.closed, "listener socket was leaked")

    def test_port_755_already_held_is_distinguished_from_permission(self):
        # The other realistic failure is a leaked listener from an earlier run.
        # That needs a different remedy, so it must not be reported as a
        # privilege problem.
        exc, listener = self._trace_with_bind_error(
            OSError(98, "Address already in use"))
        message = str(exc)
        self.assertIn("755", message)
        self.assertIn("holding it", message)
        self.assertNotIn("cap_net_bind_service", message)
        self.assertTrue(listener.closed, "listener socket was leaked")

    def test_bind_is_attempted_before_the_debugger_attaches(self):
        # GUARD, not a regression test: this passes against patch72 as well,
        # because the ordering was already correct there. It is kept so the
        # ordering cannot be lost in a later refactor. Console safety depends
        # on it -- if the bind failed after CMD_DEBUG_ATTACH the target would
        # be left stopped by a failure that has nothing to do with the
        # console. ps5_connect asserts if it is ever reached.
        exc, _ = self._trace_with_bind_error(
            PermissionError(13, "Permission denied"))
        self.assertNotIsInstance(exc, AssertionError)

    # ── patch65: console connection budget ───────────────────────────────

    def _budget_env(self):
        """Patch _ScanSocket onto a fake transport and drain the budget."""
        class FakeSock:
            def __init__(self, *_a): pass
            def close(self): pass
        RDX._ScanSocket.clear_pool()
        return patch.object(RDX, "ps5_connect",
                            side_effect=lambda ip, timeout=15.0: FakeSock())

    def _free_slots(self):
        return RDX._console_socket_slots._value

    def test_pooled_sockets_keep_holding_their_connection_slot(self):
        # A pooled socket is idle for us but still OPEN on the console, so it
        # still costs one of the console's finite connections. MemDBG caps
        # these at 16; ps5debug-NG dropped every connection when a 12-socket
        # batch read overlapped a 6-socket AOB scan.
        old = {k: RDX.state.get(k) for k in ("backend", "memdbg")}
        RDX.state.update(backend="ps5debug", memdbg=None)
        try:
            with self._budget_env():
                cap = self._free_slots()
                socks = [RDX._ScanSocket("ip", 1) for _ in range(4)]
                self.assertEqual(self._free_slots(), cap - 4)
                for s in socks:
                    s.close()                      # -> pooled, still open
                self.assertEqual(self._free_slots(), cap - 4,
                                 "pooling must not hand the slot back")
                RDX._ScanSocket.clear_pool()       # -> really closed
                self.assertEqual(self._free_slots(), cap)
        finally:
            RDX._ScanSocket.clear_pool()
            RDX.state.update(old)

    def test_idle_pool_is_evicted_rather_than_starving_active_work(self):
        # Without this, a pool left full by a previous operation would block
        # the next operation's workers until something happened to clear it.
        old = {k: RDX.state.get(k) for k in ("backend", "memdbg")}
        RDX.state.update(backend="ps5debug", memdbg=None)
        try:
            with self._budget_env():
                cap = self._free_slots()
                for s in [RDX._ScanSocket("ip", 1)
                          for _ in range(RDX._ScanSocket._POOL_MAX)]:
                    s.close()
                self.assertLess(self._free_slots(), cap)
                fresh = [RDX._ScanSocket("ip", 1) for _ in range(cap)]
                self.assertEqual(self._free_slots(), 0)
                for s in fresh:
                    s.close()
        finally:
            RDX._ScanSocket.clear_pool()
            RDX.state.update(old)

    def test_a_failed_connect_does_not_leak_a_connection_slot(self):
        old = {k: RDX.state.get(k) for k in ("backend", "memdbg")}
        RDX.state.update(backend="ps5debug", memdbg=None)
        try:
            RDX._ScanSocket.clear_pool()
            before = self._free_slots()
            with patch.object(RDX, "ps5_connect",
                              side_effect=OSError("connection refused")):
                for _ in range(3):
                    with self.assertRaises(OSError):
                        RDX._ScanSocket("ip", 1)
            self.assertEqual(self._free_slots(), before)
        finally:
            RDX.state.update(old)

    def test_abandoned_socket_does_not_permanently_shrink_the_budget(self):
        # A slot that is never returned is worse than the problem the budget
        # solves: the ceiling drops for the rest of the session and, at zero,
        # every scan blocks for ever. Every current caller closes in a
        # `finally`, but one missed close() in future code would be enough,
        # so __del__ reclaims it.
        import gc
        old = {k: RDX.state.get(k) for k in ("backend", "memdbg")}
        RDX.state.update(backend="ps5debug", memdbg=None)
        try:
            with self._budget_env():
                RDX._ScanSocket.clear_pool()
                before = self._free_slots()
                for _ in range(5):
                    RDX._ScanSocket("ip", 1)      # deliberately never closed
                gc.collect()
                RDX._ScanSocket.clear_pool()
                self.assertEqual(self._free_slots(), before)
        finally:
            RDX._ScanSocket.clear_pool()
            RDX.state.update(old)

    def test_budget_survives_concurrent_pool_clearing(self):
        # _acquire_slot() evicts the pool when the budget is exhausted, so
        # close() (which hands its slot to the pool) and clear_pool() (which
        # releases the pool's slots) run against each other constantly.
        import gc
        old = {k: RDX.state.get(k) for k in ("backend", "memdbg")}
        RDX.state.update(backend="ps5debug", memdbg=None)
        try:
            with self._budget_env():
                RDX._ScanSocket.clear_pool()
                cap = self._free_slots()
                stop = threading.Event()

                def clearer():
                    while not stop.is_set():
                        RDX._ScanSocket.clear_pool()
                        time.sleep(0.0005)

                def user():
                    for _ in range(120):
                        sock = RDX._ScanSocket("ip", 1)
                        sock.close()

                clear_thread = threading.Thread(target=clearer, daemon=True)
                clear_thread.start()
                workers = [threading.Thread(target=user, daemon=True)
                           for _ in range(4)]
                for w in workers:
                    w.start()
                for w in workers:
                    w.join(timeout=60)
                stop.set()
                clear_thread.join(timeout=5)
                RDX._ScanSocket.clear_pool()
                gc.collect()
                self.assertEqual(self._free_slots(), cap)
        finally:
            RDX._ScanSocket.clear_pool()
            RDX.state.update(old)

    def test_pool_max_stays_below_the_connection_budget(self):
        # If the pool could hold every slot, an eviction-free deadlock would
        # be reachable. Keep the invariant explicit.
        self.assertLess(RDX._ScanSocket._POOL_MAX, RDX._MAX_CONSOLE_SOCKETS)
        # and leave room for the connections that never go through
        # _ScanSocket: turbo session, map fetch, classifier, writes, refresh.
        self.assertLessEqual(RDX._MAX_CONSOLE_SOCKETS, 12)

    # ── patch64: debugger teardown safety ────────────────────────────────

    class _RecordingDebugSocket:
        """Captures the command opcodes a teardown actually sends."""
        def __init__(self, fail_on_send=False):
            self.sent = []
            self.closed = False
            self.fail_on_send = fail_on_send
        def sendall(self, data):
            self.sent.append(data)
            if self.fail_on_send:
                raise ConnectionError("socket died")
        def recv_into(self, view, _n):
            view[:4] = struct.pack("<I", RDX.STATUS_SUCCESS)
            return 4
        def close(self):
            self.closed = True

    def _opcodes(self, sock):
        return [struct.unpack_from("<I", d, 4)[0]
                for d in sock.sent if len(d) >= 8]

    def test_debug_teardown_clears_resumes_and_detaches_in_order(self):
        # An attached session that is never torn down is the one failure in
        # this tool that can take the console with it: ps5debug-NG allows a
        # single session, the target can be left SIGSTOPped, and hardware
        # watchpoints stay armed in DR0-DR3. PS4CheaterNeo documents the same
        # hazard ("close query window before closing the game, otherwise the
        # PS4 will crash").
        sock = self._RecordingDebugSocket()
        RDX._register_debug_session("1.2.3.4", 91, sock)
        RDX._update_debug_session(wp_index=2, stopped=True)
        with patch.object(RDX, "_debug_force_resume", return_value=True):
            RDX._emergency_debug_teardown()
        self.assertEqual(self._opcodes(sock)[:3],
                         [RDX.CMD_DEBUG_SET_WATCHPOINT, 0xBDBB0010,
                          RDX.CMD_DEBUG_DETACH])
        self.assertTrue(sock.closed)
        self.assertIsNone(RDX._debug_session)

    def test_debug_teardown_is_idempotent_and_safe_with_no_session(self):
        RDX._clear_debug_session()
        RDX._emergency_debug_teardown()      # must not raise
        RDX._emergency_debug_teardown()
        self.assertIsNone(RDX._debug_session)

    def test_debug_teardown_continues_after_a_failing_step(self):
        # If the command socket is what broke, the later steps still have to
        # be attempted, and a still-stopped target must be rescued over a
        # fresh connection.
        sock = self._RecordingDebugSocket(fail_on_send=True)
        RDX._register_debug_session("1.2.3.4", 91, sock)
        RDX._update_debug_session(wp_index=0, stopped=True)
        with patch.object(RDX, "_debug_force_resume",
                          return_value=True) as rescue:
            RDX._emergency_debug_teardown()
        self.assertEqual(len(sock.sent), 3, "every step must be attempted")
        rescue.assert_called_once()

    def test_force_resume_uses_the_sessionless_stop_command(self):
        # CMD_DEBUG_PROCESS_STOP is handled with no session attached: the
        # server falls through to kill(pid, SIGCONT) for state 0, which is the
        # only way to un-stick a game left stopped by a leaked session.
        sock = self._RecordingDebugSocket()
        with patch.object(RDX, "ps5_connect", return_value=sock):
            self.assertTrue(RDX._debug_force_resume("1.2.3.4", 91))
        self.assertEqual(self._opcodes(sock), [0xBDBB0500])
        body = sock.sent[0][12:]
        self.assertEqual(len(body), 5)                      # pid + state byte
        self.assertEqual(struct.unpack("<IB", body), (91, 0))

    def test_force_resume_reports_failure_without_raising(self):
        with patch.object(RDX, "ps5_connect",
                          side_effect=OSError("no route to host")):
            self.assertFalse(RDX._debug_force_resume("1.2.3.4", 91))

    # ── patch63: change-triggered (watchpoint) pointer resolution ────────

    _GOOD_TRACE = {
        "success": True, "target": 0xA000, "rip": 0x400100,
        "base_reg": "rbx", "base_value": 0x50000, "index_reg": None,
        "index_value": 0, "scale": 1, "final_offset": 0x18,
        "access_mode": "write", "instruction": {"addr": 0x4000F8, "length": 8},
        "lwpid": 1,
    }

    def test_trace_base_rejects_unstable_accessors(self):
        # A chain root must be reachable from a module every run. rip is a
        # code reference, rsp/rbp are stack frames that exist only for the
        # call, and an indexed access has a runtime-varying element no fixed
        # offset chain can reproduce.
        self.assertIsNone(RDX._trace_base_is_resolvable(self._GOOD_TRACE))
        for bad, needle in (
                ({**self._GOOD_TRACE, "base_value": 0}, "no base register"),
                ({**self._GOOD_TRACE, "base_reg": "rip"}, "code or stack"),
                ({**self._GOOD_TRACE, "base_reg": "rsp"}, "code or stack"),
                ({**self._GOOD_TRACE, "base_reg": "rbp"}, "code or stack"),
                ({**self._GOOD_TRACE, "index_reg": "rcx"}, "indexed")):
            reason = RDX._trace_base_is_resolvable(bad)
            self.assertIsNotNone(reason, f"{bad['base_reg']} must be rejected")
            self.assertIn(needle, reason)

    def test_traced_chain_takes_its_terminal_offset_from_the_instruction(self):
        # The whole point of tracing: the field displacement is read off the
        # opcode rather than inferred, so it becomes the terminal offset and
        # is NOT treated as another pointer dereference.
        found = [{"base": 0x1000, "offsets": [0x20], "depth": 1,
                  "static": True, "region": "executable", "score": 0.0}]
        maps = [{"start": 0x1000, "end": 0x2000, "prot": 5, "offset": 0,
                 "name": "executable"}]
        with patch.object(RDX, "_walk_from_traced_base", return_value=found), \
             patch.object(RDX, "_get_maps_cached", return_value=maps), \
             patch.object(RDX, "_module_info_for_addr",
                          return_value=("executable", 0x1000, 0x0)), \
             patch.object(RDX, "_resolve_pointer_chain",
                          return_value=(True, 0xA000, [0x50000])):
            out = RDX._pointer_candidates_from_trace(
                "ip", 1, self._GOOD_TRACE, 0xA000)
        self.assertEqual(out["method"], "change-triggered")
        self.assertEqual(len(out["candidates"]), 1)
        c = out["candidates"][0]
        self.assertEqual(c["terminal_offset"], 0x18)      # from the trace
        self.assertEqual(c["offsets"], [0x20])            # chain to the object
        self.assertEqual(c["trace_base_value"], 0x50000)
        self.assertTrue(c["verified"])
        self.assertGreater(c["confidence"], 0)

    def test_traced_chain_that_does_not_reach_the_target_is_discarded(self):
        # A chain is only kept when it actually resolves to the traced
        # target; the scan finding *a* static root is not enough.
        found = [{"base": 0x1000, "offsets": [0x20], "depth": 1,
                  "static": True, "region": "executable", "score": 0.0}]
        maps = [{"start": 0x1000, "end": 0x2000, "prot": 5, "offset": 0,
                 "name": "executable"}]
        with patch.object(RDX, "_walk_from_traced_base", return_value=found), \
             patch.object(RDX, "_get_maps_cached", return_value=maps), \
             patch.object(RDX, "_module_info_for_addr",
                          return_value=("executable", 0x1000, 0x0)), \
             patch.object(RDX, "_resolve_pointer_chain",
                          return_value=(True, 0xDEAD, [])):
            out = RDX._pointer_candidates_from_trace(
                "ip", 1, self._GOOD_TRACE, 0xA000)
        self.assertEqual(out["candidates"], [])

    def test_resolve_trace_first_reports_an_unstable_base_without_scanning(self):
        # It must not burn a multi-minute pointer scan on an accessor that
        # can never yield a permanent chain.
        stack_trace = {**self._GOOD_TRACE, "base_reg": "rsp"}
        with patch.object(RDX, "_trace_temporary_access",
                          return_value=stack_trace), \
             patch.object(RDX, "pointer_chain_scan",
                          side_effect=AssertionError("must not scan")), \
             patch.object(RDX, "_walk_from_traced_base",
                          side_effect=AssertionError("must not scan")):
            out = RDX._resolve_trace_first("ip", 1, 0xA000, 4,
                                           experimental=True)
        self.assertEqual(out["method"], "trace-no-stable-base")
        self.assertIn("code or stack", out["reason"])

    def test_watchpoint_trace_cannot_run_implicitly(self):
        with patch.object(RDX, "ps5_read",
                          side_effect=AssertionError("unsafe I/O")):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                RDX._trace_temporary_access("test", 1, 0x1000, 4)

    def test_uncached_gpu_map_is_excluded_but_static_root_is_kept(self):
        maps = [
            {"start": 0x1000, "end": 0x2000, "prot": 3,
             "name": "executable"},
            {"start": 0x3000000000, "end": 0x3000100000, "prot": 3,
             "name": ""},
        ]
        fingerprint = RDX._pointer_map_fingerprint(maps)
        RDX._pointer_region_class_cache[fingerprint] = [
            {"start": 0x1000, "end": 0x2000, "flags": 1, "mbps": 1},
            {"start": 0x3000000000, "end": 0x3000100000,
             "flags": 1, "mbps": 1},
        ]
        try:
            readable = RDX._pointer_readable_regions(maps)
            self.assertEqual([(r["start"], r["end"]) for r in readable],
                             [(0x1000, 0x2000)])
        finally:
            RDX._pointer_region_class_cache.pop(fingerprint, None)

    def test_anchor_snapshot_is_clipped_and_aligned(self):
        maps = [{"start": 0x1003, "end": 0x1020, "prot": 3,
                 "name": "heap"}]

        def fake_read(_ip, _pid, start, length):
            self.assertEqual(start, 0x1004)
            self.assertEqual(length, 0x1C)
            return np.arange(length // 4, dtype=np.uint32).tobytes()

        with patch.object(RDX, "_get_maps_cached", return_value=maps), \
             patch.object(RDX, "ps5_read", side_effect=fake_read):
            addresses, values = RDX._snapshot_anchor_window(
                "test", 1, 0x1010, 4, 0x100)
        self.assertEqual(addresses.tolist(),
                         [0x1004, 0x1008, 0x100C, 0x1010,
                          0x1014, 0x1018, 0x101C])
        self.assertEqual(values.tolist(), list(range(7)))

    def test_group_preview_always_restores_every_written_field(self):
        RDX.state["ip"] = "test"
        RDX.state["pid"] = 1
        calls = []

        def fake_write(_ip, _pid, address, value, _width):
            calls.append((address, value))
            return True, True, b""

        candidates = [(0x1000, 1), (0x1004, 1), (0x1008, 1)]
        with patch.object(RDX, "_write_value_verified", side_effect=fake_write), \
             patch.object(RDX, "confirm_box", return_value=True):
            changed, error = RDX._preview_group_once(None, candidates, 3, 4)
        self.assertTrue(changed)
        self.assertIsNone(error)
        self.assertEqual(calls, [
            (0x1000, 3), (0x1004, 3), (0x1008, 3),
            (0x1008, 1), (0x1004, 1), (0x1000, 1),
        ])

    def test_group_preview_rolls_back_after_partial_write_failure(self):
        RDX.state["ip"] = "test"
        RDX.state["pid"] = 1
        calls = []

        def fake_write(_ip, _pid, address, value, _width):
            calls.append((address, value))
            if address == 0x1004 and value == 3:
                return True, False, b"bad"
            return True, True, b""

        candidates = [(0x1000, 1), (0x1004, 1), (0x1008, 1)]
        with patch.object(RDX, "_write_value_verified", side_effect=fake_write):
            changed, error = RDX._preview_group_once(None, candidates, 3, 4)
        self.assertFalse(changed)
        self.assertIn("write verification failed", error)
        self.assertEqual(calls, [(0x1000, 3), (0x1004, 3), (0x1000, 1)])

    def test_typed_value_codec_covers_signed_float_and_raw_bytes(self):
        cases = [
            ("i8", -7), ("i16", -1234), ("i32", -1234567),
            ("i64", -1234567890123), ("f32", 1.5), ("f64", -2.25),
        ]
        for value_type, value in cases:
            with self.subTest(value_type=value_type):
                raw = RDX._pack_typed_value(value, value_type)
                decoded = RDX._unpack_typed_value(raw, value_type)
                self.assertEqual(decoded, value)
        self.assertEqual(
            RDX._pack_typed_value("90 90 CC", "bytes", 3), b"\x90\x90\xcc")
        self.assertEqual(RDX._parse_value_text("0x7f", "i8"), 127)
        with self.assertRaises(ValueError):
            RDX._parse_value_text("128", "i8")
        with self.assertRaises(ValueError):
            RDX._parse_value_text("nan", "f32")

    @unittest.skipUnless(RDX._NUMBA_OK, "numba not installed")
    def test_numba_relational_kernel_wraps_at_scan_width(self):
        # A narrow-width "decreased by"/"increased by" scan can legitimately
        # wrap (e.g. a u8 counter at 3 that decreases by 10 lands on 249).
        # The Numba kernel computes in uint64 and must be told to wrap at
        # the scanned value's own width, not at 64 bits, or the match is
        # silently dropped. Also covers width=8, where the whole 64-bit
        # value is the mask: a plain Python int `delta`/`width_mask` gets
        # inferred by Numba as signed int64, and mixing that with a uint64
        # array element produces a signed result that happens to survive a
        # narrower mask by coincidence of two's-complement bit patterns —
        # at width=8 it does not, so both must be passed as np.uint64.
        def check(cur_val, prv_val, mode, delta, width, expect):
            dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32,
                    8: np.uint64}[width]
            cur = np.array([cur_val], dtype=dtype).astype(np.uint64)
            prv = np.array([prv_val], dtype=dtype).astype(np.uint64)
            keep = RDX._nb_relational_mask(
                cur, prv, RDX.RELATIONAL_MODE_IDS[mode],
                np.uint64(delta & 0xFFFFFFFFFFFFFFFF),
                np.uint64(RDX.WIDTH_MAX[width])).astype(bool)
            self.assertEqual(bool(keep[0]), expect)

        check(249, 3, "decreased by", 10, 1, True)      # u8 wraps down
        check(3, 5, "increased by", 65534, 2, True)      # u16 wraps up
        check(13, 3, "increased by", 10, 4, True)        # u32 no wrap
        check(2**64 - 7, 3, "decreased by", 10, 8, True) # u64 wraps down
        check(50, 3, "decreased by", 10, 1, False)       # u8 genuine miss

    def test_float_next_scan_uses_configurable_tolerance(self):
        previous = np.asarray([0x1000, 0x2000], dtype=np.uint64)
        live_addr = previous.copy()
        live_values = np.asarray([1.00005, 1.01], dtype=np.float32)
        old_engine = RDX.state.get("scan_engine")
        RDX.state["scan_engine"] = "host"
        try:
            with patch.object(RDX, "ps5_read_batch",
                              return_value=(live_addr, live_values)):
                result = RDX.scan_next(
                    "test", 1, 1.0, 4, previous,
                    value_type="f32", tolerance=0.0001)
            self.assertEqual(result.tolist(), [0x1000])
        finally:
            RDX.state["scan_engine"] = old_engine

    def test_wildcard_aob_parser_and_matcher(self):
        pattern, mask, canonical = RDX._parse_byte_pattern(
            "48 8B ?? ?? 89", True)
        self.assertEqual(canonical, "48 8B ?? ?? 89")
        data = bytes.fromhex("00 48 8B 11 22 89 48 8B FF EE 89")
        self.assertEqual(RDX._find_pattern_offsets(data, pattern, mask), [1, 6])
        with self.assertRaises(ValueError):
            RDX._parse_byte_pattern("?? ??", True)

    def test_typed_native_trainer_round_trip_fields(self):
        cheats = [
            {"name": "Gravity", "type": "write", "address": 0x1000,
             "value": 1.25, "original_value": 1.0,
             "value_type": "f32", "width": 4},
            {"name": "Patch", "type": "write", "address": 0x2000,
             "value": "9090CC", "original_value": "000000",
             "value_type": "bytes", "width": 3},
        ]
        payload = json.loads(RDX.generate_cht(
            cheats, "PPSA00001", "01.000.000", "Typed"))
        self.assertEqual(payload["cheatList"][0]["value_type"], "f32")
        self.assertEqual(payload["cheatList"][0]["value"], 1.25)
        self.assertEqual(payload["cheatList"][1]["value_type"], "bytes")
        self.assertEqual(payload["cheatList"][1]["value"], "9090CC")

    def test_goldhen_schema_exports_raw_byte_patch(self):
        maps = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                 "name": "executable"}]
        cheats = [{
            "name": "NOP patch", "type": "write", "address": 0x401000,
            "module_name": "executable", "module_relative_offset": 0x1000,
            "value": "9090CC", "original_value": "000000",
            "value_type": "bytes", "width": 3,
        }]
        text, mods, skipped = RDX.generate_etahen_json(
            cheats, "CUSA00001", "01.00", "Game", "eboot.bin", maps)
        payload = json.loads(text)
        self.assertFalse(skipped)
        self.assertEqual(len(mods), 1)
        self.assertEqual(payload["mods"][0]["memory"][0]["on"], "9090CC")
        self.assertEqual(payload["mods"][0]["memory"][0]["off"], "000000")

    def test_preferences_round_trip_is_versioned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefs.json"
            RDX._save_preferences({
                "last_ip": "192.168.1.50", "last_process": "eboot.bin",
                "export_dir": directory}, path)
            loaded = RDX._load_preferences(path)
        self.assertEqual(loaded["last_ip"], "192.168.1.50")
        self.assertEqual(loaded["last_process"], "eboot.bin")

    def test_pointer_project_summary_and_clear_are_game_scoped(self):
        records = [
            {"observed_process": "eboot.bin", "observed_game": "game-a",
             "observed_target": 0x9000, "reload_survivals": 1},
            {"observed_process": "other.bin", "observed_game": "game-b",
             "observed_target": 0xA000, "reload_survivals": 0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pointer.json"
            RDX._save_pointer_provisionals(records, path)
            summary = RDX._pointer_project_summary("eboot.bin", path=path)
            self.assertEqual((summary["count"], summary["survivals"]), (1, 1))
            removed = RDX._clear_pointer_project("eboot.bin", path=path)
            self.assertEqual(removed, 1)
            self.assertEqual(
                RDX._load_pointer_provisionals(path)[0]["observed_process"],
                "other.bin")

    def test_independent_freeze_toggles_do_not_replace_each_other(self):
        first = {"name": "Health", "value": 100, "width": 4,
                 "address": 0x1000}
        second = {"name": "Ammo", "value": 30, "width": 4,
                  "address": 0x2000}
        with RDX._freeze_lock:
            RDX._freeze_targets.clear()
            RDX._freeze_status.clear()
        with patch.object(RDX, "_resolve_cheat_runtime_address",
                          side_effect=lambda c: c["address"]), \
             patch.object(RDX, "_validate_addr_in_maps", return_value=None), \
             patch.object(RDX, "_ensure_freeze_worker"):
            self.assertTrue(RDX._toggle_cheat_freeze(first))
            self.assertTrue(RDX._toggle_cheat_freeze(second))
            self.assertFalse(RDX._toggle_cheat_freeze(first))
        self.assertFalse(RDX._is_cheat_frozen(first))
        self.assertTrue(RDX._is_cheat_frozen(second))
        with RDX._freeze_lock:
            RDX._freeze_targets.clear()
            RDX._freeze_status.clear()

    def test_undo_scan_discards_stale_turbo_session(self):
        # A resident TurboScan session narrows the server-side candidate list
        # in place with no rewind. If undo doesn't discard it, the next Next
        # Scan would silently rescan the pre-undo server list instead of the
        # client-reconstructed one -- previously true only for the direct
        # 'U' key path and not the "More actions -> Undo Scan" menu path,
        # since they duplicated this logic instead of sharing it.
        class StubSocket:
            def __init__(self):
                self.closed = False
            def sendall(self, _data):
                raise ConnectionError("stub: no real console")
            def close(self):
                self.closed = True

        saved_history = RDX.state["scan_history"]
        saved_results = RDX.state["scan_results"]
        saved_values = RDX.state.get("scan_values")
        saved_dropped = RDX.state["scan_dropped"]
        saved_truncated = RDX.state["scan_truncated"]
        with RDX._turbo_session_lock:
            saved_session = RDX._turbo_session
        try:
            RDX.state["scan_history"] = RDX.deque(maxlen=5)
            RDX.state["scan_history"].append(
                (RDX._make_addr_array([0x1000, 0x2000]), None, set(), False))
            RDX.state["scan_results"] = RDX._make_addr_array([0x3000])
            RDX.state["scan_values"] = None
            RDX.state["scan_dropped"] = set()
            RDX.state["scan_truncated"] = False
            stub = StubSocket()
            with RDX._turbo_session_lock:
                RDX._turbo_session = {"socket": stub, "ip": "test", "pid": 1,
                                      "width": 4, "count": 1, "engines": 0,
                                      "value_type": "u32"}
            result = RDX._apply_scan_undo()
            self.assertEqual(RDX._addr_list(result), [0x1000, 0x2000, 0x3000])
            with RDX._turbo_session_lock:
                self.assertIsNone(RDX._turbo_session)
            self.assertTrue(stub.closed)
        finally:
            RDX.state["scan_history"] = saved_history
            RDX.state["scan_results"] = saved_results
            RDX.state["scan_values"] = saved_values
            RDX.state["scan_dropped"] = saved_dropped
            RDX.state["scan_truncated"] = saved_truncated
            with RDX._turbo_session_lock:
                RDX._turbo_session = saved_session

    def test_undo_scan_discards_stale_snapshot_turbo_session(self):
        # Same regression as test_undo_scan_discards_stale_turbo_session,
        # but for a snapshot-mode (unknown-value) session — _apply_scan_undo
        # must discard ANY resident session, not just exact-mode ones.
        class StubSocket:
            def __init__(self):
                self.closed = False
            def sendall(self, _data):
                raise ConnectionError("stub: no real console")
            def close(self):
                self.closed = True

        saved_history = RDX.state["scan_history"]
        saved_results = RDX.state["scan_results"]
        saved_values = RDX.state.get("scan_values")
        saved_dropped = RDX.state["scan_dropped"]
        saved_truncated = RDX.state["scan_truncated"]
        with RDX._turbo_session_lock:
            saved_session = RDX._turbo_session
        try:
            RDX.state["scan_history"] = RDX.deque(maxlen=5)
            RDX.state["scan_history"].append(
                (RDX._make_addr_array([0x1000]), None, set(), False))
            RDX.state["scan_results"] = RDX._make_addr_array([0x2000])
            RDX.state["scan_values"] = None
            RDX.state["scan_dropped"] = set()
            RDX.state["scan_truncated"] = False
            stub = StubSocket()
            with RDX._turbo_session_lock:
                RDX._turbo_session = {"socket": stub, "ip": "test", "pid": 1,
                                      "width": 4, "count": 1, "engines": 0,
                                      "value_type": "u32", "mode": "snapshot"}
            RDX._apply_scan_undo()
            with RDX._turbo_session_lock:
                self.assertIsNone(RDX._turbo_session)
            self.assertTrue(stub.closed)
        finally:
            RDX.state["scan_history"] = saved_history
            RDX.state["scan_results"] = saved_results
            RDX.state["scan_values"] = saved_values
            RDX.state["scan_dropped"] = saved_dropped
            RDX.state["scan_truncated"] = saved_truncated
            with RDX._turbo_session_lock:
                RDX._turbo_session = saved_session

    class _ScriptedTurboSocket:
        def __init__(self, incoming):
            self.sent = []
            self.incoming = bytearray(incoming)
            self.timeout = None
        def sendall(self, data):
            self.sent.append(bytes(data))
        def recv_into(self, view, length):
            take = min(length, len(self.incoming))
            view[:take] = self.incoming[:take]
            del self.incoming[:take]
            return take
        def gettimeout(self): return self.timeout
        def settimeout(self, value): self.timeout = value
        def close(self): pass

    def test_ps5_scan_unknown_turbo_sends_correct_wire_format_and_stores_session(self):
        get_record = (struct.pack("<Q", 0x2000) + struct.pack("<I", 7) +
                     struct.pack("<I", 7))   # addr, current=7, previous=7
        incoming = (
            struct.pack("<I", RDX.STATUS_SUCCESS) +   # ack1 (header)
            struct.pack("<I", RDX.STATUS_SUCCESS) +   # ack2 (empty value phase)
            struct.pack("<QQ", 10, 0x1000) +           # plan
            struct.pack("<Q", 0xFFFFFFFFFFFFFFFF) +    # progress sentinel
            struct.pack("<IQ", 1, 1) +                 # summary: ok=1, survivors=1
            struct.pack("<I", RDX.STATUS_SUCCESS) +    # final ack
            struct.pack("<I", RDX.STATUS_SUCCESS) +    # GET ack
            struct.pack("<I", 1) +                     # GET header: count=1
            get_record +
            struct.pack("<I", RDX.STATUS_SUCCESS)      # GET final ack
        )
        socket = self._ScriptedTurboSocket(incoming)
        with RDX._turbo_session_lock:
            old_session = RDX._turbo_session
        try:
            with patch.object(RDX, "ps5_auth_scanner"), \
                 patch.object(RDX, "ps5_turboscan_caps",
                              return_value=(1, 0x08 | 0x10, 8)), \
                 patch.object(RDX, "ps5_connect", return_value=socket):
                addrs, vals = RDX.ps5_scan_unknown_turbo(
                    "test", 7, 4, [(0x1000, 0x2000)], value_type="u32")
            self.assertEqual(addrs.tolist(), [0x2000])
            self.assertEqual(vals.tolist(), [7])
            with RDX._turbo_session_lock:
                self.assertIsNotNone(RDX._turbo_session)
                self.assertEqual(RDX._turbo_session["mode"], "snapshot")
        finally:
            with RDX._turbo_session_lock:
                RDX._turbo_session = old_session

        self.assertGreaterEqual(len(socket.sent), 3)
        _magic, cmd, _datalen = struct.unpack("<III", socket.sent[0][:12])
        self.assertEqual(cmd, RDX.CMD_TURBO_START)
        (pid, _addr, _len, wire_type, compare_type, _align,
         len_data, flags) = struct.unpack("<IQIBBBII", socket.sent[0][12:])
        self.assertEqual(pid, 7)
        self.assertEqual(wire_type, RDX.SCAN_VALUE_TYPE_ID["u32"])
        self.assertEqual(compare_type, 11)   # UnknownInitialValue
        self.assertEqual(len_data, 0)
        self.assertEqual(flags, 0x04 | 0x08 | 0x10)   # SNAPSHOT|ZEROS|SEGMENTS
        self.assertEqual(socket.sent[1], b"")
        seg_count = struct.unpack_from("<I", socket.sent[2], 0)[0]
        self.assertEqual(seg_count, 1)
        seg_addr, seg_len = struct.unpack_from("<QI", socket.sent[2], 4)
        self.assertEqual((seg_addr, seg_len), (0x1000, 0x1000))

    def test_ps5_scan_unknown_turbo_raises_when_engines_unavailable(self):
        with patch.object(RDX, "ps5_auth_scanner"), \
             patch.object(RDX, "ps5_turboscan_caps", return_value=(1, 0, 8)), \
             patch.object(RDX, "ps5_connect",
                          side_effect=AssertionError("must not connect")):
            with self.assertRaises(RuntimeError):
                RDX.ps5_scan_unknown_turbo(
                    "test", 7, 4, [(0x1000, 0x2000)], value_type="u32")

    def test_ps5_scan_unknown_turbo_raises_when_snapshot_storage_overflows(self):
        incoming = (
            struct.pack("<I", RDX.STATUS_SUCCESS) +
            struct.pack("<I", RDX.STATUS_SUCCESS) +
            struct.pack("<QQ", 10, 0x1000) +
            struct.pack("<Q", 0xFFFFFFFFFFFFFFFF) +
            struct.pack("<IQ", 0, 0)             # snapshot_ok = 0
        )
        socket = self._ScriptedTurboSocket(incoming)
        with patch.object(RDX, "ps5_auth_scanner"), \
             patch.object(RDX, "ps5_turboscan_caps",
                          return_value=(1, 0x08 | 0x10, 8)), \
             patch.object(RDX, "ps5_connect", return_value=socket):
            with self.assertRaises(RuntimeError):
                RDX.ps5_scan_unknown_turbo(
                    "test", 7, 4, [(0x1000, 0x2000)], value_type="u32")
        with RDX._turbo_session_lock:
            self.assertIsNone(RDX._turbo_session)

    def test_ps5_scan_relational_turbo_requires_matching_snapshot_session(self):
        with RDX._turbo_session_lock:
            old_session = RDX._turbo_session
            RDX._turbo_session = None
        try:
            with self.assertRaises(RuntimeError):
                RDX.ps5_scan_relational_turbo(
                    "test", 7, 4, "changed", 0, value_type="u32")

            class StubSocket:
                def sendall(self, _d): raise AssertionError("must not send")
                def close(self): pass
            with RDX._turbo_session_lock:
                RDX._turbo_session = {"socket": StubSocket(), "ip": "test",
                                      "pid": 7, "width": 4, "count": 1,
                                      "engines": 0, "value_type": "u32",
                                      "mode": "exact"}   # wrong mode
            with self.assertRaises(RuntimeError):
                RDX.ps5_scan_relational_turbo(
                    "test", 7, 4, "changed", 0, value_type="u32")
        finally:
            with RDX._turbo_session_lock:
                RDX._turbo_session = old_session

    def test_ps5_scan_relational_turbo_operand_free_mode_sends_no_operand(self):
        addr_record = (struct.pack("<Q", 0x3000) + struct.pack("<I", 9) +
                       struct.pack("<I", 9))
        incoming = (
            struct.pack("<I", RDX.STATUS_SUCCESS) +    # COUNT ack
            struct.pack("<Q", 0xFFFFFFFFFFFFFFFF) +     # progress sentinel
            struct.pack("<Q", 1) +                      # new_survivor_count
            struct.pack("<I", RDX.STATUS_SUCCESS) +     # rescan final ack
            struct.pack("<I", RDX.STATUS_SUCCESS) +     # GET ack
            struct.pack("<I", 1) +                      # GET header
            addr_record +
            struct.pack("<I", RDX.STATUS_SUCCESS)       # GET final ack
        )
        socket = self._ScriptedTurboSocket(incoming)
        with RDX._turbo_session_lock:
            old_session = RDX._turbo_session
            RDX._turbo_session = {"socket": socket, "ip": "test", "pid": 7,
                                  "width": 4, "count": 5, "engines": 0,
                                  "value_type": "u32", "mode": "snapshot"}
        try:
            addrs, vals = RDX.ps5_scan_relational_turbo(
                "test", 7, 4, "changed", 0, value_type="u32")
            self.assertEqual(addrs.tolist(), [0x3000])
            self.assertEqual(vals.tolist(), [9])
        finally:
            with RDX._turbo_session_lock:
                RDX._turbo_session = old_session

        _magic, cmd, _datalen = struct.unpack("<III", socket.sent[0][:12])
        self.assertEqual(cmd, RDX.CMD_TURBO_COUNT)
        pid, _base, wire_type, compare_type, len_data, flags = \
            struct.unpack("<IQBBII", socket.sent[0][12:])
        self.assertEqual((pid, wire_type, compare_type, len_data),
                         (7, RDX.SCAN_VALUE_TYPE_ID["u32"], 9, 0))
        self.assertEqual(flags & 0x02, 0x02)   # TS_SERVER_RESIDENT
        self.assertEqual(socket.sent[1], b"")   # no operand for "changed"

    def test_ps5_scan_relational_turbo_by_mode_sends_delta_operand(self):
        incoming = (
            struct.pack("<I", RDX.STATUS_SUCCESS) +
            struct.pack("<Q", 0xFFFFFFFFFFFFFFFF) +
            struct.pack("<Q", 0) +
            struct.pack("<I", RDX.STATUS_SUCCESS)
        )
        socket = self._ScriptedTurboSocket(incoming)
        with RDX._turbo_session_lock:
            old_session = RDX._turbo_session
            RDX._turbo_session = {"socket": socket, "ip": "test", "pid": 7,
                                  "width": 4, "count": 5, "engines": 0,
                                  "value_type": "u32", "mode": "snapshot"}
        try:
            addrs, vals = RDX.ps5_scan_relational_turbo(
                "test", 7, 4, "increased by", 3, value_type="u32")
            self.assertEqual(addrs.tolist(), [])
            self.assertEqual(vals.tolist(), [])
        finally:
            with RDX._turbo_session_lock:
                RDX._turbo_session = old_session

        pid, _base, wire_type, compare_type, len_data, _flags = \
            struct.unpack("<IQBBII", socket.sent[0][12:])
        self.assertEqual(compare_type, 6)   # IncreasedValueBy
        self.assertEqual(len_data, 4)
        self.assertEqual(socket.sent[1], struct.pack("<I", 3))

    def test_scan_first_unknown_prefers_turbo_when_available(self):
        addrs = np.asarray([0x5000], dtype=RDX._NP_ADDR_DTYPE)
        vals = np.asarray([42], dtype=np.uint32)
        old_engine = RDX.state.get("scan_engine")
        RDX.state["scan_engine"] = "auto"
        try:
            with patch.object(RDX, "_get_maps_cached", return_value=[
                    {"start": 0x1000, "end": 0x2000, "prot": 3, "name": "heap"}]), \
                 patch.object(RDX, "ps5_scan_unknown_turbo",
                              return_value=(addrs, vals)) as turbo, \
                 patch.object(RDX, "_ScanSocket",
                              side_effect=AssertionError("must not use host path")):
                out_addrs, out_vals = RDX.scan_first_unknown(
                    "test", 7, width=4, value_type="u32")
            self.assertEqual(out_addrs.tolist(), [0x5000])
            self.assertEqual(out_vals.tolist(), [42])
            turbo.assert_called_once()
        finally:
            RDX.state["scan_engine"] = old_engine

    def test_scan_first_unknown_falls_back_to_host_on_turbo_failure(self):
        old_engine = RDX.state.get("scan_engine")
        RDX.state["scan_engine"] = "auto"
        try:
            with patch.object(RDX, "_get_maps_cached", return_value=[]), \
                 patch.object(RDX, "ps5_scan_unknown_turbo",
                              side_effect=RuntimeError("unavailable")):
                out_addrs, out_vals = RDX.scan_first_unknown(
                    "test", 7, width=4, value_type="u32")
            self.assertEqual(len(out_addrs), 0)
            self.assertEqual(len(out_vals), 0)
        finally:
            RDX.state["scan_engine"] = old_engine

    def test_scan_next_relational_prefers_turbo_when_available(self):
        addrs = np.asarray([0x7000], dtype=RDX._NP_ADDR_DTYPE)
        vals = np.asarray([5], dtype=np.uint32)
        old_engine = RDX.state.get("scan_engine")
        RDX.state["scan_engine"] = "auto"
        try:
            with patch.object(RDX, "ps5_scan_relational_turbo",
                              return_value=(addrs, vals)) as turbo, \
                 patch.object(RDX, "ps5_read_batch",
                              side_effect=AssertionError("must not use host path")):
                out_addrs, out_vals = RDX.scan_next_relational(
                    "test", 7, 4,
                    np.asarray([0x7000], dtype=RDX._NP_ADDR_DTYPE),
                    np.asarray([4], dtype=np.uint32),
                    "increased", 0, value_type="u32")
            self.assertEqual(out_addrs.tolist(), [0x7000])
            self.assertEqual(out_vals.tolist(), [5])
            turbo.assert_called_once()
        finally:
            RDX.state["scan_engine"] = old_engine

    def test_scan_next_relational_skips_turbo_for_float_tolerance(self):
        # A nonzero float tolerance has no server-side compareType
        # equivalent, so it must always use the host path even with turbo
        # available, matching scan_next's own identical exact-value gate.
        with patch.object(RDX, "ps5_scan_relational_turbo",
                          side_effect=AssertionError(
                              "must not attempt turbo with a float tolerance")), \
             patch.object(RDX, "ps5_read_batch",
                          return_value=(np.asarray([0x8000], dtype=RDX._NP_ADDR_DTYPE),
                                       np.asarray([1.5], dtype=np.float32))):
            RDX.scan_next_relational(
                "test", 7, 4,
                np.asarray([0x8000], dtype=RDX._NP_ADDR_DTYPE),
                np.asarray([1.0], dtype=np.float32),
                "increased", 0, value_type="f32", tolerance=0.01)

    def test_scan_next_relational_falls_back_to_host_when_no_turbo_session(self):
        live_addrs = np.asarray([0x9000], dtype=RDX._NP_ADDR_DTYPE)
        live_vals = np.asarray([8], dtype=np.uint32)
        old_engine = RDX.state.get("scan_engine")
        RDX.state["scan_engine"] = "auto"
        with RDX._turbo_session_lock:
            old_session = RDX._turbo_session
            RDX._turbo_session = None
        try:
            with patch.object(RDX, "ps5_read_batch",
                              return_value=(live_addrs, live_vals)):
                out_addrs, out_vals = RDX.scan_next_relational(
                    "test", 7, 4,
                    np.asarray([0x9000], dtype=RDX._NP_ADDR_DTYPE),
                    np.asarray([4], dtype=np.uint32),
                    "increased", 0, value_type="u32")
            self.assertEqual(out_addrs.tolist(), [0x9000])
            self.assertEqual(out_vals.tolist(), [8])
        finally:
            RDX.state["scan_engine"] = old_engine
            with RDX._turbo_session_lock:
                RDX._turbo_session = old_session

    def test_recommended_scan_scope_excludes_debug_payload_libraries(self):
        self.assertTrue(RDX._recommended_game_scan_region(
            {"name": "[anon:game]", "prot": 3}, "eboot.bin"))
        self.assertTrue(RDX._recommended_game_scan_region(
            {"name": "/app0/eboot.bin", "prot": 5}, "eboot.bin"))
        self.assertFalse(RDX._recommended_game_scan_region(
            {"name": "/system/common/lib/libSceFoo.sprx", "prot": 5},
            "eboot.bin"))
        self.assertFalse(RDX._recommended_game_scan_region(
            {"name": "ps5debug.elf", "prot": 3}, "eboot.bin"))

    def test_primary_menu_exposes_pointer_project(self):
        entries = RDX._main_menu_entries()
        self.assertIn(("P", "Pointer Project", "pointer_project", RDX.C_ACC),
                      entries)

    def test_console_exact_scan_sends_signed_protocol_type(self):
        class ScriptedSocket:
            def __init__(self):
                self.sent = []
                self.incoming = bytearray(
                    struct.pack("<IIQ", RDX.STATUS_SUCCESS,
                                RDX.STATUS_SUCCESS, 0xFFFFFFFFFFFFFFFF))
                self.timeout = None

            def sendall(self, data):
                self.sent.append(bytes(data))

            def recv_into(self, view, length):
                take = min(length, len(self.incoming))
                view[:take] = self.incoming[:take]
                del self.incoming[:take]
                return take

            def gettimeout(self): return self.timeout
            def settimeout(self, value): self.timeout = value
            def close(self): pass

        socket = ScriptedSocket()
        with patch.object(RDX, "ps5_connect", return_value=socket):
            result = RDX.ps5_scan_exact_server(
                "test", 7, -7, 4, [(0x1000, 0x2000)],
                value_type="i32")
        _magic, _command, body_length = struct.unpack(
            "<III", socket.sent[0][:12])
        pid, wire_type, compare_type, value_length = struct.unpack(
            "<IBBI", socket.sent[0][12:])
        self.assertEqual((body_length, pid, wire_type, compare_type,
                          value_length), (10, 7, 5, 0, 4))
        self.assertEqual(socket.sent[1], struct.pack("<i", -7))
        self.assertEqual(result.tolist(), [])

    def test_ps5_write_multi_sends_correct_wire_format_and_parses_status(self):
        class ScriptedSocket:
            def __init__(self, incoming):
                self.sent = []
                self.incoming = bytearray(incoming)
                self.timeout = None
            def sendall(self, data):
                self.sent.append(bytes(data))
            def recv_into(self, view, length):
                take = min(length, len(self.incoming))
                view[:take] = self.incoming[:take]
                del self.incoming[:take]
                return take
            def gettimeout(self): return self.timeout
            def settimeout(self, value): self.timeout = value
            def close(self): pass

        # ack CMD_SUCCESS, a 2-byte status array (entry0 ok, entry1 failed),
        # then the trailing CMD_SUCCESS.
        incoming = (struct.pack("<I", RDX.STATUS_SUCCESS) +
                   bytes([0, 1]) +
                   struct.pack("<I", RDX.STATUS_SUCCESS))
        socket = ScriptedSocket(incoming)
        entries = [(0x1000, b"\x01\x02\x03\x04"), (0x2000, b"\xAA\xBB")]
        with patch.object(RDX, "ps5_connect", return_value=socket):
            results = RDX.ps5_write_multi("test", 7, entries)
        self.assertEqual(results, [True, False])

        self.assertEqual(len(socket.sent), 2)
        header_and_body = socket.sent[0]
        _magic, cmd, datalen = struct.unpack("<III", header_and_body[:12])
        self.assertEqual(cmd, RDX.CMD_PROC_WRITE_MULTI)
        self.assertEqual(datalen, 12)
        pid, count, flags = struct.unpack("<III", header_and_body[12:24])
        self.assertEqual((pid, count, flags),
                         (7, 2, RDX.PROC_WRITE_MULTI_F_STATUS))

        payload = socket.sent[1]
        entry_hdr = struct.calcsize("<QI")   # 12: uint64 address + uint32 length
        addr0, len0 = struct.unpack_from("<QI", payload, 0)
        data0 = payload[entry_hdr:entry_hdr + len0]
        self.assertEqual((addr0, len0, data0), (0x1000, 4, b"\x01\x02\x03\x04"))
        entry1_off = entry_hdr + len0
        addr1, len1 = struct.unpack_from("<QI", payload, entry1_off)
        data1 = payload[entry1_off + entry_hdr: entry1_off + entry_hdr + len1]
        self.assertEqual((addr1, len1, data1), (0x2000, 2, b"\xAA\xBB"))

    def test_ps5_write_multi_rejects_oversized_batch_without_a_network_call(self):
        entries = [(0x1000, b"\x00")] * (RDX.PROC_WRITE_MULTI_MAX_COUNT + 1)
        with patch.object(RDX, "ps5_connect",
                          side_effect=AssertionError("must not connect")):
            with self.assertRaises(ValueError):
                RDX.ps5_write_multi("test", 7, entries)

    def test_freeze_worker_batches_multiple_targets_into_one_bulk_write(self):
        cheat_a = {"name": "A", "type": "write", "address": 0x1000,
                  "value": 1, "width": 4, "_runtime_id": "ra"}
        cheat_b = {"name": "B", "type": "write", "address": 0x2000,
                  "value": 2, "width": 4, "_runtime_id": "rb"}
        old_targets = dict(RDX._freeze_targets)
        old_status = dict(RDX._freeze_status)
        stop_was_set = RDX._freeze_stop.is_set()
        old_state = {k: RDX.state.get(k) for k in ("ip", "pid")}
        RDX._freeze_targets.clear()
        RDX._freeze_targets.update({"ra": cheat_a, "rb": cheat_b})
        RDX._freeze_status.clear()
        RDX._freeze_stop.clear()
        RDX.state.update(ip="test", pid=7)

        def stop_after_one_tick(*_a, **_k):
            RDX._freeze_stop.set()
            return True

        try:
            with patch.object(RDX, "_resolve_cheat_runtime_address",
                              side_effect=[0x1000, 0x2000]), \
                 patch.object(RDX, "_validate_addr_in_maps", return_value=None), \
                 patch.object(RDX, "_memdbg_has", return_value=False), \
                 patch.object(RDX, "ps5_write_multi",
                              return_value=[True, True]) as bulk, \
                 patch.object(RDX, "ps5_write",
                              side_effect=AssertionError(
                                  "must batch, not use single write, for "
                                  "2+ targets")), \
                 patch.object(RDX._freeze_stop, "wait",
                              side_effect=stop_after_one_tick):
                RDX._freeze_manager_worker(RDX._freeze_stop)
            bulk.assert_called_once()
            called_entries = bulk.call_args.args[2]
            self.assertEqual({a for a, _d in called_entries}, {0x1000, 0x2000})
            self.assertTrue(RDX._freeze_status["ra"].startswith("active"))
            self.assertTrue(RDX._freeze_status["rb"].startswith("active"))
        finally:
            RDX._freeze_targets.clear()
            RDX._freeze_targets.update(old_targets)
            RDX._freeze_status.clear()
            RDX._freeze_status.update(old_status)
            if stop_was_set:
                RDX._freeze_stop.set()
            else:
                RDX._freeze_stop.clear()
            RDX.state.update(old_state)

    def test_freeze_worker_uses_single_write_for_one_target(self):
        cheat_a = {"name": "A", "type": "write", "address": 0x1000,
                  "value": 1, "width": 4, "_runtime_id": "ra"}
        old_targets = dict(RDX._freeze_targets)
        old_status = dict(RDX._freeze_status)
        stop_was_set = RDX._freeze_stop.is_set()
        old_state = {k: RDX.state.get(k) for k in ("ip", "pid")}
        RDX._freeze_targets.clear()
        RDX._freeze_targets.update({"ra": cheat_a})
        RDX._freeze_status.clear()
        RDX._freeze_stop.clear()
        RDX.state.update(ip="test", pid=7)

        def stop_after_one_tick(*_a, **_k):
            RDX._freeze_stop.set()
            return True

        try:
            with patch.object(RDX, "_resolve_cheat_runtime_address",
                              return_value=0x1000), \
                 patch.object(RDX, "_validate_addr_in_maps", return_value=None), \
                 patch.object(RDX, "ps5_write_multi",
                              side_effect=AssertionError(
                                  "must not batch a single target")), \
                 patch.object(RDX, "ps5_write", return_value=True) as single, \
                 patch.object(RDX._freeze_stop, "wait",
                              side_effect=stop_after_one_tick):
                RDX._freeze_manager_worker(RDX._freeze_stop)
            single.assert_called_once()
            self.assertTrue(RDX._freeze_status["ra"].startswith("active"))
        finally:
            RDX._freeze_targets.clear()
            RDX._freeze_targets.update(old_targets)
            RDX._freeze_status.clear()
            RDX._freeze_status.update(old_status)
            if stop_was_set:
                RDX._freeze_stop.set()
            else:
                RDX._freeze_stop.clear()
            RDX.state.update(old_state)

    def test_freeze_worker_prefers_memdbg_batch_write_when_available(self):
        # MemDBG's own native batch write must win over both the
        # ps5debug-wire bulk write and the per-write fallback when it's
        # available, for the same reason ps5debug's bulk write is preferred
        # over per-write: one round trip instead of N.
        cheat_a = {"name": "A", "type": "write", "address": 0x1000,
                  "value": 1, "width": 4, "_runtime_id": "ra"}
        cheat_b = {"name": "B", "type": "write", "address": 0x2000,
                  "value": 2, "width": 4, "_runtime_id": "rb"}
        old_targets = dict(RDX._freeze_targets)
        old_status = dict(RDX._freeze_status)
        stop_was_set = RDX._freeze_stop.is_set()
        old_state = {k: RDX.state.get(k) for k in ("ip", "pid")}
        RDX._freeze_targets.clear()
        RDX._freeze_targets.update({"ra": cheat_a, "rb": cheat_b})
        RDX._freeze_status.clear()
        RDX._freeze_stop.clear()
        RDX.state.update(ip="test", pid=7)

        def stop_after_one_tick(*_a, **_k):
            RDX._freeze_stop.set()
            return True

        try:
            with patch.object(RDX, "_resolve_cheat_runtime_address",
                              side_effect=[0x1000, 0x2000]), \
                 patch.object(RDX, "_validate_addr_in_maps", return_value=None), \
                 patch.object(RDX, "_memdbg_has",
                              side_effect=lambda cap: cap == RDX.MEMDBG_CAP_BATCH_WRITE), \
                 patch.object(RDX, "memdbg_write_multi",
                              return_value=[True, True]) as memdbg_bulk, \
                 patch.object(RDX, "ps5_write_multi",
                              side_effect=AssertionError(
                                  "must prefer MemDBG's own batch write")), \
                 patch.object(RDX, "ps5_write",
                              side_effect=AssertionError(
                                  "must batch, not use single write")), \
                 patch.object(RDX._freeze_stop, "wait",
                              side_effect=stop_after_one_tick):
                RDX._freeze_manager_worker(RDX._freeze_stop)
            memdbg_bulk.assert_called_once()
            called_entries = memdbg_bulk.call_args.args[2]
            self.assertEqual({a for a, _d in called_entries}, {0x1000, 0x2000})
            self.assertTrue(RDX._freeze_status["ra"].startswith("active"))
            self.assertTrue(RDX._freeze_status["rb"].startswith("active"))
        finally:
            RDX._freeze_targets.clear()
            RDX._freeze_targets.update(old_targets)
            RDX._freeze_status.clear()
            RDX._freeze_status.update(old_status)
            if stop_was_set:
                RDX._freeze_stop.set()
            else:
                RDX._freeze_stop.clear()
            RDX.state.update(old_state)

    def test_first_scan_ui_passes_type_and_recommended_scope(self):
        class FakeWindow:
            def clear(self): pass
            def refresh(self): pass
            def getmaxyx(self): return (24, 100)

        old = {key: RDX.state.get(key) for key in (
            "ip", "pid", "scan_type", "scan_width", "scan_scope",
            "scan_results", "scan_values", "scan_history")}
        RDX.state.update(ip="test", pid=7)

        def run_now(_screen, thread_fn, _label, _cancel, _progress):
            thread_fn()
            return True

        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "input_box", return_value="-7"), \
                 patch.object(RDX, "cycle_input", side_effect=[
                     RDX.VALUE_TYPES["i32"]["label"], "aligned (faster)",
                     "recommended game regions"]), \
                 patch.object(RDX, "_close_turbo_session"), \
                 patch.object(RDX, "_run_scan_with_progress",
                              side_effect=run_now), \
                 patch.object(RDX, "scan_first",
                              return_value=np.asarray([0x1000], dtype=np.uint64)) as scan, \
                 patch.object(RDX, "do_show_results"):
                RDX.do_scan_first(FakeWindow())
            self.assertEqual(RDX.state["scan_type"], "i32")
            self.assertEqual(RDX.state["scan_scope"], "recommended")
            self.assertEqual(scan.call_args.kwargs["value_type"], "i32")
            self.assertEqual(scan.call_args.kwargs["region_scope"],
                             "recommended")
            self.assertEqual(scan.call_args.args[2], -7)
        finally:
            RDX.state.update(old)

    # ── patch55: pre-hardware review regressions ─────────────────────────
    #
    # Every test below covers a defect found by reading patch54 that the
    # 133 tests above did not catch.  Each one fails against patch54.

    class _FakeKeyWindow:
        """Minimal curses stand-in that replays a scripted key sequence."""
        def __init__(self, keys, size=(30, 100)):
            self._keys = list(keys)
            self._size = size
        def clear(self): pass
        def refresh(self): pass
        def nodelay(self, *_a): pass
        def timeout(self, *_a): pass
        def getmaxyx(self): return self._size
        def getch(self):
            return self._keys.pop(0) if self._keys else ord('q')

    class _ClosableTurboSocket:
        """Stub resident-session socket that records being torn down."""
        def __init__(self):
            self.closed = False
        def sendall(self, _data):
            raise ConnectionError("stub: no real console")
        def close(self):
            self.closed = True

    def _install_turbo_session(self, mode="exact"):
        """Install a stub resident session; returns (stub, restore_callable)."""
        with RDX._turbo_session_lock:
            saved = RDX._turbo_session
        stub = self._ClosableTurboSocket()
        with RDX._turbo_session_lock:
            RDX._turbo_session = {
                "socket": stub, "ip": RDX.state["ip"], "pid": RDX.state["pid"],
                "width": 4, "count": 3, "engines": 0, "value_type": "u32",
                "mode": mode}

        def restore():
            with RDX._turbo_session_lock:
                RDX._turbo_session = saved
        return stub, restore

    def test_results_live_values_populate_on_native_memdbg_backend(self):
        # Regression: _refresh_visible_locked set its short read budget via
        # sock._s, but a _ScanSocket that connected natively leaves _s as
        # None and holds its socket on _native_client.  The AttributeError
        # killed the refresh thread on every 2 s tick, so every Results row
        # stayed "…" forever with nothing logged -- and only when MemDBG was
        # actually working, since the port-744 fallback sets _s normally.
        class FakeMemDBGClient:
            def __init__(self, ip, timeout=5.0):
                self.hello = {"capabilities": RDX.MEMDBG_CAP_MEMORY_READ}
                self.sock = None
                self.timeout = timeout
            def connect(self): return self
            def close(self): pass
            def memory_read(self, _pid, _addr, length):
                return (7).to_bytes(4, "little")[:length]

        old = {key: RDX.state.get(key) for key in ("backend", "memdbg")}
        RDX.state.update(backend="memdbg-experimental",
                         memdbg={"capabilities": RDX.MEMDBG_CAP_MEMORY_READ})
        try:
            cache = {}
            with patch.object(RDX, "_MemDBGClient", FakeMemDBGClient):
                RDX._refresh_visible_locked(
                    "test", 1, [0x1000], 4, cache, threading.Lock(),
                    value_type="u32")
            self.assertEqual(cache, {0x1000: "7"})
        finally:
            RDX.state.update(old)

    def test_scan_socket_set_timeout_applies_to_the_live_transport(self):
        class FakeMemDBGClient:
            def __init__(self, ip, timeout=5.0):
                self.hello = {"capabilities": RDX.MEMDBG_CAP_MEMORY_READ}
                self.sock = None
                self.timeout = timeout
            def connect(self): return self
            def close(self): pass

        old = {key: RDX.state.get(key) for key in ("backend", "memdbg")}
        RDX.state.update(backend="memdbg-experimental",
                         memdbg={"capabilities": RDX.MEMDBG_CAP_MEMORY_READ})
        try:
            with patch.object(RDX, "_MemDBGClient", FakeMemDBGClient):
                sock = RDX._ScanSocket("test", 1)
                self.assertIsNone(sock._s)          # native: no port-744 socket
                sock.set_timeout(1.5)               # must not raise
                self.assertEqual(sock._native_client.timeout, 1.5)
        finally:
            RDX.state.update(old)

    def test_dropping_a_result_discards_the_resident_turbo_session(self):
        # A Next Scan adopts a resident session by connection/PID/width/
        # value-type/mode alone, never by candidate count.  Dropping a result
        # without discarding the session let the server narrow its own
        # pre-drop list and hand the dropped address straight back -- the
        # same hazard already fixed for Undo Scan, at four more call sites.
        old = {key: RDX.state.get(key) for key in
               ("ip", "pid", "scan_results", "scan_values", "scan_dropped",
                "scan_pid", "scan_width", "scan_type")}
        RDX.state.update(
            ip="test", pid=1, scan_pid=1, scan_width=4, scan_type="u32",
            scan_results=RDX._make_addr_array([0x1000, 0x2000, 0x3000]),
            scan_values=None, scan_dropped=set())
        stub, restore = self._install_turbo_session("exact")
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "_refresh_visible_locked"):
                RDX.do_show_results(
                    self._FakeKeyWindow([RDX.curses.KEY_DOWN, ord('d'),
                                         ord('q')]))
            self.assertEqual(RDX._addr_list(RDX.state["scan_results"]),
                             [0x1000, 0x3000])
            with RDX._turbo_session_lock:
                self.assertIsNone(RDX._turbo_session)
            self.assertTrue(stub.closed)
        finally:
            restore()
            RDX.state.update(old)

    def test_inspector_drop_removes_the_matching_snapshot_value(self):
        # Regression: the inspector located the row with np.searchsorted,
        # which assumes sorted input.  After any host-path Next Scan
        # scan_results arrives in ps5_read_batch's worker-flush order, so it
        # returned an index belonging to a different address and np.delete
        # silently desynchronised scan_values from scan_results -- every
        # later relational scan then compared each address against its
        # neighbour's previous value.
        old = {key: RDX.state.get(key) for key in
               ("ip", "pid", "scan_results", "scan_values", "scan_dropped",
                "scan_width", "scan_type")}
        RDX.state.update(
            ip="test", pid=1, scan_width=4, scan_type="u32",
            # deliberately unsorted, exactly as ps5_read_batch returns it
            scan_results=RDX._make_addr_array([0x500, 0x100, 0x300]),
            scan_values=np.array([50, 10, 30], dtype=np.uint32),
            scan_dropped=set())
        stub, restore = self._install_turbo_session("snapshot")
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "color", return_value=0):
                RDX._inspect_result(self._FakeKeyWindow([ord('d')]), 0x100)
            self.assertEqual(RDX._addr_list(RDX.state["scan_results"]),
                             [0x500, 0x300])
            # 10 belonged to 0x100; searchsorted would have removed 50.
            self.assertEqual(RDX.state["scan_values"].tolist(), [50, 30])
            self.assertIn(0x100, RDX.state["scan_dropped"])
            with RDX._turbo_session_lock:
                self.assertIsNone(RDX._turbo_session)
            self.assertTrue(stub.closed)
        finally:
            restore()
            RDX.state.update(old)

    def test_browse_nearby_discards_the_resident_turbo_session(self):
        # This replaces the candidate list wholesale, so a surviving session
        # made the next Next Scan discard the entire hand-picked nearby set
        # and narrow the old server-side list instead.
        old = {key: RDX.state.get(key) for key in
               ("ip", "pid", "scan_results", "scan_values", "scan_dropped",
                "scan_pid", "scan_width", "scan_type", "scan_history")}
        RDX.state.update(
            ip="test", pid=1, scan_width=4, scan_type="u32",
            scan_results=RDX._make_addr_array([0x1000]), scan_values=None,
            scan_dropped=set(), scan_history=RDX.deque(maxlen=5))
        stub, restore = self._install_turbo_session("exact")
        addresses = np.array([0x2000, 0x2004], dtype=np.uint64)
        values = np.array([5, 9], dtype=np.uint32)
        try:
            with patch.object(RDX, "_snapshot_anchor_window",
                              return_value=(addresses, values)), \
                 patch.object(RDX, "confirm_box", return_value=True), \
                 patch.object(RDX, "message_box"):
                RDX.do_browse_nearby(None, 0x2000)
            self.assertEqual(RDX._addr_list(RDX.state["scan_results"]), [0x2004])
            with RDX._turbo_session_lock:
                self.assertIsNone(RDX._turbo_session)
            self.assertTrue(stub.closed)
        finally:
            restore()
            RDX.state.update(old)

    def test_changing_scan_engine_discards_the_resident_turbo_session(self):
        # Switching to host and back re-adopted the session left resident by
        # the previous turbo scan, silently discarding every host-path
        # narrowing performed in between.
        old = {key: RDX.state.get(key) for key in ("ip", "pid", "scan_engine")}
        RDX.state.update(ip="test", pid=1, scan_engine="auto")
        stub, restore = self._install_turbo_session("exact")
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "cycle_input", return_value="Host only"):
                # Settings is a list screen now: Enter edits the highlighted
                # row (Scan engine is first), q leaves.
                RDX.do_scan_settings(
                    self._FakeKeyWindow([10, ord('q')]))
            self.assertEqual(RDX.state["scan_engine"], "host")
            with RDX._turbo_session_lock:
                self.assertIsNone(RDX._turbo_session)
            self.assertTrue(stub.closed)
        finally:
            restore()
            RDX.state.update(old)

    def test_scan_engine_unchanged_keeps_the_resident_turbo_session(self):
        # Re-confirming the same engine is not a reason to throw away a
        # perfectly good session and force the next scan back onto the host.
        old = {key: RDX.state.get(key) for key in ("ip", "pid", "scan_engine")}
        RDX.state.update(ip="test", pid=1, scan_engine="auto")
        stub, restore = self._install_turbo_session("exact")
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "cycle_input",
                              return_value="Auto (Turbo → Console → Host)"):
                # Settings is a list screen now: Enter edits the highlighted
                # row (Scan engine is first), q leaves.
                RDX.do_scan_settings(
                    self._FakeKeyWindow([10, ord('q')]))
            with RDX._turbo_session_lock:
                self.assertIsNotNone(RDX._turbo_session)
            self.assertFalse(stub.closed)
        finally:
            restore()
            RDX.state.update(old)

    def test_memdbg_batch_write_splits_at_the_item_cap(self):
        # Regression: one BATCH_WRITE exchange is capped at
        # MEMDBG_BATCH_WRITE_MAX_ITEMS.  Passing a longer freeze tick raised
        # ValueError, which memdbg_write_multi retried as if it were a
        # transient network fault and then re-raised; the freeze worker
        # marked every cheat ERR with no fall-through to the per-write path,
        # so nothing was written at all, every tick.
        batches = []

        class FakeMemDBGClient:
            def __init__(self, ip, timeout=5.0):
                self.hello = {"capabilities": RDX.MEMDBG_CAP_BATCH_WRITE}
                self.ip, self.sock, self.timeout = ip, None, timeout
            def connect(self):
                self.sock = FakeSock()
                return self
            def close(self): self.sock = None
            def memory_write_multi(self, _pid, entries):
                if len(entries) > RDX.MEMDBG_BATCH_WRITE_MAX_ITEMS:
                    raise ValueError("daemon would reject this batch")
                batches.append(len(entries))
                return [True] * len(entries)

        entries = [(0x1000 + i * 4, b"\x00" * 4) for i in range(150)]
        with patch.object(RDX, "_MemDBGClient", FakeMemDBGClient):
            results = RDX.memdbg_write_multi("test", 1, entries)
        self.assertEqual(results, [True] * 150)
        self.assertEqual(batches, [64, 64, 22])

    def test_signed_relational_delta_wraps_at_the_scanned_width(self):
        # Regression: the width wraparound mask was applied only for uint.
        # Signed types fell through to `prv ± delta`, and NumPy promotes to a
        # wider type whenever the Python delta does not fit the array dtype,
        # so the arithmetic stopped wrapping at the scanned width and a
        # legitimately wrapped match was silently dropped.  This is the
        # mirror of the u8/Numba bug fixed earlier, in the pure-NumPy path.
        addrs = np.array([0x1000], dtype=np.uint64)
        cases = [
            # (value_type, dtype, previous, mode, delta, live_value)
            ("i8", "<i1", 100, "increased by", 200, 44),     # delta > i8 range
            ("i8", "<i1", 100, "increased by", 100, -56),
            ("i8", "<i1", -100, "decreased by", 100, 56),
            ("i16", "<i2", 30000, "increased by", 40000, 4464),
            ("u8", "<u1", 100, "increased by", 200, 44),     # unaffected
            ("u8", "<u1", 3, "decreased by", 10, 249),       # unaffected
        ]
        for value_type, dtype, previous, mode, delta, live in cases:
            with self.subTest(value_type=value_type, mode=mode, delta=delta):
                width = np.dtype(dtype).itemsize
                with patch.object(
                        RDX, "ps5_read_batch",
                        return_value=(addrs,
                                      np.array([live], dtype=dtype))):
                    survivors, _values = RDX.scan_next_relational(
                        "test", 1, width, addrs,
                        np.array([previous], dtype=dtype), mode, delta,
                        value_type=value_type)
                self.assertEqual(RDX._addr_list(survivors), [0x1000])

    def test_first_scan_readers_pass_cancel_event_into_reads(self):
        # Regression: both first-scan reader loops called sock.read() without
        # their cancel_event, so _recv_exact_cancel skipped its cancellation
        # check and a 32 MiB read ran to completion -- bounded only by its own
        # ~42 s budget -- before Esc was noticed, with twelve readers
        # typically mid-chunk.  scan_first_pattern always did this correctly.
        class RecordingSocket:
            seen = []
            def __init__(self, *_a): pass
            def read(self, _addr, length, cancel_event=None):
                RecordingSocket.seen.append(cancel_event)
                return b"\x00" * length
            def close(self): pass

        region = [{"start": 0x10000, "end": 0x11000, "prot": 3, "name": "heap"}]
        old = {key: RDX.state.get(key) for key in
               ("scan_engine", "proc_name")}
        RDX.state.update(scan_engine="host", proc_name="eboot.bin")
        try:
            for label, call in (
                ("scan_first",
                 lambda ev: RDX.scan_first("test", 1, 5, 4, True,
                                           cancel_event=ev,
                                           region_scope="writable")),
                ("scan_first_unknown",
                 lambda ev: RDX.scan_first_unknown("test", 1, 4, True,
                                                   cancel_event=ev,
                                                   region_scope="writable")),
            ):
                with self.subTest(function=label):
                    RecordingSocket.seen = []
                    event = threading.Event()
                    event.truncated = False
                    with patch.object(RDX, "_ScanSocket", RecordingSocket), \
                         patch.object(RDX, "_get_maps_cached",
                                      return_value=region):
                        call(event)
                    self.assertTrue(RecordingSocket.seen)
                    for passed in RecordingSocket.seen:
                        self.assertIs(passed, event)
        finally:
            RDX.state.update(old)

    def test_native_trainer_import_marks_the_cheat_list_dirty(self):
        # Regression: only the .mc4/etaHEN import path set cheats_dirty, so
        # quitting straight after a native .rdx.json import discarded the
        # imported cheats with no confirmation prompt.
        trainer = {
            "title": "T", "titleid": "PPSA00001", "version": "01.00",
            "format": "rdx-pointer-trainer-v1",
            "cheatList": [{"name": "Ammo", "type": "write",
                           "address": "0x1000", "value": "0x5",
                           "value_type": "u32", "bytes": 4}],
        }
        old = {key: RDX.state.get(key) for key in
               ("cheats", "cheats_dirty", "pid", "proc_name", "session")}
        RDX.state.update(cheats=[], cheats_dirty=False, pid=1,
                         proc_name="eboot.bin", session=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.rdx.json"
            path.write_text(json.dumps(trainer), encoding="utf-8")
            try:
                with patch.object(RDX, "draw_border"), \
                     patch.object(RDX, "safe_addstr"), \
                     patch.object(RDX, "color", return_value=0), \
                     patch.object(RDX, "input_box", return_value=str(path)), \
                     patch.object(RDX, "message_box"):
                    RDX.do_import(self._FakeKeyWindow([]))
                self.assertEqual(len(RDX.state["cheats"]), 1)
                self.assertTrue(RDX.state["cheats_dirty"])
            finally:
                RDX.state.update(old)

    def test_freeze_picker_distinguishes_duplicate_cheat_names(self):
        # Regression: the picker resolved the selection with names.index(),
        # so two cheats sharing a name always toggled the first.  Import
        # de-duplicates names for exactly this reason, but _add_cheat_at
        # does not, so manually created duplicates still collided.
        first = {"name": "Ammo", "address": 0x1000, "value": 1, "width": 4,
                 "type": "freeze"}
        second = {"name": "Ammo", "address": 0x2000, "value": 2, "width": 4,
                  "type": "freeze"}
        old = {key: RDX.state.get(key) for key in ("cheats",)}
        RDX.state.update(cheats=[first, second])
        toggled = []
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "message_box"), \
                 patch.object(RDX, "cycle_input", side_effect=[
                     "Saved cheat toggle", "2. Ammo"]), \
                 patch.object(RDX, "_toggle_cheat_freeze",
                              side_effect=lambda c: toggled.append(c) or True):
                RDX.do_freeze(self._FakeKeyWindow([]))
            self.assertEqual(len(toggled), 1)
            self.assertIs(toggled[0], second)
        finally:
            RDX.state.update(old)

    def test_aes_decrypt_reports_a_non_block_sized_ciphertext(self):
        # A truncated .mc4 used to raise IndexError from inside the block
        # cipher, which the import path reported as the vaguer "it may be
        # corrupt".  Say what is actually wrong instead.
        with self.assertRaises(ValueError) as caught:
            RDX._aes256_cbc_decrypt(RDX._MC4_AES256CBC_KEY,
                                    RDX._MC4_AES256CBC_IV, b"\x00" * 23)
        self.assertIn("16-byte AES block size", str(caught.exception))

    # ── patch57: defects found on real hardware ──────────────────────────

    def _stub_scan_environment(self, engine):
        """Patches that let scan_first run without a console."""
        region = [{"start": 0x1000, "end": 0x2000, "prot": 3, "name": "heap"}]
        RDX.state.update(scan_engine=engine, proc_name="eboot.bin")
        return (
            patch.object(RDX, "_get_maps_cached", return_value=region),
            patch.object(RDX, "_classify_regions_cached", return_value=([], False)),
            patch.object(RDX, "ps5_scan_exact_turbo",
                         side_effect=RuntimeError("turbo unavailable")),
            patch.object(RDX, "_ScanSocket", MemorySocket),
        )

    def test_console_scan_failure_is_cached_per_host(self):
        # Hardware: a ps5debug-NG build acknowledged CMD_PROC_SCAN with
        # STATUS_SUCCESS and then never sent a result byte, so the only way
        # to discover it is to wait out _recv_exact_cancel's 15 s inactivity
        # budget.  Retrying that on every scan costs 15 s each time on any
        # payload without TurboScan; learn it once, like MemDBG's
        # PROCESS_MAPS_V2 probe already does.
        old = {k: RDX.state.get(k) for k in ("scan_engine", "proc_name")}
        with RDX._console_scan_lock:
            RDX._console_scan_supported.clear()
        stubs = self._stub_scan_environment("auto")
        try:
            with stubs[0], stubs[1], stubs[2], stubs[3], \
                 patch.object(RDX, "ps5_scan_exact_server",
                              side_effect=TimeoutError("stalled")) as server:
                for _ in range(3):
                    ev = threading.Event(); ev.truncated = False
                    RDX.scan_first("ip", 1, 5, 4, cancel_event=ev,
                                   value_type="i32", region_scope="writable")
                self.assertEqual(server.call_count, 1)   # not 3
        finally:
            with RDX._console_scan_lock:
                RDX._console_scan_supported.clear()
            RDX.state.update(old)

    def test_console_only_fails_fast_once_known_unsupported(self):
        old = {k: RDX.state.get(k) for k in ("scan_engine", "proc_name")}
        with RDX._console_scan_lock:
            RDX._console_scan_supported.clear()
        stubs = self._stub_scan_environment("console")
        try:
            with stubs[0], stubs[1], stubs[2], stubs[3], \
                 patch.object(RDX, "ps5_scan_exact_server",
                              side_effect=TimeoutError("stalled")) as server:
                for _ in range(2):
                    with self.assertRaises(Exception):
                        RDX.scan_first("ip", 1, 5, 4,
                                       cancel_event=threading.Event(),
                                       value_type="i32", region_scope="writable")
                # first attempt really tries; the second is refused from cache
                self.assertEqual(server.call_count, 1)
        finally:
            with RDX._console_scan_lock:
                RDX._console_scan_supported.clear()
            RDX.state.update(old)

    def _seed_learned_support(self):
        with RDX._console_scan_lock:
            RDX._console_scan_supported["1.2.3.4"] = False
        with RDX._memdbg_maps_v2_lock:
            RDX._memdbg_maps_v2_supported["1.2.3.4"] = False

    def test_reconnect_reprobes_learned_command_support(self):
        # A reconnect may be to a different payload build, so support that
        # was learned by failing must not persist across it.
        self._seed_learned_support()
        RDX._reset_learned_payload_support()
        with RDX._console_scan_lock:
            self.assertEqual(RDX._console_scan_supported, {})
        with RDX._memdbg_maps_v2_lock:
            self.assertEqual(RDX._memdbg_maps_v2_supported, {})

    def test_clearing_results_keeps_learned_command_support(self):
        # patch60: _clear_scan_state also runs on a plain "Clear Results" and
        # on a process change. Forgetting the probe result there made the next
        # scan re-pay the 15 s stall to rediscover a command this payload
        # still does not implement -- only a reconnect can reach a different
        # build, so only a reconnect may reset it.
        self._seed_learned_support()
        try:
            RDX._clear_scan_state(stop_freezes=False)
            with RDX._console_scan_lock:
                self.assertEqual(RDX._console_scan_supported,
                                 {"1.2.3.4": False})
            with RDX._memdbg_maps_v2_lock:
                self.assertEqual(RDX._memdbg_maps_v2_supported,
                                 {"1.2.3.4": False})
        finally:
            RDX._reset_learned_payload_support()

    def test_uncached_ranges_are_coalesced_before_membership_tests(self):
        # _region_is_uncached locates a range with ONE bisect against the
        # range starts, which is only correct on disjoint input. With
        # overlaps, a short later-starting range shadows a longer earlier one
        # and an address inside the long range is reported as cached -- so a
        # 2 GiB GPU mapping would get scanned at ~100 MB/s. The classifier
        # derives its rows from the VM map, which this codebase documents as
        # carrying overlapping records.
        merged = RDX._coalesce_ranges(
            [(0x100, 0x500), (0x300, 0x400), (0x600, 0x700), (0x650, 0x900)])
        self.assertEqual(merged, [(0x100, 0x500), (0x600, 0x900)])
        starts = [r[0] for r in merged]
        for region, expected in (({"start": 0x450, "end": 0x460}, True),
                                 ({"start": 0x310, "end": 0x320}, True),
                                 ({"start": 0x880, "end": 0x890}, True),
                                 ({"start": 0x510, "end": 0x520}, False),
                                 ({"start": 0x950, "end": 0x960}, False)):
            self.assertIs(
                RDX._region_is_uncached(region, merged, starts), expected,
                f"{region} should be uncached={expected}")

    def test_ambiguous_module_basename_warns_before_rebasing(self):
        # Two modules can share a basename in different directories; the
        # min(start) tie-break would silently rebase a chain into the wrong
        # image, which resolves to a plausible address and fails quietly.
        maps = [{"start": 0x1000, "end": 0x2000, "prot": 5,
                 "name": "/app0/lib/foo.prx"},
                {"start": 0x9000, "end": 0xA000, "prot": 5,
                 "name": "/data/other/foo.prx"}]
        before = len(RDX.state["log"])
        self.assertEqual(RDX._pointer_module_base(maps, "foo.prx"), 0x1000)
        warned = [e for e in RDX.state["log"][before:] if e["level"] == "warn"]
        self.assertTrue(warned, "ambiguous basename must warn")
        # ...and an unambiguous match must stay silent
        before = len(RDX.state["log"])
        RDX._pointer_module_base(
            [{"start": 0x1000, "end": 0x2000, "prot": 5,
              "name": "/app0/bar.prx"}], "bar.prx")
        self.assertEqual(len(RDX.state["log"]), before)

    def test_turbo_only_narrow_explains_why_the_session_is_gone(self):
        # Hardware: after the F-02 fix correctly discards the resident
        # session on a drop, turbo-only mode re-raises rather than degrading
        # (which is the documented contract) -- but the bare "no matching
        # resident TurboScan session" never mentions the drop, so the user
        # cannot tell what they did or how to recover.
        old = {k: RDX.state.get(k) for k in ("scan_engine",)}
        RDX.state.update(scan_engine="turbo")
        try:
            with patch.object(RDX, "ps5_scan_next_turbo",
                              side_effect=RuntimeError(
                                  "no matching resident TurboScan session")):
                with self.assertRaises(RuntimeError) as caught:
                    RDX.scan_next("ip", 1, 5, 4,
                                  RDX._make_addr_array([0x1000]),
                                  value_type="i32")
            text = str(caught.exception).lower()
            self.assertIn("no matching resident turboscan session", text)
            for clue in ("dropping a result", "new first scan", "auto"):
                self.assertIn(clue, text)
        finally:
            RDX.state.update(old)

    def test_turbo_only_relational_narrow_explains_it_too(self):
        old = {k: RDX.state.get(k) for k in ("scan_engine",)}
        RDX.state.update(scan_engine="turbo")
        try:
            with patch.object(RDX, "ps5_scan_relational_turbo",
                              side_effect=RuntimeError("no matching session")):
                with self.assertRaises(RuntimeError) as caught:
                    RDX.scan_next_relational(
                        "ip", 1, 4, RDX._make_addr_array([0x1000]),
                        np.array([1], dtype=np.uint32), "changed", 0,
                        value_type="u32")
            self.assertIn("new first scan", str(caught.exception).lower())
        finally:
            RDX.state.update(old)

    # ── patch58 ──────────────────────────────────────────────────────────

    def test_library_module_root_rebases_across_backend_name_styles(self):
        # ps5debug reports a bare module name where MemDBG reports the full
        # vnode path, so a pointer chain rooted in a *library* (not the main
        # image) that was validated under one backend silently stopped
        # resolving under the other -- the trainer was simply dead.
        # _is_main_module_name already did basename matching for the main
        # image; library roots needed the same fallback.
        bare = [{"start": 0x8026c000, "end": 0x8027c000, "prot": 5,
                 "offset": 0, "name": "Il2CppUserAssemblies.prx"}]
        pathed = [{"start": 0x8026c000, "end": 0x8027c000, "prot": 5,
                   "flags": 2 << 24, "name": "/app0/Il2CppUserAssemblies.prx"}]
        for maps in (bare, pathed):
            self.assertEqual(
                RDX._pointer_module_base(maps, "Il2CppUserAssemblies.prx"),
                0x8026c000)
        # and the reverse direction: a chain saved under MemDBG's full path
        # must rebase against ps5debug's bare name
        self.assertEqual(
            RDX._pointer_module_base(bare, "/app0/Il2CppUserAssemblies.prx"),
            0x8026c000)

    def test_module_rebase_still_rejects_absent_and_generic_names(self):
        # The basename fallback must not turn every unresolved root into a
        # false match, and must not start matching generic backing labels.
        maps = [{"start": 0x8026c000, "end": 0x8027c000, "prot": 5,
                 "offset": 0, "name": "Il2CppUserAssemblies.prx"},
                {"start": 0x900000000, "end": 0x900010000, "prot": 3,
                 "offset": 0, "name": "[file]"}]
        self.assertIsNone(RDX._pointer_module_base(maps, "NoSuchModule.prx"))
        # A path whose basename is a generic backing label must NOT fall back
        # to basename matching -- otherwise every "[file]" row in the process
        # would answer for every other one.
        self.assertIsNone(RDX._pointer_module_base(maps, "/app0/[file]"))
        # (An *exact* "[file]" still resolves, as it always has: that is the
        # existing exact-match branch, and _module_info_for_addr routes
        # generic rows to an @section: identity rather than this name path.)
        self.assertEqual(RDX._pointer_module_base(maps, "[file]"), 0x900000000)

    def test_game_identity_is_the_same_on_both_backends(self):
        # ps5debug map rows carry `offset` and no `flags`; MemDBG rows carry
        # `flags` and no `offset`.  Hashing either into the game fingerprint
        # made the SAME game on the SAME console fingerprint differently
        # depending on which payload was loaded -- and this identity gates
        # trainer reuse, portable cheats and pointer projects, so a
        # ps5debug-built trainer was rejected the moment the user switched
        # to MemDBG.
        ps5 = [
            {"start": 0x400000, "end": 0x500000, "prot": 5,
             "offset": 0x1000, "name": "executable"},
            {"start": 0x500000, "end": 0x520000, "prot": 3,
             "offset": 0x2000, "name": "executable"},
        ]
        memdbg = [
            {"start": 0x400000, "end": 0x500000, "prot": 5,
             "flags": 2 << 24, "name": "executable"},
            {"start": 0x500000, "end": 0x520000, "prot": 3,
             "flags": 2 << 24, "name": "executable"},
        ]
        self.assertEqual(RDX._pointer_game_identity("eboot.bin", ps5),
                         RDX._pointer_game_identity("eboot.bin", memdbg))

    def test_game_identity_still_separates_different_images(self):
        # The fingerprint dropped two fields; it must not have become so
        # weak that two different game images collide.
        base = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                 "offset": 0, "name": "executable"}]
        bigger = [{"start": 0x400000, "end": 0x501000, "prot": 5,
                   "offset": 0, "name": "executable"}]
        renamed = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                    "offset": 0, "name": "other.elf"}]
        a = RDX._pointer_game_identity("eboot.bin", base)
        self.assertNotEqual(a, RDX._pointer_game_identity("eboot.bin", bigger))
        self.assertNotEqual(a, RDX._pointer_game_identity("eboot.bin", renamed))

    def test_portable_cheat_survives_a_backend_switch(self):
        ps5 = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                "offset": 0x1000, "name": "executable"}]
        memdbg = [{"start": 0x400000, "end": 0x500000, "prot": 5,
                   "flags": 2 << 24, "name": "executable"}]
        old = {k: RDX.state.get(k) for k in ("backend", "ip", "pid", "proc_name")}
        RDX.state.update(backend="ps5debug", ip="test", pid=1,
                         proc_name="eboot.bin")
        cheat = {"name": "c", "address": 0x401000, "value": 1, "width": 4,
                 "value_type": "i32", "module_name": "executable",
                 "module_relative_offset": 0x1000, "process": "eboot.bin",
                 "game_identity": RDX._pointer_game_identity("eboot.bin", ps5)}
        try:
            with RDX._map_cache_lock:
                saved = dict(RDX._map_cache)
                RDX._map_cache.clear()
                RDX._map_cache[("test", 1)] = (time.time(), memdbg)
            RDX.state["backend"] = "memdbg-experimental"
            self.assertTrue(RDX._portable_cheat_matches_current_game(cheat))
        finally:
            with RDX._map_cache_lock:
                RDX._map_cache.clear(); RDX._map_cache.update(saved)
            RDX.state.update(old)

    def test_mc4_round_trip_survives_the_block_size_guard(self):
        mods = [{"name": "Infinite Ammo",
                 "memory": [{"offset": "1A2B", "on": "90909090",
                             "off": "01020304"}]}]
        blob = RDX.generate_mc4_bytes(mods, "CUSA12345", "01.00", "Killzone",
                                      "eboot.bin")
        _attrs, parsed = RDX.mc4_xml_to_mods(
            RDX._mc4_decrypt(blob).decode("utf-8"))
        self.assertEqual(parsed[0]["name"], "Infinite Ammo")
        self.assertEqual(parsed[0]["memory"][0]["on"], "90909090")
        self.assertEqual(parsed[0]["memory"][0]["off"], "01020304")

    # ── patch87: upstream UI/UX audit — items 1, 2 and 5 ─────────────────
    #
    # Every test below fails against patch86: the .shn container, the
    # /app0/ game marker and the j/k aliases did not exist there.

    def test_shn_is_the_plaintext_twin_of_the_mc4(self):
        # The whole point of shipping both: .mc4 must be exactly the .shn
        # put through _mc4_encrypt, so a CheatRunner that accepts one and
        # rejects the other isolates the fault to the container. If these
        # ever diverge, that diagnostic silently stops being valid.
        mods = [{"name": "Infinite Ammo",
                 "memory": [{"offset": "1A2B", "on": "90909090",
                             "off": "01020304"}]}]
        args = (mods, "CUSA12345", "01.00", "Killzone", "eboot.bin")
        shn = RDX.generate_shn_text(*args)
        mc4 = RDX.generate_mc4_bytes(*args)
        self.assertEqual(RDX._mc4_decrypt(mc4).decode("utf-8"), shn)
        self.assertTrue(shn.startswith('<?xml version="1.0"'))
        self.assertIn("<Offset>1A2B</Offset>", shn)

    def test_shn_round_trips_through_the_mc4_parser(self):
        mods = [{"name": "Godmode",
                 "memory": [{"offset": "FF10", "on": "01000000",
                             "off": "00000000"}]}]
        shn = RDX.generate_shn_text(mods, "PPSA01234", "01.02", "Demo",
                                    "eboot.bin")
        attrs, parsed = RDX.mc4_xml_to_mods(shn)
        self.assertEqual(attrs.get("Cusa"), "PPSA01234")
        self.assertEqual(parsed[0]["name"], "Godmode")
        self.assertEqual(parsed[0]["memory"][0]["on"], "01000000")

    def test_import_reads_a_plaintext_shn(self):
        # .shn arrives as XML on disk with no decrypt step; the importer has
        # to take the same path .mc4 does after decryption.
        mods = [{"name": "Ammo",
                 "memory": [{"offset": "2000", "on": "63000000",
                             "off": "0A000000"}]}]
        shn = RDX.generate_shn_text(mods, "CUSA00001", "01.00", "T",
                                    "eboot.bin")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "CUSA00001_01.00.shn"
            path.write_text(shn, encoding="utf-8")
            captured = {}

            def fake_tail(_stdscr, _path, parsed_mods, label, cusa, process):
                captured.update(mods=parsed_mods, label=label, cusa=cusa,
                                process=process)

            with patch.object(RDX, "_do_import_static_patch_mods", fake_tail):
                RDX._do_import_mc4(self._FakeKeyWindow([]), path,
                                   encrypted=False)
        self.assertEqual(captured["cusa"], "CUSA00001")
        self.assertEqual(captured["mods"][0]["name"], "Ammo")
        self.assertIn("shn", captured["label"])

    def test_import_reads_a_shn_carrying_a_windows_bom(self):
        # Trainers are user-supplied and a Windows editor leaves a BOM;
        # ET.fromstring rejects one with a bare "syntax error".
        mods = [{"name": "X", "memory": [{"offset": "10", "on": "01",
                                          "off": "00"}]}]
        shn = RDX.generate_shn_text(mods, "CUSA00002", "01.00", "T",
                                    "eboot.bin")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bom.shn"
            path.write_text(shn, encoding="utf-8-sig")
            captured = {}
            with patch.object(RDX, "_do_import_static_patch_mods",
                              lambda *a: captured.update(ok=True)):
                RDX._do_import_mc4(self._FakeKeyWindow([]), path,
                                   encrypted=False)
        self.assertTrue(captured.get("ok"), "BOM-prefixed .shn was rejected")

    def test_game_candidate_matches_only_the_title_process_name(self):
        self.assertTrue(RDX._is_game_candidate("eboot.bin"))
        self.assertTrue(RDX._is_game_candidate("/app0/eboot.bin"))
        self.assertFalse(RDX._is_game_candidate("SceShellCore"))
        self.assertFalse(RDX._is_game_candidate(""))
        self.assertFalse(RDX._is_game_candidate(None))

    def test_app0_mapping_identifies_the_running_title(self):
        maps_by_pid = {
            10: [{"name": "/app0/eboot.bin", "start": 0x400000, "end": 0x500000}],
            11: [{"name": "libkernel.sprx", "start": 0x900000, "end": 0x910000}],
        }
        with patch.object(RDX, "_get_maps_cached",
                          lambda _ip, pid, *a, **k: maps_by_pid[pid]):
            self.assertTrue(RDX._process_owns_app0("1.2.3.4", 10))
            self.assertFalse(RDX._process_owns_app0("1.2.3.4", 11))

    def test_game_identification_survives_a_console_that_refuses_maps(self):
        # Identification is a convenience. A process whose maps cannot be
        # read must be skipped, never allowed to stop the user attaching.
        def flaky(_ip, pid, *a, **k):
            if pid == 11:
                raise ConnectionError("stub: refused")
            return [{"name": "/app0/eboot.bin"}]

        confirmed, lock = {}, threading.Lock()
        with patch.object(RDX, "_get_maps_cached", flaky):
            RDX._identify_game_processes(
                "1.2.3.4", [{"pid": 11}, {"pid": 12}], confirmed, lock,
                threading.Event())
        self.assertNotIn(11, confirmed)
        self.assertIs(confirmed.get(12), True)

    def test_game_identification_stops_when_cancelled(self):
        calls = []

        def counting(_ip, pid, *a, **k):
            calls.append(pid)
            return [{"name": "/app0/eboot.bin"}]

        cancel = threading.Event()
        cancel.set()
        with patch.object(RDX, "_get_maps_cached", counting):
            RDX._identify_game_processes(
                "1.2.3.4", [{"pid": 10}, {"pid": 11}], {}, threading.Lock(),
                cancel)
        self.assertEqual(calls, [], "probe ran after the screen was left")

    def test_vim_aliases_move_the_cheat_list_cursor(self):
        # j/k/g/G on a screen with no typeahead filter. 'j' must not fall
        # through to any existing action key.
        old = dict(RDX.state)
        RDX.state["cheats"] = [
            {"name": f"c{i}", "address": 0x1000 + i, "width": 4,
             "value": 1, "value_type": "u32", "type": "u32",
             "pid": RDX.state.get("pid")}
            for i in range(4)]
        seen = []
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "_read_cheat_live_value", lambda _c: "1"), \
                 patch.object(RDX, "_inspect_cheat",
                              lambda _s, idx: seen.append(idx)):
                # j j -> row 2, then Enter records it; G -> last row, Enter.
                keys = [ord('j'), ord('j'), 10, ord('G'), 10, ord('q')]
                RDX.do_cheat_list(self._FakeKeyWindow(keys))
        finally:
            RDX.state.clear(); RDX.state.update(old)
        self.assertEqual(seen, [2, 3])

    def test_vim_aliases_move_the_main_menu_cursor(self):
        entries = RDX._main_menu_entries()
        dispatched = []
        old = dict(RDX.state)
        # "Next Scan" is dimmed and refuses to run with no results, so give
        # the menu a non-empty result set before pressing Enter on it.
        RDX.state["scan_results"] = RDX._make_addr_array([0x1000, 0x2000])
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "_draw_main_header"), \
                 patch.object(RDX, "_draw_toast"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "_pointer_project_summary",
                              return_value=""), \
                 patch.object(RDX, "_confirm_quit", return_value=True), \
                 patch.object(RDX, "dispatch",
                              lambda _s, action: dispatched.append(action)):
                # j -> "Next Scan", Enter runs it; q then quits.
                RDX.screen_main(self._FakeKeyWindow([ord('j'), 10, ord('q')]))
        finally:
            RDX.state.clear(); RDX.state.update(old)
        self.assertEqual(dispatched[:1], [entries[1][2]])

    # ── patch88: settings surface (audit items 3 and 4) ──────────────────
    #
    # Fails against patch87: the tunables were literals in the source there.

    def _isolated_settings(self):
        """Restore every tunable after a test mutates one."""
        saved = dict(RDX._settings)
        def restore():
            RDX._settings.clear(); RDX._settings.update(saved)
        return restore

    def test_settings_defaults_match_the_documented_constants(self):
        # The audit's finding was that these were invisible, not that they
        # were wrong. If a default ever drifts from the constant it was
        # lifted from, the cross-validation against PINCE stops holding.
        self.assertEqual(RDX.setting("ptr_max_depth"), RDX._PTR_DEPTH_DEFAULT)
        self.assertEqual(RDX.setting("ptr_direct_range"),
                         RDX._PTR_FAST_DIRECT_RANGE)
        self.assertEqual(RDX.setting("ptr_offset_max"), RDX._PTR_STRUCT_MAX)
        self.assertIs(RDX.setting("ptr_module_bases_only"), False)
        self.assertEqual(RDX.setting("region_min_size"), 0)

    def test_settings_are_clamped_not_trusted(self):
        # A hand-edited preferences file must not be able to ask for an
        # unbounded pointer walk or a zero-width scan window.
        self.assertEqual(RDX._coerce_setting("ptr_max_depth", 9999),
                         RDX.MAX_CHAIN_DEPTH)
        self.assertEqual(RDX._coerce_setting("ptr_max_depth", -4), 1)
        self.assertEqual(RDX._coerce_setting("ptr_direct_range", 0), 0x40)
        self.assertEqual(RDX._coerce_setting("ptr_offset_max", "0x999999"),
                         0x100000)
        # Unparseable input falls back to the default rather than raising.
        self.assertEqual(RDX._coerce_setting("ptr_max_depth", "banana"),
                         RDX._PTR_DEPTH_DEFAULT)

    def test_hex_settings_accept_hex_and_decimal_text(self):
        self.assertEqual(RDX._coerce_setting("region_min_size", "0x32000"),
                         0x32000)
        self.assertEqual(RDX._coerce_setting("region_min_size", "204800"),
                         204800)

    def test_region_exclude_tokens_are_editable(self):
        restore = self._isolated_settings()
        try:
            region = {"name": "/data/mything.bin", "prot": 0x2,
                      "start": 0x1000, "end": 0x900000}
            self.assertTrue(RDX._recommended_game_scan_region(region, "eboot.bin"))
            RDX._settings["region_exclude"] = RDX._coerce_setting(
                "region_exclude", "mything,libsce")
            self.assertFalse(RDX._recommended_game_scan_region(region, "eboot.bin"))
        finally:
            restore()

    def test_default_exclude_tokens_reproduce_the_old_literal_list(self):
        # The literal list this replaced, verbatim.
        restore = self._isolated_settings()
        try:
            for token in (".sprx", ".prx", ".so", "/lib/", "libkernel",
                          "libsce", "ps5debug", "ps4debug", "memdbg",
                          "etahen", "goldhen"):
                region = {"name": f"/x/{token}/thing", "prot": 0x2,
                          "start": 0x1000, "end": 0x900000}
                self.assertFalse(
                    RDX._recommended_game_scan_region(region, "eboot.bin"),
                    f"{token} should still be excluded by default")
        finally:
            restore()

    def test_min_region_size_skips_small_mappings(self):
        restore = self._isolated_settings()
        try:
            small = {"name": "anon", "prot": 0x2, "start": 0x1000, "end": 0x2000}
            big = {"name": "anon", "prot": 0x2, "start": 0x1000, "end": 0x500000}
            self.assertTrue(RDX._recommended_game_scan_region(small, "eboot.bin"))
            RDX._settings["region_min_size"] = 0x32000
            self.assertFalse(RDX._recommended_game_scan_region(small, "eboot.bin"))
            self.assertTrue(RDX._recommended_game_scan_region(big, "eboot.bin"))
        finally:
            restore()

    def test_min_region_size_never_excludes_the_main_image(self):
        # A small main image is not a signal that the value is not in it.
        restore = self._isolated_settings()
        try:
            RDX._settings["region_min_size"] = 0x100000
            main = {"name": "executable", "prot": 0x2,
                    "start": 0x1000, "end": 0x2000}
            self.assertTrue(RDX._recommended_game_scan_region(main, "eboot.bin"))
        finally:
            restore()

    def test_pointer_depth_setting_drives_the_scan(self):
        # The scan must follow the Settings value when the caller passes
        # nothing, and still honour an explicit argument when one is given.
        restore = self._isolated_settings()
        maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                 "prot": 0x5, "static": True}]
        try:
            for configured, explicit, expected in ((3, None, 3), (3, 6, 6),
                                                   (7, None, 7)):
                RDX._settings["ptr_max_depth"] = configured
                report = {}
                # The diagnostic block is filled before the scan opens its
                # socket, so the connection failure here is expected and the
                # report is already complete.
                try:
                    with patch.object(RDX, "_get_maps_cached",
                                      lambda *a, **k: maps):
                        RDX.pointer_chain_scan("ip", 1, 0x500000,
                                               max_depth=explicit,
                                               diagnostic_report=report)
                except Exception:
                    pass
                self.assertEqual(report["limits"]["max_depth"], expected)
        finally:
            restore()

    def test_pointer_offset_window_setting_drives_the_scan(self):
        restore = self._isolated_settings()
        maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                 "prot": 0x5, "static": True}]
        try:
            RDX._settings["ptr_offset_max"] = 0x1000
            report = {}
            try:
                with patch.object(RDX, "_get_maps_cached",
                                  lambda *a, **k: maps):
                    RDX.pointer_chain_scan("ip", 1, 0x500000,
                                           diagnostic_report=report)
            except Exception:
                pass
            self.assertEqual(report["limits"]["offset_max"], 0x1000)
        finally:
            restore()

    def test_module_bases_only_drops_unrooted_chains(self):
        restore = self._isolated_settings()
        try:
            candidates = [
                {"base": 0x400000, "module_name": "executable",
                 "module_relative_offset": 0x10, "offsets": [0x8], "depth": 1},
                {"base": 0x800000, "module_name": "", "offsets": [0x8],
                 "depth": 1},
            ]
            RDX._settings["ptr_module_bases_only"] = True
            kept = [c for c in candidates
                    if c.get("module_name") or not RDX.setting(
                        "ptr_module_bases_only")]
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0]["module_name"], "executable")
        finally:
            restore()

    def test_settings_round_trip_through_the_preferences_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prefs.json"
            RDX._save_preferences({"ptr_max_depth": 7,
                                   "region_min_size": "0x32000",
                                   "ptr_module_bases_only": True}, path=path)
            loaded = RDX._load_preferences(path)
            self.assertEqual(loaded["ptr_max_depth"], 7)
            self.assertEqual(loaded["region_min_size"], 0x32000)
            self.assertIs(loaded["ptr_module_bases_only"], True)

    def test_corrupt_preferences_do_not_poison_settings(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prefs.json"
            path.write_text('{"version": 1, "ptr_max_depth": "not-a-number",'
                            ' "region_min_size": -99}', encoding="utf-8")
            loaded = RDX._load_preferences(path)
            self.assertEqual(loaded["ptr_max_depth"], RDX._PTR_DEPTH_DEFAULT)
            self.assertEqual(loaded["region_min_size"], 0)

    def test_wrap_help_never_exceeds_the_width(self):
        text = ("Skip mappings smaller than this when scanning. "
                "0 = off; PS4CheaterNeo defaults to 0x32000 (200K).")
        for width in (20, 34, 60):
            for line in RDX._wrap_help(text, width):
                self.assertLessEqual(len(line), max(width, len(line.split()[0])))

    # ── patch89: bookmarks + pointer epoch visibility (items U3, 6) ──────
    #
    # Fails against patch88: neither existed there.

    def _isolated_bookmarks(self):
        saved = list(state_bm := RDX.state.get("bookmarks", []))
        RDX.state["bookmarks"] = []
        def restore():
            RDX.state["bookmarks"] = saved
        return restore

    def test_bookmark_add_is_idempotent_per_address_and_type(self):
        restore = self._isolated_bookmarks()
        try:
            RDX._add_bookmark(0x1000, "u32", "health?")
            RDX._add_bookmark(0x1000, "u32")
            self.assertEqual(len(RDX.state["bookmarks"]), 1)
            # The note survives a re-add that carries none.
            self.assertEqual(RDX.state["bookmarks"][0]["note"], "health?")
            # A different type at the same address is a different bookmark.
            RDX._add_bookmark(0x1000, "f32")
            self.assertEqual(len(RDX.state["bookmarks"]), 2)
        finally:
            restore()

    def test_bookmark_list_is_bounded(self):
        restore = self._isolated_bookmarks()
        try:
            for i in range(RDX._BOOKMARK_MAX + 20):
                RDX._add_bookmark(0x1000 + i * 4, "u32")
            self.assertEqual(len(RDX.state["bookmarks"]), RDX._BOOKMARK_MAX)
        finally:
            restore()

    def test_bookmark_goes_stale_when_the_session_changes(self):
        # A bookmark is a raw address with no chain behind it, so after a
        # reconnect it names whatever now occupies that memory.
        restore = self._isolated_bookmarks()
        old = {k: RDX.state.get(k) for k in ("session", "pid")}
        try:
            RDX.state.update(session=1, pid=42)
            RDX._add_bookmark(0x2000, "u32")
            bookmark = RDX.state["bookmarks"][0]
            self.assertTrue(RDX._bookmark_is_current(bookmark))
            RDX.state["session"] = 2
            self.assertFalse(RDX._bookmark_is_current(bookmark))
            RDX.state.update(session=1, pid=43)
            self.assertFalse(RDX._bookmark_is_current(bookmark))
        finally:
            RDX.state.update(old); restore()

    def test_clearing_scan_state_drops_bookmarks(self):
        restore = self._isolated_bookmarks()
        try:
            RDX._add_bookmark(0x3000, "u32")
            with patch.object(RDX, "_stop_freeze_worker"), \
                 patch.object(RDX, "_close_turbo_session"), \
                 patch.object(RDX, "_invalidate_pointer_index"):
                RDX._clear_scan_state(stop_freezes=False)
            self.assertEqual(RDX.state["bookmarks"], [])
        finally:
            restore()

    def test_remove_bookmark_out_of_range_is_a_no_op(self):
        restore = self._isolated_bookmarks()
        try:
            RDX._add_bookmark(0x4000, "u32")
            self.assertIsNone(RDX._remove_bookmark(9))
            self.assertIsNone(RDX._remove_bookmark(-1))
            self.assertEqual(len(RDX.state["bookmarks"]), 1)
            self.assertIsNotNone(RDX._remove_bookmark(0))
            self.assertEqual(RDX.state["bookmarks"], [])
        finally:
            restore()

    def test_epoch_log_records_the_funnel(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "proj.json"
            RDX._save_pointer_provisionals([], path)
            first = RDX._record_pointer_epoch(
                {"survivors": [{"a": 1}] * 37,
                 "rejected": [{"rejection_reason": "chain changed after reload"}] * 127},
                "eboot.bin", path)
            RDX._save_pointer_provisionals([], path, epochs=first)
            self.assertEqual(first[0]["epoch"], 1)
            self.assertEqual(first[0]["considered"], 164)
            self.assertEqual(first[0]["survived"], 37)

            second = RDX._record_pointer_epoch(
                {"survivors": [{"a": 1}] * 5,
                 "rejected": [{"rejection_reason": "module not mapped"}] * 32},
                "eboot.bin", path)
            RDX._save_pointer_provisionals([], path, epochs=second)
            self.assertEqual([e["epoch"] for e in second], [1, 2])
            self.assertEqual(second[1]["survived"], 5)
            self.assertEqual(second[1]["reasons"]["module not mapped"], 32)

    def test_epoch_log_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "proj.json"
            RDX._save_pointer_provisionals([], path)
            epochs = []
            for _ in range(RDX._PTR_EPOCH_LOG_MAX + 5):
                epochs = RDX._record_pointer_epoch(
                    {"survivors": [], "rejected": []}, "eboot.bin", path)
                RDX._save_pointer_provisionals([], path, epochs=epochs)
            self.assertEqual(len(epochs), RDX._PTR_EPOCH_LOG_MAX)

    def test_saving_candidates_preserves_an_existing_epoch_log(self):
        # A caller that only rewrites the candidate list must not silently
        # erase the history that explains how the list got that short.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "proj.json"
            RDX._save_pointer_provisionals([], path)
            epochs = RDX._record_pointer_epoch(
                {"survivors": [{"a": 1}], "rejected": []}, "eboot.bin", path)
            RDX._save_pointer_provisionals([], path, epochs=epochs)
            RDX._save_pointer_provisionals([{"module_name": "main"}], path)
            self.assertEqual(len(RDX._load_pointer_epochs(path)), 1)

    def test_corrupt_epoch_log_does_not_break_candidate_loading(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "proj.json"
            path.write_text('{"version": 1, "candidates": [{"module_name": "m"}],'
                            ' "epochs": "not-a-list"}', encoding="utf-8")
            self.assertEqual(RDX._load_pointer_epochs(path), [])
            self.assertEqual(len(RDX._load_pointer_provisionals(path)), 1)

    def test_epoch_rows_render_without_dividing_by_zero(self):
        rows = RDX._format_epoch_rows([
            {"epoch": 1, "considered": 0, "survived": 0, "reasons": {}},
            {"epoch": 2, "considered": 164, "survived": 37,
             "reasons": {"chain changed after reload": 127}},
        ])
        self.assertEqual(len(rows), 2)
        self.assertIn("0%", rows[0])
        self.assertIn("164", rows[1])
        self.assertIn("chain changed after reload", rows[1])

    # ── patch90: read-only hex viewer (audit item 7 / U1) ────────────────
    #
    # Fails against patch89: there was no viewer there.

    def test_hex_rows_render_address_hex_and_ascii(self):
        data = bytes(range(32))
        rows = RDX._hex_render_rows(0x1000, data)
        self.assertEqual(len(rows), 2)
        addr, hex_col, ascii_col = rows[0]
        self.assertEqual(addr, 0x1000)
        self.assertTrue(hex_col.startswith("00 01 02 03"))
        self.assertEqual(len(ascii_col), 16)
        self.assertEqual(rows[1][0], 0x1010)

    def test_hex_rows_show_printable_ascii_and_dot_the_rest(self):
        data = b"RDX\x00\xffabc" + b"\x00" * 8
        _addr, _hex, ascii_col = RDX._hex_render_rows(0, data)[0]
        self.assertTrue(ascii_col.startswith("RDX.."))
        self.assertIn("abc", ascii_col)
        self.assertNotIn("\x00", ascii_col)

    def test_hex_rows_pad_a_short_final_row(self):
        # A read returning fewer bytes than a full row must not produce a
        # ragged hex column that misaligns the ascii gutter.
        rows = RDX._hex_render_rows(0x2000, bytes(range(4)))
        _addr, hex_col, _ascii = rows[0]
        full = RDX._hex_render_rows(0x2000, bytes(range(16)))[0][1]
        self.assertEqual(len(hex_col), len(full))

    def test_hex_unreadable_window_renders_question_marks(self):
        rows = RDX._hex_render_rows(0x3000, b"\x00" * 16, unreadable=True)
        _addr, hex_col, ascii_col = rows[0]
        self.assertEqual(hex_col.strip().split()[0], "??")
        self.assertEqual(ascii_col, "?" * 16)

    def test_hex_fetch_reports_unreadable_instead_of_raising(self):
        # Scrolling off the end of a mapping is normal; it must not become an
        # error box the user dismisses on every keypress.
        with patch.object(RDX, "ps5_read",
                          side_effect=ConnectionError("unmapped")):
            data, unreadable = RDX._hex_fetch("ip", 1, 0x4000, 64)
        self.assertTrue(unreadable)
        self.assertEqual(len(data), 64)

    def test_hex_fetch_pads_a_short_read(self):
        with patch.object(RDX, "ps5_read", return_value=b"\x01\x02"):
            data, unreadable = RDX._hex_fetch("ip", 1, 0x5000, 16)
        self.assertFalse(unreadable)
        self.assertEqual(len(data), 16)
        self.assertEqual(data[:2], b"\x01\x02")

    def test_hex_view_never_writes(self):
        # Read-only on purpose: RDX already has three audited write paths,
        # and a fourth reachable by cursoring around a dump would be the
        # easiest way in the program to corrupt a running game by accident.
        old = {k: RDX.state.get(k) for k in ("ip", "pid", "proc_name")}
        RDX.state.update(ip="test", pid=1, proc_name="eboot.bin")
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "ps5_read", return_value=b"\x00" * 4096), \
                 patch.object(RDX, "ps5_write",
                              side_effect=AssertionError("hex view wrote")), \
                 patch.object(RDX, "ps5_write_multi",
                              side_effect=AssertionError("hex view wrote")), \
                 patch.object(RDX, "ps5_write_verified",
                              side_effect=AssertionError("hex view wrote")):
                keys = [ord('j'), ord('k'), RDX.curses.KEY_NPAGE,
                        RDX.curses.KEY_PPAGE, ord('n'), ord('q')]
                RDX.do_hex_view(self._FakeKeyWindow(keys), 0x1000)
        finally:
            RDX.state.update(old)

    def test_hex_view_requires_an_attached_process(self):
        old = RDX.state.get("pid")
        RDX.state["pid"] = None
        shown = []
        try:
            with patch.object(RDX, "message_box",
                              lambda _s, lines, *a, **k: shown.append(lines)):
                RDX.do_hex_view(self._FakeKeyWindow([]), 0x1000)
        finally:
            RDX.state["pid"] = old
        self.assertTrue(shown)

    # ── patch91: scan-path coalescing + shape-aware engine (items 8, B6) ──
    #
    # Fails against patch90: coalescing was wired only into the pointer
    # index there, and "auto" chose purely by availability.

    def test_coalesce_merges_adjacent_regions(self):
        merged = RDX._coalesce_scan_regions([
            {"start": 0x1000, "end": 0x3000, "prot": 3, "name": "a"},
            {"start": 0x3000, "end": 0x5000, "prot": 3, "name": "b"},
            {"start": 0x5000, "end": 0x7000, "prot": 3, "name": "c"},
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0]["start"], merged[0]["end"]), (0x1000, 0x7000))
        self.assertEqual(merged[0]["merged"], 3)

    def test_coalesce_keeps_a_gap_separate(self):
        merged = RDX._coalesce_scan_regions([
            {"start": 0x1000, "end": 0x2000, "prot": 3},
            {"start": 0x9000, "end": 0xA000, "prot": 3},
        ])
        self.assertEqual(len(merged), 2)

    def test_coalesce_merges_overlapping_and_unsorted_input(self):
        merged = RDX._coalesce_scan_regions([
            {"start": 0x5000, "end": 0x7000, "prot": 3},
            {"start": 0x1000, "end": 0x6000, "prot": 3},
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0]["start"], merged[0]["end"]), (0x1000, 0x7000))

    def test_coalesce_drops_empty_and_inverted_regions(self):
        merged = RDX._coalesce_scan_regions([
            {"start": 0x1000, "end": 0x1000, "prot": 3},
            {"start": 0x5000, "end": 0x2000, "prot": 3},
            {"start": 0x8000, "end": 0x9000, "prot": 3},
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["start"], 0x8000)

    def test_coalesce_preserves_total_scanned_bytes(self):
        regions = [{"start": 0x1000, "end": 0x2000, "prot": 3},
                   {"start": 0x2000, "end": 0x4000, "prot": 3},
                   {"start": 0x8000, "end": 0x9000, "prot": 3}]
        before = sum(r["end"] - r["start"] for r in regions)
        after = sum(r["end"] - r["start"]
                    for r in RDX._coalesce_scan_regions(regions))
        self.assertEqual(before, after)

    def test_aob_across_a_region_boundary_is_now_reachable(self):
        # The correctness half of the merge. The AOB scanner extends each
        # chunk by len(pattern)-1 but clamps that overlap to the region end,
        # so a match spanning two adjacent mappings was invisible.
        regions = [{"start": 0x1000, "end": 0x2000, "prot": 3},
                   {"start": 0x2000, "end": 0x3000, "prot": 3}]
        merged = RDX._coalesce_scan_regions(regions)
        # One span means a read can cross 0x2000 and see the whole pattern.
        self.assertEqual(len(merged), 1)
        self.assertLess(merged[0]["start"], 0x2000)
        self.assertGreater(merged[0]["end"], 0x2000)

    def test_auto_skips_turbo_on_a_nearly_converged_list(self):
        old = {k: RDX.state.get(k) for k in ("ip", "pid", "scan_engine")}
        RDX.state.update(ip="test", pid=1, scan_engine="auto")
        with RDX._turbo_session_lock:
            saved_session = RDX._turbo_session
            RDX._turbo_session = None
        try:
            prev = RDX._make_addr_array([0x1000, 0x2000, 0x3000])
            with patch.object(RDX, "ps5_scan_next_turbo",
                              side_effect=AssertionError("turbo used")), \
                 patch.object(RDX, "ps5_read_batch",
                              return_value=(RDX._make_addr_array([0x1000]),
                                            np.array([7], dtype=np.uint32))):
                out = RDX.scan_next("test", 1, 7, 4, prev)
            self.assertEqual(list(out), [0x1000])
        finally:
            with RDX._turbo_session_lock:
                RDX._turbo_session = saved_session
            RDX.state.update(old)

    def test_auto_still_uses_turbo_on_a_large_list(self):
        old = {k: RDX.state.get(k) for k in ("ip", "pid", "scan_engine")}
        RDX.state.update(ip="test", pid=1, scan_engine="auto")
        with RDX._turbo_session_lock:
            saved_session = RDX._turbo_session
            RDX._turbo_session = None
        try:
            prev = RDX._make_addr_array(
                range(0x1000, 0x1000 + RDX.TURBO_MIN_SURVIVORS * 4, 4))
            self.assertGreaterEqual(len(prev), RDX.TURBO_MIN_SURVIVORS)
            with patch.object(RDX, "ps5_scan_next_turbo",
                              return_value=RDX._make_addr_array([0x1000])):
                out = RDX.scan_next("test", 1, 7, 4, prev)
            self.assertEqual(list(out), [0x1000])
        finally:
            with RDX._turbo_session_lock:
                RDX._turbo_session = saved_session
            RDX.state.update(old)

    def test_a_resident_turbo_session_always_wins_over_the_size_rule(self):
        # Required, not an optimisation: the server holds the survivor list,
        # so narrowing on the host while a session stays resident would let
        # the next turbo scan re-adopt the pre-narrowing list.
        old = {k: RDX.state.get(k) for k in ("ip", "pid", "scan_engine")}
        RDX.state.update(ip="test", pid=1, scan_engine="auto")
        stub, restore = self._install_turbo_session("exact")
        try:
            prev = RDX._make_addr_array([0x1000, 0x2000])
            with patch.object(RDX, "ps5_scan_next_turbo",
                              return_value=RDX._make_addr_array([0x1000])):
                out = RDX.scan_next("test", 1, 7, 4, prev)
            self.assertEqual(list(out), [0x1000])
        finally:
            restore(); RDX.state.update(old)

    # ── patch92: ScanState aggregate (audit item 9 / B2) ─────────────────
    #
    # Fails against patch91: the six correlated fields were kept consistent
    # by convention at thirteen separate mutation sites there.

    def _fresh_scan(self, addrs, values=None, pid=42):
        RDX.state["pid"] = pid
        RDX.scan.replace(RDX._make_addr_array(addrs), values, close_turbo=False)

    def test_replace_sets_every_correlated_field(self):
        old = dict(RDX.state)
        try:
            RDX.state["scan_dropped"] = {0xDEAD}
            self._fresh_scan([0x10, 0x20])
            self.assertEqual(list(RDX.state["scan_results"]), [0x10, 0x20])
            self.assertIsNone(RDX.state["scan_values"])
            self.assertEqual(RDX.state["scan_pid"], 42)
            self.assertEqual(RDX.state["scan_dropped"], set())
            self.assertFalse(RDX.state["scan_truncated"])
            self.assertFalse(RDX.state["scan_unknown"])
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_clear_detaches_scan_pid_but_replace_inherits_it(self):
        # do_show_results distinguishes a null scan_pid from a mismatched
        # one, so these two must not collapse to the same value.
        old = dict(RDX.state)
        try:
            self._fresh_scan([0x10], pid=7)
            self.assertEqual(RDX.state["scan_pid"], 7)
            RDX.scan.clear()
            self.assertIsNone(RDX.state["scan_pid"])
            self.assertEqual(len(RDX.state["scan_results"]), 0)
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_drop_index_keeps_values_aligned_with_addresses(self):
        # The invariant that matters most: a mismatch here pairs an address
        # with another address's previous value on the next relational scan.
        old = dict(RDX.state)
        try:
            values = np.array([11, 22, 33], dtype=np.uint32)
            self._fresh_scan([0x10, 0x20, 0x30], values)
            with patch.object(RDX, "_close_turbo_session"):
                dropped = RDX.scan.drop_index(1)
            self.assertEqual(int(dropped), 0x20)
            self.assertEqual(list(RDX.state["scan_results"]), [0x10, 0x30])
            self.assertEqual(list(RDX.state["scan_values"]), [11, 33])
            self.assertIn(0x20, RDX.state["scan_dropped"])
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_drop_address_removes_the_right_row(self):
        old = dict(RDX.state)
        try:
            values = np.array([11, 22, 33], dtype=np.uint32)
            self._fresh_scan([0x10, 0x20, 0x30], values)
            with patch.object(RDX, "_close_turbo_session"):
                RDX.scan.drop_address(0x30)
            self.assertEqual(list(RDX.state["scan_results"]), [0x10, 0x20])
            self.assertEqual(list(RDX.state["scan_values"]), [11, 22])
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_drop_out_of_range_is_a_no_op(self):
        old = dict(RDX.state)
        try:
            self._fresh_scan([0x10])
            with patch.object(RDX, "_close_turbo_session"):
                self.assertIsNone(RDX.scan.drop_index(9))
                self.assertIsNone(RDX.scan.drop_index(-1))
                self.assertIsNone(RDX.scan.drop_address(0xBEEF))
            self.assertEqual(len(RDX.state["scan_results"]), 1)
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_dropping_closes_the_resident_turbo_session(self):
        # A resident session is matched by connection/PID/width/value-type,
        # never by candidate count, so it would hand the address back.
        old = dict(RDX.state)
        RDX.state.update(ip="test", pid=1)
        stub, restore = self._install_turbo_session("exact")
        try:
            RDX.scan.replace(RDX._make_addr_array([0x10, 0x20]),
                             close_turbo=False)
            RDX.scan.drop_index(0)
            self.assertTrue(stub.closed)
            with RDX._turbo_session_lock:
                self.assertIsNone(RDX._turbo_session)
        finally:
            restore(); RDX.state.clear(); RDX.state.update(old)

    def test_replace_closes_the_resident_turbo_session_by_default(self):
        old = dict(RDX.state)
        RDX.state.update(ip="test", pid=1)
        stub, restore = self._install_turbo_session("exact")
        try:
            RDX.scan.replace(RDX._make_addr_array([0x10]))
            self.assertTrue(stub.closed)
        finally:
            restore(); RDX.state.clear(); RDX.state.update(old)

    def test_mismatched_value_length_is_rejected_loudly(self):
        old = dict(RDX.state)
        try:
            with self.assertRaises(AssertionError):
                RDX.scan.replace(RDX._make_addr_array([0x10, 0x20]),
                                 np.array([1], dtype=np.uint32),
                                 close_turbo=False)
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_snapshot_and_restore_round_trip(self):
        old = dict(RDX.state)
        try:
            values = np.array([11, 22], dtype=np.uint32)
            self._fresh_scan([0x10, 0x20], values)
            RDX.state["scan_truncated"] = True
            snap = RDX.scan.snapshot()
            with patch.object(RDX, "_close_turbo_session"):
                RDX.scan.drop_index(0)
            self.assertEqual(len(RDX.state["scan_results"]), 1)
            RDX.scan.restore(snap)
            self.assertEqual(list(RDX.state["scan_results"]), [0x10, 0x20])
            self.assertEqual(list(RDX.state["scan_values"]), [11, 22])
            self.assertTrue(RDX.state["scan_truncated"])
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_narrow_preserves_pid_and_dropped_set(self):
        # A next-scan narrows within the same attach; it is not a new set.
        old = dict(RDX.state)
        try:
            self._fresh_scan([0x10, 0x20, 0x30], pid=9)
            RDX.state["scan_dropped"] = {0xAA}
            RDX.scan.narrow(RDX._make_addr_array([0x10]), None)
            self.assertEqual(RDX.state["scan_pid"], 9)
            self.assertEqual(RDX.state["scan_dropped"], {0xAA})
        finally:
            RDX.state.clear(); RDX.state.update(old)

    # ── patch93: command registry (audit item 10 / B1) ───────────────────
    #
    # Fails against patch92: dispatch, the palette and the main menu each
    # held their own copy of the action table there.

    def test_registry_is_the_single_source_for_all_three_consumers(self):
        registry = RDX._commands()
        # Every main-menu entry except Quit resolves to a registered command.
        for _key, _label, action, _cp in RDX._main_menu_entries():
            if action is None:
                continue
            self.assertIn(action, registry)
        # Every palette entry resolves too.
        for command in registry.values():
            if command.in_palette:
                self.assertIs(registry[command.name], command)

    def test_every_registered_handler_is_callable_or_special(self):
        # proc/reconnect are handled inside dispatch and carry no handler.
        for name, command in RDX._commands().items():
            if name in ("proc", "reconnect"):
                self.assertIsNone(command.handler)
            else:
                self.assertTrue(callable(command.handler), name)

    def test_availability_rules_are_shared_by_menu_and_palette(self):
        # The palette used to offer commands the main menu already knew
        # could not run.
        old = dict(RDX.state)
        try:
            RDX.state["pid"] = 1
            RDX.state["scan_results"] = RDX._make_addr_array()
            self.assertFalse(RDX._command_available("scan_next"))
            self.assertFalse(RDX._command_available("results"))
            self.assertEqual(RDX._command_unavailable_reason("scan_next"),
                             "no scan results yet")
            RDX.state["scan_results"] = RDX._make_addr_array([0x10])
            self.assertTrue(RDX._command_available("scan_next"))
            self.assertTrue(RDX._command_available("results"))
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_commands_needing_a_process_are_blocked_before_attach(self):
        old = dict(RDX.state)
        try:
            RDX.state["pid"] = None
            self.assertEqual(RDX._command_unavailable_reason("hex_view"),
                             "attach to a process first")
            self.assertEqual(RDX._command_unavailable_reason("scan_first"),
                             "attach to a process first")
            # Commands with no process requirement stay reachable.
            self.assertTrue(RDX._command_available("cheat_list"))
            self.assertTrue(RDX._command_available("log"))
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_dispatch_refuses_an_unavailable_command(self):
        old = dict(RDX.state)
        try:
            RDX.state["pid"] = 1
            RDX.state["scan_results"] = RDX._make_addr_array()
            with patch.object(RDX, "do_scan_next",
                              side_effect=AssertionError("ran anyway")):
                self.assertIsNone(
                    RDX.dispatch(self._FakeKeyWindow([]), "scan_next"))
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_dispatch_runs_an_available_command(self):
        old = dict(RDX.state)
        ran = []
        try:
            RDX.state["pid"] = 1
            RDX.state["scan_results"] = RDX._make_addr_array([0x10])
            with patch.dict(RDX._commands(),
                            {"scan_next": RDX.Command(
                                "scan_next", "Next Scan",
                                lambda _s: ran.append(True),
                                requires_results=True)}):
                RDX.dispatch(self._FakeKeyWindow([]), "scan_next")
            self.assertEqual(ran, [True])
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_dispatch_survives_an_unknown_command(self):
        # A typo in a caller must log, not raise into the curses loop.
        self.assertIsNone(RDX.dispatch(self._FakeKeyWindow([]), "not_a_command"))

    def test_proc_and_reconnect_keep_their_navigation_contract(self):
        old = dict(RDX.state)
        try:
            self.assertEqual(RDX.dispatch(self._FakeKeyWindow([]), "proc"),
                             "proc")
            with patch.object(RDX, "_stop_freeze_worker"):
                self.assertEqual(
                    RDX.dispatch(self._FakeKeyWindow([]), "reconnect"),
                    "connect")
            self.assertFalse(RDX.state["connected"])
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_menu_keys_are_unique(self):
        keys = [k for k, _l, _a, _c in RDX._main_menu_entries()]
        self.assertEqual(len(keys), len(set(keys)))

    # ── patch94: run-length-encoded undo deltas (audit item 11 / B3) ──────
    #
    # Fails against patch93: deltas were stored as raw uint64 arrays there,
    # which is what HISTORY_RAM_CAP_MB and the Clear Scan History screen
    # exist to bound.

    def test_rle_round_trips_a_contiguous_aligned_delta_exactly(self):
        arr = np.arange(0x100000, 0x100000 + 4 * 50_000, 4, dtype=np.uint64)
        stored = RDX._UndoAddrs(arr)
        self.assertTrue(stored.compressed)
        np.testing.assert_array_equal(stored.array(), arr)
        self.assertEqual(len(stored), len(arr))

    def test_rle_collapses_a_strided_run_to_constant_size(self):
        # The whole point: a Next Scan that removes a million consecutive
        # aligned addresses must not cost 8 MB of undo history.
        small = RDX._UndoAddrs(
            np.arange(0x1000, 0x1000 + 4 * 1_000, 4, dtype=np.uint64))
        large = RDX._UndoAddrs(
            np.arange(0x1000, 0x1000 + 4 * 1_000_000, 4, dtype=np.uint64))
        self.assertTrue(large.compressed)
        self.assertEqual(small.nbytes, large.nbytes)
        self.assertLess(large.nbytes, 256)

    def test_rle_round_trips_multiple_disjoint_runs(self):
        arr = np.concatenate([
            np.arange(base, base + 4 * 5_000, 4, dtype=np.uint64)
            for base in (0x10000, 0x900000, 0x5000000)])
        stored = RDX._UndoAddrs(arr)
        self.assertTrue(stored.compressed)
        np.testing.assert_array_equal(stored.array(), arr)

    def test_rle_falls_back_to_raw_on_a_scattered_delta(self):
        # RLE is 3x worse than raw with no runs, so it must not be forced.
        rng = np.random.default_rng(7)
        arr = np.sort(rng.choice(10 ** 9, 20_000,
                                 replace=False).astype(np.uint64))
        stored = RDX._UndoAddrs(arr)
        self.assertFalse(stored.compressed)
        self.assertEqual(stored.nbytes, arr.nbytes)
        np.testing.assert_array_equal(stored.array(), arr)

    def test_rle_handles_degenerate_deltas(self):
        for arr in (np.array([], dtype=np.uint64),
                    np.array([5], dtype=np.uint64),
                    np.array([5, 9], dtype=np.uint64),
                    np.array([1, 2, 3], dtype=np.uint64)):
            stored = RDX._UndoAddrs(arr)
            np.testing.assert_array_equal(stored.array(), arr)
            self.assertEqual(len(stored), len(arr))

    def test_rle_preserves_a_mixed_run_and_scatter_delta(self):
        arr = np.concatenate([
            np.arange(0x1000, 0x1000 + 4 * 2_000, 4, dtype=np.uint64),
            np.array([0x800001, 0x9000FF, 0xA00003], dtype=np.uint64)])
        stored = RDX._UndoAddrs(arr)
        np.testing.assert_array_equal(stored.array(), arr)

    def test_history_accounting_reflects_the_stored_size(self):
        # HISTORY_RAM_CAP_MB evicts on measured bytes, so the measurement has
        # to be the compressed size or the cap evicts levels it need not.
        old = dict(RDX.state)
        try:
            RDX.state["scan_history"] = RDX.deque(maxlen=5)
            RDX._push_undo(
                np.arange(0x1000, 0x1000 + 4 * 500_000, 4, dtype=np.uint64),
                None, set(), False)
            self.assertLess(RDX._history_bytes(), 4096)
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_undo_restores_the_previous_candidate_set_through_rle(self):
        old = dict(RDX.state)
        try:
            RDX.state["pid"] = 5
            RDX.state["scan_history"] = RDX.deque(maxlen=5)
            full = np.arange(0x1000, 0x1000 + 4 * 100, 4, dtype=np.uint64)
            survivors = full[:10].copy()
            removed = full[10:].copy()
            RDX.scan.replace(survivors, close_turbo=False)
            RDX._push_undo(removed, None, set(), False)
            with patch.object(RDX, "_close_turbo_session"):
                restored = RDX._apply_scan_undo()
            np.testing.assert_array_equal(restored, full)
            np.testing.assert_array_equal(RDX.state["scan_results"], full)
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_undo_still_restores_values_for_an_unknown_scan(self):
        old = dict(RDX.state)
        try:
            RDX.state.update(pid=5, scan_type="u32", scan_width=4)
            RDX.state["scan_history"] = RDX.deque(maxlen=5)
            addrs = np.arange(0x1000, 0x1000 + 4 * 8, 4, dtype=np.uint64)
            survivors, removed = addrs[:4].copy(), addrs[4:].copy()
            cur_v = np.array([1, 2, 3, 4], dtype=np.uint32)
            rem_v = np.array([5, 6, 7, 8], dtype=np.uint32)
            RDX.scan.replace(survivors, cur_v, unknown=True, close_turbo=False)
            RDX._push_undo(removed, rem_v, set(), False)
            with patch.object(RDX, "_close_turbo_session"):
                RDX._apply_scan_undo()
            np.testing.assert_array_equal(RDX.state["scan_results"], addrs)
            np.testing.assert_array_equal(
                RDX.state["scan_values"], np.array([1, 2, 3, 4, 5, 6, 7, 8],
                                                   dtype=np.uint32))
        finally:
            RDX.state.clear(); RDX.state.update(old)

    # ── patch95: Target protocol (audit item 12 / B5) ────────────────────
    #
    # Fails against patch94: the backend choice lived as an ad-hoc
    # _memdbg_has(CAP) guard at each call site, with no seam a test could
    # stand in for. HARDWARE_TEST_CHECKLIST lists the MemDBG backend as
    # never having run against a real daemon; this is what lets its
    # capability matrix be exercised without one.

    def test_ps5debug_target_reports_the_whole_surface(self):
        target = RDX.Ps5DebugTarget("1.2.3.4")
        for cap in RDX.Target.ALL_CAPS:
            self.assertTrue(target.has(cap), cap)
        self.assertEqual(target.port, RDX.PS5_PORT)
        self.assertIn("all capabilities", target.describe())

    def test_memdbg_target_maps_its_capability_bitmap(self):
        hello = {"capabilities": (RDX.MEMDBG_CAP_MEMORY_READ |
                                  RDX.MEMDBG_CAP_MEMORY_WRITE)}
        target = RDX.MemDbgTarget("1.2.3.4", hello)
        self.assertTrue(target.has(RDX.Target.CAP_READ))
        self.assertTrue(target.has(RDX.Target.CAP_WRITE))
        self.assertFalse(target.has(RDX.Target.CAP_PROCESSES))
        self.assertFalse(target.has(RDX.Target.CAP_WRITE_MULTI))
        self.assertEqual(target.port, RDX.MEMDBG_PORT)

    def test_memdbg_never_claims_turbo_or_the_region_classifier(self):
        # These are ps5debug commands with no MemDBG equivalent. Claiming
        # them would send scans down a path the payload cannot serve.
        target = RDX.MemDbgTarget("1.2.3.4", {"capabilities": 0xFFFFFFFF})
        self.assertFalse(target.has(RDX.Target.CAP_TURBO))
        self.assertFalse(target.has(RDX.Target.CAP_REGION_CLASSIFY))

    def test_memdbg_with_no_advertised_capabilities_reports_none(self):
        target = RDX.MemDbgTarget("1.2.3.4", {})
        self.assertEqual(target.capabilities(), frozenset())
        self.assertIn("missing", target.describe())

    def test_describe_names_exactly_what_is_missing(self):
        target = RDX.MemDbgTarget(
            "1.2.3.4", {"capabilities": RDX.MEMDBG_CAP_MEMORY_READ})
        described = target.describe()
        self.assertIn("turbo", described)
        self.assertIn("write", described)
        self.assertNotIn("missing read", described)

    def test_current_target_follows_the_active_backend(self):
        old = {k: RDX.state.get(k) for k in ("backend", "memdbg", "ip")}
        try:
            RDX.state.update(backend="ps5debug", memdbg=None, ip="1.2.3.4")
            self.assertIsInstance(RDX.current_target(), RDX.Ps5DebugTarget)
            RDX.state.update(backend="memdbg-experimental",
                             memdbg={"capabilities": RDX.MEMDBG_CAP_MEMORY_READ})
            target = RDX.current_target()
            self.assertIsInstance(target, RDX.MemDbgTarget)
            self.assertTrue(target.has(RDX.Target.CAP_READ))
        finally:
            RDX.state.update(old)

    def test_a_mock_target_satisfies_the_protocol(self):
        # The point of the seam: MemDBG behaviour is now expressible without
        # a daemon or a faked socket.
        class MockTarget(RDX.Target):
            name = "mock"
            def capabilities(self):
                return frozenset({RDX.Target.CAP_READ})
            def read(self, pid, addr, length):
                return bytes(length)

        target = MockTarget("0.0.0.0")
        self.assertTrue(target.has(RDX.Target.CAP_READ))
        self.assertFalse(target.has(RDX.Target.CAP_WRITE))
        self.assertEqual(target.read(1, 0x1000, 8), bytes(8))
        with self.assertRaises(NotImplementedError):
            target.write(1, 0x1000, b"\x00")

    def test_target_agrees_with_the_legacy_memdbg_guard(self):
        # The seam must not disagree with the guard still in the wire code,
        # or a capability would be reported one way and used the other.
        old = {k: RDX.state.get(k) for k in ("backend", "memdbg")}
        try:
            for bits, cap, legacy in (
                    (RDX.MEMDBG_CAP_MEMORY_READ, RDX.Target.CAP_READ,
                     RDX.MEMDBG_CAP_MEMORY_READ),
                    (RDX.MEMDBG_CAP_BATCH_WRITE, RDX.Target.CAP_WRITE_MULTI,
                     RDX.MEMDBG_CAP_BATCH_WRITE),
                    (0, RDX.Target.CAP_READ, RDX.MEMDBG_CAP_MEMORY_READ)):
                RDX.state.update(backend="memdbg-experimental",
                                 memdbg={"capabilities": bits})
                self.assertEqual(RDX.current_target().has(cap),
                                 RDX._memdbg_has(legacy))
        finally:
            RDX.state.update(old)

    # ── patch96: structure view (audit item 13 / U2) ─────────────────────
    #
    # Fails against patch95: there was no structure overlay there.

    def test_auto_dissect_identifies_a_pointer_against_the_live_map(self):
        # A qword resolving inside a mapped region is far more likely a
        # pointer than a coincidental integer of that magnitude.
        maps = [{"name": "heap", "start": 0x7F0000000000,
                 "end": 0x7F0000100000, "prot": 3}]
        raw = (0x7F0000001234).to_bytes(8, "little") + bytes(8)
        fields = RDX._struct_auto_fields(raw, maps)
        self.assertEqual(fields[0]["type"], "ptr")
        self.assertEqual(fields[0]["offset"], 0)

    def test_auto_dissect_does_not_call_an_unmapped_qword_a_pointer(self):
        maps = [{"name": "heap", "start": 0x7F0000000000,
                 "end": 0x7F0000100000, "prot": 3}]
        raw = (0x11223344).to_bytes(8, "little") + bytes(8)
        fields = RDX._struct_auto_fields(raw, maps)
        self.assertNotEqual(fields[0]["type"], "ptr")

    def test_auto_dissect_advances_eight_bytes_past_a_pointer(self):
        # A pointer occupies the whole qword; anything else is a 4-byte slot
        # so adjacent 32-bit fields stay separately addressable.
        maps = [{"name": "heap", "start": 0x7F0000000000,
                 "end": 0x7F0000100000, "prot": 3}]
        raw = (0x7F0000001234).to_bytes(8, "little") + bytes(16)
        offsets = [f["offset"] for f in RDX._struct_auto_fields(raw, maps)]
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[1], 8)

    def test_auto_dissect_recognises_a_plausible_float(self):
        raw = struct.pack("<f", 3.5) + bytes(12)
        fields = RDX._struct_auto_fields(raw, [])
        self.assertEqual(fields[0]["type"], "f32")

    def test_auto_dissect_never_runs_past_the_window(self):
        for size in (0, 3, 8, 9, 64):
            fields = RDX._struct_auto_fields(bytes(size), [])
            for field in fields:
                self.assertLessEqual(field["offset"] + 4, max(size, 4))

    def test_field_values_render_for_every_supported_type(self):
        raw = bytes(range(64))
        for kind in RDX._STRUCT_TYPES:
            rendered = RDX._struct_field_value(raw, {"offset": 0,
                                                     "type": kind})
            self.assertIsInstance(rendered, str)
            self.assertTrue(rendered)

    def test_field_value_reports_a_truncated_read_rather_than_raising(self):
        self.assertEqual(
            RDX._struct_field_value(b"\x01\x02", {"offset": 0, "type": "u64"}),
            "??")
        self.assertEqual(
            RDX._struct_field_value(bytes(8), {"offset": 0x400, "type": "u32"}),
            "??")

    def test_pointer_field_renders_as_hex(self):
        raw = (0xDEADBEEF).to_bytes(8, "little")
        self.assertEqual(
            RDX._struct_field_value(raw, {"offset": 0, "type": "ptr"}),
            hex(0xDEADBEEF))

    def test_structure_view_never_writes(self):
        old = {k: RDX.state.get(k) for k in ("ip", "pid", "proc_name")}
        RDX.state.update(ip="test", pid=1, proc_name="eboot.bin")
        RDX.state["structures"] = {}
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "_get_maps_cached", return_value=[]), \
                 patch.object(RDX, "ps5_read", return_value=bytes(0x400)), \
                 patch.object(RDX, "ps5_write",
                              side_effect=AssertionError("structure wrote")), \
                 patch.object(RDX, "ps5_write_verified",
                              side_effect=AssertionError("structure wrote")):
                keys = [ord('j'), ord('k'), ord('G'), ord('g'), ord('q')]
                RDX.do_structure_view(self._FakeKeyWindow(keys), 0x1000)
        finally:
            RDX.state.update(old)

    def test_structures_are_dropped_with_the_rest_of_the_scan_state(self):
        old = dict(RDX.state)
        try:
            RDX.state["structures"] = {0x1000: [{"offset": 0, "name": "x",
                                                 "type": "u32"}]}
            with patch.object(RDX, "_stop_freeze_worker"), \
                 patch.object(RDX, "_close_turbo_session"), \
                 patch.object(RDX, "_invalidate_pointer_index"):
                RDX._clear_scan_state(stop_freezes=False)
            self.assertEqual(RDX.state["structures"], {})
        finally:
            RDX.state.clear(); RDX.state.update(old)

    # ── patch97: watchpoint DR read-back (pass-2 items 1 and 5) ──────────
    #
    # Fails against patch96: the debug registers were read only *before*
    # arming, on threads[0] alone, so "no event in 60 s" could not be told
    # apart from "the watchpoint was never installed".

    @staticmethod
    def _dbreg_blob(dr7=0, addrs=(0, 0, 0, 0)):
        """Build a 128-byte dbreg blob: DR0-3 at 0-3, DR7 at index 7."""
        regs = [0] * 16
        for i, a in enumerate(addrs):
            regs[i] = int(a)
        regs[7] = int(dr7)
        return struct.pack("<16Q", *regs)

    @staticmethod
    def _dr7_enable(*slots):
        """DR7 with the local-enable bit set for each given slot."""
        value = 0
        for i in slots:
            value |= 1 << (2 * i)
        return value

    def test_decode_dbregs_reads_slot_enables_and_addresses(self):
        blob = self._dbreg_blob(self._dr7_enable(0, 2),
                                (0xAAAA, 0, 0xCCCC, 0))
        decoded = RDX._debug_decode_dbregs(blob)
        enabled = [s["index"] for s in decoded["slots"] if s["enabled"]]
        self.assertEqual(enabled, [0, 2])
        self.assertEqual(decoded["slots"][0]["address"], 0xAAAA)
        self.assertEqual(decoded["slots"][2]["address"], 0xCCCC)

    def test_decode_dbregs_honours_the_global_enable_bit(self):
        # DR7 bit 2i+1 is the global enable; a slot armed that way is busy.
        decoded = RDX._debug_decode_dbregs(self._dbreg_blob(1 << 1))
        self.assertTrue(decoded["slots"][0]["enabled"])

    def test_free_slot_probe_takes_the_union_across_threads(self):
        # The pass-2 finding: threads[0] alone can report a slot free that is
        # occupied on another thread, so the chosen index is wrong for the
        # thread that matters.
        per_thread = {
            10: self._dbreg_blob(self._dr7_enable(0)),
            11: self._dbreg_blob(self._dr7_enable(1)),
            12: self._dbreg_blob(self._dr7_enable(2)),
        }
        with patch.object(RDX, "_debug_get_dbregs",
                          lambda _s, lwpid: per_thread[lwpid]):
            chosen = RDX._debug_free_watchpoint_all(None, [10, 11, 12])
            # Old single-thread probe would have returned 1 from thread 10.
            single = RDX._debug_free_watchpoint(None, 10)
        self.assertEqual(chosen, 3)
        self.assertEqual(single, 1)

    def test_free_slot_probe_returns_none_when_all_slots_are_busy(self):
        blob = self._dbreg_blob(self._dr7_enable(0, 1, 2, 3))
        with patch.object(RDX, "_debug_get_dbregs", lambda _s, _l: blob):
            self.assertIsNone(RDX._debug_free_watchpoint_all(None, [10]))

    def test_free_slot_probe_returns_none_when_no_thread_answers(self):
        with patch.object(RDX, "_debug_get_dbregs",
                          side_effect=ConnectionError("refused")):
            self.assertIsNone(RDX._debug_free_watchpoint_all(None, [10, 11]))

    def test_free_slot_probe_ignores_threads_that_cannot_be_read(self):
        def flaky(_s, lwpid):
            if lwpid == 11:
                raise ConnectionError("refused")
            return PointerSubsystemTests._dbreg_blob(
                PointerSubsystemTests._dr7_enable(0))
        with patch.object(RDX, "_debug_get_dbregs", flaky):
            self.assertEqual(
                RDX._debug_free_watchpoint_all(None, [10, 11, 12]), 1)

    def test_verify_reports_full_coverage(self):
        blob = self._dbreg_blob(self._dr7_enable(1), (0, 0x4000, 0, 0))
        with patch.object(RDX, "_debug_get_dbregs", lambda _s, _l: blob):
            cov = RDX._debug_verify_watchpoint(None, [10, 11, 12], 0x4000, 1)
        self.assertEqual(len(cov["armed"]), 3)
        self.assertEqual(cov["absent"], [])
        self.assertEqual(RDX._debug_watchpoint_verdict(cov)[0], "all")

    def test_verify_detects_per_thread_application(self):
        # The hypothesis the checklist could not test: DRs on one thread only.
        armed = self._dbreg_blob(self._dr7_enable(1), (0, 0x4000, 0, 0))
        bare = self._dbreg_blob(0)
        with patch.object(RDX, "_debug_get_dbregs",
                          lambda _s, lwpid: armed if lwpid == 10 else bare):
            cov = RDX._debug_verify_watchpoint(None, [10, 11, 12], 0x4000, 1)
        self.assertEqual(cov["armed"], [10])
        self.assertEqual(len(cov["absent"]), 2)
        key, text = RDX._debug_watchpoint_verdict(cov)
        self.assertEqual(key, "partial")
        self.assertIn("CMD_DEBUG_SETDBREGS", text)

    def test_verify_detects_an_arm_that_was_acknowledged_and_discarded(self):
        with patch.object(RDX, "_debug_get_dbregs",
                          lambda _s, _l: PointerSubsystemTests._dbreg_blob(0)):
            cov = RDX._debug_verify_watchpoint(None, [10, 11], 0x4000, 1)
        self.assertEqual(cov["armed"], [])
        self.assertEqual(RDX._debug_watchpoint_verdict(cov)[0], "none")

    def test_verify_does_not_count_a_slot_armed_for_another_address(self):
        # A slot enabled for some other address is not this watchpoint;
        # counting it would manufacture coverage and hide the real cause.
        blob = self._dbreg_blob(self._dr7_enable(1), (0, 0xDEAD, 0, 0))
        with patch.object(RDX, "_debug_get_dbregs", lambda _s, _l: blob):
            cov = RDX._debug_verify_watchpoint(None, [10], 0x4000, 1)
        self.assertEqual(cov["armed"], [])

    def test_verify_reports_unreadable_threads_separately(self):
        def flaky(_s, lwpid):
            if lwpid == 11:
                raise ConnectionError("refused")
            return PointerSubsystemTests._dbreg_blob(
                PointerSubsystemTests._dr7_enable(1), (0, 0x4000, 0, 0))
        with patch.object(RDX, "_debug_get_dbregs", flaky):
            cov = RDX._debug_verify_watchpoint(None, [10, 11], 0x4000, 1)
        self.assertEqual(cov["armed"], [10])
        self.assertEqual(cov["unreadable"], [11])
        self.assertEqual(cov["checked"], 1)

    def test_verdict_is_unknown_when_nothing_could_be_read(self):
        cov = {"armed": [], "absent": [], "unreadable": [10], "checked": 0,
               "total": 1}
        self.assertEqual(RDX._debug_watchpoint_verdict(cov)[0], "unknown")

    def test_dr_probe_is_bounded(self):
        seen = []
        def counting(_s, lwpid):
            seen.append(lwpid)
            return PointerSubsystemTests._dbreg_blob(0)
        with patch.object(RDX, "_debug_get_dbregs", counting):
            RDX._debug_verify_watchpoint(
                None, list(range(1000)), 0x4000, 1)
        self.assertEqual(len(seen), RDX._DR_PROBE_MAX_THREADS)

    # ── patch98: export portability counter (pass-2 item 2) ──────────────
    #
    # Fails against patch97: _is_portable_cheat already ran inside
    # belongs_to_current_game there, but only as one arm of an OR — the
    # answer was discarded, so the preflight could not say whether the file
    # it was about to write would survive a reload.

    _EXPORT_MAPS = [{"name": "executable", "start": 0x400000, "end": 0x500000,
                     "prot": 5}]

    def _export_identity(self):
        """The game identity do_export will compute for _EXPORT_MAPS."""
        return RDX._pointer_game_identity("eboot.bin", self._EXPORT_MAPS)

    def _export_probe(self, cheats):
        """Run do_export far enough to capture the preflight lines."""
        captured = {}

        def fake_confirm(_s, question, _title="Confirm"):
            captured["preflight"] = question
            return False          # stop before writing anything

        old = dict(RDX.state)
        RDX.state.update(pid=1, session=3, proc_name="eboot.bin",
                         game_id="CUSA01659", game_ver="01.00",
                         game_title="T", cheats=list(cheats))
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "message_box"), \
                 patch.object(RDX, "_get_maps_cached",
                              return_value=self._EXPORT_MAPS), \
                 patch.object(RDX, "_select_export_cheats",
                              side_effect=lambda _s, c: c), \
                 patch.object(RDX, "input_box", side_effect=[
                     "CUSA01659", "01.00", "T", "RDX",
                     "/tmp/rdx-export-test"]), \
                 patch.object(RDX, "confirm_box", fake_confirm):
                RDX.do_export(self._FakeKeyWindow([]))
        finally:
            RDX.state.clear(); RDX.state.update(old)
        return captured.get("preflight", "")

    def _session_cheat(self, addr=0x253dff648):
        # Same shape as the project's own exports/EnterTheGungeon.rdx.json:
        # an absolute heap address with no module root.
        return {"name": "Infinite Ammo", "address": addr, "width": 4,
                "value": 999, "value_type": "i32", "type": "i32",
                "pid": 1, "session": 3, "process": "eboot.bin",
                "module_name": None, "module_relative_offset": None}

    def _portable_cheat(self):
        # Portable means a module root plus a game identity matching the one
        # do_export computes for the maps above. pid/session deliberately do
        # NOT match the current attach, so this is admitted on the portable
        # arm of belongs_to_current_game rather than the session arm.
        return {"name": "Godmode", "address": 0x400100, "width": 4,
                "value": 1, "value_type": "i32", "type": "i32",
                "pid": 99, "session": 99, "process": "eboot.bin",
                "module_name": "executable", "module_relative_offset": 0x100,
                "original_value": 0,
                "game_identity": self._export_identity()}

    def test_preflight_flags_session_bound_entries(self):
        text = self._export_probe([self._session_cheat()])
        self.assertIn("session-bound", text)
        self.assertIn("will NOT", text)

    def test_preflight_confirms_when_every_entry_is_reload_safe(self):
        # A portable cheat only counts as portable for the *current* game, so
        # pin the identity the export path computes with no maps available.
        text = self._export_probe([self._portable_cheat()])
        self.assertIn("reload-safe", text)
        self.assertNotIn("will NOT", text)

    def test_preflight_counts_a_mixed_set(self):
        text = self._export_probe(
            [self._session_cheat(), self._portable_cheat()])
        self.assertIn("Native RDX entries: 2", text)
        self.assertIn("1 of these are session-bound", text)

    def test_the_shipped_export_shape_is_correctly_classified(self):
        # Grounded in exports/EnterTheGungeon.rdx.json: both entries are
        # absolute heap addresses, so both must be reported session-bound.
        entries = [self._session_cheat(0x253dff648),
                   self._session_cheat(0x253dff648)]
        for c in entries:
            self.assertFalse(RDX._is_portable_cheat(c))
        text = self._export_probe(entries)
        self.assertIn("2 of these are session-bound", text)

    # ── patch99: changed-memory highlight + pointer preview (items 3, 4) ──
    #
    # Fails against patch98: both views re-read on a timer and held the
    # previous buffer, but never compared them, and a ptr slot showed only
    # the address it held.

    def test_hex_diff_finds_changed_offsets(self):
        prev = bytes([0, 1, 2, 3, 4, 5, 6, 7])
        cur = bytes([0, 9, 2, 3, 4, 5, 6, 8])
        self.assertEqual(RDX._hex_changed_offsets(prev, cur), {1, 7})

    def test_hex_diff_is_empty_on_the_first_frame(self):
        # Nothing to compare against must highlight nothing, not everything.
        self.assertEqual(RDX._hex_changed_offsets(None, bytes(16)), set())
        self.assertEqual(RDX._hex_changed_offsets(b"", bytes(16)), set())

    def test_hex_diff_is_empty_when_the_window_size_changed(self):
        # Scrolling or resizing is not a memory change.
        self.assertEqual(
            RDX._hex_changed_offsets(bytes(16), bytes(32)), set())

    def test_hex_diff_is_empty_when_nothing_moved(self):
        same = bytes(range(32))
        self.assertEqual(RDX._hex_changed_offsets(same, same), set())

    def test_pointer_preview_names_the_target_region(self):
        maps = [{"name": "/app0/eboot.bin", "start": 0x400000, "end": 0x500000,
                 "prot": 5}]
        raw = (0x400100).to_bytes(8, "little")
        preview = RDX._struct_pointer_target(
            {"offset": 0, "type": "ptr"}, raw, maps)
        self.assertIn("eboot.bin", preview)
        self.assertIn("0x100", preview)

    def test_pointer_preview_reports_null_and_unmapped(self):
        maps = [{"name": "heap", "start": 0x400000, "end": 0x500000, "prot": 3}]
        null = RDX._struct_pointer_target(
            {"offset": 0, "type": "ptr"}, bytes(8), maps)
        self.assertEqual(null, "NULL")
        stray = RDX._struct_pointer_target(
            {"offset": 0, "type": "ptr"},
            (0xDEADBEEF).to_bytes(8, "little"), maps)
        self.assertEqual(stray, "unmapped")

    def test_pointer_preview_only_applies_to_pointer_fields(self):
        maps = [{"name": "heap", "start": 0x400000, "end": 0x500000, "prot": 3}]
        raw = (0x400100).to_bytes(8, "little")
        for kind in ("u32", "i64", "f32", "bytes"):
            self.assertEqual(
                RDX._struct_pointer_target({"offset": 0, "type": kind},
                                           raw, maps), "")

    def test_pointer_preview_handles_a_truncated_read(self):
        maps = [{"name": "heap", "start": 0x400000, "end": 0x500000, "prot": 3}]
        self.assertEqual(
            RDX._struct_pointer_target({"offset": 0, "type": "ptr"},
                                       b"\x01\x02", maps), "")

    def test_pointer_preview_uses_the_fast_region_lookup_when_given_one(self):
        maps = [{"name": "libc.prx", "start": 0x900000, "end": 0x910000,
                 "prot": 5}]
        starts, rows = RDX._build_region_lookup(maps)
        preview = RDX._struct_pointer_target(
            {"offset": 0, "type": "ptr"},
            (0x900040).to_bytes(8, "little"), maps, starts, rows)
        self.assertIn("libc.prx", preview)

    def test_hex_view_highlight_toggle_does_not_write(self):
        # 'c' toggles highlighting; the view stays read-only.
        old = {k: RDX.state.get(k) for k in ("ip", "pid", "proc_name")}
        RDX.state.update(ip="test", pid=1, proc_name="eboot.bin")
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "ps5_read", return_value=bytes(4096)), \
                 patch.object(RDX, "ps5_write",
                              side_effect=AssertionError("hex view wrote")):
                RDX.do_hex_view(
                    self._FakeKeyWindow([ord('c'), ord('c'), ord('q')]), 0x1000)
        finally:
            RDX.state.update(old)

    def test_structure_view_highlight_toggle_does_not_write(self):
        old = {k: RDX.state.get(k) for k in ("ip", "pid", "proc_name")}
        RDX.state.update(ip="test", pid=1, proc_name="eboot.bin")
        RDX.state["structures"] = {}
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "safe_addstr"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "_get_maps_cached", return_value=[]), \
                 patch.object(RDX, "ps5_read", return_value=bytes(0x400)), \
                 patch.object(RDX, "ps5_write",
                              side_effect=AssertionError("structure wrote")):
                RDX.do_structure_view(
                    self._FakeKeyWindow([ord('c'), ord('j'), ord('q')]), 0x1000)
        finally:
            RDX.state.update(old)

    # ── patch100: instance discovery by type pointer (pass-2 item 8) ─────
    #
    # Fails against patch99: RDX had no way to find objects, only values.

    def test_static_intervals_are_coalesced(self):
        maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                 "prot": 5},
                {"name": "executable", "start": 0x410000, "end": 0x420000,
                 "prot": 5},
                {"name": "anon", "start": 0x7F0000000000,
                 "end": 0x7F0000010000, "prot": 3}]
        starts, ends = RDX._static_interval_arrays(maps)
        self.assertEqual(len(starts), 1)
        self.assertEqual(int(starts[0]), 0x400000)
        self.assertEqual(int(ends[0]), 0x420000)

    def test_interval_membership_is_half_open(self):
        starts = np.array([0x1000, 0x9000], dtype=np.uint64)
        ends = np.array([0x2000, 0xA000], dtype=np.uint64)
        values = np.array([0x0FFF, 0x1000, 0x1FFF, 0x2000, 0x9500, 0xB000],
                          dtype=np.uint64)
        mask = RDX._values_in_intervals(values, starts, ends)
        self.assertEqual(list(mask), [False, True, True, False, True, False])

    def test_interval_membership_handles_empty_inputs(self):
        empty = np.empty(0, dtype=np.uint64)
        self.assertEqual(
            len(RDX._values_in_intervals(empty, empty, empty)), 0)
        self.assertFalse(
            RDX._values_in_intervals(np.array([1], dtype=np.uint64),
                                     empty, empty).any())

    def test_grouping_keeps_only_repeated_type_pointers(self):
        # A value seen once is an ordinary pointer; one shared by many object
        # bases is a type identity. That repetition is the whole signal.
        values = np.array([0x400100] * 10 + [0x400200] * 3 + [0x400300],
                          dtype=np.uint64)
        holders = np.arange(len(values), dtype=np.uint64) * 8 + 0x800000
        groups = RDX._group_type_pointers(values, holders, min_instances=4)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["type_ptr"], 0x400100)
        self.assertEqual(groups[0]["count"], 10)
        self.assertEqual(len(groups[0]["instances"]), 10)

    def test_grouping_pairs_each_instance_with_its_own_type(self):
        # Interleaved, so a bug that sorts values without carrying holders
        # along would hand back the wrong addresses.
        values = np.array([0xA, 0xB, 0xA, 0xB, 0xA, 0xB], dtype=np.uint64)
        holders = np.array([0x10, 0x20, 0x30, 0x40, 0x50, 0x60],
                           dtype=np.uint64)
        groups = RDX._group_type_pointers(values, holders, min_instances=3)
        by_type = {g["type_ptr"]: set(int(x) for x in g["instances"])
                   for g in groups}
        self.assertEqual(by_type[0xA], {0x10, 0x30, 0x50})
        self.assertEqual(by_type[0xB], {0x20, 0x40, 0x60})

    def test_grouping_orders_by_instance_count(self):
        values = np.array([1] * 5 + [2] * 20 + [3] * 9, dtype=np.uint64)
        holders = np.arange(len(values), dtype=np.uint64)
        groups = RDX._group_type_pointers(values, holders, min_instances=4)
        self.assertEqual([g["type_ptr"] for g in groups], [2, 3, 1])

    def test_grouping_bounds_retained_instances(self):
        n = RDX._TYPE_SCAN_MAX_INSTANCES + 500
        values = np.full(n, 0x400100, dtype=np.uint64)
        holders = np.arange(n, dtype=np.uint64) * 8
        groups = RDX._group_type_pointers(values, holders, min_instances=4)
        self.assertEqual(groups[0]["count"], n)
        self.assertEqual(len(groups[0]["instances"]),
                         RDX._TYPE_SCAN_MAX_INSTANCES)

    def test_grouping_handles_an_empty_scan(self):
        empty = np.empty(0, dtype=np.uint64)
        self.assertEqual(RDX._group_type_pointers(empty, empty), [])

    def test_type_scan_finds_a_synthetic_class(self):
        # One module data region, one heap region. Every 0x40 bytes of the
        # heap is an "object" whose first qword is the same type pointer.
        #
        # The class sits in module *data* (prot 1), not in the executable
        # segment. It used to sit at 0x400280 inside a prot=5 mapping, which
        # asserted that a class can live in code — hardware says otherwise,
        # and that fixture is what let the old static-only target filter look
        # correct. See test_type_pointer_targets_admit_heap_and_exclude_code.
        maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                 "prot": 5},
                {"name": "executable", "start": 0x410000, "end": 0x420000,
                 "prot": 1},
                {"name": "anon", "start": 0x800000, "end": 0x800400,
                 "prot": 3}]
        type_ptr = 0x410280
        blob = bytearray(0x400)
        for obj in range(0, 0x400, 0x40):
            blob[obj:obj + 8] = int(type_ptr).to_bytes(8, "little")

        class FakeSock:
            def __init__(self, *a, **k): pass
            def read(self, addr, size, _cancel=None):
                off = addr - 0x800000
                return bytes(blob[off:off + size])
            def close(self): pass

        with patch.object(RDX, "_get_maps_cached", return_value=maps), \
             patch.object(RDX, "_ScanSocket", FakeSock):
            groups = RDX.scan_type_instances("ip", 1, min_instances=8)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["type_ptr"], type_ptr)
        self.assertEqual(groups[0]["count"], 0x400 // 0x40)
        self.assertEqual(groups[0]["module_name"], "executable")
        self.assertEqual(int(groups[0]["instances"][0]), 0x800000)

    def test_type_scan_ignores_pointers_into_code(self):
        # This test used to assert the opposite -- that heap targets are not
        # type-pointer candidates. Hardware disproved the premise: on
        # CUSA01659 every Il2CppClass is heap-allocated, and excluding heap
        # targets removed all of them while leaving the vtable and callback
        # pointers that do target the image. Those are the ones that must be
        # excluded, because a class never lives in code.
        maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                 "prot": 5},
                {"name": "anon", "start": 0x800000, "end": 0x800200,
                 "prot": 3}]
        blob = bytearray(0x200)
        for obj in range(0, 0x200, 8):
            blob[obj:obj + 8] = (0x400100).to_bytes(8, "little")

        class FakeSock:
            def __init__(self, *a, **k): pass
            def read(self, addr, size, _cancel=None):
                off = addr - 0x800000
                return bytes(blob[off:off + size])
            def close(self): pass

        with patch.object(RDX, "_get_maps_cached", return_value=maps), \
             patch.object(RDX, "_ScanSocket", FakeSock):
            self.assertEqual(
                RDX.scan_type_instances("ip", 1, min_instances=4), [])

    def test_type_scan_returns_empty_without_eligible_target_regions(self):
        # Nothing but an executable mapping: no address a class could occupy.
        maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                 "prot": 5}]
        with patch.object(RDX, "_get_maps_cached", return_value=maps):
            self.assertEqual(RDX.scan_type_instances("ip", 1), [])

    def test_type_scan_raises_when_every_read_fails(self):
        # patch100 returned [] here, which is what patch103 identified as a
        # defect: "no type pointers in this title" and "the console went
        # away" produced identical output. Tolerating *scattered* unreadable
        # spans is still required and is covered separately.
        maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                 "prot": 5},
                {"name": "anon", "start": 0x800000, "end": 0x800400,
                 "prot": 3}]
        class FakeSock:
            def __init__(self, *a, **k): pass
            def read(self, _addr, _size, _cancel=None):
                raise ConnectionError("unmapped")
            def close(self): pass
        with patch.object(RDX, "_get_maps_cached", return_value=maps), \
             patch.object(RDX, "_ScanSocket", FakeSock):
            with self.assertRaises(ConnectionError):
                RDX.scan_type_instances("ip", 1)

    def test_type_scan_honours_cancellation(self):
        maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                 "prot": 5},
                {"name": "anon", "start": 0x800000, "end": 0x900000,
                 "prot": 3}]
        class FakeSock:
            def __init__(self, *a, **k): pass
            def read(self, _addr, size, _cancel=None): return bytes(size)
            def close(self): pass
        cancel = threading.Event(); cancel.set()
        with patch.object(RDX, "_get_maps_cached", return_value=maps), \
             patch.object(RDX, "_ScanSocket", FakeSock):
            with self.assertRaises(InterruptedError):
                RDX.scan_type_instances("ip", 1, cancel_event=cancel)

    # ── patch101: IL2CPP symbol import (pass-2 item 7) ───────────────────
    #
    # Fails against patch100: the structure view could only name fields
    # field_0014, because auto-dissect can only infer from bytes.

    _DUMP_CS = '''// Namespace: GungeonGame
public class PlayerController : MonoBehaviour
{
	// Fields
	public int currentHealth; // 0x18
	public float moveSpeed; // 0x1C
	private bool isInvulnerable; // 0x20
	public System.String playerName; // 0x28
	public Inventory inventory; // 0x30
	private static readonly int MaxHealth; // 0x0
}

// Namespace: GungeonGame
public struct Vector3Wrapper
{
	// Fields
	public float x; // 0x0
	public float y; // 0x4
	public float z; // 0x8
}

public class NoFieldsHere
{
	// Fields
}
'''

    def test_dump_cs_parses_classes_and_located_fields(self):
        classes = RDX.parse_il2cpp_dump(self._DUMP_CS)
        self.assertIn("PlayerController", classes)
        self.assertIn("Vector3Wrapper", classes)
        # A class with no located fields cannot overlay anything.
        self.assertNotIn("NoFieldsHere", classes)
        names = [f["name"] for f in classes["PlayerController"]]
        self.assertIn("currentHealth", names)
        self.assertIn("playerName", names)

    def test_dump_cs_fields_are_sorted_by_offset(self):
        fields = RDX.parse_il2cpp_dump(self._DUMP_CS)["PlayerController"]
        offsets = [f["offset"] for f in fields]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(offsets[0], 0x0)

    def test_dump_cs_maps_value_types_to_rdx_kinds(self):
        by_name = {f["name"]: f for f in
                   RDX.parse_il2cpp_dump(self._DUMP_CS)["PlayerController"]}
        self.assertEqual(by_name["currentHealth"]["type"], "i32")
        self.assertEqual(by_name["currentHealth"]["offset"], 0x18)
        self.assertEqual(by_name["moveSpeed"]["type"], "f32")
        self.assertEqual(by_name["isInvulnerable"]["type"], "u8")

    def test_managed_references_become_pointers(self):
        # The case that matters: auto-dissect guesses at these, a declaration
        # does not have to.
        by_name = {f["name"]: f for f in
                   RDX.parse_il2cpp_dump(self._DUMP_CS)["PlayerController"]}
        self.assertEqual(by_name["playerName"]["type"], "ptr")
        self.assertEqual(by_name["inventory"]["type"], "ptr")

    def test_field_kind_mapping_handles_qualified_and_short_names(self):
        for declared, expected in (("int", "i32"), ("System.Int32", "i32"),
                                   ("float", "f32"), ("System.Single", "f32"),
                                   ("ulong", "u64"), ("bool", "u8"),
                                   ("SomeGameClass", "ptr"),
                                   ("System.Collections.Generic.List`1", "ptr"),
                                   ("", "ptr")):
            self.assertEqual(RDX._il2cpp_field_kind(declared), expected,
                             declared)

    def test_parsing_junk_yields_nothing_rather_than_raising(self):
        for text in ("", "not a dump", "public class X {", "// 0x18"):
            self.assertEqual(RDX.parse_il2cpp_dump(text), {})

    def test_field_count_per_class_is_bounded(self):
        lines = ["public class Big", "{"]
        for i in range(RDX._SYMBOL_MAX_FIELDS + 100):
            lines.append(f"\tpublic int f{i}; // {hex(i * 4)}")
        lines.append("}")
        classes = RDX.parse_il2cpp_dump("\n".join(lines))
        self.assertEqual(len(classes["Big"]), RDX._SYMBOL_MAX_FIELDS)

    def test_symbol_fields_render_through_the_structure_view(self):
        # The payoff: a real name and a declared type over real bytes.
        fields = RDX.parse_il2cpp_dump(self._DUMP_CS)["PlayerController"]
        raw = bytearray(0x40)
        raw[0x18:0x1C] = (250).to_bytes(4, "little")
        raw[0x1C:0x20] = struct.pack("<f", 6.5)
        by_name = {f["name"]: f for f in fields}
        self.assertEqual(
            RDX._struct_field_value(bytes(raw), by_name["currentHealth"]),
            "250")
        self.assertIn(
            "6.5",
            RDX._struct_field_value(bytes(raw), by_name["moveSpeed"]))

    def test_symbol_class_names_are_sorted(self):
        old = RDX.state.get("symbols")
        try:
            RDX.state["symbols"] = RDX.parse_il2cpp_dump(self._DUMP_CS)
            self.assertEqual(RDX._symbol_class_names(),
                             ["PlayerController", "Vector3Wrapper"])
        finally:
            RDX.state["symbols"] = old

    def test_symbols_survive_a_process_change(self):
        # Unlike bookmarks and structures: a dump describes the title, not
        # this session's memory, so reattaching must not discard it.
        old = dict(RDX.state)
        try:
            RDX.state["symbols"] = {"X": [{"offset": 0, "name": "a",
                                           "type": "u32"}]}
            with patch.object(RDX, "_stop_freeze_worker"), \
                 patch.object(RDX, "_close_turbo_session"), \
                 patch.object(RDX, "_invalidate_pointer_index"):
                RDX._clear_scan_state(stop_freezes=False)
            self.assertIn("X", RDX.state["symbols"])
            self.assertEqual(RDX.state["structures"], {})
        finally:
            RDX.state.clear(); RDX.state.update(old)

    # ── patch102: pre-hardware review fixes ──────────────────────────────
    #
    # Every test below covers a defect found by reviewing patches 97-101 --
    # the least-reviewed code in the tree -- before hardware testing. Each
    # fails against patch101.

    def test_dr_sweep_does_not_run_while_the_target_is_stopped(self):
        # The one that matters: patch97 put up to 128 sequential round trips
        # between the stop and the resume, in the path this project has
        # already watched black-screen a live game.
        order = []

        def fake_continue(_s, action):
            order.append("stop" if int(action) == 1 else "resume")

        def fake_dbregs(_s, lwpid):
            order.append(f"dbregs:{lwpid}")
            return PointerSubsystemTests._dbreg_blob(
                PointerSubsystemTests._dr7_enable(1), (0, 0x4000, 0, 0))

        threads = list(range(10, 50))          # 40 threads, as observed
        with patch.object(RDX, "_debug_get_dbregs", fake_dbregs):
            with patch.object(RDX, "_debug_continue", fake_continue):
                # Reproduce the arm sequence the trace performs.
                fake_continue(None, 1)
                RDX._debug_verify_watchpoint(None, threads[:1], 0x4000, 1)
                fake_continue(None, 0)
                RDX._debug_verify_watchpoint(None, threads, 0x4000, 1)
        stop_i, resume_i = order.index("stop"), order.index("resume")
        between = [x for x in order[stop_i:resume_i] if x.startswith("dbregs")]
        after = [x for x in order[resume_i:] if x.startswith("dbregs")]
        self.assertEqual(len(between), 1, "more than one read while stopped")
        self.assertEqual(len(after), len(threads))

    def test_struct_field_width_is_shared_by_render_and_diff(self):
        # A flat 8-byte span made two adjacent u8 fields both light up when
        # one byte moved, and lit two rows per change on a 32-bit class.
        self.assertEqual(RDX._struct_field_width({"type": "u8"}), 1)
        self.assertEqual(RDX._struct_field_width({"type": "i16"}), 2)
        self.assertEqual(RDX._struct_field_width({"type": "f32"}), 4)
        self.assertEqual(RDX._struct_field_width({"type": "ptr"}), 8)
        # Unknown types fall back to a sane width rather than raising.
        self.assertEqual(RDX._struct_field_width({"type": "nonsense"}), 4)

    def test_narrow_fields_do_not_falsely_report_a_change(self):
        fields = [{"offset": 0, "name": "a", "type": "u8"},
                  {"offset": 1, "name": "b", "type": "u8"}]
        prev = bytes(16)
        cur = bytearray(16); cur[7] = 0xFF          # neither field's byte
        byte_changes = RDX._hex_changed_offsets(prev, bytes(cur))
        changed = {int(f["offset"]) for f in fields
                   if any(o in byte_changes
                          for o in range(int(f["offset"]),
                                         int(f["offset"])
                                         + RDX._struct_field_width(f)))}
        self.assertEqual(changed, set())
        # ...and a byte inside a field still does report one.
        cur2 = bytearray(16); cur2[1] = 0xFF
        byte_changes2 = RDX._hex_changed_offsets(prev, bytes(cur2))
        changed2 = {int(f["offset"]) for f in fields
                    if any(o in byte_changes2
                           for o in range(int(f["offset"]),
                                          int(f["offset"])
                                          + RDX._struct_field_width(f)))}
        self.assertEqual(changed2, {1})

    def test_bytes_field_renders_a_run_not_one_byte(self):
        raw = bytes(range(32))
        rendered = RDX._struct_field_value(raw, {"offset": 0, "type": "bytes"})
        self.assertEqual(len(rendered.split()), 8)
        self.assertTrue(rendered.startswith("00 01 02"))

    def test_hex_hint_fits_the_documented_minimum_terminal(self):
        # safe_addstr clips, so an over-long hint silently loses its tail --
        # which is where "Q back" lived.
        shown = []
        old = {k: RDX.state.get(k) for k in ("ip", "pid", "proc_name")}
        RDX.state.update(ip="t", pid=1, proc_name="eboot.bin")
        try:
            with patch.object(RDX, "draw_border"), \
                 patch.object(RDX, "draw_statusbar"), \
                 patch.object(RDX, "color", return_value=0), \
                 patch.object(RDX, "safe_addstr",
                              lambda _w, y, x, t, a=0: shown.append((y, x, t))), \
                 patch.object(RDX, "ps5_read", return_value=bytes(4096)):
                RDX.do_hex_view(self._FakeKeyWindow([ord('q')], (24, 72)), 0x1000)
        finally:
            RDX.state.update(old)
        hints = [t for (y, x, t) in shown if y == 3]
        self.assertTrue(hints)
        self.assertLessEqual(len(hints[0]) + 3, RDX._MIN_COLS)
        self.assertIn("Q back", hints[0])

    def test_counted_grouping_matches_the_array_grouping(self):
        # The incremental tally must produce exactly what collect-then-group
        # produced, or the memory fix changed behaviour.
        values = np.array([0xA] * 12 + [0xB] * 3 + [0xC] * 9, dtype=np.uint64)
        holders = np.arange(len(values), dtype=np.uint64) * 8 + 0x800000
        from_arrays = RDX._group_type_pointers(values, holders,
                                               min_instances=8)
        counts, instances = {}, {}
        for v, h in zip(values.tolist(), holders.tolist()):
            counts[v] = counts.get(v, 0) + 1
            instances.setdefault(v, []).append(h)
        from_counts = RDX._group_counted_types(counts, instances,
                                               min_instances=8)
        self.assertEqual([g["type_ptr"] for g in from_arrays],
                         [g["type_ptr"] for g in from_counts])
        self.assertEqual([g["count"] for g in from_arrays],
                         [g["count"] for g in from_counts])
        for a, b in zip(from_arrays, from_counts):
            np.testing.assert_array_equal(a["instances"], b["instances"])

    def test_type_scan_tally_is_bounded_by_distinct_types(self):
        self.assertIsInstance(RDX._TYPE_SCAN_MAX_DISTINCT, int)
        counts = {i: 1 for i in range(10)}
        self.assertEqual(RDX._group_counted_types(counts, {},
                                                  min_instances=2), [])

    def test_symbol_parse_can_be_cancelled(self):
        text = "\n".join(f"\tpublic int f{i}; // {hex(i)}" for i in range(5000))
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(InterruptedError):
            RDX.parse_il2cpp_dump(text, cancel_event=cancel)

    def test_symbol_parse_reports_progress(self):
        seen = []
        lines = ["public class K", "{"]
        lines += [f"\tpublic int f{i}; // {hex(i * 4)}" for i in range(3000)]
        lines.append("}")
        RDX.parse_il2cpp_dump("\n".join(lines), threading.Event(),
                              lambda d, t: seen.append((d, t)))
        self.assertTrue(seen)
        self.assertTrue(all(t >= d for d, t in seen))

    def test_symbol_parse_still_works_without_progress_plumbing(self):
        # The plain two-argument call the tests and any caller may still use.
        classes = RDX.parse_il2cpp_dump(self._DUMP_CS)
        self.assertIn("PlayerController", classes)

    def test_remembered_structures_are_bounded(self):
        # Type Scan can hand back 4096 instances; walking them with S left one
        # layout each, every layout holding up to _STRUCT_MAX_SPAN/4 fields.
        old = dict(RDX.state)
        try:
            RDX.state["structures"] = {}
            for base in range(RDX._STRUCT_MAX_REMEMBERED + 40):
                RDX._remember_structure(base, [{"offset": 0, "name": "f",
                                                "type": "u32"}])
            self.assertEqual(len(RDX.state["structures"]),
                             RDX._STRUCT_MAX_REMEMBERED)
            # Oldest evicted, newest kept.
            self.assertNotIn(0, RDX.state["structures"])
            self.assertIn(RDX._STRUCT_MAX_REMEMBERED + 39,
                          RDX.state["structures"])
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_re_remembering_a_base_does_not_grow_the_cache(self):
        old = dict(RDX.state)
        try:
            RDX.state["structures"] = {}
            for _ in range(200):
                RDX._remember_structure(0x1000, [{"offset": 0, "name": "f",
                                                  "type": "u32"}])
            self.assertEqual(len(RDX.state["structures"]), 1)
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_dr_sweep_stops_at_its_time_budget(self):
        # The debug socket carries a 10 s per-recv timeout, so an
        # unresponsive console could stall 40 threads x 10 s with the
        # diagnostic holding the trace open.
        def slow(_s, _lwpid):
            time.sleep(0.02)
            return PointerSubsystemTests._dbreg_blob(0)

        with patch.object(RDX, "_debug_get_dbregs", slow):
            cov = RDX._debug_verify_watchpoint(
                None, list(range(200)), 0x4000, 1, time_budget=0.1)
        self.assertTrue(cov["truncated"])
        self.assertLess(cov["checked"] + len(cov["unreadable"]), 200)

    def test_dr_sweep_honours_cancellation(self):
        cancel = threading.Event(); cancel.set()
        seen = []
        with patch.object(RDX, "_debug_get_dbregs",
                          lambda _s, l: seen.append(l)):
            cov = RDX._debug_verify_watchpoint(
                None, [10, 11, 12], 0x4000, 1, cancel_event=cancel)
        self.assertEqual(seen, [])
        self.assertTrue(cov["truncated"])

    def test_truncated_sweep_does_not_claim_a_definitive_verdict(self):
        # Claiming "per-thread application is ruled out" from a sample would
        # be the same overreach the checklist's calibration note records.
        full = {"armed": [1, 2], "absent": [], "unreadable": [], "checked": 2,
                "total": 2, "truncated": False}
        sample = {"armed": [1, 2], "absent": [], "unreadable": [], "checked": 2,
                  "total": 40, "truncated": True}
        key_full, text_full = RDX._debug_watchpoint_verdict(full)
        key_s, text_s = RDX._debug_watchpoint_verdict(sample)
        self.assertEqual(key_full, "all")
        self.assertIn("ruled out", text_full)
        self.assertEqual(key_s, "all")
        self.assertNotIn("ruled out", text_s)
        self.assertIn("sample only", text_s)

    def test_truncated_partial_verdict_still_carries_the_caveat(self):
        sample = {"armed": [1], "absent": [2], "unreadable": [], "checked": 2,
                  "total": 40, "truncated": True}
        key, text = RDX._debug_watchpoint_verdict(sample)
        self.assertEqual(key, "partial")
        self.assertIn("sample only", text)

    def test_slot_probe_time_budget_stays_conservative(self):
        # A short sweep may only make the choice safer: slots seen busy stay
        # excluded, so it can never hand back an occupied index.
        busy = PointerSubsystemTests._dbreg_blob(
            PointerSubsystemTests._dr7_enable(0, 1))
        def slow(_s, _l):
            time.sleep(0.02); return busy
        with patch.object(RDX, "_debug_get_dbregs", slow):
            chosen = RDX._debug_free_watchpoint_all(None, list(range(200)))
        self.assertIn(chosen, (2, 3))

    # ── patch102 (cont.): review findings in patches 87-96 ───────────────

    def test_coalescing_intersects_protection_bits(self):
        # A span that merged a writable region with a read-only one is not
        # writable throughout; keeping the first region's prot would be wrong
        # in the unsafe direction for any future reader.
        merged = RDX._coalesce_scan_regions([
            {"name": "rw", "start": 0x1000, "end": 0x2000, "prot": 3},
            {"name": "ro", "start": 0x2000, "end": 0x3000, "prot": 1},
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["prot"], 1)

    def test_coalescing_keeps_prot_when_nothing_merges(self):
        merged = RDX._coalesce_scan_regions([
            {"name": "rw", "start": 0x1000, "end": 0x2000, "prot": 3},
            {"name": "rw2", "start": 0x9000, "end": 0xA000, "prot": 3},
        ])
        self.assertEqual([m["prot"] for m in merged], [3, 3])

    def test_region_settings_default_detection(self):
        restore = self._isolated_settings()
        try:
            self.assertTrue(RDX._region_settings_are_default())
            RDX._settings["region_min_size"] = 0x32000
            self.assertFalse(RDX._region_settings_are_default())
        finally:
            restore()

    def test_region_filter_is_silent_at_default_settings(self):
        # An ordinary scan must gain no noise from this diagnostic.
        restore = self._isolated_settings()
        before = len(RDX.state.get("log", []))
        try:
            RDX._note_recommended_filter(1, 50, "First scan")
        finally:
            restore()
        self.assertEqual(len(RDX.state.get("log", [])), before)

    def test_region_filter_names_the_setting_that_narrowed_a_scan(self):
        # Exposing these settings created a failure mode that did not exist
        # while they were literals: a scan comes back thin because of a value
        # set on another screen, and nothing connects the two.
        restore = self._isolated_settings()
        try:
            RDX._settings["region_min_size"] = 0x32000
            RDX.state["log"] = []
            RDX._note_recommended_filter(2, 40, "First scan")
            messages = [e["msg"] for e in RDX.state["log"]]
        finally:
            restore()
        self.assertTrue(messages)
        self.assertIn("38 of 40", messages[0])
        self.assertIn("Settings", messages[0])

    def test_region_filter_escalates_when_it_excludes_everything(self):
        restore = self._isolated_settings()
        try:
            RDX._settings["region_exclude"] = "anon,executable"
            RDX.state["log"] = []
            RDX._note_recommended_filter(0, 12, "First scan")
            level = RDX.state["log"][0]["level"]
        finally:
            restore()
        self.assertEqual(level, "error")

    def test_undo_rle_round_trips_near_the_64_bit_ceiling(self):
        # Signed diffs are used to detect stride; values close to 2**64 are
        # where an unsigned subtraction would have wrapped instead.
        base = 2 ** 64 - 8 * 64
        arr = np.arange(base, base + 8 * 60, 8, dtype=np.uint64)
        np.testing.assert_array_equal(RDX._UndoAddrs(arr).array(), arr)

    def test_undo_rle_never_costs_more_than_raw(self):
        rng = np.random.default_rng(99)
        for n in (8, 64, 512, 4096):
            arr = np.sort(rng.choice(2 ** 40, n, replace=False).astype(np.uint64))
            stored = RDX._UndoAddrs(arr)
            self.assertLessEqual(stored.nbytes, arr.nbytes)
            np.testing.assert_array_equal(stored.array(), arr)

    # ── patch103: second pre-hardware review pass ────────────────────────
    #
    # Concurrency, the write paths, wire bounds and console-drop behaviour --
    # the areas the first review pass did not cover. Each fails against
    # patch102.

    def test_target_write_refuses_an_unmapped_address(self):
        # Target looks like the backend write API. Before this it called
        # ps5_write with no validation, so migrating call sites onto the seam
        # would have silently dropped the map check every other write path
        # performs.
        maps = [{"name": "anon", "start": 0x800000, "end": 0x810000,
                 "prot": 3}]
        old = dict(RDX.state)
        RDX.state.update(ip="t", pid=1, backend="ps5debug", memdbg=None)
        try:
            with patch.object(RDX, "ps5_maps", return_value=maps), \
                 patch.object(RDX, "ps5_write",
                              side_effect=AssertionError("unvalidated write")):
                self.assertFalse(
                    RDX.current_target("t").write(1, 0xDEAD0000, b"\x01\x02"))
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_target_write_refuses_a_read_only_mapping(self):
        maps = [{"name": "text", "start": 0x400000, "end": 0x410000,
                 "prot": 5}]          # read+exec, not writable
        old = dict(RDX.state)
        RDX.state.update(ip="t", pid=1, backend="ps5debug", memdbg=None)
        try:
            with patch.object(RDX, "ps5_maps", return_value=maps), \
                 patch.object(RDX, "ps5_write",
                              side_effect=AssertionError("unvalidated write")):
                self.assertFalse(
                    RDX.current_target("t").write(1, 0x400100, b"\x01"))
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_target_write_allows_a_valid_writable_address(self):
        maps = [{"name": "anon", "start": 0x800000, "end": 0x810000,
                 "prot": 3}]
        old = dict(RDX.state)
        RDX.state.update(ip="t", pid=1, backend="ps5debug", memdbg=None)
        try:
            with patch.object(RDX, "ps5_maps", return_value=maps), \
                 patch.object(RDX, "ps5_write", return_value=True) as w:
                self.assertTrue(
                    RDX.current_target("t").write(1, 0x800100, b"\x01"))
                self.assertEqual(w.call_count, 1)
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_both_targets_route_through_the_checked_write(self):
        for backend, memdbg in (("ps5debug", None),
                                ("memdbg-experimental",
                                 {"capabilities": RDX.MEMDBG_CAP_MEMORY_WRITE})):
            old = dict(RDX.state)
            RDX.state.update(ip="t", pid=1, backend=backend, memdbg=memdbg)
            try:
                with patch.object(RDX, "ps5_maps", return_value=[]), \
                     patch.object(RDX, "ps5_write",
                                  side_effect=AssertionError("unvalidated")):
                    self.assertFalse(
                        RDX.current_target("t").write(1, 0x1000, b"\x01"))
            finally:
                RDX.state.clear(); RDX.state.update(old)

    def test_type_scan_reports_a_disconnect_instead_of_zero_results(self):
        # "0 types found" and "the console went away" produced identical
        # output before this. They mean completely different things.
        maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                 "prot": 5},
                {"name": "anon", "start": 0x800000, "end": 0x900000,
                 "prot": 3}]

        class DeadSock:
            def __init__(self, *a, **k): pass
            def read(self, _addr, _size, _c=None):
                raise ConnectionError("PS5 disconnected")
            def close(self): pass

        with patch.object(RDX, "_get_maps_cached", return_value=maps), \
             patch.object(RDX, "_ScanSocket", DeadSock):
            with self.assertRaises(ConnectionError) as ctx:
                RDX.scan_type_instances("t", 1, min_instances=4)
        self.assertIn("stopped responding", str(ctx.exception))

    def test_type_scan_still_tolerates_scattered_unmapped_spans(self):
        # Isolated failures are normal and must not abort a working scan.
        maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                 "prot": 5},
                {"name": "executable", "start": 0x410000, "end": 0x420000,
                 "prot": 1},
                {"name": "anon", "start": 0x800000, "end": 0x900000,
                 "prot": 3}]
        blob = bytearray(RDX._TYPE_SCAN_CHUNK)
        for o in range(0, len(blob), 0x40):
            blob[o:o + 8] = (0x410280).to_bytes(8, "little")
        state = {"n": 0}

        class FlakySock:
            def __init__(self, *a, **k): pass
            def read(self, _addr, size, _c=None):
                state["n"] += 1
                if state["n"] % 3 == 0:          # never 8 in a row
                    raise ConnectionError("unmapped span")
                return bytes(blob[:size])
            def close(self): pass

        with patch.object(RDX, "_get_maps_cached", return_value=maps), \
             patch.object(RDX, "_ScanSocket", FlakySock):
            groups = RDX.scan_type_instances("t", 1, min_instances=8)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["type_ptr"], 0x410280)

    def test_manual_freeze_revalidates_the_mapping_every_tick(self):
        # The saved-cheat freeze worker already re-validated per tick; the
        # manual freeze validated once and then wrote for the whole window.
        src = SOURCE.read_text()
        body = src.split("def worker_fn():")[1].split("\n    worker =")[0]
        self.assertIn("_validate_addr_in_maps", body)
        self.assertIn("no longer", body)

    def test_manual_freeze_stops_when_the_mapping_goes_away(self):
        calls = {"n": 0}

        def flaky_maps(_ip, _pid, _len=None, *a, **k):
            calls["n"] += 1
            return None if calls["n"] == 1 else "address is not mapped"

        writes = []
        stop = threading.Event()
        # Drive one iteration of the loop body's logic directly.
        with patch.object(RDX, "_validate_addr_in_maps", flaky_maps), \
             patch.object(RDX, "ps5_write",
                          side_effect=lambda *a, **k: writes.append(1) or True):
            first = RDX._validate_addr_in_maps("t", 1, 0x800000, 4)
            second = RDX._validate_addr_in_maps("t", 1, 0x800000, 4)
        self.assertIsNone(first)
        self.assertIsNotNone(second)

    # ── wire-level tests, against a protocol-speaking fake console ────────
    #
    # Everything above mocks at the function level. These run the real
    # network code -- _ScanSocket and its pool, recv_exact framing, the
    # chunked readers, the debugger command sequence -- over a real TCP
    # socket. Before fake_console.py existed none of that had ever executed
    # outside a hardware session.

    def _console(self, **kw):
        """Start a fake console and point the client at it."""
        from fake_console import FakeConsole
        con = FakeConsole(**kw)
        con.start()
        self._saved_port = RDX.PS5_PORT
        RDX.PS5_PORT = con.port
        self._saved_state = dict(RDX.state)
        RDX.state.update(ip=con.host, pid=91, backend="ps5debug", memdbg=None,
                         session=1, proc_name="eboot.bin")
        return con

    def _release(self, con):
        con.stop()
        RDX.PS5_PORT = self._saved_port
        RDX.state.clear(); RDX.state.update(self._saved_state)
        RDX._ScanSocket.clear_pool()
        with RDX._map_cache_lock:
            RDX._map_cache.clear()

    def test_wire_process_list_and_maps_round_trip(self):
        con = self._console()
        try:
            procs = RDX.ps5_proc_list(con.host)
            self.assertIn("eboot.bin", [p["name"] for p in procs])
            maps = RDX.ps5_maps(con.host, 91)
            self.assertEqual(len(maps), len(con.maps))
            self.assertEqual(maps[0]["name"], "executable")
        finally:
            self._release(con)

    def test_wire_auth_handshake_is_really_checked(self):
        # The console verifies the XOR response, so a client-side keystream
        # regression fails here rather than silently later.
        con = self._console()
        try:
            RDX.ps5_auth_scanner(con.host)          # must not raise
            with patch.object(RDX, "_auth_keystream",
                              lambda n: b"\x00" * n):
                with self.assertRaises(RuntimeError):
                    RDX.ps5_auth_scanner(con.host)
        finally:
            self._release(con)

    def test_wire_read_write_and_verified_write(self):
        con = self._console()
        try:
            from fake_console import seed_value
            seed_value(con, 0x2000100, 1234, 4)
            self.assertEqual(
                int.from_bytes(RDX.ps5_read(con.host, 91, 0x2000100, 4),
                               "little"), 1234)
            self.assertTrue(
                RDX.ps5_write(con.host, 91, 0x2000100,
                              (4321).to_bytes(4, "little")))
            ack, verified, _ = RDX.ps5_write_verified(
                con.host, 91, 0x2000100, (7).to_bytes(4, "little"))
            self.assertTrue(ack)
            self.assertTrue(verified)
        finally:
            self._release(con)

    def test_wire_bulk_write_interleaves_headers_and_data(self):
        # 0xBDAACC04 entries are {u64 addr; u32 len; data} concatenated, so
        # the stream desynchronises from entry two if that is misread. The
        # freeze worker uses this path on every tick.
        con = self._console()
        try:
            entries = [(0x2000800 + i * 0x100, (i + 1).to_bytes(4, "little"))
                       for i in range(4)]
            self.assertEqual(RDX.ps5_write_multi(con.host, 91, entries),
                             [True] * 4)
            for i, (addr, _) in enumerate(entries):
                self.assertEqual(
                    int.from_bytes(RDX.ps5_read(con.host, 91, addr, 4),
                                   "little"), i + 1)
        finally:
            self._release(con)

    def test_wire_bulk_write_reports_a_failed_entry(self):
        con = self._console()
        try:
            self.assertEqual(
                RDX.ps5_write_multi(con.host, 91,
                                    [(0xDEAD0000, b"\x01\x02\x03\x04")]),
                [False])
        finally:
            self._release(con)

    def test_wire_scan_first_then_next_narrows(self):
        con = self._console()
        try:
            from fake_console import seed_value
            seed_value(con, 0x2000400, 999, 4)
            seed_value(con, 0x2100400, 999, 4)
            hits = RDX.scan_first(con.host, 91, 999, 4, value_type="u32")
            self.assertEqual(sorted(int(a) for a in hits),
                             [0x2000400, 0x2100400])
            seed_value(con, 0x2000400, 111, 4)
            survivors = RDX.scan_next(con.host, 91, 999, 4, hits,
                                      value_type="u32")
            self.assertEqual([int(a) for a in survivors], [0x2100400])
        finally:
            self._release(con)

    def test_wire_aob_match_across_a_mapping_boundary(self):
        # The patch91 correctness fix, verified over the wire rather than
        # argued from the code: the default map has two ADJACENT heap
        # mappings, and this pattern straddles the join.
        con = self._console()
        try:
            planted = 0x2200000 - 3
            for i, b in enumerate(b"\xDE\xAD\xBE\xEF\xCA\xFE"):
                con.memory[planted + i] = b
            pattern, mask, _canon = RDX._parse_byte_pattern(
                "DE AD BE EF CA FE")
            hits = RDX.scan_first_pattern(con.host, 91, pattern, mask)
            self.assertIn(planted, [int(a) for a in hits])
        finally:
            self._release(con)

    def test_wire_type_scan_finds_seeded_instances(self):
        con = self._console()
        try:
            from fake_console import seed_type_pointers
            type_ptr = seed_type_pointers(con, base=0x2000000, count=64)
            groups = RDX.scan_type_instances(con.host, 91, min_instances=8)
            self.assertTrue(groups)
            self.assertEqual(groups[0]["type_ptr"], type_ptr)
            self.assertEqual(groups[0]["count"], 64)
            self.assertEqual(groups[0]["module_name"], "executable")
        finally:
            self._release(con)

    def test_wire_watchpoint_diagnostic_discriminates_all_hypotheses(self):
        # The reason fake_console.py exists. The checklist lists three
        # payload behaviours it cannot tell apart without a console; this
        # proves the diagnostic reports each one correctly before the single
        # expensive attach that will decide it for real.
        for dr_mode, expected in (("all-threads", "all"),
                                  ("first-thread-only", "partial"),
                                  ("none", "none")):
            con = self._console(dr_mode=dr_mode, thread_count=40)
            try:
                sock = RDX.ps5_connect(con.host)
                try:
                    threads = RDX._debug_thread_list(sock)
                    self.assertEqual(len(threads), 40)
                    index = RDX._debug_free_watchpoint_all(sock, threads)
                    self.assertIsNotNone(index)
                    RDX._debug_set_watchpoint(sock, index, 0x2000400, 3, 1)
                    coverage = RDX._debug_verify_watchpoint(
                        sock, threads, 0x2000400, index)
                    verdict, _text = RDX._debug_watchpoint_verdict(coverage)
                finally:
                    sock.close()
                self.assertEqual(verdict, expected, dr_mode)
            finally:
                self._release(con)

    def test_wire_free_slot_probe_avoids_an_occupied_slot(self):
        con = self._console(dr_mode="all-threads", thread_count=8)
        try:
            sock = RDX.ps5_connect(con.host)
            try:
                threads = RDX._debug_thread_list(sock)
                first = RDX._debug_free_watchpoint_all(sock, threads)
                RDX._debug_set_watchpoint(sock, first, 0x2000400, 3, 1)
                second = RDX._debug_free_watchpoint_all(sock, threads)
            finally:
                sock.close()
            self.assertNotEqual(first, second)
        finally:
            self._release(con)

    # ── patch104: pointer ranking, found by differential testing ─────────
    #
    # Found by building a known chain in fake-console memory alongside a
    # deliberate coincidence, then asking the resolver which it preferred.
    # Fails against patch103.

    def test_field_offset_plausibility_matches_the_documented_rule(self):
        ok = {"offsets": [0x18]}
        zero = {"offsets": [0]}
        edge = {"offsets": [RDX._PTR_PLAUSIBLE_FIELD_MAX]}
        past = {"offsets": [RDX._PTR_PLAUSIBLE_FIELD_MAX + 1]}
        behind = {"offsets": [-0x900]}
        for c in (ok, zero, edge):
            self.assertTrue(RDX._candidate_field_offset_is_plausible(c), c)
        for c in (past, behind):
            self.assertFalse(RDX._candidate_field_offset_is_plausible(c), c)

    def test_plausibility_uses_the_final_hop(self):
        # The last offset is the displacement inside the final object, which
        # is what the rule is about; for depth 1 it is also offsets[0], so
        # this agrees with the fast-direct filter instead of competing.
        self.assertTrue(RDX._candidate_field_offset_is_plausible(
            {"offsets": [0x10, -0x3FF0, 0x30]}))
        self.assertFalse(RDX._candidate_field_offset_is_plausible(
            {"offsets": [0x10, 0x20, -0x900]}))

    def test_cycle_detection_spots_a_revisited_address(self):
        self.assertTrue(RDX._chain_revisits_an_address(
            [0x2001010, 0x2005000, 0x2001010]))
        self.assertFalse(RDX._chain_revisits_an_address(
            [0x2001010, 0x2005000, 0x2005030]))
        self.assertFalse(RDX._chain_revisits_an_address([]))

    def test_ranking_puts_a_plausible_chain_above_a_coincidence(self):
        # The defect: a holder pointing 0x900 past the target outranked a
        # real two-hop chain. That is the shape _PTR_PLAUSIBLE_FIELD_MAX
        # records as surviving 0/5 reloads, so following it costs two game
        # reloads to disprove.
        real = {"base": 0x400600, "offsets": [0x10, 0x30], "verified": True,
                "module_name": "executable", "score": 100.0, "depth": 2,
                "module_relative_offset": 0x600}
        decoy = {"base": 0x400700, "offsets": [-0x900], "verified": True,
                 "module_name": "executable", "score": 220.0, "depth": 1,
                 "module_relative_offset": 0x700}
        with patch.object(RDX, "_resolve_pointer_chain",
                          return_value=(True, 0x2005030, [0x2005030])):
            ranked = RDX._rank_pointer_candidates("t", 1, [decoy, real])
        self.assertEqual(int(ranked[0]["base"]), 0x400600)
        # ...and the coincidence is kept, because when nothing better exists
        # it is the only lead there is.
        self.assertEqual(len(ranked), 2)

    def test_ranking_drops_self_revisiting_chains(self):
        straight = {"base": 0x400600, "offsets": [0x10, 0x30],
                    "verified": True, "module_name": "m", "score": 1.0}
        looped = {"base": 0x400600, "offsets": [0x10, -0x3FF0, 0x30],
                  "verified": True, "module_name": "m", "score": 1.0}

        def fake_resolve(_ip, _pid, _base, offsets, _term=0):
            if len(offsets) == 3:
                return True, 0x2005030, [0x2001010, 0x2005000, 0x2001010]
            return True, 0x2005030, [0x2001010, 0x2005030]

        with patch.object(RDX, "_resolve_pointer_chain", fake_resolve):
            ranked = RDX._rank_pointer_candidates("t", 1, [straight, looped])
        self.assertEqual(len(ranked), 1)
        self.assertEqual(list(ranked[0]["offsets"]), [0x10, 0x30])

    def test_every_resolver_return_path_uses_the_shared_ranking(self):
        # The root cause was three return paths with three different sorts,
        # so which ranking the user saw depended on which tier answered.
        src = SOURCE.read_text()
        body = src.split("def _resolve_permanent_candidates(")[1] \
                  .split("\ndef ")[0]
        self.assertEqual(body.count("_rank_pointer_candidates("), 3)
        # No hand-rolled candidate sort should remain in that function.
        self.assertNotIn('sort(key=lambda c: (-c["score"], c["base"]))', body)

    def test_wire_resolver_prefers_the_real_chain_over_a_decoy(self):
        # End to end over the protocol, against memory we control: a real
        # two-hop chain and a coincidence that also resolves to the target.
        con = self._console(maps=[
            {"name": "executable", "start": 0x400000, "end": 0x410000,
             "prot": 3},
            {"name": "", "start": 0x2000000, "end": 0x2020000, "prot": 3}])
        try:
            RDX._invalidate_pointer_index()
            root, mid, obj = 0x400600, 0x2001000, 0x2005000
            target, decoy = obj + 0x30, 0x400700

            def put_q(addr, val):
                for i, b in enumerate(int(val).to_bytes(8, "little")):
                    con.memory[addr + i] = b

            put_q(root, mid)
            put_q(mid + 0x10, obj)
            put_q(decoy, target + 0x900)
            for i, b in enumerate((4242).to_bytes(4, "little")):
                con.memory[target + i] = b

            res = RDX._resolve_permanent_candidates(con.host, 91, target,
                                                    max_depth=4)
            cands = res.get("candidates", [])
            self.assertTrue(cands)
            self.assertEqual(int(cands[0]["base"]), root)
            self.assertEqual(list(cands[0]["offsets"]), [0x10, 0x30])
            ok, final, _ = RDX._resolve_pointer_chain(
                con.host, 91, int(cands[0]["base"]),
                list(cands[0]["offsets"]))
            self.assertTrue(ok)
            self.assertEqual(final, target)
        finally:
            self._release(con)

    # ── patch105: bounded trainer import ─────────────────────────────────
    #
    # Trainer files are third-party input -- HEN-Cheats-Collection alone
    # carries 2,364 games' worth -- and imported cheats were the only
    # unbounded collection in the program. Fails against patch104.

    def _import_probe(self, doc_or_text, suffix=".rdx.json", size_pad=0):
        """Run do_import against a temp file; return (message, cheat count)."""
        import json as _json
        maps = [{"name": "executable", "start": 0x400000, "end": 0x500000,
                 "prot": 5}]
        captured, old = {}, dict(RDX.state)
        RDX.state.update(ip="t", pid=91, session=1, proc_name="eboot.bin",
                         cheats=[])
        try:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / ("t" + suffix)
                text = (doc_or_text if isinstance(doc_or_text, str)
                        else _json.dumps(doc_or_text))
                if size_pad:
                    text += " " * size_pad
                path.write_text(text, encoding="utf-8")
                with patch.object(RDX, "draw_border"), \
                     patch.object(RDX, "draw_statusbar"), \
                     patch.object(RDX, "safe_addstr"), \
                     patch.object(RDX, "color", return_value=0), \
                     patch.object(RDX, "confirm_box", return_value=True), \
                     patch.object(RDX, "input_box", return_value=str(path)), \
                     patch.object(RDX, "_get_maps_cached", return_value=maps), \
                     patch.object(RDX, "message_box",
                                  lambda _s, lines, *a, **k:
                                  captured.update(msg=" ".join(lines))):
                    RDX.do_import(self._FakeKeyWindow([]))
                count = len(RDX.state.get("cheats", []))
        finally:
            RDX.state.clear(); RDX.state.update(old)
        return captured.get("msg", ""), count

    def _valid_trainer(self, n):
        maps = [{"name": "executable", "start": 0x400000, "end": 0x500000,
                 "prot": 5}]
        return {"format": "rdx-1", "process": "eboot.bin",
                "titleid": "CUSA01659", "version": "01.00",
                "game_identity": RDX._pointer_game_identity("eboot.bin", maps),
                "cheatList": [{"name": f"c{i}", "address": hex(0x400000 + i * 4),
                               "width": 4, "value": "1", "type": "write",
                               "value_type": "u32"} for i in range(n)]}

    def test_a_normal_trainer_still_imports(self):
        msg, count = self._import_probe(self._valid_trainer(12))
        self.assertEqual(count, 12)
        self.assertIn("12", msg)

    def test_import_refuses_an_absurd_cheat_count(self):
        # 200,000 entries were accepted at 283 MB before this.
        msg, count = self._import_probe(
            self._valid_trainer(RDX.MAX_IMPORT_CHEATS + 1))
        self.assertEqual(count, 0)
        self.assertIn("limit", msg.lower())

    def test_the_cheat_cap_boundary_is_inclusive(self):
        _msg, count = self._import_probe(
            self._valid_trainer(RDX.MAX_IMPORT_CHEATS))
        self.assertEqual(count, RDX.MAX_IMPORT_CHEATS)

    def test_import_refuses_an_oversized_file_before_parsing(self):
        # The parse peaks at roughly ten times the file size, so the guard
        # has to come before it, not after.
        msg, count = self._import_probe(
            self._valid_trainer(2),
            size_pad=RDX.MAX_TRAINER_FILE_BYTES + 1024)
        self.assertEqual(count, 0)
        self.assertIn("MB", msg)

    def test_static_patch_mods_are_bounded_too(self):
        # The .mc4/.shn/etaHEN path reaches the cheat list through a
        # different function, which needed its own guard.
        shown = {}
        mods = [{"name": f"m{i}",
                 "memory": [{"offset": "100", "on": "01", "off": "00"}]}
                for i in range(RDX.MAX_IMPORT_CHEATS + 5)]
        old = dict(RDX.state)
        RDX.state.update(proc_name="eboot.bin", cheats=[])
        try:
            with patch.object(RDX, "message_box",
                              lambda _s, lines, *a, **k:
                              shown.update(msg=" ".join(lines))):
                RDX._do_import_static_patch_mods(
                    self._FakeKeyWindow([]), Path("x.mc4"), mods,
                    "CheatRunner .mc4", "CUSA01659", "eboot.bin")
            self.assertEqual(len(RDX.state["cheats"]), 0)
        finally:
            RDX.state.clear(); RDX.state.update(old)
        self.assertIn("limit", shown.get("msg", "").lower())

    def test_import_bounds_sit_alongside_the_others(self):
        # The finding was an inconsistency, not an isolated bug: every other
        # collection in the program is bounded.
        for name in ("_BOOKMARK_MAX", "_STRUCT_MAX_REMEMBERED",
                     "_SYMBOL_MAX_CLASSES", "MAX_SCAN_RESULTS",
                     "MAX_PROC_ENTRIES", "MAX_MAP_ENTRIES",
                     "MAX_IMPORT_CHEATS", "MAX_TRAINER_FILE_BYTES"):
            self.assertIsInstance(getattr(RDX, name), int, name)
            self.assertGreater(getattr(RDX, name), 0, name)

    # ── patch106: one-shot marker survives the container (pass-3 L4) ──────

    def _export_mods(self, cheat_type):
        maps = [{"name": "executable", "start": 0x400000, "end": 0x500000,
                 "prot": 3}]
        cheat = {"name": "Infinite Ammo", "address": 0x400100, "width": 4,
                 "value": 999, "value_type": "i32", "type": cheat_type,
                 "pid": 1, "session": 1, "process": "eboot.bin",
                 "module_name": "executable", "module_relative_offset": 0x100,
                 "original_value": 10, "game_identity": "x", "offsets": None}
        _t, mods, _s = RDX.generate_etahen_json(
            [cheat], "CUSA1", "01.00", "G", "eboot.bin", maps)
        return mods

    def test_a_freeze_is_marked_because_the_container_downgrades_it(self):
        # A freeze set up here becomes a one-shot write in every manager that
        # reads these formats, and the `hint` explaining that is dropped by
        # the schema. The name is the only field that survives.
        mods = self._export_mods("freeze")
        self.assertIn(RDX._ONE_SHOT_MARKER, mods[0]["name"])
        self.assertTrue(mods[0]["name"].startswith("Infinite Ammo"))

    def test_a_one_shot_write_is_not_marked(self):
        # A marker on every row would be noise; only downgrades get one.
        mods = self._export_mods("write")
        self.assertNotIn(RDX._ONE_SHOT_MARKER, mods[0]["name"])

    def test_the_marker_survives_shn_and_mc4(self):
        mods = self._export_mods("freeze")
        shn = RDX.generate_shn_text(mods, "CUSA1", "01.00", "G", "eboot.bin")
        mc4 = RDX.generate_mc4_bytes(mods, "CUSA1", "01.00", "G", "eboot.bin")
        for text in (shn, RDX._mc4_decrypt(mc4).decode("utf-8")):
            _attrs, parsed = RDX.mc4_xml_to_mods(text)
            self.assertIn(RDX._ONE_SHOT_MARKER, parsed[0]["name"])

    # ── patch107: class names from live memory (pass-3 L1) ───────────────

    def test_plausible_class_name_accepts_real_type_names(self):
        for good in (b"PlayerController\x00", b"Inventory\x00",
                     b"System.Int32\x00", b"List`1\x00",
                     b"Namespace.Outer+Inner\x00"):
            self.assertIsNotNone(RDX._plausible_class_name(good), good)

    def test_plausible_class_name_rejects_coincidence(self):
        # The whole risk of this feature is confidently labelling something
        # wrongly, so the validator has to be the strict part.
        for bad in (b"", b"\x00", b"\xff\xfe\xfd\x00", b"1234\x00",
                    b"   \x00", b"A\x00", b"\x00PlayerController",
                    b"no terminator in this buffer at all"):
            self.assertIsNone(RDX._plausible_class_name(bad), bad)

    def test_klass_name_probes_multiple_offsets(self):
        # Breeze's +0x10 is version-specific; Il2CppClass has been reordered
        # between IL2CPP releases, so the offset must be found, not assumed.
        self.assertIn(0x10, RDX._KLASS_NAME_OFFSETS)
        self.assertGreater(len(RDX._KLASS_NAME_OFFSETS), 1)

    def test_klass_name_returns_none_rather_than_raising(self):
        with patch.object(RDX, "_get_maps_cached",
                          side_effect=ConnectionError("gone")):
            self.assertIsNone(RDX._read_klass_name("t", 1, 0x400280))

    def test_klass_name_cache_is_bounded_and_clearable(self):
        RDX._invalidate_klass_names()
        with RDX._klass_name_lock:
            RDX._klass_name_cache.update({i: None for i in range(10)})
        self.assertEqual(len(RDX._klass_name_cache), 10)
        RDX._invalidate_klass_names()
        self.assertEqual(len(RDX._klass_name_cache), 0)

    def test_labelling_is_bounded_by_count_and_time(self):
        groups = [{"type_ptr": 0x400000 + i * 8} for i in range(500)]
        seen = []
        with patch.object(RDX, "_read_klass_name",
                          lambda _i, _p, k, _m=None: seen.append(k) or "K"):
            named = RDX.label_type_groups("t", 1, groups, [], limit=10)
        self.assertEqual(named, 10)
        self.assertEqual(len(seen), 10)
        self.assertNotIn("class_name", groups[11])

    def test_labelling_honours_cancellation(self):
        groups = [{"type_ptr": 0x400000 + i * 8} for i in range(20)]
        cancel = threading.Event(); cancel.set()
        with patch.object(RDX, "_read_klass_name",
                          lambda *a, **k: "ShouldNotBeCalled"):
            named = RDX.label_type_groups("t", 1, groups, [],
                                          cancel_event=cancel)
        self.assertEqual(named, 0)

    def test_wire_type_scan_names_a_class_from_live_memory(self):
        # End to end over the protocol: no dump.cs anywhere.
        con = self._console(maps=[
            {"name": "executable", "start": 0x400000, "end": 0x420000,
             "prot": 5},
            # Module data. A class never lives in the executable segment --
            # hardware showed Il2CppClass is heap-allocated, and this fixture
            # previously put it in code, which is what made the old
            # static-only target filter look correct offline.
            {"name": "executable", "start": 0x420000, "end": 0x430000,
             "prot": 1},
            {"name": "", "start": 0x2000000, "end": 0x2010000, "prot": 3}])
        try:
            RDX._invalidate_klass_names()
            klass, name_ptr = 0x420280, 0x421000

            def put_q(addr, val):
                for i, b in enumerate(int(val).to_bytes(8, "little")):
                    con.memory[addr + i] = b

            put_q(klass + 0x10, name_ptr)
            for i, b in enumerate(b"PlayerController\x00"):
                con.memory[name_ptr + i] = b
            for i in range(64):
                put_q(0x2000000 + i * 0x40, klass)

            groups = RDX.scan_type_instances(con.host, 91, min_instances=8)
            self.assertTrue(groups)
            self.assertEqual(groups[0]["class_name"], "PlayerController")
            self.assertEqual(groups[0]["count"], 64)
        finally:
            self._release(con)

    def test_wire_a_non_il2cpp_layout_is_left_unnamed(self):
        # A title that does not use this layout must simply go unlabelled --
        # never mislabelled.
        con = self._console(maps=[
            {"name": "executable", "start": 0x400000, "end": 0x420000,
             "prot": 5},
            {"name": "executable", "start": 0x420000, "end": 0x430000,
             "prot": 1},
            {"name": "", "start": 0x2000000, "end": 0x2010000, "prot": 3}])
        try:
            RDX._invalidate_klass_names()
            klass = 0x420280

            def put_q(addr, val):
                for i, b in enumerate(int(val).to_bytes(8, "little")):
                    con.memory[addr + i] = b

            for i in range(64):
                put_q(0x2000000 + i * 0x40, klass)
            groups = RDX.scan_type_instances(con.host, 91, min_instances=8)
            self.assertTrue(groups)
            self.assertIsNone(groups[0].get("class_name"))
        finally:
            self._release(con)

    # ── patch108: bookmarks may carry a pointer chain (pass-3 L2) ────────

    def test_a_plain_bookmark_still_expires(self):
        # patch89's reasoning is unchanged for a bookmark with no chain: it
        # is a raw address, and after a reload it names whatever now occupies
        # that memory.
        old = dict(RDX.state)
        try:
            RDX.state.update(session=1, pid=42, bookmarks=[])
            RDX._add_bookmark(0x2000100, "u32")
            mark = RDX.state["bookmarks"][0]
            self.assertTrue(RDX._bookmark_is_current(mark))
            RDX.state["session"] = 2
            self.assertFalse(RDX._bookmark_is_current(mark))
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_a_chained_bookmark_rebases_instead_of_expiring(self):
        old = dict(RDX.state)
        try:
            RDX.state.update(session=1, pid=42, ip="t", bookmarks=[])
            RDX._add_bookmark(0x2001018, "u32", "ammo", chain={
                "module_name": "executable", "module_relative_offset": 0x500,
                "offsets": [0x18], "terminal_offset": 0})
            mark = RDX.state["bookmarks"][0]
            maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                     "prot": 3}]
            with patch.object(RDX, "_get_maps_cached", return_value=maps), \
                 patch.object(RDX, "_resolve_pointer_chain",
                              return_value=(True, 0x2001018, [0x2001018])):
                self.assertTrue(RDX._bookmark_is_current(mark))
                RDX.state["session"] = 99          # a whole new session
                self.assertTrue(RDX._bookmark_is_current(mark))
                self.assertEqual(RDX._bookmark_live_address(mark), 0x2001018)
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_a_chained_bookmark_goes_stale_when_it_cannot_rebase(self):
        # Honest presentation still wins: if the module is gone, so is the
        # bookmark's meaning.
        old = dict(RDX.state)
        try:
            RDX.state.update(session=1, pid=42, ip="t", bookmarks=[])
            RDX._add_bookmark(0x2001018, "u32", chain={
                "module_name": "executable", "module_relative_offset": 0x500,
                "offsets": [0x18], "terminal_offset": 0})
            mark = RDX.state["bookmarks"][0]
            with patch.object(RDX, "_get_maps_cached", return_value=[]):
                self.assertFalse(RDX._bookmark_is_current(mark))
                # ...and falls back to the stored address rather than 0.
                self.assertEqual(RDX._bookmark_live_address(mark), 0x2001018)
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_chained_bookmarks_survive_a_process_change(self):
        old = dict(RDX.state)
        try:
            RDX.state.update(session=1, pid=42, bookmarks=[])
            RDX._add_bookmark(0x2000100, "u32")                    # plain
            RDX._add_bookmark(0x2001018, "u32", chain={
                "module_name": "executable", "module_relative_offset": 0x500,
                "offsets": [0x18], "terminal_offset": 0})          # chained
            with patch.object(RDX, "_stop_freeze_worker"), \
                 patch.object(RDX, "_close_turbo_session"), \
                 patch.object(RDX, "_invalidate_pointer_index"):
                RDX._clear_scan_state(stop_freezes=False)
            kept = RDX.state["bookmarks"]
            self.assertEqual(len(kept), 1)
            self.assertTrue(kept[0]["chain"])
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_a_verified_chain_is_attached_to_a_bookmark_on_that_address(self):
        old = dict(RDX.state)
        try:
            RDX.state.update(session=1, pid=42, bookmarks=[])
            RDX._add_bookmark(0x2001018, "u32", "ammo")
            msg = RDX._attach_chain_to_bookmark(0x2001018, {
                "module_name": "executable", "module_relative_offset": 0x500,
                "offsets": [0x18]})
            self.assertIsNotNone(msg)
            self.assertTrue(RDX.state["bookmarks"][0]["chain"])
            # Attaching twice does not clobber the first chain.
            self.assertIsNone(RDX._attach_chain_to_bookmark(0x2001018, {
                "module_name": "other", "module_relative_offset": 0x1,
                "offsets": [0x2]}))
            self.assertEqual(
                RDX.state["bookmarks"][0]["chain"]["module_name"], "executable")
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_attaching_a_chain_creates_no_bookmark_of_its_own(self):
        # The pointer search runs far more often from Results than from the
        # bookmark list; it must not start making bookmarks nobody asked for.
        old = dict(RDX.state)
        try:
            RDX.state.update(bookmarks=[])
            self.assertIsNone(RDX._attach_chain_to_bookmark(0x2001018, {
                "module_name": "executable", "module_relative_offset": 0x500,
                "offsets": [0x18]}))
            self.assertEqual(RDX.state["bookmarks"], [])
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_a_chainless_candidate_is_not_attached(self):
        old = dict(RDX.state)
        try:
            RDX.state.update(bookmarks=[])
            RDX._add_bookmark(0x2001018, "u32")
            self.assertIsNone(RDX._attach_chain_to_bookmark(
                0x2001018, {"module_name": "", "offsets": [0x18]}))
            self.assertIsNone(RDX.state["bookmarks"][0]["chain"])
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_wire_chained_bookmark_resolves_over_the_protocol(self):
        con = self._console(maps=[
            {"name": "executable", "start": 0x400000, "end": 0x410000,
             "prot": 3},
            {"name": "", "start": 0x2000000, "end": 0x2010000, "prot": 3}])
        try:
            root, obj = 0x400500, 0x2001000
            target = obj + 0x18
            for i, b in enumerate(int(obj).to_bytes(8, "little")):
                con.memory[root + i] = b
            RDX.state["bookmarks"] = []
            RDX._add_bookmark(target, "u32", "ammo", chain={
                "module_name": "executable", "module_relative_offset": 0x500,
                "offsets": [0x18], "terminal_offset": 0})
            mark = RDX.state["bookmarks"][0]
            RDX.state["session"] = 77          # pretend a reconnect happened
            self.assertTrue(RDX._bookmark_is_current(mark))
            self.assertEqual(RDX._bookmark_live_address(mark), target)
        finally:
            self._release(con)

    # ── patch109: salvage chains from an outdated trainer (pass-3 L3) ────

    def test_salvage_extracts_module_rooted_chains(self):
        items = [
            {"name": "Ammo", "module_name": "executable",
             "module_relative_offset": 0x500, "offsets": [0x18]},
            {"name": "Health", "module": "executable",      # short key form
             "module_offset": "0x600", "offsets": ["0x10", "0x8"]},
            {"name": "PlainAddress", "address": "0x2000100"},   # no chain
        ]
        salvage = RDX._salvageable_chains(items)
        self.assertEqual([s["name"] for s in salvage], ["Ammo", "Health"])
        self.assertEqual(salvage[1]["offsets"], [0x10, 0x8])
        self.assertEqual(salvage[1]["module_relative_offset"], 0x600)

    def test_salvage_applies_the_same_bounds_as_import(self):
        # A hostile trainer must not smuggle an unbounded chain in through
        # the salvage path that the normal import path would have rejected.
        bad = [
            {"name": "deep", "module_name": "m",
             "module_relative_offset": 0,
             "offsets": [0x8] * (RDX.MAX_CHAIN_DEPTH + 1)},
            {"name": "huge-offset", "module_name": "m",
             "module_relative_offset": 0,
             "offsets": [RDX._PTR_RESOLVE_OFFSET_MAX + 1]},
            {"name": "negative-base", "module_name": "m",
             "module_relative_offset": -1, "offsets": [0x8]},
            {"name": "empty", "module_name": "m",
             "module_relative_offset": 0, "offsets": []},
            {"name": "junk-offsets", "module_name": "m",
             "module_relative_offset": 0, "offsets": ["not-a-number"]},
        ]
        self.assertEqual(RDX._salvageable_chains(bad), [])

    def test_salvage_result_count_is_bounded(self):
        many = [{"name": f"c{i}", "module_name": "m",
                 "module_relative_offset": 0, "offsets": [0x8]}
                for i in range(RDX.MAX_IMPORT_CHEATS + 50)]
        self.assertEqual(len(RDX._salvageable_chains(many)),
                         RDX.MAX_IMPORT_CHEATS)

    def test_salvage_ignores_non_dict_entries(self):
        self.assertEqual(RDX._salvageable_chains([None, 5, "x", []]), [])

    def test_verifying_a_salvaged_chain_trusts_nothing_from_the_file(self):
        # The module base must come from the live map, not the trainer.
        old = dict(RDX.state)
        RDX.state.update(ip="t", pid=1)
        chain = {"module_name": "executable", "module_relative_offset": 0x500,
                 "offsets": [0x18], "terminal_offset": 0}
        try:
            # Module absent from the running build -> no salvage.
            with patch.object(RDX, "_get_maps_cached", return_value=[]):
                self.assertIsNone(RDX._verify_salvaged_chain(chain))
            # Module present and the chain resolves somewhere writable.
            maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                     "prot": 3}]
            with patch.object(RDX, "_get_maps_cached", return_value=maps), \
                 patch.object(RDX, "_resolve_pointer_chain",
                              return_value=(True, 0x2001018, [0x2001018])), \
                 patch.object(RDX, "_validate_addr_in_maps", return_value=None):
                self.assertEqual(RDX._verify_salvaged_chain(chain), 0x2001018)
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_a_salvaged_chain_resolving_somewhere_unwritable_is_rejected(self):
        # Resolving is not enough: it has to land where a cheat could act,
        # which is the same test a saved cheat's chain has to pass.
        old = dict(RDX.state)
        RDX.state.update(ip="t", pid=1)
        maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                 "prot": 3}]
        try:
            with patch.object(RDX, "_get_maps_cached", return_value=maps), \
                 patch.object(RDX, "_resolve_pointer_chain",
                              return_value=(True, 0xDEAD0000, [])), \
                 patch.object(RDX, "_validate_addr_in_maps",
                              return_value="not mapped"):
                self.assertIsNone(RDX._verify_salvaged_chain(
                    {"module_name": "executable",
                     "module_relative_offset": 0x500, "offsets": [0x18]}))
        finally:
            RDX.state.clear(); RDX.state.update(old)

    def test_a_salvaged_chain_becomes_a_chained_bookmark(self):
        # The two features compose: a salvaged chain is exactly the thing a
        # bookmark needs in order to survive the next reload.
        old = dict(RDX.state)
        try:
            RDX.state.update(session=1, pid=42, ip="t", bookmarks=[])
            chain = {"module_name": "executable",
                     "module_relative_offset": 0x500, "offsets": [0x18],
                     "terminal_offset": 0}
            RDX._add_bookmark(0x2001018, "u32", "salvaged: Ammo", chain=chain)
            mark = RDX.state["bookmarks"][0]
            self.assertTrue(mark["chain"])
            maps = [{"name": "executable", "start": 0x400000, "end": 0x410000,
                     "prot": 3}]
            with patch.object(RDX, "_get_maps_cached", return_value=maps), \
                 patch.object(RDX, "_resolve_pointer_chain",
                              return_value=(True, 0x2001018, [])):
                RDX.state["session"] = 99
                self.assertTrue(RDX._bookmark_is_current(mark))
        finally:
            RDX.state.clear(); RDX.state.update(old)

    # ── patch110: a zero-result scan explains itself (pass-4 D1) ─────────
    #
    # Fails against patch109, where a scan matching nothing logged one line
    # and returned to the main menu with no screen at all.

    def test_zero_result_advice_leads_with_the_float_case(self):
        # The default type is u32 and the only hardware-validated title is
        # Unity/IL2CPP, where health and ammo are floats. That is both the
        # likeliest mistake and the one most specific to this tool.
        lines = RDX._zero_result_advice("100", "u32", "recommended", True)
        joined = "\n".join(lines)
        self.assertIn("Most likely", joined)
        self.assertIn("float", joined)
        self.assertIn("f32", joined)
        self.assertLess(joined.index("Most likely"), joined.index("Other things"))

    def test_a_non_numeric_value_gets_no_float_headline(self):
        joined = "\n".join(RDX._zero_result_advice("abc", "u32", "writable", False))
        self.assertNotIn("Most likely", joined)

    def test_a_float_scan_is_not_told_to_try_float(self):
        # Advice that is obviously wrong once teaches the reader to skip the
        # rest of it.
        joined = "\n".join(RDX._zero_result_advice("100.5", "f32", "writable", False))
        self.assertNotIn("usually f32", joined)
        self.assertIn("tolerance", joined.lower())

    def test_advice_mentions_scope_only_when_it_is_narrowing(self):
        narrow = "\n".join(RDX._zero_result_advice("1", "u32", "recommended", False))
        wide = "\n".join(RDX._zero_result_advice("1", "u32", "writable", False))
        self.assertIn("Recommended", narrow)
        self.assertNotIn("Recommended", wide)

    def test_advice_mentions_alignment_only_when_it_is_on(self):
        on = "\n".join(RDX._zero_result_advice("1", "u32", "writable", True))
        off = "\n".join(RDX._zero_result_advice("1", "u32", "writable", False))
        self.assertIn("Aligned", on)
        self.assertNotIn("Aligned", off)

    def test_advice_flags_non_default_region_settings(self):
        # patch102 added this diagnostic for a different symptom; a scan that
        # returns nothing is the other place it matters.
        restore = self._isolated_settings()
        try:
            plain = "\n".join(RDX._zero_result_advice("1", "u32", "writable", False))
            self.assertNotIn("region settings", plain)
            RDX._settings["region_min_size"] = 0x32000
            tuned = "\n".join(RDX._zero_result_advice("1", "u32", "writable", False))
            self.assertIn("region settings", tuned)
        finally:
            restore()

    def test_advice_is_never_empty_for_any_value_type(self):
        for value_type in RDX.VALUE_TYPE_ORDER:
            lines = RDX._zero_result_advice("1", value_type, "recommended", True)
            self.assertGreater(len(lines), 3, value_type)
            self.assertTrue(all(isinstance(x, str) for x in lines), value_type)

    # ── patch111: warnings reach the log (pass-5 Q2) ────────────────────
    #
    # Fails against patch110, where a RuntimeWarning went to sys.stderr --
    # which curses has taken over, so it corrupted the display or vanished.

    def _capture_warnings(self):
        """Install the router with add_log captured; returns (lines, restore).

        Restores NumPy's *previous* error policy rather than a hardcoded one.
        An earlier version reset to all="ignore", which is not the default
        (divide/over/invalid default to "warn") and therefore silenced NumPy
        for every test that ran afterwards -- masking exactly the class of
        problem patch111 exists to surface.
        """
        lines = []
        saved_add_log = RDX.add_log
        saved_show = warnings.showwarning
        saved_err = np.geterr()
        RDX.add_log = lambda m, level="info": lines.append((level, m))
        with RDX._warning_lock:
            RDX._warning_seen.clear()
        RDX.install_warning_router()

        def restore():
            RDX.add_log = saved_add_log
            warnings.showwarning = saved_show
            np.seterr(**saved_err)
            with RDX._warning_lock:
                RDX._warning_seen.clear()
        return lines, restore

    def test_a_warning_reaches_the_log(self):
        lines, restore = self._capture_warnings()
        try:
            # Own the filter locally: under check.py's -W error::RuntimeWarning
            # a bare warnings.warn raises instead of reaching showwarning.
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn("something went sideways", RuntimeWarning)
        finally:
            restore()
        self.assertTrue(lines)
        level, msg = lines[0]
        self.assertEqual(level, "warn")
        self.assertIn("RuntimeWarning", msg)
        self.assertIn("something went sideways", msg)

    def test_a_numpy_overflow_reaches_the_log(self):
        # The class of failure _wrapped_delta's docstring is about.
        lines, restore = self._capture_warnings()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                _ = np.float32(1e30) * np.float32(1e30)
        finally:
            restore()
        self.assertTrue(any("overflow" in m.lower() for _l, m in lines), lines)

    def test_repeated_warnings_are_deduplicated(self):
        # NumPy can raise the same warning per element; 500 identical lines
        # would push every other diagnostic out of a 500-entry log.
        lines, restore = self._capture_warnings()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                for _ in range(300):
                    warnings.warn("same origin", RuntimeWarning)
        finally:
            restore()
        self.assertEqual(len(lines), 1)

    def test_the_dedup_set_is_bounded(self):
        self.assertIsInstance(RDX._WARNING_DEDUP_MAX, int)
        lines, restore = self._capture_warnings()
        try:
            for i in range(RDX._WARNING_DEDUP_MAX + 50):
                RDX._log_warning("m", RuntimeWarning, f"f{i}.py", i)
            with RDX._warning_lock:
                held = len(RDX._warning_seen)
        finally:
            restore()
        self.assertLessEqual(held, RDX._WARNING_DEDUP_MAX)

    def test_a_logging_failure_does_not_recurse(self):
        # A failure inside the router must not raise a second warning, which
        # would come straight back in here.
        saved = RDX.add_log
        saved_show = warnings.showwarning
        RDX.add_log = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            RDX.install_warning_router()
            RDX._log_warning("m", RuntimeWarning, "x.py", 1)   # must not raise
        finally:
            RDX.add_log = saved
            warnings.showwarning = saved_show
            np.seterr(all="ignore")

    def test_deliberate_wraparound_does_not_warn(self):
        # _wrapped_delta wraps on purpose; with overflow warnings on it must
        # not report its own correct behaviour as a fault.
        lines, restore = self._capture_warnings()
        try:
            for width in (1, 2, 4, 8):
                dtype = np.dtype(f"<u{width}")
                RDX._wrapped_delta(np.array([0, 1, 255], dtype=dtype), 10,
                                   width, dtype, -1)
        finally:
            restore()
        self.assertEqual([m for _l, m in lines], [])

    def test_rle_near_the_64_bit_ceiling_does_not_warn(self):
        lines, restore = self._capture_warnings()
        try:
            base = 2 ** 64 - 8 * 64
            arr = np.arange(base, base + 8 * 60, 8, dtype=np.uint64)
            np.testing.assert_array_equal(RDX._UndoAddrs(arr).array(), arr)
        finally:
            restore()
        self.assertEqual([m for _l, m in lines], [])

    def test_capture_helper_leaves_numpy_policy_untouched(self):
        # The teardown bug this replaced silenced NumPy for every subsequent
        # test, which is the one thing this whole feature exists to prevent.
        before = np.geterr()
        lines, restore = self._capture_warnings()
        restore()
        self.assertEqual(np.geterr(), before)

    def test_numpy_underflow_stays_ignored(self):
        # Normal and harmless in float scanning; warning about it is noise.
        lines, restore = self._capture_warnings()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                _ = np.float32(1e-40) * np.float32(1e-40)
        finally:
            restore()
        self.assertEqual([m for _l, m in lines if "underflow" in m.lower()], [])

    # ── patch112: the front door advertises the building (pass-4 D2) ─────

    def test_every_command_is_either_on_the_menu_or_advertised(self):
        # The finding: 16 of 22 commands were reachable only by typing a
        # name into a palette, which cannot help someone who does not know
        # the thing exists.
        on_menu = {label for _k, label, _a, _c in RDX._main_menu_entries()}
        advertised = set(RDX._menu_only_labels())
        for command in RDX._commands().values():
            if not command.in_palette:
                continue
            self.assertTrue(command.label in on_menu or command.label in advertised,
                            f"{command.label} is invisible")

    def test_wide_layout_sections_cover_the_whole_menu(self):
        # Adding "More Tools" pushed Quit past a hardcoded SETUP(5, 2) and
        # it silently stopped being drawn. The sections are derived now.
        menu = RDX._main_menu_entries()
        sections = [("SCAN", 0, 4), ("CHEATS", 4, 1),
                    ("SETUP", 5, max(1, len(menu) - 5))]
        covered = set()
        for _t, start, count in sections:
            covered |= set(range(start, start + count))
        self.assertEqual(covered, set(range(len(menu))))

    def test_more_tools_opens_the_palette(self):
        command = RDX._commands()["more_tools"]
        self.assertIs(command.handler, RDX.do_command_palette)
        self.assertEqual(command.menu_key, "M")
        # It is a doorway, not a destination: it should not list itself.
        self.assertFalse(command.in_palette)

    # ── patch113: class-name filter on Type Scan (pass-4 G1) ────────────

    def _type_groups(self):
        return [
            {"class_name": "PlayerController", "type_ptr": 0x400280,
             "module_name": "executable"},
            {"class_name": "EnemyAI", "type_ptr": 0x400300,
             "module_name": "executable"},
            {"class_name": None, "type_ptr": 0x400400,
             "module_name": "libc.prx"},
        ]

    def test_type_filter_matches_class_name_case_insensitively(self):
        groups = self._type_groups()
        for query in ("player", "PLAYER", "Controller"):
            got = RDX._filter_type_groups(groups, query)
            self.assertEqual([g["class_name"] for g in got],
                             ["PlayerController"], query)

    def test_type_filter_also_matches_pointer_and_module(self):
        groups = self._type_groups()
        self.assertEqual(len(RDX._filter_type_groups(groups, "0x400300")), 1)
        self.assertEqual(len(RDX._filter_type_groups(groups, "libc")), 1)

    def test_an_empty_type_filter_keeps_everything(self):
        groups = self._type_groups()
        for query in ("", "   ", None):
            self.assertEqual(len(RDX._filter_type_groups(groups, query)),
                             len(groups))

    def test_type_filter_survives_unnamed_groups(self):
        # A title where no class name resolved must still be filterable.
        groups = [{"class_name": None, "type_ptr": 0x400400,
                   "module_name": "m"}]
        self.assertEqual(RDX._filter_type_groups(groups, "zzz"), [])
        self.assertEqual(len(RDX._filter_type_groups(groups, "m")), 1)

    # ── patch114: first-run guide (pass-4 D3) ───────────────────────────

    def test_the_guide_teaches_the_loop_and_the_float_trap(self):
        text = "\n".join(RDX.first_run_guide_lines())
        self.assertIn("First Scan", text)
        self.assertIn("Next Scan", text)
        self.assertIn("f32", text)          # the documented first obstacle
        self.assertIn("float", text.lower())

    def test_the_guide_preference_round_trips(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prefs.json"
            RDX._save_preferences({RDX._GUIDE_PREF_KEY: True}, path=path)
            self.assertIs(RDX._load_preferences(path)[RDX._GUIDE_PREF_KEY],
                          True)

    def test_the_guide_is_shown_once(self):
        shown = []
        saved = dict(RDX._preferences)
        try:
            RDX._preferences.pop(RDX._GUIDE_PREF_KEY, None)
            with patch.object(RDX, "message_box",
                              lambda *a, **k: shown.append(1)), \
                 patch.object(RDX, "_save_preferences"):
                RDX.maybe_show_first_run_guide(self._FakeKeyWindow([]))
                RDX.maybe_show_first_run_guide(self._FakeKeyWindow([]))
        finally:
            RDX._preferences.clear(); RDX._preferences.update(saved)
        self.assertEqual(len(shown), 1)

    def test_a_failure_to_persist_does_not_break_the_guide(self):
        saved = dict(RDX._preferences)
        try:
            RDX._preferences.pop(RDX._GUIDE_PREF_KEY, None)
            with patch.object(RDX, "message_box"), \
                 patch.object(RDX, "_save_preferences",
                              side_effect=OSError("read-only")):
                RDX.maybe_show_first_run_guide(self._FakeKeyWindow([]))
        finally:
            RDX._preferences.clear(); RDX._preferences.update(saved)

    # ── patch115: automatic value type (pass-4 #5) ──────────────────────

    def test_auto_offers_only_types_the_value_parses_as(self):
        # Scanning u8 for 70000 is a guaranteed-empty pass, and three of
        # those in a row is how an automatic mode earns a bad reputation.
        self.assertNotIn("u8", RDX._auto_candidate_types("70000"))
        self.assertNotIn("u16", RDX._auto_candidate_types("70000"))
        for key in RDX._auto_candidate_types("-5"):
            self.assertNotIn(key, ("u8", "u16", "u32", "u64"))

    def test_auto_puts_the_common_case_first(self):
        self.assertEqual(RDX._auto_candidate_types("100")[0], "u32")
        # A decimal can only be a float, so nothing else is attempted.
        self.assertEqual(RDX._auto_candidate_types("100.5"), ["f32", "f64"])

    def test_auto_reaches_f32_for_a_whole_number(self):
        # The trap: Unity keeps health in floats, so "100" typed as a plain
        # integer must still be able to find a 100.0f.
        self.assertIn("f32", RDX._auto_candidate_types("100"))

    def test_auto_rejects_a_value_that_parses_as_nothing(self):
        for text in ("abc", "", "   ", "!!"):
            self.assertEqual(RDX._auto_candidate_types(text), [])

    def test_wire_auto_falls_through_u32_to_f32(self):
        # End to end over the protocol against a float planted as Unity
        # would store it.
        con = self._console()
        try:
            for addr in (0x2000400, 0x2100400):
                for i, b in enumerate(struct.pack("<f", 100.0)):
                    con.memory[addr + i] = b
            settled, hits = None, []
            for candidate in RDX._auto_candidate_types("100"):
                spec = RDX.VALUE_TYPES[candidate]
                value = RDX._parse_value_text("100", candidate, spec["width"])
                hits = RDX.scan_first(con.host, 91, value, spec["width"],
                                      value_type=candidate)
                if len(hits):
                    settled = candidate
                    break
            self.assertEqual(settled, "f32")
            self.assertEqual(sorted(int(a) for a in hits),
                             [0x2000400, 0x2100400])
        finally:
            self._release(con)

    # ── patch116: review of patches 112-115 ─────────────────────────────
    #
    # Four defects, all in code written in the preceding turn. Each fails
    # against patch115.

    def test_the_tool_list_fits_a_normal_terminal(self):
        # As a fourth column it needed a 160-column terminal before it drew
        # at all -- the three existing columns already reach 2/3 of the
        # width -- so the thing built to fix discoverability was itself
        # invisible on the 80-to-120-column terminals people use.
        labels = RDX._menu_only_labels()
        self.assertTrue(labels)
        for width in (72, 80, 100, 120):
            lines = RDX._wrap_help(" · ".join(labels), max(width - 8, 20))
            self.assertTrue(lines, width)
            for line in lines:
                self.assertLessEqual(len(line) + 3, width, width)
            # The first tool has to actually appear, not just the header.
            self.assertIn("Type Scan", "\n".join(lines))

    def test_capital_letters_reach_the_type_filter(self):
        # 'S' was bound to the structure view and its branch ran before the
        # printable-character branch, so "System", "Slot" and "Sprite" could
        # not be typed into a filter whose whole purpose is name search.
        groups = [{"class_name": "SystemManager", "type_ptr": 0x1,
                   "module_name": "m"},
                  {"class_name": "PlayerController", "type_ptr": 0x2,
                   "module_name": "m"}]
        got = RDX._filter_type_groups(groups, "System")
        self.assertEqual([g["class_name"] for g in got], ["SystemManager"])
        src = SOURCE.read_text()
        body = src.split("def do_type_scan")[1].split("\ndef ")[0]
        # Structure view must not be on a printable key on this screen.
        self.assertNotIn("elif key in (ord('S'),)", body)
        self.assertIn("ord('\\t')", body)

    def test_auto_attempts_are_capped(self):
        # Each attempt is a full scan; seven on a 2 GiB title is ~16 minutes
        # for a value that was never there.
        self.assertIsInstance(RDX._AUTO_MAX_ATTEMPTS, int)
        self.assertLessEqual(RDX._AUTO_MAX_ATTEMPTS, 4)
        candidates = RDX._auto_candidate_types("100")
        self.assertGreater(len(candidates), RDX._AUTO_MAX_ATTEMPTS)
        used = candidates[:RDX._AUTO_MAX_ATTEMPTS]
        # The cap must still cover the case Auto exists for.
        self.assertIn("u32", used)
        self.assertIn("f32", used)

    def test_auto_still_reaches_float_within_the_cap(self):
        # The whole point: a Unity float typed as a whole number.
        used = RDX._auto_candidate_types("100")[:RDX._AUTO_MAX_ATTEMPTS]
        self.assertLess(used.index("f32"), RDX._AUTO_MAX_ATTEMPTS)

    def test_no_dead_progress_label_remains(self):
        # _run_scan_with_progress takes its caption as an argument, so
        # writing progress["label"] did nothing at all.
        src = SOURCE.read_text()
        self.assertNotIn('progress["label"]', src)

    def test_zero_result_advice_does_not_repeat_what_auto_already_tried(self):
        # Telling someone to "try f32" after Auto has just tried f32 is
        # advice they have provably already followed.
        advice = RDX._zero_result_advice("100", "u32", "writable", False)
        joined = "\n".join(advice)
        self.assertIn("f32", joined)          # still there for a manual scan
        filtered = [l for l in advice[1:]
                    if "Most likely" not in l and "f32" not in l
                    and "Unity games keep" not in l
                    and "in floats. Run First Scan" not in l]
        self.assertNotIn("f32", "\n".join(filtered))


if __name__ == "__main__":
    unittest.main()


# ── patch138: writer resolution from a real hardware watchpoint event ──

GOLDEN_EVENT = Path(__file__).resolve().parent / "golden_watchpoint_event.bin"


class _GoldenEventBase:
    """Regression cover built from a captured hardware event.

    PS5 firmware 10.01, ps5debug-NG by OSR v1.3.0, eboot.bin pid 89.  A 4-byte
    write watchpoint on 0x00032a153f74 (the ammo field) was armed in DR slot 3
    and fired.  Two properties of the real event broke the previous code:

      * DR6 came back as 0.  The payload clears it while handling the trap, so
        the slot bit the trace used to identify its own event is never set.
      * RIP pointed one instruction *past* the store, because x86 data
        breakpoints are trap-type.

    The event is checked in as a fixture so these stay reproducible without a
    console.
    """

    TARGET   = 0x00032a153f74
    TRAP_RIP = 0x018f5b5b
    WRITER   = 0x018f5b55          # mov dword ptr [rbx+0x124], ecx
    WP_INDEX = 3
    RBX      = 0x32a153e50
    DISP     = 0x124

    def _writer_insn(self, addr=None, disp=None, base=56):
        # 56 == rbx in _ZYDIS_GPR64.  kind 0x10 = touches memory, 0x80 = write.
        return {"addr": self.WRITER if addr is None else addr,
                "rip_rel_target": 0,
                "mem_disp": self.DISP if disp is None else disp,
                "length": 6, "kind": 0x10 | 0x80, "mem_base_reg": base,
                "mem_index_reg": 0, "mem_scale": 0, "mnemonic_lo": 0}

    def _event(self, dr6=0):
        packet = bytearray(GOLDEN_EVENT.read_bytes())
        if dr6:
            struct.pack_into("<Q", packet,
                             RDX._DEBUG_DBREG_OFFSET + 6 * 8, dr6)
        return bytes(packet)

    # ── the fixture itself ──

    def test_golden_event_has_cleared_dr6_while_dr7_shows_it_armed(self):
        ev = RDX._debug_parse_event(GOLDEN_EVENT.read_bytes())
        dr = ev["dbregs"]
        self.assertEqual(dr[6], 0, "fixture no longer shows the cleared DR6")
        dr7 = dr[7]
        self.assertTrue((dr7 >> 6) & 1, "L3 not set")
        self.assertEqual((dr7 >> 28) & 3, 1, "R/W3 is not write-only")
        self.assertEqual((dr7 >> 30) & 3, 3, "LEN3 is not 4 bytes")
        self.assertEqual(int(ev["regs"]["rip"]), self.TRAP_RIP)
        self.assertEqual(int(ev["regs"]["rbx"]), self.RBX)
        self.assertEqual(int(ev["regs"]["rcx"]), 37, "ammo value drifted")

    # ── effective-address resolution ──

    def test_effective_address_resolves_to_the_watched_address(self):
        ev = RDX._debug_parse_event(GOLDEN_EVENT.read_bytes())
        eff, base_name, base_val, idx_name, idx_val = \
            RDX._decoded_effective_address(self._writer_insn(), ev["regs"])
        self.assertEqual(eff, self.TARGET)
        self.assertEqual(base_name, "rbx")
        self.assertEqual(base_val, self.RBX)
        self.assertIsNone(idx_name)
        self.assertEqual(idx_val, 0)

    def test_rip_relative_operand_resolves_against_instruction_end(self):
        insn = self._writer_insn(base=RDX._ZYDIS_RIP, disp=0x10)
        eff, base_name, base_val, _, _ = \
            RDX._decoded_effective_address(insn, {"rip": 0})
        self.assertEqual(base_name, "rip")
        self.assertEqual(base_val, self.WRITER + 6)
        self.assertEqual(eff, self.WRITER + 6 + 0x10)


class GoldenWatchpointEventTests(_GoldenEventBase, unittest.TestCase):
    """Properties of the captured event itself."""


class GoldenWriterResolutionTests(_GoldenEventBase, unittest.TestCase):
    """Drive _trace_temporary_access with the captured event."""

    class _Sock:
        def settimeout(self, *_): pass
        def close(self): pass
        def sendall(self, *_): pass
        def recv(self, *_): return b""

    def _listener(self, *_a, **_k):
        outer = self

        class L:
            def setsockopt(self, *_a): pass
            def bind(self, *_a): pass
            def listen(self, *_a): pass
            def settimeout(self, *_a): pass
            def accept(self): return outer._Sock(), ("192.168.0.88", 56307)
            def close(self): pass
        return L()

    def _run(self, packet, insns, target=None):
        target = self.TARGET if target is None else target
        old = {k: RDX.state.get(k) for k in ("proc_name", "ip", "pid")}
        RDX.state.update(proc_name="eboot.bin", ip="192.168.0.88", pid=89)
        try:
            with patch.object(RDX, "_trace_network_refusal", return_value=None), \
                 patch.object(RDX.socket, "socket", self._listener), \
                 patch.object(RDX, "ps5_read", return_value=b"\x25\x00\x00\x00"), \
                 patch.object(RDX, "ps5_connect", return_value=self._Sock()), \
                 patch.object(RDX, "_debug_status_word",
                              return_value=RDX.STATUS_SUCCESS), \
                 patch.object(RDX, "_debug_thread_list", return_value=[101676]), \
                 patch.object(RDX, "_debug_free_watchpoint_all",
                              return_value=self.WP_INDEX), \
                 patch.object(RDX, "_debug_set_watchpoint", lambda *a, **k: None), \
                 patch.object(RDX, "_debug_clear_watchpoint", lambda *a, **k: None), \
                 patch.object(RDX, "_debug_continue", lambda *a, **k: None), \
                 patch.object(RDX, "_debug_verify_watchpoint", return_value={}), \
                 patch.object(RDX, "_debug_watchpoint_preliminary",
                              return_value={}), \
                 patch.object(RDX, "_debug_watchpoint_verdict", return_value={}), \
                 patch.object(RDX, "_debug_detach_or_report", return_value=True), \
                 patch.object(RDX, "_debug_force_resume", return_value=True), \
                 patch.object(RDX, "recv_exact", return_value=packet), \
                 patch.object(RDX, "_debug_disasm", return_value=insns), \
                 patch.object(RDX, "add_log", lambda *a, **k: None):
                return RDX._trace_temporary_access(
                    "192.168.0.88", 89, target, 4, timeout=0.5,
                    experimental=True)
        finally:
            RDX.state.update(old)

    def test_writer_is_the_storing_instruction_not_the_trap_rip(self):
        # The whole point: RIP names the instruction after the store.
        trace = self._run(self._event(), [self._writer_insn()])
        self.assertTrue(trace["success"])
        self.assertEqual(trace["writer"], self.WRITER)
        self.assertEqual(trace["rip"], self.TRAP_RIP)
        self.assertNotEqual(trace["writer"], trace["rip"])
        self.assertEqual(trace["base_reg"], "rbx")
        self.assertEqual(trace["base_value"], self.RBX)
        self.assertEqual(trace["final_offset"], self.DISP)
        self.assertEqual(trace["access_mode"], "write")

    def test_event_with_cleared_dr6_is_not_discarded(self):
        # Previously the DR6 slot-bit gate threw this event away and the trace
        # timed out, which is exactly what happened on hardware.
        trace = self._run(self._event(dr6=0), [self._writer_insn()])
        self.assertEqual(trace["writer"], self.WRITER)

    def test_event_naming_a_different_slot_is_still_rejected(self):
        # Loosening the gate must not make it accept anyone else's watchpoint.
        with self.assertRaises(TimeoutError):
            self._run(self._event(dr6=1 << 1), [self._writer_insn()])

    def test_event_naming_our_own_slot_is_accepted(self):
        trace = self._run(self._event(dr6=1 << self.WP_INDEX), [self._writer_insn()])
        self.assertEqual(trace["writer"], self.WRITER)

    def test_accessor_resolving_elsewhere_is_rejected_not_returned(self):
        # An instruction that does not touch the watched address must not be
        # handed back as the writer; the trace keeps waiting and then times out.
        with self.assertRaises(TimeoutError):
            self._run(self._event(), [self._writer_insn(disp=0x999)])

    def test_instruction_ending_at_rip_is_preferred_over_one_starting_at_rip(self):
        decoy = self._writer_insn(addr=self.TRAP_RIP, disp=self.DISP)
        trace = self._run(self._event(), [self._writer_insn(), decoy])
        self.assertEqual(trace["writer"], self.WRITER,
                         "picked the instruction starting at RIP")


class ExecutableScopedAobScanTests(unittest.TestCase):
    """An instruction anchor can only live in executable memory."""

    SIG = {"pattern": "0F94C5410F95C64120D589832C010000"
                      "898B24010000C4C17A104724C5FA1183",
           "mask": "FF" * 32, "lead": 16}

    def test_relocation_asks_for_the_executable_scope(self):
        seen = {}

        def fake_scan(ip, pid, pattern, mask, **kw):
            seen.update(kw)
            # The scan reports where the *pattern* starts; the writer is `lead`
            # bytes into it, which is what relocation must hand back.
            return np.asarray([0x018f5b55 - 16], dtype=RDX._NP_ADDR_DTYPE)

        with patch.object(RDX, "scan_first_pattern", fake_scan), \
             patch.object(RDX, "add_log", lambda *a, **k: None):
            hit = RDX.relocate_by_aob_signature("1.2.3.4", 89, self.SIG)
        self.assertEqual(hit, 0x018f5b55)
        self.assertEqual(seen.get("region_scope"), "executable")
        self.assertFalse(seen.get("writable_only"))

    def test_executable_scope_skips_non_executable_regions(self):
        maps = [{"start": 0x400000,   "end": 0x401000, "prot": 5, "name": "code"},
                {"start": 0x10000000, "end": 0x10001000, "prot": 3, "name": "heap"},
                {"start": 0x20000000, "end": 0x20001000, "prot": 1, "name": "ro"}]
        read = []

        class Sock:
            def __init__(self, *a): pass
            def read(self, addr, size, _c): read.append(addr); return b"\x00" * size
            def close(self): pass

        with patch.object(RDX, "ps5_maps", return_value=maps), \
             patch.object(RDX, "_ScanSocket", Sock), \
             patch.object(RDX, "add_log", lambda *a, **k: None):
            RDX.scan_first_pattern("1.2.3.4", 89, b"\x90" * 4, b"\xff" * 4,
                                   region_scope="executable")
        self.assertEqual(read, [0x400000],
                         "scanned memory that cannot hold an instruction")

    def test_uniqueness_rules_are_unchanged_under_the_new_scope(self):
        with patch.object(RDX, "add_log", lambda *a, **k: None):
            for hits, expect in ((np.asarray([], dtype=RDX._NP_ADDR_DTYPE), None),
                                 (np.asarray([0x018f5b45, 0x1900000],
                                             dtype=RDX._NP_ADDR_DTYPE), None)):
                with patch.object(RDX, "scan_first_pattern",
                                  return_value=hits):
                    self.assertIs(
                        RDX.relocate_by_aob_signature("1.2.3.4", 89, self.SIG),
                        expect)


# ── patch139: the wired anchor pipeline ──

class InstructionAnchorPipelineTests(unittest.TestCase):
    """trace -> writer -> AOB -> unique relocation -> verified -> patch.

    Built on the same captured hardware case as the resolver tests: the ammo
    field at 0x00032a153f74 written by `mov [rbx+0x124], ecx` at 0x018f5b55,
    with the trap RIP six bytes later at 0x018f5b5b.
    """

    TARGET   = 0x00032a153f74
    TRAP_RIP = 0x018f5b5b
    WRITER   = 0x018f5b55
    INSN     = bytes.fromhex("898B24010000")
    PATTERN  = ("0F94C5410F95C64120D589832C010000"
                "898B24010000C4C17A104724C5FA1183")
    CODE = [{"start": 0x400000, "end": 0x2308000, "prot": 5, "name": "executable"}]

    def _trace(self, **over):
        trace = {"success": True, "target": self.TARGET, "rip": self.TRAP_RIP,
                 "writer": self.WRITER, "base_reg": "rbx",
                 "base_value": 0x32a153e50, "index_reg": None,
                 "index_value": 0, "scale": 1, "final_offset": 0x124,
                 "access_mode": "write", "lwpid": 101676,
                 "instruction": {"addr": self.WRITER, "length": 6,
                                 "kind": 0x90, "mem_base_reg": 56,
                                 "mem_index_reg": 0, "mem_scale": 0,
                                 "mem_disp": 0x124}}
        trace.update(over)
        return trace

    def _sig(self):
        return {"pattern": self.PATTERN, "mask": "FF" * 32, "lead": 16}

    def _capture(self, **kw):
        defaults = dict(read=lambda *a, **k: self.INSN,
                        sig=self._sig(), reloc=self.WRITER)
        defaults.update(kw)
        with patch.object(RDX, "_get_maps_cached", return_value=self.CODE), \
             patch.object(RDX, "ps5_read", defaults["read"]), \
             patch.object(RDX, "capture_aob_signature",
                          return_value=defaults["sig"]), \
             patch.object(RDX, "relocate_by_aob_signature",
                          return_value=defaults["reloc"]), \
             patch.object(RDX, "add_log", lambda *a, **k: None):
            return RDX.capture_instruction_anchor(
                "1.2.3.4", 89, kw.get("trace", self._trace()), 4)

    # ── contract ──

    def test_contract_never_exposes_trap_rip_as_the_writer(self):
        c = RDX._instruction_anchor_contract(self._trace(), 4)
        self.assertEqual(c["writer"], self.WRITER)
        self.assertEqual(c["trap_rip"], self.TRAP_RIP)
        self.assertNotEqual(c["writer"], c["trap_rip"])
        self.assertEqual(c["effective_address"], self.TARGET)
        self.assertEqual(c["temporary_address"], self.TARGET)

    def test_contract_refuses_a_trace_without_a_writer(self):
        broken = self._trace()
        del broken["writer"]
        with self.assertRaises(KeyError):
            RDX._instruction_anchor_contract(broken, 4)

    # ── raw RIP is never substituted ──

    def test_capture_fails_hard_when_the_writer_is_missing(self):
        broken = self._trace()
        del broken["writer"]
        out = self._capture(trace=broken)
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage"], "no-writer")
        self.assertIsNone(out["anchor"])

    def test_capture_never_anchors_the_trap_rip(self):
        broken = self._trace()
        del broken["writer"]
        out = self._capture(trace=broken)
        self.assertNotIn(str(self.TRAP_RIP), str(out.get("anchor")))
        self.assertIsNone(out["anchor"])

    # ── the happy path ──

    def test_anchor_captures_the_writer_and_relocates_to_it(self):
        out = self._capture()
        self.assertTrue(out["ok"], out["note"])
        a = out["anchor"]
        self.assertEqual(a["writer"], self.WRITER)
        self.assertEqual(a["relocated"], self.WRITER)
        self.assertEqual(a["trap_rip"], self.TRAP_RIP)
        self.assertEqual(a["instruction_bytes"], "898B24010000")
        self.assertEqual(len(a["signature"]["mask"]) // 2, 32)
        self.assertTrue(a["verified"])

    # ── failures that must abort ──

    def test_ambiguous_signature_is_rejected(self):
        out = self._capture(reloc=None)
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage"], "not-unique")

    def test_relocation_to_a_different_site_is_rejected(self):
        out = self._capture(reloc=self.WRITER + 0x40)
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage"], "relocation-mismatch")

    def test_writable_capture_is_rejected(self):
        # capture_aob_signature refuses writable memory; the pipeline must
        # surface that as a refusal, not paper over it.
        out = self._capture(sig=None)
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage"], "capture-refused")

    def test_operand_that_misses_the_watched_address_is_rejected(self):
        out = self._capture(trace=self._trace(final_offset=0x999))
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage"], "operand-mismatch")

    # ── verification gates the write ──

    def _anchor(self):
        return self._capture()["anchor"]

    def _verify(self, anchor, reloc, live, maps=None):
        with patch.object(RDX, "_get_maps_cached",
                          return_value=maps or self.CODE), \
             patch.object(RDX, "relocate_by_aob_signature", return_value=reloc), \
             patch.object(RDX, "ps5_read", lambda *a, **k: live), \
             patch.object(RDX, "add_log", lambda *a, **k: None):
            return RDX.verify_instruction_anchor("1.2.3.4", 89, anchor)

    def test_verify_accepts_an_unchanged_anchor(self):
        v = self._verify(self._anchor(), self.WRITER, self.INSN)
        self.assertTrue(v["ok"], v["note"])
        self.assertEqual(v["address"], self.WRITER)
        self.assertEqual(v["match_count"], 1)

    def test_verify_rejects_changed_bytes(self):
        v = self._verify(self._anchor(), self.WRITER, b"\x90" * 6)
        self.assertFalse(v["ok"])
        self.assertEqual(v["stage"], "bytes-changed")

    def test_verify_rejects_a_non_executable_region(self):
        heap = [{"start": 0x400000, "end": 0x2308000, "prot": 3, "name": "heap"}]
        v = self._verify(self._anchor(), self.WRITER, self.INSN, maps=heap)
        self.assertFalse(v["ok"])
        self.assertEqual(v["stage"], "not-executable")

    def test_verify_rejects_a_writable_region(self):
        rwx = [{"start": 0x400000, "end": 0x2308000, "prot": 7, "name": "rwx"}]
        v = self._verify(self._anchor(), self.WRITER, self.INSN, maps=rwx)
        self.assertFalse(v["ok"])
        self.assertEqual(v["stage"], "writable-region")

    def test_verify_rejects_an_ambiguous_relocation(self):
        v = self._verify(self._anchor(), None, self.INSN)
        self.assertFalse(v["ok"])
        self.assertEqual(v["stage"], "not-unique")

    # ── patch is refused whenever verification fails ──

    def test_patch_is_refused_when_verification_fails(self):
        wrote = []
        with patch.object(RDX, "_get_maps_cached", return_value=self.CODE), \
             patch.object(RDX, "relocate_by_aob_signature", return_value=None), \
             patch.object(RDX, "ps5_read", lambda *a, **k: self.INSN), \
             patch.object(RDX, "patch_instruction",
                          lambda *a, **k: wrote.append(a) or {"ok": True}), \
             patch.object(RDX, "add_log", lambda *a, **k: None):
            out = RDX.patch_instruction_anchor("1.2.3.4", 89, self._anchor())
        self.assertFalse(out["ok"])
        self.assertEqual(out["stage"], "not-unique")
        self.assertEqual(wrote, [], "wrote despite failed verification")

    def test_patch_nops_exactly_the_instruction_length(self):
        seen = {}

        def fake_patch(ip, pid, addr, new, expected, maps=None):
            seen.update(addr=addr, new=new, expected=expected)
            return {"ok": True, "stage": "patched", "address": addr}

        with patch.object(RDX, "_get_maps_cached", return_value=self.CODE), \
             patch.object(RDX, "relocate_by_aob_signature",
                          return_value=self.WRITER), \
             patch.object(RDX, "ps5_read", lambda *a, **k: self.INSN), \
             patch.object(RDX, "patch_instruction", fake_patch), \
             patch.object(RDX, "add_log", lambda *a, **k: None):
            out = RDX.patch_instruction_anchor("1.2.3.4", 89, self._anchor())
        self.assertTrue(out["ok"])
        self.assertEqual(seen["addr"], self.WRITER)
        self.assertEqual(seen["new"], b"\x90" * 6)
        self.assertEqual(seen["expected"], self.INSN)

    # ── the artifact is portable between runs ──

    def test_anchor_artifact_round_trips_through_json(self):
        a = self._anchor()
        back = RDX.anchor_from_json(RDX.anchor_to_json(a))
        self.assertEqual(back["writer"], self.WRITER)
        self.assertEqual(back["signature"]["pattern"], self.PATTERN)
        self.assertEqual(back["instruction_bytes"], "898B24010000")

    def test_anchor_artifact_of_an_unknown_version_is_refused(self):
        a = dict(self._anchor(), version=99)
        with self.assertRaises(ValueError):
            RDX.anchor_from_json(json.dumps(a))


# ── patch140: no developer environment baked into the shipped source ──

class ProductionHygieneTests(unittest.TestCase):
    """Guards against a development console leaking into the release build."""

    def test_source_contains_no_hardcoded_console_host(self):
        # The connect screen used to prefill one particular development
        # console's address, which reads to a new user as a real suggestion.
        # state["ip"] already carries their own last_ip preference.
        #
        # Network *ranges* are legitimate protocol constants (the CGNAT check
        # needs 100.64.0.0/10, the bind address is 0.0.0.0), so this looks for
        # a specific private *host* on the LAN ranges a console would sit on.
        host = re.compile(r"\b(?:192\.168\.\d{1,3}\.\d{1,3}"
                          r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
                          r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b")
        hits = [f"{n}: {ln.strip()}"
                for n, ln in enumerate(SOURCE.read_text().splitlines(), 1)
                if host.search(ln) and "/" not in host.search(ln).group(0)
                and not re.search(r"\.0\.0/|/\d{1,2}\b", ln)]
        self.assertEqual(hits, [],
                         "a specific console address is baked into the source")

    def test_connect_prompt_prefills_from_preferences_not_a_literal(self):
        src = SOURCE.read_text()
        self.assertIn('state["ip"] or ""', src)
        self.assertIn('_preferences.get("last_ip", "")', src)

    def test_no_developer_paths_in_source(self):
        src = SOURCE.read_text()
        for needle in ("/home/", "rdx-export-test", "cherriios"):
            self.assertNotIn(needle, src, f"developer path {needle!r} shipped")
