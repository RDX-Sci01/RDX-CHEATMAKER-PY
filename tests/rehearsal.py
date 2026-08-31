#!/usr/bin/env python3
"""
Rehearse the hardware checklist offline, against a protocol-speaking console.

    python3 rehearsal.py

`HARDWARE_TEST_CHECKLIST.md` carries 80-odd open items and every one of them
currently costs console time -- and some cost an attach that has already been
observed to black-screen a live title. A number of those items are not really
asking about the console at all; they are asking whether RDX's own wire code
does the right thing, and that can be settled here for free.

This does not replace the hardware session. It splits the checklist into the
part that can be answered offline and the part that genuinely cannot, so the
console time is spent on the second kind.

What it cannot answer, and why:
  * whether the real payload honours debug registers for a given store;
  * whether grouping the qword at offset 0 really surfaces Il2CppClass
    pointers on a real IL2CPP title;
  * timings, and anything about firmware behaviour.

Exit code is 0 when every rehearsed item passes.
"""

import pathlib
import re
import struct
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fake_console import (FakeConsole, seed_type_pointers,  # noqa: E402
                          seed_value)

import importlib.util  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
LAUNCHER = HERE.parent / "RDX-CHEATMAKER-UI-final.py"


def load_rdx():
    # Since the 1.0.0 consolidation the launcher IS the implementation; there
    # is no longer a numbered patch file to resolve.
    spec = importlib.util.spec_from_file_location("rdx", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, LAUNCHER.name


RDX, IMPL = load_rdx()


class Rehearsal:
    def __init__(self):
        self.results = []

    def item(self, section, name, fn):
        started = time.time()
        try:
            detail = fn() or ""
            ok = True
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            ok = False
        self.results.append((section, name, ok, detail,
                             time.time() - started))

    def report(self):
        width = max(len(n) for _s, n, _o, _d, _t in self.results) + 2
        section = None
        for sec, name, ok, detail, secs in self.results:
            if sec != section:
                print(f"\n{sec}")
                print("-" * (width + 34))
                section = sec
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:<{width}} {secs:5.2f}s  {detail}")
        failed = [r for r in self.results if not r[2]]
        print()
        print(f"{len(self.results) - len(failed)}/{len(self.results)} "
              f"rehearsed items passed against {IMPL}")
        return 0 if not failed else 1


def _attach(con):
    RDX.PS5_PORT = con.port
    RDX.state.update(ip=con.host, pid=91, backend="ps5debug", memdbg=None,
                     session=1, proc_name="eboot.bin")
    RDX._ScanSocket.clear_pool()
    with RDX._map_cache_lock:
        RDX._map_cache.clear()


def main():
    r = Rehearsal()

    # ── §0 environment ──
    def connect_and_list():
        with FakeConsole() as con:
            _attach(con)
            procs = RDX.ps5_proc_list(con.host)
            maps = RDX.ps5_maps(con.host, 91)
            return f"{len(procs)} processes, {len(maps)} map rows"
    r.item("§0 Environment", "connect, list processes, fetch maps",
           connect_and_list)

    def auth():
        with FakeConsole() as con:
            _attach(con)
            RDX.ps5_auth_scanner(con.host)
            return "challenge/response accepted"
    r.item("§0 Environment", "scanner auth handshake", auth)

    def game_marker():
        with FakeConsole() as con:
            _attach(con)
            con.maps.append({"name": "/app0/eboot.bin", "start": 0x3000000,
                             "end": 0x3001000, "prot": 5})
            owns = RDX._process_owns_app0(con.host, 91)
            return f"/app0/ ownership detected: {owns}"
    r.item("§0 Environment", "game marker probe (/app0/)", game_marker)

    # ── §2 scanning ──
    def first_and_next():
        with FakeConsole() as con:
            _attach(con)
            seed_value(con, 0x2000400, 999, 4)
            seed_value(con, 0x2100400, 999, 4)
            hits = RDX.scan_first(con.host, 91, 999, 4, value_type="u32")
            assert sorted(int(a) for a in hits) == [0x2000400, 0x2100400], hits
            seed_value(con, 0x2000400, 111, 4)
            left = RDX.scan_next(con.host, 91, 999, 4, hits, value_type="u32")
            assert [int(a) for a in left] == [0x2100400], left
            return "2 hits -> 1 survivor after an in-game change"
    r.item("§2 Scanning", "exact scan, then relational narrowing",
           first_and_next)

    def aob_boundary():
        with FakeConsole() as con:
            _attach(con)
            planted = 0x2200000 - 3          # straddles two adjacent mappings
            for i, b in enumerate(b"\xDE\xAD\xBE\xEF\xCA\xFE"):
                con.memory[planted + i] = b
            pattern, mask, _ = RDX._parse_byte_pattern("DE AD BE EF CA FE")
            hits = [int(a) for a in
                    RDX.scan_first_pattern(con.host, 91, pattern, mask)]
            assert planted in hits, hits
            return f"matched at {hex(planted)}, across the mapping join"
    r.item("§2 Scanning", "AOB match spanning two adjacent mappings",
           aob_boundary)

    def region_warning():
        with FakeConsole() as con:
            _attach(con)
            saved = dict(RDX._settings)
            RDX.state["log"] = []
            try:
                RDX._settings["region_min_size"] = 0x10000000
                RDX.scan_first(con.host, 91, 1, 4, value_type="u32",
                               region_scope="recommended")
                msgs = " ".join(e["msg"] for e in RDX.state["log"])
            finally:
                RDX._settings.clear(); RDX._settings.update(saved)
            assert "region settings" in msgs, msgs
            return "a non-default region setting is named as the cause"
    r.item("§2 Scanning", "region-settings warning fires", region_warning)

    # ── §3 writes and freezes ──
    def verified_write():
        with FakeConsole() as con:
            _attach(con)
            ack, verified, _ = RDX.ps5_write_verified(
                con.host, 91, 0x2000100, (7).to_bytes(4, "little"))
            assert ack and verified
            return "write acknowledged and read back"
    r.item("§3 Writes", "verified write", verified_write)

    def bulk_write():
        with FakeConsole() as con:
            _attach(con)
            entries = [(0x2000800 + i * 0x100, (i + 1).to_bytes(4, "little"))
                       for i in range(8)]
            acks = RDX.ps5_write_multi(con.host, 91, entries)
            assert all(acks), acks
            vals = [int.from_bytes(RDX.ps5_read(con.host, 91, a, 4), "little")
                    for a, _ in entries]
            assert vals == list(range(1, 9)), vals
            return "8 entries in one exchange, all applied in order"
    r.item("§3 Writes", "bulk write (the freeze tick path)", bulk_write)

    def refuses_unmapped():
        with FakeConsole() as con:
            _attach(con)
            err = RDX._validate_addr_in_maps(con.host, 91, 0xDEAD0000, 4)
            assert err, "unmapped address was accepted"
            ro = RDX._validate_addr_in_maps(con.host, 91, 0x400100, 4)
            assert ro, "read-only mapping was accepted for a write"
            return "unmapped and read-only both refused"
    r.item("§3 Writes", "write validation refuses bad targets",
           refuses_unmapped)

    # ── §4 type scan ──
    def type_scan():
        with FakeConsole() as con:
            _attach(con)
            ptr = seed_type_pointers(con, base=0x2000000, count=64)
            groups = RDX.scan_type_instances(con.host, 91, min_instances=8)
            assert groups and groups[0]["type_ptr"] == ptr, groups[:1]
            return (f"{len(groups)} type(s); {groups[0]['count']} instances of "
                    f"{hex(ptr)} in {groups[0]['module_name']}")
    r.item("§4 Type scan", "groups instances by type pointer", type_scan)

    def type_scan_disconnect():
        with FakeConsole() as con:
            _attach(con)
            con.stop()                        # console goes away mid-scan
            try:
                RDX.scan_type_instances(con.host, 91, min_instances=8)
            except Exception as exc:
                return f"reported as a fault, not as 0 results ({type(exc).__name__})"
            raise AssertionError("a dead console produced a clean empty result")
    r.item("§4 Type scan", "a disconnect is not reported as 'no types'",
           type_scan_disconnect)

    # ── §5 watchpoint diagnostic ──
    def dr_verdict(mode, expected):
        def run():
            with FakeConsole(dr_mode=mode, thread_count=40) as con:
                _attach(con)
                sock = RDX.ps5_connect(con.host)
                try:
                    threads = RDX._debug_thread_list(sock)
                    index = RDX._debug_free_watchpoint_all(sock, threads)
                    RDX._debug_set_watchpoint(sock, index, 0x2000400, 3, 1)
                    cov = RDX._debug_verify_watchpoint(sock, threads,
                                                       0x2000400, index)
                    key, _text = RDX._debug_watchpoint_verdict(cov)
                finally:
                    sock.close()
                assert key == expected, f"got {key!r}, wanted {expected!r}"
                return (f"{len(cov['armed'])}/{cov['checked']} threads armed "
                        f"-> verdict {key!r}")
        return run

    for mode, expected in (("all-threads", "all"),
                           ("first-thread-only", "partial"),
                           ("none", "none")):
        r.item("§5 Watchpoint diagnostic",
               f"payload behaviour: {mode}", dr_verdict(mode, expected))

    # ── §6 export ──
    def shn_mc4_pair():
        mods = [{"name": "Infinite Ammo",
                 "memory": [{"offset": "1A2B", "on": "90909090",
                             "off": "01020304"}]}]
        args = (mods, "CUSA01659", "01.00", "Enter the Gungeon", "eboot.bin")
        shn = RDX.generate_shn_text(*args)
        mc4 = RDX.generate_mc4_bytes(*args)
        assert RDX._mc4_decrypt(mc4).decode("utf-8") == shn
        return ".mc4 is exactly the .shn encrypted (rejection is attributable)"
    r.item("§6 Export", ".shn / .mc4 pair agree", shn_mc4_pair)

    code = r.report()
    print("\nStill requires real hardware, and is NOT rehearsed above:")
    print("  * whether the payload honours DRs for a given store (§5 decides")
    print("    only that RDX reports what it sees correctly)")
    print("  * whether type pointers on a real IL2CPP title behave as assumed")
    print("  * CheatRunner accepting a .mc4/.shn, and its Address Mode")
    print("  * all timings, and any firmware-specific behaviour")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
