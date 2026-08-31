#!/usr/bin/env python3
"""
Run every check this project has, in one command.

    python3 check.py              # unit suite, rehearsal, pty smoke
    python3 check.py --quick      # unit suite only (~25 s)
    python3 check.py --regression # also prove the new tests fail on patch N-1

Why this exists
---------------
The project accumulated three separate suites -- `test_pointer_subsystem.py`,
`rehearsal.py` and `ui_smoke.py` -- plus a convention enforced entirely by
hand:

    If you fix a bug, the regression test for it should **fail against the
    previous patch file** and pass against yours.
                                                        -- README.md

That convention is the project's main defence against tests that only look
like regressions, and performing it means editing `SOURCE` in the test file,
running the suite, and editing it back. It is exactly the kind of step that
stops happening the first time somebody is in a hurry.

For comparison, cheat-engine-linux runs its ~320-check regression suite plus
sanitisers and offscreen-GUI smoke tests in CI. RDX has more checks than that
and ran none of them automatically.

This is not CI -- the tree is not a git repository -- but it is the thing CI
would call, and it makes the whole quality story one command instead of four
remembered ones.

Warnings are treated as errors under `-X dev`, so a NumPy overflow or a
deprecation surfaces here rather than being written to a terminal curses has
taken over. patch111 routes them into the in-app log at runtime; this catches
them at build time.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LAUNCHER = ROOT / "RDX-CHEATMAKER-UI-final.py"
SUITE = HERE / "test_pointer_subsystem.py"

# Warnings-as-errors, so a NumPy overflow fails the build instead of scrolling
# past. RuntimeWarning is the one that matters for this codebase; the others
# are cheap to keep clean and expensive to let rot.
STRICT = ["-X", "dev", "-W", "error::RuntimeWarning",
          "-W", "error::DeprecationWarning"]


def current_impl() -> str:
    """The implementation under test.

    The launcher used to indirect to a numbered patch file; since the
    consolidation it *is* the implementation, so there is nothing to resolve.
    """
    return LAUNCHER.name


def previous_impl(impl: str):
    """The archived patch immediately before `impl`, if there is one.

    Numbered patch archives were removed at consolidation -- git history is the
    development history now -- so this normally returns None and the
    regression-convention check reports that it has nothing to compare against.
    The lookup is kept for a release workflow that reintroduces numbered
    builds.
    """
    match = re.search(r"patch(\d+)\.py$", impl)
    if not match:
        return None
    prior = HERE / f"RDX-CHEATMAKER-UI-patch{int(match.group(1)) - 1}.py"
    return prior if prior.exists() else None


def run(name: str, argv: list, cwd=HERE, timeout=900) -> tuple:
    """Run one check; return (ok, seconds, tail)."""
    started = time.time()
    try:
        proc = subprocess.run(argv, cwd=str(cwd), capture_output=True,
                              text=True, timeout=timeout)
        out = (proc.stdout + proc.stderr).strip().splitlines()
        return proc.returncode == 0, time.time() - started, out[-6:]
    except subprocess.TimeoutExpired:
        return False, time.time() - started, [f"timed out after {timeout}s"]


def check_regression_convention() -> tuple:
    """Confirm the suite genuinely fails against the previous patch.

    A suite that passes against both patches proves nothing about the fix it
    claims to cover -- the README calls that "a guard, not a regression test".
    This automates the SOURCE-swap that was being done by hand.
    """
    impl = current_impl()
    prior = previous_impl(impl)
    if prior is None:
        return None, 0.0, ["no previous patch archived; nothing to compare"]
    text = SUITE.read_text()
    match = re.search(r'^SOURCE = .*$', text, re.M)
    if not match:
        return False, 0.0, ["could not find SOURCE in the test file"]
    original = match.group(0)
    swapped = f'SOURCE = Path(r"{prior}")'
    started = time.time()
    try:
        SUITE.write_text(text.replace(original, swapped))
        proc = subprocess.run([sys.executable, "-m", "unittest",
                               "test_pointer_subsystem"],
                              cwd=str(HERE), capture_output=True, text=True,
                              timeout=900)
        combined = proc.stdout + proc.stderr
        tail = [l for l in combined.splitlines()
                if l.startswith(("OK", "FAILED", "Ran "))]
        # A non-zero exit is not enough. If the suite could not even be
        # collected against the older patch -- an import error, a missing
        # name, a syntax problem -- it also exits non-zero, and reporting
        # that as "the new tests correctly fail" would be exactly the false
        # assurance this check exists to prevent. Require evidence that the
        # tests actually ran.
        ran = re.search(r"^Ran (\d+) tests?", combined, re.M)
        if ran is None:
            return False, time.time() - started, tail + [
                "the suite did not run against " + prior.name +
                " (collection error?) — this proves nothing"]
        failed = proc.returncode != 0
    finally:
        # Always restore, including on Ctrl-C or a crash: leaving the suite
        # pointed at an archived patch would be a nasty thing to discover.
        SUITE.write_text(text)
    note = (f"{ran.group(1)} tests ran; new ones correctly fail against "
            + prior.name if failed else
            "suite PASSES against " + prior.name + " — the new tests are "
            "guards, not regressions")
    return failed, time.time() - started, tail + [note]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="unit suite only")
    ap.add_argument("--regression", action="store_true",
                    help="also prove the new tests fail against patch N-1")
    args = ap.parse_args()

    impl = current_impl()
    print(f"RDX checks — {impl}\n")

    checks = [("unit suite (strict warnings)",
               [sys.executable] + STRICT + ["-m", "unittest",
                                            "test_pointer_subsystem"])]
    if not args.quick:
        checks += [
            ("offline rehearsal", [sys.executable, "rehearsal.py"]),
            ("real-terminal smoke", [sys.executable, "ui_smoke.py"]),
        ]

    results = []
    for name, argv in checks:
        ok, secs, tail = run(name, argv)
        results.append((name, ok, secs, tail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<32} {secs:6.1f}s")
        if not ok:
            for line in tail:
                print(f"         {line}")

    if args.regression:
        ok, secs, tail = check_regression_convention()
        label = "regression convention"
        if ok is None:
            print(f"  [SKIP] {label:<32} {secs:6.1f}s  {tail[0]}")
        else:
            results.append((label, ok, secs, tail))
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:<32} {secs:6.1f}s")
            for line in tail:
                print(f"         {line}")

    failed = [r for r in results if not r[1]]
    total = sum(r[2] for r in results)
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed "
          f"in {total:.0f}s")
    if failed:
        print("failed: " + ", ".join(r[0] for r in failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
