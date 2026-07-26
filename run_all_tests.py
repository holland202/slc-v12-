#!/usr/bin/env python3
"""SLC v12 — run all gated test suites.

Only suites with fixed, machine-independent expectations are gated here.
tests/test_governance_loop.py is deliberately EXCLUDED: it reads live device
temperature, so its PASS/FAIL reflects how warm the phone is, not whether the
code is correct. Gating it would produce failures on a hot day.
"""
import subprocess
import sys
import os

SUITES = [
    ("Thermal ceiling", "tests/test_thermal_ceiling.py"),
    ("Thermal discrimination", "tests/test_thermal_discrimination.py"),
]

if __name__ == "__main__":
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.abspath(__file__)))
    all_ok = True
    for name, path in SUITES:
        print("\n" + "=" * 60)
        print(f"RUNNING: {name}  ({path})")
        print("=" * 60)
        rc = subprocess.run([sys.executable, path], env=env).returncode
        all_ok = all_ok and rc == 0
    print("\n" + "=" * 60)
    print("ALL SUITES PASSED" if all_ok else "SOME SUITES FAILED")
    sys.exit(0 if all_ok else 1)
