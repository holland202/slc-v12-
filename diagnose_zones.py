"""
diagnose_zones.py - Per-zone thermal diagnostic

WHAT THIS FILE IS NOW: a record of two refuted hypotheses and the vendor
trip points that replaced them. The tool still runs and still reports
per-zone statistics, but it NO LONGER RECOMMENDS EXCLUDING ANY ZONE, and it
must not be used to populate an exclusion list.

ORIGINAL HYPOTHESIS (REFUTED, KEPT)
-----------------------------------
Written because aggregate "MAX across zones" calibration produced 72-76C
when idle --list snapshots never showed above ~48C. Hypothesis: one zone
glitches intermittently and poisons the max. Remedy: name it, exclude it.

Refuted. Measured on the S25 Ultra (Snapdragon 8 Elite), 2026-08-23, five
runs, same 35s busy-loop load unless noted:

    run  cooling      start    peak      flagged
    1    no fan       ~38C     87.6C     24 of 28 zones      (criterion A)
    2    Black Shark  ~37C     60.2C     zone17, 19          (criterion A)
    3    Black Shark  ~27C     54.4C     zone16, 17, 19      (criterion A)
    4    Black Shark  ~23C     103.4C    zones 16-21         (criterion B)
    5    8x `yes`     ~43C     103.8C    none - see below    (0.5s sampling)

Run 5 settles it. Sampling every 0.5s under 8 parallel processes, ALL zones
ramp together (cpu-0 cluster 43->96C, cpu-1 cluster 41->103C) and ALL decay
together when load stops. No teleporting values, no zone diverging from its
neighbours. Every sensor tracks reality. The 72-76C was a real phone getting
genuinely hot; the "no zone above 48C" comparison came from idle snapshots
taken at a different time. A STATE MISMATCH, not a sensor fault.

TWO CRITERIA, BOTH REFUTED (KEPT)
---------------------------------
Criterion A - `max - mean > 10C`. Flags any zone that heated during the
window, because a ramp puts the mean mid-range and the max at the top. In
run 1 it flagged 24 of 28 compute zones. Acting on it would have blinded the
governor to nearly every sensor it has, including its hottest cores.

Criterion B - `max |t[i] - t[i-1]| > 15C`. Written to fix A by testing for a
discontinuity in time rather than spread. Also refuted: this hardware climbs
~45C in ONE SECOND under load (run 5, t=0.0->1.0s: 58.4 -> 91.1 -> 101.5C).
At 1Hz sampling a genuine ramp aliases into an apparent 30-45C jump. B was
the same error as A with a different threshold.

LESSON: any variance- or rate-based "unreliable sensor" heuristic must first
be shown to separate a ramp from a glitch ON THIS HARDWARE. Neither of these
was. The selftest below now encodes that requirement.

VENDOR TRIP POINTS - the ground truth that ends the argument
------------------------------------------------------------
Read from the device tree via sysfs, identical on cpu-0 and cpu-1 clusters:

    trip1  105000  passive     <- first throttle
    trip2  118000  passive
    trip3  120000  passive
    trip4  125000  hot
    trip0  135000  passive     <- ceiling

Measured peak under sustained load: 96-103C. That is BELOW the vendor's
first passive trip. The hardware considers this range normal operation and
has not begun throttling.

CONSEQUENCE FOR THE GOVERNOR (open, not addressed by this file): SLC's
temp_critical is 37.0C - 68C below the vendor's first throttle point and
about 10C above idle. calibrate_thermal.py derived that envelope by sampling
at IDLE, so the governor halts the moment real work starts. The defect is in
the threshold and its derivation, not in the sensors. Per Kumar, "The
Thermal Framework in the Linux Kernel" (IJSR 13(11), 2024), the kernel
distinguishes active / passive / critical trip types and warns specifically
about single thresholds without hysteresis. Fixing that is a design change
to the governor's state transitions and is deliberately NOT done here.

COOLING DOMINATES
-----------------
Identical load, 33C spread in peak between fan and no fan (87.6C vs 60.2C).
Starting temperature moved peaks by a further ~6C. Any threshold is valid
only for the cooling setup and starting temperature it was calibrated under.
Record both alongside any result from this tool.

Usage:
    python3 diagnose_zones.py
    python3 diagnose_zones.py --keywords cpu gpu
    python3 diagnose_zones.py --selftest
"""
import os
import sys
import time
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.hardware_link import discover_compute_zones, DEFAULT_KEYWORDS

SAMPLES = 30
DELAY = 1.0

# RETRACTED AS A VERDICT. This was criterion B, and it is refuted: run 5
# measured a genuine 45C rise in one second, so at 1Hz sampling a real ramp
# exceeds any threshold set below that. The value is retained ONLY to
# annotate the report with the largest consecutive delta observed, so a
# reader can see the sampling-rate problem for themselves. It does not
# recommend excluding anything.
JUMP_LIMIT = 15.0
MEASURED_MAX_RAMP_RATE_C_PER_S = 45.0   # run 5, 0.5s sampling, 8x load


def read_one(path):
    try:
        with open(path) as f:
            raw = float(f.read().strip())
        if raw < -200 or raw == 0:
            return None
        return raw / 1000.0 if raw > 1000 else raw
    except (OSError, ValueError):
        return None


def max_jump(vals):
    """Largest absolute change between consecutive samples."""
    if len(vals) < 2:
        return 0.0
    return max(abs(b - a) for a, b in zip(vals, vals[1:]))


def analyse(history, names):
    """Returns (rows, spiky). A row is (name, mn, mean, mx, sd, jump)."""
    rows, spiky = [], []
    for name in names:
        vals = history.get(name) or []
        if not vals:
            continue
        mn, mean, mx = min(vals), statistics.mean(vals), max(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        jump = max_jump(vals)
        if jump > JUMP_LIMIT:
            spiky.append(name)
        rows.append((name, mn, mean, mx, sd, jump))
    return rows, spiky


def report(rows, spiky):
    print(f"{'Zone':<16} {'Min':>8} {'Mean':>8} {'Max':>8} {'StdDev':>8} "
          f"{'MaxJump':>8}  Flag")
    print("-" * 72)
    for name, mn, mean, mx, sd, jump in rows:
        flag = "<-- DISCONTINUITY" if jump > JUMP_LIMIT else ""
        print(f"{name:<16} {mn:>7.2f}C {mean:>7.2f}C {mx:>7.2f}C {sd:>7.2f} "
              f"{jump:>7.2f}C   {flag}")

    if rows:
        ramp = max(r[3] - r[1] for r in rows)
        print(f"\nLargest total rise across the window: {ramp:.1f}C "
              f"(this is RAMP, not spike -- it is expected under load)")

    print()
    print("NO ZONE IS RECOMMENDED FOR EXCLUSION. This tool does not identify")
    print("faulty sensors; two criteria that claimed to have been refuted.")
    print("See the header of this file for all five runs.")
    if spiky:
        print(f"\n{len(spiky)} zone(s) showed a consecutive delta above "
              f"{JUMP_LIMIT:.0f}C: {spiky}")
        print(f"At 1Hz sampling this is EXPECTED: measured ramp rate on this")
        print(f"hardware reaches {MEASURED_MAX_RAMP_RATE_C_PER_S:.0f}C/s under load, so a genuine")
        print("ramp aliases into an apparent jump. Sample faster before")
        print("concluding anything about a sensor.")
    print(f"\nVendor trip points (sysfs, this device): first passive throttle")
    print(f"at 105C, 'hot' at 125C. Peaks measured here are BELOW the first")
    print(f"trip -- the hardware regards this as normal operation.")


def selftest():
    """The instrument must separate a RAMP from a SPIKE. If it flags both or
    neither, it is measuring nothing and no zone should be excluded on its
    word -- which is exactly how the previous criterion failed."""
    print("=" * 72)
    print("ANTI-VACUITY SELFTEST")
    print("=" * 72)
    ok = True

    # A: a steep but physical ramp, 27C -> 54C over 30 samples (run 3, real).
    ramp = [27.0 + i * 0.9 for i in range(30)]
    rows, spiky = analyse({"ramp": ramp}, ["ramp"])
    pa = not spiky
    print(f"\n  A steep ramp 27->54C is NOT flagged: {'PASS' if pa else 'FAIL'}"
          f"  (max jump {rows[0][5]:.2f}C, max-mean {rows[0][3]-rows[0][2]:.2f}C)")
    print(f"     the OLD criterion (max-mean>10) would have flagged this: "
          f"{rows[0][3] - rows[0][2] > 10.0}")
    ok = ok and pa

    # B: a real glitch -- one sample 40C off its neighbours.
    glitch = [40.0] * 15 + [92.0] + [40.5] * 14
    rows, spiky = analyse({"glitch": glitch}, ["glitch"])
    pb = bool(spiky)
    print(f"  B a 52C single-sample glitch IS flagged: "
          f"{'PASS' if pb else 'FAIL'}  (max jump {rows[0][5]:.2f}C)")
    ok = ok and pb

    # C: a flat idle trace must be clean.
    flat = [36.0 + (i % 3) * 0.1 for i in range(30)]
    _, spiky = analyse({"flat": flat}, ["flat"])
    pc = not spiky
    print(f"  C a flat idle trace is NOT flagged: {'PASS' if pc else 'FAIL'}")
    ok = ok and pc

    # D: THE REFUTATION, encoded. Real 1Hz samples from run 5's ramp. This
    #    IS flagged, and it is NOT a glitch -- it is a genuine 45C/s rise
    #    aliased by the sampling rate. The fixture must fail-to-distinguish,
    #    proving the criterion cannot be trusted as a verdict on this
    #    hardware. If a future edit makes this fixture pass cleanly, that
    #    edit has hidden the refutation rather than fixed it.
    real_ramp_1hz = [58.4, 101.5, 103.0, 103.4, 103.4, 103.4]
    rows, spiky_d = analyse({"real": real_ramp_1hz}, ["real"])
    pd_ = bool(spiky_d)   # we EXPECT the criterion to misfire here
    print(f"  D a REAL 45C/s ramp at 1Hz is (wrongly) flagged: "
          f"{'PASS' if pd_ else 'FAIL'}  (delta {rows[0][5]:.1f}C)")
    print("     this fixture exists to keep the refutation visible: the")
    print("     criterion cannot separate ramp from glitch at 1Hz.")
    ok = ok and pd_

    print("\n" + "=" * 72)
    if ok:
        print("SELFTEST PASS -- the analysis functions behave as specified.")
        print("NOTE: passing does NOT mean this tool can identify a faulty")
        print("sensor. Fixture D shows it misclassifies a real ramp at 1Hz.")
        print("It reports statistics; it does not issue exclusion verdicts.")
    else:
        print("SELFTEST FAIL -- do not trust this tool until fixed.")
    print("=" * 72)
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    zones = discover_compute_zones(args.keywords)
    if not zones:
        print(f"No zones matched {args.keywords}.")
        return 1

    names = [z.split("/")[-2] for z in zones]
    print(f"Tracking {len(zones)} zones individually for {SAMPLES} samples "
          f"({SAMPLES * DELAY:.0f}s): {names}\n")
    print("NOTE: record your cooling setup and starting temperature with any")
    print("result from this tool. Identical load produced a 33C spread in")
    print("peak temperature between fan and no fan on this device.\n")

    history = {name: [] for name in names}
    for i in range(SAMPLES):
        for path, name in zip(zones, names):
            v = read_one(path)
            if v is not None:
                history[name].append(v)
        print(f"Sample {i+1}/{SAMPLES}", end="\r")
        time.sleep(DELAY)
    print("\n")

    rows, spiky = analyse(history, names)
    report(rows, spiky)
    return 0


if __name__ == "__main__":
    sys.exit(main())
