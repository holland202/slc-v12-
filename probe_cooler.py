#!/usr/bin/env python3
"""
probe_cooler.py — P13: does the attached cooler move this substrate under the
defense profile's 36.5 C threshold?

Registered in P13_PREREG.md BEFORE the fan was first switched on.

Conditions:
  G      instrument gate  — ThermalMonitor must be reading real zones, not sim
  c1     fan OFF, idle    — max compute zone > 36.5 C   (anti-vacuity)
  c2     fan ON,  idle    — max compute zone < 36.5 C sustained for --duration
  c3     fan ON,  load    — VeritasGate does not HALT   (EXPECTED TO FAIL)

Why G exists: core/hardware_link.py ThermalMonitor.read() falls back to
simulation mode (synthetic 30-45 C drift) if every bound zone fails to read.
Without G, c2 can PASS on fabricated numbers. A sensor dropout and a cool
phone must not look the same.

Usage:
  python3 probe_cooler.py --config fan_off --ambient 24.5 --phase idle
  python3 probe_cooler.py --config fan_on  --ambient 24.5 --phase idle
  python3 probe_cooler.py --config fan_on  --ambient 24.5 --phase load
  python3 probe_cooler.py --selftest
  python3 probe_cooler.py --sabotage-gate --config fan_off --ambient 0 --phase idle
"""
import argparse
import json
import os
import sys
import time

import numpy as np

THRESHOLD_C = 36.5          # defense profile temp_threshold, core/params_sector.py
SECTOR = "defense"


# ----------------------------------------------------------------- instrument
def build_monitor(force_sim=False):
    from core.hardware_link import ThermalMonitor
    mon = ThermalMonitor()
    if force_sim:
        mon.sim = True          # sabotage: prove G can fail
    return mon


def gate_check(mon, where):
    """G. Returns None if the instrument is real; an error string otherwise."""
    if mon.sim:
        return f"GATE FAIL at {where}: ThermalMonitor is in SIMULATION mode."
    if not mon.zones:
        return f"GATE FAIL at {where}: no compute zones bound."
    return None


def read_zones(mon):
    """Per-zone reading so the hottest zone can be named, not just its value."""
    out = {}
    for path in mon.zones:
        name = path.split("/")[-2]
        v = mon._read_one(path)
        if v is not None:
            out[name] = v
    return out


def sample_idle(mon, duration, interval=1.0):
    samples = []
    t_end = time.time() + duration
    while time.time() < t_end:
        zones = read_zones(mon)
        err = gate_check(mon, "sampling")
        if err:
            return None, err
        if not zones:
            return None, "GATE FAIL at sampling: every bound zone returned None."
        hot = max(zones.items(), key=lambda kv: kv[1])
        samples.append({"t": round(time.time(), 3), "max_c": hot[1], "zone": hot[0]})
        time.sleep(interval)
    return samples, None


# ---------------------------------------------------------------- conditions
def run_idle(mon, cfg_name, duration):
    samples, err = sample_idle(mon, duration, 1.0)
    if err:
        return None, err
    vals = [s["max_c"] for s in samples]
    hottest = max(samples, key=lambda s: s["max_c"])
    stats = {
        "n_samples": len(vals),
        "max_c": max(vals),
        "min_c": min(vals),
        "median_c": float(np.median(vals)),
        "hottest_zone": hottest["zone"],
        "all_below_threshold": all(v < THRESHOLD_C for v in vals),
    }
    if cfg_name == "fan_off":
        stats["condition"] = "c1"
        stats["registered"] = f"max compute zone > {THRESHOLD_C} C"
        stats["verdict"] = "PASS" if stats["max_c"] > THRESHOLD_C else "FAIL"
        if stats["verdict"] == "FAIL":
            stats["consequence"] = ("c2 and c3 are VOID: no thermal deficit "
                                    "exists to close, so the cooler cannot be "
                                    "credited with closing one.")
    else:
        stats["condition"] = "c2"
        stats["registered"] = (f"max compute zone < {THRESHOLD_C} C sustained "
                               f"for {duration}s")
        stats["verdict"] = "PASS" if stats["all_below_threshold"] else "FAIL"
    return stats, None


def run_load(mon, cycles=20):
    """c3 — real governor, not a reimplementation. VeritasGate.evaluate()."""
    from core.params import RuntimeConfig
    from core.veritas_gate import VeritasGate
    cfg = RuntimeConfig(SECTOR)
    gate = VeritasGate(cfg, mon)

    halts, temps = 0, []
    rng = np.random.default_rng(42)
    for _ in range(cycles):
        A = rng.normal(0, 1.0, (256, 256))
        _ = A @ A.T                                    # thermal load
        _aT, _dG, is_safe, T = gate.evaluate()
        err = gate_check(mon, "load cycle")
        if err:
            return None, err
        temps.append(T)
        if not is_safe:
            halts += 1
    stats = {
        "condition": "c3",
        "registered": "VeritasGate does not HALT under load (EXPECTED TO FAIL)",
        "cycles": cycles,
        "halts": halts,
        "max_c": max(temps),
        "min_c": min(temps),
        "threshold_c": cfg.temp_threshold,
        "verdict": "PASS" if halts == 0 else "FAIL",
    }
    stats["as_registered"] = ("CONFIRMED (failure was predicted)"
                              if halts > 0 else "SURPRISE (predicted failure did not occur)")
    return stats, None


# ------------------------------------------------------------------ selftest
def selftest():
    """Prove G returns BOTH verdicts and that c1/c2 track the data."""
    import tempfile
    import core.hardware_link as hw
    ok = True

    def fake_tree(root, temps_mc):
        for i, mc in enumerate(temps_mc):
            d = os.path.join(root, f"thermal_zone{i}")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "type"), "w").write(f"cpu-{i}\n")
            open(os.path.join(d, "temp"), "w").write(f"{mc}\n")

    with tempfile.TemporaryDirectory() as root:
        fake_tree(root, [40000, 30000, -273000])   # incl. a dead sensor
        orig_base, orig_android = hw.THERMAL_BASE, hw.ThermalMonitor._is_android
        hw.THERMAL_BASE = root
        hw.ThermalMonitor._is_android = lambda self: True
        try:
            mon = hw.ThermalMonitor()
            # T1: gate passes on a real tree
            t1 = gate_check(mon, "selftest") is None
            print(f"  T1 gate PASSES on real zones          : {'ok' if t1 else 'FAIL'}")
            ok &= t1
            # T2: dead sensor filtered, max is 40.0 not -273
            z = read_zones(mon)
            t2 = (max(z.values()) == 40.0) and len(z) == 2
            print(f"  T2 -273 sensor filtered, max = 40.0 C : {'ok' if t2 else 'FAIL'}")
            ok &= t2
            # T3: c1 PASSES at 40 C (hot phone, deficit exists)
            r, _ = run_idle(mon, "fan_off", 1)
            t3 = r["verdict"] == "PASS"
            print(f"  T3 c1 PASS at 40.0 C                  : {'ok' if t3 else 'FAIL'}")
            ok &= t3
            # T4: c2 FAILS at 40 C (cooler did not close it)
            r, _ = run_idle(mon, "fan_on", 1)
            t4 = r["verdict"] == "FAIL"
            print(f"  T4 c2 FAIL at 40.0 C                  : {'ok' if t4 else 'FAIL'}")
            ok &= t4
            # T5: same instrument, cold tree -> verdicts INVERT
            fake_tree(root, [30000, 29000, -273000])
            mon2 = hw.ThermalMonitor()
            r1, _ = run_idle(mon2, "fan_off", 1)
            r2, _ = run_idle(mon2, "fan_on", 1)
            t5 = (r1["verdict"] == "FAIL") and (r2["verdict"] == "PASS")
            print(f"  T5 cold tree inverts c1/c2            : {'ok' if t5 else 'FAIL'}")
            ok &= t5
            # T6: gate FAILS when forced to sim — the sabotage direction
            mon2.sim = True
            t6 = gate_check(mon2, "selftest") is not None
            print(f"  T6 gate FAILS in simulation mode      : {'ok' if t6 else 'FAIL'}")
            ok &= t6
        finally:
            hw.THERMAL_BASE = orig_base
            hw.ThermalMonitor._is_android = orig_android

    print(f"\nSELFTEST: {'6/6 PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["fan_off", "fan_on"])
    ap.add_argument("--ambient", type=float, help="ambient room temperature, C")
    ap.add_argument("--phase", choices=["idle", "load"], default="idle")
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sabotage-gate", action="store_true",
                    help="force simulation mode; G must exit 1")
    ap.add_argument("--out", default="p13_results.json")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if a.config is None or a.ambient is None:
        print("ERROR: --config and --ambient are both required.")
        print("Ambient is not optional: c1 and c2 are not comparable without it.")
        return 2

    print(f"=== P13 cooler probe | config={a.config} phase={a.phase} "
          f"ambient={a.ambient} C ===")
    mon = build_monitor(force_sim=a.sabotage_gate)

    err = gate_check(mon, "startup")
    if err:
        print(err)
        print("No condition reported in either direction. Exit 1.")
        return 1
    print(f"[G] instrument OK — {len(mon.zones)} zones bound")

    if a.phase == "idle":
        stats, err = run_idle(mon, a.config, a.duration)
    else:
        if a.config != "fan_on":
            print("ERROR: c3 is registered fan ON only.")
            return 2
        stats, err = run_load(mon, a.cycles)

    if err:
        print(err)
        print("No condition reported in either direction. Exit 1.")
        return 1

    record = {
        "claim": "P13",
        "hardware_config": a.config,
        "ambient_c": a.ambient,
        "sector": SECTOR,
        "threshold_c": THRESHOLD_C,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **stats,
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    with open(a.out, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"\nappended to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
