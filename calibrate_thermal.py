#!/usr/bin/env python3
"""
calibrate_thermal.py - Substrate Baseline Calibration (v2)

Fixes two real problems found in the v1 script:
1. It only ever read thermal_zone0 ("aoss-0" -- the Always-On
   Subsystem, which is not necessarily representative of CPU/GPU
   compute load). This version lists ALL available thermal zones so
   you can pick one that's actually relevant.
2. A single ~60-second run isn't a stable baseline -- two consecutive
   runs in this project produced means of 37.10C and 35.74C, a 1.36C
   spread. This version appends every run to a persistent log and
   computes thresholds from the FULL history, not one session, and
   tells you plainly when you don't have enough sessions yet.

Usage:
    python3 calibrate_thermal.py --list                    # show all zones
    python3 calibrate_thermal.py                            # calibrates MAX across
                                                             # cpu/cpuss/gpuss zones,
                                                             # same set hardware_link.py reads
    python3 calibrate_thermal.py --keywords cpu gpu tsens   # customize the match
"""
import time
import json
import os
import sys
import statistics
import argparse
from datetime import datetime, timezone

# Reuse the exact same zone-discovery logic hardware_link.py uses, so
# calibration measures precisely what read() will return at runtime.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.hardware_link import discover_compute_zones, THERMAL_BASE, DEFAULT_KEYWORDS

LOG_FILE = "thermal_calibration_log.json"
CONFIG_FILE = "thermal_calibration.json"
SAMPLES_PER_SESSION = 60
DELAY = 1.0
MIN_SESSIONS_FOR_TRUST = 3


def list_zones():
    if not os.path.isdir(THERMAL_BASE):
        print("No /sys/class/thermal on this system (not running on the real device, "
              "or thermal sysfs isn't exposed to Termux).")
        return []
    zones = []
    for entry in sorted(os.listdir(THERMAL_BASE)):
        if not entry.startswith("thermal_zone"):
            continue
        type_path = os.path.join(THERMAL_BASE, entry, "type")
        temp_path = os.path.join(THERMAL_BASE, entry, "temp")
        try:
            with open(type_path) as f:
                ztype = f.read().strip()
            with open(temp_path) as f:
                raw = float(f.read().strip())
                temp = raw / 1000.0 if raw > 1000 else raw
            zones.append((entry, ztype, temp))
        except (OSError, ValueError):
            continue
    return zones


def read_one(path: str):
    try:
        with open(path, "r") as f:
            raw = float(f.read().strip())
        if raw < -200 or raw == 0:
            return None  # inactive/unpopulated sensor placeholder
        return raw / 1000.0 if raw > 1000 else raw
    except (OSError, ValueError):
        return None


def read_max(zone_paths):
    readings = [r for r in (read_one(p) for p in zone_paths) if r is not None]
    return max(readings) if readings else None


def run_session(zone_paths, label):
    print(f"Calibrating against MAX of {len(zone_paths)} zone(s) [{label}]. "
          f"Collecting {SAMPLES_PER_SESSION} samples at {DELAY}s intervals "
          f"({SAMPLES_PER_SESSION * DELAY:.0f}s total).")
    print("Let the device sit at whatever activity level you want this baseline "
          "to represent (idle, or under real load -- be consistent about which).")

    temps = []
    for i in range(SAMPLES_PER_SESSION):
        t = read_max(zone_paths)
        if t is not None:
            temps.append(t)
            print(f"Sample {i+1:02d}/{SAMPLES_PER_SESSION}: {t:.2f}C", end="\r")
        time.sleep(DELAY)
    print()
    return temps


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return {"sessions": []}


def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List all thermal zones and exit")
    parser.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS,
                         help=f"Zone-type keywords to match (default: {DEFAULT_KEYWORDS})")
    args = parser.parse_args()

    if args.list:
        zones = list_zones()
        if not zones:
            return
        print(f"{'Zone':<16} {'Type':<20} {'Current temp'}")
        for zone, ztype, temp in zones:
            print(f"{zone:<16} {ztype:<20} {temp:.2f}C")
        print(f"\nDefault calibration will use MAX across zones matching {DEFAULT_KEYWORDS} "
              f"-- the same set core/hardware_link.py reads at runtime. Override with "
              f"--keywords if you want a different set (e.g. --keywords cpu gpu tsens apps).")
        return

    zone_paths = discover_compute_zones(args.keywords)
    if not zone_paths:
        print(f"No zones matched keywords {args.keywords}. Run --list to see what's available.")
        return
    label = ",".join(args.keywords)

    temps = run_session(zone_paths, label)
    if not temps:
        print("Calibration failed. No data collected.")
        return

    session_stats = {
        "zone": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(temps),
        "min": min(temps),
        "max": max(temps),
        "mean": statistics.mean(temps),
        "median": statistics.median(temps),
        "stdev": statistics.stdev(temps) if len(temps) > 1 else 0.0,
    }

    log = load_log()
    log["sessions"].append(session_stats)
    save_log(log)

    sessions_this_zone = [s for s in log["sessions"] if s["zone"] == label]
    n_sessions = len(sessions_this_zone)
    all_means = [s["mean"] for s in sessions_this_zone]
    all_maxes = [s["max"] for s in sessions_this_zone]

    print("\n--- THIS SESSION ---")
    print(f"Mean: {session_stats['mean']:.2f}C | Max: {session_stats['max']:.2f}C | "
          f"StdDev: {session_stats['stdev']:.3f}")

    print(f"\n--- ACROSS ALL {n_sessions} LOGGED SESSION(S) FOR {label} ---")
    print(f"Session means: {[round(m, 2) for m in all_means]}")
    if n_sessions > 1:
        spread = max(all_means) - min(all_means)
        print(f"Spread between session means: {spread:.2f}C")
        if spread > 1.0:
            print(f"WARNING: {spread:.2f}C spread across sessions is large. This zone's "
                  f"reading is noisy or activity-dependent -- thresholds derived from it "
                  f"will drift. Consider a different zone (--list) or expect to "
                  f"re-calibrate periodically.")

    if n_sessions < MIN_SESSIONS_FOR_TRUST:
        print(f"\nOnly {n_sessions}/{MIN_SESSIONS_FOR_TRUST} sessions logged. "
              f"Run this {MIN_SESSIONS_FOR_TRUST - n_sessions} more time(s) "
              f"(at different times / activity levels) before trusting these "
              f"thresholds. Not writing {CONFIG_FILE} yet.")
        return

    baseline_mean = statistics.mean(all_means)
    baseline_max = max(all_maxes)
    # Conservative: threshold = worst observed max + margin, not just mean + margin
    research_threshold = baseline_max + 1.0
    research_critical = baseline_max + 2.0
    defense_threshold = baseline_max + 2.0
    defense_critical = baseline_max + 3.5

    config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "zone": label,
        "baseline_mean": round(baseline_mean, 2),
        "baseline_max_observed": round(baseline_max, 2),
        "n_sessions": n_sessions,
        "sectors": {
            "research": {
                "temp_threshold": round(research_threshold, 2),
                "temp_critical": round(research_critical, 2),
            },
            "defense": {
                "temp_threshold": round(defense_threshold, 2),
                "temp_critical": round(defense_critical, 2),
            },
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{n_sessions} sessions logged -- writing {CONFIG_FILE}.")
    print(json.dumps(config, indent=2))
    print(f"\ncore/params.py will now read this automatically for research/defense "
          f"sectors. healthcare/edge/desktop still use core/params_sector.py "
          f"defaults -- calibrate those the same way if you use them.")


if __name__ == "__main__":
    main()
