#!/usr/bin/env python3
"""
core/params.py - Unified Runtime Configurations
Coordinates selected sector profiles with baseline dynamics constants.

Calibration: if thermal_calibration.json exists (written by
calibrate_thermal.py), its per-sector temp_threshold/temp_critical
override the SECTOR_PROFILES defaults below. This lets real device
data feed into the system without deleting the sector structure that
the README, tests, and every other module depend on.
"""
import sys
import os
import json
from core.params_sector import SECTOR_PROFILES

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "..", "thermal_calibration.json")


class RuntimeConfig:
    def __init__(self, sector_name: str = "research"):
        if sector_name not in SECTOR_PROFILES:
            print(f"[PARAMS-ERROR] Unknown sector: {sector_name}. Defaulting to 'research'.")
            sector_name = "research"

        self.profile = SECTOR_PROFILES[sector_name]

        # Coupled invariants used by veritas_gate.py -- do not remove
        # any of these without checking every module that reads
        # RuntimeConfig, per CONTRIBUTING.md's "Surgical Changes" rule.
        self.dim = 64
        self.mass = 1.0
        self.dt = 0.01
        self.eta = 2.0
        self.dH = -0.1
        self.dS = 0.02

        # Defaults from the sector profile
        self.temp_threshold = self.profile.temp_threshold
        self.temp_critical = self.profile.temp_critical
        self.max_rank = self.profile.max_rank

        # Real-device calibration override, if present
        self.calibration_applied = False
        self._apply_calibration(sector_name)

    def _apply_calibration(self, sector_name: str):
        if not os.path.exists(CALIBRATION_FILE):
            return
        try:
            with open(CALIBRATION_FILE, "r") as f:
                cal = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[PARAMS-WARN] thermal_calibration.json unreadable ({e}), using sector defaults.")
            return

        sector_cal = cal.get("sectors", {}).get(sector_name)
        if not sector_cal:
            return

        self.temp_threshold = sector_cal["temp_threshold"]
        self.temp_critical = sector_cal["temp_critical"]
        self.calibration_applied = True
        self._calibration_meta = {
            "baseline_mean": cal.get("baseline_mean"),
            "n_sessions": cal.get("n_sessions"),
            "generated_at": cal.get("generated_at"),
        }

    def dump(self):
        print(f"--- CONFIGURATION VECTOR BOUND TO [{self.profile.name.upper()}] ---")
        print(f"| Dimension: {self.dim} | Mass: {self.mass} | dt: {self.dt}")
        print(f"| Soft Threshold: {self.temp_threshold}C")
        print(f"| Critical Limit: {self.temp_critical}C")
        print(f"| Max Operator Rank: {self.max_rank}")
        if self.calibration_applied:
            print(f"| Source: thermal_calibration.json "
                  f"(baseline_mean={self._calibration_meta['baseline_mean']}, "
                  f"n_sessions={self._calibration_meta['n_sessions']})")
        else:
            print("| Source: core/params_sector.py defaults (no calibration file found)")
        print("-" * 50)


if __name__ == "__main__":
    sector = sys.argv[1] if len(sys.argv) > 1 else "research"
    cfg = RuntimeConfig(sector)
    cfg.dump()
