#!/usr/bin/env python3
"""
core/hardware_link.py - Substrate Sensor Fusion

Revision 2: the previous version hardcoded a search order
[thermal_zone0, thermal_zone1, thermal_zone2, battery] and used the
FIRST one that existed -- which on this device is always
thermal_zone0 ("aoss-0", an always-on low-power monitoring block),
never the actual CPU/GPU zones where compute heat shows up.

This version auto-discovers every zone whose reported "type" matches
a compute-relevant keyword (cpu, cpuss, gpuss by default) and reads
the MAX across all of them -- the correct approach for a protective
governor, since workload can migrate between cores/clusters and you
want to catch whichever one is hottest, not one arbitrarily chosen
sensor. This must stay consistent with calibrate_thermal.py, which
uses the same keyword-matching and max-across-zones logic so the
numbers calibration produces are the numbers read() will return.
"""
import os
import numpy as np
from typing import List, Optional

THERMAL_BASE = "/sys/class/thermal"
DEFAULT_KEYWORDS = ["cpu", "cpuss", "gpuss"]


def discover_compute_zones(keywords: List[str] = None) -> List[str]:
    """Returns thermal_zone paths whose type matches any keyword."""
    keywords = keywords or DEFAULT_KEYWORDS
    if not os.path.isdir(THERMAL_BASE):
        return []
    matched = []
    for entry in sorted(os.listdir(THERMAL_BASE)):
        if not entry.startswith("thermal_zone"):
            continue
        type_path = os.path.join(THERMAL_BASE, entry, "type")
        temp_path = os.path.join(THERMAL_BASE, entry, "temp")
        try:
            with open(type_path) as f:
                ztype = f.read().strip().lower()
        except OSError:
            continue
        if any(kw in ztype for kw in keywords):
            if os.path.exists(temp_path):
                matched.append(temp_path)
    return matched


class ThermalMonitor:
    def __init__(self, keywords: List[str] = None):
        self.sim = not self._is_android()
        self._sim_temp = 35.0
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.zones = discover_compute_zones(self.keywords) if not self.sim else []

        if self.zones:
            print(f"[Hardware Link] Bound to {len(self.zones)} compute zone(s) "
                  f"matching {self.keywords}: "
                  f"{[z.split('/')[-2] for z in self.zones]}")
        else:
            self.sim = True
            print("[Hardware Link] Simulation mode active. No matching compute "
                  "zones found (or not running on-device).")

    def _is_android(self) -> bool:
        return ("ANDROID_ROOT" in os.environ or
                os.path.exists("/system/bin/app_process") or
                "termux" in os.environ.get("PREFIX", "").lower())

    def _read_one(self, path: str) -> Optional[float]:
        try:
            with open(path) as f:
                raw = float(f.read().strip())
            if raw < -200 or raw == 0:
                # Inactive/unpopulated sensor placeholder (seen on this
                # SoC as exactly -273000 or 0) -- skip it.
                return None
            return raw / 1000.0 if raw > 1000 else raw
        except (OSError, ValueError):
            return None

    def read(self) -> float:
        """Returns the MAX temperature (C) across all bound compute zones."""
        if self.sim or not self.zones:
            self._sim_temp += 0.005 * (1 + 0.1 * np.random.randn())
            self._sim_temp = max(30.0, min(self._sim_temp, 45.0))
            return self._sim_temp

        readings = [r for r in (self._read_one(p) for p in self.zones) if r is not None]
        if not readings:
            self.sim = True
            return self.read()
        return max(readings)
