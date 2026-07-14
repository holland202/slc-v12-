#!/usr/bin/env python3
"""
core/hardware_link.py — Substrate Sensor Fusion
Directly polls Snapdragon sysfs for thermal and hardware telemetry.
"""
import os
import numpy as np
from typing import Optional

class ThermalMonitor:
    ZONES = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone2/temp",
        "/sys/class/power_supply/battery/temp",
    ]
    
    def __init__(self):
        self.sim = not self._is_android()
        self._sim_temp = 35.0
        self._path = self._find_zone()
        if self._path:
            ttype = self._read_type()
            print(f"[Hardware Link] Bound to substrate: {self._path} ({ttype})")
        else:
            print("[Hardware Link] Simulation mode active. No physical sysfs found.")
    
    def _is_android(self) -> bool:
        return ("ANDROID_ROOT" in os.environ or
                os.path.exists("/system/bin/app_process") or
                "termux" in os.environ.get("PREFIX", "").lower())
    
    def _find_zone(self) -> Optional[str]:
        for p in self.ZONES:
            if os.path.exists(p):
                return p
        return None
    
    def _read_type(self) -> str:
        tpath = self._path.replace("/temp", "/type")
        try:
            with open(tpath) as f:
                return f.read().strip()
        except:
            return "unknown"
    
    def read(self) -> float:
        """Returns the current hardware temperature in Celsius."""
        if self.sim or not self._path:
            self._sim_temp += 0.005 * (1 + 0.1 * np.random.randn())
            self._sim_temp = max(30.0, min(self._sim_temp, 45.0))
            return self._sim_temp
        try:
            with open(self._path) as f:
                v = int(f.read().strip())
            return v / 10.0 if "battery" in self._path else v / 1000.0
        except:
            self.sim = True
            return self.read()
