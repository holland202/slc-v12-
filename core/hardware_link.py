"""hardware_link.py — SM8750 Hardware Interface · SLC v12"""
import os, glob, time

_SM8750_PRIORITY = [19,14,17,7,18,15,16,8,9,6,5,4,3,2,1,0]

class HardwareLink:
    def __init__(self):
        self._zone = self._find_zone()

    def _find_zone(self):
        base = "/sys/class/thermal"
        try:
            zones = [int(d.replace("thermal_zone",""))
                     for d in os.listdir(base) if d.startswith("thermal_zone")]
        except Exception:
            return 7
        ordered = _SM8750_PRIORITY + [z for z in sorted(zones) if z not in _SM8750_PRIORITY]
        for zone in ordered:
            try:
                raw = int(open(f"{base}/thermal_zone{zone}/temp").read().strip())
                if 20000 <= raw <= 80000:
                    return zone
            except Exception:
                continue
        return 7

    def get_thermal_zone_0(self):
        try:
            raw = int(open(f"/sys/class/thermal/thermal_zone{self._zone}/temp").read().strip())
            return raw / 1000.0
        except Exception:
            return 35.0

    def get_substrate_id(self):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "Hardware" in line:
                        return line.split(":")[-1].strip()
        except Exception:
            pass
        return "SM8750"

    def all_zones(self):
        zones = []
        for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
            try:
                raw = int(open(path).read().strip())
                temp = raw / 1000.0
                if -10 < temp < 200:
                    name = path.split("/")[-2]
                    zones.append((name, temp))
            except Exception:
                continue
        return zones

def get_temperature():
    return HardwareLink().get_thermal_zone_0()

def get_substrate_id():
    return HardwareLink().get_substrate_id()
