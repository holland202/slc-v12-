"""Sector thermal profiles — SLC v12"""

SECTOR_THERMAL_PROFILES = {
    "healthcare": {"T_warn":36.0,"T_throttle":38.5,"T_critical":40.0,"T_recovery":34.0,"duty_cycle_warn":0.5,"duty_cycle_throttle":0.2},
    "defense":    {"T_warn":43.0,"T_throttle":48.0,"T_critical":52.0,"T_recovery":38.0,"duty_cycle_warn":0.7,"duty_cycle_throttle":0.3},
    "research":   {"T_warn":43.0,"T_throttle":48.0,"T_critical":50.0,"T_recovery":38.0,"duty_cycle_warn":0.7,"duty_cycle_throttle":0.3},
    "edge":       {"T_warn":40.0,"T_throttle":44.0,"T_critical":48.0,"T_recovery":36.0,"duty_cycle_warn":0.6,"duty_cycle_throttle":0.2},
    "desktop":    {"T_warn":65.0,"T_throttle":75.0,"T_critical":85.0,"T_recovery":55.0,"duty_cycle_warn":0.9,"duty_cycle_throttle":0.7},
}

def get_sector_thermal(sector="research"):
    return SECTOR_THERMAL_PROFILES.get(sector, SECTOR_THERMAL_PROFILES["research"]).copy()
