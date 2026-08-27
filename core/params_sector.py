#!/usr/bin/env python3
"""
core/params_sector.py — Sector-Specific Thermodynamic Profiles
Defines operational boundaries under a 38.5°C POLICY budget.

This budget is NOT a hardware limit. Vendor trip points read from sysfs
on this device (see diagnose_zones.py, commit 9233c76) are: first passive
throttle 105000, then 118000/120000, hot 125000, ceiling 135000 -- all in
millidegrees, identical across the cpu-0 and cpu-1 clusters. Measured peaks
of 96-103°C sit below the first trip, i.e. the hardware regards that range
as normal operation. 38.5°C is a chosen operating budget (battery wear,
skin temperature, sustained-load headroom); its basis is not recorded here
and it is roughly 66°C below the first vendor intervention.
"""
from dataclasses import dataclass

@dataclass
class SectorProfile:
    name: str
    temp_threshold: float  # °C: Soft limit / Throttling trigger
    temp_critical: float   # °C: Hard limit / ATOMIC_REDUCTION trigger
    max_rank: int          # Max matrix rank allocation
    data_integrity: float  # Target fidelity bound
    latency_target: float  # ms

SECTOR_PROFILES = {
    "healthcare": SectorProfile(
        name="healthcare",
        temp_threshold=32.0,
        temp_critical=34.0,
        max_rank=8,
        data_integrity=1.00,
        latency_target=45.0
    ),
    "edge": SectorProfile(
        name="edge",
        temp_threshold=34.0,
        temp_critical=35.5,
        max_rank=10,
        data_integrity=0.75,
        latency_target=55.0
    ),
    "research": SectorProfile(
        name="research",
        temp_threshold=35.5,
        temp_critical=37.0,
        max_rank=16,
        data_integrity=0.70,
        latency_target=28.0
    ),
    "defense": SectorProfile(
        name="defense",
        temp_threshold=36.5,
        temp_critical=38.0,
        max_rank=12,
        data_integrity=0.85,
        latency_target=32.0
    ),
    "desktop": SectorProfile(
        name="desktop",
        temp_threshold=38.0,
        temp_critical=38.5,  # Top of the policy budget (NOT a hardware ceiling)
        max_rank=32,
        data_integrity=0.60,
        latency_target=18.0
    )
}
