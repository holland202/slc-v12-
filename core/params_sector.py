#!/usr/bin/env python3
"""
core/params_sector.py — Sector-Specific Thermodynamic Profiles
Defines operational boundaries matching the 38.5°C hardware wall.
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
        temp_critical=38.5,  # Absolute Hardware Ceiling
        max_rank=32,
        data_integrity=0.60,
        latency_target=18.0
    )
}
