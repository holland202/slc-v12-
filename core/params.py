#!/usr/bin/env python3
"""
core/params.py — Unified Runtime Configurations
Coordinates selected sector profiles with baseline variables.
"""
import sys
from core.params_sector import SECTOR_PROFILES

class RuntimeConfig:
    def __init__(self, sector_name: str = "research"):
        if sector_name not in SECTOR_PROFILES:
            print(f"[PARAMS-ERROR] Unknown sector: {sector_name}. Defaulting to 'research'.")
            sector_name = "research"
            
        self.profile = SECTOR_PROFILES[sector_name]
        
        # Coupled Invariants
        self.dim = 64
        self.mass = 1.0
        self.dt = 0.01
        self.eta = 2.0
        self.dH = -0.1
        self.dS = 0.02
        
        # Pull parameters dynamically from bounded sector
        self.temp_threshold = self.profile.temp_threshold
        self.temp_critical = self.profile.temp_critical
        self.max_rank = self.profile.max_rank
        
    def dump(self):
        print(f"--- CONFIGURATION VECTOR BOUND TO [{self.profile.name.upper()}] ---")
        print(f"| Dimension: {self.dim} | Mass: {self.mass} | dt: {self.dt}")
        print(f"| Soft Threshold: {self.temp_threshold}°C")
        print(f"| Critical Limit: {self.temp_critical}°C (ATOMIC_REDUCTION)")
        print(f"| Max Operator Rank: {self.max_rank}")
        print("-" * 50)

if __name__ == "__main__":
    sector = sys.argv[1] if len(sys.argv) > 1 else "research"
    cfg = RuntimeConfig(sector)
    cfg.dump()
