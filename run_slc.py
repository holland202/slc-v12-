#!/usr/bin/env python3
import sys, time
import numpy as np
from core.params import RuntimeConfig
from core.hardware_link import ThermalMonitor
from core.veritas_gate import VeritasGate
from core.sic import SICManifold
from core.vest import VESTunnel
from core.ume import UmbraManifoldEngine

def main():
    sector = sys.argv[1] if len(sys.argv) > 1 else "defense"
    cfg = RuntimeConfig(sector)
    monitor = ThermalMonitor()
    gate = VeritasGate(cfg, monitor)
    sic = SICManifold(dim=64, rank=8)
    vest = VESTunnel(fidelity_threshold=4.5)
    ume = UmbraManifoldEngine(T_0=cfg.temp_threshold)
    
    print(f"--- SLC v12.1 ACTIVE [{sector.upper()} MODE | T_LIMIT={cfg.temp_threshold}C] ---")
    
    rng = np.random.default_rng(42)
    for step in range(1, 4):
        aT, dG, is_safe, T = gate.evaluate()
        if not is_safe:
            print(f"[CYCLE {step:02d}] HALT: T={T:.2f}C | Threshold exceeded.")
            continue
            
        X_t = rng.normal(0, 0.5, 64)
        X_explored, mode = ume.explore(X_t, T)
        
        is_authentic, distance = vest.authenticate(X_explored, sic.U, sic.V)
        if is_authentic:
            w_t = sic.scar_update(X_explored, rng.normal(0, 0.5, 64), 0.3)
            print(f"[CYCLE {step:02d}] SUCCESS: T={T:.2f}C | ScarWeight={w_t:.6f}")
        else:
            print(f"[CYCLE {step:02d}] REJECTED: Distance {distance:.4f}")

if __name__ == "__main__":
    main()
