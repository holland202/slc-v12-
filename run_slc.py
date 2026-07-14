#!/usr/bin/env python3
"""
run_slc.py — Sovereign Logic Core v12.0 Entry Point
Coordinates the DMIA, Veritas Gate, and thermal subsystem execution.
"""
import time, sys
from core.params import RuntimeConfig
from core.hardware_link import ThermalMonitor
from core.veritas_gate import VeritasGate

def main():
    sector = sys.argv[1] if len(sys.argv) > 1 else "research"
    cfg = RuntimeConfig(sector)
    monitor = ThermalMonitor()
    gate = VeritasGate(cfg, monitor)
    
    print("=" * 55)
    print(f"SOVEREIGN LOGIC CORE v12.0 — [{sector.upper()}] SECTOR")
    print("=" * 55)
    
    steps = 10
    print(f"Initiating {steps}-step operational loop...\n")
    
    for step in range(1, steps + 1):
        aT, dG, is_safe, T = gate.evaluate()
        
        if not is_safe and T >= cfg.temp_critical:
            print(f"[STEP {step:02d}] T={T:.2f}°C | [ATOMIC_REDUCTION] Hardware lock engaged.")
            time.sleep(1.5)  # Pause to allow passive cooling
            continue
            
        if not is_safe:
            print(f"[STEP {step:02d}] T={T:.2f}°C | dG={dG:.4f} | [VERITAS-REJECT] Gibbs mandate violated.")
            time.sleep(1.0)
            continue
            
        print(f"[STEP {step:02d}] T={T:.2f}°C | dG={dG:.4f} | aT={aT:.4f} | [VERITAS-PASS] Manifold update authorized.")
        time.sleep(0.5)

    print("-" * 55)
    print(f"Run complete. Commits: {gate.commits} | Rejects: {gate.rejects}")
    print("=" * 55)

if __name__ == "__main__":
    main()
