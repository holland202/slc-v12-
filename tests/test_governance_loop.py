#!/usr/bin/env python3
"""
tests/test_governance_loop.py
Locally verifies the Veritas Gate and Hardware Link.
"""
from core.params import RuntimeConfig
from core.hardware_link import ThermalMonitor
from core.veritas_gate import VeritasGate

def run_test():
    print("--- INITIATING LOCAL GOVERNANCE TEST ---")
    
    # Instantiate with the 'defense' sector to test the 38.0°C critical limit
    cfg = RuntimeConfig("defense") 
    cfg.dump()
    
    monitor = ThermalMonitor()
    gate = VeritasGate(cfg, monitor)
    
    aT, dG, is_safe, T = gate.evaluate()
    
    print("\n[VERITAS DIAGNOSTIC]")
    print(f"Substrate Temp : {T:.2f}°C")
    print(f"Critical Limit : {cfg.temp_critical:.2f}°C")
    print(f"Gibbs Energy   : {dG:.4f} (Must be < 0)")
    print(f"Gate Status    : {'PASS' if is_safe else 'FAIL/THROTTLE'}")
    print(f"Thermal Multi  : {aT:.4f}")
    print("--- TEST COMPLETE ---")

if __name__ == "__main__":
    run_test()
