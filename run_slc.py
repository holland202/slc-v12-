#!/usr/bin/env python3
"""
run_slc.py — Sovereign Logic Core v12.0 Unified Orchestrator
Binds the Governor, Memory, Authenticator, and Explorer into a single closed loop.
"""
import time, sys
import numpy as np
from core.params import RuntimeConfig
from core.hardware_link import ThermalMonitor
from core.veritas_gate import VeritasGate
from core.sic import SICManifold
from core.vest import VESTunnel
from core.ume import UmbraManifoldEngine

def main():
    # Default to research sector if not specified
    sector = sys.argv[1] if len(sys.argv) > 1 else "research"
    cfg = RuntimeConfig(sector)
    
    # Initialize all 4 core modules
    monitor = ThermalMonitor()
    gate = VeritasGate(cfg, monitor)
    sic = SICManifold(dim=64, rank=8)
    vest = VESTunnel(fidelity_threshold=4.5)
    ume = UmbraManifoldEngine(T_0=cfg.temp_threshold)
    
    print("=" * 60)
    print(f"SOVEREIGN LOGIC CORE v12.0 — UNIFIED LOOP [{sector.upper()}]")
    print("=" * 60)
    
    steps = 5
    rng = np.random.default_rng(42)
    
    for step in range(1, steps + 1):
        print(f"\n[CYCLE {step:02d}] INITIATING...")
        
        # 1. GOVERNANCE: Poll substrate and check Gibbs mandate
        aT, dG, is_safe, T = gate.evaluate()
        if not is_safe:
            print(f"  -> [VERITAS] HALT: T={T:.2f}°C | dG={dG:.4f}. Hardware lock engaged.")
            time.sleep(1.0)
            continue
        print(f"  -> [VERITAS] PASS: T={T:.2f}°C | dG={dG:.4f}")

        # 2. EXPLORATION: Generate latent vector via Langevin diffusion
        X_t = rng.normal(0, 0.5, 64)
        grad_U = rng.normal(0, 0.1, 64)
        X_explored = ume.langevin_step(X_t, grad_U, T)
        print(f"  -> [UME] EXPLORED: Diffusion Applied (λ norm: {np.linalg.norm(X_explored):.4f})")
        
        # 3. AUTHENTICATION: Check against topological holes
        is_authentic, distance = vest.authenticate(X_explored, sic.U, sic.V)
        if not is_authentic:
            print(f"  -> [VEST] BLOCKED: Distance {distance:.4f} > Threshold. Topological Hole detected.")
            continue
        print(f"  -> [VEST] AUTHENTICATED: Distance {distance:.4f}")
        
        # 4. MEMORY: Crystallize the state into the manifold
        A_t = rng.normal(0, 0.5, 64) # Attractor target
        entropy_val = 0.5 # Simulated entropy
        w_t = sic.scar_update(X_explored, A_t, entropy_val)
        print(f"  -> [SIC] SCAR FORMED: Weight {w_t:.6f} | Total Scars: {sic.scars_formed}")
        
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("UNIFIED LOOP DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
