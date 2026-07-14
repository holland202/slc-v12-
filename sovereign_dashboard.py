#!/usr/bin/env python3
"""
sovereign_dashboard.py - Live terminal dashboard for SLC v12.0

Revision 2: the previous version printed hardcoded strings
("THERMAL: 34.2C STABLE", "MANIFOLD COHERENCE: 99.997%") and a
random.random() jitter loop labeled "STABLE" regardless of value. It
imported nothing from core and reflected no real system state.

This version runs the real unified loop's modules and prints numbers
that are actually computed from them each step.
"""
import sys
import time
import numpy as np
from core.params import RuntimeConfig
from core.hardware_link import ThermalMonitor
from core.veritas_gate import VeritasGate
from core.sic import SICManifold
from core.vest import VESTunnel
from core.ume import UmbraManifoldEngine


def main():
    sector = sys.argv[1] if len(sys.argv) > 1 else "research"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    cfg = RuntimeConfig(sector)
    monitor = ThermalMonitor()
    gate = VeritasGate(cfg, monitor)
    sic = SICManifold(dim=64, rank=8)
    vest = VESTunnel(fidelity_threshold=6.0)
    ume = UmbraManifoldEngine(T_0=cfg.temp_threshold)
    rng = np.random.default_rng(1)

    print("SOVEREIGN LOGIC CORE v12.0 - LIVE DASHBOARD")
    print("=" * 70)
    print(f"Sector: {sector.upper()} | Critical limit: {cfg.temp_critical}C "
          f"| Max rank: {cfg.max_rank}")
    print("=" * 70)

    for i in range(steps):
        aT, dG, is_safe, T = gate.evaluate()
        status = "STABLE" if is_safe else "HALTED"

        if is_safe:
            X_t = rng.normal(0, 0.5, 64)
            grad_U = rng.normal(0, 0.1, 64)
            X_explored = ume.langevin_step(X_t, grad_U, T)
            is_authentic, distance = vest.authenticate(X_explored, sic.U, sic.V)
            A_t = rng.normal(0, 0.5, 64)
            H_x = float(np.var(X_explored)) / 4.0
            if is_authentic:
                w_t = sic.scar_update(X_explored, A_t, H_x)
            else:
                w_t = 0.0
        else:
            distance, w_t = float("nan"), 0.0

        print(
            f"STEP {i:03d} | T={T:6.2f}C | dG={dG:8.4f} | "
            f"scar_w={w_t:.6f} | scars={sic.scars_formed:3d} | "
            f"vest_dist={distance:6.3f} | {status}"
        )
        time.sleep(0.15)

    print()
    print(f"Session complete. Veritas commits/rejects: {gate.commits}/{gate.rejects} | "
          f"VEST passed/blocked: {vest.tunnels_passed}/{vest.tunnels_blocked} | "
          f"Total scars formed: {sic.scars_formed}")


if __name__ == "__main__":
    main()
