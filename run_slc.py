#!/usr/bin/env python3
"""
run_slc.py - Sovereign Logic Core v12.0 Unified Orchestrator
Binds all ten core modules into a single closed loop:
Veritas Gate -> UME -> Pre-Inference Gate -> VEST -> SIC -> Crystallization
Memory -> Transfer Controller -> SMA (periodic hyperparameter optimization).

Revision 2: previously sma.py, pre_inference_gate.py, transfer_controller.py,
and crystallization_memory.py existed but were never imported anywhere. This
revision wires all of them into the actual loop.
"""
import time
import sys
import numpy as np
from core.params import RuntimeConfig
from core.hardware_link import ThermalMonitor
from core.veritas_gate import VeritasGate
from core.sic import SICManifold
from core.vest import VESTunnel
from core.ume import UmbraManifoldEngine
from core.pre_inference_gate import PreInferenceGate
from core.transfer_controller import TransferController
from core.crystallization_memory import CrystallizationMemory
from core.sma import SlimeMoldOptimizer


def main():
    sector = sys.argv[1] if len(sys.argv) > 1 else "research"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    cfg = RuntimeConfig(sector)

    # Initialize all ten core modules
    monitor = ThermalMonitor()
    gate = VeritasGate(cfg, monitor)
    sic = SICManifold(dim=64, rank=8)
    # NOTE: fidelity_threshold=4.5 (the original unwired default) combined with
    # PreInferenceGate's rejection_history factor creates a feedback loop: VEST
    # blocks -> rejection_history drops -> pre-gate score drops -> permanent
    # DEFER, with no way to recover since no new crystallizations get recorded
    # once deferred. This interaction was invisible while the modules were never
    # actually connected. 6.0 reduces but does not eliminate the effect --
    # this combination needs real tuning against real data, not this guess.
    vest = VESTunnel(fidelity_threshold=6.0)
    ume = UmbraManifoldEngine(T_0=cfg.temp_threshold)
    pre_gate = PreInferenceGate()
    transfer = TransferController()
    cryst_memory = CrystallizationMemory(window_size=20)
    sma = SlimeMoldOptimizer(n_agents=8)

    print("=" * 60)
    print(f"SOVEREIGN LOGIC CORE v12.0 - UNIFIED LOOP [{sector.upper()}]")
    print("=" * 60)

    rng = np.random.default_rng(42)

    for step in range(1, steps + 1):
        print(f"\n[CYCLE {step:02d}] INITIATING...")

        # 1. GOVERNANCE: poll substrate and check Gibbs mandate
        aT, dG, is_safe, T = gate.evaluate()
        if not is_safe:
            print(f"  -> [VERITAS] HALT: T={T:.2f}C | dG={dG:.4f}. Hardware lock engaged.")
            cryst_memory.record_deferral(reason="thermal_lock")
            time.sleep(1.0)
            continue
        print(f"  -> [VERITAS] PASS: T={T:.2f}C | dG={dG:.4f}")

        # 2. EXPLORATION: generate latent vector via Langevin diffusion
        X_t = rng.normal(0, 0.5, 64)
        grad_U = rng.normal(0, 0.1, 64)
        X_explored = ume.langevin_step(X_t, grad_U, T)
        print(f"  -> [UME] EXPLORED: Diffusion Applied (norm: {np.linalg.norm(X_explored):.4f})")

        # 3. PRE-INFERENCE RISK GATE: composite risk score from history
        prompt = f"exploration cycle {step} sector {sector}"
        gate_pass, risk_score, factors = pre_gate.evaluate(prompt, sic, cryst_memory)
        print(f"  -> [PRE-GATE] score={risk_score:.4f} ({'PASS' if gate_pass else 'DEFER'}) {factors}")
        if not gate_pass:
            cryst_memory.record_deferral(reason="pre_inference_risk")
            continue

        # 4. AUTHENTICATION: check against topological holes
        is_authentic, distance = vest.authenticate(X_explored, sic.U, sic.V)
        if not is_authentic:
            print(f"  -> [VEST] BLOCKED: Distance {distance:.4f} > Threshold. Topological Hole detected.")
            cryst_memory.record_crystallization(
                logit_variance=float(np.var(X_explored)),
                topological_strain=distance / 50.0,
                was_rejected=True,
            )
            continue
        print(f"  -> [VEST] AUTHENTICATED: Distance {distance:.4f}")

        # 5. MEMORY: crystallize the state into the manifold + record history
        A_t = rng.normal(0, 0.5, 64)  # Attractor target
        entropy_val = float(np.var(X_explored)) / 4.0  # proxy entropy from explored state
        w_t = sic.scar_update(X_explored, A_t, entropy_val)
        cryst_memory.record_crystallization(
            logit_variance=float(np.var(X_explored)),
            topological_strain=distance / 50.0,
            was_rejected=False,
        )
        print(f"  -> [SIC] SCAR FORMED: Weight {w_t:.6f} | Total Scars: {sic.scars_formed}")

        # 6. TRANSFER: draft and audit a commit delta before it's allowed to persist
        gguf_output = {"text": prompt}
        delta = transfer.draft_delta(gguf_output, sic)
        audit = transfer.commit_gate_audit(delta, sic, cryst_memory)
        status = "COMMITTED" if audit.passed else f"REJECTED ({audit.rejection_reason})"
        print(f"  -> [TRANSFER] {status} | fisher={audit.fisher_sharpness:.4f} "
              f"spectral={audit.spectral_norm:.4f} geo={audit.geodesic_distance:.4f}")

        # 7. SMA: background hyperparameter optimization (runs every cycle,
        # recommendations are logged; applying them to SIC/VEST live is a
        # future extension, not implemented here)
        best = sma.step(
            vest_distance=distance,
            spectral_entropy=entropy_val * 10,
            thermal_energy=T,
        )
        print(f"  -> [SMA] gen={sma._generation} best_fitness={best.fitness:.4f} "
              f"(alpha={best.alpha:.4f}, beta={best.beta:.4f}, rank={best.rank})")

        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("UNIFIED LOOP DIAGNOSTIC COMPLETE")
    print(f"Veritas commits/rejects: {gate.commits}/{gate.rejects}")
    print(f"VEST passed/blocked: {vest.tunnels_passed}/{vest.tunnels_blocked}")
    print(f"Transfer submissions/accepts: {transfer._submissions}/{transfer._accepts}")
    print(f"Crystallization history: {cryst_memory.state_summary()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
