#!/usr/bin/env python3
"""
probe_commit_gate_checks.py — P9. Can each Commit Gate check actually refuse?

The Commit Gate runs five checks and admits only if all five pass. P8 proved
check 1 (fisher) flips cleanly at its registered threshold. Checks 2-5 passed at
every input tested during the P8 work, which is not evidence that they CAN
refuse -- it is evidence that nothing tested was extreme enough to find out.

This searches, per check, for an input that makes THAT check return False. Three
outcomes per check, and the distinction matters:

    REFUSES        a refusing input was found; the check is live
    NO INPUT FOUND nothing in the swept range refused. NOT proof of inertness --
                   proof that this sweep did not find one. Candidate inert gate.
    STRUCTURAL     the check is provably unable to refuse, by arithmetic

Exit 0 only if every check either REFUSES or is explicitly declared intentional.
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from core.transfer_controller import TransferController, CrystallizationDelta
from core.sic_enhanced import ScarredIdentityChronicle
from core.crystallization_memory import CrystallizationMemory

RESULTS = []


class Delta:
    """Minimal stand-in matching what commit_gate_audit reads off a delta."""
    def __init__(self, U_delta, V_delta, logit_variance):
        self.U_delta = U_delta
        self.V_delta = V_delta
        self.logit_variance = logit_variance


def fresh(d=64, rank=8, seed=42):
    return ScarredIdentityChronicle(d=d, rank=rank, seed=seed)


def audit(tc, sic, mem, scale, logvar=0.99, rng=None):
    rng = rng or np.random.default_rng(7)
    dU = (rng.standard_normal(sic.U.shape) * scale).astype(np.float32)
    dV = (rng.standard_normal(sic.V.shape) * scale).astype(np.float32)
    return tc.commit_gate_audit(Delta(dU, dV, logvar), sic, mem)


def report(check, verdict, detail):
    RESULTS.append((check, verdict, detail))
    print(f"\n  {check}")
    print(f"    verdict : {verdict}")
    print(f"    detail  : {detail}")


tc = TransferController()
print("Commit Gate thresholds as configured:")
print(f"  fisher_threshold      = {tc.fisher_threshold}")
print(f"  spectral_norm_max     = {tc.spectral_norm_max}")
print(f"  geodesic_distance_max = {tc.geodesic_distance_max}")
print(f"  thermal_multiplier    = {tc.thermal_multiplier}")

# ------------------------------------------------------------------ check 4
# STRUCTURAL ANALYSIS FIRST. geodesic = f/(1+f) is bounded in [0,1), so a
# threshold >= 1.0 would make this check unable to refuse for any input.
gmax = tc.geodesic_distance_max
if gmax >= 1.0:
    report("check 4 geodesic_distance", "STRUCTURAL",
           f"threshold {gmax} >= 1.0 but geodesic = f/(1+f) < 1 always. "
           f"Cannot refuse for any input.")
else:
    f_needed = gmax / (1.0 - gmax)
    found = None
    for scale in (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5):
        a = audit(tc, fresh(), CrystallizationMemory(), scale)
        if not a.all_checks.get("geodesic_distance", True):
            found = (scale, a.geodesic_distance)
            break
    report("check 4 geodesic_distance",
           "REFUSES" if found else "NO INPUT FOUND",
           f"threshold {gmax}; refuses once ||dU||_F+||dV||_F > {f_needed:.4f}; "
           + (f"first refusal at delta scale {found[0]} (d={found[1]:.4f})"
              if found else "no refusal up to scale 0.5"))

# ------------------------------------------------------------------ check 2
found = None
for scale in (1e-3, 1e-2, 0.05, 0.1, 0.3, 1.0, 3.0):
    a = audit(tc, fresh(), CrystallizationMemory(), scale)
    if not a.all_checks.get("spectral_norm", True):
        found = (scale, a.spectral_norm)
        break
report("check 2 spectral_norm",
       "REFUSES" if found else "NO INPUT FOUND",
       f"threshold {tc.spectral_norm_max}; "
       + (f"first refusal at delta scale {found[0]} "
          f"(||UV^T||_2 = {found[1]:.4f})" if found
          else "no refusal up to delta scale 3.0"))

# ------------------------------------------------------------------ check 3
found = None
probes = []
for scale in (1e-3, 1e-2, 0.1, 1.0, 10.0):
    a = audit(tc, fresh(), CrystallizationMemory(), scale)
    probes.append(f"scale {scale}: {'pass' if a.all_checks.get('rank_preserved') else 'REFUSE'}")
    if not a.all_checks.get("rank_preserved", True):
        found = scale
        break
if not found:
    # Rank-deficient start: a rank-1 update to a degenerate operator should
    # change the rank if the check measures anything.
    s = fresh()
    s.V = (s.V * 0.0).astype(np.float32)
    s.V[:, 0] = 1.0
    a = audit(tc, s, CrystallizationMemory(), 0.1)
    probes.append(f"rank-deficient V: {'pass' if a.all_checks.get('rank_preserved') else 'REFUSE'}")
    if not a.all_checks.get("rank_preserved", True):
        found = "rank-deficient start"
report("check 3 rank_preserved",
       "REFUSES" if found else "NO INPUT FOUND",
       "; ".join(probes))

# ------------------------------------------------------------------ check 5
found = None
trace = []
mem = CrystallizationMemory()
for i in range(1, 61):
    mem.record_crystallization(logit_variance=0.99, topological_strain=5.0,
                               was_rejected=False)
    if i % 10 == 0:
        a = audit(tc, fresh(), mem, 1e-3)
        p = mem.cumulative_pressure(decay_factor=0.95)
        trace.append(f"n={i} pressure={p:.3f} "
                     f"{'pass' if a.all_checks.get('thermal_coupling') else 'REFUSE'}")
        if not a.all_checks.get("thermal_coupling", True):
            found = i
            break
report("check 5 thermal_coupling",
       "REFUSES" if found else "NO INPUT FOUND",
       f"T_eff = 0.8*exp(-0.5*pressure) vs T_min = {0.3 * tc.thermal_multiplier}; "
       + "; ".join(trace))

# ------------------------------------------------------------------ summary
print("\n" + "=" * 68)
live = [c for c, v, _ in RESULTS if v == "REFUSES"]
dead = [c for c, v, _ in RESULTS if v != "REFUSES"]
for c, v, _ in RESULTS:
    print(f"  {v:<15} {c}")
print(f"\n{len(live)}/{len(RESULTS)} non-Fisher checks demonstrated a refusing input.")
if dead:
    print("Checks with no demonstrated refusal are CANDIDATE inert gates. A check "
          "that cannot refuse contributes nothing to an all() over five checks, "
          "and should be given a real bound, removed, or declared intentional.")
print("=" * 68)
sys.exit(0 if not dead else 1)
