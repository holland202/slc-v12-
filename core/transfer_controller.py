"""transfer_controller.py — One-Way Crystallization Membrane (Stage 6) · SLC v12"""
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class AuditResult:
    passed: bool
    fisher_sharpness: float
    spectral_norm: float
    rank_preserved: bool
    geodesic_distance: float
    thermal_ok: bool
    all_checks: Dict
    rejection_reason: Optional[str]

class TransferController:
    def __init__(self, fisher_threshold=0.85, spectral_norm_max=2.0,
                 geodesic_distance_max=0.15, thermal_multiplier=1.0):
        self.fisher_threshold      = fisher_threshold
        self.spectral_norm_max     = spectral_norm_max
        self.geodesic_distance_max = geodesic_distance_max
        self.thermal_multiplier    = thermal_multiplier
        self._submissions = 0
        self._accepts     = 0

    def draft_delta(self, gguf_output, sic_state):
        import hashlib
        text = gguf_output.get("text","")
        h    = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4],"big")
        rng  = np.random.default_rng(seed)
        d, r = sic_state.U.shape
        delta = rng.normal(0, 0.01, (d,r)).astype(np.float32)
        delta_obj = type('Delta', (), {})()
        delta_obj.matrix = delta
        delta_obj.topology_strain = float(np.linalg.norm(delta,"fro")) * 0.1
        return delta_obj

    def commit_gate_audit(self, delta, sic_state, cryst_memory) -> AuditResult:
        self._submissions += 1
        U  = sic_state.U
        sv = np.linalg.svd(U, compute_uv=False)

        total = float(np.sum(sv))
        top_k = float(np.sum(sv[:max(1,len(sv)//4)]))
        fisher = top_k / (total + 1e-10)
        eff_thresh = self.fisher_threshold * self.thermal_multiplier
        c1 = fisher >= eff_thresh

        proposed = U + delta.matrix
        spec = float(np.linalg.norm(proposed, ord=2))
        c2 = spec <= self.spectral_norm_max

        rank_before = int(np.sum(sv > 1e-4))
        sv_after    = np.linalg.svd(proposed, compute_uv=False)
        rank_after  = int(np.sum(sv_after > 1e-4))
        c3 = rank_after >= rank_before

        geo = float(np.linalg.norm(delta.matrix,"fro")) / (float(np.linalg.norm(U,"fro"))+1e-10)
        c4 = geo <= self.geodesic_distance_max

        lv = cryst_memory.mean_logit_variance()
        c5 = lv >= 0.60

        all_pass = c1 and c2 and c3 and c4 and c5
        checks = {
            "C1_fisher":   {"value":round(fisher,4), "threshold":eff_thresh, "pass":c1},
            "C2_spectral": {"value":round(spec,4),   "max":self.spectral_norm_max, "pass":c2},
            "C3_rank":     {"before":rank_before,    "after":rank_after, "pass":c3},
            "C4_geodesic": {"value":round(geo,4),    "max":self.geodesic_distance_max, "pass":c4},
            "C5_logit":    {"value":round(lv,4),     "min":0.60, "pass":c5},
        }
        rejection_reason = None if all_pass else next(
            (k for k,v in checks.items() if not v["pass"]), "unknown"
        )
        if all_pass: self._accepts += 1

        return AuditResult(
            passed=all_pass, fisher_sharpness=round(fisher,4),
            spectral_norm=round(spec,4), rank_preserved=c3,
            geodesic_distance=round(geo,4), thermal_ok=c5,
            all_checks=checks, rejection_reason=rejection_reason,
        )

    def state_summary(self):
        return {
            "total_submissions": self._submissions,
            "accept_rate": round(self._accepts/max(1,self._submissions),4),
            "fisher_threshold": self.fisher_threshold,
        }
