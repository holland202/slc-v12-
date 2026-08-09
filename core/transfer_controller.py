"""
Transfer Controller (TC) & Commit Gate
=======================================
One-way crystallization membrane between GGUF output and SIC state.

The membrane enforces five audits before any scar is written to SIC:
  1. Fisher Sharpness: logit confidence (F_t ≥ 0.85)
  2. Spectral Norm: bounded manifold energy (‖UV^T‖₂ ≤ λ_max)
  3. Rank Preservation: topological invariant check
  4. Geodesic Distance: proposed delta near current manifold
  5. Thermal Coupling: effective temperature supports crystallization

All checks must pass (AND logic). Any failure → rejection + detailed log.
No LLM output ever writes directly to SIC.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class CrystallizationDelta:
    """
    Proposed state change before Commit Gate audit.
    
    Attributes:
        U_delta: Proposed change to U factor
        V_delta: Proposed change to V factor
        logit_variance: Fisher estimate from GGUF
        topology_strain: Estimated manifold deformation cost
        explanation: Human-readable description of the delta
    """
    U_delta: np.ndarray
    V_delta: np.ndarray
    logit_variance: float
    topology_strain: float
    explanation: str = ""


@dataclass
class CommitGateAudit:
    """
    Result of Commit Gate validation.
    
    Attributes:
        passed: All checks passed (go ahead with crystallization)
        fisher_sharpness: Fisher information eigenvalue (≥ 0.85 required)
        spectral_norm: ‖U_new @ V_new.T‖₂
        rank_preserved: rank(U_new @ V_new.T) == rank(U @ V.T)
        geodesic_distance: Approximate distance on Stiefel manifold
        thermal_ok: Effective temperature supports crystallization
        all_checks: Dict of individual check results
        rejection_reason: If passed=False, explanation
    """
    passed: bool
    fisher_sharpness: float
    spectral_norm: float
    rank_preserved: bool
    geodesic_distance: float
    thermal_ok: bool
    all_checks: Dict[str, bool]
    rejection_reason: str = ""


class TransferController:
    """
    One-way membrane + Commit Gate for SIC crystallization.
    
    Enforces all topological, thermodynamic, and information-theoretic
    constraints before any scar update to SIC.
    
    Attributes:
        fisher_threshold: Min eigenvalue of Fisher matrix (default 0.85)
        spectral_norm_max: Max ‖UV^T‖₂ (default 2.0)
        geodesic_distance_max: Max Stiefel distance (default 0.15)
        thermal_multiplier: Scaling for effective temperature (default 1.0)
    """
    
    def __init__(
        self,
        fisher_threshold: float = 0.85,
        spectral_norm_max: float = 2.0,
        geodesic_distance_max: float = 0.15,
        thermal_multiplier: float = 1.0,
    ):
        """
        Args:
            fisher_threshold: Min Fisher eigenvalue for high confidence.
            spectral_norm_max: Energy bound on manifold.
            geodesic_distance_max: Max Stiefel distance from current state.
            thermal_multiplier: Scale effective temperature (>1 = more permissive).
        """
        self.fisher_threshold = fisher_threshold
        self.spectral_norm_max = spectral_norm_max
        self.geodesic_distance_max = geodesic_distance_max
        self.thermal_multiplier = thermal_multiplier
        
        # Diagnostics
        self.total_submissions = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.rejection_reasons = {}  # Counter by reason
    
    def draft_delta(
        self,
        gguf_output: Dict[str, Any],
        sic_state: Any,
    ) -> Optional[CrystallizationDelta]:
        """
        Draft a proposed state change from GGUF output.
        
        This is pre-Commit Gate: estimates what the delta would be if accepted.
        Does NOT write to SIC yet.
        
        Args:
            gguf_output: Dict with keys:
                - "success": bool
                - "text": str (generated text)
                - "logit_variance": float (Fisher-estimated confidence)
                - Optional: "logits": np.ndarray
            sic_state: Current SIC object (with .U, .V, .update() method)
        
        Returns:
            CrystallizationDelta object, or None if GGUF failed.
        """
        if not gguf_output.get("success", False):
            return None
        
        logit_variance = gguf_output.get("logit_variance", 0.7)
        
        # Estimate the update (dry run on SIC without writing)
        try:
            # Get current state
            U_current = np.asarray(sic_state.U, dtype=np.float32)
            V_current = np.asarray(sic_state.V, dtype=np.float32)
            
            # Simulate a scar: small rank-1 update
            # (In real SIC.update(), this would be formalized via your QR retraction)
            text = gguf_output.get("text", "")
            x = self._text_to_manifold_vector(text, dim=U_current.shape[0])
            
            # Estimate U_delta as a small perturbation
            # (In production, compute from SIC's internal scar logic)
            alpha = 0.01  # step size (tunable)
            # FIX A (shape): the original wrote (d,1) @ ((k,d) @ (1,d)), which
            # cannot broadcast and raised on every call, so draft_delta always
            # returned None and the Commit Gate rejected 100% of cycles.
            # A rank-1 delta on a (d,k) factor is outer(x, V^T x).
            U_delta = alpha * np.outer(x, V_current.T @ x)
            V_delta = alpha * np.outer(x, U_current.T @ x)
            
            # Estimate topological strain (Frobenius norm of delta)
            topology_strain = float(
                np.linalg.norm(U_delta, 'fro') + np.linalg.norm(V_delta, 'fro')
            ) / (np.linalg.norm(U_current, 'fro') + 1e-10)
            topology_strain = min(1.0, topology_strain)
            
            explanation = f"GGUF scar from {len(text)} chars, logvar={logit_variance:.3f}"
            
            return CrystallizationDelta(
                U_delta=U_delta,
                V_delta=V_delta,
                logit_variance=logit_variance,
                topology_strain=topology_strain,
                explanation=explanation,
            )
        
        except Exception as e:
            print(f"[TransferController] Warning: draft_delta failed: {e}")
            return None
    
    def commit_gate_audit(
        self,
        delta: CrystallizationDelta,
        sic_state: Any,
        cryst_memory: Any,
    ) -> CommitGateAudit:
        """
        Audit a proposed delta against all five Commit Gate criteria.
        
        Args:
            delta: CrystallizationDelta from draft_delta()
            sic_state: Current SIC state
            cryst_memory: CrystallizationMemory for thermal coupling
        
        Returns:
            CommitGateAudit with detailed results for all checks.
        """
        self.total_submissions += 1
        
        U_current = np.asarray(sic_state.U, dtype=np.float32)
        V_current = np.asarray(sic_state.V, dtype=np.float32)
        U_proposed = U_current + delta.U_delta
        V_proposed = V_current + delta.V_delta
        
        all_checks = {}
        
        # --- Check 1: Fisher Sharpness ---
        fisher_sharpness = delta.logit_variance
        check_fisher = fisher_sharpness >= self.fisher_threshold
        all_checks["fisher_sharpness"] = check_fisher
        
        # --- Check 2: Spectral Norm ---
        try:
            manifold_current = U_current @ V_current.T
            manifold_proposed = U_proposed @ V_proposed.T
            spectral_norm = float(np.linalg.norm(manifold_proposed, ord=2))
            check_spectral = spectral_norm <= self.spectral_norm_max
        except:
            spectral_norm = np.inf
            check_spectral = False
        all_checks["spectral_norm"] = check_spectral
        
        # --- Check 3: Rank Preservation ---
        try:
            rank_current = np.linalg.matrix_rank(U_current @ V_current.T, tol=1e-6)
            rank_proposed = np.linalg.matrix_rank(U_proposed @ V_proposed.T, tol=1e-6)
            rank_preserved = (rank_current == rank_proposed)
        except:
            rank_preserved = False
        all_checks["rank_preserved"] = rank_preserved
        
        # --- Check 4: Geodesic Distance ---
        try:
            # Approximate Stiefel distance via Frobenius norm
            # Full formula: d(U, U') = ‖arccos(σ_min(U^T @ U'))‖_F
            # Approximation: d(U, U') ≈ ‖U - U'‖_F / (1 + ‖U - U'‖_F)
            frob_diff = np.linalg.norm(U_proposed - U_current, 'fro')
            frob_diff += np.linalg.norm(V_proposed - V_current, 'fro')
            geodesic_distance = frob_diff / (1.0 + frob_diff)
            check_geodesic = geodesic_distance <= self.geodesic_distance_max
        except:
            geodesic_distance = np.inf
            check_geodesic = False
        all_checks["geodesic_distance"] = check_geodesic
        
        # --- Check 5: Thermal Coupling ---
        # Effective temperature must be high enough to allow crystallization
        # T_eff(t) = T_base * exp(-k * Π_t)
        # where Π_t = cumulative_crystallization_pressure
        try:
            pressure = cryst_memory.cumulative_pressure(decay_factor=0.95)
            T_base = 0.8  # Baseline temperature
            k = 0.5  # Decay constant (tunable)
            T_effective = T_base * np.exp(-k * pressure)
            T_min_required = 0.3 * self.thermal_multiplier
            thermal_ok = T_effective >= T_min_required
        except:
            T_effective = 0.0
            thermal_ok = False
        all_checks["thermal_coupling"] = thermal_ok
        
        # --- Final Decision: ALL checks must pass ---
        passed = all(all_checks.values())
        
        rejection_reason = ""
        if not passed:
            failed_checks = [k for k, v in all_checks.items() if not v]
            rejection_reason = f"Commit Gate rejections: {', '.join(failed_checks)}"
        
        if passed:
            self.total_accepted += 1
        else:
            self.total_rejected += 1
            self.rejection_reasons[rejection_reason] = self.rejection_reasons.get(rejection_reason, 0) + 1
        
        return CommitGateAudit(
            passed=passed,
            fisher_sharpness=fisher_sharpness,
            spectral_norm=spectral_norm,
            rank_preserved=rank_preserved,
            geodesic_distance=geodesic_distance,
            thermal_ok=thermal_ok,
            all_checks=all_checks,
            rejection_reason=rejection_reason,
        )
    
    def crystallize(
        self,
        delta: CrystallizationDelta,
        sic_state: Any,
        cryst_memory: Any,
    ) -> Tuple[bool, CommitGateAudit]:
        """
        Execute full pipeline: Commit Gate audit → SIC update (if passed).
        
        This is the ONE place where SIC is written (one-way membrane).
        
        Args:
            delta: Proposed crystallization delta
            sic_state: SIC object to update
            cryst_memory: History buffer to record outcome
        
        Returns:
            (success, audit_result)
            where success=True only if all Commit Gate checks passed AND SIC.update() succeeded.
        """
        # Run Commit Gate
        audit = self.commit_gate_audit(delta, sic_state, cryst_memory)
        
        if not audit.passed:
            # Log rejection
            cryst_memory.record_crystallization(
                logit_variance=delta.logit_variance,
                topological_strain=delta.topology_strain,
                was_rejected=True,
            )
            return False, audit
        
        # Gate passed: attempt to write to SIC
        try:
            # Reconstruct the scar input from delta
            # (In production, this would use SIC's internal protocol)
            x = np.random.randn(100)  # Placeholder (real: recover from delta)
            
            # Call SIC.update()
            sic_state.update(x, alpha=0.01)
            
            # Log successful crystallization
            cryst_memory.record_crystallization(
                logit_variance=delta.logit_variance,
                topological_strain=delta.topology_strain,
                was_rejected=False,
            )
            
            return True, audit
        
        except Exception as e:
            print(f"[TransferController] Warning: SIC.update() failed: {e}")
            cryst_memory.record_crystallization(
                logit_variance=delta.logit_variance,
                topological_strain=delta.topology_strain,
                was_rejected=True,
            )
            return False, audit
    
    def _text_to_manifold_vector(self, text: str, dim: int) -> np.ndarray:
        """
        Convert text to a manifold-embedded vector (placeholder).
        
        In production, this would use the tokenizer + embedding layer.
        For now: hash text → seed RNG → draw from manifold.
        
        Args:
            text: Input text
            dim: Target dimension
        
        Returns:
            np.ndarray of shape (dim,)
        """
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vector = rng.randn(dim)
        vector /= np.linalg.norm(vector) + 1e-10
        return vector
    
    def state_summary(self) -> Dict[str, float]:
        """
        Diagnostic summary of transfer controller state.
        """
        accept_rate = (
            self.total_accepted / self.total_submissions
            if self.total_submissions > 0
            else 0.0
        )
        return {
            "total_submissions": self.total_submissions,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "accept_rate": accept_rate,
            "fisher_threshold": self.fisher_threshold,
            "spectral_norm_max": self.spectral_norm_max,
            "geodesic_distance_max": self.geodesic_distance_max,
            "thermal_multiplier": self.thermal_multiplier,
        }
