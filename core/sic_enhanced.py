"""
Enhanced sic.py with governance integration
============================================
Adds lightweight helper method for PreInferenceGate and Commit Gate
to read SIC state without deep coupling.
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger("slc.sic")


class ScarredIdentityChronicle:
    """
    Scarred Identity Chronicle (SIC) — v12.0 Canonical.
    Enforces Stiefel manifold constraints and spectral clamping to prevent topological holes.
    
    Phase 2 Enhancement: Added get_state_for_gate() for governance layer integration.
    """

    def __init__(self, d: int = 512, rank: int = 64, spectral_bound: float = 3.0):
        self.d = d
        self.rank = rank
        self.spectral_bound = spectral_bound
        
        # Initialize U on Stiefel Manifold (Orthogonal Columns)
        U_init = np.random.randn(d, rank)
        Q, _ = np.linalg.qr(U_init)
        self.U = Q.astype(np.float32)
        # FIX C (root cause of the null manifold): V was initialized to zeros
        # and is never written anywhere in this class, so (V.T @ x) == 0, the
        # update term was identically zero, U never moved and U@V.T stayed
        # exactly 0.0 while scars_admitted counted up. The shipped
        # core/sic.py (SICManifold) initializes BOTH factors randomly; this
        # matches that, rather than inventing a new rule.
        self.V = (np.random.randn(d, rank) * 0.1).astype(np.float32)

        self.mu = np.zeros(d, dtype=np.float32)
        self.scars_admitted = 0

    def _retract_and_clamp(self):
        """
        QR retraction + spectral clamping.
        Ensures U stays orthonormal, V stays bounded.
        """
        Q, R = np.linalg.qr(self.U)
        self.U = Q
        self.V = self.V @ R.T

        v_norm = np.linalg.norm(self.V, ord=2)
        if v_norm > self.spectral_bound:
            self.V *= (self.spectral_bound / v_norm)

    def update(self, x: np.ndarray, alpha: float = 0.01) -> bool:
        """
        Scar formation: rank-1 update to manifest residual.
        
        Args:
            x: Input vector (e.g., from GGUF output embedding)
            alpha: Step size (default 0.01)
        
        Returns:
            bool: True if scar was admitted, False if residual too small
        """
        residual = x - (self.U @ (self.V.T @ x))
        res_norm = np.linalg.norm(residual)
        
        if res_norm < 1e-4:
            return False
            
        delta_u = (residual / res_norm).reshape(-1, 1)
        self.U = self.U + alpha * (delta_u @ (self.V.T @ x).reshape(1, -1))
        
        self.scars_admitted += 1
        self._retract_and_clamp()
        return True

    def get_identity_operator(self) -> np.ndarray:
        """
        Returns the low-rank identity approximation: U @ V.T
        """
        return self.U @ self.V.T

    def state_summary(self) -> Dict[str, Any]:
        """
        High-level summary for diagnostics.
        """
        return {
            "scars_admitted": self.scars_admitted,
            "rank": self.rank,
            "spectral_norm_U": float(np.linalg.norm(self.U, ord=2))
        }

    # ========== NEW: Phase 2 Governance Integration ==========

    def get_state_for_gate(self) -> Dict[str, Any]:
        """
        Lightweight state snapshot for PreInferenceGate and Commit Gate.
        
        Called frequently (once per inference) — must be O(1) or O(d) at worst.
        Does NOT include heavy computations.
        
        Returns:
            Dict with:
              - U: Shape (d, rank). Current U factor.
              - V: Shape (d, rank). Current V factor.
              - U_norm: Spectral norm of U (2-norm, largest singular value).
              - rank: Rank of the manifold.
              - scars_admitted: Total scars written so far.
              - spectral_bound: Max allowed V norm.
        """
        return {
            "U": self.U.copy(),  # Copy for safety (gates shouldn't modify)
            "V": self.V.copy(),
            "U_norm": float(np.linalg.norm(self.U, ord=2)),
            "rank": self.rank,
            "scars_admitted": self.scars_admitted,
            "spectral_bound": self.spectral_bound,
        }

    def compute_scar_cost(self, x: np.ndarray, alpha: float = 0.01) -> Tuple[float, float]:
        """
        Estimate the topological + energetic cost of a proposed scar.
        
        Called by TransferController.draft_delta() to estimate topology_strain.
        Dry-run (does not modify state).
        
        Args:
            x: Proposed scar input
            alpha: Step size
        
        Returns:
            (topology_strain, residual_norm) where:
              - topology_strain: Frobenius norm change of U @ V.T, normalized
              - residual_norm: ‖x - U(V^T x)‖ (manifold distance of x)
        """
        # Current manifold
        current_manifold = self.U @ self.V.T
        current_norm = np.linalg.norm(current_manifold, 'fro')
        
        # Residual
        residual = x - (self.U @ (self.V.T @ x))
        res_norm = np.linalg.norm(residual)
        
        if res_norm < 1e-8:
            return 0.0, 0.0
        
        # Estimate new U after scar
        delta_u = (residual / res_norm).reshape(-1, 1)
        U_proposed = self.U + alpha * (delta_u @ (self.V.T @ x).reshape(1, -1))
        
        # Proposed manifold
        proposed_manifold = U_proposed @ self.V.T
        proposed_norm = np.linalg.norm(proposed_manifold, 'fro')
        
        # Topology strain: relative change
        topology_strain = abs(proposed_norm - current_norm) / (current_norm + 1e-10)
        topology_strain = min(1.0, topology_strain)  # Clip to [0, 1]
        
        return float(topology_strain), float(res_norm)
