"""sic.py — Scarred Identity Chronicle · SLC v12

Entropy-gated, irreversible low-rank continual learning system.
Encodes significant operational events as permanent rank-1 deformations
of an identity manifold S_t ≈ U_t V_t^T ∈ R^{d×d}.
"""
import numpy as np
from typing import Optional, Dict, Any
import time

class ScarEvent:
    """A single event candidate for scar admission."""
    def __init__(self, embedding: np.ndarray, entropy: float,
                 metadata: Optional[Dict] = None):
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.entropy = float(entropy)
        self.metadata = metadata or {}
        self.timestamp = time.time()

class ScarredIdentityChronicle:
    """SIC: low-rank identity manifold with entropy-gated scar accumulation.
    
    Parameters (from SLC-TDD-12.0):
      d=512, r=64, α=0.012, β=0.85, γ=1e-4
      θ_H = 0.35 (entropy admission threshold)
      τ_σ = 28.5 (spectral entropy ceiling)
      τ_auth = 0.18 (VEST authentication threshold)
    """
    
    def __init__(self, d: int = 512, rank: int = 64,
                 alpha: float = 0.012, beta: float = 0.85,
                 gamma: float = 1e-4, entropy_threshold: float = 0.35,
                 spectral_entropy_ceiling: float = 28.5,
                 auth_threshold: float = 0.18):
        self.d = d
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.entropy_threshold = entropy_threshold
        self.spectral_entropy_ceiling = spectral_entropy_ceiling
        self.auth_threshold = auth_threshold
        
        # Initialize U on Stiefel manifold (orthonormal columns)
        self.U = np.linalg.qr(np.random.randn(d, rank).astype(np.float32))[0]
        # V as dense right factor
        self.V = np.random.normal(0, 0.1, (rank, d)).astype(np.float32)
        # Adaptive anchor (slow drift)
        self.anchor = np.random.normal(0.0, 0.085, (d,)).astype(np.float32)
        
        self.events = []
        self.scars_admitted = 0
        self._update_counter = 0
        self._qr_period = 50  # QR retraction every N updates
        
    def record_event(self, event: ScarEvent) -> bool:
        """Attempt to scar an event. Returns True if admitted."""
        # Stage 1: Entropy gate
        H = event.entropy
        if H < self.entropy_threshold:
            return False
        
        # Stage 2: Compute residual and weight
        delta = event.embedding - self.anchor
        weight = self.alpha * np.exp(-self.beta * H)
        
        # Stage 3: Rank-1 scar update (natural gradient step)
        z = self.V @ delta  # z ∈ R^r
        
        # Symmetric update to both factors
        self.U += weight * np.outer(delta, z)
        self.V += weight * np.outer(z, delta)
        
        # Stage 4: Anchor drift (slow exponential moving average)
        self.anchor = (1 - self.gamma) * self.anchor + self.gamma * event.embedding
        
        # Stage 5: Periodic QR retraction (Stiefel manifold projection)
        self._update_counter += 1
        if self._update_counter % self._qr_period == 0:
            self._qr_retraction()
        
        # Stage 6: Spectral stability check
        if not self._check_spectral_stability():
            # Rollback: undo this update
            self.U -= weight * np.outer(delta, z)
            self.V -= weight * np.outer(z, delta)
            self.anchor = (self.anchor - self.gamma * event.embedding) / (1 - self.gamma)
            return False
        
        self.events.append(event)
        self.scars_admitted += 1
        return True
    
    def _qr_retraction(self):
        """Project U back onto Stiefel manifold."""
        Q, _ = np.linalg.qr(self.U)
        self.U = Q[:, :self.rank]
    
    def _check_spectral_stability(self) -> bool:
        """Check spectral entropy of symmetrized metric tensor."""
        M = self.metric_tensor()
        sv = np.linalg.svd(M, compute_uv=False)
        # Normalize singular values to probability distribution
        sv_norm = sv / (np.sum(sv) + 1e-10)
        # Compute spectral entropy
        H_sigma = -np.sum(sv_norm * np.log(sv_norm + 1e-10))
        return H_sigma < self.spectral_entropy_ceiling
    
    def metric_tensor(self) -> np.ndarray:
        """Symmetrized metric tensor M_s = ½(U V^T + V U^T)."""
        M = self.U @ self.V
        return 0.5 * (M + M.T)
    
    def verify_identity(self, candidate: np.ndarray,
                        threshold: Optional[float] = None) -> Dict[str, Any]:
        """VEST identity verification via projected quadratic distance.
        
        d(x, A) = δ^T (U V^T) δ,  δ = x - A
        Authentication succeeds iff d < τ.
        """
        if threshold is None:
            threshold = self.auth_threshold
        delta = np.asarray(candidate, dtype=np.float32) - self.anchor
        M = self.U @ self.V
        dist = float(delta.T @ M @ delta)
        return {
            "authentic": dist < threshold,
            "distance": dist,
            "confidence": float(np.exp(-dist * 4.5)),
            "events": len(self.events),
            "scars_admitted": self.scars_admitted,
        }
    
    def spectral_entropy(self) -> float:
        """Current spectral entropy H_σ of the metric tensor."""
        M = self.metric_tensor()
        sv = np.linalg.svd(M, compute_uv=False)
        sv_norm = sv / (np.sum(sv) + 1e-10)
        return float(-np.sum(sv_norm * np.log(sv_norm + 1e-10)))
    
    def effective_rank(self, temperature: float = 35.0,
                       T0: float = 38.0, eta: float = 0.1) -> int:
        """Thermal-adaptive rank suppression: r_eff = r · exp(-η·max(0, T-T0))."""
        suppression = np.exp(-eta * max(0, temperature - T0))
        return max(1, int(self.rank * suppression))
    
    def state_dict(self) -> Dict[str, Any]:
        """Serialize manifold state for persistence."""
        return {
            "U": self.U.copy(),
            "V": self.V.copy(),
            "anchor": self.anchor.copy(),
            "events": self.events,
            "scars_admitted": self.scars_admitted,
            "config": {
                "d": self.d, "rank": self.rank,
                "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
            }
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Restore manifold state from serialized dict."""
        self.U = state["U"].copy()
        self.V = state["V"].copy()
        self.anchor = state["anchor"].copy()
        self.events = list(state.get("events", []))
        self.scars_admitted = state.get("scars_admitted", 0)
