"""vest.py — Veritas-Encoded Semantic Tunneling · SLC v12

Trajectory-dependent geometric authentication. Not classical cryptography,
but manifold-agreement verification intrinsically bound to the full scar
history of the runtime.
"""
import numpy as np
from typing import Tuple, Dict, Any

class VESTAuthenticator:
    """VEST: manifold-trajectory-dependent geometric authentication."""
    
    def __init__(self, d: int = 512, rank: int = 64,
                 epsilon: float = 1e-6, threshold: float = 0.18):
        self.d = d
        self.rank = rank
        self.epsilon = epsilon
        self.threshold = threshold
        
    def challenge(self, sic_state) -> np.ndarray:
        """Generate a challenge vector from standard normal."""
        return np.random.normal(0, 1, self.d).astype(np.float32)
    
    def respond(self, challenge: np.ndarray, sic_state) -> np.ndarray:
        """Compute VEST response: Φ_t(c) = U_t · tanh(V_t^T · c)."""
        U = sic_state.U
        V = sic_state.V
        latent = V @ challenge
        gated = np.tanh(latent)
        response = U @ gated
        return response
    
    def authenticate(self, challenge: np.ndarray, response: np.ndarray,
                     sic_state) -> Dict[str, Any]:
        """Verify response under Fisher-Riemannian metric."""
        expected = self.respond(challenge, sic_state)
        
        # Fisher-Riemannian distance using Woodbury identity
        U = sic_state.U
        diff = response - expected
        
        eps = self.epsilon
        UtU = U.T @ U
        inner = np.eye(self.rank) + UtU / eps
        inner_inv = np.linalg.inv(inner)
        
        term1 = (diff @ diff) / eps
        temp = U.T @ diff
        term2 = (temp @ inner_inv @ temp) / (eps ** 2)
        d_fr = float(term1 - term2)
        d_fr = max(0.0, d_fr)
        
        return {
            "authentic": d_fr < self.threshold,
            "distance": d_fr,
            "confidence": float(np.exp(-d_fr * 4.5)),
            "threshold": self.threshold,
        }
    
    def mutual_verify(self, local_sic, remote_challenge: np.ndarray,
                      remote_response: np.ndarray) -> Dict[str, Any]:
        """Verify a remote party's challenge-response against local manifold."""
        return self.authenticate(remote_challenge, remote_response, local_sic)
