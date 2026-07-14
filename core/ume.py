"""ume.py — Umbra Manifold Engine · SLC v12

Stochastic exploratory cognition via Langevin diffusion:
  dX_t = -∇U(X_t) dt + √(2λ(T)) dW_t
"""
import numpy as np
from typing import Optional, Tuple, List

class UmbraManifoldEngine:
    """UME: stochastic exploration branch of DMIA."""
    
    def __init__(self, dim: int = 512, lambda0: float = 1.0,
                 T0: float = 38.0, sigma_T: float = 5.0,
                 dt: float = 0.01):
        self.dim = dim
        self.lambda0 = lambda0
        self.T0 = T0
        self.sigma_T = sigma_T
        self.dt = dt
        
        self.state = np.zeros(dim, dtype=np.float32)
        self.trajectory: List[np.ndarray] = []
        self._step_count = 0
        
    def thermal_diffusion_coeff(self, temperature: float) -> float:
        """λ(T) = λ₀ · exp(-(T-T₀)²/σ_T²)"""
        return self.lambda0 * np.exp(-((temperature - self.T0) ** 2) / (self.sigma_T ** 2))
    
    def potential(self, x: np.ndarray, sic_state) -> float:
        """U(X) = ½ ||X - SIC_anchor||² + ¼ ||X||⁴"""
        anchor = sic_state.anchor
        diff = x - anchor
        return 0.5 * np.dot(diff, diff) + 0.25 * (np.dot(x, x) ** 2)
    
    def gradient(self, x: np.ndarray, sic_state) -> np.ndarray:
        """∇U(X) = (X - anchor) + ||X||² · X"""
        anchor = sic_state.anchor
        return (x - anchor) + np.dot(x, x) * x
    
    def step(self, sic_state, temperature: float = 35.0) -> np.ndarray:
        """Single Langevin step."""
        lam = self.thermal_diffusion_coeff(temperature)
        grad = self.gradient(self.state, sic_state)
        
        drift = -grad * self.dt
        noise = np.random.normal(0, 1, self.dim)
        diffusion = np.sqrt(2 * lam * self.dt) * noise
        
        self.state += drift + diffusion
        self.trajectory.append(self.state.copy())
        self._step_count += 1
        
        return self.state
    
    def generate_hypothesis(self, sic_state, temperature: float = 35.0,
                            n_steps: int = 100) -> np.ndarray:
        for _ in range(n_steps):
            self.step(sic_state, temperature)
        return self.state
    
    def project_to_slc(self, sic_state) -> np.ndarray:
        """Project UME state onto SLC manifold."""
        U = sic_state.U
        return U @ (U.T @ self.state)
    
    def state_summary(self) -> dict:
        return {
            "dim": self.dim,
            "steps": self._step_count,
            "trajectory_length": len(self.trajectory),
            "state_norm": float(np.linalg.norm(self.state)),
        }
