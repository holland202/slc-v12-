#!/usr/bin/env python3
"""
core/ume.py — Umbra Manifold Engine
Explores the latent space using temperature-controlled Langevin dynamics.
dX_t = -∇U(X_t) dt + √(2λ(T)) dW_t
"""
import numpy as np
import math

class UmbraManifoldEngine:
    def __init__(self, lambda_0: float = 1.0, T_0: float = 34.0, sigma_T: float = 2.0, dt: float = 0.01):
        self.lambda_0 = lambda_0
        self.T_0 = T_0
        self.sigma_T = sigma_T
        self.dt = dt

    def _diffusion_coefficient(self, T: float) -> float:
        """Calculates temperature-dependent diffusion λ(T)."""
        return self.lambda_0 * math.exp(-((T - self.T_0)**2) / (self.sigma_T**2))

    def langevin_step(self, X_t: np.ndarray, grad_U: np.ndarray, T: float) -> np.ndarray:
        """
        Executes one step of Langevin diffusion using Euler-Maruyama integration.
        grad_U: The natural gradient -∇U(X_t)
        T: Current hardware temperature in °C
        """
        # Calculate current diffusion coefficient based on thermal state
        lambda_T = self._diffusion_coefficient(T)
        
        # Generate discrete Wiener process step (Brownian noise)
        dW_t = np.random.normal(0, np.sqrt(self.dt), size=X_t.shape)
        
        # Langevin update
        dX_t = -grad_U * self.dt + np.sqrt(2 * lambda_T) * dW_t
        
        return X_t + dX_t

if __name__ == "__main__":
    print("--- INITIATING LOCAL UME DIAGNOSTIC ---")
    ume = UmbraManifoldEngine(lambda_0=1.0, T_0=34.0, sigma_T=2.0, dt=0.01)
    
    # Mock parameters
    X_initial = np.zeros(64)
    gradient = np.ones(64) * 0.1  # Mock gradient pulling the state
    
    # Test 1: Optimal Temperature (T = 34.0°C) -> High exploration (max diffusion)
    X_opt = ume.langevin_step(X_initial.copy(), gradient, T=34.0)
    
    # Test 2: Throttled Temperature (T = 38.0°C) -> Low exploration (diffusion collapse)
    X_hot = ume.langevin_step(X_initial.copy(), gradient, T=38.0)
    
    print(f"Optimal Temp (34.0°C) Diffusion λ(T): {ume._diffusion_coefficient(34.0):.6f}")
    print(f"Throttled Temp (38.0°C) Diffusion λ(T): {ume._diffusion_coefficient(38.0):.6f}")
    
    print(f"\n[UME TRAJECTORY SPREAD]")
    print(f"Optimal State Vector Norm  : {np.linalg.norm(X_opt):.4f}")
    print(f"Throttled State Vector Norm: {np.linalg.norm(X_hot):.4f}")
    
    if np.linalg.norm(X_opt) > np.linalg.norm(X_hot):
        print("Status: PASS (Thermal diffusion collapse verified)")
    else:
        print("Status: FAIL (Check Gaussian noise scaling)")
        
    print("--- DIAGNOSTIC COMPLETE ---")
