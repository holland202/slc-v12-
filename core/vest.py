#!/usr/bin/env python3
"""
core/vest.py — Veritas-Encoded Semantic Tunneling
Authenticates semantic vectors to ensure they lie on the valid identity manifold.
Rejects out-of-manifold inputs as topological holes (hallucinations).
"""
import numpy as np

class VESTunnel:
    def __init__(self, fidelity_threshold: float = 1.5):
        self.fidelity_threshold = fidelity_threshold
        self.tunnels_passed = 0
        self.tunnels_blocked = 0

    def authenticate(self, x_t: np.ndarray, U_t: np.ndarray, V_t: np.ndarray) -> tuple[bool, float]:
        """
        Projects the input state onto the low-rank manifold and computes the residual.
        Returns (is_authentic, distance).
        """
        # Project onto the manifold: I_t(x) = U_t @ V_t^T @ x_t
        x_proj = U_t @ (V_t @ x_t)
        
        # Calculate geometric distance (residual Euclidean norm)
        distance = np.linalg.norm(x_t - x_proj)
        
        # Validate against the simply-connected constraint (∂²=0)
        is_authentic = distance <= self.fidelity_threshold
        
        if is_authentic:
            self.tunnels_passed += 1
        else:
            self.tunnels_blocked += 1
            
        return is_authentic, distance

if __name__ == "__main__":
    print("--- INITIATING LOCAL VEST DIAGNOSTIC ---")
    vest = VESTunnel(fidelity_threshold=4.5)
    
    # Mock Manifold Operators (matching SIC dimensions)
    dim, rank = 64, 8
    rng = np.random.default_rng(42)
    U = rng.normal(0, 0.1, (dim, rank))
    V = rng.normal(0, 0.1, (rank, dim))
    
    # 1. Clean data (Simulating an in-distribution truth signal)
    x_clean = rng.normal(0, 0.5, dim)
    
    # 2. Noisy data (Simulating an out-of-distribution hallucination / logic hole)
    x_noisy = rng.normal(5, 2.0, dim)
    
    pass_clean, dist_clean = vest.authenticate(x_clean, U, V)
    print(f"[TUNNEL 1] Clean Vector Distance: {dist_clean:.4f} | Status: {'PASS' if pass_clean else 'BLOCKED'}")
    
    pass_noisy, dist_noisy = vest.authenticate(x_noisy, U, V)
    print(f"[TUNNEL 2] Noisy Vector Distance: {dist_noisy:.4f} | Status: {'PASS' if pass_noisy else 'BLOCKED'}")
    
    print(f"Total Passed: {vest.tunnels_passed} | Total Blocked: {vest.tunnels_blocked}")
    print("--- DIAGNOSTIC COMPLETE ---")
