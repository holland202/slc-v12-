#!/usr/bin/env python3
"""
core/ume.py — Umbra Manifold Engine

Langevin (Ornstein-Uhlenbeck) exploration, Euler-Maruyama discretised:

    X_{t+1} = X_t - eta * X_t * dt + sqrt(2 * lambda(T) * dt) * xi,   xi ~ N(0, I)

The diffusion coefficient is the thermal governor, per the SLC v12 spec:
lambda -> 0 as the substrate junction temperature T approaches T_critical, so
the stochastic term collapses onto the deterministic fixed point instead of
adding heat to a device that is already throttling.

    lambda(T) = lambda_0 * clip((T_c - T) / (T_c - T_0), 0, 1)

REFUTATION HISTORY (kept). The prior implementation was:

    noise = np.random.normal(0, 0.1, size=X_t.shape)
    return X_t + noise, "CLASSICAL_DIFFUSION"

It accepted T and never read it. Registered probe P3 measured mean|dx| =
0.789665 at T=1 and 0.789665 at T=1000, bitwise identical, and the
"Langevin diffusion" claim was marked REFUTED. This is the fix. lambda_0 is
set to 0.5 so that sqrt(2*lambda_0*dt) = 0.1 at dt=0.01, preserving the old
step scale at full thermal headroom; what changes is that it now decays.
"""
import numpy as np


class UmbraManifoldEngine:
    def __init__(self, T_0: float = 38.0, T_critical: float = None,
                 eta: float = 2.0, dt: float = 0.01, lambda_0: float = 0.5,
                 seed: int = None):
        self.T_0 = T_0
        self.T_critical = T_critical if T_critical is not None else T_0 + 2.0
        self.eta = eta
        self.dt = dt
        self.lambda_0 = lambda_0
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def diffusion_coefficient(self, T: float) -> float:
        """lambda(T): full below T_0, linear decay to exactly 0.0 at T_critical."""
        span = self.T_critical - self.T_0
        if span <= 0:
            return 0.0 if T >= self.T_critical else self.lambda_0
        frac = (self.T_critical - T) / span
        return self.lambda_0 * float(np.clip(frac, 0.0, 1.0))

    def explore(self, X_t: np.ndarray, T: float) -> tuple[np.ndarray, str]:
        lam = self.diffusion_coefficient(T)
        drift = -self.eta * X_t * self.dt
        if lam > 0.0:
            xi = (self.rng.standard_normal(X_t.shape) if self.rng is not None
                  else np.random.normal(0.0, 1.0, size=X_t.shape))
            noise = np.sqrt(2.0 * lam * self.dt) * xi
        else:
            noise = np.zeros_like(X_t)
        X_next = X_t + drift + noise

        if T >= self.T_critical:
            mode = "THERMAL_COLLAPSE"
        elif T > self.T_0:
            mode = "LANGEVIN_THROTTLED"
        else:
            mode = "LANGEVIN_EXPLORE"
        return X_next, mode
