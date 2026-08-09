#!/usr/bin/env python3
"""
core/veritas_gate.py — SLC Thermodynamic Governor
Enforces the Gibbs stability mandate. Rejects topological updates if ΔG >= 0.
"""
import math
import numpy as np
from typing import Tuple
from core.params import RuntimeConfig
from core.hardware_link import ThermalMonitor

class VeritasGate:
    def __init__(self, cfg: RuntimeConfig, monitor: ThermalMonitor):
        self.cfg = cfg
        self.mon = monitor
        self.commits = 0
        self.rejects = 0
    
    def evaluate(self) -> Tuple[float, float, bool, float]:
        """
        Polls the substrate and calculates thermodynamic feasibility.
        Returns: (thermal_multiplier, gibbs_energy, pass_gate, temperature)
        """
        T = self.mon.read()
        
        # Absolute Hardware Lock
        if T >= self.cfg.temp_critical:
            return 0.0, 0.0, False, T

        # Thermal scaling (Langevin diffusion collapse)
        if T <= self.cfg.temp_threshold:
            aT = 1.0
        else:
            aT = math.exp(-self.cfg.eta * (T - self.cfg.temp_threshold))
            
        # Gibbs Free Energy Mandate, constant-coefficient path.
        # REFUTED AND KEPT: dH and dS here are fixed config constants, so
        # dG = -0.1 - 0.02*T is negative at every temperature a substrate
        # reaches (dG >= 0 only below -5.0 C). Registered probe P1 measured
        # 0 rejecting temperatures across 0-100 C. This branch cannot refuse
        # and is retained only so evaluate() keeps its old behaviour for
        # run_slc.py and the thermal suites. State-dependent admission lives
        # in evaluate_transition() below, which CAN refuse.
        dG = self.cfg.dH - T * self.cfg.dS
        gate = dG < 0
        
        if gate:
            self.commits += 1
        else:
            self.rejects += 1
            
        return aT, dG, gate, T

    # -------------------------------------------------------------------
    # State-dependent admission (the enforcing path)
    # -------------------------------------------------------------------

    @staticmethod
    def spectral_entropy(M: np.ndarray) -> float:
        """Shannon entropy of the normalised singular-value spectrum of M."""
        sv = np.linalg.svd(np.asarray(M, dtype=np.float64), compute_uv=False)
        total = sv.sum()
        if total <= 0:
            return 0.0
        p = np.clip(sv / total, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    @classmethod
    def state_deltas(cls, U, V, U_new, V_new) -> Tuple[float, float]:
        """
        Enthalpy and entropy change of a proposed operator update, per the
        SLC v12 spec: E_t = ||U_t||_F^2 + ||V_t||_F^2 and S_t is the
        spectral entropy of the left factor. Returns (dH, dS).
        """
        E_old = float(np.sum(np.asarray(U) ** 2) + np.sum(np.asarray(V) ** 2))
        E_new = float(np.sum(np.asarray(U_new) ** 2) + np.sum(np.asarray(V_new) ** 2))
        return E_new - E_old, cls.spectral_entropy(U_new) - cls.spectral_entropy(U)

    def evaluate_transition(self, dH: float, dS: float,
                            T: float = None) -> Tuple[float, float, bool, float]:
        """
        Gibbs admission for a PROPOSED state transition. Unlike evaluate(),
        dH and dS are measured from the actual operator change, so dG can
        take either sign and the gate can refuse.

        Returns: (thermal_multiplier, dG, admit, temperature)
        """
        T = self.mon.read() if T is None else float(T)

        if T >= self.cfg.temp_critical:
            self.rejects += 1
            return 0.0, 0.0, False, T

        if T <= self.cfg.temp_threshold:
            aT = 1.0
        else:
            aT = math.exp(-self.cfg.eta * (T - self.cfg.temp_threshold))

        dG = float(dH) - T * float(dS)
        admit = dG < 0

        if admit:
            self.commits += 1
        else:
            self.rejects += 1

        return aT, dG, admit, T
