#!/usr/bin/env python3
"""
core/veritas_gate.py — SLC Thermodynamic Governor
Enforces the Gibbs stability mandate. Rejects topological updates if ΔG >= 0.
"""
import math
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
            
        # Gibbs Free Energy Mandate
        dG = self.cfg.dH - T * self.cfg.dS
        gate = dG < 0
        
        if gate:
            self.commits += 1
        else:
            self.rejects += 1
            
        return aT, dG, gate, T
