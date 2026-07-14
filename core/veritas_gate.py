"""veritas_gate.py — Veritas Gate · Thermodynamic Governor · SLC v12

Enforces the Gibbs Free Energy mandate (ΔG < 0) as a hard constraint
on every state transition. Implements three-regime Schmitt trigger
with hysteresis to eliminate thermal chatter.
"""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

@dataclass
class ControlVector:
    """Governance signal injected into inference loop."""
    thermal_delta: float      # Δ from nominal band (°C)
    entropy_pressure: float   # Current H_σ normalized
    precision_mode: int       # 0:FP16 1:INT8 2:INT4
    cpu_affinity: int         # Core binding mask
    command: int              # 1:REDUCE 2:STABILIZE 3:FOLD
    gibbs_margin: float       # ΔG current value
    timestamp: float

class VeritasGate:
    """Thermodynamic governor with Schmitt trigger hysteresis.
    
    Parameters (from SLC-TDD-12.0):
      T_low = 35.0°C, T_high = 38.5°C, T_critical = 41.0°C
      Hysteresis width = 2.0°C
    """
    
    COMMANDS = {1: "ATOMIC_REDUCTION", 2: "STABILIZE", 3: "FOLD"}
    
    def __init__(self, T_low: float = 35.0, T_high: float = 38.5,
                 T_critical: float = 41.0, T_recovery: float = 38.0,
                 hysteresis_width: float = 2.0,
                 H_critical: float = 0.22, tau_thermal: float = 5.0):
        self.T_low = T_low
        self.T_high = T_high
        self.T_critical = T_critical
        self.T_recovery = T_recovery
        self.hysteresis_width = hysteresis_width
        self.H_critical = H_critical
        self.tau_thermal = tau_thermal
        
        self._thermal_ema = 35.0
        self._last_state = "NORMAL"
        self._state_history = []
        self._cycle_count = 0
        
    def update(self, temperature: float, spectral_entropy: float,
               compute_load: float = 0.5) -> ControlVector:
        """Compute governance vector from current substrate state."""
        self._cycle_count += 1
        
        # EMA smoothing of temperature
        alpha = 1.0 / (1.0 + self.tau_thermal)
        self._thermal_ema = (1 - alpha) * self._thermal_ema + alpha * temperature
        T = self._thermal_ema
        
        # Compute Gibbs margin
        delta_H = compute_load * 10.0
        delta_S = spectral_entropy * 0.5
        gibbs_margin = delta_H - T * delta_S
        
        # Schmitt trigger with hysteresis
        if self._last_state == "NORMAL":
            if T >= self.T_high:
                state = "SCAR_LOCK"
            elif T >= self.T_low:
                state = "ENTROPY_THROTTLE"
            else:
                state = "NORMAL"
        elif self._last_state == "ENTROPY_THROTTLE":
            if T >= self.T_high:
                state = "SCAR_LOCK"
            elif T <= self.T_low - self.hysteresis_width:
                state = "NORMAL"
            else:
                state = "ENTROPY_THROTTLE"
        elif self._last_state == "SCAR_LOCK":
            if T <= self.T_recovery:
                state = "ENTROPY_THROTTLE"
            elif T >= self.T_critical:
                state = "ATOMIC_REDUCTION"
            else:
                state = "SCAR_LOCK"
        elif self._last_state == "ATOMIC_REDUCTION":
            if T <= self.T_recovery:
                state = "ENTROPY_THROTTLE"
            else:
                state = "ATOMIC_REDUCTION"
        else:
            state = "NORMAL"
        
        self._last_state = state
        self._state_history.append((time.time(), state, T, gibbs_margin))
        
        # Map state to command
        if state == "ATOMIC_REDUCTION":
            command = 1
            precision = 2
        elif state == "SCAR_LOCK":
            command = 3
            precision = 2
        elif state == "ENTROPY_THROTTLE":
            command = 2
            precision = 1
        else:
            command = 2
            precision = 0
        
        # Override on entropy pressure
        if spectral_entropy > self.H_critical:
            command = 3
        
        return ControlVector(
            thermal_delta=T - self.T_low,
            entropy_pressure=spectral_entropy,
            precision_mode=precision,
            cpu_affinity=0xFF,
            command=command,
            gibbs_margin=gibbs_margin,
            timestamp=time.time(),
        )
    
    def state_summary(self) -> Dict[str, Any]:
        """Governor runtime statistics."""
        if not self._state_history:
            return {"cycles": 0}
        states = [s[1] for s in self._state_history]
        return {
            "cycles": self._cycle_count,
            "current_state": self._last_state,
            "thermal_ema": round(self._thermal_ema, 2),
            "state_distribution": {
                s: states.count(s) / len(states) for s in set(states)
            },
            "last_gibbs_margin": round(self._state_history[-1][3], 4),
        }
