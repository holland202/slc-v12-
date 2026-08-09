"""
Crystallization Memory (CrystMemory)
====================================
Tracks the history of crystallization events (scar formations) to enable
memory-aware governance in PreInferenceGate and Commit Gate.

Maintains:
  - delta_history: logit_variance from each GGUF crystallization
  - pressure_history: topological strain/thermal burden from each scar
  - rejection_history: count of deferred/rejected inferences (for recency bias)

Used by both PreInferenceGate (to estimate variance_term) and
Commit Gate (to compute effective thermal temperature with memory).
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional


class CrystallizationMemory:
    """
    Lightweight history buffer for crystallization governance.
    
    Attributes:
        delta_history: Deque of logit variance from each crystallization
        pressure_history: Deque of topological/thermal strain per event
        rejection_history: Deque of rejection flags (0=accepted, 1=rejected)
        window_size: Max length of history buffers
    """
    
    def __init__(self, window_size: int = 20):
        """
        Args:
            window_size: Max number of recent events to track. 
                         Typical: 10–30 (on-device memory constraint).
        """
        self.window_size = window_size
        self.delta_history = deque(maxlen=window_size)
        self.pressure_history = deque(maxlen=window_size)
        self.rejection_history = deque(maxlen=window_size)
        
        # Metadata for diagnostics
        self.total_crystallizations = 0
        self.total_rejections = 0
        self.total_deferrals = 0
    
    def record_crystallization(
        self,
        logit_variance: float,
        topological_strain: float,
        was_rejected: bool = False
    ) -> None:
        """
        Record a crystallization event (or rejection/deferral).
        
        Args:
            logit_variance: Fisher-estimated output variance from GGUF inference.
                           (Already in [0, 1] range, typically 0.7–0.95 for high confidence)
            topological_strain: Geometric cost of this scar update.
                               Approximated as: ‖U_new - U_old‖_F or rank change penalty.
                               In [0, 1] scale.
            was_rejected: True if this was a rejection/deferral (no scar written).
        """
        self.delta_history.append(logit_variance)
        self.pressure_history.append(topological_strain)
        self.rejection_history.append(int(was_rejected))
        
        self.total_crystallizations += 1
        if was_rejected:
            self.total_rejections += 1
    
    def record_deferral(self, reason: str = "low_fisher") -> None:
        """
        Record a deferral (PreInferenceGate rejection, not reaching Commit Gate).
        
        Args:
            reason: Tag for deferral reason (e.g., "low_fisher", "high_entropy", etc.)
        """
        self.total_deferrals += 1
    
    def variance_term(self, use_recent_only: bool = True) -> float:
        """
        Compute the variance of recent crystallizations.
        
        This term feeds into PreInferenceGate risk scoring:
        High variance = unstable recent history = raise gate threshold.
        
        Args:
            use_recent_only: If True, only use non-rejection events.
                            If False, include all events.
        
        Returns:
            Float in [0, 1]. 0 = perfectly stable, 1 = maximum variance.
        """
        if not self.delta_history:
            return 0.0
        
        if use_recent_only:
            # Only variance from successful crystallizations
            accepted = [
                var for var, rej in zip(self.delta_history, self.rejection_history)
                if rej == 0
            ]
            if not accepted or len(accepted) < 2:
                return 0.0
            return float(np.var(accepted))
        else:
            # Variance of all events
            if len(self.delta_history) < 2:
                return 0.0
            return float(np.var(self.delta_history))
    
    def rejection_rate(self, window: Optional[int] = None) -> float:
        """
        Compute rejection rate in the recent window.
        
        Args:
            window: Number of recent events to include. If None, use full history.
        
        Returns:
            Float in [0, 1]. Higher = more recent rejections.
        """
        if not self.rejection_history:
            return 0.0
        
        if window is None:
            history = list(self.rejection_history)
        else:
            history = list(self.rejection_history)[-window:]
        
        if not history:
            return 0.0
        return float(np.mean(history))
    
    def cumulative_pressure(self, decay_factor: float = 0.95) -> float:
        """
        Compute cumulative (discounted) topological pressure.
        
        Recent events weighted more heavily (exponential decay into past).
        Used by Commit Gate to compute effective thermal temperature.
        
        Args:
            decay_factor: Discount factor per step into past (e.g., 0.95 = 5% decay/step).
        
        Returns:
            Float >= 0. Higher = more accumulated strain.
        """
        if not self.pressure_history:
            return 0.0
        
        pressure = list(self.pressure_history)
        n = len(pressure)
        weighted_sum = sum(
            p * (decay_factor ** (n - 1 - i))
            for i, p in enumerate(pressure)
        )
        normalization = sum(decay_factor ** j for j in range(n))
        
        return float(weighted_sum / normalization)
    
    def mean_logvar(self) -> float:
        """
        Mean logit variance across recent crystallizations.
        Diagnostic: indicates typical output confidence level.
        """
        if not self.delta_history:
            return 0.0
        return float(np.mean(self.delta_history))
    
    def state_summary(self) -> Dict[str, float]:
        """
        Diagnostic summary of memory state.
        
        Returns:
            Dict with: total_events, acceptance_rate, mean_logvar, variance_term, 
                       cumulative_pressure, recent_rejection_rate
        """
        return {
            "total_crystallizations": self.total_crystallizations,
            "total_rejections": self.total_rejections,
            "total_deferrals": self.total_deferrals,
            "acceptance_rate": 1.0 - self.rejection_rate() if self.total_crystallizations > 0 else 1.0,
            "mean_logvar": self.mean_logvar(),
            "variance_term": self.variance_term(),
            "cumulative_pressure": self.cumulative_pressure(),
            "recent_rejection_rate": self.rejection_rate(window=5),
        }
    
    def reset(self) -> None:
        """
        Clear all history (e.g., after a system reboot or recalibration).
        """
        self.delta_history.clear()
        self.pressure_history.clear()
        self.rejection_history.clear()
        self.total_crystallizations = 0
        self.total_rejections = 0
        self.total_deferrals = 0
