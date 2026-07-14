"""crystallization_memory.py — Crystallization History Buffer · SLC v12"""
from collections import deque
import time

class CrystallizationMemory:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self._history   = deque(maxlen=window_size)
        self._deferrals = deque(maxlen=window_size)

    def record_crystallization(self, logit_variance, topological_strain, was_rejected):
        self._history.append({
            "timestamp": time.time(),
            "logit_variance": logit_variance,
            "topological_strain": topological_strain,
            "was_rejected": was_rejected,
        })

    def record_deferral(self, reason="unknown"):
        self._deferrals.append({"timestamp": time.time(), "reason": reason})

    def rejection_rate(self):
        if not self._history: return 0.0
        return sum(1 for h in self._history if h["was_rejected"]) / len(self._history)

    def mean_logit_variance(self):
        if not self._history: return 0.7
        return sum(h["logit_variance"] for h in self._history) / len(self._history)

    def mean_topological_strain(self):
        if not self._history: return 0.0
        return sum(h["topological_strain"] for h in self._history) / len(self._history)

    def state_summary(self):
        return {
            "window_size": self.window_size,
            "entries": len(self._history),
            "rejection_rate": round(self.rejection_rate(), 4),
            "mean_logit_variance": round(self.mean_logit_variance(), 4),
            "mean_topo_strain": round(self.mean_topological_strain(), 4),
            "deferrals": len(self._deferrals),
        }
