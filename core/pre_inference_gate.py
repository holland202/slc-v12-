"""pre_inference_gate.py — Predictive Risk Evaluator (Stage 4) · SLC v12"""
import numpy as np
from typing import Tuple, Dict, Any, Optional

class PreInferenceGate:
    def __init__(self, weights=(0.25,0.30,0.25,0.20), threshold=0.65, steepness=5.0):
        self.weights   = weights
        self.threshold = threshold
        self.steepness = steepness
        self._evals = 0
        self._passes = 0

    def evaluate(self, prompt, sic_state, cryst_memory,
                 prompt_embedding=None) -> Tuple[bool, float, Dict]:
        self._evals += 1
        f1 = 1.0 - cryst_memory.rejection_rate()
        if prompt:
            words = prompt.lower().split()
            f2 = min(1.0, len(set(words)) / max(len(words),1))
        else:
            f2 = 0.5
        f3 = min(1.0, cryst_memory.mean_logit_variance())
        f4 = max(0.0, 1.0 - cryst_memory.mean_topological_strain() * 5)
        factors = {
            "rejection_history": round(f1,4),
            "prompt_entropy":    round(f2,4),
            "logit_variance":    round(f3,4),
            "topo_strain":       round(f4,4),
        }
        score = (self.weights[0]*f1 + self.weights[1]*f2 +
                 self.weights[2]*f3 + self.weights[3]*f4)
        gate_pass = score >= self.threshold
        if gate_pass: self._passes += 1
        return gate_pass, float(score), factors

    def state_summary(self):
        return {
            "total_evaluations": self._evals,
            "pass_rate": round(self._passes/max(1,self._evals), 4),
            "threshold": self.threshold,
        }
