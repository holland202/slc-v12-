"""
PreInferenceGate (PIG)
======================
Predictive risk evaluation that runs BEFORE any LLM inference.

Evaluates four factors:
  1. prompt_entropy: Shannon entropy + rank deficiency of input embedding
  2. identity_distance: Approximate geodesic distance from current SIC to recent stable states
  3. cognitive_pressure: Cumulative topological/thermal load from recent crystallizations
  4. variance_term: Variance of recent crystallization logit outputs (history memory)

Returns a soft probability (not hard pass/fail) + detailed risk breakdown.
This allows the router to make nuanced decisions: accept, defer, or escalate.

Mathematical Foundation:
  - Risk aggregation via weighted sum (configurable weights, default [0.25, 0.30, 0.25, 0.20])
  - Soft decision boundary: pass_prob = sigmoid(5 * (threshold - risk))
  - threshold typically 0.65 (can be tuned per deployment)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any


def sigmoid(z):
    """
    Numerically stable logistic. Replaces scipy.special.expit: requirements.txt
    pins numpy only, and scipy has no aarch64 wheel on Termux, so the scipy
    import made this module unimportable on the target device.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out if out.ndim else float(out)



class PreInferenceGate:
    """
    Predictive governance gate that evaluates manifest risk before GGUF invocation.
    
    Attributes:
        weights: [w_entropy, w_distance, w_pressure, w_variance] for 4-factor aggregation
        threshold: Midpoint of soft decision boundary (risk = 0.65 ↔ 50% pass probability)
        steepness: Slope of sigmoid boundary (default 5 = sharp but smooth)
        entropy_estimator: Reference to any Shannon entropy estimator (if available)
    """
    
    def __init__(
        self,
        weights: Optional[Tuple[float, float, float, float]] = None,
        threshold: float = 0.65,
        steepness: float = 5.0,
    ):
        """
        Args:
            weights: 4-tuple of factor weights. Default: [0.25, 0.30, 0.25, 0.20]
                    (entropy, distance, pressure, variance).
                    Must sum to 1.0 (will be normalized if not).
            threshold: Risk midpoint for pass/fail decision. Default 0.65.
            steepness: Sigmoid slope parameter. Higher = sharper boundary.
        """
        if weights is None:
            weights = (0.25, 0.30, 0.25, 0.20)
        
        # Normalize weights
        weights_array = np.array(weights, dtype=np.float32)
        weights_array /= weights_array.sum()
        self.weights = tuple(weights_array)
        
        self.threshold = threshold
        self.steepness = steepness
        
        # Diagnostic counters
        self.total_evaluations = 0
        self.total_passes = 0
        self.total_deferrals = 0
    
    def evaluate(
        self,
        prompt: str,
        sic_state: Any,
        cryst_memory: Any,
        prompt_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Evaluate manifest risk before GGUF inference.
        
        Args:
            prompt: The input prompt (used for entropy estimation)
            sic_state: Current SIC object (has .U, .V, state_summary(), etc.)
            cryst_memory: CrystallizationMemory object with history
            prompt_embedding: Optional pre-computed embedding (e.g., from tokenizer).
                             If None, use prompt text directly.
        
        Returns:
            (pass_decision, risk_score, factor_breakdown)
            where:
              - pass_decision: bool (True if pass_prob > 0.5)
              - risk_score: float in [0, 1] (lower is safer)
              - factor_breakdown: dict with individual factor values + diagnostics
        """
        self.total_evaluations += 1
        
        # --- Factor 1: Prompt Entropy ---
        prompt_entropy = self._estimate_prompt_entropy(prompt, prompt_embedding)
        
        # --- Factor 2: Identity Distance ---
        identity_distance = self._estimate_identity_distance(sic_state)
        
        # --- Factor 3: Cognitive Pressure ---
        cognitive_pressure = cryst_memory.cumulative_pressure(decay_factor=0.95)
        
        # --- Factor 4: Variance Term ---
        variance_term = cryst_memory.variance_term(use_recent_only=True)
        
        # Weighted aggregation
        risk_score = (
            self.weights[0] * prompt_entropy +
            self.weights[1] * identity_distance +
            self.weights[2] * cognitive_pressure +
            self.weights[3] * variance_term
        )
        
        # Soft decision boundary
        pass_prob = sigmoid(self.steepness * (self.threshold - risk_score))
        pass_decision = pass_prob > 0.5
        
        if pass_decision:
            self.total_passes += 1
        else:
            self.total_deferrals += 1
        
        factor_breakdown = {
            "prompt_entropy": float(prompt_entropy),
            "identity_distance": float(identity_distance),
            "cognitive_pressure": float(cognitive_pressure),
            "variance_term": float(variance_term),
            "risk_score": float(risk_score),
            "pass_probability": float(pass_prob),
            "pass_decision": bool(pass_decision),
        }
        
        return pass_decision, float(risk_score), factor_breakdown
    
    def _estimate_prompt_entropy(
        self,
        prompt: str,
        embedding: Optional[np.ndarray] = None,
    ) -> float:
        """
        Estimate Shannon entropy + rank deficiency of prompt.
        
        If embedding is provided, use it (expected from tokenizer).
        Otherwise, fall back to character distribution in prompt.
        
        Returns:
            Float in [0, 1]. 0 = completely predictable, 1 = maximum entropy.
        """
        if embedding is not None:
            # Use provided embedding (e.g., from tokenizer)
            # Normalize and compute Shannon entropy
            embedding = np.asarray(embedding, dtype=np.float32)
            embedding = np.abs(embedding)
            embedding /= (np.linalg.norm(embedding) + 1e-10)
            
            # Shannon entropy
            prob = embedding / (embedding.sum() + 1e-10)
            entropy = -np.sum(prob[prob > 1e-10] * np.log2(prob[prob > 1e-10] + 1e-10))
            
            # Normalize to [0, 1]
            max_entropy = np.log2(len(embedding))
            entropy_normalized = min(1.0, entropy / (max_entropy + 1e-10))
            
            # Rank deficiency penalty
            rank = np.linalg.matrix_rank(embedding.reshape(1, -1), tol=0.01)
            rank_penalty = 1.0 - (rank / len(embedding))
            
            return 0.7 * entropy_normalized + 0.3 * rank_penalty
        else:
            # Fallback: character distribution entropy
            if not prompt:
                return 0.0
            
            char_counts = {}
            for c in prompt.lower():
                char_counts[c] = char_counts.get(c, 0) + 1
            
            probs = np.array(list(char_counts.values())) / len(prompt)
            entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
            
            # Normalize to [0, 1] (max entropy for alphabet size)
            max_entropy = np.log2(min(len(char_counts), 26))
            entropy_normalized = min(1.0, entropy / (max_entropy + 1e-10))
            
            return entropy_normalized
    
    def _estimate_identity_distance(self, sic_state: Any) -> float:
        """
        Approximate geodesic distance of current SIC from stable manifold regions.
        
        Uses three cheap topological proxies (no full Stiefel metric needed):
          1. Spectral norm check: ‖UV^T‖₂ relative to bounds
          2. Singular value spread: λ_max / λ_min (condition number)
          3. Frobenius drift: ‖U - U_baseline‖_F (if baseline available)
        
        Returns:
            Float in [0, 1]. 0 = stable/canonical position, 1 = far from stable.
        """
        try:
            # Get SIC factors
            U = np.asarray(sic_state.U, dtype=np.float32)
            V = np.asarray(sic_state.V, dtype=np.float32)
            
            # Proxy 1: Spectral norm (should be bounded by λ_max)
            # Compute via largest singular value of UV^T
            try:
                # Efficient: only compute largest SV
                _, s, _ = np.linalg.svd(U @ V.T, full_matrices=False)
                spectral_norm = float(s[0]) if len(s) > 0 else 0.0
            except:
                spectral_norm = float(np.linalg.norm(U @ V.T, ord=2))
            
            # Expected range: [0, 2] (rough bound); clip to [0, 1]
            spectral_proxy = min(1.0, spectral_norm / 2.0)
            
            # Proxy 2: Singular value spread (condition number)
            # High condition number = manifold near singular = risky
            try:
                u_sing, s_u, _ = np.linalg.svd(U, full_matrices=False)
                s_nonzero = s_u[s_u > 1e-8]
                if len(s_nonzero) >= 2:
                    condition_number = s_nonzero[0] / s_nonzero[-1]
                    condition_proxy = min(1.0, (condition_number - 1.0) / 10.0)  # normalize to [0, 1]
                else:
                    condition_proxy = 0.0
            except:
                condition_proxy = 0.0
            
            # Proxy 3: Rank check
            # If rank < r (target rank), penalize
            rank_actual = np.linalg.matrix_rank(U @ V.T, tol=1e-6)
            rank_expected = U.shape[1]  # Assume full rank in U
            rank_proxy = 1.0 - (rank_actual / (rank_expected + 1e-10))
            
            # Aggregate proxies
            distance = 0.4 * spectral_proxy + 0.4 * condition_proxy + 0.2 * rank_proxy
            return float(min(1.0, distance))
        
        except Exception as e:
            # Fallback: return neutral (moderate risk)
            print(f"[PIG] Warning: identity_distance estimation failed: {e}")
            return 0.5
    
    def state_summary(self) -> Dict[str, float]:
        """
        Diagnostic summary of gate state.
        
        Returns:
            Dict with: total_evals, pass_rate, deferral_rate, weights, threshold
        """
        pass_rate = (
            self.total_passes / self.total_evaluations
            if self.total_evaluations > 0
            else 0.0
        )
        return {
            "total_evaluations": self.total_evaluations,
            "pass_rate": pass_rate,
            "deferral_rate": 1.0 - pass_rate,
            "total_passes": self.total_passes,
            "total_deferrals": self.total_deferrals,
            "weights": {
                "entropy": self.weights[0],
                "distance": self.weights[1],
                "pressure": self.weights[2],
                "variance": self.weights[3],
            },
            "threshold": self.threshold,
        }
