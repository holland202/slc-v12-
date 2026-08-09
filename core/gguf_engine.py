#!/usr/bin/env python3
"""
core/gguf_engine.py — real local-model backend for the 10-step loop.

Replaces run_engine.py's MockGGUF. Exposes the one method the Engine calls:

    generate(prompt, max_tokens, logprobs) -> {
        "success": bool, "text": str, "logit_variance": float
    }

NAMING, STATED PLAINLY. The Commit Gate reads `delta.logit_variance` and admits
when it is >= fisher_threshold (0.85), i.e. HIGHER means MORE confident. A
variance would go the other way, and this is not Fisher information. The field
carries a token-confidence score in [0, 1]. The key name is kept because the
Engine and TransferController already use it; the label is inherited and wrong,
and renaming it is a separate change across three files.

Two metrics, both computed from real per-token logprobs:

  mean_token_prob          exp(mean log p(selected token)). 1.0 = the model
                           never hesitated. Default.
  one_minus_norm_entropy   1 - H/H_max over the top-k distribution per token,
                           averaged. Uses the shape of the whole distribution
                           rather than only the chosen token, so it penalises a
                           model that was nearly indifferent between two options.

CALIBRATION WARNING. The 0.85 default threshold was set against a mock that
returned a constant. It has never been calibrated against a real model's output
distribution. Run calibrate_gguf_threshold.py FIRST and set the threshold from
measured data. Do not tune it until everything passes — that is choosing the
answer, and this repo's whole argument is against it.

Requires llama-cpp-python:
    pip install llama-cpp-python --break-system-packages
"""
import math
import sys
from typing import Any, Dict, List, Optional

METRICS = ("mean_token_prob", "one_minus_norm_entropy")


def mean_token_prob(token_logprobs: List[float]) -> float:
    """exp of the mean selected-token logprob. In (0, 1]."""
    lp = [x for x in token_logprobs if x is not None and math.isfinite(x)]
    if not lp:
        return 0.0
    return float(math.exp(sum(lp) / len(lp)))


def one_minus_norm_entropy(top_logprobs: List[Dict[str, float]]) -> float:
    """
    1 - H/H_max averaged over tokens, from the top-k distribution.
    H_max = log(k), so a uniform top-k scores 0.0 and a one-hot scores 1.0.
    """
    scores = []
    for dist in top_logprobs:
        if not dist:
            continue
        ps = [math.exp(v) for v in dist.values() if v is not None and math.isfinite(v)]
        total = sum(ps)
        if total <= 0 or len(ps) < 2:
            continue
        ps = [p / total for p in ps]
        H = -sum(p * math.log(p) for p in ps if p > 0)
        H_max = math.log(len(ps))
        scores.append(1.0 - (H / H_max if H_max > 0 else 0.0))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


class GGUFEngine:
    """
    Thin wrapper over llama_cpp.Llama. Holds no governance logic — every
    admission decision stays in the Engine, where it can be probed.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4,
                 temperature: float = 0.0, seed: int = 42, top_k: int = 5,
                 metric: str = "mean_token_prob", verbose: bool = False):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        try:
            from llama_cpp import Llama
        except Exception as e:
            # NOT just ImportError. On Termux/aarch64 llama-cpp-python imports
            # fine and then raises RuntimeError("Unsupported platform") from
            # its own shared-library loader, which the narrower except missed
            # and turned into a raw traceback. Measured on an S25 Ultra.
            raise RuntimeError(
                f"llama-cpp-python unusable here ({type(e).__name__}: {e}). "
                "On Android/Termux the bundled shared library is not built "
                "for aarch64 — use core.llama_server_engine.LlamaServerEngine "
                "against a local llama-server instead."
            ) from e

        self.metric = metric
        self.top_k = top_k
        self.temperature = temperature
        self.calls = 0
        self.failures = 0
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx,
                         n_threads=n_threads, seed=seed, logits_all=True,
                         verbose=verbose)

    def generate(self, prompt: str, max_tokens: int = 256,
                 logprobs: bool = True) -> Dict[str, Any]:
        self.calls += 1
        try:
            out = self.llm.create_completion(
                prompt=prompt, max_tokens=max_tokens,
                temperature=self.temperature,
                logprobs=self.top_k if logprobs else None,
            )
            choice = out["choices"][0]
            text = choice.get("text", "")
            score = self._score(choice.get("logprobs") or {})
            return {"success": True, "text": text,
                    "logit_variance": score,
                    "metric": self.metric,
                    "n_tokens": len(text.split())}
        except Exception as e:
            self.failures += 1
            # Reported, not swallowed: the Engine treats success=False as a
            # deferral, which is the correct response to a backend that broke.
            return {"success": False, "text": "", "logit_variance": 0.0,
                    "error": f"{type(e).__name__}: {e}"}

    def _score(self, lp: Dict[str, Any]) -> float:
        if self.metric == "mean_token_prob":
            return mean_token_prob(lp.get("token_logprobs") or [])
        return one_minus_norm_entropy(lp.get("top_logprobs") or [])


# --------------------------------------------------------------------------
# Selftest. Exercises the SCORING layer against constructed logprob streams.
# It does NOT prove model integration -- no model is loaded here. Integration
# is verified by calibrate_gguf_threshold.py against a real GGUF on device.
# --------------------------------------------------------------------------

def _selftest() -> int:
    results = []

    def check(name, predicted, measured, ok):
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"       predicted: {predicted}")
        print(f"       measured : {measured}")

    # T1 -- a model that never hesitated scores 1.0
    v = mean_token_prob([0.0, 0.0, 0.0])
    check("T1 certain stream -> 1.0", "1.0", f"{v:.6f}", abs(v - 1.0) < 1e-9)

    # T2 -- ANTI-VACUITY. The metric must be able to return a LOW value.
    v = mean_token_prob([math.log(0.10)] * 5)
    check("T2 uncertain stream -> ~0.10 (metric can score low)",
          "0.10", f"{v:.6f}", abs(v - 0.10) < 1e-9)

    # T3 -- monotone: more confident input must not score lower
    a = mean_token_prob([math.log(0.5)] * 4)
    b = mean_token_prob([math.log(0.9)] * 4)
    check("T3 monotone in confidence", "score(0.9) > score(0.5)",
          f"{b:.4f} > {a:.4f}", b > a)

    # T4 -- one-hot top-k scores 1.0 under the entropy metric
    v = one_minus_norm_entropy([{ "a": 0.0, "b": -60.0, "c": -60.0 }])
    check("T4 one-hot distribution -> ~1.0", ">= 0.999", f"{v:.6f}", v >= 0.999)

    # T5 -- ANTI-VACUITY. A uniform top-k must score 0.0, not something.
    u = math.log(1.0 / 4)
    v = one_minus_norm_entropy([{ "a": u, "b": u, "c": u, "d": u }])
    check("T5 uniform distribution -> 0.0", "0.0", f"{v:.6f}", abs(v) < 1e-9)

    # T6 -- empty / malformed input degrades to 0.0, never to a confident score.
    # A backend that returns nothing must not be read as certainty.
    v1, v2 = mean_token_prob([]), one_minus_norm_entropy([])
    v3 = mean_token_prob([float("nan"), float("-inf")])
    check("T6 empty or malformed -> 0.0, never high",
          "all three 0.0", f"{v1:.4f}, {v2:.4f}, {v3:.4f}",
          v1 == 0.0 and v2 == 0.0 and v3 == 0.0)

    # T7 -- the gate boundary. Scores must land on both sides of 0.85.
    lo = mean_token_prob([math.log(0.84)] * 3)
    hi = mean_token_prob([math.log(0.86)] * 3)
    check("T7 straddles the 0.85 commit threshold",
          "one score < 0.85 and one >= 0.85",
          f"{lo:.4f} and {hi:.4f}", lo < 0.85 <= hi)

    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} scoring checks passed")
    print("NOTE: this tests the scoring layer only. No model was loaded, so it "
          "says nothing about llama.cpp integration -- run "
          "calibrate_gguf_threshold.py against a real GGUF for that.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(_selftest())
