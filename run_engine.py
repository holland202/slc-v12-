#!/usr/bin/env python3
"""
run_engine.py — entry point for the 10-step Phase-2 governance loop.

The repo's run_slc.py drives a 5-module / 3-cycle loop. This drives
core/engine.py, which is the 10-step loop the architecture docs describe.

Usage:
    python3 run_engine.py [cycles] [sector]

The inference backend is a MockGGUF: deterministic, seeded, no model file.
It exists so the governance path (steps 4-7) can be exercised and audited
without a GGUF present. It generates nothing meaningful and must never be
reported as model output. Swap it for a real backend via
engine.set_gguf_engine(obj) where obj.generate(prompt, max_tokens, logprobs)
returns {"success": bool, "text": str, "logit_variance": float}.
"""
import sys
import hashlib
import numpy as np

sys.path.insert(0, ".")
from core.engine import Engine


class MockGGUF:
    """Deterministic stand-in for a GGUF inference engine."""

    def __init__(self, seed: int = 42, logit_variance: float = 0.90):
        self.rng = np.random.default_rng(seed)
        self.logit_variance = logit_variance
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int = 256, logprobs: bool = False):
        self.calls += 1
        h = hashlib.sha256(f"{prompt}|{self.calls}".encode()).hexdigest()
        return {
            "success": True,
            "text": f"[mock-{self.calls}] {h[:64]}",
            "logit_variance": float(self.logit_variance),
        }


def main():
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sector = sys.argv[2] if len(sys.argv) > 2 else "defense"

    engine = Engine(d=64, rank=8, seed=42)
    engine.veritas_gate = type(engine.veritas_gate)(sector=sector)
    engine.set_gguf_engine(MockGGUF())

    print(f"--- SLC 10-STEP LOOP | sector={sector} | cycles={cycles} ---")
    print("--- inference backend: MockGGUF (deterministic, not a model) ---\n")

    counts = {}
    reasons = {}
    for _ in range(cycles):
        r = engine.step()
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        cg = r["steps"].get("06_commit_gate", {})
        if cg.get("rejection_reason"):
            reasons[cg["rejection_reason"]] = reasons.get(cg["rejection_reason"], 0) + 1
        elif cg.get("error"):
            reasons[f"exception: {cg['error']}"] = reasons.get(f"exception: {cg['error']}", 0) + 1

    print(f"cycles run        : {cycles}")
    print("status counts     :")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {v:4d}  {k}")
    if reasons:
        print("commit-gate stops :")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {v:4d}  {k}")
    print(f"\nscars admitted    : {engine.sic.scars_admitted}")
    print(f"||U @ V.T||       : {np.linalg.norm(engine.sic.U @ engine.sic.V.T):.6e}")
    print("                    (0.0 means the identity operator never moved)")

    reached = counts.get("success", 0)
    print(f"\ncycles reaching step 10: {reached}/{cycles}")
    return 0 if reached > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
