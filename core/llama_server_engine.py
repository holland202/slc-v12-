#!/usr/bin/env python3
"""
core/llama_server_engine.py — real model backend over llama.cpp's HTTP server.

Why this and not core/gguf_engine.py: llama-cpp-python installs on Termux but
its bundled shared library is not built for aarch64, so importing it raises
`RuntimeError: Unsupported platform` at load time. llama-server, however, is
already compiled on the device. This talks to it over HTTP with stdlib urllib
only — no bindings, no compilation.

Start the server first:

    ~/native_backend/build/bin/llama-server -m ~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \\
        --host 127.0.0.1 --port 8080 -c 2048

Then:

    from core.llama_server_engine import LlamaServerEngine
    engine.set_gguf_engine(LlamaServerEngine())

RESPONSE SHAPE. llama.cpp changed its per-token probability format across
versions, and a device may hold several builds of different ages. Two shapes
are parsed:

  legacy   completion_probabilities[i].probs[j] = {"tok_str": ..., "prob": p}
  current  completion_probabilities[i].top_logprobs[j] = {"token": ..., "logprob": lp}
           (some builds also put "logprob" directly on the token entry)

If NEITHER shape is present the call returns success=False with an explicit
error naming what it looked for. It does not fall through to a score of 0.0:
a parse failure and a genuinely unconfident model are different events, and
collapsing them would let a broken parser look like a cautious model.
"""
import json
import math
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")
from core.gguf_engine import METRICS, mean_token_prob, one_minus_norm_entropy


class LogprobShapeError(RuntimeError):
    pass


def extract_token_stats(payload: Dict[str, Any]) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Returns (selected_token_logprobs, per_token_topk_logprob_dicts).
    Raises LogprobShapeError if no recognised shape is present.
    """
    probs_list = payload.get("completion_probabilities")
    if not probs_list:
        raise LogprobShapeError(
            "no 'completion_probabilities' in response — start llama-server "
            "and pass n_probs > 0 in the request (this client sends it)"
        )

    selected: List[float] = []
    topk: List[Dict[str, float]] = []
    shape_seen = False

    for entry in probs_list:
        # --- current shape: top_logprobs with logprob values
        cand = entry.get("top_logprobs")
        if cand:
            shape_seen = True
            d = {}
            for c in cand:
                tok = c.get("token", c.get("tok_str", ""))
                lp = c.get("logprob")
                if lp is None and c.get("prob") is not None:
                    p = float(c["prob"])
                    lp = math.log(p) if p > 0 else -60.0
                if lp is not None:
                    d[tok] = float(lp)
            if d:
                topk.append(d)
            own = entry.get("logprob")
            if own is None:
                chosen = entry.get("content", entry.get("token", ""))
                own = d.get(chosen, max(d.values()) if d else None)
            if own is not None:
                selected.append(float(own))
            continue

        # --- legacy shape: probs with linear probabilities
        cand = entry.get("probs")
        if cand:
            shape_seen = True
            d = {}
            for c in cand:
                tok = c.get("tok_str", c.get("token", ""))
                p = c.get("prob")
                if p is not None:
                    p = float(p)
                    d[tok] = math.log(p) if p > 0 else -60.0
            if d:
                topk.append(d)
            chosen = entry.get("content", entry.get("tok_str", ""))
            own = d.get(chosen, max(d.values()) if d else None)
            if own is not None:
                selected.append(float(own))

    if not shape_seen:
        raise LogprobShapeError(
            "'completion_probabilities' present but held neither "
            "'top_logprobs' nor 'probs' — unrecognised llama.cpp build. "
            f"keys seen: {sorted(probs_list[0].keys()) if probs_list else []}"
        )
    return selected, topk


class LlamaServerEngine:
    def __init__(self, base_url: str = "http://127.0.0.1:8080",
                 metric: str = "mean_token_prob", temperature: float = 0.0,
                 top_k_probs: int = 5, timeout: float = 300.0,
                 seed: int = 42):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        self.base_url = base_url.rstrip("/")
        self.metric = metric
        self.temperature = temperature
        self.top_k_probs = top_k_probs
        self.timeout = timeout
        self.seed = seed
        self.calls = 0
        self.failures = 0

    def health(self) -> Tuple[bool, str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health",
                                        timeout=10.0) as r:
                return r.status == 200, f"HTTP {r.status}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def generate(self, prompt: str, max_tokens: int = 256,
                 logprobs: bool = True) -> Dict[str, Any]:
        self.calls += 1
        body = json.dumps({
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "n_probs": self.top_k_probs if logprobs else 0,
            "cache_prompt": False,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/completion", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode())
        except Exception as e:
            self.failures += 1
            return {"success": False, "text": "", "logit_variance": 0.0,
                    "error": f"{type(e).__name__}: {e}"}

        text = payload.get("content", "")
        try:
            selected, topk = extract_token_stats(payload)
        except LogprobShapeError as e:
            self.failures += 1
            # Deliberately NOT a score of 0.0 — see module docstring.
            return {"success": False, "text": text, "logit_variance": 0.0,
                    "error": f"LogprobShapeError: {e}"}

        score = (mean_token_prob(selected) if self.metric == "mean_token_prob"
                 else one_minus_norm_entropy(topk))
        return {"success": True, "text": text, "logit_variance": score,
                "metric": self.metric, "n_tokens": len(selected)}


# --------------------------------------------------------------------------
# Selftest. Runs a local HTTP server returning canned llama.cpp payloads in
# BOTH shapes, so the parsing and scoring path is exercised end to end over
# real sockets. No model is loaded: this proves the client, not the model.
# --------------------------------------------------------------------------

def _selftest() -> int:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    lp = math.log
    CASES = {
        # confident, current shape
        "current_confident": {"content": " Paris", "completion_probabilities": [
            {"content": " Paris", "logprob": lp(0.97),
             "top_logprobs": [{"token": " Paris", "logprob": lp(0.97)},
                              {"token": " Lyon", "logprob": lp(0.03)}]}]},
        # unconfident, legacy shape
        "legacy_unconfident": {"content": " maybe", "completion_probabilities": [
            {"content": " maybe", "probs": [{"tok_str": " maybe", "prob": 0.26},
                                            {"tok_str": " perhaps", "prob": 0.25},
                                            {"tok_str": " possibly", "prob": 0.25},
                                            {"tok_str": " likely", "prob": 0.24}]}]},
        # unrecognised build
        "broken": {"content": " x", "completion_probabilities": [{"content": " x"}]},
        # no probabilities at all
        "empty": {"content": " x"},
    }
    state = {"case": "current_confident"}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps(CASES[state["case"]]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    results = []

    def check(name, predicted, measured, ok):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"       predicted: {predicted}")
        print(f"       measured : {measured}")

    eng = LlamaServerEngine(base_url=url)

    state["case"] = "current_confident"
    r = eng.generate("The capital of France is", max_tokens=4)
    check("S1 current shape parses, scores high",
          "success and score >= 0.95",
          f"success={r['success']} score={r['logit_variance']:.4f}",
          r["success"] and r["logit_variance"] >= 0.95)

    state["case"] = "legacy_unconfident"
    r = eng.generate("x", max_tokens=4)
    check("S2 legacy shape parses, scores LOW (anti-vacuity)",
          "success and score <= 0.35",
          f"success={r['success']} score={r['logit_variance']:.4f}",
          r["success"] and r["logit_variance"] <= 0.35)

    state["case"] = "legacy_unconfident"
    eng_e = LlamaServerEngine(base_url=url, metric="one_minus_norm_entropy")
    r = eng_e.generate("x", max_tokens=4)
    check("S3 entropy metric scores a near-uniform top-k near 0",
          "score <= 0.05", f"score={r['logit_variance']:.4f}",
          r["logit_variance"] <= 0.05)

    state["case"] = "broken"
    r = eng.generate("x", max_tokens=4)
    check("S4 unrecognised build -> success=False, NOT a confident score",
          "success=False and error mentions the shapes",
          f"success={r['success']} error={r.get('error', '')[:60]}",
          r["success"] is False and "top_logprobs" in r.get("error", ""))

    state["case"] = "empty"
    r = eng.generate("x", max_tokens=4)
    check("S5 no probabilities -> success=False, not silent zero",
          "success=False", f"success={r['success']}", r["success"] is False)

    ok, msg = LlamaServerEngine(base_url="http://127.0.0.1:1").health()
    check("S6 health() on a dead server reports False cleanly",
          "False, no traceback", f"{ok}, {msg[:40]}", ok is False)

    srv.shutdown()
    n = sum(results)
    print(f"\n{n}/{len(results)} client checks passed")
    print("NOTE: this proves the HTTP client and the parsing of both llama.cpp "
          "response shapes. No model was loaded. Real integration is verified "
          "by calibrate_gguf_threshold.py against a running llama-server.")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(_selftest())
