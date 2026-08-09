#!/usr/bin/env python3
"""
calibrate_gguf_threshold.py — measure a real model's confidence distribution.

The Commit Gate admits when logit_variance >= fisher_threshold (default 0.85).
That default was chosen against MockGGUF, which returned a constant. It has
never met a real model. Run this before wiring GGUFEngine into the loop.

    llama-server -m model.gguf --host 127.0.0.1 --port 8080 -c 2048 &
    python3 calibrate_gguf_threshold.py http://127.0.0.1:8080
    python3 calibrate_gguf_threshold.py /path/to/model.gguf --metric one_minus_norm_entropy

What it does: runs a fixed prompt set at temperature 0 (the confident arm) and
at temperature 1.2 (the uncertain arm), and reports both distributions.

WHY TWO ARMS. A single distribution tells you where to put a threshold; it does
not tell you whether the threshold separates anything. If the two arms overlap
completely, the metric is not measuring confidence on this model and no
threshold will make the gate meaningful. That is a finding, not a tuning
problem, and this script says so rather than picking a number anyway.

WHAT NOT TO DO: do not lower the threshold until your outputs pass. That is
choosing the answer. If a real model's confident arm sits below 0.85, the
honest move is to record the measured separation and set the threshold from it
— then re-run probe_slc_claims.py so P8 re-proves the flip at the NEW value.
"""
import argparse
import statistics
import sys

sys.path.insert(0, ".")

PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Water boils at 100 degrees",
    "The first three prime numbers are",
    "Complete the sequence: 1, 2, 3,",
    "A triangle has how many sides?",
    "The opposite of hot is",
    "Name a primary color:",
]


def make_engine(model_path, metric, temperature, backend):
    if backend == "server":
        from core.llama_server_engine import LlamaServerEngine
        eng = LlamaServerEngine(base_url=model_path, metric=metric,
                                temperature=temperature)
        ok, msg = eng.health()
        if not ok:
            raise RuntimeError(
                f"no llama-server at {model_path} ({msg}). Start it with:\n"
                f"  llama-server -m /path/to/model.gguf --host 127.0.0.1 "
                f"--port 8080 -c 2048")
        return eng
    from core.gguf_engine import GGUFEngine
    return GGUFEngine(model_path=model_path, metric=metric,
                      temperature=temperature, verbose=False)


def arm(model_path, metric, temperature, max_tokens, label, backend="server"):
    eng = make_engine(model_path, metric, temperature, backend)
    scores, fails = [], 0
    for p in PROMPTS:
        r = eng.generate(prompt=p, max_tokens=max_tokens, logprobs=True)
        if r["success"]:
            scores.append(r["logit_variance"])
            print(f"  [{label}] {r['logit_variance']:.4f}  {p!r} -> "
                  f"{r['text'].strip()[:44]!r}")
        else:
            fails += 1
            print(f"  [{label}] FAILED  {p!r}: {r.get('error')}")
    return scores, fails


def describe(name, xs):
    if not xs:
        return f"{name}: no successful generations"
    return (f"{name}: n={len(xs)} min={min(xs):.4f} "
            f"median={statistics.median(xs):.4f} max={max(xs):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", help="llama-server base URL (default "
                    "backend), or a .gguf path with --backend bindings")
    ap.add_argument("--backend", default="server",
                    choices=("server", "bindings"),
                    help="server = llama.cpp HTTP (works on Termux); "
                         "bindings = llama-cpp-python (broken on aarch64)")
    ap.add_argument("--metric", default="mean_token_prob",
                    choices=("mean_token_prob", "one_minus_norm_entropy"))
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--current-threshold", type=float, default=0.85)
    a = ap.parse_args()

    print(f"model  : {a.model_path}")
    print(f"metric : {a.metric}")
    print(f"prompts: {len(PROMPTS)}\n")

    print("CONFIDENT ARM (temperature 0.0)")
    conf, cf = arm(a.model_path, a.metric, 0.0, a.max_tokens, "conf", a.backend)
    print("\nUNCERTAIN ARM (temperature 1.2)")
    unc, uf = arm(a.model_path, a.metric, 1.2, a.max_tokens, "unc ", a.backend)

    print("\n" + "=" * 64)
    print(describe("confident", conf))
    print(describe("uncertain", unc))

    if not conf or not unc:
        print("\nVERDICT: could not measure. Backend failures: "
              f"{cf} confident, {uf} uncertain.")
        return 2

    gap = min(conf) - max(unc)
    print(f"\nseparation (min confident - max uncertain): {gap:+.4f}")

    if gap > 0:
        suggested = round(max(unc) + gap / 2, 3)
        print(f"VERDICT: the arms SEPARATE. A threshold anywhere in "
              f"({max(unc):.4f}, {min(conf):.4f}) discriminates. "
              f"Midpoint: {suggested}")
    else:
        print("VERDICT: the arms OVERLAP. On this model the metric does not "
              "separate confident from uncertain generation, so NO threshold "
              "makes this gate meaningful. Try --metric "
              "one_minus_norm_entropy, or treat it as a finding: the Commit "
              "Gate's check 1 is not measuring what it claims on this model.")

    inside = sum(1 for x in conf if x >= a.current_threshold)
    print(f"\nAt the current threshold {a.current_threshold}: "
          f"{inside}/{len(conf)} confident generations would commit, "
          f"{sum(1 for x in unc if x >= a.current_threshold)}/{len(unc)} "
          f"uncertain ones would also commit.")
    print("If you change the threshold, re-run probe_slc_claims.py so P8 "
          "re-proves the flip at the new value.")
    print("=" * 64)
    return 0 if gap > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
