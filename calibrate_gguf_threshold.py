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
import math
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
        print(f"\nVERDICT: could not measure. Backend failures: "
              f"{cf} confident, {uf} uncertain.")
        return 2

    # ---------------- PAIRED analysis ----------------
    # CORRECTION (2026-08-09): the first version of this script compared
    # min(confident) to max(uncertain) and declared OVERLAP if they crossed.
    # That is a worst-case bound over a handful of samples: on TinyLlama one
    # tie and one inversion made it report "no threshold works" when the
    # paired data showed 6 wins, 1 tie, 1 loss. Same prompt, both
    # temperatures, is the comparison that carries information.
    pairs = list(zip(PROMPTS, conf, unc))
    wins = sum(1 for _, c, u in pairs if c > u)
    losses = sum(1 for _, c, u in pairs if c < u)
    ties = len(pairs) - wins - losses
    print(f"\npaired by prompt: {wins} wins, {ties} ties, {losses} losses")
    for pr, c, u in pairs:
        mark = "win " if c > u else ("tie " if c == u else "LOSS")
        print(f"  {mark}  conf {c:.4f}  unc {u:.4f}  {pr!r}")

    n_eff = wins + losses
    pval = (sum(math.comb(n_eff, k) for k in range(wins, n_eff + 1)) / 2 ** n_eff
            if n_eff else 1.0)
    print(f"\nsign test (one-sided): p = {pval:.4f} over {n_eff} non-tied pairs")
    if pval > 0.05:
        print("  NOT significant at this sample size. Treat any threshold "
              "below as provisional and re-run with more prompts.")

    # ---------------- threshold sweep ----------------
    best = None
    for i in range(1001):
        th = i / 1000.0
        tp = sum(1 for x in conf if x >= th)
        fp = sum(1 for x in unc if x >= th)
        J = tp / len(conf) - fp / len(unc)
        if best is None or J > best[1]:
            best = (th, J, tp, fp)
    th, J, tp, fp = best
    cur = a.current_threshold
    tpc = sum(1 for x in conf if x >= cur)
    fpc = sum(1 for x in unc if x >= cur)
    Jc = tpc / len(conf) - fpc / len(unc)

    print(f"\nthreshold sweep (Youden J = TPR - FPR):")
    print(f"  current {cur:.3f}: {tpc}/{len(conf)} confident commit, "
          f"{fpc}/{len(unc)} uncertain commit, J = {Jc:+.3f}")
    print(f"  best    {th:.3f}: {tp}/{len(conf)} confident commit, "
          f"{fp}/{len(unc)} uncertain commit, J = {J:+.3f}")

    print("\n" + "-" * 64)
    if J <= 0.0:
        print("VERDICT: NO threshold separates the arms on this model. The "
              "metric is not measuring confidence here. That is a finding, "
              "not a tuning problem -- try --metric one_minus_norm_entropy "
              "before concluding the gate is unusable.")
        rc = 1
    elif Jc <= 0.0:
        print(f"VERDICT: the arms SEPARATE (best J = {J:+.3f} at {th:.3f}), but "
              f"the CURRENT threshold {cur:.3f} has J = {Jc:+.3f} -- it "
              f"discriminates nothing on this model. It admits "
              f"{tpc}/{len(conf)} good generations and {fpc}/{len(unc)} bad "
              f"ones. A gate set here refuses almost everything, which is the "
              f"mirror image of a gate that refuses nothing. Both are inert.")
        rc = 1
    else:
        print(f"VERDICT: the arms separate and the current threshold "
              f"{cur:.3f} has J = {Jc:+.3f}. Best available is {J:+.3f} at "
              f"{th:.3f}.")
        rc = 0

    print("\nCONFOUND, read before trusting any number above: this metric "
          "measures the model's CONFIDENCE, not its CORRECTNESS. Degenerate "
          "repetition ('A. B. C. D.') is highly predictable and scores HIGH. "
          "A confident wrong answer commits. The Commit Gate's other four "
          "checks are what bound the damage.")
    print("If you change the threshold, re-run probe_slc_claims.py so P8 "
          "re-proves the flip at the new value.")
    print("=" * 64)
    return rc


if __name__ == "__main__":
    sys.exit(main())
