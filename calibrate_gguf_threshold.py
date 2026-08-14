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
    "The largest ocean on Earth is",
    "The chemical symbol for gold is",
    "Ice melts at",
    "The author of Hamlet was",
    "10 divided by 2 equals",
    "The planet closest to the Sun is",
    "A hexagon has",
    "The past tense of the verb go is",
    "Sound travels faster through",
    "The square root of 81 is",
    "Photosynthesis requires",
    "The capital of Japan is",
    "A baby cat is called a",
    "Iron rusts when exposed to",
    "The speed of light is approximately",
    "Seven times eight equals",
    "The freezing point of water in Fahrenheit is",
    "Mount Everest is located in",
    "The human heart has how many chambers?",
    "Bread is made primarily from",
    "The opposite of ancient is",
    "The atomic number of oxygen is",
]

INDETERMINATE_PROMPTS = [
    "Should I move to Portland or Austin? List three reasons.",
    "Write a haiku about rain.",
    "Continue this story: The door creaked open and\u2014",
    "What is the best flavor of ice cream? Explain your choice.",
    "Describe a color you have never seen.",
    "Invent a new holiday and explain its traditions.",
    "Is it better to be kind or to be honest? Argue both sides.",
    "Rewrite the ending of Romeo and Juliet so that everyone survives.",
    "Write a short poem about a locked door.",
    "What would you name a newly discovered planet? Explain.",
    "Describe the smell of a room you have never entered.",
    "Continue this line: The last train had already left, and",
    "Which season is best for starting something new? Argue it.",
    "Invent a word for the feeling of finishing a long project.",
    "Should a museum charge admission? Make the case either way.",
    "Describe a piece of music that does not exist.",
    "What should a city do with an abandoned railway? Propose something.",
    "Write the opening sentence of a novel set underwater.",
    "Is it better to travel alone or with others? Take a side.",
    "Imagine a sport played in low gravity and describe one rule.",
    "What color should a hospital waiting room be? Justify it.",
    "Continue this: She opened the letter and immediately",
    "Invent a superstition and explain where it came from.",
    "Describe an animal that evolved on a planet with no light.",
    "Should children learn cursive? Argue both positions.",
    "Write a two-line farewell note from someone leaving a job.",
    "What is the most underrated tool in a kitchen? Defend it.",
    "Describe the texture of a dream you cannot remember.",
    "Invent a tradition for the first day of winter.",
    "Would you rather read minds or become invisible? Explain.",
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


def arm(model_path, metric, temperature, max_tokens, label, backend="server", prompts=None):
    eng = make_engine(model_path, metric, temperature, backend)
    scores, fails = [], 0
    for p in (prompts or PROMPTS):
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

    print("\nINDETERMINATE \u2014 CONFIDENT ARM (temperature 0.0)")
    iconf, icf = arm(a.model_path, a.metric, 0.0, a.max_tokens, "icon", a.backend, prompts=INDETERMINATE_PROMPTS)
    print("\nINDETERMINATE \u2014 UNCERTAIN ARM (temperature 1.2)")
    iunc, iuf = arm(a.model_path, a.metric, 1.2, a.max_tokens, "iunc", a.backend, prompts=INDETERMINATE_PROMPTS)

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

    # --- P12: Indeterminate paired analysis ---
    ipairs = list(zip(INDETERMINATE_PROMPTS, iconf, iunc))
    iwins = sum(1 for _, c, u in ipairs if c > u)
    ilosses = sum(1 for _, c, u in ipairs if c < u)
    ities = len(ipairs) - iwins - ilosses
    print(f"\nINDETERMINATE paired by prompt: {iwins} wins, {ities} ties, {ilosses} losses")
    for pr, c, u in ipairs:
        mark = "win " if c > u else ("tie " if c == u else "LOSS")
        print(f"  {mark}  conf {c:.4f}  unc {u:.4f}  {pr}")
    in_eff = iwins + ilosses
    ipval = (sum(math.comb(in_eff, k) for k in range(iwins, in_eff + 1)) / 2 ** in_eff
             if in_eff else 1.0)
    print(f"\nINDETERMINATE sign test (one-sided): p = {ipval:.4f} over {in_eff} non-tied pairs")
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

    # --- P12 predictions ---
    print("\n" + "-" * 64)
    print("P12 ANTI-VACUITY PREDICTIONS (registered before run):")
    print("  P12a: Temperature slope on indeterminate class is materially flatter than on factual class.")
    factual_slope = (sum(unc)/len(unc) - sum(conf)/len(conf)) if conf and unc else 0
    indet_slope = (sum(iunc)/len(iunc) - sum(iconf)/len(iconf)) if iconf and iunc else 0
    print(f"  Measured \u2014 factual slope: {factual_slope:.4f}, indeterminate slope: {indet_slope:.4f}")
    # P12a needs BOTH a gap wider than the band AND a confidence interval on the
    # indeterminate slope that excludes the factual slope. At n=8 the gap cleared
    # the band by 0.0166 while the standard error was 0.0498 - the band's full
    # width - so a bare gap test reported noise as a result. Gap alone is not
    # enough; the instrument has to be able to tell the two slopes apart.
    gap = abs(indet_slope - factual_slope)
    ipairs = [a - b for a, b in zip(iconf, iunc)]
    ni = len(ipairs)
    if ni >= 2:
        mi = sum(ipairs) / ni
        var = sum((x - mi) ** 2 for x in ipairs) / (ni - 1)
        se = (var ** 0.5) / (ni ** 0.5)
    else:
        se = float("inf")
    # 1.96 is the large-sample normal quantile; at n<30 this is mildly optimistic
    # and the interval should be read as approximate.
    lo, hi = abs(mi) - 1.96 * se, abs(mi) + 1.96 * se
    excludes = not (lo <= abs(factual_slope) <= hi)
    print(f"  Indeterminate paired diffs: n={ni}, se={se:.4f}, "
          f"95% CI [{lo:.4f}, {hi:.4f}]; factual |slope|={abs(factual_slope):.4f}")
    if gap <= 0.05:
        status = "FAIL — slopes within ±0.05"
    elif not excludes:
        status = (f"UNDECIDED — gap {gap:.4f} clears the band but the CI on the "
                  f"indeterminate slope contains the factual slope (se={se:.4f}). "
                  f"Underpowered at n={ni}.")
    else:
        status = (f"PASS — gap {gap:.4f} > 0.05 and the CI excludes the factual "
                  f"slope (se={se:.4f}, {gap/se:.2f} SE)")
    print(f"  P12a status: {status}")
    print("  P12b: Triangle prompt does not invert on indeterminate arm.")
    print("  P12b status: VOID — no triangle prompt is in the indeterminate set, "
          "so no outcome could have failed this prediction. A prediction that can "
          "only pass is not a prediction. Do not record it as PASS.")
    print("  P12c: Sign test on indeterminate arm yields p > 0.05.")
    status2 = "PASS — p > 0.05" if ipval > 0.05 else "FAIL — p <= 0.05"
    print(f"  P12c status: {status2}")
    if p12a_fail and ipval <= 0.05:
        print("\nNULL: All three P12 predictions failed. mean_token_prob is a temperature tracker,")
        print("      not an uncertainty metric. This finding is registered; threshold not lowered.")

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
