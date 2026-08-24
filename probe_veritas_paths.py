#!/usr/bin/env python3
"""
probe_veritas_paths.py — which Veritas Gate path is actually enforcing?

core/veritas_gate.py exposes two admission paths:

    evaluate()              constant-coefficient Gibbs. dH and dS are fixed
                            config constants, so dG = dH - T*dS has the same
                            sign at every reachable temperature. Its comment
                            already records this as REFUTED AND KEPT.

    evaluate_transition()   state-dependent Gibbs. dH and dS are measured from
                            the actual proposed operator change, so dG can take
                            either sign and the gate can genuinely refuse.

The refutation of evaluate() is documented. What was NOT measured is whether
the enforcing path is the one the system actually runs. This probe measures
both, and fails if the refuted path is still reachable from a live entry point
without being explicitly declared as thermal-lock-only.

REGISTERED PREDICTIONS (registered before this script was first run):

  P1  evaluate()'s Gibbs term cannot refuse anywhere in the operating band.
      Sweep T over [0, 100] C and count temperatures where the Gibbs term
      alone (excluding the hard temp_critical lock) returns refuse.
      PREDICTED: 0 refusing temperatures.

  P2  evaluate_transition() CAN refuse. Over N random proposed operator
      updates at a fixed sub-critical temperature, both ADMIT and REFUSE
      occur. PREDICTED: refusal count > 0 AND admit count > 0.
      A path that always refuses is as uninformative as one that never does.

  P3  The refuted path is more widely wired than the enforcing path.
      Count call sites of each across the repo.
      PREDICTED: evaluate() call sites > evaluate_transition() call sites.

GATE (anti-vacuity, not a prediction): P2 must produce BOTH outcomes. If the
enforcing path returns a single verdict for every input, this probe is
measuring nothing about admission and NO prediction is reported in either
direction — exit 1.

Exit 0 when the gate passes, including when a prediction is refuted: a
refutation is a kept finding, not a build failure. Exit 1 only when the gate
fails, or when the refuted path is reachable from an undeclared entry point.

Run:  python3 probe_veritas_paths.py [--seeds 2000] [--json out.json]
      python3 probe_veritas_paths.py --selftest
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from core.params import RuntimeConfig
from core.veritas_gate import VeritasGate


class FixedTemp:
    """Deterministic substrate stand-in. The probe must not depend on the
    ambient temperature of whatever device happens to run it."""

    def __init__(self, t):
        self.t = float(t)

    def read(self):
        return self.t


# Entry points that may legitimately call evaluate(): they consume only the
# hard temp_critical lock and the thermal multiplier, not the Gibbs verdict.
# Anything NOT listed here that calls evaluate() is an undeclared reliance on
# a path documented as unable to refuse.
DECLARED_THERMAL_ONLY = {
    "probe_slc_claims.py",      # measures the refutation itself
    "probe_cooler.py",          # thermal governor probe
    "tests/test_thermal_discrimination.py",
    "core/phase1_adapters.py",  # exposes both; transition path at :80
    "core/veritas_gate.py",
}

# A file may also declare itself in-band, the way vacuity_lint.py already
# expects, rather than being listed above. Honouring the existing marker
# matters: a probe that flags a file its author has already labelled teaches
# the author to ignore the probe.
INTENTIONAL_MARKER = re.compile(r"#\s*vacuity-lint:\s*intentional", re.IGNORECASE)


def declares_intentional(path):
    """True if the file carries an in-band intentional marker near its top."""
    try:
        with open(path, errors="replace") as f:
            head = "".join(next(f, "") for _ in range(15))
    except Exception:
        return False
    return bool(INTENTIONAL_MARKER.search(head))


def p1_sweep(cfg, lo=0.0, hi=100.0, step=0.5):
    """Count temperatures where the Gibbs term ALONE would refuse."""
    refusing = []
    T = lo
    while T <= hi:
        dG = cfg.dH - T * cfg.dS
        if not (dG < 0):
            refusing.append(round(T, 2))
        T += step
    # the temperature at which the sign flips, if it is real at all
    flip = (cfg.dH / cfg.dS) if cfg.dS else float("nan")
    return refusing, flip


def p2_transition(cfg, n, temp, seed=0):
    """Drive the enforcing path with random proposed operator updates."""
    gate = VeritasGate(cfg, FixedTemp(temp))
    rng = np.random.default_rng(seed)
    admits = rejects = 0
    dGs = []
    for _ in range(n):
        U = rng.normal(size=(8, 4))
        V = rng.normal(size=(8, 4))
        scale = rng.choice([0.05, 0.5, 2.0])
        U_new = U + rng.normal(size=(8, 4)) * scale
        V_new = V + rng.normal(size=(8, 4)) * scale
        dH, dS = VeritasGate.state_deltas(U, V, U_new, V_new)
        _, dG, admit, _ = gate.evaluate_transition(dH, dS, T=temp)
        dGs.append(dG)
        if admit:
            admits += 1
        else:
            rejects += 1
    return admits, rejects, dGs


def p3_call_sites(root="."):
    """Where is each path wired in?"""
    ev, tr = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "__pycache__", ".github"}]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            try:
                src = open(full, errors="replace").read()
            except Exception:
                continue
            for i, line in enumerate(src.splitlines(), 1):
                if re.search(r"\.evaluate_transition\s*\(", line):
                    tr.append((rel, i))
                elif re.search(r"\.evaluate\s*\(\s*\)", line):
                    ev.append((rel, i))
    return ev, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--temp", type=float, default=None,
                    help="fixed sub-critical temperature for P2")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    cfg = RuntimeConfig()
    temp = a.temp if a.temp is not None else max(0.0, cfg.temp_threshold - 5.0)

    print("=" * 70)
    print("VERITAS GATE — WHICH PATH IS ENFORCING?")
    print("=" * 70)
    print(f"  dH={cfg.dH}  dS={cfg.dS}  "
          f"temp_threshold={cfg.temp_threshold}  temp_critical={cfg.temp_critical}")
    print(f"  P2 probe temperature: {temp} C (sub-critical, fixed)")

    # ---------------------------------------------------------------- P1
    refusing, flip = p1_sweep(cfg)
    print("\n[P1] can evaluate()'s Gibbs term refuse in the operating band?")
    print(f"  dG = dH - T*dS ; sign flips at T = {flip:.2f} C")
    print(f"  refusing temperatures in [0,100] C, step 0.5: {len(refusing)}")
    p1_confirmed = (len(refusing) == 0)
    print(f"  -> P1 {'CONFIRMED' if p1_confirmed else 'REFUTED'} "
          f"(predicted 0)")
    if not p1_confirmed:
        print(f"     refusing at: {refusing[:8]}{' ...' if len(refusing) > 8 else ''}")

    # ---------------------------------------------------------------- P2
    admits, rejects, dGs = p2_transition(cfg, a.seeds, temp)
    print(f"\n[P2] can evaluate_transition() refuse? "
          f"({a.seeds} random proposed updates)")
    print(f"  ADMIT  {admits}")
    print(f"  REFUSE {rejects}")
    rate = 100.0 * rejects / max(1, admits + rejects)
    print(f"  refusal rate {rate:.1f}%   "
          f"dG range [{min(dGs):.3f}, {max(dGs):.3f}]")
    p2_confirmed = (admits > 0 and rejects > 0)
    print(f"  -> P2 {'CONFIRMED' if p2_confirmed else 'REFUTED'} "
          f"(predicted both outcomes occur)")

    # ------------------------------------------------------------- GATE
    print("\n[GATE] the enforcing path must produce BOTH verdicts")
    if not p2_confirmed:
        print("  GATE FAILED. evaluate_transition() returned a single verdict")
        print("  for every input, so this probe measures nothing about")
        print("  admission. No prediction is reported. Exit 1.")
        return 1
    print("  GATE PASS")

    # ---------------------------------------------------------------- P3
    ev, tr = p3_call_sites(os.path.dirname(os.path.abspath(__file__)))
    print(f"\n[P3] which path is wired in?")
    root = os.path.dirname(os.path.abspath(__file__))
    print(f"  evaluate()            {len(ev)} call sites")
    for rel, ln in ev:
        if rel in DECLARED_THERMAL_ONLY:
            flag = ""
        elif declares_intentional(os.path.join(root, rel)):
            flag = "   (declared in-band)"
        else:
            flag = "   <-- UNDECLARED"
        print(f"      {rel}:{ln}{flag}")
    print(f"  evaluate_transition() {len(tr)} call sites")
    for rel, ln in tr:
        print(f"      {rel}:{ln}")
    p3_confirmed = len(ev) > len(tr)
    print(f"  -> P3 {'CONFIRMED' if p3_confirmed else 'REFUTED'} "
          f"(predicted refuted path more widely wired)")

    undeclared = [f"{r}:{l}" for r, l in ev
                  if r not in DECLARED_THERMAL_ONLY
                  and not declares_intentional(os.path.join(root, r))]

    print("\n" + "=" * 70)
    print(f"P1 {'CONFIRMED' if p1_confirmed else 'REFUTED'} · "
          f"P2 {'CONFIRMED' if p2_confirmed else 'REFUTED'} · "
          f"P3 {'CONFIRMED' if p3_confirmed else 'REFUTED'}")
    if undeclared:
        print("\nUNDECLARED RELIANCE on the refuted path:")
        for u in undeclared:
            print(f"  {u}")
        print("\nThese entry points consume the Gibbs verdict from evaluate(),")
        print("which cannot refuse above "
              f"{flip:.1f} C. Either migrate them to evaluate_transition()")
        print("or add them to DECLARED_THERMAL_ONLY with a comment saying they")
        print("rely only on the hard temp_critical lock. Exit 1.")
        rc = 1
    else:
        print("\nNo undeclared reliance on the refuted path. Exit 0.")
        rc = 0
    print("=" * 70)

    if a.json:
        json.dump({
            "config": {"dH": cfg.dH, "dS": cfg.dS,
                       "temp_threshold": cfg.temp_threshold,
                       "temp_critical": cfg.temp_critical},
            "probe_temp": temp,
            "P1": {"refusing_temps": len(refusing), "flip_at": flip,
                   "confirmed": p1_confirmed},
            "P2": {"admits": admits, "rejects": rejects,
                   "refusal_rate": rate, "confirmed": p2_confirmed},
            "P3": {"evaluate_sites": ev, "transition_sites": tr,
                   "confirmed": p3_confirmed},
            "undeclared": undeclared,
        }, open(a.json, "w"), indent=2)
        print(f"\n-> {a.json}")

    return rc


# ------------------------------------------------------------------ selftest
def selftest():
    """The probe must be able to report BOTH a refusing and a non-refusing
    path. A probe that says 'can refuse' regardless of input is exactly the
    defect it exists to detect."""
    print("=" * 70)
    print("ANTI-VACUITY SELFTEST")
    print("=" * 70)
    ok = True
    cfg = RuntimeConfig()

    # A: the real enforcing path produces both verdicts
    admits, rejects, _ = p2_transition(cfg, 400, max(0.0, cfg.temp_threshold - 5.0))
    pa = admits > 0 and rejects > 0
    print(f"\n  A real evaluate_transition() gives both verdicts: "
          f"{'PASS' if pa else 'FAIL'} (admit={admits} refuse={rejects})")
    ok = ok and pa

    # B: a deliberately vacuous gate must be caught. Force dS so large that
    #    dG = dH - T*dS is negative for every proposed update -> always admit.
    class AlwaysAdmit(VeritasGate):
        def evaluate_transition(self, dH, dS, T=None):
            return 1.0, -1.0, True, 30.0

    g = AlwaysAdmit(cfg, FixedTemp(30.0))
    rng = np.random.default_rng(1)
    a2 = r2 = 0
    for _ in range(200):
        _, _, admit, _ = g.evaluate_transition(0.0, 0.0, T=30.0)
        a2 += admit
        r2 += not admit
    pb = not (a2 > 0 and r2 > 0)   # gate SHOULD fail on this
    print(f"  B a stuck always-admit gate is detected as vacuous: "
          f"{'PASS' if pb else 'FAIL'} (admit={a2} refuse={r2})")
    ok = ok and pb

    # C: P1's sweep must be able to find a refusing temperature when one
    #    genuinely exists -- otherwise it is not measuring anything.
    class Cfg2:
        dH, dS = 1.0, 0.02      # dG = 1.0 - 0.02T, refuses below 50 C
    refusing, flip = p1_sweep(Cfg2())
    pc = len(refusing) > 0
    print(f"  C P1 sweep finds refusals when they exist: "
          f"{'PASS' if pc else 'FAIL'} ({len(refusing)} temps, flip {flip:.1f} C)")
    ok = ok and pc

    print("\n" + "=" * 70)
    if ok:
        print("SELFTEST PASS -- the probe distinguishes a gate that can refuse")
        print("from one that cannot, and its temperature sweep can find")
        print("refusals when they exist. It is not stuck on one verdict.")
    else:
        print("SELFTEST FAIL -- do not trust this probe until fixed.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
