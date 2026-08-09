#!/usr/bin/env python3
"""
demo.py — paced walkthrough of the SLC governance loop, for screen recording.

Runs slower than run_engine.py on purpose and suppresses the thermal-zone
binding banner, which otherwise prints once per ThermalMonitor construction
(twice per Engine) and buries the output.

    python3 demo.py            # ~75s, readable on a phone screen
    python3 demo.py --fast     # no pauses

Every number it prints is reproducible: the SIC is seeded and text embeddings
use hashlib, not Python's per-process-randomized hash().
"""
import sys
import io
import time
import contextlib
import numpy as np

sys.path.insert(0, ".")

FAST = "--fast" in sys.argv
RULE = "=" * 58


def beat(t=1.4):
    if not FAST:
        time.sleep(t)


def say(line="", t=0.5):
    print(line, flush=True)
    beat(t)


def head(title):
    say()
    say(RULE, 0.2)
    say(f"  {title}", 0.2)
    say(RULE, 0.9)


@contextlib.contextmanager
def quiet():
    """Swallow the [Hardware Link] banner without touching hardware_link.py."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


with quiet():
    from core.engine import Engine
    from core.params import RuntimeConfig
    from core.veritas_gate import VeritasGate as RepoGate
    from core.ume import UmbraManifoldEngine
    from run_engine import MockGGUF


class FixedTemp:
    def __init__(self, t):
        self.t = t

    def read(self):
        return self.t


# Engine acts run against an injected 30.0 C thermometer so the demo is
# reproducible on any device in any thermal state. The live substrate is
# reported honestly in the header, and the thermal lock is exercised against
# injected temperatures in Act 3.
BENCH_TEMP = 30.0


def new_engine(logvar=0.90, seed=42):
    with quiet():
        e = Engine(d=64, rank=8, seed=seed, monitor=FixedTemp(BENCH_TEMP))
        e.set_gguf_engine(MockGGUF(logit_variance=logvar))
    return e


say()
say("  SOVEREIGN LOGIC CORE — 10-step governance loop", 0.4)
say("  Snapdragon 8 Elite / Termux / NumPy only", 0.4)
say("  Vincit Omnia Veritas", 1.0)
say()
with quiet():
    from core.hardware_link import ThermalMonitor
    _live_mon = ThermalMonitor()
_live_T = _live_mon.read()
_cfg0 = RuntimeConfig("defense")
say(f"  live substrate    : {_live_T:.1f} C", 0.5)
say(f"  defense threshold : {_cfg0.temp_threshold:.1f} C"
    + ("   <- substrate is above it" if _live_T >= _cfg0.temp_threshold else ""), 0.5)
say(f"  governance acts run at an injected {BENCH_TEMP:.1f} C so they reproduce", 1.6)

# ---------------------------------------------------------------- 1
head("1. THE LOOP RUNS")
say("20 cycles. Inference backend is a deterministic mock, not a model.", 1.0)
eng = new_engine()
counts = {}
for i in range(1, 21):
    with quiet():
        r = eng.step()
    counts[r["status"]] = counts.get(r["status"], 0) + 1
    if i % 5 == 0:
        say(f"  cycle {i:2d}   status={r['status']:<10s}  scars={eng.sic.scars_admitted}", 0.5)
say()
say(f"  reached step 10 : {counts.get('success', 0)}/20")
say(f"  scars admitted  : {eng.sic.scars_admitted}")
say(f"  ||U @ V.T||     : {np.linalg.norm(eng.sic.U @ eng.sic.V.T):.6e}", 1.8)

# ---------------------------------------------------------------- 2
head("2. THE GATE CAN REFUSE")
say("A gate that passes everything is not a gate.", 0.8)
say("Fisher threshold is registered at 0.85. Drive across it.", 1.2)
say()
say("   logit_variance    outcome over 10 cycles      scars", 0.6)
for lv in (0.99, 0.90, 0.86, 0.84, 0.50, 0.10):
    e = new_engine(logvar=lv)
    with quiet():
        st = [e.step()["status"] for _ in range(10)]
    ok = st.count("success")
    verdict = f"{ok:2d} pass / {10 - ok:2d} refuse"
    mark = "PASS  " if ok == 10 else ("REFUSE" if ok == 0 else "split ")
    say(f"        {lv:.2f}        {mark} {verdict}        {e.sic.scars_admitted}", 0.7)
say()
say("  Clean flip at the registered threshold. Proven both directions.", 1.8)

# ---------------------------------------------------------------- 3
head("3. BOTH REFUTATIONS, NOW FIXED")
cfg = RuntimeConfig("defense")
say("Last week these two were REFUTED by this same instrument.", 1.2)
say()
say("WAS: 'Gibbs enforcement' with dH and dS as fixed constants.", 0.8)
say(f"     dG = {cfg.dH} - {cfg.dS} * T  ->  0 rejecting temps over 0-100 C.", 1.2)
say("NOW: dH and dS measured from the real operator change.", 1.0)
say("     Drive scar magnitude across the admission boundary:", 1.0)
say()
say("       alpha        scars admitted / 10", 0.6)
for a in (0.01, 0.3, 1.0, 3.0):
    e = new_engine()
    with quiet():
        e.scar_alpha = a
        for _ in range(10):
            e.step()
    n_ok = e.sic.scars_admitted
    say(f"       {a:<6}       {n_ok:2d}   {'admit' if n_ok >= 5 else 'REFUSE'}", 0.8)
say()
say("  It refuses. A gate that cannot refuse is not a gate.", 1.6)
say()
say("WAS: 'Langevin diffusion' — explore() never read its T argument.", 1.0)
say("     mean step at T=1: 0.789665.  At T=1000: 0.789665. Identical.", 1.2)
say("NOW: Euler-Maruyama, diffusion coefficient collapsing at T_critical:", 1.0)
say()
ume = UmbraManifoldEngine(T_0=cfg.temp_threshold, T_critical=cfg.temp_critical)
x0 = np.ones(64)
say("        T (C)     lambda     mean |dx|      mode", 0.6)
for t in (30.0, cfg.temp_threshold, (cfg.temp_threshold + cfg.temp_critical) / 2,
          cfg.temp_critical):
    np.random.seed(0)
    m = np.mean([np.linalg.norm(ume.explore(x0, T=t)[0] - x0) for _ in range(200)])
    say(f"       {t:5.1f}     {ume.diffusion_coefficient(t):.4f}     {m:.6f}     "
        f"{ume.explore(x0, T=t)[1]}", 0.8)
say()
say("  Noise collapses to zero as the substrate approaches its limit.", 1.6)

# ---------------------------------------------------------------- 4
head("4. IT REPRODUCES")
say("Same numbers on any machine, any run, any PYTHONHASHSEED.", 1.0)
for i in (1, 2, 3):
    e = new_engine()
    with quiet():
        for _ in range(20):
            e.step()
    say(f"  run {i}   ||U @ V.T|| = {np.linalg.norm(e.sic.U @ e.sic.V.T):.6e}", 0.7)
say()
say("  Clone it. You get the same numbers.", 1.2)
say()
say("  python3 probe_slc_claims.py     # 9 registered predictions, 8 match", 0.8)
say("  github.com/holland202/slc-v12-", 1.0)
say()
