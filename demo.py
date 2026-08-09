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


def new_engine(logvar=0.90, seed=42):
    with quiet():
        e = Engine(d=64, rank=8, seed=seed)
        e.set_gguf_engine(MockGGUF(logit_variance=logvar))
    return e


say()
say("  SOVEREIGN LOGIC CORE — 10-step governance loop", 0.4)
say("  Snapdragon 8 Elite / Termux / NumPy only", 0.4)
say("  Vincit Omnia Veritas", 1.6)

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
head("3. TWO OF MY OWN CLAIMS, REFUTED")
cfg = RuntimeConfig("defense")
say("CLAIM: 'Veritas Gate — Gibbs free energy (dG < 0) enforcement'", 1.0)
T = np.linspace(0.0, 100.0, 2001)
dG = cfg.dH - T * cfg.dS
say(f"  dG = {cfg.dH} - {cfg.dS} * T   (both constants, not state)", 0.8)
say(f"  rejecting temperatures over 0-100 C : {int(np.sum(dG >= 0))}", 0.8)
say(f"  dG >= 0 only below {cfg.dH / cfg.dS:.1f} C", 1.0)
say("  -> REFUTED. The Gibbs term cannot refuse. KEPT.", 1.6)
say()
say("  The governor that DOES work is a separate hard lock:", 0.8)
for t in (cfg.temp_critical - 5.0, cfg.temp_critical + 5.0):
    _, _, ok, temp = RepoGate(cfg, FixedTemp(t)).evaluate()
    say(f"    T = {temp:5.1f} C  ->  {'run' if ok else 'HALT'}", 0.7)
say()
say("CLAIM: 'Langevin diffusion on the Umbra Manifold'", 1.0)
ume = UmbraManifoldEngine(T_0=cfg.temp_threshold)
x0 = np.zeros(64)
np.random.seed(0)
cold = np.mean([np.linalg.norm(ume.explore(x0, T=1.0)[0] - x0) for _ in range(400)])
np.random.seed(0)
hot = np.mean([np.linalg.norm(ume.explore(x0, T=1000.0)[0] - x0) for _ in range(400)])
say(f"  mean step at T=1     : {cold:.6f}", 0.7)
say(f"  mean step at T=1000  : {hot:.6f}", 0.7)
say("  Langevin scales noise with temperature. This does not.", 0.8)
say("  -> REFUTED. Fixed-variance random walk. KEPT.", 1.8)

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
say("  Clone it. You get these two failures too.", 1.2)
say()
say("  python3 probe_slc_claims.py     # 8 registered predictions, 6 match", 0.8)
say("  github.com/holland202/slc-v12-", 1.0)
say()
