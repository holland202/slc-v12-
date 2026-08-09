#!/usr/bin/env python3
"""
probe_slc_claims.py — registered probes against SLC v12 as shipped.

Each probe prints PREDICTED beside MEASURED. A probe that only prints a
number is a log line, not a guard (CLAUDE.md, anti-vacuity rule).

Run from the repo root:  python3 probe_slc_claims.py
Exit 0 = every probe's measured result matched its registered prediction.
Exit 1 = at least one prediction was wrong (that is a finding, not a crash).
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from core.params import RuntimeConfig
from core.veritas_gate import VeritasGate
from core.sic import SICManifold
from core.ume import UmbraManifoldEngine

RESULTS = []


def record(pid, claim, predicted, measured, ok, blocked=False):
    """
    ok=True MATCH, ok=False MISMATCH, blocked=True BLOCKED.

    BLOCKED is not a failure. It means the instrument could not execute -- the
    substrate was thermally locked, a file was missing -- so the prediction was
    never tested. Reporting that as MISMATCH would mean a hot phone reads as a
    refuted claim, which is how a thermal event gets mistaken for broken code.
    """
    v = "BLOCKED" if blocked else ("MATCH" if ok else "MISMATCH")
    RESULTS.append((pid, claim, predicted, measured, ok, blocked))
    print(f"\n[{pid}] {claim}")
    print(f"      PREDICTED : {predicted}")
    print(f"      MEASURED  : {measured}")
    print(f"      VERDICT   : {v}")


class FixedTemp:
    """Thermometer stub. Lets us inject temperatures the device never reaches."""
    def __init__(self, t):
        self.t = t

    def read(self):
        return self.t


# Engine probes run against an INJECTED thermometer, not the live substrate.
# Measured on the S25 Ultra 2026-08-09: cpu-0-1-1 idles at 44.8 C and passes
# 65 C under numpy load, while the defense profile's temp_threshold is 36.5 C.
# Probes that read the live sensor therefore reported MISMATCH for P1/P7/P8
# whenever the phone was warm -- a weather report, not an experiment. The
# thermal lock itself is still tested, against injected temperatures, in P2.
BENCH_TEMP = 30.0

# ---------------------------------------------------------------- P0
# Device reality check. Not a code claim -- a substrate claim.
try:
    from core.hardware_link import ThermalMonitor
    import io as _io0, contextlib as _ctx0
    _b = _io0.StringIO()
    with _ctx0.redirect_stdout(_b):
        _mon = ThermalMonitor()
    _live = _mon.read()
    _cfg0 = RuntimeConfig("defense")
    record(
        "P0", "This substrate can run the defense profile (live T < temp_threshold)",
        f"live compute temperature below {_cfg0.temp_threshold} C",
        f"live max compute zone = {_live:.2f} C vs threshold {_cfg0.temp_threshold} C",
        _live < _cfg0.temp_threshold,
    )
except Exception as _e:
    record("P0", "This substrate can run the defense profile",
           "readable thermometer", f"{type(_e).__name__}: {_e}", False)

# ---------------------------------------------------------------- P1
# Claim under test: "Veritas Gate - Gibbs free energy (dG < 0) enforcement".
# A gate that enforces must be able to refuse. dH and dS are now measured from
# the actual operator change, so drive scar magnitude across the admission
# boundary and require BOTH outcomes.
cfg = RuntimeConfig("defense")
import io as _io, contextlib as _ctx
from core.engine import Engine as _Eng
from run_engine import MockGGUF as _Mock


def _admitted_at(alpha):
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        e = _Eng(d=64, rank=8, seed=42, scar_alpha=alpha,
                 monitor=FixedTemp(BENCH_TEMP))
        e.set_gguf_engine(_Mock(logit_variance=0.90))
        for _ in range(10):
            e.step()
    return e.sic.scars_admitted


small, large = _admitted_at(0.01), _admitted_at(3.0)
record(
    "P1", "Gibbs admission can BOTH admit and refuse a real state transition",
    "alpha=0.01 -> 10/10 admitted; alpha=3.0 -> fewer than 5/10 admitted",
    f"alpha=0.01 -> {small}/10 admitted; alpha=3.0 -> {large}/10 admitted",
    small == 10 and large < 5,
)

# ---------------------------------------------------------------- P1b
# The legacy evaluate() path, which run_slc.py still uses, was left unchanged
# on purpose so the 3-cycle loop and the thermal suites keep their behaviour.
# It is still structurally unable to refuse. REFUTED, KEPT.
temps = np.linspace(0.0, 100.0, 2001)
dG_vals = cfg.dH - temps * cfg.dS
n_reject = int(np.sum(dG_vals >= 0))
record(
    "P1b", "Legacy evaluate() constant-coefficient path can refuse (run_slc.py uses this)",
    "at least 1 rejecting temperature in [0, 100] C",
    f"{n_reject} rejecting temps; dG = {cfg.dH} - {cfg.dS}*T, so dG>=0 only for "
    f"T <= {cfg.dH / cfg.dS:.1f} C. The 10-step engine uses evaluate_transition() "
    f"instead, which P1 shows CAN refuse; run_slc.py has not been migrated.",
    n_reject > 0,
)

# ---------------------------------------------------------------- P2
# The hard thermal lock is a separate mechanism. Does IT discriminate?
below = VeritasGate(cfg, FixedTemp(cfg.temp_critical - 5.0)).evaluate()
above = VeritasGate(cfg, FixedTemp(cfg.temp_critical + 5.0)).evaluate()
p2_ok = (below[2] is True) and (above[2] is False)
record(
    "P2", "Hard thermal lock halts above temp_critical and passes below",
    f"pass at {cfg.temp_critical - 5.0}C, halt at {cfg.temp_critical + 5.0}C",
    f"pass={below[2]} at {below[3]}C, pass={above[2]} at {above[3]}C",
    p2_ok,
)

# ---------------------------------------------------------------- P3
# Claim under test: "Langevin diffusion on the Umbra Manifold".
# Langevin diffusion scales noise with temperature. Vary T by 3 orders of
# magnitude and compare the step-size distribution.
ume = UmbraManifoldEngine(T_0=cfg.temp_threshold)
x0 = np.zeros(64)
np.random.seed(0)
cold = np.array([np.linalg.norm(ume.explore(x0, T=1.0)[0] - x0) for _ in range(400)])
np.random.seed(0)
hot = np.array([np.linalg.norm(ume.explore(x0, T=1000.0)[0] - x0) for _ in range(400)])
identical = np.array_equal(cold, hot)
lam_lo = ume.diffusion_coefficient(cfg.temp_threshold - 5.0)
lam_hi = ume.diffusion_coefficient(ume.T_critical)
record(
    "P3", "UME step size responds to temperature (Langevin thermal coupling)",
    "mean step at T=1000 differs from T=1, and lambda -> 0.0 at T_critical",
    f"mean|dx| T=1: {cold.mean():.6f}  T=1000: {hot.mean():.6f}  "
    f"bitwise-identical={identical}; lambda(cool)={lam_lo:.4f}, "
    f"lambda(T_critical={ume.T_critical:.1f})={lam_hi:.4f}",
    (not identical) and lam_hi == 0.0 and lam_lo > 0.0,
)

# ---------------------------------------------------------------- P4
# Claim under test: SIC is "path-dependent operator evolution" that is stable
# enough to run as a daemon. run_slc.py only ever executes 3 cycles.
sic = SICManifold(dim=64, rank=8)
rng = np.random.default_rng(7)
norms = []
diverged_at = None
for i in range(1, 1001):
    x = rng.normal(0, 0.5, 64)
    a = rng.normal(0, 0.5, 64)
    ax = np.abs(x)
    p = np.clip(ax / (ax.sum() + 1e-12), 1e-12, 1.0)
    H = float(-np.sum(p * np.log(p)))
    sic.scar_update(x, a, H)
    n = float(np.linalg.norm(sic.U))
    norms.append(n)
    if diverged_at is None and (not np.isfinite(n) or n > 1e6):
        diverged_at = i
record(
    "P4", "SIC operator norm stays finite over 1000 scars",
    "||U|| finite and < 1e6 at cycle 1000",
    f"||U|| at cycle 1: {norms[0]:.4f}, 10: {norms[9]:.4e}, "
    f"1000: {norms[-1]:.4e}; first blowup at cycle {diverged_at}",
    diverged_at is None,
)

# ---------------------------------------------------------------- P5
# The uploaded Phase-2 SIC (core/sic_enhanced.py) is the one core/engine.py
# writes scars through. Does a "successful" scar change the manifold?
try:
    from core.sic_enhanced import ScarredIdentityChronicle
    esic = ScarredIdentityChronicle(d=64, rank=8, seed=42)
    U_before = esic.U.copy()
    V_before = esic.V.copy()
    _prng = np.random.default_rng(42)
    admitted = [esic.update(_prng.standard_normal(64).astype(np.float32), alpha=0.01)
                for _ in range(50)]
    dU = float(np.linalg.norm(esic.U - U_before))
    dV = float(np.linalg.norm(esic.V - V_before))
    # CORRECTION (first draft asked only ||dU|| > 1e-9 and PASSED on 4.6e-08,
    # which is float32 QR-retraction jitter, not a scar. The claim is that
    # identity is a deformation of the operator I = U V^T, so that is the bar.)
    p5_ok = float(np.linalg.norm(esic.U @ esic.V.T)) > 1e-6
    record(
        "P5", "Enhanced SIC: an admitted scar actually deforms the identity operator U@V.T",
        "||U@V.T|| > 1e-6 after 50 admitted scars",
        f"scars_admitted={esic.scars_admitted}/50 (rejected {50 - sum(admitted)}), "
        f"||dU||={dU:.3e}, ||dV||={dV:.3e}, ||U@V.T||={np.linalg.norm(esic.U @ esic.V.T):.3e}",
        p5_ok,
    )
except ImportError as e:
    record("P5", "Enhanced SIC present", "importable", f"ImportError: {e}", False)

# ---------------------------------------------------------------- P6
# Claim under test: the 10-step Phase-2 loop in core/engine.py runs.
try:
    import core.engine  # noqa: F401
    record("P6", "core/engine.py (10-step governance loop) imports",
           "imports clean", "imported", True)
except Exception as e:
    record("P6", "core/engine.py (10-step governance loop) imports",
           "imports clean", f"{type(e).__name__}: {e}", False)

# ---------------------------------------------------------------- P7
# Does the 10-step loop reach step 10?
try:
    from core.engine import Engine
    from run_engine import MockGGUF
    eng = Engine(d=64, rank=8, seed=42, monitor=FixedTemp(BENCH_TEMP))
    eng.set_gguf_engine(MockGGUF(logit_variance=0.90))
    statuses = [eng.step()["status"] for _ in range(20)]
    n_success = statuses.count("success")
    record(
        "P7", "10-step loop completes a cycle end to end",
        "at least 1 of 20 cycles reaches step 10",
        f"{n_success}/20 reached step 10; scars={eng.sic.scars_admitted}; "
        f"||U@V.T||={np.linalg.norm(eng.sic.U @ eng.sic.V.T):.6e}; "
        f"other statuses={ {s: statuses.count(s) for s in set(statuses) if s != 'success'} }",
        n_success > 0,
    )

    # ------------------------------------------------------------ P8
    # ANTI-VACUITY. A gate that passes everything is not a gate. Drive the
    # Fisher input across the registered 0.85 threshold in both directions.
    def run_at(lv):
        e = Engine(d=64, rank=8, seed=42, monitor=FixedTemp(BENCH_TEMP))
        e.set_gguf_engine(MockGGUF(logit_variance=lv))
        return [e.step()["status"] for _ in range(10)]

    hi = run_at(0.86)
    lo = run_at(0.84)
    p8_ok = (hi.count("success") == 10) and (lo.count("rejected_by_commit_gate") == 10)
    record(
        "P8", "Commit Gate discriminates at its registered Fisher threshold (0.85)",
        "logvar 0.86 -> 10/10 pass; logvar 0.84 -> 10/10 reject",
        f"0.86 -> {hi.count('success')}/10 pass; "
        f"0.84 -> {lo.count('rejected_by_commit_gate')}/10 reject",
        p8_ok,
    )
except Exception as e:
    record("P7", "10-step loop completes a cycle end to end",
           "at least 1 of 20 cycles reaches step 10", f"{type(e).__name__}: {e}", False)

# ---------------------------------------------------------------- summary
print("\n" + "=" * 68)
blocked = [r for r in RESULTS if r[5]]
fails = [r for r in RESULTS if not r[4] and not r[5]]
for pid, claim, _, _, ok, blk in RESULTS:
    v = "BLOCKED " if blk else ("MATCH   " if ok else "MISMATCH")
    print(f"  {pid:<4} {v}  {claim}")
tested = len(RESULTS) - len(blocked)
print(f"\n{tested - len(fails)}/{tested} predictions matched"
      + (f" ({len(blocked)} blocked, not tested)." if blocked else "."))
print("=" * 68)
sys.exit(1 if fails else 0)
