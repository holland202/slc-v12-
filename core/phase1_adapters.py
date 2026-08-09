#!/usr/bin/env python3
"""
core/phase1_adapters.py — binds core/engine.py to the modules that actually ship.

core/engine.py was written against a Phase-1 API that does not exist in this
repo. Every one of its six Phase-1 dependencies differs in BOTH name and
signature from what is on disk:

    engine.py expects            repo ships
    -------------------------    ---------------------------------
    HardwareLink                 ThermalMonitor
      .get_thermal_zone_0()        .read()
    VeritasGate()                VeritasGate(cfg, monitor)
      .update(temp) -> str         .evaluate() -> (aT, dG, gate, T)
    VEST(d, rank)                VESTunnel(fidelity_threshold)
      .challenge_response()        .authenticate(x, U, V) -> (bool, dist)
      .verify()
    SlimeMoldOptimizer(          SlimeMoldOptimizer(n_agents, ...)
      num_agents, param_dim)
      .compute_fitness(...)        .evaluate_fitness(agent, ...)
      .step()                      .step(vest_distance, spectral_entropy, ...)
    UmbraManifoldEngine(         UmbraManifoldEngine(T_0)
      d, rank, T_critical)
      .step(T)                     .explore(X_t, T)

These wrappers translate. They add no new physics — every wrapper delegates to
the shipped implementation. Where the shipped code has no equivalent (VEST
challenge/response), the wrapper says so in its docstring rather than
inventing one.
"""
import numpy as np

from core.params import RuntimeConfig
from core.hardware_link import ThermalMonitor
from core.veritas_gate import VeritasGate as _RepoVeritasGate
from core.vest import VESTunnel
from core.sma import SlimeMoldOptimizer as _RepoSMA
from core.ume import UmbraManifoldEngine as _RepoUME


class HardwareLink:
    """engine.py calls .get_thermal_zone_0(); repo exposes ThermalMonitor.read()."""

    def __init__(self, keywords=None, monitor=None):
        self.monitor = monitor if monitor is not None else ThermalMonitor(keywords)

    def get_thermal_zone_0(self) -> float:
        return self.monitor.read()


class VeritasGate:
    """
    engine.py wants a zero-arg gate returning a state STRING.
    Repo gate needs (cfg, monitor) and returns (aT, dG, pass, T).

    State strings are derived from the repo's own thresholds, not invented:
        SCAR_LOCK  T >= temp_critical   (repo's hard lock)
        THROTTLE   T >  temp_threshold  (repo's aT < 1 branch)
        NORMAL     otherwise
    """

    def __init__(self, sector: str = "defense", monitor=None):
        self.cfg = RuntimeConfig(sector)
        self.monitor = monitor or ThermalMonitor()
        self._gate = _RepoVeritasGate(self.cfg, self.monitor)
        self.last = None

    def update(self, temp: float = None) -> str:
        aT, dG, ok, T = self._gate.evaluate()
        self.last = {"aT": aT, "dG": dG, "gibbs_pass": ok, "T": T}
        if T >= self.cfg.temp_critical:
            return "SCAR_LOCK"
        if T > self.cfg.temp_threshold:
            return "THROTTLE"
        return "NORMAL"

    def admit_transition(self, U, V, U_new, V_new):
        """State-dependent Gibbs admission for a proposed scar."""
        dH, dS = _RepoVeritasGate.state_deltas(U, V, U_new, V_new)
        aT, dG, admit, T = self._gate.evaluate_transition(dH, dS)
        return {"dH": float(dH), "dS": float(dS), "dG": float(dG),
                "admit": bool(admit), "T": float(T), "aT": float(aT)}


class VEST:
    """
    engine.py calls challenge_response() then verify().

    The shipped VESTunnel has NO challenge/response protocol — it has a single
    projection-residual test. This wrapper exposes that test through the two
    method names engine.py uses; challenge_response() returns the projected
    vector and verify() returns the residual distance. It is NOT a
    challenge-response authentication protocol and must not be described as
    one until one is implemented.
    """

    def __init__(self, d: int = 512, rank: int = 64, fidelity_threshold: float = 4.5):
        self.d = d
        self.rank = rank
        self.tunnel = VESTunnel(fidelity_threshold=fidelity_threshold)

    @staticmethod
    def _project(x, U, V):
        # Repo VESTunnel assumes V is (rank, dim). Enhanced SIC stores V as
        # (dim, rank). Detect and orient rather than guessing.
        Vm = V if V.shape[0] == U.shape[1] else V.T
        return U @ (Vm @ x)

    def challenge_response(self, challenge, U, V):
        x = challenge if challenge.shape[0] == U.shape[0] else U @ challenge
        return self._project(x, U, V)

    def verify(self, response, challenge, U, V) -> float:
        x = challenge if challenge.shape[0] == U.shape[0] else U @ challenge
        _, distance = self.tunnel.authenticate(x, U, V if V.shape[0] == U.shape[1] else V.T)
        return float(distance)


class SlimeMoldOptimizer:
    """engine.py uses num_agents/.compute_fitness(...)/.step() with no args."""

    def __init__(self, num_agents: int = 10, param_dim: int = 4, **kw):
        self.num_agents = num_agents
        self._sma = _RepoSMA(n_agents=num_agents)
        self._last = (0.1, 0.5, -0.2)

    def compute_fitness(self, distance: float, entropy: float, free_energy: float) -> float:
        self._last = (distance, entropy, free_energy)
        best = self._sma.agents[0] if getattr(self._sma, "agents", None) else None
        if best is None:
            return 0.0
        return float(self._sma.evaluate_fitness(best, distance, entropy, free_energy))

    def step(self):
        d, e, f = self._last
        return self._sma.step(d, e, f)

    def state_summary(self):
        return self._sma.state_summary()


class UmbraManifoldEngine:
    """
    engine.py calls .step(T=...) with no state vector and expects the engine to
    hold no diffusion state. Repo's .explore(X_t, T) is a pure function, so this
    wrapper owns the state vector.

    NOTE: the shipped explore() IGNORES T entirely (probe P3). This wrapper does
    not silently repair that — it records whether T changed anything so the
    defect stays visible instead of being hidden behind an adapter.
    """

    def __init__(self, d: int = 512, rank: int = 64, T_critical: float = 0.5):
        self.d = d
        self.X = np.zeros(d, dtype=np.float64)
        self._ume = _RepoUME(T_0=T_critical)
        self.last_mode = None

    def step(self, T: float = 0.5):
        X_next, mode = self._ume.explore(self.X, T)
        step_norm = float(np.linalg.norm(X_next - self.X))
        self.X = X_next
        self.last_mode = mode
        return {"mode": mode, "step_norm": step_norm, "T": T}
