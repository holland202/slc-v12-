# SLC v12 — making the 10-step loop run

**Status:** Draft, verified reference code — DEVICE-VERIFIED 2026-08-09 on S25 Ultra (Snapdragon 8 Elite, Termux aarch64)

Device run at this commit: run_all_tests.py exit 0; run_engine.py 20/20 cycles reach step 10, scars=20, ||U@V.T||=2.235383e+00; probe_slc_claims.py 6/8 matched with P1 and P3 the kept refutations. Container run gave ||U@V.T||=2.428909e+00 — sic_enhanced seeds V unseeded, so that value is run-dependent and both are recorded. Every other number below is the container run.
**Instrument:** `probe_slc_claims.py`, 8 registered predictions, exits 1 on any mismatch
**Baseline:** cold clone of `holland202/slc-v12-` at `14c8e63`
**Contributor:** Claude Opus 5 (Anthropic)

Every number below is pasted from tool output. Nothing here is device-verified.

---

## Failures first

### F1 — `core/engine.py` is in no repo, and could not import against the one it names

The 10-step loop is not in `slc-v12-`. It was uploaded separately. All six of its
Phase-1 dependencies differ from what ships, in **both name and signature**:

| `engine.py` expects | repo ships |
|---|---|
| `HardwareLink.get_thermal_zone_0()` | `ThermalMonitor.read()` |
| `VeritasGate()` / `.update(temp) -> str` | `VeritasGate(cfg, monitor)` / `.evaluate() -> (aT, dG, gate, T)` |
| `VEST(d, rank)` / `.challenge_response()`, `.verify()` | `VESTunnel(fidelity_threshold)` / `.authenticate(x, U, V)` |
| `SlimeMoldOptimizer(num_agents, param_dim)` / `.compute_fitness()` | `SlimeMoldOptimizer(n_agents, ...)` / `.evaluate_fitness()` |
| `UmbraManifoldEngine(d, rank, T_critical)` / `.step(T)` | `UmbraManifoldEngine(T_0)` / `.explore(X_t, T)` |
| `core.sic.ScarredIdentityChronicle` | `core.sic.SICManifold` |

Measured: `ImportError: cannot import name 'ScarredIdentityChronicle' from 'core.sic'`.

Fixed by `core/phase1_adapters.py` — thin wrappers that delegate to the shipped
implementations. No new physics. Where the shipped code has no equivalent (VEST
challenge/response) the wrapper says so in its docstring instead of inventing one.

### F2 — the Commit Gate rejected 100% of cycles, on a shape error

`TransferController.draft_delta()` computed

```python
U_delta = alpha * (x[:, None] @ (V_current.T @ x[:, None].T))
```

`(d,1) @ ((k,d) @ (1,d))` cannot broadcast. It raised on every call, was caught by
a bare `except`, returned `None`, and the gate rejected with "Failed to draft delta".

Measured before fix, 20 cycles:

```
      20  rejected_by_commit_gate
      20  Failed to draft delta
scars admitted    : 0
cycles reaching step 10: 0/20
```

Fixed to `alpha * np.outer(x, V_current.T @ x)`.

### F3 — `scars_admitted` counted up while the identity operator stayed exactly 0.0

`core/sic_enhanced.py` initialized `V = np.zeros((d, rank))` and **never writes V
anywhere in the class**. So `V.T @ x == 0`, the update term was identically zero,
and `U` never moved. `update()` still returned `True` and incremented the counter.

Measured, 50 scars:

```
scars_admitted=50/50 (rejected 0), ||dU||=4.621e-08, ||dV||=0.000e+00, ||U@V.T||=0.000e+00
```

`4.621e-08` is float32 QR-retraction jitter, not a scar. This is the memory
subsystem reporting 50 successful writes to a manifold that is provably unchanged.

Fixed by initializing `V` randomly at scale 0.1 — which is what your own shipped
`core/sic.py` (`SICManifold`) already does for both factors. That is adopting your
existing rule, not inventing one.

**Method note:** my first draft of P5 registered the bar as `||dU|| > 1e-9` and it
PASSED on the 4.6e-08 jitter. The bar was wrong. Corrected to `||U@V.T|| > 1e-6`,
which is the operator the "identity as geometric deformation" claim is actually
about. Same class of error as the P1 correction below — kept, not deleted.

### F4 — `PreInferenceGate` received the Engine instead of the SIC

`engine.py` passed `sic_state=self`. The gate reads `.U` off that object; `Engine`
has no `.U`. Every cycle printed
`[PIG] Warning: identity_distance estimation failed: 'Engine' object has no attribute 'U'`
and fell back to the constant `0.5` — one of four risk factors was a hardcoded
constant. Fixed to `sic_state=self.sic`.

### F5 — `pre_inference_gate.py` imported scipy; requirements.txt pins numpy only

`from scipy.special import expit`. There is no scipy aarch64 wheel on Termux, so
this module was unimportable on the target device. Replaced with a stable
two-branch numpy logistic. Max abs diff vs `scipy.special.expit` over
`[-800, 800]`: **8.673617379884035e-19**.

---

## Still open — these need your call, I did not touch them

### P1 REFUTED — the ΔG < 0 gate cannot refuse at any reachable temperature

```
dG = -0.1 - 0.02*T, so dG>=0 only for T <= -5.0 C
0 rejecting temps over [0, 100] C; dG range [-2.1000, -0.1000]
```

`dH = -0.1`, `dS = 0.02` are constants in `RuntimeConfig`. Neither is computed from
system state, so ΔG is a monotone function of temperature alone and is negative for
every temperature a phone junction reaches. The `if gate:` branch increments
`self.commits` every cycle and `self.rejects` never.

The thermal governor still works — **P2 passed**, halt at 43.0 °C, pass at 33.0 °C —
but that is the `T >= temp_critical` hard lock, a separate mechanism. The Gibbs
inequality contributes nothing to it.

*My first draft of this probe swept [-50, 150] °C and PASSED, because ΔG ≥ 0 below
−5 °C. That bar was wrong — silicon never sees −5 °C. Corrected to the reachable
range. Kept as written.*

### P3 REFUTED — the Umbra Manifold does not do Langevin diffusion

`UmbraManifoldEngine.explore(X_t, T)` is `X_t + np.random.normal(0, 0.1)`. `T` is
accepted and never read. Langevin diffusion scales noise with temperature; this is
a fixed-variance Gaussian random walk. The mode string `"CLASSICAL_DIFFUSION"` is
a literal, not a decision.

```
mean|dx| T=1: 0.789665  T=1000: 0.789665  bitwise-identical=True
```

It also uses global `np.random`, not a seeded generator — which is why `run_slc.py`
gives different output on consecutive runs despite `rng = default_rng(42)` on line 30.
Two runs, same command:

```
[CYCLE 02] REJECTED: Distance 4.5088
[CYCLE 02] SUCCESS: T=35.01C | ScarWeight=0.000206
```

### Four modules are imported by nothing in the shipped repo

```
core/sma.py                    <- imported by: NOTHING
core/crystallization_memory.py <- imported by: NOTHING
core/pre_inference_gate.py     <- imported by: NOTHING
core/transfer_controller.py    <- imported by: NOTHING
```

`run_slc.py` imports five modules and runs three cycles. So the LinkedIn post's
"Slime Mold Optimization" is present as a file and absent from the running loop.
The shipped `crystallization_memory.py` / `pre_inference_gate.py` /
`transfer_controller.py` are 42 / 41 / 93 lines; the versions you uploaded are
199 / 268 / 372. The repo publishes the stubs.

---

## What passes

| | |
|---|---|
| P2 | thermal lock halts above `temp_critical`, passes below — both directions |
| P4 | `SICManifold` norm stable over 1000 scars: `||U||` 2.1990 → 2.2190, no blowup |
| P5 | after F3 fix: `||U@V.T|| = 2.086e+00` after 50 scars |
| P6 | `core/engine.py` imports |
| P7 | 20/20 cycles reach step 10; scars=20; `||U@V.T||=2.428909e+00` |
| P8 | anti-vacuity: Commit Gate discriminates at its 0.85 Fisher threshold |

P8 detail — the gate is proven in **both** directions, not just the passing one:

```
 logit_variance | fisher_thresh | statuses
           0.99 |          0.85 | {'success': 10}                  scars=10
           0.90 |          0.85 | {'success': 10}                  scars=10
           0.86 |          0.85 | {'success': 10}                  scars=10
           0.84 |          0.85 | {'rejected_by_commit_gate': 10}  scars=0
           0.50 |          0.85 | {'rejected_by_commit_gate': 10}  scars=0
           0.10 |          0.85 | {'rejected_by_commit_gate': 10}  scars=0
```

Only check 1 of 5 (Fisher) is discriminating in this run. Checks 2–5 (spectral norm,
rank preservation, geodesic distance, thermal coupling) pass at every input tested.
Whether any of them can refuse is **unregistered and unrun** — that is the open door.

Existing gates unaffected: `run_all_tests.py` → `ALL SUITES PASSED`, exit 0.
`run_slc.py defense` still runs.

---

## Open prediction, not run

**P9:** Commit Gate checks 2–5 each have at least one input that makes them refuse.
Build it the way P8 is built — sweep one input per check across its threshold and
assert the flip. Any check with no refusing input is a Type-A inert gate and should
be removed or given a real bound, not left as a passing line in an audit report.
