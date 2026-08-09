# SLC v12 — making the 10-step loop run

**Status:** Draft, verified reference code — DEVICE-VERIFIED 2026-08-09 on S25 Ultra (Snapdragon 8 Elite, Termux aarch64)

Device-verified on the S25 Ultra: run_all_tests.py exit 0; run_engine.py 20/20 cycles reach step 10, scars=20; probe_slc_claims.py 6/8 matched, with P1 and P3 the kept refutations.

**F6 (found after the first push, fixed in the follow-up):** `||U@V.T||` was different on every single run — 2.428909e+00 in the container, then 2.235383e+00, 2.255300e+00 and 2.247910e+00 on three device runs. Seeding the SIC did **not** fix it. Root cause was `_text_to_manifold_vector`, in both `core/engine.py` and `core/transfer_controller.py`:

```python
seed = hash(text) % (2**31)
```

Python randomizes `str.__hash__` per process (PYTHONHASHSEED), so every launch produced different embeddings from the same prompt. Replaced with a SHA-256 derived seed. Now **2.247657e+00** across seven consecutive runs including two with `PYTHONHASHSEED` forced random. The SIC also takes an explicit `seed` (default `None` preserves the old behaviour); `run_engine.py` and the probes pass 42.

This is the defect that would have broken the reproducibility claim for the first person to clone the repo. Every other number below is from the container run.
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

## FIXED — both refutations closed

### P1 — the Gibbs gate now enforces

`dH` and `dS` were fixed config constants (`-0.1`, `0.02`), so
`dG = -0.1 - 0.02*T` was negative at every reachable temperature: **0 rejecting
temperatures over 0-100 C**. Now `VeritasGate.evaluate_transition(dH, dS)` takes
deltas measured from the actual operator change, per the SLC v12 spec's
`E_t = ||U_t||_F^2 + ||V_t||_F^2` and spectral entropy `S_t`. Step 7 snapshots
the operator, applies the proposed scar, measures the real dH/dS, and rolls back
if `dG >= 0`.

Anti-vacuity — driven across the admission boundary, both outcomes occur:

```
   alpha | admitted |      mean dG | outcome
   0.001 |       10 |      -0.0001 | ADMIT
    0.01 |       10 |      -0.0014 | ADMIT
     0.3 |       10 |      -0.0322 | ADMIT
     1.0 |        9 |      -0.0501 | ADMIT
     3.0 |        1 |      +0.2963 | REFUSE
```

**P1b, kept:** `evaluate()`'s constant path is deliberately unchanged so
`run_slc.py` and the thermal suites keep their behaviour. That path still cannot
refuse, and `run_slc.py` still uses it. Registered as a standing mismatch rather
than quietly migrated.

### P3 — the Umbra Manifold now does Langevin diffusion

Was `X_t + np.random.normal(0, 0.1)` with `T` accepted and never read. Now
Euler-Maruyama on an Ornstein-Uhlenbeck process,
`X_{t+1} = X_t - eta*X_t*dt + sqrt(2*lambda(T)*dt)*xi`, with
`lambda(T) = lambda_0 * clip((T_c - T)/(T_c - T_0), 0, 1)` — the spec's thermal
governor, `lambda -> 0` as `T -> T_c`. `lambda_0 = 0.5` so `sqrt(2*lambda_0*dt) = 0.1`
at `dt=0.01`, preserving the old step scale at full headroom.

```
   T (C)     lambda     mean |dx|      mode
    30.0     0.5000     0.808874     LANGEVIN_EXPLORE
    36.5     0.5000     0.808874     LANGEVIN_EXPLORE
    37.2     0.2500     0.583786     LANGEVIN_THROTTLED
    38.0     0.0000     0.160000     THERMAL_COLLAPSE
```

At `T_critical` the stochastic term is exactly zero and only the deterministic
drift remains — the spec's collapse onto the fixed point.

Probes now report **10 registered predictions**. Container: 9/10. On the S25 Ultra 2026-08-09: **8/10**, with P0 and P1b the two mismatches.

### P0 — this device cannot run the defense profile

Registered as a SUBSTRATE claim, not a code claim, and it fails:

```
PREDICTED : live compute temperature below 36.5 C
MEASURED  : live max compute zone = 66.90 C vs threshold 36.5 C
```

68 thermal zones dumped on device: max 50.0 C, median 38.4 C at rest. The two hottest are modem radios (mmw_ific0, sdr0) which never matched the cpu/cpuss/gpuss keywords. Three report -273.0 C — disabled sensors at absolute zero, harmless under a max but fatal to any mean. The binding constraint is that cpu-0-1-1 idles at 44.8 C and passes 65 C under numpy load, while defense temp_threshold is 36.5 C. run_slc.py printed HALT at 65.30 / 66.10 / 67.20 C on three consecutive cycles. The governor was right; the profile is not reachable on this silicon.

### Instrument defect this exposed

P1, P7 and P8 read the live thermometer, so a warm phone reported them as MISMATCH — a weather report, not an experiment, and precisely how a thermal event gets mistaken for broken code. Engine now takes an injectable monitor; probes and demo run governance acts at 30.0 C. BLOCKED added as a third verdict, distinct from MISMATCH: instrument could not execute is not the same result as prediction was wrong.

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


---

## F7 — the Commit Gate confidence threshold has zero discriminative power on a real model

First calibration against an actual GGUF. TinyLlama-1.1B-Q4_K_M via llama-server on the S25 Ultra, 8 prompts x 2 arms (temperature 0.0 and 1.2), 57-66 tok/s eval.

```
paired by prompt: 6 wins, 1 tie, 1 loss
  win   conf 0.4627  unc 0.2343  The capital of France is
  win   conf 0.6630  unc 0.3311  2 + 2 =
  win   conf 0.4966  unc 0.2667  Water boils at 100 degrees
  win   conf 0.6014  unc 0.4391  The first three prime numbers are
  tie   conf 0.9093  unc 0.9093  Complete the sequence: 1, 2, 3,
  LOSS  conf 0.4849  unc 0.6683  A triangle has how many sides?
  win   conf 0.5507  unc 0.2298  The opposite of hot is
  win   conf 0.5922  unc 0.2410  Name a primary color:

sign test (one-sided): p = 0.0625 over 7 non-tied pairs — NOT significant
current 0.850: 1/8 confident commit, 1/8 uncertain commit, J = +0.000
best    0.440: 8/8 confident commit, 2/8 uncertain commit, J = +0.750
```

The 0.85 threshold was chosen against MockGGUF, which returned a constant. Against a real model it admits good and bad generations at the same rate. A gate that refuses almost everything is as inert as one that refuses nothing — the mirror image of the ΔG defect, reached from the opposite direction.

**Threshold NOT changed.** p = 0.0625 at n = 7 is suggestive, not significant, and eight prompts is too few to set a production value. Registered as measured; the fix is more prompts, not a new number.

The tie and the loss both explain themselves. The sequence prompt produced byte-identical output at both temperatures — deterministic enough that sampling changed nothing. The triangle inversion was degenerate repetition, 'A. B. C. D. E.', which is highly predictable once started: the metric correctly scored a confident model producing garbage. **Confidence is not correctness.** The other four Commit Gate checks are what bound that damage.

**Second instrument defect of the day.** The calibrator originally compared min(confident) to max(uncertain) and reported 'the arms OVERLAP, no threshold works' — a worst-case bound over 8 samples that one tie and one inversion destroyed, while the paired view showed clear separation. Rewritten to paired comparison, sign test, and a Youden-J sweep, and it now distinguishes 'no threshold separates' from 'the arms separate but yours is inert'. Same family as the live-thermometer probes: the instrument was wrong, not the system.
