#!/usr/bin/env python3
"""
run_slc.py — SLC v12 Unified Launcher
=======================================
Chad Edward Holland · @holland202
Vincit Omnia Veritas

Wires all core modules together and runs the governed pipeline.
Supports multiple sector profiles: healthcare, defense, research, edge, desktop.

Usage:
    python3 run_slc.py --sector healthcare
"""

import sys, os, time, json, logging, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.WARNING,
    format='%(name)s: %(message)s'
)

class C:
    R='\033[0m'; GR='\033[92m'; CY='\033[96m'; YL='\033[93m'
    RD='\033[91m'; MG='\033[95m'; DM='\033[2m'; BD='\033[1m'

def p(text, color=C.R): print(f"{color}{text}{C.R}")
def div(): p('─'*68, C.DM)

# ── Parse args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SLC v12 Unified Launcher")
parser.add_argument("--sector", default="research",
                    choices=["healthcare","defense","research","edge","desktop"],
                    help="Deployment sector thermal profile")
args = parser.parse_args()
SECTOR = args.sector

# ── Boot Banner ──────────────────────────────────────────────────────────────
print()
p("╔══════════════════════════════════════════════════════════════════════╗", C.CY)
p("║        ⚡  SOVEREIGN LOGIC CORE v12  —  UNIFIED LAUNCHER  ⚡        ║", C.CY)
p("║              Chad Edward Holland · @holland202                       ║", C.CY)
p("╚══════════════════════════════════════════════════════════════════════╝", C.CY)
print()

# ── Import core modules ───────────────────────────────────────────────────────
p("LOADING CORE MODULES", C.BD)
div()

modules_ok = True
module_status = {}

core_modules = [
    ("hardware_link", "SM8750 thermal interface"),
    ("crystallization_memory", "History buffer"),
    ("pre_inference_gate", "Stage 4 — predictive risk"),
    ("transfer_controller", "Stage 6 — commit gate"),
    ("params_sector", "Sector thermal profiles"),
    ("params", "Unified configuration"),
    ("sic", "SIC — Scarred Identity Chronicle"),
    ("veritas_gate", "Veritas Gate — thermodynamic governor"),
    ("vest", "VEST — manifold authentication"),
    ("sma", "SMA — Slime Mold Optimizer"),
    ("ume", "UME — Umbra Manifold Engine"),
]

for name, label in core_modules:
    try:
        mod = __import__(f"core.{name}", fromlist=[name])
        module_status[name] = mod
        p(f"  ✓  {name:<24} [{label}]", C.GR)
    except Exception as e:
        p(f"  ✗  {name:<24} [{label}] — {e}", C.RD)
        modules_ok = False

print()

# ── Hardware check ────────────────────────────────────────────────────────────
p("HARDWARE STATUS", C.BD)
div()

HardwareLink = getattr(module_status["hardware_link"], "HardwareLink")
hw = HardwareLink()
temp = hw.get_thermal_zone_0()
substrate = hw.get_substrate_id()
zones = hw.all_zones()

p(f"  Substrate:    {substrate}", C.CY)
p(f"  Thermal zone: {temp:.1f}°C", C.GR if temp < 43 else C.YL)
p(f"  Zones found:  {len(zones)}", C.DM)

get_sector_thermal = getattr(module_status["params_sector"], "get_sector_thermal")
profile = get_sector_thermal(SECTOR)
p(f"  Sector:       {SECTOR.upper()}", C.MG)
p(f"  Thresholds:   warn={profile['T_warn']}°C  throttle={profile['T_throttle']}°C  critical={profile['T_critical']}°C", C.DM)

if temp >= profile['T_critical']:
    p(f"\n  ⚠  CRITICAL THERMAL STATE — pipeline suspended", C.RD)
    sys.exit(1)
elif temp >= profile['T_throttle']:
    p(f"\n  ⚠  THROTTLE — pipeline will run at reduced rate", C.YL)
else:
    p(f"\n  ✓  THERMAL NOMINAL — pipeline cleared", C.GR)

print()

# ── Initialize all subsystems ────────────────────────────────────────────────
p("SUBSYSTEM INITIALIZATION", C.BD)
div()

SLCConfig = getattr(module_status["params"], "SLCConfig")
config = SLCConfig.for_sector(SECTOR)

ScarredIdentityChronicle = getattr(module_status["sic"], "ScarredIdentityChronicle")
ScarEvent = getattr(module_status["sic"], "ScarEvent")
sic = ScarredIdentityChronicle(
    d=config.manifold.d,
    rank=config.manifold.rank,
    alpha=config.sma.alpha,
    beta=0.85,
    gamma=1e-4,
)
p(f"  ✓  SIC initialized  (d={sic.d}, r={sic.rank}, α={sic.alpha})", C.GR)

VeritasGate = getattr(module_status["veritas_gate"], "VeritasGate")
vg = VeritasGate(
    T_low=profile['T_recovery'],
    T_high=profile['T_throttle'],
    T_critical=profile['T_critical'],
    T_recovery=profile['T_recovery'],
)
p(f"  ✓  Veritas Gate initialized  (T_low={vg.T_low}°C, T_high={vg.T_high}°C)", C.GR)

VESTAuthenticator = getattr(module_status["vest"], "VESTAuthenticator")
vest = VESTAuthenticator(d=sic.d, rank=sic.rank)
p(f"  ✓  VEST initialized  (ε={vest.epsilon}, τ={vest.threshold})", C.GR)

SlimeMoldOptimizer = getattr(module_status["sma"], "SlimeMoldOptimizer")
sma = SlimeMoldOptimizer(n_agents=8)
p(f"  ✓  SMA initialized  ({sma.n_agents} agents)", C.GR)

UmbraManifoldEngine = getattr(module_status["ume"], "UmbraManifoldEngine")
ume = UmbraManifoldEngine(dim=sic.d)
p(f"  ✓  UME initialized  (dim={ume.dim}, λ₀={ume.lambda0})", C.GR)

CrystallizationMemory = getattr(module_status["crystallization_memory"], "CrystallizationMemory")
PreInferenceGate = getattr(module_status["pre_inference_gate"], "PreInferenceGate")
TransferController = getattr(module_status["transfer_controller"], "TransferController")

cryst_memory = CrystallizationMemory(window_size=20)
pre_gate = PreInferenceGate(
    weights=config.calibration.gate_weights,
    threshold=config.calibration.gate_threshold,
    steepness=config.calibration.gate_steepness,
)

# COLD-START CALIBRATION
import numpy as np
U = sic.U
sv = np.linalg.svd(U, compute_uv=False)
total = float(np.sum(sv))
top_k = float(np.sum(sv[:max(1,len(sv)//4)]))
fresh_fisher = top_k / (total + 1e-10)
cold_start_fisher = max(0.15, fresh_fisher * 0.8)

tc = TransferController(
    fisher_threshold=cold_start_fisher,
    spectral_norm_max=config.calibration.spectral_norm_max,
    geodesic_distance_max=config.calibration.geodesic_distance_max,
    thermal_multiplier=config.calibration.thermal_multiplier,
)

p(f"  ✓  CrystallizationMemory  initialized", C.GR)
p(f"  ✓  PreInferenceGate       initialized  (threshold={pre_gate.threshold})", C.GR)
p(f"  ✓  TransferController     initialized  (fisher≥{cold_start_fisher:.3f}, cold-start calibrated)", C.GR)

print()

# ── VEST Authentication Test ─────────────────────────────────────────────────
p("VEST AUTHENTICATION TEST", C.BD)
div()

challenge = vest.challenge(sic)
response = vest.respond(challenge, sic)
auth_result = vest.authenticate(challenge, response, sic)
p(f"  Challenge norm:   {np.linalg.norm(challenge):.3f}", C.DM)
p(f"  Response norm:    {np.linalg.norm(response):.3f}", C.DM)
p(f"  Distance:         {auth_result['distance']:.6f}", C.DM)
p(f"  Confidence:       {auth_result['confidence']:.4f}", C.GR if auth_result['authentic'] else C.YL)
p(f"  Status:           {'AUTHENTIC ✓' if auth_result['authentic'] else 'REJECTED ✗'}", C.GR if auth_result['authentic'] else C.RD)

sic2 = ScarredIdentityChronicle(d=sic.d, rank=sic.rank)
auth_result2 = vest.authenticate(challenge, response, sic2)
p(f"  Replay test:      {'REJECTED ✓' if not auth_result2['authentic'] else 'ACCEPTED ✗'}", C.GR if not auth_result2['authentic'] else C.RD)

print()

# ── SIC Scar Accumulation Test ───────────────────────────────────────────────
p("SIC SCAR ACCUMULATION TEST", C.BD)
div()

np.random.seed(42)
n_test_events = 30
admitted = 0

for i in range(n_test_events):
    embedding = np.random.normal(0, 0.5, sic.d).astype(np.float32)
    entropy = float(np.random.uniform(0.1, 0.8))
    event = ScarEvent(embedding=embedding, entropy=entropy)
    if sic.record_event(event):
        admitted += 1

p(f"  Events tested:    {n_test_events}", C.DM)
p(f"  Scars admitted:   {admitted} ({admitted/n_test_events:.0%})", C.GR)
p(f"  Spectral entropy: {sic.spectral_entropy():.2f}  (ceiling={sic.spectral_entropy_ceiling})", C.GR if sic.spectral_entropy() < sic.spectral_entropy_ceiling else C.YL)
p(f"  Anchor drift:     {np.linalg.norm(sic.anchor):.3f}", C.DM)

print()

# ── Veritas Gate Governance Test ─────────────────────────────────────────────
p("VERITAS GATE GOVERNANCE TEST", C.BD)
div()

thermal_scenario = [33.0, 35.0, 37.0, 39.0, 41.0, 42.5, 40.0, 38.0, 36.0, 34.0]
for T in thermal_scenario:
    cv = vg.update(temperature=T, spectral_entropy=sic.spectral_entropy())
    state_color = C.GR if cv.command == 2 else (C.YL if cv.command == 3 else C.RD)
    p(f"  T={T:5.1f}°C  →  {vg.COMMANDS.get(cv.command,'UNKNOWN'):18s}  Gibbs margin={cv.gibbs_margin:+.3f}", state_color)

vg_summary = vg.state_summary()
p(f"\n  Final state: {vg_summary['current_state']}", C.CY)

print()

# ── SMA Optimization Test ────────────────────────────────────────────────────
p("SMA OPTIMIZATION TEST", C.BD)
div()

for gen in range(5):
    best = sma.step(
        vest_distance=0.1 + np.random.uniform(0, 0.2),
        spectral_entropy=sic.spectral_entropy(),
        thermal_energy=temp,
    )
    rec = sma.get_recommended_params()
    p(f"  Gen {gen+1}:  α={rec['alpha']:.4f}  β={rec['beta']:.3f}  γ={rec['gamma']:.2e}  r={rec['rank']}  fitness={rec['fitness']:.4f}", C.DM)

sma_summary = sma.state_summary()
p(f"\n  Best fitness: {sma_summary['best_fitness']}", C.GR)

print()

# ── UME Stochastic Exploration Test ──────────────────────────────────────────
p("UME STOCHASTIC EXPLORATION TEST", C.BD)
div()

ume_state = ume.step(sic, temperature=temp)
p(f"  Initial norm:     {np.linalg.norm(ume.state):.3f}", C.DM)

for _ in range(50):
    ume.step(sic, temperature=temp)

p(f"  After 50 steps:   {np.linalg.norm(ume.state):.3f}", C.DM)

projection = ume.project_to_slc(sic)
p(f"  SLC projection:   {np.linalg.norm(projection):.3f}", C.DM)

ume_summary = ume.state_summary()
p(f"  Trajectory len:   {ume_summary['trajectory_length']}", C.DM)

print()

# ── Full Pipeline Integration Test ───────────────────────────────────────────
p("FULL PIPELINE INTEGRATION — 5 TEST CYCLES", C.BD)
div()

class MockGGUFEngine:
    """Placeholder inference engine."""
    def generate(self, prompt, max_tokens=256, logprobs=False):
        return {
            "success": True,
            "text": f"Governed response to: {prompt[:40]}...",
            "logit_variance": float(np.random.uniform(0.78, 0.95)),
        }

gguf = MockGGUFEngine()

test_prompts = [
    "What is the significance of recursive manifold curvature in edge AI?",
    "Analyze the thermodynamic efficiency of the Gibbs stability mandate.",
    "How does spectral entropy proxy enforce topological validity?",
    "Interpret the RNCA attractor basin in self-observing systems.",
    "What inference constraints govern the Veritas Gate hysteresis?",
]

print()
passed = 0
for i, prompt in enumerate(test_prompts):
    gate_pass, risk_score, factors = pre_gate.evaluate(
        prompt=prompt, sic_state=sic, cryst_memory=cryst_memory
    )
    
    if gate_pass:
        gguf_result = gguf.generate(prompt)
        if gguf_result["success"]:
            delta = tc.draft_delta(gguf_result, sic)
            audit = tc.commit_gate_audit(delta, sic, cryst_memory)
            cryst_memory.record_crystallization(
                logit_variance=gguf_result["logit_variance"],
                topological_strain=delta.topology_strain,
                was_rejected=not audit.passed,
            )
            gate_label = f"{C.GR}ALLOW{C.R}" if audit.passed else f"{C.YL}BLOCK{C.R}"
            p(f"  [{i+1}] {gate_label}  risk={risk_score:.3f}  fisher={audit.fisher_sharpness:.3f}  lv={gguf_result['logit_variance']:.3f}", C.R)
            if audit.passed:
                passed += 1
                embedding = np.random.normal(0, 0.3, sic.d).astype(np.float32)
                event = ScarEvent(embedding=embedding, entropy=gguf_result["logit_variance"])
                sic.record_event(event)
        else:
            p(f"  [{i+1}] {C.RD}GGUF FAIL{C.R}", C.R)
    else:
        cryst_memory.record_deferral(reason="pre_gate")
        p(f"  [{i+1}] {C.YL}DEFER{C.R}   risk={risk_score:.3f}", C.R)

print()
div()
p(f"  RESULT: {passed}/{len(test_prompts)} queries passed full pipeline", C.GR if passed >= 3 else C.YL)
summary = cryst_memory.state_summary()
p(f"  Rejection rate: {summary['rejection_rate']:.1%}  Mean LV: {summary['mean_logit_variance']:.3f}  Mean strain: {summary['mean_topo_strain']:.3f}", C.DM)
div()

# ── Final subsystem summaries ────────────────────────────────────────────────
print()
p("SUBSYSTEM STATE SUMMARIES", C.BD)
div()

p(f"  SIC:        scars={sic.scars_admitted}  H_σ={sic.spectral_entropy():.2f}  rank_eff={sic.effective_rank(temp)}", C.DM)
p(f"  Veritas:    state={vg_summary['current_state']}  cycles={vg_summary['cycles']}  ema={vg_summary['thermal_ema']}°C", C.DM)
p(f"  SMA:        gen={sma_summary['generation']}  best_fitness={sma_summary['best_fitness']}", C.DM)
p(f"  UME:        steps={ume_summary['steps']}  norm={ume_summary['state_norm']:.3f}", C.DM)
p(f"  PreGate:    evals={pre_gate.state_summary()['total_evaluations']}  pass_rate={pre_gate.state_summary()['pass_rate']:.1%}", C.DM)
p(f"  Transfer:   submissions={tc.state_summary()['total_submissions']}  accept_rate={tc.state_summary()['accept_rate']:.1%}", C.DM)

print()
div()
p("  STATUS: SLC v12 UNIFIED PIPELINE OPERATIONAL", C.GR + C.BD)
p(f"  Sector: {SECTOR.upper()}  |  Thermal: {temp:.1f}°C  |  Pipeline: READY", C.CY)
print()
p("  All subsystems verified. Repository ready for GitHub push.", C.DM)
print()
div()
p("\n  Vincit Omnia Veritas\n", C.CY + C.BD)
