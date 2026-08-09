"""
Engine — Sovereign Logic Core v12.0 Orchestrator
=================================================
Phase 2 Integration: Governance-aware 10-step cycle with PreInferenceGate, 
TransferController, and CrystallizationMemory.

Key Invariant:
  LLM output NEVER writes directly to SIC. All writes go through:
    PreInferenceGate → GGUF → Commit Gate → SIC
  
  This is the one-way crystallization membrane.
"""

import hashlib
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from collections import deque

# Phase 2 imports (new)
from core.crystallization_memory import CrystallizationMemory
from core.pre_inference_gate import PreInferenceGate
from core.transfer_controller import TransferController

# Existing imports
from core.sic_enhanced import ScarredIdentityChronicle
from core.phase1_adapters import (
    VeritasGate,
    VEST,
    SlimeMoldOptimizer,
    UmbraManifoldEngine,
    HardwareLink,
)

logger = logging.getLogger("slc.engine")


class Engine:
    """
    Sovereign Logic Core Orchestrator (v12.0 + Phase 2).
    
    10-Step Cycle with Governance:
      1. [EXISTING] Thermal status check (Veritas Gate)
      2. [EXISTING] VEST authentication (identity verification)
      3. [EXISTING] SMA hyperparameter optimization
      4. [NEW] PreInferenceGate evaluation (predictive risk)
      5. [NEW] GGUF inference (with logprobs=1 for Fisher estimate)
      6. [NEW] Commit Gate audit (5 topological/thermodynamic checks)
      7. [EXISTING] Scar formation (SIC.update if Commit Gate passes)
      8. [EXISTING] UME Itô dynamics (stochastic perturbation)
      9. [EXISTING] SMA fitness update + slime mold flow
      10. [EXISTING] Telemetry logging + checkpoint save
    
    Attributes:
        sic: Scarred Identity Chronicle (core memory)
        veritas_gate: Schmitt trigger thermal governor
        vest: VEST authenticator
        sma: Slime Mold Optimizer
        ume: Umbra Manifold Engine (Langevin diffusion)
        hardware_link: Thermal telemetry + substrate info
        
        [NEW Phase 2]
        cryst_memory: Crystallization history buffer
        pre_gate: Predictive risk evaluator
        transfer_controller: One-way membrane + Commit Gate
        gguf_engine: External GGUF inference engine (injected)
    """

    def __init__(
        self,
        d: int = 512,
        rank: int = 64,
        gguf_engine: Optional[Any] = None,
        cryst_memory_window: int = 20,
        pre_gate_config: Optional[Dict[str, Any]] = None,
        transfer_controller_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the engine with all Phase 1 + Phase 2 components.
        
        Args:
            d: SIC dimension (default 512)
            rank: SIC rank (default 64)
            gguf_engine: External GGUF inference engine. Must have:
                - .generate(prompt, logprobs=True) → dict with "success", "text", "logit_variance"
            cryst_memory_window: History buffer size (default 20)
            pre_gate_config: Dict of PreInferenceGate hyperparams
            transfer_controller_config: Dict of TransferController hyperparams
        """
        
        # ===== Phase 1: Core Components =====
        self.sic = ScarredIdentityChronicle(d=d, rank=rank, seed=seed)
        self.veritas_gate = VeritasGate()
        self.vest = VEST(d=d, rank=rank)
        self.sma = SlimeMoldOptimizer(num_agents=10, param_dim=4)
        self.ume = UmbraManifoldEngine(d=d, rank=rank, T_critical=0.5)
        self.hardware_link = HardwareLink()
        
        # ===== Phase 2: Governance Components =====
        self.cryst_memory = CrystallizationMemory(window_size=cryst_memory_window)
        
        pre_gate_cfg = pre_gate_config or {
            "weights": (0.25, 0.30, 0.25, 0.20),
            "threshold": 0.65,
            "steepness": 5.0,
        }
        self.pre_gate = PreInferenceGate(**pre_gate_cfg)
        
        tc_cfg = transfer_controller_config or {
            "fisher_threshold": 0.85,
            "spectral_norm_max": 2.0,
            "geodesic_distance_max": 0.15,
            "thermal_multiplier": 1.0,
        }
        self.transfer_controller = TransferController(**tc_cfg)
        
        # External GGUF engine (injected at init or set later)
        self.gguf_engine = gguf_engine
        
        # ===== Cycle Tracking =====
        self.cycle_count = 0
        self.step_log = deque(maxlen=100)  # Recent 100 steps for diagnostics
        
        logger.info(
            f"Engine initialized: d={d}, rank={rank}, "
            f"cryst_window={cryst_memory_window}, gguf_engine={'ready' if gguf_engine else 'not set'}"
        )
    
    def set_gguf_engine(self, engine: Any) -> None:
        """
        Set (or update) the external GGUF inference engine.
        Can be called after __init__ if engine wasn't ready yet.
        """
        self.gguf_engine = engine
        logger.info("GGUF engine set")
    
    # =========================================================================
    # 10-STEP CYCLE (Phase 1 + Phase 2 Integrated)
    # =========================================================================
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one cycle of the 10-step governance loop.
        
        Returns:
            step_report: Dict with cycle_count, status, and detailed results from each step.
        """
        self.cycle_count += 1
        report = {
            "cycle_count": self.cycle_count,
            "status": "success",
            "steps": {},
        }
        
        try:
            # ===== STEP 1: Thermal Status (Veritas Gate) =====
            report["steps"]["01_thermal"] = self._step_01_thermal()
            
            # ===== STEP 2: VEST Authentication =====
            report["steps"]["02_vest"] = self._step_02_vest()
            
            # ===== STEP 3: SMA Optimization =====
            report["steps"]["03_sma"] = self._step_03_sma()
            
            # ===== STEP 4: PreInferenceGate (NEW Phase 2) =====
            pre_gate_pass, pre_gate_report = self._step_04_pre_inference_gate()
            report["steps"]["04_pre_gate"] = pre_gate_report
            
            if not pre_gate_pass:
                # PreInferenceGate rejected — skip inference
                # Record deferral in history
                self.cryst_memory.record_deferral(reason="pre_gate_rejection")
                report["status"] = "deferred_by_pre_gate"
                self.step_log.append(report)
                logger.debug(f"[Cycle {self.cycle_count}] PreInferenceGate rejected inference")
                return report
            
            # ===== STEP 5: GGUF Inference (NEW Phase 2) =====
            gguf_result, gguf_report = self._step_05_gguf_inference()
            report["steps"]["05_gguf"] = gguf_report
            
            if not gguf_result:
                # GGUF failed — skip to deferral
                self.cryst_memory.record_deferral(reason="gguf_failure")
                report["status"] = "deferred_by_gguf"
                self.step_log.append(report)
                logger.debug(f"[Cycle {self.cycle_count}] GGUF inference failed")
                return report
            
            # ===== STEP 6: Commit Gate Audit (NEW Phase 2) =====
            commit_gate_pass, commit_gate_report = self._step_06_commit_gate(gguf_result)
            report["steps"]["06_commit_gate"] = commit_gate_report
            
            if not commit_gate_pass:
                # Commit Gate rejected — record rejection in history
                self.cryst_memory.record_crystallization(
                    logit_variance=gguf_result.get("logit_variance", 0.7),
                    topological_strain=commit_gate_report.get("topology_strain", 0.0),
                    was_rejected=True,
                )
                report["status"] = "rejected_by_commit_gate"
                self.step_log.append(report)
                logger.debug(
                    f"[Cycle {self.cycle_count}] Commit Gate rejected: "
                    f"{commit_gate_report.get('rejection_reason', 'unknown')}"
                )
                return report
            
            # ===== STEP 7: Scar Formation (SIC Update) =====
            scar_success, scar_report = self._step_07_scar_formation(gguf_result)
            report["steps"]["07_scar"] = scar_report
            
            if scar_success:
                # Record successful crystallization
                self.cryst_memory.record_crystallization(
                    logit_variance=gguf_result.get("logit_variance", 0.7),
                    topological_strain=commit_gate_report.get("topology_strain", 0.0),
                    was_rejected=False,
                )
            else:
                # SIC.update() failed despite Commit Gate passing
                self.cryst_memory.record_crystallization(
                    logit_variance=gguf_result.get("logit_variance", 0.7),
                    topological_strain=commit_gate_report.get("topology_strain", 0.0),
                    was_rejected=True,
                )
                report["status"] = "sic_update_failed"
                self.step_log.append(report)
                logger.warning(f"[Cycle {self.cycle_count}] SIC.update() failed after Commit Gate passed")
                return report
            
            # ===== STEP 8: UME Dynamics =====
            report["steps"]["08_ume"] = self._step_08_ume()
            
            # ===== STEP 9: SMA Fitness + Flow =====
            report["steps"]["09_sma_flow"] = self._step_09_sma_flow()
            
            # ===== STEP 10: Telemetry & Checkpoint =====
            report["steps"]["10_telemetry"] = self._step_10_telemetry()
            
            report["status"] = "success"
            self.step_log.append(report)
            logger.info(f"[Cycle {self.cycle_count}] Complete: success")
            
        except Exception as e:
            report["status"] = "error"
            report["error"] = str(e)
            self.step_log.append(report)
            logger.error(f"[Cycle {self.cycle_count}] Exception: {e}")
        
        return report
    
    # =========================================================================
    # STEP IMPLEMENTATIONS
    # =========================================================================
    
    def _step_01_thermal(self) -> Dict[str, Any]:
        """
        STEP 1: Thermal status check (Veritas Gate).
        Existing Phase 1 logic — unchanged.
        """
        try:
            temp = self.hardware_link.get_thermal_zone_0()
            state = self.veritas_gate.update(temp)
            return {
                "thermal_temp": float(temp),
                "gate_state": state,
                "ok": True,
            }
        except Exception as e:
            return {"thermal_temp": None, "gate_state": "error", "ok": False, "error": str(e)}
    
    def _step_02_vest(self) -> Dict[str, Any]:
        """
        STEP 2: VEST authentication.
        Existing Phase 1 logic — unchanged.
        """
        try:
            challenge = np.random.randn(self.sic.rank)
            response = self.vest.challenge_response(challenge, self.sic.U, self.sic.V)
            distance = self.vest.verify(response, challenge, self.sic.U, self.sic.V)
            authenticated = distance < 0.18
            return {
                "distance": float(distance),
                "authenticated": authenticated,
                "ok": True,
            }
        except Exception as e:
            return {"distance": None, "authenticated": False, "ok": False, "error": str(e)}
    
    def _step_03_sma(self) -> Dict[str, Any]:
        """
        STEP 3: SMA hyperparameter optimization.
        Existing Phase 1 logic — unchanged.
        """
        try:
            # SMA fitness depends on VEST distance + entropy + free energy
            fitness = self.sma.compute_fitness(
                distance=0.1,  # Placeholder
                entropy=0.5,   # Placeholder
                free_energy=-0.2  # Placeholder
            )
            self.sma.step()
            return {
                "fitness": float(fitness),
                "agent_count": self.sma.num_agents,
                "ok": True,
            }
        except Exception as e:
            return {"fitness": None, "ok": False, "error": str(e)}
    
    def _step_04_pre_inference_gate(self) -> Tuple[bool, Dict[str, Any]]:
        """
        STEP 4: PreInferenceGate evaluation (NEW Phase 2).
        
        Checks manifest risk BEFORE calling GGUF.
        
        Returns:
            (pass_decision, report_dict)
        """
        try:
            # Get SIC state for the gate
            sic_state_snapshot = self.sic.get_state_for_gate()
            
            # Use a simple prompt generator (in production: your goal-setting module)
            prompt = self._generate_next_prompt()
            
            # Evaluate PreInferenceGate
            gate_pass, risk_score, factors = self.pre_gate.evaluate(
                prompt=prompt,
                # FIX B: was `sic_state=self` (the Engine). PreInferenceGate
                # reads .U off this object; Engine has no .U, so the estimate
                # silently fell back to the constant 0.5 every cycle.
                sic_state=self.sic,
                cryst_memory=self.cryst_memory,
                prompt_embedding=None,  # Could be computed from tokenizer
            )
            
            report = {
                "prompt": prompt[:50],  # First 50 chars (for logging)
                "gate_pass": gate_pass,
                "risk_score": float(risk_score),
                "factors": factors,
                "ok": True,
            }
            
            return gate_pass, report
        
        except Exception as e:
            logger.error(f"PreInferenceGate exception: {e}")
            return False, {"ok": False, "error": str(e)}
    
    def _step_05_gguf_inference(self) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        STEP 5: GGUF inference with logprobs (NEW Phase 2).
        
        Calls external GGUF engine with logprobs=1 to get Fisher estimate.
        
        Returns:
            (gguf_result_dict, report_dict) or (None, error_report)
        """
        if not self.gguf_engine:
            return None, {
                "ok": False,
                "error": "GGUF engine not set",
            }
        
        try:
            prompt = self._generate_next_prompt()
            
            gguf_result = self.gguf_engine.generate(
                prompt=prompt,
                max_tokens=256,
                logprobs=True,  # Enable Fisher estimate
            )
            
            success = gguf_result.get("success", False)
            
            report = {
                "prompt": prompt[:50],
                "success": success,
                "text": gguf_result.get("text", "")[:100] if success else "",
                "logit_variance": gguf_result.get("logit_variance", 0.0),
                "ok": True,
            }
            
            return (gguf_result if success else None), report
        
        except Exception as e:
            logger.error(f"GGUF inference exception: {e}")
            return None, {"ok": False, "error": str(e)}
    
    def _step_06_commit_gate(self, gguf_result: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        STEP 6: Commit Gate audit (NEW Phase 2).
        
        Five checks: Fisher sharpness, spectral norm, rank preservation, 
        geodesic distance, thermal coupling.
        
        Returns:
            (all_checks_passed, detailed_audit_report)
        """
        try:
            # Draft proposed delta
            delta = self.transfer_controller.draft_delta(
                gguf_output=gguf_result,
                sic_state=self.sic,
            )
            
            if delta is None:
                return False, {
                    "ok": False,
                    "rejection_reason": "Failed to draft delta",
                    "topology_strain": 0.0,
                }
            
            # Run Commit Gate audit
            audit = self.transfer_controller.commit_gate_audit(
                delta=delta,
                sic_state=self.sic,
                cryst_memory=self.cryst_memory,
            )
            
            report = {
                "passed": audit.passed,
                "fisher_sharpness": audit.fisher_sharpness,
                "spectral_norm": audit.spectral_norm,
                "rank_preserved": audit.rank_preserved,
                "geodesic_distance": audit.geodesic_distance,
                "thermal_ok": audit.thermal_ok,
                "all_checks": audit.all_checks,
                "rejection_reason": audit.rejection_reason,
                "topology_strain": delta.topology_strain,
                "ok": True,
            }
            
            return audit.passed, report
        
        except Exception as e:
            logger.error(f"Commit Gate exception: {e}")
            return False, {"ok": False, "error": str(e), "topology_strain": 0.0}
    
    def _step_07_scar_formation(self, gguf_result: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        STEP 7: Scar formation (SIC.update).
        
        ONE place where SIC is written. Commit Gate has already passed.
        
        Returns:
            (success, report_dict)
        """
        try:
            # Convert GGUF text to manifold vector
            # (In production: use tokenizer embedding)
            text = gguf_result.get("text", "")
            x = self._text_to_manifold_vector(text)
            
            # Update SIC
            success = self.sic.update(x, alpha=0.01)
            
            report = {
                "success": success,
                "text_len": len(text),
                "scars_total": self.sic.scars_admitted,
                "ok": True,
            }
            
            if success:
                logger.debug(f"Scar #{self.sic.scars_admitted} admitted")
            
            return success, report
        
        except Exception as e:
            logger.error(f"Scar formation exception: {e}")
            return False, {"ok": False, "error": str(e)}
    
    def _step_08_ume(self) -> Dict[str, Any]:
        """
        STEP 8: UME Itô dynamics.
        Existing Phase 1 logic — unchanged.
        """
        try:
            # UME stochastic perturbation
            self.ume.step(T=0.5)
            return {
                "ume_state": "updated",
                "ok": True,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def _step_09_sma_flow(self) -> Dict[str, Any]:
        """
        STEP 9: SMA fitness update + slime mold flow.
        Existing Phase 1 logic — unchanged.
        """
        try:
            self.sma.step()
            return {
                "sma_agents": self.sma.num_agents,
                "ok": True,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def _step_10_telemetry(self) -> Dict[str, Any]:
        """
        STEP 10: Telemetry & checkpoint.
        Existing Phase 1 logic — enhanced with Phase 2 diagnostics.
        """
        try:
            # Collect all diagnostic summaries
            sic_summary = self.sic.state_summary()
            cryst_summary = self.cryst_memory.state_summary()
            pre_gate_summary = self.pre_gate.state_summary()
            tc_summary = self.transfer_controller.state_summary()
            
            report = {
                "sic": sic_summary,
                "cryst_memory": cryst_summary,
                "pre_gate": pre_gate_summary,
                "transfer_controller": tc_summary,
                "ok": True,
            }
            
            logger.debug(f"Telemetry: {report}")
            return report
        
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _generate_next_prompt(self) -> str:
        """
        Generate next prompt (placeholder).
        In production: replace with your goal-setting / task manager.
        """
        prompts = [
            "What is the nature of identity?",
            "How should systems govern themselves?",
            "Explain topological safety.",
            "Define manifold learning.",
        ]
        idx = self.cycle_count % len(prompts)
        return prompts[idx]
    
    def _text_to_manifold_vector(self, text: str, dim: Optional[int] = None) -> np.ndarray:
        """
        Convert text to manifold embedding (placeholder).
        In production: use tokenizer + embedding layer.
        """
        if dim is None:
            dim = self.sic.d
        
        # REPRODUCIBILITY FIX (2026-08-09): was hash(text), and Python
        # randomizes str hashing per process (PYTHONHASHSEED). Every launch
        # produced different embeddings, so ||U@V.T|| differed on every run
        # (2.428909 container, 2.235383 / 2.255300 / 2.247910 on device).
        # hashlib is stable across processes and machines.
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") % (2**31)
        rng = np.random.RandomState(seed)
        vector = rng.randn(dim).astype(np.float32)
        vector /= np.linalg.norm(vector) + 1e-10
        return vector
    
    def get_state_for_gate(self) -> Dict[str, Any]:
        """
        Expose SIC state to PreInferenceGate via delegation.
        """
        return self.sic.get_state_for_gate()
    
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    
    def state_summary(self) -> Dict[str, Any]:
        """
        Complete diagnostic summary of engine state.
        """
        return {
            "cycle_count": self.cycle_count,
            "sic": self.sic.state_summary(),
            "cryst_memory": self.cryst_memory.state_summary(),
            "pre_gate": self.pre_gate.state_summary(),
            "transfer_controller": self.transfer_controller.state_summary(),
            "recent_steps": len(self.step_log),
        }
    
    def print_diagnostics(self) -> None:
        """
        Pretty-print full engine diagnostics.
        """
        summary = self.state_summary()
        
        print("\n" + "=" * 70)
        print("ENGINE DIAGNOSTICS (Phase 1 + Phase 2)")
        print("=" * 70)
        
        print(f"\nCycle Count: {summary['cycle_count']}")
        print(f"Recent Steps in Log: {summary['recent_steps']}")
        
        print("\n[SIC]")
        for k, v in summary["sic"].items():
            print(f"  {k}: {v}")
        
        print("\n[CRYSTALLIZATION MEMORY]")
        for k, v in summary["cryst_memory"].items():
            print(f"  {k}: {v}")
        
        print("\n[PRE-INFERENCE GATE]")
        pre_gate = summary["pre_gate"]
        print(f"  total_evaluations: {pre_gate['total_evaluations']}")
        print(f"  pass_rate: {pre_gate['pass_rate']:.2%}")
        print(f"  threshold: {pre_gate['threshold']}")
        
        print("\n[TRANSFER CONTROLLER]")
        tc = summary["transfer_controller"]
        print(f"  total_submissions: {tc['total_submissions']}")
        print(f"  accept_rate: {tc['accept_rate']:.2%}")
        print(f"  fisher_threshold: {tc['fisher_threshold']}")
        
        print("\n" + "=" * 70)


# ============================================================================
# ENTRY POINT (for testing)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("[Engine] Initializing with Phase 2 governance...\n")
    
    # Mock GGUF engine for testing
    class MockGGUFEngine:
        def generate(self, prompt: str, max_tokens: int = 256, logprobs: bool = False) -> Dict[str, Any]:
            return {
                "success": np.random.rand() > 0.15,  # 85% success rate
                "text": f"Response to: {prompt}",
                "logit_variance": np.random.uniform(0.75, 0.95),
            }
    
    # Initialize engine
    engine = Engine(
        d=512,
        rank=64,
        gguf_engine=MockGGUFEngine(),
        cryst_memory_window=20,
    )
    
    # Run a few test cycles
    print("[Engine] Running 5 test cycles...\n")
    for i in range(5):
        report = engine.step()
        status = report["status"]
        print(f"Cycle {i+1}: {status}")
    
    # Print diagnostics
    engine.print_diagnostics()
