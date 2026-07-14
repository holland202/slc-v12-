"""params.py — SLC v12 Configuration & Parameters"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

@dataclass
class ManifoldParams:
    d: int = 512
    rank: int = 64
    spectral_bound_U: float = 1.0
    spectral_bound_V: float = 3.0

    @property
    def compression_ratio(self) -> float:
        return self.rank / self.d

    def __post_init__(self):
        assert self.d > 0
        assert 0 < self.rank < self.d

@dataclass
class ThermalParams:
    T_ambient: float = 28.0
    T_warn: float = 43.0
    T_throttle: float = 48.0
    T_critical: float = 50.0
    T_recovery: float = 38.0
    hysteresis_width: float = 2.0
    tau_thermal: float = 5.0
    duty_cycle_normal: float = 1.0
    duty_cycle_warn: float = 0.7
    duty_cycle_throttle: float = 0.3
    gpu_layers_baseline: int = 33
    gpu_layers_min: int = 8

    def __post_init__(self):
        assert self.T_ambient < self.T_warn < self.T_throttle < self.T_critical
        assert self.T_recovery < self.T_warn

@dataclass
class InferenceParams:
    max_tokens: int = 256
    context_window: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    timeout_generate_s: float = 30.0
    timeout_response_s: float = 5.0
    enable_logprobs: bool = True
    logprob_max_tokens: int = 5

    def __post_init__(self):
        assert self.max_tokens > 0
        assert self.context_window >= self.max_tokens

@dataclass
class SMAParams:
    alpha: float = 0.01
    sma_factor: float = 0.001
    max_iterations_per_scar: int = 3
    decay_schedule: str = "none"
    decay_rate: float = 0.9999

    def current_alpha(self, cycle: int) -> float:
        if self.decay_schedule == "none":
            return self.alpha
        elif self.decay_schedule == "exponential":
            return self.alpha * (self.decay_rate ** cycle)
        elif self.decay_schedule == "linear":
            return max(self.alpha * 0.1, self.alpha * (1.0 - cycle * 0.00001))
        return self.alpha

@dataclass
class UMEParams:
    epoch_size: int = 50
    max_epochs: int = 4
    embedding_dim: int = 128
    compress: bool = True
    compress_dtype: str = "float16"

    @property
    def max_buffer_size(self) -> int:
        return self.epoch_size * self.max_epochs

@dataclass
class VeritasGateParams:
    check_rank_preservation: bool = True
    check_spectral_norm: bool = True
    check_determinant_sign: bool = True
    min_logit_variance: float = 0.70
    min_fisher_sharpness: float = 0.85
    thermal_gate_coupling: float = 0.8
    max_duty_cycle_high_temp: float = 0.3
    max_consecutive_rejections: int = 10
    rejection_window_cycles: int = 100
    max_rejection_rate: float = 0.95

@dataclass
class CalibrationParams:
    gate_threshold: float = 0.65
    gate_weights: Tuple[float, float, float, float] = (0.25, 0.30, 0.25, 0.20)
    gate_steepness: float = 5.0
    fisher_threshold: float = 0.85
    spectral_norm_max: float = 2.0
    geodesic_distance_max: float = 0.15
    thermal_multiplier: float = 1.0
    cryst_memory_window: int = 20
    sweep_gate_thr_range: Tuple[float, float] = (0.55, 0.75)
    sweep_fisher_thr_range: Tuple[float, float] = (0.75, 0.95)
    sweep_grid_size: int = 5

@dataclass
class SLCConfig:
    manifold: ManifoldParams = field(default_factory=ManifoldParams)
    thermal: ThermalParams = field(default_factory=ThermalParams)
    inference: InferenceParams = field(default_factory=InferenceParams)
    sma: SMAParams = field(default_factory=SMAParams)
    ume: UMEParams = field(default_factory=UMEParams)
    veritas: VeritasGateParams = field(default_factory=VeritasGateParams)
    calibration: CalibrationParams = field(default_factory=CalibrationParams)

    device_name: str = "Samsung Galaxy S25 Ultra (SM8750-AB)"
    target_soc: str = "Snapdragon 8 Elite"
    log_level: str = "INFO"
    save_checkpoints: bool = False
    checkpoint_dir: str = "./checkpoints"

    @classmethod
    def s25_ultra_default(cls) -> "SLCConfig":
        return cls(
            manifold=ManifoldParams(d=512, rank=64),
            thermal=ThermalParams(T_warn=43.0, T_throttle=48.0, T_critical=50.0,
                                  gpu_layers_baseline=33, gpu_layers_min=8),
            inference=InferenceParams(max_tokens=256, context_window=2048,
                                      temperature=0.7, enable_logprobs=True),
            sma=SMAParams(alpha=0.01, sma_factor=0.001),
            ume=UMEParams(epoch_size=50, max_epochs=4),
        )

    @classmethod
    def mobile_lightweight(cls) -> "SLCConfig":
        return cls(
            manifold=ManifoldParams(d=256, rank=32),
            thermal=ThermalParams(T_warn=41.0, T_throttle=45.0,
                                  gpu_layers_baseline=16),
            inference=InferenceParams(max_tokens=128, context_window=1024),
        )

    @classmethod
    def desktop_unrestricted(cls) -> "SLCConfig":
        return cls(
            manifold=ManifoldParams(d=2048, rank=256),
            thermal=ThermalParams(T_critical=85.0, duty_cycle_normal=1.0),
            inference=InferenceParams(max_tokens=512, context_window=4096),
        )

    @classmethod
    def healthcare(cls) -> "SLCConfig":
        from .params_sector import get_sector_thermal
        cfg = cls.s25_ultra_default()
        profile = get_sector_thermal("healthcare")
        cfg.thermal.T_warn = profile["T_warn"]
        cfg.thermal.T_throttle = profile["T_throttle"]
        cfg.thermal.T_critical = profile["T_critical"]
        cfg.thermal.T_recovery = profile["T_recovery"]
        cfg.thermal.duty_cycle_warn = profile["duty_cycle_warn"]
        cfg.thermal.duty_cycle_throttle = profile["duty_cycle_throttle"]
        cfg.inference.temperature = 0.3
        cfg.inference.max_tokens = 512
        cfg.veritas.min_fisher_sharpness = 0.90
        cfg.veritas.min_logit_variance = 0.80
        return cfg

    @classmethod
    def for_sector(cls, sector: str) -> "SLCConfig":
        from .params_sector import get_sector_thermal
        if sector == "healthcare":
            return cls.healthcare()
        cfg = cls.s25_ultra_default()
        profile = get_sector_thermal(sector)
        cfg.thermal.T_warn = profile["T_warn"]
        cfg.thermal.T_throttle = profile["T_throttle"]
        cfg.thermal.T_critical = profile["T_critical"]
        cfg.thermal.T_recovery = profile["T_recovery"]
        cfg.thermal.duty_cycle_warn = profile["duty_cycle_warn"]
        cfg.thermal.duty_cycle_throttle = profile["duty_cycle_throttle"]
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifold": {
                "d": self.manifold.d,
                "rank": self.manifold.rank,
                "compression_ratio": self.manifold.compression_ratio,
            },
            "thermal": {
                "T_warn": self.thermal.T_warn,
                "T_throttle": self.thermal.T_throttle,
                "T_critical": self.thermal.T_critical,
            },
            "inference": {
                "max_tokens": self.inference.max_tokens,
                "context_window": self.inference.context_window,
                "temperature": self.inference.temperature,
            },
            "device": self.device_name,
        }

DEFAULT_CONFIG: SLCConfig = SLCConfig.s25_ultra_default()

def get_default_config() -> SLCConfig:
    return DEFAULT_CONFIG
