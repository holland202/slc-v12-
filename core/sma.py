"""sma.py — Slime Mold Optimization Layer (OMOL) · SLC v12

Adaptive manifold optimization without backpropagation.
Agents explore parameter space θ = (α, β, γ, r) under composite fitness:
  F_i = λ₁·d_VEST + λ₂·H_σ + λ₃·E_t
"""
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SMAAgent:
    """Single OMOL agent with position in parameter space."""
    alpha: float
    beta: float
    gamma: float
    rank: int
    fitness: float = float('inf')
    
    def to_array(self) -> np.ndarray:
        return np.array([self.alpha, self.beta, self.gamma, float(self.rank)])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SMAAgent":
        return cls(
            alpha=float(arr[0]),
            beta=float(arr[1]),
            gamma=float(arr[2]),
            rank=int(max(1, round(arr[3]))),
        )

class SlimeMoldOptimizer:
    """OMOL: oscillatory swarm optimization over SIC hyperparameters."""
    
    def __init__(self, n_agents: int = 8,
                 lambda_vest: float = 0.4,
                 lambda_spectral: float = 0.35,
                 lambda_thermal: float = 0.25):
        self.n_agents = n_agents
        self.lambda_vest = lambda_vest
        self.lambda_spectral = lambda_spectral
        self.lambda_thermal = lambda_thermal
        
        self.bounds = np.array([
            [0.001, 0.05],
            [0.1, 2.0],
            [1e-6, 1e-3],
            [16.0, 128.0],
        ])
        
        self.agents: List[SMAAgent] = []
        for _ in range(n_agents):
            pos = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            self.agents.append(SMAAgent.from_array(pos))
        
        self.best_agent = self.agents[0]
        self._generation = 0
        
    def evaluate_fitness(self, agent: SMAAgent,
                         vest_distance: float,
                         spectral_entropy: float,
                         thermal_energy: float) -> float:
        f_vest = vest_distance / 0.5
        f_spec = spectral_entropy / 28.5
        f_therm = thermal_energy / 50.0
        return (self.lambda_vest * f_vest +
                self.lambda_spectral * f_spec +
                self.lambda_thermal * f_therm)
    
    def step(self, vest_distance: float, spectral_entropy: float,
             thermal_energy: float) -> SMAAgent:
        for agent in self.agents:
            agent.fitness = self.evaluate_fitness(
                agent, vest_distance, spectral_entropy, thermal_energy
            )
        
        self.best_agent = min(self.agents, key=lambda a: a.fitness)
        
        positions = np.stack([a.to_array() for a in self.agents])
        centroid = np.mean(positions, axis=0)
        
        v_b = 0.3 + 0.2 * np.sin(self._generation * 0.1)
        v_c = 0.2 + 0.1 * np.cos(self._generation * 0.15)
        
        for i, agent in enumerate(self.agents):
            pos = agent.to_array()
            best_pos = self.best_agent.to_array()
            
            new_pos = pos + v_b * (best_pos - pos) + v_c * (centroid - pos)
            noise = np.random.normal(0, 0.01, 4)
            new_pos += noise
            new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
            self.agents[i] = SMAAgent.from_array(new_pos)
        
        self._generation += 1
        return self.best_agent
    
    def get_recommended_params(self) -> Dict[str, float]:
        return {
            "alpha": self.best_agent.alpha,
            "beta": self.best_agent.beta,
            "gamma": self.best_agent.gamma,
            "rank": self.best_agent.rank,
            "fitness": self.best_agent.fitness,
        }
    
    def state_summary(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "n_agents": self.n_agents,
            "best_fitness": round(self.best_agent.fitness, 4),
            "best_params": self.get_recommended_params(),
            "mean_fitness": round(np.mean([a.fitness for a in self.agents]), 4),
        }
