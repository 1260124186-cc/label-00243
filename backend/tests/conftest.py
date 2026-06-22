import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.network import (
    NonDifferentiableAttentionLayer,
    DifferentiableAttentionLayer,
    NonDifferentiableNetwork,
    DifferentiableNetwork,
)
from src.models.ppo_agent import PPOAgent, Trajectory, ValueNetwork
from src.models.genetic_algorithm import Individual, WeightGenerator, GeneticAlgorithm


@pytest.fixture
def state_dim():
    return 8


@pytest.fixture
def action_dim():
    return 4


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def non_diff_attention_layer(state_dim):
    return NonDifferentiableAttentionLayer(input_dim=state_dim, output_dim=state_dim * 3)


@pytest.fixture
def diff_attention_layer(state_dim):
    return DifferentiableAttentionLayer(input_dim=state_dim, output_dim=state_dim * 3, temperature=1.0)


@pytest.fixture
def non_diff_network(state_dim, action_dim):
    return NonDifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)


@pytest.fixture
def diff_network(state_dim, action_dim):
    return DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim, initial_temperature=1.0)


@pytest.fixture
def value_network(state_dim):
    return ValueNetwork(state_dim=state_dim)


@pytest.fixture
def ppo_agent(state_dim, action_dim):
    return PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=3e-4,
        ppo_epochs=2,
        batch_size=32,
        device="cpu",
    )


@pytest.fixture
def trajectory():
    return Trajectory(
        states=[np.random.randn(8) for _ in range(10)],
        actions=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        rewards=[1.0, -0.5, 0.3, 2.0, -1.0, 0.5, 1.5, -0.2, 0.8, 0.1],
        log_probs=[-1.2, -0.8, -1.5, -0.3, -2.0, -0.9, -0.6, -1.1, -0.7, -1.3],
        values=[0.5, 0.3, 0.6, 0.8, 0.2, 0.4, 0.7, 0.1, 0.3, 0.5],
        dones=[False, False, False, False, False, False, False, False, False, True],
    )


@pytest.fixture
def individual():
    seeds = np.array([
        [100, 200, 300, 400, 7, 3],
        [500, 600, 700, 800, 11, 5],
        [900, 1000, 1100, 1200, 13, 7],
        [1300, 1400, 1500, 1600, 17, 11],
    ])
    return Individual(seeds=seeds, fitness=100.0, generation=5)


@pytest.fixture
def weight_generator():
    return WeightGenerator()


@pytest.fixture
def genetic_algorithm():
    return GeneticAlgorithm(
        population_size=10,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elite_size=2,
        seed_range=(0, 100),
        traversal_enabled=False,
        alpha=0.9,
    )


@pytest.fixture
def sample_state(state_dim):
    return np.random.randn(state_dim).astype(np.float32)


@pytest.fixture
def sample_batch(state_dim, batch_size):
    return torch.randn(batch_size, state_dim)


@pytest.fixture
def network_shapes():
    return {
        "layer1.linear_q1.weight": (24, 8),
        "layer1.linear_q1.bias": (24,),
        "layer1.linear_k1.weight": (24, 8),
        "layer1.linear_k1.bias": (24,),
    }
