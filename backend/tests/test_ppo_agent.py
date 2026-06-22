import torch
import numpy as np
import pytest

from src.models.ppo_agent import PPOAgent, Trajectory, TrainingResult, ValueNetwork
from src.models.network import NonDifferentiableNetwork


def test_trajectory_default_fields_are_empty_lists():
    t = Trajectory()
    assert t.states == []
    assert t.actions == []
    assert t.rewards == []
    assert t.log_probs == []
    assert t.values == []
    assert t.dones == []


def test_trajectory_clear_empties_all_lists(trajectory):
    trajectory.clear()
    assert trajectory.states == []
    assert trajectory.actions == []
    assert trajectory.rewards == []
    assert trajectory.log_probs == []
    assert trajectory.values == []
    assert trajectory.dones == []


def test_training_result_to_dict_contains_expected_keys():
    result = TrainingResult()
    d = result.to_dict()
    expected_keys = {
        'episode_rewards', 'episode_lengths', 'policy_losses',
        'value_losses', 'temperatures', 'best_reward', 'avg_reward_last_100',
    }
    assert set(d.keys()) == expected_keys


def test_training_result_avg_reward_last_100_empty():
    result = TrainingResult()
    d = result.to_dict()
    assert d['avg_reward_last_100'] == 0.0


def test_training_result_avg_reward_last_100_calculation():
    result = TrainingResult(episode_rewards=list(range(200)))
    d = result.to_dict()
    expected = float(np.mean(list(range(100, 200))))
    assert d['avg_reward_last_100'] == pytest.approx(expected)


def test_training_result_avg_reward_last_100_fewer_than_100():
    rewards = [1.0, 2.0, 3.0]
    result = TrainingResult(episode_rewards=rewards)
    d = result.to_dict()
    assert d['avg_reward_last_100'] == pytest.approx(float(np.mean(rewards)))


def test_value_network_output_shape_batch(value_network, state_dim):
    x = torch.randn(5, state_dim)
    out = value_network(x)
    assert out.shape == (5,)


def test_value_network_output_shape_single(value_network, state_dim):
    x = torch.randn(state_dim)
    out = value_network(x)
    assert out.shape == ()
    assert out.ndim == 0


def test_ppo_agent_select_action_returns_valid_action(ppo_agent, sample_state, action_dim):
    action, log_prob, value = ppo_agent.select_action(sample_state)
    assert 0 <= action < action_dim
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_ppo_agent_select_action_deterministic(ppo_agent, sample_state):
    actions = [ppo_agent.select_action(sample_state, deterministic=True)[0] for _ in range(20)]
    assert len(set(actions)) == 1


def test_ppo_agent_normalize_state_updates_stats(ppo_agent, state_dim):
    state = np.ones(state_dim, dtype=np.float32) * 5.0
    ppo_agent.normalize_state(state, update_stats=True)
    assert ppo_agent.state_count == 1
    np.testing.assert_allclose(ppo_agent.state_mean, state, atol=1e-5)


def test_ppo_agent_normalize_state_no_update(ppo_agent, state_dim):
    original_mean = ppo_agent.state_mean.copy()
    original_std = ppo_agent.state_std.copy()
    original_count = ppo_agent.state_count
    state = np.ones(state_dim, dtype=np.float32) * 99.0
    ppo_agent.normalize_state(state, update_stats=False)
    np.testing.assert_array_equal(ppo_agent.state_mean, original_mean)
    np.testing.assert_array_equal(ppo_agent.state_std, original_std)
    assert ppo_agent.state_count == original_count


def test_ppo_agent_compute_gae_returns_tensors(ppo_agent, trajectory):
    next_value = 0.0
    advantages, returns = ppo_agent.compute_gae(
        trajectory.rewards, trajectory.values, trajectory.dones, next_value
    )
    assert isinstance(advantages, torch.Tensor)
    assert isinstance(returns, torch.Tensor)
    assert advantages.shape == (len(trajectory.rewards),)
    assert returns.shape == (len(trajectory.rewards),)


def test_ppo_agent_update_returns_loss_dict(ppo_agent, trajectory):
    next_value = 0.0
    result = ppo_agent.update(trajectory, next_value)
    expected_keys = {'policy_loss', 'value_loss', 'entropy', 'reg_loss', 'temperature', 'target_updated'}
    assert set(result.keys()) == expected_keys
    for key in ['policy_loss', 'value_loss', 'entropy', 'reg_loss']:
        assert isinstance(result[key], float)
    assert isinstance(result['temperature'], float)
    assert isinstance(result['target_updated'], bool)


def test_ppo_agent_save_load_roundtrip(ppo_agent, sample_state, tmp_path):
    ppo_agent.normalize_state(sample_state, update_stats=True)
    mean_before = ppo_agent.state_mean.copy()
    std_before = ppo_agent.state_std.copy()
    count_before = ppo_agent.state_count

    policy_params_before = {k: v.clone() for k, v in ppo_agent.policy_net.state_dict().items()}
    value_params_before = {k: v.clone() for k, v in ppo_agent.value_net.state_dict().items()}

    save_path = str(tmp_path / "ppo_model.pt")
    ppo_agent.save(save_path)

    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
    try:
        ppo_agent.load(save_path)
    finally:
        torch.load = original_load

    np.testing.assert_array_equal(ppo_agent.state_mean, mean_before)
    np.testing.assert_array_equal(ppo_agent.state_std, std_before)
    assert ppo_agent.state_count == count_before

    for k, v in ppo_agent.policy_net.state_dict().items():
        assert torch.allclose(v, policy_params_before[k], atol=1e-6)
    for k, v in ppo_agent.value_net.state_dict().items():
        assert torch.allclose(v, value_params_before[k], atol=1e-6)


def test_ppo_agent_set_target_network(ppo_agent, state_dim, action_dim):
    target = NonDifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
    ppo_agent.set_target_network(target)
    assert ppo_agent.target_network is not None
    for param in ppo_agent.target_network.parameters():
        assert param.requires_grad is False


def test_ppo_agent_device_auto_selection(state_dim, action_dim):
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device="auto",
        ppo_epochs=2,
        batch_size=32,
    )
    assert agent.device.type in ["cpu", "cuda"]


def test_ppo_agent_device_manual_cpu(state_dim, action_dim):
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device="cpu",
        ppo_epochs=2,
        batch_size=32,
    )
    assert agent.device.type == "cpu"


def test_ppo_agent_normalize_state_multiple_states(ppo_agent, state_dim):
    states = [np.ones(state_dim) * i for i in range(10)]

    for state in states:
        ppo_agent.normalize_state(state, update_stats=True)

    assert ppo_agent.state_count == 10
    assert ppo_agent.state_mean.shape == (state_dim,)
    assert ppo_agent.state_std.shape == (state_dim,)


def test_ppo_agent_normalize_state_std_clamped(ppo_agent, state_dim):
    state = np.ones(state_dim) * 5.0
    ppo_agent.normalize_state(state, update_stats=True)

    assert (ppo_agent.state_std >= 1e-6).all()


def test_ppo_agent_compute_gae_with_dones(ppo_agent, trajectory):
    next_value = 0.5
    advantages, returns = ppo_agent.compute_gae(
        trajectory.rewards, trajectory.values, trajectory.dones, next_value
    )

    assert advantages.shape == (len(trajectory.rewards),)
    assert returns.shape == (len(trajectory.rewards),)
    assert not torch.isnan(advantages).any()
    assert not torch.isinf(advantages).any()


def test_ppo_agent_compute_gae_no_dones(ppo_agent):
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.6, 0.7]
    dones = [False, False, False]
    next_value = 0.8

    advantages, returns = ppo_agent.compute_gae(rewards, values, dones, next_value)

    assert advantages.shape == (3,)
    assert returns.shape == (3,)


def test_ppo_agent_compute_gae_all_dones(ppo_agent):
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.6, 0.7]
    dones = [False, False, True]
    next_value = 0.0

    advantages, returns = ppo_agent.compute_gae(rewards, values, dones, next_value)

    assert advantages.shape == (3,)
    assert returns.shape == (3,)


def test_ppo_agent_compute_gae_with_zero_rewards(ppo_agent):
    rewards = [0.0, 0.0, 0.0]
    values = [0.0, 0.0, 0.0]
    dones = [False, False, True]
    next_value = 0.0

    advantages, returns = ppo_agent.compute_gae(rewards, values, dones, next_value)

    assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-5)
    assert torch.allclose(returns, torch.zeros_like(returns), atol=1e-5)


def test_ppo_agent_update_with_target_network(ppo_agent, trajectory, state_dim, action_dim):
    target = NonDifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
    ppo_agent.set_target_network(target)

    next_value = 0.0
    result = ppo_agent.update(trajectory, next_value)

    assert 'reg_loss' in result
    assert result['reg_loss'] >= 0.0


def test_ppo_agent_update_losses_decrease(ppo_agent, trajectory):
    next_value = 0.0
    result = ppo_agent.update(trajectory, next_value)

    assert result['policy_loss'] is not None
    assert result['value_loss'] is not None
    assert result['entropy'] >= 0.0


def test_ppo_agent_temperature_decays_on_update(ppo_agent, trajectory):
    initial_temp = ppo_agent.policy_net.temperature
    next_value = 0.0
    result = ppo_agent.update(trajectory, next_value)

    assert result['temperature'] <= initial_temp
    assert result['temperature'] >= ppo_agent.min_temperature


def test_ppo_agent_select_action_log_prob_range(ppo_agent, sample_state):
    action, log_prob, value = ppo_agent.select_action(sample_state)

    assert log_prob <= 0.0
    assert isinstance(value, float)


def test_ppo_agent_multiple_select_actions(ppo_agent, sample_state, action_dim):
    actions = []
    for _ in range(20):
        action, _, _ = ppo_agent.select_action(sample_state, deterministic=False)
        actions.append(action)

    assert all(0 <= a < action_dim for a in actions)


def test_ppo_agent_value_network_batch(value_network, state_dim):
    batch = torch.randn(10, state_dim)
    values = value_network(batch)

    assert values.shape == (10,)
    assert not torch.isnan(values).any()


def test_ppo_agent_value_network_gradient_flow(value_network, state_dim):
    batch = torch.randn(5, state_dim, requires_grad=True)
    values = value_network(batch)
    loss = values.sum()
    loss.backward()

    assert batch.grad is not None
    assert batch.grad.shape == (5, state_dim)
    assert not torch.isnan(batch.grad).any()


def test_trajectory_add_elements():
    t = Trajectory()
    t.states.append(np.random.randn(8))
    t.actions.append(1)
    t.rewards.append(1.0)
    t.log_probs.append(-0.5)
    t.values.append(0.3)
    t.dones.append(False)

    assert len(t.states) == 1
    assert len(t.actions) == 1
    assert len(t.rewards) == 1
    assert len(t.log_probs) == 1
    assert len(t.values) == 1
    assert len(t.dones) == 1


def test_training_result_best_reward_tracking():
    result = TrainingResult()
    rewards = [10.0, 20.0, 15.0, 30.0, 25.0]

    for r in rewards:
        result.episode_rewards.append(r)
        if r > result.best_reward:
            result.best_reward = r

    assert result.best_reward == 30.0
    assert len(result.episode_rewards) == 5


def test_training_result_to_dict_with_data():
    result = TrainingResult(
        episode_rewards=[100.0, 150.0, 200.0],
        episode_lengths=[50, 75, 100],
        policy_losses=[0.5, 0.3, 0.1],
        value_losses=[1.0, 0.8, 0.6],
        temperatures=[1.0, 0.9, 0.8],
        best_reward=200.0,
    )

    d = result.to_dict()
    assert d['best_reward'] == 200.0
    assert d['avg_reward_last_100'] == pytest.approx(150.0)
    assert len(d['episode_rewards']) == 3


def test_ppo_agent_evaluate_deterministic(ppo_agent, sample_state):
    actions = []
    for _ in range(10):
        action, _, _ = ppo_agent.select_action(sample_state, deterministic=True)
        actions.append(action)

    assert all(a == actions[0] for a in actions)


def test_ppo_agent_evaluate_stochastic_diversity(ppo_agent, sample_state, action_dim):
    if action_dim > 1:
        actions = []
        for _ in range(50):
            action, _, _ = ppo_agent.select_action(sample_state, deterministic=False)
            actions.append(action)

        unique_actions = set(actions)
        assert len(unique_actions) >= 1


def test_ppo_agent_target_mode_random(state_dim, action_dim):
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        ppo_epochs=1,
        batch_size=8,
        device="cpu",
        target_mode="random"
    )
    assert agent.target_mode == "random"
    assert agent.target_network is not None
    from src.models.network import NonDifferentiableNetwork
    assert isinstance(agent.target_network, NonDifferentiableNetwork)
    for param in agent.target_network.parameters():
        assert param.requires_grad is False


def test_ppo_agent_target_mode_frozen_differentiable(state_dim, action_dim):
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        ppo_epochs=1,
        batch_size=8,
        device="cpu",
        target_mode="frozen_differentiable",
        target_quantize_bits=4,
        weight_copy_interval=5,
        harden_on_copy=True
    )
    assert agent.target_mode == "frozen_differentiable"
    assert agent.target_quantize_bits == 4
    assert agent.weight_copy_interval == 5
    assert agent.harden_on_copy is True
    assert agent.target_network is not None
    from src.models.network import DifferentiableNetwork
    assert isinstance(agent.target_network, DifferentiableNetwork)
    for param in agent.target_network.parameters():
        assert param.requires_grad is False


def test_ppo_agent_target_mode_seed_based(state_dim, action_dim):
    seeds = list(range(1, 25))
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        ppo_epochs=1,
        batch_size=8,
        device="cpu",
        target_mode="seed_based",
        target_seeds=seeds
    )
    assert agent.target_mode == "seed_based"
    assert agent.target_seeds == seeds
    assert agent.target_network is not None
    from src.models.network import DifferentiableNetwork
    assert isinstance(agent.target_network, DifferentiableNetwork)
    for param in agent.target_network.parameters():
        assert param.requires_grad is False


def test_ppo_agent_target_mode_seed_based_no_seeds_raises(state_dim, action_dim):
    with pytest.raises(ValueError, match="target_seeds is required"):
        PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            ppo_epochs=1,
            batch_size=8,
            device="cpu",
            target_mode="seed_based",
            target_seeds=None
        )


def test_ppo_agent_target_mode_seed_based_wrong_seed_count(state_dim, action_dim):
    with pytest.raises(ValueError, match="Expected 24 seeds"):
        PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            ppo_epochs=1,
            batch_size=8,
            device="cpu",
            target_mode="seed_based",
            target_seeds=list(range(1, 10))
        )


def test_ppo_agent_periodic_target_update(state_dim, action_dim, trajectory):
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        ppo_epochs=1,
        batch_size=8,
        device="cpu",
        target_mode="frozen_differentiable",
        weight_copy_interval=2,
        harden_on_copy=True
    )

    assert agent._update_count == 0

    result1 = agent.update(trajectory, 0.0)
    assert result1['target_updated'] is False
    assert agent._update_count == 1

    result2 = agent.update(trajectory, 0.0)
    assert result2['target_updated'] is True
    assert agent._update_count == 2

    result3 = agent.update(trajectory, 0.0)
    assert result3['target_updated'] is False
    assert agent._update_count == 3

    result4 = agent.update(trajectory, 0.0)
    assert result4['target_updated'] is True
    assert agent._update_count == 4


def test_ppo_agent_regularization_loss_with_target(ppo_agent, trajectory):
    from src.models.network import NonDifferentiableNetwork
    target = NonDifferentiableNetwork(
        state_dim=ppo_agent.state_dim,
        action_dim=ppo_agent.action_dim
    )
    ppo_agent.set_target_network(target)

    result = ppo_agent.update(trajectory, 0.0)
    assert result['reg_loss'] >= 0.0
    assert 'reg_loss' in result


def test_ppo_agent_update_count_increments(ppo_agent, trajectory):
    initial_count = ppo_agent._update_count
    ppo_agent.update(trajectory, 0.0)
    assert ppo_agent._update_count == initial_count + 1
