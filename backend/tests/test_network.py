import pytest
import torch
import numpy as np
from src.models.network import (
    NonDifferentiableAttentionLayer,
    DifferentiableAttentionLayer,
    NonDifferentiableNetwork,
    DifferentiableNetwork,
)


def test_non_diff_attention_layer_2d_output_shape(non_diff_attention_layer, state_dim, batch_size, sample_batch):
    output = non_diff_attention_layer(sample_batch)
    assert output.shape == (batch_size, state_dim * 3)


def test_non_diff_attention_layer_1d_output_shape(non_diff_attention_layer, state_dim, sample_state):
    x = torch.from_numpy(sample_state).float()
    output = non_diff_attention_layer(x)
    assert output.shape == (state_dim * 3,)


def test_non_diff_attention_layer_no_nan_2d(non_diff_attention_layer, sample_batch):
    output = non_diff_attention_layer(sample_batch)
    assert not torch.isnan(output).any()


def test_non_diff_attention_layer_no_nan_1d(non_diff_attention_layer, sample_state):
    x = torch.from_numpy(sample_state).float()
    output = non_diff_attention_layer(x)
    assert not torch.isnan(output).any()


def test_non_diff_attention_layer_different_input_dims():
    layer = NonDifferentiableAttentionLayer(input_dim=16, output_dim=48)
    x = torch.randn(2, 16)
    output = layer(x)
    assert output.shape == (2, 48)


def test_diff_attention_layer_2d_output_shape(diff_attention_layer, state_dim, batch_size, sample_batch):
    output = diff_attention_layer(sample_batch)
    assert output.shape == (batch_size, state_dim * 3)


def test_diff_attention_layer_1d_output_shape(diff_attention_layer, state_dim, sample_state):
    x = torch.from_numpy(sample_state).float()
    output = diff_attention_layer(x)
    assert output.shape == (state_dim * 3,)


def test_diff_attention_layer_gradients_flow(diff_attention_layer, state_dim, sample_batch):
    x = sample_batch.requires_grad_(True)
    output = diff_attention_layer(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == sample_batch.shape
    assert not torch.isnan(x.grad).any()


def test_diff_attention_layer_set_temperature(diff_attention_layer):
    diff_attention_layer.set_temperature(0.5)
    assert diff_attention_layer.temperature == 0.5


def test_diff_attention_layer_set_temperature_clamp(diff_attention_layer):
    diff_attention_layer.set_temperature(0.0)
    assert diff_attention_layer.temperature == 1e-6


def test_diff_attention_layer_set_temperature_negative(diff_attention_layer):
    diff_attention_layer.set_temperature(-1.0)
    assert diff_attention_layer.temperature == 1e-6


def test_non_diff_network_batch_output_shape(non_diff_network, state_dim, action_dim, sample_batch):
    output = non_diff_network(sample_batch)
    assert output.shape == (sample_batch.shape[0], action_dim)


def test_non_diff_network_single_input(non_diff_network, action_dim, sample_state):
    x = torch.from_numpy(sample_state).float()
    output = non_diff_network(x)
    assert output.shape == (action_dim,)


def test_non_diff_network_has_parameters(non_diff_network):
    params = list(non_diff_network.parameters())
    assert len(params) > 0
    for p in params:
        assert p.requires_grad


def test_diff_network_batch_output_shape(diff_network, action_dim, sample_batch):
    output = diff_network(sample_batch)
    assert output.shape == (sample_batch.shape[0], action_dim)


def test_diff_network_single_input(diff_network, action_dim, sample_state):
    x = torch.from_numpy(sample_state).float()
    output = diff_network(x)
    assert output.shape == (action_dim,)


def test_diff_network_temperature_annealing_decreases(diff_network):
    initial_temp = diff_network.temperature
    new_temp = diff_network.anneal_temperature(decay_rate=0.995, min_temperature=0.01)
    assert new_temp < initial_temp
    assert new_temp == pytest.approx(initial_temp * 0.995, rel=1e-5)


def test_diff_network_temperature_annealing_respects_min(state_dim, action_dim):
    network = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim, initial_temperature=0.005)
    new_temp = network.anneal_temperature(decay_rate=0.995, min_temperature=0.01)
    assert new_temp == 0.01


def test_diff_network_set_temperature(state_dim, action_dim):
    network = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim, initial_temperature=1.0)
    network.set_temperature(0.1)
    assert network.temperature == 0.1
    assert network.layer1.temperature == 0.1
    assert network.layer2.temperature == 0.1


def test_diff_network_set_temperature_clamp(state_dim, action_dim):
    network = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim, initial_temperature=1.0)
    network.set_temperature(0.0)
    assert network.temperature == 1e-6
    assert network.layer1.temperature == 1e-6
    assert network.layer2.temperature == 1e-6


def test_diff_network_copy_weights_from(state_dim, action_dim):
    source = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim, initial_temperature=1.0)
    target = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim, initial_temperature=1.0)
    source_state = source.state_dict()
    target.load_state_dict(source_state)
    for (n1, p1), (n2, p2) in zip(source.named_parameters(), target.named_parameters()):
        assert torch.equal(p1, p2)


def test_diff_network_get_regularization_loss_non_negative(diff_network, state_dim, action_dim):
    target = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim, initial_temperature=1.0)
    loss = diff_network.get_regularization_loss(target)
    assert loss.item() >= 0.0


def test_diff_network_get_regularization_loss_zero_when_same(state_dim, action_dim):
    source = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim, initial_temperature=1.0)
    target = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim, initial_temperature=1.0)
    target.load_state_dict(source.state_dict())
    loss = source.get_regularization_loss(target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_diff_network_gradients_flow(diff_network, sample_batch):
    x = sample_batch.requires_grad_(True)
    output = diff_network(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == sample_batch.shape
    assert not torch.isnan(x.grad).any()


def test_non_diff_attention_layer_internal_shapes(non_diff_attention_layer, sample_batch):
    q1 = non_diff_attention_layer.linear_q1(sample_batch).view(4, 8, 3)
    k1 = non_diff_attention_layer.linear_k1(sample_batch).view(4, 8, 3)
    attention = torch.bmm(q1, k1.transpose(1, 2))
    assert attention.shape == (4, 8, 8)

    idx = attention.argmax(dim=-1)
    assert idx.shape == (4, 8)

    q2 = non_diff_attention_layer.linear_q2(sample_batch).view(4, 8, 3)
    k2 = non_diff_attention_layer.linear_k2(sample_batch).view(4, 3, 3)
    v = torch.bmm(q2, k2.transpose(1, 2))
    assert v.shape == (4, 8, 3)


def test_diff_attention_layer_softmax_output(diff_attention_layer, sample_batch):
    output = diff_attention_layer(sample_batch)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_diff_attention_layer_temperature_effect(diff_attention_layer, sample_batch):
    diff_attention_layer.set_temperature(1.0)
    output_high_temp = diff_attention_layer(sample_batch).clone()

    diff_attention_layer.set_temperature(0.1)
    output_low_temp = diff_attention_layer(sample_batch).clone()

    assert not torch.allclose(output_high_temp, output_low_temp, atol=1e-3)


def test_non_diff_network_output_logits_range(non_diff_network, sample_batch):
    output = non_diff_network(sample_batch)
    assert output.shape == (4, 4)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_diff_network_output_logits_range(diff_network, sample_batch):
    output = diff_network(sample_batch)
    assert output.shape == (4, 4)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_non_diff_network_eval_mode(non_diff_network, sample_state):
    non_diff_network.eval()
    x = torch.from_numpy(sample_state).float()

    with torch.no_grad():
        output1 = non_diff_network(x)
        output2 = non_diff_network(x)

    assert torch.allclose(output1, output2)


def test_diff_network_eval_mode(diff_network, sample_state):
    diff_network.eval()
    x = torch.from_numpy(sample_state).float()

    with torch.no_grad():
        output1 = diff_network(x)
        output2 = diff_network(x)

    assert torch.allclose(output1, output2)


def test_non_diff_attention_layer_dtype_preservation(non_diff_attention_layer, sample_batch):
    assert sample_batch.dtype == torch.float32
    output = non_diff_attention_layer(sample_batch)
    assert output.dtype == torch.float32


def test_diff_attention_layer_dtype_preservation(diff_attention_layer, sample_batch):
    assert sample_batch.dtype == torch.float32
    output = diff_attention_layer(sample_batch)
    assert output.dtype == torch.float32


def test_non_diff_network_layer2_projection(non_diff_network, state_dim):
    x = torch.randn(2, state_dim)
    layer1_out = non_diff_network.layer1(x)
    assert layer1_out.shape == (2, state_dim * 3)

    layer2_out = non_diff_network.layer2(layer1_out)
    assert layer2_out.shape == (2, (state_dim * 3) * 3)

    proj_out = non_diff_network.layer2_proj(layer2_out)
    assert proj_out.shape == (2, 6)


def test_diff_network_layer2_projection(diff_network, state_dim):
    x = torch.randn(2, state_dim)
    layer1_out = diff_network.layer1(x)
    assert layer1_out.shape == (2, state_dim * 3)

    layer2_out = diff_network.layer2(layer1_out)
    assert layer2_out.shape == (2, (state_dim * 3) * 3)

    proj_out = diff_network.layer2_proj(layer2_out)
    assert proj_out.shape == (2, 6)


def test_non_diff_network_custom_dims():
    network = NonDifferentiableNetwork(state_dim=4, action_dim=2)
    x = torch.randn(3, 4)
    output = network(x)
    assert output.shape == (3, 2)


def test_diff_network_custom_dims():
    network = DifferentiableNetwork(state_dim=4, action_dim=2, initial_temperature=0.5)
    x = torch.randn(3, 4)
    output = network(x)
    assert output.shape == (3, 2)
    assert network.temperature == 0.5


def test_diff_network_anneal_multiple_times(diff_network):
    initial_temp = diff_network.temperature
    for _ in range(10):
        new_temp = diff_network.anneal_temperature(decay_rate=0.99, min_temperature=0.01)

    assert new_temp < initial_temp
    assert new_temp >= 0.01


def test_diff_network_get_regularization_loss_scales_with_difference(diff_network, state_dim, action_dim):
    target = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim, initial_temperature=1.0)

    loss_same = diff_network.get_regularization_loss(diff_network)
    loss_diff = diff_network.get_regularization_loss(target)

    assert loss_same.item() == pytest.approx(0.0, abs=1e-6)
    assert loss_diff.item() > 0
