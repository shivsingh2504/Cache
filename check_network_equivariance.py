
from __future__ import annotations

import torch

from RL.config import NetworkConfig
from RL.network import QNetwork


def check_shape_contract(network: QNetwork, config: NetworkConfig) -> None:
    batch_size = 4
    state = torch.randn(batch_size, config.cache_capacity, config.num_state_features)
    q_values = network(state)
    assert q_values.shape == (batch_size, config.cache_capacity), (
        f"expected shape ({batch_size}, {config.cache_capacity}), got {tuple(q_values.shape)}"
    )
    print(f"[Check 1/5] shape contract: PASS  (input {tuple(state.shape)} -> output {tuple(q_values.shape)})")


def check_single_batch_inference(network: QNetwork, config: NetworkConfig) -> None:
    state = torch.randn(1, config.cache_capacity, config.num_state_features)
    with torch.no_grad():
        q_values = network(state)
    assert q_values.shape == (1, config.cache_capacity)
    assert torch.isfinite(q_values).all(), "non-finite Q-values on single-batch inference"
    print(f"[Check 2/5] single-batch inference: PASS  (output {tuple(q_values.shape)}, all finite)")


def check_identical_slots_identical_q(network: QNetwork, config: NetworkConfig) -> None:
    torch.manual_seed(0)
    state = torch.randn(1, config.cache_capacity, config.num_state_features)
    # Force every slot to share the exact same feature vector.
    probe = state[0, 0].clone()
    for slot in range(config.cache_capacity):
        state[0, slot] = probe

    network.eval()
    with torch.no_grad():
        q_values = network(state)

    spread = (q_values.max() - q_values.min()).item()
    assert spread < 1e-6, (
        f"identical per-slot inputs produced non-identical Q-values, spread={spread}"
    )
    print(
        f"[Check 3/5] identical-slot-features -> identical Q-values: PASS  "
        f"(spread={spread:.2e} across {config.cache_capacity} identical slots)"
    )


def check_permutation_equivariance(network: QNetwork, config: NetworkConfig) -> None:
    torch.manual_seed(1)
    state = torch.randn(2, config.cache_capacity, config.num_state_features)

    permutation = torch.randperm(config.cache_capacity)
    permuted_state = state[:, permutation, :]

    network.eval()
    with torch.no_grad():
        q_values = network(state)
        q_values_permuted_input = network(permuted_state)

    expected = q_values[:, permutation]
    max_abs_diff = (q_values_permuted_input - expected).abs().max().item()
    assert max_abs_diff < 1e-5, (
        f"permutation equivariance violated, max abs diff={max_abs_diff}"
    )
    print(
        f"[Check 4/5] permutation equivariance: PASS  "
        f"(max abs diff={max_abs_diff:.2e} after permuting {config.cache_capacity} slots)"
    )


def check_no_slot_specific_parameters(network: QNetwork, config: NetworkConfig) -> None:
    offending = []
    for name, param in network.named_parameters():
        if config.cache_capacity in param.shape:
            offending.append((name, tuple(param.shape)))
    assert not offending, (
        f"found parameter(s) whose shape depends on cache_capacity "
        f"({config.cache_capacity}), implying per-slot weights: {offending}"
    )
    total_params = sum(p.numel() for p in network.parameters())
    print(
        f"[Check 5/5] no slot-specific parameters: PASS  "
        f"({sum(1 for _ in network.parameters())} parameter tensors, "
        f"none shaped by cache_capacity, {total_params:,} total params)"
    )


def main() -> None:
    config = NetworkConfig()
    network = QNetwork(config)

    total_params = sum(p.numel() for p in network.parameters())
    print(f"[Info] NetworkConfig: cache_capacity={config.cache_capacity} "
          f"num_state_features={config.num_state_features} "
          f"hidden_sizes={config.hidden_sizes}")
    print(f"[Info] total parameters: {total_params:,}\n")

    check_shape_contract(network, config)
    check_single_batch_inference(network, config)
    check_identical_slots_identical_q(network, config)
    check_permutation_equivariance(network, config)
    check_no_slot_specific_parameters(network, config)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()