
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from simulator.config import DistributionType, WorkloadConfig
from simulator.generator import WorkloadGenerator

from feature.state import CacheState

from lstm.config import ModelConfig
from lstm.predictor import Predictor

from RL.config import EnvironmentConfig, NetworkConfig
from RL.environment import CacheEvictionEnvironment
from RL.normalizer import StateNormalizer
from RL.network import QNetwork
from RL.baseline import LFUPolicy, _FREQUENCY_COL, _IS_EMPTY_COL
from RL.frequency_only import mask_frequency_only

# Hardcoded from the prior confirmed run on this exact seed=43,
# num_requests=5000, num_keys=5000 trace, imitation.pt (5-feature model).
# NOT recomputed here -- printed for direct side-by-side comparison only.
_ORIGINAL_OPEN_LOOP_AGREEMENT = 0.261012
_ORIGINAL_CLOSED_LOOP_AGREEMENT = 0.091670


@dataclass
class DecisionRecord:
    index: int
    raw_state: np.ndarray
    masked_state: np.ndarray
    lfu_action: int
    net_action: int
    q_values: np.ndarray
    agree: bool


def _load_network(checkpoint_path: str, network_config: NetworkConfig) -> QNetwork:
    network = QNetwork(network_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    network.load_state_dict(checkpoint["online_state_dict"])
    network.eval()
    return network


def _validate_masked_tensor(masked_state: np.ndarray, raw_state: np.ndarray) -> None:
    """Hard checks that masking actually happened correctly at the
    inference boundary, every single decision -- not assumed once."""
    assert masked_state.shape == (16, 5), f"masked shape {masked_state.shape} != (16, 5)"
    assert masked_state.dtype == np.float32, f"masked dtype {masked_state.dtype} != float32"
    assert np.array_equal(masked_state[:, 0], raw_state[:, 0]), "column 0 (frequency) altered by mask"
    assert np.array_equal(masked_state[:, 4], raw_state[:, 4]), "column 4 (is_empty) altered by mask"
    assert np.all(masked_state[:, 1] == 0.0), "column 1 (recency) not zeroed"
    assert np.all(masked_state[:, 2] == 0.0), "column 2 (key_age) not zeroed"
    assert np.all(masked_state[:, 3] == 0.0), "column 3 (predicted_popularity) not zeroed"


def _freq_only_net_action(network: QNetwork, raw_state: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """The actual inference path: mask happens HERE, immediately before
    the tensor is built, mirroring FrequencyOnlyGreedyAgentPolicy's
    select_action -- not merely illustrated after the fact."""
    masked_state = mask_frequency_only(raw_state)
    _validate_masked_tensor(masked_state, raw_state)

    with torch.no_grad():
        state_tensor = torch.from_numpy(masked_state).float().unsqueeze(0)
        assert state_tensor.shape == (1, 16, 5), f"tensor shape {tuple(state_tensor.shape)} != (1, 16, 5)"
        assert state_tensor.dtype == torch.float32
        q_values = network(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())

    return action, masked_state, q_values.squeeze(0).numpy()


def _lfu_action(state: np.ndarray) -> int:
    frequency = np.where(
        state[:, _IS_EMPTY_COL] > 0.5,
        np.inf,
        state[:, _FREQUENCY_COL],
    )
    return int(np.argmin(frequency))


def _build_environment(
    environment_config: EnvironmentConfig,
    workload_config: WorkloadConfig,
    model_config: ModelConfig,
    lstm_checkpoint: str,
    normalizer: StateNormalizer,
) -> CacheEvictionEnvironment:
    workload = WorkloadGenerator(workload_config)
    cache_state = CacheState()
    predictor = Predictor.from_checkpoint(
        checkpoint_path=lstm_checkpoint,
        model_config=model_config,
        state=cache_state,
    )
    return CacheEvictionEnvironment(
        config=environment_config,
        workload_generator=workload,
        predictor=predictor,
        cache_state=cache_state,
        normalizer=normalizer,
    )


def run_open_loop(environment: CacheEvictionEnvironment, network: QNetwork) -> list[DecisionRecord]:
    """LFU drives the trajectory. Frequency-only network only observes."""
    records: list[DecisionRecord] = []
    state = environment.reset()
    idx = 0
    while not environment.done:
        lfu_action = _lfu_action(state)
        net_action, masked_state, q_values = _freq_only_net_action(network, state)
        records.append(
            DecisionRecord(
                index=idx,
                raw_state=state.copy(),
                masked_state=masked_state,
                lfu_action=lfu_action,
                net_action=net_action,
                q_values=q_values,
                agree=(lfu_action == net_action),
            )
        )
        idx += 1
        result = environment.step(lfu_action)  # LFU drives
        state = result.next_state
    return records


def run_closed_loop(environment: CacheEvictionEnvironment, network: QNetwork) -> list[DecisionRecord]:
    """Frequency-only network drives the trajectory via masked inference.
    LFU action computed at each step for comparison ONLY -- never
    executed, never affects the rollout."""
    records: list[DecisionRecord] = []
    state = environment.reset()
    idx = 0
    while not environment.done:
        lfu_action = _lfu_action(state)
        net_action, masked_state, q_values = _freq_only_net_action(network, state)
        records.append(
            DecisionRecord(
                index=idx,
                raw_state=state.copy(),
                masked_state=masked_state,
                lfu_action=lfu_action,
                net_action=net_action,
                q_values=q_values,
                agree=(lfu_action == net_action),
            )
        )
        idx += 1
        result = environment.step(net_action)  # frequency-only network drives
        state = result.next_state
    return records


def print_first_mismatch(name: str, records: list[DecisionRecord], top_k: int = 5) -> None:
    first = next((r for r in records if not r.agree), None)
    if first is None:
        print(f"[{name}] no mismatches found.\n")
        return

    print(f"[{name}] FIRST MISMATCH at decision {first.index}")
    print(f"  LFU action: {first.lfu_action}")
    print(f"  Network action: {first.net_action}\n")

    print(f"  {'slot':<5}{'raw_freq':<22}{'masked_freq':<22}{'is_empty':<10}{'Q-value':<14}")
    for slot in range(16):
        print(
            f"  {slot:<5}"
            f"{first.raw_state[slot, 0]:<22.7f}"
            f"{first.masked_state[slot, 0]:<22.7f}"
            f"{first.masked_state[slot, 4]:<10.1f}"
            f"{first.q_values[slot]:<14.7f}"
        )

    ranked = sorted(range(16), key=lambda i: first.q_values[i], reverse=True)
    print(f"\n  Top {top_k} network actions:")
    for rank, slot in enumerate(ranked[:top_k], start=1):
        print(f"    {rank}. slot {slot}  Q={first.q_values[slot]:.7f}")

    tied = [
        slot for slot in range(16)
        if first.masked_state[slot, 4] < 0.5
        and first.masked_state[slot, 0] == np.min(
            np.where(first.masked_state[:, 4] > 0.5, np.inf, first.masked_state[:, 0])
        )
    ]
    if len(tied) > 1:
        tied_q = [first.q_values[s] for s in tied]
        print(
            f"\n  Slots tied at minimum masked frequency: {tied}\n"
            f"  Their Q-values: {[round(q, 4) for q in tied_q]}\n"
            f"  Q-value spread among tied slots: {max(tied_q) - min(tied_q):.7f}"
        )
    print()


def summarize(name: str, records: list[DecisionRecord]) -> float:
    n = len(records)
    agreements = sum(r.agree for r in records)
    rate = agreements / n if n > 0 else 0.0
    print(f"=== {name} ===")
    print(f"total decisions: {n:,}")
    print(f"agreement count: {agreements:,}")
    print(f"agreement rate: {rate:.4%}")
    first_mismatch = next((r.index for r in records if not r.agree), None)
    print(f"first mismatch index: {first_mismatch if first_mismatch is not None else 'none'}")
    print()
    return rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imitation-checkpoint", default="checkpoints_lfu_imitation/imitation.pt")
    parser.add_argument("--freq-only-checkpoint", default="checkpoints_lfu_imitation/frequency_only.pt")
    parser.add_argument("--lstm-checkpoint", default="lstm_popularity_predictor.pt")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--num-requests", type=int, default=5_000)
    parser.add_argument("--num-keys", type=int, default=5_000)
    args = parser.parse_args()

    environment_config = EnvironmentConfig()
    network_config = NetworkConfig()
    model_config = ModelConfig(event_features=8, candidate_features=3)
    normalizer = StateNormalizer()

    workload_config = WorkloadConfig(
        num_keys=args.num_keys,
        num_requests=args.num_requests,
        distribution=DistributionType.ZIPFIAN,
        seed=args.seed,
    )

    # Loaded to confirm existence / architecture compatibility, and to
    # make explicit that this diagnostic reads imitation.pt read-only for
    # reference context only -- it is never used to drive any rollout
    # here (its own open/closed-loop numbers were already established
    # and are only reprinted, not recomputed, per the fixed constants
    # above).
    _ = _load_network(args.imitation_checkpoint, network_config)
    freq_only_network = _load_network(args.freq_only_checkpoint, network_config)

    print(f"[Diagnostic] seed={args.seed} num_requests={args.num_requests} num_keys={args.num_keys}")
    print(f"[Diagnostic] imitation checkpoint (reference only): {args.imitation_checkpoint}")
    print(f"[Diagnostic] frequency-only checkpoint under test: {args.freq_only_checkpoint}\n")

    open_loop_env = _build_environment(
        environment_config, workload_config, model_config, args.lstm_checkpoint, normalizer
    )
    closed_loop_env = _build_environment(
        environment_config, workload_config, model_config, args.lstm_checkpoint, normalizer
    )

    open_records = run_open_loop(open_loop_env, freq_only_network)
    open_rate = summarize("TEST 1: OPEN LOOP (frequency-only, seed=43)", open_records)
    print_first_mismatch("TEST 1 OPEN LOOP", open_records)

    closed_records = run_closed_loop(closed_loop_env, freq_only_network)
    closed_rate = summarize("TEST 2: CLOSED LOOP (frequency-only, seed=43)", closed_records)
    print_first_mismatch("TEST 2 CLOSED LOOP", closed_records)

    print("=== Direct comparison (seed=43, num_requests=5000, num_keys=5000) ===")
    print(f"{'':<28}{'Original 5-feature':>22}{'Frequency-only':>18}")
    print(
        f"{'Open-loop agreement':<28}"
        f"{_ORIGINAL_OPEN_LOOP_AGREEMENT:>21.4%} "
        f"{open_rate:>17.4%}"
    )
    print(
        f"{'Closed-loop agreement':<28}"
        f"{_ORIGINAL_CLOSED_LOOP_AGREEMENT:>21.4%} "
        f"{closed_rate:>17.4%}"
    )


if __name__ == "__main__":
    main()