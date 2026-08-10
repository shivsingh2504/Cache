
from __future__ import annotations

import argparse

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

_COL_NAMES = ("frequency", "recency", "key_age", "predicted_popularity", "is_empty")


def _validate_state(state: np.ndarray, decision_index: int, cache_capacity: int, num_features: int) -> None:
    """Hard check. Raises loudly instead of letting a malformed state
    propagate into a printout that looks plausible but isn't."""
    problems = []
    if state.shape != (cache_capacity, num_features):
        problems.append(
            f"shape mismatch: got {state.shape}, expected {(cache_capacity, num_features)}"
        )
    if state.dtype != np.float32:
        problems.append(f"dtype mismatch: got {state.dtype}, expected float32")
    if not np.all(np.isfinite(state)):
        bad = np.argwhere(~np.isfinite(state))
        problems.append(f"non-finite values at rows/cols: {bad.tolist()}")

    if problems:
        print(f"\n!!! STATE VALIDATION FAILED at decision {decision_index} !!!")
        for p in problems:
            print(f"    - {p}")
        print(f"    raw state repr:\n{repr(state)}\n")
        raise AssertionError(
            f"Malformed state at decision {decision_index}; see printout above. "
            f"Refusing to proceed with unreliable data."
        )


def _load_network(checkpoint_path: str, network_config: NetworkConfig) -> QNetwork:
    network = QNetwork(network_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    network.load_state_dict(checkpoint["online_state_dict"])
    network.eval()
    print(f"[Checkpoint] loaded {checkpoint_path}")
    print(f"[Checkpoint] NetworkConfig used for loading: {network_config}")
    if "step" in checkpoint:
        print(f"[Checkpoint] recorded step={checkpoint.get('step')}")
    if "hit_rate" in checkpoint:
        print(f"[Checkpoint] recorded hit_rate={checkpoint.get('hit_rate')}")
    return network


def _q_values(network: QNetwork, state: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    with torch.no_grad():
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        print(
            f"    tensor passed to network: shape={tuple(state_tensor.shape)} "
            f"dtype={state_tensor.dtype}"
        )
        q_values = network(state_tensor)
    return q_values.squeeze(0).numpy(), state_tensor


def _lfu_diagnostics(state: np.ndarray) -> dict:
    frequency = np.where(
        state[:, _IS_EMPTY_COL] > 0.5,
        np.inf,
        state[:, _FREQUENCY_COL],
    )
    min_freq = float(np.min(frequency))
    tied_candidates = [i for i, f in enumerate(frequency) if f == min_freq]
    selected = int(np.argmin(frequency))
    return {
        "frequency_per_slot": frequency,
        "min_freq": min_freq,
        "tied_candidates": tied_candidates,
        "selected": selected,
    }


def print_decision_table(
    decision_index: int,
    state: np.ndarray,
    slot_keys: list,
    q_values: np.ndarray,
    lfu_action: int,
    net_action: int,
    lfu_diag: dict,
    top_k: int = 5,
) -> None:
    print(f"\nDecision {decision_index}")
    print("-" * 11)
    header = f"{'slot':<5}{'key':<8}"
    for name in _COL_NAMES:
        header += f"{name:<22}"
    header += f"{'Q-value':<12}"
    print(header)

    for slot in range(state.shape[0]):
        row = f"{slot:<5}{str(slot_keys[slot]):<8}"
        for col in range(state.shape[1]):
            row += f"{state[slot, col]:<22.7f}"
        row += f"{q_values[slot]:<12.7f}"
        print(row)

    print(f"\nLFU action: {lfu_action}")
    print(f"Network action: {net_action}")

    ranked = sorted(range(len(q_values)), key=lambda i: q_values[i], reverse=True)
    print(f"\nTop {top_k} network actions:")
    for rank, slot in enumerate(ranked[:top_k], start=1):
        print(f"  {rank}. slot {slot}  Q={q_values[slot]:.7f}  key={slot_keys[slot]}")

    print(f"\nLFU frequency minimum (log1p-normalized): {lfu_diag['min_freq']:.7f}")
    print(f"LFU tie candidates (slots at minimum): {lfu_diag['tied_candidates']}")
    print(f"LFU selected slot: {lfu_diag['selected']}  (first index among tied candidates)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_lfu_imitation/imitation.pt")
    parser.add_argument("--lstm-checkpoint", default="lstm_popularity_predictor.pt")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--num-requests", type=int, default=5_000)
    parser.add_argument("--num-keys", type=int, default=5_000)
    parser.add_argument(
        "--inspect-until",
        type=int,
        default=5,
        help="Print the full per-slot table for decisions 0..this index inclusive.",
    )
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
    workload = WorkloadGenerator(workload_config)
    cache_state = CacheState()
    predictor = Predictor.from_checkpoint(
        checkpoint_path=args.lstm_checkpoint,
        model_config=model_config,
        state=cache_state,
    )
    environment = CacheEvictionEnvironment(
        config=environment_config,
        workload_generator=workload,
        predictor=predictor,
        cache_state=cache_state,
        normalizer=normalizer,
    )

    network = _load_network(args.checkpoint, network_config)
    lfu = LFUPolicy()

    print(
        f"\n[Diagnostic] seed={args.seed} num_requests={args.num_requests} "
        f"num_keys={args.num_keys} inspect_until={args.inspect_until}"
    )
    print(f"[Diagnostic] EnvironmentConfig: {environment_config}")

    state = environment.reset()
    decision_index = 0
    first_mismatch_reported = False

    while not environment.done and decision_index <= args.inspect_until:
        _validate_state(
            state,
            decision_index,
            environment_config.cache_capacity,
            environment_config.num_state_features,
        )

        # environment.py has no public accessor for _slot_keys; this reads
        # the private attribute for diagnostic purposes ONLY (read-only,
        # never mutated here). If a raw_state()/resident_keys() accessor
        # gets added later (as noted in prior project memory), swap this
        # for that instead.
        slot_keys = list(environment._slot_keys)  # noqa: SLF001

        lfu_diag = _lfu_diagnostics(state)
        lfu_action = lfu_diag["selected"]

        print(f"\n[Decision {decision_index}] computing Q-values...")
        q_values, state_tensor = _q_values(network, state)
        net_action = int(np.argmax(q_values))

        print_decision_table(
            decision_index, state, slot_keys, q_values, lfu_action, net_action, lfu_diag
        )

        if lfu_action != net_action and not first_mismatch_reported:
            print(f"\n>>> FIRST MISMATCH at decision {decision_index} <<<")
            first_mismatch_reported = True

        result = environment.step(lfu_action)  # LFU drives the trajectory (open loop)
        state = result.next_state
        decision_index += 1


if __name__ == "__main__":
    main()