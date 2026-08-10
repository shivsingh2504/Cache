
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from simulator.config import DistributionType, WorkloadConfig
from simulator.generator import WorkloadGenerator

from feature.state import CacheState

from lstm.config import ModelConfig
from lstm.predictor import Predictor

from RL.config import EnvironmentConfig
from RL.environment import CacheEvictionEnvironment
from RL.normalizer import StateNormalizer
from RL.baseline import LFUPolicy, _FREQUENCY_COL, _IS_EMPTY_COL
from RL.frequency_only import mask_frequency_only


@dataclass
class DecisionRecord:
    index: int
    lfu_action: int
    trivial_action: int
    agree: bool


def _trivial_argmin_frequency(state: np.ndarray) -> int:
    """Identical logic to LFUPolicy.select_action, reimplemented
    independently (not by calling LFUPolicy) so this is a genuinely
    separate check, not the same code path compared to itself."""
    frequency = np.where(
        state[:, _IS_EMPTY_COL] > 0.5,
        np.inf,
        state[:, _FREQUENCY_COL],
    )
    return int(np.argmin(frequency))


def _lfu_action(state: np.ndarray) -> int:
    lfu = LFUPolicy()
    return lfu.select_action(state)


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


def run_open_loop(environment: CacheEvictionEnvironment, use_mask: bool) -> list[DecisionRecord]:
    """LFU drives the trajectory (identical to the real diagnostics).
    `use_mask` mirrors the frequency-only harness path -- masking must
    not change the trivial policy's answer, since it only zeros columns
    the trivial policy never reads."""
    records: list[DecisionRecord] = []
    state = environment.reset()
    idx = 0
    while not environment.done:
        lfu_action = _lfu_action(state)
        probe_state = mask_frequency_only(state) if use_mask else state
        trivial_action = _trivial_argmin_frequency(probe_state)
        records.append(
            DecisionRecord(index=idx, lfu_action=lfu_action, trivial_action=trivial_action, agree=(lfu_action == trivial_action))
        )
        idx += 1
        result = environment.step(lfu_action)
        state = result.next_state
    return records


def run_closed_loop(environment: CacheEvictionEnvironment, use_mask: bool) -> list[DecisionRecord]:
    """The trivial policy drives the trajectory itself -- exact same
    control-flow shape as run_closed_loop() in the real diagnostics,
    just with the trivial function standing in for a trained network."""
    records: list[DecisionRecord] = []
    state = environment.reset()
    idx = 0
    while not environment.done:
        lfu_action = _lfu_action(state)
        probe_state = mask_frequency_only(state) if use_mask else state
        trivial_action = _trivial_argmin_frequency(probe_state)
        records.append(
            DecisionRecord(index=idx, lfu_action=lfu_action, trivial_action=trivial_action, agree=(lfu_action == trivial_action))
        )
        idx += 1
        result = environment.step(trivial_action)
        state = result.next_state
    return records


def summarize(name: str, records: list[DecisionRecord]) -> None:
    n = len(records)
    agreements = sum(r.agree for r in records)
    rate = agreements / n if n > 0 else 0.0
    print(f"=== {name} ===")
    print(f"total decisions: {n:,}")
    print(f"agreement count: {agreements:,}")
    print(f"agreement rate: {rate:.4%}")
    if agreements != n:
        mismatches = [r.index for r in records if not r.agree][:10]
        print(f"!!! MISMATCHES FOUND at indices (first 10): {mismatches}")
        print("!!! This means the HARNESS has a bug -- investigate before trusting")
        print("!!! any network-based conclusion from the other diagnostic scripts.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm-checkpoint", default="lstm_popularity_predictor.pt")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--num-requests", type=int, default=5_000)
    parser.add_argument("--num-keys", type=int, default=5_000)
    args = parser.parse_args()

    environment_config = EnvironmentConfig()
    model_config = ModelConfig(event_features=8, candidate_features=3)
    normalizer = StateNormalizer()

    workload_config = WorkloadConfig(
        num_keys=args.num_keys,
        num_requests=args.num_requests,
        distribution=DistributionType.ZIPFIAN,
        seed=args.seed,
    )

    print(f"[Sanity check] seed={args.seed} num_requests={args.num_requests} num_keys={args.num_keys}\n")

    # Unmasked variant: mirrors imitation.pt's harness shape.
    env_a = _build_environment(environment_config, workload_config, model_config, args.lstm_checkpoint, normalizer)
    env_b = _build_environment(environment_config, workload_config, model_config, args.lstm_checkpoint, normalizer)
    summarize("UNMASKED open loop (trivial argmin(frequency) vs LFU)", run_open_loop(env_a, use_mask=False))
    summarize("UNMASKED closed loop (trivial argmin(frequency) vs LFU)", run_closed_loop(env_b, use_mask=False))

    # Masked variant: mirrors frequency_only.pt's harness shape exactly,
    # including mask_frequency_only() in the inference path.
    env_c = _build_environment(environment_config, workload_config, model_config, args.lstm_checkpoint, normalizer)
    env_d = _build_environment(environment_config, workload_config, model_config, args.lstm_checkpoint, normalizer)
    summarize("MASKED open loop (trivial argmin(frequency) vs LFU)", run_open_loop(env_c, use_mask=True))
    summarize("MASKED closed loop (trivial argmin(frequency) vs LFU)", run_closed_loop(env_d, use_mask=True))

    print("Expected result: 100.0000% agreement in all four blocks above.")
    print("If any block is below 100%, stop -- the harness itself needs fixing")
    print("before any conclusion about network architecture can be trusted.")


if __name__ == "__main__":
    main()