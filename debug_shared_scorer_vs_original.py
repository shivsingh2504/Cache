"""Diagnostic for the shared-scorer frequency-only LFU imitation ablation.

A new script rather than a reuse of debug_freq_only_vs_original.py /
debug_policy_equivalence.py because those are written against QNetwork's
(16, 5) flattened input and mask_frequency_only's 5-column output --
incompatible with SharedScorerNetwork's (16, 2) per-slot input. This
script mirrors their methodology (same open-loop / closed-loop
construction, same seed=43/num_requests=5000/num_keys=5000 evaluation
trace) but adds the tied-vs-unique-minimum-frequency breakdown the
shared-scorer experiment specifically needs: the central question is
whether the shared scorer, unlike the flattened MLP, gives IDENTICAL
scores to slots with identical [frequency, is_empty] inputs (as a
correctly-implemented shared scorer must, by construction), and whether
that identical-scoring behavior is what recovers LFU's tie-breaking
behavior.
"""
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

from RL.config import EnvironmentConfig
from RL.environment import CacheEvictionEnvironment
from RL.normalizer import StateNormalizer
from RL.baseline import _FREQUENCY_COL, _IS_EMPTY_COL
from RL.shared_scorer_network import SharedScorerNetwork, to_shared_scorer_input


@dataclass
class DecisionRecord:
    index: int
    raw_state: np.ndarray
    reduced_state: np.ndarray
    lfu_action: int
    net_action: int
    scores: np.ndarray
    agree: bool
    is_tied: bool
    tied_score_spread: float


def _load_network(checkpoint_path: str) -> SharedScorerNetwork:
    network = SharedScorerNetwork()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    network.load_state_dict(checkpoint["online_state_dict"])
    network.eval()
    return network


def _lfu_action(state: np.ndarray) -> int:
    """Reimplemented independently (not by calling LFUPolicy) so this is
    a genuinely separate check, matching the convention already used in
    debug_harness_sanity_check.py."""
    frequency = np.where(
        state[:, _IS_EMPTY_COL] > 0.5,
        np.inf,
        state[:, _FREQUENCY_COL],
    )
    return int(np.argmin(frequency))


def _resident_min_frequency_slots(state: np.ndarray) -> list[int]:
    """Slots occupied (is_empty <= 0.5) whose frequency equals the
    minimum frequency among occupied slots -- i.e. the slots LFU is
    choosing among."""
    is_empty = state[:, _IS_EMPTY_COL] > 0.5
    frequency = np.where(is_empty, np.inf, state[:, _FREQUENCY_COL])
    min_freq = np.min(frequency)
    return [i for i in range(state.shape[0]) if not is_empty[i] and frequency[i] == min_freq]


def _net_action_and_scores(
    network: SharedScorerNetwork, raw_state: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    """Inference path: reduction happens HERE, immediately before the
    tensor is built, mirroring SharedScorerNetwork.select_action."""
    reduced_state = to_shared_scorer_input(raw_state)  # (16, 2)
    assert reduced_state.shape == (16, 2), f"reduced shape {reduced_state.shape} != (16, 2)"
    assert np.array_equal(reduced_state[:, 0], raw_state[:, 0]), "frequency column altered by reduction"
    assert np.array_equal(reduced_state[:, 1], raw_state[:, 4]), "is_empty column altered by reduction"

    with torch.no_grad():
        state_tensor = torch.from_numpy(reduced_state).float().unsqueeze(0)
        scores = network(state_tensor)
        action = int(torch.argmax(scores, dim=1).item())

    return action, reduced_state, scores.squeeze(0).numpy()


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


def _run(
    environment: CacheEvictionEnvironment,
    network: SharedScorerNetwork,
    driver: str,
) -> list[DecisionRecord]:
    """driver='lfu' -> open loop (LFU drives, network only observes).
    driver='net' -> closed loop (network drives; LFU action computed for
    comparison only, never executed)."""
    assert driver in ("lfu", "net")
    records: list[DecisionRecord] = []
    state = environment.reset()
    idx = 0
    while not environment.done:
        lfu_action = _lfu_action(state)
        net_action, reduced_state, scores = _net_action_and_scores(network, state)

        tied_slots = _resident_min_frequency_slots(state)
        is_tied = len(tied_slots) > 1
        tied_score_spread = (
            float(np.max(scores[tied_slots]) - np.min(scores[tied_slots]))
            if is_tied
            else 0.0
        )

        records.append(
            DecisionRecord(
                index=idx,
                raw_state=state.copy(),
                reduced_state=reduced_state,
                lfu_action=lfu_action,
                net_action=net_action,
                scores=scores,
                agree=(lfu_action == net_action),
                is_tied=is_tied,
                tied_score_spread=tied_score_spread,
            )
        )
        idx += 1
        drive_action = lfu_action if driver == "lfu" else net_action
        result = environment.step(drive_action)
        state = result.next_state
    return records


def _agreement_rate(records: list[DecisionRecord]) -> float:
    n = len(records)
    return sum(r.agree for r in records) / n if n > 0 else 0.0


def summarize(name: str, records: list[DecisionRecord]) -> None:
    n = len(records)
    agreements = sum(r.agree for r in records)
    rate = agreements / n if n > 0 else 0.0

    tied = [r for r in records if r.is_tied]
    unique = [r for r in records if not r.is_tied]

    print(f"=== {name} ===")
    print(f"total decisions: {n:,}")
    print(f"agreement count: {agreements:,}")
    print(f"agreement rate: {rate:.4%}")
    print(f"unique-minimum-frequency decisions: {len(unique):,}  agreement: {_agreement_rate(unique):.4%}")
    print(f"tied-minimum-frequency decisions:   {len(tied):,}  agreement: {_agreement_rate(tied):.4%}")

    if tied:
        spreads = np.array([r.tied_score_spread for r in tied])
        print(
            f"score spread among tied-minimum slots -- "
            f"mean={spreads.mean():.8f} max={spreads.max():.8f}"
        )
        non_zero = int(np.sum(spreads > 1e-6))
        if non_zero:
            print(
                f"!!! {non_zero:,}/{len(tied):,} tied-decisions have NON-ZERO score "
                f"spread among identically-featured slots -- this should be "
                f"impossible for a correctly-implemented shared scorer and "
                f"indicates a bug (e.g. reduction not applied per-slot "
                f"consistently), not a modeling result."
            )
        else:
            print(
                "All tied-minimum slots score identically (spread ~0), as required "
                "by construction for a true shared scorer."
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_lfu_imitation/shared_scorer_freq_only.pt")
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

    print(f"[Diagnostic] seed={args.seed} num_requests={args.num_requests} num_keys={args.num_keys}")
    print(f"[Diagnostic] shared-scorer checkpoint under test: {args.checkpoint}\n")

    network = _load_network(args.checkpoint)

    open_loop_env = _build_environment(
        environment_config, workload_config, model_config, args.lstm_checkpoint, normalizer
    )
    closed_loop_env = _build_environment(
        environment_config, workload_config, model_config, args.lstm_checkpoint, normalizer
    )

    open_records = _run(open_loop_env, network, driver="lfu")
    summarize("TEST 1: OPEN LOOP (shared scorer, seed=43)", open_records)

    closed_records = _run(closed_loop_env, network, driver="net")
    summarize("TEST 2: CLOSED LOOP (shared scorer, seed=43)", closed_records)

    print("=== Verdict ===")
    print(f"open-loop agreement:   {_agreement_rate(open_records):.4%}")
    print(f"closed-loop agreement: {_agreement_rate(closed_records):.4%}")
    print(
        "Compare these against the flattened frequency-only ablation's "
        "0.0000% open-loop / 7.9985% closed-loop, and the original "
        "5-feature imitation's 26.1012% open-loop / 9.1670% closed-loop, "
        "to judge whether weight-sharing across slots recovered "
        "generalization of the frequency rule -- do not conclude success "
        "from training accuracy alone."
    )


if __name__ == "__main__":
    main()