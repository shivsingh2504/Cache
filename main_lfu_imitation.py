from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from simulator.config import WorkloadConfig, DistributionType
from simulator.generator import WorkloadGenerator

from feature.state import CacheState

from lstm.config import ModelConfig
from lstm.predictor import Predictor

from RL.config import EnvironmentConfig, NetworkConfig, ReplayBufferConfig, AgentConfig
from RL.environment import CacheEvictionEnvironment
from RL.normalizer import StateNormalizer
from RL.network import QNetwork
from RL.baseline import LFUPolicy
from RL.agent import DQNAgent
from RL.replay_buffer import ReplayBuffer
from RL.benchmark import run_baseline_comparison, LEARNED_POLICY_NAME

_CHECKPOINT_DIR = Path("checkpoints_lfu_imitation")
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "imitation.pt"

# Same training trace as main.py / main_freq_only.py / main_rl_fix.py, so
# the states LFU is cloned from come from the identical distribution the
# RL agent trains on.
_TRAIN_WORKLOAD_SEED = 42
_TRAIN_WORKLOAD_REQUESTS = 200_000

# NOTE (forensic finding, see debug_policy_equivalence.py): a chronological
# split of decisions FROM THE SAME 200k-request trace used for training is
# NOT an independent generalization test. Under Zipfian sampling the same
# ~num_keys hot keys recur throughout the entire trace, so the "held-out"
# tail shares almost all of its frequency/recency/key_age correlational
# structure with the training portion. That produced misleadingly high
# (~100%) "val_action_accuracy" that did not predict closed-loop benchmark
# performance (46.71% vs LFU's 56.73%). Validation now uses a SEPARATE
# trace with its own seed, collected independently of the training
# rollout, so accuracy here is a real (if still open-loop) generalization
# signal instead of near-memorization of a single trace's structure.
_VAL_WORKLOAD_SEED = 142  # disjoint from _TRAIN_WORKLOAD_SEED
_VAL_WORKLOAD_REQUESTS = 30_000

# Identical independent evaluation seeds used in
# evaluate_checkpoint_multiseed.py, so results are directly comparable to
# the reported 44.55% mean DQN / 56.73% mean LFU.
_EVAL_SEEDS = [43, 44, 45, 46, 47, 48]

_BATCH_SIZE = 64
_NUM_EPOCHS = 10
_LEARNING_RATE = 1e-3


def _collect_lfu_transitions(
    environment: CacheEvictionEnvironment,
) -> tuple[np.ndarray, np.ndarray]:
    """Drives `environment` end-to-end under LFU's own policy, recording
    the (state, lfu_action) pair presented at every eviction decision.

    Rolling out under LFU's own actions (rather than random actions)
    means the recorded states are exactly the states LFU visits -- the
    on-policy distribution behavioral cloning is meant to learn from --
    rather than states from an arbitrary/untrained policy's differently-
    evolving cache contents.
    """
    policy = LFUPolicy()
    states: list[np.ndarray] = []
    actions: list[int] = []

    state = environment.reset()
    while not environment.done:
        action = policy.select_action(state)
        states.append(state)
        actions.append(action)
        result = environment.step(action)
        state = result.next_state

    return np.stack(states, axis=0), np.array(actions, dtype=np.int64)





def _train_classifier(
    network: QNetwork,
    train_states: np.ndarray,
    train_actions: np.ndarray,
    val_states: np.ndarray,
    val_actions: np.ndarray,
) -> None:
    optimizer = torch.optim.Adam(network.parameters(), lr=_LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    train_states_t = torch.from_numpy(train_states).float()
    train_actions_t = torch.from_numpy(train_actions).long()
    val_states_t = torch.from_numpy(val_states).float()
    val_actions_t = torch.from_numpy(val_actions).long()

    num_train = train_states_t.shape[0]
    print(f"[Imitation] train examples={num_train:,} val examples={val_states_t.shape[0]:,}")

    for epoch in range(1, _NUM_EPOCHS + 1):
        network.train()
        # Shuffling minibatch order WITHIN an epoch is ordinary SGD
        # practice and does not violate the chronological split -- the
        # split itself (which decisions are ever seen during training at
        # all) is chronological; only the draw order within that fixed
        # training set is randomized per epoch.
        permutation = torch.randperm(num_train)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, num_train, _BATCH_SIZE):
            idx = permutation[start : start + _BATCH_SIZE]
            batch_states = train_states_t[idx]
            batch_actions = train_actions_t[idx]

            logits = network(batch_states)
            loss = loss_fn(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        train_loss = epoch_loss / max(num_batches, 1)

        network.eval()
        with torch.no_grad():
            val_logits = network(val_states_t)
            val_pred = torch.argmax(val_logits, dim=1)
            val_accuracy = (val_pred == val_actions_t).float().mean().item()
            val_loss = loss_fn(val_logits, val_actions_t).item()

        print(
            f"[Imitation] epoch {epoch:>2}/{_NUM_EPOCHS} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_action_accuracy={val_accuracy:.2%}"
        )


def main() -> None:
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig(event_features=8, candidate_features=3)
    network_config = NetworkConfig()
    environment_config = EnvironmentConfig()
    normalizer = StateNormalizer()

    train_workload_config = WorkloadConfig(
        num_keys=5000,
        num_requests=_TRAIN_WORKLOAD_REQUESTS,
        distribution=DistributionType.ZIPFIAN,
        seed=_TRAIN_WORKLOAD_SEED,
    )
    train_workload = WorkloadGenerator(train_workload_config)
    train_cache_state = CacheState()
    train_predictor = Predictor.from_checkpoint(
        checkpoint_path="lstm_popularity_predictor.pt",
        model_config=model_config,
        state=train_cache_state,
    )
    train_environment = CacheEvictionEnvironment(
        config=environment_config,
        workload_generator=train_workload,
        predictor=train_predictor,
        cache_state=train_cache_state,
        normalizer=normalizer,
    )

    print("[Imitation] collecting (state, LFU action) pairs by rolling out LFU (train trace)...")
    train_states, train_actions = _collect_lfu_transitions(train_environment)
    print(f"[Imitation] collected {len(train_states):,} decisions from the seed={_TRAIN_WORKLOAD_SEED} trace.")

    # Independent validation trace: different seed, different request
    # count, its own fresh CacheState/Predictor -- not a slice of the
    # training rollout. See NOTE above _VAL_WORKLOAD_SEED.
    val_workload_config = WorkloadConfig(
        num_keys=5000,
        num_requests=_VAL_WORKLOAD_REQUESTS,
        distribution=DistributionType.ZIPFIAN,
        seed=_VAL_WORKLOAD_SEED,
    )
    val_workload = WorkloadGenerator(val_workload_config)
    val_cache_state = CacheState()
    val_predictor = Predictor.from_checkpoint(
        checkpoint_path="lstm_popularity_predictor.pt",
        model_config=model_config,
        state=val_cache_state,
    )
    val_environment = CacheEvictionEnvironment(
        config=environment_config,
        workload_generator=val_workload,
        predictor=val_predictor,
        cache_state=val_cache_state,
        normalizer=normalizer,
    )
    print("[Imitation] collecting (state, LFU action) pairs by rolling out LFU (independent val trace)...")
    val_states, val_actions = _collect_lfu_transitions(val_environment)
    print(f"[Imitation] collected {len(val_states):,} decisions from the seed={_VAL_WORKLOAD_SEED} trace.")

    network = QNetwork(network_config)
    _train_classifier(network, train_states, train_actions, val_states, val_actions)

    torch.save({"online_state_dict": network.state_dict()}, _CHECKPOINT_PATH)
    print(f"[Imitation] saved trained network to {_CHECKPOINT_PATH}")

    # Wrap into a DQNAgent purely to reuse run_baseline_comparison()
    # unmodified. target_network/optimizer/replay_buffer below are never
    # used again -- no train_step() is called anywhere in this script.
    agent_config = AgentConfig()
    replay_buffer = ReplayBuffer(ReplayBufferConfig())
    agent = DQNAgent(
        agent_config=agent_config,
        network_config=network_config,
        replay_buffer=replay_buffer,
    )
    agent.online_network.load_state_dict(network.state_dict())
    agent.online_network.eval()

    results_by_policy: dict[str, list[float]] = {}

    for seed in _EVAL_SEEDS:
        print(f"=== Evaluating on independent trace, seed={seed} ===")
        eval_workload_config = WorkloadConfig(
            num_keys=5000,
            num_requests=5_000,
            distribution=DistributionType.ZIPFIAN,
            seed=seed,
        )
        benchmark_workload = WorkloadGenerator(eval_workload_config)
        benchmark_cache_state = CacheState()
        benchmark_predictor = Predictor.from_checkpoint(
            checkpoint_path="lstm_popularity_predictor.pt",
            model_config=model_config,
            state=benchmark_cache_state,
        )
        benchmark_environment = CacheEvictionEnvironment(
            config=environment_config,
            workload_generator=benchmark_workload,
            predictor=benchmark_predictor,
            cache_state=benchmark_cache_state,
            normalizer=normalizer,
        )

        entries = run_baseline_comparison(environment=benchmark_environment, agent=agent)
        for entry in entries:
            results_by_policy.setdefault(entry.name, []).append(entry.result.hit_rate)
            print(f"  {entry.name:<40}{entry.result.hit_rate:.2%}")
        print()

    print("=" * 72)
    print(f"LFU-IMITATION MULTI-SEED SUMMARY (n={len(_EVAL_SEEDS)} independent traces)")
    print("=" * 72)
    print(f"{'Policy':<40}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    print("-" * 72)
    for name, values in results_by_policy.items():
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        print(f"{name:<40}{mean:>7.2%} {std:>7.2%} {min(values):>7.2%} {max(values):>7.2%}")
    print("=" * 72)

    if LEARNED_POLICY_NAME in results_by_policy and "LFU" in results_by_policy:
        imitation_vals = results_by_policy[LEARNED_POLICY_NAME]
        lfu_vals = results_by_policy["LFU"]
        wins = sum(1 for i, l in zip(imitation_vals, lfu_vals) if i > l)
        print()
        print(
            f"Imitation-trained network beat LFU on {wins}/{len(_EVAL_SEEDS)} "
            f"independent traces (mean imitation={statistics.mean(imitation_vals):.2%} "
            f"vs mean LFU={statistics.mean(lfu_vals):.2%})"
        )


if __name__ == "__main__":
    main()