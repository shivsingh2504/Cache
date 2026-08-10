
from __future__ import annotations

import statistics
from pathlib import Path

import torch

from simulator.config import WorkloadConfig, DistributionType
from simulator.generator import WorkloadGenerator

from feature.state import CacheState

from lstm.config import ModelConfig
from lstm.predictor import Predictor

from RL.config import (
    EnvironmentConfig,
    ReplayBufferConfig,
    AgentConfig,
    NetworkConfig,
)

from RL.environment import CacheEvictionEnvironment
from RL.normalizer import StateNormalizer
from RL.replay_buffer import ReplayBuffer
from RL.agent import DQNAgent
from RL.frequency_only import run_frequency_only_benchmark

# The trained artifact under test. Read-only: this script never writes
# to this path.
_CHECKPOINT_PATH = Path("checkpoints_rl_fix") / "best.pt"

# Multiple independent evaluation traces. Same WorkloadConfig shape
# (num_keys=5000, num_requests=5_000, ZIPFIAN) as the original seed-43
# benchmark -- only the seed varies, so each is a fresh, independent
# draw from the identical distribution. Includes the original seed=43
# so its single-trace result is directly visible alongside the others,
# not just summarized away.
_EVAL_SEEDS = [43, 44, 45, 46, 47, 48]


def _load_checkpoint(agent: DQNAgent, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    agent.online_network.load_state_dict(checkpoint["online_state_dict"])
    agent.online_network.eval()
    return checkpoint


def main() -> None:
    if not _CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{_CHECKPOINT_PATH} not found. This script only evaluates an "
            "already-trained checkpoint -- run main_rl_fix.py first."
        )

    model_config = ModelConfig(event_features=8, candidate_features=3)
    network_config = NetworkConfig()
    # gamma/epsilon/etc are irrelevant here -- evaluation is always greedy
    # (select_action(..., greedy=True) bypasses epsilon entirely) and no
    # training step ever runs in this script. Kept at the same values as
    # main_rl_fix.py purely so nothing about the agent's construction
    # differs from the run that produced this checkpoint.
    agent_config = AgentConfig(epsilon_decay_steps=15_000)
    environment_config = EnvironmentConfig(gamma=agent_config.gamma)
    normalizer = StateNormalizer()

    # A replay buffer is required to construct DQNAgent, but is never
    # pushed to or sampled from in this script -- no training occurs.
    replay_buffer = ReplayBuffer(ReplayBufferConfig())

    agent = DQNAgent(
        agent_config=agent_config,
        network_config=network_config,
        replay_buffer=replay_buffer,
    )
    checkpoint = _load_checkpoint(agent, _CHECKPOINT_PATH)
    print(
        f"[Checkpoint] loaded {_CHECKPOINT_PATH} "
        f"(step={checkpoint['step']:,}, "
        f"recorded_hit_rate={checkpoint['hit_rate']:.2%})"
    )
    print(
        "[Note] recorded_hit_rate above is the 5-episode-averaged periodic-eval "
        "score that selected this checkpoint as best -- it is NOT expected to "
        "match any single-seed benchmark result below exactly; that mismatch "
        "is the subject of this script.\n"
    )

    # policy name -> list of hit rates, one per seed
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

        # Unmodified existing benchmark methodology -- same function used
        # by main_rl_fix.py, just called once per seed here.
        entries = run_frequency_only_benchmark(environment=benchmark_environment, agent=agent)

        for entry in entries:
            results_by_policy.setdefault(entry.name, []).append(entry.result.hit_rate)
            print(f"  {entry.name:<40}{entry.result.hit_rate:.2%}")
        print()

    print("=" * 72)
    print("MULTI-SEED SUMMARY (n={} independent traces)".format(len(_EVAL_SEEDS)))
    print("=" * 72)
    print(f"{'Policy':<40}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    print("-" * 72)
    for name, values in results_by_policy.items():
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        print(
            f"{name:<40}{mean:>7.2%} {std:>7.2%} {min(values):>7.2%} {max(values):>7.2%}"
        )
    print("=" * 72)

    if "Cache Intelligence (Frequency-only DQN)" in results_by_policy and "LFU" in results_by_policy:
        dqn_vals = results_by_policy["Cache Intelligence (Frequency-only DQN)"]
        lfu_vals = results_by_policy["LFU"]
        dqn_mean = statistics.mean(dqn_vals)
        lfu_mean = statistics.mean(lfu_vals)
        wins = sum(1 for d, l in zip(dqn_vals, lfu_vals) if d > l)
        print()
        print(
            f"DQN beat LFU on {wins}/{len(_EVAL_SEEDS)} independent traces "
            f"(mean DQN={dqn_mean:.2%} vs mean LFU={lfu_mean:.2%})"
        )


if __name__ == "__main__":
    main()