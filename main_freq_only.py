
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
    TrainerConfig,
)

from RL.environment import CacheEvictionEnvironment
from RL.normalizer import StateNormalizer
from RL.replay_buffer import ReplayBuffer
from RL.agent import DQNAgent
from RL.trainer import Trainer
from RL.benchmark import print_comparison
from RL.frequency_only import FrequencyOnlyEnvironment, run_frequency_only_benchmark

_CHECKPOINT_DIR = "checkpoints_freq_only"
_BEST_CHECKPOINT_PATH = Path(_CHECKPOINT_DIR) / "best.pt"


def _load_best_checkpoint(agent: DQNAgent, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    agent.online_network.load_state_dict(checkpoint["online_state_dict"])
    agent.online_network.eval()
    return checkpoint


def main() -> None:
    # Identical to main.py -- same distribution, same seeds, same sizes.
    # This must be an apples-to-apples ablation against the 49.16% result.
    workload_config = WorkloadConfig(
        num_keys=5000,
        num_requests=200_000,
        distribution=DistributionType.ZIPFIAN,
        seed=42,
    )
    eval_workload_config = WorkloadConfig(
        num_keys=5000,
        num_requests=5_000,
        distribution=DistributionType.ZIPFIAN,
        seed=43,
    )

    model_config = ModelConfig(
        event_features=8,
        candidate_features=3,
    )

    replay_buffer_config = ReplayBufferConfig()
    network_config = NetworkConfig()
    # Same config-only diagnostic settings as the corrected production run.
    agent_config = AgentConfig(epsilon_decay_steps=15_000)
    trainer_config = TrainerConfig(
        num_training_steps=30_000,
        eval_interval=2_500,
    )
    environment_config = EnvironmentConfig(gamma=agent_config.gamma)

    assert network_config.cache_capacity == environment_config.cache_capacity, (
        "NetworkConfig.cache_capacity must match EnvironmentConfig.cache_capacity "
        f"(got {network_config.cache_capacity} vs {environment_config.cache_capacity})"
    )
    assert network_config.num_state_features == environment_config.num_state_features, (
        "NetworkConfig.num_state_features must match EnvironmentConfig.num_state_features "
        f"(got {network_config.num_state_features} vs {environment_config.num_state_features})"
    )

    workload = WorkloadGenerator(workload_config)
    eval_workload = WorkloadGenerator(eval_workload_config)

    cache_state = CacheState()
    eval_cache_state = CacheState()

    predictor = Predictor.from_checkpoint(
        checkpoint_path="lstm_popularity_predictor.pt",
        model_config=model_config,
        state=cache_state,
    )
    eval_predictor = Predictor.from_checkpoint(
        checkpoint_path="lstm_popularity_predictor.pt",
        model_config=model_config,
        state=eval_cache_state,
    )

    replay_buffer = ReplayBuffer(replay_buffer_config)
    normalizer = StateNormalizer()

    raw_environment = CacheEvictionEnvironment(
        config=environment_config,
        workload_generator=workload,
        predictor=predictor,
        cache_state=cache_state,
        normalizer=normalizer,
    )
    raw_eval_environment = CacheEvictionEnvironment(
        config=environment_config,
        workload_generator=eval_workload,
        predictor=eval_predictor,
        cache_state=eval_cache_state,
        normalizer=normalizer,
    )

    # Only these two lines differ structurally from main.py: the DQN sees
    # a masked state for both training and periodic in-training eval.
    environment = FrequencyOnlyEnvironment(raw_environment)
    eval_environment = FrequencyOnlyEnvironment(raw_eval_environment)

    agent = DQNAgent(
        agent_config=agent_config,
        network_config=network_config,
        replay_buffer=replay_buffer,
    )

    trainer = Trainer(
        config=trainer_config,
        environment=environment,
        eval_environment=eval_environment,
        agent=agent,
        replay_buffer=replay_buffer,
        checkpoint_dir=_CHECKPOINT_DIR,
    )

    trainer.train()

    # --- Corrected benchmark methodology, frequency-only variant --------
    #
    # 1. Restore this ablation's own best checkpoint (never the production
    #    checkpoints/best.pt).
    best_checkpoint = _load_best_checkpoint(agent, _BEST_CHECKPOINT_PATH)
    print(
        f"[Checkpoint] loaded {_BEST_CHECKPOINT_PATH} "
        f"(step={best_checkpoint['step']:,}, "
        f"recorded_hit_rate={best_checkpoint['hit_rate']:.2%})"
    )

    # 2. Fresh seed-43 workload/environment -- NOT the already-consumed
    #    eval_workload/eval_environment used during training. Raw
    #    (unwrapped) CacheEvictionEnvironment: masking for the DQN row
    #    happens inside run_frequency_only_benchmark() at the policy
    #    level, so FIFO/LRU/LFU/Belady see this environment completely
    #    unmasked, exactly like the production benchmark.
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

    benchmark_entries = run_frequency_only_benchmark(environment=benchmark_environment, agent=agent)
    print_comparison(benchmark_entries)


if __name__ == "__main__":
    main()