import os
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
from RL.agent_double_dqn import DoubleDQNAgent
from RL.trainer import Trainer
from RL.benchmark import run_baseline_comparison, print_comparison
from RL.export import export_dashboard
from RL.seeding import set_experiment_seed

_ARCHITECTURE_TAG = os.environ.get("CACHE_RL_ARCHITECTURE_TAG", "deepsets_v1")

# Algorithm selector -- purely additive, same pattern as
# _ARCHITECTURE_TAG above. Defaults to "vanilla" so existing behavior,
# checkpoint paths, and dashboard paths are byte-identical when this
# variable is unset. Only "double_dqn" changes anything, and the only
# thing it changes is which DQNAgent subclass is constructed (see
# AgentClass below) plus the artifact path suffix (see _ALGO_SUFFIX
# below) -- nothing else in this file branches on it.
_ALGO_TAG = os.environ.get("CACHE_RL_ALGO_TAG", "vanilla")

# Experiment RNG seed -- conceptually distinct from the workload seeds
# below (train=42, eval=43), which are fixed regardless of this value.
# Controls model init / epsilon-greedy / warmup actions / replay sampling
# via RL/seeding.py::set_experiment_seed(). Optional and additive: if
# unset, behavior is byte-identical to before this change (same
# checkpoint dir, same dashboard path, no seeding call made), so existing
# checkpoints_deepsets_v1/ and checkpoints_flattened_v1/ artifacts from
# the completed architecture comparison are never at risk of being
# overwritten by a seeded run.
_EXPERIMENT_SEED_RAW = os.environ.get("CACHE_RL_EXPERIMENT_SEED")
_EXPERIMENT_SEED = int(_EXPERIMENT_SEED_RAW) if _EXPERIMENT_SEED_RAW is not None else None

# Seed-tagged suffix, applied only when an experiment seed is actually
# set, so unseeded runs keep writing to the exact same
# checkpoints_<tag>/ and dashboard/results_<tag>.html paths as before.
_SEED_SUFFIX = f"_seed{_EXPERIMENT_SEED}" if _EXPERIMENT_SEED is not None else ""

# Algo-tagged suffix, applied only when _ALGO_TAG is not the default
# "vanilla", so existing vanilla-DQN checkpoint/dashboard paths are
# completely unaffected. When CACHE_RL_ALGO_TAG=double_dqn, this
# guarantees Double-DQN artifacts land in a distinct, non-colliding
# path (e.g. checkpoints_deepsets_v1_double_dqn_seed42) and can never
# overwrite the existing checkpoints_deepsets_v1_seed42/ baseline.
_ALGO_SUFFIX = f"_{_ALGO_TAG}" if _ALGO_TAG != "vanilla" else ""

_CHECKPOINT_DIR = Path(f"checkpoints_{_ARCHITECTURE_TAG}{_ALGO_SUFFIX}{_SEED_SUFFIX}")
_BEST_CHECKPOINT_PATH = _CHECKPOINT_DIR / "best.pt"

# Dashboard output likewise tagged per architecture (and, when set, per
# experiment seed) so old-vs-new results don't overwrite each other and
# can be compared side by side afterward.
_DASHBOARD_PATH = Path(f"dashboard/results_{_ARCHITECTURE_TAG}{_ALGO_SUFFIX}{_SEED_SUFFIX}.html")


def _load_best_checkpoint(agent: DQNAgent, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    try:
        agent.online_network.load_state_dict(checkpoint["online_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}' into the "
            f"network architecture currently active in RL/network.py "
            f"(CACHE_RL_ARCHITECTURE_TAG='{_ARCHITECTURE_TAG}'). This "
            f"almost always means the checkpoint was written by a "
            f"DIFFERENT architecture (e.g. the old flattened QNetwork vs "
            f"the new permutation-equivariant QNetwork) than whatever is "
            f"currently copied into RL/network.py. Each architecture "
            f"writes to its own checkpoint_dir (checkpoints_<tag>/), so "
            f"this typically means RL/network.py was swapped after this "
            f"checkpoint was produced, or CACHE_RL_ARCHITECTURE_TAG does "
            f"not match the architecture that's actually live in "
            f"RL/network.py right now. Original error: {exc}"
        ) from exc
    agent.online_network.eval()
    return checkpoint


def main() -> None:
    print(f"[Architecture] CACHE_RL_ARCHITECTURE_TAG={_ARCHITECTURE_TAG}")
    print(f"[Algorithm] CACHE_RL_ALGO_TAG={_ALGO_TAG}")
    print(f"[Architecture] checkpoint_dir={_CHECKPOINT_DIR}")
    print(f"[Architecture] dashboard_path={_DASHBOARD_PATH}")
    if _EXPERIMENT_SEED is not None:
        print(f"[Reproducibility] CACHE_RL_EXPERIMENT_SEED={_EXPERIMENT_SEED}")
    else:
        print(
            "[Reproducibility] CACHE_RL_EXPERIMENT_SEED not set -- "
            "training RNG is uncontrolled (pre-existing behavior)."
        )

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
    # Diagnostic run (Gap B, config-only): epsilon_decay_steps shortened
    # from the default 50,000 to 15,000 so the greedy regime is reached
    # within a 30,000-step run instead of never being reached. This is
    # intentional and changes the exploration schedule -- this run does
    # NOT reproduce the original 100k-step / 50k-decay experiment. See
    # diagnostic run notes: goal is to observe Q/target divergence as
    # epsilon approaches its floor around step 15k, not to reproduce the
    # original 20k-peak -> 70k-degradation curve.
    agent_config = AgentConfig(epsilon_decay_steps=15_000)
    # Diagnostic run (Gap B, config-only): fewer training steps and a much
    # denser eval_interval so the existing [Diag] Q/target print statements
    # (already implemented in trainer.py::log_diagnostics, gated on
    # step % eval_interval == 0) surface at a much finer granularity.
    # No instrumentation code changed -- only how often the existing
    # diagnostics fire.
    trainer_config = TrainerConfig(
        num_training_steps=30_000,
        eval_interval=2_500,
    )
    # environment_config.gamma must equal agent_config.gamma: the
    # environment discounts intra-option rewards by it, and the agent
    # discounts the Semi-MDP bootstrap by gamma**k against the same
    # rewards. Sourced from agent_config so the two can't drift apart.
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

    # A single stateless normalizer, shared across the training environment,
    # the held-out evaluation environment, and every benchmark environment
    # constructed later in run_baseline_comparison(). Safe to share because
    # the transform is fixed-form (log1p) with no fitted statistics.
    normalizer = StateNormalizer()

    environment = CacheEvictionEnvironment(
        config=environment_config,
        workload_generator=workload,
        predictor=predictor,
        cache_state=cache_state,
        normalizer=normalizer,
    )
    eval_environment = CacheEvictionEnvironment(
        config=environment_config,
        workload_generator=eval_workload,
        predictor=eval_predictor,
        cache_state=eval_cache_state,
        normalizer=normalizer,
    )

    # Experiment seed applied here: immediately before the first
    # QNetwork() is constructed (inside DQNAgent.__init__), and after
    # every workload/environment object above -- none of which consume
    # this RNG stream (WorkloadGenerator uses its own independent
    # WorkloadConfig.seed). This is the only set_experiment_seed() call
    # in the entry point; nothing downstream seeds anything itself.
    if _EXPERIMENT_SEED is not None:
        set_experiment_seed(_EXPERIMENT_SEED)

    # AgentClass selection is the only behavioral branch _ALGO_TAG
    # controls. DoubleDQNAgent subclasses DQNAgent and overrides only
    # compute_td_targets (see RL/agent_double_dqn.py) -- construction
    # signature, network_config, replay_buffer wiring, and every other
    # code path below this point are identical regardless of which
    # class is chosen.
    AgentClass = DoubleDQNAgent if _ALGO_TAG == "double_dqn" else DQNAgent
    agent = AgentClass(
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
        # Tagged per architecture (see _CHECKPOINT_DIR above) -- the only
        # line in this function that differs from before the architecture
        # investigation. Everything else in this Trainer construction is
        # unchanged.
        checkpoint_dir=str(_CHECKPOINT_DIR),
    )

    trainer.train()

    # --- Evaluation correction (audit-authorized, evaluation-only) -----
    #
    # 1. Restore the best checkpoint. eval_environment's periodic
    #    Trainer.evaluate() calls left the live agent's online_network at
    #    whatever it happened to be after the final training step, which
    #    the audit showed is not necessarily its best observed policy.
    #    Reloading checkpoints_<tag>/best.pt into agent.online_network
    #    makes the benchmark reflect the best checkpoint Trainer ever
    #    saved instead.
    best_checkpoint = _load_best_checkpoint(agent, _BEST_CHECKPOINT_PATH)
    print(
        f"[Checkpoint] loaded {_BEST_CHECKPOINT_PATH} "
        f"(step={best_checkpoint['step']:,}, "
        f"recorded_hit_rate={best_checkpoint['hit_rate']:.2%})"
    )

    # 2. Build a fresh evaluation workload/environment for the final
    #    benchmark instead of reusing eval_environment. eval_workload's
    #    WorkloadGenerator seeds its RNG once at construction and never
    #    reseeds (see generator.py) -- eval_environment's own periodic
    #    evaluations during training already consumed many draws from
    #    that RNG stream, so reusing it here would silently benchmark
    #    against an unreproducible, non-seed-43-aligned trace. Same
    #    eval_workload_config, same seed (43), same num_requests (5,000)
    #    -- only the WorkloadGenerator instance (and its backing
    #    CacheState/Predictor, which are themselves stateful and must not
    #    be shared with the already-used eval_environment) is fresh.
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

    benchmark_entries = run_baseline_comparison(environment=benchmark_environment, agent=agent)
    print_comparison(benchmark_entries)

    dashboard_results = {
        entry.name: entry.result for entry in benchmark_entries
    }
    dashboard_path = export_dashboard(trainer, dashboard_results, output_path=_DASHBOARD_PATH)
    print(f"Dashboard written to {dashboard_path}")
    print(f"[Architecture] this run used CACHE_RL_ARCHITECTURE_TAG='{_ARCHITECTURE_TAG}'")
    print(f"[Algorithm] this run used CACHE_RL_ALGO_TAG='{_ALGO_TAG}'")


if __name__ == "__main__":
    main()