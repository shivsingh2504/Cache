from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from simulator.config import WorkloadConfig, DistributionType
from simulator.generator import WorkloadGenerator

from feature.state import CacheState

from lstm.config import ModelConfig
from lstm.predictor import Predictor

from RL.config import EnvironmentConfig
from RL.environment import CacheEvictionEnvironment
from RL.normalizer import StateNormalizer
from RL.shared_scorer_network import SharedScorerNetwork, to_shared_scorer_input

# Reused unmodified from main_lfu_imitation.py -- do not duplicate this
# logic. If main_lfu_imitation.py's function signatures ever change,
# this import breaks loudly rather than silently drifting out of sync.
# _train_classifier is architecture-agnostic (it only calls
# network(batch_states), network.parameters(), .train()/.eval()), so it
# works unmodified for SharedScorerNetwork even though it was written
# against QNetwork.
from main_lfu_imitation import (
    _collect_lfu_transitions,
    _train_classifier,
    _TRAIN_WORKLOAD_SEED,
    _TRAIN_WORKLOAD_REQUESTS,
    _VAL_WORKLOAD_SEED,
    _VAL_WORKLOAD_REQUESTS,
)

_CHECKPOINT_DIR = Path("checkpoints_lfu_imitation")
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "shared_scorer_freq_only.pt"
_PROTECTED_PATHS = (
    _CHECKPOINT_DIR / "imitation.pt",
    _CHECKPOINT_DIR / "frequency_only.pt",
)


def main() -> None:
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if _CHECKPOINT_PATH.exists():
        raise FileExistsError(
            f"{_CHECKPOINT_PATH} already exists. Refusing to silently "
            f"overwrite -- delete it manually first if you intend to "
            f"retrain."
        )

    model_config = ModelConfig(event_features=8, candidate_features=3)
    environment_config = EnvironmentConfig()
    normalizer = StateNormalizer()

    # --- Training trace: identical to main_lfu_imitation.py and
    # main_lfu_imitation_freq_only.py (same seed, same request count, same
    # LFU-driven collection), so this is a controlled architecture-only
    # comparison. --------------------------------------------------------
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

    print("[SharedScorer] collecting (state, LFU action) pairs (train trace)...")
    train_states, train_actions = _collect_lfu_transitions(train_environment)
    print(f"[SharedScorer] collected {len(train_states):,} decisions from seed={_TRAIN_WORKLOAD_SEED}.")

    # --- Validation trace: independent, same seed/size as
    # main_lfu_imitation_freq_only.py's val trace. -----------------------
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

    print("[SharedScorer] collecting (state, LFU action) pairs (independent val trace)...")
    val_states, val_actions = _collect_lfu_transitions(val_environment)
    print(f"[SharedScorer] collected {len(val_states):,} decisions from seed={_VAL_WORKLOAD_SEED}.")

    # --- The ONE deliberate difference from main_lfu_imitation_freq_only.py:
    # reduce each state to exactly [frequency, is_empty] per slot instead
    # of masking-in-place to 5 columns. -----------------------------------
    train_states_reduced = to_shared_scorer_input(train_states)
    val_states_reduced = to_shared_scorer_input(val_states)

    # Sanity check: confirm the reduction kept exactly frequency/is_empty,
    # in that order, and nothing else. Cheap, and directly guards against
    # silently training on the wrong columns if the column convention
    # ever drifts from environment.py's.
    assert train_states_reduced.shape[-1] == 2, "expected exactly 2 selected columns"
    assert np.array_equal(train_states_reduced[..., 0], train_states[..., 0]), "frequency column altered"
    assert np.array_equal(train_states_reduced[..., 1], train_states[..., 4]), "is_empty column altered"
    print("[SharedScorer] to_shared_scorer_input verified: columns are exactly [frequency, is_empty].")

    network = SharedScorerNetwork()
    _train_classifier(
        network, train_states_reduced, train_actions, val_states_reduced, val_actions
    )

    # _train_classifier() (reused unmodified from main_lfu_imitation.py)
    # only prints per-epoch stats and returns nothing, so the final
    # train/val loss and accuracy needed for the experiment report are
    # recomputed here, once, post-hoc -- this does not affect training
    # (network is already trained; .eval() + no_grad only).
    loss_fn = torch.nn.CrossEntropyLoss()
    network.eval()
    with torch.no_grad():
        train_logits = network(torch.from_numpy(train_states_reduced).float())
        train_loss = loss_fn(train_logits, torch.from_numpy(train_actions).long()).item()
        train_acc = (
            (torch.argmax(train_logits, dim=1) == torch.from_numpy(train_actions).long())
            .float().mean().item()
        )
        val_logits = network(torch.from_numpy(val_states_reduced).float())
        val_loss = loss_fn(val_logits, torch.from_numpy(val_actions).long()).item()
        val_acc = (
            (torch.argmax(val_logits, dim=1) == torch.from_numpy(val_actions).long())
            .float().mean().item()
        )
    print("[SharedScorer] --- final metrics (post-hoc, network already trained) ---")
    print(f"[SharedScorer] final train_loss={train_loss:.4f} train_accuracy={train_acc:.2%}")
    print(f"[SharedScorer] final val_loss={val_loss:.4f} val_accuracy={val_acc:.2%}")

    protected_existed_before = {path: path.exists() for path in _PROTECTED_PATHS}

    torch.save({"online_state_dict": network.state_dict()}, _CHECKPOINT_PATH)

    created = _CHECKPOINT_PATH.exists()
    print(f"[SharedScorer] checkpoint path: {_CHECKPOINT_PATH.resolve()}")
    print(f"[SharedScorer] file created: {created}")
    for path in _PROTECTED_PATHS:
        print(
            f"[SharedScorer] {path.name} untouched: "
            f"{path.exists() == protected_existed_before[path]}"
        )
    if not created:
        raise RuntimeError("shared_scorer_freq_only.pt was not created -- torch.save silently failed.")


if __name__ == "__main__":
    main()
    