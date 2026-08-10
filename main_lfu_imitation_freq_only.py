
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from simulator.config import WorkloadConfig, DistributionType
from simulator.generator import WorkloadGenerator

from feature.state import CacheState

from lstm.config import ModelConfig
from lstm.predictor import Predictor

from RL.config import EnvironmentConfig, NetworkConfig
from RL.environment import CacheEvictionEnvironment
from RL.normalizer import StateNormalizer
from RL.network import QNetwork
from RL.frequency_only import mask_frequency_only

# Reused unmodified from main_lfu_imitation.py -- do not duplicate this
# logic. If main_lfu_imitation.py's function signatures ever change,
# this import breaks loudly rather than silently drifting out of sync.
from main_lfu_imitation import (
    _collect_lfu_transitions,
    _train_classifier,
    _TRAIN_WORKLOAD_SEED,
    _TRAIN_WORKLOAD_REQUESTS,
    _VAL_WORKLOAD_SEED,
    _VAL_WORKLOAD_REQUESTS,
)

_CHECKPOINT_DIR = Path("checkpoints_lfu_imitation")
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "frequency_only.pt"
_PROTECTED_PATH = _CHECKPOINT_DIR / "imitation.pt"


def main() -> None:
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if _CHECKPOINT_PATH.exists():
        raise FileExistsError(
            f"{_CHECKPOINT_PATH} already exists. Refusing to silently "
            f"overwrite -- delete it manually first if you intend to "
            f"retrain."
        )

    model_config = ModelConfig(event_features=8, candidate_features=3)
    network_config = NetworkConfig()
    environment_config = EnvironmentConfig()
    normalizer = StateNormalizer()

    # --- Training trace: identical to main_lfu_imitation.py -------------
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

    print("[FreqOnly] collecting (state, LFU action) pairs (train trace)...")
    train_states, train_actions = _collect_lfu_transitions(train_environment)
    print(f"[FreqOnly] collected {len(train_states):,} decisions from seed={_TRAIN_WORKLOAD_SEED}.")

    # --- Validation trace: independent, matches the corrected
    # main_lfu_imitation.py methodology (see module docstring) -----------
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

    print("[FreqOnly] collecting (state, LFU action) pairs (independent val trace)...")
    val_states, val_actions = _collect_lfu_transitions(val_environment)
    print(f"[FreqOnly] collected {len(val_states):,} decisions from seed={_VAL_WORKLOAD_SEED}.")

    # --- The ONE deliberate difference from main_lfu_imitation.py -------
    train_states_masked = mask_frequency_only(train_states)
    val_states_masked = mask_frequency_only(val_states)

    # Sanity check: confirm the mask actually did something and did not
    # touch frequency/is_empty. Cheap, and directly guards against
    # silently training on unmasked data if mask_frequency_only's column
    # convention ever drifts from environment.py's.
    assert np.array_equal(train_states_masked[:, 0], train_states[:, 0]), "frequency column altered"
    assert np.array_equal(train_states_masked[:, 4], train_states[:, 4]), "is_empty column altered"
    assert np.all(train_states_masked[:, 1] == 0.0), "recency not zeroed"
    assert np.all(train_states_masked[:, 2] == 0.0), "key_age not zeroed"
    assert np.all(train_states_masked[:, 3] == 0.0), "predicted_popularity not zeroed"
    print("[FreqOnly] mask_frequency_only verified: cols 1-3 zeroed, cols 0/4 preserved.")

    network = QNetwork(network_config)
    _train_classifier(
        network, train_states_masked, train_actions, val_states_masked, val_actions
    )

    imitation_existed_before = _PROTECTED_PATH.exists()

    torch.save({"online_state_dict": network.state_dict()}, _CHECKPOINT_PATH)

    created = _CHECKPOINT_PATH.exists()
    print(f"[FreqOnly] checkpoint path: {_CHECKPOINT_PATH.resolve()}")
    print(f"[FreqOnly] file created: {created}")
    print(
        "[FreqOnly] imitation.pt untouched: "
        f"{_PROTECTED_PATH.exists() == imitation_existed_before}"
    )
    if not created:
        raise RuntimeError("frequency_only.pt was not created -- torch.save silently failed.")


if __name__ == "__main__":
    main()