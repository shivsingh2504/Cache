"""Shared-scorer frequency-only LFU imitation ablation.

Companion to RL/frequency_only.py's flattened-MLP frequency-only
ablation. That experiment found the flattened MLP produces radically
different Q-values for slots with byte-identical [frequency, is_empty]
inputs (see debug_freq_only_vs_original.py), which is consistent with the
flattened architecture learning to exploit fixed slot position rather
than the frequency value itself -- every slot occupies a distinct,
fixed region of the flattened 80-dimensional input vector, so nothing
prevents position-dependent weights from substituting for the intended
frequency rule.

This module tests the natural fix: force the SAME small scorer to be
applied independently to every slot, so slot position cannot be encoded
in per-slot weights at all.

    score_i = shared_scorer([frequency_i, is_empty_i])   for i in 0..15
    action  = argmax_i score_i

Diagnostic experiment only -- not a production component. Composes
existing, untouched pieces (RL/environment.py, RL/baseline.py,
main_lfu_imitation.py's collection/training helpers) rather than editing
any of them. Does not touch RL/network.py's QNetwork, which remains the
architecture used by the original 5-feature imitation and the flattened
frequency-only ablation.

Verified state layout, from RL/environment.py::
CacheEvictionEnvironment.current_state():

    column 0 = frequency
    column 1 = recency
    column 2 = key_age
    column 3 = predicted_popularity
    column 4 = is_empty

The shared scorer sees ONLY columns 0 and 4, and never sees a flattened
16-slot vector -- each slot is scored independently by the same weights.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Mirrors the column-index convention already used in baseline.py
# (_FREQUENCY_COL, _IS_EMPTY_COL) and frequency_only.py.
_FREQUENCY_COL = 0
_IS_EMPTY_COL = 4
_SELECTED_COLUMNS = (_FREQUENCY_COL, _IS_EMPTY_COL)

_DEFAULT_HIDDEN_SIZE = 32


def to_shared_scorer_input(state: np.ndarray) -> np.ndarray:
    """Returns a COPY of `state` reduced to exactly the two per-slot
    columns the shared scorer is allowed to see: frequency (column 0)
    and is_empty (column 4), in that order.

    Unlike frequency_only.py::mask_frequency_only, this does not zero
    columns 1-3 and return a same-shaped 5-column array -- it selects
    and returns only the 2 columns, since the shared scorer's input
    dimensionality is 2, not 5. This is a deliberate difference: feeding
    the five-column masked representation (with three zeroed columns)
    into a per-slot scorer would waste input capacity on constant
    columns but would not itself violate the "no flattening, no slot
    index" requirement. Using exactly 2 columns keeps the input to
    'shared_scorer' unambiguous and matches the architecture spec
    exactly: [frequency, is_empty].

    Accepts either a single state, shape (cache_capacity,
    num_state_features), or a batch, shape (batch, cache_capacity,
    num_state_features) -- the selected columns are taken from the last
    axis in both cases, so the output shape is (..., 2).

    Returns a copy rather than mutating in place: `state` may be a
    reference the caller (the environment, the replay buffer, or another
    consumer) still holds and depends on being unmodified.
    """
    return state[..., list(_SELECTED_COLUMNS)].copy()


class SharedScorerNetwork(nn.Module):
    """Scores every cache slot independently with ONE shared small MLP.

        score_i = shared_scorer([frequency_i, is_empty_i])

    The same scorer weights are used for all cache_capacity slots -- the
    network never flattens the slot dimension into its input (contrast
    RL/network.py::QNetwork, which reshapes (cache_capacity,
    num_state_features) to a single (cache_capacity * num_state_features,)
    vector before its MLP) and slot index is never provided as a
    feature.

    Architecture, per the experiment spec:

        Linear(2, hidden_size)
        ReLU
        Linear(hidden_size, 1)

    applied identically to each of the 16 (frequency, is_empty) pairs.

    Input:  (batch, cache_capacity, 2)
    Output: (batch, cache_capacity)  -- one scalar score per slot, with
        the same shape contract as QNetwork's output, so this network is
        a drop-in replacement for QNetwork wherever only that shape
        contract matters (e.g. main_lfu_imitation.py::_train_classifier,
        which calls network(batch_states) and computes
        CrossEntropyLoss(logits, actions) against it) -- no other API is
        assumed.
    """

    def __init__(self, hidden_size: int = _DEFAULT_HIDDEN_SIZE) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.shared_scorer = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() != 3 or state.shape[-1] != 2:
            raise ValueError(
                "SharedScorerNetwork expects input shape "
                f"(batch, cache_capacity, 2) -- [frequency, is_empty] "
                f"per slot -- got {tuple(state.shape)}."
            )
        batch_size, cache_capacity, num_features = state.shape

        # Collapse batch and slot dims together (NOT slot into feature
        # dim) so the exact same Linear weights see every slot of every
        # example as an independent row. This is what makes the scorer
        # "shared": there is no per-slot parameter anywhere in this
        # module.
        flattened_slots = state.reshape(batch_size * cache_capacity, num_features)
        scores = self.shared_scorer(flattened_slots)  # (batch*cache_capacity, 1)
        return scores.reshape(batch_size, cache_capacity)

    def select_action(self, state: np.ndarray) -> int:
        """Deterministic greedy action selection for a single raw
        (cache_capacity, num_state_features) state. Applies
        to_shared_scorer_input() internally so callers can pass the
        environment's normal state unchanged, mirroring
        FrequencyOnlyGreedyAgentPolicy's select_action in
        RL/frequency_only.py.
        """
        reduced = to_shared_scorer_input(state)  # (cache_capacity, 2)
        self.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(reduced).float().unsqueeze(0)
            scores = self.forward(state_tensor)
            action = int(torch.argmax(scores, dim=1).item())
        return action