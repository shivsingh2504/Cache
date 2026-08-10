
import torch
import torch.nn as nn
from RL.config import NetworkConfig


class QNetwork(nn.Module):
    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config

        num_features = config.num_state_features
        hidden_size = config.hidden_sizes[0]
        self.hidden_size = hidden_size

        # Shared per-slot encoder (phi). Same weights process every slot
        # -- no per-slot Linear layer, no slot-index input.
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Shared per-slot head (psi), consuming [local_embedding ; global
        # context], hence 2 * hidden_size input features. Same weights
        # process every slot.
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() != 3:
            raise ValueError(
                f"QNetwork expects input shape (batch, cache_capacity, "
                f"num_state_features), got {tuple(state.shape)}."
            )
        batch_size, cache_capacity, num_features = state.shape
        if num_features != self.config.num_state_features:
            raise ValueError(
                f"QNetwork configured for num_state_features="
                f"{self.config.num_state_features}, got {num_features}."
            )

        # --- shared encoder phi, applied independently per slot ---------
        # Collapse batch and slot dims together (NOT slot into the
        # feature dim -- that would reintroduce flattening/positional
        # weights). Every row of (batch*cache_capacity, num_features) is
        # one slot's raw feature vector, processed by the same weights.
        flattened_slots = state.reshape(batch_size * cache_capacity, num_features)
        embeddings_flat = self.encoder(flattened_slots)  # (batch*capacity, H)
        embeddings = embeddings_flat.reshape(
            batch_size, cache_capacity, self.hidden_size
        )  # (batch, capacity, H)

        # --- permutation-invariant pooling across the slot axis ---------
        # Mean pooling only, per the approved design -- no max/attention.
        # A symmetric function of the *set* of embeddings: reordering the
        # slot axis before this line does not change the result.
        context = embeddings.mean(dim=1)  # (batch, H)

        # --- broadcast + concatenate -------------------------------------
        context_broadcast = context.unsqueeze(1).expand(-1, cache_capacity, -1)
        combined = torch.cat([embeddings, context_broadcast], dim=-1)  # (batch, capacity, 2H)

        # --- shared head psi, applied independently per slot -------------
        combined_flat = combined.reshape(batch_size * cache_capacity, self.hidden_size * 2)
        q_values_flat = self.head(combined_flat)  # (batch*capacity, 1)
        q_values = q_values_flat.reshape(batch_size, cache_capacity)  # (batch, capacity)

        return q_values