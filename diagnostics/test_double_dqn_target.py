

from __future__ import annotations

import torch
import torch.nn as nn


class TinyQNetwork(nn.Module):

    def __init__(self, capacity: int, features: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.capacity = capacity

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        b, c, f = state.shape
        flat = state.reshape(b * c, f)
        q = self.net(flat).reshape(b, c)
        return q


def vanilla_target(target_net, next_states, rewards, dones, steps_elapsed, gamma):
    with torch.no_grad():
        next_q = target_net(next_states)
        max_next_q = next_q.max(dim=1).values
    discount = gamma ** steps_elapsed
    return rewards + discount * max_next_q * (1.0 - dones)


def double_target(online_net, target_net, next_states, rewards, dones, steps_elapsed, gamma):
    with torch.no_grad():
        next_actions = online_net(next_states).argmax(dim=1, keepdim=True)
        next_value = target_net(next_states).gather(1, next_actions).squeeze(1)
    discount = gamma ** steps_elapsed
    return rewards + discount * next_value * (1.0 - dones)


def main() -> None:
    capacity, features, batch = 16, 5, 32
    gamma = 0.99

    next_states = torch.randn(batch, capacity, features)
    rewards = torch.randn(batch)
    steps_elapsed = torch.randint(1, 5, (batch,)).float()

    # --- Test 1 & 4: differing networks + discounting -------------------
    online_net = TinyQNetwork(capacity, features, seed=1)
    target_net = TinyQNetwork(capacity, features, seed=2)  # different weights
    dones = torch.zeros(batch)

    v = vanilla_target(target_net, next_states, rewards, dones, steps_elapsed, gamma)
    d = double_target(online_net, target_net, next_states, rewards, dones, steps_elapsed, gamma)

    frac_diff = (v - d).abs().gt(1e-6).float().mean().item()
    assert frac_diff > 0.0, "Double DQN target identical to vanilla on all rows -- override not active."
    print(f"[Test 1] targets differ on {frac_diff:.0%} of rows when online != target network (expected >0%). PASS")

    # --- Test 2: online == target -> Double DQN reduces to vanilla ------
    target_net_same = TinyQNetwork(capacity, features, seed=1)
    target_net_same.load_state_dict(online_net.state_dict())  # simulate hard copy

    v2 = vanilla_target(target_net_same, next_states, rewards, dones, steps_elapsed, gamma)
    d2 = double_target(online_net, target_net_same, next_states, rewards, dones, steps_elapsed, gamma)
    assert torch.allclose(v2, d2, atol=1e-6), "Double DQN should equal vanilla DQN when online==target."
    print("[Test 2] Double DQN target == vanilla DQN target when online/target networks are identical. PASS")

    # --- Test 3: terminal transitions never bootstrap --------------------
    dones_terminal = torch.ones(batch)
    v3 = vanilla_target(target_net, next_states, rewards, dones_terminal, steps_elapsed, gamma)
    d3 = double_target(online_net, target_net, next_states, rewards, dones_terminal, steps_elapsed, gamma)
    assert torch.allclose(v3, rewards, atol=1e-6), "Vanilla target should equal reward alone when done=1."
    assert torch.allclose(d3, rewards, atol=1e-6), "Double DQN target should equal reward alone when done=1."
    print("[Test 3] Terminal transitions bootstrap nothing in either formula (target == reward). PASS")

    print("\nAll diagnostic checks passed.")


if __name__ == "__main__":
    main()