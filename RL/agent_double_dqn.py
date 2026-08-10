from __future__ import annotations

import torch

from RL.agent import DQNAgent
from RL.replay_buffer import TransitionBatch


class DoubleDQNAgent(DQNAgent):
    def compute_td_targets(self, batch: TransitionBatch) -> torch.Tensor:
        next_states = torch.from_numpy(batch.next_states).float()
        rewards = torch.from_numpy(batch.rewards).float()
        dones = torch.from_numpy(batch.dones).float()
        steps_elapsed = torch.from_numpy(batch.steps_elapsed).float()

        with torch.no_grad():
            # Action selection uses the ONLINE network (this is the only
            # substantive difference from vanilla DQN).
            next_q_online = self.online_network(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)  # (B, 1)

            # Action evaluation uses the TARGET network, gathered at the
            # action selected above -- not a fresh max over the target
            # network's own values.
            next_q_target = self.target_network(next_states)
            next_value = next_q_target.gather(1, next_actions).squeeze(1)

        # Unchanged: same Semi-MDP discount, same terminal masking.
        discount = self.config.gamma ** steps_elapsed
        targets = rewards + discount * next_value * (1.0 - dones)
        return targets