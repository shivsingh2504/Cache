from dataclasses import dataclass
import numpy as np
from RL.config import ReplayBufferConfig
import random
from collections import deque
@dataclass
class Transition:
  states : np.ndarray
  actions : int
  rewards: float
  next_states : np.ndarray
  dones : bool

@dataclass
class TransitionBatch:
  states : np.ndarray
  actions : np.ndarray
  rewards : np.ndarray
  next_states : np.ndarray
  dones : np.ndarray
  
class ReplayBuffer:
  def __init__(self,config:ReplayBufferConfig)->None:
    self.config = config
    self._buffer = deque[Transition] = deque(maxlen=config.capacity)
    
    
  def push(self,transition:Transition)->None:
    self._buffer.append(transition)
    
  
  def sample(self,batch_size:int|None)->None:
    resolved_batch_size = batch_size if batch_size is not None else self.config.batch_size
    if len(self._buffer) < resolved_batch_size:
      raise ValueError(
                f"Cannot sample batch_size={resolved_batch_size} from a "
                f"buffer containing only {len(self._buffer)} transitions."
            )
    sampled = random.sample(self._buffer,resolved_batch_size)
    return TransitionBatch(states=np.stack([t.state for t in self.sampled],axis=0),
                           actions=np.stack([t.action for t in sampled]),
                           rewards=np.array([t.reward for t in sampled], dtype=np.float32),
                           next_states=np.stack([t.next_state for t in sampled], axis=0),
                           dones=np.array([t.done for t in sampled], dtype=np.bool_)
                           )
    
    
  
  def __len__(self)->int:
    return len(self._buffer)
  