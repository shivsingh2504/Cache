from dataclasses import dataclass
import numpy as np
from RL.config import ReplayBufferConfig

@dataclass
class Transition:
  states : np.ndarray
  actions : np.ndarray
  rewards : np.ndarray
  next_states : np.ndarray
  dones : np.ndarray
  
class ReplayBuffer:
  def __init__(self,config:ReplayBufferConfig)->None:
    self.config = config
    
  def push(self,transition:Transition)->None:
    raise NotImplementedError
  
  def sample(self,batch_size:int|None)->None:
    raise NotImplementedError
  
  def __len__(self)->int:
    raise NotImplementedError
  