import random
from collections import deque
from dataclasses import dataclass

from feature.state import CacheState
from lstm.config import DatasetConfig
from model_input.candidate_builder import CandidateFeatureBuilder
from model_input.schema import TrainingSample

class CandidatePool:
  def __init__(self,window_size:int)->None:
    self._window_size = window_size
    self._recent_keys : deque[int] = deque(maxlen=window_size)
    
  def observe(self,key:int)->None:
    self._recent_keys.append(key)
    
  def sample(self,n:int,rng:random.Random)->list[int]:
    pool = list(set(self._recent_keys))
    if not pool:
      return []
    n = min(n,len(pool))
    return rng.sample(pool,n)
  