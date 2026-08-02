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


class LabelGenerator:
  def __init__(self,horizon:int)->None:
    self._horizon = horizon
    
  def label_for(self,future_keys: list[int],candidate_key:int)->float:
    window = future_keys[: self._horizon]
    return float(window.count(candidate_key))
  
@dataclass
class TrainingSampleBuilder:
  dataset_config: DatasetConfig
  state : CacheState
  candidate_pool : CandidatePool
  label_generator : LabelGenerator
  rng : random.Random
  
  def _post_init__(self)->None:
    self._feature_builder = CandidateFeatureBuilder(self.state)
    
  def build_sample(self,context:tuple,future_keys : list[int])->list[TrainingSample]:
    candidates = self.candidate_pool.sample(self.dataset_config.candidate_per_context,self.rng)
    samples = []
    for key in candidates:
      candidate_feature = self._feature_builder.build(key)
      label = self.label_generator.label_for(future_keys,key)
      samples.append(TrainingSample(context=context,candidate=candidate_feature,label=label))
    return samples