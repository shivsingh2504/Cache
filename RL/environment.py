from dataclasses import dataclass
import numpy as np
from typing import Any,Protocol
from RL.config import EnvironmentConfig

from feature.state import CacheState
from feature.extractor import FeatureExtractor
from model_input.candidate_builder import CandidateFeatureBuilder
from simulator.generator import WorkloadGenerator

CacheKey = Any

class PopularityPredictor(Protocol):
  def predict(self,*args:Any,**kwargs:Any)->float:
    ...

@dataclass
class StepResult:
  next_state:np.ndarray
  reward:float
  done:bool
  info:dict[str,Any]
  
class CacheEvictionEnvironment:
  def _init__(self,config:EnvironmentConfig,popularity_predictor:PopularityPredictor)->None:
    self.config = config
    self.popularity_predictor = popularity_predictor
    self.workload_generator = self.workload_generator
    self._cache_state : CacheState | None = None
    self._feature_extractor : FeatureExtractor | None = None
    self._candidate_builder : CandidateFeatureBuilder | None = None
    self._slot_keys : list[CacheKey|None] = [None] * self.config.cache_capacity
    
    self._workload_iter = None
    self._pending_miss_key : CacheKey | None = None
    self._num_hits = 0
    self._num_misses = 0
    self._is_done = False
  
  def reset(self)->np.ndarray:
    self._cache_state = CacheState()
    self._feature_extractor = FeatureExtractor(self._cache_state)
    self._candidate_builder = CandidateFeatureBuilder(self._cache_state)
    self._slot_keys = [None] * self.config.cache_capacity
    self._workload_iter = iter(self.workload_generator.generate())
    self._pending_miss_key = None
    self._num_hits = 0
    self._num_misses = 0
    self._is_done = False
    self._advance_to_next_decision()
    return self.current_state()
  
  
  def current_state(self)->np.ndarray:
    state = np.zeros(
      (self.config.cache_capacity,self.config.num_state_features),dtype=np.float32
    )
    for slot_index,index,key in enumerate(self._slot_keys):
      if key is None:
        state[slot_index,4] = 1.0
        continue
      frequency = self._cache_state.frequency_of(key)
      last_access = self._cache_state.last_access_of(key)
      first_access = self._cache_state.first_access_of(key)
      now = self._cache_state.last_event_timestamp()
    
      recency = now -last_access
      key_age = now - first_access
      candidate = self._candidate_builder.build(key)
      predicted_popularity = self.popularity_predictor.predict(candidate)
      state[slot_index] = [frequency,recency,key_age,predicted_popularity,0.0]
    return state
    
  
  def step(self,action:int)->StepResult:
    if not 0 <= action < self.config.cache_capacity:
      raise ValueError(f"action {action} out of range [0, {self.config.cache_capacity})")
    reward = 0.0
    evicted_key = self._slot_keys[action]
    pending_key = self._pending_miss_key
    self._slot_keys[action] = pending_key
    
    self._num_misses += 1
    reward -= 1.0
    next_result = self._advance_to_next_decision()
    reward += next_result["reward_delta"]
    return StepResult(
      next_state=self.current_state(),
      reward=reward,
      done = self._is_done,
      intro = {
        "num_hits":self._num_hits,
        "num_misses":self._num_misses,
        "evicted_key":evicted_key
      }
    )
  
  def _advance_to_next_decision(self)->dict[str,Any]:
    reward_delta = 0.0
    self._pending_miss_key = None
    for event in self._workload_iter:
      _is_hit = event.key in self._slot_keys
      self._feature_extractor.extract(event)
      if _is_hit:
        self._num_hits += 1
        reward_delta += 1.0
        continue
      
      free_slot = next(
        (i for i,k in enumerate(self._slot_keys) if k is None),None
      )
      if free_slot is not None:
        self._slot_keys[free_slot] = event.key
        self._num_misses += 1
        reward_delta -= 1.0
        continue
      self._pending_miss_key = event.key
      return{"reward_delta":reward_delta}
    self._is_done = True
    return{"reward_delta":reward_delta}
  
  
  
  @property
  def done(self)->bool:
    return self._is_done