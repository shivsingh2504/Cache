from dataclasses import dataclass
import numpy as np
from typing import Any,Protocol
from RL.config import EnvironmentConfig

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
  
  def reset(self)->np.ndarray:
    raise NotImplementedError
  
  def current_state(self)->np.ndarray:
    raise NotImplementedError
  
  def step(self,action:int)->StepResult:
    raise NotImplementedError
  
  @property
  def done(self)->bool:
    raise NotImplementedError