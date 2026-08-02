from feature.state import CacheState
from model_input.schema import CandidateFeatures

class CandidateFeatureBuilder:
  def __init__(self,state:CacheState)->None:
    self.state  = state
    
  def build(self,key:int)->CandidateFeatures:
    if not self.state.has_seen(key):
      raise ValueError(f"Cannot build candidate features for unseen key: {key}")
    now = self.state.last_event_timestamp()
    assert now is not None
    
    last_access = self.state.last_access_of(key)
    first_access = self.state.first_access_of(key)
    
    return CandidateFeatures(key=key,frequency=self.state.frequency_of(key),recency=now-last_access,key_age=now-first_access)
  
  