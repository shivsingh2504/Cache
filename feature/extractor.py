from feature.schema import FeatureVector
from feature.state import CacheState
from simulator.schema import AccessEvent

class FeatureExtractor:
  def __init__(self,state:CacheState | None = None)->None:
    self.state : CacheState = state if state is not None else CacheState()
    
  def extract(self,event:AccessEvent)->FeatureVector:
    
    _is_first_access = not self.state.has_seen(event.key)
    frequency = self._compute_frequency(event.key)
    recency = self._compute_recency(event)
    first_access_time  = self._compute_first_access_time(event)
    key_age = self._compute_key_age(event,first_access_time)
    inter_arrival_time = self._compute_inter_arrival_time(event)
    
    self.state.apply_event(event.key,event.timestamp)
    
    return FeatureVector(
      timestamp=event.timestamp,
      key=event.key,
      frequency=frequency,
      recency=recency,
      first_access_time=first_access_time,
      key_age=key_age,
      inter_arrival_time=inter_arrival_time,
      _is_first_access = _is_first_access,
    )
    