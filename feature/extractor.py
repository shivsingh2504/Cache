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
  
  def _compute_frequency(self,key:int)->int:
    return self.state.frequency_of(key)
  
  def _compute_recency(self,event:AccessEvent)->int | None:
    last_seen = self.state.last_access_of(event.key)
    if last_seen is None:
      return None
    return event.timestamp - last_seen
  
  def _compute_first_access_time(self,event:AccessEvent)->int:
    first_seen  = self.state.first_access_of(event.key)
    return first_seen if first_seen is not None else event.timestamp
  
  def _compute_key_age(self,event:AccessEvent,first_access_time:int)->int:
    return event.timestamp - first_access_time
  
  def _compute_inter_arrival_time(self,event:AccessEvent)->int | None:
    last_event_ts = self.state.last_event_timestamp()
    if last_event_ts is None:
      return None
    return event.timestamp - last_event_ts
    