from dataclasses import dataclass,field

@dataclass(slots=True)
class CacheState:
  frequency : dict[int,int] = field(default_factory=dict)
  first_access_time : dict[int,int] = field(default_factory=dict)
  last_access_time : dict[int,int] = field(default_factory=dict)
  _last_event_timestamp : int | None = field(default=None, repr=None)
  
  def has_seen(self,key:int)->bool:
    return key in self.frequency
  
  def frequency_of(self,key:int)->int:
    return self.frequency.get(key,0)
  
  def last_access_of(self,key:int)->int | None:
    return self.last_access_time.get(key)
  
  def first_access_of(self,key:int)->int|None:
    return self.first_access_time.get(key)
  
  def last_event_timestamp(self)-> int | None:
    return self._last_event_timestamp
  
  def apply_event(self, key:int, timestamp:int)->None:
    if key not in self.first_access_time:
      self.first_access_time[key] = timestamp
    self.frequency[key] = self.frequency.get(key,0) + 1
    self.last_access_time[key] = timestamp
    self._last_event_timestamp = timestamp
    