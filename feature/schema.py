from dataclasses import dataclass

@dataclass(slots=True,frozen=True)
class FeatureVector:
  timestamp : int
  key : int
  frequency : int
  key_age: int
  recency : int | None
  _is_first_access : bool
  inter_arrival_time : int | None
  first_access_time : int