from dataclasses import dataclass

@dataclass(frozen=True,slots=True)
class DatasetConfig:
  trailing_window : int 
  label_horizon: int
  candidate_per_context : int  = 1