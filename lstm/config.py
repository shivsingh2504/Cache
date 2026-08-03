from dataclasses import dataclass

@dataclass(frozen=True,slots=True)
class DatasetConfig:
  trailing_window : int 
  label_horizon: int
  candidate_per_context : int  = 1
  
@dataclass(frozen=True,slots=True)
class ModelConfig:
  event_features : int
  candidate_features : int
  
  lstm_hidden_size : int = 64
  lstm_layers : int = 1
  
  candidate_hidden_size : int = 16
  fusion_hidden_size : int = 32
  
  dropout : float = 0.0
  