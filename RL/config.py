from dataclasses import dataclass

@dataclass
class EnvironmentConfig:
  cache_capacity : int = 16
  num_state_features : int = 5
  
@dataclass
class ReplayBufferConfig:
  capacity : int =100_000
  batch_size : int = 64
  
@dataclass
class AgentConfig:
  gamma : float = 0.99
  learning_rate : float = 1e-4
  epsilon_start : 1.0
  epsilon_end: float = 0.05
  epsilon_decay_steps: int = 50_000
  target_update_interval : int = 1_000
  
@dataclass
class TrainerConfig:
  num_training_steps : int = 200_000
  warmup_steps : int = 5_000
  eval_interval : int = 10_000
  log_interval : int = 1_000
  