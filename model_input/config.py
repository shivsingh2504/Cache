from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True,slots=True)

class  SequenceConfig:
  window_size : int = 50
  
  def __post_init__(self)->None:
    self._validate_window_size()
    
  def _validate_window_size(self)->None:
    if self.window_size <= 0:
      raise ValueError(f"Window size must be a positive integer")
