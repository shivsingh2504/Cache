from __future__ import annotations
from collections.abc import Iterator
from feature.schema import FeatureVector
from collections import deque
from model_input.config import SequenceConfig

class SequenceBuilder:
  
  def __init__(self,config:SequenceConfig)->None:
    
    self._config = config
    
  def build(self,feature_stream: Iterator[FeatureVector])-> Iterator[tuple[FeatureVector]]:
    window_size = self._config.window_size
    buffer : deque[FeatureVector] = deque(maxlen=window_size)
    
    for feature_vector in feature_stream:
      buffer.append(feature_vector)
      if len(buffer)==window_size:
        yield tuple(buffer)