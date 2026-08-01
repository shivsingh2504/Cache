from __future__ import annotations
import dataclasses

import numpy as np
from feature.schema import FeatureVector

class Tensorizer:
  def tensorize(self,sequence: tuple[FeatureVector])->np.ndarray:
    if not sequence:
      raise ValueError("Cannot tensorize an empty sequence")
    rows = [dataclasses.astuple(feature_vector) for feature_vector in sequence]
    return np.ndarray(rows, dtype=np.float32)