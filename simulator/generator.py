from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterator
from pathlib import Path
from .config import DistributionType, WorkloadConfig
from .schema import AccessEvent

def __init__(self,config:WorkloadConfig)->None:
  self.config = config
  self._rng : np.random.Generator = np.random.default_rng(config.seed)
  self._num_keys : int = config.num_keys
  self._rank_prob : np.ndarray | None = None
  self._rank_to_key : np.ndarray | None = None
  self._request_until_drift : int = config.drift_interval
  if config.distribution in(DistributionType.ZIPFIAN,DistributionType.HOTSPOT_DRIFT):
    self._rank_prob = self.compute_zipf_probabilities(num_keys=self._num_keys,alpha=config.zipf_alpha)
  if config.distribution is DistributionType.HOTSPOT_DRIFT:
    self._rank_to_key = np.arange(self._num_keys)
  self._burst_remaining : int = 0
  self._burst_key : int = 0
  self._burst_enabled : bool = config.burst_prob > 0.0
  
def generate(self) -> Iterator[AccessEvent]:
  total_requests = config._num_request
  produced = 0
  remaining = 0
  while produced < total_requests:
    remaining = total_requests - produced
    chunk_size = self._next_chunk_size(remaining)
    key_chunk = self._sample_base_keys(chunk_size)
    if self._bursts_enabled:
      self._inject_bursts(key_chunk)
    for key in key_chunk:
      yield AccessEvent(timestamp = timestamp , key = int(key))
      timestamp+=1
    produced += chunk_size
    self._advance_drift_state(chunk_size)
    