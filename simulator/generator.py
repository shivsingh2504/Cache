from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterator
from pathlib import Path
from .config import DistributionType, WorkloadConfig
from .schema import AccessEvent
class WorkloadGenerator:
  _DEFAULT_CHUNK_SIZE=10000
  def __init__(self,config:WorkloadConfig)->None:
    self.config = config
    self._rng : np.random.Generator = np.random.default_rng(config.seed)
    self._num_keys : int = config.num_keys
    self._rank_prob : np.ndarray | None = None
    self._rank_to_key : np.ndarray | None = None
    self._request_until_drift : int = config.drift_interval
    if config.distribution in(DistributionType.ZIPFIAN,DistributionType.HOTSPOT_DRIFT):
      self._rank_prob = self._compute_zipf_prob(num_keys=self._num_keys,alpha=config.zipf_alpha)
    if config.distribution is DistributionType.HOTSPOT_DRIFT:
      self._rank_to_key = np.arange(self._num_keys)
    self._burst_remaining : int = 0
    self._burst_key : int = 0
    self._bursts_enabled : bool = config.burst_prob > 0.0
  
  def generate(self) -> Iterator[AccessEvent]:
    total_requests = self.config.num_requests
    produced = 0
    timestamp = 0

    while produced < total_requests:
        remaining = total_requests - produced
        chunk_size = self._next_chunk_size(remaining)
        keys_chunk = self._sample_base_keys(chunk_size)

        if self._bursts_enabled:
            self._inject_bursts(keys_chunk)

        for key in keys_chunk:
            yield AccessEvent(timestamp=timestamp, key=int(key))
            timestamp += 1

        produced += chunk_size
        self._advance_drift_state(chunk_size)
    
  def to_dataframe(self) -> pd.DataFrame:
    num_requests = self.config.num_requests
    timestamp = np.empty(num_requests,dtype=np.int64)
    key = np.empty(num_requests,dtype=np.int64)
    for index, event in enumerate(self.generate()):
      timestamp[index] = event.timestamp
      key[index] = event.key
    return pd.DataFrame({"timestamp": timestamp, "key": key})
  
  def save(self,path:Path)->None:
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    self.to_dataframe().to_parquet(path,index=False)
  
  def _next_chunk_size(self,remaining:int)->int:
    chunk_size = min(self._DEFAULT_CHUNK_SIZE,remaining)
    if self.config.distribution is DistributionType.HOTSPOT_DRIFT:
      chunk_size = min(chunk_size,self._request_until_drift)
    return chunk_size
  
  def _sample_base_keys(self,size:int)->np.ndarray:
    distribution = self.config.distribution
    if distribution is DistributionType.UNIFORM:
      return self._rng.integers(0,self._num_keys,size=size)
    if distribution is DistributionType.ZIPFIAN:
      assert self._rank_prob is not None
      return self._rng.choice(self._num_keys,size=size,p=self._rank_prob)
    assert self._rank_prob is not None
    assert self._num_keys is not None
    ranks = self._rng.choice(self._num_keys, size=size, p=self._rank_prob)
    return self._rank_to_key[ranks]
  
  def _inject_bursts(self,keys:np.ndarray)->None:
    burst_prob = self.config.burst_prob
    low , high = self.config.burst_length_range
    
    for index in range(len(keys)):
      if self._burst_remaining > 0:
        keys[index] = self._burst_key
        self._burst_remaining -= 1
        continue
      
      if self._rng.random() < burst_prob:
        self._burst_key = int(keys[index])
        burst_length = int(self._rng.integers(low,high+1))
        keys[index] = self._burst_key
        self._burst_remaining = burst_length - 1
  
  def _advance_drift_state(self,chunk_size:int)->None:
    if self.config.distribution is not DistributionType.HOTSPOT_DRIFT:
      return
    self._request_until_drift -= chunk_size
    if(self._request_until_drift <= 0):
      self._drift_hotspot()
      self._request_until_drift = self.config.drift_interval
      
  def _drift_hotspot(self)->None:
    assert self._rank_to_key is not None
    shift = max(1,round(self.config.drift_fraction * self._num_keys))
    self._rank_to_key = np.roll(self._rank_to_key,shift)
    
  @staticmethod
  def _compute_zipf_prob(num_keys:int , alpha:float)->np.ndarray:
    ranks = np.arange(1,num_keys+1,dtype=np.float64)
    weights = ranks ** (-alpha)
    return weights / weights.sum()