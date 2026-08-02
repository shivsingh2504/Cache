from dataclasses import dataclass

@dataclass(frozen=True,slots=True)
class CandidateFeatures:
  key: int
  frequency : int
  recency : int
  key_age : int
  
@dataclass(frozen=True,slots=True)
class TrainingSample:
  context : tuple
  candidate: CandidateFeatures
  label : float