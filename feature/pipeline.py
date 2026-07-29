from collections.abc import Iterator
from feature.extractor import FeatureExtractor
from feature.schema import FeatureVector
from feature.state import CacheState
from simulator.schema import AccessEvent

def extract_features(events:Iterator[AccessEvent],state:CacheState | None = None,)->Iterator[FeatureVector]:
  extractor = FeatureExtractor(state=state)
  for event in events:
    yield extractor.extract(event)
    