import random
from collections import deque
from dataclasses import dataclass

from feature.state import CacheState
from lstm.config import DatasetConfig
from model_input.candidate_builder import CandidateFeatureBuilder
from model_input.schema import TrainingSample

