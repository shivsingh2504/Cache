from __future__ import annotations
import dataclasses

import numpy as np
from feature.schema import FeatureVector
from model_input.schema import CandidateFeatures


class Tensorizer:
    def tensorize(self, sequence: tuple[FeatureVector]) -> np.ndarray:
        if not sequence:
            raise ValueError("Cannot tensorize an empty sequence")
        rows = [
            tuple(0.0 if v is None else v for v in dataclasses.astuple(feature_vector))
            for feature_vector in sequence
        ]
        return np.array(rows, dtype=np.float32)


def candidate_to_array(candidate: CandidateFeatures) -> np.ndarray:
    return np.array(
        [candidate.frequency, candidate.recency, candidate.key_age],
        dtype=np.float32,
    )