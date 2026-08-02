

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from feature.schema import FeatureVector

from model_input.builder import SequenceBuilder
from model_input.config import SequenceConfig
from model_input.tensorizer import Tensorizer


class ModelInputPipeline:
    def __init__(self, config: SequenceConfig) -> None:
        self._builder = SequenceBuilder(config)
        self._tensorizer = Tensorizer()

    def run(self, feature_stream: Iterator[FeatureVector]) -> Iterator[np.ndarray]:
        for sequence in self._builder.build(feature_stream):
            yield self._tensorizer.tensorize(sequence)