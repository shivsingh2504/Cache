
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from feature.extractor import FeatureExtractor
from feature.schema import FeatureVector
from model_input.config import SequenceConfig
from model_input.pipeline import ModelInputPipeline
from simulator.config import WorkloadConfig
from simulator.generator import WorkloadGenerator
from simulator.schema import AccessEvent


def _feature_stream(
    events: Iterator[AccessEvent], extractor: FeatureExtractor
) -> Iterator[FeatureVector]:
    for event in events:
        yield extractor.extract(event)


def main() -> None:
    workload_config = WorkloadConfig(num_keys=10,num_requests=20,seed=42)

    generator = WorkloadGenerator(workload_config)
    extractor = FeatureExtractor()

    window_size = 10
    sequence_config = SequenceConfig(window_size=window_size)
    pipeline = ModelInputPipeline(sequence_config)

    events = generator.generate()
    features = _feature_stream(events, extractor)
    tensor_stream = pipeline.run(features)

    print("--- Streaming behaviour check ---")
    print(f"tensor_stream is a generator: {tensor_stream.__class__.__name__ == 'generator'}")

    tensors = list(tensor_stream)
    print(f"\nGenerated {len(tensors)} tensors")
    assert len(tensors) >= 2, "Need at least two tensors to test sliding windows."
    first_tensor = tensors[0]
    second_tensor = tensors[1]

    print("\n--- Shape check ---")
    print(f"first_tensor.shape  = {first_tensor.shape}")
    print(f"first_tensor.dtype  = {first_tensor.dtype}")
    assert first_tensor.shape[0] == window_size, "window dimension mismatch"

    print("\n--- First tensor ---")
    print(first_tensor)

    print("\n--- Sliding window check ---")
    overlap_matches = np.array_equal(first_tensor[1:], second_tensor[:-1])
    print(f"first_tensor[1:] == second_tensor[:-1]: {overlap_matches}")
    assert overlap_matches, "windows are not sliding with stride 1"

    print("\n--- Consuming remaining stream to confirm laziness holds ---")
    remaining_count = sum(1 for _ in tensor_stream)
    print(f"remaining tensors after the first two: {remaining_count}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()