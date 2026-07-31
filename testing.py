from simulator.config import WorkloadConfig, DistributionType
from simulator.generator import WorkloadGenerator

from feature.pipeline import extract_features


def main():
    config = WorkloadConfig(
        num_keys=5,
        num_requests=15,
        distribution=DistributionType.UNIFORM,
        seed=42,
    )

    generator = WorkloadGenerator(config)

    events = list(generator.generate())

    features = extract_features(iter(events))

    for event, feature in zip(events, features):
        print(event)
        print(feature)
        print("-" * 50)


if __name__ == "__main__":
    main()