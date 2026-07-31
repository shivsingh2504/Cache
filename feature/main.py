from simulator.config import WorkloadConfig, DistributionType
from simulator.generator import WorkloadGenerator

from feature.pipeline import extract_features

config = WorkloadConfig(
    num_keys=5,
    num_requests=15,
    distribution=DistributionType.UNIFORM,
)
generator = WorkloadGenerator(config)

events = generator.generate()
features = extract_features(events)
for feature in features:
    print(feature)
    