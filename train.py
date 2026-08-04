import random

import torch

from simulator.generator import WorkloadGenerator
from simulator.config import WorkloadConfig, DistributionType
from feature.state import CacheState
from feature.pipeline import extract_features
from model_input.config import SequenceConfig
from model_input.builder import SequenceBuilder
from model_input.tensorizer import Tensorizer
from lstm.config import DatasetConfig, ModelConfig, TrainerConfig
from lstm.dataset import CandidatePool, LabelGenerator, PendingSampleQueue, TrainingSampleBuilder
from lstm.model import PopularityPredictor
from lstm.trainer import Trainer, inspect_label_distribution


def _tap(feature_stream, candidate_pool, pending_queue):
    for fv in feature_stream:
        candidate_pool.observe(fv.key)
        pending_queue.observe_key(fv.key)
        yield fv


def main() -> None:
    workload_config = WorkloadConfig(
        num_keys=5000,
        num_requests=200_000,
        distribution=DistributionType.ZIPFIAN,
        seed=42,
    )  # explicit values — WorkloadConfig's own defaults (num_keys=10, num_requests=10) are toy-sized

    dataset_config = DatasetConfig(trailing_window=1000, label_horizon=50, candidate_per_context=4)
    model_config = ModelConfig(event_features=8, candidate_features=3)  # event_features still unconfirmed — see note below
    trainer_config = TrainerConfig()
    sequence_config = SequenceConfig(window_size=50)  # only window_size confirmed from SequenceBuilder usage — see note below

    workload = WorkloadGenerator(workload_config)
    state = CacheState()

    candidate_pool = CandidatePool(window_size=dataset_config.trailing_window)
    label_generator = LabelGenerator(horizon=dataset_config.label_horizon)
    pending_queue = PendingSampleQueue(horizon=dataset_config.label_horizon)
    rng = random.Random(42)

    sample_builder = TrainingSampleBuilder(
        dataset_config=dataset_config,
        state=state,
        candidate_pool=candidate_pool,
        label_generator=label_generator,
        rng=rng,
    )

    tensorizer = Tensorizer()
    sequence_builder = SequenceBuilder(sequence_config)

    events = workload.generate()
    feature_stream = extract_features(events, state=state)  # shares `state` so TrainingSampleBuilder reads live, correctly-ordered updates
    tapped_stream = _tap(feature_stream, candidate_pool, pending_queue)

    training_samples = []
    for context in sequence_builder.build(tapped_stream):
        context_tensor = tensorizer.tensorize(context)  # matches ModelInputPipeline's own internal call
        candidates = candidate_pool.sample(dataset_config.candidate_per_context, rng)
        pending_queue.enqueue(context_tensor, candidates)

        for ctx, cands, future_keys in pending_queue.ready_batches():
            training_samples.extend(
                sample_builder.build_samples(context=ctx, candidates=cands, future_keys=future_keys)
            )

    stats = inspect_label_distribution(training_samples)
    print("label distribution:", stats)

    model = PopularityPredictor(model_config)
    trainer = Trainer(model=model, config=trainer_config)
    history = trainer.fit(training_samples)

    torch.save(model.state_dict(), "lstm_popularity_predictor.pt")
    print("training complete, final val_loss:", history["val_loss"][-1])


if __name__ == "__main__":
    main()