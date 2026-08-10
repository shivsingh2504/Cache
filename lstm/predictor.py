import torch
from collections.abc import Sequence

from feature.schema import FeatureVector
from feature.state import CacheState
from lstm.config import ModelConfig
from lstm.model import PopularityPredictor
from model_input.candidate_builder import CandidateFeatureBuilder
from model_input.tensorizer import Tensorizer, candidate_to_array


class Predictor:
    def __init__(
        self,
        model: PopularityPredictor,
        state: CacheState,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.candidate_builder = CandidateFeatureBuilder(state)
        self.tensorizer = Tensorizer()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config: ModelConfig,
        state: CacheState,
        device: str = "cpu",
    ) -> "Predictor":
        model = PopularityPredictor(model_config)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        model.eval()

        return cls(
            model=model,
            state=state,
            device=device,
        )

    def set_state(self, state: CacheState) -> None:
        self.candidate_builder = CandidateFeatureBuilder(state)

    @torch.no_grad()
    def score(
        self,
        feature_window: Sequence[FeatureVector],
        candidate_keys: list[int],
    ) -> dict[int, float]:

        if not candidate_keys:
            return {}

        if not feature_window:
            return {key: 0.0 for key in candidate_keys}

        context = self.tensorizer.tensorize(tuple(feature_window))

        context_tensor = (
            torch.from_numpy(context)
            .float()
            .unsqueeze(0)
            .expand(len(candidate_keys), -1, -1)
            .to(self.device)
        )

        candidate_features = [
            self.candidate_builder.build(key)
            for key in candidate_keys
        ]

        candidate_tensor = torch.stack(
            [
                torch.from_numpy(candidate_to_array(feature)).float()
                for feature in candidate_features
            ]
        ).to(self.device)

        predictions = self.model(context_tensor, candidate_tensor)
        predictions = predictions.expm1()

        return {
            key: prediction.item()
            for key, prediction in zip(candidate_keys, predictions)
        }