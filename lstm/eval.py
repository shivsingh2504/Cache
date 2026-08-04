import random

import torch

from lstm.config import ModelConfig
from lstm.model import PopularityPredictor
from model_input.schema import TrainingSample


def spot_check(
    model_path: str,
    model_config: ModelConfig,
    val_samples: list[TrainingSample],
    n: int = 30,
    device: str = "cpu",
    seed: int = 7,
) -> None:
    """Loads a checkpoint and prints true vs. predicted popularity on a
    random sample of validation examples, with a bias toward including
    non-zero-label examples — since with 93%+ zero labels, a random
    sample would mostly show trivial zero-vs-near-zero comparisons and
    hide whether the model discriminates the tail that actually matters."""

    model = PopularityPredictor(model_config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    rng = random.Random(seed)

    nonzero = [s for s in val_samples if s.label > 0]
    zero = [s for s in val_samples if s.label == 0]

    n_nonzero = min(len(nonzero), n // 2)
    n_zero = min(len(zero), n - n_nonzero)

    chosen = rng.sample(nonzero, n_nonzero) + rng.sample(zero, n_zero)
    rng.shuffle(chosen)

    contexts = torch.stack([torch.as_tensor(s.context, dtype=torch.float32) for s in chosen]).to(device)
    candidates = torch.stack([
        torch.tensor(
            [s.candidate.frequency, s.candidate.recency, s.candidate.key_age],
            dtype=torch.float32,
        )
        for s in chosen
    ]).to(device)

    with torch.no_grad():
        raw_predictions = model(contexts, candidates)
        predictions = raw_predictions.expm1()  # invert log1p — matches predictor.py's inversion

    print(f"{'true':>8} {'pred':>10}")
    for s, pred in zip(chosen, predictions):
        print(f"{s.label:8.2f} {pred.item():10.3f}")

    true_vals = torch.tensor([s.label for s in chosen])
    pred_vals = predictions.cpu()
    if true_vals.std() > 0 and pred_vals.std() > 0:
        corr = torch.corrcoef(torch.stack([true_vals, pred_vals]))[0, 1].item()
        print(f"\nPearson correlation (true vs pred), this sample: {corr:.3f}")
    else:
        print("\ncould not compute correlation (zero variance in true or pred)")


if __name__ == "__main__":
    import pickle

    model_config = ModelConfig(event_features=8, candidate_features=3)

    with open("checkpoints/val_samples.pkl", "rb") as f:
        val_samples = pickle.load(f)

    spot_check(
        model_path="checkpoints/best_model.pt",
        model_config=model_config,
        val_samples=val_samples,
    )