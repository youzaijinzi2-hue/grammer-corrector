from pathlib import Path

import torch
from transformer.model import Transformer


def save_checkpoint(
    path: Path,
    model: Transformer,
    model_config: dict[str, int | float],
    vocabulary: dict[str, int],
) -> None:
    """Save weights and the configuration used to construct the model."""
    torch.save(
        {
            "version": 1,
            "model_config": model_config,
            "vocabulary": vocabulary,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[Transformer, dict[str, int]]:
    """Restore a model and its vocabulary for CPU inference."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["version"] != 1:
        raise ValueError("Unsupported checkpoint version.")

    model = Transformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["vocabulary"]
