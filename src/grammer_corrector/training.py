import torch
from torch.nn.functional import cross_entropy
from transformer.model import Transformer


def train_step(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    """Update the model once and return mean loss over non-padding labels."""
    source, decoder_input, labels = batch
    model.train()
    optimizer.zero_grad()
    logits = model(source, decoder_input)
    loss = cross_entropy(logits.transpose(1, 2), labels, ignore_index=model.tgt_pad_id)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    epochs: int,
) -> list[float]:
    """Train on one full batch per epoch and return the loss history."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    losses = []
    for _ in range(epochs):
        losses.append(train_step(model, optimizer, batch))
    return losses
