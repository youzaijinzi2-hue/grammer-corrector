import torch
from torch.nn.functional import cross_entropy
from transformer.model import Transformer


from grammer_corrector.batching import make_batch
from grammer_corrector.data import encode_pairs
from grammer_corrector.tokenizer import build_vocabulary

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


def fit_model(pairs: list[tuple[str, str]], *, epochs: int, seed: int, max_len: int = 16):
    vocabulary = build_vocabulary([text for pair in pairs for text in pair])
    pad_id = vocabulary["<pad>"]
    encoded = encode_pairs(pairs, vocabulary)
    if any(len(ids) > max_len for pair in encoded for ids in pair):
        raise ValueError(f"Training sentences must fit within {max_len} tokens including markers.")
    torch.manual_seed(seed)
    model_config = dict(
        src_vocab_size=len(vocabulary), tgt_vocab_size=len(vocabulary),
        d_model=16, num_heads=2,
        d_ff=32, num_layers=1, dropout=0.0, max_len=max_len,
        src_pad_id=pad_id, tgt_pad_id=pad_id,
    )
    model = Transformer(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    train(model, optimizer, make_batch(encoded, pad_id), epochs)
    return model, model_config, vocabulary
