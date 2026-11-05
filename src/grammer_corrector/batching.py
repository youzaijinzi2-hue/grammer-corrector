import torch
from torch.nn.utils.rnn import pad_sequence


def make_batch(
    pairs: list[tuple[list[int], list[int]]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return encoder inputs, decoder inputs, and next-token labels."""
    if not pairs:
        raise ValueError("A batch must contain at least one pair.")

    sources, targets = zip(*pairs)
    source_batch = pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in sources],
        batch_first=True,
        padding_value=pad_id,
    )
    target_batch = pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in targets],
        batch_first=True,
        padding_value=pad_id,
    )
    return source_batch, target_batch[:, :-1], target_batch[:, 1:]
