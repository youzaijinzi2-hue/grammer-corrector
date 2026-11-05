import pytest
import torch

from grammer_corrector.batching import make_batch


def test_pads_sources_and_targets_and_shifts_targets():
    pairs = [
        ([1, 5, 6, 2], [1, 5, 7, 2]),
        ([1, 8, 2], [1, 8, 9, 10, 2]),
    ]

    source, decoder, labels = make_batch(pairs, pad_id=0)

    assert source.tolist() == [[1, 5, 6, 2], [1, 8, 2, 0]]
    assert decoder.tolist() == [[1, 5, 7, 2], [1, 8, 9, 10]]
    assert labels.tolist() == [[5, 7, 2, 0], [8, 9, 10, 2]]
    assert source.dtype == decoder.dtype == labels.dtype == torch.long


def test_rejects_an_empty_batch():
    with pytest.raises(ValueError, match="at least one pair"):
        make_batch([], pad_id=0)
