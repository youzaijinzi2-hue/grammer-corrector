import pytest
import torch
from torch.nn.functional import cross_entropy
from transformer.model import Transformer

from grammer_corrector.batching import make_batch
from grammer_corrector.data import encode_pairs, generate_agreement_pairs
from grammer_corrector.tokenizer import build_vocabulary
from grammer_corrector.training import train, train_step


def test_train_step_ignores_padding_and_updates_transformer():
    pairs = generate_agreement_pairs()
    vocabulary = build_vocabulary([text for pair in pairs for text in pair])
    pad_id = vocabulary["<pad>"]
    batch = make_batch(encode_pairs(pairs, vocabulary), pad_id)
    model = Transformer(
        len(vocabulary), len(vocabulary), d_model=16, num_heads=2,
        d_ff=32, num_layers=1, dropout=0.0, max_len=16,
        src_pad_id=pad_id, tgt_pad_id=pad_id,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    source, decoder_input, labels = batch
    valid = labels != pad_id
    assert (~valid).any()
    with torch.no_grad():
        expected_loss = cross_entropy(model(source, decoder_input)[valid], labels[valid]).item()
        initial_weights = model.output_projection.weight.clone()

    loss = train_step(model, optimizer, batch)

    assert loss == pytest.approx(expected_loss)
    assert not torch.equal(model.output_projection.weight, initial_weights)


def test_train_reduces_loss_on_six_agreement_pairs():
    pairs = generate_agreement_pairs()
    vocabulary = build_vocabulary([text for pair in pairs for text in pair])
    pad_id = vocabulary["<pad>"]
    batch = make_batch(encode_pairs(pairs, vocabulary), pad_id)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = Transformer(
            len(vocabulary), len(vocabulary), d_model=16, num_heads=2,
            d_ff=32, num_layers=1, dropout=0.0, max_len=16,
            src_pad_id=pad_id, tgt_pad_id=pad_id,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    losses = train(model, optimizer, batch, epochs=15)

    assert len(losses) == 15
    assert 0 < losses[-1] < losses[0]
