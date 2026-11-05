import torch
from transformer.model import Transformer

from grammer_corrector.batching import make_batch
from grammer_corrector.checkpoint import load_checkpoint, save_checkpoint
from grammer_corrector.data import encode_pairs, generate_agreement_pairs
from grammer_corrector.inference import greedy_decode
from grammer_corrector.tokenizer import build_vocabulary, decode
from grammer_corrector.training import train


def test_greedy_decode_reproduces_a_training_sentence_and_stops(tmp_path):
    pairs = generate_agreement_pairs()
    vocabulary = build_vocabulary([text for pair in pairs for text in pair])
    encoded_pairs = encode_pairs(pairs, vocabulary)
    pad_id = vocabulary["<pad>"]
    batch = make_batch(encoded_pairs, pad_id)
    model_config = dict(
        src_vocab_size=len(vocabulary), tgt_vocab_size=len(vocabulary),
        d_model=16, num_heads=2, d_ff=32, num_layers=1, dropout=0.0,
        max_len=16, src_pad_id=pad_id, tgt_pad_id=pad_id,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = Transformer(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    train(model, optimizer, batch, epochs=100)
    source_ids, target_ids = encoded_pairs[0]

    generated = greedy_decode(model, source_ids, vocabulary, max_new_tokens=12)

    assert generated == target_ids
    assert decode(generated, vocabulary) == "He goes to school."
    assert greedy_decode(model, source_ids, vocabulary, max_new_tokens=2) == target_ids[:3]

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, model_config, vocabulary)
    restored_model, restored_vocabulary = load_checkpoint(path)

    assert restored_vocabulary == vocabulary
    assert not restored_model.training
    restored_ids = greedy_decode(restored_model, source_ids, restored_vocabulary, 12)
    assert restored_ids == generated
