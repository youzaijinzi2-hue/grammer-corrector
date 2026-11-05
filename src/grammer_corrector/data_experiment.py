import argparse
from pathlib import Path

import torch
from transformer.model import Transformer

from grammer_corrector.batching import make_batch
from grammer_corrector.checkpoint import save_checkpoint
from grammer_corrector.data import encode_pairs, load_agreement_dataset
from grammer_corrector.evaluation import (
    AGREEMENT_EVALUATION_PAIRS,
    EVALUATION_PAIRS,
    evaluate_pairs,
)
from grammer_corrector.tokenizer import build_vocabulary, encode
from grammer_corrector.training import train

EXTRA_EVALUATION = EVALUATION_PAIRS[len(AGREEMENT_EVALUATION_PAIRS):]


def training_pool() -> list[tuple[str, str]]:
    return load_agreement_dataset("train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three agreement training sets.")
    parser.add_argument("--save-dir", type=Path, help="Directory for model checkpoints")
    args = parser.parse_args()
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    epochs = 100
    torch.set_num_threads(1)
    pool = training_pool()
    for size in (12, 50, 100):
        corrections = pool[:size]
        pairs = corrections + [(target, target) for _, target in corrections]
        vocabulary = build_vocabulary([text for pair in pairs for text in pair])
        pad_id = vocabulary["<pad>"]
        torch.manual_seed(seed)
        model_config = dict(
            src_vocab_size=len(vocabulary), tgt_vocab_size=len(vocabulary),
            d_model=16, num_heads=2,
            d_ff=32, num_layers=1, dropout=0.0, max_len=16,
            src_pad_id=pad_id, tgt_pad_id=pad_id,
        )
        model = Transformer(**model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        train(model, optimizer, make_batch(encode_pairs(pairs, vocabulary), pad_id), epochs)
        if args.save_dir is not None:
            filename = f"{size}-correction-{size}-identity-seed{seed}-epochs{epochs}.pt"
            path = args.save_dir / filename
            save_checkpoint(path, model, model_config, vocabulary)
            print(f"Saved checkpoint: {path}")
        print(f"{size} correction pairs + {size} identity pairs")
        for name, examples in (("original", AGREEMENT_EVALUATION_PAIRS), ("extra", EXTRA_EVALUATION)):
            results = evaluate_pairs(model, examples, vocabulary, 12)
            for label, identity in (("correction", False), ("retention", True)):
                group = [result for result in results if (result.source == result.expected) == identity]
                matches = sum(result.exact_match for result in group)
                unknown = sum(vocabulary["<unk>"] in encode(result.source, vocabulary) for result in group)
                print(f"  {name} {label}: {matches}/{len(group)} (unknown inputs: {unknown})")


if __name__ == "__main__":
    main()
