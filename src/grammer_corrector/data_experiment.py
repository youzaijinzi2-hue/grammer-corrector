import argparse
import random
from pathlib import Path

import torch
from transformer.model import Transformer

from grammer_corrector.batching import make_batch
from grammer_corrector.checkpoint import save_checkpoint
from grammer_corrector.data import encode_pairs, generate_agreement_pairs
from grammer_corrector.evaluation import AGREEMENT_EVALUATION_PAIRS, evaluate_pairs
from grammer_corrector.tokenizer import build_vocabulary, encode
from grammer_corrector.training import train

EXTRA_EVALUATION = (
    ("He walk to work.", "He walks to work."),
    ("She read at home.", "She reads at home."),
    ("My sister work outside.", "My sister works outside."),
    ("The teacher cook at school.", "The teacher cooks at school."),
    ("My brother walks to work.", "My brother walks to work."),
    ("The student reads at home.", "The student reads at home."),
    ("The teacher works outside.", "The teacher works outside."),
    ("My sister cooks at school.", "My sister cooks at school."),
)


def training_pool() -> list[tuple[str, str]]:
    baseline = generate_agreement_pairs(expanded=True)
    held_out = {text for pair in (*AGREEMENT_EVALUATION_PAIRS, *EXTRA_EVALUATION) for text in pair}
    subjects = ("He", "She", "My sister", "My brother", "The teacher", "The student",
                "Her friend", "His father")
    phrases = (
        ("go", "goes", "to school"), ("go", "goes", "home"),
        ("go", "goes", "outside"), ("study", "studies", "at home"),
        ("study", "studies", "at school"), ("study", "studies", "outside"),
        ("play", "plays", "outside"), ("play", "plays", "at school"),
        ("play", "plays", "at home"), ("walk", "walks", "to work"),
        ("walk", "walks", "to school"), ("read", "reads", "at home"),
        ("read", "reads", "outside"), ("work", "works", "outside"),
        ("work", "works", "at school"), ("cook", "cooks", "at school"),
        ("cook", "cooks", "at home"),
    )
    extras = []
    for subject in subjects:
        for base, third, complement in phrases:
            pair = (f"{subject} {base} {complement}.", f"{subject} {third} {complement}.")
            if pair not in baseline and held_out.isdisjoint(pair):
                extras.append(pair)
    random.Random(42).shuffle(extras)
    return baseline + extras


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
