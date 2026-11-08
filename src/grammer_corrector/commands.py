import argparse
from pathlib import Path

import torch

from grammer_corrector.checkpoint import load_checkpoint, save_checkpoint
from grammer_corrector.data import load_pairs
from grammer_corrector.evaluation import evaluate_pairs
from grammer_corrector.tokenizer import encode
from grammer_corrector.training import fit_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="grammer-corrector")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train a model and save a checkpoint")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--include-identity", action="store_true", help="Add target-to-target pairs")
    evaluation = commands.add_parser("evaluate", help="Evaluate a saved model")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)

    try:
        pairs = load_pairs(args.data)
        if args.command == "train":
            if args.epochs < 1:
                training.error("--epochs must be at least 1")
            if args.include_identity:
                existing = set(pairs)
                identity_pairs = dict.fromkeys((target, target) for _, target in pairs)
                pairs.extend(pair for pair in identity_pairs if pair not in existing)
            model, config, vocabulary = fit_model(pairs, epochs=args.epochs, seed=args.seed)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(args.output, model, config, vocabulary)
            print(f"Saved checkpoint: {args.output} ({len(pairs)} training pairs)")
            return

        model, vocabulary = load_checkpoint(args.checkpoint)
        source_limit = model.encoder.embedding.position.pe.shape[1]
        if any(len(encode(source, vocabulary)) > source_limit for source, _ in pairs):
            evaluation.error(f"Evaluation inputs must fit within {source_limit} tokens including markers.")
        generation_limit = model.decoder.embedding.position.pe.shape[1]
        results = evaluate_pairs(model, pairs, vocabulary, generation_limit)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"exact match: {sum(result.exact_match for result in results)}/{len(results)}")
    for label, identity in (("correction", False), ("retention", True)):
        group = [result for result in results if (result.source == result.expected) == identity]
        unknown = sum(vocabulary["<unk>"] in encode(result.source, vocabulary) for result in group)
        print(f"{label}: {sum(result.exact_match for result in group)}/{len(group)} (unknown inputs: {unknown})")
