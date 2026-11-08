import argparse
from pathlib import Path

import torch

from grammer_corrector.checkpoint import save_checkpoint
from grammer_corrector.data import load_agreement_dataset
from grammer_corrector.evaluation import (
    AGREEMENT_EVALUATION_PAIRS,
    EVALUATION_PAIRS,
    evaluate_pairs,
)
from grammer_corrector.tokenizer import encode
from grammer_corrector.training import fit_model

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
        model, model_config, vocabulary = fit_model(pairs, epochs=epochs, seed=seed)
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
