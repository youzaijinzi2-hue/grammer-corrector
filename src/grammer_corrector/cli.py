import argparse
from pathlib import Path

from grammer_corrector.checkpoint import load_checkpoint
from grammer_corrector.inference import greedy_decode
from grammer_corrector.tokenizer import decode, encode


def main() -> None:
    parser = argparse.ArgumentParser(description="Correct English sentences interactively.")
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    try:
        model, vocabulary = load_checkpoint(args.checkpoint)
    except FileNotFoundError:
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    source_limit = model.encoder.embedding.position.pe.shape[1]
    generation_limit = model.decoder.embedding.position.pe.shape[1]
    unknown_id = vocabulary["<unk>"]
    print("Enter one English sentence at a time. Type :quit to exit.")

    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.strip() == ":quit":
            return
        if not text.strip():
            continue

        source_ids = encode(text, vocabulary)
        if unknown_id in source_ids:
            print("Unknown token. Returning the original sentence.")
            print(text)
            continue
        if len(source_ids) > source_limit:
            print("Sentence is too long. Returning the original sentence.")
            print(text)
            continue

        generated = greedy_decode(model, source_ids, vocabulary, generation_limit)
        correction = decode(generated, vocabulary)
        if generated[-1] != vocabulary["<eos>"] or unknown_id in generated or not correction:
            print("No complete correction. Returning the original sentence.")
            print(text)
            continue
        print(correction)


if __name__ == "__main__":
    main()
