import json
from importlib.resources import files

from grammer_corrector.tokenizer import encode


def load_agreement_materials() -> dict:
    resource = files("grammer_corrector").joinpath("resources", "agreement.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_agreement_dataset(split: str) -> list[tuple[str, str]]:
    """Read source and target sentences in their recorded order."""
    if split not in ("train", "eval"):
        raise ValueError(f"Unknown agreement dataset split: {split}")
    resource = files("grammer_corrector").joinpath(
        "resources", "datasets", "agreement", f"{split}.jsonl"
    )
    pairs = []
    with resource.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            pairs.append((row["source"], row["target"]))
    return pairs


def generate_agreement_pairs(
    *, expanded: bool = False, include_identity: bool = False
) -> list[tuple[str, str]]:
    """Return correction pairs and optionally pairs that preserve correct sentences."""
    materials = load_agreement_materials()
    groups = ("basic", "expanded") if expanded else ("basic",)
    phrases = [
        (base, third_person, complement)
        for group in groups
        for base, third_person, complement, phrase_group in materials["verb_phrases"]
        if phrase_group == group
    ]
    pairs = []
    for subject in materials["subjects"]:
        for base, third_person, complement in phrases:
            source = f"{subject} {base} {complement}."
            target = f"{subject} {third_person} {complement}."
            pairs.append((source, target))
    if include_identity:
        pairs += [(target, target) for _, target in pairs]
    return pairs


def encode_pairs(
    pairs: list[tuple[str, str]], vocabulary: dict[str, int]
) -> list[tuple[list[int], list[int]]]:
    """Encode each source and target sentence with the same vocabulary."""
    return [(encode(source, vocabulary), encode(target, vocabulary)) for source, target in pairs]
