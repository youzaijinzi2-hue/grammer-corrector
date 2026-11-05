from grammer_corrector.tokenizer import encode



SUBJECTS = ("He", "She")
VERB_PHRASES = (
    ("go", "goes", "to school"),
    ("study", "studies", "at home"),
    ("play", "plays", "outside"),
)


EXTRA_VERB_PHRASES = (
    ("go", "goes", "home"),
    ("study", "studies", "at school"),
    ("play", "plays", "at school"),
)


def generate_agreement_pairs(
    *, expanded: bool = False, include_identity: bool = False
) -> list[tuple[str, str]]:
    """Return correction pairs and optionally pairs that preserve correct sentences."""
    phrases = VERB_PHRASES + EXTRA_VERB_PHRASES if expanded else VERB_PHRASES
    pairs = []
    for subject in SUBJECTS:
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
