from dataclasses import dataclass

from transformer.model import Transformer

from grammer_corrector.inference import greedy_decode
from grammer_corrector.tokenizer import decode, encode


# Familiar words in combinations absent from the six training pairs.
AGREEMENT_EVALUATION_PAIRS = (
    ("He go outside.", "He goes outside."),
    ("She study outside.", "She studies outside."),
    ("He goes outside.", "He goes outside."),
    ("She plays at home.", "She plays at home."),
)


@dataclass(frozen=True)
class EvaluationResult:
    source: str
    expected: str
    prediction: str

    @property
    def exact_match(self) -> bool:
        return self.prediction == self.expected


def evaluate_pairs(
    model: Transformer,
    pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    vocabulary: dict[str, int],
    max_new_tokens: int,
) -> list[EvaluationResult]:
    """Generate a correction for each source and retain its expected text."""
    results = []
    for source, expected in pairs:
        source_ids = encode(source, vocabulary)
        generated_ids = greedy_decode(model, source_ids, vocabulary, max_new_tokens)
        results.append(EvaluationResult(source, expected, decode(generated_ids, vocabulary)))
    return results
