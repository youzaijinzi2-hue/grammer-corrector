import re


def tokenize(text: str) -> list[str]:
    """Split English text into words and individual non-whitespace symbols."""
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*|[^\s]", text)


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


def build_vocabulary(texts: list[str]) -> dict[str, int]:
    """Assign stable IDs to special tokens and tokens observed in texts."""
    tokens = {token for text in texts for token in tokenize(text)}
    ordered_tokens = (*SPECIAL_TOKENS, *sorted(tokens))
    return {token: token_id for token_id, token in enumerate(ordered_tokens)}


def encode(text: str, vocabulary: dict[str, int]) -> list[int]:
    """Convert text to token IDs, with start and end markers."""
    unknown_id = vocabulary["<unk>"]
    token_ids = [vocabulary.get(token, unknown_id) for token in tokenize(text)]
    return [vocabulary["<bos>"], *token_ids, vocabulary["<eos>"]]


def decode(token_ids: list[int], vocabulary: dict[str, int]) -> str:
    """Convert token IDs back to text, stopping at the end marker."""
    id_to_token = {token_id: token for token, token_id in vocabulary.items()}
    tokens = []
    for token_id in token_ids:
        token = id_to_token.get(token_id, "<unk>")
        if token == "<eos>":
            break
        if token not in ("<bos>", "<pad>"):
            tokens.append(token)

    return re.sub(r" ([.,!?;:])", r"\1", " ".join(tokens))
