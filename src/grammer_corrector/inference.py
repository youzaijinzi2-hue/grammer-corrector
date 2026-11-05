import torch
from transformer.model import Transformer


@torch.inference_mode()
def greedy_decode(
    model: Transformer,
    source_ids: list[int],
    vocabulary: dict[str, int],
    max_new_tokens: int,
) -> list[int]:
    """Return the generated IDs, including the start and any end marker."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    model.eval()
    device = next(model.parameters()).device
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    generated = [vocabulary["<bos>"]]
    eos_id = vocabulary["<eos>"]
    blocked_ids = [vocabulary["<pad>"], vocabulary["<bos>"]]

    for _ in range(max_new_tokens):
        decoder_input = torch.tensor([generated], dtype=torch.long, device=device)
        scores = model(source, decoder_input)[0, -1].clone()
        scores[blocked_ids] = -torch.inf
        next_id = scores.argmax().item()
        generated.append(next_id)
        if next_id == eos_id:
            break

    return generated
