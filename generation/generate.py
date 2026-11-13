import torch
import torch.nn.functional as F

from tokenizer.bpe import BPETokenizer
from model.gpt import MiniGPT


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    repetition_penalty=1.1,
    eos_token_id=None,
):
    """
    Generate tokens autoregressively.

    Args:
        model: Trained MiniGPT.
        input_ids: Tensor with shape [1, sequence_length].
        max_new_tokens: Maximum number of new tokens.
        temperature: Controls randomness. Lower means more predictable.
        top_k: Only sample from the top K predictions. None disables it.
        repetition_penalty: Penalizes tokens already generated.
        eos_token_id: Stops generation when this token is produced.
    """

    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("input_ids must have shape [1, sequence_length].")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative.")

    if temperature < 0:
        raise ValueError("temperature cannot be negative.")

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 or None.")

    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive.")

    model.eval()

    generated = input_ids

    for _ in range(max_new_tokens):

        # Keep only the most recent tokens that fit the context window.
        context = generated[:, -model.max_sequence_length:]

        # Forward pass: [1, context_length, vocab_size]
        logits = model(context)

        # Get predictions for the next token.
        next_token_logits = logits[:, -1, :].clone()

        # Repetition penalty.
        if repetition_penalty != 1.0:
            previous_tokens = generated[0].unique()

            for token_id in previous_tokens:
                token_id = int(token_id.item())

                if next_token_logits[0, token_id] < 0:
                    next_token_logits[0, token_id] *= repetition_penalty
                else:
                    next_token_logits[0, token_id] /= repetition_penalty

        # Temperature = 0 means greedy decoding.
        if temperature == 0:
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )
        else:
            next_token_logits = next_token_logits / temperature

            # Top-k filtering.
            if top_k is not None:
                k = min(top_k, next_token_logits.size(-1))

                top_values, _ = torch.topk(
                    next_token_logits,
                    k=k,
                    dim=-1,
                )

                threshold = top_values[:, -1].unsqueeze(-1)

                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold,
                    float("-inf"),
                )

            # Convert logits into probabilities.
            probabilities = F.softmax(
                next_token_logits,
                dim=-1,
            )

            # Sample the next token.
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

        # Append the new token.
        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

        # Stop if EOS is generated.
        if eos_token_id is not None:
            if next_token.item() == eos_token_id:
                break

    return generated