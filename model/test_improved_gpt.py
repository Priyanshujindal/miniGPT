
import torch

from model.gpt import MiniGPT


def main():

    vocab_size = 513

    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=256,
        num_heads=4,
        num_layers=4,
        max_sequence_length=128,
        dropout=0.1,
    )

    model.train()

    input_ids = torch.randint(
        0,
        vocab_size,
        (2, 64),
    )

    # Forward pass
    logits = model(input_ids)

    assert logits.shape == (2, 64, vocab_size)
    assert torch.isfinite(logits).all()

    # Verify weight tying.
    embedding_weight = (
        model.embeddings.token_embedding.weight
    )

    assert embedding_weight is model.embeddings.token_embedding.weight

    # Check that there is no independent output-head parameter.
    assert not hasattr(model, "output_head")

    # Backpropagation
    loss = logits.square().mean()
    loss.backward()

    assert embedding_weight.grad is not None

    print("Improved GPT tests passed.")
    print("Logits shape:", tuple(logits.shape))
    print("Weight tying: verified")
    print("Embedding gradients: verified")

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Total parameters: {total_params:,}")


if __name__ == "__main__":
    main()
