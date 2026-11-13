
import torch

from tokenizer.bpe import BPETokenizer
from model.gpt import MiniGPT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BPETokenizer.load(
    "data/tokenizer/bpe_tokenizer.json"
)

VOCAB_SIZE = tokenizer.vocab_size + 1

BATCH_SIZE = 2
SEQUENCE_LENGTH = 64

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    embedding_dim=256,
    num_heads=4,
    num_layers=4,
    max_sequence_length=128
).to(DEVICE)


def main():

    print("Device:", DEVICE)
    print("Vocabulary:", VOCAB_SIZE)

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH_SIZE, SEQUENCE_LENGTH),
        device=DEVICE
    )

    # Forward pass.
    logits = model(input_ids)

    print("Input shape:", input_ids.shape)
    print("Output shape:", logits.shape)

    assert logits.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        VOCAB_SIZE
    )

    assert torch.isfinite(logits).all()

    print("\nForward pass passed!")

    # Check causal behavior.
    model.eval()

    modified_input = input_ids.clone()

    # Change future tokens from position 32 onwards.
    modified_input[:, 32:] = torch.randint(
        0,
        VOCAB_SIZE,
        modified_input[:, 32:].shape,
        device=DEVICE
    )

    with torch.no_grad():

        original_logits = model(input_ids)

        modified_logits = model(modified_input)

    # Earlier predictions must not change.
    assert torch.allclose(
        original_logits[:, :32],
        modified_logits[:, :32],
        atol=1e-5,
        rtol=1e-5
    )

    print("Causal masking test passed!")

    # Check gradients.
    model.train()

    logits = model(input_ids)

    loss = logits.square().mean()

    loss.backward()

    assert (
        model.blocks[0].attention.qkv.weight.grad
        is not None
    )

    print("Gradient test passed!")

    # Count parameters.
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print("\nALL GPT TESTS PASSED!")


if __name__ == "__main__":
    main()
