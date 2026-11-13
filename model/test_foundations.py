
import torch

from tokenizer.bpe import BPETokenizer

from model.embeddings import (
    TokenEmbedding,
    PositionalEmbedding,
    GPTEmbeddings
)

from model.layers import Linear, GELU

from model.normalization import LayerNorm


# -----------------------------
# CONFIGURATION
# -----------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EMBEDDING_DIM = 256
MAX_SEQUENCE_LENGTH = 128
BATCH_SIZE = 2
SEQUENCE_LENGTH = 16

tokenizer = BPETokenizer.load(
    "data/tokenizer/bpe_tokenizer.json"
)

VOCAB_SIZE = tokenizer.vocab_size + 1


def check(condition, message):

    assert condition, message

    print("[PASS]", message)


def main():

    print("Device:", DEVICE)
    print("Vocabulary:", VOCAB_SIZE)

    # -------------------------
    # TEST TOKEN EMBEDDING
    # -------------------------

    print("\nTEST 1: Token Embedding")

    token_embedding = TokenEmbedding(
        VOCAB_SIZE,
        EMBEDDING_DIM
    ).to(DEVICE)

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH_SIZE, SEQUENCE_LENGTH),
        device=DEVICE
    )

    token_vectors = token_embedding(input_ids)

    check(
        token_vectors.shape == (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            EMBEDDING_DIM
        ),
        "Token embedding output shape"
    )

    # -------------------------
    # TEST POSITION EMBEDDING
    # -------------------------

    print("\nTEST 2: Positional Embedding")

    position_embedding = PositionalEmbedding(
        MAX_SEQUENCE_LENGTH,
        EMBEDDING_DIM
    ).to(DEVICE)

    position_vectors = position_embedding(input_ids)

    check(
        position_vectors.shape == (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            EMBEDDING_DIM
        ),
        "Positional embedding output shape"
    )

    # -------------------------
    # TEST GPT EMBEDDINGS
    # -------------------------

    print("\nTEST 3: Combined GPT Embeddings")

    embeddings = GPTEmbeddings(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        MAX_SEQUENCE_LENGTH
    ).to(DEVICE)

    combined = embeddings(input_ids)

    check(
        combined.shape == (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            EMBEDDING_DIM
        ),
        "Combined embedding shape"
    )

    # -------------------------
    # TEST LINEAR LAYER
    # -------------------------

    print("\nTEST 4: Custom Linear Layer")

    linear = Linear(
        EMBEDDING_DIM,
        512
    ).to(DEVICE)

    linear_output = linear(combined)

    check(
        linear_output.shape == (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            512
        ),
        "Linear layer output shape"
    )

    # -------------------------
    # TEST GELU
    # -------------------------

    print("\nTEST 5: GELU Activation")

    gelu = GELU()

    activated = gelu(linear_output)

    check(
        activated.shape == linear_output.shape,
        "GELU output shape"
    )

    check(
        torch.isfinite(activated).all().item(),
        "GELU contains no NaN or Inf"
    )

    # -------------------------
    # TEST LAYER NORMALIZATION
    # -------------------------

    print("\nTEST 6: Layer Normalization")

    layer_norm = LayerNorm(
        512
    ).to(DEVICE)

    normalized = layer_norm(activated)

    check(
        normalized.shape == activated.shape,
        "LayerNorm output shape"
    )

    means = normalized.mean(
        dim=-1
    )

    check(
        torch.allclose(
            means,
            torch.zeros_like(means),
            atol=1e-5
        ),
        "LayerNorm feature means approximately zero"
    )

    # -------------------------
    # TEST BACKPROPAGATION
    # -------------------------

    print("\nTEST 7: Gradient Calculation")

    loss = normalized.square().mean()

    loss.backward()

    check(
        embeddings.token_embedding.weight.grad is not None,
        "Token embedding receives gradients"
    )

    check(
        embeddings.position_embedding.weight.grad is not None,
        "Position embedding receives gradients"
    )

    check(
        linear.weight.grad is not None,
        "Linear layer receives gradients"
    )

    check(
        layer_norm.gamma.grad is not None,
        "LayerNorm gamma receives gradients"
    )

    # -------------------------
    # TEST POSITION LIMIT
    # -------------------------

    print("\nTEST 8: Position Limit")

    try:

        invalid_ids = torch.zeros(
            (1, MAX_SEQUENCE_LENGTH + 1),
            dtype=torch.long,
            device=DEVICE
        )

        position_embedding(invalid_ids)

        raise AssertionError(
            "Expected sequence length error"
        )

    except ValueError:

        print("[PASS] Position limit validation")

    print("\nALL FOUNDATION TESTS PASSED!")


if __name__ == "__main__":
    main()
