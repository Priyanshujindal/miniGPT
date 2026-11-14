
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.gpt import MiniGPT

from training.config_v2 import (
    TOKENIZER_PATH,
    CHECKPOINT_DIR,
    TOKENIZER_VOCAB_SIZE,
    EOS_TOKEN_ID,
    VOCAB_SIZE,
    DEVICE,
)


def load_trained_model_v2(
    checkpoint_path=None,
    tokenizer_path=None,
    device=None,
):

    if device is None:
        device = DEVICE

    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "best.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    if tokenizer_path is None:
        tokenizer_path = TOKENIZER_PATH
    else:
        tokenizer_path = Path(tokenizer_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"V2 checkpoint not found: {checkpoint_path}\n"
            "Train the V2 model before starting generation."
        )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"V2 tokenizer not found: {tokenizer_path}"
        )

    print("Loading V2 tokenizer...")

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    actual_tokenizer_size = tokenizer.get_vocab_size()

    if actual_tokenizer_size != TOKENIZER_VOCAB_SIZE:
        raise ValueError(
            f"Tokenizer vocabulary mismatch. "
            f"Expected {TOKENIZER_VOCAB_SIZE}, "
            f"got {actual_tokenizer_size}."
        )

    print(f"Tokenizer vocabulary: {actual_tokenizer_size}")

    print("\nLoading V2 checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint_vocab = checkpoint.get("vocab_size")
    checkpoint_tokenizer_vocab = checkpoint.get(
        "tokenizer_vocab_size"
    )
    checkpoint_eos = checkpoint.get("eos_token_id")

    if checkpoint_vocab != VOCAB_SIZE:
        raise ValueError(
            "Checkpoint is incompatible with tokenizer V2. "
            f"Expected model vocabulary {VOCAB_SIZE}, "
            f"got {checkpoint_vocab}."
        )

    if checkpoint_tokenizer_vocab != TOKENIZER_VOCAB_SIZE:
        raise ValueError(
            "This checkpoint was not trained with tokenizer V2."
        )

    if checkpoint_eos != EOS_TOKEN_ID:
        raise ValueError(
            "Checkpoint EOS token ID does not match tokenizer V2."
        )

    config = checkpoint.get("model_config")

    if not config:
        raise ValueError(
            "Checkpoint does not contain model configuration."
        )

    print(f"Checkpoint step: {checkpoint.get('step')}")
    print(f"Validation loss: {checkpoint.get('val_loss')}")

    print("\nRecreating V2 model...")

    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_sequence_length=config["max_sequence_length"],
        dropout=config.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nV2 model loaded successfully.")
    print(f"Model vocabulary: {VOCAB_SIZE}")
    print(f"EOS token ID: {EOS_TOKEN_ID}")
    print(f"Device: {device}")

    return model, tokenizer, device
