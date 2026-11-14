
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.gpt import MiniGPT
from training.config_v3 import (
    TOKENIZER_PATH,
    CHECKPOINT_DIR,
    TOKENIZER_VOCAB_SIZE,
    EOS_TOKEN_ID,
    VOCAB_SIZE,
    DEVICE,
)


def load_model_v3(checkpoint_name="best.pt"):

    checkpoint_path = CHECKPOINT_DIR / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"V3 checkpoint not found: {checkpoint_path}"
        )

    print("Loading V3 tokenizer...")

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    if tokenizer.get_vocab_size() != TOKENIZER_VOCAB_SIZE:
        raise ValueError("V3 tokenizer vocabulary mismatch.")

    print("Loading V3 checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if checkpoint.get("version") != "v3":
        raise ValueError("The selected checkpoint is not V3.")

    if checkpoint.get("vocab_size") != VOCAB_SIZE:
        raise ValueError("V3 model vocabulary mismatch.")

    if checkpoint.get("eos_token_id") != EOS_TOKEN_ID:
        raise ValueError("V3 EOS token ID mismatch.")

    config = checkpoint["model_config"]

    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_sequence_length=config["max_sequence_length"],
        dropout=config["dropout"],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nV3 model loaded successfully.")
    print("Checkpoint:", checkpoint_path)
    print("Training step:", checkpoint["step"])
    print("Validation loss:", checkpoint["val_loss"])
    print("Device:", DEVICE)

    return model, tokenizer, DEVICE


if __name__ == "__main__":
    load_model_v3()
