import sys
from pathlib import Path

# Ensure miniGPT root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tokenizers import Tokenizer

from model.gpt import MiniGPT
from training.config_v4 import (
    PROJECT_ROOT,
    TOKENIZER_V4_PATH,
    CHECKPOINT_DIR_TRANSFER,
    CHECKPOINT_DIR_SCRATCH,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    BLOCK_SIZE,
    DROPOUT,
    DEVICE,
)


def load_model_v4(model_type="transfer", checkpoint_name="best.pt"):
    """
    Load a trained MiniGPT V4 model and its V4 tokenizer.

    Args:
        model_type: 'transfer' or 'scratch'
        checkpoint_name: 'best.pt' or 'latest.pt'

    Returns:
        tuple: (model, tokenizer, device)
    """
    if model_type not in ["transfer", "scratch"]:
        raise ValueError(f"Invalid model_type: '{model_type}'. Expected 'transfer' or 'scratch'.")

    checkpoint_dir = CHECKPOINT_DIR_TRANSFER if model_type == "transfer" else CHECKPOINT_DIR_SCRATCH
    checkpoint_path = checkpoint_dir / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"V4-{model_type.capitalize()} checkpoint not found at: {checkpoint_path}\n"
            f"Make sure training has completed and saved {checkpoint_name}."
        )

    if not TOKENIZER_V4_PATH.exists():
        raise FileNotFoundError(f"V4 tokenizer not found at: {TOKENIZER_V4_PATH}")

    print(f"Loading V4 tokenizer from: {TOKENIZER_V4_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_V4_PATH))

    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError(
            f"Tokenizer vocab mismatch: expected {VOCAB_SIZE}, got {tokenizer.get_vocab_size()}"
        )

    print(f"Loading V4-{model_type} checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    expected_version = f"v4-{model_type}"
    if checkpoint.get("version") != expected_version:
        raise ValueError(
            f"Checkpoint version mismatch: expected '{expected_version}', got '{checkpoint.get('version')}'"
        )

    config = checkpoint.get("model_config", {})
    model = MiniGPT(
        vocab_size=checkpoint.get("vocab_size", VOCAB_SIZE),
        embedding_dim=config.get("embedding_dim", EMBEDDING_DIM),
        num_heads=config.get("num_heads", NUM_HEADS),
        num_layers=config.get("num_layers", NUM_LAYERS),
        max_sequence_length=config.get("max_sequence_length", BLOCK_SIZE),
        dropout=config.get("dropout", DROPOUT),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\nMiniGPT V4-{model_type.capitalize()} loaded successfully.")
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Step        : {checkpoint.get('step')}")
    print(f"  Val Loss    : {checkpoint.get('val_loss'):.4f}")
    print(f"  Device      : {DEVICE}")

    return model, tokenizer, DEVICE


if __name__ == "__main__":
    import sys
    mtype = sys.argv[1] if len(sys.argv) > 1 else "transfer"
    load_model_v4(model_type=mtype)
