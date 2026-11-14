import torch

from model.gpt import MiniGPT
from tokenizer.bpe import BPETokenizer


def load_trained_model(
    checkpoint_path="checkpoints_v2/best.pt",
    tokenizer_path="data/tokenizer/bpe_tokenizer.json",
    device=None,
):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Load tokenizer
    tokenizer = BPETokenizer.load(tokenizer_path)

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    print(f"Checkpoint training step: {checkpoint.get('step', 'Unknown')}")
    print(f"Checkpoint validation loss: {checkpoint.get('val_loss', 'Unknown')}")

    config = checkpoint["model_config"]

    # Check tokenizer/model vocabulary compatibility.
    expected_vocab_size = tokenizer.vocab_size + 1

    if checkpoint["vocab_size"] != expected_vocab_size:
        raise ValueError(
            "Tokenizer vocabulary does not match the trained model. "
            "Use the tokenizer that was used during training."
        )

    # Recreate the exact model architecture.
    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_sequence_length=config["max_sequence_length"],
        dropout=config.get("dropout", 0.1),
    ).to(device)

    # Load learned parameters.
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Training step: {checkpoint['step']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Device: {device}")

    return model, tokenizer, device