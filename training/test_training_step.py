import torch
import torch.nn.functional as F

from tokenizer.bpe import BPETokenizer
from model.gpt import MiniGPT
from training.dataset import create_dataloader
from training.config import (
    TOKENIZER_PATH,
    TRAIN_DATA_PATH,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    BLOCK_SIZE,
    BATCH_SIZE,
    DEVICE,
)

def main():
    print("Loading tokenizer...")

    tokenizer = BPETokenizer.load(str(TOKENIZER_PATH))
    vocab_size = tokenizer.vocab_size + 1

    print("Loading training data...")

    train_loader = create_dataloader(
        TRAIN_DATA_PATH,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        shuffle=True,
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            "Training DataLoader is empty. Check your tokenized dataset "
            "and batch size."
        )

    x, y = next(iter(train_loader))

    x = x.to(DEVICE)
    y = y.to(DEVICE)

    print("Creating model...")

    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_sequence_length=BLOCK_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
    )

    model.train()

    # Forward pass
    logits = model(x)

    # Next-token prediction loss
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
    )

    print(f"Initial loss: {loss.item():.4f}")

    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update parameters
    optimizer.step()

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is not finite.")

    print("Training step completed successfully.")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Target shape: {tuple(y.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()