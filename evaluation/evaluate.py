
import math
import torch
import torch.nn.functional as F

from generation.load_model import load_trained_model
from training.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
)
from training.dataset import create_dataloader


@torch.no_grad()
def evaluate_model(model, loader, vocab_size, max_batches=100):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        # Sum individual token losses so batches are weighted
        # correctly even if their sizes differ.
        loss_sum = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
            reduction="sum",
        )

        total_loss += loss_sum.item()
        total_tokens += y.numel()

    if total_tokens == 0:
        raise RuntimeError("No evaluation tokens were processed.")

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss) if average_loss < 100 else float("inf")

    return average_loss, perplexity


def main():
    model, tokenizer, device = load_trained_model()

    vocab_size = tokenizer.vocab_size + 1

    train_loader = create_dataloader(
        TRAIN_DATA_PATH,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        shuffle=False,
    )

    val_loader = create_dataloader(
        VAL_DATA_PATH,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        shuffle=False,
    )

    print("\nEvaluating model...")

    train_loss, train_perplexity = evaluate_model(
        model, train_loader, vocab_size
    )

    val_loss, val_perplexity = evaluate_model(
        model, val_loader, vocab_size
    )

    print("\n" + "=" * 45)
    print("             MODEL EVALUATION")
    print("=" * 45)

    print(f"Training loss:      {train_loss:.4f}")
    print(f"Training perplexity:{train_perplexity:10.2f}")

    print("-" * 45)

    print(f"Validation loss:      {val_loss:.4f}")
    print(f"Validation perplexity:{val_perplexity:10.2f}")

    print("=" * 45)

    if val_loss > train_loss * 1.5:
        print("Warning: Possible overfitting.")
    else:
        print("No strong overfitting signal from this comparison.")

    model.train()


if __name__ == "__main__":
    main()
