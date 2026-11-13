import csv
import math
import torch
import torch.nn.functional as F

from training.config import (
    TOKENIZER_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    CHECKPOINT_DIR,
    LOG_DIR,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    BLOCK_SIZE,
    BATCH_SIZE,
    MAX_STEPS,
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    GRAD_CLIP,
    EVAL_INTERVAL,
    EVAL_BATCHES,
    LOG_INTERVAL,
    DEVICE,
)

from training.dataset import create_dataloader
from tokenizer.bpe import BPETokenizer
from model.gpt import MiniGPT


def get_learning_rate(step):
    """Linear warmup followed by cosine decay."""

    if WARMUP_STEPS > 0 and step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS

    decay_steps = max(1, MAX_STEPS - WARMUP_STEPS)

    progress = (step - WARMUP_STEPS) / decay_steps
    progress = min(max(progress, 0.0), 1.0)

    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    return MIN_LEARNING_RATE + cosine_factor * (
        LEARNING_RATE - MIN_LEARNING_RATE
    )


def calculate_loss(model, x, y, vocab_size):
    """Calculate next-token prediction cross-entropy loss."""

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
    )

    return loss


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, vocab_size):
    """Estimate average training and validation loss."""

    model.eval()

    results = {}

    for name, loader in [
        ("train", train_loader),
        ("val", val_loader),
    ]:
        losses = []
        iterator = iter(loader)

        for _ in range(EVAL_BATCHES):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(loader)

                try:
                    x, y = next(iterator)
                except StopIteration:
                    raise RuntimeError(
                        f"The {name} DataLoader contains no batches."
                    )

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            loss = calculate_loss(model, x, y, vocab_size)
            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


def save_checkpoint(
    model,
    optimizer,
    step,
    val_loss,
    vocab_size,
    filename,
):
    """Save model and optimizer state."""

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "vocab_size": vocab_size,
        "model_config": {
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "max_sequence_length": BLOCK_SIZE,
        },
    }

    torch.save(checkpoint, CHECKPOINT_DIR / filename)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MiniGPT Training")
    print("=" * 60)

    # 1. Load tokenizer
    print("\nLoading tokenizer...")

    tokenizer = BPETokenizer.load(str(TOKENIZER_PATH))

    # EOS token is an additional token beyond the BPE vocabulary.
    vocab_size = tokenizer.vocab_size + 1

    print(f"Tokenizer vocabulary: {tokenizer.vocab_size}")
    print(f"Model vocabulary: {vocab_size}")

    # 2. Load datasets
    print("\nLoading datasets...")

    train_loader = create_dataloader(
        TRAIN_DATA_PATH,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        shuffle=True,
    )

    val_loader = create_dataloader(
        VAL_DATA_PATH,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        shuffle=False,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training DataLoader is empty.")

    if len(val_loader) == 0:
        raise RuntimeError("Validation DataLoader is empty.")

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # 3. Create model
    print("\nCreating MiniGPT...")

    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_sequence_length=BLOCK_SIZE,
    ).to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Parameters: {total_parameters:,}")
    print(f"Device: {DEVICE}")

    # 4. Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 5. Prepare logging
    log_path = LOG_DIR / "training_log.csv"

    with open(log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "step",
            "train_loss",
            "val_loss",
            "learning_rate",
        ])

    best_val_loss = float("inf")
    train_iterator = iter(train_loader)

    model.train()

    print("\nStarting training...\n")

    # 6. Training loop
    for step in range(MAX_STEPS):

        # Update learning rate
        current_lr = get_learning_rate(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Get next training batch
        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        # Forward pass
        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
        )

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Prevent excessively large gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        # Update model parameters
        optimizer.step()

        # Print training progress
        if step % LOG_INTERVAL == 0:
            print(
                f"Step {step:5d}/{MAX_STEPS} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.7f}"
            )

        # 7. Evaluate model
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:

            losses = estimate_loss(
                model,
                train_loader,
                val_loader,
                vocab_size,
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            print("\n" + "-" * 60)
            print(f"Evaluation at step {step}")
            print(f"Training loss:   {train_loss:.4f}")
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < 20:
                print(f"Validation perplexity: {math.exp(val_loss):.2f}")

            print("-" * 60 + "\n")

            # Save CSV log
            with open(log_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    step,
                    train_loss,
                    val_loss,
                    current_lr,
                ])

            # Save latest checkpoint
            save_checkpoint(
                model,
                optimizer,
                step,
                val_loss,
                vocab_size,
                "latest.pt",
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    val_loss,
                    vocab_size,
                    "best.pt",
                )

                print("New best model saved.")

            print()

    print("=" * 60)
    print("Training completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"Training log: {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()