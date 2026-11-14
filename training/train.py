
import argparse
import csv
import math
from pathlib import Path

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
    DROPOUT,
    USE_AMP,
)

from training.dataset import create_dataloader
from tokenizer.bpe import BPETokenizer
from model.gpt import MiniGPT


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Train or resume MiniGPT."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Total target training steps.",
    )

    return parser.parse_args()


# --------------------------------------------------
# Learning rate scheduler
# --------------------------------------------------

def get_learning_rate(step, max_steps):
    """Linear warmup followed by cosine decay."""

    if WARMUP_STEPS > 0 and step < WARMUP_STEPS:

        return LEARNING_RATE * (step + 1) / WARMUP_STEPS

    decay_steps = max(1, max_steps - WARMUP_STEPS)

    progress = (step - WARMUP_STEPS) / decay_steps
    progress = min(max(progress, 0.0), 1.0)

    cosine_factor = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return MIN_LEARNING_RATE + cosine_factor * (
        LEARNING_RATE - MIN_LEARNING_RATE
    )


# --------------------------------------------------
# Loss calculation
# --------------------------------------------------

def calculate_loss(model, x, y, vocab_size):
    """Calculate next-token prediction cross-entropy loss."""

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
    )

    return loss


# --------------------------------------------------
# Loss evaluation
# --------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, vocab_size):

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

            loss = calculate_loss(
                model,
                x,
                y,
                vocab_size,
            )

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


# --------------------------------------------------
# Random-number generator states
# --------------------------------------------------

def get_rng_state():

    state = {
        "torch_rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():

        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state):

    if not state:
        return

    if "torch_rng_state" in state:

        torch.set_rng_state(
            state["torch_rng_state"].cpu()
        )

    if (
        torch.cuda.is_available()
        and "cuda_rng_state" in state
    ):

        torch.cuda.set_rng_state_all(
            [item.cpu() for item in state["cuda_rng_state"]]
        )


# --------------------------------------------------
# Checkpoint saving
# --------------------------------------------------

def save_checkpoint(
    model,
    optimizer,
    scaler,
    step,
    val_loss,
    best_val_loss,
    vocab_size,
    filename,
):

    checkpoint = {

        "step": step,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "val_loss": val_loss,

        "best_val_loss": best_val_loss,

        "vocab_size": vocab_size,

        "model_config": {
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "max_sequence_length": BLOCK_SIZE,
            "dropout": DROPOUT,
        },

        "rng_state": get_rng_state(),
    }

    torch.save(
        checkpoint,
        CHECKPOINT_DIR / filename,
    )


# --------------------------------------------------
# Checkpoint loading
# --------------------------------------------------

def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scaler,
    vocab_size,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Only load checkpoints you trust.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("vocab_size") != vocab_size:

        raise ValueError(
            "Checkpoint vocabulary does not match the current tokenizer."
        )

    saved_config = checkpoint.get("model_config", {})

    expected_config = {
        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "max_sequence_length": BLOCK_SIZE,
    }

    for key, expected_value in expected_config.items():

        saved_value = saved_config.get(key)

        if (
            saved_value is not None
            and saved_value != expected_value
        ):

            raise ValueError(
                f"Checkpoint model configuration mismatch: "
                f"{key} is {saved_value}, expected {expected_value}."
            )

    # Restore model weights.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Restore optimizer.
    if "optimizer_state_dict" in checkpoint:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    # Restore mixed-precision scaler.
    scaler_state = checkpoint.get("scaler_state_dict")

    if scaler_state:

        scaler.load_state_dict(scaler_state)

    # Restore random-number generator states.
    restore_rng_state(
        checkpoint.get("rng_state")
    )

    start_step = checkpoint["step"] + 1

    best_val_loss = checkpoint.get(
        "best_val_loss",
        checkpoint.get("val_loss", float("inf")),
    )

    print("Checkpoint loaded successfully.")
    print(f"Previous step: {checkpoint['step']}")
    print(f"Next training step: {start_step}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return start_step, best_val_loss


# --------------------------------------------------
# Main training function
# --------------------------------------------------

def main():

    args = parse_arguments()

    max_steps = args.max_steps

    if max_steps <= 0:

        raise ValueError(
            "Maximum training steps must be greater than zero."
        )

    if EVAL_BATCHES <= 0:

        raise ValueError(
            "EVAL_BATCHES must be greater than zero."
        )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("MiniGPT Improved Training")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = BPETokenizer.load(
        str(TOKENIZER_PATH)
    )

    vocab_size = tokenizer.vocab_size + 1

    print(f"Tokenizer vocabulary: {tokenizer.vocab_size}")
    print(f"Model vocabulary: {vocab_size}")

    # --------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------

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

        raise RuntimeError(
            "Training DataLoader is empty."
        )

    if len(val_loader) == 0:

        raise RuntimeError(
            "Validation DataLoader is empty."
        )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # --------------------------------------------------
    # 3. Create model
    # --------------------------------------------------

    print("\nCreating improved MiniGPT...")

    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_sequence_length=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Parameters: {total_parameters:,}")
    print(f"Device: {DEVICE}")
    print(f"Dropout: {DROPOUT}")

    # --------------------------------------------------
    # 4. Create optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # 5. Configure mixed precision
    # --------------------------------------------------

    amp_enabled = USE_AMP and DEVICE.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    print(f"Mixed precision enabled: {amp_enabled}")

    # --------------------------------------------------
    # 6. Resume or start fresh
    # --------------------------------------------------

    resume_requested = (
        args.resume or args.checkpoint is not None
    )

    start_step = 0
    best_val_loss = float("inf")

    if resume_requested:

        checkpoint_path = (
            Path(args.checkpoint)
            if args.checkpoint
            else CHECKPOINT_DIR / "latest.pt"
        )

        start_step, best_val_loss = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scaler,
            vocab_size,
        )

        if start_step >= max_steps:

            raise ValueError(
                f"Checkpoint is already at step {start_step - 1}, "
                f"but max_steps is {max_steps}. "
                "Increase --max-steps to continue training."
            )

    else:

        print("\nStarting a new training run.")

    # --------------------------------------------------
    # 7. Prepare logging
    # --------------------------------------------------

    log_path = LOG_DIR / "training_log.csv"

    if resume_requested and log_path.exists():

        print("Appending to existing training log.")

    else:

        with open(
            log_path,
            "w",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "step",
                "train_loss",
                "val_loss",
                "learning_rate",
            ])

    # Create iterator after restoring RNG state.
    train_iterator = iter(train_loader)

    model.train()

    print("\nStarting training...\n")

    # --------------------------------------------------
    # 8. Training loop
    # --------------------------------------------------

    for step in range(start_step, max_steps):

        # Update learning rate.
        current_lr = get_learning_rate(
            step,
            max_steps,
        )

        for param_group in optimizer.param_groups:

            param_group["lr"] = current_lr

        # Get next batch.
        try:

            x, y = next(train_iterator)

        except StopIteration:

            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        # --------------------------------------------------
        # Forward pass with mixed precision
        # --------------------------------------------------

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=(
                torch.float16
                if DEVICE.type == "cuda"
                else torch.bfloat16
            ),
            enabled=amp_enabled,
        ):

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size).float(),
                y.reshape(-1),
            )

        # --------------------------------------------------
        # Backward pass
        # --------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(optimizer)
        scaler.update()

        # --------------------------------------------------
        # Training progress
        # --------------------------------------------------

        if step % LOG_INTERVAL == 0:

            print(
                f"Step {step:5d}/{max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.7f}"
            )

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        if (
            step % EVAL_INTERVAL == 0
            or step == max_steps - 1
        ):

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

                print(
                    f"Validation perplexity: "
                    f"{math.exp(val_loss):.2f}"
                )

            print("-" * 60 + "\n")

            # --------------------------------------------------
            # Save CSV log
            # --------------------------------------------------

            with open(
                log_path,
                "a",
                newline="",
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    step,
                    train_loss,
                    val_loss,
                    current_lr,
                ])

            # --------------------------------------------------
            # Update best validation loss
            # --------------------------------------------------

            is_best = val_loss < best_val_loss

            if is_best:

                best_val_loss = val_loss

                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    step,
                    val_loss,
                    best_val_loss,
                    vocab_size,
                    "best.pt",
                )

                print("New best model saved.")

            # --------------------------------------------------
            # Save latest checkpoint
            # --------------------------------------------------

            save_checkpoint(
                model,
                optimizer,
                scaler,
                step,
                val_loss,
                best_val_loss,
                vocab_size,
                "latest.pt",
            )

            print("Latest checkpoint saved.")
            print()

    # --------------------------------------------------
    # 9. Training completed
    # --------------------------------------------------

    print("=" * 60)
    print("Training completed.")

    print(
        f"Best validation loss: {best_val_loss:.4f}"
    )

    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"Training log: {log_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
