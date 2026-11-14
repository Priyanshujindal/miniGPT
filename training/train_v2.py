
import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from training.config_v2 import (
    PROJECT_ROOT,
    TOKENIZER_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    CHECKPOINT_DIR,
    LOG_DIR,
    TOKENIZER_VOCAB_SIZE,
    EOS_TOKEN_ID,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    BLOCK_SIZE,
    DROPOUT,
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
    USE_AMP,
)

from training.dataset_v2 import create_dataloader_v2
from model.gpt import MiniGPT


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Train or resume MiniGPT with tokenizer V2."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the V2 latest checkpoint.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific V2 checkpoint to resume from.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Total target training steps.",
    )

    return parser.parse_args()


def get_learning_rate(step, max_steps):

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


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader):

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

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
            )

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


def save_checkpoint(
    model,
    optimizer,
    scaler,
    step,
    val_loss,
    best_val_loss,
    filename,
):

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "vocab_size": VOCAB_SIZE,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "eos_token_id": EOS_TOKEN_ID,
        "tokenizer_path": str(TOKENIZER_PATH.relative_to(PROJECT_ROOT)),
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


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scaler,
):

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(f"\nLoading V2 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("vocab_size") != VOCAB_SIZE:
        raise ValueError(
            "Checkpoint vocabulary does not match tokenizer V2."
        )

    if checkpoint.get("tokenizer_vocab_size") != TOKENIZER_VOCAB_SIZE:
        raise ValueError(
            "Checkpoint was not created with tokenizer V2."
        )

    if checkpoint.get("eos_token_id") != EOS_TOKEN_ID:
        raise ValueError(
            "Checkpoint EOS token ID does not match tokenizer V2."
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

        if saved_value != expected_value:
            raise ValueError(
                f"Model configuration mismatch for {key}: "
                f"saved={saved_value}, expected={expected_value}."
            )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scaler_state = checkpoint.get("scaler_state_dict")

    if scaler_state:
        scaler.load_state_dict(scaler_state)

    restore_rng_state(
        checkpoint.get("rng_state")
    )

    start_step = checkpoint["step"] + 1

    best_val_loss = checkpoint.get(
        "best_val_loss",
        checkpoint.get("val_loss", float("inf")),
    )

    print("V2 checkpoint loaded successfully.")
    print(f"Previous step: {checkpoint['step']}")
    print(f"Next step: {start_step}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return start_step, best_val_loss


def main():

    args = parse_arguments()
    max_steps = args.max_steps

    if max_steps <= 0:
        raise ValueError("Maximum training steps must be positive.")

    if EVAL_BATCHES <= 0:
        raise ValueError("EVAL_BATCHES must be positive.")

    if WARMUP_STEPS >= max_steps:
        raise ValueError(
            "WARMUP_STEPS must be smaller than max_steps."
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MINIGPT V2 TRAINING")
    print("=" * 65)

    # 1. Load and validate the V2 tokenizer.

    print("\nLoading tokenizer V2...")

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    if tokenizer.get_vocab_size() != TOKENIZER_VOCAB_SIZE:
        raise ValueError(
            f"Expected tokenizer vocabulary {TOKENIZER_VOCAB_SIZE}, "
            f"got {tokenizer.get_vocab_size()}."
        )

    print(f"Tokenizer vocabulary: {tokenizer.get_vocab_size()}")
    print(f"EOS token ID: {EOS_TOKEN_ID}")
    print(f"Model vocabulary: {VOCAB_SIZE}")

    # 2. Load V2 datasets.

    print("\nLoading V2 datasets...")

    train_loader = create_dataloader_v2(
        TRAIN_DATA_PATH,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        shuffle=True,
    )

    val_loader = create_dataloader_v2(
        VAL_DATA_PATH,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        shuffle=False,
    )

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError("A V2 DataLoader is empty.")

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # 3. Initialize a completely new model.

    print("\nCreating a fresh MiniGPT V2 model...")

    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
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

    # 4. Optimizer.

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 5. Mixed precision.

    amp_enabled = USE_AMP and DEVICE.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    print(f"Mixed precision enabled: {amp_enabled}")

    # 6. Resume or start fresh.

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
        )

        if start_step >= max_steps:
            raise ValueError(
                f"Checkpoint is already at step {start_step - 1}, "
                f"but max_steps is {max_steps}. "
                "Increase --max-steps to continue."
            )

    else:
        print("\nStarting a fresh V2 training run.")

    # 7. CSV logging.

    log_path = LOG_DIR / "training_log.csv"

    if not (resume_requested and log_path.exists()):

        with open(log_path, "w", newline="") as file:

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

    print("\nStarting V2 training...\n")

    # 8. Training loop.

    for step in range(start_step, max_steps):

        current_lr = get_learning_rate(
            step,
            max_steps,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        try:
            x, y = next(train_iterator)

        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

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
                logits.reshape(-1, VOCAB_SIZE).float(),
                y.reshape(-1),
            )

        optimizer.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(optimizer)
        scaler.update()

        if step % LOG_INTERVAL == 0:

            print(
                f"Step {step:5d}/{max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.7f}"
            )

        # 9. Evaluation and checkpoint saving.

        if (
            step % EVAL_INTERVAL == 0
            or step == max_steps - 1
        ):

            losses = estimate_loss(
                model,
                train_loader,
                val_loader,
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            print("\n" + "-" * 65)
            print(f"V2 evaluation at step {step}")
            print(f"Training loss:   {train_loss:.4f}")
            print(f"Validation loss: {val_loss:.4f}")
            print(f"Validation perplexity: {math.exp(val_loss):.2f}")
            print("-" * 65)

            with open(log_path, "a", newline="") as file:

                writer = csv.writer(file)

                writer.writerow([
                    step,
                    train_loss,
                    val_loss,
                    current_lr,
                ])

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
                    "best.pt",
                )

                print("New best V2 model saved.")

            save_checkpoint(
                model,
                optimizer,
                scaler,
                step,
                val_loss,
                best_val_loss,
                "latest.pt",
            )

            print("Latest V2 checkpoint saved.\n")

    print("=" * 65)
    print("V2 TRAINING COMPLETED")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"Training log: {log_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
