import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.gpt import MiniGPT
from training.config_v4 import (
    TOKENIZER_V4_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    CHECKPOINT_DIR_SCRATCH,
    LOG_DIR_SCRATCH,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    BLOCK_SIZE,
    DROPOUT,
    BATCH_SIZE,
    LEARNING_RATE_SCRATCH,
    MIN_LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    GRAD_CLIP,
    EVAL_INTERVAL,
    EVAL_BATCHES,
    LOG_INTERVAL,
    SAVE_INTERVAL,
    MAX_STEPS_SCRATCH,
    USE_AMP,
    DEVICE,
)
from training.dataset_v4 import create_dataloader_v4


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train MiniGPT V4 from Scratch (Random Initialization)."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest V4-Scratch checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific V4-Scratch checkpoint to resume from.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS_SCRATCH,
        help="Total target training steps.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a 10-step verification test to confirm training and checkpointing.",
    )
    return parser.parse_args()


def get_learning_rate(step, max_steps):
    if WARMUP_STEPS > 0 and step < WARMUP_STEPS:
        return LEARNING_RATE_SCRATCH * (step + 1) / WARMUP_STEPS

    decay_steps = max(1, max_steps - WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + cosine_factor * (LEARNING_RATE_SCRATCH - MIN_LEARNING_RATE)


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_state"]])


@torch.no_grad()
def estimate_loss(model, val_loader, eval_batches=EVAL_BATCHES):
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    amp_enabled = USE_AMP and DEVICE.type == "cuda"

    for _ in range(eval_batches):
        try:
            x, y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            x, y = next(val_iter)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16,
            enabled=amp_enabled,
        ):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE).float(),
                y.reshape(-1),
                ignore_index=-100,
            )

        if not torch.isnan(loss):
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def save_v4_checkpoint(
    model,
    optimizer,
    scaler,
    step,
    val_loss,
    best_val_loss,
    checkpoint_dir,
    filename,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": "v4-scratch",
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "vocab_size": VOCAB_SIZE,
        "model_config": {
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "max_sequence_length": BLOCK_SIZE,
            "dropout": DROPOUT,
        },
        "rng_state": get_rng_state(),
    }
    torch.save(checkpoint, checkpoint_dir / filename)


def main():
    args = parse_arguments()
    max_steps = 10 if args.smoke_test else args.max_steps

    print("=" * 70)
    print(f"MINIGPT V4-SCRATCH TRAINING {'(SMOKE TEST MODE)' if args.smoke_test else ''}")
    print("=" * 70)

    # 1. Verify tokenizer exists
    if not TOKENIZER_V4_PATH.exists():
        raise FileNotFoundError(f"V4 tokenizer not found: {TOKENIZER_V4_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_V4_PATH))
    assert tokenizer.get_vocab_size() == VOCAB_SIZE, "Tokenizer vocab mismatch!"
    print(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()} | Model vocab: {VOCAB_SIZE}")

    # 2. Dataloaders
    print("\nLoading V4 conversation datasets...")
    train_loader = create_dataloader_v4(
        TRAIN_DATA_PATH,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        shuffle=True,
        tokenizer=tokenizer,
    )
    val_loader = create_dataloader_v4(
        VAL_DATA_PATH,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        shuffle=False,
        tokenizer=tokenizer,
    )
    print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")

    # 3. Model instantiation from scratch
    print("\nInitializing fresh MiniGPT V4 model from scratch (seed=42)...")
    torch.manual_seed(42)
    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_sequence_length=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {DEVICE}")

    # 4. Optimizer and Scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_SCRATCH,
        weight_decay=WEIGHT_DECAY,
    )
    amp_enabled = USE_AMP and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # 5. Resume logic
    start_step = 0
    best_val_loss = float("inf")

    if args.resume or args.checkpoint:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR_SCRATCH / "latest.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"\nResuming from V4-Scratch checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        restore_rng_state(ckpt.get("rng_state"))
        print(f"Resumed at step {start_step} with best validation loss {best_val_loss:.4f}")

    # 6. Logging setup
    LOG_DIR_SCRATCH.mkdir(parents=True, exist_ok=True)
    log_file_path = LOG_DIR_SCRATCH / "training_log.csv"
    if not (args.resume and log_file_path.exists()):
        with open(log_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss", "learning_rate"])

    # 7. Initial baseline evaluation
    print("Computing baseline validation loss...")
    initial_val_loss = estimate_loss(model, val_loader, eval_batches=min(5, len(val_loader)))
    print(f"Initial Validation Loss: {initial_val_loss:.4f}\n")
    if initial_val_loss < best_val_loss:
        best_val_loss = initial_val_loss

    # 8. Training loop
    train_iter = iter(train_loader)
    model.train()
    print(f"Beginning training loop (steps {start_step} to {max_steps})...\n")

    for step in range(start_step, max_steps):
        lr = get_learning_rate(step, max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16,
            enabled=amp_enabled,
        ):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE).float(),
                y.reshape(-1),
                ignore_index=-100,
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Step logging
        if (step + 1) % LOG_INTERVAL == 0 or step == max_steps - 1 or args.smoke_test:
            print(f"Step {step + 1:5d}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.2e}")

        # Evaluation & checkpointing
        if (step + 1) % EVAL_INTERVAL == 0 or step == max_steps - 1 or args.smoke_test:
            val_loss = estimate_loss(model, val_loader, eval_batches=min(5, len(val_loader)))
            print(f"  --> Eval @ Step {step + 1}: Val Loss = {val_loss:.4f} (Best: {best_val_loss:.4f})")

            with open(log_file_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([step + 1, loss.item(), val_loss, lr])

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_v4_checkpoint(
                    model, optimizer, scaler, step + 1, val_loss, best_val_loss,
                    CHECKPOINT_DIR_SCRATCH, "best.pt"
                )
                print(f"  --> Saved new best checkpoint to: {CHECKPOINT_DIR_SCRATCH / 'best.pt'}")

            save_v4_checkpoint(
                model, optimizer, scaler, step + 1, val_loss, best_val_loss,
                CHECKPOINT_DIR_SCRATCH, "latest.pt"
            )

    print("\n" + "=" * 70)
    print(f"V4-SCRATCH {'SMOKE TEST' if args.smoke_test else 'TRAINING'} COMPLETED SUCCESSFULLY!")
    print(f"Final Best Validation Loss: {best_val_loss:.4f}")
    print(f"Checkpoints directory: {CHECKPOINT_DIR_SCRATCH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
