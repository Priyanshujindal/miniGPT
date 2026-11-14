
import math
import torch
import torch.nn.functional as F

from generation.load_model import load_trained_model
from training.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    BLOCK_SIZE,
)
from training.dataset import create_dataloader


MAX_EVAL_BATCHES = 100

CHECKPOINTS = [
    CHECKPOINT_DIR / "best.pt",
    CHECKPOINT_DIR / "latest.pt",
]


@torch.no_grad()
def evaluate_model(model, loader, vocab_size, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= MAX_EVAL_BATCHES:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

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
    perplexity = math.exp(average_loss)

    return average_loss, perplexity


def main():
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

    results = []

    print("=" * 65)
    print("MINIGPT CHECKPOINT COMPARISON")
    print(f"Evaluation batches per dataset: {MAX_EVAL_BATCHES}")
    print("=" * 65)

    for checkpoint_path in CHECKPOINTS:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        model, tokenizer, device = load_trained_model(
            checkpoint_path=str(checkpoint_path)
        )

        vocab_size = tokenizer.vocab_size + 1

        train_loss, train_ppl = evaluate_model(
            model,
            train_loader,
            vocab_size,
            device,
        )

        val_loss, val_ppl = evaluate_model(
            model,
            val_loader,
            vocab_size,
            device,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        result = {
            "name": checkpoint_path.name,
            "step": checkpoint["step"],
            "stored_val_loss": checkpoint["val_loss"],
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
        }

        results.append(result)

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 85)
    print("                 FAIR CHECKPOINT COMPARISON")
    print("=" * 85)

    print(
        f"{'Checkpoint':<15}"
        f"{'Step':>10}"
        f"{'Train Loss':>13}"
        f"{'Val Loss':>13}"
        f"{'Val PPL':>12}"
    )

    print("-" * 85)

    for result in results:
        print(
            f"{result['name']:<15}"
            f"{result['step']:>10}"
            f"{result['train_loss']:>13.4f}"
            f"{result['val_loss']:>13.4f}"
            f"{result['val_ppl']:>12.2f}"
        )

    print("-" * 85)

    print("\nStored checkpoint validation losses:")
    for result in results:
        print(
            f"{result['name']}: "
            f"{result['stored_val_loss']:.4f}"
        )

    best = min(results, key=lambda item: item["val_loss"])

    print("\nResult:")
    print(
        f"Better checkpoint on this evaluation sample: "
        f"{best['name']}"
    )
    print(f"Validation loss: {best['val_loss']:.4f}")
    print(f"Validation perplexity: {best['val_ppl']:.2f}")

    print("\nNote: This comparison uses the first 100 validation")
    print("batches, not necessarily the entire validation dataset.")
    print("=" * 85)


if __name__ == "__main__":
    main()
