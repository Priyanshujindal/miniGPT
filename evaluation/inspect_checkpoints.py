
import torch
from pathlib import Path


CHECKPOINTS = [
    ("V1 best", Path("checkpoints_v2/best.pt")),
    ("V1 latest", Path("checkpoints_v2/latest.pt")),
    ("V2 best", Path("checkpoints_tokenizer_v2/best.pt")),
    ("V2 latest", Path("checkpoints_tokenizer_v2/latest.pt")),
]


def main():
    print("=" * 65)
    print("MINIGPT CHECKPOINT INSPECTION")
    print("=" * 65)

    for name, path in CHECKPOINTS:
        print(f"\n{name}")
        print("-" * 45)

        if not path.exists():
            print("Checkpoint not found:", path)
            continue

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        print("Path:", path)
        print("Step:", checkpoint.get("step"))
        print("Validation loss:", checkpoint.get("val_loss"))
        print("Best validation loss:", checkpoint.get("best_val_loss"))

        model_config = checkpoint.get("model_config", {})
        print("Model config:", model_config)

        print(
            "Checkpoint size:",
            round(path.stat().st_size / (1024 * 1024), 2),
            "MB"
        )


if __name__ == "__main__":
    main()
