
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

TRAIN_RAW = ROOT / "data" / "raw" / "train.jsonl"
VAL_RAW = ROOT / "data" / "raw" / "validation.jsonl"

DATASETS = {
    "V1": {
        "train": ROOT / "data" / "tokenized" / "train.npy",
        "validation": ROOT / "data" / "tokenized" / "validation.npy",
        "eos_id": 512,
        "vocab_size": 513,
    },
    "V2": {
        "train": ROOT / "data" / "tokenized_v2" / "train.npy",
        "validation": ROOT / "data" / "tokenized_v2" / "validation.npy",
        "eos_id": 2048,
        "vocab_size": 2049,
    },
}

RAW_FILES = {
    "train": TRAIN_RAW,
    "validation": VAL_RAW,
}


def inspect_raw_file(filepath):
    stories = 0
    characters = 0

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            stories += 1
            characters += len(record["text"])

    return stories, characters


def inspect_token_file(filepath, eos_id, vocab_size, expected_stories):
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    tokens = np.load(filepath, mmap_mode="r")

    if tokens.dtype != np.uint16:
        raise ValueError(f"Unexpected dtype: {tokens.dtype}")

    minimum_id = int(tokens.min())
    maximum_id = int(tokens.max())

    if minimum_id < 0 or maximum_id >= vocab_size:
        raise ValueError("Token ID outside expected vocabulary.")

    eos_count = int(np.count_nonzero(tokens == eos_id))

    non_eos_tokens = len(tokens) - eos_count

    eos_valid = eos_count == expected_stories

    return {
        "total_tokens": len(tokens),
        "non_eos_tokens": non_eos_tokens,
        "eos_count": eos_count,
        "eos_valid": eos_valid,
        "minimum_id": minimum_id,
        "maximum_id": maximum_id,
        "characters_per_token": None,
        "file_size_mb": filepath.stat().st_size / 1_000_000,
    }


def main():
    print("=" * 65)
    print("FULL DATASET TOKENIZER COMPARISON")
    print("=" * 65)

    results = {}

    for split, raw_file in RAW_FILES.items():
        stories, characters = inspect_raw_file(raw_file)

        print(f"\n{split.upper()} RAW DATA")
        print("-" * 50)
        print("Stories:", stories)
        print("Characters:", characters)

        results[split] = {}

        for version, config in DATASETS.items():
            stats = inspect_token_file(
                config[split],
                config["eos_id"],
                config["vocab_size"],
                stories,
            )

            stats["characters_per_token"] = (
                characters / stats["non_eos_tokens"]
                if stats["non_eos_tokens"] else 0
            )

            results[split][version] = stats

            print(f"\n{version}")
            print("Total tokens:", stats["total_tokens"])
            print("Non-EOS tokens:", stats["non_eos_tokens"])
            print("EOS count:", stats["eos_count"])
            print("EOS count valid:", stats["eos_valid"])
            print("Minimum ID:", stats["minimum_id"])
            print("Maximum ID:", stats["maximum_id"])
            print(
                "Characters per token:",
                round(stats["characters_per_token"], 4)
            )
            print("File size MB:", round(stats["file_size_mb"], 3))

            if not stats["eos_valid"]:
                raise ValueError(
                    f"{version} {split}: Expected exactly "
                    f"{stories} EOS tokens, got {stats['eos_count']}"
                )

        v1_ratio = results[split]["V1"]["characters_per_token"]
        v2_ratio = results[split]["V2"]["characters_per_token"]

        improvement = ((v2_ratio / v1_ratio) - 1) * 100

        print("\nCOMPRESSION IMPROVEMENT")
        print(f"V2 vs V1: {improvement:.2f}%")

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    for split in RAW_FILES:
        v1 = results[split]["V1"]
        v2 = results[split]["V2"]

        token_reduction = (
            (1 - v2["non_eos_tokens"] / v1["non_eos_tokens"]) * 100
        )

        print(f"\n{split.upper()}")
        print(
            "V1 non-EOS tokens:",
            f"{v1['non_eos_tokens']:,}"
        )
        print(
            "V2 non-EOS tokens:",
            f"{v2['non_eos_tokens']:,}"
        )
        print(f"Token count reduction: {token_reduction:.2f}%")

    print("\nAll EOS and vocabulary checks passed.")
    print("No datasets or checkpoints were modified.")


if __name__ == "__main__":
    main()
