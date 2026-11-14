
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenizer"
    / "bpe_tokenizer_v2.json"
)

TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "train.jsonl"
VAL_FILE = PROJECT_ROOT / "data" / "raw" / "validation.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "data" / "tokenized_v2"

TRAIN_OUTPUT = OUTPUT_DIR / "train.npy"
VAL_OUTPUT = OUTPUT_DIR / "validation.npy"
STATS_OUTPUT = OUTPUT_DIR / "dataset_stats.json"

EOS_TOKEN_ID = 2048
MODEL_VOCAB_SIZE = 2049


def tokenize_file(input_file, output_file, tokenizer, label):
    all_tokens = []

    total_characters = 0
    total_stories = 0
    total_tokens_without_eos = 0

    with open(input_file, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc=f"Tokenizing {label}"):

            record = json.loads(line)
            text = record["text"]

            token_ids = tokenizer.encode(text).ids

            if not token_ids and text:
                raise ValueError(
                    f"Tokenization failed for story {record.get('id')}"
                )

            if any(token_id < 0 or token_id >= EOS_TOKEN_ID
                   for token_id in token_ids):
                raise ValueError(
                    "Tokenizer produced an invalid token ID."
                )

            total_characters += len(text)
            total_tokens_without_eos += len(token_ids)
            total_stories += 1

            # Append EOS after every story.
            token_ids.append(EOS_TOKEN_ID)

            all_tokens.extend(token_ids)

    token_array = np.asarray(all_tokens, dtype=np.uint16)

    if token_array.size == 0:
        raise ValueError(f"No tokens generated for {label}")

    if int(token_array.max()) >= MODEL_VOCAB_SIZE:
        raise ValueError("Token ID exceeds model vocabulary size.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, token_array)

    stats = {
        "split": label,
        "stories": total_stories,
        "characters": total_characters,
        "tokens_without_eos": total_tokens_without_eos,
        "eos_tokens": total_stories,
        "total_tokens": int(token_array.size),
        "characters_per_token": round(
            total_characters / total_tokens_without_eos, 4
        ) if total_tokens_without_eos else 0,
        "dtype": str(token_array.dtype),
        "minimum_token_id": int(token_array.min()),
        "maximum_token_id": int(token_array.max()),
        "output_file": str(output_file.relative_to(PROJECT_ROOT))
    }

    print(f"\n{label.upper()} RESULTS")
    print("-" * 45)
    print("Stories:", stats["stories"])
    print("Characters:", stats["characters"])
    print("Tokens without EOS:", stats["tokens_without_eos"])
    print("EOS tokens:", stats["eos_tokens"])
    print("Total tokens:", stats["total_tokens"])
    print("Characters per token:", stats["characters_per_token"])
    print("Data type:", stats["dtype"])
    print("Minimum token ID:", stats["minimum_token_id"])
    print("Maximum token ID:", stats["maximum_token_id"])
    print("Saved:", output_file)

    return stats


def main():
    print("=" * 60)
    print("MINIGPT TOKENIZER V2 DATA PREPARATION")
    print("=" * 60)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(TOKENIZER_PATH)

    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError("Raw training or validation data missing.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    actual_vocab_size = tokenizer.get_vocab_size()

    if actual_vocab_size != EOS_TOKEN_ID:
        raise ValueError(
            f"Expected tokenizer vocabulary {EOS_TOKEN_ID}, "
            f"got {actual_vocab_size}"
        )

    print("Tokenizer vocabulary:", actual_vocab_size)
    print("EOS token ID:", EOS_TOKEN_ID)
    print("Model vocabulary size:", MODEL_VOCAB_SIZE)

    start_time = time.perf_counter()

    train_stats = tokenize_file(
        TRAIN_FILE,
        TRAIN_OUTPUT,
        tokenizer,
        "train"
    )

    val_stats = tokenize_file(
        VAL_FILE,
        VAL_OUTPUT,
        tokenizer,
        "validation"
    )

    elapsed = time.perf_counter() - start_time

    dataset_stats = {
        "tokenizer": str(TOKENIZER_PATH.relative_to(PROJECT_ROOT)),
        "tokenizer_vocab_size": actual_vocab_size,
        "eos_token_id": EOS_TOKEN_ID,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "training_duration_seconds": round(elapsed, 2),
        "train": train_stats,
        "validation": val_stats
    }

    with open(STATS_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(dataset_stats, file, indent=2)

    print("\n" + "=" * 60)
    print("V2 DATA PREPARATION COMPLETE")
    print("=" * 60)
    print("Duration:", round(elapsed, 2), "seconds")
    print("Output directory:", OUTPUT_DIR)
    print("Statistics:", STATS_OUTPUT)


if __name__ == "__main__":
    main()
