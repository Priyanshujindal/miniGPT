
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = ROOT / "data" / "tokenizer" / "bpe_tokenizer_v3.json"

TRAIN_DATA_PATH = ROOT / "data" / "cleaned" / "train.jsonl"
VAL_DATA_PATH = ROOT / "data" / "cleaned" / "validation.jsonl"

OUTPUT_DIR = ROOT / "data" / "tokenized_v3"

EXPECTED_TRAIN_STORIES = 20000
EXPECTED_VAL_STORIES = 2000


def load_stories(file_path):
    stories = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path.name}, "
                    f"line {line_number}: {error}"
                ) from error

            text = record.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Missing or empty story in {file_path.name}, "
                    f"line {line_number}"
                )

            stories.append(text)

    return stories


def tokenize_dataset(stories, tokenizer, eos_token_id, name):
    all_tokens = []

    total_characters = 0
    total_bytes = 0
    total_story_tokens = 0

    for index, story in enumerate(stories, start=1):
        encoded = tokenizer.encode(story)
        story_tokens = encoded.ids

        if not story_tokens:
            raise ValueError(
                f"Story {index} in {name} produced no tokens."
            )

        if max(story_tokens) >= eos_token_id:
            raise ValueError(
                f"Unexpected token ID in {name}, story {index}."
            )

        all_tokens.extend(story_tokens)
        all_tokens.append(eos_token_id)

        total_story_tokens += len(story_tokens)
        total_characters += len(story)
        total_bytes += len(story.encode("utf-8"))

    token_array = np.asarray(all_tokens, dtype=np.uint16)

    if token_array.size == 0:
        raise ValueError(f"{name} dataset is empty.")

    if int(token_array.max()) > eos_token_id:
        raise ValueError(f"Invalid token ID found in {name}.")

    stats = {
        "stories": len(stories),
        "characters": total_characters,
        "utf8_bytes": total_bytes,
        "tokens_without_eos": total_story_tokens,
        "eos_tokens": len(stories),
        "total_tokens": int(token_array.size),
        "characters_per_token": (
            total_characters / total_story_tokens
            if total_story_tokens else 0
        ),
        "bytes_per_token": (
            total_bytes / total_story_tokens
            if total_story_tokens else 0
        ),
        "dtype": str(token_array.dtype),
        "maximum_token_id": int(token_array.max()),
    }

    print(f"\n{name} dataset:")
    print("-" * 60)

    for key, value in stats.items():
        print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")

    return token_array, stats


def main():
    print("=" * 70)
    print("MINIGPT V3 TOKEN DATA PREPARATION")
    print("=" * 70)

    for path in [TOKENIZER_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    vocab_size = tokenizer.get_vocab_size()
    eos_token_id = vocab_size

    if vocab_size != 2048:
        raise ValueError(
            f"Expected tokenizer vocabulary 2048, got {vocab_size}."
        )

    print("\nTokenizer loaded:", TOKENIZER_PATH)
    print("Vocabulary size:", vocab_size)
    print("EOS token ID:", eos_token_id)
    print("Model vocabulary size:", vocab_size + 1)

    print("\nLoading cleaned datasets...")

    train_stories = load_stories(TRAIN_DATA_PATH)
    val_stories = load_stories(VAL_DATA_PATH)

    print("Training stories:", len(train_stories))
    print("Validation stories:", len(val_stories))

    if len(train_stories) != EXPECTED_TRAIN_STORIES:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_STORIES} training stories, "
            f"got {len(train_stories)}."
        )

    if len(val_stories) != EXPECTED_VAL_STORIES:
        raise ValueError(
            f"Expected {EXPECTED_VAL_STORIES} validation stories, "
            f"got {len(val_stories)}."
        )

    print("\nTokenizing training dataset...")

    train_tokens, train_stats = tokenize_dataset(
        train_stories,
        tokenizer,
        eos_token_id,
        "TRAIN"
    )

    print("\nTokenizing validation dataset...")

    val_tokens, val_stats = tokenize_dataset(
        val_stories,
        tokenizer,
        eos_token_id,
        "VALIDATION"
    )

    train_output = OUTPUT_DIR / "train.npy"
    val_output = OUTPUT_DIR / "validation.npy"
    stats_output = OUTPUT_DIR / "tokenization_stats.json"

    np.save(train_output, train_tokens)
    np.save(val_output, val_tokens)

    report = {
        "version": "v3",
        "tokenizer": "data/tokenizer/bpe_tokenizer_v3.json",
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "model_vocab_size": vocab_size + 1,
        "dtype": "uint16",
        "training": train_stats,
        "validation": val_stats,
    }

    with stats_output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("V3 TOKENIZATION COMPLETED")
    print("=" * 70)

    print("\nSaved files:")
    print(train_output)
    print(val_output)
    print(stats_output)

    print("\nFinal array checks:")

    for name, tokens in [
        ("Training", train_tokens),
        ("Validation", val_tokens),
    ]:
        print(
            f"{name}: shape={tokens.shape}, "
            f"dtype={tokens.dtype}, "
            f"min={tokens.min()}, "
            f"max={tokens.max()}"
        )

        assert tokens.dtype == np.uint16
        assert int(tokens.max()) <= eos_token_id

    print("\nAll V3 tokenization checks passed.")


if __name__ == "__main__":
    main()
