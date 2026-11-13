
import json
import hashlib
from pathlib import Path
from collections import Counter


TRAIN_FILE = Path("data/raw/train.jsonl")
VAL_FILE = Path("data/raw/validation.jsonl")


def load_records(filename):

    records = []

    with open(filename, encoding="utf-8") as file:

        for line in file:
            records.append(json.loads(line))

    return records


def fingerprint(text):

    normalized = " ".join(text.lower().split())

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def inspect(records, name):

    texts = [record["text"] for record in records]

    char_counts = [len(text) for text in texts]

    word_counts = [
        len(text.split()) for text in texts
    ]

    total_chars = sum(char_counts)
    total_words = sum(word_counts)

    print(f"\n{'=' * 45}")
    print(f"{name} DATASET")
    print(f"{'=' * 45}")

    print("Stories:", len(texts))
    print("Total characters:", total_chars)
    print("Total words:", total_words)

    print(
        "Average characters:",
        round(total_chars / len(texts), 2)
    )

    print(
        "Average words:",
        round(total_words / len(texts), 2)
    )

    print("Shortest story:", min(char_counts))
    print("Longest story:", max(char_counts))

    print(
        "Unique stories:",
        len(set(fingerprint(t) for t in texts))
    )

    print("\nSample stories:")

    for i, text in enumerate(texts[:3]):

        print(f"\nStory {i + 1}:")
        print(text[:500])


def main():

    train = load_records(TRAIN_FILE)
    validation = load_records(VAL_FILE)

    inspect(train, "TRAINING")
    inspect(validation, "VALIDATION")

    train_hashes = {
        fingerprint(record["text"])
        for record in train
    }

    val_hashes = {
        fingerprint(record["text"])
        for record in validation
    }

    overlap = train_hashes.intersection(val_hashes)

    print("\n" + "=" * 45)
    print("DATA LEAKAGE CHECK")
    print("=" * 45)

    print("Overlapping stories:", len(overlap))

    if overlap:
        print("WARNING: Duplicate stories across splits!")
    else:
        print("PASS: No exact duplicate stories across splits.")


if __name__ == "__main__":
    main()
