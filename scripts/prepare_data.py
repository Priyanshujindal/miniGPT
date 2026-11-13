
import json
import re
import hashlib
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# -----------------------------
# CONFIGURATION
# -----------------------------

DATASET_NAME = "roneneldan/TinyStories"

TRAIN_LIMIT = 20_000
VAL_LIMIT = 2_000

OUTPUT_DIR = Path("data")

RAW_DIR = OUTPUT_DIR / "raw"
PROCESSED_DIR = OUTPUT_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# TEXT CLEANING
# -----------------------------

def clean_text(text):

    if not isinstance(text, str):
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_hash(text):

    normalized = " ".join(text.lower().split())

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# -----------------------------
# SAVE DATA
# -----------------------------

def save_records(records, filename):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


def save_text(records, filename):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:
            file.write(record["text"])
            file.write("\n\n")


# -----------------------------
# DOWNLOAD DATASET
# -----------------------------

def download_split(split, limit):

    print(f"\nDownloading {split} data...")

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True
    )

    if split == "train":
        dataset = dataset.shuffle(
            seed=42,
            buffer_size=10_000
        )

    records = []
    seen_hashes = set()

    for example in tqdm(
        dataset,
        desc=f"Processing {split}",
        total=limit
    ):

        text = clean_text(example.get("text", ""))

        # Remove empty or very short samples
        if len(text) < 20:
            continue

        text_hash = get_hash(text)

        # Remove duplicates within this split
        if text_hash in seen_hashes:
            continue

        seen_hashes.add(text_hash)

        records.append({
            "id": len(records),
            "text": text,
            "characters": len(text)
        })

        if len(records) >= limit:
            break

    if len(records) < limit:
        raise RuntimeError(
            f"Only collected {len(records)} records "
            f"for {split}; expected {limit}."
        )

    return records


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():

    train_records = download_split(
        "train",
        TRAIN_LIMIT
    )

    val_records = download_split(
        "validation",
        VAL_LIMIT
    )

    # Save raw JSONL records
    save_records(
        train_records,
        RAW_DIR / "train.jsonl"
    )

    save_records(
        val_records,
        RAW_DIR / "validation.jsonl"
    )

    # Save processed plain-text corpus
    save_text(
        train_records,
        PROCESSED_DIR / "train.txt"
    )

    save_text(
        val_records,
        PROCESSED_DIR / "validation.txt"
    )

    print("\nDATA PREPARATION COMPLETE")

    print("Training stories:", len(train_records))
    print("Validation stories:", len(val_records))

    print(
        "Training characters:",
        sum(r["characters"] for r in train_records)
    )

    print(
        "Validation characters:",
        sum(r["characters"] for r in val_records)
    )


if __name__ == "__main__":
    main()
