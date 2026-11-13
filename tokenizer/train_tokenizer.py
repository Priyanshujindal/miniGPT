
import json
from pathlib import Path
from tqdm import tqdm

from bpe import train_bpe, BPETokenizer


# -----------------------------
# CONFIGURATION
# -----------------------------

TRAIN_FILE = Path("data/raw/train.jsonl")

OUTPUT_FILE = Path(
    "data/tokenizer/bpe_tokenizer.json"
)

VOCAB_SIZE = 512

MAX_TRAINING_BYTES = 100_000


# -----------------------------
# LOAD TRAINING TEXT
# -----------------------------

def load_training_sequences():

    sequences = []
    total_bytes = 0

    with open(
        TRAIN_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in tqdm(
            file,
            desc="Reading training stories"
        ):

            record = json.loads(line)

            text = record["text"]

            encoded = text.encode("utf-8")

            sequences.append(list(encoded))

            total_bytes += len(encoded)

            if total_bytes >= MAX_TRAINING_BYTES:
                break

    print("\nDataset loaded.")

    print("Number of sequences:", len(sequences))
    print("Total bytes:", total_bytes)

    return sequences


# -----------------------------
# TRAIN BPE
# -----------------------------

def main():

    sequences = load_training_sequences()

    print("\nStarting BPE training...")

    merges = train_bpe(
        sequences,
        vocab_size=VOCAB_SIZE
    )

    tokenizer = BPETokenizer(merges)

    tokenizer.save(OUTPUT_FILE)

    print("\nTOKENIZER TRAINING COMPLETE")

    print("Base vocabulary:", 256)
    print("Learned merges:", len(merges))
    print("Final vocabulary:", tokenizer.vocab_size)

    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
