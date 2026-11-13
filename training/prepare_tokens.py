
import json
from pathlib import Path
import numpy as np

from tokenizer.bpe import BPETokenizer


# -----------------------------
# CONFIGURATION
# -----------------------------

TOKENIZER_PATH = Path(
    "data/tokenizer/bpe_tokenizer.json"
)

TRAIN_FILE = Path("data/raw/train.jsonl")
VAL_FILE = Path("data/raw/validation.jsonl")

OUTPUT_DIR = Path("data/tokenized")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# LOAD TOKENIZER
# -----------------------------

tokenizer = BPETokenizer.load(TOKENIZER_PATH)

EOS_TOKEN_ID = tokenizer.vocab_size

print("Tokenizer vocabulary:", tokenizer.vocab_size)
print("EOS token ID:", EOS_TOKEN_ID)


# -----------------------------
# TOKENIZE DATASET
# -----------------------------

def tokenize_file(input_file, output_file):

    all_tokens = []

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            record = json.loads(line)

            text = record["text"]

            token_ids = tokenizer.encode(text)

            # Append end-of-story marker.
            token_ids.append(EOS_TOKEN_ID)

            all_tokens.extend(token_ids)

    token_array = np.asarray(
        all_tokens,
        dtype=np.uint16
    )

    np.save(output_file, token_array)

    print("\nProcessed:", input_file)
    print("Total tokens:", len(token_array))
    print("Saved:", output_file)


# -----------------------------
# MAIN
# -----------------------------

def main():

    tokenize_file(
        TRAIN_FILE,
        OUTPUT_DIR / "train.npy"
    )

    tokenize_file(
        VAL_FILE,
        OUTPUT_DIR / "validation.npy"
    )

    print("\nTOKENIZATION COMPLETE")


if __name__ == "__main__":
    main()
