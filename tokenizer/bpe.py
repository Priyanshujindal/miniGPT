
from collections import Counter


class BPETokenizer:

    def __init__(self, merges):

        self.merges = [
            tuple(pair) for pair in merges
        ]

        # Each merge gets a new token ID,
        # starting after the 256 base byte tokens.

        self.ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        # Store the bytes represented by each token.
        self.token_bytes = [
            bytes([i]) for i in range(256)
        ]

        for left, right in self.merges:

            merged_bytes = (
                self.token_bytes[left]
                + self.token_bytes[right]
            )

            self.token_bytes.append(merged_bytes)

        self.vocab_size = len(self.token_bytes)

    def encode(self, text):

        # Convert UTF-8 text into initial byte token IDs.
        tokens = list(text.encode("utf-8"))

        # Repeatedly apply the highest-priority merge.
        while len(tokens) >= 2:

            candidates = []

            for i in range(len(tokens) - 1):

                pair = (tokens[i], tokens[i + 1])

                if pair in self.ranks:

                    candidates.append(
                        (self.ranks[pair], i, pair)
                    )

            if not candidates:
                break

            # Select the merge learned earliest.
            _, _, best_pair = min(candidates)

            new_token_id = (
                256 + self.ranks[best_pair]
            )

            # Merge every non-overlapping occurrence.
            merged = []
            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1])
                    == best_pair
                ):

                    merged.append(new_token_id)
                    i += 2

                else:

                    merged.append(tokens[i])
                    i += 1

            tokens = merged

        return tokens

    def decode(self, token_ids):

        decoded_bytes = bytearray()

        for token_id in token_ids:

            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"Invalid token ID: {token_id}"
                )

            decoded_bytes.extend(
                self.token_bytes[token_id]
            )

        return decoded_bytes.decode(
            "utf-8",
            errors="replace"
        )

    def save(self, filepath):

        import json
        from pathlib import Path

        filepath = Path(filepath)
        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "vocab_size": self.vocab_size,
            "merges": [
                list(pair)
                for pair in self.merges
            ]
        }

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=2
            )

    @classmethod
    def load(cls, filepath):

        import json

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        tokenizer = cls(data["merges"])

        if tokenizer.vocab_size != data["vocab_size"]:
            raise ValueError(
                "Tokenizer vocabulary size mismatch"
            )

        return tokenizer


def merge_pair(tokens, pair, new_token_id):

    merged = []
    i = 0

    while i < len(tokens):

        if (
            i < len(tokens) - 1
            and (tokens[i], tokens[i + 1]) == pair
        ):

            merged.append(new_token_id)
            i += 2

        else:

            merged.append(tokens[i])
            i += 1

    return merged


def train_bpe(sequences, vocab_size):

    merges = []

    current_sequences = [
        list(sequence) for sequence in sequences
    ]

    while 256 + len(merges) < vocab_size:

        # Count every adjacent pair.
        pair_counts = Counter(
            pair
            for sequence in current_sequences
            for pair in zip(
                sequence,
                sequence[1:]
            )
        )

        if not pair_counts:
            break

        # Find the most frequent pair.
        best_pair, frequency = pair_counts.most_common(1)[0]

        # Stop if no pair occurs more than once.
        if frequency < 2:
            break

        new_token_id = 256 + len(merges)

        # Apply the merge to every sequence.
        current_sequences = [
            merge_pair(
                sequence,
                best_pair,
                new_token_id
            )
            for sequence in current_sequences
        ]

        merges.append(best_pair)

        print(
            f"Merge {len(merges)}: "
            f"{best_pair} | "
            f"frequency: {frequency}"
        )

    return merges
