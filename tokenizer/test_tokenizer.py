
import json
from pathlib import Path

from bpe import BPETokenizer


TOKENIZER_FILE = Path(
    "data/tokenizer/bpe_tokenizer.json"
)

TEST_FILE = Path("data/raw/validation.jsonl")


def main():

    tokenizer = BPETokenizer.load(
        TOKENIZER_FILE
    )

    print("Vocabulary size:", tokenizer.vocab_size)

    # Test normal and Unicode text.
    test_cases = [
        "Hello world!",
        "Artificial intelligence is interesting.",
        "I am building my own language model.",
        "Machine learning and deep learning.",
        "Hello! नमस्ते! 你好!",
        "Numbers: 12345, symbols: @#$%",
        ""
    ]

    print("\nBASIC TOKENIZER TESTS")

    for text in test_cases:

        token_ids = tokenizer.encode(text)

        decoded = tokenizer.decode(token_ids)

        assert decoded == text, (
            f"Round-trip failed for: {text}"
        )

        print("\nOriginal:", repr(text))
        print("Token IDs:", token_ids)
        print("Token count:", len(token_ids))
        print("Decoded:", repr(decoded))

    print("\nAll basic tests passed!")

    # Test real validation stories.
    print("\nVALIDATION DATASET TEST")

    with open(
        TEST_FILE,
        encoding="utf-8"
    ) as file:

        records = [
            json.loads(next(file))
            for _ in range(5)
        ]

    total_chars = 0
    total_tokens = 0

    for record in records:

        text = record["text"]

        token_ids = tokenizer.encode(text)

        decoded = tokenizer.decode(token_ids)

        assert decoded == text

        total_chars += len(text)
        total_tokens += len(token_ids)

    print("Stories tested:", len(records))
    print("Total characters:", total_chars)
    print("Total tokens:", total_tokens)

    print(
        "Characters per token:",
        round(total_chars / total_tokens, 3)
        if total_tokens else 0
    )

    # Test saving and loading.
    tokenizer.save("data/tokenizer/test_saved.json")

    reloaded = BPETokenizer.load(
        "data/tokenizer/test_saved.json"
    )

    sample = "Testing tokenizer persistence!"

    assert (
        tokenizer.encode(sample)
        == reloaded.encode(sample)
    )

    print("\nSave/load test passed!")
    print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    main()
