
import json
import time
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "train.jsonl"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "tokenizer"
    / "bpe_tokenizer_v2.json"
)

VOCAB_SIZE = 2048
TARGET_TRAINING_BYTES = 1_000_000
MIN_FREQUENCY = 2


def load_training_texts():
    texts = []
    total_bytes = 0

    with open(TRAIN_FILE, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            text = record["text"]

            texts.append(text)
            total_bytes += len(text.encode("utf-8"))

            if total_bytes >= TARGET_TRAINING_BYTES:
                break

    print("Training stories:", len(texts))
    print("Training bytes:", total_bytes)
    print("Training MB:", round(total_bytes / 1_000_000, 3))

    return texts


def create_tokenizer():
    tokenizer = Tokenizer(
        models.BPE(unk_token=None)
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    return tokenizer, trainer


def test_tokenizer(tokenizer):
    test_cases = [
        "Hello world!",
        "Once upon a time",
        "The little girl went to the garden.",
        "Artificial intelligence is changing the world.",
        "I am building my own language model.",
        "Numbers: 12345, symbols: @#$%",
        "Hello! नमस्ते! 你好!",
        ""
    ]

    print("\nTOKENIZER ROUND-TRIP TESTS")
    print("-" * 50)

    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)

        assert decoded == text, (
            f"Round-trip failed: {repr(text)} -> {repr(decoded)}"
        )

        print(
            repr(text),
            "->",
            len(encoded.ids),
            "tokens"
        )

    print("\nAll round-trip tests passed!")


def measure_efficiency(tokenizer, texts, label):
    total_characters = 0
    total_tokens = 0

    for text in texts:
        encoded = tokenizer.encode(text)

        total_characters += len(text)
        total_tokens += len(encoded.ids)

    chars_per_token = (
        total_characters / total_tokens
        if total_tokens else 0
    )

    print(f"\n{label}")
    print("-" * 50)
    print("Stories:", len(texts))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)
    print("Characters per token:", round(chars_per_token, 3))


def main():
    print("=" * 60)
    print("MINIGPT TOKENIZER V2 TRAINING")
    print("=" * 60)

    texts = load_training_texts()

    tokenizer, trainer = create_tokenizer()

    print("\nStarting BPE training...")

    start_time = time.perf_counter()

    tokenizer.train_from_iterator(
        texts,
        trainer=trainer,
        length=len(texts)
    )

    training_duration = time.perf_counter() - start_time

    actual_vocab_size = tokenizer.get_vocab_size()

    print("\nTRAINING COMPLETE")
    print("-" * 50)
    print("Requested vocabulary:", VOCAB_SIZE)
    print("Actual vocabulary:", actual_vocab_size)
    print("Training duration:", round(training_duration, 2), "seconds")

    if actual_vocab_size != VOCAB_SIZE:
        raise RuntimeError(
            f"Expected vocabulary size {VOCAB_SIZE}, "
            f"but got {actual_vocab_size}. "
            "Do not proceed until this is investigated."
        )

    # EOS will be added separately during dataset tokenization.
    # Its ID is immediately after the tokenizer vocabulary.
    eos_token_id = actual_vocab_size

    print("EOS token ID:", eos_token_id)
    print("Model vocabulary size:", actual_vocab_size + 1)

    test_tokenizer(tokenizer)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer.save(str(OUTPUT_FILE))

    print("\nTokenizer saved to:")
    print(OUTPUT_FILE)

    # Reload the saved tokenizer to verify persistence.
    reloaded = Tokenizer.from_file(str(OUTPUT_FILE))

    assert reloaded.get_vocab_size() == actual_vocab_size

    sample = "Testing tokenizer persistence!"
    assert (
        reloaded.decode(reloaded.encode(sample).ids)
        == sample
    )

    print("\nSave/load test passed!")

    # Compare tokenization efficiency using the same sample.
    measure_efficiency(
        reloaded,
        texts[:500],
        "V2 TRAINING SAMPLE"
    )

    print("\n" + "=" * 60)
    print("TOKENIZER V2 FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
