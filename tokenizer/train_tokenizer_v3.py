
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers.trainers import BpeTrainer


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DATA_PATH = ROOT / "data" / "cleaned" / "train.jsonl"
TOKENIZER_PATH = (
    ROOT / "data" / "tokenizer" / "bpe_tokenizer_v3.json"
)
REPORT_PATH = (
    ROOT / "data" / "tokenizer" / "tokenizer_v3_report.json"
)

VOCAB_SIZE = 2048
MIN_FREQUENCY = 2


def training_iterator():
    """Stream cleaned training stories into the tokenizer."""

    with TRAIN_DATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            text = record.get("text")

            if not isinstance(text, str) or not text.strip():
                continue

            yield text


def main():
    print("=" * 70)
    print("MINIGPT V3 TOKENIZER TRAINING")
    print("=" * 70)

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(TRAIN_DATA_PATH)

    TOKENIZER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = Tokenizer(
        models.BPE(unk_token=None)
    )

    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print("\nTokenizer configuration:")
    print("Algorithm: Byte-level BPE")
    print("Target vocabulary:", VOCAB_SIZE)
    print("Minimum frequency:", MIN_FREQUENCY)
    print("Training source:", TRAIN_DATA_PATH)

    print("\nTraining on the complete cleaned training corpus...")

    tokenizer.train_from_iterator(
        training_iterator(),
        trainer=trainer,
    )

    actual_vocab_size = tokenizer.get_vocab_size()

    print("\nTraining completed.")
    print("Actual vocabulary size:", actual_vocab_size)

    if actual_vocab_size != VOCAB_SIZE:
        raise RuntimeError(
            f"Expected vocabulary size {VOCAB_SIZE}, "
            f"got {actual_vocab_size}"
        )

    tokenizer.save(str(TOKENIZER_PATH))

    print("\nSaved tokenizer:", TOKENIZER_PATH)

    # Round-trip tests
    test_cases = [
        "Hello world!",
        "Once upon a time, there was a little girl.",
        "He said, \"Hello!\"",
        "The girl's toy was beautiful.",
        "Hello! नमस्ते! 你好!",
        "They played together in the garden.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print("\nROUND-TRIP TESTS")
    print("-" * 70)

    for original in test_cases:
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded.ids)

        passed = decoded == original

        print(
            f"{'PASS' if passed else 'FAIL'} | "
            f"Tokens: {len(encoded.ids):3d} | "
            f"{original!r}"
        )

        if not passed:
            print("Original:", repr(original))
            print("Decoded: ", repr(decoded))
            raise RuntimeError("Tokenizer round-trip test failed.")

    report = {
        "version": "v3",
        "algorithm": "Byte-level BPE",
        "vocab_size": actual_vocab_size,
        "min_frequency": MIN_FREQUENCY,
        "training_source": str(TRAIN_DATA_PATH.relative_to(ROOT)),
        "training_scope": "complete cleaned training corpus",
        "eos_token_id": actual_vocab_size,
        "model_vocab_size": actual_vocab_size + 1,
        "round_trip_tests": len(test_cases),
        "round_trip_passed": True,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("V3 TOKENIZER READY")
    print("=" * 70)
    print("Tokenizer vocabulary:", actual_vocab_size)
    print("EOS token ID:", actual_vocab_size)
    print("Model vocabulary:", actual_vocab_size + 1)
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
