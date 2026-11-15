import json
from pathlib import Path
from tokenizers import Tokenizer, AddedToken

from training.config_v4 import (
    TOKENIZER_V3_PATH,
    TOKENIZER_V4_PATH,
    TOKENIZER_V4_REPORT_PATH,
    TOKENIZER_BASE_VOCAB_SIZE,
    VOCAB_SIZE,
    SPECIAL_TOKENS,
    END_TOKEN_ID,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    ASSISTANT_TOKEN_ID,
    END_OF_TURN_TOKEN_ID,
)


def build_tokenizer_v4():
    print("=" * 70)
    print("MINIGPT V4 TOKENIZER CREATION & VERIFICATION")
    print("=" * 70)

    if not TOKENIZER_V3_PATH.exists():
        raise FileNotFoundError(f"Base V3 tokenizer not found: {TOKENIZER_V3_PATH}")

    print(f"\n1. Loading base V3 tokenizer from: {TOKENIZER_V3_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_V3_PATH))
    base_vocab_size = tokenizer.get_vocab_size()
    print(f"Base V3 vocabulary size: {base_vocab_size}")

    if base_vocab_size != TOKENIZER_BASE_VOCAB_SIZE:
        raise ValueError(
            f"Expected base vocabulary size {TOKENIZER_BASE_VOCAB_SIZE}, got {base_vocab_size}"
        )

    print("\n2. Adding V4 conversational special tokens...")
    added_tokens = [
        AddedToken(token, single_word=False, normalized=False, special=True)
        for token in SPECIAL_TOKENS
    ]
    num_added = tokenizer.add_special_tokens(added_tokens)
    print(f"Successfully added {num_added} special tokens.")

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"New V4 vocabulary size: {actual_vocab_size}")

    if actual_vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"Expected vocabulary size {VOCAB_SIZE}, got {actual_vocab_size}"
        )

    # Verify exact ID assignments
    expected_ids = {
        "<|end|>": END_TOKEN_ID,
        "<|system|>": SYSTEM_TOKEN_ID,
        "<|user|>": USER_TOKEN_ID,
        "<|assistant|>": ASSISTANT_TOKEN_ID,
        "<|endofturn|>": END_OF_TURN_TOKEN_ID,
    }

    print("\n3. Validating special token ID assignments:")
    for token_str, expected_id in expected_ids.items():
        actual_id = tokenizer.token_to_id(token_str)
        print(f"  {token_str:15s} -> ID {actual_id} (expected {expected_id})")
        if actual_id != expected_id:
            raise ValueError(
                f"Token ID mismatch for {token_str}: got {actual_id}, expected {expected_id}"
            )

    # Save V4 tokenizer
    TOKENIZER_V4_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_V4_PATH))
    print(f"\n4. Saved V4 tokenizer to: {TOKENIZER_V4_PATH}")

    # Comprehensive Round-Trip & Encoding Tests
    print("\n5. Running round-trip and token isolation tests...")
    test_cases = [
        "Hello world!",
        "Once upon a time, there was a little girl.",
        "What is the capital of France?",
        "def add(a, b): return a + b",
        "2 + 2 = 4 and 10 * 5 = 50.",
        "<|system|>You are a helpful assistant.<|endofturn|>",
        "<|user|>Hello!<|endofturn|><|assistant|>Hi there! How can I help?<|endofturn|><|end|>",
    ]

    all_passed = True
    for original in test_cases:
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        passed = (decoded == original)
        if not passed:
            all_passed = False
            print(f"FAIL | Original: {original!r}")
            print(f"     | Decoded:  {decoded!r}")
        else:
            print(f"PASS | Tokens: {len(encoded.ids):3d} | {original!r}")

    if not all_passed:
        raise RuntimeError("One or more round-trip tests failed!")

    # Verify existing V3 token invariance
    v3_base_test = "The quick brown fox jumps over the lazy dog."
    v3_tok = Tokenizer.from_file(str(TOKENIZER_V3_PATH))
    v3_ids = v3_tok.encode(v3_base_test).ids
    v4_ids = tokenizer.encode(v3_base_test).ids
    if v3_ids != v4_ids:
        raise RuntimeError(
            f"V3 token invariance violated! V3: {v3_ids} != V4: {v4_ids}"
        )
    print("PASS | V3 token invariance verified (identical IDs for standard text).")

    # Save detailed report
    report = {
        "version": "v4",
        "base_version": "v3",
        "base_vocab_size": base_vocab_size,
        "vocab_size": actual_vocab_size,
        "special_tokens": expected_ids,
        "round_trip_tests_passed": all_passed,
        "v3_invariance_verified": True,
    }

    with open(TOKENIZER_V4_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"\nReport written to: {TOKENIZER_V4_REPORT_PATH}")
    print("\nV4 Tokenizer build complete and verified!")


if __name__ == "__main__":
    build_tokenizer_v4()
