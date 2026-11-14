
from tokenizers import Tokenizer


TOKENIZER_PATH = "data/tokenizer/bpe_tokenizer_v2.json"

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

test_cases = [
    "Hello world!",
    "box that had an idea",
    "Hello! नमस्ते! 你好!",
    'He said, "Hello!"',
    "The girl's toy was beautiful.",
    "They played together in the garden."
]

print("=" * 65)
print("V2 TOKENIZER ROUND-TRIP TEST")
print("=" * 65)

all_passed = True

for original in test_cases:
    encoded = tokenizer.encode(original)
    decoded = tokenizer.decode(encoded.ids)

    passed = original == decoded
    all_passed = all_passed and passed

    print("\nOriginal:", repr(original))
    print("Decoded: ", repr(decoded))
    print("Token IDs:", encoded.ids)
    print("Result:", "PASS" if passed else "FAIL")

print("\n" + "=" * 65)
print("OVERALL:", "ALL TESTS PASSED" if all_passed else "FAILURES FOUND")
print("=" * 65)
