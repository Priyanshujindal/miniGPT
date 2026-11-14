
import json
from pathlib import Path
from collections import Counter

FILES = [
    Path("data/raw/train.jsonl"),
    Path("data/raw/validation.jsonl"),
]

# Exact suspicious sequences, not broad character matches.
MOJIBAKE_PATTERNS = [
    "â€™",  # Often corrupted right apostrophe
    "â€œ",  # Often corrupted left double quote
    "â€�",  # Often corrupted right double quote
    "â€“",  # Often corrupted en dash
    "â€”",  # Often corrupted em dash
    "Ã",
]

# Check whether spaces are missing in the actual source text.
SPACE_PATTERNS = [
    "fenceand",
    "makinga",
    "toyou",
    "everytuesday",
    "andsaid",
    "car,kitty",
    "bigleaves",
]


def load_records(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def inspect_file(path):
    records = load_records(path)

    print("\n" + "=" * 75)
    print("FILE:", path)
    print("TOTAL STORIES:", len(records))
    print("=" * 75)

    for label, patterns in [
        ("MOJIBAKE", MOJIBAKE_PATTERNS),
        ("POSSIBLY MISSING SPACES", SPACE_PATTERNS),
    ]:
        print(f"\n--- {label} ---")

        for pattern in patterns:
            matches = []

            for record in records:
                text = record.get("text", "")
                count = text.lower().count(pattern.lower())

                if count:
                    matches.append((record, count))

            total_occurrences = sum(count for _, count in matches)

            print(
                f"\n{pattern!r}: "
                f"{total_occurrences} occurrences "
                f"in {len(matches)} stories"
            )

            for record, _ in matches[:2]:
                story = record.get("text", "")
                index = story.lower().find(pattern.lower())

                start = max(0, index - 70)
                end = min(len(story), index + len(pattern) + 70)
                sample = story[start:end]

                print("  ID:", record.get("id"))
                print("  TEXT:", repr(sample))
                print(
                    "  CODEPOINTS:",
                    [f"U+{ord(char):04X}" for char in sample]
                )

    print("\n--- FIRST THREE STORIES ---")

    for record in records[:3]:
        print("\nID:", record.get("id"))
        print("TEXT:", repr(record.get("text", "")[:500]))


for file_path in FILES:
    inspect_file(file_path)
