
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ROOT / "data" / "raw" / "train.jsonl",
    ROOT / "data" / "raw" / "validation.jsonl",
]


def main():
    suffix_counts = Counter()
    examples = defaultdict(list)

    for path in FILES:
        print(f"\nInspecting: {path.name}")

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                text = record["text"]

                start = 0

                while True:
                    position = text.find("â€", start)

                    if position == -1:
                        break

                    suffix_position = position + len("â€")

                    if suffix_position < len(text):
                        suffix = text[suffix_position]
                    else:
                        suffix = "<END>"

                    suffix_counts[suffix] += 1

                    if len(examples[suffix]) < 8:
                        left = max(0, position - 45)
                        right = min(len(text), position + 65)

                        examples[suffix].append({
                            "file": path.name,
                            "id": record.get("id"),
                            "context": text[left:right].replace("\n", " "),
                        })

                    start = position + len("â€")

    print("\n" + "=" * 70)
    print("CHARACTERS FOLLOWING THE â€ PREFIX")
    print("=" * 70)

    for suffix, count in suffix_counts.most_common():
        print(f"\nSuffix: {suffix!r} | Unicode: {ord(suffix):04X}" if suffix != "<END>" else "\nSuffix: <END>")
        print("Count:", count)

        for example in examples[suffix]:
            print(
                f"  {example['file']} | ID {example['id']}: "
                f"{example['context']!r}"
            )

    print("\nTotal â€ prefixes:", sum(suffix_counts.values()))


if __name__ == "__main__":
    main()
