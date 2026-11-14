
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ROOT / "data" / "cleaned" / "train.jsonl",
    ROOT / "data" / "cleaned" / "validation.jsonl",
]

REPLACEMENTS = {
    "smileing": "smiling",
}


def main():
    total_replacements = 0

    for path in FILES:
        records = []
        file_replacements = 0

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)

                original = record["text"]
                cleaned = original

                for wrong, correct in REPLACEMENTS.items():
                    count = cleaned.count(wrong)
                    cleaned = cleaned.replace(wrong, correct)
                    file_replacements += count

                record["text"] = cleaned
                records.append(record)

        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        total_replacements += file_replacements

        print(f"{path.name}: {file_replacements} corrections")

    print(f"\nTotal corrections: {total_replacements}")
    print("Finalization completed.")


if __name__ == "__main__":
    main()
