
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ROOT / "data" / "cleaned" / "train.jsonl",
    ROOT / "data" / "cleaned" / "validation.jsonl",
]

REPLACEMENTS = {
    "dance.So": "dance. So",
    "doit every Tuesday": "do it every Tuesday",
}


def main():
    total = 0

    for path in FILES:
        records = []
        corrected = 0

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = record["text"]

                for wrong, right in REPLACEMENTS.items():
                    count = text.count(wrong)
                    text = text.replace(wrong, right)
                    corrected += count

                record["text"] = text
                records.append(record)

        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        print(f"{path.name}: {corrected} corrections")
        total += corrected

    print(f"\nTotal corrections: {total}")


if __name__ == "__main__":
    main()
