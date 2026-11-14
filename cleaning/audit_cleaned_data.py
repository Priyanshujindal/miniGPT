
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "train": (
        ROOT / "data" / "raw" / "train.jsonl",
        ROOT / "data" / "cleaned" / "train.jsonl",
    ),
    "validation": (
        ROOT / "data" / "raw" / "validation.jsonl",
        ROOT / "data" / "cleaned" / "validation.jsonl",
    ),
}


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def show_context(text, position, radius=65):
    start = max(0, position - radius)
    end = min(len(text), position + radius)

    return text[start:end].replace("\n", " ")


def audit_dataset(name, raw_path, cleaned_path):
    raw_records = load_jsonl(raw_path)
    cleaned_records = load_jsonl(cleaned_path)

    print("\n" + "=" * 75)
    print(f"{name.upper()} DATASET AUDIT")
    print("=" * 75)

    if len(raw_records) != len(cleaned_records):
        raise RuntimeError("Story count mismatch.")

    # Inspect residual â€ sequences in the original data.
    # Exclude known complete sequences that were intentionally replaced.
    residual_pattern = re.compile(r"â€(?![™œ–—])")

    residual_count = 0

    print("\n1. ORIGINAL RESIDUAL â€ SEQUENCES")
    print("-" * 75)

    for record in raw_records:
        text = record["text"]

        for match in residual_pattern.finditer(text):
            residual_count += 1

            if residual_count <= 30:
                print(
                    f"ID {record.get('id')}: "
                    f"{show_context(text, match.start())!r}"
                )

    print("Total residual sequences:", residual_count)

    # Inspect remaining Ã markers in cleaned data.
    print("\n2. REMAINING Ã MARKERS IN CLEANED DATA")
    print("-" * 75)

    remaining_count = 0

    for record in cleaned_records:
        text = record["text"]

        for match in re.finditer("Ã", text):
            remaining_count += 1

            if remaining_count <= 30:
                print(
                    f"ID {record.get('id')}: "
                    f"{show_context(text, match.start())!r}"
                )

    print("Total remaining Ã markers:", remaining_count)

    # Show representative before/after quotation replacements.
    print("\n3. BEFORE/AFTER QUOTATION EXAMPLES")
    print("-" * 75)

    examples_shown = 0

    for raw, cleaned in zip(raw_records, cleaned_records):
        raw_text = raw["text"]
        cleaned_text = cleaned["text"]

        if "â€" in raw_text and "”" in cleaned_text:
            position = raw_text.find("â€")

            print("Story ID:", raw.get("id"))
            print("BEFORE:", show_context(raw_text, position))
            print("AFTER: ", show_context(cleaned_text, position))

            examples_shown += 1

            if examples_shown >= 10:
                break

    print("\nAudit completed for", name)


def main():
    for name, paths in DATASETS.items():
        audit_dataset(name, paths[0], paths[1])


if __name__ == "__main__":
    main()
