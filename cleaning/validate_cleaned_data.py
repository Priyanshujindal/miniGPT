
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "train": ROOT / "data" / "cleaned" / "train.jsonl",
    "validation": ROOT / "data" / "cleaned" / "validation.jsonl",
}

EXPECTED_COUNTS = {
    "train": 20000,
    "validation": 2000,
}

BAD_MARKERS = [
    "â€™", "â€œ", "â€˜", "â€“", "â€”",
    "â€¦", "â€‹", "â€Š", "Ã©", "Ã±",
    "Ã¡", "Ã³", "Â¡",
]


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text):
    normalized = normalize_text(text)
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def inspect_file(name, path):
    records = []
    ids = []
    hashes = []
    marker_counts = Counter()
    invalid_json = 0
    empty_text = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON at line {line_number}: {exc}")
                invalid_json += 1
                continue

            if not isinstance(record, dict):
                invalid_json += 1
                continue

            records.append(record)
            ids.append(record.get("id"))

            text = record.get("text", "")

            if not isinstance(text, str) or not text.strip():
                empty_text += 1
                continue

            hashes.append(text_hash(text))

            for marker in BAD_MARKERS:
                marker_counts[marker] += text.count(marker)

    duplicate_ids = [
        value for value, count in Counter(ids).items()
        if count > 1
    ]

    duplicate_stories = [
        value for value, count in Counter(hashes).items()
        if count > 1
    ]

    remaining = {
        marker: count
        for marker, count in marker_counts.items()
        if count > 0
    }

    report = {
        "count_ok": len(records) == EXPECTED_COUNTS[name],
        "duplicate_ids": duplicate_ids,
        "duplicate_stories": duplicate_stories,
        "invalid_json": invalid_json,
        "empty_text": empty_text,
        "markers": remaining,
        "hashes": set(hashes),
    }

    print(f"\n{'=' * 65}")
    print(f"{name.upper()} DATASET")
    print("=" * 65)
    print("Stories:", len(records))
    print("Expected:", EXPECTED_COUNTS[name])
    print("Count correct:", report["count_ok"])
    print("Unique IDs:", len(set(ids)))
    print("Duplicate IDs within file:", len(duplicate_ids))
    print("Duplicate stories within file:", len(duplicate_stories))
    print("Invalid JSON:", invalid_json)
    print("Empty/missing text:", empty_text)
    print("Remaining known mojibake:", remaining or "None")

    return report


def main():
    reports = {}

    for name, path in FILES.items():
        reports[name] = inspect_file(name, path)

    train_hashes = reports["train"]["hashes"]
    validation_hashes = reports["validation"]["hashes"]

    overlap = train_hashes & validation_hashes

    print(f"\n{'=' * 65}")
    print("TRAIN / VALIDATION CONTENT LEAKAGE")
    print("=" * 65)
    print("Matching normalized stories:", len(overlap))

    passed = True

    for report in reports.values():
        if (
            not report["count_ok"]
            or report["duplicate_ids"]
            or report["duplicate_stories"]
            or report["invalid_json"]
            or report["empty_text"]
            or report["markers"]
        ):
            passed = False

    if overlap:
        passed = False

    print("\nFINAL RESULT:", "PASS" if passed else "FAIL")

    if not passed:
        raise SystemExit(
            "Dataset validation failed. Do not train."
        )


if __name__ == "__main__":
    main()
