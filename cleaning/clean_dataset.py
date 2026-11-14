
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILES = {
    "train": PROJECT_ROOT / "data" / "raw" / "train.jsonl",
    "validation": PROJECT_ROOT / "data" / "raw" / "validation.jsonl",
}

OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"

# Exact, verified replacements. Process these before truncated â€ sequences.
EXACT_REPLACEMENTS = {
    "â€™": "’",
    "â€œ": "“",
    "â€˜": "‘",
    "â€“": "–",
    "â€”": "—",
    "â€¦": "…",
    "â€‹": "",
    "â€Š": " ",
    "Ã©": "é",
    "Ã±": "ñ",
    "Ã¡": "á",
    "Ã³": "ó",
    "Â¡": "¡",
}

SPACING_REVIEW_TERMS = [
    "andonce",
    "andthrew",
    "namedjack",
    "overto",
    "mattressback",
    "sheclimbed",
    "climbthe",
    "therewasa",
    "cometo",
    "said,\"",
    "herewe",
    "helpyou",
    "canuse",
    "thoughtit",
    "smiledback",
]

SPELLING_REVIEW_TERMS = [
    "smileing",
]


def repair_text(text):
    """Repair known corruption while preserving unrelated Unicode."""

    counts = Counter()
    cleaned = text

    # First, repair complete and unambiguous sequences.
    for corrupted, corrected in EXACT_REPLACEMENTS.items():
        occurrences = cleaned.count(corrupted)

        if occurrences:
            cleaned = cleaned.replace(corrupted, corrected)
            counts[corrupted] += occurrences

    # Remaining â€ sequences are truncated or malformed.
    # A following lowercase letter or punctuation/whitespace generally
    # indicates a closing quotation mark in the inspected corpus.
    closing_pattern = re.compile(
        r"â€(?=[a-z\s.,!?;:)'\"]|$)"
    )

    cleaned, closing_count = closing_pattern.subn("”", cleaned)

    if closing_count:
        counts["â€ [closing quote]"] += closing_count

    # In the inspected corpus, truncated opening quotes are followed
    # immediately by an uppercase letter and preceded by a boundary.
    opening_pattern = re.compile(
        r"(?<=[\s,:;.!?])â€(?=[A-Z])"
    )

    cleaned, opening_count = opening_pattern.subn("“", cleaned)

    if opening_count:
        counts["â€ [opening quote]"] += opening_count

    return cleaned, counts


def load_records(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}, line {line_number}"
                )

            if not isinstance(record.get("text"), str):
                raise ValueError(
                    f"Missing or invalid text at {path}, line {line_number}"
                )

            records.append(record)

    return records


def find_terms(records, terms):
    results = {}

    for term in terms:
        matches = []

        for record in records:
            story = record["text"]

            for match in re.finditer(re.escape(term), story, re.IGNORECASE):
                start = max(0, match.start() - 45)
                end = min(len(story), match.end() + 45)

                matches.append({
                    "id": record.get("id"),
                    "matched_text": match.group(),
                    "context": story[start:end],
                })

        if matches:
            results[term] = {
                "occurrences": len(matches),
                "examples": matches[:5],
            }

    return results


def find_unresolved_sequences(records):
    markers = [
        "â€",
        "Ã",
        "Â¡",
    ]

    remaining = Counter()
    examples = defaultdict(list)

    for record in records:
        text = record["text"]

        for marker in markers:
            start = 0

            while True:
                position = text.find(marker, start)

                if position == -1:
                    break

                remaining[marker] += 1

                if len(examples[marker]) < 10:
                    left = max(0, position - 45)
                    right = min(len(text), position + 65)

                    examples[marker].append({
                        "id": record.get("id"),
                        "context": text[left:right].replace("\n", " "),
                    })

                start = position + len(marker)

    return dict(remaining), dict(examples)


def process_dataset(name, input_path):
    original_records = load_records(input_path)

    cleaned_records = []
    replacement_counts = Counter()
    changed_stories = 0

    for original in original_records:
        cleaned_record = dict(original)

        original_text = original["text"]
        cleaned_text, counts = repair_text(original_text)

        cleaned_record["text"] = cleaned_text

        if cleaned_text != original_text:
            changed_stories += 1

        replacement_counts.update(counts)
        cleaned_records.append(cleaned_record)

    output_path = OUTPUT_DIR / f"{name}.jsonl"

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in cleaned_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # Reload the output and verify that the JSON is valid.
    verified_records = load_records(output_path)

    if len(original_records) != len(verified_records):
        raise RuntimeError(f"{name}: story count changed.")

    # Verify that record IDs and non-text fields remain unchanged.
    for original, cleaned in zip(original_records, verified_records):
        if original.get("id") != cleaned.get("id"):
            raise RuntimeError(f"{name}: story ID changed.")

        original_other = {
            key: value for key, value in original.items() if key != "text"
        }
        cleaned_other = {
            key: value for key, value in cleaned.items() if key != "text"
        }

        if original_other != cleaned_other:
            raise RuntimeError(
                f"{name}: a non-text field changed for ID {original.get('id')}"
            )

    unresolved, unresolved_examples = find_unresolved_sequences(
        verified_records
    )

    report = {
        "stories": len(verified_records),
        "stories_changed": changed_stories,
        "total_replacements": sum(replacement_counts.values()),
        "replacement_counts": dict(replacement_counts),
        "unresolved_markers": unresolved,
        "unresolved_examples": unresolved_examples,
        "spacing_review": find_terms(
            verified_records,
            SPACING_REVIEW_TERMS,
        ),
        "spelling_review": find_terms(
            verified_records,
            SPELLING_REVIEW_TERMS,
        ),
        "output_verified": True,
    }

    print("\n" + "=" * 70)
    print(f"{name.upper()} DATASET")
    print("=" * 70)
    print("Stories:", report["stories"])
    print("Stories changed:", changed_stories)
    print("Total replacements:", report["total_replacements"])

    print("\nReplacement counts:")
    for key, value in replacement_counts.items():
        print(f"  {key!r}: {value}")

    print("\nUnresolved markers:")
    print(unresolved or "None detected")

    if unresolved_examples:
        print("\nUnresolved examples:")
        for marker, examples in unresolved_examples.items():
            for example in examples:
                print(
                    f"  {marker!r} | ID {example['id']}: "
                    f"{example['context']!r}"
                )

    print("\nSpacing candidates:", {
        key: value["occurrences"]
        for key, value in report["spacing_review"].items()
    })

    print("Spelling candidates:", {
        key: value["occurrences"]
        for key, value in report["spelling_review"].items()
    })

    print("Output:", output_path)

    return report


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reports = {}

    for name, path in INPUT_FILES.items():
        reports[name] = process_dataset(name, path)

    report_path = OUTPUT_DIR / "cleaning_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(reports, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("CLEANING FINISHED")
    print("=" * 70)
    print("Report:", report_path)
    print("Original raw files were not modified.")


if __name__ == "__main__":
    main()
