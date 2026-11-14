
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ROOT / "data" / "cleaned" / "train.jsonl",
    ROOT / "data" / "cleaned" / "validation.jsonl",
]

PATTERNS = {
    "andonce": r"\bandonce\b",
    "andthrew": r"\bandthrew\b",
    "namedJack": r"\bnamedJack\b",
    "overto": r"\boverto\b",
    "mattressback": r"\bmattressback\b",
    "birds_comma_her": r'birds,"her',
    "Sheclimbed": r"\bSheclimbed\b",
    "climbthe": r"\bclimbthe\b",
    "therewasa": r"\btherewasa\b",
    "cometo": r"\bcometo\b",
    "Wow_Kitty": r"Wow,Kitty",
    "Herewe": r"\bHerewe\b",
    "canuse": r"\bcanuse\b",
    "thoughtit": r"\bthoughtit\b",
    "smiledback": r"\bsmiledback\b",
}


def main():
    total = 0

    for path in FILES:
        print(f"\n{'=' * 70}")
        print(f"FILE: {path.name}")
        print("=" * 70)

        file_count = 0

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = record["text"]

                for label, pattern in PATTERNS.items():
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        file_count += 1
                        total += 1

                        start = max(0, match.start() - 65)
                        end = min(len(text), match.end() + 65)

                        print(f"\nPattern: {label}")
                        print(f"Story ID: {record.get('id')}")
                        print(f"Match: {match.group()!r}")
                        print(f"Context: {text[start:end]!r}")

        print(f"\nMatches in {path.name}: {file_count}")

    print(f"\nTotal targeted matches: {total}")
    print("No files were modified.")


if __name__ == "__main__":
    main()
