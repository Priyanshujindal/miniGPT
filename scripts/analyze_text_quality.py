
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tokenizer.bpe import BPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "train.jsonl"
VAL_FILE = PROJECT_ROOT / "data" / "raw" / "validation.jsonl"
TOKENIZER_FILE = PROJECT_ROOT / "data" / "tokenizer" / "bpe_tokenizer.json"

# Sample sizes for tokenizer efficiency measurement.
TRAIN_TOKENIZER_SAMPLE = 500
VAL_TOKENIZER_SAMPLE = 200

# Words that can legitimately contain apparent word combinations.
COMMON_VALID_WORDS = {
    "another", "anything", "anyone", "anywhere",
    "everyone", "everything", "everywhere",
    "something", "someone", "somewhere",
    "without", "within", "throughout",
    "inside", "outside", "into", "onto",
    "before", "after", "under", "over",
    "already", "always", "almost",
    "sometimes", "together", "himself",
    "herself", "themselves", "yourself",
    "himself", "cannot", "maybe",
    "nothing", "nobody", "nowhere",
    "upon", "therefore", "although",
    "because", "around", "another",
}


def load_records(filepath):
    records = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records


def get_words(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def analyze_word_frequencies(train_records, val_records):
    frequencies = Counter()

    for record in train_records + val_records:
        frequencies.update(get_words(record["text"]))

    return frequencies


def find_suspicious_joined_words(records, word_frequencies):
    """
    Heuristic:
    Find whitespace-separated tokens that could be two frequent
    English words accidentally joined together.

    This is not a definitive spelling checker.
    """

    candidates = defaultdict(lambda: {
        "count": 0,
        "examples": []
    })

    for story_index, record in enumerate(records):
        text = record["text"]

        for raw_token in text.split():
            token = re.sub(r"[^a-zA-Z]", "", raw_token).lower()

            if len(token) < 6:
                continue

            if token in COMMON_VALID_WORDS:
                continue

            best_split = None
            best_score = 0

            for split_position in range(1, len(token)):
                left = token[:split_position]
                right = token[split_position:]

                if len(left) < 1 or len(right) < 2:
                    continue

                left_frequency = word_frequencies[left]
                right_frequency = word_frequencies[right]

                if left_frequency < 10 or right_frequency < 10:
                    continue

                score = min(left_frequency, right_frequency)

                if score > best_score:
                    best_score = score
                    best_split = (left, right)

            if best_split:
                joined_word = raw_token.lower()

                candidates[joined_word]["count"] += 1

                if len(candidates[joined_word]["examples"]) < 3:
                    candidates[joined_word]["examples"].append(
                        {
                            "story_id": record.get("id", story_index),
                            "text": raw_token,
                            "suggested_split": " ".join(best_split),
                        }
                    )

    return candidates


def find_punctuation_issues(records):
    """
    Detect punctuation immediately followed by a letter,
    for example: 'Kitty,your' or 'hello.World'.

    Some cases may be legitimate, so inspect examples.
    """

    pattern = re.compile(r"[,.!?;:][A-Za-z]")

    total_matches = 0
    examples = []

    for story_index, record in enumerate(records):
        text = record["text"]

        for match in pattern.finditer(text):
            total_matches += 1

            if len(examples) < 30:
                start = max(0, match.start() - 35)
                end = min(len(text), match.end() + 45)

                examples.append({
                    "story_id": record.get("id", story_index),
                    "context": text[start:end].replace("\n", " "),
                })

    return total_matches, examples


def analyze_tokenizer(tokenizer, records, label):
    total_characters = 0
    total_tokens = 0
    stories_processed = 0

    for record in records:
        text = record["text"]
        token_ids = tokenizer.encode(text)

        total_characters += len(text)
        total_tokens += len(token_ids)
        stories_processed += 1

    chars_per_token = (
        total_characters / total_tokens
        if total_tokens else 0
    )

    print(f"\n{label}")
    print("-" * 45)
    print("Stories processed:", stories_processed)
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)
    print("Characters per token:", round(chars_per_token, 3))


def main():
    print("=" * 65)
    print("MINIGPT DATA QUALITY ANALYSIS")
    print("=" * 65)

    train_records = load_records(TRAIN_FILE)
    val_records = load_records(VAL_FILE)

    print("\nDataset sizes")
    print("-" * 45)
    print("Training stories:", len(train_records))
    print("Validation stories:", len(val_records))

    all_records = train_records + val_records

    word_frequencies = analyze_word_frequencies(
        train_records,
        val_records
    )

    print("\n1. SUSPICIOUS JOINED WORDS")
    print("-" * 45)

    joined_candidates = find_suspicious_joined_words(
        all_records,
        word_frequencies
    )

    sorted_candidates = sorted(
        joined_candidates.items(),
        key=lambda item: item[1]["count"],
        reverse=True
    )

    print("Candidate word types:", len(sorted_candidates))
    print(
        "Total candidate occurrences:",
        sum(item["count"] for item in joined_candidates.values())
    )

    if not sorted_candidates:
        print("No candidates found.")

    for word, details in sorted_candidates[:30]:
        print(f"\n{word} | occurrences: {details['count']}")

        for example in details["examples"]:
            print(
                f"  Story {example['story_id']}: "
                f"{example['text']} -> "
                f"{example['suggested_split']}"
            )

    print("\n2. PUNCTUATION WITHOUT FOLLOWING SPACE")
    print("-" * 45)

    punctuation_count, punctuation_examples = (
        find_punctuation_issues(all_records)
    )

    print("Total occurrences:", punctuation_count)

    for example in punctuation_examples:
        print(
            f"  Story {example['story_id']}: "
            f"...{example['context']}..."
        )

    print("\n3. TOKENIZER EFFICIENCY")
    print("-" * 45)

    tokenizer = BPETokenizer.load(TOKENIZER_FILE)

    print("Tokenizer vocabulary:", tokenizer.vocab_size)
    print("Learned merges:", len(tokenizer.merges))

    analyze_tokenizer(
        tokenizer,
        train_records[:TRAIN_TOKENIZER_SAMPLE],
        "TRAINING SAMPLE"
    )

    analyze_tokenizer(
        tokenizer,
        val_records[:VAL_TOKENIZER_SAMPLE],
        "VALIDATION SAMPLE"
    )

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)
    print("No existing data or checkpoints were modified.")


if __name__ == "__main__":
    main()
