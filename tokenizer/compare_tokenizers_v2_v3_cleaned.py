
import json
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "train": ROOT / "data" / "cleaned" / "train.jsonl",
    "validation": ROOT / "data" / "cleaned" / "validation.jsonl",
}

TOKENIZER_PATHS = {
    "V2": ROOT / "data" / "tokenizer" / "bpe_tokenizer_v2.json",
    "V3": ROOT / "data" / "tokenizer" / "bpe_tokenizer_v3.json",
}


def load_stories(path):
    stories = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            text = record.get("text")

            if isinstance(text, str) and text.strip():
                stories.append(text)

    return stories


def evaluate_tokenizer(tokenizer, stories):
    total_characters = 0
    total_bytes = 0
    total_tokens = 0

    for story in stories:
        total_characters += len(story)
        total_bytes += len(story.encode("utf-8"))
        total_tokens += len(tokenizer.encode(story).ids)

    return {
        "stories": len(stories),
        "characters": total_characters,
        "utf8_bytes": total_bytes,
        "tokens": total_tokens,
        "characters_per_token": total_characters / total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
    }


def main():
    print("=" * 75)
    print("MINIGPT V2 VS V3 TOKENIZER COMPARISON")
    print("Same cleaned dataset for both tokenizers")
    print("=" * 75)

    tokenizers = {
        name: Tokenizer.from_file(str(path))
        for name, path in TOKENIZER_PATHS.items()
    }

    for name, tokenizer in tokenizers.items():
        print(f"{name} vocabulary: {tokenizer.get_vocab_size()}")

    for dataset_name, dataset_path in DATASETS.items():
        stories = load_stories(dataset_path)

        print(f"\n{dataset_name.upper()} DATASET")
        print("-" * 75)
        print("Stories:", len(stories))

        results = {}

        for name, tokenizer in tokenizers.items():
            results[name] = evaluate_tokenizer(tokenizer, stories)

        for name in ("V2", "V3"):
            result = results[name]

            print(f"\n{name}:")
            print("Tokens:", f"{result['tokens']:,}")
            print(
                "Characters per token:",
                f"{result['characters_per_token']:.6f}",
            )
            print(
                "Bytes per token:",
                f"{result['bytes_per_token']:.6f}",
            )

        v2_tokens = results["V2"]["tokens"]
        v3_tokens = results["V3"]["tokens"]

        reduction = (v2_tokens - v3_tokens) / v2_tokens * 100

        print("\nV3 token count reduction vs V2:")
        print(f"{reduction:.2f}%")

    print("\n" + "=" * 75)
    print("COMPARISON COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
