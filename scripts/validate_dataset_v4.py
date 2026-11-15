import json
import sys
from pathlib import Path
from tokenizers import Tokenizer

from training.config_v4 import (
    TOKENIZER_V4_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    DATASET_REPORT_PATH,
    BLOCK_SIZE,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_OF_TURN_TOKEN,
    END_TOKEN,
)


def format_messages(messages):
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "system":
            formatted += f"{SYSTEM_TOKEN}{content}{END_OF_TURN_TOKEN}"
        elif role == "user":
            formatted += f"{USER_TOKEN}{content}{END_OF_TURN_TOKEN}"
        elif role == "assistant":
            formatted += f"{ASSISTANT_TOKEN}{content}{END_OF_TURN_TOKEN}"
    formatted += END_TOKEN
    return formatted


def validate_file(path, tokenizer, name):
    print(f"\nValidating {name}: {path}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []
    prompts_seen = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num} in {name} is invalid JSON: {e}")

            if "messages" not in data or not isinstance(data["messages"], list):
                raise ValueError(f"Line {line_num} in {name} missing 'messages' list")

            messages = data["messages"]
            if len(messages) < 2:
                raise ValueError(f"Line {line_num} in {name} has fewer than 2 messages")

            has_assistant = False
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if role not in ["system", "user", "assistant"]:
                    raise ValueError(f"Line {line_num} in {name} has invalid role: {role}")
                if not content or not isinstance(content, str) or not content.strip():
                    raise ValueError(f"Line {line_num} in {name} has empty content for role: {role}")
                if role == "assistant":
                    has_assistant = True

            if not has_assistant:
                raise ValueError(f"Line {line_num} in {name} has no assistant message!")

            # Verify token length <= BLOCK_SIZE
            formatted = format_messages(messages)
            encoded = tokenizer.encode(formatted)
            token_count = len(encoded.ids)

            if token_count > BLOCK_SIZE:
                raise ValueError(
                    f"Line {line_num} in {name} exceeds context window: {token_count} > {BLOCK_SIZE} tokens"
                )

            # Extract user prompt for leakage detection
            user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
            prompts_seen.add(user_text)
            records.append((data, token_count))

    print(f"  Total verified records: {len(records)}")
    print(f"  Min tokens: {min(r[1] for r in records)}")
    print(f"  Max tokens: {max(r[1] for r in records)}")
    print(f"  Avg tokens: {sum(r[1] for r in records) / len(records):.1f}")
    return records, prompts_seen


def main():
    print("=" * 70)
    print("MINIGPT V4 DATASET COMPREHENSIVE VALIDATION SUITE")
    print("=" * 70)

    print(f"Loading V4 tokenizer from: {TOKENIZER_V4_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_V4_PATH))

    # 1. Validate train, validation, and test files
    train_records, train_prompts = validate_file(TRAIN_DATA_PATH, tokenizer, "TRAIN")
    val_records, val_prompts = validate_file(VAL_DATA_PATH, tokenizer, "VALIDATION")
    test_records, test_prompts = validate_file(TEST_DATA_PATH, tokenizer, "TEST")

    # 2. Strict Leakage Verification
    print("\nChecking for data leakage between splits...")
    train_val_overlap = train_prompts.intersection(val_prompts)
    train_test_overlap = train_prompts.intersection(test_prompts)
    val_test_overlap = val_prompts.intersection(test_prompts)

    if train_val_overlap:
        raise AssertionError(f"Leakage between Train and Val: {len(train_val_overlap)} overlaps")
    if train_test_overlap:
        raise AssertionError(f"Leakage between Train and Test: {len(train_test_overlap)} overlaps")
    if val_test_overlap:
        raise AssertionError(f"Leakage between Val and Test: {len(val_test_overlap)} overlaps")
    print("Zero prompt leakage confirmed across Train, Validation, and Test sets!")

    # 3. Validate dataset_report.json
    print(f"\nChecking dataset report: {DATASET_REPORT_PATH}")
    if not DATASET_REPORT_PATH.exists():
        raise FileNotFoundError(f"Report not found: {DATASET_REPORT_PATH}")

    with open(DATASET_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    print("Report metadata verified:")
    print(f"  Dataset Name       : {report.get('dataset_name')}")
    print(f"  Version            : {report.get('version')}")
    print(f"  Total Samples      : {report.get('total_samples')}")
    print(f"  Context Window Max : {report.get('context_window_limit')}")
    print(f"  Split Counts       : {report.get('split_counts')}")

    print("\n" + "=" * 70)
    print("ALL STAGE 2 DATASET & TOKENIZER VALIDATION CHECKS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
