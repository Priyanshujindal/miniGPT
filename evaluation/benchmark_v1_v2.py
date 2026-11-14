
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tokenizer.bpe import BPETokenizer
from tokenizers import Tokenizer

from generation.load_model import load_trained_model
from generation.load_model_v2 import load_trained_model_v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_VALIDATION_FILE = PROJECT_ROOT / "data" / "raw" / "validation.jsonl"

V1_CHECKPOINT = PROJECT_ROOT / "checkpoints_v2" / "best.pt"
V2_CHECKPOINT = PROJECT_ROOT / "checkpoints_tokenizer_v2" / "best.pt"

V1_TOKENIZER_FILE = PROJECT_ROOT / "data" / "tokenizer" / "bpe_tokenizer.json"
V2_TOKENIZER_FILE = PROJECT_ROOT / "data" / "tokenizer" / "bpe_tokenizer_v2.json"

OUTPUT_DIR = PROJECT_ROOT / "logs"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_K = 30
REPETITION_PENALTY = 1.15
RANDOM_SEED = 42

PROMPTS = [
    "Once upon a time",
    "The little girl went to the garden.",
    "A small boy found a mysterious box.",
    "A dog and a cat became friends.",
    "One day, a young boy decided to help his mother."
]


def load_validation_stories():
    if not RAW_VALIDATION_FILE.exists():
        raise FileNotFoundError(RAW_VALIDATION_FILE)

    stories = []

    with open(RAW_VALIDATION_FILE, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            text = record["text"]

            if not isinstance(text, str) or not text:
                continue

            stories.append(text)

    if not stories:
        raise RuntimeError("No validation stories were loaded.")

    return stories


def encode_stories(stories, tokenizer, eos_token_id, tokenizer_type):
    all_tokens = []
    total_bytes = 0

    for text in stories:
        if tokenizer_type == "v1":
            token_ids = tokenizer.encode(text)
        else:
            token_ids = tokenizer.encode(text).ids

        if not token_ids:
            continue

        if any(
            token_id < 0 or token_id >= eos_token_id
            for token_id in token_ids
        ):
            raise ValueError(
                f"Invalid token ID encountered in {tokenizer_type}."
            )

        all_tokens.extend(token_ids)
        all_tokens.append(eos_token_id)

        total_bytes += len(text.encode("utf-8"))

    if len(all_tokens) < 2:
        raise RuntimeError("Not enough validation tokens.")

    return np.asarray(all_tokens, dtype=np.int64), total_bytes


def make_batches(tokens, block_size, batch_size):
    windows = []

    for start in range(0, len(tokens) - 1, block_size):
        x = tokens[start:start + block_size]
        y = tokens[start + 1:start + block_size + 1]

        if len(x) == 0 or len(x) != len(y):
            continue

        windows.append((x, y))

    for start in range(0, len(windows), batch_size):
        group = windows[start:start + batch_size]

        x_batch = np.stack([item[0] for item in group])
        y_batch = np.stack([item[1] for item in group])

        yield (
            torch.from_numpy(x_batch).long(),
            torch.from_numpy(y_batch).long()
        )


@torch.inference_mode()
def evaluate_model(model, tokens, total_bytes, device):
    model.eval()

    block_size = model.max_sequence_length

    total_loss = 0.0
    total_tokens = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for x, y in make_batches(tokens, block_size, BATCH_SIZE):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        loss_sum = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            y.reshape(-1),
            reduction="sum"
        )

        total_loss += loss_sum.item()
        total_tokens += y.numel()

        del x, y, logits, loss_sum

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    if total_tokens == 0 or total_bytes == 0:
        raise RuntimeError("Evaluation processed no tokens or bytes.")

    average_loss = total_loss / total_tokens

    try:
        perplexity = math.exp(average_loss)
    except OverflowError:
        perplexity = float("inf")

    bits_per_byte = total_loss / (math.log(2) * total_bytes)

    peak_memory_mb = 0.0

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

    return {
        "loss": average_loss,
        "perplexity": perplexity,
        "bits_per_byte": bits_per_byte,
        "tokens_evaluated": total_tokens,
        "bytes_evaluated": total_bytes,
        "evaluation_seconds": elapsed,
        "evaluation_tokens_per_second": total_tokens / elapsed,
        "peak_gpu_memory_mb": peak_memory_mb
    }


def encode_prompt(tokenizer, prompt, tokenizer_type):
    if tokenizer_type == "v1":
        return tokenizer.encode(prompt)

    return tokenizer.encode(prompt).ids


def decode_tokens(tokenizer, token_ids, tokenizer_type):
    if tokenizer_type == "v1":
        return tokenizer.decode(token_ids)

    return tokenizer.decode(token_ids)


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt,
    tokenizer_type,
    eos_token_id,
    device
):
    model.eval()

    input_ids = encode_prompt(tokenizer, prompt, tokenizer_type)

    if not input_ids:
        raise ValueError("Prompt produced no tokens.")

    generated = list(input_ids)

    for _ in range(MAX_NEW_TOKENS):
        context = generated[-model.max_sequence_length:]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device
        )

        logits = model(x)
        next_logits = logits[0, -1, :].clone()

        # Apply repetition penalty.
        if REPETITION_PENALTY != 1.0:
            previous_tokens = set(generated)

            for token_id in previous_tokens:
                if next_logits[token_id] < 0:
                    next_logits[token_id] *= REPETITION_PENALTY
                else:
                    next_logits[token_id] /= REPETITION_PENALTY

        if TEMPERATURE <= 0:
            next_token = torch.argmax(next_logits).item()
        else:
            next_logits = next_logits / TEMPERATURE

            if TOP_K > 0:
                k = min(TOP_K, next_logits.numel())
                values, _ = torch.topk(next_logits, k)

                threshold = values[-1]
                next_logits[next_logits < threshold] = float("-inf")

            probabilities = F.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            ).item()

        if next_token == eos_token_id:
            break

        generated.append(next_token)

    new_tokens = generated[len(input_ids):]

    return decode_tokens(tokenizer, new_tokens, tokenizer_type), len(new_tokens)


def benchmark_generation(model, tokenizer, tokenizer_type, eos_token_id, device):
    results = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    total_generated_tokens = 0

    for prompt_index, prompt in enumerate(PROMPTS):
        torch.manual_seed(RANDOM_SEED + prompt_index)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(RANDOM_SEED + prompt_index)

        generated_text, token_count = generate_text(
            model,
            tokenizer,
            prompt,
            tokenizer_type,
            eos_token_id,
            device
        )

        total_generated_tokens += token_count

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "generated_tokens": token_count
        })

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    speed = (
        total_generated_tokens / elapsed
        if elapsed > 0 else 0
    )

    return results, elapsed, speed


def load_v1():
    model, tokenizer, device = load_trained_model(
        checkpoint_path=str(V1_CHECKPOINT),
        tokenizer_path=str(V1_TOKENIZER_FILE)
    )

    checkpoint = torch.load(
        V1_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    if tokenizer.vocab_size != 512:
        raise ValueError("Unexpected V1 tokenizer vocabulary.")

    if checkpoint["vocab_size"] != 513:
        raise ValueError("Unexpected V1 model vocabulary.")

    return model, tokenizer, device, checkpoint, tokenizer.vocab_size


def load_v2():
    model, tokenizer, device = load_trained_model_v2(
        checkpoint_path=V2_CHECKPOINT,
        tokenizer_path=V2_TOKENIZER_FILE
    )

    checkpoint = torch.load(
        V2_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    if tokenizer.get_vocab_size() != 2048:
        raise ValueError("Unexpected V2 tokenizer vocabulary.")

    if checkpoint["vocab_size"] != 2049:
        raise ValueError("Unexpected V2 model vocabulary.")

    return model, tokenizer, device, checkpoint, 2048


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [
        RAW_VALIDATION_FILE,
        V1_CHECKPOINT,
        V2_CHECKPOINT,
        V1_TOKENIZER_FILE,
        V2_TOKENIZER_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("=" * 70)
    print("MINIGPT V1 vs V2 BENCHMARK")
    print("=" * 70)

    print("\nLoading raw validation stories...")
    stories = load_validation_stories()

    total_bytes = sum(
        len(text.encode("utf-8"))
        for text in stories
    )

    print("Validation stories:", len(stories))
    print("Raw validation bytes:", total_bytes)

    models = [
        ("V1", load_v1, "v1"),
        ("V2", load_v2, "v2")
    ]

    benchmark_results = []
    generation_results = {}

    report_lines = []

    for name, loader, tokenizer_type in models:
        print("\n" + "=" * 70)
        print(f"BENCHMARKING {name}")
        print("=" * 70)

        model, tokenizer, device, checkpoint, eos_token_id = loader()

        if tokenizer_type == "v1":
            tokenized, byte_count = encode_stories(
                stories, tokenizer, eos_token_id, "v1"
            )
        else:
            tokenized, byte_count = encode_stories(
                stories, tokenizer, eos_token_id, "v2"
            )

        print("\nValidation tokens:", len(tokenized))
        print("Validation bytes:", byte_count)

        evaluation = evaluate_model(
            model,
            tokenized,
            byte_count,
            device
        )

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        checkpoint_path = (
            V1_CHECKPOINT if name == "V1" else V2_CHECKPOINT
        )

        tokenizer_path = (
            V1_TOKENIZER_FILE if name == "V1" else V2_TOKENIZER_FILE
        )

        result = {
            "model": name,
            "checkpoint_step": checkpoint["step"],
            "parameters": parameter_count,
            "tokenizer_vocab_size": tokenizer.vocab_size
                if name == "V1" else tokenizer.get_vocab_size(),
            "validation_loss": evaluation["loss"],
            "perplexity": evaluation["perplexity"],
            "bits_per_byte": evaluation["bits_per_byte"],
            "validation_tokens": evaluation["tokens_evaluated"],
            "validation_bytes": evaluation["bytes_evaluated"],
            "evaluation_seconds": evaluation["evaluation_seconds"],
            "evaluation_tokens_per_second":
                evaluation["evaluation_tokens_per_second"],
            "peak_gpu_memory_mb": evaluation["peak_gpu_memory_mb"],
            "checkpoint_size_mb":
                checkpoint_path.stat().st_size / (1024 ** 2),
            "tokenizer_size_mb":
                tokenizer_path.stat().st_size / (1024 ** 2)
        }

        print("\nValidation results:")
        for key, value in result.items():
            print(f"{key}: {value}")

        print("\nBenchmarking generation...")

        generation, generation_seconds, generation_speed = benchmark_generation(
            model,
            tokenizer,
            tokenizer_type,
            eos_token_id,
            device
        )

        result["generation_seconds"] = generation_seconds
        result["generation_tokens_per_second"] = generation_speed

        generation_results[name] = generation

        benchmark_results.append(result)

        del model, tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save CSV.
    csv_path = OUTPUT_DIR / "v1_v2_benchmark.csv"

    fieldnames = list(benchmark_results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(benchmark_results)

    # Save generation comparison.
    generation_path = OUTPUT_DIR / "v1_v2_generation.txt"

    with open(generation_path, "w", encoding="utf-8") as file:
        file.write("MINIGPT V1 vs V2 GENERATION COMPARISON\n")
        file.write("=" * 80 + "\n")

        for index, prompt in enumerate(PROMPTS):
            file.write(f"\nPROMPT {index + 1}: {prompt}\n")
            file.write("-" * 80 + "\n")

            for name in ["V1", "V2"]:
                result = generation_results[name][index]

                file.write(f"\n{name}:\n")
                file.write(result["generated_text"] + "\n")
                file.write(
                    f"Generated tokens: {result['generated_tokens']}\n"
                )

    # Save readable report.
    report_path = OUTPUT_DIR / "v1_v2_benchmark_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("MINIGPT V1 vs V2 FINAL BENCHMARK REPORT\n")
        file.write("=" * 80 + "\n\n")

        file.write(f"Validation stories: {len(stories)}\n")
        file.write(f"Raw validation bytes: {total_bytes}\n\n")

        for result in benchmark_results:
            file.write(f"\n{result['model']} RESULTS\n")
            file.write("-" * 60 + "\n")

            for key, value in result.items():
                file.write(f"{key}: {value}\n")

        v1 = benchmark_results[0]
        v2 = benchmark_results[1]

        file.write("\n\nCROSS-MODEL COMPARISON\n")
        file.write("=" * 80 + "\n")

        file.write(
            f"V1 BPB: {v1['bits_per_byte']:.6f}\n"
        )
        file.write(
            f"V2 BPB: {v2['bits_per_byte']:.6f}\n"
        )

        if v2["bits_per_byte"] < v1["bits_per_byte"]:
            file.write("V2 has better validation BPB.\n")
        elif v1["bits_per_byte"] < v2["bits_per_byte"]:
            file.write("V1 has better validation BPB.\n")
        else:
            file.write("Both models have equal validation BPB.\n")

        file.write(
            "\nBPB measures negative log likelihood in bits per raw byte. "
            "Lower is better.\n"
        )

        file.write(
            "\nGeneration results must also be reviewed qualitatively. "
            "A numerical score does not establish story quality.\n"
        )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED")
    print("=" * 70)

    print("CSV:", csv_path)
    print("Generation comparison:", generation_path)
    print("Report:", report_path)

    print("\nFinal BPB comparison:")

    for result in benchmark_results:
        print(
            f"{result['model']}: "
            f"{result['bits_per_byte']:.6f} bits/byte"
        )

    print("\nGeneration speed:")

    for result in benchmark_results:
        print(
            f"{result['model']}: "
            f"{result['generation_tokens_per_second']:.2f} tokens/sec"
        )


if __name__ == "__main__":
    main()
