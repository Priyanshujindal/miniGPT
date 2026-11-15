import json
import sys
import time
from pathlib import Path

# Ensure miniGPT project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from generation.load_model_v4 import load_model_v4
from generation.generate import generate
from training.config_v4 import (
    TEST_DATA_PATH,
    BLOCK_SIZE,
    VOCAB_SIZE,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_OF_TURN_TOKEN,
    END_TOKEN,
    END_OF_TURN_TOKEN_ID,
    END_TOKEN_ID,
    DEVICE,
)
from training.dataset_v4 import create_dataloader_v4

RESULTS_DIR = Path("evaluation") / "results"


def format_prompt(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "system":
            prompt += f"{SYSTEM_TOKEN}{content}{END_OF_TURN_TOKEN}"
        elif role == "user":
            prompt += f"{USER_TOKEN}{content}{END_OF_TURN_TOKEN}"
        elif role == "assistant":
            prompt += f"{ASSISTANT_TOKEN}{content}{END_OF_TURN_TOKEN}"
    prompt += ASSISTANT_TOKEN
    return prompt


@torch.no_grad()
def evaluate_test_loss(model, test_loader):
    model.eval()
    losses = []
    for x, y in test_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE).float(),
                y.reshape(-1),
                ignore_index=-100,
            )
        if not torch.isnan(loss):
            losses.append(loss.item())
    return sum(losses) / len(losses) if losses else float("inf")


def generate_answer(model, tokenizer, prompt_text, max_new_tokens=40):
    encoded = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=DEVICE)

    start_time = time.time()
    with torch.inference_mode():
        output_ids = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_k=20,
            repetition_penalty=1.15,
            eos_token_id=END_OF_TURN_TOKEN_ID,
        )
    elapsed = time.time() - start_time

    new_tokens = output_ids[0].tolist()[input_ids.shape[1] :]
    clean_tokens = []
    for tid in new_tokens:
        if tid in [END_OF_TURN_TOKEN_ID, END_TOKEN_ID]:
            break
        clean_tokens.append(tid)

    tokens_per_sec = len(new_tokens) / elapsed if elapsed > 0 else 0
    response_text = tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()
    return response_text, tokens_per_sec


def benchmark_prompts(model, tokenizer, test_prompts):
    results = []
    speeds = []

    for item in test_prompts:
        prompt_text = format_prompt(item["messages"][:-1])  # Exclude reference answer
        ref_answer = item["messages"][-1]["content"]
        user_query = item["messages"][-2]["content"]
        cat = item.get("category", "general")

        generated, tps = generate_answer(model, tokenizer, prompt_text)
        speeds.append(tps)

        results.append({
            "category": cat,
            "query": user_query,
            "reference": ref_answer,
            "generated": generated,
            "tokens_per_second": round(tps, 1),
        })

    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    return results, avg_speed


def main():
    print("=" * 75)
    print("MINIGPT V4: FAIR COMPARISON HARNESS (TRANSFER vs SCRATCH)")
    print("=" * 75)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Test Dataloader
    test_loader = create_dataloader_v4(TEST_DATA_PATH, batch_size=8, shuffle=False)
    print(f"Loaded held-out test set: {len(test_loader.dataset)} conversations\n")

    # 2. Extract benchmark prompts from test.jsonl
    test_prompts = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_prompts.append(json.loads(line.strip()))

    models_to_eval = [("transfer", "V4-Transfer (Pretrained Warm-Start)"), ("scratch", "V4-Scratch (Trained from Scratch)")]
    comparison = {}

    for model_key, model_label in models_to_eval:
        print("-" * 75)
        print(f"Evaluating {model_label}...")
        print("-" * 75)

        try:
            model, tokenizer, device = load_model_v4(model_type=model_key, checkpoint_name="best.pt")
        except FileNotFoundError as e:
            print(f"Skipping {model_key}: {e}")
            continue

        # Test Loss
        test_loss = evaluate_test_loss(model, test_loader)
        print(f"  Test Loss (Assistant-only): {test_loss:.4f}")

        # GPU memory usage
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        print(f"  Peak GPU VRAM Allocated  : {vram_mb:.1f} MB")

        # Benchmark generation
        print("  Generating responses on held-out test prompts...")
        responses, avg_speed = benchmark_prompts(model, tokenizer, test_prompts[:15])
        print(f"  Average Generation Speed : {avg_speed:.1f} tokens/sec\n")

        comparison[model_key] = {
            "label": model_label,
            "test_loss": round(test_loss, 4),
            "peak_vram_mb": round(vram_mb, 1),
            "tokens_per_sec": round(avg_speed, 1),
            "sample_responses": responses,
        }

    # Save Results
    output_path = RESULTS_DIR / "v4_comparison_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)
    print(f"Comparison report saved to: {output_path}")

    # Display comparison table
    print("\n" + "=" * 75)
    print(f"{'METRIC':<30} | {'V4-TRANSFER':<20} | {'V4-SCRATCH':<20}")
    print("=" * 75)
    t_loss = comparison.get("transfer", {}).get("test_loss", "N/A")
    s_loss = comparison.get("scratch", {}).get("test_loss", "N/A")
    t_spd = comparison.get("transfer", {}).get("tokens_per_sec", "N/A")
    s_spd = comparison.get("scratch", {}).get("tokens_per_sec", "N/A")
    t_ram = comparison.get("transfer", {}).get("peak_vram_mb", "N/A")
    s_ram = comparison.get("scratch", {}).get("peak_vram_mb", "N/A")

    print(f"{'Test Loss (Assistant-only)':<30} | {str(t_loss):<20} | {str(s_loss):<20}")
    print(f"{'Generation Speed (tok/s)':<30} | {str(t_spd):<20} | {str(s_spd):<20}")
    print(f"{'Peak GPU Memory (MB)':<30} | {str(t_ram):<20} | {str(s_ram):<20}")
    print("=" * 75)


if __name__ == "__main__":
    main()
