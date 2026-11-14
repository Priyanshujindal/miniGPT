
import torch

from training.config import CHECKPOINT_DIR, PROJECT_ROOT, LOG_DIR
from generation.load_model import load_trained_model
from generation.generate import generate


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROMPTS_PATH = PROJECT_ROOT / "evaluation" / "prompts.txt"

MAX_NEW_TOKENS = 80
RANDOM_SEED = 42

CHECKPOINTS = [
    CHECKPOINT_DIR / "best.pt",
    CHECKPOINT_DIR / "latest.pt",
]

DECODING_SETTINGS = [
    {
        "name": "Greedy",
        "temperature": 0,
        "top_k": None,
        "repetition_penalty": 1.0,
    },
    {
        "name": "Low Randomness",
        "temperature": 0.4,
        "top_k": 20,
        "repetition_penalty": 1.1,
    },
    {
        "name": "Balanced",
        "temperature": 0.8,
        "top_k": 40,
        "repetition_penalty": 1.1,
    },
    {
        "name": "High Randomness",
        "temperature": 1.2,
        "top_k": 50,
        "repetition_penalty": 1.0,
    },
]


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

def set_seed(seed, device):

    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# LOAD PROMPTS
# --------------------------------------------------

def load_prompts():

    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {PROMPTS_PATH}"
        )

    with open(PROMPTS_PATH, "r", encoding="utf-8") as file:

        prompts = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not prompts:
        raise ValueError("No prompts found in prompts.txt.")

    return prompts


# --------------------------------------------------
# GENERATION
# --------------------------------------------------

def generate_continuation(
    model,
    tokenizer,
    device,
    prompt,
    settings,
):

    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        return "[Prompt produced no tokens.]"

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    # Reset the seed before every corresponding test.
    set_seed(RANDOM_SEED, device)

    output = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=settings["temperature"],
        top_k=settings["top_k"],
        repetition_penalty=settings["repetition_penalty"],
        eos_token_id=tokenizer.vocab_size,
    )

    generated_tokens = output[0, len(token_ids):].tolist()

    # Remove EOS before passing tokens to the tokenizer.
    eos_token_id = tokenizer.vocab_size

    if eos_token_id in generated_tokens:

        eos_position = generated_tokens.index(eos_token_id)

        generated_tokens = generated_tokens[:eos_position]

    return tokenizer.decode(generated_tokens)


# --------------------------------------------------
# MAIN COMPARISON
# --------------------------------------------------

def main():

    prompts = load_prompts()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    report_path = LOG_DIR / "checkpoint_generation_comparison.txt"

    report_lines = []

    print("=" * 75)
    print("MINIGPT CHECKPOINT GENERATION COMPARISON")
    print("=" * 75)

    print(f"\nPrompts: {len(prompts)}")
    print(f"Decoding strategies: {len(DECODING_SETTINGS)}")
    print(f"Maximum new tokens: {MAX_NEW_TOKENS}")
    print(f"Random seed: {RANDOM_SEED}")

    for checkpoint_path in CHECKPOINTS:

        if not checkpoint_path.exists():

            print(
                f"\nWARNING: Checkpoint not found: {checkpoint_path}"
            )

            continue

        checkpoint_name = checkpoint_path.name

        print("\n\n" + "#" * 75)
        print(f"CHECKPOINT: {checkpoint_name}")
        print("#" * 75)

        report_lines.append("\n" + "#" * 75)
        report_lines.append(f"CHECKPOINT: {checkpoint_name}")
        report_lines.append("#" * 75)

        model, tokenizer, device = load_trained_model(
            checkpoint_path=str(checkpoint_path)
        )

        print(f"Device: {device}")

        for prompt in prompts:

            print("\n" + "-" * 75)
            print(f"PROMPT: {prompt}")
            print("-" * 75)

            report_lines.append("\n" + "-" * 75)
            report_lines.append(f"PROMPT: {prompt}")
            report_lines.append("-" * 75)

            for settings in DECODING_SETTINGS:

                continuation = generate_continuation(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    settings=settings,
                )

                result = (
                    f"\n[{settings['name']}]\n"
                    f"{continuation}\n"
                )

                print(result)

                report_lines.append(result)

        # Release this checkpoint before loading the next one.
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save the complete comparison.
    with open(report_path, "w", encoding="utf-8") as file:

        file.write(
            "MINIGPT CHECKPOINT GENERATION COMPARISON\n"
        )

        file.write(f"Random seed: {RANDOM_SEED}\n")
        file.write(f"Maximum new tokens: {MAX_NEW_TOKENS}\n")

        file.write("\n".join(report_lines))

    print("\n" + "=" * 75)
    print("COMPARISON COMPLETED")
    print("=" * 75)

    print(f"\nResults saved to:\n{report_path}")


if __name__ == "__main__":
    main()
