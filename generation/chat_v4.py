import argparse
import sys
from pathlib import Path

# Ensure miniGPT root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from generation.load_model_v4 import load_model_v4
from generation.generate import generate
from training.config_v4 import (
    BLOCK_SIZE,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_OF_TURN_TOKEN,
    END_TOKEN,
    END_OF_TURN_TOKEN_ID,
    END_TOKEN_ID,
)


def format_chat_prompt(messages):
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
    # Prime assistant for response
    prompt += ASSISTANT_TOKEN
    return prompt


def chat_loop(model_type="transfer", checkpoint_name="best.pt"):
    print("=" * 70)
    print(f"MINIGPT V4 ({model_type.upper()}) CONVERSATIONAL CHAT")
    print("=" * 70)

    model, tokenizer, device = load_model_v4(
        model_type=model_type,
        checkpoint_name=checkpoint_name,
    )

    system_prompt = "You are MiniGPT, a polite and helpful assistant."
    messages = [{"role": "system", "content": system_prompt}]

    print("\nCommands:")
    print("  'exit' or 'quit' : End conversation")
    print("  'clear' or 'reset': Start fresh conversation")
    print("\nAsk a question or enter an instruction.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if user_input.lower() in ["clear", "reset"]:
            messages = [{"role": "system", "content": system_prompt}]
            print("[System] Conversation cleared. New session started.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        # Format full conversation
        prompt_text = format_chat_prompt(messages)
        encoded = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

        # Context pruning if context exceeds budget
        max_prompt_len = BLOCK_SIZE - 40  # reserve at least 40 tokens for completion
        if input_ids.shape[1] > max_prompt_len:
            # Prune older user/assistant turns while preserving system prompt
            while len(messages) > 2 and input_ids.shape[1] > max_prompt_len:
                # Remove oldest user & assistant pair
                messages = [messages[0]] + messages[3:]
                prompt_text = format_chat_prompt(messages)
                encoded = tokenizer.encode(prompt_text)
                input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

        try:
            with torch.inference_mode():
                output_ids = generate(
                    model=model,
                    input_ids=input_ids,
                    max_new_tokens=60,
                    temperature=0.7,
                    top_k=40,
                    repetition_penalty=1.15,
                    eos_token_id=END_OF_TURN_TOKEN_ID,
                )

            new_tokens = output_ids[0].tolist()[input_ids.shape[1] :]

            # Truncate at end of turn or end token
            clean_tokens = []
            for tid in new_tokens:
                if tid in [END_OF_TURN_TOKEN_ID, END_TOKEN_ID]:
                    break
                clean_tokens.append(tid)

            response_text = tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()

            if not response_text:
                response_text = "I am not sure how to answer that yet."

            print(f"\nMiniGPT ({model_type}): {response_text}\n")
            messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            print(f"\n[Generation Error]: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with MiniGPT V4")
    parser.add_argument(
        "--model-type",
        choices=["transfer", "scratch"],
        default="transfer",
        help="Which model variant to chat with (transfer or scratch)",
    )
    parser.add_argument(
        "--checkpoint",
        default="best.pt",
        help="Checkpoint name (default: best.pt)",
    )
    args = parser.parse_args()
    chat_loop(model_type=args.model_type, checkpoint_name=args.checkpoint)
