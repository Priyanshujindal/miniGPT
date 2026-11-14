
import torch

from generation.load_model_v3 import load_model_v3
from generation.generate import generate
from training.config_v3 import (
    EOS_TOKEN_ID,
    BLOCK_SIZE,
)


def main():

    print("=" * 65)
    print("MINIGPT V3 CHAT")
    print("=" * 65)

    model, tokenizer, device = load_model_v3("best.pt")

    print("\nType 'exit' to quit.")
    print("Enter a prompt to generate text.\n")

    while True:

        prompt = input("You: ").strip()

        if prompt.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not prompt:
            continue

        encoded = tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [encoded.ids],
            dtype=torch.long,
            device=device,
        )

        if input_ids.shape[1] == 0:
            continue

        if input_ids.shape[1] > BLOCK_SIZE:
            input_ids = input_ids[:, -BLOCK_SIZE:]

        try:

            output_ids = generate(
                model=model,
                input_ids=input_ids,
                max_new_tokens=100,
                temperature=0.8,
                top_k=40,
                repetition_penalty=1.15,
                eos_token_id=EOS_TOKEN_ID,
            )

            generated_ids = output_ids[0].tolist()

            # Remove the original prompt tokens.
            new_tokens = generated_ids[input_ids.shape[1]:]

            # Remove EOS before decoding.
            if EOS_TOKEN_ID in new_tokens:
                new_tokens = new_tokens[
                    :new_tokens.index(EOS_TOKEN_ID)
                ]

            generated_text = tokenizer.decode(new_tokens)

            print("\nMiniGPT:", generated_text)
            print()

        except Exception as error:
            print("\nGeneration error:", error)


if __name__ == "__main__":
    main()
