
import torch

from generation.generate import generate
from generation.load_model_v2 import load_trained_model_v2

from training.config_v2 import EOS_TOKEN_ID


def main():

    model, tokenizer, device = load_trained_model_v2()

    print("\n" + "=" * 60)
    print("              MiniGPT V2 Text Generator")
    print("=" * 60)

    print("Tokenizer: Hugging Face ByteLevel BPE V2")
    print(f"Vocabulary: {tokenizer.get_vocab_size()}")
    print(f"EOS token ID: {EOS_TOKEN_ID}")
    print(f"Device: {device}")

    print("\nEnter a prompt to generate text.")
    print("Type 'exit' to close the application.")
    print("=" * 60)

    while True:

        prompt = input("\nYou: ").strip()

        if prompt.lower() in ["exit", "quit"]:
            print("MiniGPT V2 closed.")
            break

        if not prompt:
            print("Please enter a prompt.")
            continue

        # Encode prompt using Hugging Face tokenizer.
        encoded = tokenizer.encode(prompt)

        token_ids = encoded.ids

        if not token_ids:
            print("The tokenizer returned no tokens.")
            continue

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=device,
        )

        try:

            output_ids = generate(
                model=model,
                input_ids=input_ids,
                max_new_tokens=150,
                temperature=0.7,
                top_k=30,
                repetition_penalty=1.15,
                eos_token_id=EOS_TOKEN_ID,
            )

            # Extract only newly generated tokens.
            new_tokens = output_ids[
                0,
                len(token_ids):
            ].tolist()

            # Remove EOS before decoding.
            if new_tokens and new_tokens[-1] == EOS_TOKEN_ID:
                new_tokens = new_tokens[:-1]

            # Decode with the V2 Hugging Face tokenizer.
            generated_text = tokenizer.decode(
                new_tokens,
                skip_special_tokens=False,
            )

            print("\nMiniGPT V2:")
            print(
                generated_text
                if generated_text
                else "(No continuation generated.)"
            )

        except Exception as error:

            print(f"\nGeneration error: {error}")


if __name__ == "__main__":
    main()
