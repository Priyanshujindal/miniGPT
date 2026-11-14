
import torch

from generation.generate import generate
from generation.load_model import load_trained_model


def main():

    model, tokenizer, device = load_trained_model()

    # EOS is the special token immediately after the BPE vocabulary.
    eos_token_id = tokenizer.vocab_size

    print("\n" + "=" * 55)
    print("             MiniGPT Text Generator")
    print("=" * 55)

    print("Enter a prompt and MiniGPT will continue it.")
    print("Type 'exit' to close the application.")
    print("=" * 55)

    while True:

        prompt = input("\nYou: ").strip()

        if prompt.lower() in ["exit", "quit"]:
            print("MiniGPT closed.")
            break

        if not prompt:
            print("Please enter a prompt.")
            continue

        # Encode the prompt.
        token_ids = tokenizer.encode(prompt)

        if not token_ids:
            print("The tokenizer returned no tokens for this prompt.")
            continue

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=device,
        )

        # Generate continuation.
        output_ids = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_k=30,
            repetition_penalty=1.15,
            eos_token_id=eos_token_id,
        )

        # Extract only newly generated tokens.
        new_tokens = output_ids[0, len(token_ids):].tolist()

        # Remove EOS before passing tokens to the BPE tokenizer.
        if new_tokens and new_tokens[-1] == eos_token_id:
            new_tokens = new_tokens[:-1]

        # Decode only normal BPE tokens.
        generated_text = tokenizer.decode(new_tokens)

        print("\nMiniGPT:")
        print(generated_text if generated_text else "(No continuation generated.)")


if __name__ == "__main__":
    main()
