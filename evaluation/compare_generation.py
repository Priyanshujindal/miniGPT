
import torch

from generation.load_model import load_trained_model
from generation.generate import generate


def main():
    model, tokenizer, device = load_trained_model()

    prompt = "Once upon a time"

    token_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    settings = [
        {
            "name": "Greedy",
            "temperature": 0,
            "top_k": None,
            "repetition_penalty": 1.0,
        },
        {
            "name": "Low randomness",
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
            "name": "High randomness",
            "temperature": 1.2,
            "top_k": 50,
            "repetition_penalty": 1.0,
        },
    ]

    for setting in settings:
        output_ids = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=100,
            temperature=setting["temperature"],
            top_k=setting["top_k"],
            repetition_penalty=setting["repetition_penalty"],
            eos_token_id=tokenizer.vocab_size,
        )

        new_tokens = output_ids[0, len(token_ids):].tolist()

        # Remove EOS from the displayed continuation.
        if tokenizer.vocab_size in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(tokenizer.vocab_size)]

        text = tokenizer.decode(new_tokens)

        print("\n" + "=" * 60)
        print(setting["name"])
        print("=" * 60)
        print(text)


if __name__ == "__main__":
    main()
