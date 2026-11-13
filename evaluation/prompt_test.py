
import torch

from generation.load_model import load_trained_model
from generation.generate import generate


def main():
    model, tokenizer, device = load_trained_model()

    prompts_path = "evaluation/prompts.txt"

    with open(prompts_path, "r", encoding="utf-8") as file:
        prompts = [
            line.strip()
            for line in file
            if line.strip()
        ]

    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)

        if not token_ids:
            print(f"Skipping empty tokenization: {prompt}")
            continue

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=device,
        )

        output = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=80,
            temperature=0.8,
            top_k=40,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.vocab_size,
        )

        generated_tokens = output[0, len(token_ids):].tolist()

        if tokenizer.vocab_size in generated_tokens:
            generated_tokens = generated_tokens[
                :generated_tokens.index(tokenizer.vocab_size)
            ]

        continuation = tokenizer.decode(generated_tokens)

        print("\n" + "-" * 60)
        print("PROMPT:", prompt)
        print("CONTINUATION:", continuation)


if __name__ == "__main__":
    main()
