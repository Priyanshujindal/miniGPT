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

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=30,
        temperature=0.8,
        top_k=40,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.vocab_size,
    )

    generated_ids = output_ids[0].tolist()

    # Remove the prompt and EOS before decoding.
    new_tokens = generated_ids[len(token_ids):]
    new_tokens = [
        token for token in new_tokens
        if token != tokenizer.vocab_size
    ]

    generated_text = tokenizer.decode(new_tokens)

    print("\nPrompt:")
    print(prompt)

    print("\nGenerated continuation:")
    print(generated_text)

    print("\nGeneration test completed successfully.")


if __name__ == "__main__":
    main()