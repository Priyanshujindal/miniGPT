
import torch

from generation.load_model import load_trained_model


def main():
    model, tokenizer, device = load_trained_model()

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    parameter_memory = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    buffer_memory = sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )

    print("\n" + "=" * 45)
    print("             MODEL STATISTICS")
    print("=" * 45)

    print(f"Vocabulary size:       {tokenizer.vocab_size + 1:,}")
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")

    print(
        f"Parameter memory:      "
        f"{parameter_memory / (1024 ** 2):.2f} MB"
    )

    print(
        f"Buffer memory:         "
        f"{buffer_memory / (1024 ** 2):.2f} MB"
    )

    print(f"Device:                {device}")

    if device.type == "cuda":
        torch.cuda.synchronize()

        print(
            f"Allocated GPU memory:  "
            f"{torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB"
        )

        print(
            f"Reserved GPU memory:   "
            f"{torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB"
        )

        print(
            f"GPU name:              "
            f"{torch.cuda.get_device_name(device)}"
        )

    print("=" * 45)


if __name__ == "__main__":
    main()
