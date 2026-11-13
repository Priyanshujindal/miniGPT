
import time
import torch

from model.attention import CausalSelfAttention


def benchmark():

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    device = torch.device("cuda")

    embedding_dim = 256
    num_heads = 4
    batch_size = 8
    sequence_length = 128
    iterations = 100

    model = CausalSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout=0.1,
    ).to(device)

    model.train()

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        device=device,
        requires_grad=True,
    )

    # Warmup.
    for _ in range(10):
        output = model(x)
        loss = output.mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        x.grad = None

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    for _ in range(iterations):

        output = model(x)

        loss = output.mean()
        loss.backward()

        model.zero_grad(set_to_none=True)
        x.grad = None

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    average_time = elapsed / iterations * 1000

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    )

    print("\nSDPA Attention Benchmark")
    print("-" * 40)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Iterations: {iterations}")
    print(f"Average forward + backward: {average_time:.2f} ms")
    print(f"Peak allocated GPU memory: {peak_memory:.2f} MB")


if __name__ == "__main__":
    benchmark()
