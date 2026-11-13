
import torch

from training.dataset import create_dataloader


TRAIN_FILE = "data/tokenized/train.npy"

VAL_FILE = "data/tokenized/validation.npy"

BLOCK_SIZE = 128
BATCH_SIZE = 16


def test_loader(file_path, name):

    print(f"\nTesting {name} dataset")

    loader = create_dataloader(
        file_path=file_path,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        shuffle=False
    )

    print("Number of batches:", len(loader))

    x, y = next(iter(loader))

    print("Input shape:", x.shape)
    print("Target shape:", y.shape)

    print("Input dtype:", x.dtype)
    print("Target dtype:", y.dtype)

    # Verify dimensions.
    assert x.shape == (
        BATCH_SIZE,
        BLOCK_SIZE
    )

    assert y.shape == (
        BATCH_SIZE,
        BLOCK_SIZE
    )

    # Verify next-token shifting.
    assert torch.equal(
        x[:, 1:],
        y[:, :-1]
    )

    # Check valid token IDs.
    assert torch.all(x >= 0)
    assert torch.all(y >= 0)

    print("\nFirst input sequence:")
    print(x[0].tolist())

    print("\nFirst target sequence:")
    print(y[0].tolist())

    print(f"\n{name} dataset passed!")


def main():

    test_loader(TRAIN_FILE, "TRAINING")

    test_loader(VAL_FILE, "VALIDATION")

    print("\nALL DATA PIPELINE TESTS PASSED!")


if __name__ == "__main__":
    main()
