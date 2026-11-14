
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from training.config_v3 import (
    BLOCK_SIZE,
    VOCAB_SIZE,
)


class GPTDatasetV3(Dataset):

    def __init__(
        self,
        file_path,
        block_size=BLOCK_SIZE,
    ):

        self.data = np.load(
            file_path,
            mmap_mode="r",
        )

        self.block_size = block_size

        if self.data.dtype != np.uint16:
            raise ValueError(
                f"Expected uint16 token data, got {self.data.dtype}"
            )

        if len(self.data) <= block_size:
            raise ValueError(
                "Dataset must contain more tokens "
                "than the context length."
            )

        minimum_id = int(self.data.min())
        maximum_id = int(self.data.max())

        if minimum_id < 0 or maximum_id >= VOCAB_SIZE:
            raise ValueError(
                f"Invalid token ID range: {minimum_id}-{maximum_id}. "
                f"Expected IDs between 0 and {VOCAB_SIZE - 1}."
            )

        self.num_samples = (
            len(self.data) - 1
        ) // block_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        start = index * self.block_size
        end = start + self.block_size

        x = np.array(
            self.data[start:end],
            dtype=np.int64,
        )

        y = np.array(
            self.data[start + 1:end + 1],
            dtype=np.int64,
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
        )


def create_dataloader_v3(
    file_path,
    batch_size=8,
    block_size=BLOCK_SIZE,
    shuffle=True,
):

    dataset = GPTDatasetV3(
        file_path=file_path,
        block_size=block_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    return loader
