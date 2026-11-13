
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):

    def __init__(
        self,
        file_path,
        block_size=128
    ):

        self.data = np.load(
            file_path,
            mmap_mode="r"
        )

        self.block_size = block_size

        if len(self.data) <= block_size:
            raise ValueError(
                "Dataset must contain more tokens "
                "than the context length."
            )

        # Non-overlapping context windows.
        self.num_samples = (
            len(self.data) - 1
        ) // block_size

    def __len__(self):

        return self.num_samples

    def __getitem__(self, index):

        start = index * self.block_size

        end = start + self.block_size

        # Copy the read-only memory-mapped slices.
        x = np.array(
            self.data[start:end],
            dtype=np.int64
        )

        y = np.array(
            self.data[start + 1:end + 1],
            dtype=np.int64
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(y)
        )


def create_dataloader(
    file_path,
    batch_size=16,
    block_size=128,
    shuffle=True
):

    dataset = GPTDataset(
        file_path=file_path,
        block_size=block_size
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    return loader
