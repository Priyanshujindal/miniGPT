import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from training.config_v4 import (
    TOKENIZER_V4_PATH,
    BLOCK_SIZE,
    VOCAB_SIZE,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_OF_TURN_TOKEN,
    END_TOKEN,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    ASSISTANT_TOKEN_ID,
    END_OF_TURN_TOKEN_ID,
    END_TOKEN_ID,
)


class ConversationDatasetV4(Dataset):
    """
    Dataset for conversational instruction tuning with Assistant-Only Loss Masking.
    Masks out system and user tokens with label -100 so cross-entropy loss is computed
    only on the assistant's output and turn termination tokens.
    """

    def __init__(
        self,
        file_path,
        tokenizer=None,
        block_size=BLOCK_SIZE,
    ):
        self.file_path = file_path
        self.block_size = block_size

        if tokenizer is None:
            self.tokenizer = Tokenizer.from_file(str(TOKENIZER_V4_PATH))
        else:
            self.tokenizer = tokenizer

        self.samples = []
        self._load_and_process()

    def _load_and_process(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line_str = line.strip()
                if not line_str:
                    continue

                record = json.loads(line_str)
                messages = record["messages"]

                # Process conversation turns into token IDs and assistant mask
                token_ids, mask = self._tokenize_and_mask(messages)

                # Ensure sequence fits within context window
                if len(token_ids) > self.block_size:
                    token_ids = token_ids[: self.block_size]
                    mask = mask[: self.block_size]

                # We need at least 2 tokens to have input x and target y
                if len(token_ids) < 2:
                    continue

                # Ensure at least one assistant token is unmasked
                # (mask[1:] corresponds to target positions in y)
                if not any(mask[1:]):
                    continue

                self.samples.append((token_ids, mask))

        if len(self.samples) == 0:
            raise ValueError(f"No valid conversational samples found in {self.file_path}")

    def _tokenize_and_mask(self, messages):
        """
        Tokenize messages and create a boolean mask where True = assistant token (active for loss).
        """
        token_ids = []
        mask = []  # True for tokens where loss should be calculated, False otherwise

        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()

            if role == "system":
                # <|system|>content<|endofturn|> -> MASKED
                turn_str = f"{SYSTEM_TOKEN}{content}{END_OF_TURN_TOKEN}"
                encoded = self.tokenizer.encode(turn_str)
                token_ids.extend(encoded.ids)
                mask.extend([False] * len(encoded.ids))

            elif role == "user":
                # <|user|>content<|endofturn|> -> MASKED
                turn_str = f"{USER_TOKEN}{content}{END_OF_TURN_TOKEN}"
                encoded = self.tokenizer.encode(turn_str)
                token_ids.extend(encoded.ids)
                mask.extend([False] * len(encoded.ids))

            elif role == "assistant":
                # <|assistant|> -> MASKED prefix
                asst_prefix = self.tokenizer.encode(ASSISTANT_TOKEN).ids
                token_ids.extend(asst_prefix)
                mask.extend([False] * len(asst_prefix))

                # content<|endofturn|> -> ACTIVE (loss computed!)
                asst_body = f"{content}{END_OF_TURN_TOKEN}"
                encoded = self.tokenizer.encode(asst_body)
                token_ids.extend(encoded.ids)
                mask.extend([True] * len(encoded.ids))

        # Append final <|end|> token -> ACTIVE (model must learn sequence termination)
        end_id = self.tokenizer.token_to_id(END_TOKEN)
        token_ids.append(end_id)
        mask.append(True)

        return token_ids, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, mask = self.samples[idx]

        # x: tokens[:-1], y: tokens[1:]
        seq_len = len(tokens) - 1
        x = list(tokens[:-1])
        y = list(tokens[1:])
        target_mask = mask[1:]

        # Apply loss masking: where target_mask is False, target y is -100
        for i in range(seq_len):
            if not target_mask[i]:
                y[i] = -100

        # Pad to block_size with END_TOKEN_ID for x and -100 for y
        pad_len = self.block_size - seq_len
        if pad_len > 0:
            x.extend([END_TOKEN_ID] * pad_len)
            y.extend([-100] * pad_len)

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def create_dataloader_v4(
    file_path,
    batch_size=16,
    block_size=BLOCK_SIZE,
    shuffle=True,
    tokenizer=None,
):
    dataset = ConversationDatasetV4(
        file_path=file_path,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return loader
