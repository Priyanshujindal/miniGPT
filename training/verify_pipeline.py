
import numpy as np

from tokenizer.bpe import BPETokenizer


TOKENIZER_PATH = "data/tokenizer/bpe_tokenizer.json"

TRAIN_PATH = "data/tokenized/train.npy"

VAL_PATH = "data/tokenized/validation.npy"


def main():

    tokenizer = BPETokenizer.load(
        TOKENIZER_PATH
    )

    train = np.load(
        TRAIN_PATH,
        mmap_mode="r"
    )

    val = np.load(
        VAL_PATH,
        mmap_mode="r"
    )

    eos_id = tokenizer.vocab_size

    model_vocab_size = tokenizer.vocab_size + 1

    print("Tokenizer vocabulary:", tokenizer.vocab_size)
    print("EOS token ID:", eos_id)
    print("Model vocabulary size:", model_vocab_size)

    print("\nTraining tokens:", len(train))
    print("Validation tokens:", len(val))

    print(
        "Training storage:",
        round(train.nbytes / (1024 ** 2), 2),
        "MB"
    )

    print(
        "Validation storage:",
        round(val.nbytes / (1024 ** 2), 2),
        "MB"
    )

    print("\nToken range:")

    print("Training minimum:", int(train.min()))
    print("Training maximum:", int(train.max()))

    print("Validation minimum:", int(val.min()))
    print("Validation maximum:", int(val.max()))

    assert train.min() >= 0
    assert val.min() >= 0

    assert train.max() < model_vocab_size
    assert val.max() < model_vocab_size

    assert eos_id in train
    assert eos_id in val

    print("\nPipeline verification passed!")


if __name__ == "__main__":
    main()
