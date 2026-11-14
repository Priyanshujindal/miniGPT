
from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = (
    PROJECT_ROOT / "data" / "tokenizer" / "bpe_tokenizer_v2.json"
)

TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "tokenized_v2" / "train.npy"
)

VAL_DATA_PATH = (
    PROJECT_ROOT / "data" / "tokenized_v2" / "validation.npy"
)

# Keep V2 checkpoints and logs separate from V1.
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_tokenizer_v2"
LOG_DIR = PROJECT_ROOT / "logs_tokenizer_v2"

# Tokenizer configuration
TOKENIZER_VOCAB_SIZE = 2048
EOS_TOKEN_ID = 2048
VOCAB_SIZE = TOKENIZER_VOCAB_SIZE + 1

# Model configuration
EMBEDDING_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
BLOCK_SIZE = 128
DROPOUT = 0.1

# Training configuration
BATCH_SIZE = 8
MAX_STEPS = 20000

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
GRAD_CLIP = 1.0

EVAL_INTERVAL = 100
EVAL_BATCHES = 20
LOG_INTERVAL = 20

USE_AMP = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"V2 training device: {DEVICE}")
print(f"V2 tokenizer vocabulary: {TOKENIZER_VOCAB_SIZE}")
print(f"V2 model vocabulary: {VOCAB_SIZE}")
