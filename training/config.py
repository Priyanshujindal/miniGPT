from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = PROJECT_ROOT / "data" / "tokenizer" / "bpe_tokenizer.json"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "tokenized" / "train.npy"
VAL_DATA_PATH = PROJECT_ROOT / "data" / "tokenized" / "validation.npy"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Model configuration
EMBEDDING_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
BLOCK_SIZE = 128

# Training configuration
BATCH_SIZE = 8
MAX_STEPS = 2000
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
GRAD_CLIP = 1.0

EVAL_INTERVAL = 100
EVAL_BATCHES = 20
LOG_INTERVAL = 20

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training device: {DEVICE}")