from pathlib import Path
import torch

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Tokenizer paths
TOKENIZER_V3_PATH = PROJECT_ROOT / "data" / "tokenizer" / "bpe_tokenizer_v3.json"
TOKENIZER_V4_PATH = PROJECT_ROOT / "data" / "tokenizer" / "bpe_tokenizer_v4.json"
TOKENIZER_V4_REPORT_PATH = PROJECT_ROOT / "data" / "tokenizer" / "tokenizer_v4_report.json"

# V4 Dataset paths
DATA_DIR_V4 = PROJECT_ROOT / "data" / "conversations_v4"
TRAIN_DATA_PATH = DATA_DIR_V4 / "train.jsonl"
VAL_DATA_PATH = DATA_DIR_V4 / "validation.jsonl"
TEST_DATA_PATH = DATA_DIR_V4 / "test.jsonl"
DATASET_REPORT_PATH = DATA_DIR_V4 / "dataset_report.json"

# Checkpoint directories (strictly separated)
CHECKPOINT_DIR_TRANSFER = PROJECT_ROOT / "checkpoints_tokenizer_v4" / "transfer"
CHECKPOINT_DIR_SCRATCH = PROJECT_ROOT / "checkpoints_tokenizer_v4" / "scratch"

# Log directories (strictly separated)
LOG_DIR_TRANSFER = PROJECT_ROOT / "logs_tokenizer_v4" / "transfer"
LOG_DIR_SCRATCH = PROJECT_ROOT / "logs_tokenizer_v4" / "scratch"

# V3 checkpoint source for transfer
V3_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints_tokenizer_v3" / "best.pt"

# Tokenizer configuration
TOKENIZER_BASE_VOCAB_SIZE = 2048  # V3 BPE tokens: 0..2047

# V4 Special Tokens
END_TOKEN = "<|end|>"            # 2048: Conversation / sequence end (matches V3 EOS)
SYSTEM_TOKEN = "<|system|>"      # 2049: System instruction boundary
USER_TOKEN = "<|user|>"          # 2050: User message prefix
ASSISTANT_TOKEN = "<|assistant|>"# 2051: Assistant message prefix
END_OF_TURN_TOKEN = "<|endofturn|>" # 2052: Turn delimiter

SPECIAL_TOKENS = [
    END_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_OF_TURN_TOKEN,
]

END_TOKEN_ID = 2048
SYSTEM_TOKEN_ID = 2049
USER_TOKEN_ID = 2050
ASSISTANT_TOKEN_ID = 2051
END_OF_TURN_TOKEN_ID = 2052

VOCAB_SIZE = TOKENIZER_BASE_VOCAB_SIZE + len(SPECIAL_TOKENS)  # 2053

# Model architecture (identical to V3, with expanded vocab)
EMBEDDING_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
BLOCK_SIZE = 128  # Context length
DROPOUT = 0.1

# Training configuration
BATCH_SIZE = 16
LEARNING_RATE_TRANSFER = 2e-4  # Slightly lower for fine-tuning
LEARNING_RATE_SCRATCH = 3e-4   # Standard rate for scratch training
MIN_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
GRAD_CLIP = 1.0

EVAL_INTERVAL = 100
EVAL_BATCHES = 25
LOG_INTERVAL = 20
SAVE_INTERVAL = 500

MAX_STEPS_TRANSFER = 4000
MAX_STEPS_SCRATCH = 4000

USE_AMP = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
