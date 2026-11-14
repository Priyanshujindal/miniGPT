import asyncio
from typing import Optional, Tuple
import torch

from generation.load_model_v3 import load_model_v3
from generation.generate import generate
from training.config_v3 import (
    EOS_TOKEN_ID,
    BLOCK_SIZE,
)


class ModelService:
    def __init__(self, checkpoint_name: str = "best.pt"):
        self.checkpoint_name = checkpoint_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_version = "v3"
        self._lock: Optional[asyncio.Lock] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loaded = False

    def get_lock(self) -> asyncio.Lock:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._lock is None or self._loop is not current_loop:
            self._lock = asyncio.Lock()
            self._loop = current_loop
        return self._lock

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Load the V3 model and tokenizer once at startup."""
        if self._loaded:
            return

        try:
            self.model, self.tokenizer, self.device = load_model_v3(self.checkpoint_name)
            self.model.eval()
            self._loaded = True
        except Exception as e:
            self._loaded = False
            raise RuntimeError(f"Failed to load MiniGPT V3 model ({self.checkpoint_name}): {e}") from e

    def _run_inference(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        repetition_penalty: float = 1.15,
    ) -> str:
        """Synchronous inference execution inside torch.inference_mode()."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        # Encode input
        encoded = self.tokenizer.encode(prompt)
        token_count = len(encoded.ids)

        if token_count == 0:
            raise ValueError("Prompt produced 0 tokens. Please provide valid text.")

        if token_count > BLOCK_SIZE:
            raise ValueError(
                f"Input length ({token_count} tokens) exceeds maximum sequence length of {BLOCK_SIZE} tokens. "
                "Please shorten your prompt."
            )

        input_ids = torch.tensor(
            [encoded.ids],
            dtype=torch.long,
            device=self.device,
        )

        try:
            with torch.inference_mode():
                output_ids = generate(
                    model=self.model,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=EOS_TOKEN_ID,
                )

            generated_ids = output_ids[0].tolist()
            new_tokens = generated_ids[input_ids.shape[1] :]

            # Truncate at EOS token if produced
            if EOS_TOKEN_ID in new_tokens:
                new_tokens = new_tokens[: new_tokens.index(EOS_TOKEN_ID)]

            generated_text = self.tokenizer.decode(new_tokens)
            return generated_text

        except torch.cuda.OutOfMemoryError as oom_err:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError("CUDA Out of Memory during generation. Try reducing max_new_tokens.") from oom_err

    async def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        repetition_penalty: float = 1.15,
    ) -> str:
        """Asynchronous wrapper that serializes inference requests."""
        if not self.is_loaded:
            raise RuntimeError("MiniGPT V3 model is not loaded.")

        async with self.get_lock():
            return self._run_inference(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )


# Singleton instance
model_service = ModelService(checkpoint_name="best.pt")

