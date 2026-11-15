import threading
from typing import Optional, Tuple
import torch

from generation.load_model_v3 import load_model_v3
from generation.load_model_v4 import load_model_v4
from generation.generate import generate
from training.config_v3 import (
    EOS_TOKEN_ID as V3_EOS_TOKEN_ID,
    BLOCK_SIZE as V3_BLOCK_SIZE,
)
from training.config_v4 import (
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_OF_TURN_TOKEN,
    END_OF_TURN_TOKEN_ID,
    END_TOKEN_ID,
    BLOCK_SIZE as V4_BLOCK_SIZE,
)


class ModelService:
    def __init__(self, checkpoint_name: str = "best.pt"):
        self.checkpoint_name = checkpoint_name
        self.model_v3 = None
        self.tokenizer_v3 = None
        self.model_v4 = None
        self.tokenizer_v4 = None
        self.device = None
        self.model_version = "v3"
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and (self.model_v4 is not None or self.model_v3 is not None)

    def load(self) -> None:
        """Load V4 (Transfer) and V3 models at startup."""
        if self._loaded:
            return

        # 1. Attempt loading V4-Transfer (conversational assistant)
        try:
            self.model_v4, self.tokenizer_v4, self.device = load_model_v4(
                model_type="transfer",
                checkpoint_name=self.checkpoint_name,
            )
            self.model_v4.eval()
            print(">>> [MODEL SERVICE] MiniGPT V4-Transfer loaded successfully.")
        except Exception as e:
            print(f">>> [MODEL SERVICE] Notice: MiniGPT V4-Transfer not loaded: {e}")

        # 2. Attempt loading V3 (storyteller)
        try:
            self.model_v3, self.tokenizer_v3, self.device = load_model_v3(
                checkpoint_name=self.checkpoint_name
            )
            self.model_v3.eval()
            print(">>> [MODEL SERVICE] MiniGPT V3 loaded successfully.")
        except Exception as e:
            print(f">>> [MODEL SERVICE] Notice: MiniGPT V3 not loaded: {e}")

        if self.model_v4 is None and self.model_v3 is None:
            self._loaded = False
            raise RuntimeError("Neither MiniGPT V4 nor V3 model could be loaded.")

        self.model_version = "v3"
        self._loaded = True

    def _run_inference(
        self,
        prompt: str,
        model_version: str = "v4",
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        repetition_penalty: float = 1.15,
    ) -> Tuple[str, str]:
        """Synchronous inference execution inside torch.inference_mode()."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        is_v4 = (model_version.lower() == "v4" and self.model_v4 is not None) or self.model_v3 is None

        if is_v4:
            model = self.model_v4
            tokenizer = self.tokenizer_v4
            model_name = "MiniGPT V4 (Transfer)"

            prompt_text = (
                f"{SYSTEM_TOKEN}You are MiniGPT, a polite and helpful assistant.{END_OF_TURN_TOKEN}"
                f"{USER_TOKEN}{prompt.strip()}{END_OF_TURN_TOKEN}"
                f"{ASSISTANT_TOKEN}"
            )
            encoded = tokenizer.encode(prompt_text)
            token_count = len(encoded.ids)

            if token_count > (V4_BLOCK_SIZE - 20):
                # Fallback to shorter prompt if user message is long
                prompt_text = f"{USER_TOKEN}{prompt.strip()}{END_OF_TURN_TOKEN}{ASSISTANT_TOKEN}"
                encoded = tokenizer.encode(prompt_text)

            input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)

            try:
                with torch.inference_mode():
                    output_ids = generate(
                        model=model,
                        input_ids=input_ids,
                        max_new_tokens=min(max_new_tokens, 80),
                        temperature=temperature,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        eos_token_id=END_OF_TURN_TOKEN_ID,
                    )

                generated_ids = output_ids[0].tolist()
                new_tokens = generated_ids[input_ids.shape[1] :]

                clean_tokens = []
                for tid in new_tokens:
                    if tid in [END_OF_TURN_TOKEN_ID, END_TOKEN_ID]:
                        break
                    clean_tokens.append(tid)

                response_text = tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()
                if not response_text:
                    response_text = "I am not sure how to answer that yet."
                return response_text, model_name

            except torch.cuda.OutOfMemoryError as oom_err:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("CUDA Out of Memory during generation.") from oom_err

        else:
            # MiniGPT V3 Storyteller logic
            model = self.model_v3
            tokenizer = self.tokenizer_v3
            model_name = "MiniGPT V3"

            encoded = tokenizer.encode(prompt)
            token_count = len(encoded.ids)

            if token_count == 0:
                raise ValueError("Prompt produced 0 tokens. Please provide valid text.")

            if token_count > V3_BLOCK_SIZE:
                raise ValueError(
                    f"Input length ({token_count} tokens) exceeds maximum sequence length of {V3_BLOCK_SIZE} tokens. "
                    "Please shorten your prompt."
                )

            input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)

            try:
                with torch.inference_mode():
                    output_ids = generate(
                        model=model,
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        eos_token_id=V3_EOS_TOKEN_ID,
                    )

                generated_ids = output_ids[0].tolist()
                new_tokens = generated_ids[input_ids.shape[1] :]

                if V3_EOS_TOKEN_ID in new_tokens:
                    new_tokens = new_tokens[: new_tokens.index(V3_EOS_TOKEN_ID)]

                generated_text = tokenizer.decode(new_tokens).strip()
                return generated_text, model_name

            except torch.cuda.OutOfMemoryError as oom_err:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("CUDA Out of Memory during generation.") from oom_err

    async def generate_text(
        self,
        prompt: str,
        model_version: str = "v4",
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        repetition_penalty: float = 1.15,
    ) -> Tuple[str, str]:
        """Asynchronous wrapper that serializes inference requests."""
        if not self.is_loaded:
            raise RuntimeError("MiniGPT model is not loaded.")

        with self._lock:
            return self._run_inference(
                prompt=prompt,
                model_version=model_version,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )


# Singleton instance
model_service = ModelService(checkpoint_name="best.pt")
