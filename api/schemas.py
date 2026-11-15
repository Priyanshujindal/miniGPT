from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., description="Prompt text to continue from")
    model_version: Optional[str] = Field(
        default="v4",
        description="Target model version: 'v4' (Conversational Assistant) or 'v3' (Story Generator)",
    )
    max_new_tokens: int = Field(
        default=100,
        ge=1,
        le=128,
        description="Maximum number of new tokens to generate (1-128)",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 for greedy decoding)",
    )
    top_k: Optional[int] = Field(
        default=40,
        ge=1,
        le=200,
        description="Top-k filtering threshold",
    )
    repetition_penalty: float = Field(
        default=1.15,
        gt=0.0,
        le=2.0,
        description="Penalty for repeating tokens",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message prompt cannot be empty.")
        return stripped


class ChatResponse(BaseModel):
    response: str
    model: str = "MiniGPT V3"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    checkpoint_name: str
    device: str

