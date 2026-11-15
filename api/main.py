from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import ChatRequest, ChatResponse, HealthResponse
from api.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> [LIFESPAN] Starting up...", flush=True)
    try:
        model_service.load()
        print(">>> [LIFESPAN] MiniGPT models loaded successfully.", flush=True)
    except Exception as e:
        print(f">>> [LIFESPAN] Warning: Failed to load MiniGPT models: {e}", flush=True)
    yield
    print(">>> [LIFESPAN] Shutting down...", flush=True)
    # Shutdown logic if any (e.g. cleanup)


app = FastAPI(
    title="MiniGPT V3 API",
    description="Backend API for interacting with the locally trained MiniGPT V3 language model.",
    version="3.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint exposing model and device status."""
    is_loaded = model_service.is_loaded
    device_str = str(model_service.device).upper() if model_service.device else "UNKNOWN"

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_version=model_service.model_version,
        checkpoint_name=model_service.checkpoint_name,
        device=device_str,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate text completion from a prompt using MiniGPT V3."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable.",
        )

    try:
        generated_text, model_name = await model_service.generate_text(
            prompt=request.message,
            model_version=request.model_version or "v4",
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )

        return ChatResponse(
            response=generated_text,
            model=model_name,
        )

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except RuntimeError as run_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(run_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during generation: {str(err)}",
        )

