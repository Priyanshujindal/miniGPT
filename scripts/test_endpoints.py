import json
import sys
from fastapi.testclient import TestClient
from api.main import app
from api.model_service import model_service

def main():
    print("=" * 65, flush=True)
    print("MINIGPT V3 ENDPOINTS VERIFICATION TEST SUITE", flush=True)
    print("=" * 65, flush=True)

    print("Pre-loading model on main thread...", flush=True)
    model_service.load()
    print("Model ready on main thread!", flush=True)

    with TestClient(app) as client:
        # 1. Health check
        print("\n[1] Testing GET /api/health...", flush=True)
        res = client.get("/api/health")
        print("Status code:", res.status_code, flush=True)
        data = res.json()
        print("Response:", json.dumps(data, indent=2), flush=True)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert data["status"] == "healthy", f"Expected healthy, got {data['status']}"
        assert data["model_loaded"] is True
        assert data["model_version"] == "v3"
        assert data["checkpoint_name"] == "best.pt"
        assert data["device"] in ("CUDA", "CPU")
        print(">>> /api/health PASSED!\n", flush=True)

        # 2. Chat completion
        print("[2] Testing POST /api/chat (Standard Story Generation)...", flush=True)
        chat_req = {
            "message": "Once upon a time",
            "max_new_tokens": 30,
            "temperature": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.15
        }
        res = client.post("/api/chat", json=chat_req)
        print("Status code:", res.status_code, flush=True)
        data = res.json()
        print("Response:", json.dumps(data, indent=2), flush=True)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert "response" in data
        assert data["model"] == "MiniGPT V3"
        assert len(data["response"].strip()) > 0
        print(">>> POST /api/chat PASSED! Real story generated.\n", flush=True)

        # 3. Chat completion with custom parameters
        print("[3] Testing POST /api/chat with custom parameters...", flush=True)
        chat_req2 = {
            "message": "A dog discovers a mysterious forest",
            "max_new_tokens": 25,
            "temperature": 0.3,
            "top_k": 20,
            "repetition_penalty": 1.2
        }
        res2 = client.post("/api/chat", json=chat_req2)
        print("Status code:", res2.status_code, flush=True)
        data2 = res2.json()
        print("Response:", json.dumps(data2, indent=2), flush=True)
        assert res2.status_code == 200
        assert len(data2["response"].strip()) > 0
        print(">>> Custom parameters test PASSED!\n", flush=True)

        # 4. Validation: Empty message
        print("[4] Testing POST /api/chat (Validation: Empty message)...", flush=True)
        bad_res1 = client.post("/api/chat", json={"message": "   "})
        print("Status code:", bad_res1.status_code, flush=True)
        assert bad_res1.status_code in (400, 422), f"Expected 400/422, got {bad_res1.status_code}"
        print(">>> Empty message validation PASSED!\n", flush=True)

        # 5. Validation: Max tokens out of bounds (> 128)
        print("[5] Testing POST /api/chat (Validation: max_new_tokens > 128)...", flush=True)
        bad_res2 = client.post("/api/chat", json={"message": "Hello", "max_new_tokens": 500})
        print("Status code:", bad_res2.status_code, flush=True)
        assert bad_res2.status_code == 422, f"Expected 422, got {bad_res2.status_code}"
        print(">>> Max tokens bound validation PASSED!\n", flush=True)

        # 6. Validation: Invalid temperature (< 0)
        print("[6] Testing POST /api/chat (Validation: temperature < 0)...", flush=True)
        bad_res3 = client.post("/api/chat", json={"message": "Hello", "temperature": -1.0})
        print("Status code:", bad_res3.status_code, flush=True)
        assert bad_res3.status_code == 422, f"Expected 422, got {bad_res3.status_code}"
        print(">>> Temperature validation PASSED!\n", flush=True)

        # 7. Validation: Invalid repetition penalty (<= 0)
        print("[7] Testing POST /api/chat (Validation: repetition_penalty <= 0)...", flush=True)
        bad_res4 = client.post("/api/chat", json={"message": "Hello", "repetition_penalty": 0})
        print("Status code:", bad_res4.status_code, flush=True)
        assert bad_res4.status_code == 422, f"Expected 422, got {bad_res4.status_code}"
        print(">>> Repetition penalty validation PASSED!\n", flush=True)

    print("=" * 65, flush=True)
    print("ALL 7 ENDPOINT AND VALIDATION TESTS PASSED!", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
