/**
 * API Service for MiniGPT V3 Backend
 * Communicates with FastAPI server at http://localhost:8000 (proxied via Vite /api in dev).
 */

const API_BASE = '/api';

/**
 * Fetch health status of the backend and V3 model
 */
export async function getHealthStatus() {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`Backend returned HTTP ${response.status}`);
    }

    const data = await response.json();
    return {
      success: true,
      data,
    };
  } catch (error) {
    return {
      success: false,
      error: error.message || 'Unable to connect to MiniGPT backend at http://localhost:8000',
    };
  }
}

/**
 * Send a prompt to MiniGPT V3 for autoregressive text generation
 *
 * @param {Object} params
 * @param {string} params.message
 * @param {number} params.max_new_tokens
 * @param {number} params.temperature
 * @param {number} params.top_k
 * @param {number} params.repetition_penalty
 */
export async function generateCompletion({
  message,
  model_version = 'v4',
  max_new_tokens = 80,
  temperature = 0.7,
  top_k = 40,
  repetition_penalty = 1.15,
}) {
  try {
    const payload = {
      message: message.trim(),
      model_version: model_version,
      max_new_tokens: Number(max_new_tokens),
      temperature: Number(temperature),
      top_k: top_k !== null && top_k !== undefined ? Number(top_k) : null,
      repetition_penalty: Number(repetition_penalty),
    };

    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let errorMessage = `Server error (${response.status})`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          if (Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail.map(d => d.msg || JSON.stringify(d)).join(', ');
          } else {
            errorMessage = errorData.detail;
          }
        }
      } catch {
        // Fallback to status text
        errorMessage = response.statusText || errorMessage;
      }
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return {
      success: true,
      response: data.response,
      model: data.model || 'MiniGPT V3',
    };
  } catch (error) {
    return {
      success: false,
      error: error.message || 'Generation failed. Please verify that the backend is running.',
    };
  }
}

